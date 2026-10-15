// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/thread_worker.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/shader/generator/pica_fs_config.h"

namespace Pica {
struct RegsInternal;
}

namespace Vulkan {

/// Each class of texture 0 has its own variant of the ubershader. MoltenVK can't alias a 2D and a
/// cube sampler at one binding.
enum class UberTex0 : u32 {
    Texture2D,
    TextureCube,
    Shadow2D,
};

constexpr std::size_t NumUberTex0 = 3;

/// The words the ubershader reads after FSUniformData in the fragment uniform block: the
/// profile-applied FSConfig, then the texture units it samples.
using UberConfig = std::array<u32, 72>;

/// Packs a profile-applied FSConfig for the ubershader.
[[nodiscard]] UberConfig PackUberConfig(const Pica::Shader::FSConfig& applied);

/// The ubershader variant that draws a fragment config, or nothing when its specialised shader
/// doesn't come from the SPIR-V generator or does something the ubershader leaves out.
[[nodiscard]] std::optional<UberTex0> UbershaderVariant(const Pica::Shader::FSConfig& config,
                                                        const Pica::Shader::UserConfig& user,
                                                        const Pica::Shader::Profile& profile);

/// Compiles a variant of the ubershader to SPIR-V.
[[nodiscard]] std::vector<u32> CompileUbershader(UberTex0 variant);

/// How a draw is drawn.
enum class DrawOutcome {
    /// With its own pipeline, which is built.
    Bind,
    /// A CPU vertex path draw, with an ubershader fallback pipeline.
    Fallback,
    /// A hardware draw runs its vertex shader on the CPU instead, and that draw takes Fallback.
    CpuVertexPath,
    /// With its own pipeline, and the GPU work waits for it to build.
    Wait,
    /// Not at all.
    Skip,
};

struct DrawPlan {
    DrawOutcome outcome;
    /// The pipeline to bind. For CpuVertexPath, the fallback the CPU vertex path draw binds, or
    /// null when that draw decides for itself.
    GraphicsPipeline* pipeline;
};

/// What an ubershader fallback pipeline depends on, normalised so draws that render alike share
/// one: the variant, the blend state and the attachments, plus raster and depth state on devices
/// that can't set them dynamically.
struct FallbackKey {
    UberTex0 variant;
    BlendingState blending;
    AttachmentInfo attachments;
    DepthStencilState depth_stencil;
    RasterizationState rasterization;

    [[nodiscard]] u64 Hash() const noexcept {
        return Common::ComputeStructHash64(*this);
    }

    bool operator==(const FallbackKey& other) const noexcept {
        return std::memcmp(this, &other, sizeof(FallbackKey)) == 0;
    }
};
// No padding, so its bytes are its value: it's compared, hashed and saved as bytes.
static_assert(sizeof(FallbackKey) == 32 && std::is_trivially_copyable_v<FallbackKey>);

/// The fallback key of a draw's pipeline state.
[[nodiscard]] FallbackKey MakeFallbackKey(const StaticPipelineInfo& state, UberTex0 variant,
                                          bool logic_op_emulated, bool extended_dynamic_state);

/// The fallback keys saved in a file, or nothing when there's no file of saved keys there.
[[nodiscard]] std::optional<std::vector<FallbackKey>> ReadFallbackKeys(const std::string& path);

/// Adds a key to a file of saved keys, starting the file over when it isn't one.
void SaveFallbackKey(const std::string& path, const FallbackKey& key);

/**
 * The fragment ubershader. With async shaders and the ubershader fallback setting on, a draw whose
 * own pipeline isn't built yet runs its vertex shader on the CPU and draws through a fallback
 * pipeline with the ubershader, while its own pipeline builds. The fallback pipelines are built
 * when the renderer starts, from the keys a title used before and a few seeds, and the keys a title
 * uses are saved per title.
 *
 * AZAHAR_UBERSHADER_FORCE=1 draws every draw it can take through the CPU vertex path and the
 * ubershader, to compare it with the specialised shaders. AZAHAR_UBERSHADER_FORCE=cpu takes the
 * same draws through the CPU vertex path but keeps their specialised shaders, which separates the
 * two sources of difference.
 */
class Ubershader {
public:
    explicit Ubershader(const Instance& instance, RenderManager& renderpass_cache,
                        Common::ThreadWorker& shader_workers, std::size_t num_workers);
    ~Ubershader();

    /// Packs the config of the current fragment shader. Returns true when it changed.
    bool UseFragmentShader(const Pica::RegsInternal& regs, const Pica::Shader::UserConfig& user,
                           const Pica::Shader::Profile& profile, u64 fs_hash);

    [[nodiscard]] const UberConfig& Config() const noexcept {
        return config;
    }

    /// The variant that draws the current fragment shader, or nothing when it can't or the
    /// ubershader is off.
    [[nodiscard]] std::optional<UberTex0> Variant() const noexcept {
        return mode == Mode::Off ? std::nullopt : variant;
    }

    /// Draws whose own pipeline isn't built take the fallback.
    [[nodiscard]] bool FallbackOn() const noexcept {
        return mode == Mode::Fallback;
    }

    /// Every draw the ubershader can take goes through the CPU vertex path.
    [[nodiscard]] bool Forced() const noexcept {
        return mode == Mode::Force || mode == Mode::ForceCpu;
    }

    /// Forced draws keep their own fragment shader.
    [[nodiscard]] bool ForcedCpu() const noexcept {
        return mode == Mode::ForceCpu;
    }

    /// Builds the fallback pipelines a title is known to use. Before this, nothing draws.
    void Start(vk::PipelineLayout layout, Shader* vertex_shader,
               const VertexLayout& software_layout, std::string keys_dir, u64 title_id);

    /// Builds the fallback pipelines another title is known to use, and saves keys for it.
    void SwitchTitle(u64 title_id);

    /// The fallback pipeline for a draw of the current fragment shader, queued when it's new.
    GraphicsPipeline* Fallback(const StaticPipelineInfo& state);

private:
    enum class Mode {
        Off,
        Fallback,
        Force,
        ForceCpu,
    };

    Shader* VariantShader(UberTex0 tex0);
    /// The fallback pipeline for a key, and whether it's new. A new one isn't queued yet.
    std::pair<GraphicsPipeline*, bool> Emplace(const FallbackKey& key, const char* source);
    void Prewarm(std::span<const std::pair<FallbackKey, const char*>> keys);
    std::string KeysPath() const;

    const Instance& instance;
    RenderManager& renderpass_cache;
    Common::ThreadWorker& shader_workers;
    Mode mode{Mode::Off};
    std::array<std::unique_ptr<Shader>, NumUberTex0> shaders;
    UberConfig config{};
    std::optional<UberTex0> variant;
    std::optional<u64> config_key;

    vk::PipelineLayout layout;
    Shader* vertex_shader{};
    VertexLayout software_layout{};
    std::string keys_dir;
    u64 title_id{};
    std::unordered_set<u64> saved_keys;
    std::unordered_map<u64, std::unique_ptr<GraphicsPipeline>> fallbacks;
    // Declared last so its threads stop before the pipelines they build go away. It's apart from
    // the pipeline workers so that switching pipeline caches doesn't wait for fallback builds.
    std::unique_ptr<Common::ThreadWorker> workers;
};

} // namespace Vulkan
