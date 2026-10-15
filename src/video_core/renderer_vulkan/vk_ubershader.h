// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

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

/**
 * The fragment ubershader. AZAHAR_UBERSHADER_FORCE=1 draws every draw it can take through the CPU
 * vertex path and the ubershader, to compare it with the specialised shaders.
 * AZAHAR_UBERSHADER_FORCE=cpu takes the same draws through the CPU vertex path but keeps their
 * specialised shaders, which separates the two sources of difference.
 */
class Ubershader {
public:
    explicit Ubershader(const Instance& instance, RenderManager& renderpass_cache,
                        Common::ThreadWorker& shader_workers,
                        Common::ThreadWorker& pipeline_workers);
    ~Ubershader();

    /// Packs the config of the current fragment shader. Returns true when it changed.
    bool UseFragmentShader(const Pica::RegsInternal& regs, const Pica::Shader::UserConfig& user,
                           const Pica::Shader::Profile& profile, u64 fs_hash);

    [[nodiscard]] const UberConfig& Config() const noexcept {
        return config;
    }

    /// Whether hardware draws of the current fragment shader go through the CPU vertex path.
    [[nodiscard]] bool UsesCpuVertexPath() const noexcept {
        return mode != Mode::Off && variant.has_value();
    }

    /// The pipeline that draws a CPU vertex path draw of the current fragment shader with the
    /// ubershader, or null when its specialised pipeline draws it.
    GraphicsPipeline* Pipeline(const PipelineInfo& info, Shader* vertex_shader,
                               vk::PipelineLayout layout);

private:
    enum class Mode {
        Off,
        Ubershader,
        SoftwareOnly,
    };

    const Instance& instance;
    RenderManager& renderpass_cache;
    Common::ThreadWorker& pipeline_workers;
    Mode mode{Mode::Off};
    std::array<std::unique_ptr<Shader>, NumUberTex0> shaders;
    std::unordered_map<u64, std::unique_ptr<GraphicsPipeline>> pipelines;
    UberConfig config{};
    std::optional<UberTex0> variant;
    std::optional<u64> config_key;
};

} // namespace Vulkan
