// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>

#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"

namespace Pica {
struct RegsInternal;
}

namespace Pica::Shader {
union UserConfig;
}

namespace Vulkan {

class PipelineCache;

enum class UberTex0 : u32 {
    Texture2D,
    TextureCube,
    Shadow2D,
};

constexpr u32 NumUberTex0 = 3;

/// The applied FSConfig followed by values derived from it, appended to the fragment uniform block.
using UberConfig = std::array<u32, 72>;

/**
 * Draws whose specialised pipeline isn't built yet through a general fragment shader that reads
 * the FSConfig from the uniform block. Fallback draws use the software vertex path, so a fallback
 * pipeline only depends on the tex0 class, the blend state and the attachments.
 */
class Ubershaders {
public:
    explicit Ubershaders(PipelineCache& parent);
    ~Ubershaders();

    /// The fallback is on: async shaders with AZAHAR_UBERSHADER=1, or AZAHAR_UBERSHADER_FORCE=1.
    [[nodiscard]] bool Active() const noexcept {
        return active;
    }

    /// Every eligible draw uses the fallback, even when its specialised pipeline is built.
    [[nodiscard]] bool Forced() const noexcept {
        return forced;
    }

    /// Redirected draws still use their specialised pipeline (AZAHAR_UBERSHADER_SOFTWARE_ONLY=1),
    /// to tell differences from the software vertex path apart from the ubershader's.
    [[nodiscard]] bool SoftwareOnly() const noexcept {
        return software_only;
    }

    void Init();

    void SetSoftwareLayout(const VertexLayout& layout);

    /// Packs the config of the current fragment shader. Returns true when it changed.
    bool UseFragmentShader(const Pica::RegsInternal& regs, const Pica::Shader::UserConfig& user,
                           u64 fs_hash);

    [[nodiscard]] const UberConfig& Config() const noexcept {
        return config;
    }

    /// The tex0 class of the current fragment shader, or nothing when the ubershader can't draw it.
    [[nodiscard]] std::optional<UberTex0> Eligible() const noexcept {
        return eligible;
    }

    /// Returns the fallback pipeline for the draw, queueing its build the first time.
    GraphicsPipeline* Get(const PipelineInfo& info, UberTex0 tex0);

    /// Queues the fallback pipeline for a pipeline the disk cache knows.
    void Prewarm(const StaticPipelineInfo& state);

    /// Queues the fallback pipelines from the built-in seed list.
    void PrewarmSeeds();

    /// Remembers that the next software draw stands in for a hardware draw.
    void SetRedirected(u64 fs_hash) noexcept {
        redirected = true;
        redirected_fs_hash = fs_hash;
    }

    /// Whether the current software draw stands in for a hardware draw.
    bool TakeRedirected(u64 fs_hash) noexcept {
        return std::exchange(redirected, false) && redirected_fs_hash == fs_hash;
    }

private:
    StaticPipelineInfo FallbackState(const StaticPipelineInfo& state, UberTex0 tex0) const;

    GraphicsPipeline* GetState(const StaticPipelineInfo& state, const char* source);

    PipelineCache& parent;
    bool active{};
    bool forced{};
    bool software_only{};
    std::array<std::unique_ptr<Shader>, NumUberTex0> shaders;
    VertexLayout software_layout{};
    std::unordered_map<u64, std::unique_ptr<GraphicsPipeline>> pipelines;
    UberConfig config{};
    u64 config_fs_hash{};
    bool config_valid{};
    std::optional<UberTex0> eligible;
    bool redirected{};
    u64 redirected_fs_hash{};
};

} // namespace Vulkan
