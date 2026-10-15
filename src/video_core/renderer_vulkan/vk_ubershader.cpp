// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string_view>

#include "common/assert.h"
#include "common/hash.h"
#include "video_core/host_shaders/vulkan_fs_ubershader_frag.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_ubershader.h"
#include "video_core/shader/generator/shader_uniforms.h"

namespace Vulkan {

namespace {

using Pica::Shader::FSConfig;
using Pica::Shader::Generator::FSUniformData;
using TevStageConfig = Pica::TexturingRegs::TevStageConfig;
using TextureType = Pica::TexturingRegs::TextureConfig::TextureType;

// The ubershader reads these words by index (CFG_* in vulkan_fs_ubershader.frag), and declares
// its config at this offset of the uniform block.
static_assert(sizeof(FSUniformData) == 1328);
static_assert(sizeof(FSConfig) == 68 * sizeof(u32));
static_assert(offsetof(FSConfig, framebuffer) == 0);
static_assert(offsetof(FSConfig, texture) == 8 * sizeof(u32));
static_assert(offsetof(FSConfig, texture) +
                  offsetof(Pica::Shader::TextureConfig, texture_border_color) ==
              9 * sizeof(u32));
static_assert(offsetof(FSConfig, texture) + offsetof(Pica::Shader::TextureConfig, tev_stages) ==
              12 * sizeof(u32));
static_assert(offsetof(FSConfig, lighting) == 42 * sizeof(u32));
static_assert(offsetof(FSConfig, lighting) + offsetof(Pica::Shader::LightConfig, lut_d0) ==
              43 * sizeof(u32));
static_assert(offsetof(FSConfig, lighting) + offsetof(Pica::Shader::LightConfig, lut_rb) ==
              55 * sizeof(u32));
static_assert(offsetof(FSConfig, lighting) + offsetof(Pica::Shader::LightConfig, lights) ==
              57 * sizeof(u32));
static_assert(offsetof(FSConfig, proctex) == 61 * sizeof(u32));
static_assert(offsetof(FSConfig, proctex) + offsetof(Pica::Shader::ProcTexConfig, lut_width) ==
              62 * sizeof(u32));
static_assert(offsetof(FSConfig, proctex) + offsetof(Pica::Shader::ProcTexConfig, lut_offset0) ==
              63 * sizeof(u32));
static_assert(offsetof(FSConfig, proctex) + offsetof(Pica::Shader::ProcTexConfig, lod_min) ==
              67 * sizeof(u32));

constexpr std::size_t TexturesUsedWord = 68;

constexpr std::array<std::string_view, NumUberTex0> Preambles = {
    "",
    "#define TEX0_CUBE\n",
    "#define TEX0_SHADOW\n",
};

// Stands in for a fragment shader hash in the ubershader pipelines' keys.
constexpr u64 UbershaderId = 0x7562'6572'0000'0000ULL;

bool IsPassThrough(const TevStageConfig& stage) {
    return stage.color_op == TevStageConfig::Operation::Replace &&
           stage.alpha_op == TevStageConfig::Operation::Replace &&
           stage.color_source1 == TevStageConfig::Source::Previous &&
           stage.alpha_source1 == TevStageConfig::Source::Previous &&
           stage.color_modifier1 == TevStageConfig::ColorModifier::SourceColor &&
           stage.alpha_modifier1 == TevStageConfig::AlphaModifier::SourceAlpha &&
           stage.GetColorMultiplier() == 1 && stage.GetAlphaMultiplier() == 1;
}

// The texture units the specialised shader samples. The ubershader samples only these, which
// keeps it off units whose bound view doesn't match its sampler type.
u32 TexturesUsed(const FSConfig& config) {
    using Source = TevStageConfig::Source;
    u32 used = 0;
    const auto add = [&used](Source source) {
        const u32 value = static_cast<u32>(source);
        if (value >= static_cast<u32>(Source::Texture0) &&
            value <= static_cast<u32>(Source::Texture3)) {
            used |= 1u << (value - static_cast<u32>(Source::Texture0));
        }
    };
    for (u32 index = 0; index < config.texture.tev_stages.size(); index++) {
        const TevStageConfig stage = config.texture.tev_stages[index];
        if (IsPassThrough(stage)) {
            continue;
        }
        const auto resolve_previous = [index](Source source, Source third) {
            return index == 0 && source == Source::Previous ? third : source;
        };
        add(resolve_previous(stage.color_source1, stage.color_source3));
        add(resolve_previous(stage.color_source2, stage.color_source3));
        add(stage.color_source3);
        if (stage.color_op != TevStageConfig::Operation::Dot3_RGBA) {
            add(resolve_previous(stage.alpha_source1, stage.alpha_source3));
            add(resolve_previous(stage.alpha_source2, stage.alpha_source3));
            add(stage.alpha_source3);
        }
    }
    const auto& lighting = config.lighting;
    if (lighting.enable) {
        if (lighting.bump_mode == Pica::LightingRegs::LightingBumpMode::NormalMap ||
            lighting.bump_mode == Pica::LightingRegs::LightingBumpMode::TangentMap) {
            used |= 1u << lighting.bump_selector;
        }
        if (lighting.enable_shadow) {
            used |= 1u << lighting.shadow_selector;
        }
    }
    return used;
}

} // Anonymous namespace

UberConfig PackUberConfig(const FSConfig& applied) {
    UberConfig words{};
    std::memcpy(words.data(), &applied, sizeof(applied));
    words[TexturesUsedWord] = TexturesUsed(applied);
    return words;
}

std::optional<UberTex0> UbershaderVariant(const FSConfig& config,
                                          const Pica::Shader::UserConfig& user,
                                          const Pica::Shader::Profile& profile) {
    if (!profile.vk_use_spirv_generator || config.UsesSpirvIncompatibleConfig() ||
        config.texture.fog_mode == Pica::TexturingRegs::FogMode::Gas || user.use_custom_normal ||
        (profile.has_fragment_shader_barycentric && config.lighting.enable)) {
        return std::nullopt;
    }
    switch (config.texture.texture0_type) {
    case TextureType::TextureCube:
        return UberTex0::TextureCube;
    case TextureType::Shadow2D:
        return UberTex0::Shadow2D;
    default:
        return UberTex0::Texture2D;
    }
}

std::vector<u32> CompileUbershader(UberTex0 variant) {
    return CompileGLSL(HostShaders::VULKAN_FS_UBERSHADER_FRAG, vk::ShaderStageFlagBits::eFragment,
                       Preambles[static_cast<std::size_t>(variant)]);
}

Ubershader::Ubershader(const Instance& instance_, RenderManager& renderpass_cache_,
                       Common::ThreadWorker& shader_workers,
                       Common::ThreadWorker& pipeline_workers_)
    : instance{instance_}, renderpass_cache{renderpass_cache_},
      pipeline_workers{pipeline_workers_} {
    const char* force = std::getenv("AZAHAR_UBERSHADER_FORCE");
    if (!force) {
        return;
    }
    const std::string_view value{force};
    if (value == "1") {
        mode = Mode::Ubershader;
    } else if (value == "cpu") {
        mode = Mode::SoftwareOnly;
        return;
    } else {
        return;
    }
    for (std::size_t i = 0; i < NumUberTex0; i++) {
        shaders[i] = std::make_unique<Shader>(instance);
        shader_workers.QueueWork([this, i] {
            const auto spirv = CompileUbershader(static_cast<UberTex0>(i));
            ASSERT_MSG(!spirv.empty(), "Failed to compile ubershader variant {}", i);
            shaders[i]->module = CompileSPV(spirv, instance.GetDevice());
            shaders[i]->MarkDone();
        });
    }
}

Ubershader::~Ubershader() = default;

bool Ubershader::UseFragmentShader(const Pica::RegsInternal& regs,
                                   const Pica::Shader::UserConfig& user,
                                   const Pica::Shader::Profile& profile, u64 fs_hash) {
    if (mode == Mode::Off) {
        return false;
    }
    const u64 key = Common::HashCombine(fs_hash, static_cast<u64>(user.raw));
    if (config_key == key) {
        return false;
    }
    FSConfig fs_config{regs};
    variant = UbershaderVariant(fs_config, user, profile);
    fs_config.ApplyProfile(profile);
    config = PackUberConfig(fs_config);
    config_key = key;
    return true;
}

GraphicsPipeline* Ubershader::Pipeline(const PipelineInfo& info, Shader* vertex_shader,
                                       vk::PipelineLayout layout) {
    if (mode != Mode::Ubershader || !variant) {
        return nullptr;
    }
    const std::size_t index = static_cast<std::size_t>(*variant);
    PipelineInfo fallback = info;
    fallback.state.shader_ids = {0, UbershaderId + index, 0};
    auto& pipeline = pipelines[fallback.state.OptimizedHash(instance)];
    if (!pipeline) {
        pipeline = std::make_unique<GraphicsPipeline>(
            instance, renderpass_cache, fallback, vk::PipelineCache{}, nullptr, layout,
            std::array<Shader*, MAX_SHADER_STAGES>{vertex_shader, shaders[index].get(), nullptr},
            &pipeline_workers);
    }
    return pipeline.get();
}

} // namespace Vulkan
