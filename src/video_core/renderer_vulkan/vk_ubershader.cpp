// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string_view>
#include <tuple>
#include <fmt/format.h>

#include "common/assert.h"
#include "common/common_paths.h"
#include "common/file_util.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/loader/loader.h"
#include "video_core/host_shaders/vulkan_fs_ubershader_frag.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_stats.h"
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

// The blend and attachment combinations Pokémon X draws with, from the shader harness. They cover
// a title's first boot, before it has saved keys of its own.
struct Seed {
    u32 blend;
    VideoCore::PixelFormat depth;
};
constexpr std::array<Seed, 8> Seeds = {{
    {0x00000801, VideoCore::PixelFormat::D24S8},
    {0x0003b076, VideoCore::PixelFormat::D24S8},
    {0x0000b016, VideoCore::PixelFormat::D24S8},
    {0x00008010, VideoCore::PixelFormat::D24S8},
    {0x00000810, VideoCore::PixelFormat::D24S8},
    {0x000040e0, VideoCore::PixelFormat::D24S8},
    {0x00002004, VideoCore::PixelFormat::D24S8},
    {0x0003b076, VideoCore::PixelFormat::Invalid},
}};

struct KeysHeader {
    u32 magic;
    u32 version;
    u64 layout;
};

constexpr u32 KeysMagic = 0x4B465542;
constexpr u32 KeysVersion = 1;
constexpr u64 KeysLayout = Common::HashCombine(StaticPipelineInfo::StructHash(),
                                               static_cast<u64>(sizeof(FallbackKey)));

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

FallbackKey MakeFallbackKey(const StaticPipelineInfo& state, UberTex0 variant,
                            bool logic_op_emulated, bool extended_dynamic_state) {
    using LogicOp = Pica::FramebufferRegs::LogicOp;
    FallbackKey key{};
    key.variant = variant;
    key.attachments = state.attachments;
    BlendingState& blending = key.blending;
    blending = state.blending;
    if (blending.blend_enable) {
        blending.logic_op = LogicOp::Copy;
    } else {
        if (logic_op_emulated && blending.logic_op == LogicOp::NoOp) {
            blending.color_write_mask = 0;
        }
        blending.value = 0;
        if (logic_op_emulated) {
            blending.logic_op = LogicOp::Copy;
        }
    }
    if (!extended_dynamic_state) {
        key.rasterization.cull_mode.Assign(state.rasterization.cull_mode);
        key.rasterization.flip_viewport.Assign(state.rasterization.flip_viewport);
        key.depth_stencil = state.depth_stencil;
    }
    return key;
}

std::optional<std::vector<FallbackKey>> ReadFallbackKeys(const std::string& path) {
    if (!FileUtil::Exists(path)) {
        return std::nullopt;
    }
    FileUtil::IOFile file{path, "rb"};
    KeysHeader header{};
    if (!file.IsOpen() || file.ReadBytes(&header, sizeof(header)) != sizeof(header) ||
        header.magic != KeysMagic || header.version != KeysVersion ||
        header.layout != KeysLayout) {
        return std::nullopt;
    }
    std::vector<FallbackKey> keys((file.GetSize() - sizeof(header)) / sizeof(FallbackKey));
    keys.resize(file.ReadArray(keys.data(), keys.size()));
    return keys;
}

void SaveFallbackKey(const std::string& path, const FallbackKey& key) {
    std::vector<FallbackKey> keys = ReadFallbackKeys(path).value_or(std::vector<FallbackKey>{});
    keys.push_back(key);
    const std::string temp = path + ".tmp";
    {
        FileUtil::IOFile file{temp, "wb"};
        const KeysHeader header{.magic = KeysMagic, .version = KeysVersion, .layout = KeysLayout};
        if (!file.IsOpen() || file.WriteObject(header) != 1 ||
            file.WriteArray(keys.data(), keys.size()) != keys.size()) {
            LOG_ERROR(Render_Vulkan, "Unable to save ubershader fallback keys to {}", path);
            return;
        }
    }
    FileUtil::Rename(temp, path);
}

Ubershader::Ubershader(const Instance& instance_, RenderManager& renderpass_cache_,
                       Common::ThreadWorker& shader_workers_, std::size_t num_workers)
    : instance{instance_}, renderpass_cache{renderpass_cache_}, shader_workers{shader_workers_} {
    if (const char* force = std::getenv("AZAHAR_UBERSHADER_FORCE")) {
        const std::string_view value{force};
        if (value == "1") {
            mode = Mode::Force;
        } else if (value == "cpu") {
            mode = Mode::ForceCpu;
        }
    } else if (Settings::values.async_shader_compilation.GetValue() &&
               Settings::values.ubershader_fallback.GetValue()) {
        mode = Mode::Fallback;
    }
    if (mode == Mode::Fallback || mode == Mode::Force) {
        workers = std::make_unique<Common::ThreadWorker>(num_workers, "Ubershader workers");
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

void Ubershader::Start(vk::PipelineLayout layout_, Shader* vertex_shader_,
                       const VertexLayout& software_layout_, std::string keys_dir_, u64 title_id) {
    layout = layout_;
    vertex_shader = vertex_shader_;
    software_layout = software_layout_;
    keys_dir = std::move(keys_dir_);
    SwitchTitle(title_id);
}

void Ubershader::SwitchTitle(u64 title_id_) {
    if (mode != Mode::Fallback) {
        return;
    }
    title_id = title_id_;
    saved_keys.clear();
    std::vector<std::pair<FallbackKey, const char*>> keys;
    if (!keys_dir.empty()) {
        const auto saved = ReadFallbackKeys(KeysPath()).value_or(std::vector<FallbackKey>{});
        for (const FallbackKey& key : saved) {
            saved_keys.insert(key.Hash());
            keys.emplace_back(key, "disk");
        }
    }
    // The seeds go in every time, since a first boot that stops early saves only part of its set.
    for (const Seed& seed : Seeds) {
        StaticPipelineInfo state{};
        state.blending.blend_enable = 1;
        state.blending.color_write_mask = 0xF;
        state.blending.value = seed.blend;
        state.attachments.color = VideoCore::PixelFormat::RGBA8;
        state.attachments.depth = seed.depth;
        keys.emplace_back(MakeFallbackKey(state, UberTex0::Texture2D,
                                          instance.NeedsLogicOpEmulation(),
                                          instance.IsExtendedDynamicStateSupported()),
                          "seed");
    }
    Prewarm(keys);
}

GraphicsPipeline* Ubershader::Fallback(const StaticPipelineInfo& state) {
    ASSERT(variant.has_value());
    const FallbackKey key =
        MakeFallbackKey(state, *variant, instance.NeedsLogicOpEmulation(),
                        instance.IsExtendedDynamicStateSupported());
    const auto [pipeline, created] = Emplace(key, "draw");
    if (created) {
        workers->QueueWork([pipeline] { pipeline->Build(); });
    }
    if (mode == Mode::Fallback && !keys_dir.empty() && saved_keys.insert(key.Hash()).second) {
        SaveFallbackKey(KeysPath(), key);
    }
    return pipeline;
}

Shader* Ubershader::VariantShader(UberTex0 tex0) {
    auto& shader = shaders[static_cast<std::size_t>(tex0)];
    if (!shader) {
        shader = std::make_unique<Shader>(instance);
        shader_workers.QueueWork([this, tex0, compiled = shader.get()] {
            const u64 started = PipelineStats::Enabled() ? PipelineStats::Now() : 0;
            const auto spirv = CompileUbershader(tex0);
            ASSERT_MSG(!spirv.empty(), "Failed to compile ubershader variant {}",
                       static_cast<u32>(tex0));
            compiled->module = CompileSPV(spirv, instance.GetDevice());
            if (PipelineStats::Enabled()) {
                PipelineStats::Ubershader(static_cast<u32>(tex0), PipelineStats::Now() - started,
                                          spirv.size() * sizeof(u32));
            }
            compiled->MarkDone();
        });
    }
    return shader.get();
}

std::pair<GraphicsPipeline*, bool> Ubershader::Emplace(const FallbackKey& key, const char* source) {
    auto [it, created] = fallbacks.try_emplace(key.Hash());
    if (!created) {
        return {it->second.get(), false};
    }
    PipelineInfo info{};
    info.state.shader_ids = {0, UbershaderId + static_cast<u64>(key.variant), 0};
    info.state.blending = key.blending;
    info.state.attachments = key.attachments;
    info.state.vertex_layout = software_layout;
    info.state.rasterization = key.rasterization;
    info.state.depth_stencil = key.depth_stencil;
    it->second = std::make_unique<GraphicsPipeline>(
        instance, renderpass_cache, info, vk::PipelineCache{}, nullptr, layout,
        std::array<Shader*, MAX_SHADER_STAGES>{vertex_shader, VariantShader(key.variant), nullptr},
        workers.get());
    if (PipelineStats::Enabled()) {
        PipelineStats::FallbackPipeline(it->second->Hash(), static_cast<u32>(key.variant), source,
                                        key.blending.blend_enable, key.blending.value,
                                        key.blending.color_write_mask,
                                        static_cast<u32>(key.attachments.color),
                                        static_cast<u32>(key.attachments.depth));
    }
    return {it->second.get(), true};
}

// MoltenVK compiles a variant to its own Metal library for each set of attachments. On an M1 Pro
// that pipeline takes 400-450 ms, and one that reuses the library 100-140 ms. So the first
// pipeline of each variant and attachments builds alone, and the others wait to reuse its library.
void Ubershader::Prewarm(std::span<const std::pair<FallbackKey, const char*>> keys) {
    std::map<std::tuple<UberTex0, VideoCore::PixelFormat, VideoCore::PixelFormat>,
             std::vector<GraphicsPipeline*>>
        groups;
    for (const auto& [key, source] : keys) {
        const auto [pipeline, created] = Emplace(key, source);
        if (created) {
            groups[{key.variant, key.attachments.color, key.attachments.depth}].push_back(pipeline);
        }
    }
    for (const auto& [_, pipelines] : groups) {
        GraphicsPipeline* const first = pipelines.front();
        workers->QueueWork([first] { first->Build(); });
    }
    for (const auto& [_, pipelines] : groups) {
        GraphicsPipeline* const first = pipelines.front();
        for (std::size_t i = 1; i < pipelines.size(); i++) {
            workers->QueueWork([first, pipeline = pipelines[i]] {
                first->WaitDone();
                pipeline->Build();
            });
        }
    }
}

std::string Ubershader::KeysPath() const {
    return fmt::format("{}{}{:016X}_fallback.bin", keys_dir, DIR_SEP, title_id);
}

void PipelineCache::PrewarmFallbacks(const VertexLayout& software_layout) {
    if (!ubershader.FallbackOn() && !ubershader.Forced()) {
        return;
    }
    u64 title_id{};
    if (ubershader.FallbackOn()) {
        Core::System::GetInstance().GetAppLoader().ReadProgramId(title_id);
    }
    const bool disk = Settings::values.use_disk_shader_cache.GetValue() && EnsureDirectories();
    ubershader.Start(*pipeline_layout, &trivial_vertex_shader, software_layout,
                     disk ? GetTransferableDir() : std::string{}, title_id);
}

DrawPlan PipelineCache::PlanDraw(PipelineInfo& info, u32 vertices, bool async) {
    using Pica::Shader::Generator::ProgramType;
    for (u32 i = 0; i < MAX_SHADER_STAGES; i++) {
        info.state.shader_ids[i] = shader_hashes[i];
    }
    const bool hardware = current_shaders[ProgramType::VS] != &trivial_vertex_shader;
    const bool stats = PipelineStats::Enabled();
    const auto variant = ubershader.Variant();

    if (variant && ubershader.Forced()) {
        if (hardware) {
            if (stats) {
                PipelineStats::Redirect();
            }
            return {DrawOutcome::CpuVertexPath,
                    ubershader.ForcedCpu() ? nullptr : ubershader.Fallback(info.state)};
        }
        if (!ubershader.ForcedCpu()) {
            return {DrawOutcome::Fallback, ubershader.Fallback(info.state)};
        }
    }

    GraphicsPipeline* const pipeline = curr_disk_cache->GetPipeline(info);
    if (stats) {
        PipelineStats::Request(instance, info, pipeline->IsDone(), hardware, vertices, stats_tex0,
                               stats_tex0_on);
    }
    if (pipeline->IsDone()) {
        return {DrawOutcome::Bind, pipeline};
    }

    if (ubershader.FallbackOn()) {
        if (!variant) {
            if (stats) {
                PipelineStats::Ineligible();
            }
        } else if (GraphicsPipeline* const fallback = ubershader.Fallback(info.state);
                   fallback->IsDone()) {
            if (pipeline->TryBuild(false) && pipeline->IsDone()) {
                return {DrawOutcome::Bind, pipeline};
            }
            pipeline->QueueBuild();
            if (!hardware) {
                return {DrawOutcome::Fallback, fallback};
            }
            if (stats) {
                PipelineStats::Redirect();
            }
            return {DrawOutcome::CpuVertexPath, fallback};
        } else if (stats) {
            PipelineStats::FallbackMissing();
        }
    }

    const bool wait_built = !hardware || !async || vertices <= 6;
    if (!pipeline->TryBuild(wait_built)) {
        if (stats) {
            PipelineStats::Skip(info.state.OptimizedHash(instance), hardware, vertices);
        }
        return {DrawOutcome::Skip, nullptr};
    }
    return {pipeline->IsDone() ? DrawOutcome::Bind : DrawOutcome::Wait, pipeline};
}

} // namespace Vulkan
