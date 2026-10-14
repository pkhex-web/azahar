// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <unordered_set>
#include <fmt/format.h>

#include "common/settings.h"
#include "video_core/host_shaders/vulkan_fs_ubershader_frag.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_stats.h"
#include "video_core/renderer_vulkan/vk_shader_disk_cache.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_ubershader.h"
#include "video_core/shader/generator/pica_fs_config.h"

namespace Vulkan {

using namespace Pica::Shader::Generator;

namespace {

using Pica::Shader::FSConfig;
using TevStageConfig = Pica::TexturingRegs::TevStageConfig;
using TextureType = Pica::TexturingRegs::TextureConfig::TextureType;

static_assert(sizeof(FSConfig) == 68 * sizeof(u32));
static_assert(offsetof(FSConfig, texture) == 8 * sizeof(u32));
static_assert(offsetof(FSConfig, lighting) == 42 * sizeof(u32));
static_assert(offsetof(FSConfig, proctex) == 61 * sizeof(u32));
static_assert(offsetof(Pica::Shader::TextureConfig, tev_stages) == 4 * sizeof(u32));
static_assert(offsetof(Pica::Shader::LightConfig, lights) == 15 * sizeof(u32));
static_assert(offsetof(Pica::Shader::ProcTexConfig, lod_min) == 6 * sizeof(u32));

constexpr u32 TexturesUsedWord = 68;
constexpr u64 UberShaderId = 0x75626572'00000000ULL;

constexpr std::array<const char*, NumUberVariants> Preambles = {
    "",
    "#define TEX0_CUBE\n",
    "#define TEX0_SHADOW\n",
    "#define TEX0_ALIASED\n",
};

// The (blend, attachments) combinations Pokémon X draws with, read from the shader harness.
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

bool EnvOn(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] == '1';
}

bool IsPassThrough(const TevStageConfig& stage) {
    return stage.color_op == TevStageConfig::Operation::Replace &&
           stage.alpha_op == TevStageConfig::Operation::Replace &&
           stage.color_source1 == TevStageConfig::Source::Previous &&
           stage.alpha_source1 == TevStageConfig::Source::Previous &&
           stage.color_modifier1 == TevStageConfig::ColorModifier::SourceColor &&
           stage.alpha_modifier1 == TevStageConfig::AlphaModifier::SourceAlpha &&
           stage.GetColorMultiplier() == 1 && stage.GetAlphaMultiplier() == 1;
}

u32 TexturesUsed(const FSConfig& config) {
    u32 used = 0;
    const auto add = [&used](TevStageConfig::Source source) {
        const u32 value = static_cast<u32>(source);
        if (value >= 3 && value <= 6) {
            used |= 1u << (value - 3);
        }
    };
    for (u32 index = 0; index < config.texture.tev_stages.size(); index++) {
        const TevStageConfig stage = config.texture.tev_stages[index];
        if (IsPassThrough(stage)) {
            continue;
        }
        const auto source = [index](TevStageConfig::Source source, TevStageConfig::Source third) {
            return index == 0 && source == TevStageConfig::Source::Previous ? third : source;
        };
        add(source(stage.color_source1, stage.color_source3));
        add(source(stage.color_source2, stage.color_source3));
        add(stage.color_source3);
        if (stage.color_op != TevStageConfig::Operation::Dot3_RGBA) {
            add(source(stage.alpha_source1, stage.alpha_source3));
            add(source(stage.alpha_source2, stage.alpha_source3));
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

std::optional<UberTex0> Classify(const FSConfig& config, const Pica::Shader::UserConfig& user,
                                 const Pica::Shader::Profile& profile) {
    if (config.UsesSpirvIncompatibleConfig() ||
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

} // namespace

Ubershaders::Ubershaders(PipelineCache& parent_) : parent{parent_} {}

Ubershaders::~Ubershaders() = default;

void Ubershaders::Init() {
    forced = EnvOn("AZAHAR_UBERSHADER_FORCE");
    software_only = EnvOn("AZAHAR_UBERSHADER_SOFTWARE_ONLY");
    aliased = EnvOn("AZAHAR_UBERSHADER_ALIASED");
    prewarm_all = EnvOn("AZAHAR_UBERSHADER_PREWARM_ALL");
    early = EnvOn("AZAHAR_UBERSHADER_EARLY");
    active = forced ||
             (Settings::values.async_shader_compilation.GetValue() && EnvOn("AZAHAR_UBERSHADER"));
    if (!active) {
        return;
    }
    for (u32 i = 0; i < NumUberVariants; i++) {
        if (i == NumUberTex0 && !aliased) {
            continue;
        }
        shaders[i] = std::make_unique<Shader>(parent.instance);
        Shader* const shader = shaders[i].get();
        parent.shader_workers.QueueWork([this, shader, i] {
            const u64 started = PipelineStats::Now();
            const auto spirv = CompileGLSL(HostShaders::VULKAN_FS_UBERSHADER_FRAG,
                                           vk::ShaderStageFlagBits::eFragment, Preambles[i]);
            shader->module = CompileSPV(spirv, parent.instance.GetDevice());
            if (PipelineStats::Enabled()) {
                PipelineStats::UberShader(i, PipelineStats::Now() - started,
                                          spirv.size() * sizeof(u32));
            }
            shader->MarkDone();
        });
    }
}

void Ubershaders::SetSoftwareLayout(const VertexLayout& layout) {
    software_layout = layout;
    if (early) {
        PrewarmSeeds();
    }
}

bool Ubershaders::UseFragmentShader(const Pica::RegsInternal& regs,
                                    const Pica::Shader::UserConfig& user, u64 fs_hash) {
    if (!active) {
        return false;
    }
    const u64 hash = Common::HashCombine(fs_hash, static_cast<u64>(user.raw));
    if (config_valid && hash == config_fs_hash) {
        return false;
    }
    FSConfig fs_config{regs};
    fs_config.ApplyProfile(parent.profile);
    std::memcpy(config.data(), &fs_config, sizeof(fs_config));
    config[TexturesUsedWord] = TexturesUsed(fs_config);
    eligible = Classify(fs_config, user, parent.profile);
    config_fs_hash = hash;
    config_valid = true;
    return true;
}

StaticPipelineInfo Ubershaders::FallbackState(const StaticPipelineInfo& state,
                                              UberTex0 tex0) const {
    const Instance& instance = parent.instance;
    StaticPipelineInfo fallback{};
    fallback.shader_ids = {0, UberShaderId + Variant(tex0), 0};
    fallback.blending = state.blending;
    fallback.attachments = state.attachments;
    fallback.vertex_layout = software_layout;
    fallback.rasterization.topology.Assign(Pica::PipelineRegs::TriangleTopology::List);
    if (!instance.IsExtendedDynamicStateSupported()) {
        fallback.rasterization.cull_mode.Assign(state.rasterization.cull_mode);
        fallback.rasterization.flip_viewport.Assign(state.rasterization.flip_viewport);
        fallback.depth_stencil = state.depth_stencil;
    }

    BlendingState& blending = fallback.blending;
    const bool emulated = instance.NeedsLogicOpEmulation();
    if (!blending.blend_enable) {
        if (emulated && blending.logic_op == Pica::FramebufferRegs::LogicOp::NoOp) {
            blending.color_write_mask = 0;
        }
        blending.value = 0;
        if (emulated) {
            blending.logic_op = Pica::FramebufferRegs::LogicOp::Copy;
        }
    } else {
        blending.logic_op = Pica::FramebufferRegs::LogicOp::Copy;
    }
    return fallback;
}

u32 Ubershaders::Variant(UberTex0 tex0) const {
    return aliased && tex0 != UberTex0::Shadow2D ? NumUberTex0 : static_cast<u32>(tex0);
}

GraphicsPipeline* Ubershaders::GetState(const StaticPipelineInfo& state, const char* source,
                                        bool before_caches) {
    const u64 hash = state.OptimizedHash(parent.instance);
    auto [it, created] = pipelines.try_emplace(hash);
    if (created) {
        PipelineInfo info{};
        info.state = state;
        const u32 tex0 = static_cast<u32>(state.shader_ids[ProgramType::FS] - UberShaderId);
        const std::array<Shader*, MAX_SHADER_STAGES> stages = {
            &parent.trivial_vertex_shader, shaders[tex0].get(), nullptr};
        it->second = std::make_unique<GraphicsPipeline>(
            parent.instance, parent.renderpass_cache, info,
            before_caches ? vk::PipelineCache{} : *parent.driver_pipeline_cache,
            before_caches ? nullptr : &parent.worker_pipeline_cache_handles,
            *parent.pipeline_layout, stages, &parent.pipeline_workers);
        if (PipelineStats::Enabled()) {
            PipelineStats::FallbackPipeline(hash, tex0, source, state.blending.blend_enable,
                                            state.blending.value,
                                            state.blending.color_write_mask,
                                            static_cast<u32>(state.attachments.color),
                                            static_cast<u32>(state.attachments.depth));
        }
        it->second->QueueBuild();
    }
    return it->second.get();
}

GraphicsPipeline* Ubershaders::Get(const PipelineInfo& info, UberTex0 tex0) {
    return GetState(FallbackState(info.state, tex0), "draw");
}

void Ubershaders::Prewarm(const StaticPipelineInfo& state) {
    if (!active || state.blending.rgb_blend_emulation || state.blending.alpha_blend_emulation) {
        return;
    }
    GetState(FallbackState(state, UberTex0::Texture2D), "disk");
}

void Ubershaders::PrewarmSeeds() {
    if (!active) {
        return;
    }
    const bool early_seed = !parent.driver_pipeline_cache;
    for (const Seed& seed : Seeds) {
        StaticPipelineInfo state{};
        state.blending.blend_enable = 1;
        state.blending.color_write_mask = 0xF;
        state.blending.value = seed.blend;
        state.attachments.color = VideoCore::PixelFormat::RGBA8;
        state.attachments.depth = seed.depth;
        for (u32 tex0 = 0; tex0 < (prewarm_all ? NumUberTex0 : 1); tex0++) {
            GetState(FallbackState(state, static_cast<UberTex0>(tex0)), early_seed ? "early" : "seed",
                     early_seed);
        }
    }
}

bool PipelineCache::RedirectToFallback(PipelineInfo& info, u32 vertices) {
    if (!ubershaders.Active()) {
        return false;
    }
    for (u32 i = 0; i < MAX_SHADER_STAGES; i++) {
        info.state.shader_ids[i] = shader_hashes[i];
    }
    const auto tex0 = ubershaders.Eligible();
    GraphicsPipeline* const pipeline = curr_disk_cache->GetPipeline(info);
    if (!tex0 || (pipeline->IsDone() && !ubershaders.Forced())) {
        return false;
    }
    GraphicsPipeline* const fallback = ubershaders.Get(info, *tex0);
    const bool stats = PipelineStats::Enabled();
    if (!fallback->IsDone() && !ubershaders.Forced()) {
        if (stats) {
            PipelineStats::FallbackMissing();
        }
        return false;
    }
    if (!pipeline->IsDone()) {
        pipeline->QueueBuild();
    }
    if (stats) {
        PipelineStats::Request(instance, info, pipeline->IsDone(), true, vertices, stats_tex0,
                               stats_tex0_on);
        PipelineStats::Redirect();
    }
    ubershaders.SetRedirected(shader_hashes[ProgramType::FS]);
    return true;
}

GraphicsPipeline* PipelineCache::GetPipelineOrFallback(PipelineInfo& info, bool hardware,
                                                       u32 vertices) {
    const bool stats = PipelineStats::Enabled();
    const auto tex0 = ubershaders.Eligible();
    if (!hardware && ubershaders.TakeRedirected(shader_hashes[ProgramType::FS]) && tex0 &&
        !ubershaders.SoftwareOnly()) {
        if (stats) {
            PipelineStats::FallbackDraw();
        }
        return ubershaders.Get(info, *tex0);
    }
    GraphicsPipeline* const pipeline = curr_disk_cache->GetPipeline(info);
    if (stats) {
        PipelineStats::Request(instance, info, pipeline->IsDone(), hardware, vertices, stats_tex0,
                               stats_tex0_on);
    }
    if (!ubershaders.Active() || ubershaders.SoftwareOnly() ||
        (pipeline->IsDone() && !ubershaders.Forced())) {
        return pipeline;
    }
    if (!tex0) {
        if (stats && !pipeline->IsDone()) {
            PipelineStats::Ineligible();
        }
        return pipeline;
    }
    if (hardware) {
        return pipeline;
    }
    GraphicsPipeline* const fallback = ubershaders.Get(info, *tex0);
    if (!fallback->IsDone() && !ubershaders.Forced()) {
        if (stats) {
            PipelineStats::FallbackMissing();
        }
        return pipeline;
    }
    if (!pipeline->IsDone()) {
        pipeline->QueueBuild();
    }
    if (stats) {
        PipelineStats::FallbackDraw();
    }
    return fallback;
}

bool PipelineCache::DebugSkipDraw(const PipelineInfo& info, u32 vertices, bool accelerate) {
    static const std::unordered_set<u64> skipped = [] {
        std::unordered_set<u64> hashes;
        if (const char* list = std::getenv("AZAHAR_SKIP_FS")) {
            std::stringstream stream{list};
            std::string item;
            while (std::getline(stream, item, ',')) {
                if (!item.empty()) {
                    hashes.insert(std::stoull(item, nullptr, 16));
                }
            }
        }
        return hashes;
    }();
    static const u64 log_frame = [] {
        const char* frame = std::getenv("AZAHAR_LOG_DRAWS_FRAME");
        return frame ? std::stoull(frame) : ~0ULL;
    }();
    const u64 fs = shader_hashes[ProgramType::FS];
    const bool skip = skipped.contains(fs);
    if (PipelineStats::Enabled() && PipelineStats::Frame() == log_frame) {
        PipelineStats::Write(fmt::format(
            R"({{"e":"draw","f":{},"fs":"{:016x}","n":{},"accel":{},"be":{},"bf":"{:08x}","tex0":{},"skip":{}}})",
            log_frame, fs, vertices, accelerate, info.state.blending.blend_enable,
            info.state.blending.value, stats_tex0, skip));
    }
    return skip;
}

} // namespace Vulkan
