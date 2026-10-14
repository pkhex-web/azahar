// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_set>
#include <fmt/format.h>
#if defined(__APPLE__)
#include <time.h>
#endif

#include "common/hash.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_pipeline_stats.h"
#include "video_core/shader/generator/pica_fs_config.h"

namespace Vulkan::PipelineStats {

namespace {

struct Sink {
    std::FILE* file = nullptr;
    bool markers = false;
    std::mutex mutex;
    std::unordered_set<u64> seen;
    std::atomic<u64> frame{0};
    std::atomic<u32> draws{0};
    std::atomic<u32> hardware{0};
    std::atomic<u32> pending{0};
    std::atomic<u32> skips{0};
    std::atomic<u32> waits{0};
    std::atomic<u64> wait_ns{0};

    Sink() {
        const char* path = std::getenv("AZAHAR_PIPELINE_STATS");
        if (path && *path) {
            file = std::fopen(path, "w");
        }
        markers = std::getenv("AZAHAR_PIPELINE_STATS_MARKERS") != nullptr;
    }
};

Sink& GetSink() {
    static Sink sink;
    return sink;
}

} // namespace

bool Enabled() {
    static const bool enabled = GetSink().file != nullptr;
    return enabled;
}

u64 Now() {
#if defined(__APPLE__)
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
#endif
}

u64 Frame() {
    return GetSink().frame.load(std::memory_order_relaxed);
}

void Write(const std::string& line) {
    Sink& sink = GetSink();
    std::scoped_lock lock{sink.mutex};
    std::fputs(line.c_str(), sink.file);
    std::fputc('\n', sink.file);
    std::fflush(sink.file);
}

void EndFrame() {
    if (!Enabled()) {
        return;
    }
    Sink& sink = GetSink();
    const u64 frame = sink.frame.fetch_add(1, std::memory_order_relaxed);
    Write(fmt::format(
        R"({{"e":"frame","t":{},"f":{},"draws":{},"hw":{},"pending":{},"skips":{},"waits":{},"wait_ns":{}}})",
        Now(), frame, sink.draws.exchange(0), sink.hardware.exchange(0), sink.pending.exchange(0),
        sink.skips.exchange(0), sink.waits.exchange(0), sink.wait_ns.exchange(0)));
}

void Request(const Instance& instance, const PipelineInfo& info, bool done, bool hardware,
             u32 vertices, u32 tex0, bool tex0_on) {
    Sink& sink = GetSink();
    sink.draws.fetch_add(1, std::memory_order_relaxed);
    if (hardware) {
        sink.hardware.fetch_add(1, std::memory_order_relaxed);
    }
    if (!done) {
        sink.pending.fetch_add(1, std::memory_order_relaxed);
    }
    const u64 pipeline = info.state.OptimizedHash(instance);
    bool first;
    {
        std::scoped_lock lock{sink.mutex};
        first = sink.seen.insert(pipeline).second;
    }
    if (!first && done) {
        return;
    }
    const auto& state = info.state;
    const u64 blend_attachments = Common::HashCombine(Common::ComputeStructHash64(state.blending),
                                                      Common::ComputeStructHash64(state.attachments));
    Write(fmt::format(
        R"({{"e":"req","t":{},"f":{},"pl":"{:016x}","full":"{:016x}","vs":"{:016x}","fs":"{:016x}","gs":"{:016x}","ba":"{:016x}","vl":"{:016x}","rd":"{:016x}","hw":{},"n":{},"built":{},"first":{},"tex0":{},"tex0_on":{},"be":{},"wm":{},"fwm":{},"lo":{},"bf":"{:08x}","color":{},"depth":{},"raster":{},"ds":"{:08x}"}})",
        Now(), Frame(), pipeline, info.Hash(), state.shader_ids[0], state.shader_ids[1],
        state.shader_ids[2], blend_attachments, Common::ComputeStructHash64(state.vertex_layout),
        Common::HashCombine(Common::ComputeStructHash64(state.rasterization),
                            Common::ComputeStructHash64(state.depth_stencil)),
        hardware, vertices, done, first, tex0, tex0_on, state.blending.blend_enable,
        state.blending.color_write_mask,
        const_cast<PipelineInfo&>(info).GetFinalColorWriteMask(instance),
        static_cast<u32>(state.blending.logic_op), state.blending.value,
        static_cast<u32>(state.attachments.color), static_cast<u32>(state.attachments.depth),
        state.rasterization.value, state.depth_stencil.value));
}

void Skip(u64 pipeline, bool hardware, u32 vertices) {
    GetSink().skips.fetch_add(1, std::memory_order_relaxed);
    Write(fmt::format(R"({{"e":"skip","t":{},"f":{},"pl":"{:016x}","hw":{},"n":{}}})", Now(),
                      Frame(), pipeline, hardware, vertices));
}

void Wait(u64 pipeline, u64 ns) {
    Sink& sink = GetSink();
    sink.waits.fetch_add(1, std::memory_order_relaxed);
    sink.wait_ns.fetch_add(ns, std::memory_order_relaxed);
    Write(fmt::format(R"({{"e":"wait","t":{},"f":{},"pl":"{:016x}","ns":{}}})", Now(), Frame(),
                      pipeline, ns));
}

void Building(u64 pipeline) {
    if (GetSink().markers) {
        std::fprintf(stderr, "[pipeline-stats] building %016llx\n",
                     static_cast<unsigned long long>(pipeline));
    }
}

void Built(u64 pipeline, u64 ns, bool fail_on_compile_required, bool succeeded) {
    if (GetSink().markers) {
        std::fprintf(stderr, "[pipeline-stats] built %016llx %llu ns\n",
                     static_cast<unsigned long long>(pipeline), static_cast<unsigned long long>(ns));
    }
    Write(fmt::format(
        R"({{"e":"built","t":{},"f":{},"pl":"{:016x}","ns":{},"fail_on_compile":{},"ok":{}}})",
        Now(), Frame(), pipeline, ns, fail_on_compile_required, succeeded));
}

void FragmentShader(const Pica::Shader::FSConfig& config, u64 hash, u64 ns, bool glsl) {
    Write(fmt::format(
        R"({{"e":"fs","t":{},"f":{},"fs":"{:016x}","ns":{},"glsl":{},"tex0":{},"lighting":{},"lights":{},"proctex":{},"fog":{},"shadow_rendering":{},"shadow_light":{},"minmax":{},"alpha_test":{}}})",
        Now(), Frame(), hash, ns, glsl, static_cast<u32>(config.texture.texture0_type.Value()),
        config.lighting.enable.Value(), config.lighting.src_num.Value(),
        config.proctex.enable.Value(), static_cast<u32>(config.texture.fog_mode.Value()),
        config.framebuffer.shadow_rendering.Value(), config.lighting.enable_shadow.Value(),
        config.framebuffer.requested_rgb_blend.RequiresMinMaxEmulation() ||
            config.framebuffer.requested_alpha_blend.RequiresMinMaxEmulation(),
        static_cast<u32>(config.framebuffer.alpha_test_func.Value())));
}

void VertexShader(u64 config, u64 program, u64 ns) {
    Write(fmt::format(R"({{"e":"vs","t":{},"f":{},"vs":"{:016x}","program":"{:016x}","ns":{}}})",
                      Now(), Frame(), config, program, ns));
}

} // namespace Vulkan::PipelineStats
