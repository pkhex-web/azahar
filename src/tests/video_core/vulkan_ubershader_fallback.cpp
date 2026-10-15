// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <filesystem>
#include <random>
#include <string>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "common/file_util.h"
#include "video_core/renderer_vulkan/vk_ubershader.h"

namespace {

using Vulkan::FallbackKey;
using Vulkan::MakeFallbackKey;
using Vulkan::StaticPipelineInfo;
using Vulkan::UberTex0;
using BlendFactor = Pica::FramebufferRegs::BlendFactor;
using LogicOp = Pica::FramebufferRegs::LogicOp;
using Topology = Pica::PipelineRegs::TriangleTopology;
using CullMode = Pica::RasterizerRegs::CullMode;

StaticPipelineInfo Draw() {
    StaticPipelineInfo state{};
    state.shader_ids = {0x1111, 0x2222, 0};
    state.blending.blend_enable = 1;
    state.blending.color_write_mask = 0xF;
    state.blending.logic_op = LogicOp::Copy;
    state.blending.src_color_blend_factor.Assign(BlendFactor::SourceAlpha);
    state.blending.dst_color_blend_factor.Assign(BlendFactor::OneMinusSourceAlpha);
    state.attachments.color = VideoCore::PixelFormat::RGBA8;
    state.attachments.depth = VideoCore::PixelFormat::D24S8;
    state.rasterization.topology.Assign(Topology::List);
    return state;
}

struct Device {
    bool logic_op_emulated;
    bool extended_dynamic_state;
};

constexpr Device AppleGpu{.logic_op_emulated = true, .extended_dynamic_state = true};
constexpr Device LogicOpGpu{.logic_op_emulated = false, .extended_dynamic_state = true};
constexpr Device NoDynamicStateGpu{.logic_op_emulated = true, .extended_dynamic_state = false};

FallbackKey Key(const StaticPipelineInfo& state, Device device = AppleGpu,
                UberTex0 variant = UberTex0::Texture2D) {
    return MakeFallbackKey(state, variant, device.logic_op_emulated,
                           device.extended_dynamic_state);
}

std::string TempPath(const char* name) {
    const auto path = std::filesystem::temp_directory_path() /
                      fmt::format("azahar-fallback-keys-{}-{}", name, std::random_device{}());
    std::filesystem::remove(path);
    return path.string();
}

} // Anonymous namespace

TEST_CASE("Fallback keys leave out what the ubershader pipeline doesn't depend on",
          "[video_core][ubershader]") {
    SECTION("the draw's shaders and vertex layout") {
        StaticPipelineInfo other = Draw();
        other.shader_ids = {0x3333, 0x4444, 0x5555};
        other.vertex_layout.attribute_count = 3;
        CHECK(Key(other) == Key(Draw()));
    }

    SECTION("blend factors when blending is off") {
        StaticPipelineInfo a = Draw();
        a.blending.blend_enable = 0;
        StaticPipelineInfo b = a;
        b.blending.src_color_blend_factor.Assign(BlendFactor::One);
        b.blending.dst_alpha_blend_factor.Assign(BlendFactor::DestColor);
        CHECK(Key(a) == Key(b));
        CHECK(Key(a, LogicOpGpu) == Key(b, LogicOpGpu));
    }

    SECTION("but not blend factors when blending is on") {
        StaticPipelineInfo other = Draw();
        other.blending.src_color_blend_factor.Assign(BlendFactor::One);
        CHECK(Key(other) != Key(Draw()));
    }

    SECTION("the logic op where the fragment shader emulates it") {
        StaticPipelineInfo a = Draw();
        a.blending.blend_enable = 0;
        a.blending.logic_op = LogicOp::Xor;
        StaticPipelineInfo b = a;
        b.blending.logic_op = LogicOp::And;
        CHECK(Key(a) == Key(b));
        CHECK(Key(a, LogicOpGpu) != Key(b, LogicOpGpu));
    }

    SECTION("the logic op when blending is on") {
        StaticPipelineInfo other = Draw();
        other.blending.logic_op = LogicOp::Xor;
        CHECK(Key(other, LogicOpGpu) == Key(Draw(), LogicOpGpu));
    }

    SECTION("NoOp, which becomes a write mask of 0 where the logic op is emulated") {
        StaticPipelineInfo noop = Draw();
        noop.blending.blend_enable = 0;
        noop.blending.logic_op = LogicOp::NoOp;
        StaticPipelineInfo masked = noop;
        masked.blending.logic_op = LogicOp::Copy;
        masked.blending.color_write_mask = 0;
        CHECK(Key(noop) == Key(masked));
        CHECK(Key(noop, LogicOpGpu) != Key(masked, LogicOpGpu));
    }

    SECTION("raster and depth state the device sets dynamically") {
        StaticPipelineInfo other = Draw();
        other.rasterization.cull_mode.Assign(CullMode::KeepCounterClockWise);
        other.depth_stencil.depth_test_enable.Assign(1);
        CHECK(Key(other) == Key(Draw()));
        CHECK(Key(other, NoDynamicStateGpu) != Key(Draw(), NoDynamicStateGpu));
    }

    SECTION("the topology, since fallback draws are triangle lists") {
        StaticPipelineInfo strip = Draw();
        strip.rasterization.topology.Assign(Topology::Strip);
        CHECK(Key(strip, NoDynamicStateGpu) == Key(Draw(), NoDynamicStateGpu));
    }
}

TEST_CASE("Fallback keys tell the ubershader variants and attachments apart",
          "[video_core][ubershader]") {
    CHECK(Key(Draw(), AppleGpu, UberTex0::TextureCube) != Key(Draw()));
    StaticPipelineInfo no_depth = Draw();
    no_depth.attachments.depth = VideoCore::PixelFormat::Invalid;
    CHECK(Key(no_depth) != Key(Draw()));
}

TEST_CASE("Saved fallback keys", "[video_core][ubershader]") {
    SECTION("a title without a file has none") {
        CHECK_FALSE(Vulkan::ReadFallbackKeys(TempPath("missing")).has_value());
    }

    SECTION("come back in the order they were saved") {
        const std::string path = TempPath("round-trip");
        StaticPipelineInfo cube = Draw();
        cube.attachments.depth = VideoCore::PixelFormat::Invalid;
        const FallbackKey first = Key(Draw());
        const FallbackKey second = Key(cube, AppleGpu, UberTex0::TextureCube);

        Vulkan::SaveFallbackKey(path, first);
        Vulkan::SaveFallbackKey(path, second);

        const auto keys = Vulkan::ReadFallbackKeys(path);
        REQUIRE(keys.has_value());
        REQUIRE(keys->size() == 2);
        CHECK((*keys)[0] == first);
        CHECK((*keys)[1] == second);
        std::filesystem::remove(path);
    }

    SECTION("leave out a key cut short") {
        const std::string path = TempPath("cut-short");
        Vulkan::SaveFallbackKey(path, Key(Draw()));
        Vulkan::SaveFallbackKey(path, Key(Draw(), AppleGpu, UberTex0::Shadow2D));
        std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);

        const auto keys = Vulkan::ReadFallbackKeys(path);
        REQUIRE(keys.has_value());
        CHECK(keys->size() == 1);
        std::filesystem::remove(path);
    }

    SECTION("start over when the file isn't ours") {
        const std::string path = TempPath("foreign");
        {
            FileUtil::IOFile file{path, "wb"};
            const std::string text = "not a list of fallback keys at all";
            file.WriteBytes(text.data(), text.size());
        }
        CHECK_FALSE(Vulkan::ReadFallbackKeys(path).has_value());

        Vulkan::SaveFallbackKey(path, Key(Draw()));

        const auto keys = Vulkan::ReadFallbackKeys(path);
        REQUIRE(keys.has_value());
        CHECK(keys->size() == 1);
        std::filesystem::remove(path);
    }
}
