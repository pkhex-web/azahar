// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <string>
#include "common/common_types.h"

namespace Pica::Shader {
struct FSConfig;
}

namespace Vulkan {

class Instance;
struct PipelineInfo;

namespace PipelineStats {

bool Enabled();

u64 Now();

u64 Frame();

void Write(const std::string& line);

void EndFrame();

void Request(const Instance& instance, const PipelineInfo& info, bool done, bool hardware,
             u32 vertices, u32 tex0, bool tex0_on);

void Skip(u64 pipeline, bool hardware, u32 vertices);

void Wait(u64 pipeline, u64 ns);

void Building(u64 pipeline);

void Built(u64 pipeline, u64 ns, bool fail_on_compile_required, bool succeeded);

void FragmentShader(const Pica::Shader::FSConfig& config, u64 hash, u64 ns, bool glsl);

void VertexShader(u64 config, u64 program, u64 ns);

} // namespace PipelineStats

} // namespace Vulkan
