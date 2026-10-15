// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

// Renders fragment configs through the SPIR-V generator and through the ubershader on an offscreen
// target and compares the pixels. Needs a Vulkan driver: set AZAHAR_TEST_VULKAN_LIBRARY to its
// path (libMoltenVK.dylib on macOS), or the test skips.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "common/common_types.h"
#include "common/dynamic_library.h"
#include "common/vector_math.h"
#include "video_core/pica/regs_internal.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_ubershader.h"
#include "video_core/shader/generator/glsl_shader_gen.h"
#include "video_core/shader/generator/pica_fs_config.h"
#include "video_core/shader/generator/shader_uniforms.h"
#include "video_core/shader/generator/spv_fs_shader_gen.h"

namespace {

using Pica::FramebufferRegs;
using Pica::LightingRegs;
using Pica::TexturingRegs;
using Pica::Shader::FSConfig;
using Pica::Shader::Generator::FSUniformData;
using Source = TexturingRegs::TevStageConfig::Source;
using Operation = TexturingRegs::TevStageConfig::Operation;
using TextureType = TexturingRegs::TextureConfig::TextureType;
using Vulkan::UberConfig;
using Vulkan::UberTex0;

constexpr u32 Size = 64;
constexpr u32 Cells = 8;
constexpr vk::Format ColorFormat = vk::Format::eR8G8B8A8Unorm;
constexpr vk::Format DepthFormat = vk::Format::eD32Sfloat;
constexpr vk::DeviceSize UniformAlignment = 256;
constexpr vk::DeviceSize FsUniformOffset = 2 * UniformAlignment;
constexpr u32 LightingLutEntries = 256 * LightingRegs::NumLightingSampler;
constexpr u32 FogLutOffset = LightingLutEntries;

struct Vertex {
    Common::Vec4f position;
    Common::Vec4f color;
    Common::Vec2f tex_coord0;
    Common::Vec2f tex_coord1;
    Common::Vec2f tex_coord2;
    float tex_coord0_w;
    Common::Vec4f normquat;
    Common::Vec3f view;
};
static_assert(sizeof(Vertex) == 22 * sizeof(float));

struct Buffer {
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    void* data{};
};

struct Image {
    vk::UniqueImage image;
    vk::UniqueDeviceMemory memory;
    vk::UniqueImageView view;
};

struct Frame {
    std::vector<u8> color;
    std::vector<float> depth;
};

class Random {
public:
    explicit Random(u32 seed) : engine{seed} {}

    float Float(float min = 0.f, float max = 1.f) {
        return std::uniform_real_distribution<float>{min, max}(engine);
    }

    u32 Uint(u32 min, u32 max) {
        return std::uniform_int_distribution<u32>{min, max}(engine);
    }

    bool Bool() {
        return Uint(0, 1) == 1;
    }

    template <typename T, std::size_t N>
    T Pick(const std::array<T, N>& values) {
        return values[Uint(0, N - 1)];
    }

    Common::Vec3f Vec3(float min = 0.f, float max = 1.f) {
        return {Float(min, max), Float(min, max), Float(min, max)};
    }

    Common::Vec4f Vec4(float min = 0.f, float max = 1.f) {
        return {Float(min, max), Float(min, max), Float(min, max), Float(min, max)};
    }

    std::mt19937& Engine() {
        return engine;
    }

private:
    std::mt19937 engine;
};

/// The device, the offscreen target and every resource both shaders read.
class Renderer {
public:
    static std::unique_ptr<Renderer> Open(std::string& reason);

    /// Draws the grid with the given fragment shader and reads the target back. With
    /// base_level_only the textures have one level, so the LOD doesn't change what they return.
    Frame Draw(std::span<const u32> fragment_spirv, UberTex0 tex0, bool base_level_only,
               const FSUniformData& uniforms, const UberConfig& config,
               std::span<const Vertex> vertices);

private:
    Renderer() = default;

    void CreateDevice();
    void CreateTarget();
    void CreateResources();
    void CreateLayouts();
    u32 FindMemory(u32 type_bits, vk::MemoryPropertyFlags flags) const;
    Buffer MakeBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage);
    Image MakeImage(vk::Format format, u32 width, u32 height, u32 layers, u32 levels,
                    vk::ImageUsageFlags usage, vk::ImageViewType view_type,
                    vk::ImageAspectFlags aspect);
    void Upload(vk::Image image, u32 layers, u32 levels, std::span<const u8> data, u32 texel_size,
                u32 width, u32 height);
    void Submit(const std::function<void(vk::CommandBuffer)>& record);
    vk::Pipeline GetPipeline(std::span<const u32> fragment_spirv);

    Common::DynamicLibrary library;
    vk::UniqueInstance instance;
    vk::PhysicalDevice physical_device;
    vk::UniqueDevice device;
    vk::Queue queue;
    u32 queue_family{};
    vk::UniqueCommandPool command_pool;

    Image color;
    Image depth;
    vk::UniqueRenderPass render_pass;
    vk::UniqueFramebuffer framebuffer;
    Buffer readback;

    Buffer uniforms;
    Buffer vertex_buffer;
    Buffer lut_lf;
    Buffer lut_rg;
    Buffer lut_rgba;
    vk::UniqueBufferView lut_lf_view;
    vk::UniqueBufferView lut_rg_view;
    vk::UniqueBufferView lut_rgba_view;
    std::array<Image, 3> textures_2d;
    std::array<vk::UniqueImageView, 3> base_level_views;
    Image texture_cube;
    Image texture_shadow;
    vk::UniqueSampler sampler;
    vk::UniqueSampler nearest_sampler;

    vk::UniqueDescriptorSetLayout buffer_layout;
    vk::UniqueDescriptorSetLayout texture_layout;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniqueDescriptorPool descriptor_pool;
    vk::DescriptorSet buffer_set;
    std::array<std::array<vk::DescriptorSet, Vulkan::NumUberTex0>, 2> texture_sets;

    vk::UniqueShaderModule vertex_shader;
    std::vector<vk::UniquePipeline> pipelines;
};

std::unique_ptr<Renderer> Renderer::Open(std::string& reason) {
#ifdef __APPLE__
    // Metal's fast math contracts the two shaders' arithmetic differently, which moves a few
    // pixels by a rounding step and hides real differences.
    setenv("MVK_CONFIG_FAST_MATH_ENABLED", "0", 0);
#endif

    std::vector<std::string> names;
    if (const char* path = std::getenv("AZAHAR_TEST_VULKAN_LIBRARY")) {
        names.emplace_back(path);
    }
    names.push_back(Common::DynamicLibrary::GetLibraryName("vulkan", 1));
    names.push_back(Common::DynamicLibrary::GetLibraryName("vulkan"));
    names.push_back(Common::DynamicLibrary::GetLibraryName("MoltenVK"));

    std::unique_ptr<Renderer> renderer{new Renderer};
    for (const auto& name : names) {
        if (renderer->library.Load(name)) {
            break;
        }
    }
    if (!renderer->library.IsLoaded()) {
        reason = "no Vulkan library; set AZAHAR_TEST_VULKAN_LIBRARY";
        return nullptr;
    }
    const auto get_instance_proc_addr =
        renderer->library.GetSymbol<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    if (!get_instance_proc_addr) {
        reason = "the Vulkan library has no vkGetInstanceProcAddr";
        return nullptr;
    }
    VULKAN_HPP_DEFAULT_DISPATCHER.init(get_instance_proc_addr);
    try {
        renderer->CreateDevice();
    } catch (const vk::SystemError& error) {
        reason = fmt::format("no Vulkan device: {}", error.what());
        return nullptr;
    }
    if (!renderer->device) {
        reason = "no Vulkan device with a graphics queue";
        return nullptr;
    }
    renderer->CreateTarget();
    renderer->CreateLayouts();
    renderer->CreateResources();
    return renderer;
}

void Renderer::CreateDevice() {
    const auto has = [](const auto& properties, std::string_view name) {
        return std::ranges::any_of(properties, [name](const auto& property) {
            return std::string_view{property.extensionName} == name;
        });
    };

    std::vector<const char*> instance_extensions;
    vk::InstanceCreateFlags instance_flags{};
    const auto instance_properties = vk::enumerateInstanceExtensionProperties();
    if (has(instance_properties, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        instance_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        instance_flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
    }
    const vk::ApplicationInfo application = {
        .pApplicationName = "azahar-tests",
        .apiVersion = VK_API_VERSION_1_1,
    };
    instance = vk::createInstanceUnique({
        .flags = instance_flags,
        .pApplicationInfo = &application,
        .enabledExtensionCount = static_cast<u32>(instance_extensions.size()),
        .ppEnabledExtensionNames = instance_extensions.data(),
    });
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);

    const auto physical_devices = instance->enumeratePhysicalDevices();
    for (const auto& candidate : physical_devices) {
        const auto families = candidate.getQueueFamilyProperties();
        for (u32 i = 0; i < families.size(); i++) {
            if (families[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                physical_device = candidate;
                queue_family = i;
                break;
            }
        }
        if (physical_device) {
            break;
        }
    }
    if (!physical_device) {
        return;
    }

    std::vector<const char*> device_extensions;
    if (has(physical_device.enumerateDeviceExtensionProperties(), "VK_KHR_portability_subset")) {
        device_extensions.push_back("VK_KHR_portability_subset");
    }
    const float priority = 1.f;
    const vk::DeviceQueueCreateInfo queue_info = {
        .queueFamilyIndex = queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    device = physical_device.createDeviceUnique({
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .enabledExtensionCount = static_cast<u32>(device_extensions.size()),
        .ppEnabledExtensionNames = device_extensions.data(),
    });
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
    queue = device->getQueue(queue_family, 0);
    command_pool = device->createCommandPoolUnique({
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queue_family,
    });
}

u32 Renderer::FindMemory(u32 type_bits, vk::MemoryPropertyFlags flags) const {
    const auto properties = physical_device.getMemoryProperties();
    for (u32 i = 0; i < properties.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) && (properties.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    throw std::runtime_error("no suitable memory type");
}

Buffer Renderer::MakeBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage) {
    Buffer result;
    result.buffer = device->createBufferUnique({.size = size, .usage = usage});
    const auto requirements = device->getBufferMemoryRequirements(*result.buffer);
    result.memory = device->allocateMemoryUnique({
        .allocationSize = requirements.size,
        .memoryTypeIndex =
            FindMemory(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible |
                                                        vk::MemoryPropertyFlagBits::eHostCoherent),
    });
    device->bindBufferMemory(*result.buffer, *result.memory, 0);
    result.data = device->mapMemory(*result.memory, 0, size);
    return result;
}

Image Renderer::MakeImage(vk::Format format, u32 width, u32 height, u32 layers, u32 levels,
                          vk::ImageUsageFlags usage, vk::ImageViewType view_type,
                          vk::ImageAspectFlags aspect) {
    Image result;
    result.image = device->createImageUnique({
        .flags = view_type == vk::ImageViewType::eCube ? vk::ImageCreateFlagBits::eCubeCompatible
                                                       : vk::ImageCreateFlags{},
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = levels,
        .arrayLayers = layers,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    const auto requirements = device->getImageMemoryRequirements(*result.image);
    result.memory = device->allocateMemoryUnique({
        .allocationSize = requirements.size,
        .memoryTypeIndex =
            FindMemory(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
    });
    device->bindImageMemory(*result.image, *result.memory, 0);
    result.view = device->createImageViewUnique({
        .image = *result.image,
        .viewType = view_type,
        .format = format,
        .subresourceRange = {aspect, 0, levels, 0, layers},
    });
    return result;
}

void Renderer::Submit(const std::function<void(vk::CommandBuffer)>& record) {
    const auto buffers = device->allocateCommandBuffersUnique({
        .commandPool = *command_pool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    });
    const vk::CommandBuffer cmdbuf = *buffers[0];
    cmdbuf.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    record(cmdbuf);
    cmdbuf.end();
    const auto fence = device->createFenceUnique({});
    queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &cmdbuf}, *fence);
    static_cast<void>(device->waitForFences(*fence, true, UINT64_MAX));
}

// Data holds every layer's levels, level after level, each level half the size of the last.
void Renderer::Upload(vk::Image image, u32 layers, u32 levels, std::span<const u8> data,
                      u32 texel_size, u32 width, u32 height) {
    Buffer staging = MakeBuffer(data.size(), vk::BufferUsageFlagBits::eTransferSrc);
    std::memcpy(staging.data, data.data(), data.size());
    std::vector<vk::BufferImageCopy> copies;
    vk::DeviceSize offset = 0;
    for (u32 layer = 0; layer < layers; layer++) {
        for (u32 level = 0; level < levels; level++) {
            const u32 level_width = std::max(width >> level, 1u);
            const u32 level_height = std::max(height >> level, 1u);
            copies.push_back({
                .bufferOffset = offset,
                .imageSubresource = {vk::ImageAspectFlagBits::eColor, level, layer, 1},
                .imageExtent = {level_width, level_height, 1},
            });
            offset += level_width * level_height * texel_size;
        }
    }
    const vk::ImageSubresourceRange range = {vk::ImageAspectFlagBits::eColor, 0, levels, 0, layers};
    Submit([&](vk::CommandBuffer cmdbuf) {
        const vk::ImageMemoryBarrier to_transfer = {
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = range,
        };
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                               vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, to_transfer);
        cmdbuf.copyBufferToImage(*staging.buffer, image, vk::ImageLayout::eTransferDstOptimal,
                                 copies);
        const vk::ImageMemoryBarrier to_shader = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = range,
        };
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                               vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, to_shader);
    });
}

void Renderer::CreateTarget() {
    color =
        MakeImage(ColorFormat, Size, Size, 1, 1,
                  vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
                  vk::ImageViewType::e2D, vk::ImageAspectFlagBits::eColor);
    depth = MakeImage(DepthFormat, Size, Size, 1, 1,
                      vk::ImageUsageFlagBits::eDepthStencilAttachment |
                          vk::ImageUsageFlagBits::eTransferSrc,
                      vk::ImageViewType::e2D, vk::ImageAspectFlagBits::eDepth);

    const std::array attachments = {
        vk::AttachmentDescription{
            .format = ColorFormat,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::eTransferSrcOptimal,
        },
        vk::AttachmentDescription{
            .format = DepthFormat,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::eTransferSrcOptimal,
        },
    };
    const vk::AttachmentReference color_ref = {0, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::AttachmentReference depth_ref = {1, vk::ImageLayout::eDepthStencilAttachmentOptimal};
    const vk::SubpassDescription subpass = {
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_ref,
        .pDepthStencilAttachment = &depth_ref,
    };
    render_pass = device->createRenderPassUnique({
        .attachmentCount = static_cast<u32>(attachments.size()),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
    });
    const std::array views = {*color.view, *depth.view};
    framebuffer = device->createFramebufferUnique({
        .renderPass = *render_pass,
        .attachmentCount = static_cast<u32>(views.size()),
        .pAttachments = views.data(),
        .width = Size,
        .height = Size,
        .layers = 1,
    });
    readback = MakeBuffer(Size * Size * 8, vk::BufferUsageFlagBits::eTransferDst);
}

void Renderer::CreateLayouts() {
    const std::array buffer_bindings = {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eUniformBuffer, 1,
                                       vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eUniformBuffer, 1,
                                       vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eUniformBuffer, 1,
                                       vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{3, vk::DescriptorType::eUniformTexelBuffer, 1,
                                       vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{4, vk::DescriptorType::eUniformTexelBuffer, 1,
                                       vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{5, vk::DescriptorType::eUniformTexelBuffer, 1,
                                       vk::ShaderStageFlagBits::eFragment},
    };
    const std::array texture_bindings = {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eCombinedImageSampler, 1,
                                       vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eCombinedImageSampler, 1,
                                       vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eCombinedImageSampler, 1,
                                       vk::ShaderStageFlagBits::eFragment},
    };
    buffer_layout = device->createDescriptorSetLayoutUnique({
        .bindingCount = static_cast<u32>(buffer_bindings.size()),
        .pBindings = buffer_bindings.data(),
    });
    texture_layout = device->createDescriptorSetLayoutUnique({
        .bindingCount = static_cast<u32>(texture_bindings.size()),
        .pBindings = texture_bindings.data(),
    });
    const std::array set_layouts = {*buffer_layout, *texture_layout};
    pipeline_layout = device->createPipelineLayoutUnique({
        .setLayoutCount = static_cast<u32>(set_layouts.size()),
        .pSetLayouts = set_layouts.data(),
    });

    const std::array pool_sizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 3},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformTexelBuffer, 3},
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 18},
    };
    descriptor_pool = device->createDescriptorPoolUnique({
        .maxSets = 7,
        .poolSizeCount = static_cast<u32>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data(),
    });
    std::array<vk::DescriptorSetLayout, 7> allocate_layouts;
    allocate_layouts.fill(*texture_layout);
    allocate_layouts[0] = *buffer_layout;
    const auto sets = device->allocateDescriptorSets({
        .descriptorPool = *descriptor_pool,
        .descriptorSetCount = static_cast<u32>(allocate_layouts.size()),
        .pSetLayouts = allocate_layouts.data(),
    });
    buffer_set = sets[0];
    std::copy(sets.begin() + 1, sets.begin() + 4, texture_sets[0].begin());
    std::copy(sets.begin() + 4, sets.end(), texture_sets[1].begin());

    const auto vertex_spirv =
        Vulkan::CompileGLSL(Pica::Shader::Generator::GLSL::GenerateTrivialVertexShader(false, true),
                            vk::ShaderStageFlagBits::eVertex);
    vertex_shader = device->createShaderModuleUnique({
        .codeSize = vertex_spirv.size() * sizeof(u32),
        .pCode = vertex_spirv.data(),
    });
}

void Renderer::CreateResources() {
    Random random{1};

    uniforms =
        MakeBuffer(FsUniformOffset + UniformAlignment * 8, vk::BufferUsageFlagBits::eUniformBuffer);
    std::memset(uniforms.data, 0, FsUniformOffset);
    vertex_buffer =
        MakeBuffer(sizeof(Vertex) * Cells * Cells * 6, vk::BufferUsageFlagBits::eVertexBuffer);

    // Value and difference pairs, the way the rasterizer uploads the LUTs.
    const auto fill_pairs = [&random](Buffer& buffer, u32 count) {
        auto* pairs = static_cast<Common::Vec2f*>(buffer.data);
        for (u32 i = 0; i < count; i++) {
            pairs[i] = {random.Float(), random.Float(-0.05f, 0.05f)};
        }
    };
    lut_lf = MakeBuffer(sizeof(Common::Vec2f) * (LightingLutEntries + 128),
                        vk::BufferUsageFlagBits::eUniformTexelBuffer);
    fill_pairs(lut_lf, LightingLutEntries + 128);
    lut_rg =
        MakeBuffer(sizeof(Common::Vec2f) * 128 * 3, vk::BufferUsageFlagBits::eUniformTexelBuffer);
    fill_pairs(lut_rg, 128 * 3);
    lut_rgba =
        MakeBuffer(sizeof(Common::Vec4f) * 512, vk::BufferUsageFlagBits::eUniformTexelBuffer);
    auto* colors = static_cast<Common::Vec4f*>(lut_rgba.data);
    for (u32 i = 0; i < 256; i++) {
        colors[i] = random.Vec4();
    }
    for (u32 i = 0; i < 256; i++) {
        colors[256 + i] = colors[std::min(i + 1, 255u)] - colors[i];
    }
    const auto make_view = [this](const Buffer& buffer, vk::Format format) {
        return device->createBufferViewUnique({
            .buffer = *buffer.buffer,
            .format = format,
            .offset = 0,
            .range = VK_WHOLE_SIZE,
        });
    };
    lut_lf_view = make_view(lut_lf, vk::Format::eR32G32Sfloat);
    lut_rg_view = make_view(lut_rg, vk::Format::eR32G32Sfloat);
    lut_rgba_view = make_view(lut_rgba, vk::Format::eR32G32B32A32Sfloat);

    const auto random_bytes = [&random](std::size_t count) {
        std::vector<u8> bytes(count);
        for (auto& byte : bytes) {
            byte = static_cast<u8>(random.Uint(0, 255));
        }
        return bytes;
    };
    constexpr u32 TextureSize = 32;
    constexpr u32 Levels = 6;
    std::size_t mip_bytes = 0;
    for (u32 level = 0; level < Levels; level++) {
        mip_bytes += (TextureSize >> level) * (TextureSize >> level) * 4;
    }
    for (u32 i = 0; i < textures_2d.size(); i++) {
        auto& texture = textures_2d[i];
        texture = MakeImage(ColorFormat, TextureSize, TextureSize, 1, Levels,
                            vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
                            vk::ImageViewType::e2D, vk::ImageAspectFlagBits::eColor);
        Upload(*texture.image, 1, Levels, random_bytes(mip_bytes), 4, TextureSize, TextureSize);
        base_level_views[i] = device->createImageViewUnique({
            .image = *texture.image,
            .viewType = vk::ImageViewType::e2D,
            .format = ColorFormat,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        });
    }
    constexpr u32 CubeSize = 16;
    texture_cube =
        MakeImage(ColorFormat, CubeSize, CubeSize, 6, 1,
                  vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
                  vk::ImageViewType::eCube, vk::ImageAspectFlagBits::eColor);
    Upload(*texture_cube.image, 6, 1, random_bytes(CubeSize * CubeSize * 4 * 6), 4, CubeSize,
           CubeSize);
    constexpr u32 ShadowSize = 16;
    texture_shadow =
        MakeImage(vk::Format::eR32Uint, ShadowSize, ShadowSize, 1, 1,
                  vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
                  vk::ImageViewType::e2D, vk::ImageAspectFlagBits::eColor);
    Upload(*texture_shadow.image, 1, 1, random_bytes(ShadowSize * ShadowSize * 4), 4, ShadowSize,
           ShadowSize);

    sampler = device->createSamplerUnique({
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eRepeat,
        .addressModeV = vk::SamplerAddressMode::eMirroredRepeat,
        .addressModeW = vk::SamplerAddressMode::eRepeat,
        .maxLod = 16.f,
    });
    nearest_sampler = device->createSamplerUnique({
        .magFilter = vk::Filter::eNearest,
        .minFilter = vk::Filter::eNearest,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
    });

    const std::array buffer_infos = {
        vk::DescriptorBufferInfo{*uniforms.buffer, 0,
                                 sizeof(Pica::Shader::Generator::VSPicaUniformData)},
        vk::DescriptorBufferInfo{*uniforms.buffer, UniformAlignment,
                                 sizeof(Pica::Shader::Generator::VSUniformData)},
        vk::DescriptorBufferInfo{*uniforms.buffer, FsUniformOffset,
                                 sizeof(FSUniformData) + sizeof(UberConfig)},
    };
    const std::array texel_views = {*lut_lf_view, *lut_rg_view, *lut_rgba_view};
    std::vector<vk::WriteDescriptorSet> writes;
    for (u32 i = 0; i < buffer_infos.size(); i++) {
        writes.push_back({
            .dstSet = buffer_set,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &buffer_infos[i],
        });
    }
    for (u32 i = 0; i < texel_views.size(); i++) {
        writes.push_back({
            .dstSet = buffer_set,
            .dstBinding = 3 + i,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformTexelBuffer,
            .pTexelBufferView = &texel_views[i],
        });
    }
    const auto image_info = [](vk::ImageView view, vk::Sampler image_sampler) {
        return vk::DescriptorImageInfo{image_sampler, view,
                                       vk::ImageLayout::eShaderReadOnlyOptimal};
    };
    std::array<std::array<vk::DescriptorImageInfo, 5>, 2> infos;
    for (u32 base_level_only = 0; base_level_only < 2; base_level_only++) {
        const auto view_2d = [&](u32 unit) {
            return base_level_only ? *base_level_views[unit] : *textures_2d[unit].view;
        };
        auto& unit_infos = infos[base_level_only];
        unit_infos = {
            image_info(view_2d(0), *sampler),
            image_info(*texture_cube.view, *sampler),
            image_info(*texture_shadow.view, *nearest_sampler),
            image_info(view_2d(1), *sampler),
            image_info(view_2d(2), *sampler),
        };
        for (u32 variant = 0; variant < Vulkan::NumUberTex0; variant++) {
            const std::array bindings = {&unit_infos[variant], &unit_infos[3], &unit_infos[4]};
            for (u32 binding = 0; binding < bindings.size(); binding++) {
                writes.push_back({
                    .dstSet = texture_sets[base_level_only][variant],
                    .dstBinding = binding,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo = bindings[binding],
                });
            }
        }
    }
    device->updateDescriptorSets(writes, {});
}

vk::Pipeline Renderer::GetPipeline(std::span<const u32> fragment_spirv) {
    const auto fragment_shader = device->createShaderModuleUnique({
        .codeSize = fragment_spirv.size() * sizeof(u32),
        .pCode = fragment_spirv.data(),
    });
    const std::array stages = {
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = *vertex_shader,
            .pName = "main",
        },
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = *fragment_shader,
            .pName = "main",
        },
    };
    const vk::VertexInputBindingDescription binding = {0, static_cast<u32>(sizeof(Vertex))};
    constexpr std::array<u32, 8> components = {4, 4, 2, 2, 2, 1, 4, 3};
    std::array<vk::VertexInputAttributeDescription, 8> attributes;
    u32 offset = 0;
    for (u32 i = 0; i < attributes.size(); i++) {
        constexpr std::array formats = {vk::Format::eR32Sfloat, vk::Format::eR32G32Sfloat,
                                        vk::Format::eR32G32B32Sfloat,
                                        vk::Format::eR32G32B32A32Sfloat};
        attributes[i] = {i, 0, formats[components[i] - 1], offset};
        offset += components[i] * sizeof(float);
    }
    const vk::PipelineVertexInputStateCreateInfo vertex_input = {
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = static_cast<u32>(attributes.size()),
        .pVertexAttributeDescriptions = attributes.data(),
    };
    const vk::PipelineInputAssemblyStateCreateInfo input_assembly = {
        .topology = vk::PrimitiveTopology::eTriangleList,
    };
    const vk::Viewport viewport = {0.f, 0.f, static_cast<float>(Size), static_cast<float>(Size),
                                   0.f, 1.f};
    const vk::Rect2D scissor = {{0, 0}, {Size, Size}};
    const vk::PipelineViewportStateCreateInfo viewport_state = {
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };
    const vk::PipelineRasterizationStateCreateInfo rasterization = {
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .lineWidth = 1.f,
    };
    const vk::PipelineMultisampleStateCreateInfo multisample = {
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };
    const vk::PipelineDepthStencilStateCreateInfo depth_stencil = {
        .depthTestEnable = true,
        .depthWriteEnable = true,
        .depthCompareOp = vk::CompareOp::eAlways,
    };
    const vk::PipelineColorBlendAttachmentState blend_attachment = {
        .blendEnable = false,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };
    const vk::PipelineColorBlendStateCreateInfo blend = {
        .attachmentCount = 1,
        .pAttachments = &blend_attachment,
    };
    auto result =
        device->createGraphicsPipelineUnique({}, {
                                                     .stageCount = static_cast<u32>(stages.size()),
                                                     .pStages = stages.data(),
                                                     .pVertexInputState = &vertex_input,
                                                     .pInputAssemblyState = &input_assembly,
                                                     .pViewportState = &viewport_state,
                                                     .pRasterizationState = &rasterization,
                                                     .pMultisampleState = &multisample,
                                                     .pDepthStencilState = &depth_stencil,
                                                     .pColorBlendState = &blend,
                                                     .layout = *pipeline_layout,
                                                     .renderPass = *render_pass,
                                                 });
    if (result.result != vk::Result::eSuccess) {
        throw std::runtime_error("pipeline creation failed");
    }
    pipelines.push_back(std::move(result.value));
    return *pipelines.back();
}

Frame Renderer::Draw(std::span<const u32> fragment_spirv, UberTex0 tex0, bool base_level_only,
                     const FSUniformData& fs_uniforms, const UberConfig& config,
                     std::span<const Vertex> vertices) {
    const vk::Pipeline pipeline = GetPipeline(fragment_spirv);
    auto* fs_data = static_cast<u8*>(uniforms.data) + FsUniformOffset;
    std::memcpy(fs_data, &fs_uniforms, sizeof(fs_uniforms));
    std::memcpy(fs_data + sizeof(fs_uniforms), config.data(), sizeof(config));
    std::memcpy(vertex_buffer.data, vertices.data(), vertices.size_bytes());

    Submit([&](vk::CommandBuffer cmdbuf) {
        const std::array clear_values = {
            vk::ClearValue{.color =
                               vk::ClearColorValue{.float32 = std::array{0.2f, 0.4f, 0.6f, 0.8f}}},
            vk::ClearValue{.depthStencil =
                               vk::ClearDepthStencilValue{.depth = 0.25f, .stencil = 0}},
        };
        cmdbuf.beginRenderPass(
            {
                .renderPass = *render_pass,
                .framebuffer = *framebuffer,
                .renderArea = {{0, 0}, {Size, Size}},
                .clearValueCount = static_cast<u32>(clear_values.size()),
                .pClearValues = clear_values.data(),
            },
            vk::SubpassContents::eInline);
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        const std::array sets = {buffer_set, texture_sets[base_level_only][static_cast<u32>(tex0)]};
        cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline_layout, 0, sets, {});
        cmdbuf.bindVertexBuffers(0, *vertex_buffer.buffer, vk::DeviceSize{0});
        cmdbuf.draw(static_cast<u32>(vertices.size()), 1, 0, 0);
        cmdbuf.endRenderPass();
        cmdbuf.copyImageToBuffer(*color.image, vk::ImageLayout::eTransferSrcOptimal,
                                 *readback.buffer,
                                 vk::BufferImageCopy{
                                     .bufferOffset = 0,
                                     .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                                     .imageExtent = {Size, Size, 1},
                                 });
        cmdbuf.copyImageToBuffer(*depth.image, vk::ImageLayout::eTransferSrcOptimal,
                                 *readback.buffer,
                                 vk::BufferImageCopy{
                                     .bufferOffset = Size * Size * 4,
                                     .imageSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1},
                                     .imageExtent = {Size, Size, 1},
                                 });
        const vk::MemoryBarrier to_host = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eHostRead,
        };
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                               vk::PipelineStageFlagBits::eHost, {}, to_host, {}, {});
    });

    Frame frame;
    const auto* bytes = static_cast<const u8*>(readback.data);
    frame.color.assign(bytes, bytes + Size * Size * 4);
    frame.depth.resize(Size * Size);
    std::memcpy(frame.depth.data(), bytes + Size * Size * 4, Size * Size * sizeof(float));
    return frame;
}

Common::Vec4f RandomQuaternion(Random& random) {
    Common::Vec4f q = random.Vec4(-1.f, 1.f);
    q /= std::sqrt(q.Length2());
    return q;
}

/// A grid of triangles covering the target, each with its own attributes.
std::vector<Vertex> MakeVertices(Random& random) {
    std::vector<Vertex> vertices;
    const float cell = 2.f / Cells;
    for (u32 y = 0; y < Cells; y++) {
        for (u32 x = 0; x < Cells; x++) {
            const float left = -1.f + x * cell;
            const float top = -1.f + y * cell;
            const std::array<std::array<float, 2>, 6> corners = {{
                {left, top},
                {left + cell, top},
                {left, top + cell},
                {left + cell, top},
                {left + cell, top + cell},
                {left, top + cell},
            }};
            for (u32 triangle = 0; triangle < 2; triangle++) {
                const Common::Vec4f normquat = RandomQuaternion(random);
                const Common::Vec3f view = random.Vec3(-2.f, 2.f);
                for (u32 corner = 0; corner < 3; corner++) {
                    const auto [px, py] = corners[triangle * 3 + corner];
                    const float w = random.Float(0.5f, 2.f);
                    const float z = random.Float(0.05f, 0.95f);
                    Common::Vec4f quat = normquat + random.Vec4(-0.2f, 0.2f);
                    vertices.push_back({
                        .position = {px * w, py * w, -z * w, w},
                        .color = random.Vec4(),
                        .tex_coord0 = {random.Float(-0.25f, 1.25f), random.Float(-0.25f, 1.25f)},
                        .tex_coord1 = {random.Float(-0.25f, 1.25f), random.Float(-0.25f, 1.25f)},
                        .tex_coord2 = {random.Float(-0.25f, 1.25f), random.Float(-0.25f, 1.25f)},
                        .tex_coord0_w = random.Float(0.5f, 1.5f),
                        .normquat = quat,
                        .view = view + random.Vec3(-0.3f, 0.3f),
                    });
                }
            }
        }
    }
    return vertices;
}

FSUniformData MakeUniforms(Random& random) {
    FSUniformData data{};
    data.framebuffer_scale = 1;
    data.alphatest_ref = static_cast<int>(random.Uint(0, 255));
    data.depth_scale = random.Float(-1.f, -0.5f);
    data.depth_offset = random.Float(0.f, 0.25f);
    data.scissor_x1 = static_cast<int>(random.Uint(0, Size / 2));
    data.scissor_y1 = static_cast<int>(random.Uint(0, Size / 2));
    data.scissor_x2 = static_cast<int>(random.Uint(Size / 2, Size));
    data.scissor_y2 = static_cast<int>(random.Uint(Size / 2, Size));
    data.fog_lut_offset = FogLutOffset;
    data.proctex_noise_lut_offset = 0;
    data.proctex_color_map_offset = 128;
    data.proctex_alpha_map_offset = 256;
    data.proctex_lut_offset = 0;
    data.proctex_diff_lut_offset = 256;
    data.proctex_bias = random.Bool() ? 0.f : random.Float(0.5f, 4.f);
    data.shadow_texture_bias = static_cast<int>(random.Uint(0, 1000));
    for (u32 i = 0; i < LightingRegs::NumLightingSampler; i++) {
        data.lighting_lut_offset[i / 4][i % 4] = static_cast<int>(i * 256);
    }
    data.fog_color = random.Vec3();
    data.proctex_noise_f = {random.Float(0.1f, 2.f), random.Float(0.1f, 2.f)};
    data.proctex_noise_a = {random.Float(-0.5f, 0.5f), random.Float(-0.5f, 0.5f)};
    data.proctex_noise_p = {random.Float(0.f, 1.f), random.Float(0.f, 1.f)};
    data.lighting_global_ambient = random.Vec3(0.f, 0.3f);
    for (auto& light : data.light_src) {
        light.specular_0 = random.Vec3();
        light.specular_1 = random.Vec3();
        light.diffuse = random.Vec3();
        light.ambient = random.Vec3(0.f, 0.3f);
        light.position = random.Vec3(-3.f, 3.f);
        Common::Vec3f spot = random.Vec3(-1.f, 1.f);
        light.spot_direction = spot / std::sqrt(spot.Length2());
        light.dist_atten_bias = random.Float(0.f, 0.5f);
        light.dist_atten_scale = random.Float(0.f, 0.5f);
    }
    for (auto& color : data.const_color) {
        color = random.Vec4();
    }
    data.tev_combiner_buffer_color = random.Vec4();
    data.tex_lod_bias = random.Vec3(-1.f, 1.f);
    for (auto& color : data.tex_border_color) {
        color = random.Vec4();
    }
    data.blend_color = random.Vec4();
    return data;
}

Pica::Shader::Profile MakeProfile() {
    return Pica::Shader::Profile{
        .has_separable_shaders = true,
        .has_custom_border_color = false,
        .has_logic_op = false,
        .vk_use_spirv_generator = true,
        .is_vulkan = true,
    };
}

Pica::Shader::TevStageConfigRaw Stage(Source color1, Source color2, Source color3,
                                      Operation color_op, Source alpha1, Source alpha2,
                                      Source alpha3, Operation alpha_op) {
    TexturingRegs::TevStageConfig stage{};
    stage.color_source1.Assign(color1);
    stage.color_source2.Assign(color2);
    stage.color_source3.Assign(color3);
    stage.alpha_source1.Assign(alpha1);
    stage.alpha_source2.Assign(alpha2);
    stage.alpha_source3.Assign(alpha3);
    stage.color_op.Assign(color_op);
    stage.alpha_op.Assign(alpha_op);
    return {stage.sources_raw, stage.modifiers_raw, stage.ops_raw, stage.scales_raw};
}

Pica::Shader::TevStageConfigRaw PassThrough() {
    return Stage(Source::Previous, Source::Previous, Source::Previous, Operation::Replace,
                 Source::Previous, Source::Previous, Source::Previous, Operation::Replace);
}

Pica::Shader::TevStageConfigRaw Replace(Source source) {
    return Stage(source, source, source, Operation::Replace, source, source, source,
                 Operation::Replace);
}

/// Texture 0 times the primary colour, then pass-through stages; nothing else on.
FSConfig BaseConfig() {
    Pica::RegsInternal regs{};
    regs.lighting.disable.Assign(1);
    FSConfig config{regs};
    config.framebuffer.requested_logic_op = FramebufferRegs::LogicOp::Copy;
    config.framebuffer.depthmap_enable.Assign(Pica::RasterizerRegs::DepthBuffering::WBuffering);
    config.texture.texture0_type.Assign(TextureType::Texture2D);
    config.texture.tev_stages[0] =
        Stage(Source::Texture0, Source::PrimaryColor, Source::Previous, Operation::Modulate,
              Source::Texture0, Source::PrimaryColor, Source::Previous, Operation::Modulate);
    for (u32 i = 1; i < config.texture.tev_stages.size(); i++) {
        config.texture.tev_stages[i] = PassThrough();
    }
    return config;
}

/// Any raw stage, unknown sources, modifiers and operations included.
Pica::Shader::TevStageConfigRaw RandomStage(Random& random) {
    if (random.Uint(0, 3) == 0) {
        return PassThrough();
    }
    constexpr std::array sources = {
        Source::PrimaryColor,
        Source::PrimaryFragmentColor,
        Source::SecondaryFragmentColor,
        Source::Texture0,
        Source::Texture1,
        Source::Texture2,
        Source::Texture3,
        Source::PreviousBuffer,
        Source::Constant,
        Source::Previous,
        static_cast<Source>(7),
    };
    TexturingRegs::TevStageConfig stage{};
    stage.color_source1.Assign(random.Pick(sources));
    stage.color_source2.Assign(random.Pick(sources));
    stage.color_source3.Assign(random.Pick(sources));
    stage.alpha_source1.Assign(random.Pick(sources));
    stage.alpha_source2.Assign(random.Pick(sources));
    stage.alpha_source3.Assign(random.Pick(sources));
    stage.modifiers_raw = random.Engine()() & 0x777FFF;
    stage.color_op.Assign(static_cast<Operation>(random.Uint(0, 10)));
    stage.alpha_op.Assign(static_cast<Operation>(random.Uint(0, 10)));
    stage.color_scale.Assign(random.Uint(0, 3));
    stage.alpha_scale.Assign(random.Uint(0, 3));
    if (stage.color_op == Operation::Dot3_RGBA) {
        stage.sources_raw &= 0xFFF;
        stage.modifiers_raw &= 0xFFF;
        stage.ops_raw &= 0xF;
    }
    return {stage.sources_raw, stage.modifiers_raw, stage.ops_raw, stage.scales_raw};
}

void RandomTev(FSConfig& config, Random& random) {
    for (auto& stage : config.texture.tev_stages) {
        stage = RandomStage(random);
    }
    config.texture.combiner_buffer_input.Assign(random.Uint(0, 255));
}

void EnableProcTex(FSConfig& config, Random& random) {
    auto& proctex = config.proctex;
    proctex.enable.Assign(1);
    proctex.coord.Assign(random.Uint(0, 2));
    proctex.u_clamp.Assign(static_cast<TexturingRegs::ProcTexClamp>(random.Uint(0, 4)));
    proctex.v_clamp.Assign(static_cast<TexturingRegs::ProcTexClamp>(random.Uint(0, 4)));
    proctex.color_combiner.Assign(static_cast<TexturingRegs::ProcTexCombiner>(random.Uint(0, 9)));
    proctex.alpha_combiner.Assign(static_cast<TexturingRegs::ProcTexCombiner>(random.Uint(0, 9)));
    proctex.lut_filter.Assign(static_cast<TexturingRegs::ProcTexFilter>(random.Uint(0, 5)));
    proctex.separate_alpha.Assign(random.Bool());
    proctex.noise_enable.Assign(random.Bool());
    proctex.u_shift.Assign(static_cast<TexturingRegs::ProcTexShift>(random.Uint(0, 2)));
    proctex.v_shift.Assign(static_cast<TexturingRegs::ProcTexShift>(random.Uint(0, 2)));
    proctex.lut_width = 1 << random.Uint(4, 7);
    proctex.lut_offset0 = 0;
    proctex.lut_offset1 = proctex.lut_width;
    proctex.lut_offset2 = proctex.lut_offset1 + proctex.lut_width / 2;
    proctex.lut_offset3 = proctex.lut_offset2 + proctex.lut_width / 4;
    proctex.lod_min = static_cast<u16>(random.Uint(0, 3));
    proctex.lod_max = static_cast<u16>(random.Uint(proctex.lod_min, 7));
    config.texture.tev_stages[0] = Replace(Source::Texture3);
}

Pica::Shader::LutConfig RandomLut(Random& random) {
    Pica::Shader::LutConfig lut{};
    lut.enable.Assign(random.Uint(0, 3) != 0);
    lut.abs_input.Assign(random.Bool());
    lut.type.Assign(static_cast<LightingRegs::LightingLutInput>(random.Uint(0, 5)));
    lut.SetScale(random.Pick(std::array{0.25f, 0.5f, 1.f, 2.f, 4.f, 8.f}));
    return lut;
}

void EnableLighting(FSConfig& config, Random& random) {
    auto& lighting = config.lighting;
    lighting.enable.Assign(1);
    lighting.src_num.Assign(random.Uint(1, 8));
    lighting.config.Assign(random.Pick(
        std::array{LightingRegs::LightingConfig::Config0, LightingRegs::LightingConfig::Config1,
                   LightingRegs::LightingConfig::Config2, LightingRegs::LightingConfig::Config3,
                   LightingRegs::LightingConfig::Config4, LightingRegs::LightingConfig::Config5,
                   LightingRegs::LightingConfig::Config6, LightingRegs::LightingConfig::Config7}));
    lighting.clamp_highlights.Assign(random.Bool());
    lighting.enable_primary_alpha.Assign(random.Bool());
    lighting.enable_secondary_alpha.Assign(random.Bool());
    lighting.lut_d0 = RandomLut(random);
    lighting.lut_d1 = RandomLut(random);
    lighting.lut_sp = RandomLut(random);
    lighting.lut_sp.enable.Assign(1);
    lighting.lut_fr = RandomLut(random);
    lighting.lut_rr = RandomLut(random);
    lighting.lut_rg = RandomLut(random);
    lighting.lut_rb = RandomLut(random);
    std::array<u16, 8> nums = {0, 1, 2, 3, 4, 5, 6, 7};
    std::shuffle(nums.begin(), nums.end(), random.Engine());
    for (u32 i = 0; i < lighting.lights.size(); i++) {
        auto& light = lighting.lights[i];
        light.raw = static_cast<u16>(random.Uint(0, 0x3FF));
        light.num.Assign(nums[i]);
    }
    config.texture.tev_stages[0] =
        Stage(Source::PrimaryFragmentColor, Source::SecondaryFragmentColor, Source::Previous,
              Operation::Add, Source::PrimaryFragmentColor, Source::SecondaryFragmentColor,
              Source::Previous, Operation::Modulate);
}

void EnableShadow(FSConfig& config, Random& random) {
    auto& lighting = config.lighting;
    lighting.enable_shadow.Assign(1);
    lighting.shadow_primary.Assign(random.Bool());
    lighting.shadow_secondary.Assign(random.Bool());
    lighting.shadow_invert.Assign(random.Bool());
    lighting.shadow_alpha.Assign(random.Bool());
    lighting.shadow_selector.Assign(random.Uint(0, 2));
    for (auto& light : lighting.lights) {
        light.shadow_enable.Assign(1);
    }
}

struct Difference {
    u32 color_pixels{};
    u32 max_color_step{};
    u32 depth_pixels{};
    float max_depth{};
    u32 drawn{};
};

Difference Compare(const Frame& a, const Frame& b) {
    Difference difference;
    for (u32 pixel = 0; pixel < Size * Size; pixel++) {
        u32 step = 0;
        for (u32 channel = 0; channel < 4; channel++) {
            const int x = a.color[pixel * 4 + channel];
            const int y = b.color[pixel * 4 + channel];
            step = std::max<u32>(step, static_cast<u32>(std::abs(x - y)));
        }
        if (step != 0) {
            difference.color_pixels++;
            difference.max_color_step = std::max(difference.max_color_step, step);
        }
        const float depth = std::abs(a.depth[pixel] - b.depth[pixel]);
        if (depth != 0.f) {
            difference.depth_pixels++;
            difference.max_depth = std::max(difference.max_depth, depth);
        }
        if (a.depth[pixel] != 0.25f) {
            difference.drawn++;
        }
    }
    return difference;
}

class Comparison {
public:
    explicit Comparison(Renderer& renderer_) : renderer{renderer_} {}

    /// Draws the config through both shaders and checks they agree.
    void Check(const std::string& name, const FSConfig& config, u32 seed,
               bool base_level_only = false) {
        const auto profile = MakeProfile();
        const auto variant = Vulkan::UbershaderVariant(config, {}, profile);
        INFO(name);
        REQUIRE(variant.has_value());

        FSConfig applied = config;
        applied.ApplyProfile(profile);
        const UberConfig words = Vulkan::PackUberConfig(applied);
        Random random{seed};
        const FSUniformData uniforms = MakeUniforms(random);
        const auto vertices = MakeVertices(random);

        const auto specialised =
            Pica::Shader::Generator::SPIRV::GenerateFragmentShader(config, profile);
        const Frame expected =
            renderer.Draw(specialised, *variant, base_level_only, uniforms, words, vertices);
        const Frame actual = renderer.Draw(Ubershader(*variant), *variant, base_level_only,
                                           uniforms, words, vertices);
        const Difference difference = Compare(expected, actual);
        INFO(fmt::format("{} px differ in colour (up to {} steps), {} px in depth (up to {}), {} "
                         "px drawn",
                         difference.color_pixels, difference.max_color_step,
                         difference.depth_pixels, difference.max_depth, difference.drawn));
        // Metal can still fuse a multiply and an add in one shader and not the other, which
        // moves the odd pixel by a step.
        CHECK(difference.max_color_step <= 1);
        CHECK(difference.color_pixels <= Size * Size / 1000);
        CHECK(difference.depth_pixels == 0);
    }

private:
    const std::vector<u32>& Ubershader(UberTex0 variant) {
        auto& spirv = ubershaders[static_cast<u32>(variant)];
        if (spirv.empty()) {
            spirv = Vulkan::CompileUbershader(variant);
            REQUIRE(!spirv.empty());
        }
        return spirv;
    }

    Renderer& renderer;
    std::array<std::vector<u32>, Vulkan::NumUberTex0> ubershaders;
};

Renderer* OpenRenderer() {
    static std::string reason;
    static const std::unique_ptr<Renderer> renderer = Renderer::Open(reason);
    if (!renderer) {
        SKIP(reason);
    }
    return renderer.get();
}

} // Anonymous namespace

TEST_CASE("Ubershader matches the SPIR-V generator", "[video_core][vulkan][ubershader]") {
    Comparison comparison{*OpenRenderer()};

    SECTION("TEV stages, unknown modifiers and operations included") {
        for (u32 seed = 0; seed < 24; seed++) {
            FSConfig config = BaseConfig();
            Random random{seed};
            RandomTev(config, random);
            comparison.Check(fmt::format("tev {}", seed), config, seed);
        }
    }

    SECTION("Alpha test") {
        for (u32 func = 0; func < 8; func++) {
            FSConfig config = BaseConfig();
            config.framebuffer.alpha_test_func.Assign(
                static_cast<FramebufferRegs::CompareFunc>(func));
            comparison.Check(fmt::format("alpha test {}", func), config, 100 + func);
        }
    }

    SECTION("Scissor, depth and fog") {
        for (u32 mode = 0; mode < 4; mode++) {
            for (u32 w_buffer = 0; w_buffer < 2; w_buffer++) {
                FSConfig config = BaseConfig();
                config.framebuffer.scissor_test_mode.Assign(
                    static_cast<Pica::RasterizerRegs::ScissorMode>(mode));
                config.framebuffer.depthmap_enable.Assign(
                    w_buffer ? Pica::RasterizerRegs::DepthBuffering::WBuffering
                             : Pica::RasterizerRegs::DepthBuffering::ZBuffering);
                config.texture.fog_mode.Assign(TexturingRegs::FogMode::Fog);
                config.texture.fog_flip.Assign(mode & 1);
                comparison.Check(fmt::format("scissor {} w-buffer {}", mode, w_buffer), config,
                                 200 + mode * 2 + w_buffer);
            }
        }
    }

    SECTION("Logic op emulation") {
        constexpr std::array ops = {FramebufferRegs::LogicOp::Clear, FramebufferRegs::LogicOp::Set,
                                    FramebufferRegs::LogicOp::Copy, FramebufferRegs::LogicOp::NoOp,
                                    FramebufferRegs::LogicOp::CopyInverted};
        for (const auto op : ops) {
            FSConfig config = BaseConfig();
            config.framebuffer.requested_logic_op = op;
            comparison.Check(fmt::format("logic op {}", static_cast<u32>(op)), config,
                             300 + static_cast<u32>(op));
        }
    }

    SECTION("Texture 0 types") {
        constexpr std::array types = {TextureType::Texture2D, TextureType::Projection2D,
                                      TextureType::TextureCube, TextureType::Shadow2D,
                                      TextureType::Disabled};
        for (const auto type : types) {
            for (u32 orthographic = 0; orthographic < 2; orthographic++) {
                FSConfig config = BaseConfig();
                config.texture.texture0_type.Assign(type);
                config.texture.shadow_texture_orthographic.Assign(orthographic);
                config.texture.texture2_use_coord1.Assign(orthographic);
                config.texture.tev_stages[1] =
                    Stage(Source::Previous, Source::Texture1, Source::Texture2,
                          Operation::MultiplyThenAdd, Source::Previous, Source::Texture2,
                          Source::Texture1, Operation::Lerp);
                comparison.Check(fmt::format("texture 0 type {} orthographic {}",
                                             static_cast<u32>(type), orthographic),
                                 config, 500 + static_cast<u32>(type) * 2 + orthographic);
            }
        }
    }

    // The generator samples inside the border branch, so the derivatives that pick a level are
    // undefined next to a border. These draws use one level to compare the border alone.
    SECTION("Border emulation") {
        constexpr std::array types = {TextureType::Texture2D, TextureType::Projection2D,
                                      TextureType::TextureCube, TextureType::Shadow2D};
        for (u32 unit = 0; unit < 3; unit++) {
            for (u32 axes = 1; axes < 4; axes++) {
                for (const auto type : types) {
                    if (unit != 0 && type != TextureType::Texture2D) {
                        continue;
                    }
                    FSConfig config = BaseConfig();
                    config.texture.texture0_type.Assign(type);
                    config.texture.tev_stages[0] =
                        Replace(static_cast<Source>(static_cast<u32>(Source::Texture0) + unit));
                    config.texture.texture2_use_coord1.Assign(axes == 2);
                    auto& wrap = config.texture.requested_wrap[unit];
                    wrap.s = (axes & 1) ? TexturingRegs::TextureConfig::ClampToBorder
                                        : TexturingRegs::TextureConfig::Repeat;
                    wrap.t = (axes & 2) ? TexturingRegs::TextureConfig::ClampToBorder
                                        : TexturingRegs::TextureConfig::Repeat;
                    comparison.Check(fmt::format("border unit {} axes {} type {}", unit, axes,
                                                 static_cast<u32>(type)),
                                     config, 400 + unit * 16 + axes * 4 + static_cast<u32>(type),
                                     true);
                }
            }
        }
    }

    SECTION("Procedural textures") {
        for (u32 seed = 0; seed < 48; seed++) {
            FSConfig config = BaseConfig();
            Random random{600 + seed};
            EnableProcTex(config, random);
            config.proctex.lut_filter.Assign(static_cast<TexturingRegs::ProcTexFilter>(seed % 6));
            comparison.Check(fmt::format("proctex {}", seed), config, 600 + seed);
        }
    }

    SECTION("Fragment lighting") {
        for (u32 seed = 0; seed < 32; seed++) {
            FSConfig config = BaseConfig();
            Random random{700 + seed};
            EnableLighting(config, random);
            comparison.Check(fmt::format("lighting {}", seed), config, 700 + seed);
        }
    }

    SECTION("Bump and tangent maps") {
        for (u32 seed = 0; seed < 12; seed++) {
            FSConfig config = BaseConfig();
            Random random{800 + seed};
            EnableLighting(config, random);
            const bool tangent = seed % 2 == 1;
            config.lighting.bump_mode.Assign(tangent ? LightingRegs::LightingBumpMode::TangentMap
                                                     : LightingRegs::LightingBumpMode::NormalMap);
            config.lighting.bump_selector.Assign(seed % 3);
            config.lighting.bump_renorm.Assign(seed % 4 < 2);
            if (tangent) {
                config.lighting.config.Assign(LightingRegs::LightingConfig::Config7);
                config.lighting.lut_d0.enable.Assign(1);
                config.lighting.lut_d0.type.Assign(LightingRegs::LightingLutInput::CP);
            }
            comparison.Check(fmt::format("bump {}", seed), config, 800 + seed);
        }
    }

    SECTION("Fresnel") {
        for (u32 seed = 0; seed < 8; seed++) {
            FSConfig config = BaseConfig();
            Random random{900 + seed};
            EnableLighting(config, random);
            config.lighting.config.Assign(LightingRegs::LightingConfig::Config7);
            config.lighting.lut_fr.enable.Assign(1);
            config.lighting.enable_primary_alpha.Assign(seed & 1);
            config.lighting.enable_secondary_alpha.Assign((seed >> 1) & 1 || seed == 0);
            comparison.Check(fmt::format("fresnel {}", seed), config, 900 + seed);
        }
    }

    SECTION("Shadow attenuation") {
        for (u32 seed = 0; seed < 12; seed++) {
            FSConfig config = BaseConfig();
            Random random{1000 + seed};
            EnableLighting(config, random);
            EnableShadow(config, random);
            if (seed % 4 == 3) {
                config.lighting.shadow_selector.Assign(3);
                EnableProcTex(config, random);
                config.texture.tev_stages[0] =
                    Stage(Source::PrimaryFragmentColor, Source::SecondaryFragmentColor,
                          Source::Previous, Operation::Add, Source::PrimaryFragmentColor,
                          Source::SecondaryFragmentColor, Source::Previous, Operation::Modulate);
            }
            comparison.Check(fmt::format("shadow {}", seed), config, 1000 + seed);
        }
    }
}
