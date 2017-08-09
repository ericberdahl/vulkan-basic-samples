/*
 * Vulkan Samples
 *
 * Copyright (C) 2015-2016 Valve Corporation
 * Copyright (C) 2015-2016 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
VULKAN_SAMPLE_SHORT_DESCRIPTION
create and destroy a Vulkan physical device
*/

/* This is part of the draw cube progression */

#include <algorithm>
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <util_init.hpp>

struct buffer {
    VkBuffer        buf;
    VkDeviceMemory  mem;
};

struct float4 {
    float x;
    float y;
    float z;
    float w;
};

struct fill_kernel_scalar_args {
    int     inPitch;        // offset 0
    int     inDeviceFormat; // DevicePixelFormat offset 4
    int     inOffsetX;      // offset 8
    int     inOffsetY;      // offset 12
    int     inWidth;        // offset 16
    int     inHeight;       // offset 20
    int     unused[2];      // offset 24, 28
    float4  inColor;        // offset 32
};
static_assert(0 == offsetof(fill_kernel_scalar_args, inPitch), "inPitch offset incorrect");
static_assert(4 == offsetof(fill_kernel_scalar_args, inDeviceFormat), "inDeviceFormat offset incorrect");
static_assert(8 == offsetof(fill_kernel_scalar_args, inOffsetX), "inOffsetX offset incorrect");
static_assert(12 == offsetof(fill_kernel_scalar_args, inOffsetY), "inOffsetY offset incorrect");
static_assert(16 == offsetof(fill_kernel_scalar_args, inWidth), "inWidth offset incorrect");
static_assert(20 == offsetof(fill_kernel_scalar_args, inHeight), "inHeight offset incorrect");
static_assert(32 == offsetof(fill_kernel_scalar_args, inColor), "inColor offset incorrect");

VKAPI_ATTR VkBool32 VKAPI_CALL dbgFunc(VkDebugReportFlagsEXT msgFlags, VkDebugReportObjectTypeEXT objType, uint64_t srcObject,
                                       size_t location, int32_t msgCode, const char *pLayerPrefix, const char *pMsg,
                                       void *pUserData) {
    std::ostringstream message;

    if (msgFlags & VK_DEBUG_REPORT_ERROR_BIT_EXT) {
        message << "ERROR: ";
    } else if (msgFlags & VK_DEBUG_REPORT_WARNING_BIT_EXT) {
        message << "WARNING: ";
    } else if (msgFlags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT) {
        message << "PERFORMANCE WARNING: ";
    } else if (msgFlags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT) {
        message << "INFO: ";
    } else if (msgFlags & VK_DEBUG_REPORT_DEBUG_BIT_EXT) {
        message << "DEBUG: ";
    }
    message << "[" << pLayerPrefix << "] Code " << msgCode << " : " << pMsg;

    std::cout << message.str() << std::endl;

    /*
     * false indicates that layer should not bail-out of an
     * API call that had validation failures. This may mean that the
     * app dies inside the driver due to invalid parameter(s).
     * That's what would happen without validation layers, so we'll
     * keep that behavior here.
     */
    return false;
}

void init_compute_queue_family_index(struct sample_info &info) {
    /* This routine simply finds a compute queue for a later vkCreateDevice.
     */
    bool found = false;
    for (unsigned int i = 0; i < info.queue_props.size(); i++) {
        if (info.queue_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            info.graphics_queue_family_index = i;
            found = true;
            break;
        }
    }
    assert(found);
}

VkShaderModule create_shader(struct sample_info &info, const char* spvFileName) {
    std::FILE* spv_file = AndroidFopen(spvFileName, "rb");

    std::fseek(spv_file, 0, SEEK_END);
    // Use vector of uint32_t to ensure alignment is satisfied.
    const auto num_bytes = std::ftell(spv_file);
    assert(0 == (num_bytes % sizeof(uint32_t)));
    const auto num_words = (num_bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    std::vector<uint32_t> spvModule(num_words);
    assert(num_bytes == (spvModule.size() * sizeof(uint32_t)));

    std::fseek(spv_file, 0, SEEK_SET);
    std::fread(spvModule.data(), 1, num_bytes, spv_file);

    std::fclose(spv_file);

    VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.pNext = NULL;
    shaderModuleCreateInfo.flags = 0;
    shaderModuleCreateInfo.codeSize = num_bytes;
    shaderModuleCreateInfo.pCode = spvModule.data();

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkResult U_ASSERT_ONLY res = vkCreateShaderModule(info.device, &shaderModuleCreateInfo, NULL, &shaderModule);
    assert(res == VK_SUCCESS);

    return shaderModule;
}

VkDescriptorSetLayout create_sampler_descriptor_set(VkDevice device, int numSamplers) {
    std::vector<VkDescriptorSetLayoutBinding> bindingSet;

    VkDescriptorSetLayoutBinding binding = {};
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    binding.descriptorCount = 1;

    for (int i = 0; i < numSamplers; ++i) {
        binding.binding = i;
        bindingSet.push_back(binding);
    }

    VkDescriptorSetLayoutCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    createInfo.bindingCount = bindingSet.size();
    createInfo.pBindings = createInfo.bindingCount ? bindingSet.data() : NULL;

    VkDescriptorSetLayout result = VK_NULL_HANDLE;
    VkResult U_ASSERT_ONLY res = vkCreateDescriptorSetLayout(device, &createInfo, NULL, &result);
    assert(res == VK_SUCCESS);

    return result;
}

VkDescriptorSetLayout create_buffer_descriptor_set(VkDevice device, int numBuffers) {
    std::vector<VkDescriptorSetLayoutBinding> bindingSet;

    VkDescriptorSetLayoutBinding binding = {};
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;

    for (int i = 0; i < numBuffers; ++i) {
        binding.binding = i;
        bindingSet.push_back(binding);
    }

    VkDescriptorSetLayoutCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    createInfo.bindingCount = bindingSet.size();
    createInfo.pBindings = createInfo.bindingCount ? bindingSet.data() : NULL;

    VkDescriptorSetLayout result = VK_NULL_HANDLE;
    VkResult U_ASSERT_ONLY res = vkCreateDescriptorSetLayout(device, &createInfo, NULL, &result);
    assert(res == VK_SUCCESS);

    return result;
}

buffer create_buffer(struct sample_info &info, VkDeviceSize num_bytes) {
    buffer result = {};

    // Allocate the buffer
    VkBufferCreateInfo buf_info = {};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    buf_info.size = num_bytes;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult U_ASSERT_ONLY res = vkCreateBuffer(info.device, &buf_info, NULL, &result.buf);
    assert(res == VK_SUCCESS);

    // Find out what we need in order to allocate memory for the buffer
    VkMemoryRequirements mem_reqs = {};
    vkGetBufferMemoryRequirements(info.device, result.buf, &mem_reqs);

    // Allocate memory for the buffer
    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    bool U_ASSERT_ONLY pass = memory_type_from_properties(info, mem_reqs.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                       &alloc_info.memoryTypeIndex);
    assert(pass && "No mappable, coherent memory");
    res = vkAllocateMemory(info.device, &alloc_info, NULL, &result.mem);
    assert(res == VK_SUCCESS);

    // Bind the memory to the buffer object
    res = vkBindBufferMemory(info.device, result.buf, result.mem, 0);
    assert(res == VK_SUCCESS);

    return result;
}

void destroy_buffer(struct sample_info &info, const buffer& buf) {
    vkDestroyBuffer(info.device, buf.buf, NULL);
    vkFreeMemory(info.device, buf.mem, NULL);
}

void init_compute_pipeline_layout(struct sample_info &info, int num_samplers, int num_buffers) {
    info.desc_layout.resize(0);
    info.desc_layout.push_back(create_sampler_descriptor_set(info.device, num_samplers));
    info.desc_layout.push_back(create_buffer_descriptor_set(info.device, num_buffers));

    VkPipelineLayoutCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    createInfo.setLayoutCount = info.desc_layout.size();
    createInfo.pSetLayouts = createInfo.setLayoutCount ? info.desc_layout.data() : NULL;

    VkResult U_ASSERT_ONLY res = vkCreatePipelineLayout(info.device, &createInfo, NULL, &info.pipeline_layout);
    assert(res == VK_SUCCESS);
}

void init_compute_pipeline(struct sample_info &info, VkShaderModule shaderModule, const char* entryName) {
    VkComputePipelineCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    createInfo.layout = info.pipeline_layout;

    createInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    createInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    createInfo.stage.module = shaderModule;
    createInfo.stage.pName = entryName;

    VkResult U_ASSERT_ONLY res = vkCreateComputePipelines(info.device, VK_NULL_HANDLE, 1, &createInfo, NULL, &info.pipeline);
    assert(res == VK_SUCCESS);
}

int sample_main(int argc, char *argv[]) {
    const int buffer_height = 32;
    const int buffer_width = 32;
    const fill_kernel_scalar_args scalar_args = {
            buffer_width,               // inPitch
            1,                          // inDeviceFormat - kDevicePixelFormat_BGRA_4444_32f
            0,                          // inOffsetX
            0,                          // inOffsetY
            buffer_width,               // inWidth
            buffer_height,              // inHeight
            {0,0},                      // unused
            { 1.0f, 0.0f, 0.0f, 1.0f }  // inColor
    };
    const std::size_t buffer_size = scalar_args.inPitch * scalar_args.inHeight * sizeof(float4);

    struct sample_info info = {};
    init_global_layer_properties(info);

    info.instance_extension_names.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    init_instance(info, "vulkansamples_device");

    init_enumerate_device(info);
    init_compute_queue_family_index(info);

    // The clspv solution we're using requires two Vulkan extensions to be enabled.
    info.device_extension_names.push_back("VK_KHR_storage_buffer_storage_class");
    info.device_extension_names.push_back("VK_KHR_variable_pointers");
    init_device(info);

    init_debug_report_callback(info, dbgFunc);

    // We cannot use the shader support built into the sample framework because it is too tightly
    // tied to a graphics pipeline. Instead, track our compute shader externally.
    const VkShaderModule compute_shader = create_shader(info, "fills-opt.spv");

    std::vector<VkSampler> samplers(4, VK_NULL_HANDLE);
    std::for_each(samplers.begin(), samplers.end(), [&info](VkSampler& s) { init_sampler(info, s); });

    // create memory buffers
    std::vector<buffer> buffers;
    buffers.push_back(create_buffer(info, buffer_size));
    buffers.push_back(create_buffer(info, sizeof(fill_kernel_scalar_args)));

    void* data = NULL;

    // fill scalar args buffer with contents
    VkResult U_ASSERT_ONLY res = vkMapMemory(info.device, buffers[1].mem, 0, VK_WHOLE_SIZE, 0, &data);
    assert(res == VK_SUCCESS);
    memcpy(data, &scalar_args, sizeof(scalar_args));
    vkUnmapMemory(info.device, buffers[1].mem);
    data = NULL;

    // clear image buffer
    res = vkMapMemory(info.device, buffers[0].mem, 0, VK_WHOLE_SIZE, 0, &data);
    assert(res == VK_SUCCESS);
    memset(data, 0, buffer_size);
    vkUnmapMemory(info.device, buffers[0].mem);
    data = NULL;

    // create the pipeline
    init_compute_pipeline_layout(info, samplers.size(), buffers.size());
    init_compute_pipeline(info, compute_shader, "FillWithColorKernel");

    // bind descriptor sets
    // invoke kernel

    // examine result buffer contents

    //
    // Clean up
    //

    destroy_pipeline(info);

    // Cannot use the descriptor set and pipeline layout destruction built into the sample framework
    // because it is too tightly tied to the graphics pipeline (e.g. hard-coding the number of
    // descriptor set layouts).
    std::for_each(info.desc_layout.begin(), info.desc_layout.end(), [&info](VkDescriptorSetLayout l) { vkDestroyDescriptorSetLayout(info.device, l, NULL); });
    vkDestroyPipelineLayout(info.device, info.pipeline_layout, NULL);

    std::for_each(buffers.begin(), buffers.end(), [&info](const buffer& b) { destroy_buffer(info, b); });
    std::for_each(samplers.begin(), samplers.end(), [&info](VkSampler s) { vkDestroySampler(info.device, s, NULL); });

    // Cannot use the shader module desctruction built into the sampel framework because it is too
    // tightly tied to the graphics pipeline (e.g. hard-coding the number and type of shaders).
    vkDestroyShaderModule(info.device, compute_shader, NULL);

    destroy_debug_report_callback(info);
    destroy_device(info);
    destroy_instance(info);

    LOGI("Complete!");

    return 0;
}
