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
#include <functional>
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <util_init.hpp>

struct buffer {
    buffer() : device(VK_NULL_HANDLE), buf(VK_NULL_HANDLE), mem(VK_NULL_HANDLE) {};
    buffer(sample_info &info, VkDeviceSize num_bytes) : buffer() {
        allocate(info, num_bytes);
    };

    void allocate(sample_info &info, VkDeviceSize num_bytes);
    void reset();

    VkDevice        device;
    VkBuffer        buf;
    VkDeviceMemory  mem;
};

struct alignas(16) float4 {
    float x;
    float y;
    float z;
    float w;
};

struct spv_map {
    struct sampler {
        sampler() : opencl_flags(0), binding(-1) {};

        int opencl_flags;
        int binding;
    };

    struct kernel {
        struct arg {
            arg() : binding(-1), offset(0) {};

            int binding;
            int offset;
        };

        kernel() : name(), descriptor_set(-1), args() {};

        std::string             name;
        int                     descriptor_set;
        std::vector<arg> args;
    };

    spv_map() : samplers(), kernels() {};

    std::vector<sampler>    samplers;
    std::vector<kernel>     kernels;
};

enum {
    CLK_ADDRESS_NONE = 0x0000,
    CLK_ADDRESS_CLAMP_TO_EDGE = 0x0002,
    CLK_ADDRESS_CLAMP = 0x0004,
    CLK_ADDRESS_REPEAT = 0x0006,
    CLK_ADDRESS_MIRRORED_REPEAT = 0x0008,
    CLK_ADDRESS_MASK = 0x000E,

    CLK_NORMALIZED_COORDS_FALSE = 0x0000,
    CLK_NORMALIZED_COORDS_TRUE = 0x0001,
    CLK_NORMALIZED_COORDS_MASK = 0x0001,

    CLK_FILTER_NEAREST = 0x0010,
    CLK_FILTER_LINEAR = 0x0020,
    CLK_FILTER_MASK = 0x0030
};

bool operator==(const float4& l, const float4& r) {
    return (l.w == r.w && l.x == r.x && l.y == r.y && l.z == r.z);
}

bool operator!=(const float4& l, const float4& r) {
    return !(l == r);
}

struct fill_kernel_scalar_args {
    int     inPitch;        // offset 0
    int     inDeviceFormat; // DevicePixelFormat offset 4
    int     inOffsetX;      // offset 8
    int     inOffsetY;      // offset 12
    int     inWidth;        // offset 16
    int     inHeight;       // offset 20
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
    return VK_FALSE;
}

std::string read_csv_field(std::istream& in) {
    std::string result;

    if (in.good()) {
        const bool is_quoted = (in.peek() == '"');

        if (is_quoted) {
            in.ignore(std::numeric_limits<std::streamsize>::max(), '"');
        }

        std::getline(in, result, is_quoted ? '"' : ',');

        if (is_quoted) {
            in.ignore(std::numeric_limits<std::streamsize>::max(), ',');
        }
    }

    return result;
}

spv_map create_spv_map(std::istream& in) {
    spv_map result;

    while (!in.eof()) {
        // read one line
        std::string line;
        std::getline(in, line);

        std::istringstream in_line(line);
        std::string key = read_csv_field(in_line);
        std::string value = read_csv_field(in_line);
        if ("sampler" == key) {
            auto s = result.samplers.insert(result.samplers.end(), spv_map::sampler());
            assert(s != result.samplers.end());

            s->opencl_flags = std::atoi(value.c_str());

            while (!in_line.eof()) {
                key = read_csv_field(in_line);
                value = read_csv_field(in_line);

                if ("descriptorSet" == key) {
                    // all samplers, if any, are documented to share descriptor set 0
                    const int ds = std::atoi(value.c_str());
                    assert(ds == 0);
                }
                else if ("binding" == key) {
                    s->binding = std::atoi(value.c_str());
                }
            }
        }
        else if ("kernel" == key) {
            auto kernel = std::find_if(result.kernels.begin(), result.kernels.end(), [&value](const spv_map::kernel& iter) { return iter.name == value; });
            if (kernel == result.kernels.end()) {
                kernel = result.kernels.insert(kernel, spv_map::kernel());
                kernel->name = value;
            }
            assert(kernel != result.kernels.end());

            auto ka = kernel->args.end();

            while (!in_line.eof()) {
                key = read_csv_field(in_line);
                value = read_csv_field(in_line);

                if ("argOrdinal" == key) {
                    assert(ka == kernel->args.end());

                    const int arg_index = std::atoi(value.c_str());

                    if (kernel->args.size() <= arg_index) {
                        ka = kernel->args.insert(ka, arg_index - kernel->args.size() + 1, spv_map::kernel::arg());
                    }
                    else {
                        ka = std::next(kernel->args.begin(), arg_index);
                    }

                    assert(ka != kernel->args.end());
                }
                else if ("descriptorSet" == key) {
                    const int ds = std::atoi(value.c_str());
                    if (-1 == kernel->descriptor_set) {
                        kernel->descriptor_set = ds;
                    }

                    // all args for a kernel are documented to share the same descriptor set
                    assert(ds == kernel->descriptor_set);
                }
                else if ("binding" == key) {
                    ka->binding = std::atoi(value.c_str());
                }
                else if ("offset" == key) {
                    ka->offset = std::atoi(value.c_str());
                }
            }
        }
    }

    return result;
}

spv_map create_spv_map(const char* spvmapFilename) {
    // Read the spvmap file into a string buffer
    std::FILE *spvmap_file = AndroidFopen(spvmapFilename, "rb");
    assert(spvmap_file != NULL);
    std::fseek(spvmap_file, 0, SEEK_END);
    std::string buffer(std::ftell(spvmap_file), ' ');
    std::fseek(spvmap_file, 0, SEEK_SET);
    std::fread(&buffer.front(), 1, buffer.length(), spvmap_file);
    std::fclose(spvmap_file);

    // parse the spvmap file contents
    std::istringstream in(buffer);
    return create_spv_map(in);
}

std::vector<int> count_kernel_bindings(const spv_map& spvMap) {
    std::vector<int> result;

    for (auto &k : spvMap.kernels) {
        auto max_arg = std::max_element(k.args.begin(), k.args.end(), [](const spv_map::kernel::arg& a, const spv_map::kernel::arg& b) {
            return a.binding < b.binding;
        });

        if (result.size() <= k.descriptor_set) {
            result.insert(result.end(), k.descriptor_set - result.size() + 1, 0);
        }

        result[k.descriptor_set] = max_arg->binding + 1;
    }

    if (0 < spvMap.samplers.size()) {
        // There should be no kernel bindings for descriptor set 0 if there are samplers in the
        // SPIR-V module.
        assert(result[0] == 0);

        // Remove the first element of the result (because it's a misnomer to say there are 0 kernel
        // bindings for the first set
        result.erase(result.begin());
    }

    return result;
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

void my_init_descriptor_pool(struct sample_info &info) {
    const VkDescriptorPoolSize type_count[] = {
            {
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // type
                4                                   // descriptorCount
            },
            {
                VK_DESCRIPTOR_TYPE_SAMPLER, // type
                4                           // descriptorCount
            }
    };

    VkDescriptorPoolCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    createInfo.maxSets = 2;
    createInfo.poolSizeCount = sizeof(type_count) / sizeof(type_count[0]);
    createInfo.pPoolSizes = type_count;

    VkResult U_ASSERT_ONLY res = vkCreateDescriptorPool(info.device, &createInfo, NULL, &info.desc_pool);
    assert(res == VK_SUCCESS);
}

void my_init_descriptor_set(struct sample_info &info) {
    VkDescriptorSetAllocateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    createInfo.descriptorPool = info.desc_pool;
    createInfo.descriptorSetCount = info.desc_layout.size();
    createInfo.pSetLayouts = info.desc_layout.data();

    info.desc_set.resize(createInfo.descriptorSetCount);
    VkResult U_ASSERT_ONLY res = vkAllocateDescriptorSets(info.device, &createInfo, info.desc_set.data());
    assert(res == VK_SUCCESS);
}

void update_descriptor_sets(struct sample_info &info, const std::vector<VkSampler>& samplers, const std::vector<buffer>& buffers) {
    VkWriteDescriptorSet baseWriteSet = {};
    baseWriteSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    baseWriteSet.descriptorCount = 1;

    std::vector<VkWriteDescriptorSet> writeSets;

    // Update the samplers

    VkDescriptorImageInfo baseImageInfo = {};
    std::vector<VkDescriptorImageInfo> imageInfo(samplers.size(), baseImageInfo);
    for (int i = 0; i < samplers.size(); ++i) imageInfo[i].sampler = samplers[i];

    baseWriteSet.dstSet = info.desc_set[0];
    baseWriteSet.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    writeSets.resize(samplers.size(), baseWriteSet);
    for (int i = 0; i < samplers.size(); ++i) {
        writeSets[i].dstBinding = i;
        writeSets[i].pImageInfo = &imageInfo[i];
    }

    // Update the buffers

    VkDescriptorBufferInfo baseBufferInfo = {};
    baseBufferInfo.offset = 0;
    baseBufferInfo.range = VK_WHOLE_SIZE;
    std::vector<VkDescriptorBufferInfo> bufferInfo(buffers.size(), baseBufferInfo);
    for (int i = 0; i < buffers.size(); ++i) bufferInfo[i].buffer = buffers[i].buf;

    baseWriteSet.dstSet = info.desc_set[1];
    baseWriteSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    auto prevSize = writeSets.size();
    writeSets.resize(prevSize + buffers.size(), baseWriteSet);
    for (int i = 0; i < buffers.size(); ++i) {
        writeSets[i + prevSize].dstBinding = i;
        writeSets[i + prevSize].pBufferInfo = &bufferInfo[i];
    }

    vkUpdateDescriptorSets(info.device, writeSets.size(), writeSets.data(), 0, NULL);

}

VkShaderModule create_shader(struct sample_info &info, const char* spvFileName) {
    std::FILE* spv_file = AndroidFopen(spvFileName, "rb");
    assert(spv_file != NULL);

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

VkDescriptorSetLayout create_descriptor_set_layout(VkDevice device, int numBindings, VkDescriptorType descriptorType) {
    std::vector<VkDescriptorSetLayoutBinding> bindingSet;

    VkDescriptorSetLayoutBinding binding = {};
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binding.descriptorType = descriptorType;
    binding.descriptorCount = 1;

    for (int i = 0; i < numBindings; ++i) {
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

void buffer::allocate(sample_info &info, VkDeviceSize inNumBytes) {
    reset();

    device = info.device;

    // Allocate the buffer
    VkBufferCreateInfo buf_info = {};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buf_info.size = inNumBytes;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult U_ASSERT_ONLY res = vkCreateBuffer(device, &buf_info, NULL, &buf);
    assert(res == VK_SUCCESS);

    // Find out what we need in order to allocate memory for the buffer
    VkMemoryRequirements mem_reqs = {};
    vkGetBufferMemoryRequirements(device, buf, &mem_reqs);

    // Allocate memory for the buffer
    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    bool U_ASSERT_ONLY pass = memory_type_from_properties(info, mem_reqs.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                       &alloc_info.memoryTypeIndex);
    assert(pass && "No mappable, coherent memory");
    res = vkAllocateMemory(device, &alloc_info, NULL, &mem);
    assert(res == VK_SUCCESS);

    // Bind the memory to the buffer object
    res = vkBindBufferMemory(device, buf, mem, 0);
    assert(res == VK_SUCCESS);
}

void buffer::reset() {
    if (buf != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buf, NULL);
        buf = VK_NULL_HANDLE;
    }
    if (mem != VK_NULL_HANDLE) {
        vkFreeMemory(device, mem, NULL);
        mem = VK_NULL_HANDLE;
    }

    device = VK_NULL_HANDLE;
}

void init_compute_pipeline_layout(struct sample_info &info, const spv_map& spvMap) {
    info.desc_layout.clear();

    const int num_samplers = spvMap.samplers.size();
    if (0 < num_samplers) {
        info.desc_layout.push_back(create_descriptor_set_layout(info.device, num_samplers,
                                                                VK_DESCRIPTOR_TYPE_SAMPLER));
    }

    for (auto &nb : count_kernel_bindings(spvMap)) {
        info.desc_layout.push_back(create_descriptor_set_layout(info.device, nb,
                                                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER));
    };

    VkPipelineLayoutCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    createInfo.setLayoutCount = info.desc_layout.size();
    createInfo.pSetLayouts = createInfo.setLayoutCount ? info.desc_layout.data() : NULL;

    VkResult U_ASSERT_ONLY res = vkCreatePipelineLayout(info.device, &createInfo, NULL, &info.pipeline_layout);
    assert(res == VK_SUCCESS);
}

void submit_command(struct sample_info &info) {
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &info.cmd;

    VkResult U_ASSERT_ONLY res = vkQueueSubmit(info.graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
    assert(res == VK_SUCCESS);

}

void init_compute_pipeline(struct sample_info &info, VkShaderModule shaderModule, const char* entryName, int workGroupSizeX, int workGroupSizeY) {
    const unsigned int num_workgroup_sizes = 3;
    const int32_t workGroupSizes[num_workgroup_sizes] = { workGroupSizeX, workGroupSizeY, 1 };
    const VkSpecializationMapEntry specializationEntries[num_workgroup_sizes] = {
            {
                    0,                          // specialization constant 0 - workgroup size X
                    0*sizeof(int32_t),          // offset - start of workGroupSizes array
                    sizeof(workGroupSizes[0])   // sizeof the first element
            },
            {
                    1,                          // specialization constant 1 - workgroup size Y
                    1*sizeof(int32_t),            // offset - one element into the array
                    sizeof(workGroupSizes[1])   // sizeof the second element
            },
            {
                    2,                          // specialization constant 2 - workgroup size Z
                    2*sizeof(int32_t),          // offset - two elements into the array
                    sizeof(workGroupSizes[2])   // sizeof the second element
            }
    };
    VkSpecializationInfo specializationInfo = {};
    specializationInfo.mapEntryCount = num_workgroup_sizes;
    specializationInfo.pMapEntries = specializationEntries;
    specializationInfo.dataSize = sizeof(workGroupSizes);
    specializationInfo.pData = workGroupSizes;

    VkComputePipelineCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    createInfo.layout = info.pipeline_layout;

    createInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    createInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    createInfo.stage.module = shaderModule;
    createInfo.stage.pName = entryName;
    createInfo.stage.pSpecializationInfo = &specializationInfo;

    VkResult U_ASSERT_ONLY res = vkCreateComputePipelines(info.device, VK_NULL_HANDLE, 1, &createInfo, NULL, &info.pipeline);
    assert(res == VK_SUCCESS);
}

VkSampler create_compatible_sampler(VkDevice device, int opencl_flags) {
    typedef std::pair<int,VkSamplerAddressMode> address_mode_map;
    const address_mode_map address_mode_translator[] = {
            { CLK_ADDRESS_NONE, VK_SAMPLER_ADDRESS_MODE_REPEAT },
            { CLK_ADDRESS_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE},
            { CLK_ADDRESS_CLAMP, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER },
            { CLK_ADDRESS_REPEAT, VK_SAMPLER_ADDRESS_MODE_REPEAT },
            { CLK_ADDRESS_MIRRORED_REPEAT, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT }
    };

    const VkFilter filter = ((opencl_flags & CLK_FILTER_MASK) == CLK_FILTER_LINEAR ?
                             VK_FILTER_LINEAR :
                             VK_FILTER_NEAREST);
    const VkBool32 unnormalizedCoordinates = ((opencl_flags & CLK_NORMALIZED_COORDS_MASK) == CLK_NORMALIZED_COORDS_FALSE ? VK_FALSE : VK_TRUE);

    const auto found_map = std::find_if(std::begin(address_mode_translator), std::end(address_mode_translator), [&opencl_flags](const address_mode_map& am) {
        return (am.first == (opencl_flags & CLK_ADDRESS_MASK));
    });
    const VkSamplerAddressMode addressMode = (found_map == std::end(address_mode_translator) ? VK_SAMPLER_ADDRESS_MODE_REPEAT : found_map->second);

    VkSamplerCreateInfo samplerCreateInfo = {};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = filter;
    samplerCreateInfo.minFilter = filter ;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreateInfo.addressModeU = addressMode;
    samplerCreateInfo.addressModeV = samplerCreateInfo.addressModeU;
    samplerCreateInfo.addressModeW = samplerCreateInfo.addressModeU;
    samplerCreateInfo.anisotropyEnable = VK_FALSE;
    samplerCreateInfo.compareEnable = VK_FALSE;
    samplerCreateInfo.unnormalizedCoordinates = unnormalizedCoordinates;

    VkSampler result = VK_NULL_HANDLE;
    VkResult U_ASSERT_ONLY res = vkCreateSampler(device, &samplerCreateInfo, NULL, &result);
    assert(res == VK_SUCCESS);

    return result;
}

void memset_buffer(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, int value) {
    void* data = NULL;
    VkResult U_ASSERT_ONLY res = vkMapMemory(device, memory, offset, size, 0, &data);
    assert(res == VK_SUCCESS);
    memset(data, 0, size);
    vkUnmapMemory(device, memory);
}

void memcpy_buffer(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, const void* source) {
    void* data = NULL;
    VkResult U_ASSERT_ONLY res = vkMapMemory(device, memory, offset, size, 0, &data);
    assert(res == VK_SUCCESS);
    memcpy(data, source, size);
    vkUnmapMemory(device, memory);
}

void fill_command_buffer(struct sample_info &info, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) {
    execute_begin_command_buffer(info);
    vkCmdBindPipeline(info.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, info.pipeline);

    vkCmdBindDescriptorSets(info.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            info.pipeline_layout,
                            0,
                            info.desc_set.size(), info.desc_set.data(),
                            0, NULL);

    // NOTE: Current logic is inefficient, 1 work item per work group
    vkCmdDispatch(info.cmd, groupCountX, groupCountY, groupCountZ);
    execute_end_command_buffer(info);
}

void check_results(struct sample_info &info,
                   VkDeviceMemory   memory,
                   int              width,
                   int              height,
                   int              pitch,
                   const float4&    expected,
                   const char*      label,
                   bool             logIncorrect = false,
                   bool             logCorrect = false) {
    void* data = NULL;

    unsigned int num_correct_pixels = 0;
    unsigned int num_incorrect_pixels = 0;
    VkResult U_ASSERT_ONLY res = vkMapMemory(info.device, memory, 0, VK_WHOLE_SIZE, 0, &data);
    assert(res == VK_SUCCESS);
    {
        const float4* pixels = static_cast<const float4*>(data);

        const float4* row = pixels;
        for (int r = 0; r < height; ++r, row += pitch) {
            const float4* p = row;
            for (int c = 0; c < width; ++c, ++p) {
                const bool pixel_is_correct = (*p == expected);
                if (pixel_is_correct) {
                    ++num_correct_pixels;
                    if (logCorrect) {
                        LOGE("%s:  CORRECT pixels{row:%d, col%d} = {x=%f, y=%f, z=%f, w=%f}",
                             label, r, c, p->x, p->y, p->z, p->w);
                    }
                }
                else {
                    ++num_incorrect_pixels;
                    if (logIncorrect) {
                        LOGE("%s: INCORRECT pixels{row:%d, col%d} = {x=%f, y=%f, z=%f, w=%f}",
                             label, r, c, p->x, p->y, p->z, p->w);
                    }
                }
            }
        }
    }
    vkUnmapMemory(info.device, memory);

    LOGE("%s: Correct pixels=%d; Incorrect pixels=%d", label, num_correct_pixels, num_incorrect_pixels);
}

void run_kernel(sample_info&                    info,
                const char*                     module_name,
                const char*                     entry_point,
                const std::vector<VkSampler>&   samplers,
                const fill_kernel_scalar_args&  scalar_args) {
    init_command_buffer(info);

    const std::size_t buffer_size = scalar_args.inPitch * scalar_args.inHeight * sizeof(float4);

    const int workgroup_size_x = 32;
    const int workgroup_size_y = 32;
    const int num_workgroups_x = (scalar_args.inWidth + workgroup_size_x - 1) / workgroup_size_x;
    const int num_workgroups_y = (scalar_args.inHeight + workgroup_size_y - 1) / workgroup_size_y;

    // create memory buffers
    std::vector<buffer> buffers;
    buffers.push_back(buffer(info, buffer_size));
    buffers.push_back(buffer(info, sizeof(scalar_args)));

    // fill scalar args buffer with contents
    memcpy_buffer(info.device, buffers[1].mem, 0, sizeof(scalar_args), &scalar_args);

    // clear image buffer
    memset_buffer(info.device, buffers[0].mem, 0, buffer_size, 0);

    // We cannot use the shader support built into the sample framework because it is too tightly
    // tied to a graphics pipeline. Instead, track our compute shader externally.
    const VkShaderModule compute_shader = create_shader(info, module_name);

    // create the pipeline
    init_compute_pipeline(info, compute_shader, entry_point, workgroup_size_x, workgroup_size_y);

    my_init_descriptor_set(info);
    update_descriptor_sets(info, samplers, buffers);
    fill_command_buffer(info, num_workgroups_x, num_workgroups_y, 1);
    submit_command(info);

    vkQueueWaitIdle(info.graphics_queue);

    // examine result buffer contents
    std::string label(module_name);
    label += '/';
    label += entry_point;
    check_results(info, buffers[0].mem, scalar_args.inWidth, scalar_args.inHeight,
                  scalar_args.inPitch, scalar_args.inColor, label.c_str());

    VkResult U_ASSERT_ONLY res = vkFreeDescriptorSets(info.device,
                                                      info.desc_pool,
                                                      info.desc_set.size(),
                                                      info.desc_set.data());
    assert(res == VK_SUCCESS);
    info.desc_set.clear();

    destroy_pipeline(info);
    std::for_each(buffers.begin(), buffers.end(), std::mem_fun_ref(&buffer::reset));

    vkDestroyShaderModule(info.device, compute_shader, NULL);

    destroy_command_buffer(info);
}

int sample_main(int argc, char *argv[]) {
    const int buffer_height = 64;
    const int buffer_width = 64;
    const fill_kernel_scalar_args scalar_args = {
            buffer_width,               // inPitch
            1,                          // inDeviceFormat - kDevicePixelFormat_BGRA_4444_32f
            0,                          // inOffsetX
            0,                          // inOffsetY
            buffer_width,               // inWidth
            buffer_height,              // inHeight
            { 0.25f, 0.50f, 0.75f, 1.0f }  // inColor
    };

    const char* const spv_module_mapname = "fills.spvmap";

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
    init_device_queue(info);

    init_debug_report_callback(info, dbgFunc);

    init_command_pool(info);
    my_init_descriptor_pool(info);

    // Parse the spvmap file and create the pipeline layout from that description
    const spv_map shader_arg_map = create_spv_map(spv_module_mapname);
    init_compute_pipeline_layout(info, shader_arg_map);

    // This sample presumes that all OpenCL C kernels were compiled with the same samplermap file,
    // whose contents and order are statically known to the application. Thus, the app can create
    // a set of compatible samplers thusly.
    const int sampler_flags[] = {
            CLK_ADDRESS_CLAMP_TO_EDGE   | CLK_FILTER_LINEAR     | CLK_NORMALIZED_COORDS_FALSE,
            CLK_ADDRESS_CLAMP_TO_EDGE   | CLK_FILTER_NEAREST    | CLK_NORMALIZED_COORDS_FALSE,
            CLK_ADDRESS_NONE            | CLK_FILTER_NEAREST    | CLK_NORMALIZED_COORDS_FALSE,
            CLK_ADDRESS_CLAMP_TO_EDGE   | CLK_FILTER_LINEAR     | CLK_NORMALIZED_COORDS_TRUE
    };
    std::vector<VkSampler> samplers;
    std::transform(std::begin(sampler_flags), std::end(sampler_flags),
                   std::back_inserter(samplers),
                   std::bind(create_compatible_sampler, info.device, std::placeholders::_1));

    // run one kernel
    run_kernel(info, "fills_glsl.spv", "main", samplers, scalar_args);
    run_kernel(info, "fills.spv", "FillWithColorKernel", samplers, scalar_args);

    //
    // Clean up
    //

    // Cannot use the descriptor set and pipeline layout destruction built into the sample framework
    // because it is too tightly tied to the graphics pipeline (e.g. hard-coding the number of
    // descriptor set layouts).
    std::for_each(info.desc_layout.begin(), info.desc_layout.end(), std::bind(vkDestroyDescriptorSetLayout, info.device, std::placeholders::_1, nullptr));
    vkDestroyPipelineLayout(info.device, info.pipeline_layout, NULL);

    std::for_each(samplers.begin(), samplers.end(), std::bind(vkDestroySampler, info.device, std::placeholders::_1, nullptr));

    // Cannot use the shader module desctruction built into the sampel framework because it is too
    // tightly tied to the graphics pipeline (e.g. hard-coding the number and type of shaders).

    destroy_descriptor_pool(info);
    destroy_command_pool(info);
    destroy_debug_report_callback(info);
    destroy_device(info);
    destroy_instance(info);

    LOGI("Complete!");

    return 0;
}
