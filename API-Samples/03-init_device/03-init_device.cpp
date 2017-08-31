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

struct pipeline_layout {
    pipeline_layout() : device(VK_NULL_HANDLE), descriptors(), pipeline(VK_NULL_HANDLE) {};

    void    reset();

    VkDevice                            device;
    std::vector<VkDescriptorSetLayout>  descriptors;
    VkPipelineLayout                    pipeline;
};

struct buffer {
    buffer() : device(VK_NULL_HANDLE), buf(VK_NULL_HANDLE), mem(VK_NULL_HANDLE) {}
    buffer(sample_info &info, VkDeviceSize num_bytes) : buffer(info.device, info.memory_properties, num_bytes) {}

    buffer(VkDevice dev, const VkPhysicalDeviceMemoryProperties memoryProperties, VkDeviceSize num_bytes) : buffer() {
        allocate(dev, memoryProperties, num_bytes);
    };

    void allocate(VkDevice dev, const VkPhysicalDeviceMemoryProperties& memory_properties, VkDeviceSize num_bytes);
    void reset();

    void memset(VkDeviceSize offset, VkDeviceSize size, int value);
    void memcpy(VkDeviceSize offset, VkDeviceSize size, const void* source);

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
static_assert(sizeof(float4) == 16, "bad size for float4");

struct uchar4 {
    unsigned char   x;
    unsigned char   y;
    unsigned char   z;
    unsigned char   w;
};
static_assert(sizeof(uchar4) == 4, "bad size for uchar4");

struct spv_map {
    struct sampler {
        sampler() : opencl_flags(0), binding(-1) {};

        int opencl_flags;
        int binding;
    };

    struct arg {
        enum kind_t { kind_unknown, kind_pod, kind_buffer, kind_ro_image, kind_wo_image, kind_sampler };

        arg() : kind(kind_unknown), binding(-1), offset(0) {};

        kind_t  kind;
        int     binding;
        int     offset;
    };

    struct kernel {
        kernel() : name(), descriptor_set(-1), args() {};

        std::string         name;
        int                 descriptor_set;
        std::vector<arg>    args;
    };

    static arg::kind_t parse_argType(const std::string& argType);
    static spv_map   parse(std::istream& in);

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

class kernel_invocation {
public:
    typedef std::tuple<int,int> WorkgroupDimensions;

public:
    kernel_invocation(VkDevice              device,
                      VkCommandPool         cmdPool,
                      VkDescriptorPool      descPool,
                      const VkPhysicalDeviceMemoryProperties&   memoryProperties,
                      const std::string&    moduleName,
                      std::string           entryPoint,
                      const std::string&    spvmapName,
                      const std::string&    spvmapKernelName);

    ~kernel_invocation();

    template <typename Iterator>
    void    addLiteralSamplers(Iterator first, Iterator last);

    void    addBufferArgument(VkBuffer buf);
    void    addReadOnlyImageArgument(VkImageView image);
    void    addWriteOnlyImageArgument(VkImageView image);
    void    addSamplerArgument(VkSampler samp);

    template <typename T>
    void    addPodArgument(const T& pod);

    void    run(VkQueue                    queue,
                const WorkgroupDimensions& workgroup_sizes,
                const WorkgroupDimensions& num_workgroups);

private:
    void        fillCommandBuffer(VkPipeline                    pipeline,
                                  const WorkgroupDimensions&    num_workgroups);
    VkPipeline  createPipeline(const std::tuple<int,int>& work_group_sizes);
    void        updateDescriptorSets();
    void        submitCommand(VkQueue queue);

private:
    struct arg {
        VkDescriptorType    type;
        VkBuffer            buffer;
        VkSampler           sampler;
        VkImageView         image;
    };

private:
    std::string                         mEntryPoint;
    pipeline_layout                     mPipelineLayout;

    VkDevice                            mDevice;
    VkCommandPool                       mCmdPool;
    VkDescriptorPool                    mDescriptorPool;
    VkCommandBuffer                     mCommand;
    VkShaderModule                      mShaderModule;
    VkPhysicalDeviceMemoryProperties    mMemoryProperties;

    VkDescriptorSet                     mLiteralSamplerDescSet;
    VkDescriptorSet                     mArgumentsDescSet;

    std::vector<VkSampler>              mLiteralSamplers;
    std::vector<arg>                    mArguments;
    std::vector<buffer>                 mPodBuffers;

    std::vector<VkDescriptorSet>        mDescriptors;
};

template <typename Iterator>
void kernel_invocation::addLiteralSamplers(Iterator first, Iterator last) {
    mLiteralSamplers.insert(mLiteralSamplers.end(), first, last);
}

template <typename T>
void kernel_invocation::addPodArgument(const T& pod) {
    buffer scalar_args(mDevice, mMemoryProperties, sizeof(T));
    mPodBuffers.push_back(scalar_args);

    scalar_args.memcpy(0, sizeof(T), &pod);

    addBufferArgument(scalar_args.buf);
}

/* ============================================================================================== */

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

/* ============================================================================================== */

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

spv_map::arg::kind_t spv_map::parse_argType(const std::string& argType) {
    arg::kind_t result = arg::kind_unknown;

    if (argType == "pod") {
        result = arg::kind_pod;
    }
    else if (argType == "buffer") {
        result = arg::kind_buffer;
    }
    else if (argType == "ro_image") {
        result = arg::kind_ro_image;
    }
    else if (argType == "wo_image") {
        result = arg::kind_wo_image;
    }
    else if (argType == "sampler") {
        result = arg::kind_sampler;
    }

    return result;
}

spv_map spv_map::parse(std::istream& in) {
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
                        ka = kernel->args.insert(ka, arg_index - kernel->args.size() + 1, spv_map::arg());
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
                else if ("argType" == key) {
                    ka->kind = parse_argType(value);
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
    return spv_map::parse(in);
}

std::vector<int> count_kernel_bindings(const spv_map& spvMap) {
    std::vector<int> result;

    for (auto &k : spvMap.kernels) {
        auto max_arg = std::max_element(k.args.begin(), k.args.end(), [](const spv_map::arg& a, const spv_map::arg& b) {
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

/* ============================================================================================== */

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

std::vector<VkDescriptorSet> allocate_descriptor_set(VkDevice device, VkDescriptorPool pool, const pipeline_layout& layout) {
    std::vector<VkDescriptorSet> result;

    VkDescriptorSetAllocateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    createInfo.descriptorPool = pool;
    createInfo.descriptorSetCount = layout.descriptors.size();
    createInfo.pSetLayouts = layout.descriptors.data();

    result.resize(createInfo.descriptorSetCount, VK_NULL_HANDLE);
    VkResult U_ASSERT_ONLY res = vkAllocateDescriptorSets(device, &createInfo, result.data());
    assert(res == VK_SUCCESS);

    return result;
}

VkShaderModule create_shader(VkDevice device, const char* spvFileName) {
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
    shaderModuleCreateInfo.codeSize = num_bytes;
    shaderModuleCreateInfo.pCode = spvModule.data();

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkResult U_ASSERT_ONLY res = vkCreateShaderModule(device, &shaderModuleCreateInfo, NULL, &shaderModule);
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


uint32_t find_compatible_memory_index(const VkPhysicalDeviceMemoryProperties& memory_properties,
                                      uint32_t   typeBits,
                                      VkFlags    requirements_mask) {
    uint32_t result = std::numeric_limits<uint32_t>::max();
    assert(memory_properties.memoryTypeCount < std::numeric_limits<uint32_t>::max());

    // Search memtypes to find first index with those properties
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
        if ((typeBits & 1) == 1) {
            // Type is available, does it match user properties?
            if ((memory_properties.memoryTypes[i].propertyFlags & requirements_mask) == requirements_mask) {
                result = i;
                break;
            }
        }
        typeBits >>= 1;
    }

    return result;
}

pipeline_layout init_compute_pipeline_layout(VkDevice device, const spv_map& spvMap) {
    pipeline_layout result;
    result.device = device;

    const int num_samplers = spvMap.samplers.size();
    if (0 < num_samplers) {
        result.descriptors.push_back(create_descriptor_set_layout(device, num_samplers,
                                                                  VK_DESCRIPTOR_TYPE_SAMPLER));
    }

    for (auto &nb : count_kernel_bindings(spvMap)) {
        result.descriptors.push_back(create_descriptor_set_layout(device, nb,
                                                                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER));
    };

    VkPipelineLayoutCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    createInfo.setLayoutCount = result.descriptors.size();
    createInfo.pSetLayouts = createInfo.setLayoutCount ? result.descriptors.data() : NULL;

    VkResult U_ASSERT_ONLY res = vkCreatePipelineLayout(device, &createInfo, NULL, &result.pipeline);
    assert(res == VK_SUCCESS);

    return result;
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

VkCommandBuffer allocate_command_buffer(VkDevice device, VkCommandPool cmd_pool) {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = cmd_pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer result = VK_NULL_HANDLE;
    VkResult U_ASSERT_ONLY res = vkAllocateCommandBuffers(device, &allocInfo, &result);
    assert(res == VK_SUCCESS);

    return result;
}

/* ============================================================================================== */

void buffer::memset(VkDeviceSize offset, VkDeviceSize size, int value) {
    void* data = NULL;
    VkResult U_ASSERT_ONLY res = vkMapMemory(device, mem, offset, size, 0, &data);
    assert(res == VK_SUCCESS);
    ::memset(data, 0, size);
    vkUnmapMemory(device, mem);
}

void buffer::memcpy(VkDeviceSize offset, VkDeviceSize size, const void* source) {
    void* data = NULL;
    VkResult U_ASSERT_ONLY res = vkMapMemory(device, mem, offset, size, 0, &data);
    assert(res == VK_SUCCESS);
    ::memcpy(data, source, size);
    vkUnmapMemory(device, mem);
}

void buffer::allocate(VkDevice                                  dev,
                      const VkPhysicalDeviceMemoryProperties&   memory_properties,
                      VkDeviceSize                              inNumBytes) {
    reset();

    device = dev;

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
    alloc_info.memoryTypeIndex = find_compatible_memory_index(memory_properties,
                                                              mem_reqs.memoryTypeBits,
                                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    assert(alloc_info.memoryTypeIndex < std::numeric_limits<uint32_t>::max() && "No mappable, coherent memory");
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

/* ============================================================================================== */

void pipeline_layout::reset() {
    std::for_each(descriptors.begin(), descriptors.end(), std::bind(vkDestroyDescriptorSetLayout, device, std::placeholders::_1, nullptr));
    descriptors.clear();

    if (VK_NULL_HANDLE != device && VK_NULL_HANDLE != pipeline) {
        vkDestroyPipelineLayout(device, pipeline, NULL);
    }

    device = VK_NULL_HANDLE;
    pipeline = VK_NULL_HANDLE;
}

/* ============================================================================================== */

kernel_invocation::kernel_invocation(VkDevice           device,
                                     VkCommandPool      cmdPool,
                                     VkDescriptorPool   descPool,
                                     const VkPhysicalDeviceMemoryProperties&    memoryProperties,
                                     const std::string& moduleName,
                                     std::string        entryPoint,
                                     const std::string& spvmapName,
                                     const std::string& spvmapKernelName) :
        mEntryPoint(entryPoint),
        mPipelineLayout(),
        mDevice(device),
        mCmdPool(cmdPool),
        mDescriptorPool(descPool),
        mMemoryProperties(memoryProperties),
        mLiteralSamplerDescSet(VK_NULL_HANDLE),
        mArgumentsDescSet(VK_NULL_HANDLE),
        mCommand(VK_NULL_HANDLE),
        mShaderModule(VK_NULL_HANDLE),
        mLiteralSamplers(),
        mArguments(),
        mDescriptors() {
    const spv_map shader_arg_map = create_spv_map(spvmapName.c_str());
    const auto kernel_arg_map = std::find_if(shader_arg_map.kernels.begin(),
                                             shader_arg_map.kernels.end(),
                                             [&spvmapKernelName](const spv_map::kernel& k) {
                                                 return k.name == spvmapKernelName;
                                             });
    assert(kernel_arg_map != shader_arg_map.kernels.end());

    // Create the pipeline layout from the spvmap description
    mPipelineLayout = init_compute_pipeline_layout(mDevice, shader_arg_map);

    mCommand = allocate_command_buffer(mDevice, mCmdPool);
    mShaderModule = create_shader(device, moduleName.c_str());

    mDescriptors = allocate_descriptor_set(mDevice, mDescriptorPool, mPipelineLayout);
    mLiteralSamplerDescSet = mDescriptors[0];
    mArgumentsDescSet = mDescriptors[kernel_arg_map->descriptor_set];
}

kernel_invocation::~kernel_invocation() {
    std::for_each(mPodBuffers.begin(), mPodBuffers.end(), std::mem_fun_ref(&buffer::reset));

    if (mDevice) {
        if (!mDescriptors.empty()) {
            VkResult U_ASSERT_ONLY res = vkFreeDescriptorSets(mDevice,
                                                              mDescriptorPool,
                                                              mDescriptors.size(),
                                                              mDescriptors.data());
            assert(res == VK_SUCCESS);
        }

        if (mShaderModule) {
            vkDestroyShaderModule(mDevice, mShaderModule, NULL);
        }

        if (mCmdPool && mCommand) {
            vkFreeCommandBuffers(mDevice, mCmdPool, 1, &mCommand);
        }
    }

    mPipelineLayout.reset();
}

void kernel_invocation::addBufferArgument(VkBuffer buf) {
    arg item = {};

    item.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    item.buffer = buf;

    mArguments.push_back(item);
}

void kernel_invocation::addSamplerArgument(VkSampler samp) {
    arg item = {};

    item.type = VK_DESCRIPTOR_TYPE_SAMPLER;
    item.sampler = samp;

    mArguments.push_back(item);
}

void kernel_invocation::addReadOnlyImageArgument(VkImageView im) {
    arg item = {};

    item.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    item.image = im;

    mArguments.push_back(item);
}

void kernel_invocation::addWriteOnlyImageArgument(VkImageView im) {
    arg item = {};

    item.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    item.image = im;

    mArguments.push_back(item);
}

void kernel_invocation::updateDescriptorSets() {
    std::vector<VkDescriptorImageInfo>  imageList;
    std::vector<VkDescriptorBufferInfo> bufferList;

    //
    // Collect information about the literal samplers
    //
    for (auto s : mLiteralSamplers) {
        VkDescriptorImageInfo samplerInfo = {};
        samplerInfo.sampler = s;

        imageList.push_back(samplerInfo);
    }

    //
    // Collect information about the arguments
    //
    for (auto& a : mArguments) {
        switch (a.type) {
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: {
                VkDescriptorImageInfo imageInfo = {};
                imageInfo.imageView = a.image;
                imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                imageList.push_back(imageInfo);
                break;
            }

            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: {
                VkDescriptorBufferInfo bufferInfo = {};
                bufferInfo.range = VK_WHOLE_SIZE;
                bufferInfo.buffer = a.buffer;

                bufferList.push_back(bufferInfo);
                break;
            }

            case VK_DESCRIPTOR_TYPE_SAMPLER: {
                VkDescriptorImageInfo samplerInfo = {};
                samplerInfo.sampler = a.sampler;

                imageList.push_back(samplerInfo);
                break;
            }

            default:
                assert(0 && "unkown argument type");
        }
    }

    //
    // Set up to create the descriptor set write structures
    // We will iterate the param lists in the same order,
    // picking up image and buffer infos in order.
    //

    std::vector<VkWriteDescriptorSet> writeSets;
    auto nextImage = imageList.begin();
    auto nextBuffer = bufferList.begin();


    //
    // Update the literal samplers' descriptor set
    //

    VkWriteDescriptorSet literalSamplerSet = {};
    literalSamplerSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    literalSamplerSet.dstSet = mLiteralSamplerDescSet;
    literalSamplerSet.dstBinding = 0;
    literalSamplerSet.descriptorCount = 1;
    literalSamplerSet.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;

    for (auto s : mLiteralSamplers) {
        literalSamplerSet.pImageInfo = &(*nextImage);
        ++nextImage;

        writeSets.push_back(literalSamplerSet);

        ++literalSamplerSet.dstBinding;
    }

    //
    // Update the kernel's argument descriptor set
    //

    VkWriteDescriptorSet argSet = {};
    argSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    argSet.dstSet = mArgumentsDescSet;
    argSet.dstBinding = 0;
    argSet.descriptorCount = 1;

    for (auto& a : mArguments) {
        switch (a.type) {
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                argSet.descriptorType = a.type;
                argSet.pImageInfo = &(*nextImage);
                ++nextImage;
                break;

            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                argSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                argSet.pBufferInfo = &(*nextBuffer);
                ++nextBuffer;
                break;

            case VK_DESCRIPTOR_TYPE_SAMPLER:
                argSet.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                literalSamplerSet.pImageInfo = &(*nextImage);
                ++nextImage;
                break;

            default:
                assert(0 && "unkown argument type");
        }

        writeSets.push_back(argSet);

        ++argSet.dstBinding;
    }

    //
    // Do the actual descriptor set updates
    //
    vkUpdateDescriptorSets(mDevice, writeSets.size(), writeSets.data(), 0, nullptr);
}

void kernel_invocation::fillCommandBuffer(VkPipeline                    pipeline,
                                          const WorkgroupDimensions&    num_workgroups) {
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VkResult U_ASSERT_ONLY res = vkBeginCommandBuffer(mCommand, &beginInfo);
    assert(res == VK_SUCCESS);

    vkCmdBindPipeline(mCommand, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    vkCmdBindDescriptorSets(mCommand, VK_PIPELINE_BIND_POINT_COMPUTE,
                            mPipelineLayout.pipeline,
                            0,
                            mDescriptors.size(), mDescriptors.data(),
                            0, NULL);

    vkCmdDispatch(mCommand, std::get<0>(num_workgroups), std::get<1>(num_workgroups), 1);

    res = vkEndCommandBuffer(mCommand);
    assert(res == VK_SUCCESS);
}

void kernel_invocation::submitCommand(VkQueue queue) {
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &mCommand;

    VkResult U_ASSERT_ONLY res = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    assert(res == VK_SUCCESS);

}

VkPipeline kernel_invocation::createPipeline(const WorkgroupDimensions& work_group_sizes) {
    const unsigned int num_workgroup_sizes = 3;
    const int32_t workGroupSizes[num_workgroup_sizes] = {
            std::get<0>(work_group_sizes),
            std::get<1>(work_group_sizes),
            1
    };
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
    createInfo.layout = mPipelineLayout.pipeline;

    createInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    createInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    createInfo.stage.module = mShaderModule;
    createInfo.stage.pName = mEntryPoint.c_str();
    createInfo.stage.pSpecializationInfo = &specializationInfo;

    VkPipeline result = VK_NULL_HANDLE;
    VkResult U_ASSERT_ONLY res = vkCreateComputePipelines(mDevice, VK_NULL_HANDLE, 1, &createInfo, NULL, &result);
    assert(res == VK_SUCCESS);

    return result;
}

void kernel_invocation::run(VkQueue                     queue,
                            const WorkgroupDimensions&  workgroup_sizes,
                            const WorkgroupDimensions&  num_workgroups) {
    const VkPipeline pipeline = createPipeline(workgroup_sizes);

    updateDescriptorSets();
    fillCommandBuffer(pipeline, num_workgroups);
    submitCommand(queue);

    vkQueueWaitIdle(queue);

    vkDestroyPipeline(mDevice, pipeline, NULL);
}

/* ============================================================================================== */

void check_fill_results(const buffer&    buf,
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
    VkResult U_ASSERT_ONLY res = vkMapMemory(buf.device, buf.mem, 0, VK_WHOLE_SIZE, 0, &data);
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
    vkUnmapMemory(buf.device, buf.mem);

    LOGE("%s: Correct pixels=%d; Incorrect pixels=%d", label, num_correct_pixels, num_incorrect_pixels);
}

void run_fill_kernel(struct sample_info& info, const std::vector<VkSampler>& samplers) {
    const int buffer_height = 64;
    const int buffer_width = 64;

    struct scalar_args {
        int     inPitch;        // offset 0
        int     inDeviceFormat; // DevicePixelFormat offset 4
        int     inOffsetX;      // offset 8
        int     inOffsetY;      // offset 12
        int     inWidth;        // offset 16
        int     inHeight;       // offset 20
        float4  inColor;        // offset 32
    };
    static_assert(0 == offsetof(scalar_args, inPitch), "inPitch offset incorrect");
    static_assert(4 == offsetof(scalar_args, inDeviceFormat), "inDeviceFormat offset incorrect");
    static_assert(8 == offsetof(scalar_args, inOffsetX), "inOffsetX offset incorrect");
    static_assert(12 == offsetof(scalar_args, inOffsetY), "inOffsetY offset incorrect");
    static_assert(16 == offsetof(scalar_args, inWidth), "inWidth offset incorrect");
    static_assert(20 == offsetof(scalar_args, inHeight), "inHeight offset incorrect");
    static_assert(32 == offsetof(scalar_args, inColor), "inColor offset incorrect");

    const scalar_args scalars = {
            buffer_width,               // inPitch
            1,                          // inDeviceFormat - kDevicePixelFormat_BGRA_4444_32f
            0,                          // inOffsetX
            0,                          // inOffsetY
            buffer_width,               // inWidth
            buffer_height,              // inHeight
            { 0.25f, 0.50f, 0.75f, 1.0f }  // inColor
    };

    // allocate image buffer
    const std::size_t buffer_size = scalars.inPitch * scalars.inHeight * sizeof(float4);
    buffer image_buffer(info, buffer_size);

    const auto workgroup_sizes = std::make_tuple(32, 32);
    const auto num_workgroups = std::make_tuple((scalars.inWidth + std::get<0>(workgroup_sizes) - 1) / std::get<0>(workgroup_sizes),
                                                (scalars.inHeight + std::get<1>(workgroup_sizes) - 1) / std::get<1>(workgroup_sizes));

    image_buffer.memset(0, buffer_size, 0);
    {
        kernel_invocation glslInvocation(info.device,
                                         info.cmd_pool,
                                         info.desc_pool,
                                         info.memory_properties,
                                         "fills_glsl.spv", "main",
                                         "fills.spvmap", "FillWithColorKernel");
        glslInvocation.addLiteralSamplers(samplers.begin(), samplers.end());
        glslInvocation.addBufferArgument(image_buffer.buf);
        glslInvocation.addPodArgument(scalars);
        glslInvocation.run(info.graphics_queue, workgroup_sizes, num_workgroups);
    }
    check_fill_results(image_buffer, scalars.inWidth, scalars.inHeight,
                       scalars.inPitch, scalars.inColor, "fills_glsl.spv/main");

    image_buffer.memset(0, buffer_size, 0);
    {
        kernel_invocation oclInvocation(info.device,
                                        info.cmd_pool,
                                        info.desc_pool,
                                        info.memory_properties,
                                        "fills.spv", "FillWithColorKernel",
                                        "fills.spvmap", "FillWithColorKernel");
        oclInvocation.addLiteralSamplers(samplers.begin(), samplers.end());
        oclInvocation.addBufferArgument(image_buffer.buf);
        oclInvocation.addPodArgument(scalars);
        oclInvocation.run(info.graphics_queue, workgroup_sizes, num_workgroups);
    }
    check_fill_results(image_buffer, scalars.inWidth, scalars.inHeight,
                       scalars.inPitch, scalars.inColor, "fills.spv/FillWithColorKernel");

    image_buffer.reset();
}

/* ============================================================================================== */

void run_copytoimage_kernel(struct sample_info& info, const std::vector<VkSampler>& samplers) {
    const int buffer_height = 64;
    const int buffer_width = 64;

    struct scalar_args {
        int inSrcOffset;        // offset 0
        int inSrcPitch;         // offset 4
        int inSrcChannelOrder;  // offset 8 -- cl_channel_order
        int inSrcChannelType;   // offset 12 -- cl_channel_type
        int inSwapComponents;   // offset 16 -- bool
        int inPremultiply;      // offset 20 -- bool
        int inWidth;            // offset 24
        int inHeight;           // offset 28
    };
    static_assert(0 == offsetof(scalar_args, inSrcOffset), "inSrcOffset offset incorrect");
    static_assert(4 == offsetof(scalar_args, inSrcPitch), "inSrcPitch offset incorrect");
    static_assert(8 == offsetof(scalar_args, inSrcChannelOrder), "inSrcChannelOrder offset incorrect");
    static_assert(12 == offsetof(scalar_args, inSrcChannelType), "inSrcChannelType offset incorrect");
    static_assert(16 == offsetof(scalar_args, inSwapComponents), "inSwapComponents offset incorrect");
    static_assert(20 == offsetof(scalar_args, inPremultiply), "inPremultiply offset incorrect");
    static_assert(24 == offsetof(scalar_args, inWidth), "inWidth offset incorrect");
    static_assert(28 == offsetof(scalar_args, inHeight), "inHeight offset incorrect");

    const scalar_args scalars = {
            0,              // inSrcOffset
            buffer_width,   // inSrcPitch
            0x10B6,         // inSrcChannelOrder -- CL_BGRA
            0x10D2,         // inSrcChannelType -- CL_UNORM_INT8
            0,              // inSwapComponents
            0,              // inPremultiply
            buffer_width,   // inWidth
            buffer_height   // inHeight
    };

    // allocate image buffer
    const std::size_t buffer_size = scalars.inSrcPitch * scalars.inHeight * sizeof(uchar4);

    buffer input_buffer(info, buffer_size);

    {
        VkImageCreateInfo imageInfo = {};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = scalars.inWidth;
        imageInfo.extent.height = scalars.inHeight;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = VK_FORMAT_R8G8B8A8_UINT;
        imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    const auto workgroup_sizes = std::make_tuple(32, 32);
    const auto num_workgroups = std::make_tuple((scalars.inWidth + std::get<0>(workgroup_sizes) - 1) / std::get<0>(workgroup_sizes),
                                                (scalars.inHeight + std::get<1>(workgroup_sizes) - 1) / std::get<1>(workgroup_sizes));

    if (0) {    // STILL TESTING!!
        kernel_invocation copyToImageInvocation(info.device,
                                                info.cmd_pool,
                                                info.desc_pool,
                                                info.memory_properties,
                                                "memory.spv", "CopyBufferToImageKernel",
                                                "memory.spvmap", "CopyBufferToImageKernel");
    }

    input_buffer.reset();
}

/* ============================================================================================== */

int sample_main(int argc, char *argv[]) {
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

    run_fill_kernel(info, samplers);
    run_copytoimage_kernel(info, samplers);

    //
    // Clean up
    //

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
