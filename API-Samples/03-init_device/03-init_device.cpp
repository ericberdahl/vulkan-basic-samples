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

#include "half.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <util_init.hpp>

/* ============================================================================================== */

namespace vulkan_utils {

    class error : public std::runtime_error {
        VkResult    mResult;
    public:
        error(const std::string& s, VkResult result) : runtime_error(s), mResult(result) {}

        VkResult get_result() const { return mResult; }
    };

    void throwIfNotSuccess(VkResult result, const std::string& label);

    struct device_memory {
        device_memory() : device(VK_NULL_HANDLE), mem(VK_NULL_HANDLE) {}
        device_memory(VkDevice                                  dev,
                      const VkMemoryRequirements&               mem_reqs,
                      const VkPhysicalDeviceMemoryProperties    memoryProperties)
                : device_memory() {
            allocate(dev, mem_reqs, memoryProperties);
        };

        void    allocate(VkDevice                                   dev,
                         const VkMemoryRequirements&                mem_reqs,
                         const VkPhysicalDeviceMemoryProperties&    memory_properties);
        void    reset();

        VkDevice        device;
        VkDeviceMemory  mem;
    };

    struct buffer {
        buffer() : mem(), buf(VK_NULL_HANDLE) {}
        buffer(const sample_info &info, VkDeviceSize num_bytes) : buffer(info.device, info.memory_properties, num_bytes) {}

        buffer(VkDevice dev, const VkPhysicalDeviceMemoryProperties memoryProperties, VkDeviceSize num_bytes) : buffer() {
            allocate(dev, memoryProperties, num_bytes);
        };

        void    allocate(VkDevice dev, const VkPhysicalDeviceMemoryProperties& memory_properties, VkDeviceSize num_bytes);
        void    reset();

        device_memory   mem;
        VkBuffer        buf;
    };

    struct image {
        image() : mem(), im(VK_NULL_HANDLE), view(VK_NULL_HANDLE) {}
        image(const sample_info&  info,
              uint32_t      width,
              uint32_t      height,
              VkFormat      format) : image(info.device, info.memory_properties, width, height, format) {}

        image(VkDevice dev,
              const VkPhysicalDeviceMemoryProperties memoryProperties,
              uint32_t                                   width,
              uint32_t                                   height,
              VkFormat                                   format) : image() {
            allocate(dev, memoryProperties, width, height, format);
        };

        void    allocate(VkDevice                                   dev,
                         const VkPhysicalDeviceMemoryProperties&    memory_properties,
                         uint32_t                                   width,
                         uint32_t                                   height,
                         VkFormat                                   format);
        void    reset();

        device_memory   mem;
        VkImage         im;
        VkImageView     view;
    };

    struct memory_map {
        memory_map(VkDevice dev, VkDeviceMemory mem);
        memory_map(const device_memory& mem) : memory_map(mem.device, mem.mem) {}
        memory_map(const buffer& buf) : memory_map(buf.mem) {}
        memory_map(const image& im) : memory_map(im.mem) {}
        ~memory_map();

        VkDevice        dev;
        VkDeviceMemory  mem;
        void*           data;
    };
}

namespace vulkan_utils {

    namespace {
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
    }

/* ============================================================================================== */

    void throwIfNotSuccess(VkResult result, const std::string& label) {
        if (VK_SUCCESS != result) {
            throw error(label, result);
        }
    }

/* ============================================================================================== */

    memory_map::memory_map(VkDevice device, VkDeviceMemory memory) :
            dev(device), mem(memory), data(nullptr) {
        throwIfNotSuccess(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &data),
                          "vkMapMemory");
    }

    memory_map::~memory_map() {
        if (dev && mem) {
            vkUnmapMemory(dev, mem);
        }
    }

/* ============================================================================================== */

    void device_memory::allocate(VkDevice dev,
                                 const VkMemoryRequirements &mem_reqs,
                                 const VkPhysicalDeviceMemoryProperties &memory_properties) {
        reset();

        device = dev;

        // Allocate memory for the buffer
        VkMemoryAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_reqs.size;
        alloc_info.memoryTypeIndex = find_compatible_memory_index(memory_properties,
                                                                  mem_reqs.memoryTypeBits,
                                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        assert(alloc_info.memoryTypeIndex < std::numeric_limits<uint32_t>::max() &&
               "No mappable, coherent memory");
        throwIfNotSuccess(vkAllocateMemory(device, &alloc_info, NULL, &mem),
                          "vkAllocateMemory");
    }

    void device_memory::reset() {
        if (mem != VK_NULL_HANDLE) {
            vkFreeMemory(device, mem, NULL);
            mem = VK_NULL_HANDLE;
        }

        device = VK_NULL_HANDLE;
    }


/* ============================================================================================== */

    void buffer::allocate(VkDevice dev,
                          const VkPhysicalDeviceMemoryProperties &memory_properties,
                          VkDeviceSize inNumBytes) {
        reset();

        // Allocate the buffer
        VkBufferCreateInfo buf_info = {};
        buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        buf_info.size = inNumBytes;
        buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        throwIfNotSuccess(vkCreateBuffer(dev, &buf_info, NULL, &buf),
                          "vkCreateBuffer");

        // Find out what we need in order to allocate memory for the buffer
        VkMemoryRequirements mem_reqs = {};
        vkGetBufferMemoryRequirements(dev, buf, &mem_reqs);

        mem.allocate(dev, mem_reqs, memory_properties);

        // Bind the memory to the buffer object
        throwIfNotSuccess(vkBindBufferMemory(dev, buf, mem.mem, 0),
                          "vkBindBufferMemory");
    }

    void buffer::reset() {
        if (buf != VK_NULL_HANDLE) {
            vkDestroyBuffer(mem.device, buf, NULL);
            buf = VK_NULL_HANDLE;
        }

        mem.reset();
    }

/* ============================================================================================== */

    void image::allocate(VkDevice dev,
                         const VkPhysicalDeviceMemoryProperties &memory_properties,
                         uint32_t width,
                         uint32_t height,
                         VkFormat format) {
        reset();

        VkImageCreateInfo imageInfo = {};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = format;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_GENERAL;

        throwIfNotSuccess(vkCreateImage(dev, &imageInfo, nullptr, &im),
                          "vkCreateImage");

        // Find out what we need in order to allocate memory for the image
        VkMemoryRequirements mem_reqs = {};
        vkGetImageMemoryRequirements(dev, im, &mem_reqs);

        mem.allocate(dev, mem_reqs, memory_properties);

        // Bind the memory to the image object
        throwIfNotSuccess(vkBindImageMemory(dev, im, mem.mem, 0),
                          "vkBindImageMemory");

        // Allocate the image view
        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = im;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = 1;

        throwIfNotSuccess(vkCreateImageView(dev, &viewInfo, nullptr, &view),
                          "vkCreateImageView");
    }

    void image::reset() {
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(mem.device, view, NULL);
            view = VK_NULL_HANDLE;
        }
        if (im != VK_NULL_HANDLE) {
            vkDestroyImage(mem.device, im, NULL);
            im = VK_NULL_HANDLE;
        }

        mem.reset();
    }

/* ============================================================================================== */

} // namespace vulkan_utils

/* ============================================================================================== */

namespace clspv_utils {

    namespace details {
        struct spv_map {
            struct sampler {
                sampler() : opencl_flags(0), binding(-1) {};

                int opencl_flags;
                int binding;
            };

            struct arg {
                enum kind_t {
                    kind_unknown, kind_pod, kind_buffer, kind_ro_image, kind_wo_image, kind_sampler
                };

                arg() : kind(kind_unknown), binding(-1), offset(0) {};

                kind_t kind;
                int binding;
                int offset;
            };

            struct kernel {
                kernel() : name(), descriptor_set(-1), args() {};

                std::string name;
                int descriptor_set;
                std::vector<arg> args;
            };

            static arg::kind_t parse_argType(const std::string &argType);

            static spv_map parse(std::istream &in);

            spv_map() : samplers(), kernels(), samplers_desc_set(-1) {};

            std::vector<sampler> samplers;
            int samplers_desc_set;
            std::vector<kernel> kernels;
        };

        struct pipeline_layout {
            pipeline_layout() : device(VK_NULL_HANDLE), descriptors(), pipeline(VK_NULL_HANDLE) {};

            void    reset();

            VkDevice                            device;
            std::vector<VkDescriptorSetLayout>  descriptors;
            VkPipelineLayout                    pipeline;
        };
    } // namespace details

    struct WorkgroupDimensions {
        WorkgroupDimensions(int xDim = 1, int yDim = 1) : x(xDim), y(yDim) {}

        int x;
        int y;
    };

    class kernel_module {
    public:
        kernel_module(VkDevice              device,
                      VkDescriptorPool      pool,
                      const std::string&    moduleName);

        ~kernel_module();

        VkDescriptorSet getLiteralSamplersDescriptorSet() const;
        VkDescriptorSet getKernelArgumentDescriptorSet(const std::string& entryPoint) const;

        std::string                 getName() const { return mName; }
        std::vector<std::string>    getEntryPoints() const;

        VkPipeline      createPipeline(const std::string&           entryPoint,
                                       const WorkgroupDimensions&   work_group_sizes) const;

        void bindDescriptors(VkCommandBuffer command) const;

    private:
        std::vector<VkDescriptorSet>    allocateDescriptorSet(VkDescriptorPool pool) const;
        VkDescriptorSetLayout           createDescriptorSetLayout(const std::vector<VkDescriptorType>& descriptorTypes);
        details::pipeline_layout        createPipelineLayout(const details::spv_map& spvMap);

    private:
        std::string                         mName;
        details::pipeline_layout            mPipelineLayout;

        VkDevice                            mDevice;
        std::vector<VkDescriptorSet>        mDescriptors;
        VkDescriptorPool                    mDescriptorPool;
        VkShaderModule                      mShaderModule;
        details::spv_map                    mSpvMap;
    };

    class kernel {
    public:
        kernel(VkDevice                     device,
               const kernel_module&         module,
               std::string                  entryPoint,
               const WorkgroupDimensions&   workgroup_sizes);

        ~kernel();

        void bindPipeline(VkCommandBuffer command) const;

        std::string getEntryPoint() const { return mEntryPoint; }
        WorkgroupDimensions getWorkgroupSize() const { return mWorkgroupSizes; }

        VkDescriptorSet getLiteralSamplerDescSet() const { return mLiteralSamplerDescSet; }
        VkDescriptorSet getArgumentDescSet() const { return mArgumentDescSet; }

    private:
        VkDescriptorSetLayout   createDescriptorSetLayout(const std::vector<VkDescriptorType>& descriptorTypes);
        VkPipeline              createPipeline(const std::tuple<int,int>& work_group_sizes);

    private:
        std::string         mEntryPoint;
        WorkgroupDimensions mWorkgroupSizes;
        VkDevice            mDevice;
        VkPipeline          mPipeline;
        VkDescriptorSet     mLiteralSamplerDescSet;
        VkDescriptorSet     mArgumentDescSet;
    };

    class kernel_invocation {
    public:
        kernel_invocation(VkDevice              device,
                          VkCommandPool         cmdPool,
                          const VkPhysicalDeviceMemoryProperties&   memoryProperties);

        ~kernel_invocation();

        template <typename Iterator>
        void    addLiteralSamplers(Iterator first, Iterator last);

        void    addBufferArgument(VkBuffer buf);
        void    addReadOnlyImageArgument(VkImageView image);
        void    addWriteOnlyImageArgument(VkImageView image);
        void    addSamplerArgument(VkSampler samp);

        template <typename T>
        void    addPodArgument(const T& pod);

        void    run(VkQueue                     queue,
                    const kernel_module&        module,
                    const kernel&               kern,
                    const WorkgroupDimensions&  num_workgroups);

    private:
        void        fillCommandBuffer(const kernel_module&          module,
                                      const kernel&                 kern,
                                      const WorkgroupDimensions&    num_workgroups);
        void        updateDescriptorSets(VkDescriptorSet literalSamplerSet,
                                         VkDescriptorSet argumentSet);
        void        submitCommand(VkQueue queue);

    private:
        struct arg {
            VkDescriptorType    type;
            VkBuffer            buffer;
            VkSampler           sampler;
            VkImageView         image;
        };

    private:
        VkDevice                            mDevice;
        VkCommandPool                       mCmdPool;
        VkCommandBuffer                     mCommand;
        VkPhysicalDeviceMemoryProperties    mMemoryProperties;

        std::vector<VkSampler>              mLiteralSamplers;
        std::vector<arg>                    mArguments;
        std::vector<vulkan_utils::buffer>   mPodBuffers;
    };

    template <typename Iterator>
    void kernel_invocation::addLiteralSamplers(Iterator first, Iterator last) {
        mLiteralSamplers.insert(mLiteralSamplers.end(), first, last);
    }

    template <typename T>
    void kernel_invocation::addPodArgument(const T& pod) {
        vulkan_utils::buffer scalar_args(mDevice, mMemoryProperties, sizeof(T));
        mPodBuffers.push_back(scalar_args);

        {
            vulkan_utils::memory_map scalar_map(scalar_args);
            memcpy(scalar_map.data, &pod, sizeof(T));
        }

        addBufferArgument(scalar_args.buf);
    }
}

namespace clspv_utils {

    namespace {

        details::spv_map create_spv_map(const char *spvmapFilename) {
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
            return details::spv_map::parse(in);
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

        VkShaderModule create_shader(VkDevice device, const std::string& spvFilename) {
            std::FILE* spv_file = AndroidFopen(spvFilename.c_str(), "rb");
            if (!spv_file) {
                throw std::runtime_error("can't open file: " + spvFilename);
            }

            std::fseek(spv_file, 0, SEEK_END);
            // Use vector of uint32_t to ensure alignment is satisfied.
            const auto num_bytes = std::ftell(spv_file);
            if (0 != (num_bytes % sizeof(uint32_t))) {
                throw std::runtime_error("file size of " + spvFilename + " inappropriate for spv file");
            }
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
            vulkan_utils::throwIfNotSuccess(vkCreateShaderModule(device,
                                                                 &shaderModuleCreateInfo,
                                                                 NULL,
                                                                 &shaderModule),
                                            "vkCreateShaderModule");

            return shaderModule;
        }

        VkCommandBuffer allocate_command_buffer(VkDevice device, VkCommandPool cmd_pool) {
            VkCommandBufferAllocateInfo allocInfo = {};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = cmd_pool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1;

            VkCommandBuffer result = VK_NULL_HANDLE;
            vulkan_utils::throwIfNotSuccess(vkAllocateCommandBuffers(device, &allocInfo, &result),
                                            "vkAllocateCommandBuffers");

            return result;
        }

    } // anonymous namespace

    namespace details {

        spv_map::arg::kind_t spv_map::parse_argType(const std::string &argType) {
            arg::kind_t result = arg::kind_unknown;

            if (argType == "pod") {
                result = arg::kind_pod;
            } else if (argType == "buffer") {
                result = arg::kind_buffer;
            } else if (argType == "ro_image") {
                result = arg::kind_ro_image;
            } else if (argType == "wo_image") {
                result = arg::kind_wo_image;
            } else if (argType == "sampler") {
                result = arg::kind_sampler;
            } else {
                assert(0 && "unknown spvmap arg type");
            }

            return result;
        }

        spv_map spv_map::parse(std::istream &in) {
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

                            if (-1 == result.samplers_desc_set) {
                                result.samplers_desc_set = ds;
                            }
                        } else if ("binding" == key) {
                            s->binding = std::atoi(value.c_str());
                        }
                    }
                } else if ("kernel" == key) {
                    auto kernel = std::find_if(result.kernels.begin(), result.kernels.end(),
                                               [&value](const spv_map::kernel &iter) {
                                                   return iter.name == value;
                                               });
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
                                ka = kernel->args.insert(ka, arg_index - kernel->args.size() + 1,
                                                         spv_map::arg());
                            } else {
                                ka = std::next(kernel->args.begin(), arg_index);
                            }

                            assert(ka != kernel->args.end());
                        } else if ("descriptorSet" == key) {
                            const int ds = std::atoi(value.c_str());
                            if (-1 == kernel->descriptor_set) {
                                kernel->descriptor_set = ds;
                            }

                            // all args for a kernel are documented to share the same descriptor set
                            assert(ds == kernel->descriptor_set);
                        } else if ("binding" == key) {
                            ka->binding = std::atoi(value.c_str());
                        } else if ("offset" == key) {
                            ka->offset = std::atoi(value.c_str());
                        } else if ("argType" == key) {
                            ka->kind = parse_argType(value);
                        }
                    }
                }
            }

            std::sort(result.kernels.begin(),
                      result.kernels.end(),
                      [](const spv_map::kernel &a, const spv_map::kernel &b) {
                          return a.descriptor_set < b.descriptor_set;
                      });

            return result;
        }

        void pipeline_layout::reset() {
            std::for_each(descriptors.begin(), descriptors.end(), std::bind(vkDestroyDescriptorSetLayout, device, std::placeholders::_1, nullptr));
            descriptors.clear();

            if (VK_NULL_HANDLE != device && VK_NULL_HANDLE != pipeline) {
                vkDestroyPipelineLayout(device, pipeline, NULL);
            }

            device = VK_NULL_HANDLE;
            pipeline = VK_NULL_HANDLE;
        }

    } // namespace details

    kernel_module::kernel_module(VkDevice           device,
                                 VkDescriptorPool   pool,
                                 const std::string& moduleName) :
            mName(moduleName),
            mPipelineLayout(),
            mDevice(device),
            mDescriptors(),
            mDescriptorPool(pool),
            mShaderModule(VK_NULL_HANDLE),
            mSpvMap() {
        const std::string spvFilename = moduleName + ".spv";
        mShaderModule = create_shader(device, spvFilename.c_str());

        const std::string mapFilename = moduleName + ".spvmap";
        mSpvMap = create_spv_map(mapFilename.c_str());

        mPipelineLayout = createPipelineLayout(mSpvMap);
        mDescriptors = allocateDescriptorSet(pool);
    }

    kernel_module::~kernel_module() {
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
        }

        mPipelineLayout.reset();
    }

    VkDescriptorSet kernel_module::getLiteralSamplersDescriptorSet() const {
        VkDescriptorSet result = VK_NULL_HANDLE;

        if (-1 != mSpvMap.samplers_desc_set) {
            result = mDescriptors[mSpvMap.samplers_desc_set];
        }

        return result;
    }

    VkDescriptorSet kernel_module::getKernelArgumentDescriptorSet(const std::string& entryPoint) const {
        VkDescriptorSet result = VK_NULL_HANDLE;

        const auto kernel_arg_map = std::find_if(mSpvMap.kernels.begin(),
                                                 mSpvMap.kernels.end(),
                                                 [&entryPoint](const details::spv_map::kernel& k) {
                                                     return k.name == entryPoint;
                                                 });
        if (kernel_arg_map != mSpvMap.kernels.end() && -1 != kernel_arg_map->descriptor_set) {
            result = mDescriptors[kernel_arg_map->descriptor_set];
        }

        return result;

    }

    std::vector<std::string> kernel_module::getEntryPoints() const {
        std::vector<std::string> result;

        std::transform(mSpvMap.kernels.begin(), mSpvMap.kernels.end(),
                       std::back_inserter(result),
                       [](const details::spv_map::kernel& k) { return k.name; });

        return result;
    }

    std::vector<VkDescriptorSet> kernel_module::allocateDescriptorSet(VkDescriptorPool pool) const {
        std::vector<VkDescriptorSet> result;

        VkDescriptorSetAllocateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        createInfo.descriptorPool = pool;
        createInfo.descriptorSetCount = mPipelineLayout.descriptors.size();
        createInfo.pSetLayouts = mPipelineLayout.descriptors.data();

        result.resize(createInfo.descriptorSetCount, VK_NULL_HANDLE);
        vulkan_utils::throwIfNotSuccess(vkAllocateDescriptorSets(mDevice,
                                                                 &createInfo,
                                                                 result.data()),
                                        "vkAllocateDescriptorSets");

        return result;
    }

    details::pipeline_layout kernel_module::createPipelineLayout(const details::spv_map& spvMap) {
        details::pipeline_layout result;
        result.device = mDevice;

        std::vector<VkDescriptorType> descriptorTypes;

        const int num_samplers = spvMap.samplers.size();
        if (0 < num_samplers) {
            descriptorTypes.clear();
            descriptorTypes.resize(num_samplers, VK_DESCRIPTOR_TYPE_SAMPLER);
            result.descriptors.push_back(createDescriptorSetLayout(descriptorTypes));
        }

        for (auto &k : spvMap.kernels) {
            descriptorTypes.clear();


            for (auto &ka : k.args) {
                // ignore any argument not in offset 0
                if (0 != ka.offset) continue;

                VkDescriptorType argType;

                switch (ka.kind) {
                    case details::spv_map::arg::kind_pod:
                    case details::spv_map::arg::kind_buffer:
                        argType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        break;

                    case details::spv_map::arg::kind_ro_image:
                        argType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                        break;

                    case details::spv_map::arg::kind_wo_image:
                        argType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                        break;

                    case details::spv_map::arg::kind_sampler:
                        argType = VK_DESCRIPTOR_TYPE_SAMPLER;
                        break;

                    default:
                        assert(0 && "unkown argument type");
                }

                descriptorTypes.push_back(argType);
            }
            result.descriptors.push_back(createDescriptorSetLayout(descriptorTypes));
        };

        VkPipelineLayoutCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        createInfo.setLayoutCount = result.descriptors.size();
        createInfo.pSetLayouts = createInfo.setLayoutCount ? result.descriptors.data() : NULL;

        vulkan_utils::throwIfNotSuccess(vkCreatePipelineLayout(mDevice,
                                                               &createInfo,
                                                               NULL,
                                                               &result.pipeline),
                                        "vkCreatePipelineLayout");

        return result;
    }

    VkDescriptorSetLayout kernel_module::createDescriptorSetLayout(const std::vector<VkDescriptorType>& descriptorTypes) {
        std::vector<VkDescriptorSetLayoutBinding> bindingSet;

        VkDescriptorSetLayoutBinding binding = {};
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        binding.descriptorCount = 1;
        binding.binding = 0;

        for (auto type : descriptorTypes) {
            binding.descriptorType = type;
            bindingSet.push_back(binding);

            ++binding.binding;
        }

        VkDescriptorSetLayoutCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        createInfo.bindingCount = bindingSet.size();
        createInfo.pBindings = createInfo.bindingCount ? bindingSet.data() : NULL;

        VkDescriptorSetLayout result = VK_NULL_HANDLE;
        vulkan_utils::throwIfNotSuccess(vkCreateDescriptorSetLayout(mDevice,
                                                                    &createInfo,
                                                                    NULL,
                                                                    &result),
                                        "vkCreateDescriptorSetLayout");

        return result;
    }

    VkPipeline kernel_module::createPipeline(const std::string& entryPoint, const WorkgroupDimensions& work_group_sizes) const {
        const unsigned int num_workgroup_sizes = 3;
        const int32_t workGroupSizes[num_workgroup_sizes] = {
                work_group_sizes.x,
                work_group_sizes.y,
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
        createInfo.stage.pName = entryPoint.c_str();
        createInfo.stage.pSpecializationInfo = &specializationInfo;

        VkPipeline result = VK_NULL_HANDLE;
        vulkan_utils::throwIfNotSuccess(vkCreateComputePipelines(mDevice,
                                                                 VK_NULL_HANDLE,
                                                                 1,
                                                                 &createInfo,
                                                                 NULL,
                                                                 &result),
                                        "vkCreateComputePipelines");

        return result;
    }

    void kernel_module::bindDescriptors(VkCommandBuffer command) const {
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                mPipelineLayout.pipeline,
                                0,
                                mDescriptors.size(), mDescriptors.data(),
                                0, NULL);
    }

    kernel::kernel(VkDevice                     device,
                   const kernel_module&         module,
                   std::string                  entryPoint,
                   const WorkgroupDimensions&   workgroup_sizes) :
            mEntryPoint(entryPoint),
            mWorkgroupSizes(workgroup_sizes),
            mDevice(device),
            mPipeline(VK_NULL_HANDLE),
            mLiteralSamplerDescSet(VK_NULL_HANDLE),
            mArgumentDescSet(VK_NULL_HANDLE){
        mLiteralSamplerDescSet = module.getLiteralSamplersDescriptorSet();
        mArgumentDescSet = module.getKernelArgumentDescriptorSet(entryPoint);
        mPipeline = module.createPipeline(entryPoint, workgroup_sizes);
    }

    kernel::~kernel() {
        if (mDevice) {
            if (mPipeline) {
                vkDestroyPipeline(mDevice, mPipeline, NULL);
            }
        }
    }

    void kernel::bindPipeline(VkCommandBuffer command) const {
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, mPipeline);
    }

    kernel_invocation::kernel_invocation(VkDevice           device,
                                         VkCommandPool      cmdPool,
                                         const VkPhysicalDeviceMemoryProperties&    memoryProperties) :
            mDevice(device),
            mCmdPool(cmdPool),
            mMemoryProperties(memoryProperties),
            mCommand(VK_NULL_HANDLE),
            mLiteralSamplers(),
            mArguments() {
        mCommand = allocate_command_buffer(mDevice, mCmdPool);
    }

    kernel_invocation::~kernel_invocation() {
        std::for_each(mPodBuffers.begin(), mPodBuffers.end(), std::mem_fun_ref(&vulkan_utils::buffer::reset));

        if (mDevice) {
            if (mCmdPool && mCommand) {
                vkFreeCommandBuffers(mDevice, mCmdPool, 1, &mCommand);
            }
        }
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

    void kernel_invocation::updateDescriptorSets(VkDescriptorSet literalSamplerDescSet,
                                                 VkDescriptorSet argumentDescSet) {
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
        literalSamplerSet.dstSet = literalSamplerDescSet;
        literalSamplerSet.dstBinding = 0;
        literalSamplerSet.descriptorCount = 1;
        literalSamplerSet.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;

        assert(mLiteralSamplers.empty() || literalSamplerDescSet);

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
        argSet.dstSet = argumentDescSet;
        argSet.dstBinding = 0;
        argSet.descriptorCount = 1;

        assert(mArguments.empty() || argumentDescSet);

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
                    argSet.pImageInfo = &(*nextImage);
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

    void kernel_invocation::fillCommandBuffer(const kernel_module&          module,
                                              const kernel&                 inKernel,
                                              const WorkgroupDimensions&    num_workgroups) {
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vulkan_utils::throwIfNotSuccess(vkBeginCommandBuffer(mCommand, &beginInfo),
                                        "vkBeginCommandBuffer");

        inKernel.bindPipeline(mCommand);
        module.bindDescriptors(mCommand);

        vkCmdDispatch(mCommand, num_workgroups.x, num_workgroups.y, 1);

        vulkan_utils::throwIfNotSuccess(vkEndCommandBuffer(mCommand),
                                        "vkEndCommandBuffer");
    }

    void kernel_invocation::submitCommand(VkQueue queue) {
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &mCommand;

        vulkan_utils::throwIfNotSuccess(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE),
                                        "vkQueueSubmit");

    }

    void kernel_invocation::run(VkQueue                     queue,
                                const kernel_module&        module,
                                const kernel&               kern,
                                const WorkgroupDimensions&  num_workgroups) {
        updateDescriptorSets(kern.getLiteralSamplerDescSet(), kern.getArgumentDescSet());
        fillCommandBuffer(module, kern, num_workgroups);
        submitCommand(queue);

        vkQueueWaitIdle(queue);
    }

} // namespace clspv_utils

/* ============================================================================================== */

template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
almost_equal(T x, T y, int ulp)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x-y) < std::numeric_limits<T>::epsilon() * std::abs(x+y) * ulp
           // unless the result is subnormal
           || std::abs(x-y) < std::numeric_limits<T>::min();
}

/* ============================================================================================== */

enum cl_channel_order {
    CL_R = 0x10B0,
    CL_A = 0x10B1,
    CL_RG = 0x10B2,
    CL_RA = 0x10B3,
    CL_RGB = 0x10B4,
    CL_RGBA = 0x10B5,
    CL_BGRA = 0x10B6,
    CL_ARGB = 0x10B7,
    CL_INTENSITY = 0x10B8,
    CL_LUMINANCE = 0x10B9,
    CL_Rx = 0x10BA,
    CL_RGx = 0x10BB,
    CL_RGBx = 0x10BC,
    CL_DEPTH = 0x10BD,
    CL_DEPTH_STENCIL = 0x10BE,
};

enum cl_channel_type {
    CL_SNORM_INT8 = 0x10D0,
    CL_SNORM_INT16 = 0x10D1,
    CL_UNORM_INT8 = 0x10D2,
    CL_UNORM_INT16 = 0x10D3,
    CL_UNORM_SHORT_565 = 0x10D4,
    CL_UNORM_SHORT_555 = 0x10D5,
    CL_UNORM_INT_101010 = 0x10D6,
    CL_SIGNED_INT8 = 0x10D7,
    CL_SIGNED_INT16 = 0x10D8,
    CL_SIGNED_INT32 = 0x10D9,
    CL_UNSIGNED_INT8 = 0x10DA,
    CL_UNSIGNED_INT16 = 0x10DB,
    CL_UNSIGNED_INT32 = 0x10DC,
    CL_HALF_FLOAT = 0x10DD,
    CL_FLOAT = 0x10DE,
    CL_UNORM_INT24 = 0x10DF,
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

/* ============================================================================================== */

std::pair<int,int> operator+=(std::pair<int,int>& l, const std::pair<int,int>& r) {
    l.first += r.first;
    l.second += r.second;
    return l;
};

/* ============================================================================================== */

template <typename T>
struct alignas(2*sizeof(T)) vec2 {
    vec2(T a, T b) : x(a), y(b) {}
    vec2(T a) : vec2(a, T(0)) {}
    vec2() : vec2(T(0), T(0)) {}

    vec2(const vec2<T>& other) : x(other.x), y(other.y) {}
    vec2(vec2<T>&& other) : vec2() {
        swap(*this, other);
    }

    vec2<T>& operator=(vec2<T> other) {
        swap(*this, other);
        return *this;
    }

    T   x;
    T   y;
};

template <typename T>
void swap(vec2<T>& first, vec2<T>& second) {
    using std::swap;

    swap(first.x, second.x);
    swap(first.y, second.y);
}

template <typename T>
bool operator==(const vec2<T>& l, const vec2<T>& r) {
    return (l.x == r.x) && (l.y == r.y);
}

template <typename T>
struct alignas(4*sizeof(T)) vec4 {
    vec4(T a, T b, T c, T d) : x(a), y(b), z(c), w(d) {}
    vec4(T a, T b, T c) : vec4(a, b, c, T(0)) {}
    vec4(T a, T b) : vec4(a, b, T(0), T(0)) {}
    vec4(T a) : vec4(a, T(0), T(0), T(0)) {}
    vec4() : vec4(T(0), T(0), T(0), T(0)) {}

    vec4(const vec4<T>& other) : x(other.x), y(other.y), z(other.z), w(other.w) {}
    vec4(vec4<T>&& other) : vec4() {
        swap(*this, other);
    }

    vec4<T>& operator=(vec4<T> other) {
        swap(*this, other);
        return *this;
    }

    T   x;
    T   y;
    T   z;
    T   w;
};

template <typename T>
void swap(vec4<T>& first, vec4<T>& second) {
    using std::swap;

    swap(first.x, second.x);
    swap(first.y, second.y);
    swap(first.z, second.z);
    swap(first.w, second.w);
}

template <typename T>
bool operator==(const vec4<T>& l, const vec4<T>& r) {
    return (l.x == r.x) && (l.y == r.y) && (l.z == r.z) && (l.w == r.w);
}

static_assert(sizeof(float) == 4, "bad size for float");

typedef vec2<float> float2;
static_assert(sizeof(float2) == 8, "bad size for float2");

template<>
bool operator==(const float2& l, const float2& r) {
    const int ulp = 2;
    return almost_equal(l.x, r.x, ulp)
           && almost_equal(l.y, r.y, ulp);
}

typedef vec4<float> float4;
static_assert(sizeof(float4) == 16, "bad size for float4");

template<>
bool operator==(const float4& l, const float4& r) {
    const int ulp = 2;
    return almost_equal(l.x, r.x, ulp)
           && almost_equal(l.y, r.y, ulp)
           && almost_equal(l.z, r.z, ulp)
           && almost_equal(l.w, r.w, ulp);
}

typedef half_float::half    half;
static_assert(sizeof(half) == 2, "bad size for half");

template <>
struct std::is_floating_point<half> : std::true_type {};
static_assert(std::is_floating_point<half>::value, "half should be floating point");

typedef vec2<half> half2;
static_assert(sizeof(half2) == 4, "bad size for half2");

typedef vec4<half> half4;
static_assert(sizeof(half4) == 8, "bad size for half4");

typedef unsigned short   ushort;
static_assert(sizeof(ushort) == 2, "bad size for ushort");

typedef vec2<ushort> ushort2;
static_assert(sizeof(ushort2) == 4, "bad size for ushort2");

typedef vec4<ushort> ushort4;
static_assert(sizeof(ushort4) == 8, "bad size for ushort4");

typedef unsigned char   uchar;
static_assert(sizeof(uchar) == 1, "bad size for uchar");

typedef vec2<uchar> uchar2;
static_assert(sizeof(uchar2) == 2, "bad size for uchar2");

typedef vec4<uchar> uchar4;
static_assert(sizeof(uchar4) == 4, "bad size for uchar4");

/* ============================================================================================== */

template <typename ComponentType, int N>
struct pixel_vector { };

template <typename ComponentType>
struct pixel_vector<ComponentType,1> {
    typedef ComponentType   type;
};

template <typename ComponentType>
struct pixel_vector<ComponentType,2> {
    typedef vec2<ComponentType> type;
};

template <typename ComponentType>
struct pixel_vector<ComponentType,4> {
    typedef vec4<ComponentType> type;
};

/* ============================================================================================== */

template <typename T>
struct pixel_traits {};

template <>
struct pixel_traits<float> {
    typedef float   component_t;

    static constexpr const int num_components = 1;
    static constexpr const char* const type_name = "float";

    static const cl_channel_order cl_pixel_order = CL_R;
    static const cl_channel_type cl_pixel_type = CL_FLOAT;
    static const VkFormat vk_pixel_type = VK_FORMAT_R32_SFLOAT;

    static float translate(const float& pixel) { return pixel; }

    static float translate(half pixel) {
        return pixel;
    }

    static float translate(uchar pixel) {
        return (pixel / (float) std::numeric_limits<uchar>::max());
    }

    template <typename T>
    static float translate(const vec2<T>& pixel) {
        return translate(pixel.x);
    }

    template <typename T>
    static float translate(const vec4<T>& pixel) {
        return translate(pixel.x);
    }
};

template <>
struct pixel_traits<float2> {
    typedef float   component_t;

    static constexpr const int num_components = 2;
    static constexpr const char* const type_name = "float2";

    static const cl_channel_order cl_pixel_order = CL_RG;
    static const cl_channel_type cl_pixel_type = pixel_traits<component_t>::cl_pixel_type;
    static const VkFormat vk_pixel_type = VK_FORMAT_R32G32_SFLOAT;

    template <typename T>
    static float2 translate(const vec4<T>& pixel) {
        return translate((vec2<T>){ pixel.x, pixel.y });
    }

    template <typename T>
    static float2 translate(const vec2<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y)
        };
    }

    template <typename T>
    static float2 translate(T pixel) {
        return {
                pixel_traits<component_t>::translate(pixel),
                component_t(0)
        };
    }
};

template <>
struct pixel_traits<float4> {
    typedef float   component_t;

    static constexpr const int num_components = 4;
    static constexpr const char* const type_name = "float4";

    static const int device_pixel_format = 1; // kDevicePixelFormat_BGRA_4444_32f
    static const cl_channel_order cl_pixel_order = CL_RGBA;
    static const cl_channel_type cl_pixel_type = pixel_traits<component_t>::cl_pixel_type;
    static const VkFormat vk_pixel_type = VK_FORMAT_R32G32B32A32_SFLOAT;

    template <typename T>
    static float4 translate(const vec4<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y),
                pixel_traits<component_t>::translate(pixel.z),
                pixel_traits<component_t>::translate(pixel.w)
        };
    }

    template <typename T>
    static float4 translate(const vec2<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y),
                component_t(0),
                component_t(0)
        };
    }

    template <typename T>
    static float4 translate(T pixel) {
        return {
                pixel_traits<component_t>::translate(pixel),
                component_t(0),
                component_t(0),
                component_t(0)
        };
    }
};

template <>
struct pixel_traits<half> {
    typedef half    component_t;

    static constexpr const int num_components = 1;
    static constexpr const char* const type_name = "half";

    static const cl_channel_order cl_pixel_order = CL_R;
    static const cl_channel_type cl_pixel_type = CL_HALF_FLOAT;
    static const VkFormat vk_pixel_type = VK_FORMAT_R16_SFLOAT;

    static half translate(float pixel) { return half(pixel); }

    static half translate(const half& pixel) { return pixel; }

    static half translate(uchar pixel) {
        return translate(pixel / (float) std::numeric_limits<uchar>::max());
    }

    template <typename T>
    static half translate(const vec2<T>& pixel) {
        return translate(pixel.x);
    }

    template <typename T>
    static half translate(const vec4<T>& pixel) {
        return translate(pixel.x);
    }
};

template <>
struct pixel_traits<half2> {
    typedef half    component_t;

    static constexpr const int num_components = 2;
    static constexpr const char* const type_name = "half2";

    static const cl_channel_order cl_pixel_order = CL_RG;
    static const cl_channel_type cl_pixel_type = pixel_traits<component_t>::cl_pixel_type;
    static const VkFormat vk_pixel_type = VK_FORMAT_R16G16_SFLOAT;

    template <typename T>
    static half2 translate(const vec4<T>& pixel) {
        return translate((vec2<T>){ pixel.x, pixel.y });
    }

    template <typename T>
    static half2 translate(const vec2<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y)
        };
    }

    template <typename T>
    static half2 translate(T pixel) {
        return {
                pixel_traits<component_t>::translate(pixel),
                component_t(0)
        };
    }
};

template <>
struct pixel_traits<half4> {
    typedef half    component_t;

    static constexpr const int num_components = 4;
    static constexpr const char* const type_name = "half4";

    static const int device_pixel_format = 0; // kDevicePixelFormat_BGRA_4444_16f
    static const cl_channel_order cl_pixel_order = CL_RGBA;
    static const cl_channel_type cl_pixel_type = pixel_traits<component_t>::cl_pixel_type;
    static const VkFormat vk_pixel_type = VK_FORMAT_R16G16B16A16_SFLOAT;

    template <typename T>
    static half4 translate(const vec4<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y),
                pixel_traits<component_t>::translate(pixel.z),
                pixel_traits<component_t>::translate(pixel.w)
        };
    }

    template <typename T>
    static half4 translate(const vec2<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y),
                component_t(0),
                component_t(0)
        };
    }

    template <typename T>
    static half4 translate(T pixel) {
        return {
                pixel_traits<component_t>::translate(pixel),
                component_t(0),
                component_t(0),
                component_t(0)
        };
    }
};

template <>
struct pixel_traits<ushort> {
    typedef ushort    component_t;

    static constexpr const int num_components = 1;
    static constexpr const char* const type_name = "ushort";

    static const cl_channel_order cl_pixel_order = CL_R;
    static const cl_channel_type cl_pixel_type = CL_UNSIGNED_INT16;
    static const VkFormat vk_pixel_type = VK_FORMAT_R16_UINT;

    static ushort translate(float pixel) {
        return (ushort) (pixel * std::numeric_limits<ushort>::max());
    }

    static ushort translate(ushort pixel) { return pixel; }

    template <typename T>
    static ushort translate(const vec2<T>& pixel) {
        return translate(pixel.x);
    }

    template <typename T>
    static ushort translate(const vec4<T>& pixel) {
        return translate(pixel.x);
    }
};

template <>
struct pixel_traits<ushort2> {
    typedef ushort    component_t;

    static constexpr const int num_components = 2;
    static constexpr const char* const type_name = "ushort2";

    static const cl_channel_order cl_pixel_order = CL_RG;
    static const cl_channel_type cl_pixel_type = pixel_traits<component_t>::cl_pixel_type;
    static const VkFormat vk_pixel_type = VK_FORMAT_R16G16_UINT;

    template <typename T>
    static ushort2 translate(const vec4<T>& pixel) {
        return translate((vec2<T>){ pixel.x, pixel.y });
    }

    template <typename T>
    static ushort2 translate(const vec2<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y)
        };
    }

    template <typename T>
    static ushort2 translate(T pixel) {
        return {
                pixel_traits<component_t>::translate(pixel),
                0
        };
    }
};

template <>
struct pixel_traits<ushort4> {
    typedef ushort    component_t;

    static constexpr const int num_components = 4;
    static constexpr const char* const type_name = "ushort4";

    static const cl_channel_order cl_pixel_order = CL_RGBA;
    static const cl_channel_type cl_pixel_type = pixel_traits<component_t>::cl_pixel_type;
    static const VkFormat vk_pixel_type = VK_FORMAT_R16G16B16A16_UINT;

    template <typename T>
    static ushort4 translate(const vec4<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y),
                pixel_traits<component_t>::translate(pixel.z),
                pixel_traits<component_t>::translate(pixel.w)
        };
    }

    template <typename T>
    static ushort4 translate(const vec2<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y),
                0,
                0
        };
    }

    template <typename T>
    static ushort4 translate(T pixel) {
        return {
                pixel_traits<component_t>::translate(pixel),
                0,
                0,
                0
        };
    }
};

template <>
struct pixel_traits<uchar> {
    typedef uchar    component_t;

    static constexpr const int num_components = 1;
    static constexpr const char* const type_name = "uchar";

    static const cl_channel_order cl_pixel_order = CL_R;
    static const cl_channel_type cl_pixel_type = CL_UNORM_INT8;
    static const VkFormat vk_pixel_type = VK_FORMAT_R8_UNORM;

    static uchar translate(float pixel) {
        return (uchar) (pixel * std::numeric_limits<uchar>::max());
    }

    static uchar translate(half pixel) {
        return (uchar) (pixel * std::numeric_limits<uchar>::max());
    }

    static uchar translate(uchar pixel) { return pixel; }

    template <typename T>
    static uchar translate(const vec2<T>& pixel) {
        return translate(pixel.x);
    }

    template <typename T>
    static uchar translate(const vec4<T>& pixel) {
        return translate(pixel.x);
    }
};

template <>
struct pixel_traits<uchar2> {
    typedef uchar    component_t;

    static constexpr const int num_components = 2;
    static constexpr const char* const type_name = "uchar2";

    static const cl_channel_order cl_pixel_order = CL_RG;
    static const cl_channel_type cl_pixel_type = pixel_traits<component_t>::cl_pixel_type;
    static const VkFormat vk_pixel_type = VK_FORMAT_R8G8_UNORM;

    template <typename T>
    static uchar2 translate(const vec4<T>& pixel) {
        return translate((vec2<T>){ pixel.x, pixel.y });
    }

    template <typename T>
    static uchar2 translate(const vec2<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y)
        };
    }

    template <typename T>
    static uchar2 translate(T pixel) {
        return {
                pixel_traits<component_t>::translate(pixel),
                0
        };
    }
};

template <>
struct pixel_traits<uchar4> {
    typedef uchar    component_t;

    static constexpr const int num_components = 4;
    static constexpr const char* const type_name = "uchar4";

    static const cl_channel_order cl_pixel_order = CL_RGBA;
    static const cl_channel_type cl_pixel_type = pixel_traits<component_t>::cl_pixel_type;
    static const VkFormat vk_pixel_type = VK_FORMAT_R8G8B8A8_UNORM;

    template <typename T>
    static uchar4 translate(const vec4<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y),
                pixel_traits<component_t>::translate(pixel.z),
                pixel_traits<component_t>::translate(pixel.w)
        };
    }

    template <typename T>
    static uchar4 translate(const vec2<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y),
                0,
                0
        };
    }

    template <typename T>
    static uchar4 translate(T pixel) {
        return {
                pixel_traits<component_t>::translate(pixel),
                0,
                0,
                0
        };
    }
};

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
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,    16 },
            { VK_DESCRIPTOR_TYPE_SAMPLER,           16 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,     16 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,     16 }
    };

    VkDescriptorPoolCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    createInfo.maxSets = 64;
    createInfo.poolSizeCount = sizeof(type_count) / sizeof(type_count[0]);
    createInfo.pPoolSizes = type_count;
    createInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    vulkan_utils::throwIfNotSuccess(vkCreateDescriptorPool(info.device,
                                                           &createInfo,
                                                           NULL,
                                                           &info.desc_pool),
                                    "vkCreateDescriptorPool");
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
    vulkan_utils::throwIfNotSuccess(vkCreateSampler(device, &samplerCreateInfo, NULL, &result),
                                    "vkCreateSampler");

    return result;
}

/* ============================================================================================== */

void invoke_fill_kernel(const clspv_utils::kernel_module&   module,
                        const clspv_utils::kernel&          kernel,
                        const sample_info&                  info,
                        const std::vector<VkSampler>&       samplers,
                        VkBuffer                            dst_buffer,
                        int                                 pitch,
                        int                                 device_format,
                        int                                 offset_x,
                        int                                 offset_y,
                        int                                 width,
                        int                                 height,
                        const float4&                       color) {
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
            pitch,
            device_format,
            offset_x,
            offset_y,
            width,
            height,
            color
    };

    const clspv_utils::WorkgroupDimensions workgroup_sizes = kernel.getWorkgroupSize();
    const clspv_utils::WorkgroupDimensions num_workgroups(
            (scalars.inWidth + workgroup_sizes.x - 1) / workgroup_sizes.x,
            (scalars.inHeight + workgroup_sizes.y - 1) / workgroup_sizes.y);

    clspv_utils::kernel_invocation invocation(info.device, info.cmd_pool, info.memory_properties);

    invocation.addLiteralSamplers(samplers.begin(), samplers.end());
    invocation.addBufferArgument(dst_buffer);
    invocation.addPodArgument(scalars);
    invocation.run(info.graphics_queue, module, kernel, num_workgroups);
}

void invoke_copybuffertoimage_kernel(const clspv_utils::kernel_module&   module,
                                     const clspv_utils::kernel&          kernel,
                                     const sample_info& info,
                                  const std::vector<VkSampler>& samplers,
                                  VkBuffer  src_buffer,
                                  VkImageView   dst_image,
                                  int src_offset,
                                  int src_pitch,
                                     cl_channel_order src_channel_order,
                                     cl_channel_type src_channel_type,
                                  bool swap_components,
                                  bool premultiply,
                                  int width,
                                  int height) {
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
            src_offset,
            src_pitch,
            src_channel_order,
            src_channel_type,
            (swap_components ? 1 : 0),
            (premultiply ? 1 : 0),
            width,
            height
    };

    const clspv_utils::WorkgroupDimensions workgroup_sizes = kernel.getWorkgroupSize();
    const clspv_utils::WorkgroupDimensions num_workgroups(
            (width + workgroup_sizes.x - 1) / workgroup_sizes.x,
            (height + workgroup_sizes.y - 1) / workgroup_sizes.y);

    clspv_utils::kernel_invocation invocation(info.device, info.cmd_pool, info.memory_properties);

    invocation.addLiteralSamplers(samplers.begin(), samplers.end());
    invocation.addBufferArgument(src_buffer);
    invocation.addWriteOnlyImageArgument(dst_image);
    invocation.addPodArgument(scalars);

    invocation.run(info.graphics_queue, module, kernel, num_workgroups);
}

void run_copybuffertoimage_kernel(const sample_info& info,
                                  const std::vector<VkSampler>& samplers,
                                  VkBuffer  src_buffer,
                                  VkImageView   dst_image,
                                  int src_offset,
                                  int src_pitch,
                                  cl_channel_order src_channel_order,
                                  cl_channel_type src_channel_type,
                                  bool swap_components,
                                  bool premultiply,
                                  int width,
                                  int height) {
    const clspv_utils::WorkgroupDimensions workgroup_sizes(32, 32);

    clspv_utils::kernel_module     module(info.device, info.desc_pool, "Memory");
    clspv_utils::kernel            kernel(info.device, module, "CopyBufferToImageKernel", workgroup_sizes);
    invoke_copybuffertoimage_kernel(module, kernel, info, samplers,
                                    src_buffer, dst_image,
                                    src_offset, src_pitch,
                                    src_channel_order, src_channel_type,
                                    swap_components,
                                    premultiply,
                                    width, height);
}

void invoke_copyimagetobuffer_kernel(const clspv_utils::kernel_module&   module,
                                     const clspv_utils::kernel&          kernel,
                                     const sample_info& info,
                                  const std::vector<VkSampler>& samplers,
                                  VkImageView src_image,
                                  VkBuffer dst_buffer,
                                  int dst_offset,
                                  int dst_pitch,
                                     cl_channel_order dst_channel_order,
                                     cl_channel_type dst_channel_type,
                                  bool swap_components,
                                  int width,
                                  int height) {
    struct scalar_args {
        int inDestOffset;       // offset 0
        int inDestPitch;        // offset 4
        int inDestChannelOrder; // offset 8 -- cl_channel_order
        int inDestChannelType;  // offset 12 -- cl_channel_type
        int inSwapComponents;   // offset 16 -- bool
        int inWidth;            // offset 20
        int inHeight;           // offset 24
    };
    static_assert(0 == offsetof(scalar_args, inDestOffset), "inDestOffset offset incorrect");
    static_assert(4 == offsetof(scalar_args, inDestPitch), "inDestPitch offset incorrect");
    static_assert(8 == offsetof(scalar_args, inDestChannelOrder), "inDestChannelOrder offset incorrect");
    static_assert(12 == offsetof(scalar_args, inDestChannelType), "inDestChannelType offset incorrect");
    static_assert(16 == offsetof(scalar_args, inSwapComponents), "inSwapComponents offset incorrect");
    static_assert(20 == offsetof(scalar_args, inWidth), "inWidth offset incorrect");
    static_assert(24 == offsetof(scalar_args, inHeight), "inHeight offset incorrect");

    const scalar_args scalars = {
            dst_offset,
            width,
            dst_channel_order,
            dst_channel_type,
            (swap_components ? 1 : 0),
            width,
            height
    };

    const clspv_utils::WorkgroupDimensions workgroup_sizes = kernel.getWorkgroupSize();
    const clspv_utils::WorkgroupDimensions num_workgroups(
            (width + workgroup_sizes.x - 1) / workgroup_sizes.x,
            (height + workgroup_sizes.y - 1) / workgroup_sizes.y);

    clspv_utils::kernel_invocation invocation(info.device, info.cmd_pool, info.memory_properties);

    invocation.addLiteralSamplers(samplers.begin(), samplers.end());
    invocation.addReadOnlyImageArgument(src_image);
    invocation.addBufferArgument(dst_buffer);
    invocation.addPodArgument(scalars);

    invocation.run(info.graphics_queue, module, kernel, num_workgroups);
}

void run_copyimagetobuffer_kernel(const sample_info& info,
                                  const std::vector<VkSampler>& samplers,
                                  VkImageView src_image,
                                  VkBuffer dst_buffer,
                                  int dst_offset,
                                  int dst_pitch,
                                  cl_channel_order dst_channel_order,
                                  cl_channel_type dst_channel_type,
                                  bool swap_components,
                                  int width,
                                  int height) {
    const clspv_utils::WorkgroupDimensions workgroup_sizes(32, 32);

    clspv_utils::kernel_module     module(info.device, info.desc_pool, "Memory");
    clspv_utils::kernel            kernel(info.device, module, "CopyImageToBufferKernel", workgroup_sizes);
    invoke_copyimagetobuffer_kernel(module, kernel, info, samplers,
                                    src_image, dst_buffer,
                                    dst_offset, dst_pitch,
                                    dst_channel_order, dst_channel_type,
                                    swap_components,
                                    width, height);
}

std::tuple<int,int,int> invoke_localsize_kernel(const clspv_utils::kernel_module&   module,
                                                const clspv_utils::kernel&          kernel,
                                                const sample_info&                  info,
                                                const std::vector<VkSampler>&       samplers) {
    struct scalar_args {
        int outWorkgroupX;  // offset 0
        int outWorkgroupY;  // offset 4
        int outWorkgroupZ;  // offset 8
    };
    static_assert(0 == offsetof(scalar_args, outWorkgroupX), "outWorkgroupX offset incorrect");
    static_assert(4 == offsetof(scalar_args, outWorkgroupY), "outWorkgroupY offset incorrect");
    static_assert(8 == offsetof(scalar_args, outWorkgroupZ), "outWorkgroupZ offset incorrect");

    vulkan_utils::buffer outArgs(info, sizeof(scalar_args));

    // The localsize kernel needs only a single workgroup with a single workitem
    const clspv_utils::WorkgroupDimensions num_workgroups(1, 1);

    clspv_utils::kernel_invocation invocation(info.device, info.cmd_pool, info.memory_properties);

    invocation.addBufferArgument(outArgs.buf);

    invocation.run(info.graphics_queue, module, kernel, num_workgroups);

    vulkan_utils::memory_map argMap(outArgs);
    auto outScalars = static_cast<const scalar_args*>(argMap.data);

    const auto result = std::make_tuple(outScalars->outWorkgroupX,
                                        outScalars->outWorkgroupY,
                                        outScalars->outWorkgroupZ);
    return result;
}


/* ============================================================================================== */

namespace test_utils {

    typedef std::pair<int,int>  Results;

    int count_successes(Results r) { return r.first; }
    int count_failures(Results r) { return r.second; }

    const Results   no_result = std::make_pair(0, 0);
    const Results   success = std::make_pair(1, 0);
    const Results   failure = std::make_pair(0, 1);

    typedef Results (*test_kernel_fn)(const clspv_utils::kernel_module& module,
                                      const clspv_utils::kernel&        kernel,
                                      const sample_info&                info,
                                      const std::vector<VkSampler>&     samplers,
                                      bool                              logIncorrect,
                                      bool                              logCorrect);

    struct kernel_test_map {
        std::string                         entry;
        test_kernel_fn                      test;
        clspv_utils::WorkgroupDimensions    workgroupSize;
    };

    struct module_test_bundle {
        std::string                     name;
        std::vector<kernel_test_map>    kernelTests;
    };

    template<typename ExpectedPixelType, typename ObservedPixelType>
    struct pixel_promotion {
        static constexpr const int expected_vec_size = pixel_traits<ExpectedPixelType>::num_components;
        static constexpr const int observed_vec_size = pixel_traits<ObservedPixelType>::num_components;
        static constexpr const int vec_size = (expected_vec_size > observed_vec_size
                                               ? observed_vec_size : expected_vec_size);

        typedef typename pixel_traits<ExpectedPixelType>::component_t expected_comp_type;
        typedef typename pixel_traits<ObservedPixelType>::component_t observed_comp_type;

        static constexpr const bool expected_is_smaller =
                sizeof(expected_comp_type) < sizeof(observed_comp_type);
        typedef typename std::conditional<expected_is_smaller, expected_comp_type, observed_comp_type>::type smaller_comp_type;
        typedef typename std::conditional<!expected_is_smaller, expected_comp_type, observed_comp_type>::type larger_comp_type;

        static constexpr const bool smaller_is_floating = std::is_floating_point<smaller_comp_type>::value;
        typedef typename std::conditional<smaller_is_floating, smaller_comp_type, larger_comp_type>::type comp_type;

        typedef typename pixel_vector<comp_type, vec_size>::type promotion_type;
    };

    template<typename T>
    struct pixel_comparator {
    };

    template<>
    struct pixel_comparator<float> {
        static bool is_equal(float l, float r) {
            const int ulp = 2;
            return almost_equal(l, r, ulp);
        }
    };

    template<>
    struct pixel_comparator<half> {
        static bool is_equal(half l, half r) {
            const int ulp = 2;
            return almost_equal(l, r, ulp);
        }
    };

    template<>
    struct pixel_comparator<uchar> {
        static bool is_equal(uchar l, uchar r) {
            return pixel_comparator<float>::is_equal(pixel_traits<float>::translate(l),
                                                     pixel_traits<float>::translate(r));
        }
    };

    template<typename T>
    struct pixel_comparator<vec2<T> > {
        static bool is_equal(const vec2<T> &l, const vec2<T> &r) {
            return pixel_comparator<T>::is_equal(l.x, r.x)
                   && pixel_comparator<T>::is_equal(l.y, r.y);
        }
    };

    template<typename T>
    struct pixel_comparator<vec4<T> > {
        static bool is_equal(const vec4<T> &l, const vec4<T> &r) {
            return pixel_comparator<T>::is_equal(l.x, r.x)
                   && pixel_comparator<T>::is_equal(l.y, r.y)
                   && pixel_comparator<T>::is_equal(l.z, r.z)
                   && pixel_comparator<T>::is_equal(l.w, r.w);
        }
    };

    template<typename T>
    bool pixel_compare(const T &l, const T &r) {
        return pixel_comparator<T>::is_equal(l, r);
    }

    template<typename ExpectedPixelType, typename ObservedPixelType>
    bool check_result(ExpectedPixelType expected_pixel,
                      ObservedPixelType observed_pixel,
                      const char *label,
                      int row,
                      int column,
                      bool logIncorrect = false,
                      bool logCorrect = false) {
        typedef typename pixel_promotion<ExpectedPixelType, ObservedPixelType>::promotion_type promotion_type;

        auto expected = pixel_traits<promotion_type>::translate(expected_pixel);
        auto observed = pixel_traits<promotion_type>::translate(observed_pixel);

        auto t_expected = pixel_traits<float4>::translate(expected);
        auto t_observed = pixel_traits<float4>::translate(observed);

        const bool pixel_is_correct = pixel_compare(observed, expected);
        if (pixel_is_correct) {
            if (logCorrect) {
                LOGE("%s:  CORRECT pixel{row:%d, col%d}", label, row, column);
            }
        } else {
            if (logIncorrect) {
                const float4 log_expected = pixel_traits<float4>::translate(expected_pixel);
                const float4 log_observed = pixel_traits<float4>::translate(observed_pixel);

                LOGE("%s: INCORRECT pixel{row:%d, col%d} expected{x=%f, y=%f, z=%f, w=%f} observed{x=%f, y=%f, z=%f, w=%f}",
                     label, row, column,
                     log_expected.x, log_expected.y, log_expected.z, log_expected.w,
                     log_observed.x, log_observed.y, log_observed.z, log_observed.w);
            }
        }

        return pixel_is_correct;
    }

    template<typename ObservedPixelType, typename ExpectedPixelType>
    bool check_results(const ObservedPixelType *observed_pixels,
                       int width,
                       int height,
                       int pitch,
                       ExpectedPixelType expected,
                       const char *label,
                       bool logIncorrect = false,
                       bool logCorrect = false) {
        unsigned int num_correct_pixels = 0;
        unsigned int num_incorrect_pixels = 0;

        auto row = observed_pixels;
        for (int r = 0; r < height; ++r, row += pitch) {
            auto p = row;
            for (int c = 0; c < width; ++c, ++p) {
                if (check_result(expected, *p, label, r, c, logIncorrect, logCorrect)) {
                    ++num_correct_pixels;
                } else {
                    ++num_incorrect_pixels;
                }
            }
        }

        LOGE("%s: Correct pixels=%d; Incorrect pixels=%d", label, num_correct_pixels,
             num_incorrect_pixels);

        return (0 == num_incorrect_pixels && 0 < num_correct_pixels);
    }

    template<typename ExpectedPixelType, typename ObservedPixelType>
    bool check_results(const ExpectedPixelType *expected_pixels,
                       const ObservedPixelType *observed_pixels,
                       int width,
                       int height,
                       int pitch,
                       const char *label,
                       bool logIncorrect = false,
                       bool logCorrect = false) {
        unsigned int num_correct_pixels = 0;
        unsigned int num_incorrect_pixels = 0;

        auto expected_row = expected_pixels;
        auto observed_row = observed_pixels;
        for (int r = 0; r < height; ++r, expected_row += pitch, observed_row += pitch) {
            auto expected_p = expected_row;
            auto observed_p = observed_row;
            for (int c = 0; c < width; ++c, ++expected_p, ++observed_p) {
                if (check_result(*expected_p, *observed_p, label, r, c, logIncorrect, logCorrect)) {
                    ++num_correct_pixels;
                } else {
                    ++num_incorrect_pixels;
                }
            }
        }

        LOGE("%s: Correct pixels=%d; Incorrect pixels=%d", label, num_correct_pixels,
             num_incorrect_pixels);

        return (0 == num_incorrect_pixels && 0 < num_correct_pixels);
    }

    template<typename ExpectedPixelType, typename ObservedPixelType>
    bool check_results(const vulkan_utils::device_memory &expected,
                       const vulkan_utils::device_memory &observed,
                       int width,
                       int height,
                       int pitch,
                       const char *label,
                       bool logIncorrect = false,
                       bool logCorrect = false) {
        vulkan_utils::memory_map src_map(expected);
        vulkan_utils::memory_map dst_map(observed);
        auto src_pixels = static_cast<const ExpectedPixelType *>(src_map.data);
        auto dst_pixels = static_cast<const ObservedPixelType *>(dst_map.data);

        return check_results(src_pixels, dst_pixels, width, height, pitch, label, logIncorrect,
                             logCorrect);
    }

    template<typename Fn>
    Results runInExceptionContext(const std::string& label, const std::string& stage, Fn f) {
        Results result = no_result;

        try {
            result = f();
        }
        catch(const vulkan_utils::error& e) {
            LOGE("%s/%s: Vulkan error (%s, VkResult=%d)",
                 label.c_str(), stage.c_str(),
                 e.what(), e.get_result());
            result = failure;
        }
        catch(const std::exception& e) {
            LOGE("%s/%s: unkonwn error (%s)", label.c_str(), stage.c_str(), e.what());
            result = failure;
        }
        catch(...) {
            LOGE("%s/%s: unknown error", label.c_str(), stage.c_str());
            result = failure;
        }

        return result;
    }

    template<typename ObservedPixelType>
    bool check_results(const vulkan_utils::device_memory &observed,
                       int width,
                       int height,
                       int pitch,
                       const float4 &expected,
                       const char *label,
                       bool logIncorrect = false,
                       bool logCorrect = false) {
        vulkan_utils::memory_map map(observed);
        auto pixels = static_cast<const ObservedPixelType *>(map.data);
        return check_results(pixels, width, height, pitch, expected, label, logIncorrect,
                             logCorrect);
    }

    Results test_kernel_invocation(const clspv_utils::kernel_module&    module,
                        const clspv_utils::kernel&                      kernel,
                        test_utils::test_kernel_fn                      testFn,
                        const sample_info&                              info,
                        const std::vector<VkSampler>&                   samplers,
                        bool                                            logIncorrect = false,
                        bool                                            logCorrect = false) {
        Results result = no_result;

        if (testFn) {
            result += runInExceptionContext(module.getName() + "/" + kernel.getEntryPoint(),
                                  "invoking kernel",
                                  [&]() {
                                      return testFn(module, kernel, info, samplers, logIncorrect, logCorrect);
                                  });
        }

        return result;
    }

    Results test_kernel_invocation(const clspv_utils::kernel_module&    module,
                                   const clspv_utils::kernel&           kernel,
                                   const test_utils::test_kernel_fn*    first,
                                   const test_utils::test_kernel_fn*    last,
                                   const sample_info&                   info,
                                   const std::vector<VkSampler>&        samplers,
                                   bool                                 logIncorrect = false,
                                   bool                                 logCorrect = false) {
        Results result = no_result;

        for (; first != last; ++first) {
            result += test_kernel_invocation(module, kernel, *first, info, samplers,
                                             logIncorrect, logCorrect);
        }

        return result;
    }

    Results test_kernel(const clspv_utils::kernel_module&       module,
                        const std::string&                      entryPoint,
                        test_utils::test_kernel_fn              testFn,
                        const clspv_utils::WorkgroupDimensions& numWorkgroups,
                        const sample_info&                      info,
                        const std::vector<VkSampler>&           samplers,
                        bool                                    logIncorrect = false,
                        bool                                    logCorrect = false) {
        return runInExceptionContext(module.getName() + "/" + entryPoint,
                                     "compiling kernel",
                                     [&]() {
                                         Results results = no_result;

                                         clspv_utils::kernel kernel(info.device, module,
                                                                    entryPoint, numWorkgroups);
                                         results += success;
                                         results += test_kernel_invocation(module,
                                                                           kernel,
                                                                           testFn,
                                                                           info,
                                                                           samplers,
                                                                           logIncorrect,
                                                                           logCorrect);

                                         return results;
                                     });
    }

    Results test_module(const std::string&                  moduleName,
                        const std::vector<kernel_test_map>& kernelTests,
                        const sample_info&                  info,
                        const std::vector<VkSampler>&       samplers,
                        bool                                logIncorrect = false,
                        bool                                logCorrect = false) {
        return runInExceptionContext(moduleName, "loading module", [&]() {
            Results result = no_result;

            clspv_utils::kernel_module module(info.device, info.desc_pool, moduleName);
            result += success;

            std::vector<std::string> entryPoints(module.getEntryPoints());
            for (auto ep : entryPoints) {
                const auto epTest = std::find_if(kernelTests.begin(), kernelTests.end(),
                                                 [&ep](const kernel_test_map& ktm) {
                                                     return ktm.entry == ep;
                                                 });

                result += test_kernel(module,
                                      ep,
                                      epTest == kernelTests.end() ? nullptr : epTest->test,
                                      epTest == kernelTests.end() ? clspv_utils::WorkgroupDimensions() : epTest->workgroupSize,
                                      info,
                                      samplers,
                                      logIncorrect,
                                      logCorrect);
            }

            return result;
        });
    }
} // namespace test_utils

/* ============================================================================================== */

test_utils::Results test_readlocalsize(const clspv_utils::kernel_module& module,
                                       const clspv_utils::kernel&        kernel,
                                       const sample_info&                info,
                                       const std::vector<VkSampler>&     samplers,
                                       bool                              logIncorrect = false,
                                       bool                              logCorrect = false) {
    const clspv_utils::WorkgroupDimensions expected = kernel.getWorkgroupSize();

    const auto observed = invoke_localsize_kernel(module, kernel, info, samplers);

    const bool success = (expected.x == std::get<0>(observed) &&
                          expected.y == std::get<1>(observed) &&
                          1 == std::get<2>(observed));
    if (success) {
        if (logCorrect) {
            const std::string label = module.getName() + "/" + kernel.getEntryPoint();
            LOGE("%s:  CORRECT workgroup_size{x:%d, y:%d, z:%d}", label.c_str(),
                 std::get<0>(observed), std::get<1>(observed), std::get<2>(observed));
        }
    }
    else {
        if (logIncorrect) {
            const std::string label = module.getName() + "/" + kernel.getEntryPoint();
            LOGE("%s: INCORRECT workgroup_size expected{x=%d, y=%d, z=1} observed{x=%d, y=%d, z=%d}",
                 label.c_str(),
                 expected.x, expected.y,
                 std::get<0>(observed), std::get<1>(observed), std::get<2>(observed));
        }
    }

    return (success ? std::make_pair(1, 0) : std::make_pair(0, 1));
};

template <typename PixelType>
test_utils::Results test_fill(const clspv_utils::kernel_module&  module,
                              const clspv_utils::kernel&         kernel,
                              const sample_info&            info,
                              const std::vector<VkSampler>& samplers,
                              bool                          logIncorrect = false,
                              bool                          logCorrect = false) {
    const std::string typeLabel = pixel_traits<PixelType>::type_name;

    std::string testLabel = "fills.spv/FillWithColorKernel/";
    testLabel += typeLabel;

    const int buffer_height = 64;
    const int buffer_width = 64;
    const float4 color = { 0.25f, 0.50f, 0.75f, 1.0f };

    // allocate image buffer
    const std::size_t buffer_size = buffer_width * buffer_height * sizeof(PixelType);
    vulkan_utils::buffer dst_buffer(info, buffer_size);

    {
        const PixelType src_value = pixel_traits<PixelType>::translate((float4){ 0.0f, 0.0f, 0.0f, 0.0f });

        vulkan_utils::memory_map dst_map(dst_buffer);
        auto dst_data = static_cast<PixelType*>(dst_map.data);
        std::fill(dst_data, dst_data + (buffer_width * buffer_height), src_value);
    }

    invoke_fill_kernel(module,
                       kernel,
                       info,
                       samplers,
                       dst_buffer.buf, // dst_buffer
                       buffer_width,   // pitch
                       pixel_traits<PixelType>::device_pixel_format, // device_format
                       0, 0, // offset_x, offset_y
                       buffer_width, buffer_height, // width, height
                       color); // color

    const bool success = test_utils::check_results<PixelType>(dst_buffer.mem,
                                                  buffer_width, buffer_height,
                                                  buffer_width,
                                                  color,
                                                  testLabel.c_str(),
                                                  logIncorrect,
                                                  logCorrect);

    dst_buffer.reset();

    return (success ? test_utils::success : std::make_pair(0, 1));
}

test_utils::Results test_fill_series(const clspv_utils::kernel_module&  module,
                                     const clspv_utils::kernel&         kernel,
                                     const sample_info&                 info,
                                     const std::vector<VkSampler>&      samplers,
                                     bool                               logIncorrect = false,
                                     bool                               logCorrect = false) {
    const test_utils::test_kernel_fn tests[] = {
            test_fill<float4>,
            test_fill<half4>,
    };

    return test_utils::test_kernel_invocation(module,
                                              kernel,
                                              std::begin(tests), std::end(tests),
                                              info,
                                              samplers,
                                              logIncorrect, logCorrect);
}

/* ============================================================================================== */

template <typename BufferPixelType, typename ImagePixelType>
test_utils::Results test_copytoimage_kernel(const sample_info&           info,
                                           const std::vector<VkSampler>& samplers,
                                           bool                          logIncorrect = false,
                                           bool                          logCorrect = false) {
    std::string typeLabel = pixel_traits<BufferPixelType>::type_name;
    typeLabel += '-';
    typeLabel += pixel_traits<ImagePixelType>::type_name;

    std::string testLabel = "memory.spv/CopyBufferToImageKernel/";
    testLabel += typeLabel;

    const int buffer_height = 64;
    const int buffer_width = 64;

    const std::size_t buffer_size = buffer_width * buffer_height * sizeof(BufferPixelType);

    // allocate buffers and images
    vulkan_utils::buffer  src_buffer(info, buffer_size);
    vulkan_utils::image   dstImage(info, buffer_width, buffer_height, pixel_traits<ImagePixelType>::vk_pixel_type);

    // initialize source and destination buffers
    {
        auto src_value = pixel_traits<BufferPixelType>::translate((float4){ 0.2f, 0.4f, 0.8f, 1.0f });
        vulkan_utils::memory_map src_map(src_buffer);
        auto src_data = static_cast<decltype(src_value)*>(src_map.data);
        std::fill(src_data, src_data + (buffer_width * buffer_height), src_value);
    }

    {
        auto dst_value = pixel_traits<ImagePixelType>::translate((float4){ 0.1f, 0.3f, 0.5f, 0.7f });
        vulkan_utils::memory_map dst_map(dstImage);
        auto dst_data = static_cast<decltype(dst_value)*>(dst_map.data);
        std::fill(dst_data, dst_data + (buffer_width * buffer_height), dst_value);
    }

    run_copybuffertoimage_kernel(info,
                                 samplers,
                                 src_buffer.buf,
                                 dstImage.view,
                                 0,
                                 buffer_width,
                                 pixel_traits<BufferPixelType>::cl_pixel_order,
                                 pixel_traits<BufferPixelType>::cl_pixel_type,
                                 false,
                                 false,
                                 buffer_width,
                                 buffer_height);

    const bool success = test_utils::check_results<BufferPixelType, ImagePixelType>(src_buffer.mem, dstImage.mem,
                                                                        buffer_width, buffer_height,
                                                                        buffer_height,
                                                                        testLabel.c_str(),
                                                                        logIncorrect,
                                                                        logCorrect);

    dstImage.reset();
    src_buffer.reset();

    return (success ? test_utils::success : std::make_pair(0, 1));
}

/* ============================================================================================== */

template <typename BufferPixelType, typename ImagePixelType>
test_utils::Results test_copyfromimage_kernel(const sample_info&            info,
                                             const std::vector<VkSampler>& samplers,
                                             bool                          logIncorrect = false,
                                             bool                          logCorrect = false) {
    std::string typeLabel = pixel_traits<BufferPixelType>::type_name;
    typeLabel += '-';
    typeLabel += pixel_traits<ImagePixelType>::type_name;

    std::string testLabel = "memory.spv/CopyImageToBufferKernel/";
    testLabel += typeLabel;

    const int buffer_height = 64;
    const int buffer_width = 64;

    const std::size_t buffer_size = buffer_width * buffer_height * sizeof(BufferPixelType);

    // allocate buffers and images
    vulkan_utils::buffer  dst_buffer(info, buffer_size);
    vulkan_utils::image   srcImage(info, buffer_width, buffer_height, pixel_traits<ImagePixelType>::vk_pixel_type);

    // initialize source and destination buffers
    {
        auto src_value = pixel_traits<ImagePixelType>::translate((float4){ 0.2f, 0.4f, 0.8f, 1.0f });
        vulkan_utils::memory_map src_map(srcImage);
        auto src_data = static_cast<decltype(src_value)*>(src_map.data);
        std::fill(src_data, src_data + (buffer_width * buffer_height), src_value);
    }

    {
        auto dst_value = pixel_traits<BufferPixelType>::translate((float4){ 0.1f, 0.3f, 0.5f, 0.7f });
        vulkan_utils::memory_map dst_map(dst_buffer);
        auto dst_data = static_cast<decltype(dst_value)*>(dst_map.data);
        std::fill(dst_data, dst_data + (buffer_width * buffer_height), dst_value);
    }

    run_copyimagetobuffer_kernel(info,
                                 samplers,
                                 srcImage.view,
                                 dst_buffer.buf,
                                 0,
                                 buffer_width,
                                 pixel_traits<BufferPixelType>::cl_pixel_order,
                                 pixel_traits<BufferPixelType>::cl_pixel_type,
                                 false,
                                 buffer_width,
                                 buffer_height);

    const bool success = test_utils::check_results<ImagePixelType, BufferPixelType>(srcImage.mem, dst_buffer.mem,
                                                                        buffer_width, buffer_height,
                                                                        buffer_height,
                                                                        testLabel.c_str(),
                                                                        logIncorrect,
                                                                        logCorrect);

    srcImage.reset();
    dst_buffer.reset();

    return (success ? test_utils::success : test_utils::failure);
}

/* ============================================================================================== */

struct test_t {
    typedef test_utils::Results (*fn)(const sample_info&, const std::vector<VkSampler>&, bool, bool);

    fn          func;
    std::string label;
};

test_utils::Results test_series(const test_t*                    first,
                                const test_t*                    last,
                                const sample_info&               info,
                                const std::vector<VkSampler>&    samplers,
                                bool                             logIncorrect = false,
                                bool                             logCorrect = false) {
    auto results = std::make_pair(0, 0);

    for (; first != last; ++first) {
        results += test_utils::runInExceptionContext(first->label.c_str(), "", [&]() {
            return (*first->func)(info, samplers, logIncorrect, logCorrect);
        });
    }

    return results;
};

template <typename ImagePixelType>
test_utils::Results test_copytoimage_series(const sample_info&           info,
                                           const std::vector<VkSampler>& samplers,
                                           bool                          logIncorrect = false,
                                           bool                          logCorrect = false) {
    std::string labelEnd = pixel_traits<ImagePixelType>::type_name;
    labelEnd += ">";

    const test_t tests[] = {
            { test_copytoimage_kernel<uchar, ImagePixelType>, "test_copytoimage_kernel<uchar, " + labelEnd },
            { test_copytoimage_kernel<uchar4, ImagePixelType>, "test_copytoimage_kernel<uchar4, " + labelEnd },
            { test_copytoimage_kernel<half, ImagePixelType>, "test_copytoimage_kernel<half, " + labelEnd },
            { test_copytoimage_kernel<half4, ImagePixelType>, "test_copytoimage_kernel<half4, " + labelEnd },
            { test_copytoimage_kernel<float, ImagePixelType>, "test_copytoimage_kernel<float, " + labelEnd },
            { test_copytoimage_kernel<float2, ImagePixelType>, "test_copytoimage_kernel<float2, " + labelEnd },
            { test_copytoimage_kernel<float4, ImagePixelType>, "test_copytoimage_kernel<float4, " + labelEnd },
    };

    return test_series(std::begin(tests), std::end(tests), info, samplers, logIncorrect, logCorrect);
}

template <typename ImagePixelType>
test_utils::Results test_copyfromimage_series(const sample_info&            info,
                                              const std::vector<VkSampler>& samplers,
                                              bool                          logIncorrect = false,
                                              bool                          logCorrect = false) {
    std::string labelEnd = pixel_traits<ImagePixelType>::type_name;
    labelEnd += ">";

    const test_t tests[] = {
            { test_copyfromimage_kernel<uchar, ImagePixelType>, "test_copyfromimage_kernel<uchar, " + labelEnd },
            { test_copyfromimage_kernel<uchar4, ImagePixelType>, "test_copyfromimage_kernel<uchar4, " + labelEnd },
            { test_copyfromimage_kernel<half, ImagePixelType>, "test_copyfromimage_kernel<half, " + labelEnd },
            { test_copyfromimage_kernel<half4, ImagePixelType>, "test_copyfromimage_kernel<half4, " + labelEnd },
            { test_copyfromimage_kernel<float, ImagePixelType>, "test_copyfromimage_kernel<float, " + labelEnd },
            { test_copyfromimage_kernel<float2, ImagePixelType>, "test_copyfromimage_kernel<float2, " + labelEnd },
            { test_copyfromimage_kernel<float4, ImagePixelType>, "test_copyfromimage_kernel<float4, " + labelEnd }
    };

    return test_series(std::begin(tests), std::end(tests), info, samplers, logIncorrect, logCorrect);
}

test_utils::Results test_copytoimage_matrix(const sample_info&            info,
                                            const std::vector<VkSampler>& samplers,
                                            bool                          logIncorrect = false,
                                            bool                          logCorrect = false) {
    const test_t tests[] = {
            { test_copytoimage_series<float4>, "test_copytoimage_series<float4>" },
            { test_copytoimage_series<half4>, "test_copytoimage_series<half4>" },
            { test_copytoimage_series<uchar4>, "test_copytoimage_series<uchar4>" },
            { test_copytoimage_series<float2>, "test_copytoimage_series<float2>" },
            { test_copytoimage_series<half2>, "test_copytoimage_series<half2>" },
            { test_copytoimage_series<uchar2>, "test_copytoimage_series<uchar2>" },
            { test_copytoimage_series<float>, "test_copytoimage_series<float>" },
            { test_copytoimage_series<half>, "test_copytoimage_series<half>" },
            { test_copytoimage_series<uchar>, "test_copytoimage_series<uchar>" },
    };

    return test_series(std::begin(tests), std::end(tests), info, samplers, logIncorrect, logCorrect);
}

test_utils::Results test_copyfromimage_matrix(const sample_info&            info,
                                              const std::vector<VkSampler>& samplers,
                                              bool                          logIncorrect = false,
                                              bool                          logCorrect = false) {
    const test_t tests[] = {
            { test_copyfromimage_series<float4>, "test_copyfromimage_series<float4>" },
            { test_copyfromimage_series<half4>, "test_copyfromimage_series<half4>" },
            { test_copyfromimage_series<uchar4>, "test_copyfromimage_series<uchar4>" },
            { test_copyfromimage_series<float2>, "test_copyfromimage_series<float2>" },
            { test_copyfromimage_series<half2>, "test_copyfromimage_series<half2>" },
            { test_copyfromimage_series<uchar2>, "test_copyfromimage_series<uchar2>" },
            { test_copyfromimage_series<float>, "test_copyfromimage_series<float>" },
            { test_copyfromimage_series<half>, "test_copyfromimage_series<half>" },
            { test_copyfromimage_series<uchar>, "test_copyfromimage_series<uchar>" },
    };

    return test_series(std::begin(tests), std::end(tests), info, samplers, logIncorrect, logCorrect);
}

/* ============================================================================================== */

const test_utils::module_test_bundle module_tests[] = {
        {
                "localsize", {
                                     { "ReadLocalSize", test_readlocalsize }
                             }
        },
        {
                "Fills", {
                                 { "FillWithColorKernel", test_fill_series }
                         }
        },
        {
                "Memory", {}
        },
};

test_utils::Results run_all_tests(const sample_info& info, const std::vector<VkSampler>& samplers) {
    auto test_results = test_utils::no_result;

    for (auto m : module_tests) {
        test_results += test_utils::test_module(m.name, m.kernelTests, info, samplers, false, false);
    }

    return test_results;
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


    auto test_results = run_all_tests(info, samplers);

    const test_t tests[] = {
            { test_copytoimage_matrix, "test_copytoimage_matrix" },
            { test_copyfromimage_matrix, "test_copyfromimage_matrix" },
    };

    test_results += test_series(std::begin(tests), std::end(tests), info, samplers);

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

    LOGI("Complete! %d tests passed. %d tests failed",
         test_utils::count_successes(test_results),
         test_utils::count_failures(test_results));

    return 0;
}
