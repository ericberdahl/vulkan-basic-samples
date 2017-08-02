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

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <util_init.hpp>

VkShaderModule create_shader(VkDevice device, const char* spvFileName) {
    std::FILE* spv_file = AndroidFopen(spvFileName, "rb");

    std::fseek(spv_file, 0, SEEK_END);
    std::vector<char> spvModule(ftell(spv_file));

    std::fseek(spv_file, 0, SEEK_SET);
    std::fread(spvModule.data(), 1, spvModule.size(), spv_file);

    std::fclose(spv_file);

    VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.pNext = NULL;
    shaderModuleCreateInfo.flags = 0;
    shaderModuleCreateInfo.codeSize = spvModule.size();
    shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(spvModule.data());

    VkShaderModule shaderModule;
    VkResult U_ASSERT_ONLY res = vkCreateShaderModule(device, &shaderModuleCreateInfo, NULL, &shaderModule);
    assert(res == VK_SUCCESS);

    return shaderModule;
}

int sample_main(int argc, char *argv[]) {
    struct sample_info info = {};
    init_global_layer_properties(info);
    init_instance(info, "vulkansamples_device");

    init_enumerate_device(info);

    /* VULKAN_KEY_START */

    VkDeviceQueueCreateInfo queue_info = {};

    vkGetPhysicalDeviceQueueFamilyProperties(info.gpus[0], &info.queue_family_count, NULL);
    assert(info.queue_family_count >= 1);

    info.queue_props.resize(info.queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(info.gpus[0], &info.queue_family_count, info.queue_props.data());
    assert(info.queue_family_count >= 1);

    bool found = false;
    for (unsigned int i = 0; i < info.queue_family_count; i++) {
        if (info.queue_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queue_info.queueFamilyIndex = i;
            found = true;
            break;
        }
    }
    assert(found);
    assert(info.queue_family_count >= 1);

    info.device_extension_names.push_back("VK_KHR_variable_pointers");

    float queue_priorities[1] = {0.0};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.pNext = NULL;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = queue_priorities;

    VkDeviceCreateInfo device_info = {};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = NULL;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = info.device_extension_names.size();
    device_info.ppEnabledExtensionNames = device_info.enabledExtensionCount ? info.device_extension_names.data() : NULL;
    device_info.enabledLayerCount = 0;
    device_info.ppEnabledLayerNames = NULL;
    device_info.pEnabledFeatures = NULL;

    VkDevice device;
    VkResult U_ASSERT_ONLY res = vkCreateDevice(info.gpus[0], &device_info, NULL, &device);
    assert(res == VK_SUCCESS);

    VkShaderModule compute_shader = create_shader(device, "fills.spv");

    vkDestroyDevice(device, NULL);

    /* VULKAN_KEY_END */

    destroy_instance(info);

    return 0;
}
