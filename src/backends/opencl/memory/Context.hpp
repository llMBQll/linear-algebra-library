#pragma once

#include "../common.hpp"

namespace mbq::opencl
{
    class Context
    {
    public:
        cl_platform_id platform_id = nullptr;
        cl_device_id device_id = nullptr;
        cl_context context = nullptr;
        cl_command_queue command_queue = nullptr;
    public:
        Context()
        {
            cl_uint ret_num_devices;
            cl_uint ret_num_platforms;
            cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
            ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
            context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret);
            command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
        }

        ~Context()
        {
            clFlush(command_queue);
            clFinish(command_queue);
            clReleaseCommandQueue(command_queue);
            clReleaseContext(context);
            clReleaseDevice(device_id);
        }

        static Context* get_default()
        {
            static Context ctx;
            return &ctx;
        }
    };
} // namespace mbq::opencl