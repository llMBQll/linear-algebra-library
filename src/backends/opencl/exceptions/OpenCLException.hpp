#pragma once

#include "backends/opencl/common.hpp"
#include "exceptions/Exception.hpp"

#define MBQ_OPENCL_STATUS_CASE(m__status)                                                                              \
    case (m__status):                                                                                                  \
        return #m__status

namespace mbq
{
    namespace detail
    {
        constexpr const char* opencl_get_error_string(cl_int status)
        {
            switch (status)
            {
                MBQ_OPENCL_STATUS_CASE(CL_SUCCESS);
                MBQ_OPENCL_STATUS_CASE(CL_DEVICE_NOT_FOUND);
                MBQ_OPENCL_STATUS_CASE(CL_DEVICE_NOT_AVAILABLE);
                MBQ_OPENCL_STATUS_CASE(CL_COMPILER_NOT_AVAILABLE);
                MBQ_OPENCL_STATUS_CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE);
                MBQ_OPENCL_STATUS_CASE(CL_OUT_OF_RESOURCES);
                MBQ_OPENCL_STATUS_CASE(CL_OUT_OF_HOST_MEMORY);
                MBQ_OPENCL_STATUS_CASE(CL_PROFILING_INFO_NOT_AVAILABLE);
                MBQ_OPENCL_STATUS_CASE(CL_MEM_COPY_OVERLAP);
                MBQ_OPENCL_STATUS_CASE(CL_IMAGE_FORMAT_MISMATCH);
                MBQ_OPENCL_STATUS_CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED);
                MBQ_OPENCL_STATUS_CASE(CL_BUILD_PROGRAM_FAILURE);
                MBQ_OPENCL_STATUS_CASE(CL_MAP_FAILURE);
#ifdef CL_VERSION_1_1
                MBQ_OPENCL_STATUS_CASE(CL_MISALIGNED_SUB_BUFFER_OFFSET);
                MBQ_OPENCL_STATUS_CASE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
#endif
#ifdef CL_VERSION_1_2
                MBQ_OPENCL_STATUS_CASE(CL_COMPILE_PROGRAM_FAILURE);
                MBQ_OPENCL_STATUS_CASE(CL_LINKER_NOT_AVAILABLE);
                MBQ_OPENCL_STATUS_CASE(CL_LINK_PROGRAM_FAILURE);
                MBQ_OPENCL_STATUS_CASE(CL_DEVICE_PARTITION_FAILED);
                MBQ_OPENCL_STATUS_CASE(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
#endif
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_VALUE);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_DEVICE_TYPE);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_PLATFORM);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_DEVICE);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_CONTEXT);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_QUEUE_PROPERTIES);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_COMMAND_QUEUE);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_HOST_PTR);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_MEM_OBJECT);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_IMAGE_SIZE);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_SAMPLER);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_BINARY);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_BUILD_OPTIONS);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_PROGRAM);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_PROGRAM_EXECUTABLE);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_KERNEL_NAME);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_KERNEL_DEFINITION);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_KERNEL);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_ARG_INDEX);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_ARG_VALUE);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_ARG_SIZE);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_KERNEL_ARGS);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_WORK_DIMENSION);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_WORK_GROUP_SIZE);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_WORK_ITEM_SIZE);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_GLOBAL_OFFSET);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_EVENT_WAIT_LIST);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_EVENT);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_OPERATION);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_GL_OBJECT);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_BUFFER_SIZE);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_MIP_LEVEL);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_GLOBAL_WORK_SIZE);
#ifdef CL_VERSION_1_1
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_PROPERTY);
#endif
#ifdef CL_VERSION_1_2
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_IMAGE_DESCRIPTOR);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_COMPILER_OPTIONS);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_LINKER_OPTIONS);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_DEVICE_PARTITION_COUNT);
#endif
#ifdef CL_VERSION_2_0
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_PIPE_SIZE);
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_DEVICE_QUEUE);
#endif
#ifdef CL_VERSION_2_2
                MBQ_OPENCL_STATUS_CASE(CL_INVALID_SPEC_ID);
                MBQ_OPENCL_STATUS_CASE(CL_MAX_SIZE_RESTRICTION_EXCEEDED);
#endif
            default:
                return "OPENCL_STATUS_UNKNOWN";
            }
        }
    } // namespace detail

    class OpenCLException : public Exception
    {
    public:
        using status_t = cl_int;
        constexpr inline static status_t UNKNOWN{1};
    private:
        status_t _status;
    public:
        explicit OpenCLException(const std::source_location& location = {}) : Exception(location), _status(UNKNOWN) { }

        explicit OpenCLException(status_t status, const std::source_location& location = {})
            : Exception(location), _status(status)
        { }

        OpenCLException(const OpenCLException&) noexcept = default;

        ~OpenCLException() noexcept override = default;

        [[nodiscard]] const char* what() const noexcept override
        {
            return detail::opencl_get_error_string(_status);
        }

        [[nodiscard]] status_t status() const noexcept
        {
            return _status;
        }
    };
} // namespace mbq