#pragma once

#include "exceptions/Exception.hpp"

#include <clblast.h>

#define MBQ_CLBLAST_STATUS_CASE(m__status)                                                                             \
    case (m__status):                                                                                                  \
        return #m__status

namespace mbq
{
    namespace detail
    {
        constexpr const char* clblast_get_error_string(clblast::StatusCode status)
        {
            using enum clblast::StatusCode;
            switch (status)
            {
                MBQ_CLBLAST_STATUS_CASE(kSuccess);
                MBQ_CLBLAST_STATUS_CASE(kTempBufferAllocFailure);
                MBQ_CLBLAST_STATUS_CASE(kOpenCLOutOfResources);
                MBQ_CLBLAST_STATUS_CASE(kOpenCLOutOfHostMemory);
                MBQ_CLBLAST_STATUS_CASE(kOpenCLBuildProgramFailure);
                MBQ_CLBLAST_STATUS_CASE(kInvalidValue);
                MBQ_CLBLAST_STATUS_CASE(kInvalidCommandQueue);
                MBQ_CLBLAST_STATUS_CASE(kInvalidMemObject);
                MBQ_CLBLAST_STATUS_CASE(kInvalidBinary);
                MBQ_CLBLAST_STATUS_CASE(kInvalidBuildOptions);
                MBQ_CLBLAST_STATUS_CASE(kInvalidProgram);
                MBQ_CLBLAST_STATUS_CASE(kInvalidProgramExecutable);
                MBQ_CLBLAST_STATUS_CASE(kInvalidKernelName);
                MBQ_CLBLAST_STATUS_CASE(kInvalidKernelDefinition);
                MBQ_CLBLAST_STATUS_CASE(kInvalidKernel);
                MBQ_CLBLAST_STATUS_CASE(kInvalidArgIndex);
                MBQ_CLBLAST_STATUS_CASE(kInvalidArgValue);
                MBQ_CLBLAST_STATUS_CASE(kInvalidArgSize);
                MBQ_CLBLAST_STATUS_CASE(kInvalidKernelArgs);
                MBQ_CLBLAST_STATUS_CASE(kInvalidLocalNumDimensions);
                MBQ_CLBLAST_STATUS_CASE(kInvalidLocalThreadsTotal);
                MBQ_CLBLAST_STATUS_CASE(kInvalidLocalThreadsDim);
                MBQ_CLBLAST_STATUS_CASE(kInvalidGlobalOffset);
                MBQ_CLBLAST_STATUS_CASE(kInvalidEventWaitList);
                MBQ_CLBLAST_STATUS_CASE(kInvalidEvent);
                MBQ_CLBLAST_STATUS_CASE(kInvalidOperation);
                MBQ_CLBLAST_STATUS_CASE(kInvalidBufferSize);
                MBQ_CLBLAST_STATUS_CASE(kInvalidGlobalWorkSize);

                // Status codes in common with the clBLAS library
                MBQ_CLBLAST_STATUS_CASE(kNotImplemented);
                MBQ_CLBLAST_STATUS_CASE(kInvalidMatrixA);
                MBQ_CLBLAST_STATUS_CASE(kInvalidMatrixB);
                MBQ_CLBLAST_STATUS_CASE(kInvalidMatrixC);
                MBQ_CLBLAST_STATUS_CASE(kInvalidVectorX);
                MBQ_CLBLAST_STATUS_CASE(kInvalidVectorY);
                MBQ_CLBLAST_STATUS_CASE(kInvalidDimension);
                MBQ_CLBLAST_STATUS_CASE(kInvalidLeadDimA);
                MBQ_CLBLAST_STATUS_CASE(kInvalidLeadDimB);
                MBQ_CLBLAST_STATUS_CASE(kInvalidLeadDimC);
                MBQ_CLBLAST_STATUS_CASE(kInvalidIncrementX);
                MBQ_CLBLAST_STATUS_CASE(kInvalidIncrementY);
                MBQ_CLBLAST_STATUS_CASE(kInsufficientMemoryA);
                MBQ_CLBLAST_STATUS_CASE(kInsufficientMemoryB);
                MBQ_CLBLAST_STATUS_CASE(kInsufficientMemoryC);
                MBQ_CLBLAST_STATUS_CASE(kInsufficientMemoryX);
                MBQ_CLBLAST_STATUS_CASE(kInsufficientMemoryY);

                // Custom additional status codes for CLBlast
                MBQ_CLBLAST_STATUS_CASE(kInsufficientMemoryTemp);
                MBQ_CLBLAST_STATUS_CASE(kInvalidBatchCount);
                MBQ_CLBLAST_STATUS_CASE(kInvalidOverrideKernel);
                MBQ_CLBLAST_STATUS_CASE(kMissingOverrideParameter);
                MBQ_CLBLAST_STATUS_CASE(kInvalidLocalMemUsage);
                MBQ_CLBLAST_STATUS_CASE(kNoHalfPrecision);
                MBQ_CLBLAST_STATUS_CASE(kNoDoublePrecision);
                MBQ_CLBLAST_STATUS_CASE(kInvalidVectorScalar);
                MBQ_CLBLAST_STATUS_CASE(kInsufficientMemoryScalar);
                MBQ_CLBLAST_STATUS_CASE(kDatabaseError);
                MBQ_CLBLAST_STATUS_CASE(kUnknownError);
                MBQ_CLBLAST_STATUS_CASE(kUnexpectedError);
            default:
                return "CLBLAST_STATUS_UNKNOWN";
            }
        }
    } // namespace detail

    class CLBlastException : public Exception
    {
    public:
        using status_t = clblast::StatusCode;
        constexpr inline static status_t UNKNOWN{1};
    private:
        status_t _status;
    public:
        explicit CLBlastException(const std::source_location& location = {}) : Exception(location), _status(UNKNOWN) { }

        explicit CLBlastException(status_t status, const std::source_location& location = {})
            : Exception(location), _status(status)
        { }

        CLBlastException(const CLBlastException&) noexcept = default;

        ~CLBlastException() noexcept override = default;

        [[nodiscard]] const char* what() const noexcept override
        {
            return detail::clblast_get_error_string(_status);
        }
    };
} // namespace mbq
