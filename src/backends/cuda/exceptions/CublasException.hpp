#pragma once

#include "exceptions/Exception.hpp"

#include <cublas_v2.h>

#define MBQ_CUBLAS_STATUS_CASE(m__status)                                                                              \
    case m__status:                                                                                                    \
        return #m__status

namespace mbq
{
    namespace detail
    {
        constexpr const char* cublas_get_error_string(cublasStatus_t status)
        {
            switch (status)
            {
                MBQ_CUBLAS_STATUS_CASE(CUBLAS_STATUS_SUCCESS);
                MBQ_CUBLAS_STATUS_CASE(CUBLAS_STATUS_NOT_INITIALIZED);
                MBQ_CUBLAS_STATUS_CASE(CUBLAS_STATUS_ALLOC_FAILED);
                MBQ_CUBLAS_STATUS_CASE(CUBLAS_STATUS_INVALID_VALUE);
                MBQ_CUBLAS_STATUS_CASE(CUBLAS_STATUS_ARCH_MISMATCH);
                MBQ_CUBLAS_STATUS_CASE(CUBLAS_STATUS_MAPPING_ERROR);
                MBQ_CUBLAS_STATUS_CASE(CUBLAS_STATUS_EXECUTION_FAILED);
                MBQ_CUBLAS_STATUS_CASE(CUBLAS_STATUS_INTERNAL_ERROR);
                MBQ_CUBLAS_STATUS_CASE(CUBLAS_STATUS_NOT_SUPPORTED);
                MBQ_CUBLAS_STATUS_CASE(CUBLAS_STATUS_LICENSE_ERROR);
            default:
                return "CUBLAS_STATUS_UNKNOWN";
            }
        }
    } // namespace detail

    class CublasException : public Exception
    {
    private:
        cublasStatus_t _status;
    public:
        explicit CublasException(const std::source_location& location = {})
            : Exception(location), _status(static_cast<cublasStatus_t>(-1))
        { }

        explicit CublasException(cublasStatus_t status, const std::source_location& location = {})
            : Exception(location), _status(status)
        { }

        CublasException(const CublasException&) noexcept = default;

        ~CublasException() noexcept override = default;

        [[nodiscard]] const char* what() const noexcept override
        {
            return detail::cublas_get_error_string(_status);
        }

        [[nodiscard]] cublasStatus_t status() const noexcept
        {
            return _status;
        }
    };
} // namespace mbq
