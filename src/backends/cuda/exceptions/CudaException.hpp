#pragma once

#include "exceptions/Exception.hpp"

#include <cuda_runtime.h>

namespace mbq
{
    class CudaException : public Exception
    {
    private:
        cudaError_t _error;
    public:
        CudaException(const std::source_location& location = {})
            : Exception(cudaGetErrorString(cudaError::cudaErrorUnknown), location), _error(cudaError::cudaErrorUnknown)
        { }

        CudaException(cudaError_t error, const std::source_location& location = {})
            : Exception(cudaGetErrorString(error), location), _error(error)
        { }

        CudaException(const CudaException&) noexcept = default;

        ~CudaException() noexcept override = default;

        [[nodiscard]] cudaError_t error() const noexcept
        {
            return _error;
        }
    };
} // namespace mbq