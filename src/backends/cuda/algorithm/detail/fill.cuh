#pragma once

#include <cuda_runtime.h>

namespace mbq::cuda::detail
{
    template <typename T>
    cudaError_t fill(T* ptr, size_t count, T value);
} // namespace mbq::cuda::detail