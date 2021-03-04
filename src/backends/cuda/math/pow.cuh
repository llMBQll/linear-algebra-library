#pragma once

namespace mbq::cuda::detail
{
    template <typename T>
    cudaError_t pow(const T* in, T* out, size_t cnt, const T& exponent);
} // namespace mbq::cuda::detail
