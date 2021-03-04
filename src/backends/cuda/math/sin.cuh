#pragma once

namespace mbq::cuda::detail
{
    template <typename T>
    cudaError_t sin(const T* in, T* out, size_t cnt);

    template <typename T>
    cudaError_t sinh(const T* in, T* out, size_t cnt);

    template <typename T>
    cudaError_t cos(const T* in, T* out, size_t cnt);

    template <typename T>
    cudaError_t cosh(const T* in, T* out, size_t cnt);

    template <typename T>
    cudaError_t tan(const T* in, T* out, size_t cnt);

    template <typename T>
    cudaError_t tanh(const T* in, T* out, size_t cnt);
} // namespace mbq::cuda::detail
