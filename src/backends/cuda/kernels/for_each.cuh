#pragma once

#include <cuda_runtime.h>
#include <utility>

#ifndef MBQ_CUDA_WAIT_FOR_KERNEL_FINISHED
    #define MBQ_CUDA_WAIT_FOR_KERNEL_FINISHED 0
#endif

namespace mbq::cuda::detail
{
    using dim_size_type = unsigned int;

    template <typename T>
    __global__ void for_each(T* inout, size_t size, T value)
    {
        dim_size_type idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < size)
            inout[idx] = value;
    }

    template <typename T, typename UnaryFn>
    __global__ void for_each(const T* in, T* out, size_t size)
    {
        dim_size_type idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < size)
            out[idx] = UnaryFn{}(in[idx]);
    }

    template <typename T, typename UnaryFn>
    __global__ void for_each(const T* in, T* out, size_t size, UnaryFn fn)
    {
        dim_size_type idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < size)
            out[idx] = fn(in[idx]);
    }

    template <typename T, typename BinaryFn>
    __global__ void for_each(const T* x_in, const T* y_in, T* out, size_t size)
    {
        dim_size_type idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < size)
            out[idx] = BinaryFn{}(x_in[idx], y_in[idx]);
    }

    template <typename T, typename BinaryFn>
    __global__ void for_each(const T* x_in, const T* y_in, T* out, size_t size, BinaryFn fn)
    {
        dim_size_type idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < size)
            out[idx] = fn(x_in[idx], y_in[idx]);
    }

    inline cudaError_t kernel_last_error() noexcept
    {
#if MBQ_CUDA_WAIT_FOR_KERNEL_FINISHED
        return cudaDeviceSynchronize();
#else
        return cudaError::cudaSuccess;
#endif
    }

    inline std::pair<dim_size_type, dim_size_type> get_launch_params(size_t size) noexcept
    {
        // use maximum block size which is 1'024
        dim_size_type block_size = 1'024;
        dim_size_type num_blocks = (size + block_size - 1) / block_size;
        return {num_blocks, block_size};
    }

    template <typename T>
    inline cudaError_t launch_for_each(T* inout, size_t size, T value) noexcept
    {
        auto [num_blocks, block_size] = get_launch_params(size);
        for_each<T><<<num_blocks, block_size>>>(inout, size, value);
        return kernel_last_error();
    }

    template <typename T, typename UnaryFn>
    inline cudaError_t launch_for_each(const T* in, T* out, size_t size) noexcept
    {
        auto [num_blocks, block_size] = get_launch_params(size);
        for_each<T, UnaryFn><<<num_blocks, block_size>>>(in, out, size);
        return kernel_last_error();
    }

    template <typename T>
    inline cudaError_t launch_for_each(const T* in, T* out, size_t size, T value) noexcept
    {
        auto [num_blocks, block_size] = get_launch_params(size);
        for_each<T><<<num_blocks, block_size>>>(in, out, size, value);
        return kernel_last_error();
    }

    template <typename T, typename UnaryFn>
    inline cudaError_t launch_for_each(const T* in, T* out, size_t size, UnaryFn fn) noexcept
    {
        auto [num_blocks, block_size] = get_launch_params(size);
        for_each<T, UnaryFn><<<num_blocks, block_size>>>(in, out, size, fn);
        return kernel_last_error();
    }

    template <typename T, typename BinaryFn>
    inline cudaError_t launch_for_each(const T* x_in, const T* y_in, T* out, size_t size) noexcept
    {
        auto [num_blocks, block_size] = get_launch_params(size);
        for_each<T, BinaryFn><<<num_blocks, block_size>>>(x_in, y_in, out, size);
        return kernel_last_error();
    }

    template <typename T, typename BinaryFn>
    inline cudaError_t launch_for_each(const T* x_in, const T* y_in, T* out, size_t size, BinaryFn fn) noexcept
    {
        auto [num_blocks, block_size] = get_launch_params(size);
        for_each<T, BinaryFn><<<num_blocks, block_size>>>(x_in, y_in, out, size, fn);
        return kernel_last_error();
    }
} // namespace mbq::cuda::detail