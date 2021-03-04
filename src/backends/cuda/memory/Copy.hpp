#pragma once

#include "../../../memory/Copy.hpp"
#include "backends/cuda/exceptions/CudaException.hpp"
#include "concepts.hpp"

#include <cuda_runtime.h>

namespace mbq
{
    template <non_void T>
    class Copy<cuda::Allocator<T>, cuda::Allocator<T>>
    {
    public:
        using dst_pointer = pointer_of_t<cuda::Allocator<T>>;
        using src_pointer = const_pointer_of_t<cuda::Allocator<T>>;
        using value_type = value_type_of_t<cuda::Allocator<T>>;
    public:
        void operator()(dst_pointer dst, cuda::Allocator<T>, src_pointer src, cuda::Allocator<T>, size_t count) const
        {
            auto res = cudaMemcpy(dst, src, count * sizeof(value_type), cudaMemcpyDeviceToDevice);
            if (res != cudaError::cudaSuccess)
                MBQ_THROW_EXCEPTION(CudaException, res);
        }
    };

    template <non_void T, typename Allocator>
        requires host_allocator<T, Allocator>
    class Copy<cuda::Allocator<T>, Allocator>
    {
    public:
        using dst_pointer = pointer_of_t<cuda::Allocator<T>>;
        using src_pointer = const_pointer_of_t<Allocator>;
        using value_type = value_type_of_t<cuda::Allocator<T>>;
    public:
        void operator()(dst_pointer dst, cuda::Allocator<T>, src_pointer src, Allocator, size_t count) const
        {
            auto res = cudaMemcpy(dst, src, count * sizeof(value_type), cudaMemcpyHostToDevice);
            if (res != cudaError::cudaSuccess)
                MBQ_THROW_EXCEPTION(CudaException, res);
        }
    };

    template <non_void T, typename Allocator>
        requires host_allocator<T, Allocator>
    class Copy<Allocator, cuda::Allocator<T>>
    {
    public:
        using dst_pointer = pointer_of_t<Allocator>;
        using src_pointer = const_pointer_of_t<cuda::Allocator<T>>;
        using value_type = value_type_of_t<cuda::Allocator<T>>;
    public:
        void operator()(dst_pointer dst, Allocator, src_pointer src, cuda::Allocator<T>, size_t count) const
        {
            auto res = cudaMemcpy(dst, src, count * sizeof(value_type), cudaMemcpyDeviceToHost);
            if (res != cudaError::cudaSuccess)
                MBQ_THROW_EXCEPTION(CudaException, res);
        }
    };
} // namespace mbq