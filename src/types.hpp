#pragma once

#include "Array.hpp"
#include "ArrayView.hpp"
#include "backends/host/memory.hpp"
#include "memory/Memory.hpp"
#include "memory/MemoryView.hpp"

#if MBQ_CUDA_BACKEND == ON
    #include "backends/cuda/memory.hpp"
#endif

#if MBQ_OPENCL_BACKEND == ON
    #include "backends/opencl/memory.hpp"
#endif

namespace mbq
{
    template <non_void T, size_t N>
    using HostArray = mbq::Array<T, mbq::host::Allocator<T>, N>;

    template <non_void T, size_t N>
    using HostArrayView = mbq::ArrayView<T, mbq::host::Allocator<T>, N>;

    template <non_void T>
    using HostMemory = mbq::Memory<T, mbq::host::Allocator<T>>;

    template <non_void T>
    using HostMemoryView = mbq::MemoryView<T, mbq::host::Allocator<T>>;

#if MBQ_CUDA_BACKEND == ON
    template <non_void T, size_t N>
    using CudaArray = mbq::Array<T, mbq::cuda::Allocator<T>, N>;

    template <non_void T, size_t N>
    using CudaArrayView = mbq::ArrayView<T, mbq::cuda::Allocator<T>, N>;

    template <non_void T>
    using CudaMemory = mbq::Memory<T, mbq::cuda::Allocator<T>>;

    template <non_void T>
    using CudaMemoryView = mbq::MemoryView<T, mbq::cuda::Allocator<T>>;
#endif

#if MBQ_OPENCL_BACKEND == ON
    template <non_void T, size_t N>
    using OpenCLArray = mbq::Array<T, mbq::opencl::Allocator<T>, N>;

    template <non_void T, size_t N>
    using OpenCLArrayView = mbq::ArrayView<T, mbq::opencl::Allocator<T>, N>;

    template <non_void T>
    using OpenCLMemory = mbq::Memory<T, mbq::opencl::Allocator<T>>;

    template <non_void T>
    using OpenCLMemoryView = mbq::MemoryView<T, mbq::opencl::Allocator<T>>;
#endif
} // namespace mbq