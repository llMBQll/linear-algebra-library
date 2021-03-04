#pragma once

#include "../../../memory/Copy.hpp"
#include "backends/opencl/exceptions/OpenCLException.hpp"
#include "concepts.hpp"
#include "Context.hpp"
#include "Pointer.hpp"

#include <vector>

namespace mbq
{
    template <non_void T>
    class Copy<opencl::Allocator<T>, opencl::Allocator<T>>
    {
    public:
        using dst_pointer = pointer_of_t<opencl::Allocator<T>>;
        using src_pointer = const_pointer_of_t<opencl::Allocator<T>>;
        using value_type = value_type_of_t<opencl::Allocator<T>>;
    public:
        void operator()(dst_pointer dst, opencl::Allocator<T> dst_alloc, src_pointer src, opencl::Allocator<T>,
                        size_t count) const
        {
            auto [dst_ptr, dst_off] = dst.get();
            auto [src_ptr, src_off] = src.get();

            auto res =
                clEnqueueCopyBuffer(dst_alloc.state()->command_queue, src_ptr, dst_ptr, src_off * sizeof(value_type),
                                    dst_off * sizeof(value_type), count * sizeof(value_type), 0, nullptr, nullptr);
            if (res != CL_SUCCESS)
                MBQ_THROW_EXCEPTION(OpenCLException, res);
        }
    };

    template <non_void T, typename Allocator>
        requires host_allocator<T, Allocator>
    class Copy<opencl::Allocator<T>, Allocator>
    {
    public:
        using dst_pointer = pointer_of_t<opencl::Allocator<T>>;
        using src_pointer = const_pointer_of_t<Allocator>;
        using value_type = value_type_of_t<opencl::Allocator<T>>;
    public:
        void operator()(dst_pointer dst, opencl::Allocator<T> dst_alloc, src_pointer src, Allocator, size_t count) const
        {
            auto [dst_ptr, dst_off] = dst.get();

            auto res = clEnqueueWriteBuffer(dst_alloc.state()->command_queue, dst_ptr, CL_BLOCKING,
                                            dst_off * sizeof(value_type), count * sizeof(value_type), src, 0, nullptr,
                                            nullptr);
            if (res != CL_SUCCESS)
                MBQ_THROW_EXCEPTION(OpenCLException, res);
        }
    };

    template <non_void T, typename Allocator>
        requires host_allocator<T, Allocator>
    class Copy<Allocator, opencl::Allocator<T>>
    {
    public:
        using dst_pointer = pointer_of_t<Allocator>;
        using src_pointer = const_pointer_of_t<opencl::Allocator<T>>;
        using value_type = value_type_of_t<opencl::Allocator<T>>;
    public:
        void operator()(dst_pointer dst, Allocator, src_pointer src, opencl::Allocator<T> src_alloc, size_t count) const
        {
            auto [src_ptr, src_off] = src.get();

            auto res =
                clEnqueueReadBuffer(src_alloc.state()->command_queue, src_ptr, CL_BLOCKING,
                                    src_off * sizeof(value_type), count * sizeof(value_type), dst, 0, nullptr, nullptr);
            if (res != CL_SUCCESS)
                MBQ_THROW_EXCEPTION(OpenCLException, res);
        }
    };
} // namespace mbq