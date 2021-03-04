#pragma once

#include "backends/opencl/exceptions/CLBlastException.hpp"
#include "backends/opencl/exceptions/OpenCLException.hpp"
#include "backends/opencl/kernels/Kernel.hpp"

#include <clblast.h>

namespace mbq
{
    template <typename T>
    struct Add<opencl::Allocator<T>>
    {
        using value_type = T;

        template <typename ConstIterX, typename ConstIterY, typename Iter>
        void operator()(ConstIterX x_begin, ConstIterX x_end, ConstIterY y_begin, ConstIterY, Iter z_begin, Iter) const
        {
            auto state = x_begin.get_allocator().state();
            opencl::detail::add(state, &(*x_begin), &(*y_begin), &(*z_begin), x_end - x_begin);
        }

        template <typename ConstIter, typename Iter>
        void operator()(ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter, const value_type& value) const
        {
            auto state = x_begin.get_allocator().state();
            opencl::detail::add_arg(state, &(*x_begin), &(*y_begin), x_end - x_begin, value);
        }
    };

    template <typename T>
    struct Subtract<opencl::Allocator<T>>
    {
        using value_type = T;

        template <typename ConstIterX, typename ConstIterY, typename Iter>
        void operator()(ConstIterX x_begin, ConstIterX x_end, ConstIterY y_begin, ConstIterY, Iter z_begin, Iter) const
        {
            auto state = x_begin.get_allocator().state();
            opencl::detail::subtract(state, &(*x_begin), &(*y_begin), &(*z_begin), x_end - x_begin);
        }

        template <typename ConstIter, typename Iter>
        void operator()(ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter, const value_type& value) const
        {
            auto state = x_begin.get_allocator().state();
            opencl::detail::subtract_arg(state, &(*x_begin), &(*y_begin), x_end - x_begin, value);
        }

        template <typename ConstIter, typename Iter>
        void operator()(const value_type& value, ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter) const
        {
            auto state = x_begin.get_allocator().state();
            opencl::detail::subtract_reverse_arg(state, &(*x_begin), &(*y_begin), x_end - x_begin, value);
        }
    };

    template <typename T>
    struct Multiply<opencl::Allocator<T>>
    {
        using value_type = T;

        template <typename ConstIterX, typename ConstIterY, typename Iter>
        void operator()(ConstIterX x_begin, ConstIterX, ConstIterY y_begin, ConstIterY, Iter z_begin, Iter, int m,
                        int n, int k) const
        {
            auto [x, x_off] = (&(*x_begin)).get();
            auto [y, y_off] = (&(*y_begin)).get();
            auto [z, z_off] = (&(*z_begin)).get();
            auto queue = x_begin.get_allocator().state()->command_queue;
            T alpha{1};
            T beta{0};

            auto status = clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo, m,
                                        n, k, alpha, x, x_off, k, y, y_off, n, beta, z, z_off, n, &queue);
            if (status != clblast::StatusCode::kSuccess)
                MBQ_THROW_EXCEPTION(CLBlastException, status);
        }

        template <typename ConstIterX, typename ConstIterY, typename Iter>
        void operator()(ConstIterX x_begin, ConstIterX x_end, ConstIterY y_begin, ConstIterY, Iter z_begin, Iter) const
        {
            auto state = x_begin.get_allocator().state();
            opencl::detail::multiply(state, &(*x_begin), &(*y_begin), &(*z_begin), x_end - x_begin);
        }

        template <typename ConstIter, typename Iter>
        void operator()(ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter, const value_type& value) const
        {
            auto state = x_begin.get_allocator().state();
            opencl::detail::multiply_arg(state, &(*x_begin), &(*y_begin), x_end - x_begin, value);
        }
    };

    template <typename T>
    struct Divide<opencl::Allocator<T>>
    {
        using value_type = T;

        template <typename ConstIterX, typename ConstIterY, typename Iter>
        void operator()(ConstIterX x_begin, ConstIterX x_end, ConstIterY y_begin, ConstIterY, Iter z_begin, Iter) const
        {
            auto state = x_begin.get_allocator().state();
            opencl::detail::divide(state, &(*x_begin), &(*y_begin), &(*z_begin), x_end - x_begin);
        }

        template <typename ConstIter, typename Iter>
        void operator()(ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter, const value_type& value) const
        {
            auto state = x_begin.get_allocator().state();
            opencl::detail::divide_arg(state, &(*x_begin), &(*y_begin), x_end - x_begin, value);
        }

        template <typename ConstIter, typename Iter>
        void operator()(const value_type& value, ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter) const
        {
            auto state = x_begin.get_allocator().state();
            opencl::detail::divide_reverse_arg(state, &(*x_begin), &(*y_begin), x_end - x_begin, value);
        }
    };
} // namespace mbq