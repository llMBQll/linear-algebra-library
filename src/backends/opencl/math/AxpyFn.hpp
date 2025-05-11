#pragma once

#include "backends/opencl/exceptions/OpenCLException.hpp"
#include "math/axpy.hpp"

#include <clblast.h>

namespace mbq
{
    template <typename T, size_t N>
    struct Axpy<opencl::Allocator<T>, N>
    {
        using value_type = T;

        template <typename ConstIter, typename Iter>
        void operator()(T alpha, ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter) const
        {
            auto [x, x_off] = (&(*x_begin)).get();
            auto [y, y_off] = (&(*y_begin)).get();
            auto n = static_cast<size_t>(x_end - x_begin);
            auto queue = y_begin.get_allocator().state()->command_queue;

            auto status = clblast::Axpy(n, alpha, x, x_off, 1, y, y_off, 1, &queue);
            if (status != clblast::StatusCode::kSuccess)
                MBQ_THROW_EXCEPTION(CLBlastException, status);
        }
    };
} // namespace mbq