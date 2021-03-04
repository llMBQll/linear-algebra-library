#pragma once

#include "backends/opencl/memory/Allocator.hpp"
#include "math/transpose.hpp"

#include <clblast.h>

namespace mbq
{
    namespace opencl::detail
    {
        template <typename T>
        clblast::StatusCode transpose(cl_mem in, size_t in_off, cl_mem out, size_t out_off, size_t rows, size_t cols,
                                      cl_command_queue queue)
        {
            const T alpha{1};

            return clblast::Omatcopy<T>(clblast::Layout::kRowMajor, clblast::Transpose::kYes, rows, cols, alpha, in,
                                        in_off, cols, out, out_off, rows, &queue);
        }
    } // namespace opencl::detail

    template <typename T>
    class Transpose<mbq::opencl::Allocator<T>>
    {
    public:
        using value_type = T;
    public:
        template <typename ConstIter, typename Iter>
        void operator()(ConstIter in_begin, ConstIter, Iter out_begin, Iter, size_t rows, size_t cols) const
        {
            auto [in, in_off] = (&(*in_begin)).get();
            auto [out, out_off] = (&(*out_begin)).get();
            auto queue = out_begin.get_allocator().state()->command_queue;

            auto status = opencl::detail::transpose<T>(in, in_off, out, out_off, rows, cols, queue);
            if (status != clblast::StatusCode::kSuccess)
                MBQ_THROW_EXCEPTION(CLBlastException, status);
        }
    };
} // namespace mbq