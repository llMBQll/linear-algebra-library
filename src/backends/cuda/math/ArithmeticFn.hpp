#pragma once

#include "arithmetic.cuh"
#include "backends/cuda/exceptions/CublasException.hpp"
#include "backends/cuda/exceptions/CudaException.hpp"
#include "backends/cuda/memory.hpp"
#include "math/arithmetic.hpp"

namespace mbq
{
    template <typename T>
    struct Add<cuda::Allocator<T>>
    {
        using value_type = T;

        template <typename ConstIterX, typename ConstIterY, typename Iter>
        void operator()(ConstIterX x_begin, ConstIterX x_end, ConstIterY y_begin, ConstIterY, Iter z_begin, Iter) const
        {
            auto res = cuda::detail::add(&(*x_begin), &(*y_begin), &(*z_begin), x_end - x_begin);
            if (res != cudaError::cudaSuccess)
                MBQ_THROW_EXCEPTION(CudaException, res);
        }

        template <typename Iter, typename ConstIter>
        void operator()(ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter, const value_type& value) const
        {
            auto res = cuda::detail::add(&(*x_begin), &(*y_begin), value, x_end - x_begin);
            if (res != cudaError::cudaSuccess)
                MBQ_THROW_EXCEPTION(CudaException, res);
        }
    };

    template <typename T>
    struct Subtract<cuda::Allocator<T>>
    {
        using value_type = T;

        template <typename ConstIterX, typename ConstIterY, typename Iter>
        void operator()(ConstIterX x_begin, ConstIterX x_end, ConstIterY y_begin, ConstIterY, Iter z_begin, Iter) const
        {
            auto res = cuda::detail::subtract(&(*x_begin), &(*y_begin), &(*z_begin), x_end - x_begin);
            if (res != cudaError::cudaSuccess)
                MBQ_THROW_EXCEPTION(CudaException, res);
        }

        template <typename Iter, typename ConstIter>
        void operator()(ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter, const value_type& value) const
        {
            auto res = cuda::detail::subtract(&(*x_begin), &(*y_begin), value, x_end - x_begin);
            if (res != cudaError::cudaSuccess)
                MBQ_THROW_EXCEPTION(CudaException, res);
        }

        template <typename Iter, typename ConstIter>
        void operator()(const value_type& value, ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter) const
        {
            auto res = cuda::detail::subtract(value, &(*x_begin), &(*y_begin), x_end - x_begin);
            if (res != cudaError::cudaSuccess)
                MBQ_THROW_EXCEPTION(CudaException, res);
        }
    };

    template <typename T>
    struct Multiply<cuda::Allocator<T>>
    {
        using value_type = T;

        template <typename ConstIterX, typename ConstIterY, typename Iter>
        void operator()(ConstIterX x_begin, ConstIterX, ConstIterY y_begin, ConstIterY, Iter z_begin, Iter, int m,
                        int n, int k) const
        {
            const T* x = &(*x_begin);
            const T* y = &(*y_begin);
            T* z = &(*z_begin);
            auto handle = y_begin.get_allocator().state()->handle;

            auto status = cuda::detail::multiply(handle, m, n, k, x, y, z);
            if (status != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
                MBQ_THROW_EXCEPTION(CublasException, status);
        }

        template <typename ConstIterX, typename ConstIterY, typename Iter>
        void operator()(ConstIterX x_begin, ConstIterX x_end, ConstIterY y_begin, ConstIterY, Iter z_begin, Iter) const
        {
            auto res = cuda::detail::multiply(&(*x_begin), &(*y_begin), &(*z_begin), x_end - x_begin);
            if (res != cudaError::cudaSuccess)
                MBQ_THROW_EXCEPTION(CudaException, res);
        }

        template <typename Iter, typename ConstIter>
        void operator()(ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter, const value_type& value) const
        {
            auto res = cuda::detail::multiply(&(*x_begin), &(*y_begin), value, x_end - x_begin);
            if (res != cudaError::cudaSuccess)
                MBQ_THROW_EXCEPTION(CudaException, res);
        }
    };

    template <typename T>
    struct Divide<cuda::Allocator<T>>
    {
        using value_type = T;

        template <typename ConstIterX, typename ConstIterY, typename Iter>
        void operator()(ConstIterX x_begin, ConstIterX x_end, ConstIterY y_begin, ConstIterY, Iter z_begin, Iter) const
        {
            auto res = cuda::detail::divide(&(*x_begin), &(*y_begin), &(*z_begin), x_end - x_begin);
            if (res != cudaError::cudaSuccess)
                MBQ_THROW_EXCEPTION(CudaException, res);
        }

        template <typename Iter, typename ConstIter>
        void operator()(ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter, const value_type& value) const
        {
            auto res = cuda::detail::divide(&(*x_begin), &(*y_begin), value, x_end - x_begin);
            if (res != cudaError::cudaSuccess)
                MBQ_THROW_EXCEPTION(CudaException, res);
        }

        template <typename Iter, typename ConstIter>
        void operator()(const value_type& value, ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter) const
        {
            auto res = cuda::detail::divide(value, &(*x_begin), &(*y_begin), x_end - x_begin);
            if (res != cudaError::cudaSuccess)
                MBQ_THROW_EXCEPTION(CudaException, res);
        }
    };
} // namespace mbq