#pragma once

#include "math/arithmetic.hpp"
#include "memory/AllocatorTraits.hpp"

#include <openblas/cblas.h>

namespace mbq
{
    template <typename Allocator>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    struct Add<Allocator>
    {
        using value_type = value_type_of_t<Allocator>;

        template <typename ConstIterX, typename ConstIterY, typename Iter>
        void operator()(ConstIterX x_begin, ConstIterX x_end, ConstIterY y_begin, ConstIterY, Iter z_begin, Iter) const
        {
            while (x_begin != x_end)
            {
                *z_begin = *x_begin + *y_begin;
                ++x_begin;
                ++y_begin;
                ++z_begin;
            }
        }

        template <typename Iter, typename ConstIter>
        void operator()(ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter, const value_type& value) const
        {
            while (x_begin != x_end)
            {
                *y_begin = *x_begin + value;
                ++x_begin;
                ++y_begin;
            }
        }
    };

    template <typename Allocator>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    struct Subtract<Allocator>
    {
        using value_type = value_type_of_t<Allocator>;

        template <typename ConstIterX, typename ConstIterY, typename Iter>
        void operator()(ConstIterX x_begin, ConstIterX x_end, ConstIterY y_begin, ConstIterY, Iter z_begin, Iter) const
        {
            while (x_begin != x_end)
            {
                *z_begin = *x_begin - *y_begin;
                ++x_begin;
                ++y_begin;
                ++z_begin;
            }
        }

        template <typename Iter, typename ConstIter>
        void operator()(ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter, const value_type& value) const
        {
            while (x_begin != x_end)
            {
                *y_begin = *x_begin - value;
                ++x_begin;
                ++y_begin;
            }
        }

        template <typename Iter, typename ConstIter>
        void operator()(const value_type& value, ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter) const
        {
            while (x_begin != x_end)
            {
                *y_begin = value - *x_begin;
                ++x_begin;
                ++y_begin;
            }
        }
    };

    namespace host::detail
    {
        template <typename T>
        void multiply(int m, int n, int k, const T* A, const T* B, T* C)
        {
            T alpha{1};
            T beta{0};

            if constexpr (std::is_same_v<T, float>)
                return cblas_sgemm(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans,
                                   CBLAS_TRANSPOSE::CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);
            else if constexpr (std::is_same_v<T, double>)
                return cblas_dgemm(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans,
                                   CBLAS_TRANSPOSE::CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);
            else if constexpr (std::is_same_v<T, std::complex<float>>)
                return cblas_cgemm(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans,
                                   CBLAS_TRANSPOSE::CblasNoTrans, m, n, k, static_cast<const void*>(&alpha),
                                   static_cast<const void*>(A), k, static_cast<const void*>(B), n,
                                   static_cast<const void*>(&beta), static_cast<void*>(C), n);
            else if constexpr (std::is_same_v<T, std::complex<double>>)
                return cblas_zgemm(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans,
                                   CBLAS_TRANSPOSE::CblasNoTrans, m, n, k, static_cast<const void*>(&alpha),
                                   static_cast<const void*>(A), k, static_cast<const void*>(B), n,
                                   static_cast<const void*>(&beta), static_cast<void*>(C), n);
        }
    } // namespace host::detail

    template <typename Allocator>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    struct Multiply<Allocator>
    {
        using value_type = value_type_of_t<Allocator>;

        template <typename ConstIterX, typename ConstIterY, typename Iter>
        void operator()(ConstIterX x_begin, ConstIterX, ConstIterY y_begin, ConstIterY, Iter z_begin, Iter, int m,
                        int n, int k) const
        {
            const value_type* x = &(*x_begin);
            const value_type* y = &(*y_begin);
            value_type* z = &(*z_begin);

            host::detail::multiply(m, n, k, x, y, z);
        }

        template <typename ConstIterX, typename ConstIterY, typename Iter>
        void operator()(ConstIterX x_begin, ConstIterX x_end, ConstIterY y_begin, ConstIterY, Iter z_begin, Iter) const
        {
            while (x_begin != x_end)
            {
                *z_begin = *x_begin * *y_begin;
                ++x_begin;
                ++y_begin;
                ++z_begin;
            }
        }

        template <typename Iter, typename ConstIter>
        void operator()(ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter, const value_type& value) const
        {
            while (x_begin != x_end)
            {
                *y_begin = *x_begin * value;
                ++x_begin;
                ++y_begin;
            }
        }
    };

    template <typename Allocator>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    struct Divide<Allocator>
    {
        using value_type = value_type_of_t<Allocator>;

        template <typename ConstIterX, typename ConstIterY, typename Iter>
        void operator()(ConstIterX x_begin, ConstIterX x_end, ConstIterY y_begin, ConstIterY, Iter z_begin, Iter) const
        {
            while (x_begin != x_end)
            {
                *z_begin = *x_begin / *y_begin;
                ++x_begin;
                ++y_begin;
                ++z_begin;
            }
        }

        template <typename Iter, typename ConstIter>
        void operator()(ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter, const value_type& value) const
        {
            while (x_begin != x_end)
            {
                *y_begin = *x_begin / value;
                ++x_begin;
                ++y_begin;
            }
        }

        template <typename Iter, typename ConstIter>
        void operator()(const value_type& value, ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter) const
        {
            while (x_begin != x_end)
            {
                *y_begin = value / *x_begin;
                ++x_begin;
                ++y_begin;
            }
        }
    };
} // namespace mbq