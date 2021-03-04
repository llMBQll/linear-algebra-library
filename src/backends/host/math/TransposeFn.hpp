#pragma once

#include "math/transpose.hpp"
#include "memory/AllocatorTraits.hpp"

#include <openblas/cblas.h>

namespace mbq
{
    namespace host::detail
    {
        template <typename T>
        void transpose(const T* in, T* out, blasint rows, blasint cols)
        {
            const T alpha{1};

            if constexpr (std::is_same_v<T, float>)
                return cblas_somatcopy(CblasRowMajor, CblasTrans, rows, cols, alpha, in, cols, out, rows);
            else if constexpr (std::is_same_v<T, double>)
                return cblas_domatcopy(CblasRowMajor, CblasTrans, rows, cols, alpha, in, cols, out, rows);
            else if constexpr (std::is_same_v<T, std::complex<float>>)
                return cblas_comatcopy(CblasRowMajor, CblasTrans, rows, cols, std::bit_cast<const float*>(&alpha),
                                       std::bit_cast<const float*>(in), cols, std::bit_cast<float*>(out), rows);
            else if constexpr (std::is_same_v<T, std::complex<double>>)
                return cblas_zomatcopy(CblasRowMajor, CblasTrans, rows, cols, std::bit_cast<const double*>(&alpha),
                                       std::bit_cast<const double*>(in), cols, std::bit_cast<double*>(out), rows);
        }
    } // namespace host::detail

    template <typename Allocator>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    class Transpose<Allocator>
    {
    public:
        using value_type = value_type_of_t<Allocator>;
    public:
        template <typename ConstIter, typename Iter>
        void operator()(ConstIter in_begin, ConstIter, Iter out_begin, Iter, size_t rows, size_t cols) const
        {
            const value_type* in = &(*in_begin);
            value_type* out = &(*out_begin);

            host::detail::transpose(in, out, static_cast<blasint>(rows), static_cast<blasint>(cols));
        }
    };
} // namespace mbq