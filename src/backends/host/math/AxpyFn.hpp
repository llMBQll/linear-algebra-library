#pragma once

#include "math/axpy.hpp"
#include "memory/AllocatorTraits.hpp"

#include <openblas/cblas.h>

namespace mbq
{
    template <typename Allocator, size_t N>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    struct Axpy<Allocator, N>
    {
        using value_type = value_type_of_t<Allocator>;

        template <typename ConstIter, typename Iter>
        void operator()(const value_type& alpha, ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter) const
        {
            const value_type* x = &(*x_begin);
            value_type* y = &(*y_begin);
            auto n = static_cast<blasint>(x_end - x_begin);

            if constexpr (std::is_same_v<value_type, float>)
                return cblas_saxpy(n, alpha, x, 1, y, 1);
            else if constexpr (std::is_same_v<value_type, double>)
                return cblas_daxpy(n, alpha, x, 1, y, 1);
            else if constexpr (std::is_same_v<value_type, std::complex<float>>)
                return cblas_caxpy(n, std::bit_cast<const void*>(&alpha), std::bit_cast<const void*>(x), 1,
                                   std::bit_cast<void*>(y), 1);
            else if constexpr (std::is_same_v<value_type, std::complex<double>>)
                return cblas_zaxpy(n, std::bit_cast<const void*>(&alpha), std::bit_cast<const void*>(x), 1,
                                   std::bit_cast<void*>(y), 1);
        }
    };
} // namespace mbq