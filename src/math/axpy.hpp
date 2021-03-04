#pragma once

#include "Array.hpp"
#include "util.hpp"

namespace mbq
{
    template <typename Allocator, size_t N>
    struct Axpy
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };

    template <typename Allocator, size_t N>
    inline constexpr Axpy<Allocator, N> axpy_impl;

    template <typename A, template <non_void, typename, size_t> typename X,
              template <non_void, typename, size_t> typename Y, non_void T, typename Allocator, size_t N>
        requires convertible_to<A, T>
    Array<T, Allocator, N> axpy(A alpha, const X<T, Allocator, N>& x, const Y<T, Allocator, N>& y)
    {
        Array<T, Allocator, N> tmp{y};
        axpy_impl<Allocator, N>(static_cast<T>(alpha), x.begin(), x.end(), tmp.begin(), tmp.end());
        return tmp;
    }
} // namespace mbq