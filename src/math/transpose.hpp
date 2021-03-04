#pragma once

#include "Array.hpp"
#include "util.hpp"

namespace mbq
{
    template <typename Allocator>
    struct Transpose
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };

    template <typename Allocator>
    inline constexpr Transpose<Allocator> transpose_impl;

    template <template <non_void, typename, size_t> typename X, non_void T, typename Allocator, size_t N>
    mbq::Array<T, Allocator, N> transpose(const X<T, Allocator, N>& x)
    {
        auto rows = x.dimensions()[0];
        auto cols = x.dimensions()[1];
        auto tmp = mbq::Array<T, Allocator, N>(cols, rows, x.get_allocator());

        transpose_impl<Allocator>(x.cbegin(), x.cend(), tmp.begin(), tmp.end(), rows, cols);

        return tmp;
    }
} // namespace mbq