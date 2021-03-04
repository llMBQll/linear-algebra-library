#pragma once

#include "backends/cuda/memory.hpp"
#include "math/pow.hpp"
#include "memory/AllocatorTraits.hpp"
#include "pow.cuh"

#include <algorithm>

namespace mbq
{
    template <typename T>
    struct Pow<cuda::Allocator<T>>
    {
        using value_type = value_type_of_t<cuda::Allocator<T>>;

        template <typename Iter>
        constexpr auto operator()(Iter x_begin, Iter x_end, const value_type& exponent) const
        {
            auto res = cuda::detail::pow(&(*x_begin), &(*x_begin), x_end - x_begin, exponent);
            if (res != cudaError::cudaSuccess)
                MBQ_THROW_EXCEPTION(CudaException, res);
            return x_end;
        }
    };
} // namespace mbq