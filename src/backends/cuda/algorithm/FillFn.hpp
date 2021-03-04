#pragma once

#include "algorithm/fill.hpp"
#include "backends/cuda/exceptions/CudaException.hpp"
#include "backends/cuda/memory/Allocator.hpp"
#include "detail/fill.cuh"

namespace mbq
{
    template <typename T>
    struct Fill<cuda::Allocator<T>>
    {
        using value_type = T;

        template <std::output_iterator<const value_type&> O, std::sentinel_for<O> S>
        constexpr O operator()(O first, S last, const value_type& value) const
        {
            auto res = cuda::detail::fill(&(*first), last - first, value);
            if (res != cudaError::cudaSuccess)
                MBQ_THROW_EXCEPTION(CudaException, res);
            return last;
        }
    };
} // namespace mbq