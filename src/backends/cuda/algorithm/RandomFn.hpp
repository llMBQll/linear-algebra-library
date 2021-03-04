#pragma once

#include "algorithm/random.hpp"
#include "backends/cuda/memory/Allocator.hpp"
#include "detail/random.cuh"

namespace mbq
{
    template <typename T>
    struct Random<cuda::Allocator<T>>
    {
        using value_type = T;

        template <std::output_iterator<const value_type&> O, std::sentinel_for<O> S>
        constexpr O operator()(O first, S last, const value_type& min, const value_type& max) const
        {
            auto ptr = &(*first);
            auto count = last - first;

            auto res = cuda::detail::random(ptr, count, min, max);
            // TODO add curand exception support
            if (res != curandStatus_t::CURAND_STATUS_SUCCESS)
                std::cerr << "Curand status: " << static_cast<int>(res) << '\n';
            return last;
        }
    };
} // namespace mbq