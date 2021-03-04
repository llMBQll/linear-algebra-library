#pragma once

#include "backends/opencl/kernels/Kernel.hpp"
#include "backends/opencl/memory/Allocator.hpp"
#include "math/pow.hpp"

namespace mbq
{
    template <typename T>
    struct Pow<opencl::Allocator<T>>
    {
        using value_type = T;

        template <std::output_iterator<const value_type&> O, std::sentinel_for<O> S,
                  std::convertible_to<value_type> Exponent>
        constexpr O operator()(O first, S last, const Exponent& exponent) const
        {
            auto state = first.get_allocator().state();
            opencl::detail::pow(state, &(*first), last - first, static_cast<const value_type&>(exponent));
            return last;
        }
    };
} // namespace mbq