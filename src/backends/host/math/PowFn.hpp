#pragma once

#include "math/pow.hpp"
#include "memory/AllocatorTraits.hpp"

#include <algorithm>

namespace mbq
{
    template <typename Allocator>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    struct Pow<Allocator>
    {
        using value_type = value_type_of_t<Allocator>;

        template <std::output_iterator<const value_type&> O, std::sentinel_for<O> S>
        constexpr O operator()(O first, S last, const value_type& exponent) const
        {
            using std::pow;
            return std::transform(first, last, first, [exponent](const value_type& x) { return pow(x, exponent); });
        }
    };
} // namespace mbq