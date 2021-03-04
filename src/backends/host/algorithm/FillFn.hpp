#pragma once

#include "algorithm/fill.hpp"
#include "memory/AllocatorTraits.hpp"

namespace mbq
{
    template <typename Allocator>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    struct Fill<Allocator>
    {
        using value_type = value_type_of_t<Allocator>;

        template <std::output_iterator<const value_type&> O, std::sentinel_for<O> S>
        constexpr O operator()(O first, S last, const value_type& value) const
        {
            std::fill(first, last, value);
            return last;
        }
    };
} // namespace mbq