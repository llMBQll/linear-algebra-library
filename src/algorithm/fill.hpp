#pragma once

#include "concepts.hpp"
#include "util.hpp"

namespace mbq
{
    template <typename Allocator>
    struct Fill
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };

    template <typename Allocator>
    inline constexpr Fill<Allocator> fill_impl;

    template <typename First, typename Last, typename V>
        requires output_iterator_pair<First, Last> && std::convertible_to<V, typename First::value_type>
    constexpr auto fill(First first, Last last, const V& value)
    {
        using allocator_type = typename First::allocator_type;
        using value_type = typename allocator_type::value_type;
        return fill_impl<allocator_type>(first, last, static_cast<const value_type&>(value));
    }

    template <typename Range, typename V>
        requires output_range<Range&> && std::convertible_to<V, typename std::decay_t<Range>::value_type>
    constexpr auto fill(Range& r, const V& value)
    {
        // explicit mbq namespace because ADL was finding std::fill instead
        return mbq::fill(std::ranges::begin(r), std::ranges::end(r), value);
    }
} // namespace mbq