#pragma once

#include "concepts.hpp"
#include "util.hpp"

namespace mbq
{
    template <typename Allocator>
    struct Random
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };

    template <typename Allocator>
    inline constexpr Random<Allocator> random_impl;

    template <typename First, typename Last, typename V>
        requires output_iterator_pair<First, Last> && std::convertible_to<V, typename First::value_type>
    constexpr auto random(First first, Last last, const V& min, const V& max)
    {
        using allocator_type = typename First::allocator_type;
        using value_type = typename allocator_type::value_type;
        return random_impl<allocator_type>(first, last, static_cast<const value_type&>(min),
                                           static_cast<const value_type&>(max));
    }

    template <typename First, typename Last>
        requires output_iterator_pair<First, Last>
    constexpr auto random(First first, Last last)
    {
        using allocator_type = typename First::allocator_type;
        using value_type = typename allocator_type::value_type;
        return random_impl<allocator_type>(first, last, value_type{0}, value_type{1});
    }

    template <typename Range, typename V>
        requires output_range<Range&> && std::convertible_to<V, typename std::decay_t<Range>::value_type>
    constexpr auto random(Range& r, const V& min, const V& max)
    {
        using value_type = typename std::remove_cvref_t<Range>::value_type;
        return random(std::ranges::begin(r), std::ranges::end(r), static_cast<const value_type&>(min),
                      static_cast<const value_type&>(max));
    }

    template <typename Range>
        requires output_range<Range&>
    constexpr auto random(Range& r)
    {
        using value_type = typename std::remove_cvref_t<Range>::value_type;
        return random(r, value_type{0}, value_type{1});
    }
} // namespace mbq