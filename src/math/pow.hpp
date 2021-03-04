#pragma once

#include "concepts.hpp"
#include "util.hpp"

namespace mbq
{
    template <typename Allocator>
    struct Pow
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };

    template <typename Allocator>
    inline constexpr Pow<Allocator> pow_impl;

    template <typename First, typename Last, typename Exponent>
        requires output_iterator_pair<First, Last> && std::convertible_to<Exponent, typename First::value_type>
    constexpr auto pow(First first, Last last, const Exponent& exponent)
    {
        using allocator_type = typename First::allocator_type;
        using value_type = typename allocator_type::value_type;
        return pow_impl<allocator_type>(first, last, static_cast<const value_type&>(exponent));
    }

    template <typename T, typename Exponent>
        requires output_range<T&> && std::convertible_to<Exponent, typename std::decay_t<T>::value_type>
    constexpr auto pow(T& r, const Exponent& exponent)
    {
        return pow(std::ranges::begin(r), std::ranges::end(r), exponent);
    }
} // namespace mbq