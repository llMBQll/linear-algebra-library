#pragma once

#include "algorithm/random.hpp"
#include "memory/AllocatorTraits.hpp"

#include <random>

namespace mbq
{
    namespace host::detail
    {
        template <typename>
        struct is_complex_value : std::false_type
        { };

        template <typename T>
        struct is_complex_value<std::complex<T>> : std::true_type
        { };

        template <typename T>
        concept complex_value = is_complex_value<T>::value;

        inline std::mt19937& get_default_engine()
        {
            thread_local static std::mt19937 engine{std::random_device{}()};
            return engine;
        }

        template <typename Iter, typename Sentinel, typename T>
        void random(Iter first, Sentinel last, const T& min, const T& max)
        {
            std::uniform_real_distribution<T> dist{min, max};
            auto& engine = host::detail::get_default_engine();

            std::transform(first, last, first, [&engine, &dist](const auto&) -> T { return dist(engine); });
        }

        template <typename Iter, typename Sentinel, complex_value T>
        void random(Iter first, Sentinel last, const T& min, const T& max)
        {
            using value_type = decltype(min.real());
            std::uniform_real_distribution<value_type> dist{min.real(), max.real()};
            auto& engine = host::detail::get_default_engine();

            std::transform(first, last, first, [&engine, &dist](const auto&) -> T {
                return {dist(engine), dist(engine)};
            });
        }
    } // namespace host::detail

    template <typename Allocator>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    struct Random<Allocator>
    {
        using value_type = value_type_of_t<Allocator>;

        template <std::output_iterator<const value_type&> O, std::sentinel_for<O> S>
        constexpr O operator()(O first, S last, const value_type& min, const value_type& max) const
        {
            host::detail::random(first, last, min, max);
            return last;
        }
    };
} // namespace mbq