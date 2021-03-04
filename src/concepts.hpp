#pragma once

#include <complex>
#include <concepts>
#include <ranges>

namespace mbq
{
    template <typename T>
    concept non_void = !
    std::is_same_v<T, void>;

    template <typename T>
    concept integral = std::is_integral_v<T>;

    template <typename T>
    concept signed_integral = integral<T> && std::is_signed_v<T>;

    template <typename T>
    concept unsigned_integral = integral<T> && std::is_unsigned_v<T>;

    template <typename T>
    concept floating_point = std::is_floating_point_v<T>;

    template <typename T>
    concept arithmetic = integral<T> || floating_point<T>;

    template <typename To, typename... From>
    concept convertible_to = std::conjunction_v<std::is_convertible<From, To>...>;

    template <size_t N, typename... Ts>
    concept has_size = (sizeof...(Ts) == N);

    template <size_t Required, size_t N>
    concept equal_dimensions = (Required == N);

    template <typename T>
    concept supported_blas_type = std::same_as<T, float> || std::same_as<T, double> ||
                                  std::same_as<T, std::complex<float>> || std::same_as<T, std::complex<double>>;

    namespace detail
    {
        template <typename... Ts>
        struct last
        {
            using type = typename decltype((std::type_identity<Ts>{}, ...))::type;
        };

        template <size_t Current, size_t N, typename T, typename... Ts>
        constexpr bool valid_alloc_helper()
        {
            if constexpr (!std::is_convertible_v<T, size_t>)
                return false;
            if constexpr (Current < N)
                return valid_alloc_helper<Current + 1, N, Ts...>();
            return true;
        }
    } // namespace detail

    template <typename T>
    concept stateful_allocator = requires { typename T::stateful; };

    template <size_t N, typename... Ts>
    concept valid_dimensions = has_size<N, Ts...> && convertible_to<size_t, Ts...>;

    template <typename Allocator, size_t N, typename... Ts>
    concept valid_dimensions_with_allocator =
        has_size<N + 1, Ts...> &&
        std::is_same_v<Allocator, std::remove_reference_t<std::remove_const_t<typename detail::last<Ts...>::type>>> &&
        requires { detail::valid_alloc_helper<0, N, Ts...>(); };

    template <typename Allocator, size_t N, typename... Ts>
    concept valid = valid_dimensions<N, Ts...> || valid_dimensions_with_allocator<Allocator, N, Ts...>;

    template <typename First, typename Last>
    concept output_iterator_pair =
        std::output_iterator<First, typename First::value_type> && std::sentinel_for<First, Last>;

    template <typename R>
    concept output_range = std::ranges::output_range<R, typename std::decay_t<R>::value_type>;
} // namespace mbq