#pragma once

#include "math/sin.hpp"
#include "memory/AllocatorTraits.hpp"

namespace mbq
{
    template <typename Allocator>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    struct Sin<Allocator>
    {
        using value_type = value_type_of_t<Allocator>;

        template <typename First, typename Last>
        constexpr First operator()(First first, Last last) const
        {
            return std::transform(first, last, first, [](const value_type& x) { return std::sin(x); });
        }
    };

    template <typename Allocator>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    struct Sinh<Allocator>
    {
        using value_type = value_type_of_t<Allocator>;

        template <typename First, typename Last>
        constexpr First operator()(First first, Last last) const
        {
            return std::transform(first, last, first, [](const value_type& x) { return std::sinh(x); });
        }
    };

    template <typename Allocator>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    struct Asin<Allocator>
    {
        using value_type = value_type_of_t<Allocator>;

        template <typename First, typename Last>
        constexpr First operator()(First first, Last last) const
        {
            return std::transform(first, last, first, [](const value_type& x) { return std::asin(x); });
        }
    };

    template <typename Allocator>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    struct Asinh<Allocator>
    {
        using value_type = value_type_of_t<Allocator>;

        template <typename First, typename Last>
        constexpr First operator()(First first, Last last) const
        {
            return std::transform(first, last, first, [](const value_type& x) { return std::asinh(x); });
        }
    };

    template <typename Allocator>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    struct Cos<Allocator>
    {
        using value_type = value_type_of_t<Allocator>;

        template <typename First, typename Last>
        constexpr First operator()(First first, Last last) const
        {
            return std::transform(first, last, first, [](const value_type& x) { return std::cos(x); });
        }
    };

    template <typename Allocator>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    struct Cosh<Allocator>
    {
        using value_type = value_type_of_t<Allocator>;

        template <typename First, typename Last>
        constexpr First operator()(First first, Last last) const
        {
            return std::transform(first, last, first, [](const value_type& x) { return std::cosh(x); });
        }
    };

    template <typename Allocator>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    struct Acos<Allocator>
    {
        using value_type = value_type_of_t<Allocator>;

        template <typename First, typename Last>
        constexpr First operator()(First first, Last last) const
        {
            return std::transform(first, last, first, [](const value_type& x) { return std::acos(x); });
        }
    };

    template <typename Allocator>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    struct Acosh<Allocator>
    {
        using value_type = value_type_of_t<Allocator>;

        template <typename First, typename Last>
        constexpr First operator()(First first, Last last) const
        {
            return std::transform(first, last, first, [](const value_type& x) { return std::acosh(x); });
        }
    };

    template <typename Allocator>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    struct Tan<Allocator>
    {
        using value_type = value_type_of_t<Allocator>;

        template <typename First, typename Last>
        constexpr First operator()(First first, Last last) const
        {
            return std::transform(first, last, first, [](const value_type& x) { return std::tan(x); });
        }
    };

    template <typename Allocator>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    struct Tanh<Allocator>
    {
        using value_type = value_type_of_t<Allocator>;

        template <typename First, typename Last>
        constexpr First operator()(First first, Last last) const
        {
            return std::transform(first, last, first, [](const value_type& x) { return std::tanh(x); });
        }
    };

    template <typename Allocator>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    struct Atan<Allocator>
    {
        using value_type = value_type_of_t<Allocator>;

        template <typename First, typename Last>
        constexpr First operator()(First first, Last last) const
        {
            return std::transform(first, last, first, [](const value_type& x) { return std::atan(x); });
        }
    };

    template <typename Allocator>
        requires host_allocator<value_type_of_t<Allocator>, Allocator>
    struct Atanh<Allocator>
    {
        using value_type = value_type_of_t<Allocator>;

        template <typename First, typename Last>
        constexpr First operator()(First first, Last last) const
        {
            return std::transform(first, last, first, [](const value_type& x) { return std::atanh(x); });
        }
    };
} // namespace mbq