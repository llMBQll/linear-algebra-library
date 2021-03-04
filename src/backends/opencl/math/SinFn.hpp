#pragma once

#include "backends/opencl/kernels/Kernel.hpp"
#include "math/sin.hpp"

namespace mbq
{
    template <typename T>
    struct Sin<opencl::Allocator<T>>
    {
        using value_type = T;

        template <std::output_iterator<const value_type&> O, std::sentinel_for<O> S>
        constexpr O operator()(O first, S last) const
        {
            auto state = first.get_allocator().state();
            opencl::detail::sin(state, &(*first), last - first);
            return last;
        }
    };

    template <typename T>
    struct Sinh<opencl::Allocator<T>>
    {
        using value_type = T;

        template <std::output_iterator<const value_type&> O, std::sentinel_for<O> S>
        constexpr O operator()(O first, S last) const
        {
            auto state = first.get_allocator().state();
            opencl::detail::sinh(state, &(*first), last - first);
            return last;
        }
    };

    template <typename T>
    struct Cos<opencl::Allocator<T>>
    {
        using value_type = T;

        template <std::output_iterator<const value_type&> O, std::sentinel_for<O> S>
        constexpr O operator()(O first, S last) const
        {
            auto state = first.get_allocator().state();
            opencl::detail::cos(state, &(*first), last - first);
            return last;
        }
    };

    template <typename T>
    struct Cosh<opencl::Allocator<T>>
    {
        using value_type = T;

        template <std::output_iterator<const value_type&> O, std::sentinel_for<O> S>
        constexpr O operator()(O first, S last) const
        {
            auto state = first.get_allocator().state();
            opencl::detail::cosh(state, &(*first), last - first);
            return last;
        }
    };

    template <typename T>
    struct Tan<opencl::Allocator<T>>
    {
        using value_type = T;

        template <std::output_iterator<const value_type&> O, std::sentinel_for<O> S>
        constexpr O operator()(O first, S last) const
        {
            auto state = first.get_allocator().state();
            opencl::detail::tan(state, &(*first), last - first);
            return last;
        }
    };

    template <typename T>
    struct Tanh<opencl::Allocator<T>>
    {
        using value_type = T;

        template <std::output_iterator<const value_type&> O, std::sentinel_for<O> S>
        constexpr O operator()(O first, S last) const
        {
            auto state = first.get_allocator().state();
            opencl::detail::tanh(state, &(*first), last - first);
            return last;
        }
    };
} // namespace mbq