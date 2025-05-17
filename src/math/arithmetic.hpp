#pragma once

#include "Array.hpp"
#include "util.hpp"

namespace mbq
{
    template <typename Allocator>
    struct Add
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };

    template <typename Allocator>
    inline constexpr Add<Allocator> add_fn;

    template <template <non_void, typename, size_t> typename X, template <non_void, typename, size_t> typename Y,
              non_void T, typename Allocator, size_t N>
    auto& operator+=(X<T, Allocator, N>& x, const Y<T, Allocator, N>& y)
    {
        if (x.dimensions() != y.dimensions())
            MBQ_THROW_EXCEPTION(Exception, "Array dimensions are not equal");

        add_fn<Allocator>(x.begin(), x.end(), y.begin(), y.end(), x.begin(), x.end());
        return x;
    }

    template <template <non_void, typename, size_t> typename Y, non_void T, typename Allocator, size_t N>
    auto operator+=(ArrayView<T, Allocator, N>&& x, const Y<T, Allocator, N>& y)
    {
        return x += y;
    }

    template <template <non_void, typename, size_t> typename X, template <non_void, typename, size_t> typename Y,
              non_void T, typename Allocator, size_t N>
    Array<T, Allocator, N> operator+(const X<T, Allocator, N>& x, const Y<T, Allocator, N>& y)
    {
        if (x.dimensions() != y.dimensions())
            MBQ_THROW_EXCEPTION(Exception, "Array dimensions are not equal");

        Array<T, Allocator, N> tmp{x.dimensions(), x.get_allocator()};
        add_fn<Allocator>(x.begin(), x.end(), y.begin(), y.end(), tmp.begin(), tmp.end());
        return tmp;
    }

    template <typename A, template <non_void, typename, size_t> typename X, non_void T, typename Allocator, size_t N>
        requires std::convertible_to<A, T>
    auto& operator+=(X<T, Allocator, N>& x, const A& a)
    {
        add_fn<Allocator>(x.begin(), x.end(), x.begin(), x.end(), static_cast<const T&>(a));
        return x;
    }

    template <typename A, non_void T, typename Allocator, size_t N>
        requires std::convertible_to<A, T>
    auto operator+=(ArrayView<T, Allocator, N>&& x, const A& a)
    {
        return x += a;
    }

    template <typename A, template <non_void, typename, size_t> typename X, non_void T, typename Allocator, size_t N>
        requires std::convertible_to<A, T>
    Array<T, Allocator, N> operator+(const X<T, Allocator, N>& x, const A& a)
    {
        Array<T, Allocator, N> tmp{x.dimensions(), x.get_allocator()};
        add_fn<Allocator>(x.begin(), x.end(), tmp.begin(), tmp.end(), static_cast<const T&>(a));
        return tmp;
    }

    template <typename A, template <non_void, typename, size_t> typename X, non_void T, typename Allocator, size_t N>
        requires std::convertible_to<A, T>
    Array<T, Allocator, N> operator+(const A& a, const X<T, Allocator, N>& x)
    {
        return x + a;
    }

    template <typename Allocator>
    struct Subtract
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };

    template <typename Allocator>
    inline constexpr Subtract<Allocator> subtract_impl;

    template <template <non_void, typename, size_t> typename X, template <non_void, typename, size_t> typename Y,
              non_void T, typename Allocator, size_t N>
    auto& operator-=(X<T, Allocator, N>& x, const Y<T, Allocator, N>& y)
    {
        if (x.dimensions() != y.dimensions())
            MBQ_THROW_EXCEPTION(Exception, "Array dimensions are not equal");

        subtract_impl<Allocator>(x.begin(), x.end(), y.begin(), y.end(), x.begin(), x.end());
        return x;
    }

    template <template <non_void, typename, size_t> typename Y, non_void T, typename Allocator, size_t N>
    auto operator-=(ArrayView<T, Allocator, N>&& x, const Y<T, Allocator, N>& y)
    {
        return x -= y;
    }

    template <template <non_void, typename, size_t> typename X, template <non_void, typename, size_t> typename Y,
              non_void T, typename Allocator, size_t N>
    Array<T, Allocator, N> operator-(const X<T, Allocator, N>& x, const Y<T, Allocator, N>& y)
    {
        if (x.dimensions() != y.dimensions())
            MBQ_THROW_EXCEPTION(Exception, "Array dimensions are not equal");

        Array<T, Allocator, N> tmp{x.dimensions(), x.get_allocator()};
        subtract_impl<Allocator>(x.begin(), x.end(), y.begin(), y.end(), tmp.begin(), tmp.end());
        return tmp;
    }

    template <typename A, template <non_void, typename, size_t> typename X, non_void T, typename Allocator, size_t N>
        requires std::convertible_to<A, T>
    auto& operator-=(X<T, Allocator, N>& x, const A& a)
    {
        subtract_impl<Allocator>(x.begin(), x.end(), x.begin(), x.end(), static_cast<const T&>(a));
        return x;
    }

    template <typename A, non_void T, typename Allocator, size_t N>
        requires std::convertible_to<A, T>
    auto operator-=(ArrayView<T, Allocator, N>&& x, const A& a)
    {
        return x -= a;
    }

    template <typename A, template <non_void, typename, size_t> typename X, non_void T, typename Allocator, size_t N>
        requires std::convertible_to<A, T>
    Array<T, Allocator, N> operator-(const X<T, Allocator, N>& x, const A& a)
    {
        Array<T, Allocator, N> tmp{x.dimensions(), x.get_allocator()};
        subtract_impl<Allocator>(x.begin(), x.end(), tmp.begin(), tmp.end(), static_cast<const T&>(a));
        return tmp;
    }

    template <typename A, template <non_void, typename, size_t> typename X, non_void T, typename Allocator, size_t N>
        requires std::convertible_to<A, T>
    Array<T, Allocator, N> operator-(const A& a, const X<T, Allocator, N>& x)
    {
        Array<T, Allocator, N> tmp{x.dimensions(), x.get_allocator()};
        subtract_impl<Allocator>(static_cast<const T&>(a), x.begin(), x.end(), tmp.begin(), tmp.end());
        return tmp;
    }

    template <typename Allocator>
    struct Multiply
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };
    template <typename Allocator>
    inline constexpr Multiply<Allocator> multiply_impl;

    template <template <non_void, typename, size_t> typename X, template <non_void, typename, size_t> typename Y,
              non_void T, typename Allocator, size_t N>
        requires equal_dimensions<2, N>
    Array<T, Allocator, N> multiply(const X<T, Allocator, N>& x, const Y<T, Allocator, N>& y)
    {
        if (x.dimensions()[1] != y.dimensions()[0])
            MBQ_THROW_EXCEPTION(Exception, "Array dimensions are not equal");

        auto m = x.dimensions()[0];
        auto n = y.dimensions()[1];
        auto k = x.dimensions()[1];
        Array<T, Allocator, N> tmp(m, n);
        multiply_impl<Allocator>(x.cbegin(), y.cbegin(), y.cbegin(), y.cend(), tmp.begin(), tmp.end(),
                                 static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
        return tmp;
    }

    template <template <non_void, typename, size_t> typename X, template <non_void, typename, size_t> typename Y,
              non_void T, typename Allocator, size_t N>
    auto& operator*=(X<T, Allocator, N>& x, const Y<T, Allocator, N>& y)
    {
        if (x.dimensions() != y.dimensions())
            MBQ_THROW_EXCEPTION(Exception, "Array dimensions are not equal");

        multiply_impl<Allocator>(x.begin(), x.end(), y.begin(), y.end(), x.begin(), x.end());
        return x;
    }

    template <template <non_void, typename, size_t> typename Y, non_void T, typename Allocator, size_t N>
    auto operator*=(ArrayView<T, Allocator, N>&& x, const Y<T, Allocator, N>& y)
    {
        return x *= y;
    }

    template <template <non_void, typename, size_t> typename X, template <non_void, typename, size_t> typename Y,
              non_void T, typename Allocator, size_t N>
    Array<T, Allocator, N> operator*(const X<T, Allocator, N>& x, const Y<T, Allocator, N>& y)
    {
        if (x.dimensions() != y.dimensions())
            MBQ_THROW_EXCEPTION(Exception, "Array dimensions are not equal");

        Array<T, Allocator, N> tmp{x.dimensions(), x.get_allocator()};
        multiply_impl<Allocator>(x.begin(), x.end(), y.begin(), y.end(), tmp.begin(), tmp.end());
        return tmp;
    }

    template <typename A, template <non_void, typename, size_t> typename X, non_void T, typename Allocator, size_t N>
        requires std::convertible_to<A, T>
    auto& operator*=(X<T, Allocator, N>& x, const A& a)
    {
        multiply_impl<Allocator>(x.begin(), x.end(), x.begin(), x.end(), static_cast<const T&>(a));
        return x;
    }

    template <typename A, non_void T, typename Allocator, size_t N>
        requires std::convertible_to<A, T>
    auto operator*=(ArrayView<T, Allocator, N>&& x, const A& a)
    {
        return x *= a;
    }

    template <typename A, template <non_void, typename, size_t> typename X, non_void T, typename Allocator, size_t N>
        requires std::convertible_to<A, T>
    Array<T, Allocator, N> operator*(const X<T, Allocator, N>& x, const A& a)
    {
        Array<T, Allocator, N> tmp{x.dimensions(), x.get_allocator()};
        multiply_impl<Allocator>(x.begin(), x.end(), tmp.begin(), tmp.end(), static_cast<const T&>(a));
        return tmp;
    }

    template <typename A, template <non_void, typename, size_t> typename X, non_void T, typename Allocator, size_t N>
        requires std::convertible_to<A, T>
    Array<T, Allocator, N> operator*(const A& a, const X<T, Allocator, N>& x)
    {
        return x * a;
    }

    template <typename Allocator>
    struct Divide
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };

    template <typename Allocator>
    inline constexpr Divide<Allocator> divide_impl;

    template <template <non_void, typename, size_t> typename X, template <non_void, typename, size_t> typename Y,
              non_void T, typename Allocator, size_t N>
    auto& operator/=(X<T, Allocator, N>& x, const Y<T, Allocator, N>& y)
    {
        if (x.dimensions() != y.dimensions())
            MBQ_THROW_EXCEPTION(Exception, "Array dimensions are not equal");

        divide_impl<Allocator>(x.begin(), x.end(), y.begin(), y.end(), x.begin(), x.end());
        return x;
    }

    template <template <non_void, typename, size_t> typename Y, non_void T, typename Allocator, size_t N>
    auto operator/=(ArrayView<T, Allocator, N>&& x, const Y<T, Allocator, N>& y)
    {
        return x /= y;
    }

    template <template <non_void, typename, size_t> typename X, template <non_void, typename, size_t> typename Y,
              non_void T, typename Allocator, size_t N>
    Array<T, Allocator, N> operator/(const X<T, Allocator, N>& x, const Y<T, Allocator, N>& y)
    {
        if (x.dimensions() != y.dimensions())
            MBQ_THROW_EXCEPTION(Exception, "Array dimensions are not equal");

        Array<T, Allocator, N> tmp{x.dimensions(), x.get_allocator()};
        divide_impl<Allocator>(x.begin(), x.end(), y.begin(), y.end(), tmp.begin(), tmp.end());
        return tmp;
    }

    template <typename A, template <non_void, typename, size_t> typename X, non_void T, typename Allocator, size_t N>
        requires std::convertible_to<A, T>
    auto& operator/=(X<T, Allocator, N>& x, const A& a)
    {
        divide_impl<Allocator>(x.begin(), x.end(), x.begin(), x.end(), a);
        return x;
    }

    template <typename A, non_void T, typename Allocator, size_t N>
        requires std::convertible_to<A, T>
    auto operator/=(ArrayView<T, Allocator, N>&& x, const A& a)
    {
        return x /= a;
    }

    template <typename A, template <non_void, typename, size_t> typename X, non_void T, typename Allocator, size_t N>
        requires std::convertible_to<A, T>
    Array<T, Allocator, N> operator/(const X<T, Allocator, N>& x, const A& a)
    {
        Array<T, Allocator, N> tmp{x.dimensions(), x.get_allocator()};
        divide_impl<Allocator>(x.begin(), x.end(), tmp.begin(), tmp.end(), a);
        return tmp;
    }

    template <typename A, template <non_void, typename, size_t> typename X, non_void T, typename Allocator, size_t N>
        requires std::convertible_to<A, T>
    Array<T, Allocator, N> operator/(const A& a, const X<T, Allocator, N>& x)
    {
        Array<T, Allocator, N> tmp{x.dimensions(), x.get_allocator()};
        divide_impl<Allocator>(a, x.begin(), x.end(), tmp.begin(), tmp.end());
        return tmp;
    }
} // namespace mbq