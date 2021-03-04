#pragma once

#include "util.hpp"

namespace mbq
{
    template <typename Allocator>
    struct Sin
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };

    template <typename Allocator>
    inline constexpr Sin<Allocator> sin_impl;

    template <typename First, typename Last>
        requires output_iterator_pair<First, Last>
    constexpr auto sin(First first, Last last)
    {
        using allocator_type = typename First::allocator_type;
        return sin_impl<allocator_type>(first, last);
    }

    template <typename T>
        requires output_range<T&>
    constexpr auto sin(T& r)
    {
        return sin(std::ranges::begin(r), std::ranges::end(r));
    }

    template <typename Allocator>
    struct Sinh
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };

    template <typename Allocator>
    inline constexpr Sinh<Allocator> sinh_impl;

    template <typename First, typename Last>
        requires output_iterator_pair<First, Last>
    constexpr auto sinh(First first, Last last)
    {
        using allocator_type = typename First::allocator_type;
        return sinh_impl<allocator_type>(first, last);
    }

    template <typename T>
        requires output_range<T&>
    constexpr auto sinh(T& r)
    {
        return sinh(std::ranges::begin(r), std::ranges::end(r));
    }

    template <typename Allocator>
    struct Asin
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };

    template <typename Allocator>
    inline constexpr Asin<Allocator> asin_impl;

    template <typename First, typename Last>
        requires output_iterator_pair<First, Last>
    constexpr auto asin(First first, Last last)
    {
        using allocator_type = typename First::allocator_type;
        return asin_impl<allocator_type>(first, last);
    }

    template <typename T>
        requires output_range<T&>
    constexpr auto asin(T& r)
    {
        return asin(std::ranges::begin(r), std::ranges::end(r));
    }

    template <typename Allocator>
    struct Asinh
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };

    template <typename Allocator>
    inline constexpr Asinh<Allocator> asinh_impl;

    template <typename First, typename Last>
        requires output_iterator_pair<First, Last>
    constexpr auto asinh(First first, Last last)
    {
        using allocator_type = typename First::allocator_type;
        return asinh_impl<allocator_type>(first, last);
    }

    template <typename T>
        requires output_range<T&>
    constexpr auto asinh(T& r)
    {
        return asinh(std::ranges::begin(r), std::ranges::end(r));
    }

    template <typename Allocator>
    struct Cos
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };

    template <typename Allocator>
    inline constexpr Cos<Allocator> cos_impl;

    template <typename First, typename Last>
        requires output_iterator_pair<First, Last>
    constexpr auto cos(First first, Last last)
    {
        using allocator_type = typename First::allocator_type;
        return cos_impl<allocator_type>(first, last);
    }

    template <typename T>
        requires output_range<T&>
    constexpr auto cos(T& r)
    {
        return cos(std::ranges::begin(r), std::ranges::end(r));
    }

    template <typename Allocator>
    struct Cosh
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };

    template <typename Allocator>
    inline constexpr Cosh<Allocator> cosh_impl;

    template <typename First, typename Last>
        requires output_iterator_pair<First, Last>
    constexpr auto cosh(First first, Last last)
    {
        using allocator_type = typename First::allocator_type;
        return cosh_impl<allocator_type>(first, last);
    }

    template <typename T>
        requires output_range<T&>
    constexpr auto cosh(T& r)
    {
        return cosh(std::ranges::begin(r), std::ranges::end(r));
    }

    template <typename Allocator>
    struct Acos
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };

    template <typename Allocator>
    inline constexpr Acos<Allocator> acos_impl;

    template <typename First, typename Last>
        requires output_iterator_pair<First, Last>
    constexpr auto acos(First first, Last last)
    {
        using allocator_type = typename First::allocator_type;
        return acos_impl<allocator_type>(first, last);
    }

    template <typename T>
        requires output_range<T&>
    constexpr auto acos(T& r)
    {
        return acos(std::ranges::begin(r), std::ranges::end(r));
    }

    template <typename Allocator>
    struct Acosh
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };

    template <typename Allocator>
    inline constexpr Acosh<Allocator> acosh_impl;

    template <typename First, typename Last>
        requires output_iterator_pair<First, Last>
    constexpr auto acosh(First first, Last last)
    {
        using allocator_type = typename First::allocator_type;
        return acosh_impl<allocator_type>(first, last);
    }

    template <typename T>
        requires output_range<T&>
    constexpr auto acosh(T& r)
    {
        return acosh(std::ranges::begin(r), std::ranges::end(r));
    }

    template <typename Allocator>
    struct Tan
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };

    template <typename Allocator>
    inline constexpr Tan<Allocator> tan_impl;

    template <typename First, typename Last>
        requires output_iterator_pair<First, Last>
    constexpr auto tan(First first, Last last)
    {
        using allocator_type = typename First::allocator_type;
        return tan_impl<allocator_type>(first, last);
    }

    template <typename T>
        requires output_range<T&>
    constexpr auto tan(T& r)
    {
        return tan(std::ranges::begin(r), std::ranges::end(r));
    }

    template <typename Allocator>
    struct Tanh
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };

    template <typename Allocator>
    inline constexpr Tanh<Allocator> tanh_impl;

    template <typename First, typename Last>
        requires output_iterator_pair<First, Last>
    constexpr auto tanh(First first, Last last)
    {
        using allocator_type = typename First::allocator_type;
        return tanh_impl<allocator_type>(first, last);
    }

    template <typename T>
        requires output_range<T&>
    constexpr auto tanh(T& r)
    {
        return tanh(std::ranges::begin(r), std::ranges::end(r));
    }

    template <typename Allocator>
    struct Atan
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };

    template <typename Allocator>
    inline constexpr Atan<Allocator> atan_impl;

    template <typename First, typename Last>
        requires output_iterator_pair<First, Last>
    constexpr auto atan(First first, Last last)
    {
        using allocator_type = typename First::allocator_type;
        return atan_impl<allocator_type>(first, last);
    }

    template <typename T>
        requires output_range<T&>
    constexpr auto atan(T& r)
    {
        return atan(std::ranges::begin(r), std::ranges::end(r));
    }

    template <typename Allocator>
    struct Atanh
    {
        MBQ_FN_NOT_IMPLEMENTED;
    };

    template <typename Allocator>
    inline constexpr Atanh<Allocator> atanh_impl;

    template <typename First, typename Last>
        requires output_iterator_pair<First, Last>
    constexpr auto atanh(First first, Last last)
    {
        using allocator_type = typename First::allocator_type;
        return atanh_impl<allocator_type>(first, last);
    }

    template <typename T>
        requires output_range<T&>
    constexpr auto atanh(T& r)
    {
        return atanh(std::ranges::begin(r), std::ranges::end(r));
    }
} // namespace mbq