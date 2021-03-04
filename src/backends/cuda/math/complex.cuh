#pragma once

#include <cmath>
#include <cuComplex.h>
#include <limits>

#define ENABLE_IF_CU_COMPLEX                                                                                           \
    typename std::enable_if<std::disjunction_v<std::is_same<T, cuFloatComplex>, std::is_same<T, cuDoubleComplex>>,     \
                            bool>::type = true

// silence spurious 'missing return statement at end of non-void function' warning in functions using constexpr if
#define SILENCE_MISSING_RETURN return {};

namespace mbq::cuda
{
    // Returns precision of a complex type - float for cuFloatComplex and double for cuDoubleComplex
    template <typename T>
    struct PrecisionOf;

    template <>
    struct PrecisionOf<cuFloatComplex>
    {
        using type = float;
    };

    template <>
    struct PrecisionOf<cuDoubleComplex>
    {
        using type = double;
    };

    template <typename T, ENABLE_IF_CU_COMPLEX>
    using precision_of_t = typename PrecisionOf<T>::type;

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T make_complex(precision_of_t<T> real, precision_of_t<T> imag)
    {
        if constexpr (std::is_same_v<T, cuFloatComplex>)
            return ::make_cuFloatComplex(real, imag);
        else
            return ::make_cuDoubleComplex(real, imag);

        SILENCE_MISSING_RETURN
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T make_complex(precision_of_t<T> real)
    {
        if constexpr (std::is_same_v<T, cuFloatComplex>)
            return ::make_cuFloatComplex(real, 0);
        else
            return ::make_cuDoubleComplex(real, 0);

        SILENCE_MISSING_RETURN
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline precision_of_t<T> real(T x)
    {
        return x.x;
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline precision_of_t<T> imag(T x)
    {
        return x.y;
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T operator+(T x, T y)
    {
        if constexpr (std::is_same_v<T, cuFloatComplex>)
            return ::cuCaddf(x, y);
        else
            return ::cuCadd(x, y);

        SILENCE_MISSING_RETURN
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T operator+(T x, precision_of_t<T> a)
    {
        return make_complex<T>(real(x) + a, imag(x));
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T operator+(precision_of_t<T> a, T x)
    {
        return x + a;
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T operator-(T x, T y)
    {
        if constexpr (std::is_same_v<T, cuFloatComplex>)
            return ::cuCsubf(x, y);
        else
            return ::cuCsub(x, y);

        SILENCE_MISSING_RETURN
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T operator-(T x, precision_of_t<T> a)
    {
        return make_complex<T>(real(x) - a, imag(x));
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T operator-(precision_of_t<T> a, T x)
    {
        return make_complex<T>(a - real(x), imag(x));
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T operator*(T x, T y)
    {
        if constexpr (std::is_same_v<T, cuFloatComplex>)
            return ::cuCmulf(x, y);
        else
            return ::cuCmul(x, y);

        SILENCE_MISSING_RETURN
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T operator*(T x, precision_of_t<T> a)
    {
        return make_complex<T>(real(x) * a, imag(x) * a);
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T operator*(precision_of_t<T> a, T x)
    {
        return x * a;
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T operator/(T x, T y)
    {
        if constexpr (std::is_same_v<T, cuFloatComplex>)
            return ::cuCdivf(x, y);
        else
            return ::cuCdiv(x, y);

        SILENCE_MISSING_RETURN
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T operator/(T x, precision_of_t<T> a)
    {
        return make_complex<T>(real(x) / a, imag(x) / a);
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T operator/(precision_of_t<T> a, T x)
    {
        return make_complex<T>(a / real(x), a / imag(x));
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline bool operator==(T x, T y)
    {
        return real(x) == real(y) && imag(x) == imag(y);
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline bool operator==(T x, precision_of_t<T> a)
    {
        return imag(x) == 0 && real(x) == a;
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline bool operator==(precision_of_t<T> a, T x)
    {
        return x == a;
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline bool operator!=(T x, T y)
    {
        return !(x == y);
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline bool operator!=(T x, precision_of_t<T> a)
    {
        return !(x == a);
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline bool operator!=(precision_of_t<T> a, T x)
    {
        return x != a;
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T conj(T x)
    {
        if constexpr (std::is_same_v<T, cuFloatComplex>)
            return ::cuConjf(x);
        else
            return ::cuConj(x);

        SILENCE_MISSING_RETURN
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline precision_of_t<T> norm(T x)
    {
        return real(x) * real(x) + imag(x) * imag(x);
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline precision_of_t<T> proj(T x)
    {
        if (::isinf(real(x)) || ::isinf(imag(x)))
        {
            precision_of_t<T> imag = copysign(precision_of_t<T>{0}, imag(x));
            return make_complex<T>(std::numeric_limits<precision_of_t<T>>::infiniy(), imag);
        }
        return x;
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T abs(T x)
    {
        if constexpr (std::is_same_v<T, cuFloatComplex>)
            return ::cuCabsf(x);
        else
            return ::cuCabs(x);

        SILENCE_MISSING_RETURN
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T exp(T x)
    {
        precision_of_t<T> e = ::exp(real(x));
        return make_complex<T>(e * ::cos(imag(x)), e * ::sin(imag(x)));
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T log(T x)
    {
        precision_of_t<T> log_abs = std::log(std::hypot(real(x), imag(x)));
        precision_of_t<T> theta = std::atan2(imag(x), real(x));
        return make_complex<T>(log_abs, theta);
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T pow(T x, precision_of_t<T> exponent)
    {
        if (imag(x) == 0)
        {
            if (std::signbit(imag(x)))
                return conj(make_complex<T>(std::pow(real(x), exponent)));
            else
                return make_complex<T>(std::pow(real(x), exponent));
        }
        else
            return exp(exponent * log(x));
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T pow(T x, T exponent)
    {
        if (imag(x) == 0)
            return pow(x, real(exponent));
        else if (imag(x) == 0 && real(x) > 0)
            return exp(exponent * std::log(real(x)));
        else
            return exp(exponent * log(x));
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T polar(precision_of_t<T> r, precision_of_t<T> theta)
    {
        return make_complex<T>(r * ::cos(theta), r * ::sin(theta));
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T polar(precision_of_t<T> r)
    {
        return make_complex<T>(r, precision_of_t<T>{0});
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T arg(T x)
    {
        return make_complex<T>(atan2(imag(x), real(x)));
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T sin(T x)
    {
        return make_complex<T>(::sin(real(x)) * ::cosh(imag(x)), ::cos(real(x)) * ::sinh(imag(x)));
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T sinh(T x)
    {
        return make_complex<T>(::sinh(real(x)) * ::cos(imag(x)), ::cosh(real(x)) * ::sin(imag(x)));
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T cos(T x)
    {
        return make_complex<T>(::cos(real(x)) * ::cosh(imag(x)), -::sin(real(x)) * ::sinh(imag(x)));
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T cosh(T x)
    {
        return make_complex<T>(::cosh(real(x)) * ::cos(imag(x)), ::sinh(real(x)) * ::sin(imag(x)));
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T tan(T x)
    {
        return sin(x) / cos(x);
    }

    template <typename T, ENABLE_IF_CU_COMPLEX>
    __host__ __device__ inline T tanh(T x)
    {
        return sinh(x) / cosh(x);
    }
} // namespace mbq::cuda

#undef SILENCE_MISSING_RETURN
#undef ENABLE_IF_CU_COMPLEX
