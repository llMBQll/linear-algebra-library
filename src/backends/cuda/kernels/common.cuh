#pragma once

#include <complex>
#include <cuComplex.h>

namespace mbq::cuda::detail
{
    inline float cuda_cast(float x)
    {
        return x;
    }

    inline float* cuda_cast(float* x)
    {
        return x;
    }

    inline const float* cuda_cast(const float* x)
    {
        return x;
    }

    inline double cuda_cast(double x)
    {
        return x;
    }

    inline double* cuda_cast(double* x)
    {
        return x;
    }

    inline const double* cuda_cast(const double* x)
    {
        return x;
    }

    inline cuFloatComplex cuda_cast(std::complex<float> x)
    {
        return *reinterpret_cast<cuFloatComplex*>(&x);
    }

    inline cuFloatComplex* cuda_cast(std::complex<float>* x)
    {
        return reinterpret_cast<cuFloatComplex*>(x);
    }

    inline const cuFloatComplex* cuda_cast(const std::complex<float>* x)
    {
        return reinterpret_cast<const cuFloatComplex*>(x);
    }

    inline cuDoubleComplex cuda_cast(std::complex<double> x)
    {
        return *reinterpret_cast<cuDoubleComplex*>(&x);
    }

    inline cuDoubleComplex* cuda_cast(std::complex<double>* x)
    {
        return reinterpret_cast<cuDoubleComplex*>(x);
    }

    inline const cuDoubleComplex* cuda_cast(const std::complex<double>* x)
    {
        return reinterpret_cast<const cuDoubleComplex*>(x);
    }
} // namespace mbq::cuda::detail