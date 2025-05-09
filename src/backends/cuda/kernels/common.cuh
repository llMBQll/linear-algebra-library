#pragma once

#include <bit>
#include <complex>
#include <cuComplex.h>

namespace mbq::cuda::detail
{
    __inline__ float cuda_cast(float x)
    {
        return x;
    }

    __inline__ float* cuda_cast(float* x)
    {
        return x;
    }

    __inline__ const float* cuda_cast(const float* x)
    {
        return x;
    }

    __inline__ double cuda_cast(double x)
    {
        return x;
    }

    __inline__ double* cuda_cast(double* x)
    {
        return x;
    }

    __inline__ const double* cuda_cast(const double* x)
    {
        return x;
    }

    __inline__ cuFloatComplex cuda_cast(std::complex<float> x)
    {
        return std::bit_cast<cuFloatComplex>(x);
    }

    __inline__ cuFloatComplex* cuda_cast(std::complex<float>* x)
    {
        return std::bit_cast<cuFloatComplex*>(x);
    }

    __inline__ const cuFloatComplex* cuda_cast(const std::complex<float>* x)
    {
        return std::bit_cast<const cuFloatComplex*>(x);
    }

    __inline__ cuDoubleComplex cuda_cast(std::complex<double> x)
    {
        return std::bit_cast<cuDoubleComplex>(x);
    }

    __inline__ cuDoubleComplex* cuda_cast(std::complex<double>* x)
    {
        return std::bit_cast<cuDoubleComplex*>(x);
    }

    __inline__ const cuDoubleComplex* cuda_cast(const std::complex<double>* x)
    {
        return std::bit_cast<const cuDoubleComplex*>(x);
    }
} // namespace mbq::cuda::detail