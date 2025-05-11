#pragma once

#include "backends/cuda/exceptions/CublasException.hpp"
#include "backends/cuda/memory/Allocator.hpp"
#include "math/axpy.hpp"

#include <bit>
#include <cublas_v2.h>

namespace mbq
{
    template <typename T, size_t N>
    struct Axpy<cuda::Allocator<T>, N>
    {
        using value_type = T;

        template <typename ConstIter, typename Iter>
        void operator()(T alpha, ConstIter x_begin, ConstIter x_end, Iter y_begin, Iter) const
        {
            const T* x = &(*x_begin);
            T* y = &(*y_begin);
            auto n = static_cast<int>(x_end - x_begin);
            auto handle = y_begin.get_allocator().state()->handle;

            cublasStatus_t status;
            if constexpr (std::is_same_v<T, float>)
                status = cublasSaxpy_v2(handle, n, &alpha, x, 1, y, 1);
            else if constexpr (std::is_same_v<T, double>)
                status = cublasDaxpy_v2(handle, n, &alpha, x, 1, y, 1);
            else if constexpr (std::is_same_v<T, std::complex<float>>)
                status = cublasCaxpy_v2(handle, n, std::bit_cast<const float2*>(&alpha),
                                        std::bit_cast<const float2*>(x), 1, std::bit_cast<float2*>(y), 1);
            else if constexpr (std::is_same_v<T, std::complex<double>>)
                status = cublasZaxpy_v2(handle, n, std::bit_cast<const double2*>(&alpha),
                                        std::bit_cast<const double2*>(x), 1, std::bit_cast<double2*>(y), 1);

            if (status != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
                MBQ_THROW_EXCEPTION(CublasException, status);
        }
    };
} // namespace mbq