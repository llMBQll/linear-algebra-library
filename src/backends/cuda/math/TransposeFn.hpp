#pragma once

#include "backends/cuda/exceptions/CublasException.hpp"
#include "backends/cuda/memory/Allocator.hpp"
#include "backends/cuda/memory/Context.hpp"
#include "math/transpose.hpp"

#include <cublas_v2.h>

namespace mbq
{
    namespace cuda::detail
    {
        template <typename T>
        cublasStatus_t transpose(cublasHandle_t handle, const T* in, T* out, size_t rows, size_t cols)
        {
            const T alpha{1};
            const T beta{0};

            if constexpr (std::is_same_v<T, float>)
                return cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, &alpha, in, cols, &beta, nullptr, rows,
                                   out, rows);
            else if constexpr (std::is_same_v<T, double>)
                return cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, &alpha, in, cols, &beta, nullptr, rows,
                                   out, rows);
            else if constexpr (std::is_same_v<T, std::complex<float>>)
                return cublasCgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, cuda_cast(&alpha), cuda_cast(in), cols,
                                   cuda_cast(&beta), nullptr, rows, cuda_cast(out), rows);
            else if constexpr (std::is_same_v<T, std::complex<double>>)
                return cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, cuda_cast(&alpha), cuda_cast(in), cols,
                                   cuda_cast(&beta), nullptr, rows, cuda_cast(out), rows);
        }
    } // namespace cuda::detail

    template <typename T>
    struct Transpose<mbq::cuda::Allocator<T>>
    {
    public:
        using value_type = T;
    public:
        template <typename ConstIter, typename Iter>
        void operator()(ConstIter in_begin, ConstIter, Iter out_begin, Iter, size_t rows, size_t cols) const
        {
            const T* in = &(*in_begin);
            T* out = &(*out_begin);
            auto handle = out_begin.get_allocator().state()->handle;

            auto status = cuda::detail::transpose(handle, in, out, rows, cols);
            if (status != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
                MBQ_THROW_EXCEPTION(CublasException, status);
        }
    };
} // namespace mbq