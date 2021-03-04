#pragma once

#include "../kernels/common.cuh"

#include <cublas_v2.h>

namespace mbq::cuda::detail
{

    template <typename T>
    cudaError_t add(const T* x, const T* y, T* z, size_t count);

    template <typename T>
    cudaError_t add(const T* x, T* y, const T& value, size_t count);

    template <typename T>
    cudaError_t subtract(const T* x, const T* y, T* z, size_t count);

    template <typename T>
    cudaError_t subtract(const T* x, T* y, const T& value, size_t count);

    template <typename T>
    cudaError_t subtract(const T& value, const T* x, T* y, size_t count);

    template <typename T>
    cublasStatus_t multiply(cublasHandle_t handle, int m, int n, int k, const T* A, const T* B, T* C)
    {
        const T alpha{1};
        const T beta{0};

        if constexpr (std::is_same_v<T, float>)
            return cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n);
        else if constexpr (std::is_same_v<T, double>)
            return cublasDgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n);
        else if constexpr (std::is_same_v<T, std::complex<float>>)
            return cublasCgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, cuda_cast(&alpha), cuda_cast(B), n,
                                  cuda_cast(A), k, cuda_cast(&beta), cuda_cast(C), n);
        else if constexpr (std::is_same_v<T, std::complex<double>>)
            return cublasZgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, cuda_cast(&alpha), cuda_cast(B), n,
                                  cuda_cast(A), k, cuda_cast(&beta), cuda_cast(C), n);
    }

    template <typename T>
    cudaError_t multiply(const T* x, const T* y, T* z, size_t count);

    template <typename T>
    cudaError_t multiply(const T* x, T* y, const T& value, size_t count);

    template <typename T>
    cudaError_t divide(const T* x, const T* y, T* z, size_t count);

    template <typename T>
    cudaError_t divide(const T* x, T* y, const T& value, size_t count);

    template <typename T>
    cudaError_t divide(const T& value, const T* x, T* y, size_t count);
} // namespace mbq::cuda::detail