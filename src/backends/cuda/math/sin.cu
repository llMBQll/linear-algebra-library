#include "../kernels/common.cuh"
#include "../kernels/for_each.cuh"
#include "complex.cuh"
#include "sin.cuh"

#define MBQ_CUDA_FN_IMPL(m__cls, m__fn)                                                                                \
    template <typename T>                                                                                              \
    struct m__cls                                                                                                      \
    {                                                                                                                  \
        __host__ __device__ T operator()(T x)                                                                          \
        {                                                                                                              \
            using namespace mbq::cuda;                                                                                 \
            return m__fn(x);                                                                                           \
        }                                                                                                              \
    };                                                                                                                 \
    namespace mbq::cuda::detail                                                                                        \
    {                                                                                                                  \
        template <typename T>                                                                                          \
        cudaError_t m__fn(const T* in, T* out, size_t cnt)                                                             \
        {                                                                                                              \
            using namespace mbq::cuda::detail;                                                                         \
            using type = decltype(cuda_cast(std::declval<T>()));                                                       \
            return launch_for_each<type, m__cls<type>>(cuda_cast(in), cuda_cast(out), cnt);                            \
        }                                                                                                              \
        template cudaError_t m__fn<float>(const float* in, float* out, size_t cnt);                                    \
        template cudaError_t m__fn<double>(const double* in, double* out, size_t cnt);                                 \
        template cudaError_t m__fn<std::complex<float>>(const std::complex<float>* in, std::complex<float>* out,       \
                                                        size_t cnt);                                                   \
        template cudaError_t m__fn<std::complex<double>>(const std::complex<double>* in, std::complex<double>* out,    \
                                                         size_t cnt);                                                  \
    }

MBQ_CUDA_FN_IMPL(SinFn, sin)
MBQ_CUDA_FN_IMPL(SinhFn, sinh)
MBQ_CUDA_FN_IMPL(CosFn, cos)
MBQ_CUDA_FN_IMPL(CoshFn, cosh)
MBQ_CUDA_FN_IMPL(TanFn, tan)
MBQ_CUDA_FN_IMPL(TanhFn, tanh)