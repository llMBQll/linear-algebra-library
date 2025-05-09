#include "../kernels/common.cuh"
#include "../kernels/for_each.cuh"
#include "complex.cuh"
#include "pow.cuh"

template <typename T>
struct PowFn
{
    T exponent;

    __host__ __device__ __inline__ T operator()(T x)
    {
        using namespace mbq::cuda;
        return pow(x, exponent);
    }

    __host__ __device__ PowFn(const T& e) : exponent(e) { }
};

namespace mbq::cuda::detail
{
    template <typename T>
    cudaError_t pow(const T* in, T* out, size_t cnt, const T& exponent)
    {
        using type = decltype(cuda_cast(std::declval<T>()));
        return launch_for_each<type, PowFn<type>>(cuda_cast(in), cuda_cast(out), cnt, PowFn{cuda_cast(exponent)});
    }

    template cudaError_t pow<float>(const float* in, float* out, size_t cnt, const float&);
    template cudaError_t pow<double>(const double* in, double* out, size_t cnt, const double&);
    template cudaError_t pow<std::complex<float>>(const std::complex<float>* in, std::complex<float>* out, size_t cnt,
                                                  const std::complex<float>&);
    template cudaError_t pow<std::complex<double>>(const std::complex<double>* in, std::complex<double>* out,
                                                   size_t cnt, const std::complex<double>&);
} // namespace mbq::cuda::detail