#include "../../kernels/common.cuh"
#include "../../kernels/for_each.cuh"
#include "fill.cuh"

#include <complex>

namespace mbq::cuda::detail
{
    template <typename T>
    cudaError_t fill(T* ptr, size_t count, T value)
    {
        return launch_for_each(cuda_cast(ptr), count, cuda_cast(value));
    }

    template cudaError_t fill<float>(float* ptr, size_t count, float value);
    template cudaError_t fill<double>(double* ptr, size_t count, double value);
    template cudaError_t fill<std::complex<float>>(std::complex<float>* ptr, size_t count, std::complex<float> value);
    template cudaError_t fill<std::complex<double>>(std::complex<double>* ptr, size_t count,
                                                    std::complex<double> value);
} // namespace mbq::cuda::detail