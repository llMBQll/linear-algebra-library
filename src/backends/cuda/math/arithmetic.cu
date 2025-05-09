#include "../kernels/common.cuh"
#include "../kernels/for_each.cuh"
#include "arithmetic.cuh"
#include "complex.cuh"

namespace mbq::cuda::detail
{
    template <typename T>
    struct AddValues
    {
        __device__ __inline__ T operator()(T x, T y) const
        {
            return x + y;
        }
    };

    template <typename T>
    struct AddConstant
    {
    private:
        T _value;
    public:
        constexpr explicit AddConstant(const T& value) : _value(value) { }
        __device__ __inline__ T operator()(T x) const
        {
            return x + _value;
        }
    };

    template <typename T>
    cudaError_t add(const T* x, const T* y, T* z, size_t count)
    {
        using type = decltype(cuda_cast(std::declval<T>()));
        return launch_for_each<type, AddValues<type>>(cuda_cast(x), cuda_cast(y), cuda_cast(z), count);
    }

    template <typename T>
    cudaError_t add(const T* x, T* y, const T& value, size_t count)
    {
        using type = decltype(cuda_cast(std::declval<T>()));
        return launch_for_each<type, AddConstant<type>>(cuda_cast(x), cuda_cast(y), count,
                                                        AddConstant<type>{cuda_cast(value)});
    }

    template cudaError_t add<float>(const float* x, const float* y, float* z, size_t count);
    template cudaError_t add<double>(const double* x, const double* y, double* z, size_t count);
    template cudaError_t add<std::complex<float>>(const std::complex<float>* x, const std::complex<float>* y,
                                                  std::complex<float>* z, size_t count);
    template cudaError_t add<std::complex<double>>(const std::complex<double>* x, const std::complex<double>* y,
                                                   std::complex<double>* z, size_t count);

    template cudaError_t add<float>(const float* x, float* y, const float& value, size_t count);
    template cudaError_t add<double>(const double* x, double* y, const double& value, size_t count);
    template cudaError_t add<std::complex<float>>(const std::complex<float>* x, std::complex<float>* y,
                                                  const std::complex<float>& value, size_t count);
    template cudaError_t add<std::complex<double>>(const std::complex<double>* x, std::complex<double>* y,
                                                   const std::complex<double>& value, size_t count);

    template <typename T>
    struct SubtractValues
    {
        __device__ __inline__ T operator()(T x, T y) const
        {
            return x - y;
        }
    };

    template <typename T>
    struct SubtractConstant
    {
    private:
        T _value;
    public:
        constexpr explicit SubtractConstant(const T& value) : _value(value) { }
        __device__ __inline__ T operator()(T x) const
        {
            return x - _value;
        }
    };

    template <typename T>
    struct SubtractConstantReverse
    {
    private:
        T _value;
    public:
        constexpr explicit SubtractConstantReverse(const T& value) : _value(value) { }
        __device__ __inline__ T operator()(T x) const
        {
            return _value - x;
        }
    };

    template <typename T>
    cudaError_t subtract(const T* x_in, const T* y_in, T* out, size_t count)
    {
        using type = decltype(cuda_cast(std::declval<T>()));
        return launch_for_each<type, SubtractValues<type>>(cuda_cast(x_in), cuda_cast(y_in), cuda_cast(out), count);
    }

    template <typename T>
    cudaError_t subtract(const T* in, T* out, const T& value, size_t count)
    {
        using type = decltype(cuda_cast(std::declval<T>()));
        return launch_for_each<type, SubtractConstant<type>>(cuda_cast(in), cuda_cast(out), count,
                                                             SubtractConstant<type>{cuda_cast(value)});
    }

    template <typename T>
    cudaError_t subtract(const T& value, const T* in, T* out, size_t count)
    {
        using type = decltype(cuda_cast(std::declval<T>()));
        return launch_for_each<type, SubtractConstantReverse<type>>(cuda_cast(in), cuda_cast(out), count,
                                                                    SubtractConstantReverse<type>{cuda_cast(value)});
    }

    template cudaError_t subtract<float>(const float* x, const float* y, float* z, size_t count);
    template cudaError_t subtract<double>(const double* x, const double* y, double* z, size_t count);
    template cudaError_t subtract<std::complex<float>>(const std::complex<float>* x, const std::complex<float>* y,
                                                       std::complex<float>* z, size_t count);
    template cudaError_t subtract<std::complex<double>>(const std::complex<double>* x, const std::complex<double>* y,
                                                        std::complex<double>* z, size_t count);

    template cudaError_t subtract<float>(const float* x, float* y, const float& value, size_t count);
    template cudaError_t subtract<double>(const double* x, double* y, const double& value, size_t count);
    template cudaError_t subtract<std::complex<float>>(const std::complex<float>* x, std::complex<float>* y,
                                                       const std::complex<float>& value, size_t count);
    template cudaError_t subtract<std::complex<double>>(const std::complex<double>* x, std::complex<double>* y,
                                                        const std::complex<double>& value, size_t count);

    template cudaError_t subtract<float>(const float& value, const float* x, float* y, size_t count);
    template cudaError_t subtract<double>(const double& value, const double* x, double* y, size_t count);
    template cudaError_t subtract<std::complex<float>>(const std::complex<float>& value, const std::complex<float>* x,
                                                       std::complex<float>* y, size_t count);
    template cudaError_t subtract<std::complex<double>>(const std::complex<double>& value,
                                                        const std::complex<double>* x, std::complex<double>* y,
                                                        size_t count);

    template <typename T>
    struct MultiplyValues
    {
        __device__ __inline__ T operator()(T x, T y) const
        {
            return x * y;
        }
    };

    template <typename T>
    struct MultiplyConstant
    {
    private:
        T _value;
    public:
        constexpr explicit MultiplyConstant(const T& value) : _value(value) { }
        __device__ __inline__ T operator()(T x) const
        {
            return x * _value;
        }
    };

    template <typename T>
    cudaError_t multiply(const T* x_in, const T* y_in, T* out, size_t count)
    {
        using type = decltype(cuda_cast(std::declval<T>()));
        return launch_for_each<type, MultiplyValues<type>>(cuda_cast(x_in), cuda_cast(y_in), cuda_cast(out), count);
    }

    template <typename T>
    cudaError_t multiply(const T* in, T* out, const T& value, size_t count)
    {
        using type = decltype(cuda_cast(std::declval<T>()));
        return launch_for_each<type, MultiplyConstant<type>>(cuda_cast(in), cuda_cast(out), count,
                                                             MultiplyConstant<type>{cuda_cast(value)});
    }

    template cudaError_t multiply<float>(const float* x, const float* y, float* z, size_t count);
    template cudaError_t multiply<double>(const double* x, const double* y, double* z, size_t count);
    template cudaError_t multiply<std::complex<float>>(const std::complex<float>* x, const std::complex<float>* y,
                                                       std::complex<float>* z, size_t count);
    template cudaError_t multiply<std::complex<double>>(const std::complex<double>* x, const std::complex<double>* y,
                                                        std::complex<double>* z, size_t count);

    template cudaError_t multiply<float>(const float* x, float* y, const float& value, size_t count);
    template cudaError_t multiply<double>(const double* x, double* y, const double& value, size_t count);
    template cudaError_t multiply<std::complex<float>>(const std::complex<float>* x, std::complex<float>* y,
                                                       const std::complex<float>& value, size_t count);
    template cudaError_t multiply<std::complex<double>>(const std::complex<double>* x, std::complex<double>* y,
                                                        const std::complex<double>& value, size_t count);

    template <typename T>
    struct DivideValues
    {
        __device__ __inline__ T operator()(T x, T y) const
        {
            return x / y;
        }
    };

    template <typename T>
    struct DivideConstant
    {
    private:
        T _value;
    public:
        constexpr explicit DivideConstant(const T& value) : _value(value) { }
        __device__ __inline__ T operator()(T x) const
        {
            return x / _value;
        }
    };

    template <typename T>
    struct DivideConstantReverse
    {
    private:
        T _value;
    public:
        constexpr explicit DivideConstantReverse(const T& value) : _value(value) { }
        __device__ __inline__ T operator()(T x) const
        {
            return _value / x;
        }
    };

    template <typename T>
    cudaError_t divide(const T* x_in, const T* y_in, T* out, size_t count)
    {
        using type = decltype(cuda_cast(std::declval<T>()));
        return launch_for_each<type, DivideValues<type>>(cuda_cast(x_in), cuda_cast(y_in), cuda_cast(out), count);
    }

    template <typename T>
    cudaError_t divide(const T* in, T* out, const T& value, size_t count)
    {
        using type = decltype(cuda_cast(std::declval<T>()));
        return launch_for_each<type, DivideConstant<type>>(cuda_cast(in), cuda_cast(out), count,
                                                           DivideConstant<type>{cuda_cast(value)});
    }

    template <typename T>
    cudaError_t divide(const T& value, const T* in, T* out, size_t count)
    {
        using type = decltype(cuda_cast(std::declval<T>()));
        return launch_for_each<type, DivideConstantReverse<type>>(cuda_cast(in), cuda_cast(out), count,
                                                                  DivideConstantReverse<type>{cuda_cast(value)});
    }

    template cudaError_t divide<float>(const float* x, const float* y, float* z, size_t count);
    template cudaError_t divide<double>(const double* x, const double* y, double* z, size_t count);
    template cudaError_t divide<std::complex<float>>(const std::complex<float>* x, const std::complex<float>* y,
                                                     std::complex<float>* z, size_t count);
    template cudaError_t divide<std::complex<double>>(const std::complex<double>* x, const std::complex<double>* y,
                                                      std::complex<double>* z, size_t count);

    template cudaError_t divide<float>(const float* x, float* y, const float& value, size_t count);
    template cudaError_t divide<double>(const double* x, double* y, const double& value, size_t count);
    template cudaError_t divide<std::complex<float>>(const std::complex<float>* x, std::complex<float>* y,
                                                     const std::complex<float>& value, size_t count);
    template cudaError_t divide<std::complex<double>>(const std::complex<double>* x, std::complex<double>* y,
                                                      const std::complex<double>& value, size_t count);

    template cudaError_t divide<float>(const float& value, const float* x, float* y, size_t count);
    template cudaError_t divide<double>(const double& value, const double* x, double* y, size_t count);
    template cudaError_t divide<std::complex<float>>(const std::complex<float>& value, const std::complex<float>* x,
                                                     std::complex<float>* y, size_t count);
    template cudaError_t divide<std::complex<double>>(const std::complex<double>& value, const std::complex<double>* x,
                                                      std::complex<double>* y, size_t count);
} // namespace mbq::cuda::detail