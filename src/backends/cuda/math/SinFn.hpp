#pragma once

#include "backends/cuda/memory.hpp"
#include "math/sin.hpp"
#include "sin.cuh"

#define MBQ_CUDA_FN(m__cls, m__fn)                                                                                     \
    template <typename T>                                                                                              \
    struct m__cls<cuda::Allocator<T>>                                                                                  \
    {                                                                                                                  \
        using value_type = T;                                                                                          \
        template <typename Iter>                                                                                       \
        constexpr auto operator()(Iter x_begin, Iter x_end) const                                                      \
        {                                                                                                              \
            auto res = cuda::detail::m__fn(&(*x_begin), &(*x_begin), x_end - x_begin);                                 \
            if (res != cudaError::cudaSuccess)                                                                         \
                MBQ_THROW_EXCEPTION(CudaException, res);                                                               \
            return x_end;                                                                                              \
        }                                                                                                              \
    }

namespace mbq
{
    MBQ_CUDA_FN(Sin, sin);
    MBQ_CUDA_FN(Sinh, sinh);
    MBQ_CUDA_FN(Cos, cos);
    MBQ_CUDA_FN(Cosh, cosh);
    MBQ_CUDA_FN(Tan, tan);
    MBQ_CUDA_FN(Tanh, tanh);
} // namespace mbq

#undef MBQ_CUDA_FN