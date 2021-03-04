#pragma once

#include <curand.h>

namespace mbq::cuda::detail
{
    template <typename T>
    curandStatus_t random(T* ptr, size_t count, T min, T max);
} // namespace mbq::cuda::detail