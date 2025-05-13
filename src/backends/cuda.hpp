#pragma once

#if MBQ_CUDA_BACKEND == ON
    #include "cuda/algorithm.hpp"
    #include "cuda/math.hpp"
    #include "cuda/memory.hpp"
#else
    #error "Enable CUDA backend is disabled"
#endif