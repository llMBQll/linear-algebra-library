#pragma once

#if MBQ_ENABLE_CUDA
    #include "cuda/algorithm.hpp"
    #include "cuda/math.hpp"
    #include "cuda/memory.hpp"
#else
    #error "Enable CUDA backend is disabled"
#endif