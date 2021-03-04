#pragma once

#if MBQ_ENABLE_OPENCL
    #include "backends/opencl/algorithm.hpp"
    #include "backends/opencl/math.hpp"
    #include "backends/opencl/memory.hpp"
#else
    #error "Enable OpenCL backend is disabled"
#endif