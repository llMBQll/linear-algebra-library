cmake_minimum_required(VERSION 3.17)

option(MBQ_ENABLE_CUDA "Enable CUDA backed implementation" ON)
option(MBQ_ENABLE_OPENCL "Enable OpenCL backed implementation" ON)
option(MBQ_COMPILE_TESTS "Compile unit tests for the library" OFF)

if (MBQ_ENABLE_CUDA)
    project(linear-algebra-library CXX CUDA)
else ()
    project(linear-algebra-library CXX)
endif ()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MBQ_SOURCES
        src/algorithm/fill.hpp src/algorithm/random.hpp
        src/exceptions/Exception.hpp
        src/math/axpy.hpp src/math/pow.hpp src/math/sin.hpp src/math/arithmetic.hpp src/math/transpose.hpp
        src/memory/AllocatorTraits.hpp src/memory/Copy.hpp src/memory/Memory.hpp src/memory/MemoryBase.hpp
        src/memory/MemoryIterator.hpp src/memory/MemoryView.hpp src/memory/Reference.hpp
        src/Array.hpp src/ArrayView.hpp src/concepts.hpp src/types.hpp src/util.hpp)

##################### Configure Host #####################
include_directories("${MBQ_OPENBLAS_INCLUDE_DIRECTORY}")
find_package(OpenBLAS CONFIG REQUIRED)

# Collect all files for Host backend
set(MBQ_HOST_BACKEND
        src/backends/host/algorithm/FillFn.hpp src/backends/host/algorithm/RandomFn.hpp
        src/backends/host/math/ArithmeticFn.hpp src/backends/host/math/AxpyFn.hpp src/backends/host/math/PowFn.hpp
        src/backends/host/math/SinFn.hpp src/backends/host/math/TransposeFn.hpp
        src/backends/host/memory/Allocator.hpp src/backends/host/memory/Copy.hpp
        src/backends/host/algorithm.hpp src/backends/host/math.hpp src/backends/host/memory.hpp
        src/backends/host.hpp)

set(MBQ_SOURCES ${MBQ_SOURCES} ${MBQ_HOST_BACKEND})

##################### Configure CUDA #####################
add_compile_definitions(MBQ_ENABLE_CUDA=${MBQ_ENABLE_CUDA})

if (${MBQ_ENABLE_CUDA})
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_ARCHITECTURES "75")
    find_package(CUDAToolkit)

    # Compile kernels to a static library
    add_library(linear-algebra-library-cuda STATIC
            src/backends/cuda/algorithm/detail/fill.cu src/backends/cuda/algorithm/detail/fill.cuh
            src/backends/cuda/algorithm/detail/random.cu src/backends/cuda/algorithm/detail/random.cuh
            src/backends/cuda/kernels/common.cuh src/backends/cuda/kernels/for_each.cuh
            src/backends/cuda/math/arithmetic.cu src/backends/cuda/math/arithmetic.cuh
            src/backends/cuda/math/complex.cuh
            src/backends/cuda/math/pow.cu src/backends/cuda/math/pow.cuh
            src/backends/cuda/math/sin.cu src/backends/cuda/math/sin.cuh)
    set_target_properties(linear-algebra-library-cuda PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_link_libraries(linear-algebra-library-cuda CUDA::cudart_static CUDA::cublas CUDA::curand)

    # Collect all files for CUDA backend
    set(MBQ_CUDA_BACKEND
            src/backends/cuda/algorithm/detail/fill.cuh src/backends/cuda/algorithm/FillFn.hpp
            src/backends/cuda/algorithm/detail/random.cuh src/backends/cuda/algorithm/RandomFn.hpp
            src/backends/cuda/exceptions/CublasException.hpp src/backends/cuda/exceptions/CudaException.hpp
            src/backends/cuda/math/arithmetic.cuh src/backends/cuda/math/ArithmeticFn.hpp
            src/backends/cuda/math/AxpyFn.hpp
            src/backends/cuda/math/pow.cuh src/backends/cuda/math/PowFn.hpp
            src/backends/cuda/math/sin.cuh src/backends/cuda/math/SinFn.hpp
            src/backends/cuda/math/TransposeFn.hpp
            src/backends/cuda/memory/Allocator.hpp src/backends/cuda/memory/Context.hpp src/backends/cuda/memory/Copy.hpp
            src/backends/cuda/algorithm.hpp src/backends/cuda/math.hpp src/backends/cuda/memory.hpp
            src/backends/cuda.hpp)

    set(MBQ_SOURCES ${MBQ_SOURCES} ${MBQ_CUDA_BACKEND})
endif ()

#################### Configure OpenCL ####################
add_compile_definitions(MBQ_ENABLE_OPENCL=${MBQ_ENABLE_OPENCL})

if (${MBQ_ENABLE_OPENCL})
    find_package(OpenCL REQUIRED)
    find_package(CLBlast CONFIG REQUIRED)

    # Collect all files for OpenCL backend
    set(MBQ_OPENCL_BACKEND
            src/backends/opencl/algorithm/FillFn.hpp src/backends/opencl/algorithm/RandomFn.hpp
            src/backends/opencl/exceptions/CLBlastException.hpp src/backends/opencl/exceptions/OpenCLException.hpp
            src/backends/opencl/kernels/Kernel.hpp
            src/backends/opencl/math/AxpyFn.hpp src/backends/opencl/math/AxpyFn.hpp
            src/backends/opencl/math/complex.hpp src/backends/opencl/math/PowFn.hpp
            src/backends/opencl/math/SinFn.hpp src/backends/opencl/math/TransposeFn.hpp
            src/backends/opencl/memory/Allocator.hpp src/backends/opencl/memory/Context.hpp
            src/backends/opencl/memory/Copy.hpp src/backends/opencl/memory/Pointer.hpp
            src/backends/opencl/algorithm.hpp src/backends/opencl/common.hpp
            src/backends/opencl/math.hpp src/backends/opencl/memory.hpp
            src/backends/opencl.hpp)

    set(MBQ_SOURCES ${MBQ_SOURCES} ${MBQ_OPENCL_BACKEND})
endif ()

add_library(linear-algebra-library INTERFACE ${MBQ_SOURCES})

# Host
target_link_libraries(linear-algebra-library INTERFACE OpenBLAS::OpenBLAS)

# CUDA
if (${MBQ_ENABLE_CUDA})
    target_include_directories(linear-algebra-library INTERFACE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    target_link_libraries(linear-algebra-library INTERFACE linear-algebra-library-cuda)
endif ()

# OpenCL
if (${MBQ_ENABLE_OPENCL})
    target_link_libraries(linear-algebra-library INTERFACE OpenCL::OpenCL)
    target_link_libraries(linear-algebra-library INTERFACE clblast)
endif ()

if (${MBQ_COMPILE_TESTS})
    add_subdirectory(tests)
endif ()
