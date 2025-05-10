#pragma once

#include "backends/opencl/common.hpp"
#include "backends/opencl/exceptions/OpenCLException.hpp"
#include "backends/opencl/math/complex.hpp"
#include "backends/opencl/memory/Context.hpp"
#include "backends/opencl/memory/Pointer.hpp"

#include <chrono>
#include <print>
#include <unordered_map>

namespace mbq::opencl
{
    struct Sizes
    {
        size_t global_offset{0};
        size_t global_size{0};
        size_t local_size{0};
        size_t N{0};
    };

    class KernelBase
    {
    private:
        std::unordered_map<cl_device_id, cl_kernel> _kernels;
        const char* _name;
        const char* _src;
    private:
        template <typename... Args>
        static void set_args(cl_kernel kernel, Args... args)
        {
            constexpr auto check = [](cl_int status, const std::source_location& location = {}) {
                if (status != CL_SUCCESS)
                    MBQ_THROW_EXCEPTION(OpenCLException, status, location);
            };

            cl_uint index = 0;
            (check(clSetKernelArg(kernel, index++, sizeof(Args), static_cast<const void*>(&args))), ...);
        }

        [[noreturn]] static void exit_with_compile_error(Context* ctx, cl_program program)
        {
            cl_int status = CL_SUCCESS;

            size_t len = 0;
            status = clGetProgramBuildInfo(program, ctx->device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
            if (status != CL_SUCCESS)
            {
                std::println(stderr, "{}", OpenCLException{status});
                std::exit(status);
            }

            auto buf = std::make_unique<char[]>(len);
            status = clGetProgramBuildInfo(program, ctx->device_id, CL_PROGRAM_BUILD_LOG, len, buf.get(), nullptr);
            if (status != CL_SUCCESS)
            {
                std::println(stderr, "{}", OpenCLException{status});
                std::exit(status);
            }

            std::cerr << buf.get() << std::endl;
            std::exit(1);
        }

        cl_kernel find_or_compile(Context* ctx)
        {
            const auto it = _kernels.find(ctx->device_id);
            if (it != _kernels.cend())
                return it->second;

            cl_int status = CL_SUCCESS;

            cl_program program = clCreateProgramWithSource(ctx->context, 1, &_src, nullptr, &status);
            if (status != CL_SUCCESS)
                MBQ_THROW_EXCEPTION(OpenCLException, status);

            // 0 length device list - use devices associated with context used to create thr program object
            status = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);

            if (status != CL_SUCCESS)
                exit_with_compile_error(ctx, program);

            cl_kernel kernel = clCreateKernel(program, _name, &status);
            if (status != CL_SUCCESS)
                MBQ_THROW_EXCEPTION(OpenCLException, status);

            status = clReleaseProgram(program);
            if (status != CL_SUCCESS)
                MBQ_THROW_EXCEPTION(OpenCLException, status);

            _kernels[ctx->device_id] = kernel;
            return kernel;
        }
    protected:
        KernelBase(const char* name, const char* src) : _name(name), _src(src) { }

        template <typename Ret, typename... Args>
        Ret call(Context* ctx, Sizes sizes, Args... args)
        {
            auto kernel = find_or_compile(ctx);

            if (sizes.N > 0)
                set_args(kernel, static_cast<unsigned int>(sizes.N), args...);
            else
                set_args(kernel, args...);

            auto status = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, &sizes.global_offset,
                                                 &sizes.global_size, &sizes.local_size, 0, nullptr, nullptr);
            if (status != CL_SUCCESS)
                MBQ_THROW_EXCEPTION(OpenCLException, status);
        }
    };

    namespace detail
    {
        template <typename...>
        struct Types
        { };

        template <typename...>
        struct Bufs
        { };

        template <typename...>
        struct Args
        { };

        namespace execution_policy
        {
            struct parallel
            { };
            struct sequential
            { };
        } // namespace execution_policy
    } // namespace detail

    template <typename...>
    class Kernel;

    template <typename... BufTypes, typename... ArgTypes>
    class Kernel<detail::Bufs<BufTypes...>, detail::Args<ArgTypes...>> : public KernelBase
    {
    private:
        void call_impl(Context* ctx, Sizes sizes, Pointer<BufTypes>... bufs, ArgTypes... args)
        {
            return call<void>(ctx, sizes, bufs.ptr()..., static_cast<unsigned int>(bufs.off())..., args...);
        }
    public:
        Kernel(const char* name, const char* src) : KernelBase(name, src) { }

        void operator()(Context* ctx, Pointer<BufTypes>... bufs, size_t len, ArgTypes... args,
                        detail::execution_policy::parallel)
        {
            size_t group_size = 1'024;
            size_t mod = len % group_size;
            size_t global_size = len + (mod ? group_size - mod : 0);

            return call_impl(ctx, {0, global_size, group_size, len}, bufs..., args...);
        }

        void operator()(Context* ctx, Pointer<BufTypes>... bufs, ArgTypes... args, detail::execution_policy::sequential)
        {
            return call_impl(ctx, {0, 1, 1, 0}, bufs..., args...);
        }
    };

#define MBQ_STRINGIFY_IMPL(X) #X
#define MBQ_STRINGIFY(X) MBQ_STRINGIFY_IMPL(X)

#define MBQ_UNARY_FOREACH_BUILTIN_KERNEL(name, letter, cpp_type, type)                                                 \
    constexpr const char* name##letter##_src =                                                                         \
        "__kernel void " #name #letter "(unsigned int N, __global " #type "* x, unsigned int offset) \n"               \
        "{ \n"                                                                                                         \
        "    int index = get_global_id(0); \n"                                                                         \
        "    if (index < N) \n"                                                                                        \
        "        x[index + offset] = " #name "(x[index + offset]); \n"                                                 \
        "}";                                                                                                           \
    inline Kernel<Bufs<cpp_type>, Args<>> name##letter(#name #letter, name##letter##_src)

#define MBQ_UNARY_FOREACH_BUILTIN_KERNEL_COMPLEX(name, letter, cpp_type, type, include)                                \
    constexpr const char* name##letter##_src =                                                                         \
        MBQ_STRINGIFY(COMPLEX_##include) "__kernel void " #name #letter "(unsigned int N, __global " #type             \
                                         "* x, unsigned int offset) \n"                                                \
                                         "{ \n"                                                                        \
                                         "    int index = get_global_id(0); \n"                                        \
                                         "    if (index < N) \n"                                                       \
                                         "        x[index + offset] = " #name "_" #letter "(x[index + offset]); \n"    \
                                         "}";                                                                          \
    inline Kernel<Bufs<cpp_type>, Args<>> name##letter(#name #letter, name##letter##_src)

#define MBQ_UNARY_FOREACH_BUILTIN_KERNEL_GROUP(name)                                                                   \
    namespace detail                                                                                                   \
    {                                                                                                                  \
        MBQ_UNARY_FOREACH_BUILTIN_KERNEL(name, s, float, float);                                                       \
        MBQ_UNARY_FOREACH_BUILTIN_KERNEL(name, d, double, double);                                                     \
        MBQ_UNARY_FOREACH_BUILTIN_KERNEL_COMPLEX(name, c, std::complex<float>, float2, FLOAT);                         \
        MBQ_UNARY_FOREACH_BUILTIN_KERNEL_COMPLEX(name, z, std::complex<double>, double2, DOUBLE);                      \
        inline void name(Context* ctx, Pointer<float> ptr, size_t len)                                                 \
        {                                                                                                              \
            name##s(ctx, ptr, len, execution_policy::parallel{});                                                      \
        }                                                                                                              \
        inline void name(Context* ctx, Pointer<double> ptr, size_t len)                                                \
        {                                                                                                              \
            name##d(ctx, ptr, len, execution_policy::parallel{});                                                      \
        }                                                                                                              \
        inline void name(Context* ctx, Pointer<std::complex<float>> ptr, size_t len)                                   \
        {                                                                                                              \
            name##c(ctx, ptr, len, execution_policy::parallel{});                                                      \
        }                                                                                                              \
        inline void name(Context* ctx, Pointer<std::complex<double>> ptr, size_t len)                                  \
        {                                                                                                              \
            name##z(ctx, ptr, len, execution_policy::parallel{});                                                      \
        }                                                                                                              \
    }

#define MBQ_UNARY_FOREACH_ARG_BUILTIN_KERNEL(name, letter, cpp_type, type)                                             \
    constexpr const char* name##letter##_src = "__kernel void " #name #letter "(unsigned int N, __global " #type       \
                                               "* x, unsigned int offset, " #type " arg) \n"                           \
                                               "{ \n"                                                                  \
                                               "    int index = get_global_id(0); \n"                                  \
                                               "    if (index < N) \n"                                                 \
                                               "        x[index + offset] = " #name "(x[index + offset], arg); \n"     \
                                               "}";                                                                    \
    inline Kernel<Bufs<cpp_type>, Args<cpp_type>> name##letter(#name #letter, name##letter##_src)

#define MBQ_UNARY_FOREACH_ARG_BUILTIN_KERNEL_COMPLEX(name, letter, cpp_type, type, include)                            \
    constexpr const char* name##letter##_src =                                                                         \
        MBQ_STRINGIFY(COMPLEX_##include) "__kernel void " #name #letter "(unsigned int N, __global " #type             \
                                         "* x, unsigned int offset, " #type " arg) \n"                                 \
                                         "{ \n"                                                                        \
                                         "    int index = get_global_id(0); \n"                                        \
                                         "    if (index < N) \n"                                                       \
                                         "        x[index + offset] = " #name "_" #letter                              \
                                         "(x[index + offset], arg); \n"                                                \
                                         "}";                                                                          \
    inline Kernel<Bufs<cpp_type>, Args<cpp_type>> name##letter(#name #letter, name##letter##_src)

#define MBQ_UNARY_FOREACH_ARG_BUILTIN_KERNEL_GROUP(name)                                                               \
    namespace detail                                                                                                   \
    {                                                                                                                  \
        MBQ_UNARY_FOREACH_ARG_BUILTIN_KERNEL(name, s, float, float);                                                   \
        MBQ_UNARY_FOREACH_ARG_BUILTIN_KERNEL(name, d, double, double);                                                 \
        MBQ_UNARY_FOREACH_ARG_BUILTIN_KERNEL_COMPLEX(name, c, std::complex<float>, float2, FLOAT);                     \
        MBQ_UNARY_FOREACH_ARG_BUILTIN_KERNEL_COMPLEX(name, z, std::complex<double>, double2, DOUBLE);                  \
        inline void name(Context* ctx, Pointer<float> ptr, size_t len, const float& arg)                               \
        {                                                                                                              \
            name##s(ctx, ptr, len, arg, execution_policy::parallel{});                                                 \
        }                                                                                                              \
        inline void name(Context* ctx, Pointer<double> ptr, size_t len, const double& arg)                             \
        {                                                                                                              \
            name##d(ctx, ptr, len, arg, execution_policy::parallel{});                                                 \
        }                                                                                                              \
        inline void name(Context* ctx, Pointer<std::complex<float>> ptr, size_t len, const std::complex<float>& arg)   \
        {                                                                                                              \
            name##c(ctx, ptr, len, arg, execution_policy::parallel{});                                                 \
        }                                                                                                              \
        inline void name(Context* ctx, Pointer<std::complex<double>> ptr, size_t len, const std::complex<double>& arg) \
        {                                                                                                              \
            name##z(ctx, ptr, len, arg, execution_policy::parallel{});                                                 \
        }                                                                                                              \
    }

#define MBQ_UNARY_FOREACH_ARG_KERNEL(name, letter, cpp_type, type, include)                                            \
    constexpr const char* name##letter##_src =                                                                         \
        MBQ_STRINGIFY(include(name, letter, type)) "__kernel void " #name #letter "(unsigned int N, __global " #type   \
                                                   "* x, __global " #type                                              \
                                                   "* y, unsigned int offset_x, unsigned int offset_y, " #type         \
                                                   " arg) \n"                                                          \
                                                   "{ \n"                                                              \
                                                   "    int index = get_global_id(0); \n"                              \
                                                   "    if (index < N) \n"                                             \
                                                   "        y[index + offset_y] = " #name "_" #letter                  \
                                                   "(x[index + offset_x], arg); \n"                                    \
                                                   "}";                                                                \
    inline Kernel<Bufs<cpp_type, cpp_type>, Args<cpp_type>> name##letter(#name #letter, name##letter##_src)

#define MBQ_UNARY_FOREACH_ARG_KERNEL_GROUP(name, include)                                                              \
    namespace detail                                                                                                   \
    {                                                                                                                  \
        MBQ_UNARY_FOREACH_ARG_KERNEL(name, s, float, float, include);                                                  \
        MBQ_UNARY_FOREACH_ARG_KERNEL(name, d, double, double, include);                                                \
        MBQ_UNARY_FOREACH_ARG_KERNEL(name, c, std::complex<float>, float2, include##_COMPLEX);                         \
        MBQ_UNARY_FOREACH_ARG_KERNEL(name, z, std::complex<double>, double2, include##_COMPLEX);                       \
        inline void name(Context* ctx, Pointer<float> x, Pointer<float> y, size_t len, const float& arg)               \
        {                                                                                                              \
            name##s(ctx, x, y, len, arg, execution_policy::parallel{});                                                \
        }                                                                                                              \
        inline void name(Context* ctx, Pointer<double> x, Pointer<double> y, size_t len, const double& arg)            \
        {                                                                                                              \
            name##d(ctx, x, y, len, arg, execution_policy::parallel{});                                                \
        }                                                                                                              \
        inline void name(Context* ctx, Pointer<std::complex<float>> x, Pointer<std::complex<float>> y, size_t len,     \
                         const std::complex<float>& arg)                                                               \
        {                                                                                                              \
            name##c(ctx, x, y, len, arg, execution_policy::parallel{});                                                \
        }                                                                                                              \
        inline void name(Context* ctx, Pointer<std::complex<double>> x, Pointer<std::complex<double>> y, size_t len,   \
                         const std::complex<double>& arg)                                                              \
        {                                                                                                              \
            name##z(ctx, x, y, len, arg, execution_policy::parallel{});                                                \
        }                                                                                                              \
    }

#define MBQ_BINARY_FOREACH_KERNEL(name, letter, cpp_type, type, include)                                               \
    constexpr const char* name##letter##_src = MBQ_STRINGIFY(                                                          \
        include(name, letter, type)) "__kernel void " #name #letter "(unsigned int N, __global " #type                 \
                                     "* x, __global " #type "* y, __global " #type                                     \
                                     "* z, unsigned int x_offset, unsigned int y_offset, unsigned int z_offset) \n"    \
                                     "{ \n"                                                                            \
                                     "    int index = get_global_id(0); \n"                                            \
                                     "    if (index < N) \n"                                                           \
                                     "        z[index + z_offset] = " #name "_" #letter                                \
                                     "(x[index + x_offset], y[index + y_offset]); \n"                                  \
                                     "}";                                                                              \
    inline Kernel<Bufs<cpp_type, cpp_type, cpp_type>, Args<>> name##letter(#name #letter, name##letter##_src)

#define MBQ_BINARY_FOREACH_KERNEL_GROUP(name, include)                                                                 \
    namespace detail                                                                                                   \
    {                                                                                                                  \
        MBQ_BINARY_FOREACH_KERNEL(name, s, float, float, include);                                                     \
        MBQ_BINARY_FOREACH_KERNEL(name, d, double, double, include);                                                   \
        MBQ_BINARY_FOREACH_KERNEL(name, c, std::complex<float>, float2, include##_COMPLEX);                            \
        MBQ_BINARY_FOREACH_KERNEL(name, z, std::complex<double>, double2, include##_COMPLEX);                          \
        inline void name(Context* ctx, Pointer<float> x, Pointer<float> y, Pointer<float> z, size_t len)               \
        {                                                                                                              \
            name##s(ctx, x, y, z, len, execution_policy::parallel{});                                                  \
        }                                                                                                              \
        inline void name(Context* ctx, Pointer<double> x, Pointer<double> y, Pointer<double> z, size_t len)            \
        {                                                                                                              \
            name##d(ctx, x, y, z, len, execution_policy::parallel{});                                                  \
        }                                                                                                              \
        inline void name(Context* ctx, Pointer<std::complex<float>> x, Pointer<std::complex<float>> y,                 \
                         Pointer<std::complex<float>> z, size_t len)                                                   \
        {                                                                                                              \
            name##c(ctx, x, y, z, len, execution_policy::parallel{});                                                  \
        }                                                                                                              \
        inline void name(Context* ctx, Pointer<std::complex<double>> x, Pointer<std::complex<double>> y,               \
                         Pointer<std::complex<double>> z, size_t len)                                                  \
        {                                                                                                              \
            name##z(ctx, x, y, z, len, execution_policy::parallel{});                                                  \
        }                                                                                                              \
    }

    // TRIGONOMETRIC FUNCTIONS

    MBQ_UNARY_FOREACH_BUILTIN_KERNEL_GROUP(sin)
    MBQ_UNARY_FOREACH_BUILTIN_KERNEL_GROUP(sinh)
    MBQ_UNARY_FOREACH_BUILTIN_KERNEL_GROUP(cos)
    MBQ_UNARY_FOREACH_BUILTIN_KERNEL_GROUP(cosh)
    MBQ_UNARY_FOREACH_BUILTIN_KERNEL_GROUP(tan)
    MBQ_UNARY_FOREACH_BUILTIN_KERNEL_GROUP(tanh)

    // POW

    MBQ_UNARY_FOREACH_ARG_BUILTIN_KERNEL_GROUP(pow);

    // OPERATOR +

#define MBQ_ADD(name, letter, type)                                                                                    \
    inline type name##_##letter(type x, type y)                                                                        \
    {                                                                                                                  \
        return x + y;                                                                                                  \
    }

#define MBQ_ADD_COMPLEX(name, letter, type)                                                                            \
    inline type name##_##letter(type x, type y)                                                                        \
    {                                                                                                                  \
        type z;                                                                                                        \
        z.x = x.x + y.x;                                                                                               \
        z.y = x.y + y.y;                                                                                               \
        return z;                                                                                                      \
    }

    MBQ_BINARY_FOREACH_KERNEL_GROUP(add, MBQ_ADD)
    MBQ_UNARY_FOREACH_ARG_KERNEL_GROUP(add_arg, MBQ_ADD)

    // OPERATOR -

#define MBQ_SUBTRACT(name, letter, type)                                                                               \
    inline type name##_##letter(type x, type y)                                                                        \
    {                                                                                                                  \
        return x - y;                                                                                                  \
    }

#define MBQ_SUBTRACT_COMPLEX(name, letter, type)                                                                       \
    inline type name##_##letter(type x, type y)                                                                        \
    {                                                                                                                  \
        type z;                                                                                                        \
        z.x = x.x - y.x;                                                                                               \
        z.y = x.y - y.y;                                                                                               \
        return z;                                                                                                      \
    }

#define MBQ_SUBTRACT_REVERSE(name, letter, type)                                                                       \
    inline type name##_##letter(type x, type y)                                                                        \
    {                                                                                                                  \
        return y - x;                                                                                                  \
    }

#define MBQ_SUBTRACT_REVERSE_COMPLEX(name, letter, type)                                                               \
    inline type name##_##letter(type x, type y)                                                                        \
    {                                                                                                                  \
        type z;                                                                                                        \
        z.x = y.x - x.x;                                                                                               \
        z.y = y.y - x.y;                                                                                               \
        return z;                                                                                                      \
    }

    MBQ_BINARY_FOREACH_KERNEL_GROUP(subtract, MBQ_SUBTRACT)
    MBQ_UNARY_FOREACH_ARG_KERNEL_GROUP(subtract_arg, MBQ_SUBTRACT)
    MBQ_UNARY_FOREACH_ARG_KERNEL_GROUP(subtract_reverse_arg, MBQ_SUBTRACT_REVERSE)

    // OPERATOR *

#define MBQ_MULTIPLY(name, letter, type)                                                                               \
    inline type name##_##letter(type x, type y)                                                                        \
    {                                                                                                                  \
        return x * y;                                                                                                  \
    }

#define MBQ_MULTIPLY_COMPLEX(name, letter, type)                                                                       \
    inline type name##_##letter(type x, type y)                                                                        \
    {                                                                                                                  \
        type z;                                                                                                        \
        z.x = x.x * y.x - x.y * y.y;                                                                                   \
        z.y = x.x * y.y + x.y * y.x;                                                                                   \
        return z;                                                                                                      \
    }

    MBQ_BINARY_FOREACH_KERNEL_GROUP(multiply, MBQ_MULTIPLY)
    MBQ_UNARY_FOREACH_ARG_KERNEL_GROUP(multiply_arg, MBQ_MULTIPLY)

    // OPERATOR /

#define MBQ_DIVIDE(name, letter, type)                                                                                 \
    inline type name##_##letter(type x, type y)                                                                        \
    {                                                                                                                  \
        return x / y;                                                                                                  \
    }

#define MBQ_DIVIDE_COMPLEX(name, letter, type)                                                                         \
    inline type name##_##letter(type x, type y)                                                                        \
    {                                                                                                                  \
        type z;                                                                                                        \
        z.x = x.x - y.x;                                                                                               \
        z.y = x.y - y.y;                                                                                               \
        return z;                                                                                                      \
    }

#define MBQ_DIVIDE_REVERSE(name, letter, type)                                                                         \
    inline type name##_##letter(type x, type y)                                                                        \
    {                                                                                                                  \
        return y / x;                                                                                                  \
    }

#define MBQ_DIVIDE_REVERSE_COMPLEX(name, letter, type)                                                                 \
    inline type name##_##letter(type x, type y)                                                                        \
    {                                                                                                                  \
        type z;                                                                                                        \
        z.x = y.x - x.x;                                                                                               \
        z.y = y.y - x.y;                                                                                               \
        return z;                                                                                                      \
    }

    MBQ_BINARY_FOREACH_KERNEL_GROUP(divide, MBQ_DIVIDE)
    MBQ_UNARY_FOREACH_ARG_KERNEL_GROUP(divide_arg, MBQ_DIVIDE)
    MBQ_UNARY_FOREACH_ARG_KERNEL_GROUP(divide_reverse_arg, MBQ_DIVIDE_REVERSE)
} // namespace mbq::opencl

#undef MBQ_UNARY_FOREACH_BUILTIN_KERNEL
#undef MBQ_UNARY_FOREACH_BUILTIN_KERNEL_COMPLEX
#undef MBQ_UNARY_FOREACH_BUILTIN_KERNEL_GROUP
#undef MBQ_UNARY_FOREACH_ARG_KERNEL
#undef MBQ_UNARY_FOREACH_ARG_KERNEL_GROUP
#undef MBQ_BINARY_FOREACH_KERNEL
#undef MBQ_BINARY_FOREACH_KERNEL_GROUP
#undef MBQ_PLUS
#undef MBQ_PLUS_COMPLEX
#undef MBQ_MINUS
#undef MBQ_MINUS_COMPLEX
#undef COMPLEX_DOUBLE
#undef COMPLEX_FLOAT