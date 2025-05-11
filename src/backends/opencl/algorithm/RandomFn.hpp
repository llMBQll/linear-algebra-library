#pragma once

#include "algorithm/random.hpp"
#include "backends/opencl/kernels/Kernel.hpp"
#include "backends/opencl/memory/Allocator.hpp"

#include <random>
#include <unordered_map>

#define MBQ_MT19937_N 624
#define MBQ_MT19937_M 397
#define MBQ_MT19937_MATRIX_A 0x9908b0df
#define MBQ_UPPER_MASK 0x80000000
#define MBQ_LOWER_MASK 0x7fffffff

#define MBQ_STRINGIFY_VA_ARGS_IMPL(...) #__VA_ARGS__
#define MBQ_STRINGIFY_VA_ARGS(...) MBQ_STRINGIFY_VA_ARGS_IMPL(__VA_ARGS__)

#define MBQ_MT19937_STATE_STRUCT                                                                                       \
    typedef struct _state                                                                                              \
    {                                                                                                                  \
        unsigned int MT[MBQ_MT19937_N];                                                                                \
        int index;                                                                                                     \
    } State;

#define MBQ_MT19937_INIT_STATE                                                                                         \
    MBQ_MT19937_STATE_STRUCT                                                                                           \
    __kernel void mt19937_init_state(__global State* state, unsigned int offset, unsigned int len, unsigned int seed)  \
    {                                                                                                                  \
        state->MT[0] = seed;                                                                                           \
        for (int i = 1; i < MBQ_MT19937_N; ++i)                                                                        \
            state->MT[i] = 1812433253 * (state->MT[i - 1] ^ (state->MT[i - 1] >> 30)) + i;                             \
        state->index = MBQ_MT19937_N;                                                                                  \
    }

#define MBQ_MT19937_NEXT                                                                                               \
    unsigned int mt19937_next(State* state)                                                                            \
    {                                                                                                                  \
        unsigned int y;                                                                                                \
        unsigned int mag01[2] = {0x0, MBQ_MT19937_MATRIX_A};                                                           \
        if (state->index < MBQ_MT19937_N - MBQ_MT19937_M)                                                              \
        {                                                                                                              \
            y = (state->MT[state->index] & MBQ_UPPER_MASK) | (state->MT[state->index + 1] & MBQ_LOWER_MASK);           \
            state->MT[state->index] = state->MT[state->index + MBQ_MT19937_M] ^ (y >> 1) ^ mag01[y & 0x1];             \
        }                                                                                                              \
        else if (state->index < MBQ_MT19937_N - 1)                                                                     \
        {                                                                                                              \
            y = (state->MT[state->index] & MBQ_UPPER_MASK) | (state->MT[state->index + 1] & MBQ_LOWER_MASK);           \
            state->MT[state->index] =                                                                                  \
                state->MT[state->index + (MBQ_MT19937_M - MBQ_MT19937_N)] ^ (y >> 1) ^ mag01[y & 0x1];                 \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            y = (state->MT[MBQ_MT19937_N - 1] & MBQ_UPPER_MASK) | (state->MT[0] & MBQ_LOWER_MASK);                     \
            state->MT[MBQ_MT19937_N - 1] = state->MT[MBQ_MT19937_M - 1] ^ (y >> 1) ^ mag01[y & 0x1];                   \
            state->index = 0;                                                                                          \
        }                                                                                                              \
        y = state->MT[state->index++];                                                                                 \
        y ^= (y >> 11);                                                                                                \
        y ^= (y << 7) & 0x9d2c5680;                                                                                    \
        y ^= (y << 15) & 0xefc60000;                                                                                   \
        y ^= (y >> 18);                                                                                                \
        return y;                                                                                                      \
    }

#define MBQ_MT19937_RANDOM_BUFFER(m__type, m__suffix, m__constant)                                                     \
    MBQ_MT19937_STATE_STRUCT                                                                                           \
    MBQ_MT19937_NEXT                                                                                                   \
    __kernel void mt19937_random_buffer##m__suffix(__global m__type* x, unsigned int off, unsigned int len,            \
                                                   __global State* state)                                              \
    {                                                                                                                  \
        for (unsigned int i = off; i < len + off; ++i)                                                                 \
            x[i] = ((m__type)mt19937_next(state)) * (m__constant);                                                     \
    }

namespace mbq
{
    namespace opencl::detail
    {
        MBQ_MT19937_STATE_STRUCT

        constexpr const char* mt19937_init_state_src = MBQ_STRINGIFY_VA_ARGS(MBQ_MT19937_INIT_STATE);
        inline Kernel<Bufs<State>, Args<unsigned int, unsigned int>> mt19937_init_state("mt19937_init_state",
                                                                                        mt19937_init_state_src);

        constexpr const char* mt19937_random_buffers_src =
            MBQ_STRINGIFY_VA_ARGS(MBQ_MT19937_RANDOM_BUFFER(float, _s, 2.3283064365386962890625e-10f));
        inline Kernel<Bufs<float>, Args<unsigned int, cl_mem>> mt19937_random_buffers("mt19937_random_buffer_s",
                                                                                      mt19937_random_buffers_src);

        constexpr const char* mt19937_random_bufferd_src =
            MBQ_STRINGIFY_VA_ARGS(MBQ_MT19937_RANDOM_BUFFER(double, _d, 2.3283064365386962890625e-10));
        inline Kernel<Bufs<double>, Args<unsigned int, cl_mem>> mt19937_random_bufferd("mt19937_random_buffer_d",
                                                                                       mt19937_random_bufferd_src);

        inline void mt19937_random_buffer(Context* ctx, Pointer<float> ptr, uint32_t count, cl_mem state)
        {
            mt19937_random_buffers(ctx, ptr, count, state, execution_policy::sequential{});
        }

        inline void mt19937_random_buffer(Context* ctx, Pointer<double> ptr, uint32_t count, cl_mem state)
        {
            mt19937_random_bufferd(ctx, ptr, count, state, execution_policy::sequential{});
        }

        inline void mt19937_random_buffer(Context* ctx, Pointer<std::complex<float>> ptr, uint32_t count, cl_mem state)
        {
            auto [mem, offset] = ptr.get();
            Pointer<float> cast_ptr{mem, offset * 2};

            mt19937_random_buffers(ctx, cast_ptr, count * 2, state, execution_policy::sequential{});
        }

        inline void mt19937_random_buffer(Context* ctx, Pointer<std::complex<double>> ptr, uint32_t count, cl_mem state)
        {
            auto [mem, offset] = ptr.get();
            Pointer<double> cast_ptr{mem, offset * 2};

            mt19937_random_bufferd(ctx, cast_ptr, count * 2, state, execution_policy::sequential{});
        }

        struct MT19937
        {
            Context* _ctx{nullptr};
            Pointer<State> _state{nullptr};

            explicit MT19937(Context* ctx) : _ctx(ctx)
            {
                Allocator<State> allocator(ctx);
                _state = allocator.allocate(1);
                mt19937_init_state(ctx, _state, 1, std::random_device{}(), execution_policy::sequential{});
            }
            MT19937(const MT19937&) = delete;
            MT19937(MT19937&&) noexcept = default;
            ~MT19937()
            {
                Allocator<State> allocator(_ctx);
                allocator.deallocate(_state, 1);
            }
        };

        inline MT19937& get_default_engine(Context* ctx)
        {
            thread_local static std::unordered_map<Context*, MT19937> engines;

            auto it = engines.find(ctx);
            if (it != engines.cend())
                return it->second;
            auto [element, _] = engines.emplace(ctx, ctx);
            return element->second;
        }
    } // namespace opencl::detail

    template <typename T>
    struct Random<opencl::Allocator<T>>
    {
        using value_type = T;

        template <typename First, typename Last>
        constexpr Last operator()(First first, Last last, const value_type& /*min*/, const value_type& /*max*/) const
        {
            auto ptr = &(*first);
            auto count = last - first;
            auto ctx = first.get_allocator().state();
            auto& engine = opencl::detail::get_default_engine(ctx);
            auto [state, _] = engine._state.get();

            opencl::detail::mt19937_random_buffer(ctx, ptr, static_cast<uint32_t>(count), state);

            return last;
        }
    };
} // namespace mbq