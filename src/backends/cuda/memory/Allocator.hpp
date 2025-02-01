#pragma once

#include "backends/cuda/exceptions/CudaException.hpp"
#include "concepts.hpp"
#include "Context.hpp"

#include <cuda_runtime.h>
#include <iostream>

namespace mbq::cuda
{
    template <non_void T>
    class Allocator
    {
    public:
        using value_type = T;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using state_type = Context;
        using difference_type = std::ptrdiff_t;
        using stateful = std::true_type;
        using dereferenceable = std::false_type;
    private:
        state_type* _state;
    public:
        constexpr Allocator() noexcept : _state(state_type::get_default()) { }

        constexpr explicit Allocator(state_type* state) noexcept : _state(state) { }

        constexpr Allocator(const Allocator&) noexcept = default;

        template <typename Other>
        constexpr explicit Allocator(const Allocator<Other>& other) noexcept : _state(other._state)
        { }

        [[nodiscard]] pointer allocate(size_type count)
        {
            const auto size = count * sizeof(value_type);
            void* memory = nullptr;
            auto res = cudaMalloc(&memory, size);
            if (res != cudaError::cudaSuccess)
                MBQ_THROW_EXCEPTION(CudaException, res);
            return static_cast<pointer>(memory);
        }

        void deallocate(pointer memory, size_type) noexcept
        {
            auto res = cudaFree(memory);
            if (res != cudaError::cudaSuccess)
            {
                std::cerr << CudaException{res} << std::endl;
                exit(res);
            }
        }

        void set_state(state_type* state) noexcept
        {
            _state = state;
        }

        [[nodiscard]] state_type* state() const noexcept
        {
            return _state;
        }
    };
} // namespace mbq::cuda