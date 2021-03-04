#pragma once

#include "backends/opencl/exceptions/OpenCLException.hpp"
#include "Context.hpp"
#include "Pointer.hpp"

namespace mbq::opencl
{
    template <non_void T>
    class Allocator
    {
    public:
        using value_type = T;
        using pointer = Pointer<T>;
        using const_pointer = Pointer<T>;
        using size_type = std::size_t;
        using state_type = Context;
        using difference_type = typename pointer::difference_type;
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
            cl_int res;
            auto memory = clCreateBuffer(_state->context, CL_MEM_READ_WRITE, count * sizeof(value_type), nullptr, &res);
            if (res != CL_SUCCESS)
                MBQ_THROW_EXCEPTION(OpenCLException, res);
            return pointer(memory);
        }

        void deallocate(pointer memory, size_type)
        {
            if (!memory)
                return;
            auto res = clReleaseMemObject(memory.ptr());
            if (res != CL_SUCCESS)
                exit(res);
        }

        void set_state(state_type* state)
        {
            _state = state;
        }

        [[nodiscard]] state_type* state() const noexcept
        {
            return _state;
        }
    };
} // namespace mbq::opencl