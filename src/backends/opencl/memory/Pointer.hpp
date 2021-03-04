#pragma once

#include "../common.hpp"
#include "concepts.hpp"

namespace mbq::opencl
{
    template <typename T>
    class Pointer
    {
    public:
        using value_type = T;
        using pointer = cl_mem;
        using difference_type = std::ptrdiff_t;
    private:
        pointer _ptr{nullptr};
        difference_type _offset{0};
    public:
        Pointer() = default;

        explicit Pointer(std::nullptr_t ptr) : _ptr(ptr), _offset(0) { }

        explicit Pointer(pointer ptr) : _ptr(ptr), _offset(0) { }

        Pointer(pointer ptr, difference_type offset) : _ptr(ptr), _offset(offset) { }

        Pointer(const Pointer&) = default;

        Pointer(Pointer&&) noexcept = default;

        Pointer& operator=(const Pointer&) = default;

        Pointer& operator=(Pointer&&) noexcept = default;

        Pointer& operator=(std::nullptr_t ptr)
        {
            _ptr = ptr;
            _offset = 0;
            return *this;
        }

        Pointer& operator++() noexcept
        {
            ++_offset;
            return *this;
        }

        Pointer operator++(int) noexcept
        {
            Pointer tmp = *this;
            ++*this;
            return tmp;
        }

        Pointer& operator--() noexcept
        {
            --_offset;
            return *this;
        }

        Pointer operator--(int) noexcept
        {
            Pointer tmp = *this;
            --*this;
            return tmp;
        }

        Pointer operator+=(difference_type offset) noexcept
        {
            _offset += offset;
            return *this;
        }

        Pointer operator+(difference_type offset) const noexcept
        {
            Pointer tmp = *this;
            tmp += offset;
            return tmp;
        }

        Pointer operator-=(difference_type offset) noexcept
        {
            _offset -= offset;
            return *this;
        }

        Pointer operator-(difference_type offset) const noexcept
        {
            Pointer tmp = *this;
            tmp -= offset;
            return tmp;
        }

        difference_type operator-(const Pointer& rhs) const noexcept
        {
            return _offset - rhs._offset;
        }

        auto operator<=>(const Pointer& rhs) const
        {
            auto lhs_val = std::bit_cast<std::intptr_t>(_ptr) + _offset;
            auto rhs_val = std::bit_cast<std::intptr_t>(rhs._ptr) + rhs._offset;
            return lhs_val <=> rhs_val;
        }

        auto operator==(const Pointer& rhs) const
        {
            auto lhs_val = std::bit_cast<std::intptr_t>(_ptr) + _offset;
            auto rhs_val = std::bit_cast<std::intptr_t>(rhs._ptr) + rhs._offset;
            return lhs_val == rhs_val;
        }

        auto operator!=(const Pointer& rhs) const
        {
            return !(*this == rhs);
        }

        bool operator==(std::nullptr_t ptr) const
        {
            return _ptr == ptr;
        }

        bool operator!=(std::nullptr_t ptr) const
        {
            return !(*this == ptr);
        }

        explicit operator bool() const
        {
            return std::bit_cast<std::intptr_t>(_ptr) + _offset;
        }

        [[nodiscard]] std::pair<pointer, difference_type> get() const
        {
            return {_ptr, _offset};
        }

        [[nodiscard]] pointer ptr() const
        {
            return _ptr;
        }

        [[nodiscard]] difference_type off() const
        {
            return _offset;
        }
    };
} // namespace mbq::opencl