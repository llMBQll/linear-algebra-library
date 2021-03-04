#pragma once

#include "concepts.hpp"

namespace mbq
{
    template <typename Allocator, bool = stateful_allocator<Allocator>>
    class MemoryBase
    {
    private:
        Allocator _allocator;
    protected:
        constexpr MemoryBase() : _allocator(Allocator{}) { }

        constexpr explicit MemoryBase(Allocator allocator) : _allocator(allocator) { }

        constexpr void set_allocator(Allocator allocator)
        {
            _allocator = allocator;
        }
    public:
        [[nodiscard]] constexpr Allocator get_allocator() const
        {
            return _allocator;
        }
    };

    template <typename Allocator>
    class MemoryBase<Allocator, false>
    {
    protected:
        constexpr MemoryBase() = default;

        constexpr explicit MemoryBase(Allocator) { }

        constexpr void set_allocator(Allocator) { }
    public:
        [[nodiscard]] constexpr Allocator get_allocator() const
        {
            return Allocator{};
        }
    };
} // namespace mbq