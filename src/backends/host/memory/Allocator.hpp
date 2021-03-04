#pragma once

#include "concepts.hpp"
#include "exceptions/Exception.hpp"

namespace mbq::host
{
    template <non_void T>
    class Allocator
    {
    public:
        using value_type = T;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
    public:
        constexpr Allocator() noexcept = default;
        constexpr Allocator(const Allocator&) noexcept = default;
        template <typename Other>
        constexpr explicit Allocator(const Allocator<Other>&) noexcept
        { }

        [[nodiscard]] pointer allocate(size_type count)
        {
            auto memory = static_cast<pointer>(std::malloc(sizeof(value_type) * count));
            if (!memory)
                MBQ_THROW_EXCEPTION(Exception, "Allocation failed");
            return memory;
        }

        void deallocate(pointer memory, size_type)
        {
            std::free(memory);
        }
    };
} // namespace mbq::host