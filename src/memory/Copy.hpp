#pragma once

#include "concepts.hpp"
#include "memory/AllocatorTraits.hpp"

#include <memory>

namespace mbq
{
    template <typename AllocatorA, typename AllocatorB>
    concept same_value_type = std::same_as<value_type_of_t<AllocatorA>, value_type_of_t<AllocatorB>>;

    template <typename AllocatorDst, typename AllocatorSrc>
    class Copy;

    template <typename AllocatorDst, typename AllocatorSrc>
    inline constexpr Copy<AllocatorDst, AllocatorSrc> copy_impl;

    template <typename AllocatorDst, typename AllocatorSrc>
        requires same_value_type<AllocatorDst, AllocatorSrc>
    void copy(pointer_of_t<AllocatorDst> dst, AllocatorDst dst_alloc, const_pointer_of_t<AllocatorSrc> src,
              AllocatorSrc src_alloc, size_t count)
    {
        return copy_impl<AllocatorDst, AllocatorSrc>(dst, dst_alloc, src, src_alloc, count);
    }

    template <typename AllocatorDst, typename AllocatorSrc>
    class Copy
    {
    public:
        using dst_pointer = pointer_of_t<AllocatorDst>;
        using src_pointer = const_pointer_of_t<AllocatorSrc>;
        using value_type = value_type_of_t<AllocatorSrc>;
    public:
        void operator()(dst_pointer dst, AllocatorDst dst_alloc, src_pointer src, AllocatorSrc src_alloc,
                        size_t count) const
        {
            std::allocator<value_type> allocator;
            auto buffer = allocator.allocate(count);
            copy(buffer, allocator, src, src_alloc, count);
            copy(dst, dst_alloc, buffer, allocator, count);
            allocator.deallocate(buffer, count);
        }
    };
} // namespace mbq