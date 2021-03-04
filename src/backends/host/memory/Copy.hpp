#pragma once

#include "../../../memory/Copy.hpp"
#include "Allocator.hpp"

#include <cstring>

namespace mbq
{
    template <typename AllocatorDst, typename AllocatorSrc>
        requires host_allocator<value_type_of_t<AllocatorSrc>, AllocatorDst> &&
                 host_allocator<value_type_of_t<AllocatorDst>, AllocatorSrc>
    class Copy<AllocatorDst, AllocatorSrc>
    {
    public:
        using dst_pointer = pointer_of_t<AllocatorDst>;
        using src_pointer = const_pointer_of_t<AllocatorSrc>;
        using value_type = value_type_of_t<AllocatorSrc>;
    public:
        void operator()(dst_pointer dst, AllocatorDst, src_pointer src, AllocatorSrc, size_t count) const
        {
            std::memcpy(dst, src, count * sizeof(value_type));
        }
    };
} // namespace mbq