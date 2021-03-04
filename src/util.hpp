#pragma once

#include <type_traits>

#define MBQ_FN_NOT_IMPLEMENTED                                                                                         \
    static_assert(::mbq::detail::AlwaysFalse<Allocator>::value, "Function not implemented for specified allocator "    \
                                                                "type")

namespace mbq::detail
{
    template <typename>
    struct AlwaysFalse : std::false_type
    { };
} // namespace mbq::detail