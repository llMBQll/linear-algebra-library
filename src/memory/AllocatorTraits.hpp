#pragma once

#include <memory>

namespace mbq
{
    namespace detail
    {
        template <typename T, typename = void>
        struct get_stateful_type
        {
            using type = std::false_type;
        };

        template <typename T>
        struct get_stateful_type<T, std::void_t<typename T::stateful>>
        {
            using type = typename T::stateful;
        };

        template <typename T, typename = void>
        struct get_dereferenceable
        {
            using type = std::true_type;
        };

        template <typename T>
        struct get_dereferenceable<T, std::void_t<typename T::dereferenceable>>
        {
            using type = typename T::dereferenceable;
        };
    } // namespace detail

    template <typename Allocator>
    struct ValueTypeOf
    {
        using traits_type = std::allocator_traits<Allocator>;
        using type = typename traits_type::value_type;
    };

    template <typename Allocator>
    using value_type_of_t = typename ValueTypeOf<Allocator>::type;

    template <typename Allocator>
    struct PointerOf
    {
        using traits_type = std::allocator_traits<Allocator>;
        using type = typename traits_type::pointer;
    };

    template <typename Allocator>
    using pointer_of_t = typename PointerOf<Allocator>::type;

    template <typename Allocator>
    struct ConstPointerOf
    {
        using traits_type = std::allocator_traits<Allocator>;
        using type = typename traits_type::const_pointer;
    };

    template <typename Allocator>
    using const_pointer_of_t = typename ConstPointerOf<Allocator>::type;

    template <typename Allocator>
    struct AllocatorTraits : public std::allocator_traits<Allocator>
    {
        using std_traits = std::allocator_traits<Allocator>;
        using stateful = typename detail::get_stateful_type<Allocator>::type;
        using dereferenceable = typename detail::get_dereferenceable<Allocator>::type;
    };

    template <typename T, typename Alloc>
    concept allocator_requirement = requires(Alloc a) {
        typename Alloc::value_type;
        {
            a.allocate(std::declval<typename AllocatorTraits<Alloc>::size_type>())
        } -> std::same_as<typename AllocatorTraits<Alloc>::pointer>;
        {
            a.deallocate(std::declval<typename AllocatorTraits<Alloc>::pointer>(),
                         std::declval<typename AllocatorTraits<Alloc>::size_type>())
        };
    } && std::same_as<typename Alloc::value_type, T> && std::is_default_constructible_v<Alloc>;

    template <typename T, typename Alloc>
    concept host_allocator = std::same_as<typename AllocatorTraits<Alloc>::dereferenceable, std::true_type> &&
                             std::same_as<typename AllocatorTraits<Alloc>::value_type, T>;
} // namespace mbq