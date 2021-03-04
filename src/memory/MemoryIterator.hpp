#pragma once

#include "AllocatorTraits.hpp"
#include "MemoryBase.hpp"

#include <concepts>
#include <vector>

namespace mbq
{
    template <typename Allocator>
    struct AllocatorTraits;

    namespace detail
    {
        template <typename Ref, typename Ptr, typename Allocator>
        constexpr inline Ref make_reference(Ptr ptr, Allocator allocator)
        {
            using dereferenceable = typename AllocatorTraits<Allocator>::dereferenceable;

            if constexpr (dereferenceable::value)
                return static_cast<Ref>(*ptr);
            else
                return Ref{ptr, allocator};
        }
    } // namespace detail

    template <typename Memory>
    class MemoryIterator : public MemoryBase<typename Memory::allocator_type>,
                           std::conditional_t<Memory::dereferenceable::value, std::contiguous_iterator_tag,
                                              std::random_access_iterator_tag>
    {
    public:
        using base_type = MemoryBase<typename Memory::allocator_type>;
        using memory_type = Memory;
        using value_type = typename memory_type::value_type;
        using size_type = typename memory_type::size_type;
        using pointer = typename memory_type::pointer;
        using reference = typename memory_type::reference;
        using difference_type = typename memory_type::difference_type;
        using allocator_type = typename memory_type::allocator_type;
        using dereferenceable = typename memory_type::dereferenceable;
    private:
        pointer _ptr{nullptr};
    public:
        MemoryIterator() noexcept = default;
        MemoryIterator(pointer ptr, allocator_type allocator = allocator_type{}) noexcept
            : base_type(allocator), _ptr(ptr)
        { }

        auto operator==(const MemoryIterator& rhs) const noexcept
        {
            return _ptr == rhs._ptr;
        }
        auto operator!=(const MemoryIterator& rhs) const noexcept
        {
            return !(*this == rhs);
        }
        auto operator<=>(const MemoryIterator& rhs) const noexcept
        {
            return _ptr <=> rhs._ptr;
        }

        MemoryIterator& operator++() noexcept
        {
            ++_ptr;
            return *this;
        }

        MemoryIterator operator++(int) noexcept
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        MemoryIterator& operator--() noexcept
        {
            --_ptr;
            return *this;
        }

        MemoryIterator operator--(int) noexcept
        {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        MemoryIterator& operator+=(difference_type offset) noexcept
        {
            _ptr += offset;
            return *this;
        }

        MemoryIterator operator+(difference_type offset) const noexcept
        {
            auto tmp = *this;
            tmp += offset;
            return tmp;
        }

        MemoryIterator& operator-=(difference_type offset) noexcept
        {
            _ptr -= offset;
            return *this;
        }

        MemoryIterator operator-(difference_type offset) const noexcept
        {
            auto tmp = *this;
            tmp -= offset;
            return tmp;
        }

        friend MemoryIterator operator+(difference_type offset, const MemoryIterator& iter) noexcept
        {
            return iter + offset;
        }

        friend MemoryIterator operator-(difference_type offset, const MemoryIterator& iter) noexcept
        {
            return iter - offset;
        }

        difference_type operator-(const MemoryIterator& rhs) const noexcept
        {
            return _ptr - rhs._ptr;
        }

        reference operator*() const
        {
            return detail::make_reference<reference>(_ptr, this->get_allocator());
        }

        reference operator[](size_t offset) const noexcept
        {
            return *(*this + offset);
        }
    };

    template <typename Memory>
    class ConstMemoryIterator : public MemoryBase<typename Memory::allocator_type>,
                                std::conditional_t<Memory::dereferenceable::value, std::contiguous_iterator_tag,
                                                   std::random_access_iterator_tag>
    {
    public:
        using base_type = MemoryBase<typename Memory::allocator_type>;
        using memory_type = Memory;
        using value_type = typename memory_type::value_type;
        using size_type = typename memory_type::size_type;
        using pointer = typename memory_type::const_pointer;
        using reference = typename memory_type::const_reference;
        using difference_type = typename memory_type::difference_type;
        using allocator_type = typename memory_type::allocator_type;
        using dereferenceable = typename memory_type::dereferenceable;
    private:
        pointer _ptr{nullptr};
    public:
        ConstMemoryIterator() noexcept = default;
        ConstMemoryIterator(pointer ptr, allocator_type allocator = allocator_type{}) noexcept
            : base_type(allocator), _ptr(ptr)
        { }

        auto operator==(const ConstMemoryIterator& rhs) const noexcept
        {
            return _ptr == rhs._ptr;
        }
        auto operator!=(const ConstMemoryIterator& rhs) const noexcept
        {
            return !(*this == rhs);
        }
        auto operator<=>(const ConstMemoryIterator& rhs) const noexcept
        {
            return _ptr <=> rhs._ptr;
        }

        ConstMemoryIterator& operator++() noexcept
        {
            ++_ptr;
            return *this;
        }

        ConstMemoryIterator operator++(int) noexcept
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        ConstMemoryIterator& operator--() noexcept
        {
            --_ptr;
            return *this;
        }

        ConstMemoryIterator operator--(int) noexcept
        {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        ConstMemoryIterator& operator+=(difference_type offset) noexcept
        {
            _ptr += offset;
            return *this;
        }

        ConstMemoryIterator operator+(difference_type offset) const noexcept
        {
            auto tmp = *this;
            tmp += offset;
            return tmp;
        }

        ConstMemoryIterator& operator-=(difference_type offset) noexcept
        {
            _ptr -= offset;
            return *this;
        }

        ConstMemoryIterator operator-(difference_type offset) const noexcept
        {
            auto tmp = *this;
            tmp -= offset;
            return tmp;
        }

        friend ConstMemoryIterator operator+(difference_type offset, const ConstMemoryIterator& iter) noexcept
        {
            return iter + offset;
        }

        friend ConstMemoryIterator operator-(difference_type offset, const ConstMemoryIterator& iter) noexcept
        {
            return iter - offset;
        }

        difference_type operator-(const ConstMemoryIterator& rhs) const noexcept
        {
            return _ptr - rhs._ptr;
        }

        reference operator*() const noexcept
        {
            return detail::make_reference<reference>(_ptr, this->get_allocator());
        }

        reference operator[](size_t offset) const noexcept
        {
            return *(*this + offset);
        }
    };
} // namespace mbq