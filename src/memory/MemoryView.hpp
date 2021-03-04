#pragma once

#include "AllocatorTraits.hpp"
#include "concepts.hpp"
#include "exceptions/Exception.hpp"
#include "MemoryBase.hpp"
#include "MemoryIterator.hpp"
#include "Reference.hpp"

namespace mbq
{
    template <non_void T, typename Allocator>
    class ConstMemoryView : public MemoryBase<Allocator>
    {
    public:
        using base_type = MemoryBase<Allocator>;
        using value_type = T;
        using allocator_type = Allocator;
        using traits_type = AllocatorTraits<allocator_type>;
        using pointer = typename traits_type::const_pointer;
        using const_pointer = pointer;
        using size_type = typename traits_type::size_type;
        using iterator = ConstMemoryIterator<ConstMemoryView>;
        using dereferenceable = typename traits_type::dereferenceable;
        using reference =
            std::conditional_t<dereferenceable::value, const value_type&, ConstReference<T, ConstMemoryView>>;
        using const_reference = reference;
        using difference_type = typename traits_type::difference_type;
    public:
        constexpr inline static size_type npos = std::numeric_limits<size_type>::max();
    private:
        pointer _data{nullptr};
        size_type _size{0};
    public:
        ConstMemoryView() noexcept = default;

        ConstMemoryView(pointer data, size_type size, allocator_type allocator = allocator_type{}) noexcept
            : base_type(allocator), _data(data), _size(size)
        { }

        ~ConstMemoryView() noexcept = default;

        static ConstMemoryView from(pointer data, size_type size, allocator_type allocator = allocator_type{})
        {
            return {data, size, allocator};
        }

        [[nodiscard]] const_pointer data() const noexcept
        {
            return _data;
        }

        size_type size() const noexcept
        {
            return _size;
        }

        iterator begin() const noexcept
        {
            return iterator(_data, this->get_allocator());
        }

        iterator end() const noexcept
        {
            return iterator(_data + _size, this->get_allocator());
        }

        iterator cbegin() const noexcept
        {
            return begin();
        }

        iterator cend() const noexcept
        {
            return end();
        }

        reference operator[](size_type index) const noexcept
        {
            return detail::make_reference<reference>(_data + index, this->get_allocator());
        }

        reference at(size_t index) const
        {
            if (index >= size())
                MBQ_THROW_EXCEPTION(Exception, "Out of bounds");
            return (*this)[index];
        }

        ConstMemoryView sub_view(size_type pos = 0, size_type count = npos) const
        {
            if (pos > size())
                MBQ_THROW_EXCEPTION(Exception, "out of range");
            return {_data + pos, std::min(count, size() - pos), this->get_allocator()};
        }

        ConstMemoryView const_sub_view(size_type pos = 0, size_type count = npos) const
        {
            if (pos > size())
                MBQ_THROW_EXCEPTION(Exception, "out of range");
            return {_data + pos, std::min(count, size() - pos), this->get_allocator()};
        }
    };

    template <non_void T, typename Allocator>
    class MemoryView : public MemoryBase<Allocator>
    {
    public:
        using base_type = MemoryBase<Allocator>;
        using value_type = T;
        using allocator_type = Allocator;
        using traits_type = AllocatorTraits<allocator_type>;
        using pointer = typename traits_type::pointer;
        using const_pointer = typename traits_type::const_pointer;
        using size_type = typename traits_type::size_type;
        using iterator = MemoryIterator<MemoryView>;
        using const_iterator = ConstMemoryIterator<MemoryView>;
        using dereferenceable = typename traits_type::dereferenceable;
        using reference = std::conditional_t<dereferenceable::value, value_type&, Reference<T, MemoryView>>;
        using const_reference =
            std::conditional_t<dereferenceable::value, const value_type&, ConstReference<T, MemoryView>>;
        using difference_type = typename traits_type::difference_type;
    public:
        constexpr inline static size_type npos = std::numeric_limits<size_type>::max();
    private:
        pointer _data{nullptr};
        size_type _size{0};
    public:
        MemoryView() noexcept = default;

        MemoryView(pointer data, size_type size, allocator_type allocator = allocator_type{}) noexcept
            : base_type(allocator), _data(data), _size(size)
        { }

        MemoryView(const MemoryView&) noexcept = default;

        ~MemoryView() noexcept = default;

        static MemoryView from(pointer data, size_type size, allocator_type allocator = {})
        {
            return {data, size, allocator};
        }

        [[nodiscard]] const_pointer data() const noexcept
        {
            return _data;
        }

        size_type size() const noexcept
        {
            return _size;
        }

        iterator begin() noexcept
        {
            return iterator(_data, this->get_allocator());
        }

        iterator end() noexcept
        {
            return iterator(_data + _size, this->get_allocator());
        }

        const_iterator begin() const noexcept
        {
            return const_iterator(_data, this->get_allocator());
        }

        const_iterator end() const noexcept
        {
            return const_iterator(_data + _size, this->get_allocator());
        }

        const_iterator cbegin() const noexcept
        {
            return begin();
        }

        const_iterator cend() const noexcept
        {
            return end();
        }

        reference operator[](size_type index) noexcept
        {
            return detail::make_reference<reference>(_data + index, this->get_allocator());
        }

        const_reference operator[](size_type index) const noexcept
        {
            return detail::make_reference<const_reference>(_data + index, this->get_allocator());
        }

        reference at(size_t index)
        {
            if (index >= size())
                MBQ_THROW_EXCEPTION(Exception, "out of bounds");
            return (*this)[index];
        }

        const_reference at(size_t index) const
        {
            if (index >= size())
                MBQ_THROW_EXCEPTION(Exception, "out of bounds");
            return (*this)[index];
        }

        MemoryView sub_view(size_type pos = 0, size_type count = npos)
        {
            if (pos > size())
                MBQ_THROW_EXCEPTION(Exception, "out of range");
            return {_data + pos, std::min(count, size() - pos), this->get_allocator()};
        }

        ConstMemoryView<T, Allocator> sub_view(size_type pos = 0, size_type count = npos) const
        {
            if (pos > size())
                MBQ_THROW_EXCEPTION(Exception, "out of range");
            return {_data + pos, std::min(count, size() - pos), this->get_allocator()};
        }

        ConstMemoryView<T, Allocator> const_sub_view(size_type pos = 0, size_type count = npos) const
        {
            if (pos > size())
                MBQ_THROW_EXCEPTION(Exception, "out of range");
            return {_data + pos, std::min(count, size() - pos), this->get_allocator()};
        }

        ConstMemoryView<T, Allocator> as_const() const
        {
            return {_data, _size, this->get_allocator()};
        }
    };
} // namespace mbq