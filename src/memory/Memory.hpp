#pragma once

#include "algorithm/fill.hpp"
#include "AllocatorTraits.hpp"
#include "concepts.hpp"
#include "Copy.hpp"
#include "exceptions/Exception.hpp"
#include "MemoryBase.hpp"
#include "MemoryIterator.hpp"
#include "MemoryView.hpp"
#include "Reference.hpp"

#include <initializer_list>

namespace mbq
{
    namespace detail
    {
        template <typename Pointer, typename SizeType, typename Allocator>
        class Guard
        {
        private:
            Pointer _pointer;
            SizeType _count;
            Allocator _allocator;
        public:
            Guard(Pointer pointer, SizeType count, Allocator allocator)
                : _pointer(pointer), _count(count), _allocator(allocator)
            { }
            ~Guard()
            {
                if (_pointer)
                    _allocator.deallocate(_pointer, _count);
            }
            void release()
            {
                _pointer = nullptr;
            }
        };
    } // namespace detail

    template <non_void T, typename Allocator>
    class Memory : public MemoryBase<Allocator>
    {
    public:
        using base_type = MemoryBase<Allocator>;
        using value_type = T;
        using allocator_type = Allocator;
        using traits_type = AllocatorTraits<allocator_type>;
        using size_type = typename traits_type::size_type;
        using pointer = typename traits_type::pointer;
        using const_pointer = typename traits_type::const_pointer;
        using dereferenceable = typename traits_type::dereferenceable;
        using reference = std::conditional_t<dereferenceable::value, value_type&, Reference<T, Memory>>;
        using const_reference =
            std::conditional_t<dereferenceable::value, const value_type&, ConstReference<T, Memory>>;
        using iterator = MemoryIterator<Memory>;
        using const_iterator = ConstMemoryIterator<Memory>;
        using difference_type = typename traits_type::difference_type;

        template <non_void, typename>
        friend class Memory;
    private:
        pointer _data{nullptr};
        size_type _size{0};
    private:
        template <typename Other>
        Memory(const_pointer_of_t<Other> other_data, size_type other_size, Other other_allocator = {},
               allocator_type allocator = {})
            : base_type(allocator), _data(allocator.allocate(other_size)), _size(other_size)
        {
            copy(_data, this->get_allocator(), other_data, other_allocator, _size);
        }
    public:
        Memory() = default;

        explicit Memory(size_type size, allocator_type allocator = allocator_type{})
            : base_type(allocator), _data(allocator.allocate(size)), _size(size)
        { }

        // Memory(std::initializer_list<value_type> init, allocator_type _allocator = allocator_type{})
        //     : base_type(_allocator), _data(_allocator.allocate(init.size())), _size(init.size())
        // {
        //     copy(_data, this->get_allocator(), init.begin(), ::std::_allocator<value_type>{}, _size);
        // }

        Memory(const Memory& other) : base_type(other.get_allocator()), _size(other._size)
        {
            _data = this->get_allocator().allocate(_size);
            copy(_data, this->get_allocator(), other._data, other.get_allocator(), other._size);
        }

        template <typename OtherAllocator>
        explicit Memory(const Memory<value_type, OtherAllocator>& other, allocator_type allocator = allocator_type{})
            : base_type(allocator), _data(allocator.allocate(other._size)), _size(other._size)
        {
            copy(_data, this->get_allocator(), other._data, other.get_allocator(), other._size);
        }

        explicit Memory(const_pointer data, size_t size, allocator_type allocator = allocator_type{})
            : base_type(allocator), _data(allocator.allocate(size)), _size(size)
        {
            copy(_data, allocator, data, allocator, _size);
        }

        Memory(Memory&& other) noexcept : base_type(other.get_allocator()), _data(other._data), _size(other._size)
        {
            other._data = nullptr;
            other._size = 0;
        }

        Memory& operator=(const Memory& other)
        {
            if (this == &other)
                return *this;

            if (_data)
                this->get_allocator().deallocate(_data, _size);

            _size = other._size;
            _data = this->get_allocator().allocate(_size);
            copy(_data, this->get_allocator(), other._data, other.get_allocator(), other._size);
            return *this;
        }

        template <typename OtherAllocator>
        Memory& operator=(const Memory<value_type, OtherAllocator>& other)
        {
            if (_data)
                this->get_allocator().deallocate(_data, _size);

            _size = other._size;
            _data = this->get_allocator().allocate(_size);
            copy(_data, this->get_allocator(), other._data, other.get_allocator(), other._size);
            return *this;
        }

        Memory& operator=(Memory&& other) noexcept
        {
            if (_data)
                this->get_allocator().deallocate(_data, _size);

            this->set_allocator(other.get_allocator());
            _data = other._data;
            _size = other._size;
            other._data = nullptr;
            other._size = 0;

            return *this;
        }

        ~Memory()
        {
            this->get_allocator().deallocate(_data, _size);
        }

        template <typename OtherAllocator>
        static Memory from(const_pointer_of_t<OtherAllocator> data, size_type count,
                           OtherAllocator other_allocator = {}, allocator_type allocator = {})
        {
            return Memory(data, count, other_allocator, allocator);
        }

        static Memory from(const MemoryView<T, Allocator>& view, allocator_type allocator = {})
        {
            return Memory::from(view.data(), view.size(), view.get_allocator(), allocator);
        }

        static Memory from(const ConstMemoryView<T, Allocator>& view, allocator_type allocator = {})
        {
            return Memory::from(view.data(), view.size(), view.get_allocator(), allocator);
        }

        [[nodiscard]] const_pointer data() const noexcept
        {
            return _data;
        }

        [[nodiscard]] size_type size() const noexcept
        {
            return _size;
        }

        iterator begin() noexcept
        {
            return iterator(_data, this->get_allocator());
        }

        iterator end() noexcept
        {
            return iterator(_data + size(), this->get_allocator());
        }

        const_iterator begin() const noexcept
        {
            return const_iterator(_data, this->get_allocator());
        }

        const_iterator end() const noexcept
        {
            return const_iterator(_data + size(), this->get_allocator());
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
                MBQ_THROW_EXCEPTION(Exception, "Out of bounds");
            return (*this)[index];
        }

        const_reference at(size_t index) const
        {
            if (index >= size())
                MBQ_THROW_EXCEPTION(Exception, "Out of bounds");
            return (*this)[index];
        }

        MemoryView<T, Allocator> view() const noexcept
        {
            return {_data, size(), this->get_allocator()};
        }

        ConstMemoryView<T, Allocator> const_view() const noexcept
        {
            return {_data, size(), this->get_allocator()};
        }

        void resize(size_type count, value_type value = {})
        {
            if (count == size())
                return;

            auto allocator = this->get_allocator();
            auto data = allocator.allocate(count);

            detail::Guard<pointer, size_type, allocator_type> guard{data, count, allocator};

            copy(data, allocator, _data, allocator, std::min(_size, count));

            if (count > _size)
                fill_impl<allocator_type>(iterator{data + _size}, iterator{data + count}, value);

            allocator.deallocate(_data, _size);
            _data = data;
            _size = count;

            guard.release();
        }
    };
} // namespace mbq