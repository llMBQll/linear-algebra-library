#pragma once

#include "concepts.hpp"
#include "Copy.hpp"
#include "MemoryBase.hpp"

namespace mbq
{
    template <non_void T, typename Memory>
    class ConstReference;

    template <non_void T, typename Memory>
    class Reference : public MemoryBase<typename Memory::allocator_type>
    {
    public:
        using base_type = MemoryBase<typename Memory::allocator_type>;
        using value_type = T;
        using memory_type = Memory;
        using allocator_type = typename memory_type::allocator_type;
        using pointer = typename memory_type::pointer;
    private:
        pointer _data{nullptr};
    private:
        inline void set_value(value_type value) const
        {
            copy(_data, this->get_allocator(), &value, std::allocator<value_type>{}, 1);
        }
        inline value_type get_value() const
        {
            value_type value;
            mbq::copy(&value, std::allocator<value_type>{}, _data, this->get_allocator(), 1);
            return value;
        }
    public:
        Reference() noexcept = default;
        explicit Reference(pointer data, allocator_type allocator = allocator_type{})
            : base_type(allocator), _data(data)
        { }
        Reference(const Reference&) noexcept = default;
        Reference(Reference&&) noexcept = default;
        Reference& operator=(const Reference&) noexcept = default;
        Reference& operator=(Reference&&) noexcept = default;
        Reference& operator=(value_type value)
        {
            set_value(value);
            return *this;
        }
        Reference& operator=(value_type value) const
        {
            set_value(value);
            return *this;
        }
        operator value_type() const
        {
            return get_value();
        }

        Reference& operator++()
        {
            set_value(get_value() + value_type{1});
            return *this;
        }
        Reference operator++(int)
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        Reference& operator--()
        {
            set_value(get_value() - value_type{1});
            return *this;
        }
        Reference operator--(int)
        {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        Reference& operator+=(const Reference& rhs)
        {
            set_value(get_value() + rhs.get_value());
            return *this;
        }
        Reference& operator+=(value_type rhs)
        {
            set_value(get_value() + rhs);
            return *this;
        }
        value_type operator+(const Reference& rhs) const
        {
            return get_value() + rhs.get_value();
        }
        value_type operator+(value_type rhs) const
        {
            return get_value() + rhs;
        }
        friend value_type operator+(value_type lhs, const Reference& rhs)
        {
            return lhs + rhs.get_value();
        }

        Reference& operator-=(const Reference& rhs)
        {
            set_value(get_value() - rhs.get_value());
            return *this;
        }
        Reference& operator-=(value_type rhs)
        {
            set_value(get_value() - rhs);
            return *this;
        }
        value_type operator-(const Reference& rhs) const
        {
            return get_value() - rhs.get_value();
        }
        value_type operator-(value_type rhs) const
        {
            return get_value() - rhs;
        }
        friend value_type operator-(value_type lhs, const Reference& rhs)
        {
            return lhs - rhs.get_value();
        }

        Reference& operator*=(const Reference& rhs)
        {
            set_value(get_value() * rhs.get_value());
            return *this;
        }
        Reference& operator*=(value_type rhs)
        {
            set_value(get_value() * rhs);
            return *this;
        }
        value_type operator*(const Reference& rhs) const
        {
            return get_value() * rhs.get_value();
        }
        value_type operator*(value_type rhs) const
        {
            return get_value() * rhs;
        }
        friend value_type operator*(value_type lhs, const Reference& rhs)
        {
            return lhs * rhs.get_value();
        }

        Reference& operator/=(const Reference& rhs)
        {
            set_value(get_value() / rhs.get_value());
            return *this;
        }
        Reference& operator/=(value_type rhs)
        {
            set_value(get_value() / rhs);
            return *this;
        }
        value_type operator/(const Reference& rhs) const
        {
            return get_value() / rhs.get_value();
        }
        value_type operator/(value_type rhs) const
        {
            return get_value() / rhs;
        }
        friend value_type operator/(value_type lhs, const Reference& rhs)
        {
            return lhs / rhs.get_value();
        }

        friend std::ostream& operator<<(std::ostream& out, const Reference& rhs)
        {
            out << rhs.get_value();
            return out;
        }

        pointer operator&()
        {
            return _data;
        }

        bool operator<=>(const Reference&) const noexcept = default;

        template <typename Rhs>
            requires convertible_to<value_type, Rhs>
        bool operator==(const Rhs& rhs) const
        {
            return get_value() == rhs;
        }

        template <typename Rhs>
            requires convertible_to<value_type, Rhs>
        bool operator<=>(const Rhs& rhs) const
        {
            return get_value() <=> rhs;
        }

        explicit operator bool() const noexcept
        {
            return static_cast<bool>(get_value());
        }
    };

    template <non_void T, typename Memory>
    class ConstReference : public MemoryBase<typename Memory::allocator_type>
    {
    public:
        using base_type = MemoryBase<typename Memory::allocator_type>;
        using value_type = T;
        using memory_type = Memory;
        using allocator_type = typename memory_type::allocator_type;
        using pointer = typename memory_type::const_pointer;
    private:
        pointer _data{nullptr};
    private:
        inline value_type get_value() const
        {
            value_type value;
            copy(&value, std::allocator<value_type>{}, _data, this->get_allocator(), 1);
            return value;
        }
    public:
        ConstReference() noexcept = default;
        explicit ConstReference(pointer data, allocator_type allocator = allocator_type{})
            : base_type(allocator), _data(data)
        { }
        ConstReference(const ConstReference&) noexcept = default;
        ConstReference(ConstReference&&) noexcept = default;
        ConstReference& operator=(const ConstReference&) noexcept = default;
        ConstReference& operator=(ConstReference&&) noexcept = default;
        operator value_type() const
        {
            return get_value();
        }

        value_type operator+(const ConstReference& rhs) const
        {
            return get_value() + rhs._value;
        }
        value_type operator+(value_type rhs) const
        {
            return get_value() + rhs;
        }
        friend value_type operator+(value_type lhs, const ConstReference& rhs)
        {
            return lhs + rhs.get_value();
        }

        value_type operator-(const ConstReference& rhs) const
        {
            return get_value() - rhs._value;
        }
        value_type operator-(value_type rhs) const
        {
            return get_value() - rhs;
        }
        friend value_type operator-(value_type lhs, const ConstReference& rhs)
        {
            return lhs - rhs.get_value();
        }

        value_type operator*(const ConstReference& rhs) const
        {
            return get_value() * rhs._value;
        }
        value_type operator*(value_type rhs) const
        {
            return get_value() * rhs;
        }
        friend value_type operator*(value_type lhs, const ConstReference& rhs)
        {
            return lhs * rhs.get_value();
        }

        value_type operator/(const ConstReference& rhs) const
        {
            return get_value() / rhs._value;
        }
        value_type operator/(value_type rhs) const
        {
            return get_value() / rhs;
        }
        friend value_type operator/(value_type lhs, const ConstReference& rhs)
        {
            return lhs / rhs.get_value();
        }

        friend std::ostream& operator<<(std::ostream& out, const ConstReference& rhs)
        {
            out << rhs.get_value();
            return out;
        }

        pointer operator&()
        {
            return _data;
        }

        bool operator<=>(const ConstReference&) const = default;

        explicit operator bool() const noexcept
        {
            return static_cast<bool>(get_value());
        }
    };
} // namespace mbq