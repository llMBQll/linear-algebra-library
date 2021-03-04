#pragma once

#include "concepts.hpp"
#include "exceptions/Exception.hpp"
#include "memory/Memory.hpp"
#include "memory/MemoryView.hpp"

#include <numeric>

namespace mbq
{
    template <non_void T, typename Allocator, size_t N>
    class ConstArrayView
    {
    public:
        using value_type = T;
        using allocator_type = Allocator;
        using memory_view_type = ConstMemoryView<T, Allocator>;
        using iterator = typename memory_view_type::iterator;
        using reference = typename memory_view_type::reference;
        using ret_type = std::conditional_t<N == 1uLL, reference, ConstArrayView<T, Allocator, N - 1>>;
    private:
        template <non_void, typename, size_t>
        friend class ConstArrayView;
    private:
        memory_view_type _memory_view;
        std::array<size_t, N> _dimensions;
    private:
        ConstArrayView(memory_view_type memory_view, const std::array<size_t, N + 1>& dimensions)
            : _memory_view(memory_view)
        {
            for (size_t i = 0; i < N; i++)
                _dimensions[i] = dimensions[i + 1];
        }

        ConstArrayView(memory_view_type memory_view, const std::array<size_t, N>& dimensions)
            : _memory_view(memory_view), _dimensions(dimensions)
        { }
    public:
        static ConstArrayView from(MemoryView<T, Allocator> memory_view, const std::array<size_t, N>& dimensions)
        {
            return ConstArrayView{memory_view.as_const(), dimensions};
        }

        static ConstArrayView from(ConstMemoryView<T, Allocator> memory_view, const std::array<size_t, N>& dimensions)
        {
            return ConstArrayView{memory_view, dimensions};
        }

        [[nodiscard]] const std::array<size_t, N>& dimensions() const
        {
            return _dimensions;
        }

        ret_type operator[](size_t index) const
        {
            if constexpr (N == 1uLL)
                return _memory_view[index];
            else
            {
                auto offset = std::accumulate(_dimensions.begin() + 1, _dimensions.end(), 1uLL, std::multiplies<>());
                return ConstArrayView<T, Allocator, N - 1>(_memory_view.sub_view(index * offset, offset), _dimensions);
            }
        }

        auto at(size_t index) const
        {
            if (index >= _dimensions[0])
                MBQ_THROW_EXCEPTION(Exception, "Out of range");
            return (*this)[index];
        }

        iterator begin() const noexcept
        {
            return _memory_view.begin();
        }

        iterator cbegin() const noexcept
        {
            return _memory_view.cbegin();
        }

        iterator end() const noexcept
        {
            return _memory_view.end();
        }

        iterator cend() const noexcept
        {
            return _memory_view.cend();
        }

        [[nodiscard]] allocator_type get_allocator() const
        {
            return _memory_view.get_allocator();
        }
    };

    template <non_void T, typename Allocator, size_t N>
    class ArrayView
    {
    public:
        using value_type = T;
        using allocator_type = Allocator;
        using memory_view_type = MemoryView<T, Allocator>;
        using iterator = typename memory_view_type::iterator;
        using const_iterator = typename memory_view_type::const_iterator;
        using reference = typename memory_view_type::reference;
        using const_reference = typename memory_view_type::const_reference;
        using ret_type = std::conditional_t<N == 1uLL, reference, ArrayView<T, Allocator, N - 1>>;
        using const_ret_type = std::conditional_t<N == 1uLL, const_reference, ConstArrayView<T, Allocator, N - 1>>;
    private:
        template <non_void, typename, size_t>
        friend class ArrayView;
    private:
        memory_view_type _memory_view;
        std::array<size_t, N> _dimensions;
    private:
        ArrayView(memory_view_type memory_view, const std::array<size_t, N + 1>& dimensions) : _memory_view(memory_view)
        {
            for (size_t i = 0; i < N; ++i)
                _dimensions[i] = dimensions[i + 1];
        }

        ArrayView(memory_view_type memory_view, const std::array<size_t, N>& dimensions)
            : _memory_view(memory_view), _dimensions(dimensions)
        { }
    public:
        static ArrayView from(MemoryView<T, Allocator> memory_view, const std::array<size_t, N>& dimensions)
        {
            return {memory_view, dimensions};
        }

        [[nodiscard]] size_t size() const noexcept
        {
            return _memory_view.size();
        }

        [[nodiscard]] const std::array<size_t, N>& dimensions() const noexcept
        {
            return _dimensions;
        }

        [[nodiscard]] memory_view_type memory_view() const noexcept
        {
            return _memory_view;
        }

        ConstArrayView<T, Allocator, N> as_const() const noexcept
        {
            ConstArrayView<T, Allocator, N>::from(_memory_view, _dimensions);
        }

        ret_type operator[](size_t index)
        {
            if constexpr (N == 1uLL)
                return _memory_view[index];
            else
            {
                auto offset = std::accumulate(_dimensions.begin() + 1, _dimensions.end(), 1uLL, std::multiplies<>());
                return ArrayView<T, Allocator, N - 1>(_memory_view.sub_view(index * offset, offset), _dimensions);
            }
        }

        const_ret_type operator[](size_t index) const
        {
            if constexpr (N == 1uLL)
                return _memory_view[index];
            else
            {
                auto offset = std::accumulate(_dimensions.begin() + 1, _dimensions.end(), 1uLL, std::multiplies<>());
                return ConstArrayView<T, Allocator, N - 1>(_memory_view.sub_view(index * offset, offset), _dimensions);
            }
        }

        auto at(size_t index)
        {
            if (index >= _dimensions[0])
                MBQ_THROW_EXCEPTION(Exception, "Out of range");
            return (*this)[index];
        }

        auto at(size_t index) const
        {
            if (index >= _dimensions[0])
                MBQ_THROW_EXCEPTION(Exception, "Out of range");
            return (*this)[index];
        }

        iterator begin() noexcept
        {
            return _memory_view.begin();
        }

        const_iterator begin() const noexcept
        {
            return _memory_view.begin();
        }

        const_iterator cbegin() const noexcept
        {
            return _memory_view.cbegin();
        }

        iterator end() noexcept
        {
            return _memory_view.end();
        }

        const_iterator end() const noexcept
        {
            return _memory_view.end();
        }

        const_iterator cend() const noexcept
        {
            return _memory_view.cend();
        }

        [[nodiscard]] allocator_type get_allocator() const
        {
            return _memory_view.get_allocator();
        }
    };
} // namespace mbq