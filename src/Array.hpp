#pragma once

#include "algorithm/fill.hpp"
#include "algorithm/random.hpp"
#include "ArrayView.hpp"

#include <array>
#include <numeric>
#include <random>

namespace mbq
{
    template <non_void T, typename Allocator, size_t N>
    class Array
    {
    private:
        struct args
        {
        private:
            template <typename... Ts>
            static std::array<size_t, N> to_array(const std::tuple<Ts...>& tuple)
            {
                // tuple is actually of type <size_t, size_t, ..., Allocator> with N size_t elements
                return [&tuple]<size_t... Indices>(std::index_sequence<Indices...>) -> std::array<size_t, N> {
                    return {static_cast<size_t>(std::get<Indices>(tuple))...};
                }(std::make_index_sequence<N>());
            }
        public:
            Allocator _allocator;
            std::array<size_t, N> _array;
        public:
            template <typename... Vs>
            constexpr args(Allocator allocator, Vs... vs) : _allocator(allocator), _array{static_cast<size_t>(vs)...}
            { }

            constexpr explicit args(Allocator allocator, const std::array<size_t, N>& dimensions)
                : _allocator(allocator), _array{dimensions}
            { }

            template <typename... Ts>
            constexpr explicit args(std::tuple<Ts...> tuple) : _allocator(std::get<N>(tuple)), _array(to_array(tuple))
            { }

            [[nodiscard]] constexpr size_t accumulate() const
            {
                return std::accumulate(_array.begin(), _array.end(), 1uLL, std::multiplies<>());
            }
        };
    public:
        using value_type = T;
        using allocator_type = Allocator;
        using memory_type = Memory<T, Allocator>;
        using iterator = typename memory_type::iterator;
        using const_iterator = typename memory_type::const_iterator;
        using reference = typename memory_type::reference;

        template <non_void, typename, size_t>
        friend class Array;

        static_assert(N > 0uLL);
        friend class ArrayView<T, Allocator, N - 1>;
    private:
        memory_type _memory;
        std::array<size_t, N> _dimensions;
    private:
        explicit Array(args args) : _memory(args.accumulate(), args._allocator), _dimensions(args._array) { }

        Array(const memory_type& memory, const std::array<size_t, N>& dimensions)
            : _memory(memory), _dimensions(dimensions)
        { }

        Array(memory_type&& memory, const std::array<size_t, N>& dimensions)
            : _memory(std::move(memory)), _dimensions(dimensions)
        { }

        template <size_t Size>
        static size_t size_from_dimensions(const std::array<size_t, Size>& array)
        {
            return std::accumulate(array.begin(), array.end(), 1uLL, std::multiplies<>());
        }

        template <size_t Size>
        static void assert_size(const std::array<size_t, Size>& dimensions, size_t source_size)
        {
            if (size_from_dimensions(dimensions) != source_size)
                MBQ_THROW_EXCEPTION(
                    Exception,
                    "Number of elements in the array must be equal to the number of elements in source object");
        }
    public:
        constexpr inline static size_t n = N;

        Array() : _memory()
        {
            std::ranges::fill(_dimensions, 0);
        }

        template <typename... Ts>
            requires valid_dimensions<N, Ts...>
        explicit Array(Ts&&... ts) : Array(args{allocator_type{}, std::forward<Ts>(ts)...})
        { }

        explicit Array(const std::array<size_t, N>& dimensions) : Array(args{allocator_type{}, dimensions}) { }

        template <typename... Ts>
            requires valid_dimensions_with_allocator<allocator_type, N, Ts...>
        explicit Array(Ts&&... ts) : Array(args{std::forward_as_tuple<Ts...>(std::forward<Ts>(ts)...)})
        { }

        explicit Array(const std::array<size_t, N>& dimensions, allocator_type allocator)
            : Array(args{allocator, dimensions})
        { }

        Array(const Array& other) = default;

        template <typename OtherAlloc>
        Array(const Array<value_type, OtherAlloc, N>& other) : _memory(other._memory), _dimensions(other._dimensions)
        { }

        Array(Array&& other) noexcept = default;

        Array& operator=(const Array& other) = default;

        template <typename OtherAlloc>
        Array& operator=(const Array<value_type, OtherAlloc, N>& other)
        {
            _dimensions = other._dimensions;
            _memory = other._memory;
            return *this;
        }

        Array& operator=(Array&& other) noexcept = default;

        explicit Array(const ArrayView<T, Allocator, N>& view)
            : _memory(&(*view.begin()), view.end() - view.begin(), view.get_allocator()), _dimensions(view.dimensions())
        { }

        explicit Array(const ConstArrayView<T, Allocator, N>& view)
            : _memory(&(*view.begin()), view.end() - view.begin(), view.get_allocator()), _dimensions(view.dimensions())
        { }

        template <typename... Ts>
            requires valid<allocator_type, N, Ts...>
        static Array zeros(Ts&&... ts)
        {
            Array array(std::forward<Ts>(ts)...);
            fill(array, value_type{0});
            return array;
        }

        template <typename... Ts>
            requires valid<allocator_type, N, Ts...>
        static Array ones(Ts&&... ts)
        {
            Array array(std::forward<Ts>(ts)...);
            fill(array, value_type{1});
            return array;
        }

        template <typename... Ts>
            requires valid<allocator_type, N, Ts...>
        static Array random(Ts&&... ts)
        {
            Array array(std::forward<Ts>(ts)...);
            ::mbq::random(array);
            return array;
        }

        static Array from(const MemoryView<T, Allocator>& view, const std::array<size_t, N>& dimensions)
        {
            assert_size(dimensions, view.size());
            return {Memory<T, Allocator>::from(view), dimensions};
        }

        template <typename... Dimensions>
            requires valid_dimensions<N, Dimensions...>
        static Array from(const MemoryView<T, Allocator>& view, Dimensions... dimensions)
        {
            return Array::from(view, {dimensions...});
        }

        static Array from(const ConstMemoryView<T, Allocator>& view, const std::array<size_t, N>& dimensions)
        {
            assert_size(dimensions, view.size());
            return {Memory<T, Allocator>::from(view), dimensions};
        }

        template <typename... Dimensions>
            requires valid_dimensions<N, Dimensions...>
        static Array from(const ConstMemoryView<T, Allocator>& view, Dimensions... dimensions)
        {
            return Array::from(view, {dimensions...});
        }

        static Array from(const Memory<T, Allocator>& memory, const std::array<size_t, N>& dimensions)
        {
            assert_size(dimensions, memory.size());
            return {memory, dimensions};
        }

        template <typename... Dimensions>
            requires valid_dimensions<N, Dimensions...>
        static Array from(const Memory<T, Allocator>& memory, Dimensions... dimensions)
        {
            return Array::from(memory, {dimensions...});
        }

        static Array from(Memory<T, Allocator>&& memory, const std::array<size_t, N>& dimensions)
        {
            assert_size(dimensions, memory.size());
            return {std::forward<Memory<T, Allocator>>(memory), dimensions};
        }

        template <typename... Dimensions>
            requires valid_dimensions<N, Dimensions...>
        static Array from(Memory<T, Allocator>&& memory, Dimensions... dimensions)
        {
            return Array::from(std::forward<Memory<T, Allocator>>(memory), {dimensions...});
        }

        static Array from(const ArrayView<T, Allocator, N>& view)
        {
            return {Memory<T, Allocator>::from(view.memory_view()), view.dimensions()};
        }

        template <size_t ViewN>
        static Array from(const ArrayView<T, Allocator, ViewN>& view, const std::array<size_t, N>& dimensions)
        {
            assert_size(dimensions, view.size());
            return {Memory<T, Allocator>::from(view.memory_view()), dimensions};
        }

        template <size_t ViewN, typename... Dimensions>
            requires valid_dimensions<N, Dimensions...>
        static Array from(const ArrayView<T, Allocator, ViewN>& view, Dimensions... dimensions)
        {
            return Array::from(view, {dimensions...});
        }

        static Array from(const ConstArrayView<T, Allocator, N>& view)
        {
            return {Memory<T, Allocator>::from(view.memory_view()), view.dimensions()};
        }

        template <size_t ViewN>
        static Array from(const ConstArrayView<T, Allocator, ViewN>& view, const std::array<size_t, N>& dimensions)
        {
            assert_size(dimensions, view.size());
            return {Memory<T, Allocator>::from(view.memory_view()), dimensions};
        }

        template <size_t ViewN, typename... Dimensions>
            requires valid_dimensions<N, Dimensions...>
        static Array from(const ConstArrayView<T, Allocator, ViewN>& view, Dimensions... dimensions)
        {
            return Array::from(view, {dimensions...});
        }

        [[nodiscard]] size_t size() const noexcept
        {
            return _memory.size();
        }

        [[nodiscard]] const std::array<size_t, N>& dimensions() const
        {
            return _dimensions;
        }

        auto operator[](size_t index)
        {
            return ArrayView<T, Allocator, N>::from(_memory.view(), _dimensions)[index];
        }

        auto operator[](size_t index) const
        {
            return ConstArrayView<T, Allocator, N>::from(_memory.const_view(), _dimensions)[index];
        }

        auto at(size_t index)
        {
            return ArrayView<T, Allocator, N>(_memory.view(), _dimensions).at(index);
        }

        auto at(size_t index) const
        {
            return ConstArrayView<T, Allocator, N>(_memory.const_view(), _dimensions).at(index);
        }

        iterator begin() noexcept
        {
            return _memory.begin();
        }

        iterator end() noexcept
        {
            return _memory.end();
        }

        const_iterator begin() const noexcept
        {
            return _memory.begin();
        }

        const_iterator end() const noexcept
        {
            return _memory.end();
        }

        const_iterator cbegin() const noexcept
        {
            return _memory.begin();
        }

        const_iterator cend() const noexcept
        {
            return _memory.end();
        }

        [[nodiscard]] allocator_type get_allocator() const
        {
            return _memory.get_allocator();
        }

        ArrayView<T, Allocator, N> view()
        {
            return ArrayView<T, Allocator, N>::from(_memory.view(), _dimensions);
        }

        ConstArrayView<T, Allocator, N> view() const
        {
            return ConstArrayView<T, Allocator, N>::from(_memory.view(), _dimensions);
        }

        ConstArrayView<T, Allocator, N> const_view() const
        {
            return ConstArrayView<T, Allocator, N>::from(_memory.view(), _dimensions);
        }

        MemoryView<T, Allocator> memory_view()
        {
            return _memory.view();
        }

        ConstMemoryView<T, Allocator> memory_view() const
        {
            return _memory.view();
        }

        ConstMemoryView<T, Allocator> const_memory_view() const
        {
            return _memory.view();
        }

        template <std::convertible_to<size_t>... Dimensions>
            requires requires { sizeof...(Dimensions) > 0uLL; }
        [[nodiscard]] Array<T, Allocator, sizeof...(Dimensions)> reshape(Dimensions... new_dimensions) const
        {
            constexpr size_t SIZE = sizeof...(Dimensions);

            std::array<size_t, SIZE> array{static_cast<size_t>(new_dimensions)...};
            if (size_from_dimensions(array) != size())
                MBQ_THROW_EXCEPTION(Exception, "Number of elements in the new array must be equal");
            return Array<T, Allocator, SIZE>::from(_memory, array);
        }
    };
} // namespace mbq