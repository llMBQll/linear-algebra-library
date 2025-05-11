#pragma once

#include <Array.hpp>
#include <complex>
#include <iostream>
#include <vector>

#define MBQ_EXPECT_EQ(m__type, m__a, m__b)                                                                             \
    [](const m__type& m_a, const m__type& m_b) {                                                                       \
        if constexpr (std::is_same_v<m__type, float>)                                                                  \
        {                                                                                                              \
            EXPECT_FLOAT_EQ(m_a, m_b);                                                                                 \
        }                                                                                                              \
        else if constexpr (std::is_same_v<m__type, double>)                                                            \
        {                                                                                                              \
            EXPECT_DOUBLE_EQ(m_a, m_b);                                                                                \
        }                                                                                                              \
        else if constexpr (std::is_same_v<m__type, std::complex<float>>)                                               \
        {                                                                                                              \
            EXPECT_FLOAT_EQ(m_a.real(), m_b.real());                                                                   \
            EXPECT_FLOAT_EQ(m_a.imag(), m_b.imag());                                                                   \
        }                                                                                                              \
        else if constexpr (std::is_same_v<m__type, std::complex<double>>)                                              \
        {                                                                                                              \
            EXPECT_DOUBLE_EQ(m_a.real(), m_b.real());                                                                  \
            EXPECT_DOUBLE_EQ(m_a.imag(), m_b.imag());                                                                  \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            EXPECT_EQ(m_a, m_b);                                                                                       \
        }                                                                                                              \
    }(static_cast<const m__type&>(m__a), static_cast<const m__type&>(m__b))

namespace detail
{
    template <typename Iter>
    void print(std::ostream& out, Iter begin, Iter end, const char* brackets)
    {
        auto current = begin;
        out << brackets[0];
        while (current != end)
        {
            if (current != begin)
                out << ", ";
            out << *current++;
        }
        out << brackets[1];
    }
} // namespace detail

template <typename T, typename Allocator>
std::ostream& operator<<(std::ostream& out, const std::vector<T, Allocator>& x)
{
    detail::print(out, x.begin(), x.end(), "[]");
    return out;
}

template <typename T, typename Allocator, size_t N>
std::ostream& operator<<(std::ostream& out, const mbq::Array<T, Allocator, N>& x)
{
    detail::print(out, x.begin(), x.end(), "[]");
    return out;
}

template <typename T, typename Allocator>
std::ostream& operator<<(std::ostream& out, const mbq::Memory<T, Allocator>& x)
{
    detail::print(out, x.begin(), x.end(), "[]");
    return out;
}

template <typename>
struct is_complex_t : std::false_type
{ };

template <typename T>
struct is_complex_t<std::complex<T>> : std::true_type
{ };

template <typename T>
constexpr inline bool is_complex_v = is_complex_t<T>::value;

template <typename T>
concept complex = is_complex_v<T>;

template <typename T, typename U>
T make_value(const U& value)
{
    if constexpr (std::same_as<std::complex<float>, T>)
    {
        return T{static_cast<float>(value), 0};
    }
    else if constexpr (std::same_as<std::complex<double>, T>)
    {
        return T{static_cast<double>(value), 0};
    }
    else
    {
        return static_cast<T>(value);
    }
}