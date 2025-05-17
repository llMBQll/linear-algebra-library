#include "common.hpp"

#include <backends/cuda.hpp>
#include <backends/host.hpp>
#include <backends/opencl.hpp>
#include <gtest/gtest.h>
#include <types.hpp>

template <typename>
class ArrayTest : public testing::Test
{ };

TYPED_TEST_SUITE_P(ArrayTest);

template <complex T>
testing::AssertionResult assert_uniform_distribution(const T& sum, size_t count) noexcept
{
    const auto value = sum / make_value<T>(count).real();
    auto result = (value.real() > 0.48 && value.real() < 0.52) && (value.imag() > 0.48 && value.imag() < 0.52);
    return result ? testing::AssertionSuccess() : testing::AssertionFailure() << value;
}

template <typename T>
testing::AssertionResult assert_uniform_distribution(const T& sum, size_t count) noexcept
{
    const auto value = sum / count;
    auto result = value > 0.48 && value < 0.52;
    return result ? testing::AssertionSuccess() : testing::AssertionFailure() << value;
}

TYPED_TEST_P(ArrayTest, Fill)
{
    using allocator_type = TypeParam;
    using value_type = typename allocator_type::value_type;
    using array_t = mbq::Array<value_type, allocator_type, 1>;

    constexpr size_t SIZE = 2'000;

    auto zeros_array = array_t::zeros(SIZE);
    ASSERT_EQ(zeros_array.size(), SIZE);
    std::ranges::for_each(zeros_array, [](const auto& x) { MBQ_EXPECT_EQ(value_type, x, 0); });

    auto ones_array = array_t::ones(SIZE);
    ASSERT_EQ(ones_array.size(), SIZE);
    std::ranges::for_each(ones_array, [](const auto& x) { MBQ_EXPECT_EQ(value_type, x, 1); });

    auto random_array = array_t::random(SIZE);
    ASSERT_EQ(random_array.size(), SIZE);
    auto sum = std::accumulate(random_array.begin(), random_array.end(), value_type{0},
                               [](const auto& x, auto current) { return current + static_cast<value_type>(x); });
    EXPECT_TRUE(assert_uniform_distribution(sum, random_array.size()));
}

TYPED_TEST_P(ArrayTest, SubscriptOperator)
{
    using allocator_type = TypeParam;
    using value_type = typename allocator_type::value_type;
    using array_t = mbq::Array<value_type, allocator_type, 4>;

    array_t array(5, 20, 5, 8);
    ASSERT_EQ(array.size(), 5 * 20 * 5 * 8);
    std::ranges::generate(array, [i = 0]() mutable { return make_value<value_type>(i++); });

    int v = 0;
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = 0; j < 20; ++j)
            for (size_t k = 0; k < 5; ++k)
                for (size_t l = 0; l < 8; ++l)
                    MBQ_EXPECT_EQ(value_type, array[i][j][k][l], make_value<value_type>(v++));
}

TYPED_TEST_P(ArrayTest, Transpose)
{
    using allocator_type = TypeParam;
    using value_type = typename allocator_type::value_type;
    using array_t = mbq::Array<value_type, allocator_type, 2>;

    constexpr size_t ROWS = 3;
    constexpr size_t COLS = 7;

    auto x = array_t::random(ROWS, COLS);
    auto y = transpose(x);

    ASSERT_EQ(y.dimensions()[0], COLS);
    ASSERT_EQ(y.dimensions()[1], ROWS);

    for (size_t row = 0; row < ROWS; ++row)
        for (size_t col = 0; col < COLS; ++col)
            MBQ_EXPECT_EQ(value_type, x[row][col], y[col][row]);
}

REGISTER_TYPED_TEST_SUITE_P(ArrayTest, Fill, SubscriptOperator, Transpose);

using real_types =
    ::testing::Types<mbq::host::Allocator<float>, mbq::cuda::Allocator<float>, mbq::opencl::Allocator<float>,
                     mbq::host::Allocator<double>, mbq::cuda::Allocator<double>, mbq::opencl::Allocator<double>>;

using complex_types =
    ::testing::Types<mbq::host::Allocator<std::complex<float>>, mbq::cuda::Allocator<std::complex<float>>,
                     mbq::opencl::Allocator<std::complex<float>>, mbq::host::Allocator<std::complex<double>>,
                     mbq::cuda::Allocator<std::complex<double>>, mbq::opencl::Allocator<std::complex<double>>>;

INSTANTIATE_TYPED_TEST_SUITE_P(Real, ArrayTest, real_types);
INSTANTIATE_TYPED_TEST_SUITE_P(Complex, ArrayTest, complex_types);