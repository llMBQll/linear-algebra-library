#include "common.hpp"

#include <backends/cuda.hpp>
#include <backends/host.hpp>
#include <backends/opencl.hpp>
#include <gtest/gtest.h>
#include <numbers>
#include <types.hpp>

template <typename T>
class MathTest : public testing::Test
{ };

TYPED_TEST_SUITE_P(MathTest);

TYPED_TEST_P(MathTest, Axpy)
{
    using allocator = TypeParam;
    using value_type = typename allocator::value_type;

    constexpr size_t N = 1'000;
    const auto alpha = make_value<value_type>(std::numbers::e);
    auto x = mbq::Array<value_type, allocator, 1>::random(N);
    auto y = mbq::Array<value_type, allocator, 1>::random(N);

    auto x_view = x.view();
    auto y_view = y.view();

    auto a = mbq::axpy(alpha, x, y);
    auto b = mbq::axpy(alpha, x_view, y);
    auto c = mbq::axpy(alpha, x, y_view);
    auto d = mbq::axpy(alpha, x_view, y_view);

    for (size_t i = 0; i < N; i++)
    {
        MBQ_EXPECT_EQ(value_type, a[i], alpha * x[i] + y[i]);
        MBQ_EXPECT_EQ(value_type, a[i], b[i]);
        MBQ_EXPECT_EQ(value_type, b[i], c[i]);
        MBQ_EXPECT_EQ(value_type, c[i], d[i]);
    }
}

TYPED_TEST_P(MathTest, OperatorPlus)
{
    using allocator = TypeParam;
    using value_type = typename allocator::value_type;

    constexpr size_t N = 100;
    auto x = mbq::Array<value_type, allocator, 1>::random(N);
    auto y = mbq::Array<value_type, allocator, 1>::random(N);

    auto x_view = x.view();
    auto y_view = y.view();

    auto a = x + y;
    auto b = x_view + y;
    auto c = x + y_view;
    auto d = x_view + y_view;

    auto e = x + value_type{1};
    auto f = value_type{1} + x;
    auto g = x_view + value_type{1};
    auto h = value_type{1} + x_view;

    auto j = x * value_type{2};
    auto k = value_type{2} * x;

    auto l = x - value_type{1};
    auto m = value_type{1} - x;
    auto n = x - y;

    for (size_t i = 0; i < N; i++)
    {
        MBQ_EXPECT_EQ(value_type, a[i], x[i] + y[i]);
        MBQ_EXPECT_EQ(value_type, a[i], b[i]);
        MBQ_EXPECT_EQ(value_type, b[i], c[i]);
        MBQ_EXPECT_EQ(value_type, c[i], d[i]);
    }

    for (size_t i = 0; i < N; i++)
    {
        MBQ_EXPECT_EQ(value_type, e[i], x[i] + value_type{1});
        MBQ_EXPECT_EQ(value_type, e[i], f[i]);
        MBQ_EXPECT_EQ(value_type, f[i], g[i]);
        MBQ_EXPECT_EQ(value_type, g[i], h[i]);
    }

    for (size_t i = 0; i < N; i++)
    {
        MBQ_EXPECT_EQ(value_type, j[i], x[i] * value_type{2});
        MBQ_EXPECT_EQ(value_type, j[i], k[i]);
    }

    for (size_t i = 0; i < N; i++)
    {
        MBQ_EXPECT_EQ(value_type, l[i], x[i] - value_type{1});
        MBQ_EXPECT_EQ(value_type, m[i], value_type{1} - x[i]);
        MBQ_EXPECT_EQ(value_type, n[i], x[i] - y[i]);
    }
}

TYPED_TEST_P(MathTest, OperatorMultiply)
{
    using allocator = TypeParam;
    using value_type = typename allocator::value_type;

    auto x = mbq::Array<value_type, allocator, 2>::random(2, 3);
    auto y = mbq::Array<value_type, allocator, 2>::random(3, 2);
    auto z = multiply(x, y);
}

TYPED_TEST_P(MathTest, Trigonometric)
{
    using allocator = TypeParam;
    using value_type = typename allocator::value_type;

    constexpr size_t N = 10;
    auto x = mbq::Array<value_type, allocator, 1>::random(N);
    auto y = x;

    sin(x);
    sinh(x);
    cos(x);
    cosh(x);
    tan(x);
    tanh(x);

    sin(y.begin(), y.end());
    sinh(y.begin(), y.end());
    cos(y.begin(), y.end());
    cosh(y.begin(), y.end());
    tan(y.begin(), y.end());
    tanh(y.begin(), y.end());

    for (size_t i = 0; i < N; ++i)
        MBQ_EXPECT_EQ(value_type, x[i], y[i]);
}

TYPED_TEST_P(MathTest, Pow)
{
    using allocator = TypeParam;
    using value_type = typename allocator::value_type;

    constexpr size_t N = 1'000;
    auto x = mbq::Array<value_type, allocator, 1>::random(N);
    pow(x, make_value<value_type>(std::numbers::e));
}

REGISTER_TYPED_TEST_SUITE_P(MathTest, Axpy, OperatorPlus, OperatorMultiply, Trigonometric, Pow);

using real_types =
    ::testing::Types<mbq::host::Allocator<float>, mbq::cuda::Allocator<float>, mbq::opencl::Allocator<float>,
                     mbq::host::Allocator<double>, mbq::cuda::Allocator<double>, mbq::opencl::Allocator<double>>;

using complex_types =
    ::testing::Types<mbq::host::Allocator<std::complex<float>>, mbq::cuda::Allocator<std::complex<float>>,
                     mbq::opencl::Allocator<std::complex<float>>, mbq::host::Allocator<std::complex<double>>,
                     mbq::cuda::Allocator<std::complex<double>>, mbq::opencl::Allocator<std::complex<double>>>;

INSTANTIATE_TYPED_TEST_SUITE_P(Real, MathTest, real_types);
INSTANTIATE_TYPED_TEST_SUITE_P(Complex, MathTest, complex_types);
