#include "common.hpp"

#include <backends/cuda.hpp>
#include <backends/host.hpp>
#include <backends/opencl.hpp>
#include <gtest/gtest.h>
#include <types.hpp>

template <typename T>
class MemoryViewTest : public testing::Test
{ };

TYPED_TEST_SUITE_P(MemoryViewTest);

TYPED_TEST_P(MemoryViewTest, FromStd)
{
    using allocator_type = TypeParam;
    using value_type = typename allocator_type::value_type;
    using memory_view_type = mbq::MemoryView<value_type, allocator_type>;
    using array_view_type = mbq::ArrayView<value_type, allocator_type, 2>;

    constexpr size_t SIZE = 8;

    std::vector<value_type> vec(SIZE, value_type{1});
    std::ranges::generate(vec, [i = 0]() mutable -> value_type { return make_value<value_type>(i++); });
    auto memory_view = memory_view_type::from(vec.data(), vec.size());
    auto array_view = array_view_type::from(memory_view, {4, 2});
    std::cout << transpose(array_view) << '\n';
}

TEST(Test, TestName1)
{
    using value_type = int;
    using memory_view_type = mbq::MemoryView<value_type, std::allocator<value_type>>;
    using array_view_type = mbq::ArrayView<value_type, std::allocator<value_type>, 2>;

    std::vector<value_type> vec{1, 2, 3, 4};
    auto memory_view = memory_view_type::from(vec.data(), vec.size());
    auto array_view = array_view_type::from(memory_view, {2, 2});
    array_view[1][0] = 7;
    auto a = array_view[1][1];
    auto b = vec[2];

    assert(a == 4);
    assert(b == 7);
}

TEST(Test, TestName2)
{
    using value_type = float;
    using array_type = mbq::CudaArray<value_type, 2>;

    auto A = array_type::ones(2, 2);
    std::ranges::transform(A, A.begin(), [](const auto& x) { return x * 2.0f; });
    std::ranges::for_each(A, [](const auto& x) { EXPECT_FLOAT_EQ(x, 2.0f); });
}

REGISTER_TYPED_TEST_SUITE_P(MemoryViewTest, FromStd);

using real_types = ::testing::Types<std::allocator<float>, std::allocator<double>>;

using complex_types = ::testing::Types<std::allocator<std::complex<float>>, std::allocator<std::complex<double>>>;

INSTANTIATE_TYPED_TEST_SUITE_P(Real, MemoryViewTest, real_types);
INSTANTIATE_TYPED_TEST_SUITE_P(Complex, MemoryViewTest, complex_types);