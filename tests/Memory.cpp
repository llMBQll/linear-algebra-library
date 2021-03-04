#include "common.hpp"

#include <backends/cuda.hpp>
#include <backends/host.hpp>
#include <backends/opencl.hpp>
#include <gtest/gtest.h>
#include <types.hpp>

template <typename T>
class MemoryTest : public testing::Test
{ };

TYPED_TEST_SUITE_P(MemoryTest);

TYPED_TEST_P(MemoryTest, Constructor)
{
    using allocator_type = TypeParam;
    using value_type = typename allocator_type::value_type;
    using memory_type = mbq::Memory<value_type, allocator_type>;

    constexpr size_t SIZE = 1'000;

    memory_type memory{SIZE};
    ASSERT_EQ(memory.size(), SIZE);

    std::ranges::generate(memory, [i = 0]() mutable { return static_cast<value_type>(i++); });
    for (size_t i = 0; i < SIZE; ++i)
        MBQ_EXPECT_EQ(value_type, memory[i], i);
}

TYPED_TEST_P(MemoryTest, ResizeDown)
{
    using allocator_type = TypeParam;
    using value_type = typename allocator_type::value_type;
    using memory_type = mbq::Memory<value_type, allocator_type>;

    constexpr size_t INITIAL_SIZE = 1'000;
    constexpr size_t NEW_SIZE = 500;

    memory_type memory{INITIAL_SIZE};
    ASSERT_EQ(memory.size(), INITIAL_SIZE);
    mbq::fill(memory, value_type{0});

    memory.resize(NEW_SIZE);
    ASSERT_EQ(memory.size(), NEW_SIZE);

    for (size_t i = 0; i < NEW_SIZE; ++i)
        MBQ_EXPECT_EQ(value_type, memory[i], value_type{0});
}

TYPED_TEST_P(MemoryTest, ResizeUp)
{
    using allocator_type = TypeParam;
    using value_type = typename allocator_type::value_type;
    using memory_type = mbq::Memory<value_type, allocator_type>;

    constexpr size_t INITIAL_SIZE = 1'000;
    constexpr size_t NEW_SIZE = 2'000;

    memory_type memory{INITIAL_SIZE};
    ASSERT_EQ(memory.size(), INITIAL_SIZE);
    mbq::fill(memory, value_type{0});

    memory.resize(NEW_SIZE, value_type{1});
    ASSERT_EQ(memory.size(), NEW_SIZE);

    for (size_t i = 0; i < INITIAL_SIZE; ++i)
        MBQ_EXPECT_EQ(value_type, memory[i], value_type{0});
    for (size_t i = INITIAL_SIZE; i < NEW_SIZE; ++i)
        MBQ_EXPECT_EQ(value_type, memory[i], value_type{1});
}

TYPED_TEST_P(MemoryTest, FromVector)
{
    using allocator_type = TypeParam;
    using value_type = typename allocator_type::value_type;
    using memory_type = mbq::Memory<value_type, allocator_type>;

    constexpr size_t SIZE = 1'000;

    std::vector<value_type> vec(SIZE, value_type{1});
    auto memory = memory_type::from(vec.data(), vec.size(), vec.get_allocator());
    ASSERT_EQ(memory.size(), SIZE);

    for (size_t i = 0; i < SIZE; ++i)
        MBQ_EXPECT_EQ(value_type, memory[i], value_type{1});
}

TYPED_TEST_P(MemoryTest, Move)
{
    using allocator_type = TypeParam;
    using value_type = typename allocator_type::value_type;
    using memory_type = mbq::Memory<value_type, allocator_type>;
    constexpr size_t SIZE = 1'000;

    memory_type memory{SIZE};
    memory_type x;
    x = std::move(memory);
}

REGISTER_TYPED_TEST_SUITE_P(MemoryTest, Constructor, ResizeDown, ResizeUp, FromVector, Move);

using real_types =
    ::testing::Types<mbq::host::Allocator<double>, mbq::cuda::Allocator<double>, mbq::opencl::Allocator<double>>;

using complex_types =
    ::testing::Types<mbq::host::Allocator<std::complex<double>>, mbq::cuda::Allocator<std::complex<double>>,
                     mbq::opencl::Allocator<std::complex<double>>>;

INSTANTIATE_TYPED_TEST_SUITE_P(Real, MemoryTest, real_types);
INSTANTIATE_TYPED_TEST_SUITE_P(Complex, MemoryTest, complex_types);