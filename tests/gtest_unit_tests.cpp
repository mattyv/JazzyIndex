// Google Test migration of original unit tests from unit_tests.cpp
// Migrated tests now use Google Test framework for better organization,
// parameterization, and integration with modern tooling.

#include "jazzy_index.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace {

// Standalone helper functions that work with any segment size
template <typename IndexType, typename T>
bool expect_found(const IndexType& index,
                  const std::vector<T>& data,
                  const T& value) {
    const T* result = index.find(value);
    if (result == data.data() + data.size()) {
        return false;
    }
    return *result == value;
}

template <typename IndexType, typename T>
bool expect_missing(const IndexType& index,
                    const std::vector<T>& data,
                    const T& value) {
    const T* result = index.find(value);
    return result == data.data() + data.size();
}

// Test fixture for basic JazzyIndex operations
template <typename T, std::size_t Segments = 256>
class JazzyIndexTest : public ::testing::Test {
protected:
    jazzy::JazzyIndex<T, jazzy::to_segment_count<Segments>()> build_index(const std::vector<T>& data) {
        jazzy::JazzyIndex<T, jazzy::to_segment_count<Segments>()> index;
        index.build(data.data(), data.data() + data.size());
        return index;
    }
};

// Type aliases for common test configurations
using IntIndex = JazzyIndexTest<int, 256>;
using UInt64Index = JazzyIndexTest<std::uint64_t, 64>;

}  // namespace

// Test: Empty index returns end iterator
TEST_F(IntIndex, EmptyIndexReturnsEndIterator) {
    std::vector<int> data;
    jazzy::JazzyIndex<int> index;
    index.build(data.data(), data.data());

    EXPECT_TRUE(expect_missing(index, data, 42));
    EXPECT_EQ(index.size(), 0);
}

// Test: Single element hit and miss
TEST_F(IntIndex, SingleElementHitAndMiss) {
    std::vector<int> data{42};
    auto index = build_index(data);

    EXPECT_TRUE(expect_found(index, data, 42));
    EXPECT_TRUE(expect_missing(index, data, 43));
    EXPECT_TRUE(expect_missing(index, data, 41));
    EXPECT_EQ(index.size(), 1);
}

// Test: Uniform sequence lookups
TEST_F(IntIndex, UniformSequenceLookups) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);
    auto index = build_index(data);

    // Test various points in the sequence
    EXPECT_TRUE(expect_found(index, data, 0));
    EXPECT_TRUE(expect_found(index, data, 10));
    EXPECT_TRUE(expect_found(index, data, 500));
    EXPECT_TRUE(expect_found(index, data, 999));

    // Test missing values
    EXPECT_TRUE(expect_missing(index, data, -1));
    EXPECT_TRUE(expect_missing(index, data, 1500));
}

// Test: Duplicate values
TEST_F(IntIndex, DuplicateValues) {
    std::vector<int> data{1, 1, 1, 2, 2, 5, 5, 9};
    jazzy::JazzyIndex<int, jazzy::to_segment_count<128>()> index;
    index.build(data.data(), data.data() + data.size());

    EXPECT_TRUE(expect_found(index, data, 1));
    EXPECT_TRUE(expect_found(index, data, 2));
    EXPECT_TRUE(expect_found(index, data, 5));
    EXPECT_TRUE(expect_found(index, data, 9));

    EXPECT_TRUE(expect_missing(index, data, 3));
    EXPECT_TRUE(expect_missing(index, data, 0));
    EXPECT_TRUE(expect_missing(index, data, 10));
}

// Test: uint64_t values including large numbers
TEST_F(UInt64Index, LargeUInt64Values) {
    std::vector<std::uint64_t> data{
        0ull,
        42ull,
        1'000ull,
        1'000'000ull,
        (1ull << 32),
        (1ull << 40)
    };
    auto index = build_index(data);

    EXPECT_TRUE(expect_found(index, data, 0ull));
    EXPECT_TRUE(expect_found(index, data, 42ull));
    EXPECT_TRUE(expect_found(index, data, (1ull << 40)));

    EXPECT_TRUE(expect_missing(index, data, (1ull << 40) + 1));
    EXPECT_TRUE(expect_missing(index, data, 100ull));
}

// Parameterized test for different segment configurations
class SegmentSizeTest : public ::testing::TestWithParam<std::size_t> {
protected:
    template <std::size_t Segments>
    bool test_with_segments(std::size_t segment_size) {
        std::vector<int> data(1000);
        std::iota(data.begin(), data.end(), 0);

        jazzy::JazzyIndex<int, jazzy::to_segment_count<Segments>()> index;
        index.build(data.data(), data.data() + data.size());

        // Test various lookups
        for (int value : {0, 250, 500, 750, 999}) {
            const int* result = index.find(value);
            if (result == data.data() + data.size() || *result != value) {
                return false;
            }
        }
        return true;
    }
};

TEST_P(SegmentSizeTest, UniformDataWithVariousSegmentSizes) {
    std::size_t segment_size = GetParam();

    bool result = false;
    switch (segment_size) {
        case 64:
            result = test_with_segments<64>(segment_size);
            break;
        case 128:
            result = test_with_segments<128>(segment_size);
            break;
        case 256:
            result = test_with_segments<256>(segment_size);
            break;
        case 512:
            result = test_with_segments<512>(segment_size);
            break;
        default:
            FAIL() << "Unsupported segment size: " << segment_size;
    }

    EXPECT_TRUE(result) << "Failed with segment size: " << segment_size;
}

INSTANTIATE_TEST_SUITE_P(
    DifferentSegmentCounts,
    SegmentSizeTest,
    ::testing::Values(64, 128, 256, 512)
);

// Test: Very small datasets
TEST_F(IntIndex, TwoElements) {
    std::vector<int> data{1, 2};
    auto index = build_index(data);

    EXPECT_TRUE(expect_found(index, data, 1));
    EXPECT_TRUE(expect_found(index, data, 2));
    EXPECT_TRUE(expect_missing(index, data, 0));
    EXPECT_TRUE(expect_missing(index, data, 3));
}

TEST_F(IntIndex, ThreeElements) {
    std::vector<int> data{10, 20, 30};
    auto index = build_index(data);

    EXPECT_TRUE(expect_found(index, data, 10));
    EXPECT_TRUE(expect_found(index, data, 20));
    EXPECT_TRUE(expect_found(index, data, 30));
    EXPECT_TRUE(expect_missing(index, data, 15));
    EXPECT_TRUE(expect_missing(index, data, 25));
}

// Test: Rebuild index
TEST_F(IntIndex, RebuildIndex) {
    std::vector<int> data1{1, 2, 3, 4, 5};
    std::vector<int> data2{10, 20, 30, 40, 50};

    jazzy::JazzyIndex<int, jazzy::to_segment_count<256>()> index;

    // Build with first dataset
    index.build(data1.data(), data1.data() + data1.size());
    EXPECT_TRUE(expect_found(index, data1, 3));
    EXPECT_EQ(index.size(), 5);

    // Rebuild with second dataset
    index.build(data2.data(), data2.data() + data2.size());
    EXPECT_TRUE(expect_found(index, data2, 30));
    EXPECT_TRUE(expect_missing(index, data2, 3));  // Value from old dataset
    EXPECT_EQ(index.size(), 5);
}

// Test: Negative numbers
TEST_F(IntIndex, NegativeNumbers) {
    std::vector<int> data{-100, -50, -10, 0, 10, 50, 100};
    auto index = build_index(data);

    EXPECT_TRUE(expect_found(index, data, -100));
    EXPECT_TRUE(expect_found(index, data, -50));
    EXPECT_TRUE(expect_found(index, data, 0));
    EXPECT_TRUE(expect_found(index, data, 50));
    EXPECT_TRUE(expect_found(index, data, 100));

    EXPECT_TRUE(expect_missing(index, data, -101));
    EXPECT_TRUE(expect_missing(index, data, 101));
    EXPECT_TRUE(expect_missing(index, data, -25));
}
