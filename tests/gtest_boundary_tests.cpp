// Tests for segment boundaries and edge cases
// These tests verify correct behavior at segment boundaries, min/max values,
// and edge cases in the data range.

#include "jazzy_index.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace {

// Standalone helper for any segment size
template <typename IndexType, typename T>
bool is_found(const IndexType& index,
              const std::vector<T>& data, const T& value) {
    const T* result = index.find(value);
    return result != data.data() + data.size() && *result == value;
}

template <typename T, std::size_t Segments = 256>
class BoundaryTest : public ::testing::Test {
protected:
    jazzy::JazzyIndex<T, jazzy::to_segment_count<Segments>()> build_index(const std::vector<T>& data) {
        jazzy::JazzyIndex<T, jazzy::to_segment_count<Segments>()> index;
        index.build(data.data(), data.data() + data.size());
        return index;
    }

    bool is_found(const T* result, const std::vector<T>& data, const T& value) {
        return result != data.data() + data.size() && *result == value;
    }
};

using IntBoundaryTest = BoundaryTest<int, 256>;
using UInt64BoundaryTest = BoundaryTest<std::uint64_t, 256>;

}  // namespace

// Test: Min and max values in dataset
TEST_F(IntBoundaryTest, MinMaxValues) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);
    auto index = build_index(data);

    // Test minimum value
    EXPECT_TRUE(is_found(index.find(0), data, 0));

    // Test maximum value
    EXPECT_TRUE(is_found(index.find(999), data, 999));

    // Test just before min (should not be found)
    EXPECT_FALSE(is_found(index.find(-1), data, -1));

    // Test just after max (should not be found)
    EXPECT_FALSE(is_found(index.find(1000), data, 1000));
}

// Test: Segment boundaries with known segment count
TEST_F(IntBoundaryTest, SegmentBoundariesExplicit) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);

    // Use 10 segments for easy calculation
    jazzy::JazzyIndex<int, jazzy::to_segment_count<10>()> index;
    index.build(data.data(), data.data() + data.size());

    // Each segment should cover 100 elements
    // Test at each segment boundary (0, 100, 200, ..., 900)
    for (int i = 0; i < 10; ++i) {
        int boundary_value = i * 100;
        EXPECT_TRUE(is_found(index.find(boundary_value), data, boundary_value))
            << "Failed at segment boundary " << i << " (value " << boundary_value << ")";
    }

    // Test just before and after each boundary
    for (int i = 1; i < 10; ++i) {
        int boundary = i * 100;
        EXPECT_TRUE(is_found(index.find(boundary - 1), data, boundary - 1))
            << "Failed just before boundary " << i;
        EXPECT_TRUE(is_found(index.find(boundary + 1), data, boundary + 1))
            << "Failed just after boundary " << i;
    }
}

// Test: Segment boundaries with exact quantile splits
TEST_F(IntBoundaryTest, ExactQuantileSplits) {
    std::vector<int> data(256);
    std::iota(data.begin(), data.end(), 0);

    // 256 elements with 256 segments = 1 element per segment
    jazzy::JazzyIndex<int, jazzy::to_segment_count<256>()> index;
    index.build(data.data(), data.data() + data.size());

    // Every single element should be findable
    for (int i = 0; i < 256; ++i) {
        EXPECT_TRUE(is_found(index.find(i), data, i))
            << "Failed to find value " << i;
    }
}

// Test: More segments than data points
TEST_F(IntBoundaryTest, MoreSegmentsThanDataPoints) {
    std::vector<int> data{1, 2, 3, 4, 5};

    // 256 segments but only 5 data points
    jazzy::JazzyIndex<int, jazzy::to_segment_count<256>()> index;
    index.build(data.data(), data.data() + data.size());

    // Should still work correctly
    for (int val : data) {
        EXPECT_TRUE(is_found(index.find(val), data, val));
    }

    EXPECT_FALSE(is_found(index.find(0), data, 0));
    EXPECT_FALSE(is_found(index.find(6), data, 6));
}

// Test: Type limits - minimum values
TEST_F(IntBoundaryTest, TypeMinimumValue) {
    std::vector<int> data{
        std::numeric_limits<int>::min(),
        std::numeric_limits<int>::min() + 1,
        0,
        std::numeric_limits<int>::max() - 1,
        std::numeric_limits<int>::max()
    };
    auto index = build_index(data);

    EXPECT_TRUE(is_found(index.find(std::numeric_limits<int>::min()), data,
                         std::numeric_limits<int>::min()));
    EXPECT_TRUE(is_found(index.find(std::numeric_limits<int>::max()), data,
                         std::numeric_limits<int>::max()));
}

// Test: Type limits - unsigned 64-bit
TEST_F(UInt64BoundaryTest, UInt64TypeLimits) {
    std::vector<std::uint64_t> data{
        0ull,
        1ull,
        std::numeric_limits<std::uint64_t>::max() - 1,
        std::numeric_limits<std::uint64_t>::max()
    };
    auto index = build_index(data);

    EXPECT_TRUE(is_found(index.find(0ull), data, 0ull));
    EXPECT_TRUE(is_found(index.find(std::numeric_limits<std::uint64_t>::max()),
                         data, std::numeric_limits<std::uint64_t>::max()));
}

// Test: First and last elements in each segment
TEST_F(IntBoundaryTest, FirstLastElementsPerSegment) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 100);  // Start from 100

    jazzy::JazzyIndex<int, jazzy::to_segment_count<20>()> index;  // 20 segments, 50 elements each
    index.build(data.data(), data.data() + data.size());

    // Test first and last element of each theoretical segment
    for (int seg = 0; seg < 20; ++seg) {
        int first_idx = seg * 50;
        int last_idx = (seg + 1) * 50 - 1;

        int first_val = data[first_idx];
        int last_val = data[last_idx];

        EXPECT_TRUE(is_found(index.find(first_val), data, first_val))
            << "Segment " << seg << " first element failed";
        EXPECT_TRUE(is_found(index.find(last_val), data, last_val))
            << "Segment " << seg << " last element failed";
    }
}

// Test: Adjacent values at segment boundaries
TEST_F(IntBoundaryTest, AdjacentValuesAtBoundaries) {
    std::vector<int> data;
    // Create data with clear segment structure
    for (int i = 0; i < 100; ++i) {
        data.push_back(i * 10);  // 0, 10, 20, ..., 990
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<10>()> index;  // 10 segments, 10 elements each
    index.build(data.data(), data.data() + data.size());

    // Test around each boundary
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_TRUE(is_found(index.find(data[i]), data, data[i]))
            << "Failed at index " << i;
    }
}

// Test: Boundary with duplicate values
TEST_F(IntBoundaryTest, BoundariesWithDuplicates) {
    std::vector<int> data;

    // Segment 1: many 10s
    for (int i = 0; i < 50; ++i) {
        data.push_back(10);
    }

    // Segment 2: transition and many 20s
    for (int i = 0; i < 50; ++i) {
        data.push_back(20);
    }

    // Segment 3: many 30s
    for (int i = 0; i < 50; ++i) {
        data.push_back(30);
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<16>()> index;
    index.build(data.data(), data.data() + data.size());

    EXPECT_TRUE(is_found(index.find(10), data, 10));
    EXPECT_TRUE(is_found(index.find(20), data, 20));
    EXPECT_TRUE(is_found(index.find(30), data, 30));

    // Values not present
    EXPECT_FALSE(is_found(index.find(15), data, 15));
    EXPECT_FALSE(is_found(index.find(25), data, 25));
}

// Test: Single-element segments
TEST_F(IntBoundaryTest, SingleElementSegments) {
    std::vector<int> data{10, 20, 30, 40, 50};

    jazzy::JazzyIndex<int, jazzy::to_segment_count<5>()> index;  // Same number of segments as elements
    index.build(data.data(), data.data() + data.size());

    for (int val : data) {
        EXPECT_TRUE(is_found(index.find(val), data, val));
    }
}

// Test: Very large gaps between values
TEST_F(UInt64BoundaryTest, LargeGapsBetweenValues) {
    std::vector<std::uint64_t> data{
        1ull,
        1000ull,
        1'000'000ull,
        1'000'000'000ull,
        1'000'000'000'000ull
    };
    auto index = build_index(data);

    for (auto val : data) {
        EXPECT_TRUE(is_found(index.find(val), data, val))
            << "Failed to find value " << val;
    }

    // Test values in the gaps
    EXPECT_FALSE(is_found(index.find(500ull), data, 500ull));
    EXPECT_FALSE(is_found(index.find(500'000ull), data, 500'000ull));
}

// Test: Segment with exact range boundaries
TEST_F(IntBoundaryTest, SegmentRangeBoundaries) {
    std::vector<int> data(100);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::to_segment_count<4>()> index;  // 4 segments: [0-24], [25-49], [50-74], [75-99]
    index.build(data.data(), data.data() + data.size());

    // Test boundaries of each segment
    EXPECT_TRUE(is_found(index.find(0), data, 0));    // Start of segment 0
    EXPECT_TRUE(is_found(index.find(24), data, 24));  // End of segment 0
    EXPECT_TRUE(is_found(index.find(25), data, 25));  // Start of segment 1
    EXPECT_TRUE(is_found(index.find(49), data, 49));  // End of segment 1
    EXPECT_TRUE(is_found(index.find(50), data, 50));  // Start of segment 2
    EXPECT_TRUE(is_found(index.find(74), data, 74));  // End of segment 2
    EXPECT_TRUE(is_found(index.find(75), data, 75));  // Start of segment 3
    EXPECT_TRUE(is_found(index.find(99), data, 99));  // End of segment 3
}

// Test: Out of bounds queries at extremes
TEST_F(IntBoundaryTest, OutOfBoundsExtremes) {
    std::vector<int> data{10, 20, 30, 40, 50};
    auto index = build_index(data);

    // Way below minimum
    EXPECT_FALSE(is_found(index.find(-1000), data, -1000));
    EXPECT_FALSE(is_found(index.find(std::numeric_limits<int>::min()), data,
                          std::numeric_limits<int>::min()));

    // Way above maximum
    EXPECT_FALSE(is_found(index.find(1000), data, 1000));
    EXPECT_FALSE(is_found(index.find(std::numeric_limits<int>::max()), data,
                          std::numeric_limits<int>::max()));
}

// Test: Boundary at zero crossing
TEST_F(IntBoundaryTest, ZeroCrossingBoundary) {
    std::vector<int> data;
    for (int i = -50; i <= 50; ++i) {
        data.push_back(i);
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<10>()> index;
    index.build(data.data(), data.data() + data.size());

    // Test around zero
    EXPECT_TRUE(is_found(index.find(-1), data, -1));
    EXPECT_TRUE(is_found(index.find(0), data, 0));
    EXPECT_TRUE(is_found(index.find(1), data, 1));

    // Test extremes
    EXPECT_TRUE(is_found(index.find(-50), data, -50));
    EXPECT_TRUE(is_found(index.find(50), data, 50));
}
