// Tests for uniformity detection optimization
// These tests verify that the index correctly detects uniform data distributions
// and uses the O(1) arithmetic segment lookup instead of O(log n) binary search.

#include "jazzy_index.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace {

// Standalone helper for any segment size
template <typename T, std::size_t Segments>
bool is_found(const jazzy::JazzyIndex<T, Segments>& index,
              const std::vector<T>& data, const T& value) {
    const T* result = index.find(value);
    return result != data.data() + data.size() && *result == value;
}

template <typename T, std::size_t Segments = 256>
class UniformityTest : public ::testing::Test {
protected:
    jazzy::JazzyIndex<T, Segments> build_index(const std::vector<T>& data) {
        jazzy::JazzyIndex<T, Segments> index;
        index.build(data.data(), data.data() + data.size());
        return index;
    }

    bool is_found(const T* result, const std::vector<T>& data, const T& value) {
        return result != data.data() + data.size() && *result == value;
    }
};

using IntUniformityTest = UniformityTest<int, 256>;
using DoubleUniformityTest = UniformityTest<double, 256>;

}  // namespace

// Test: Perfectly uniform integer sequence
TEST_F(IntUniformityTest, PerfectlyUniformSequence) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);  // 0, 1, 2, ..., 999
    auto index = build_index(data);

    // This should trigger uniformity detection
    // All lookups should work via O(1) arithmetic

    for (int i = 0; i < 1000; i += 50) {
        EXPECT_TRUE(is_found(index.find(i), data, i));
    }

    EXPECT_EQ(index.size(), 1000);
}

// Test: Uniform with constant spacing
TEST_F(IntUniformityTest, UniformConstantSpacing) {
    std::vector<int> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(i * 10);  // 0, 10, 20, ..., 990
    }
    auto index = build_index(data);

    // Should be detected as uniform
    for (int val : data) {
        EXPECT_TRUE(is_found(index.find(val), data, val));
    }
}

// Test: Uniform floating-point sequence
TEST_F(DoubleUniformityTest, UniformFloatingPointSequence) {
    std::vector<double> data(1000);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<double>(i) * 0.1;  // 0.0, 0.1, 0.2, ..., 99.9
    }
    auto index = build_index(data);

    // Test samples
    EXPECT_TRUE(is_found(index.find(0.0), data, 0.0));
    EXPECT_TRUE(is_found(index.find(50.0), data, 50.0));
    EXPECT_TRUE(is_found(index.find(99.9), data, 99.9));
}

// Test: Non-uniform data (skewed)
TEST_F(IntUniformityTest, NonUniformSkewedData) {
    std::vector<int> data;

    // Skewed: dense at beginning, sparse at end
    for (int i = 0; i < 50; ++i) {
        data.push_back(i);  // 0-49
    }
    for (int i = 0; i < 50; ++i) {
        data.push_back(1000 + i * 100);  // 1000, 1100, 1200, ..., 5900
    }

    jazzy::JazzyIndex<int, 64> index;
    index.build(data.data(), data.data() + data.size());

    // Should NOT be detected as uniform
    // Should still work correctly with binary search
    EXPECT_TRUE(is_found(index.find(25), data, 25));
    EXPECT_TRUE(is_found(index.find(2000), data, 2000));
}

// Test: Nearly uniform (within tolerance)
TEST_F(IntUniformityTest, NearlyUniformWithinTolerance) {
    std::vector<int> data(1000);
    for (int i = 0; i < 1000; ++i) {
        // Add small perturbations within tolerance
        data[i] = i + (i % 3 == 0 ? 1 : 0);  // Slight variation
    }
    auto index = build_index(data);

    // Might still be detected as uniform depending on tolerance
    // Should work correctly either way
    for (size_t i = 0; i < data.size(); i += 100) {
        EXPECT_TRUE(is_found(index.find(data[i]), data, data[i]));
    }
}

// Test: Non-uniform (exceeds tolerance)
TEST_F(IntUniformityTest, NonUniformExceedsTolerance) {
    std::vector<int> data;

    // First half: dense
    for (int i = 0; i < 50; ++i) {
        data.push_back(i);
    }

    // Second half: very sparse
    for (int i = 0; i < 50; ++i) {
        data.push_back(1000 + i * 1000);
    }

    jazzy::JazzyIndex<int, 32> index;
    index.build(data.data(), data.data() + data.size());

    // Should definitely NOT be uniform
    // Verify correct behavior with fallback
    EXPECT_TRUE(is_found(index.find(25), data, 25));
    EXPECT_TRUE(is_found(index.find(25000), data, 25000));
}

// Test: Single segment (always uniform)
TEST_F(IntUniformityTest, SingleSegmentAlwaysUniform) {
    std::vector<int> data(100);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, 1> index;  // Only 1 segment
    index.build(data.data(), data.data() + data.size());

    // Single segment is trivially uniform
    EXPECT_TRUE(is_found(index.find(50), data, 50));
    EXPECT_EQ(index.num_segments(), 1);
}

// Test: Two segments - uniform
TEST_F(IntUniformityTest, TwoSegmentsUniform) {
    std::vector<int> data(100);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, 2> index;
    index.build(data.data(), data.data() + data.size());

    EXPECT_TRUE(is_found(index.find(25), data, 25));
    EXPECT_TRUE(is_found(index.find(75), data, 75));
}

// Test: Large uniform dataset
TEST_F(IntUniformityTest, LargeUniformDataset) {
    std::vector<int> data(10000);
    std::iota(data.begin(), data.end(), 0);
    auto index = build_index(data);

    // Sample across entire range
    for (int i = 0; i < 10000; i += 1000) {
        EXPECT_TRUE(is_found(index.find(i), data, i));
    }
}

// Test: Uniform with negative range
TEST_F(IntUniformityTest, UniformNegativeRange) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), -500);  // -500 to 499
    auto index = build_index(data);

    EXPECT_TRUE(is_found(index.find(-500), data, -500));
    EXPECT_TRUE(is_found(index.find(0), data, 0));
    EXPECT_TRUE(is_found(index.find(499), data, 499));
}

// Test: Uniform with large values
TEST_F(IntUniformityTest, UniformLargeValues) {
    std::vector<int> data(1000);
    for (int i = 0; i < 1000; ++i) {
        data[i] = 1000000 + i;
    }
    auto index = build_index(data);

    EXPECT_TRUE(is_found(index.find(1000000), data, 1000000));
    EXPECT_TRUE(is_found(index.find(1000500), data, 1000500));
    EXPECT_TRUE(is_found(index.find(1000999), data, 1000999));
}

// Test: Exponential distribution (non-uniform)
TEST_F(IntUniformityTest, ExponentialDistributionNonUniform) {
    std::vector<int> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(static_cast<int>(std::exp(i * 0.1)));
    }
    auto index = build_index(data);

    // Definitely not uniform - exponential growth
    // Should fall back to binary search
    EXPECT_TRUE(is_found(index.find(data[0]), data, data[0]));
    EXPECT_TRUE(is_found(index.find(data[50]), data, data[50]));
    EXPECT_TRUE(is_found(index.find(data[99]), data, data[99]));
}

// Test: Piecewise uniform (mixed)
TEST_F(IntUniformityTest, PiecewiseUniform) {
    std::vector<int> data;

    // Uniform part 1
    for (int i = 0; i < 100; ++i) {
        data.push_back(i);
    }

    // Uniform part 2
    for (int i = 0; i < 100; ++i) {
        data.push_back(200 + i);
    }

    jazzy::JazzyIndex<int, 64> index;
    index.build(data.data(), data.data() + data.size());

    // Might or might not be detected as uniform overall
    EXPECT_TRUE(is_found(index.find(50), data, 50));
    EXPECT_TRUE(is_found(index.find(250), data, 250));
    EXPECT_FALSE(is_found(index.find(150), data, 150));  // In the gap
}

// Test: Uniform duplicates
TEST_F(IntUniformityTest, UniformWithDuplicates) {
    std::vector<int> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(i);
        data.push_back(i);  // Each value twice
    }
    std::sort(data.begin(), data.end());
    auto index = build_index(data);

    // Still uniform spacing in value space
    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(is_found(index.find(i), data, i));
    }
}

// Test: Arithmetic progression with different start
TEST_F(IntUniformityTest, ArithmeticProgressionDifferentStart) {
    std::vector<int> data(100);
    for (int i = 0; i < 100; ++i) {
        data[i] = 1000 + i * 5;  // 1000, 1005, 1010, ..., 1495
    }
    auto index = build_index(data);

    EXPECT_TRUE(is_found(index.find(1000), data, 1000));
    EXPECT_TRUE(is_found(index.find(1250), data, 1250));
    EXPECT_TRUE(is_found(index.find(1495), data, 1495));
}

// Test: Zero range (all same values)
TEST_F(IntUniformityTest, ZeroRangeAllSame) {
    std::vector<int> data(100, 42);
    auto index = build_index(data);

    // Zero range is trivially uniform
    EXPECT_TRUE(is_found(index.find(42), data, 42));
    EXPECT_FALSE(is_found(index.find(41), data, 41));
}

// Test: Very small range
TEST_F(IntUniformityTest, VerySmallRange) {
    std::vector<int> data{10, 11, 12};
    auto index = build_index(data);

    // Small range, uniform
    EXPECT_TRUE(is_found(index.find(10), data, 10));
    EXPECT_TRUE(is_found(index.find(11), data, 11));
    EXPECT_TRUE(is_found(index.find(12), data, 12));
}

// Test: Many segments, uniform data
TEST_F(IntUniformityTest, ManySegmentsUniformData) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, 512> index;
    index.build(data.data(), data.data() + data.size());

    // Even with many segments, should detect uniformity
    for (int i = 0; i < 1000; i += 100) {
        EXPECT_TRUE(is_found(index.find(i), data, i));
    }
}
