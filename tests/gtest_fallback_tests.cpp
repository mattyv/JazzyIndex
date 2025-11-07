// Tests for directional binary search fallback optimization
// These tests verify that when uniform segment lookup fails, the fallback
// binary search is restricted to the correct half (left or right) based on
// which direction the actual segment is relative to the failed prediction.

#include "jazzy_index.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace {

template <typename T, std::size_t Segments = 256>
class FallbackTest : public ::testing::Test {
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

using IntFallbackTest = FallbackTest<int, 64>;
using DoubleFallbackTest = FallbackTest<double, 64>;

}  // namespace

// Test: Data that's barely uniform enough to pass threshold but has clustering
// This creates a scenario where O(1) lookup might occasionally fail
TEST_F(IntFallbackTest, BarelyUniformWithClustering) {
    std::vector<int> data;

    // Create data with slight clustering at both ends
    // First 40% of data: values 0-3999 (dense)
    for (int i = 0; i < 4000; ++i) {
        data.push_back(i);
    }

    // Middle 20% of data: values 4000-5999 (sparse, skipping every other)
    for (int i = 4000; i < 6000; i += 2) {
        data.push_back(i);
    }

    // Last 40% of data: values 6000-9999 (dense again)
    for (int i = 6000; i < 10000; ++i) {
        data.push_back(i);
    }

    std::sort(data.begin(), data.end());
    auto index = build_index(data);

    // All values should still be findable, even if O(1) lookup fails sometimes
    // The directional fallback should rescue any failed predictions

    // Test values from dense left region
    for (int i = 0; i < 4000; i += 100) {
        EXPECT_TRUE(is_found(index.find(i), data, i))
            << "Failed to find " << i << " in dense left region";
    }

    // Test values from sparse middle region
    for (int i = 4000; i < 6000; i += 100) {
        if (std::binary_search(data.begin(), data.end(), i)) {
            EXPECT_TRUE(is_found(index.find(i), data, i))
                << "Failed to find " << i << " in sparse middle region";
        }
    }

    // Test values from dense right region
    for (int i = 6000; i < 10000; i += 100) {
        EXPECT_TRUE(is_found(index.find(i), data, i))
            << "Failed to find " << i << " in dense right region";
    }
}

// Test: Floating-point data where rounding errors could cause O(1) lookup to fail
TEST_F(DoubleFallbackTest, FloatingPointRoundingEdgeCases) {
    std::vector<double> data;

    // Create data with values that might cause rounding issues
    // when computing segment_scale_
    for (int i = 0; i < 10000; ++i) {
        // Use values that aren't exactly representable in floating-point
        data.push_back(static_cast<double>(i) / 3.0);
    }

    auto index = build_index(data);

    // Test edge cases where rounding errors are most likely
    EXPECT_TRUE(is_found(index.find(0.0), data, 0.0));
    EXPECT_TRUE(is_found(index.find(1.0 / 3.0), data, 1.0 / 3.0));
    EXPECT_TRUE(is_found(index.find(100.0 / 3.0), data, 100.0 / 3.0));
    EXPECT_TRUE(is_found(index.find(1000.0 / 3.0), data, 1000.0 / 3.0));
    EXPECT_TRUE(is_found(index.find(9999.0 / 3.0), data, 9999.0 / 3.0));
}

// Test: Data with power-of-two segments to test boundary cases
TEST_F(IntFallbackTest, PowerOfTwoSegmentBoundaries) {
    std::vector<int> data;

    // Create exactly 64 * 100 = 6400 elements
    // With 64 segments, each segment should have exactly 100 elements
    for (int i = 0; i < 6400; ++i) {
        data.push_back(i);
    }

    auto index = build_index(data);

    // Test values at segment boundaries (where O(1) lookup is most likely to be off by 1)
    // Segment boundaries should be at multiples of 100
    for (int seg = 0; seg < 64; ++seg) {
        int boundary_value = seg * 100;
        if (boundary_value < 6400) {
            EXPECT_TRUE(is_found(index.find(boundary_value), data, boundary_value))
                << "Failed to find segment boundary value " << boundary_value;
        }
    }
}

// Test: Skewed distribution that might be falsely detected as uniform
// This tests the fallback when uniformity detection gives a false positive
TEST_F(IntFallbackTest, SkewedDataWithFalseUniformDetection) {
    std::vector<int> data;

    // Create heavily skewed data:
    // 80% of data in first 20% of value range
    // 20% of data in last 80% of value range

    // First segment: 0-1999 (8000 elements)
    for (int i = 0; i < 8000; ++i) {
        data.push_back(i * 2000 / 8000);
    }

    // Second segment: 2000-9999 (2000 elements)
    for (int i = 0; i < 2000; ++i) {
        data.push_back(2000 + i * 8000 / 2000);
    }

    std::sort(data.begin(), data.end());
    auto index = build_index(data);

    // Test values from dense region (where O(1) might predict too far right)
    for (int i = 0; i < 2000; i += 50) {
        if (std::binary_search(data.begin(), data.end(), i)) {
            EXPECT_TRUE(is_found(index.find(i), data, i))
                << "Failed to find " << i << " in dense left region";
        }
    }

    // Test values from sparse region (where O(1) might predict too far left)
    for (int i = 2000; i < 10000; i += 200) {
        if (std::binary_search(data.begin(), data.end(), i)) {
            EXPECT_TRUE(is_found(index.find(i), data, i))
                << "Failed to find " << i << " in sparse right region";
        }
    }
}

// Test: Large dataset with subtle non-uniformity
TEST_F(IntFallbackTest, LargeDatasetWithSubtleNonUniformity) {
    std::vector<int> data;

    // Create data with slight variations in density
    // This might pass uniformity threshold but still cause some O(1) failures
    std::mt19937 gen(12345);  // Fixed seed for reproducibility
    std::uniform_int_distribution<> dis(0, 100);

    int value = 0;
    for (int i = 0; i < 100000; ++i) {
        data.push_back(value);
        // Increment varies slightly: usually 10, but sometimes 9-11
        value += 10 + (dis(gen) % 3) - 1;
    }

    auto index = build_index(data);

    // Test random samples throughout the range
    for (size_t i = 0; i < data.size(); i += 1000) {
        EXPECT_TRUE(is_found(index.find(data[i]), data, data[i]))
            << "Failed to find data[" << i << "] = " << data[i];
    }

    // Test first and last elements specifically
    EXPECT_TRUE(is_found(index.find(data.front()), data, data.front()));
    EXPECT_TRUE(is_found(index.find(data.back()), data, data.back()));
}

// Test: Data with gaps that might cause segment boundary issues
TEST_F(IntFallbackTest, DataWithGapsCausingBoundaryIssues) {
    std::vector<int> data;

    // Create data with intentional gaps at regular intervals
    for (int seg = 0; seg < 64; ++seg) {
        // Each "segment" has 90 values, then a gap of 10
        int base = seg * 100;
        for (int i = 0; i < 90; ++i) {
            data.push_back(base + i);
        }
        // Gap: values [base+90, base+100) are missing
    }

    auto index = build_index(data);

    // Test values at the edges of gaps (most likely to trigger fallback)
    for (int seg = 0; seg < 64; ++seg) {
        int base = seg * 100;

        // Last value before gap
        EXPECT_TRUE(is_found(index.find(base + 89), data, base + 89))
            << "Failed to find value before gap at segment " << seg;
    }
}

// Test: All segment lookups succeed even with edge cases
TEST_F(IntFallbackTest, ComprehensiveEdgeCaseValidation) {
    std::vector<int> data(10000);
    std::iota(data.begin(), data.end(), 0);

    auto index = build_index(data);

    // Test every single value to ensure none fall through the cracks
    for (int i = 0; i < 10000; ++i) {
        EXPECT_TRUE(is_found(index.find(i), data, i))
            << "Failed to find value " << i;
    }
}

// Test: Verify directional optimization doesn't break find_lower_bound
TEST_F(IntFallbackTest, FindLowerBoundWithFallback) {
    std::vector<int> data;
    for (int i = 0; i < 1000; ++i) {
        data.push_back(i * 10);  // 0, 10, 20, ..., 9990
    }

    auto index = build_index(data);

    // Test find_lower_bound for values that don't exist
    // These should trigger segment search and potentially use fallback
    for (int i = 5; i < 10000; i += 100) {
        const int* result = index.find_lower_bound(i);
        if (result != data.data() + data.size()) {
            // Result should be the first element >= i
            int expected = ((i + 9) / 10) * 10;  // Round up to nearest multiple of 10
            EXPECT_EQ(*result, expected)
                << "find_lower_bound(" << i << ") returned wrong value";
        }
    }
}

// Test: Verify directional optimization doesn't break find_upper_bound
TEST_F(IntFallbackTest, FindUpperBoundWithFallback) {
    std::vector<int> data;
    for (int i = 0; i < 1000; ++i) {
        data.push_back(i * 10);
    }

    auto index = build_index(data);

    // Test find_upper_bound for exact values
    for (int i = 0; i < 9990; i += 100) {
        const int* result = index.find_upper_bound(i);
        if (result != data.data() + data.size()) {
            // Result should be the first element > i
            EXPECT_EQ(*result, i + 10)
                << "find_upper_bound(" << i << ") returned wrong value";
        }
    }
}
