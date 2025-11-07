// Tests for binary search segment finding
// These tests verify that the binary search logic in find_segment() works correctly
// for non-uniform data distributions and covers all edge cases.

#include "jazzy_index.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>

namespace {

// Helper to verify a value is found
template <typename IndexType, typename T>
bool is_found(const IndexType& index, const std::vector<T>& data, const T& value) {
    const T* result = index.find(value);
    return result != data.data() + data.size() && *result == value;
}

// Helper to verify a value is missing
template <typename IndexType, typename T>
bool is_missing(const IndexType& index, const std::vector<T>& data, const T& value) {
    const T* result = index.find(value);
    return result == data.data() + data.size();
}

template <typename T, std::size_t Segments = 256>
class BinarySearchSegmentTest : public ::testing::Test {
protected:
    jazzy::JazzyIndex<T, jazzy::to_segment_count<Segments>()> build_index(const std::vector<T>& data) {
        jazzy::JazzyIndex<T, jazzy::to_segment_count<Segments>()> index;
        index.build(data.data(), data.data() + data.size());
        return index;
    }
};

using IntBinarySearchTest = BinarySearchSegmentTest<int, 256>;
using DoubleBinarySearchTest = BinarySearchSegmentTest<double, 64>;

}  // namespace

// Test: Highly skewed data forces binary search path
TEST_F(IntBinarySearchTest, HighlySkewedDataForcesBinarySearch) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data;

    // First quarter: very dense (0-49)
    for (int i = 0; i < 50; ++i) {
        data.push_back(i);
    }

    // Last three quarters: very sparse (1000-5900)
    for (int i = 0; i < 50; ++i) {
        data.push_back(1000 + i * 100);
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos)
            << "Should log build phase";
        // Skewed data should NOT be detected as uniform
        EXPECT_EQ(build_log.find("Data is UNIFORM"), std::string::npos)
            << "Skewed data should NOT be detected as UNIFORM";
    }
    jazzy::clear_debug_log();
#endif

    // This data should NOT be uniform, forcing binary search

    // Test finds in dense region
    EXPECT_TRUE(is_found(index, data, 0));
    EXPECT_TRUE(is_found(index, data, 25));
    EXPECT_TRUE(is_found(index, data, 49));

    // Test finds in sparse region
    EXPECT_TRUE(is_found(index, data, 1000));
    EXPECT_TRUE(is_found(index, data, 3000));
    EXPECT_TRUE(is_found(index, data, 5900));

#ifdef JAZZY_DEBUG_LOGGING
    std::string find_log = jazzy::get_debug_log();
    if (!find_log.empty()) {
        // Non-uniform data should NOT use UNIFORM path
        EXPECT_EQ(find_log.find("UNIFORM path"), std::string::npos)
            << "Non-uniform data should use binary search, not UNIFORM path";
    }
    jazzy::clear_debug_log();
#endif

    // Test missing values
    EXPECT_TRUE(is_missing(index, data, 50));
    EXPECT_TRUE(is_missing(index, data, 500));
    EXPECT_TRUE(is_missing(index, data, 6000));
}

// Test: Exponential distribution (definitely non-uniform)
TEST_F(IntBinarySearchTest, ExponentialDistributionBinarySearch) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(static_cast<int>(std::pow(2.0, i * 0.2)));
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<128>()> index;
    index.build(data.data(), data.data() + data.size());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build: Building index for 100 elements"),
                 std::string::npos) << "Should log build with correct element count";
        // Exponential distribution should NOT be uniform
        EXPECT_EQ(build_log.find("Data is UNIFORM"), std::string::npos)
            << "Exponential data should NOT be uniform";
    }
    jazzy::clear_debug_log();
#endif

    // Exponential growth ensures non-uniformity
    EXPECT_TRUE(is_found(index, data, data[0]));
    EXPECT_TRUE(is_found(index, data, data[25]));
    EXPECT_TRUE(is_found(index, data, data[50]));
    EXPECT_TRUE(is_found(index, data, data[75]));
    EXPECT_TRUE(is_found(index, data, data[99]));
}

// Test: Edge case - search at last segment boundary
// This targets the untested edge case on line 568
TEST_F(IntBinarySearchTest, LastSegmentBoundaryEdgeCase) {
    std::vector<int> data;

    // Create data with distinct segments
    // Segment 1: 0-9
    for (int i = 0; i < 10; ++i) {
        data.push_back(i);
    }

    // Gap

    // Segment 2: 1000-1009
    for (int i = 1000; i < 1010; ++i) {
        data.push_back(i);
    }

    // Gap

    // Segment 3: 10000-10009
    for (int i = 10000; i < 10010; ++i) {
        data.push_back(i);
    }

    // Use very few segments to create specific boundary conditions
    jazzy::JazzyIndex<int, jazzy::SegmentCount::MINIMAL> index;
    index.build(data.data(), data.data() + data.size());

    // Search for value at the very end
    EXPECT_TRUE(is_found(index, data, 10009));
    EXPECT_TRUE(is_found(index, data, 10000));

    // Search for value just past the last valid value
    EXPECT_TRUE(is_missing(index, data, 10010));
    EXPECT_TRUE(is_missing(index, data, 10011));
}

// Test: Binary search with minimal segments
TEST_F(IntBinarySearchTest, MinimalSegmentsBinarySearch) {
    std::vector<int> data;

    // Create clustered data (non-uniform)
    for (int i = 0; i < 100; ++i) {
        data.push_back(i);
    }
    for (int i = 10000; i < 10100; ++i) {
        data.push_back(i);
    }

    jazzy::JazzyIndex<int, jazzy::SegmentCount::MINIMAL> index;  // Only 2 segments
    index.build(data.data(), data.data() + data.size());

    // With only 2 segments, binary search is simple but must work
    EXPECT_TRUE(is_found(index, data, 0));
    EXPECT_TRUE(is_found(index, data, 99));
    EXPECT_TRUE(is_found(index, data, 10000));
    EXPECT_TRUE(is_found(index, data, 10099));

    // Values in the gap
    EXPECT_TRUE(is_missing(index, data, 5000));
}

// Test: Binary search with pico segments
TEST_F(IntBinarySearchTest, PicoSegmentsBinarySearch) {
    std::vector<int> data;

    // Create 4 distinct clusters
    for (int i = 0; i < 25; ++i) data.push_back(i);
    for (int i = 1000; i < 1025; ++i) data.push_back(i);
    for (int i = 10000; i < 10025; ++i) data.push_back(i);
    for (int i = 100000; i < 100025; ++i) data.push_back(i);

    jazzy::JazzyIndex<int, jazzy::SegmentCount::PICO> index;  // 4 segments
    index.build(data.data(), data.data() + data.size());

    // Test one value from each cluster
    EXPECT_TRUE(is_found(index, data, 12));
    EXPECT_TRUE(is_found(index, data, 1012));
    EXPECT_TRUE(is_found(index, data, 10012));
    EXPECT_TRUE(is_found(index, data, 100012));
}

// Test: Binary search finds correct segment (left < mid case)
TEST_F(IntBinarySearchTest, BinarySearchLeftBranch) {
    std::vector<int> data;

    // Create data where we'll search in early segments
    for (int i = 0; i < 20; ++i) data.push_back(i);          // Early segment
    for (int i = 1000; i < 1020; ++i) data.push_back(i);    // Middle segment
    for (int i = 10000; i < 10020; ++i) data.push_back(i);  // Late segment

    jazzy::JazzyIndex<int, jazzy::SegmentCount::NANO> index;  // 8 segments
    index.build(data.data(), data.data() + data.size());

    // Search for values that trigger left branch (value < segments[mid].min_val)
    EXPECT_TRUE(is_found(index, data, 0));
    EXPECT_TRUE(is_found(index, data, 5));
    EXPECT_TRUE(is_found(index, data, 19));
}

// Test: Binary search finds correct segment (right > mid case)
TEST_F(IntBinarySearchTest, BinarySearchRightBranch) {
    std::vector<int> data;

    // Create data where we'll search in late segments
    for (int i = 0; i < 20; ++i) data.push_back(i);          // Early segment
    for (int i = 1000; i < 1020; ++i) data.push_back(i);    // Middle segment
    for (int i = 10000; i < 10020; ++i) data.push_back(i);  // Late segment

    jazzy::JazzyIndex<int, jazzy::SegmentCount::NANO> index;  // 8 segments
    index.build(data.data(), data.data() + data.size());

    // Search for values that trigger right branch (segments[mid].max_val < value)
    EXPECT_TRUE(is_found(index, data, 10000));
    EXPECT_TRUE(is_found(index, data, 10010));
    EXPECT_TRUE(is_found(index, data, 10019));
}

// Test: Binary search finds exact segment (value within segment)
TEST_F(IntBinarySearchTest, BinarySearchExactSegment) {
    std::vector<int> data;

    // Create well-separated clusters
    for (int i = 0; i < 10; ++i) data.push_back(i * 1000);

    jazzy::JazzyIndex<int, jazzy::SegmentCount::PICO> index;  // 4 segments
    index.build(data.data(), data.data() + data.size());

    // Each value should land in exact segment
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(is_found(index, data, i * 1000));
    }
}

// Test: Logarithmic distribution (non-uniform)
TEST_F(DoubleBinarySearchTest, LogarithmicDistributionBinarySearch) {
    std::vector<double> data;
    for (int i = 1; i <= 100; ++i) {
        data.push_back(std::log(static_cast<double>(i)));
    }

    auto index = build_index(data);

    // Logarithmic growth is non-uniform
    EXPECT_TRUE(is_found(index, data, data[0]));
    EXPECT_TRUE(is_found(index, data, data[50]));
    EXPECT_TRUE(is_found(index, data, data[99]));
}

// Test: Quadratic distribution (non-uniform)
TEST_F(IntBinarySearchTest, QuadraticDistributionBinarySearch) {
    std::vector<int> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(i * i);
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<128>()> index;
    index.build(data.data(), data.data() + data.size());

    // Quadratic growth is non-uniform
    EXPECT_TRUE(is_found(index, data, 0));
    EXPECT_TRUE(is_found(index, data, 100));   // 10^2
    EXPECT_TRUE(is_found(index, data, 2500));  // 50^2
    EXPECT_TRUE(is_found(index, data, 9801));  // 99^2
}

// Test: Clustered data with large gaps
TEST_F(IntBinarySearchTest, ClusteredDataWithLargeGaps) {
    std::vector<int> data;

    // Cluster 1
    for (int i = 0; i < 50; ++i) data.push_back(i);

    // Large gap (50-9999)

    // Cluster 2
    for (int i = 10000; i < 10050; ++i) data.push_back(i);

    // Large gap (10050-99999)

    // Cluster 3
    for (int i = 100000; i < 100050; ++i) data.push_back(i);

    jazzy::JazzyIndex<int, jazzy::to_segment_count<32>()> index;
    index.build(data.data(), data.data() + data.size());

    // Test values in each cluster
    EXPECT_TRUE(is_found(index, data, 25));
    EXPECT_TRUE(is_found(index, data, 10025));
    EXPECT_TRUE(is_found(index, data, 100025));

    // Test values in gaps
    EXPECT_TRUE(is_missing(index, data, 5000));
    EXPECT_TRUE(is_missing(index, data, 50000));
}

// Test: Reverse-skewed data (sparse at start, dense at end)
TEST_F(IntBinarySearchTest, ReverseSkewedData) {
    std::vector<int> data;

    // First half: very sparse (0, 1000, 2000, ...)
    for (int i = 0; i < 50; ++i) {
        data.push_back(i * 1000);
    }

    // Second half: very dense (50000-50049)
    for (int i = 50000; i < 50050; ++i) {
        data.push_back(i);
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());

    // Test sparse region
    EXPECT_TRUE(is_found(index, data, 0));
    EXPECT_TRUE(is_found(index, data, 25000));
    EXPECT_TRUE(is_found(index, data, 49000));

    // Test dense region
    EXPECT_TRUE(is_found(index, data, 50000));
    EXPECT_TRUE(is_found(index, data, 50025));
    EXPECT_TRUE(is_found(index, data, 50049));
}

// Test: Single segment with binary search fallback
TEST_F(IntBinarySearchTest, SingleSegmentFallback) {
    std::vector<int> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(i * i);  // Non-uniform
    }

    jazzy::JazzyIndex<int, jazzy::SegmentCount::SINGLE> index;
    index.build(data.data(), data.data() + data.size());

    // With single segment, still uses segment finding logic
    EXPECT_TRUE(is_found(index, data, 0));
    EXPECT_TRUE(is_found(index, data, 2500));
    EXPECT_TRUE(is_found(index, data, 9801));
}

// Test: Edge case with value exactly at segment boundary
TEST_F(IntBinarySearchTest, ValueAtSegmentBoundary) {
    std::vector<int> data;

    // Create 100 evenly-spaced but with non-uniform gaps
    for (int i = 0; i < 25; ++i) data.push_back(i);
    for (int i = 100; i < 125; ++i) data.push_back(i);
    for (int i = 200; i < 225; ++i) data.push_back(i);
    for (int i = 300; i < 325; ++i) data.push_back(i);

    jazzy::JazzyIndex<int, jazzy::SegmentCount::PICO> index;
    index.build(data.data(), data.data() + data.size());

    // Test values at segment boundaries
    EXPECT_TRUE(is_found(index, data, 0));    // First segment start
    EXPECT_TRUE(is_found(index, data, 24));   // First segment end
    EXPECT_TRUE(is_found(index, data, 100));  // Second segment start
    EXPECT_TRUE(is_found(index, data, 324));  // Last segment end
}

// Test: Many segments with non-uniform data
TEST_F(IntBinarySearchTest, ManySegmentsNonUniform) {
    std::vector<int> data;

    // Create power-law distribution
    for (int i = 1; i <= 1000; ++i) {
        data.push_back(static_cast<int>(std::pow(i, 1.5)));
    }

    jazzy::JazzyIndex<int, jazzy::SegmentCount::XLARGE> index;  // 512 segments
    index.build(data.data(), data.data() + data.size());

    // Even with many segments, binary search should work
    EXPECT_TRUE(is_found(index, data, data[0]));
    EXPECT_TRUE(is_found(index, data, data[250]));
    EXPECT_TRUE(is_found(index, data, data[500]));
    EXPECT_TRUE(is_found(index, data, data[750]));
    EXPECT_TRUE(is_found(index, data, data[999]));
}

// Test: Binary search with negative values
TEST_F(IntBinarySearchTest, BinarySearchWithNegatives) {
    std::vector<int> data;

    // Sparse negative values
    for (int i = -1000; i < -900; i += 10) data.push_back(i);

    // Dense around zero
    for (int i = -10; i <= 10; ++i) data.push_back(i);

    // Sparse positive values
    for (int i = 1000; i < 1100; i += 10) data.push_back(i);

    jazzy::JazzyIndex<int, jazzy::to_segment_count<32>()> index;
    index.build(data.data(), data.data() + data.size());

    // Test all regions
    EXPECT_TRUE(is_found(index, data, -1000));
    EXPECT_TRUE(is_found(index, data, 0));
    EXPECT_TRUE(is_found(index, data, 1000));
}

// Test: Stress test - very large gaps between segments
TEST_F(IntBinarySearchTest, VeryLargeGapsBetweenSegments) {
    std::vector<int> data;

    data.push_back(0);
    data.push_back(1000000);
    data.push_back(2000000);
    data.push_back(3000000);

    jazzy::JazzyIndex<int, jazzy::SegmentCount::PICO> index;
    index.build(data.data(), data.data() + data.size());

    EXPECT_TRUE(is_found(index, data, 0));
    EXPECT_TRUE(is_found(index, data, 1000000));
    EXPECT_TRUE(is_found(index, data, 2000000));
    EXPECT_TRUE(is_found(index, data, 3000000));

    EXPECT_TRUE(is_missing(index, data, 500000));
    EXPECT_TRUE(is_missing(index, data, 2500000));
}

// Test: Edge case targeting line 568 - value at absolute end of range
TEST_F(IntBinarySearchTest, AbsoluteEndOfRangeEdgeCase) {
    std::vector<int> data;

    // Create a very specific pattern with few segments
    // This creates a situation where the last value is at the extreme boundary
    for (int i = 0; i < 10; ++i) {
        data.push_back(i);
    }

    // Add one more value with a large gap
    data.push_back(1000);

    // Use minimal segments to create edge condition
    jazzy::JazzyIndex<int, jazzy::SegmentCount::MINIMAL> index;
    index.build(data.data(), data.data() + data.size());

    // Search for the very last value
    EXPECT_TRUE(is_found(index, data, 1000));

    // Search for values near the boundary
    EXPECT_TRUE(is_found(index, data, 9));
    EXPECT_TRUE(is_found(index, data, 0));
}

// Test: Minimal segments with specific search pattern
TEST_F(IntBinarySearchTest, MinimalSegmentsSearchBoundary) {
    std::vector<int> data;

    // Create exactly 3 elements to have very few segments
    data.push_back(1);
    data.push_back(2);
    data.push_back(1000000);

    jazzy::JazzyIndex<int, jazzy::SegmentCount::MINIMAL> index;
    index.build(data.data(), data.data() + data.size());

    // Search for each value
    EXPECT_TRUE(is_found(index, data, 1));
    EXPECT_TRUE(is_found(index, data, 2));
    EXPECT_TRUE(is_found(index, data, 1000000));

    // Missing values
    EXPECT_TRUE(is_missing(index, data, 500000));
    EXPECT_TRUE(is_missing(index, data, 0));
    EXPECT_TRUE(is_missing(index, data, 1000001));
}
