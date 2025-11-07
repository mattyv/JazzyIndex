// Tests for error recovery and exponential search fallback
// These tests verify the exponential search mechanism that kicks in
// when model predictions have errors.

#include "jazzy_index.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
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
class ErrorRecoveryTest : public ::testing::Test {
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

using IntErrorRecoveryTest = ErrorRecoveryTest<int, 256>;

}  // namespace

// Test: Data with high prediction error (non-linear distribution)
TEST_F(IntErrorRecoveryTest, HighPredictionErrorData) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    // Create data that will have poor linear predictions
    std::vector<int> data;
    for (int i = 0; i < 100; ++i) {
        // Exponential growth causes high prediction errors
        data.push_back(static_cast<int>(std::pow(1.2, i)));
    }
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos)
            << "Should log build phase";
        // Exponential data should have segments with error metrics
        if (build_log.find("analyze_segment") != std::string::npos) {
            // Should log error metrics for non-linear data
            EXPECT_TRUE(build_log.find("max_error") != std::string::npos ||
                       build_log.find("Selected") != std::string::npos)
                << "Should log error metrics or model selection";
        }
    }
    jazzy::clear_debug_log();
#endif

    // Despite poor predictions, all values should still be found
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_TRUE(is_found(index.find(data[i]), data, data[i]))
            << "Failed at index " << i << " with value " << data[i];
    }

#ifdef JAZZY_DEBUG_LOGGING
    std::string find_log = jazzy::get_debug_log();
    if (!find_log.empty()) {
        // With high prediction errors, exponential search may be used
        // Just verify that find operations are logged
        EXPECT_NE(find_log.find("find:"), std::string::npos)
            << "Should log find operations";
    }
#endif
}

// Test: Stepped data (sudden jumps)
TEST_F(IntErrorRecoveryTest, SteppedDataWithJumps) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data;

    // Region 1: dense (0-99)
    for (int i = 0; i < 100; ++i) {
        data.push_back(i);
    }

    // Region 2: sparse jump to 1000+
    for (int i = 0; i < 100; ++i) {
        data.push_back(1000 + i);
    }

    // Region 3: another jump to 10000+
    for (int i = 0; i < 100; ++i) {
        data.push_back(10000 + i);
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build: Building index for 300 elements"),
                 std::string::npos) << "Should log build with 300 elements";
        // Stepped data with jumps should NOT be uniform
        EXPECT_EQ(build_log.find("Data is UNIFORM"), std::string::npos)
            << "Stepped data with large jumps should NOT be uniform";
    }
    jazzy::clear_debug_log();
#endif

    // Test values from each region
    EXPECT_TRUE(is_found(index.find(50), data, 50));
    EXPECT_TRUE(is_found(index.find(1050), data, 1050));
    EXPECT_TRUE(is_found(index.find(10050), data, 10050));
}

// Test: Clustered data with gaps
TEST_F(IntErrorRecoveryTest, ClusteredDataWithGaps) {
    std::vector<int> data;

    // Cluster 1: 0-9
    for (int i = 0; i < 10; ++i) {
        data.push_back(i);
    }

    // Gap

    // Cluster 2: 100-109
    for (int i = 100; i < 110; ++i) {
        data.push_back(i);
    }

    // Gap

    // Cluster 3: 1000-1009
    for (int i = 1000; i < 1010; ++i) {
        data.push_back(i);
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<32>()> index;
    index.build(data.data(), data.data() + data.size());

    // All values in clusters should be findable
    for (int val : data) {
        EXPECT_TRUE(is_found(index.find(val), data, val))
            << "Failed to find " << val;
    }

    // Values in gaps should not be found
    EXPECT_FALSE(is_found(index.find(50), data, 50));
    EXPECT_FALSE(is_found(index.find(500), data, 500));
}

// Test: Zigzag pattern (alternating high/low)
TEST_F(IntErrorRecoveryTest, ZigzagPattern) {
    std::vector<int> data;
    for (int i = 0; i < 50; ++i) {
        data.push_back(i * 2);      // Even positions
        data.push_back(i * 2 + 1);  // Odd positions
    }
    std::sort(data.begin(), data.end());
    auto index = build_index(data);

    // All values should be found despite alternating pattern
    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(is_found(index.find(i), data, i));
    }
}

// Test: Power-law distribution
TEST_F(IntErrorRecoveryTest, PowerLawDistribution) {
    std::vector<int> data;
    for (int i = 1; i <= 100; ++i) {
        int value = static_cast<int>(std::pow(i, 3));  // Cubic growth
        data.push_back(value);
    }
    auto index = build_index(data);

    // Sample across the distribution
    EXPECT_TRUE(is_found(index.find(1), data, 1));        // 1^3
    EXPECT_TRUE(is_found(index.find(8), data, 8));        // 2^3
    EXPECT_TRUE(is_found(index.find(125), data, 125));    // 5^3
    EXPECT_TRUE(is_found(index.find(1000), data, 1000));  // 10^3
}

// Test: Bimodal distribution
TEST_F(IntErrorRecoveryTest, BimodalDistribution) {
    std::vector<int> data;

    // Mode 1: around 100
    for (int i = 0; i < 50; ++i) {
        data.push_back(100 + i % 10);
    }

    // Mode 2: around 1000
    for (int i = 0; i < 50; ++i) {
        data.push_back(1000 + i % 10);
    }

    std::sort(data.begin(), data.end());
    auto index = build_index(data);

    // Values from both modes should be found
    EXPECT_TRUE(is_found(index.find(100), data, 100));
    EXPECT_TRUE(is_found(index.find(105), data, 105));
    EXPECT_TRUE(is_found(index.find(1000), data, 1000));
    EXPECT_TRUE(is_found(index.find(1005), data, 1005));
}

// Test: Sparse data with many segments
TEST_F(IntErrorRecoveryTest, SparseDataManySegments) {
    std::vector<int> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(i * 100);  // 0, 100, 200, ..., 9900
    }

    // Use many segments for sparse data
    jazzy::JazzyIndex<int, jazzy::to_segment_count<512>()> index;
    index.build(data.data(), data.data() + data.size());

    for (int val : data) {
        EXPECT_TRUE(is_found(index.find(val), data, val));
    }
}

// Test: Random-like data
TEST_F(IntErrorRecoveryTest, RandomLikeData) {
    // Deterministic "random" pattern
    std::vector<int> data;
    std::uint64_t rng = 12345u;
    for (int i = 0; i < 200; ++i) {
        rng = (rng * 1103515245u + 12345u) & 0x7fffffffULL;
        data.push_back(static_cast<int>(rng % 10000u));
    }
    std::sort(data.begin(), data.end());
    data.erase(std::unique(data.begin(), data.end()), data.end());

    auto index = build_index(data);

    // All unique values should be findable
    for (int val : data) {
        EXPECT_TRUE(is_found(index.find(val), data, val))
            << "Failed to find " << val;
    }
}

// Test: Pathological case - all values at segment boundaries
TEST_F(IntErrorRecoveryTest, ValuesAtSegmentBoundaries) {
    std::vector<int> data(256);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::to_segment_count<256>()> index;  // 1 element per segment
    index.build(data.data(), data.data() + data.size());

    // Every value is at a boundary
    for (int i = 0; i < 256; ++i) {
        EXPECT_TRUE(is_found(index.find(i), data, i))
            << "Failed at boundary " << i;
    }
}

// Test: Worst case - inverse predictions
TEST_F(IntErrorRecoveryTest, InversePredictions) {
    // Create data where linear model would predict incorrectly
    std::vector<int> data;

    // First half: large values
    for (int i = 0; i < 50; ++i) {
        data.push_back(1000000 + i);
    }

    // Second half: small values
    for (int i = 0; i < 50; ++i) {
        data.push_back(i);
    }

    std::sort(data.begin(), data.end());
    jazzy::JazzyIndex<int, jazzy::to_segment_count<16>()> index;
    index.build(data.data(), data.data() + data.size());

    // Despite counter-intuitive distribution, all should be found
    EXPECT_TRUE(is_found(index.find(0), data, 0));
    EXPECT_TRUE(is_found(index.find(25), data, 25));
    EXPECT_TRUE(is_found(index.find(1000000), data, 1000000));
}

// Test: Exponential search radius expansion
TEST_F(IntErrorRecoveryTest, LargeErrorRequiringWideSearch) {
    // Create data that will force exponential search with large radius
    std::vector<int> data;

    // Most data at low values
    for (int i = 0; i < 90; ++i) {
        data.push_back(i);
    }

    // Few values at very high values
    for (int i = 0; i < 10; ++i) {
        data.push_back(10000 + i);
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<16>()> index;
    index.build(data.data(), data.data() + data.size());

    // The high values will have poor predictions
    EXPECT_TRUE(is_found(index.find(10005), data, 10005));
}

// Test: Duplicate-heavy data with poor spread
TEST_F(IntErrorRecoveryTest, DuplicateHeavyData) {
    std::vector<int> data;

    // Lots of duplicates (sorted)
    for (int i = 0; i < 100; ++i) {
        data.push_back(1);
    }
    for (int i = 0; i < 100; ++i) {
        data.push_back(2);
    }
    for (int i = 0; i < 100; ++i) {
        data.push_back(3);
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());

    EXPECT_TRUE(is_found(index.find(1), data, 1));
    EXPECT_TRUE(is_found(index.find(2), data, 2));
    EXPECT_TRUE(is_found(index.find(3), data, 3));
    EXPECT_FALSE(is_found(index.find(4), data, 4));
}

// Test: Segment with maximum error bound
TEST_F(IntErrorRecoveryTest, MaxErrorBoundSegment) {
    // Create data designed to hit max_error limits
    std::vector<int> data(1000);
    for (int i = 0; i < 1000; ++i) {
        // Non-linear: causes higher prediction errors
        data[i] = i * i / 100;
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<32>()> index;
    index.build(data.data(), data.data() + data.size());

    // All should still be found via exponential search
    for (size_t i = 0; i < data.size(); i += 50) {
        EXPECT_TRUE(is_found(index.find(data[i]), data, data[i]))
            << "Failed at index " << i;
    }
}
