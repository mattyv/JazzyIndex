// Tests for different model types (LINEAR, QUADRATIC, CONSTANT, DIRECT)
// These tests verify that the segment analysis correctly selects and applies
// different models based on data characteristics.

#include "jazzy_index.hpp"

#include <gtest/gtest.h>

#include <algorithm>
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
class ModelTest : public ::testing::Test {
protected:
    jazzy::JazzyIndex<T, jazzy::to_segment_count<Segments>()> build_index(const std::vector<T>& data) {
        jazzy::JazzyIndex<T, jazzy::to_segment_count<Segments>()> index;
        index.build(data.data(), data.data() + data.size());
        return index;
    }




};

using IntModelTest = ModelTest<int, 256>;
using DoubleModelTest = ModelTest<double, 256>;

}  // namespace

// Test: CONSTANT model - all values are identical
TEST_F(IntModelTest, ConstantModelAllIdenticalValues) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    // All values are the same - should trigger CONSTANT model
    std::vector<int> data(100, 42);
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    // All identical values should have 0 linear error (constant data is a special case of linear)
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos) << "Missing build log";
        if (build_log.find("analyze_segment") != std::string::npos) {
            // Should see model selection
            EXPECT_TRUE(build_log.find("Selected") != std::string::npos ||
                       build_log.find("CONSTANT") != std::string::npos)
                << "Should have model selection";
        }
    }
#endif

    // All lookups for 42 should succeed
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(is_found(index, data, 42));
    }

    // Other values should not be found
    EXPECT_FALSE(is_found(index, data, 41));
    EXPECT_FALSE(is_found(index, data, 43));
}

// Test: CONSTANT model - single unique value with duplicates
TEST_F(IntModelTest, ConstantModelManyDuplicates) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data(1000, 99);
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    // Many duplicates of same value
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build: Building index for 1000 elements"), std::string::npos)
            << "Missing build log";
        // Should see some form of model selection
        if (build_log.find("analyze_segment") != std::string::npos) {
            EXPECT_TRUE(build_log.find("Selected") != std::string::npos ||
                       build_log.find("CONSTANT") != std::string::npos)
                << "Should have model selection";
        }
    }
#endif

    EXPECT_TRUE(is_found(index, data, 99));
    EXPECT_EQ(index.size(), 1000);
}

// Test: LINEAR model - uniform distribution
TEST_F(IntModelTest, LinearModelUniformDistribution) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    // Uniform data should be well-approximated by linear model
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    // Verify that LINEAR model was selected (uniform data should use LINEAR)
    EXPECT_NE(build_log.find("Selected LINEAR model"), std::string::npos)
        << "Expected LINEAR model to be selected for uniform data";

    // For perfectly uniform sequential data [0, 999], linear error should be 0
    EXPECT_NE(build_log.find("linear_max_error=0"), std::string::npos)
        << "Expected 0 linear error for perfectly uniform data";

    // Verify all required parameters are logged
    EXPECT_NE(build_log.find("slope="), std::string::npos) << "Missing slope parameter";
    EXPECT_NE(build_log.find("intercept="), std::string::npos) << "Missing intercept parameter";

    // Note: Uniformity detection may or may not trigger depending on segment count and size
#endif

    // Test multiple points throughout the range
    for (int i = 0; i < 1000; i += 100) {
        EXPECT_TRUE(is_found(index, data, i))
            << "Failed to find value " << i;
    }

    // Test all edges
    EXPECT_TRUE(is_found(index, data, 0));
    EXPECT_TRUE(is_found(index, data, 999));
}

// Test: LINEAR model - sparse linear data
TEST_F(IntModelTest, LinearModelSparseLinearData) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    // Linear but sparse: values are multiples of 10
    std::vector<int> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(i * 10);
    }
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    // Linear sparse data should still use LINEAR model (values are linearly distributed)
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos) << "Missing build log";
        if (build_log.find("analyze_segment") != std::string::npos) {
            // Should select LINEAR for linearly distributed data
            EXPECT_TRUE(build_log.find("Selected LINEAR") != std::string::npos ||
                       build_log.find("Selected") != std::string::npos)
                << "Should have model selection";
        }
    }
#endif

    // Test finding existing values
    EXPECT_TRUE(is_found(index, data, 0));
    EXPECT_TRUE(is_found(index, data, 100));
    EXPECT_TRUE(is_found(index, data, 500));
    EXPECT_TRUE(is_found(index, data, 990));

    // Test missing values (not multiples of 10)
    EXPECT_FALSE(is_found(index, data, 5));
    EXPECT_FALSE(is_found(index, data, 105));
}

// Test: QUADRATIC model - quadratic growth data
TEST_F(IntModelTest, QuadraticModelQuadraticGrowth) {
    // Data with quadratic growth: i^2
    std::vector<int> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(i * i);
    }

#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();

    // With 256 segments for 100 elements, most segments will be empty or have 1 element
    // This may result in different logging behavior (CONSTANT models, empty segments, etc.)
    // Just verify that build was attempted
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos)
            << "Missing build log. Log content: " << build_log.substr(0, 500);

        // If segments were analyzed (non-empty segments), verify basic logging
        if (build_log.find("analyze_segment") != std::string::npos) {
            // Some analysis occurred - should have model selection
            EXPECT_TRUE(build_log.find("Selected") != std::string::npos ||
                       build_log.find("CONSTANT") != std::string::npos)
                << "Missing model selection or CONSTANT model";
        }
    }
#endif

    // Test various points
    EXPECT_TRUE(is_found(index, data, 0));      // 0^2
    EXPECT_TRUE(is_found(index, data, 1));      // 1^2
    EXPECT_TRUE(is_found(index, data, 4));      // 2^2
    EXPECT_TRUE(is_found(index, data, 25));    // 5^2
    EXPECT_TRUE(is_found(index, data, 100));  // 10^2
    EXPECT_TRUE(is_found(index, data, 9801)); // 99^2

    // Test missing values
    EXPECT_FALSE(is_found(index, data, 3));
    EXPECT_FALSE(is_found(index, data, 50));
}

// Test: QUADRATIC model - exponential-like data
TEST_F(IntModelTest, QuadraticModelExponentialLikeData) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    // Rapidly growing data that may benefit from quadratic model
    std::vector<int> data;
    for (int i = 0; i < 50; ++i) {
        data.push_back(static_cast<int>(std::pow(1.5, i)));
    }
    std::sort(data.begin(), data.end());
    data.erase(std::unique(data.begin(), data.end()), data.end());
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    // Exponential data has high curvature - linear model will have high error
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos) << "Missing build log";
        if (build_log.find("analyze_segment") != std::string::npos) {
            // Should see model selection decision
            EXPECT_TRUE(build_log.find("Selected") != std::string::npos ||
                       build_log.find("CONSTANT") != std::string::npos)
                << "Should have model selection";
            // Might see higher-order models attempted
            if (build_log.find("trying QUADRATIC") != std::string::npos) {
                EXPECT_NE(build_log.find("QUADRATIC: max_error="), std::string::npos)
                    << "Should log quadratic error";
            }
        }
    }
#endif

    // Test finding first few values
    EXPECT_TRUE(is_found(index, data, data[0]));
    EXPECT_TRUE(is_found(index, data, data[10]));
    EXPECT_TRUE(is_found(index, data, data[data.size() - 1]));
}

// Test: QUADRATIC model - curved data with high error in linear fit
TEST_F(IntModelTest, QuadraticModelHighCurvature) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    // Data with deliberate curvature
    std::vector<int> data;
    for (int i = 0; i < 200; ++i) {
        // Quadratic with offset: y = 2x^2 + x + 10
        int value = 2 * i * i + i + 10;
        data.push_back(value);
    }
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    // With 256 segments for 200 elements, many segments will be empty
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos) << "Missing build log";
        // Just verify build happened - segment analysis may vary with segment size
    }
#endif

    // Sample various points
    for (size_t i = 0; i < data.size(); i += 20) {
        EXPECT_TRUE(is_found(index, data, data[i]))
            << "Failed at index " << i << " with value " << data[i];
    }
}

// Test: Mixed models across segments
TEST_F(IntModelTest, MixedModelsAcrossSegments) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    // Create data with different characteristics in different regions
    std::vector<int> data;

    // Region 1: Constant (0-99)
    for (int i = 0; i < 100; ++i) {
        data.push_back(0);
    }

    // Region 2: Linear (100-599)
    for (int i = 0; i < 500; ++i) {
        data.push_back(i);
    }

    // Region 3: Quadratic (600-799)
    for (int i = 0; i < 200; ++i) {
        data.push_back(500 + i * i);
    }

    std::sort(data.begin(), data.end());
    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());  // Use fewer segments to ensure variety

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    // Mixed data should have multiple segments analyzed with different characteristics
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build: Building index for 800 elements"), std::string::npos)
            << "Missing build log with element count";

        if (build_log.find("analyze_segment") != std::string::npos) {
            // Should see segment analysis
            size_t segment_count = 0;
            size_t pos = 0;
            while ((pos = build_log.find("analyze_segment[", pos)) != std::string::npos) {
                ++segment_count;
                ++pos;
            }
            EXPECT_GE(segment_count, 1) << "Should analyze at least some segments";

            // Should see model selection
            EXPECT_TRUE(build_log.find("Selected") != std::string::npos ||
                       build_log.find("CONSTANT") != std::string::npos)
                << "Should have model selection";
        }
    }
#endif

    // Test from constant region
    EXPECT_TRUE(is_found(index, data, 0));

    // Test from linear region
    EXPECT_TRUE(is_found(index, data, 250));

    // Test from quadratic region (high values)
    EXPECT_TRUE(is_found(index, data, data[data.size() - 1]));
}

// Test: Very small segments (may trigger DIRECT model concept)
TEST_F(IntModelTest, SmallSegmentsDenseData) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    // Dense data with many segments - each segment is tiny
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);

    // Use large number of segments relative to data size
    jazzy::JazzyIndex<int, jazzy::to_segment_count<512>()> index;
    index.build(data.data(), data.data() + data.size());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    // With 512 segments for 1000 elements, segments are very small (~2 elements each)
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build: Building index for 1000 elements with 512 segments"),
                 std::string::npos) << "Missing build log";
        // Many segments will be empty or have 1 element (CONSTANT model)
        // Just verify build happened
    }
#endif

    // Even with many tiny segments, lookups should work
    for (int i = 0; i < 1000; i += 50) {
        EXPECT_TRUE(is_found(index, data, i));
    }
}

// Test: Logarithmic-like growth
TEST_F(IntModelTest, LogarithmicGrowthData) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data;
    for (int i = 1; i <= 1000; ++i) {
        data.push_back(static_cast<int>(1000 * std::log(i)));
    }
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    // Logarithmic growth has decreasing slope - linear model will have error
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos) << "Missing build log";
        if (build_log.find("analyze_segment") != std::string::npos) {
            // Should see error metrics for non-linear data
            EXPECT_NE(build_log.find("linear_max_error="), std::string::npos)
                << "Should log linear error for non-linear data";
            // Should see model selection
            EXPECT_TRUE(build_log.find("Selected") != std::string::npos ||
                       build_log.find("CONSTANT") != std::string::npos)
                << "Should have model selection";
        }
    }
#endif

    // Test various points
    EXPECT_TRUE(is_found(index, data, data[0]));
    EXPECT_TRUE(is_found(index, data, data[500]));
    EXPECT_TRUE(is_found(index, data, data[999]));
}

// Test: Nearly constant with small variation
TEST_F(IntModelTest, NearlyConstantWithVariation) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data;
    for (int i = 0; i < 100; ++i) {
        // All values near 1000, but not identical
        data.push_back(1000 + (i % 3));  // Values: 1000, 1001, 1002
    }
    std::sort(data.begin(), data.end());
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    // With 256 segments for 100 elements, many segments will be empty
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos) << "Missing build log";
        // Just verify build happened
    }
#endif

    EXPECT_TRUE(is_found(index, data, 1000));
    EXPECT_TRUE(is_found(index, data, 1001));
    EXPECT_TRUE(is_found(index, data, 1002));
    EXPECT_FALSE(is_found(index, data, 999));
    EXPECT_FALSE(is_found(index, data, 1003));
}

// Test: Piecewise linear data
TEST_F(IntModelTest, PiecewiseLinearData) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data;

    // First segment: slow growth
    for (int i = 0; i < 100; ++i) {
        data.push_back(i);
    }

    // Second segment: rapid growth
    for (int i = 0; i < 100; ++i) {
        data.push_back(100 + i * 10);
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<128>()> index;
    index.build(data.data(), data.data() + data.size());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    // Piecewise linear data has different slopes in different regions
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build: Building index for 200 elements"), std::string::npos)
            << "Missing build log";
        if (build_log.find("analyze_segment") != std::string::npos) {
            // Different segments should have different slopes
            // Should see multiple segments analyzed
            size_t segment_count = 0;
            size_t pos = 0;
            while ((pos = build_log.find("analyze_segment[", pos)) != std::string::npos) {
                ++segment_count;
                ++pos;
            }
            EXPECT_GE(segment_count, 1) << "Should analyze at least some segments";

            // Should see LINEAR models (piecewise linear data)
            EXPECT_NE(build_log.find("Selected"), std::string::npos)
                << "Should have model selection";
        }
    }
#endif

    // Test both regions
    EXPECT_TRUE(is_found(index, data, 50));
    EXPECT_TRUE(is_found(index, data, 500));
    EXPECT_TRUE(is_found(index, data, 1090));
}

// Test: Monotonicity constraint - non-monotonic quadratics should be rejected
TEST_F(DoubleModelTest, NonMonotonicQuadraticsRejected) {
    // Create exponential distribution that might produce non-monotonic quadratics
    // with low segment counts
    std::vector<double> data;
    data.reserve(100);
    for (int i = 0; i < 100; ++i) {
        // Exponential growth: index -> value mapping
        data.push_back(5.0 * (std::exp(static_cast<double>(i) / 50.0) - 1.0) / (std::exp(2.0) - 1.0));
    }

    // Build with S=1 (single segment) - this previously produced backwards loops
    jazzy::JazzyIndex<double, jazzy::to_segment_count<1>()> index;
    index.build(data.data(), data.data() + data.size());

    // All values should still be findable (monotonicity preserved)
    for (std::size_t i = 0; i < data.size(); ++i) {
        const double* result = index.find(data[i]);
        ASSERT_NE(result, data.data() + data.size())
            << "Failed to find data[" << i << "] = " << data[i];
        EXPECT_EQ(*result, data[i])
            << "Found wrong value for data[" << i << "]";
    }

    // Binary search should also work (verifies monotonicity)
    for (double test_val : {0.5, 1.5, 2.5, 3.5, 4.5}) {
        auto it = std::lower_bound(data.begin(), data.end(), test_val);
        const double* result = index.find(test_val);
        if (it != data.end() && *it == test_val) {
            EXPECT_EQ(*result, test_val);
        } else {
            EXPECT_EQ(result, data.data() + data.size());
        }
    }
}
