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
template <typename T, std::size_t Segments>
bool is_found(const jazzy::JazzyIndex<T, Segments>& index,
              const std::vector<T>& data, const T& value) {
    const T* result = index.find(value);
    return result != data.data() + data.size() && *result == value;
}

template <typename T, std::size_t Segments = 256>
class ModelTest : public ::testing::Test {
protected:
    jazzy::JazzyIndex<T, Segments> build_index(const std::vector<T>& data) {
        jazzy::JazzyIndex<T, Segments> index;
        index.build(data.data(), data.data() + data.size());
        return index;
    }




};

using IntModelTest = ModelTest<int, 256>;
using DoubleModelTest = ModelTest<double, 256>;

}  // namespace

// Test: CONSTANT model - all values are identical
TEST_F(IntModelTest, ConstantModelAllIdenticalValues) {
    // All values are the same - should trigger CONSTANT model
    std::vector<int> data(100, 42);
    auto index = build_index(data);

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
    std::vector<int> data(1000, 99);
    auto index = build_index(data);

    EXPECT_TRUE(is_found(index, data, 99));
    EXPECT_EQ(index.size(), 1000);
}

// Test: LINEAR model - uniform distribution
TEST_F(IntModelTest, LinearModelUniformDistribution) {
    // Uniform data should be well-approximated by linear model
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);
    auto index = build_index(data);

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
    // Linear but sparse: values are multiples of 10
    std::vector<int> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(i * 10);
    }
    auto index = build_index(data);

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
    auto index = build_index(data);

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
    // Rapidly growing data that may benefit from quadratic model
    std::vector<int> data;
    for (int i = 0; i < 50; ++i) {
        data.push_back(static_cast<int>(std::pow(1.5, i)));
    }
    std::sort(data.begin(), data.end());
    data.erase(std::unique(data.begin(), data.end()), data.end());
    auto index = build_index(data);

    // Test finding first few values
    EXPECT_TRUE(is_found(index, data, data[0]));
    EXPECT_TRUE(is_found(index, data, data[10]));
    EXPECT_TRUE(is_found(index, data, data[data.size() - 1]));
}

// Test: QUADRATIC model - curved data with high error in linear fit
TEST_F(IntModelTest, QuadraticModelHighCurvature) {
    // Data with deliberate curvature
    std::vector<int> data;
    for (int i = 0; i < 200; ++i) {
        // Quadratic with offset: y = 2x^2 + x + 10
        int value = 2 * i * i + i + 10;
        data.push_back(value);
    }
    auto index = build_index(data);

    // Sample various points
    for (size_t i = 0; i < data.size(); i += 20) {
        EXPECT_TRUE(is_found(index, data, data[i]))
            << "Failed at index " << i << " with value " << data[i];
    }
}

// Test: Mixed models across segments
TEST_F(IntModelTest, MixedModelsAcrossSegments) {
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
    jazzy::JazzyIndex<int, 64> index;
    index.build(data.data(), data.data() + data.size());  // Use fewer segments to ensure variety

    // Test from constant region
    EXPECT_TRUE(is_found(index, data, 0));

    // Test from linear region
    EXPECT_TRUE(is_found(index, data, 250));

    // Test from quadratic region (high values)
    EXPECT_TRUE(is_found(index, data, data[data.size() - 1]));
}

// Test: Very small segments (may trigger DIRECT model concept)
TEST_F(IntModelTest, SmallSegmentsDenseData) {
    // Dense data with many segments - each segment is tiny
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);

    // Use large number of segments relative to data size
    jazzy::JazzyIndex<int, 512> index;
    index.build(data.data(), data.data() + data.size());

    // Even with many tiny segments, lookups should work
    for (int i = 0; i < 1000; i += 50) {
        EXPECT_TRUE(is_found(index, data, i));
    }
}

// Test: Logarithmic-like growth
TEST_F(IntModelTest, LogarithmicGrowthData) {
    std::vector<int> data;
    for (int i = 1; i <= 1000; ++i) {
        data.push_back(static_cast<int>(1000 * std::log(i)));
    }
    auto index = build_index(data);

    // Test various points
    EXPECT_TRUE(is_found(index, data, data[0]));
    EXPECT_TRUE(is_found(index, data, data[500]));
    EXPECT_TRUE(is_found(index, data, data[999]));
}

// Test: Nearly constant with small variation
TEST_F(IntModelTest, NearlyConstantWithVariation) {
    std::vector<int> data;
    for (int i = 0; i < 100; ++i) {
        // All values near 1000, but not identical
        data.push_back(1000 + (i % 3));  // Values: 1000, 1001, 1002
    }
    std::sort(data.begin(), data.end());
    auto index = build_index(data);

    EXPECT_TRUE(is_found(index, data, 1000));
    EXPECT_TRUE(is_found(index, data, 1001));
    EXPECT_TRUE(is_found(index, data, 1002));
    EXPECT_FALSE(is_found(index, data, 999));
    EXPECT_FALSE(is_found(index, data, 1003));
}

// Test: Piecewise linear data
TEST_F(IntModelTest, PiecewiseLinearData) {
    std::vector<int> data;

    // First segment: slow growth
    for (int i = 0; i < 100; ++i) {
        data.push_back(i);
    }

    // Second segment: rapid growth
    for (int i = 0; i < 100; ++i) {
        data.push_back(100 + i * 10);
    }

    jazzy::JazzyIndex<int, 128> index;
    index.build(data.data(), data.data() + data.size());

    // Test both regions
    EXPECT_TRUE(is_found(index, data, 50));
    EXPECT_TRUE(is_found(index, data, 500));
    EXPECT_TRUE(is_found(index, data, 1090));
}
