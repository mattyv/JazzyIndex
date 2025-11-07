// Tests for floating-point types (float, double)
// These tests verify correct behavior with floating-point data types,
// including precision handling and special values.

#include "jazzy_index.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
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
class FloatingPointTest : public ::testing::Test {
protected:
    jazzy::JazzyIndex<T, jazzy::to_segment_count<Segments>()> build_index(const std::vector<T>& data) {
        jazzy::JazzyIndex<T, jazzy::to_segment_count<Segments>()> index;
        index.build(data.data(), data.data() + data.size());
        return index;
    }

    bool is_found(const T* result, const std::vector<T>& data, const T& value) {
        return result != data.data() + data.size() && *result == value;
    }

    // Helper for approximate equality (not needed for exact lookups)
    bool approx_equal(T a, T b, T epsilon = std::numeric_limits<T>::epsilon() * 100) {
        return std::abs(a - b) < epsilon;
    }
};

using FloatTest = FloatingPointTest<float, 256>;
using DoubleTest = FloatingPointTest<double, 256>;

}  // namespace

// Test: Basic double values
TEST_F(DoubleTest, BasicDoubleValues) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<double> data{0.0, 0.5, 1.0, 10.5, 100.0};
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    EXPECT_TRUE(is_found(index.find(0.0), data, 0.0));
    EXPECT_TRUE(is_found(index.find(0.5), data, 0.5));
    EXPECT_TRUE(is_found(index.find(1.0), data, 1.0));
    EXPECT_TRUE(is_found(index.find(10.5), data, 10.5));
    EXPECT_TRUE(is_found(index.find(100.0), data, 100.0));

    // Not in dataset
    EXPECT_FALSE(is_found(index.find(0.25), data, 0.25));
    EXPECT_FALSE(is_found(index.find(50.0), data, 50.0));
}

// Test: Basic float values
TEST_F(FloatTest, BasicFloatValues) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<float> data{0.0f, 1.5f, 3.14f, 10.0f, 99.99f};
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    EXPECT_TRUE(is_found(index.find(0.0f), data, 0.0f));
    EXPECT_TRUE(is_found(index.find(3.14f), data, 3.14f));
    EXPECT_TRUE(is_found(index.find(99.99f), data, 99.99f));

    EXPECT_FALSE(is_found(index.find(2.0f), data, 2.0f));
}

// Test: Double precision values
TEST_F(DoubleTest, DoublePrecisionValues) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<double> data;
    for (int i = 0; i < 1000; ++i) {
        data.push_back(i * 0.001);  // 0.000, 0.001, 0.002, ..., 0.999
    }
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    // Test various precision values
    EXPECT_TRUE(is_found(index.find(0.000), data, 0.000));
    EXPECT_TRUE(is_found(index.find(0.500), data, 0.500));
    EXPECT_TRUE(is_found(index.find(0.999), data, 0.999));

    // Test missing values
    EXPECT_FALSE(is_found(index.find(0.0005), data, 0.0005));
    EXPECT_FALSE(is_found(index.find(1.5), data, 1.5));
}

// Test: Negative floating-point values
TEST_F(DoubleTest, NegativeFloatingPointValues) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<double> data{-100.5, -50.25, -10.0, 0.0, 10.0, 50.25, 100.5};
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    EXPECT_TRUE(is_found(index.find(-100.5), data, -100.5));
    EXPECT_TRUE(is_found(index.find(-50.25), data, -50.25));
    EXPECT_TRUE(is_found(index.find(0.0), data, 0.0));
    EXPECT_TRUE(is_found(index.find(50.25), data, 50.25));
    EXPECT_TRUE(is_found(index.find(100.5), data, 100.5));

    EXPECT_FALSE(is_found(index.find(-75.0), data, -75.0));
}

// Test: Very small positive values
TEST_F(DoubleTest, VerySmallPositiveValues) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<double> data{1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0};
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    EXPECT_TRUE(is_found(index.find(1e-10), data, 1e-10));
    EXPECT_TRUE(is_found(index.find(1e-6), data, 1e-6));
    EXPECT_TRUE(is_found(index.find(1.0), data, 1.0));

    EXPECT_FALSE(is_found(index.find(1e-9), data, 1e-9));
}

// Test: Very large values
TEST_F(DoubleTest, VeryLargeValues) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<double> data{1e6, 1e9, 1e12, 1e15, 1e18};
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    EXPECT_TRUE(is_found(index.find(1e6), data, 1e6));
    EXPECT_TRUE(is_found(index.find(1e12), data, 1e12));
    EXPECT_TRUE(is_found(index.find(1e18), data, 1e18));

    EXPECT_FALSE(is_found(index.find(1e7), data, 1e7));
}

// Test: Mixed magnitude values
TEST_F(DoubleTest, MixedMagnitudeValues) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<double> data{
        -1e10, -1000.0, -1.0, -0.001,
        0.0,
        0.001, 1.0, 1000.0, 1e10
    };
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    for (auto val : data) {
        EXPECT_TRUE(is_found(index.find(val), data, val))
            << "Failed to find value " << val;
    }
}

// Test: Fractional sequence
TEST_F(DoubleTest, FractionalSequence) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<double> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(i + 0.5);  // 0.5, 1.5, 2.5, ..., 99.5
    }
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    EXPECT_TRUE(is_found(index.find(0.5), data, 0.5));
    EXPECT_TRUE(is_found(index.find(50.5), data, 50.5));
    EXPECT_TRUE(is_found(index.find(99.5), data, 99.5));

    // Integer values should not be found
    EXPECT_FALSE(is_found(index.find(0.0), data, 0.0));
    EXPECT_FALSE(is_found(index.find(50.0), data, 50.0));
}

// Test: Scientific notation values
TEST_F(DoubleTest, ScientificNotationValues) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<double> data{
        1.23e-5, 4.56e-3, 7.89e0, 1.11e2, 2.22e5
    };
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    EXPECT_TRUE(is_found(index.find(1.23e-5), data, 1.23e-5));
    EXPECT_TRUE(is_found(index.find(7.89e0), data, 7.89e0));
    EXPECT_TRUE(is_found(index.find(2.22e5), data, 2.22e5));
}

// Test: Uniform floating-point distribution
TEST_F(DoubleTest, UniformFloatingPointDistribution) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<double> data;
    for (int i = 0; i < 1000; ++i) {
        data.push_back(static_cast<double>(i) / 10.0);  // 0.0, 0.1, 0.2, ..., 99.9
    }
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    // Test regular intervals
    for (int i = 0; i < 1000; i += 100) {
        double val = static_cast<double>(i) / 10.0;
        EXPECT_TRUE(is_found(index.find(val), data, val))
            << "Failed at value " << val;
    }
}

// Test: Logarithmically spaced values
TEST_F(DoubleTest, LogarithmicallySpacedValues) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<double> data;
    for (int i = -10; i <= 10; ++i) {
        data.push_back(std::pow(10.0, static_cast<double>(i)));
    }
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    EXPECT_TRUE(is_found(index.find(1e-10), data, 1e-10));
    EXPECT_TRUE(is_found(index.find(1e0), data, 1e0));
    EXPECT_TRUE(is_found(index.find(1e10), data, 1e10));
}

// Test: Denormalized/subnormal numbers (very small)
TEST_F(DoubleTest, SubnormalNumbers) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<double> data{
        std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::denorm_min() * 2,
        std::numeric_limits<double>::denorm_min() * 10,
        std::numeric_limits<double>::min(),  // Smallest normal
        1.0
    };
    std::sort(data.begin(), data.end());
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    // Denormalized numbers should still be findable
    EXPECT_TRUE(is_found(index.find(std::numeric_limits<double>::denorm_min()),
                         data, std::numeric_limits<double>::denorm_min()));
    EXPECT_TRUE(is_found(index.find(std::numeric_limits<double>::min()),
                         data, std::numeric_limits<double>::min()));
}

// Test: Float type with similar operations
TEST_F(FloatTest, FloatPrecisionSequence) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<float> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(static_cast<float>(i) * 0.1f);
    }
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    // Test various points - use actual values from the array to avoid precision issues
    EXPECT_TRUE(is_found(index.find(data[0]), data, data[0]));   // 0.0
    EXPECT_TRUE(is_found(index.find(data[50]), data, data[50])); // 5.0
    EXPECT_TRUE(is_found(index.find(data[99]), data, data[99])); // 9.9
}

// Test: Quadratic floating-point growth
TEST_F(DoubleTest, QuadraticFloatingPointGrowth) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<double> data;
    for (int i = 0; i < 100; ++i) {
        double x = static_cast<double>(i) * 0.1;
        data.push_back(x * x);
    }
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    // Test some specific values - use actual values from the array
    EXPECT_TRUE(is_found(index.find(data[0]), data, data[0]));   // 0^2 = 0.0
    EXPECT_TRUE(is_found(index.find(data[1]), data, data[1]));   // 0.1^2
    EXPECT_TRUE(is_found(index.find(data[10]), data, data[10])); // 1.0^2
}

// Test: Fractional duplicates
TEST_F(DoubleTest, FractionalDuplicates) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<double> data;
    for (int i = 0; i < 50; ++i) {
        data.push_back(1.5);
        data.push_back(2.5);
        data.push_back(3.5);
    }
    std::sort(data.begin(), data.end());
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    EXPECT_TRUE(is_found(index.find(1.5), data, 1.5));
    EXPECT_TRUE(is_found(index.find(2.5), data, 2.5));
    EXPECT_TRUE(is_found(index.find(3.5), data, 3.5));

    EXPECT_FALSE(is_found(index.find(1.0), data, 1.0));
    EXPECT_FALSE(is_found(index.find(2.0), data, 2.0));
}

// Test: Zero and negative zero
TEST_F(DoubleTest, ZeroAndNegativeZero) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<double> data{-1.0, 0.0, 1.0};
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    EXPECT_TRUE(is_found(index.find(0.0), data, 0.0));
    EXPECT_TRUE(is_found(index.find(-0.0), data, -0.0));  // -0.0 == 0.0
}

// Test: Large dataset with floating-point values
TEST_F(DoubleTest, LargeFloatingPointDataset) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<double> data(10000);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<double>(i) * 0.01;
    }
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    // Sample various points
    for (size_t i = 0; i < data.size(); i += 1000) {
        EXPECT_TRUE(is_found(index.find(data[i]), data, data[i]))
            << "Failed at index " << i;
    }
}

// Test: Exponential growth
TEST_F(DoubleTest, ExponentialGrowth) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<double> data;
    for (int i = 0; i < 20; ++i) {
        data.push_back(std::exp(static_cast<double>(i) * 0.5));
    }
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    // Test first, middle, and last
    EXPECT_TRUE(is_found(index.find(data[0]), data, data[0]));
    EXPECT_TRUE(is_found(index.find(data[10]), data, data[10]));
    EXPECT_TRUE(is_found(index.find(data[19]), data, data[19]));
}

// Test: Interleaved positive and negative
TEST_F(DoubleTest, InterleavedPositiveNegative) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<double> data;
    for (int i = -50; i <= 50; ++i) {
        data.push_back(static_cast<double>(i) * 0.5);
    }
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    EXPECT_TRUE(is_found(index.find(-25.0), data, -25.0));
    EXPECT_TRUE(is_found(index.find(0.0), data, 0.0));
    EXPECT_TRUE(is_found(index.find(25.0), data, 25.0));
}
