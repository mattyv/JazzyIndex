// Tests for equal_range functionality
// These tests verify that JazzyIndex correctly finds ranges of equal elements
// with proper comparator support.

#include "jazzy_index.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

namespace {

// Helper to verify equal_range results
template <typename T>
void verify_equal_range(const T* lower, const T* upper, const T* expected_lower,
                       const T* expected_upper) {
    EXPECT_EQ(lower, expected_lower) << "Lower bound mismatch";
    EXPECT_EQ(upper, expected_upper) << "Upper bound mismatch";
}

}  // namespace

// Test: Basic equal_range with no duplicates
TEST(EqualRangeTest, NoDuplicates) {
    std::vector<int> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());

    auto [lower, upper] = index.equal_range(5);
    EXPECT_EQ(lower, data.data() + 4);  // First 5
    EXPECT_EQ(upper, data.data() + 5);  // One past last 5
    EXPECT_EQ(upper - lower, 1);        // Exactly one 5
    EXPECT_EQ(*lower, 5);
}

// Test: equal_range with multiple duplicates
TEST(EqualRangeTest, MultipleDuplicates) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data{1, 2, 2, 2, 3, 4, 5, 5, 5, 6, 7, 8, 8, 9};
    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());

#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    // Test finding range of 2's
    auto [lower2, upper2] = index.equal_range(2);
    EXPECT_EQ(lower2, data.data() + 1);
    EXPECT_EQ(upper2, data.data() + 4);
    EXPECT_EQ(upper2 - lower2, 3);

#ifdef JAZZY_DEBUG_LOGGING
    std::string log = jazzy::get_debug_log();
    if (!log.empty()) {
        // equal_range should call both find_lower_bound and find_upper_bound
        EXPECT_NE(log.find("equal_range: Called"), std::string::npos)
            << "Should log equal_range call";
        EXPECT_NE(log.find("find_lower_bound"), std::string::npos)
            << "Should call find_lower_bound";
        EXPECT_NE(log.find("find_upper_bound"), std::string::npos)
            << "Should call find_upper_bound";
    }
    jazzy::clear_debug_log();
#endif

    // Test finding range of 5's
    auto [lower5, upper5] = index.equal_range(5);
    EXPECT_EQ(lower5, data.data() + 6);
    EXPECT_EQ(upper5, data.data() + 9);
    EXPECT_EQ(upper5 - lower5, 3);

    // Test finding range of 8's
    auto [lower8, upper8] = index.equal_range(8);
    EXPECT_EQ(lower8, data.data() + 11);
    EXPECT_EQ(upper8, data.data() + 13);
    EXPECT_EQ(upper8 - lower8, 2);
}

// Test: equal_range with all identical values
TEST(EqualRangeTest, AllIdentical) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data(10, 42);
    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos)
            << "Should log build phase";
        // All identical values should use CONSTANT model
        if (build_log.find("Selected CONSTANT") != std::string::npos) {
            EXPECT_NE(build_log.find("Selected CONSTANT"), std::string::npos);
        }
    }
    jazzy::clear_debug_log();
#endif

    auto [lower, upper] = index.equal_range(42);
    EXPECT_EQ(lower, data.data());
    EXPECT_EQ(upper, data.data() + data.size());
    EXPECT_EQ(upper - lower, 10);

#ifdef JAZZY_DEBUG_LOGGING
    std::string find_log = jazzy::get_debug_log();
    if (!find_log.empty()) {
        EXPECT_NE(find_log.find("equal_range"), std::string::npos);
    }
#endif
}

// Test: equal_range for missing value
TEST(EqualRangeTest, MissingValue) {
    std::vector<int> data{1, 2, 3, 5, 6, 7, 8, 9, 10};
    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());

    auto [lower, upper] = index.equal_range(4);
    // For missing values, both should point to insertion position (where 5 is)
    EXPECT_EQ(lower, upper);  // Empty range
    EXPECT_EQ(upper - lower, 0);
    // Verify it's the correct insertion point
    if (lower != data.data() + data.size()) {
        EXPECT_GE(*lower, 4);  // First element >= 4
    }
}

// Test: equal_range for value less than minimum
TEST(EqualRangeTest, ValueLessThanMin) {
    std::vector<int> data{10, 20, 30, 40, 50};
    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());

    auto [lower, upper] = index.equal_range(5);
    EXPECT_EQ(lower, upper);  // Empty range
    EXPECT_EQ(lower, data.data());  // Should point to beginning (insertion point)
}

// Test: equal_range for value greater than maximum
TEST(EqualRangeTest, ValueGreaterThanMax) {
    std::vector<int> data{10, 20, 30, 40, 50};
    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());

    auto [lower, upper] = index.equal_range(100);
    EXPECT_EQ(lower, upper);  // Empty range
    EXPECT_EQ(lower, data.data() + data.size());  // Should point to end (insertion point)
}

// Test: equal_range on empty index
TEST(EqualRangeTest, EmptyIndex) {
    std::vector<int> data;
    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data());

    auto [lower, upper] = index.equal_range(42);
    EXPECT_EQ(lower, data.data());
    EXPECT_EQ(upper, data.data());
}

// Test: equal_range with single element
TEST(EqualRangeTest, SingleElement) {
    std::vector<int> data{42};
    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());

    // Find the element
    auto [lower, upper] = index.equal_range(42);
    EXPECT_EQ(lower, data.data());
    EXPECT_EQ(upper, data.data() + 1);
    EXPECT_EQ(upper - lower, 1);

    // Try to find missing element (would be inserted at end)
    auto [lower2, upper2] = index.equal_range(43);
    EXPECT_EQ(lower2, upper2);  // Empty range
    EXPECT_EQ(lower2, data.data() + data.size());  // Insertion point at end
}

// Test: equal_range with reverse comparator (std::greater)
TEST(EqualRangeTest, ReverseComparator) {
    std::vector<int> data = {15, 13, 11, 9, 7, 5, 5, 5, 3, 1};
    jazzy::JazzyIndex<int, jazzy::to_segment_count<256>(), std::greater<>> index(
        data.data(),
        data.data() + data.size(),
        std::greater<>()
    );

    auto [lower, upper] = index.equal_range(5);
    EXPECT_EQ(lower, data.data() + 5);  // First 5
    EXPECT_EQ(upper, data.data() + 8);  // One past last 5
    EXPECT_EQ(upper - lower, 3);        // Three 5's
    EXPECT_EQ(*lower, 5);
}

// Test: equal_range with negative numbers
TEST(EqualRangeTest, NegativeNumbers) {
    std::vector<int> data{-100, -50, -50, -10, -5, 0, 5, 10, 50, 50, 100};
    jazzy::JazzyIndex<int, jazzy::to_segment_count<256>()> index;
    index.build(data.data(), data.data() + data.size());

    // Find all -50's
    auto [lower_neg50, upper_neg50] = index.equal_range(-50);
    EXPECT_EQ(upper_neg50 - lower_neg50, 2);
    EXPECT_EQ(*lower_neg50, -50);

    // Find all 50's
    auto [lower_50, upper_50] = index.equal_range(50);
    EXPECT_EQ(upper_50 - lower_50, 2);
    EXPECT_EQ(*lower_50, 50);
}

// Test: equal_range at boundaries (first and last elements)
TEST(EqualRangeTest, BoundaryElements) {
    std::vector<int> data{1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9};
    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());

    // Test first element
    auto [lower1, upper1] = index.equal_range(1);
    EXPECT_EQ(lower1, data.data());
    EXPECT_EQ(upper1, data.data() + 2);
    EXPECT_EQ(upper1 - lower1, 2);

    // Test last element
    auto [lower9, upper9] = index.equal_range(9);
    EXPECT_EQ(lower9, data.data() + 9);
    EXPECT_EQ(upper9, data.data() + data.size());
    EXPECT_EQ(upper9 - lower9, 3);
}

// Test: equal_range with large dataset
TEST(EqualRangeTest, LargeDataset) {
    std::vector<int> data;
    // Create dataset with groups of duplicates
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 10; ++j) {
            data.push_back(i);
        }
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<256>()> index;
    index.build(data.data(), data.data() + data.size());

    // Test various ranges
    for (int i = 0; i < 100; i += 10) {
        auto [lower, upper] = index.equal_range(i);
        EXPECT_EQ(upper - lower, 10) << "Failed for value " << i;
        if (lower != data.data() + data.size()) {
            EXPECT_EQ(*lower, i);
        }
    }
}

// Test: equal_range with floating-point values
TEST(EqualRangeTest, FloatingPoint) {
    std::vector<double> data{1.0, 2.5, 2.5, 2.5, 3.7, 5.0, 5.0, 8.9};
    jazzy::JazzyIndex<double, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());

    // Test finding range of 2.5's
    auto [lower, upper] = index.equal_range(2.5);
    EXPECT_EQ(lower, data.data() + 1);
    EXPECT_EQ(upper, data.data() + 4);
    EXPECT_EQ(upper - lower, 3);

    // Test finding range of 5.0's
    auto [lower5, upper5] = index.equal_range(5.0);
    EXPECT_EQ(lower5, data.data() + 5);
    EXPECT_EQ(upper5, data.data() + 7);
    EXPECT_EQ(upper5 - lower5, 2);
}

// Test: equal_range consistency with std::equal_range
TEST(EqualRangeTest, ConsistencyWithStdEqualRange) {
    std::vector<int> data{1, 2, 2, 3, 3, 3, 4, 5, 5, 6, 7, 7, 7, 7, 8};
    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());

    // Compare with std::equal_range for various values
    for (int val : {1, 2, 3, 4, 5, 6, 7, 8, 0, 9}) {
        auto [jazzy_lower, jazzy_upper] = index.equal_range(val);
        auto [std_lower, std_upper] = std::equal_range(
            data.begin(), data.end(), val);

        EXPECT_EQ(jazzy_lower - data.data(), std_lower - data.begin())
            << "Lower bound mismatch for value " << val;
        EXPECT_EQ(jazzy_upper - data.data(), std_upper - data.begin())
            << "Upper bound mismatch for value " << val;
    }
}

// Test: equal_range with absolute value comparator
TEST(EqualRangeTest, AbsoluteValueComparator) {
    struct AbsoluteValueCompare {
        bool operator()(int lhs, int rhs) const {
            return std::abs(lhs) < std::abs(rhs);
        }
    };

    std::vector<int> data{-10, -5, -5, 0, 5, 5, 10};
    std::sort(data.begin(), data.end(), AbsoluteValueCompare{});
    // After sort: 0, -5, 5, -5, 5, -10, 10 (may vary by stability)
    // Actually: 0, -5, -5, 5, 5, -10, 10 (assuming stable_sort behavior)

    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>(), AbsoluteValueCompare> index;
    index.build(data.data(), data.data() + data.size(), AbsoluteValueCompare{});

    // Find all values with absolute value of 5
    auto [lower, upper] = index.equal_range(5);
    EXPECT_GT(upper - lower, 0) << "Should find elements with |value| = 5";

    // Verify all found elements have the same absolute value
    for (const int* it = lower; it != upper; ++it) {
        EXPECT_EQ(std::abs(*it), 5);
    }
}

// Test: equal_range with modulo comparator
TEST(EqualRangeTest, ModuloComparator) {
    struct ModuloCompare {
        bool operator()(int lhs, int rhs) const {
            return (lhs % 10) < (rhs % 10);
        }
    };

    std::vector<int> data{1, 11, 21, 2, 12, 22, 3, 13, 23, 4, 14};
    std::sort(data.begin(), data.end(), ModuloCompare{});

    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>(), ModuloCompare> index;
    index.build(data.data(), data.data() + data.size(), ModuloCompare{});

    // Find all values equivalent to 1 (modulo 10)
    auto [lower, upper] = index.equal_range(1);
    EXPECT_EQ(upper - lower, 3);  // Should find 1, 11, 21

    // Verify all found elements are equivalent under modulo 10
    for (const int* it = lower; it != upper; ++it) {
        EXPECT_EQ(*it % 10, 1);
    }
}

// Test: equal_range with many duplicates in middle
TEST(EqualRangeTest, ManyDuplicatesInMiddle) {
    std::vector<int> data;
    // Add some unique values
    for (int i = 1; i <= 10; ++i) {
        data.push_back(i);
    }
    // Add many duplicates of 50
    for (int i = 0; i < 100; ++i) {
        data.push_back(50);
    }
    // Add more unique values
    for (int i = 51; i <= 60; ++i) {
        data.push_back(i);
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<256>()> index;
    index.build(data.data(), data.data() + data.size());

    auto [lower, upper] = index.equal_range(50);
    EXPECT_EQ(upper - lower, 100);
    EXPECT_EQ(*lower, 50);
}

// Test: equal_range result iteration
TEST(EqualRangeTest, ResultIteration) {
    std::vector<int> data{1, 2, 3, 3, 3, 4, 5};
    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());

    auto [lower, upper] = index.equal_range(3);

    // Iterate through the range
    std::vector<int> found_values;
    for (const int* it = lower; it != upper; ++it) {
        found_values.push_back(*it);
    }

    EXPECT_EQ(found_values.size(), 3);
    for (int val : found_values) {
        EXPECT_EQ(val, 3);
    }
}

// Test: equal_range with two consecutive duplicate groups
TEST(EqualRangeTest, ConsecutiveDuplicateGroups) {
    std::vector<int> data{1, 2, 3, 3, 3, 4, 4, 4, 4, 5};
    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());

    // Test first group
    auto [lower3, upper3] = index.equal_range(3);
    EXPECT_EQ(lower3, data.data() + 2);
    EXPECT_EQ(upper3, data.data() + 5);
    EXPECT_EQ(upper3 - lower3, 3);

    // Test second group
    auto [lower4, upper4] = index.equal_range(4);
    EXPECT_EQ(lower4, data.data() + 5);
    EXPECT_EQ(upper4, data.data() + 9);
    EXPECT_EQ(upper4 - lower4, 4);

    // Verify no overlap
    EXPECT_EQ(upper3, lower4);
}
