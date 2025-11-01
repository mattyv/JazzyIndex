// Tests for custom comparator functionality
// These tests verify that JazzyIndex works correctly with custom comparison
// functions, including reverse order and custom comparison logic.

#include "jazzy_index.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

namespace {

// Custom comparator for reverse order
template <typename T>
struct ReverseCompare {
    bool operator()(const T& lhs, const T& rhs) const {
        return lhs > rhs;
    }
};

// Custom comparator for absolute value comparison
struct AbsoluteValueCompare {
    bool operator()(int lhs, int rhs) const {
        return std::abs(lhs) < std::abs(rhs);
    }
};

}  // namespace

// Test: std::greater for reverse order
TEST(ComparatorTest, ReverseOrderWithGreater) {
    std::vector<int> data(100);
    std::iota(data.begin(), data.end(), 0);

    // Sort in descending order
    std::sort(data.begin(), data.end(), std::greater<int>{});

    // Build index with std::greater
    jazzy::JazzyIndex<int, 256, std::greater<int>> index;
    index.build(data.data(), data.data() + data.size(), std::greater<int>{});

    // Test finding elements
    const int* result = index.find(50);
    EXPECT_NE(result, data.data() + data.size());
    EXPECT_EQ(*result, 50);

    // Test min and max (reversed)
    EXPECT_NE(index.find(99), data.data() + data.size());  // Max value
    EXPECT_NE(index.find(0), data.data() + data.size());    // Min value
}

// Test: Custom reverse comparator
TEST(ComparatorTest, CustomReverseComparator) {
    std::vector<int> data{100, 90, 80, 70, 60, 50, 40, 30, 20, 10};

    // Build index with custom reverse comparator
    jazzy::JazzyIndex<int, 64, ReverseCompare<int>> index;
    index.build(data.data(), data.data() + data.size(), ReverseCompare<int>{});

    // Test lookups
    for (int val : data) {
        const int* result = index.find(val);
        EXPECT_NE(result, data.data() + data.size())
            << "Failed to find " << val;
        EXPECT_EQ(*result, val);
    }

    // Test missing values
    EXPECT_EQ(index.find(55), data.data() + data.size());
    EXPECT_EQ(index.find(105), data.data() + data.size());
}

// Test: Absolute value comparator
TEST(ComparatorTest, AbsoluteValueComparator) {
    std::vector<int> data{-100, -50, -10, -5, 0, 5, 10, 50, 100};

    // Sort by absolute value
    std::sort(data.begin(), data.end(), AbsoluteValueCompare{});
    // After sort: 0, -5, 5, -10, 10, -50, 50, -100, 100

    jazzy::JazzyIndex<int, 64, AbsoluteValueCompare> index;
    index.build(data.data(), data.data() + data.size(), AbsoluteValueCompare{});

    // Test finding elements - check equivalence under absolute value, not exact match
    // With custom comparators, lower_bound can return any equivalent element
    for (int val : data) {
        const int* result = index.find(val);
        EXPECT_NE(result, data.data() + data.size())
            << "Failed to find " << val;
        // Check that found value is equivalent under absolute value comparison
        EXPECT_EQ(std::abs(*result), std::abs(val))
            << "Found " << *result << " when searching for " << val;
    }
}

// Test: std::less (default) explicitly specified
TEST(ComparatorTest, ExplicitStdLess) {
    std::vector<int> data(100);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, 256, std::less<int>> index;
    index.build(data.data(), data.data() + data.size(), std::less<int>{});

    EXPECT_NE(index.find(0), data.data() + data.size());
    EXPECT_NE(index.find(50), data.data() + data.size());
    EXPECT_NE(index.find(99), data.data() + data.size());
}

// Test: Lambda comparator (if supported)
TEST(ComparatorTest, LambdaComparatorViaFunctionObject) {
    std::vector<int> data{10, 20, 30, 40, 50};

    // Using std::function to wrap lambda
    auto comp = std::function<bool(int, int)>([](int a, int b) { return a < b; });

    jazzy::JazzyIndex<int, 64, decltype(comp)> index;
    index.build(data.data(), data.data() + data.size(), comp);

    EXPECT_NE(index.find(30), data.data() + data.size());
}

// Test: Reverse order with duplicates
TEST(ComparatorTest, ReverseOrderWithDuplicates) {
    std::vector<int> data{50, 50, 50, 40, 40, 30, 30, 20, 10, 10};

    jazzy::JazzyIndex<int, 64, std::greater<int>> index;
    index.build(data.data(), data.data() + data.size(), std::greater<int>{});

    EXPECT_NE(index.find(50), data.data() + data.size());
    EXPECT_NE(index.find(40), data.data() + data.size());
    EXPECT_NE(index.find(30), data.data() + data.size());
    EXPECT_NE(index.find(20), data.data() + data.size());
    EXPECT_NE(index.find(10), data.data() + data.size());

    EXPECT_EQ(index.find(25), data.data() + data.size());
}

// Test: Floating-point with reverse order
TEST(ComparatorTest, FloatingPointReverseOrder) {
    std::vector<double> data{100.0, 75.5, 50.0, 25.5, 10.0, 5.0, 1.0};

    jazzy::JazzyIndex<double, 64, std::greater<double>> index;
    index.build(data.data(), data.data() + data.size(), std::greater<double>{});

    EXPECT_NE(index.find(100.0), data.data() + data.size());
    EXPECT_NE(index.find(50.0), data.data() + data.size());
    EXPECT_NE(index.find(1.0), data.data() + data.size());

    EXPECT_EQ(index.find(60.0), data.data() + data.size());
}

// Test: Case-insensitive string comparison (conceptual with int encoding)
TEST(ComparatorTest, ModuloComparator) {
    // Custom comparator that compares values modulo 10
    struct ModuloCompare {
        bool operator()(int lhs, int rhs) const {
            return (lhs % 10) < (rhs % 10);
        }
    };

    std::vector<int> data{1, 11, 21, 2, 12, 22, 3, 13, 23};
    std::sort(data.begin(), data.end(), ModuloCompare{});
    // After sort: 1, 11, 21, 2, 12, 22, 3, 13, 23

    jazzy::JazzyIndex<int, 64, ModuloCompare> index;
    index.build(data.data(), data.data() + data.size(), ModuloCompare{});

    // Search for values - check equivalence under modulo 10, not exact match
    for (int val : data) {
        const int* result = index.find(val);
        EXPECT_NE(result, data.data() + data.size())
            << "Failed to find " << val;
        // Check that found value is equivalent under modulo 10 comparison
        EXPECT_EQ(*result % 10, val % 10)
            << "Found " << *result << " when searching for " << val;
    }
}

// Test: Large dataset with reverse order
TEST(ComparatorTest, LargeDatasetReverseOrder) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);
    std::reverse(data.begin(), data.end());

    jazzy::JazzyIndex<int, 256, std::greater<int>> index;
    index.build(data.data(), data.data() + data.size(), std::greater<int>{});

    // Sample various points
    for (int i = 0; i < 1000; i += 100) {
        const int* result = index.find(i);
        EXPECT_NE(result, data.data() + data.size())
            << "Failed to find " << i;
        EXPECT_EQ(*result, i);
    }
}

// Test: std::greater<> (transparent comparator)
TEST(ComparatorTest, TransparentComparator) {
    std::vector<int> data{100, 90, 80, 70, 60, 50, 40, 30, 20, 10};

    jazzy::JazzyIndex<int, 64, std::greater<>> index;
    index.build(data.data(), data.data() + data.size(), std::greater<>{});

    EXPECT_NE(index.find(100), data.data() + data.size());
    EXPECT_NE(index.find(50), data.data() + data.size());
    EXPECT_NE(index.find(10), data.data() + data.size());
}

// Test: Rebuild with different comparator
TEST(ComparatorTest, RebuildWithDifferentComparator) {
    std::vector<int> data1{10, 20, 30, 40, 50};
    std::vector<int> data2{50, 40, 30, 20, 10};

    jazzy::JazzyIndex<int, 64, std::less<int>> index;

    // Build with ascending order
    index.build(data1.data(), data1.data() + data1.size(), std::less<int>{});
    EXPECT_NE(index.find(30), data1.data() + data1.size());

    // Rebuild with different comparator (note: this requires the index to support it)
    // In practice, you'd create a new index with different comparator template parameter
    jazzy::JazzyIndex<int, 64, std::greater<int>> index2;
    index2.build(data2.data(), data2.data() + data2.size(), std::greater<int>{});
    EXPECT_NE(index2.find(30), data2.data() + data2.size());
}

// Test: Negative numbers with reverse order
TEST(ComparatorTest, NegativeNumbersReverseOrder) {
    std::vector<int> data{0, -10, -20, -30, -40, -50};

    jazzy::JazzyIndex<int, 64, std::greater<int>> index;
    index.build(data.data(), data.data() + data.size(), std::greater<int>{});

    EXPECT_NE(index.find(0), data.data() + data.size());
    EXPECT_EQ(index.find(-25), data.data() + data.size()) << "Value -25 not in dataset";
    EXPECT_NE(index.find(-50), data.data() + data.size());
}

// Test: Edge case with single element and custom comparator
TEST(ComparatorTest, SingleElementCustomComparator) {
    std::vector<int> data{42};

    jazzy::JazzyIndex<int, 64, std::greater<int>> index;
    index.build(data.data(), data.data() + data.size(), std::greater<int>{});

    EXPECT_NE(index.find(42), data.data() + data.size());
    EXPECT_EQ(index.find(43), data.data() + data.size());
}

// Test: Empty dataset with custom comparator
TEST(ComparatorTest, EmptyDatasetCustomComparator) {
    std::vector<int> data;

    jazzy::JazzyIndex<int, 64, std::greater<int>> index;
    index.build(data.data(), data.data(), std::greater<int>{});

    EXPECT_EQ(index.find(42), data.data());
    EXPECT_EQ(index.size(), 0);
}
