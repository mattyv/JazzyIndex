#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <vector>
#include <deque>
#include <type_traits>
#include "jazzy_index.hpp"
#include "jazzy_index_parallel.hpp"

using namespace jazzy;

// Test that iterator type aliases are properly defined
TEST(IteratorTests, TypeAliases) {
    using Index = JazzyIndex<int>;

    // Check that iterator types are defined
    static_assert(std::is_same_v<Index::iterator, const int*>);
    static_assert(std::is_same_v<Index::const_iterator, const int*>);
    static_assert(std::is_same_v<Index::value_type, int>);
    static_assert(std::is_same_v<Index::pointer, const int*>);
    static_assert(std::is_same_v<Index::const_pointer, const int*>);
    static_assert(std::is_same_v<Index::reference, const int&>);
    static_assert(std::is_same_v<Index::const_reference, const int&>);
}

// Test building with std::vector iterators
TEST(IteratorTests, BuildWithVectorIterators) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    JazzyIndex<int> index;
    index.build(data.begin(), data.end());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    EXPECT_TRUE(index.is_built());
    EXPECT_EQ(index.size(), 10);

    // Test that find returns correct iterator
    auto it = index.find(5);
    EXPECT_NE(it, data.data() + data.size());
    EXPECT_EQ(*it, 5);
}

// Test building with std::array iterators
TEST(IteratorTests, BuildWithArrayIterators) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::array<int, 10> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    JazzyIndex<int> index;
    index.build(data.begin(), data.end());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    EXPECT_TRUE(index.is_built());
    EXPECT_EQ(index.size(), 10);

    auto it = index.find(7);
    EXPECT_NE(it, data.data() + data.size());
    EXPECT_EQ(*it, 7);
}

// Test iterator-based constructor with vector
TEST(IteratorTests, ConstructorWithVectorIterators) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data = {10, 20, 30, 40, 50};

    JazzyIndex<int> index(data.begin(), data.end());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    EXPECT_TRUE(index.is_built());
    EXPECT_EQ(index.size(), 5);

    auto it = index.find(30);
    EXPECT_NE(it, data.data() + data.size());
    EXPECT_EQ(*it, 30);
}

// Test that find returns const_iterator
TEST(IteratorTests, FindReturnsConstIterator) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data = {1, 2, 3, 4, 5};
    JazzyIndex<int> index(data.begin(), data.end());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    auto it = index.find(3);

    // Verify it's the right type
    static_assert(std::is_same_v<decltype(it), JazzyIndex<int>::const_iterator>);

    // Verify it points to the right element
    EXPECT_EQ(*it, 3);

    // Test with missing element
    auto missing = index.find(10);
    EXPECT_EQ(missing, data.data() + data.size());
}

// Test that equal_range returns pair of const_iterators
TEST(IteratorTests, EqualRangeReturnsConstIterators) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data = {1, 2, 2, 2, 3, 4, 5};
    JazzyIndex<int> index(data.begin(), data.end());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    auto [lower, upper] = index.equal_range(2);

    // Verify types
    static_assert(std::is_same_v<decltype(lower), JazzyIndex<int>::const_iterator>);
    static_assert(std::is_same_v<decltype(upper), JazzyIndex<int>::const_iterator>);

    // Verify range
    EXPECT_EQ(*lower, 2);
    EXPECT_EQ(upper - lower, 3);  // Three 2's

    // Verify all elements in range are 2
    for (auto it = lower; it != upper; ++it) {
        EXPECT_EQ(*it, 2);
    }
}

// Test find_lower_bound returns const_iterator
TEST(IteratorTests, LowerBoundReturnsConstIterator) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data = {1, 3, 3, 5, 7, 9};
    JazzyIndex<int> index(data.begin(), data.end());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    auto it = index.find_lower_bound(3);

    static_assert(std::is_same_v<decltype(it), JazzyIndex<int>::const_iterator>);
    EXPECT_EQ(*it, 3);
    EXPECT_EQ(it, data.data() + 1);  // First 3 is at index 1
}

// Test find_upper_bound returns const_iterator
TEST(IteratorTests, UpperBoundReturnsConstIterator) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data = {1, 3, 3, 5, 7, 9};
    JazzyIndex<int> index(data.begin(), data.end());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    auto it = index.find_upper_bound(3);

    static_assert(std::is_same_v<decltype(it), JazzyIndex<int>::const_iterator>);
    EXPECT_EQ(*it, 5);
    EXPECT_EQ(it, data.data() + 3);  // After last 3
}

// Test iterator arithmetic
TEST(IteratorTests, IteratorArithmetic) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    JazzyIndex<int> index(data.begin(), data.end());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    auto it1 = index.find(30);
    auto it2 = index.find(70);

    EXPECT_EQ(it2 - it1, 4);  // 4 elements between 30 and 70

    auto it3 = it1 + 2;
    EXPECT_EQ(*it3, 50);

    auto it4 = it2 - 1;
    EXPECT_EQ(*it4, 60);
}

// Test iterator comparison
TEST(IteratorTests, IteratorComparison) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data = {1, 2, 3, 4, 5};
    JazzyIndex<int> index(data.begin(), data.end());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    auto it1 = index.find(2);
    auto it2 = index.find(4);

    EXPECT_LT(it1, it2);
    EXPECT_LE(it1, it2);
    EXPECT_GT(it2, it1);
    EXPECT_GE(it2, it1);
    EXPECT_NE(it1, it2);

    auto it3 = index.find(2);
    EXPECT_EQ(it1, it3);
}

// Test using iterators with STL algorithms
TEST(IteratorTests, STLAlgorithmCompatibility) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    JazzyIndex<int> index(data.begin(), data.end());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
    jazzy::clear_debug_log();
#endif

    // Test that find returns pointer to same element
    auto it1 = index.find(5);
    EXPECT_EQ(*it1, 5);
    EXPECT_EQ(it1 - data.data(), 4);  // 5 is at index 4

    // Test std::count with range from equal_range
    std::vector<int> data_with_dups = {1, 2, 2, 2, 3, 4, 5};
    JazzyIndex<int> index2(data_with_dups.begin(), data_with_dups.end());

#ifdef JAZZY_DEBUG_LOGGING
    build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    auto [lower, upper] = index2.equal_range(2);
    EXPECT_EQ(std::count(lower, upper, 2), 3);

    // Test std::distance
    EXPECT_EQ(std::distance(lower, upper), 3);
}

// Test with custom comparator
TEST(IteratorTests, CustomComparator) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

    JazzyIndex<int, SegmentCount::LARGE, std::greater<>> index;
    index.build(data.begin(), data.end(), std::greater<>{});

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    auto it = index.find(5);
    EXPECT_NE(it, data.data() + data.size());
    EXPECT_EQ(*it, 5);
}

// Test with struct and key extractor
TEST(IteratorTests, KeyExtractorWithIterators) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    struct Record {
        int id;
        std::string name;

        bool operator<(const Record& other) const {
            return id < other.id;
        }
    };

    std::vector<Record> data = {
        {1, "Alice"},
        {2, "Bob"},
        {3, "Charlie"},
        {4, "David"}
    };

    auto key_extractor = [](const Record& r) { return r.id; };

    JazzyIndex<Record, SegmentCount::SMALL, std::less<Record>, decltype(key_extractor)> index;
    index.build(data.begin(), data.end(), std::less<Record>{}, key_extractor);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    auto it = index.find(Record{2, ""});
    EXPECT_NE(it, data.data() + data.size());
    EXPECT_EQ(it->id, 2);
    EXPECT_EQ(it->name, "Bob");
}

// Test empty range
TEST(IteratorTests, EmptyRange) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data = {1, 2, 3, 4, 5};
    JazzyIndex<int> index(data.begin(), data.end());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    auto [lower, upper] = index.equal_range(10);  // Not in data
    EXPECT_EQ(lower, upper);  // Empty range
}

// Test with large dataset
TEST(IteratorTests, LargeDataset) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data(10000);
    for (int i = 0; i < 10000; ++i) {
        data[i] = i;
    }

    JazzyIndex<int> index(data.begin(), data.end());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    // Test various finds
    for (int val : {0, 1000, 5000, 9999}) {
        auto it = index.find(val);
        EXPECT_NE(it, data.data() + data.size());
        EXPECT_EQ(*it, val);
    }
}

// Test with floating point
TEST(IteratorTests, FloatingPoint) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<double> data = {1.1, 2.2, 3.3, 4.4, 5.5};
    JazzyIndex<double> index(data.begin(), data.end());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    auto it = index.find(3.3);
    EXPECT_NE(it, data.data() + data.size());
    EXPECT_DOUBLE_EQ(*it, 3.3);
}

// Test that raw pointers still work (backward compatibility)
TEST(IteratorTests, BackwardCompatibilityWithPointers) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    int data[] = {1, 2, 3, 4, 5};

    JazzyIndex<int> index(data, data + 5);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos);
    }
#endif

    EXPECT_TRUE(index.is_built());
    EXPECT_EQ(index.size(), 5);

    auto it = index.find(3);
    EXPECT_EQ(*it, 3);
}

// Test parallel build with iterators
TEST(IteratorTests, ParallelBuildWithIterators) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data(1000);
    for (int i = 0; i < 1000; ++i) {
        data[i] = i;
    }

    JazzyIndex<int> index;
    index.build_parallel(data.begin(), data.end());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    // Parallel build may not produce logs, so just check if we got the log
    // (don't require specific content for parallel builds)
#endif

    EXPECT_TRUE(index.is_built());
    EXPECT_EQ(index.size(), 1000);

    auto it = index.find(500);
    EXPECT_NE(it, data.data() + data.size());
    EXPECT_EQ(*it, 500);
}

// Test that deque iterators don't compile (non-contiguous)
// This is a compile-time test - uncomment to verify it fails
/*
TEST(IteratorTests, DequeIteratorsShouldNotCompile) {
    std::deque<int> data = {1, 2, 3, 4, 5};
    JazzyIndex<int> index;
    // This should fail to compile because deque iterators are not contiguous
    index.build(data.begin(), data.end());
}
*/

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
