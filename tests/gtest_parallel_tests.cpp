// Tests for parallel build functionality
// Verifies that parallel builds produce identical results to single-threaded builds

#include "jazzy_index.hpp"
#include "jazzy_index_parallel.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace {

// Helper to compare two indexes for equality
template <typename T, jazzy::SegmentCount Segments>
bool indexes_equal(const jazzy::JazzyIndex<T, Segments>& idx1,
                   const jazzy::JazzyIndex<T, Segments>& idx2,
                   const std::vector<T>& data) {
    if (idx1.size() != idx2.size()) return false;
    if (idx1.num_segments() != idx2.num_segments()) return false;

    // Test every value in the dataset
    for (const auto& val : data) {
        const T* result1 = idx1.find(val);
        const T* result2 = idx2.find(val);

        // Both should return same result (found or not found)
        if ((result1 == data.data() + data.size()) != (result2 == data.data() + data.size())) {
            return false;
        }

        // If found, values should match
        if (result1 != data.data() + data.size()) {
            if (*result1 != *result2) {
                return false;
            }
        }
    }

    return true;
}

// Test fixture for parallel build tests
template <typename T, std::size_t Segments = 256>
class ParallelBuildTest : public ::testing::Test {
protected:
    using IndexType = jazzy::JazzyIndex<T, jazzy::to_segment_count<Segments>()>;

    IndexType build_single_threaded(const std::vector<T>& data) {
        IndexType index;
        index.build(data.data(), data.data() + data.size());
        return index;
    }

    IndexType build_parallel(const std::vector<T>& data) {
        IndexType index;
        index.build_parallel(data.data(), data.data() + data.size());
        return index;
    }
};

using IntParallel = ParallelBuildTest<int, 256>;
using LargeIntParallel = ParallelBuildTest<int, 512>;

}  // namespace

// Test: Empty dataset
TEST_F(IntParallel, EmptyDataset) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data;

    auto idx_single = build_single_threaded(data);
    auto idx_parallel = build_parallel(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    EXPECT_EQ(idx_single.size(), 0);
    EXPECT_EQ(idx_parallel.size(), 0);
    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}

// Test: Single element
TEST_F(IntParallel, SingleElement) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data{42};

    auto idx_single = build_single_threaded(data);
    auto idx_parallel = build_parallel(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));

    const int* result = idx_parallel.find(42);
    ASSERT_NE(result, data.data() + data.size());
    EXPECT_EQ(*result, 42);
}

// Test: Uniform sequence (linear distribution)
TEST_F(IntParallel, UniformSequence) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data(10000);
    std::iota(data.begin(), data.end(), 0);

    auto idx_single = build_single_threaded(data);
    auto idx_parallel = build_parallel(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}

// Test: Large uniform sequence
TEST_F(LargeIntParallel, LargeUniformSequence) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data(100000);
    std::iota(data.begin(), data.end(), 0);

    auto idx_single = build_single_threaded(data);
    auto idx_parallel = build_parallel(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}

// Test: Skewed distribution
TEST_F(IntParallel, SkewedDistribution) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data;

    // Dense region
    for (int i = 0; i < 1000; ++i) {
        data.push_back(i);
    }

    // Sparse region
    for (int i = 1000; i < 100000; i += 100) {
        data.push_back(i);
    }

    auto idx_single = build_single_threaded(data);
    auto idx_parallel = build_parallel(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}

// Test: Quadratic distribution (requires quadratic models)
TEST_F(IntParallel, QuadraticDistribution) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data;
    for (int i = 0; i < 1000; ++i) {
        data.push_back(i * i);
    }

    auto idx_single = build_single_threaded(data);
    auto idx_parallel = build_parallel(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}

// Test: Cubic distribution (requires cubic models)
TEST_F(IntParallel, CubicDistribution) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data;
    for (int i = 0; i < 500; ++i) {
        data.push_back(i * i * i);
    }

    auto idx_single = build_single_threaded(data);
    auto idx_parallel = build_parallel(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}

// Test: Duplicates
TEST_F(IntParallel, WithDuplicates) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(i);
        data.push_back(i);  // Each value appears twice
        data.push_back(i);  // And three times
    }
    std::sort(data.begin(), data.end());

    auto idx_single = build_single_threaded(data);
    auto idx_parallel = build_parallel(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}

// Test: Custom threading model using prepare_build_tasks
TEST_F(IntParallel, CustomThreadingModel) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data(10000);
    std::iota(data.begin(), data.end(), 0);

    // Build using standard method
    auto idx_single = build_single_threaded(data);

    // Build using custom task execution
    IndexType idx_custom;
    auto tasks = idx_custom.prepare_build_tasks(data.data(), data.data() + data.size());

    // Execute tasks in single-threaded manner (simulating custom threading)
    std::vector<jazzy::detail::SegmentAnalysis<int>> results;
    results.reserve(tasks.size());

    for (auto& task : tasks) {
        results.push_back(task.execute());
    }

    idx_custom.finalize_build(results);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    EXPECT_TRUE(indexes_equal(idx_single, idx_custom, data));
}

// Test: Unsorted data throws exception
TEST_F(IntParallel, UnsortedDataThrows) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data{5, 3, 1, 2, 4};  // Not sorted

    IndexType index;
    EXPECT_THROW(
        index.build_parallel(data.data(), data.data() + data.size()),
        std::runtime_error
    );

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif
}

// Test: Various segment counts
TEST(ParallelBuildVariousSegments, TinySegments) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::SegmentCount::TINY> idx_single;
    jazzy::JazzyIndex<int, jazzy::SegmentCount::TINY> idx_parallel;

    idx_single.build(data.data(), data.data() + data.size());
    idx_parallel.build_parallel(data.data(), data.data() + data.size());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}

TEST(ParallelBuildVariousSegments, XXLargeSegments) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data(50000);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::SegmentCount::XXLARGE> idx_single;
    jazzy::JazzyIndex<int, jazzy::SegmentCount::XXLARGE> idx_parallel;

    idx_single.build(data.data(), data.data() + data.size());
    idx_parallel.build_parallel(data.data(), data.data() + data.size());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}

// Test: Floating point data
TEST(ParallelBuildFloatingPoint, DoubleData) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<double> data;
    for (int i = 0; i < 5000; ++i) {
        data.push_back(std::sqrt(static_cast<double>(i)));
    }

    jazzy::JazzyIndex<double, jazzy::SegmentCount::LARGE> idx_single;
    jazzy::JazzyIndex<double, jazzy::SegmentCount::LARGE> idx_parallel;

    idx_single.build(data.data(), data.data() + data.size());
    idx_parallel.build_parallel(data.data(), data.data() + data.size());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}

// Test: Task execution with std::async (simulating parallel execution)
TEST_F(IntParallel, TaskExecutionWithAsync) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data(10000);
    std::iota(data.begin(), data.end(), 0);

    auto idx_single = build_single_threaded(data);

    // Build using tasks with async execution
    IndexType idx_async;
    auto tasks = idx_async.prepare_build_tasks(data.data(), data.data() + data.size());

    // Launch tasks asynchronously
    std::vector<std::future<jazzy::detail::SegmentAnalysis<int>>> futures;
    for (auto& task : tasks) {
        futures.push_back(std::async(std::launch::async, [task]() {
            return task.execute();
        }));
    }

    // Collect results
    std::vector<jazzy::detail::SegmentAnalysis<int>> results;
    for (auto& future : futures) {
        results.push_back(future.get());
    }

    idx_async.finalize_build(results);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    EXPECT_TRUE(indexes_equal(idx_single, idx_async, data));
}

// Test: Mismatched result count throws
TEST_F(IntParallel, MismatchedResultCountThrows) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);

    IndexType index;
    auto tasks = index.prepare_build_tasks(data.data(), data.data() + data.size());

    // Create wrong number of results
    std::vector<jazzy::detail::SegmentAnalysis<int>> results(tasks.size() - 1);

    EXPECT_THROW(index.finalize_build(results), std::runtime_error);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif
}

// Test: Dataset smaller than segment count
TEST_F(IntParallel, DatasetSmallerThanSegments) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data{1, 2, 3, 4, 5};  // 5 elements with 256 segments configured

    auto idx_single = build_single_threaded(data);
    auto idx_parallel = build_parallel(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
    EXPECT_EQ(idx_parallel.num_segments(), data.size());  // Should use actual size
}

// Test: All identical values (constant segment)
TEST_F(IntParallel, AllIdenticalValues) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data(1000, 42);  // All values are 42

    auto idx_single = build_single_threaded(data);
    auto idx_parallel = build_parallel(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));

    // All should find 42
    const int* result = idx_parallel.find(42);
    ASSERT_NE(result, data.data() + data.size());
    EXPECT_EQ(*result, 42);
}

// Test: Custom comparator (descending order)
TEST(ParallelBuildCustomComparator, DescendingOrder) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data(10000);
    std::iota(data.begin(), data.end(), 0);
    std::reverse(data.begin(), data.end());  // Descending: 9999, 9998, ..., 0

    using IndexType = jazzy::JazzyIndex<int, jazzy::SegmentCount::LARGE, std::greater<>>;

    IndexType idx_single;
    IndexType idx_parallel;

    idx_single.build(data.data(), data.data() + data.size(), std::greater<>{});
    idx_parallel.build_parallel(data.data(), data.data() + data.size(), std::greater<>{});

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    // Test searches
    const int* result1 = idx_parallel.find(5000);
    ASSERT_NE(result1, data.data() + data.size());
    EXPECT_EQ(*result1, 5000);

    const int* result2 = idx_parallel.find(0);
    ASSERT_NE(result2, data.data() + data.size());
    EXPECT_EQ(*result2, 0);
}

// Test: Custom key extractor (key-value pairs)
TEST(ParallelBuildCustomKeyExtractor, KeyValuePairs) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    struct KeyValue {
        int key;
        std::string value;
        bool operator<(const KeyValue& other) const { return key < other.key; }
        bool operator==(const KeyValue& other) const { return key == other.key; }
    };

    auto key_extractor = [](const KeyValue& kv) { return kv.key; };

    std::vector<KeyValue> data;
    for (int i = 0; i < 5000; ++i) {
        data.push_back({i * 2, "value_" + std::to_string(i)});  // Even keys only
    }

    using IndexType = jazzy::JazzyIndex<KeyValue, jazzy::SegmentCount::LARGE,
                                        std::less<>, decltype(key_extractor)>;

    IndexType idx_single;
    IndexType idx_parallel;

    idx_single.build(data.data(), data.data() + data.size(), std::less<>{}, key_extractor);
    idx_parallel.build_parallel(data.data(), data.data() + data.size(), std::less<>{}, key_extractor);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    // Test searches
    KeyValue search_key{1000, ""};
    const KeyValue* result = idx_parallel.find(search_key);
    ASSERT_NE(result, data.data() + data.size());
    EXPECT_EQ(result->key, 1000);
    EXPECT_EQ(result->value, "value_500");
}

// Test: Multiple sequential builds on same index
TEST_F(IntParallel, MultipleSequentialBuilds) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    IndexType index;

    // First build
    std::vector<int> data1(1000);
    std::iota(data1.begin(), data1.end(), 0);
    index.build_parallel(data1.data(), data1.data() + data1.size());

    const int* result1 = index.find(500);
    ASSERT_NE(result1, data1.data() + data1.size());
    EXPECT_EQ(*result1, 500);

    // Second build (should overwrite)
    std::vector<int> data2(2000);
    std::iota(data2.begin(), data2.end(), 1000);
    index.build_parallel(data2.data(), data2.data() + data2.size());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    const int* result2 = index.find(1500);
    ASSERT_NE(result2, data2.data() + data2.size());
    EXPECT_EQ(*result2, 1500);

    // Old data should not be found
    const int* result3 = index.find(500);
    EXPECT_EQ(result3, data2.data() + data2.size());  // Not found
}

// Test: Very small datasets (edge cases)
TEST_F(IntParallel, VerySmallDatasets) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    // Two elements
    {
        std::vector<int> data{1, 2};
        auto idx_single = build_single_threaded(data);
        auto idx_parallel = build_parallel(data);
        EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
    }

    // Three elements
    {
        std::vector<int> data{1, 5, 10};
        auto idx_single = build_single_threaded(data);
        auto idx_parallel = build_parallel(data);
        EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
    }

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif
}

// Test: Segment metadata preservation
TEST_F(IntParallel, SegmentMetadataPreserved) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data(10000);
    std::iota(data.begin(), data.end(), 0);

    auto idx_single = build_single_threaded(data);
    auto idx_parallel = build_parallel(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    // Both should have same number of segments
    EXPECT_EQ(idx_single.num_segments(), idx_parallel.num_segments());
    EXPECT_EQ(idx_single.size(), idx_parallel.size());
    EXPECT_TRUE(idx_single.is_built());
    EXPECT_TRUE(idx_parallel.is_built());
}

// Test: Empty result after prepare (edge case)
TEST_F(IntParallel, EmptyTaskList) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data;  // Empty
    IndexType index;

    auto tasks = index.prepare_build_tasks(data.data(), data.data());
    EXPECT_TRUE(tasks.empty());

    // Should be safe to call finalize with empty results
    std::vector<jazzy::detail::SegmentAnalysis<int>> results;
    // This should not throw, just do nothing
    EXPECT_NO_THROW(index.finalize_build(results));

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif
}

// Test: Negative integers
TEST_F(IntParallel, NegativeIntegers) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data(5000);
    std::iota(data.begin(), data.end(), -2500);  // -2500 to 2499

    auto idx_single = build_single_threaded(data);
    auto idx_parallel = build_parallel(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));

    const int* result = idx_parallel.find(0);
    ASSERT_NE(result, data.data() + data.size());
    EXPECT_EQ(*result, 0);
}

// Test: Large sparse dataset
TEST_F(IntParallel, LargeSparseDataset) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data;
    for (int i = 0; i < 10000; i += 100) {  // Very sparse
        data.push_back(i);
    }

    auto idx_single = build_single_threaded(data);
    auto idx_parallel = build_parallel(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex"), std::string::npos);
    }
#endif

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}
