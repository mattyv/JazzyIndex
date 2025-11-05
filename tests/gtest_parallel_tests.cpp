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
    std::vector<int> data;

    auto idx_single = build_single_threaded(data);
    auto idx_parallel = build_parallel(data);

    EXPECT_EQ(idx_single.size(), 0);
    EXPECT_EQ(idx_parallel.size(), 0);
    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}

// Test: Single element
TEST_F(IntParallel, SingleElement) {
    std::vector<int> data{42};

    auto idx_single = build_single_threaded(data);
    auto idx_parallel = build_parallel(data);

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));

    const int* result = idx_parallel.find(42);
    ASSERT_NE(result, data.data() + data.size());
    EXPECT_EQ(*result, 42);
}

// Test: Uniform sequence (linear distribution)
TEST_F(IntParallel, UniformSequence) {
    std::vector<int> data(10000);
    std::iota(data.begin(), data.end(), 0);

    auto idx_single = build_single_threaded(data);
    auto idx_parallel = build_parallel(data);

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}

// Test: Large uniform sequence
TEST_F(LargeIntParallel, LargeUniformSequence) {
    std::vector<int> data(100000);
    std::iota(data.begin(), data.end(), 0);

    auto idx_single = build_single_threaded(data);
    auto idx_parallel = build_parallel(data);

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}

// Test: Skewed distribution
TEST_F(IntParallel, SkewedDistribution) {
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

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}

// Test: Quadratic distribution (requires quadratic models)
TEST_F(IntParallel, QuadraticDistribution) {
    std::vector<int> data;
    for (int i = 0; i < 1000; ++i) {
        data.push_back(i * i);
    }

    auto idx_single = build_single_threaded(data);
    auto idx_parallel = build_parallel(data);

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}

// Test: Cubic distribution (requires cubic models)
TEST_F(IntParallel, CubicDistribution) {
    std::vector<int> data;
    for (int i = 0; i < 500; ++i) {
        data.push_back(i * i * i);
    }

    auto idx_single = build_single_threaded(data);
    auto idx_parallel = build_parallel(data);

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}

// Test: Duplicates
TEST_F(IntParallel, WithDuplicates) {
    std::vector<int> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(i);
        data.push_back(i);  // Each value appears twice
        data.push_back(i);  // And three times
    }
    std::sort(data.begin(), data.end());

    auto idx_single = build_single_threaded(data);
    auto idx_parallel = build_parallel(data);

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}

// Test: Custom threading model using prepare_build_tasks
TEST_F(IntParallel, CustomThreadingModel) {
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

    EXPECT_TRUE(indexes_equal(idx_single, idx_custom, data));
}

// Test: Unsorted data throws exception
TEST_F(IntParallel, UnsortedDataThrows) {
    std::vector<int> data{5, 3, 1, 2, 4};  // Not sorted

    IndexType index;
    EXPECT_THROW(
        index.build_parallel(data.data(), data.data() + data.size()),
        std::runtime_error
    );
}

// Test: Various segment counts
TEST(ParallelBuildVariousSegments, TinySegments) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::SegmentCount::TINY> idx_single;
    jazzy::JazzyIndex<int, jazzy::SegmentCount::TINY> idx_parallel;

    idx_single.build(data.data(), data.data() + data.size());
    idx_parallel.build_parallel(data.data(), data.data() + data.size());

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}

TEST(ParallelBuildVariousSegments, XXLargeSegments) {
    std::vector<int> data(50000);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::SegmentCount::XXLARGE> idx_single;
    jazzy::JazzyIndex<int, jazzy::SegmentCount::XXLARGE> idx_parallel;

    idx_single.build(data.data(), data.data() + data.size());
    idx_parallel.build_parallel(data.data(), data.data() + data.size());

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}

// Test: Floating point data
TEST(ParallelBuildFloatingPoint, DoubleData) {
    std::vector<double> data;
    for (int i = 0; i < 5000; ++i) {
        data.push_back(std::sqrt(static_cast<double>(i)));
    }

    jazzy::JazzyIndex<double, jazzy::SegmentCount::LARGE> idx_single;
    jazzy::JazzyIndex<double, jazzy::SegmentCount::LARGE> idx_parallel;

    idx_single.build(data.data(), data.data() + data.size());
    idx_parallel.build_parallel(data.data(), data.data() + data.size());

    EXPECT_TRUE(indexes_equal(idx_single, idx_parallel, data));
}

// Test: Task execution with std::async (simulating parallel execution)
TEST_F(IntParallel, TaskExecutionWithAsync) {
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

    EXPECT_TRUE(indexes_equal(idx_single, idx_async, data));
}

// Test: Mismatched result count throws
TEST_F(IntParallel, MismatchedResultCountThrows) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);

    IndexType index;
    auto tasks = index.prepare_build_tasks(data.data(), data.data() + data.size());

    // Create wrong number of results
    std::vector<jazzy::detail::SegmentAnalysis<int>> results(tasks.size() - 1);

    EXPECT_THROW(index.finalize_build(results), std::runtime_error);
}
