// Performance and complexity assertion tests
// These tests verify that the index meets expected performance characteristics.

#include "jazzy_index.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <vector>

namespace {

template <typename T, std::size_t Segments = 256>
class PerformanceTest : public ::testing::Test {
protected:
    jazzy::JazzyIndex<T, jazzy::to_segment_count<Segments>()> build_index(const std::vector<T>& data) {
        jazzy::JazzyIndex<T, jazzy::to_segment_count<Segments>()> index;
        index.build(data.data(), data.data() + data.size());
        return index;
    }

    // Measure time in nanoseconds for a function
    template <typename Func>
    long long measure_ns(Func&& f) {
        auto start = std::chrono::high_resolution_clock::now();
        f();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
};

using IntPerformanceTest = PerformanceTest<int, 256>;

}  // namespace

// Test: Build time is reasonable for large dataset
TEST_F(IntPerformanceTest, BuildTimeReasonable) {
    std::vector<int> data(100000);
    std::iota(data.begin(), data.end(), 0);

    auto build_time_ns = measure_ns([&]() {
        auto index = build_index(data);
    });

    // Build should complete in reasonable time (< 100ms for 100k elements)
    EXPECT_LT(build_time_ns, 100'000'000)
        << "Build took " << build_time_ns / 1'000'000 << "ms";
}

// Test: Query time is fast for large dataset
TEST_F(IntPerformanceTest, QueryTimeFast) {
    std::vector<int> data(100000);
    std::iota(data.begin(), data.end(), 0);
    auto index = build_index(data);

    // Measure average query time
    const int num_queries = 1000;
    auto query_time_ns = measure_ns([&]() {
        for (int i = 0; i < num_queries; ++i) {
            int query = (i * 12345) % 100000;
            volatile const int* result = index.find(query);
            (void)result;
        }
    });

    long long avg_query_ns = query_time_ns / num_queries;

    // Average query should be fast (< 1000ns / 1us per query)
    EXPECT_LT(avg_query_ns, 1000)
        << "Average query took " << avg_query_ns << "ns";
}

// Test: Uniform data queries are faster than non-uniform
TEST_F(IntPerformanceTest, UniformFasterThanNonUniform) {
    // Uniform data
    std::vector<int> uniform_data(10000);
    std::iota(uniform_data.begin(), uniform_data.end(), 0);
    auto uniform_index = build_index(uniform_data);

    // Non-uniform data (exponential)
    std::vector<int> nonuniform_data;
    for (int i = 0; i < 10000; ++i) {
        nonuniform_data.push_back(static_cast<int>(std::pow(1.001, i)));
    }
    std::sort(nonuniform_data.begin(), nonuniform_data.end());
    nonuniform_data.erase(std::unique(nonuniform_data.begin(), nonuniform_data.end()),
                          nonuniform_data.end());
    auto nonuniform_index = build_index(nonuniform_data);

    const int num_queries = 100;

    // Measure uniform queries
    auto uniform_time_ns = measure_ns([&]() {
        for (int i = 0; i < num_queries; ++i) {
            int query = (i * 123) % 10000;
            volatile const int* result = uniform_index.find(query);
            (void)result;
        }
    });

    // Measure non-uniform queries
    auto nonuniform_time_ns = measure_ns([&]() {
        for (int i = 0; i < num_queries; ++i) {
            int idx = (i * 123) % nonuniform_data.size();
            volatile const int* result = nonuniform_index.find(nonuniform_data[idx]);
            (void)result;
        }
    });

    // Uniform should be faster (or at least not significantly slower)
    // Allow for measurement noise - just check it's not 2x slower
    EXPECT_LT(uniform_time_ns, nonuniform_time_ns * 2)
        << "Uniform: " << uniform_time_ns / num_queries << "ns/query, "
        << "Non-uniform: " << nonuniform_time_ns / num_queries << "ns/query";
}

// Test: More segments doesn't dramatically slow queries
TEST_F(IntPerformanceTest, MoreSegmentsReasonableOverhead) {
    std::vector<int> data(10000);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index64;
    jazzy::JazzyIndex<int, jazzy::to_segment_count<512>()> index512;

    index64.build(data.data(), data.data() + data.size());
    index512.build(data.data(), data.data() + data.size());

    const int num_queries = 100;

    auto time64_ns = measure_ns([&]() {
        for (int i = 0; i < num_queries; ++i) {
            int query = (i * 123) % 10000;
            volatile const int* result = index64.find(query);
            (void)result;
        }
    });

    auto time512_ns = measure_ns([&]() {
        for (int i = 0; i < num_queries; ++i) {
            int query = (i * 123) % 10000;
            volatile const int* result = index512.find(query);
            (void)result;
        }
    });

    // 512 segments should not be more than 3x slower than 64
    // (log(512) / log(64) = 1.5, so with overhead factor of 2x is reasonable)
    EXPECT_LT(time512_ns, time64_ns * 3)
        << "64 segments: " << time64_ns / num_queries << "ns/query, "
        << "512 segments: " << time512_ns / num_queries << "ns/query";
}

// Test: Successful queries are not much slower than unsuccessful
TEST_F(IntPerformanceTest, SuccessfulVsUnsuccessfulQueries) {
    std::vector<int> data;
    for (int i = 0; i < 1000; ++i) {
        data.push_back(i * 10);  // 0, 10, 20, ..., 9990 (gaps)
    }
    auto index = build_index(data);

    const int num_queries = 100;

    // Measure successful queries
    auto success_time_ns = measure_ns([&]() {
        for (int i = 0; i < num_queries; ++i) {
            int query = (i % 1000) * 10;  // Values that exist
            volatile const int* result = index.find(query);
            (void)result;
        }
    });

    // Measure unsuccessful queries
    auto fail_time_ns = measure_ns([&]() {
        for (int i = 0; i < num_queries; ++i) {
            int query = (i % 1000) * 10 + 5;  // Values in gaps
            volatile const int* result = index.find(query);
            (void)result;
        }
    });

    // Both should be similar (within 2x)
    long long min_time = std::min(success_time_ns, fail_time_ns);
    long long max_time = std::max(success_time_ns, fail_time_ns);

    EXPECT_LT(max_time, min_time * 2)
        << "Success: " << success_time_ns / num_queries << "ns/query, "
        << "Fail: " << fail_time_ns / num_queries << "ns/query";
}

// Test: Build time scales linearly (not quadratically)
TEST_F(IntPerformanceTest, BuildTimeScalesLinear) {
    std::vector<int> small_data(10000);
    std::iota(small_data.begin(), small_data.end(), 0);

    std::vector<int> large_data(100000);
    std::iota(large_data.begin(), large_data.end(), 0);

    auto small_build_ns = measure_ns([&]() {
        auto index = build_index(small_data);
    });

    auto large_build_ns = measure_ns([&]() {
        auto index = build_index(large_data);
    });

    // 10x more data should not take more than 20x time (allowing overhead)
    EXPECT_LT(large_build_ns, small_build_ns * 20)
        << "10k elements: " << small_build_ns / 1000 << "us, "
        << "100k elements: " << large_build_ns / 1000 << "us";
}

// Test: Memory footprint is reasonable
TEST_F(IntPerformanceTest, MemoryFootprintReasonable) {
    std::vector<int> data(10000);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::to_segment_count<256>()> index;

    // Index should not use excessive memory
    // Size check: sizeof(index) should be reasonable
    size_t index_size = sizeof(index);

    // With 256 segments, expect < 100KB for the index structure itself
    EXPECT_LT(index_size, 100'000)
        << "Index size: " << index_size << " bytes";
}

// Test: Duplicate-heavy data doesn't slow queries
TEST_F(IntPerformanceTest, DuplicatesNoSlowdown) {
    // Unique data
    std::vector<int> unique_data(1000);
    std::iota(unique_data.begin(), unique_data.end(), 0);
    auto unique_index = build_index(unique_data);

    // Duplicate-heavy data
    std::vector<int> dup_data;
    for (int i = 0; i < 1000; ++i) {
        for (int j = 0; j < 10; ++j) {
            dup_data.push_back(i);
        }
    }
    std::sort(dup_data.begin(), dup_data.end());
    auto dup_index = build_index(dup_data);

    const int num_queries = 100;

    auto unique_time_ns = measure_ns([&]() {
        for (int i = 0; i < num_queries; ++i) {
            int query = (i * 17) % 1000;
            volatile const int* result = unique_index.find(query);
            (void)result;
        }
    });

    auto dup_time_ns = measure_ns([&]() {
        for (int i = 0; i < num_queries; ++i) {
            int query = (i * 17) % 1000;
            volatile const int* result = dup_index.find(query);
            (void)result;
        }
    });

    // Duplicates should not cause significant slowdown
    EXPECT_LT(dup_time_ns, unique_time_ns * 3)
        << "Unique: " << unique_time_ns / num_queries << "ns/query, "
        << "Duplicates: " << dup_time_ns / num_queries << "ns/query";
}

// Test: Small datasets are not penalized
TEST_F(IntPerformanceTest, SmallDatasetsNotPenalized) {
    std::vector<int> small_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto index = build_index(small_data);

    const int num_queries = 1000;

    auto time_ns = measure_ns([&]() {
        for (int i = 0; i < num_queries; ++i) {
            int query = (i % 10) + 1;
            volatile const int* result = index.find(query);
            (void)result;
        }
    });

    long long avg_ns = time_ns / num_queries;

    // Small dataset queries should be very fast (< 500ns)
    EXPECT_LT(avg_ns, 500)
        << "Average query on small dataset: " << avg_ns << "ns";
}

// Test: Quadratic model doesn't add excessive overhead
TEST_F(IntPerformanceTest, QuadraticModelOverhead) {
    // Linear data (should use linear model)
    std::vector<int> linear_data(1000);
    std::iota(linear_data.begin(), linear_data.end(), 0);
    auto linear_index = build_index(linear_data);

    // Quadratic data (should use quadratic model)
    std::vector<int> quad_data;
    for (int i = 0; i < 1000; ++i) {
        quad_data.push_back(i * i);
    }
    auto quad_index = build_index(quad_data);

    const int num_queries = 100;

    auto linear_time_ns = measure_ns([&]() {
        for (int i = 0; i < num_queries; ++i) {
            int query = (i * 13) % 1000;
            volatile const int* result = linear_index.find(query);
            (void)result;
        }
    });

    auto quad_time_ns = measure_ns([&]() {
        for (int i = 0; i < num_queries; ++i) {
            int idx = (i * 13) % 1000;
            int query = quad_data[idx];
            volatile const int* result = quad_index.find(query);
            (void)result;
        }
    });

    // Quadratic model should not be much slower (< 2x)
    EXPECT_LT(quad_time_ns, linear_time_ns * 2)
        << "Linear: " << linear_time_ns / num_queries << "ns/query, "
        << "Quadratic: " << quad_time_ns / num_queries << "ns/query";
}
