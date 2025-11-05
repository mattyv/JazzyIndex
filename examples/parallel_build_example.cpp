// Example demonstrating parallel build functionality in JazzyIndex
//
// This example shows three ways to use parallel builds:
// 1. Simple parallel build with default threading (easiest)
// 2. Custom threading model using prepare_build_tasks() and finalize_build()
// 3. Performance comparison between single-threaded and parallel builds

#include "jazzy_index.hpp"
#include "jazzy_index_parallel.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

// Helper to measure execution time
template <typename F>
double measure_time_ms(F&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Example 1: Simple parallel build (default threading)
void example_simple_parallel_build() {
    std::cout << "=== Example 1: Simple Parallel Build ===\n";

    // Create a dataset
    std::vector<int> data(1'000'000);
    std::iota(data.begin(), data.end(), 0);

    // Build index using parallel build (uses std::async internally)
    jazzy::JazzyIndex<int, jazzy::SegmentCount::LARGE> index;

    auto time = measure_time_ms([&]() {
        index.build_parallel(data.data(), data.data() + data.size());
    });

    std::cout << "Built index with " << index.num_segments() << " segments\n";
    std::cout << "Build time: " << time << " ms\n";

    // Verify it works
    const int* result = index.find(500'000);
    if (result != data.data() + data.size()) {
        std::cout << "Found value: " << *result << "\n";
    }

    std::cout << "\n";
}

// Example 2: Custom threading model
void example_custom_threading() {
    std::cout << "=== Example 2: Custom Threading Model ===\n";

    std::vector<int> data(1'000'000);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::SegmentCount::LARGE> index;

    auto time = measure_time_ms([&]() {
        // Step 1: Prepare build tasks
        auto tasks = index.prepare_build_tasks(data.data(), data.data() + data.size());

        std::cout << "Generated " << tasks.size() << " independent build tasks\n";

        // Step 2: Execute tasks in your own threading model
        // Here we demonstrate using std::async, but you could use:
        // - Your own thread pool
        // - A task scheduler
        // - A job system
        // - Single-threaded execution for debugging

        std::vector<std::future<jazzy::detail::SegmentAnalysis<int>>> futures;
        for (auto& task : tasks) {
            futures.push_back(std::async(std::launch::async, [task]() {
                return task.execute();
            }));
        }

        // Collect results
        std::vector<jazzy::detail::SegmentAnalysis<int>> results;
        results.reserve(futures.size());
        for (auto& future : futures) {
            results.push_back(future.get());
        }

        // Step 3: Finalize the index
        index.finalize_build(results);
    });

    std::cout << "Build time: " << time << " ms\n";

    // Verify
    const int* result = index.find(750'000);
    if (result != data.data() + data.size()) {
        std::cout << "Found value: " << *result << "\n";
    }

    std::cout << "\n";
}

// Example 3: Performance comparison
void example_performance_comparison() {
    std::cout << "=== Example 3: Performance Comparison ===\n";

    // Test with various data sizes
    std::vector<std::size_t> sizes = {10'000, 100'000, 1'000'000};

    for (std::size_t size : sizes) {
        std::vector<int> data(size);
        std::iota(data.begin(), data.end(), 0);

        // Single-threaded build
        jazzy::JazzyIndex<int, jazzy::SegmentCount::LARGE> index_st;
        double time_st = measure_time_ms([&]() {
            index_st.build(data.data(), data.data() + data.size());
        });

        // Parallel build
        jazzy::JazzyIndex<int, jazzy::SegmentCount::LARGE> index_mt;
        double time_mt = measure_time_ms([&]() {
            index_mt.build_parallel(data.data(), data.data() + data.size());
        });

        double speedup = time_st / time_mt;

        std::cout << "Dataset size: " << size << "\n";
        std::cout << "  Single-threaded: " << time_st << " ms\n";
        std::cout << "  Parallel:        " << time_mt << " ms\n";
        std::cout << "  Speedup:         " << speedup << "x\n";
        std::cout << "\n";
    }
}

// Example 4: Different data distributions
void example_different_distributions() {
    std::cout << "=== Example 4: Different Data Distributions ===\n";

    const std::size_t size = 100'000;

    // Uniform distribution
    {
        std::vector<int> data(size);
        std::iota(data.begin(), data.end(), 0);

        jazzy::JazzyIndex<int, jazzy::SegmentCount::LARGE> index;
        auto time = measure_time_ms([&]() {
            index.build_parallel(data.data(), data.data() + data.size());
        });

        std::cout << "Uniform distribution: " << time << " ms\n";
    }

    // Quadratic distribution (requires more complex models)
    {
        std::vector<int> data;
        for (int i = 0; i < static_cast<int>(size); ++i) {
            data.push_back(i * i / 100);  // Scale down to avoid overflow
        }

        jazzy::JazzyIndex<int, jazzy::SegmentCount::LARGE> index;
        auto time = measure_time_ms([&]() {
            index.build_parallel(data.data(), data.data() + data.size());
        });

        std::cout << "Quadratic distribution: " << time << " ms\n";
    }

    // Skewed distribution
    {
        std::vector<int> data;
        // Dense region
        for (int i = 0; i < 10'000; ++i) {
            data.push_back(i);
        }
        // Sparse region
        for (int i = 10'000; i < 1'000'000; i += 10) {
            data.push_back(i);
        }

        jazzy::JazzyIndex<int, jazzy::SegmentCount::LARGE> index;
        auto time = measure_time_ms([&]() {
            index.build_parallel(data.data(), data.data() + data.size());
        });

        std::cout << "Skewed distribution: " << time << " ms\n";
    }

    std::cout << "\n";
}

// Example 5: Error handling
void example_error_handling() {
    std::cout << "=== Example 5: Error Handling ===\n";

    // Unsorted data will throw an exception
    std::vector<int> unsorted_data = {5, 3, 1, 4, 2};

    jazzy::JazzyIndex<int, jazzy::SegmentCount::SMALL> index;

    try {
        index.build_parallel(unsorted_data.data(),
                            unsorted_data.data() + unsorted_data.size());
        std::cout << "ERROR: Should have thrown exception!\n";
    } catch (const std::runtime_error& e) {
        std::cout << "Caught expected exception: " << e.what() << "\n";
    }

    std::cout << "\n";
}

int main() {
    std::cout << "JazzyIndex Parallel Build Examples\n";
    std::cout << "===================================\n\n";

    example_simple_parallel_build();
    example_custom_threading();
    example_performance_comparison();
    example_different_distributions();
    example_error_handling();

    std::cout << "All examples completed!\n";
    return 0;
}
