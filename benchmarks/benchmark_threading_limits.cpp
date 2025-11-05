#include <benchmark/benchmark.h>
#include <algorithm>
#include <random>
#include <thread>
#include <vector>
#include "jazzy_index.hpp"
#include "fixtures.hpp"

// This benchmark tests multi-threading limits to determine how many
// concurrent threads can query the index without performance degradation.
//
// We test:
// 1. Thread counts: 1, 2, 4, 8, 16, 32, 64, 128
// 2. Multiple data distributions (uniform, clustered, exponential)
// 3. Different dataset sizes (10K, 100K, 1M)
//
// Key metrics to watch:
// - Per-thread throughput (items/s) should remain constant
// - If throughput degrades, we've hit contention limits

// Key-value struct for benchmarking
struct KeyValue {
    std::uint64_t key;
    std::string value;

    bool operator<(const KeyValue& other) const {
        return key < other.key;
    }

    bool operator==(const KeyValue& other) const {
        return key == other.key;
    }
};

// Benchmark: Uniform distribution with concurrent queries
static void BM_Threads_Uniform_Find(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));

    // Build data once (shared across all threads)
    static std::vector<std::uint64_t> keys;
    static std::vector<KeyValue> data;
    static jazzy::JazzyIndex<KeyValue, jazzy::SegmentCount::LARGE, std::less<>,
                             decltype(&KeyValue::key)> index;
    static std::size_t last_size = 0;

    if (state.thread_index() == 0) {
        if (last_size != size) {
            // Rebuild for new size
            keys = qi::bench::make_uniform_values(size, 0, 1);

            data.clear();
            data.reserve(size);
            for (std::size_t i = 0; i < size; ++i) {
                data.push_back({keys[i], "value_" + std::to_string(keys[i])});
            }

            index.build(data.data(), data.data() + data.size(), std::less<>{}, &KeyValue::key);
            last_size = size;
        }
    }

    // Each thread gets its own search pattern (deterministic based on thread_index)
    std::mt19937_64 rng(dataset::kSeed + state.thread_index());
    std::uniform_int_distribution<std::size_t> dist(0, size - 1);

    std::vector<KeyValue> search_keys;
    search_keys.reserve(1000);
    for (std::size_t i = 0; i < 1000; ++i) {
        std::size_t idx = dist(rng);
        search_keys.push_back({keys[idx], ""});
    }

    // Benchmark concurrent queries
    std::size_t search_idx = 0;
    std::size_t found_count = 0;

    for (auto _ : state) {
        const auto& search_key = search_keys[search_idx % search_keys.size()];
        const KeyValue* result = index.find(search_key);

        if (result != data.data() + data.size()) {
            found_count++;
        }

        search_idx++;
        benchmark::DoNotOptimize(result);
    }

    state.counters["FoundRate"] = benchmark::Counter(
        static_cast<double>(found_count) / static_cast<double>(state.iterations()),
        benchmark::Counter::kIsRate | benchmark::Counter::kAvgThreads
    );
}

// Benchmark: Clustered distribution (worst-case for learned indexes)
static void BM_Threads_Clustered_Find(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));

    static std::vector<std::uint64_t> keys;
    static std::vector<KeyValue> data;
    static jazzy::JazzyIndex<KeyValue, jazzy::SegmentCount::LARGE, std::less<>,
                             decltype(&KeyValue::key)> index;
    static std::size_t last_size = 0;

    if (state.thread_index() == 0) {
        if (last_size != size) {
            keys = qi::bench::make_clustered_values(size);

            data.clear();
            data.reserve(size);
            for (std::size_t i = 0; i < size; ++i) {
                data.push_back({keys[i], "value_" + std::to_string(keys[i])});
            }

            index.build(data.data(), data.data() + data.size(), std::less<>{}, &KeyValue::key);
            last_size = size;
        }
    }

    std::mt19937_64 rng(dataset::kSeed + state.thread_index());
    std::uniform_int_distribution<std::size_t> dist(0, size - 1);

    std::vector<KeyValue> search_keys;
    search_keys.reserve(1000);
    for (std::size_t i = 0; i < 1000; ++i) {
        std::size_t idx = dist(rng);
        search_keys.push_back({keys[idx], ""});
    }

    std::size_t search_idx = 0;
    std::size_t found_count = 0;

    for (auto _ : state) {
        const auto& search_key = search_keys[search_idx % search_keys.size()];
        const KeyValue* result = index.find(search_key);

        if (result != data.data() + data.size()) {
            found_count++;
        }

        search_idx++;
        benchmark::DoNotOptimize(result);
    }

    state.counters["FoundRate"] = benchmark::Counter(
        static_cast<double>(found_count) / static_cast<double>(state.iterations()),
        benchmark::Counter::kIsRate | benchmark::Counter::kAvgThreads
    );
}

// Benchmark: Exponential distribution
static void BM_Threads_Exponential_Find(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));

    static std::vector<std::uint64_t> keys;
    static std::vector<KeyValue> data;
    static jazzy::JazzyIndex<KeyValue, jazzy::SegmentCount::LARGE, std::less<>,
                             decltype(&KeyValue::key)> index;
    static std::size_t last_size = 0;

    if (state.thread_index() == 0) {
        if (last_size != size) {
            keys = qi::bench::make_exponential_values(size);

            data.clear();
            data.reserve(size);
            for (std::size_t i = 0; i < size; ++i) {
                data.push_back({keys[i], "value_" + std::to_string(keys[i])});
            }

            index.build(data.data(), data.data() + data.size(), std::less<>{}, &KeyValue::key);
            last_size = size;
        }
    }

    std::mt19937_64 rng(dataset::kSeed + state.thread_index());
    std::uniform_int_distribution<std::size_t> dist(0, size - 1);

    std::vector<KeyValue> search_keys;
    search_keys.reserve(1000);
    for (std::size_t i = 0; i < 1000; ++i) {
        std::size_t idx = dist(rng);
        search_keys.push_back({keys[idx], ""});
    }

    std::size_t search_idx = 0;
    std::size_t found_count = 0;

    for (auto _ : state) {
        const auto& search_key = search_keys[search_idx % search_keys.size()];
        const KeyValue* result = index.find(search_key);

        if (result != data.data() + data.size()) {
            found_count++;
        }

        search_idx++;
        benchmark::DoNotOptimize(result);
    }

    state.counters["FoundRate"] = benchmark::Counter(
        static_cast<double>(found_count) / static_cast<double>(state.iterations()),
        benchmark::Counter::kIsRate | benchmark::Counter::kAvgThreads
    );
}

// Register benchmarks with thread counts: 1, 2, 4, 8, 16, 32, 64, 128
// Testing multiple dataset sizes: 10K, 100K, 1M

// Uniform distribution benchmarks
BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(10000)
    ->Threads(1)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(10000)
    ->Threads(2)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(10000)
    ->Threads(4)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(10000)
    ->Threads(8)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(10000)
    ->Threads(16)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(10000)
    ->Threads(32)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(10000)
    ->Threads(64)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(10000)
    ->Threads(128)
    ->Unit(benchmark::kNanosecond);

// 100K dataset
BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(100000)
    ->Threads(1)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(100000)
    ->Threads(2)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(100000)
    ->Threads(4)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(100000)
    ->Threads(8)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(100000)
    ->Threads(16)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(100000)
    ->Threads(32)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(100000)
    ->Threads(64)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(100000)
    ->Threads(128)
    ->Unit(benchmark::kNanosecond);

// 1M dataset
BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(1000000)
    ->Threads(1)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(1000000)
    ->Threads(2)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(1000000)
    ->Threads(4)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(1000000)
    ->Threads(8)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(1000000)
    ->Threads(16)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(1000000)
    ->Threads(32)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(1000000)
    ->Threads(64)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Uniform_Find)
    ->Arg(1000000)
    ->Threads(128)
    ->Unit(benchmark::kNanosecond);

// Clustered distribution benchmarks (worst-case)
BENCHMARK(BM_Threads_Clustered_Find)
    ->Arg(100000)
    ->Threads(1)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Clustered_Find)
    ->Arg(100000)
    ->Threads(2)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Clustered_Find)
    ->Arg(100000)
    ->Threads(4)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Clustered_Find)
    ->Arg(100000)
    ->Threads(8)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Clustered_Find)
    ->Arg(100000)
    ->Threads(16)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Clustered_Find)
    ->Arg(100000)
    ->Threads(32)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Clustered_Find)
    ->Arg(100000)
    ->Threads(64)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Clustered_Find)
    ->Arg(100000)
    ->Threads(128)
    ->Unit(benchmark::kNanosecond);

// Exponential distribution benchmarks
BENCHMARK(BM_Threads_Exponential_Find)
    ->Arg(100000)
    ->Threads(1)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Exponential_Find)
    ->Arg(100000)
    ->Threads(2)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Exponential_Find)
    ->Arg(100000)
    ->Threads(4)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Exponential_Find)
    ->Arg(100000)
    ->Threads(8)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Exponential_Find)
    ->Arg(100000)
    ->Threads(16)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Exponential_Find)
    ->Arg(100000)
    ->Threads(32)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Exponential_Find)
    ->Arg(100000)
    ->Threads(64)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Threads_Exponential_Find)
    ->Arg(100000)
    ->Threads(128)
    ->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
