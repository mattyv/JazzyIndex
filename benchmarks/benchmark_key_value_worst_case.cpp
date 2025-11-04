#include <benchmark/benchmark.h>
#include <algorithm>
#include <string>
#include <vector>
#include "jazzy_index.hpp"
#include "dataset_generators.hpp"

// Key-value struct for benchmarking
struct KeyValue {
    std::uint64_t key;
    std::string value;

    bool operator<(const KeyValue& other) const {
        return key < other.key;
    }

    bool operator>(const KeyValue& other) const {
        return key > other.key;
    }

    bool operator==(const KeyValue& other) const {
        return key == other.key;
    }
};

// Benchmark key-value index with worst-case clustered distribution
static void BM_KeyValue_Clustered_Find(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));

    // Generate clustered distribution (worst case for learned indexes)
    auto keys = dataset::generate_clustered(
        size,
        dataset::kClusterCount,     // 5 clusters
        dataset::kClusterSpread,    // 0.02 spread
        dataset::kSeed,
        0,
        std::numeric_limits<std::uint64_t>::max()
    );

    // Create key-value pairs
    std::vector<KeyValue> data;
    data.reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
        data.push_back({keys[i], "value_" + std::to_string(keys[i])});
    }

    // Build index with key extractor
    jazzy::JazzyIndex<KeyValue, jazzy::SegmentCount::LARGE, std::less<>,
                      decltype(&KeyValue::key)> index;
    index.build(data.data(), data.data() + data.size(), std::less<>{}, &KeyValue::key);

    // Generate search keys (mix of hits and misses)
    std::vector<KeyValue> search_keys;
    search_keys.reserve(1000);
    for (std::size_t i = 0; i < 1000; ++i) {
        if (i % 2 == 0) {
            // Hit: use actual key from data
            std::size_t idx = (i * 997) % size;  // Pseudo-random but deterministic
            search_keys.push_back({keys[idx], ""});
        } else {
            // Miss: use key that's likely not in data
            search_keys.push_back({keys[i % size] + 1, ""});
        }
    }

    // Benchmark
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
        benchmark::Counter::kAvgThreads
    );
    state.counters["Size"] = benchmark::Counter(
        static_cast<double>(size),
        benchmark::Counter::kDefaults
    );
}

// Benchmark comparison: std::lower_bound on clustered data
static void BM_LowerBound_Clustered_Find(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));

    auto keys = dataset::generate_clustered(
        size,
        dataset::kClusterCount,
        dataset::kClusterSpread,
        dataset::kSeed,
        0,
        std::numeric_limits<std::uint64_t>::max()
    );

    std::vector<KeyValue> data;
    data.reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
        data.push_back({keys[i], "value_" + std::to_string(keys[i])});
    }

    std::vector<KeyValue> search_keys;
    search_keys.reserve(1000);
    for (std::size_t i = 0; i < 1000; ++i) {
        if (i % 2 == 0) {
            std::size_t idx = (i * 997) % size;
            search_keys.push_back({keys[idx], ""});
        } else {
            search_keys.push_back({keys[i % size] + 1, ""});
        }
    }

    std::size_t search_idx = 0;
    std::size_t found_count = 0;
    for (auto _ : state) {
        const auto& search_key = search_keys[search_idx % search_keys.size()];

        auto it = std::lower_bound(data.begin(), data.end(), search_key,
            [](const KeyValue& a, const KeyValue& b) {
                return a.key < b.key;
            });

        if (it != data.end() && it->key == search_key.key) {
            found_count++;
        }

        search_idx++;
        benchmark::DoNotOptimize(it);
    }

    state.counters["FoundRate"] = benchmark::Counter(
        static_cast<double>(found_count) / static_cast<double>(state.iterations()),
        benchmark::Counter::kAvgThreads
    );
}

// Benchmark key-value with extreme polynomial distribution
static void BM_KeyValue_ExtremePoly_Find(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));

    // Generate extreme polynomial distribution (x^5 - very non-linear)
    auto keys = dataset::generate_extreme_polynomial(
        size,
        dataset::kSeed,
        0,
        std::numeric_limits<std::uint64_t>::max()
    );

    std::vector<KeyValue> data;
    data.reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
        data.push_back({keys[i], "value_" + std::to_string(keys[i])});
    }

    jazzy::JazzyIndex<KeyValue, jazzy::SegmentCount::LARGE, std::less<>,
                      decltype(&KeyValue::key)> index;
    index.build(data.data(), data.data() + data.size(), std::less<>{}, &KeyValue::key);

    std::vector<KeyValue> search_keys;
    search_keys.reserve(1000);
    for (std::size_t i = 0; i < 1000; ++i) {
        std::size_t idx = (i * 997) % size;
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
        benchmark::Counter::kAvgThreads
    );
}

// Register benchmarks with various sizes
BENCHMARK(BM_KeyValue_Clustered_Find)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->Arg(10000000)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_LowerBound_Clustered_Find)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->Arg(10000000)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_KeyValue_ExtremePoly_Find)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->Arg(10000000)
    ->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
