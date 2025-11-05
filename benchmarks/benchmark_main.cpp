#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <sys/stat.h>

#include "fixtures.hpp"
#include "jazzy_index_export.hpp"
#include "jazzy_index_parallel.hpp"

namespace {

// Global dataset cache for parallel pre-generation
std::unordered_map<std::string, std::shared_ptr<std::vector<std::uint64_t>>> dataset_cache;

// Optional: number of threads for benchmark execution (0 = single-threaded)
static int benchmark_threads = 0;

// Global flag to control benchmark dataset sizes
static bool use_full_benchmarks = false;

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

    bool operator!=(const KeyValue& other) const {
        return !(*this == other);
    }
};

template <typename F, std::size_t... Segments>
void for_each_segment_count_impl(F&& f, std::integer_sequence<std::size_t, Segments...>) {
    (f(std::integral_constant<std::size_t, Segments>{}), ...);
}

template <typename F>
void for_each_segment_count(F&& f) {
    for_each_segment_count_impl(std::forward<F>(f),
                                std::integer_sequence<std::size_t, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512>{});
}

// Parallel dataset generation
template <typename Generator>
std::shared_ptr<std::vector<std::uint64_t>> get_or_generate_dataset(
    const std::string& name, std::size_t size, Generator&& gen) {
    const std::string key = name + "_" + std::to_string(size);

    auto it = dataset_cache.find(key);
    if (it != dataset_cache.end()) {
        return it->second;
    }

    // Generate and cache
    auto data = std::make_shared<std::vector<std::uint64_t>>(gen(size));
    dataset_cache[key] = data;
    return data;
}

// Pre-generate all datasets in parallel (called once at startup)
void pre_generate_datasets_parallel(const std::vector<std::size_t>& sizes) {
    std::cout << "Pre-generating datasets in parallel..." << std::endl;

    struct Distribution {
        std::string name;
        std::function<std::vector<std::uint64_t>(std::size_t)> generator;
    };

    std::vector<Distribution> distributions = {
        {"Uniform", [](std::size_t s) { return qi::bench::make_uniform_values(s); }},
        {"Exponential", [](std::size_t s) { return qi::bench::make_exponential_values(s); }},
        {"Clustered", [](std::size_t s) { return qi::bench::make_clustered_values(s); }},
        {"Lognormal", [](std::size_t s) { return qi::bench::make_lognormal_values(s); }},
        {"Zipf", [](std::size_t s) { return qi::bench::make_zipf_values(s); }},
        {"Mixed", [](std::size_t s) { return qi::bench::make_mixed_values(s); }},
        {"Quadratic", [](std::size_t s) { return qi::bench::make_quadratic_values(s); }},
        {"ExtremePoly", [](std::size_t s) { return qi::bench::make_extreme_polynomial_values(s); }},
        {"InversePoly", [](std::size_t s) { return qi::bench::make_inverse_polynomial_values(s); }}
    };

    std::vector<std::future<void>> futures;

    // Launch parallel generation
    for (const auto& dist : distributions) {
        for (std::size_t size : sizes) {
            futures.push_back(std::async(std::launch::async, [&dist, size]() {
                const std::string key = dist.name + "_" + std::to_string(size);
                std::cout << "  Generating " << dist.name << " (N=" << size << ")..." << std::endl;
                auto data = std::make_shared<std::vector<std::uint64_t>>(dist.generator(size));
                dataset_cache[key] = data;
            }));
        }
    }

    // Wait for all to complete
    for (auto& f : futures) {
        f.wait();
    }

    std::cout << "Dataset generation complete! Generated " << dataset_cache.size() << " datasets." << std::endl;
}

// Helper to optionally add threading to benchmarks
template<typename BenchmarkType>
auto maybe_add_threads(BenchmarkType* bench) -> decltype(bench) {
    if (benchmark_threads > 0) {
        return bench->Threads(benchmark_threads);
    }
    return bench;
}

// Baseline: std::lower_bound benchmarks for comparison

template <typename T>
const T* find_with_lower_bound(const std::vector<T>& data, const T& value) {
    auto it = std::lower_bound(data.begin(), data.end(), value);
    if (it != data.end() && *it == value) {
        return &(*it);
    }
    return data.data() + data.size();
}

void register_lower_bound_uniform_suite(std::size_t size) {
    if (size == 0) {
        return;
    }

    auto data = std::make_shared<std::vector<std::uint64_t>>(qi::bench::make_uniform_values(size));
    const std::uint64_t mid_target = (*data)[data->size() / 2];
    const std::uint64_t end_target = data->back();
    const std::uint64_t miss_target =
        end_target == std::numeric_limits<std::uint64_t>::max() ? end_target : end_target + 1;

    const std::string base = "LowerBound/Uniform/N" + std::to_string(size);

    maybe_add_threads(
        benchmark::RegisterBenchmark((base + "/FoundMiddle").c_str(),
                                     [data, mid_target](benchmark::State& state) {
                                         for (auto _ : state) {
                                             const auto* result = find_with_lower_bound(*data, mid_target);
                                             benchmark::DoNotOptimize(result);
                                         }
                                         state.counters["size"] = static_cast<double>(data->size());
                                     })
            ->Unit(benchmark::kNanosecond));

    maybe_add_threads(
        benchmark::RegisterBenchmark((base + "/FoundEnd").c_str(),
                                     [data, end_target](benchmark::State& state) {
                                         for (auto _ : state) {
                                             const auto* result = find_with_lower_bound(*data, end_target);
                                             benchmark::DoNotOptimize(result);
                                         }
                                         state.counters["size"] = static_cast<double>(data->size());
                                     })
            ->Unit(benchmark::kNanosecond));

    maybe_add_threads(
        benchmark::RegisterBenchmark((base + "/NotFound").c_str(),
                                     [data, miss_target](benchmark::State& state) {
                                         for (auto _ : state) {
                                             const auto* result = find_with_lower_bound(*data, miss_target);
                                             benchmark::DoNotOptimize(result);
                                         }
                                         state.counters["size"] = static_cast<double>(data->size());
                                     })
            ->Unit(benchmark::kNanosecond));
}

template <typename Generator>
void register_lower_bound_distribution_suite(const std::string& name,
                                              Generator&& generator,
                                              std::size_t size) {
    if (size == 0) {
        return;
    }

    auto data = std::make_shared<std::vector<std::uint64_t>>(generator(size));
    if (data->empty()) {
        return;
    }

    const std::uint64_t mid_target = (*data)[data->size() / 2];
    const std::uint64_t end_target = data->back();
    const std::uint64_t miss_target =
        end_target == std::numeric_limits<std::uint64_t>::max() ? end_target : end_target + 1;

    const std::string base = "LowerBound/" + name + "/N" + std::to_string(size);

    maybe_add_threads(
        benchmark::RegisterBenchmark((base + "/FoundMiddle").c_str(),
                                     [data, mid_target](benchmark::State& state) {
                                         for (auto _ : state) {
                                             const auto* result = find_with_lower_bound(*data, mid_target);
                                             benchmark::DoNotOptimize(result);
                                         }
                                         state.counters["size"] = static_cast<double>(data->size());
                                     })
            ->Unit(benchmark::kNanosecond));

    maybe_add_threads(
        benchmark::RegisterBenchmark((base + "/FoundEnd").c_str(),
                                     [data, end_target](benchmark::State& state) {
                                         for (auto _ : state) {
                                             const auto* result = find_with_lower_bound(*data, end_target);
                                             benchmark::DoNotOptimize(result);
                                         }
                                         state.counters["size"] = static_cast<double>(data->size());
                                     })
            ->Unit(benchmark::kNanosecond));

    maybe_add_threads(
        benchmark::RegisterBenchmark((base + "/NotFound").c_str(),
                                     [data, miss_target](benchmark::State& state) {
                                         for (auto _ : state) {
                                             const auto* result = find_with_lower_bound(*data, miss_target);
                                             benchmark::DoNotOptimize(result);
                                         }
                                         state.counters["size"] = static_cast<double>(data->size());
                                     })
            ->Unit(benchmark::kNanosecond));
}

// Lower bound benchmarks for key-value pairs
template <typename Generator>
void register_lower_bound_keyvalue_suite(const std::string& name,
                                         Generator&& generator,
                                         std::size_t size) {
    if (size == 0) {
        return;
    }

    // Generate keys and create key-value pairs
    auto keys = std::make_shared<std::vector<std::uint64_t>>(generator(size));
    if (keys->empty()) {
        return;
    }

    auto data = std::make_shared<std::vector<KeyValue>>();
    data->reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
        data->push_back({(*keys)[i], "value_" + std::to_string((*keys)[i])});
    }

    const std::uint64_t mid_key = (*keys)[keys->size() / 2];
    const std::uint64_t end_key = keys->back();
    const std::uint64_t miss_key =
        end_key == std::numeric_limits<std::uint64_t>::max() ? end_key : end_key + 1;

    const std::string base = "LowerBound/" + name + "KV/N" + std::to_string(size);

    maybe_add_threads(
        benchmark::RegisterBenchmark((base + "/FoundMiddle").c_str(),
                                     [data, mid_key](benchmark::State& state) {
                                         KeyValue target{mid_key, ""};
                                         for (auto _ : state) {
                                             const auto* result = find_with_lower_bound(*data, target);
                                             benchmark::DoNotOptimize(result);
                                         }
                                         state.counters["size"] = static_cast<double>(data->size());
                                     })
            ->Unit(benchmark::kNanosecond));

    maybe_add_threads(
        benchmark::RegisterBenchmark((base + "/FoundEnd").c_str(),
                                     [data, end_key](benchmark::State& state) {
                                         KeyValue target{end_key, ""};
                                         for (auto _ : state) {
                                             const auto* result = find_with_lower_bound(*data, target);
                                             benchmark::DoNotOptimize(result);
                                         }
                                         state.counters["size"] = static_cast<double>(data->size());
                                     })
            ->Unit(benchmark::kNanosecond));

    maybe_add_threads(
        benchmark::RegisterBenchmark((base + "/NotFound").c_str(),
                                     [data, miss_key](benchmark::State& state) {
                                         KeyValue target{miss_key, ""};
                                         for (auto _ : state) {
                                             const auto* result = find_with_lower_bound(*data, target);
                                             benchmark::DoNotOptimize(result);
                                         }
                                         state.counters["size"] = static_cast<double>(data->size());
                                     })
            ->Unit(benchmark::kNanosecond));
}

void register_lower_bound_suites() {
    std::vector<std::size_t> sizes = {100, 1'000, 10'000};
    if (use_full_benchmarks) {
        sizes.push_back(100'000);
        sizes.push_back(1'000'000);
    }

    for (const std::size_t size : sizes) {
        register_lower_bound_uniform_suite(size);
    }

    std::vector<std::size_t> dist_sizes = {100, 10'000};
    if (use_full_benchmarks) {
        dist_sizes.push_back(100'000);
        dist_sizes.push_back(1'000'000);
    }

    for (const std::size_t size : dist_sizes) {
        register_lower_bound_distribution_suite("Exponential", qi::bench::make_exponential_values, size);
        register_lower_bound_distribution_suite("Clustered", qi::bench::make_clustered_values, size);
        register_lower_bound_distribution_suite("Lognormal", qi::bench::make_lognormal_values, size);
        register_lower_bound_distribution_suite("Zipf", qi::bench::make_zipf_values, size);
        register_lower_bound_distribution_suite("Mixed", qi::bench::make_mixed_values, size);
        register_lower_bound_distribution_suite("Quadratic", qi::bench::make_quadratic_values, size);
        register_lower_bound_distribution_suite("ExtremePoly", qi::bench::make_extreme_polynomial_values, size);
        register_lower_bound_distribution_suite("InversePoly", qi::bench::make_inverse_polynomial_values, size);
    }

    // Add key-value lower_bound baselines
    for (const std::size_t size : dist_sizes) {
        register_lower_bound_keyvalue_suite("Clustered", qi::bench::make_clustered_values, size);
        register_lower_bound_keyvalue_suite("ExtremePoly", qi::bench::make_extreme_polynomial_values, size);
    }
}

// Build time benchmarks
template <std::size_t Segments>
void register_build_benchmark(std::size_t size, const std::string& distribution,
                               const std::vector<std::uint64_t>& data) {
    const std::string name = "JazzyIndexBuild/" + distribution + "/S" +
                            std::to_string(Segments) + "/N" + std::to_string(size);

    maybe_add_threads(
        benchmark::RegisterBenchmark(name.c_str(),
                                     [data](benchmark::State& state) {
                                         for (auto _ : state) {
                                             jazzy::JazzyIndex<std::uint64_t, jazzy::to_segment_count<Segments>()> index;
                                             index.build(data.data(), data.data() + data.size());
                                             benchmark::DoNotOptimize(index);
                                         }
                                         state.counters["segments"] = Segments;
                                         state.counters["size"] = static_cast<double>(data.size());
                                     })
            ->Unit(benchmark::kMicrosecond));
}

// Parallel build time benchmarks
template <std::size_t Segments>
void register_parallel_build_benchmark(std::size_t size, const std::string& distribution,
                                       const std::vector<std::uint64_t>& data) {
    const std::string name = "JazzyIndexBuildParallel/" + distribution + "/S" +
                            std::to_string(Segments) + "/N" + std::to_string(size);

    benchmark::RegisterBenchmark(name.c_str(),
                                 [data](benchmark::State& state) {
                                     for (auto _ : state) {
                                         jazzy::JazzyIndex<std::uint64_t, jazzy::to_segment_count<Segments>()> index;
                                         index.build_parallel(data.data(), data.data() + data.size());
                                         benchmark::DoNotOptimize(index);
                                     }
                                     state.counters["segments"] = Segments;
                                     state.counters["size"] = static_cast<double>(data.size());
                                 })
        ->Unit(benchmark::kMicrosecond);
}

void register_build_suites() {
    std::vector<std::size_t> sizes = {1'000, 10'000};
    if (use_full_benchmarks) {
        sizes.push_back(100'000);
        sizes.push_back(1'000'000);
    }

    for (const std::size_t size : sizes) {
        // Uniform distribution
        auto uniform_data = qi::bench::make_uniform_values(size);
        for_each_segment_count([&](auto seg_tag) {
            register_build_benchmark<decltype(seg_tag)::value>(size, "Uniform", uniform_data);
            register_parallel_build_benchmark<decltype(seg_tag)::value>(size, "Uniform", uniform_data);
        });

        // Exponential distribution
        auto exp_data = qi::bench::make_exponential_values(size);
        for_each_segment_count([&](auto seg_tag) {
            register_build_benchmark<decltype(seg_tag)::value>(size, "Exponential", exp_data);
            register_parallel_build_benchmark<decltype(seg_tag)::value>(size, "Exponential", exp_data);
        });

        // Zipf distribution
        auto zipf_data = qi::bench::make_zipf_values(size);
        for_each_segment_count([&](auto seg_tag) {
            register_build_benchmark<decltype(seg_tag)::value>(size, "Zipf", zipf_data);
            register_parallel_build_benchmark<decltype(seg_tag)::value>(size, "Zipf", zipf_data);
        });

        // Quadratic distribution
        auto quadratic_data = qi::bench::make_quadratic_values(size);
        for_each_segment_count([&](auto seg_tag) {
            register_build_benchmark<decltype(seg_tag)::value>(size, "Quadratic", quadratic_data);
            register_parallel_build_benchmark<decltype(seg_tag)::value>(size, "Quadratic", quadratic_data);
        });

        // Extreme polynomial distribution
        auto extreme_poly_data = qi::bench::make_extreme_polynomial_values(size);
        for_each_segment_count([&](auto seg_tag) {
            register_build_benchmark<decltype(seg_tag)::value>(size, "ExtremePoly", extreme_poly_data);
            register_parallel_build_benchmark<decltype(seg_tag)::value>(size, "ExtremePoly", extreme_poly_data);
        });

        // Inverse polynomial distribution
        auto inverse_poly_data = qi::bench::make_inverse_polynomial_values(size);
        for_each_segment_count([&](auto seg_tag) {
            register_build_benchmark<decltype(seg_tag)::value>(size, "InversePoly", inverse_poly_data);
            register_parallel_build_benchmark<decltype(seg_tag)::value>(size, "InversePoly", inverse_poly_data);
        });
    }
}

// JazzyIndex benchmarks

template <std::size_t Segments>
void register_uniform_suite(std::size_t size) {
    if (size == 0) {
        return;
    }

    // Use cached dataset (already generated in parallel)
    auto data = get_or_generate_dataset("Uniform", size, [](std::size_t s) {
        return qi::bench::make_uniform_values(s);
    });
    const std::uint64_t mid_target = (*data)[data->size() / 2];
    const std::uint64_t end_target = data->back();
    const std::uint64_t miss_target =
        end_target == std::numeric_limits<std::uint64_t>::max() ? end_target : end_target + 1;

    const std::string base =
        "JazzyIndex/Uniform/S" + std::to_string(Segments) + "/N" + std::to_string(size);

    maybe_add_threads(
        benchmark::RegisterBenchmark((base + "/FoundMiddle").c_str(),
                                     [data, mid_target](benchmark::State& state) {
                                         auto index = qi::bench::make_index<Segments>(*data);
                                         for (auto _ : state) {
                                             const auto* result = index.find(mid_target);
                                             benchmark::DoNotOptimize(result);
                                         }
                                         state.counters["segments"] = Segments;
                                         state.counters["size"] = static_cast<double>(data->size());
                                     })
            ->Unit(benchmark::kNanosecond));

    maybe_add_threads(
        benchmark::RegisterBenchmark((base + "/FoundEnd").c_str(),
                                     [data, end_target](benchmark::State& state) {
                                         auto index = qi::bench::make_index<Segments>(*data);
                                         for (auto _ : state) {
                                             const auto* result = index.find(end_target);
                                             benchmark::DoNotOptimize(result);
                                         }
                                         state.counters["segments"] = Segments;
                                         state.counters["size"] = static_cast<double>(data->size());
                                     })
            ->Unit(benchmark::kNanosecond));

    maybe_add_threads(
        benchmark::RegisterBenchmark((base + "/NotFound").c_str(),
                                     [data, miss_target](benchmark::State& state) {
                                         auto index = qi::bench::make_index<Segments>(*data);
                                         for (auto _ : state) {
                                             const auto* result = index.find(miss_target);
                                             benchmark::DoNotOptimize(result);
                                         }
                                         state.counters["segments"] = Segments;
                                         state.counters["size"] = static_cast<double>(data->size());
                                     })
            ->Unit(benchmark::kNanosecond));

}

template <std::size_t Segments, typename Generator>
void register_distribution_suite(const std::string& name,
                                 Generator&& generator,
                                 std::size_t size) {
    if (size == 0) {
        return;
    }

    // Use cached dataset (already generated in parallel)
    auto data = get_or_generate_dataset(name, size, std::forward<Generator>(generator));
    if (data->empty()) {
        return;
    }

    const std::uint64_t mid_target = (*data)[data->size() / 2];
    const std::uint64_t end_target = data->back();
    const std::uint64_t miss_target =
        end_target == std::numeric_limits<std::uint64_t>::max() ? end_target : end_target + 1;

    const std::string base = "JazzyIndex/" + name + "/S" + std::to_string(Segments) +
                             "/N" + std::to_string(size);

    maybe_add_threads(
        benchmark::RegisterBenchmark((base + "/FoundMiddle").c_str(),
                                     [data, mid_target](benchmark::State& state) {
                                         auto index = qi::bench::make_index<Segments>(*data);
                                         for (auto _ : state) {
                                             const auto* result = index.find(mid_target);
                                             benchmark::DoNotOptimize(result);
                                         }
                                         state.counters["segments"] = Segments;
                                         state.counters["size"] = static_cast<double>(data->size());
                                     })
            ->Unit(benchmark::kNanosecond));

    maybe_add_threads(
        benchmark::RegisterBenchmark((base + "/FoundEnd").c_str(),
                                     [data, end_target](benchmark::State& state) {
                                         auto index = qi::bench::make_index<Segments>(*data);
                                         for (auto _ : state) {
                                             const auto* result = index.find(end_target);
                                             benchmark::DoNotOptimize(result);
                                         }
                                         state.counters["segments"] = Segments;
                                         state.counters["size"] = static_cast<double>(data->size());
                                     })
            ->Unit(benchmark::kNanosecond));

    maybe_add_threads(
        benchmark::RegisterBenchmark((base + "/NotFound").c_str(),
                                     [data, miss_target](benchmark::State& state) {
                                         auto index = qi::bench::make_index<Segments>(*data);
                                         for (auto _ : state) {
                                             const auto* result = index.find(miss_target);
                                             benchmark::DoNotOptimize(result);
                                         }
                                         state.counters["segments"] = Segments;
                                         state.counters["size"] = static_cast<double>(data->size());
                                     })
            ->Unit(benchmark::kNanosecond));

}

void register_uniform_suites() {
    std::vector<std::size_t> sizes = {100, 1'000, 10'000};
    if (use_full_benchmarks) {
        sizes.push_back(100'000);
        sizes.push_back(1'000'000);
    }

    for (const std::size_t size : sizes) {
        for_each_segment_count([size](auto seg_tag) {
            register_uniform_suite<decltype(seg_tag)::value>(size);
        });
    }
}

void register_distribution_suites() {
    std::vector<std::size_t> sizes = {100, 10'000};
    if (use_full_benchmarks) {
        sizes.push_back(100'000);
        sizes.push_back(1'000'000);
    }

    for (const std::size_t size : sizes) {
        for_each_segment_count([size](auto seg_tag) {
            constexpr std::size_t Segments = decltype(seg_tag)::value;
            register_distribution_suite<Segments>("Exponential", qi::bench::make_exponential_values, size);
            register_distribution_suite<Segments>("Clustered", qi::bench::make_clustered_values, size);
            register_distribution_suite<Segments>("Lognormal", qi::bench::make_lognormal_values, size);
            register_distribution_suite<Segments>("Zipf", qi::bench::make_zipf_values, size);
            register_distribution_suite<Segments>("Mixed", qi::bench::make_mixed_values, size);
            register_distribution_suite<Segments>("Quadratic", qi::bench::make_quadratic_values, size);
            register_distribution_suite<Segments>("ExtremePoly", qi::bench::make_extreme_polynomial_values, size);
            register_distribution_suite<Segments>("InversePoly", qi::bench::make_inverse_polynomial_values, size);
        });
    }
}

// Key-Value benchmarks (for worst-case distributions)
template <std::size_t Segments, typename Generator>
void register_keyvalue_suite(const std::string& name,
                              Generator&& generator,
                              std::size_t size) {
    if (size == 0) {
        return;
    }

    // Generate keys and create key-value pairs
    auto keys = std::make_shared<std::vector<std::uint64_t>>(generator(size));
    if (keys->empty()) {
        return;
    }

    auto data = std::make_shared<std::vector<KeyValue>>();
    data->reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
        data->push_back({(*keys)[i], "value_" + std::to_string((*keys)[i])});
    }

    const std::uint64_t mid_key = (*keys)[keys->size() / 2];
    const std::uint64_t end_key = keys->back();
    const std::uint64_t miss_key =
        end_key == std::numeric_limits<std::uint64_t>::max() ? end_key : end_key + 1;

    const std::string base = "JazzyIndex/" + name + "KV/S" + std::to_string(Segments) +
                             "/N" + std::to_string(size);

    maybe_add_threads(
        benchmark::RegisterBenchmark((base + "/FoundMiddle").c_str(),
                                     [data, mid_key](benchmark::State& state) {
                                         jazzy::JazzyIndex<KeyValue, jazzy::to_segment_count<Segments>(),
                                                           std::less<>, decltype(&KeyValue::key)> index;
                                         index.build(data->data(), data->data() + data->size(),
                                                    std::less<>{}, &KeyValue::key);
                                         KeyValue target{mid_key, ""};
                                         for (auto _ : state) {
                                             const auto* result = index.find(target);
                                             benchmark::DoNotOptimize(result);
                                         }
                                         state.counters["segments"] = Segments;
                                         state.counters["size"] = static_cast<double>(data->size());
                                     })
            ->Unit(benchmark::kNanosecond));

    maybe_add_threads(
        benchmark::RegisterBenchmark((base + "/FoundEnd").c_str(),
                                     [data, end_key](benchmark::State& state) {
                                         jazzy::JazzyIndex<KeyValue, jazzy::to_segment_count<Segments>(),
                                                           std::less<>, decltype(&KeyValue::key)> index;
                                         index.build(data->data(), data->data() + data->size(),
                                                    std::less<>{}, &KeyValue::key);
                                         KeyValue target{end_key, ""};
                                         for (auto _ : state) {
                                             const auto* result = index.find(target);
                                             benchmark::DoNotOptimize(result);
                                         }
                                         state.counters["segments"] = Segments;
                                         state.counters["size"] = static_cast<double>(data->size());
                                     })
            ->Unit(benchmark::kNanosecond));

    maybe_add_threads(
        benchmark::RegisterBenchmark((base + "/NotFound").c_str(),
                                     [data, miss_key](benchmark::State& state) {
                                         jazzy::JazzyIndex<KeyValue, jazzy::to_segment_count<Segments>(),
                                                           std::less<>, decltype(&KeyValue::key)> index;
                                         index.build(data->data(), data->data() + data->size(),
                                                    std::less<>{}, &KeyValue::key);
                                         KeyValue target{miss_key, ""};
                                         for (auto _ : state) {
                                             const auto* result = index.find(target);
                                             benchmark::DoNotOptimize(result);
                                         }
                                         state.counters["segments"] = Segments;
                                         state.counters["size"] = static_cast<double>(data->size());
                                     })
            ->Unit(benchmark::kNanosecond));
}

void register_keyvalue_suites() {
    // Only benchmark key-value for worst-case distributions
    std::vector<std::size_t> sizes = {100, 10'000};
    if (use_full_benchmarks) {
        sizes.push_back(100'000);
        sizes.push_back(1'000'000);
    }

    for (const std::size_t size : sizes) {
        for_each_segment_count([size](auto seg_tag) {
            constexpr std::size_t Segments = decltype(seg_tag)::value;
            register_keyvalue_suite<Segments>("Clustered", qi::bench::make_clustered_values, size);
            register_keyvalue_suite<Segments>("ExtremePoly", qi::bench::make_extreme_polynomial_values, size);
        });
    }
}

// Helper to create output directory
bool ensure_directory_exists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        // Directory doesn't exist, try to create it
        #ifdef _WIN32
        return _mkdir(path.c_str()) == 0;
        #else
        return mkdir(path.c_str(), 0755) == 0;
        #endif
    } else if (info.st_mode & S_IFDIR) {
        return true;
    }
    return false;
}

template <std::size_t Segments>
std::string export_metadata_for_segments(const std::vector<std::uint64_t>& data) {
    auto index = qi::bench::make_index<Segments>(data);
    return jazzy::export_index_metadata(index);
}

// Export visualization data for various index configurations
void export_visualization_data(const std::string& output_dir) {
    std::cout << "Exporting index visualization data to " << output_dir << "..." << std::endl;

    if (!ensure_directory_exists(output_dir)) {
        std::cerr << "Warning: Could not create directory " << output_dir << std::endl;
        std::cerr << "Attempting to write to current directory instead." << std::endl;
    }

    // Configuration: which distributions, sizes, and segment counts to visualize
    const std::vector<std::size_t> viz_sizes = {100, 1'000, 10'000};
    const std::vector<std::size_t> viz_segments = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

    struct Distribution {
        std::string name;
        std::function<std::vector<std::uint64_t>(std::size_t)> generator;

        Distribution(std::string n, std::function<std::vector<std::uint64_t>(std::size_t)> g)
            : name(std::move(n)), generator(std::move(g)) {}
    };

    std::vector<Distribution> distributions;
    distributions.emplace_back("Uniform", [](std::size_t size) { return qi::bench::make_uniform_values(size); });
    distributions.emplace_back("Exponential", [](std::size_t size) { return qi::bench::make_exponential_values(size); });
    distributions.emplace_back("Clustered", [](std::size_t size) { return qi::bench::make_clustered_values(size); });
    distributions.emplace_back("Lognormal", [](std::size_t size) { return qi::bench::make_lognormal_values(size); });
    distributions.emplace_back("Zipf", [](std::size_t size) { return qi::bench::make_zipf_values(size); });
    distributions.emplace_back("Mixed", [](std::size_t size) { return qi::bench::make_mixed_values(size); });
    distributions.emplace_back("Quadratic", [](std::size_t size) { return qi::bench::make_quadratic_values(size); });
    distributions.emplace_back("ExtremePoly", [](std::size_t size) { return qi::bench::make_extreme_polynomial_values(size); });
    distributions.emplace_back("InversePoly", [](std::size_t size) { return qi::bench::make_inverse_polynomial_values(size); });

    int total_exports = 0;
    int failed_exports = 0;

    for (const auto& dist : distributions) {
        for (const std::size_t size : viz_sizes) {
            auto data = dist.generator(size);
            if (data.empty()) {
                std::cerr << "Warning: Empty dataset for " << dist.name << " size " << size << std::endl;
                continue;
            }

            for (const std::size_t segments : viz_segments) {
                // Build index and export metadata
                std::string json_data;
                bool supported = true;

                switch (segments) {
                    case 1:
                        json_data = export_metadata_for_segments<1>(data);
                        break;
                    case 2:
                        json_data = export_metadata_for_segments<2>(data);
                        break;
                    case 4:
                        json_data = export_metadata_for_segments<4>(data);
                        break;
                    case 8:
                        json_data = export_metadata_for_segments<8>(data);
                        break;
                    case 16:
                        json_data = export_metadata_for_segments<16>(data);
                        break;
                    case 32:
                        json_data = export_metadata_for_segments<32>(data);
                        break;
                    case 64:
                        json_data = export_metadata_for_segments<64>(data);
                        break;
                    case 128:
                        json_data = export_metadata_for_segments<128>(data);
                        break;
                    case 256:
                        json_data = export_metadata_for_segments<256>(data);
                        break;
                    case 512:
                        json_data = export_metadata_for_segments<512>(data);
                        break;
                    default:
                        supported = false;
                        break;
                }

                if (!supported) {
                    continue;
                }

                // Write to file
                std::string filename = output_dir + "/index_" + dist.name + "_N" +
                                     std::to_string(size) + "_S" + std::to_string(segments) + ".json";

                std::ofstream out(filename);
                if (out) {
                    out << json_data;
                    out.close();
                    total_exports++;
                    std::cout << "  Exported: " << filename << std::endl;
                } else {
                    std::cerr << "Error: Could not write to " << filename << std::endl;
                    failed_exports++;
                }
            }
        }
    }

    std::cout << "Visualization data export complete: " << total_exports << " files created";
    if (failed_exports > 0) {
        std::cout << " (" << failed_exports << " failed)";
    }
    std::cout << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    // Check for custom flags before benchmark initialization
    bool visualize_mode = false;
    std::string output_dir = "index_data";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--visualize-index") {
            visualize_mode = true;
            // Remove this flag so benchmark library doesn't see it
            for (int j = i; j < argc - 1; ++j) {
                argv[j] = argv[j + 1];
            }
            --argc;
            --i;
        } else if (arg.find("--visualize-index-output=") == 0) {
            visualize_mode = true;
            output_dir = arg.substr(25);  // Length of "--visualize-index-output="
            // Remove this flag
            for (int j = i; j < argc - 1; ++j) {
                argv[j] = argv[j + 1];
            }
            --argc;
            --i;
        } else if (arg == "--full-benchmarks") {
            use_full_benchmarks = true;
            // Remove this flag so benchmark library doesn't see it
            for (int j = i; j < argc - 1; ++j) {
                argv[j] = argv[j + 1];
            }
            --argc;
            --i;
        } else if (arg.find("--benchmark_threads=") == 0) {
            try {
                benchmark_threads = std::stoi(arg.substr(20));  // Length of "--benchmark_threads="
                std::cout << "Running benchmarks with " << benchmark_threads << " threads per benchmark" << std::endl;
            } catch (...) {
                std::cerr << "Error: Invalid value for --benchmark_threads" << std::endl;
                return 1;
            }
            // Remove this flag so benchmark library doesn't see it
            for (int j = i; j < argc - 1; ++j) {
                argv[j] = argv[j + 1];
            }
            --argc;
            --i;
        }
    }

    if (visualize_mode) {
        export_visualization_data(output_dir);
        std::cout << "\nVisualization data exported. Now run:" << std::endl;
        std::cout << "  python3 scripts/plot_index_structure.py " << output_dir << std::endl;
        return 0;
    }

    // Pre-generate all datasets in parallel (major speedup!)
    std::vector<std::size_t> dataset_sizes = {100, 1'000, 10'000};
    if (use_full_benchmarks) {
        dataset_sizes.push_back(100'000);
        dataset_sizes.push_back(1'000'000);
    }
    pre_generate_datasets_parallel(dataset_sizes);

    // Register baseline std::lower_bound benchmarks first
    register_lower_bound_suites();

    // Register JazzyIndex query benchmarks
    register_uniform_suites();
    register_distribution_suites();

    // Register key-value benchmarks (worst-case distributions)
    register_keyvalue_suites();

    // Register JazzyIndex build time benchmarks
    register_build_suites();

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
