// Benchmarks for equal_range, find_lower_bound, and find_upper_bound
// These benchmarks test the new range query functions added to JazzyIndex

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "fixtures.hpp"
#include "jazzy_index_export.hpp"

namespace {

// Global dataset cache
std::unordered_map<std::string, std::shared_ptr<std::vector<std::uint64_t>>> range_dataset_cache;

// Optional: number of threads for benchmark execution (0 = single-threaded)
static int benchmark_threads = 0;

// Distribution generators map
using GeneratorFunc = std::function<std::vector<std::uint64_t>(std::size_t)>;
static const std::unordered_map<std::string, GeneratorFunc> distribution_generators = {
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

// Helper to optionally add threading to benchmarks
template<typename BenchmarkType>
auto maybe_add_threads(BenchmarkType* bench) -> decltype(bench) {
    if (benchmark_threads > 0) {
        return bench->Threads(benchmark_threads);
    }
    return bench;
}

// Helper to get or generate datasets
std::shared_ptr<std::vector<std::uint64_t>> get_or_generate_range_dataset(
    const std::string& name,
    std::size_t size,
    std::function<std::vector<std::uint64_t>(std::size_t)> gen) {
    std::string key = name + "_" + std::to_string(size);
    auto it = range_dataset_cache.find(key);
    if (it != range_dataset_cache.end()) {
        return it->second;
    }

    auto data = std::make_shared<std::vector<std::uint64_t>>(gen(size));
    range_dataset_cache[key] = data;
    return data;
}

// Benchmark JazzyIndex::equal_range
template <std::size_t Segments>
void BM_JazzyIndex_EqualRange(benchmark::State& state, const std::string& distribution, std::uint64_t target) {
    const std::size_t size = state.range(0);

    auto it = distribution_generators.find(distribution);
    if (it == distribution_generators.end()) {
        state.SkipWithError("Unknown distribution");
        return;
    }

    auto data = get_or_generate_range_dataset(distribution, size, it->second);
    auto index = qi::bench::make_index<Segments>(*data);

    for (auto _ : state) {
        auto [lower, upper] = index.equal_range(target);
        benchmark::DoNotOptimize(lower);
        benchmark::DoNotOptimize(upper);
    }

    state.counters["segments"] = Segments;
    state.counters["size"] = static_cast<double>(size);
}

// Benchmark std::equal_range for comparison
void BM_Std_EqualRange(benchmark::State& state, const std::string& distribution, std::uint64_t target) {
    const std::size_t size = state.range(0);

    auto it = distribution_generators.find(distribution);
    if (it == distribution_generators.end()) {
        state.SkipWithError("Unknown distribution");
        return;
    }

    auto data = get_or_generate_range_dataset(distribution, size, it->second);

    for (auto _ : state) {
        auto [lower, upper] = std::equal_range(data->begin(), data->end(), target);
        benchmark::DoNotOptimize(lower);
        benchmark::DoNotOptimize(upper);
    }

    state.counters["size"] = static_cast<double>(size);
}

// Benchmark JazzyIndex::find_lower_bound
template <std::size_t Segments>
void BM_JazzyIndex_LowerBound(benchmark::State& state, const std::string& distribution, std::uint64_t target) {
    const std::size_t size = state.range(0);

    auto it = distribution_generators.find(distribution);
    if (it == distribution_generators.end()) {
        state.SkipWithError("Unknown distribution");
        return;
    }

    auto data = get_or_generate_range_dataset(distribution, size, it->second);
    auto index = qi::bench::make_index<Segments>(*data);

    for (auto _ : state) {
        const auto* lower = index.find_lower_bound(target);
        benchmark::DoNotOptimize(lower);
    }

    state.counters["segments"] = Segments;
    state.counters["size"] = static_cast<double>(size);
}

// Benchmark std::lower_bound for comparison
void BM_Std_LowerBound(benchmark::State& state, const std::string& distribution, std::uint64_t target) {
    const std::size_t size = state.range(0);

    auto it = distribution_generators.find(distribution);
    if (it == distribution_generators.end()) {
        state.SkipWithError("Unknown distribution");
        return;
    }

    auto data = get_or_generate_range_dataset(distribution, size, it->second);

    for (auto _ : state) {
        auto lower = std::lower_bound(data->begin(), data->end(), target);
        benchmark::DoNotOptimize(lower);
    }

    state.counters["size"] = static_cast<double>(size);
}

// Benchmark JazzyIndex::find_upper_bound
template <std::size_t Segments>
void BM_JazzyIndex_UpperBound(benchmark::State& state, const std::string& distribution, std::uint64_t target) {
    const std::size_t size = state.range(0);

    auto it = distribution_generators.find(distribution);
    if (it == distribution_generators.end()) {
        state.SkipWithError("Unknown distribution");
        return;
    }

    auto data = get_or_generate_range_dataset(distribution, size, it->second);
    auto index = qi::bench::make_index<Segments>(*data);

    for (auto _ : state) {
        const auto* upper = index.find_upper_bound(target);
        benchmark::DoNotOptimize(upper);
    }

    state.counters["segments"] = Segments;
    state.counters["size"] = static_cast<double>(size);
}

// Benchmark std::upper_bound for comparison
void BM_Std_UpperBound(benchmark::State& state, const std::string& distribution, std::uint64_t target) {
    const std::size_t size = state.range(0);

    auto it = distribution_generators.find(distribution);
    if (it == distribution_generators.end()) {
        state.SkipWithError("Unknown distribution");
        return;
    }

    auto data = get_or_generate_range_dataset(distribution, size, it->second);

    for (auto _ : state) {
        auto upper = std::upper_bound(data->begin(), data->end(), target);
        benchmark::DoNotOptimize(upper);
    }

    state.counters["size"] = static_cast<double>(size);
}

}  // namespace

// Helper to iterate over all segment counts (matching main benchmarks)
template <typename F, std::size_t... Segments>
void for_each_segment_count_impl(F&& f, std::integer_sequence<std::size_t, Segments...>) {
    (f(std::integral_constant<std::size_t, Segments>{}), ...);
}

template <typename F>
void for_each_segment_count(F&& f) {
    for_each_segment_count_impl(std::forward<F>(f),
                                std::integer_sequence<std::size_t, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512>{});
}

// Helper template to register benchmarks for a specific segment count and scenario
template<std::size_t Segments>
void register_segment_benchmarks(const std::string& distribution, const std::string& scenario,
                                 std::uint64_t target_100, std::uint64_t target_1000, std::uint64_t target_10000) {
    // Create target map for different sizes
    std::unordered_map<std::size_t, std::uint64_t> targets = {
        {100, target_100},
        {1000, target_1000},
        {10000, target_10000}
    };

    // Register equal_range benchmarks with scenario-specific targets
    maybe_add_threads(benchmark::RegisterBenchmark(
        "BM_JazzyIndex_EqualRange_" + std::to_string(Segments) + "_" + distribution + "/" + scenario,
        [distribution, targets](benchmark::State& s) {
            auto it = targets.find(s.range(0));
            BM_JazzyIndex_EqualRange<Segments>(s, distribution, it->second);
        })
        ->RangeMultiplier(10)->Range(100, 10000)->Unit(benchmark::kNanosecond));

    // Register lower_bound benchmarks with scenario-specific targets
    maybe_add_threads(benchmark::RegisterBenchmark(
        "BM_JazzyIndex_LowerBound_" + std::to_string(Segments) + "_" + distribution + "/" + scenario,
        [distribution, targets](benchmark::State& s) {
            auto it = targets.find(s.range(0));
            BM_JazzyIndex_LowerBound<Segments>(s, distribution, it->second);
        })
        ->RangeMultiplier(10)->Range(100, 10000)->Unit(benchmark::kNanosecond));

    // Register upper_bound benchmarks with scenario-specific targets
    maybe_add_threads(benchmark::RegisterBenchmark(
        "BM_JazzyIndex_UpperBound_" + std::to_string(Segments) + "_" + distribution + "/" + scenario,
        [distribution, targets](benchmark::State& s) {
            auto it = targets.find(s.range(0));
            BM_JazzyIndex_UpperBound<Segments>(s, distribution, it->second);
        })
        ->RangeMultiplier(10)->Range(100, 10000)->Unit(benchmark::kNanosecond));
}

void register_benchmarks() {
    // Distributions to benchmark
    const std::vector<std::string> distributions = {
        "Uniform", "Exponential", "Clustered", "Lognormal", "Zipf",
        "Mixed", "Quadratic", "ExtremePoly", "InversePoly"
    };

    // Sizes to benchmark
    const std::vector<std::size_t> sizes = {100, 1000, 10000};

    // Register benchmarks for all distributions
    for (const auto& distribution : distributions) {
        auto it = distribution_generators.find(distribution);
        if (it == distribution_generators.end()) {
            continue;
        }

        // Generate datasets and compute targets for each scenario
        struct ScenarioTargets {
            std::uint64_t mid;
            std::uint64_t end;
            std::uint64_t miss;
        };
        std::unordered_map<std::size_t, ScenarioTargets> targets_by_size;

        for (std::size_t size : sizes) {
            auto data = get_or_generate_range_dataset(distribution, size, it->second);
            ScenarioTargets targets;
            targets.mid = (*data)[data->size() / 2];  // FoundMiddle
            targets.end = data->back();  // FoundEnd
            targets.miss = (data->back() == std::numeric_limits<std::uint64_t>::max())
                ? data->back() : data->back() + 1;  // NotFound
            targets_by_size[size] = targets;
        }

        // Register std:: baselines for each scenario
        for (const auto& scenario_name : {"FoundMiddle", "FoundEnd", "NotFound"}) {
            // Helper lambda to get the right target for each scenario
            auto get_target = [&targets_by_size, scenario_name](std::size_t size) -> std::uint64_t {
                const auto& targets = targets_by_size[size];
                if (std::string(scenario_name) == "FoundMiddle") return targets.mid;
                if (std::string(scenario_name) == "FoundEnd") return targets.end;
                return targets.miss;
            };

            std::unordered_map<std::size_t, std::uint64_t> scenario_targets;
            for (std::size_t size : sizes) {
                scenario_targets[size] = get_target(size);
            }

            maybe_add_threads(benchmark::RegisterBenchmark(
                "BM_Std_EqualRange/" + distribution + "/" + scenario_name,
                [distribution, scenario_targets](benchmark::State& s) {
                    auto it = scenario_targets.find(s.range(0));
                    BM_Std_EqualRange(s, distribution, it->second);
                })
                ->RangeMultiplier(10)->Range(100, 10000)->Unit(benchmark::kNanosecond));

            maybe_add_threads(benchmark::RegisterBenchmark(
                "BM_Std_LowerBound/" + distribution + "/" + scenario_name,
                [distribution, scenario_targets](benchmark::State& s) {
                    auto it = scenario_targets.find(s.range(0));
                    BM_Std_LowerBound(s, distribution, it->second);
                })
                ->RangeMultiplier(10)->Range(100, 10000)->Unit(benchmark::kNanosecond));

            maybe_add_threads(benchmark::RegisterBenchmark(
                "BM_Std_UpperBound/" + distribution + "/" + scenario_name,
                [distribution, scenario_targets](benchmark::State& s) {
                    auto it = scenario_targets.find(s.range(0));
                    BM_Std_UpperBound(s, distribution, it->second);
                })
                ->RangeMultiplier(10)->Range(100, 10000)->Unit(benchmark::kNanosecond));
        }

        // Register JazzyIndex benchmarks for all segment counts and scenarios
        const auto& targets_100 = targets_by_size[100];
        const auto& targets_1000 = targets_by_size[1000];
        const auto& targets_10000 = targets_by_size[10000];

        for_each_segment_count([&](auto seg_tag) {
            constexpr std::size_t Segments = decltype(seg_tag)::value;

            // FoundMiddle scenario
            register_segment_benchmarks<Segments>(distribution, "FoundMiddle",
                targets_100.mid, targets_1000.mid, targets_10000.mid);

            // FoundEnd scenario
            register_segment_benchmarks<Segments>(distribution, "FoundEnd",
                targets_100.end, targets_1000.end, targets_10000.end);

            // NotFound scenario
            register_segment_benchmarks<Segments>(distribution, "NotFound",
                targets_100.miss, targets_1000.miss, targets_10000.miss);
        });
    }
}

int main(int argc, char** argv) {
    // Parse custom flags before initializing benchmark library
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.find("--benchmark_threads=") == 0) {
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

    register_benchmarks();

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
