#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "fixtures.hpp"

namespace {

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

    benchmark::RegisterBenchmark((base + "/FoundMiddle").c_str(),
                                 [data, mid_target](benchmark::State& state) {
                                     for (auto _ : state) {
                                         const auto* result = find_with_lower_bound(*data, mid_target);
                                         benchmark::DoNotOptimize(result);
                                     }
                                     state.counters["size"] = static_cast<double>(data->size());
                                 })
        ->Unit(benchmark::kNanosecond);

    benchmark::RegisterBenchmark((base + "/FoundEnd").c_str(),
                                 [data, end_target](benchmark::State& state) {
                                     for (auto _ : state) {
                                         const auto* result = find_with_lower_bound(*data, end_target);
                                         benchmark::DoNotOptimize(result);
                                     }
                                     state.counters["size"] = static_cast<double>(data->size());
                                 })
        ->Unit(benchmark::kNanosecond);

    benchmark::RegisterBenchmark((base + "/NotFound").c_str(),
                                 [data, miss_target](benchmark::State& state) {
                                     for (auto _ : state) {
                                         const auto* result = find_with_lower_bound(*data, miss_target);
                                         benchmark::DoNotOptimize(result);
                                     }
                                     state.counters["size"] = static_cast<double>(data->size());
                                 })
        ->Unit(benchmark::kNanosecond);
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

    const std::uint64_t target = (*data)[data->size() / 3];
    const std::string base = "LowerBound/" + name + "/N" + std::to_string(size);

    benchmark::RegisterBenchmark((base + "/Found").c_str(),
                                 [data, target](benchmark::State& state) {
                                     for (auto _ : state) {
                                         const auto* result = find_with_lower_bound(*data, target);
                                         benchmark::DoNotOptimize(result);
                                     }
                                     state.counters["size"] = static_cast<double>(data->size());
                                 })
        ->Unit(benchmark::kNanosecond);
}

void register_lower_bound_suites() {
    const std::vector<std::size_t> sizes = {100, 1'000, 10'000, 100'000, 1'000'000};

    for (const std::size_t size : sizes) {
        register_lower_bound_uniform_suite(size);
    }

    const std::vector<std::size_t> dist_sizes = {100, 10'000, 100'000, 1'000'000};
    for (const std::size_t size : dist_sizes) {
        register_lower_bound_distribution_suite("Exponential", qi::bench::make_exponential_values, size);
        register_lower_bound_distribution_suite("Clustered", qi::bench::make_clustered_values, size);
        register_lower_bound_distribution_suite("Lognormal", qi::bench::make_lognormal_values, size);
        register_lower_bound_distribution_suite("Zipf", qi::bench::make_zipf_values, size);
        register_lower_bound_distribution_suite("Mixed", qi::bench::make_mixed_values, size);
    }
}

// JazzyIndex benchmarks

template <std::size_t Segments>
void register_uniform_suite(std::size_t size) {
    if (size == 0) {
        return;
    }

    auto data = std::make_shared<std::vector<std::uint64_t>>(qi::bench::make_uniform_values(size));
    const std::uint64_t mid_target = (*data)[data->size() / 2];
    const std::uint64_t end_target = data->back();
    const std::uint64_t miss_target =
        end_target == std::numeric_limits<std::uint64_t>::max() ? end_target : end_target + 1;

    const std::string base =
        "JazzyIndex/Uniform/S" + std::to_string(Segments) + "/N" + std::to_string(size);

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
        ->Unit(benchmark::kNanosecond);

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
        ->Unit(benchmark::kNanosecond);

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
        ->Unit(benchmark::kNanosecond);

}

template <std::size_t Segments, typename Generator>
void register_distribution_suite(const std::string& name,
                                 Generator&& generator,
                                 std::size_t size) {
    if (size == 0) {
        return;
    }

    auto data = std::make_shared<std::vector<std::uint64_t>>(generator(size));
    if (data->empty()) {
        return;
    }

    const std::uint64_t target = (*data)[data->size() / 3];
    const std::string base = "JazzyIndex/" + name + "/S" + std::to_string(Segments) +
                             "/N" + std::to_string(size);

    benchmark::RegisterBenchmark((base + "/Found").c_str(),
                                 [data, target](benchmark::State& state) {
                                     auto index = qi::bench::make_index<Segments>(*data);
                                     for (auto _ : state) {
                                         const auto* result = index.find(target);
                                         benchmark::DoNotOptimize(result);
                                     }
                                     state.counters["segments"] = Segments;
                                     state.counters["size"] = static_cast<double>(data->size());
                                 })
        ->Unit(benchmark::kNanosecond);

}

void register_uniform_suites() {
    const std::vector<std::size_t> sizes = {100, 1'000, 10'000, 100'000, 1'000'000};

    for (const std::size_t size : sizes) {
        register_uniform_suite<64>(size);
        register_uniform_suite<128>(size);
        register_uniform_suite<256>(size);
        register_uniform_suite<512>(size);
    }
}

void register_distribution_suites() {
    const std::vector<std::size_t> sizes = {100, 10'000, 100'000, 1'000'000};

    for (const std::size_t size : sizes) {
        register_distribution_suite<64>("Exponential", qi::bench::make_exponential_values, size);
        register_distribution_suite<128>("Exponential", qi::bench::make_exponential_values, size);
        register_distribution_suite<256>("Exponential", qi::bench::make_exponential_values, size);
        register_distribution_suite<512>("Exponential", qi::bench::make_exponential_values, size);

        register_distribution_suite<64>("Clustered", qi::bench::make_clustered_values, size);
        register_distribution_suite<128>("Clustered", qi::bench::make_clustered_values, size);
        register_distribution_suite<256>("Clustered", qi::bench::make_clustered_values, size);
        register_distribution_suite<512>("Clustered", qi::bench::make_clustered_values, size);

        register_distribution_suite<64>("Lognormal", qi::bench::make_lognormal_values, size);
        register_distribution_suite<128>("Lognormal", qi::bench::make_lognormal_values, size);
        register_distribution_suite<256>("Lognormal", qi::bench::make_lognormal_values, size);
        register_distribution_suite<512>("Lognormal", qi::bench::make_lognormal_values, size);

        register_distribution_suite<64>("Zipf", qi::bench::make_zipf_values, size);
        register_distribution_suite<128>("Zipf", qi::bench::make_zipf_values, size);
        register_distribution_suite<256>("Zipf", qi::bench::make_zipf_values, size);
        register_distribution_suite<512>("Zipf", qi::bench::make_zipf_values, size);

        register_distribution_suite<64>("Mixed", qi::bench::make_mixed_values, size);
        register_distribution_suite<128>("Mixed", qi::bench::make_mixed_values, size);
        register_distribution_suite<256>("Mixed", qi::bench::make_mixed_values, size);
        register_distribution_suite<512>("Mixed", qi::bench::make_mixed_values, size);
    }
}

}  // namespace

int main(int argc, char** argv) {
    // Register baseline std::lower_bound benchmarks first
    register_lower_bound_suites();

    // Then register JazzyIndex benchmarks
    register_uniform_suites();
    register_distribution_suites();

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
