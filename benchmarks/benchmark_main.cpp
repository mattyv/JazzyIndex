#include <benchmark/benchmark.h>

#include <memory>
#include <string>
#include <vector>

#include "fixtures.hpp"

namespace {

template <std::size_t Segments>
void register_uniform_suite(std::size_t size) {
    if (size == 0) {
        return;
    }

    auto data = std::make_shared<std::vector<float>>(qi::bench::make_uniform_values(size));
    auto random_queries = std::make_shared<std::vector<float>>(
        qi::bench::make_random_queries(*data, qi::bench::kQueryCount));

    const float mid_target = (*data)[data->size() / 2];
    const float end_target = data->back();
    const float miss_target = end_target + 1.0f;

    const std::string base =
        "QuantileIndex/Uniform/S" + std::to_string(Segments) + "/N" + std::to_string(size);

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

    benchmark::RegisterBenchmark((base + "/RandomHits").c_str(),
                                 [data, random_queries](benchmark::State& state) {
                                     auto index = qi::bench::make_index<Segments>(*data);
                                     for (auto _ : state) {
                                         for (const float query : *random_queries) {
                                             const auto* result = index.find(query);
                                             benchmark::DoNotOptimize(result);
                                         }
                                     }
                                     state.counters["segments"] = Segments;
                                     state.counters["size"] = static_cast<double>(data->size());
                                     state.SetItemsProcessed(
                                         static_cast<int64_t>(state.iterations()) *
                                         static_cast<int64_t>(random_queries->size()));
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

    auto data = std::make_shared<std::vector<float>>(generator(size));
    if (data->empty()) {
        return;
    }

    auto random_queries = std::make_shared<std::vector<float>>(
        qi::bench::make_random_queries(*data, qi::bench::kQueryCount));

    const float target = (*data)[data->size() / 3];
    const std::string base = "QuantileIndex/" + name + "/S" + std::to_string(Segments) +
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

    benchmark::RegisterBenchmark((base + "/RandomHits").c_str(),
                                 [data, random_queries](benchmark::State& state) {
                                     auto index = qi::bench::make_index<Segments>(*data);
                                     for (auto _ : state) {
                                         for (const float query : *random_queries) {
                                             const auto* result = index.find(query);
                                             benchmark::DoNotOptimize(result);
                                         }
                                     }
                                     state.counters["segments"] = Segments;
                                     state.counters["size"] = static_cast<double>(data->size());
                                     state.SetItemsProcessed(
                                         static_cast<int64_t>(state.iterations()) *
                                         static_cast<int64_t>(random_queries->size()));
                                 })
        ->Unit(benchmark::kNanosecond);
}

void register_uniform_suites() {
    const std::vector<std::size_t> sizes = {1'000, 100'000};

    for (const std::size_t size : sizes) {
        register_uniform_suite<64>(size);
        register_uniform_suite<128>(size);
        register_uniform_suite<256>(size);
        register_uniform_suite<512>(size);
    }
}

void register_distribution_suites() {
    constexpr std::size_t size = 100'000;

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

}  // namespace

int main(int argc, char** argv) {
    register_uniform_suites();
    register_distribution_suites();

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
