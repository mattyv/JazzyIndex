#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <sys/stat.h>

#include "fixtures.hpp"
#include "jazzy_index_export.hpp"

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

// Build time benchmarks
template <std::size_t Segments>
void register_build_benchmark(std::size_t size, const std::string& distribution,
                               const std::vector<std::uint64_t>& data) {
    const std::string name = "JazzyIndexBuild/" + distribution + "/S" +
                            std::to_string(Segments) + "/N" + std::to_string(size);

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
        ->Unit(benchmark::kMicrosecond);
}

void register_build_suites() {
    const std::vector<std::size_t> sizes = {1'000, 10'000, 100'000, 1'000'000};

    for (const std::size_t size : sizes) {
        auto uniform_data = qi::bench::make_uniform_values(size);
        register_build_benchmark<64>(size, "Uniform", uniform_data);
        register_build_benchmark<128>(size, "Uniform", uniform_data);
        register_build_benchmark<256>(size, "Uniform", uniform_data);
        register_build_benchmark<512>(size, "Uniform", uniform_data);

        auto exp_data = qi::bench::make_exponential_values(size);
        register_build_benchmark<256>(size, "Exponential", exp_data);

        auto zipf_data = qi::bench::make_zipf_values(size);
        register_build_benchmark<256>(size, "Zipf", zipf_data);
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

// Export visualization data for various index configurations
void export_visualization_data(const std::string& output_dir) {
    std::cout << "Exporting index visualization data to " << output_dir << "..." << std::endl;

    if (!ensure_directory_exists(output_dir)) {
        std::cerr << "Warning: Could not create directory " << output_dir << std::endl;
        std::cerr << "Attempting to write to current directory instead." << std::endl;
    }

    // Configuration: which distributions, sizes, and segment counts to visualize
    const std::vector<std::size_t> viz_sizes = {100, 1'000, 10'000};
    const std::vector<std::size_t> viz_segments = {64, 128, 256, 512};

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

                if (segments == 64) {
                    auto index = qi::bench::make_index<64>(data);
                    json_data = jazzy::export_index_metadata(index);
                } else if (segments == 128) {
                    auto index = qi::bench::make_index<128>(data);
                    json_data = jazzy::export_index_metadata(index);
                } else if (segments == 256) {
                    auto index = qi::bench::make_index<256>(data);
                    json_data = jazzy::export_index_metadata(index);
                } else if (segments == 512) {
                    auto index = qi::bench::make_index<512>(data);
                    json_data = jazzy::export_index_metadata(index);
                } else {
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
    // Check for --visualize-index flag before benchmark initialization
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
        }
    }

    if (visualize_mode) {
        export_visualization_data(output_dir);
        std::cout << "\nVisualization data exported. Now run:" << std::endl;
        std::cout << "  python3 scripts/plot_index_structure.py " << output_dir << std::endl;
        return 0;
    }

    // Register baseline std::lower_bound benchmarks first
    register_lower_bound_suites();

    // Register JazzyIndex query benchmarks
    register_uniform_suites();
    register_distribution_suites();

    // Register JazzyIndex build time benchmarks
    register_build_suites();

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
