#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "dataset_generators.hpp"
#include "jazzy_index.hpp"

namespace qi::bench {

constexpr std::size_t kQueryCount = 1024;
constexpr double kRandomHitRatio = 0.9;
constexpr unsigned kRandomSeed = 1337u;

inline std::vector<std::uint64_t> make_uniform_values(std::size_t size,
                                                      std::uint64_t start = 0,
                                                      std::uint64_t step = 1) {
    std::vector<std::uint64_t> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = start + step * static_cast<std::uint64_t>(i);
    }
    return values;
}

inline std::vector<std::uint64_t> make_random_queries(const std::vector<std::uint64_t>& values,
                                                      std::size_t count,
                                                      double hit_ratio = kRandomHitRatio,
                                                      unsigned seed = kRandomSeed) {
    if (values.empty()) {
        return {};
    }

    std::mt19937_64 rng(seed);
    std::bernoulli_distribution hit_dist(hit_ratio);
    std::uniform_int_distribution<std::size_t> index_dist(0, values.size() - 1);
    const std::uint64_t max_gap = 1024;
    const std::uint64_t upper_bound =
        values.back() > std::numeric_limits<std::uint64_t>::max() - max_gap
            ? std::numeric_limits<std::uint64_t>::max()
            : values.back() + max_gap;
    const std::uint64_t lower_bound =
        values.back() == std::numeric_limits<std::uint64_t>::max()
            ? values.back()
            : values.back() + 1;
    std::uniform_int_distribution<std::uint64_t> miss_dist(lower_bound, upper_bound);

    std::vector<std::uint64_t> queries(count);
    for (auto& query : queries) {
        if (hit_dist(rng)) {
            query = values[index_dist(rng)];
        } else {
            query = miss_dist(rng);
        }
    }
    return queries;
}

inline std::vector<std::uint64_t> make_exponential_values(std::size_t size) {
    return dataset::generate_exponential(size,
                                         dataset::kExponentialScale,
                                         dataset::kSeed,
                                         0,
                                         static_cast<std::uint64_t>(size));
}

inline std::vector<std::uint64_t> make_clustered_values(std::size_t size) {
    return dataset::generate_clustered(size,
                                       dataset::kClusterCount,
                                       dataset::kClusterSpread,
                                       dataset::kSeed,
                                       0,
                                       static_cast<std::uint64_t>(size));
}

inline std::vector<std::uint64_t> make_lognormal_values(std::size_t size) {
    return dataset::generate_lognormal(size,
                                       dataset::kLognormalMean,
                                       dataset::kLognormalSigma,
                                       dataset::kSeed,
                                       0,
                                       static_cast<std::uint64_t>(size));
}

inline std::vector<std::uint64_t> make_zipf_values(std::size_t size) {
    return dataset::generate_zipf(size,
                                  dataset::kZipfA,
                                  dataset::kZipfMax,
                                  dataset::kSeed,
                                  0,
                                  static_cast<std::uint64_t>(size));
}

inline std::vector<std::uint64_t> make_mixed_values(std::size_t size) {
    return dataset::generate_mixed(size,
                                   dataset::kMixedRatio,
                                   dataset::kExponentialScale,
                                   dataset::kClusterCount,
                                   dataset::kClusterSpread,
                                   dataset::kSeed,
                                   0,
                                   static_cast<std::uint64_t>(size));
}

inline std::vector<std::uint64_t> make_quadratic_values(std::size_t size) {
    return dataset::generate_quadratic(size,
                                       dataset::kSeed,
                                       0,
                                       static_cast<std::uint64_t>(size));
}

inline std::vector<std::uint64_t> make_extreme_polynomial_values(std::size_t size) {
    return dataset::generate_extreme_polynomial(size,
                                                dataset::kSeed,
                                                0,
                                                static_cast<std::uint64_t>(size));
}

inline std::vector<std::uint64_t> make_inverse_polynomial_values(std::size_t size) {
    return dataset::generate_inverse_polynomial(size,
                                                dataset::kSeed,
                                                0,
                                                static_cast<std::uint64_t>(size));
}

inline std::vector<std::uint64_t> load_real_world_dataset(const std::string& name,
                                                            std::size_t max_elements = 0) {
    // Try to load from benchmarks/datasets/ directory
    const std::string base_path = "benchmarks/datasets/";
    const std::string file_path = base_path + name;

    auto data = dataset::load_binary_file(file_path.c_str(), max_elements);

    if (data.empty()) {
        std::cerr << "Warning: Failed to load real-world dataset: " << file_path << std::endl;
        std::cerr << "         Download with: python3 scripts/download_sosd_dataset.py wiki" << std::endl;
        std::cerr << "         Or try: osm, fb, books" << std::endl;
    }

    return data;
}

// Generate random target values from a dataset for benchmarking
// Uses fixed seed for reproducibility
inline std::vector<std::uint64_t> generate_random_targets(
    const std::vector<std::uint64_t>& data,
    std::size_t num_targets = 1000,
    std::uint64_t seed = 12345) {

    std::vector<std::uint64_t> targets;
    targets.reserve(num_targets);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> dist(0, data.size() - 1);

    for (std::size_t i = 0; i < num_targets; ++i) {
        targets.push_back(data[dist(rng)]);
    }

    return targets;
}

template <std::size_t Segments>
inline jazzy::JazzyIndex<std::uint64_t, jazzy::to_segment_count<Segments>()> make_index(
    const std::vector<std::uint64_t>& values) {
    jazzy::JazzyIndex<std::uint64_t, jazzy::to_segment_count<Segments>()> index;
    if (!values.empty()) {
        index.build(values.data(), values.data() + values.size());
    }
    return index;
}

}  // namespace qi::bench
