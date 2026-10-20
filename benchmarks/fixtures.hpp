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
    std::vector<std::size_t> gaps;
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i - 1] != std::numeric_limits<std::uint64_t>::max() &&
            values[i - 1] + 1 < values[i]) {
            gaps.push_back(i);
        }
    }
    std::uniform_int_distribution<std::size_t> gap_dist(
        0, gaps.empty() ? 0 : gaps.size() - 1);

    std::vector<std::uint64_t> queries(count);
    for (auto& query : queries) {
        if (hit_dist(rng)) {
            query = values[index_dist(rng)];
        } else if (!gaps.empty()) {
            const std::size_t gap = gaps[gap_dist(rng)];
            std::uniform_int_distribution<std::uint64_t> miss_dist(
                values[gap - 1] + 1, values[gap] - 1);
            query = miss_dist(rng);
        } else if (values.front() > 0) {
            query = values.front() - 1;
        } else if (values.back() < std::numeric_limits<std::uint64_t>::max()) {
            query = values.back() + 1;
        } else {
            query = values.front();
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
