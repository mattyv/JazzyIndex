#pragma once

#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "dataset_generators.hpp"
#include "quantile_index.hpp"

namespace qi::bench {

constexpr std::size_t kQueryCount = 1024;
constexpr double kRandomHitRatio = 0.9;
constexpr unsigned kRandomSeed = 1337u;

inline std::vector<float> make_uniform_values(std::size_t size,
                                              float start = 0.0f,
                                              float step = 1.0f) {
    std::vector<float> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = start + step * static_cast<float>(i);
    }
    return values;
}

inline std::vector<float> make_random_queries(const std::vector<float>& values,
                                              std::size_t count,
                                              double hit_ratio = kRandomHitRatio,
                                              unsigned seed = kRandomSeed) {
    if (values.empty()) {
        return {};
    }

    std::mt19937_64 rng(seed);
    std::bernoulli_distribution hit_dist(hit_ratio);
    std::uniform_int_distribution<std::size_t> index_dist(0, values.size() - 1);
    std::uniform_real_distribution<float> miss_dist(values.back() + 1.0f,
                                                    values.back() + 1024.0f);

    std::vector<float> queries(count);
    for (auto& query : queries) {
        if (hit_dist(rng)) {
            query = values[index_dist(rng)];
        } else {
            query = miss_dist(rng);
        }
    }
    return queries;
}

inline std::vector<float> make_exponential_values(std::size_t size) {
    return dataset::generate_exponential(size,
                                         dataset::kExponentialScale,
                                         dataset::kSeed,
                                         0.0f,
                                         static_cast<float>(size));
}

inline std::vector<float> make_clustered_values(std::size_t size) {
    return dataset::generate_clustered(size,
                                       dataset::kClusterCount,
                                       dataset::kClusterSpread,
                                       dataset::kSeed,
                                       0.0f,
                                       static_cast<float>(size));
}

inline std::vector<float> make_lognormal_values(std::size_t size) {
    return dataset::generate_lognormal(size,
                                       dataset::kLognormalMean,
                                       dataset::kLognormalSigma,
                                       dataset::kSeed,
                                       0.0f,
                                       static_cast<float>(size));
}

inline std::vector<float> make_zipf_values(std::size_t size) {
    return dataset::generate_zipf(size,
                                  dataset::kZipfA,
                                  dataset::kZipfMax,
                                  dataset::kSeed,
                                  0.0f,
                                  static_cast<float>(size));
}

inline std::vector<float> make_mixed_values(std::size_t size) {
    return dataset::generate_mixed(size,
                                   dataset::kMixedRatio,
                                   dataset::kExponentialScale,
                                   dataset::kClusterCount,
                                   dataset::kClusterSpread,
                                   dataset::kSeed,
                                   0.0f,
                                   static_cast<float>(size));
}

template <std::size_t Segments>
inline bucket_index::QuantileIndex<float, Segments> make_index(
    const std::vector<float>& values) {
    bucket_index::QuantileIndex<float, Segments> index;
    if (!values.empty()) {
        index.build(values.data(), values.data() + values.size());
    }
    return index;
}

}  // namespace qi::bench
