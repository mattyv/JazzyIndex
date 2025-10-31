#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace dataset {

constexpr unsigned kSeed = 42u;
constexpr float kExponentialScale = 1.0f;
constexpr std::size_t kClusterCount = 5;
constexpr float kClusterSpread = 0.02f;
constexpr float kLognormalMean = 0.0f;
constexpr float kLognormalSigma = 0.9f;
constexpr double kZipfA = 1.2;
constexpr std::size_t kZipfMax = 1'000'000;
constexpr float kMixedRatio = 0.5f;

inline std::vector<float> finalize_samples(std::vector<float>&& samples,
                                           float min_value,
                                           float max_value) {
    if (samples.empty()) {
        return samples;
    }

    std::sort(samples.begin(), samples.end());
    float prev = std::clamp(samples.front(), min_value, max_value);
    samples[0] = prev;
    for (std::size_t idx = 1; idx < samples.size(); ++idx) {
        float value = std::clamp(samples[idx], min_value, max_value);
        if (value < prev) {
            value = prev;
        }
        samples[idx] = value;
        prev = value;
    }
    return samples;
}

inline std::vector<float> generate_exponential(std::size_t size,
                                               float scale,
                                               unsigned seed,
                                               float min_value,
                                               float max_value) {
    std::mt19937_64 rng(seed);
    std::exponential_distribution<float> dist(1.0f / std::max(scale, 1e-6f));
    std::vector<float> samples(size);
    for (auto& value : samples) {
        value = dist(rng);
    }
    return finalize_samples(std::move(samples), min_value, max_value);
}

inline std::vector<float> generate_clustered(std::size_t size,
                                             std::size_t cluster_count,
                                             float cluster_spread,
                                             unsigned seed,
                                             float min_value,
                                             float max_value) {
    cluster_count = std::max<std::size_t>(cluster_count, 1);
    std::mt19937_64 rng(seed);
    std::vector<float> samples;
    samples.reserve(size);

    const std::size_t base = size / cluster_count;
    const std::size_t remainder = size % cluster_count;

    for (std::size_t cluster = 0; cluster < cluster_count; ++cluster) {
        const std::size_t chunk = base + (cluster < remainder ? 1 : 0);
        if (chunk == 0) {
            continue;
        }
        const float center = cluster_count == 1
                                 ? 0.0f
                                 : static_cast<float>(cluster) /
                                       static_cast<float>(cluster_count - 1);
        std::normal_distribution<float> dist(center, std::max(cluster_spread, 1e-4f));
        for (std::size_t idx = 0; idx < chunk; ++idx) {
            samples.push_back(dist(rng));
        }
    }

    if (samples.empty()) {
        samples.push_back(min_value);
    }

    return finalize_samples(std::move(samples), min_value, max_value);
}

inline std::vector<float> generate_lognormal(std::size_t size,
                                             float mean,
                                             float sigma,
                                             unsigned seed,
                                             float min_value,
                                             float max_value) {
    std::mt19937_64 rng(seed);
    std::lognormal_distribution<float> dist(mean, std::max(sigma, 1e-3f));
    std::vector<float> samples(size);
    for (auto& value : samples) {
        value = dist(rng);
    }
    return finalize_samples(std::move(samples), min_value, max_value);
}

inline std::vector<float> generate_zipf(std::size_t size,
                                        double a,
                                        std::size_t max_rank,
                                        unsigned seed,
                                        float min_value,
                                        float clamp_upper) {
    max_rank = std::max<std::size_t>(max_rank, 1);
    std::vector<double> weights(max_rank);
    for (std::size_t idx = 0; idx < max_rank; ++idx) {
        weights[idx] = 1.0 / std::pow(static_cast<double>(idx + 1), a);
    }

    std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
    std::mt19937_64 rng(seed);

    std::vector<float> samples(size);
    for (auto& value : samples) {
        value = static_cast<float>(dist(rng) + 1);
    }

    return finalize_samples(std::move(samples), min_value, clamp_upper);
}

inline std::vector<float> generate_mixed(std::size_t size,
                                         float exponential_ratio,
                                         float exp_scale,
                                         std::size_t cluster_count,
                                         float cluster_spread,
                                         unsigned seed,
                                         float min_value,
                                         float max_value) {
    exponential_ratio = std::clamp(exponential_ratio, 0.0f, 1.0f);
    const std::size_t exp_size =
        exponential_ratio <= 0.0f ? 0 : std::max<std::size_t>(1, static_cast<std::size_t>(size * exponential_ratio));
    const std::size_t cluster_size = size > exp_size ? size - exp_size : 0;

    std::vector<float> samples;
    samples.reserve(size);

    if (exp_size > 0) {
        auto exp_samples = generate_exponential(exp_size, exp_scale, seed, min_value, max_value);
        samples.insert(samples.end(), exp_samples.begin(), exp_samples.end());
    }
    if (cluster_size > 0) {
        auto cluster_samples =
            generate_clustered(cluster_size, cluster_count, cluster_spread, seed + 1, min_value, max_value);
        samples.insert(samples.end(), cluster_samples.begin(), cluster_samples.end());
    }

    return finalize_samples(std::move(samples), min_value, max_value);
}

}  // namespace dataset

