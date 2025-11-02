#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace dataset {

constexpr unsigned kSeed = 42u;
constexpr double kExponentialScale = 1.0;
constexpr std::size_t kClusterCount = 5;
constexpr double kClusterSpread = 0.02;
constexpr double kLognormalMean = 0.0;
constexpr double kLognormalSigma = 0.9;
constexpr double kZipfA = 1.2;
constexpr std::size_t kZipfMax = 1'000'000;
constexpr double kMixedRatio = 0.5;

inline std::vector<std::uint64_t> finalize_samples(std::vector<double>&& samples,
                                                   std::uint64_t min_value,
                                                   std::uint64_t max_value) {
    if (samples.empty()) {
        return {};
    }

    std::sort(samples.begin(), samples.end());
    std::vector<std::uint64_t> result(samples.size());

    const double min_d = static_cast<double>(min_value);
    const double max_d = static_cast<double>(max_value);

    double clamped = std::clamp(samples.front(), min_d, max_d);
    std::uint64_t prev = static_cast<std::uint64_t>(std::llround(clamped));
    prev = std::clamp(prev, min_value, max_value);
    result[0] = prev;

    for (std::size_t idx = 1; idx < samples.size(); ++idx) {
        double value = std::clamp(samples[idx], min_d, max_d);
        if (value < static_cast<double>(prev)) {
            value = static_cast<double>(prev);
        }
        std::uint64_t current =
            static_cast<std::uint64_t>(std::llround(value));
        current = std::clamp(current, prev, max_value);
        result[idx] = current;
        prev = current;
    }
    return result;
}

inline std::vector<std::uint64_t> generate_exponential(std::size_t size,
                                                       double scale,
                                                       unsigned seed,
                                                       std::uint64_t min_value,
                                                       std::uint64_t max_value) {
    std::mt19937_64 rng(seed);
    std::exponential_distribution<double> dist(1.0 / std::max(scale, 1e-6));
    std::vector<double> samples(size);
    for (auto& value : samples) {
        value = dist(rng);
    }
    return finalize_samples(std::move(samples), min_value, max_value);
}

inline std::vector<std::uint64_t> generate_clustered(std::size_t size,
                                                     std::size_t cluster_count,
                                                     double cluster_spread,
                                                     unsigned seed,
                                                     std::uint64_t min_value,
                                                     std::uint64_t max_value) {
    cluster_count = std::max<std::size_t>(cluster_count, 1);
    std::mt19937_64 rng(seed);
    std::vector<double> samples;
    samples.reserve(size);

    const std::size_t base = size / cluster_count;
    const std::size_t remainder = size % cluster_count;

    for (std::size_t cluster = 0; cluster < cluster_count; ++cluster) {
        const std::size_t chunk = base + (cluster < remainder ? 1 : 0);
        if (chunk == 0) {
            continue;
        }
        const double center = cluster_count == 1
                                  ? 0.0
                                  : static_cast<double>(cluster) /
                                        static_cast<double>(cluster_count - 1);
        std::normal_distribution<double> dist(center, std::max(cluster_spread, 1e-4));
        for (std::size_t idx = 0; idx < chunk; ++idx) {
            samples.push_back(dist(rng));
        }
    }

    if (samples.empty()) {
        samples.push_back(min_value);
    }

    return finalize_samples(std::move(samples), min_value, max_value);
}

inline std::vector<std::uint64_t> generate_lognormal(std::size_t size,
                                                     double mean,
                                                     double sigma,
                                                     unsigned seed,
                                                     std::uint64_t min_value,
                                                     std::uint64_t max_value) {
    std::mt19937_64 rng(seed);
    std::lognormal_distribution<double> dist(mean, std::max(sigma, 1e-3));
    std::vector<double> samples(size);
    for (auto& value : samples) {
        value = dist(rng);
    }
    return finalize_samples(std::move(samples), min_value, max_value);
}

inline std::vector<std::uint64_t> generate_zipf(std::size_t size,
                                                double a,
                                                std::size_t max_rank,
                                                unsigned seed,
                                                std::uint64_t min_value,
                                                std::uint64_t clamp_upper) {
    max_rank = std::max<std::size_t>(max_rank, 1);
    std::vector<double> weights(max_rank);
    for (std::size_t idx = 0; idx < max_rank; ++idx) {
        weights[idx] = 1.0 / std::pow(static_cast<double>(idx + 1), a);
    }

    std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
    std::mt19937_64 rng(seed);

    std::vector<double> samples(size);
    for (auto& value : samples) {
        value = static_cast<double>(dist(rng) + 1);
    }

    return finalize_samples(std::move(samples), min_value, clamp_upper);
}

inline std::vector<std::uint64_t> generate_mixed(std::size_t size,
                                                 double exponential_ratio,
                                                 double exp_scale,
                                                 std::size_t cluster_count,
                                                 double cluster_spread,
                                                 unsigned seed,
                                                 std::uint64_t min_value,
                                                 std::uint64_t max_value) {
    exponential_ratio = std::clamp(exponential_ratio, 0.0, 1.0);
    const std::size_t exp_size = exponential_ratio <= 0.0
                                     ? 0
                                     : std::max<std::size_t>(1, static_cast<std::size_t>(size * exponential_ratio));
    const std::size_t cluster_size = size > exp_size ? size - exp_size : 0;

    std::vector<double> samples;
    samples.reserve(size);

    if (exp_size > 0) {
        std::mt19937_64 exp_rng(seed);
        std::exponential_distribution<double> exp_dist(1.0 / std::max(exp_scale, 1e-6));
        for (std::size_t i = 0; i < exp_size; ++i) {
            samples.push_back(exp_dist(exp_rng));
        }
    }
    if (cluster_size > 0) {
        const std::size_t adjusted_cluster_count = std::max<std::size_t>(cluster_count, 1);
        std::mt19937_64 cluster_rng(seed + 1);

        const std::size_t base = cluster_size / adjusted_cluster_count;
        const std::size_t remainder = cluster_size % adjusted_cluster_count;

        for (std::size_t cluster = 0; cluster < adjusted_cluster_count; ++cluster) {
            const std::size_t chunk = base + (cluster < remainder ? 1 : 0);
            if (chunk == 0) {
                continue;
            }
            const double center = adjusted_cluster_count == 1
                                      ? 0.0
                                      : static_cast<double>(cluster) /
                                            static_cast<double>(adjusted_cluster_count - 1);
            std::normal_distribution<double> cluster_dist(center, std::max(cluster_spread, 1e-4));
            for (std::size_t idx = 0; idx < chunk; ++idx) {
                samples.push_back(cluster_dist(cluster_rng));
            }
        }

        if (samples.empty()) {
            samples.push_back(static_cast<double>(min_value));
        }
    }

    return finalize_samples(std::move(samples), min_value, max_value);
}

inline std::vector<std::uint64_t> generate_quadratic(std::size_t size,
                                                      unsigned seed,
                                                      std::uint64_t min_value,
                                                      std::uint64_t max_value) {
    if (size == 0) {
        return {};
    }

    std::vector<std::uint64_t> result;
    result.reserve(size);

    // Generate S-curve using tanh to create strong curvature in middle segments
    // This creates regions where linear models fail but quadratic models work well
    const std::uint64_t range = max_value - min_value;
    const double size_d = static_cast<double>(size);

    for (std::size_t i = 0; i < size; ++i) {
        // Map index to [-3, 3] range for tanh (creates S-curve)
        const double t = (static_cast<double>(i) / (size_d - 1.0)) * 6.0 - 3.0;
        // tanh creates smooth S-curve: slow start, fast middle, slow end
        // Map from [-1, 1] to [0, range]
        const double normalized = (std::tanh(t) + 1.0) / 2.0;
        const std::uint64_t value = min_value + static_cast<std::uint64_t>(normalized * range);
        result.push_back(std::min(value, max_value));
    }

    return result;
}

inline std::vector<std::uint64_t> generate_extreme_polynomial(std::size_t size,
                                                                unsigned seed,
                                                                std::uint64_t min_value,
                                                                std::uint64_t max_value) {
    if (size == 0) {
        return {};
    }

    std::vector<std::uint64_t> result;
    result.reserve(size);

    // Generate extreme polynomial: x^5 creates very sharp curvature
    // This should force quadratic models in curved regions
    const std::uint64_t range = max_value - min_value;
    const double size_d = static_cast<double>(size);

    for (std::size_t i = 0; i < size; ++i) {
        // Normalize to [0, 1]
        const double t = static_cast<double>(i) / (size_d - 1.0);
        // Apply x^5 for extreme curvature - very slow start, explosive end
        const double normalized = std::pow(t, 5.0);
        const std::uint64_t value = min_value + static_cast<std::uint64_t>(normalized * range);
        result.push_back(std::min(value, max_value));
    }

    return result;
}

inline std::vector<std::uint64_t> generate_inverse_polynomial(std::size_t size,
                                                                unsigned seed,
                                                                std::uint64_t min_value,
                                                                std::uint64_t max_value) {
    if (size == 0) {
        return {};
    }

    std::vector<std::uint64_t> result;
    result.reserve(size);

    // Generate inverse polynomial: 1 - (1-x)^5 creates strong curvature
    // Fast start (steep slope), slow end (flat) - opposite of x^5
    const std::uint64_t range = max_value - min_value;
    const double size_d = static_cast<double>(size);

    for (std::size_t i = 0; i < size; ++i) {
        // Normalize to [0, 1]
        const double t = static_cast<double>(i) / (size_d - 1.0);
        // Apply 1 - (1-t)^5 for inverse curvature - fast start, slow end
        const double normalized = 1.0 - std::pow(1.0 - t, 5.0);
        const std::uint64_t value = min_value + static_cast<std::uint64_t>(normalized * range);
        result.push_back(std::min(value, max_value));
    }

    return result;
}

}  // namespace dataset
