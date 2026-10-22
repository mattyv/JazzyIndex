#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "jazzy_index_utility.hpp"  // detail::clamp_value and arithmetic trait
#include "jazzy_index_debug.hpp"    // DEBUG_LOG macro (conditional compilation)

namespace jazzy {

// Identity functor for key extraction (default: no transformation)
// Provides a fallback for std::identity when not available
struct identity {
    template <typename T>
    constexpr T&& operator()(T&& t) const noexcept {
        return std::forward<T>(t);
    }
    using is_transparent = void;
};

namespace detail {

// Subtract before rounding to double, without signed overflow at integer limits.
template <typename Key>
double key_delta(Key key, Key origin) {
    if constexpr (std::is_integral_v<Key> && !std::is_same_v<Key, bool>) {
        using Unsigned = std::make_unsigned_t<Key>;
        return key >= origin
            ? static_cast<double>(static_cast<Unsigned>(Unsigned(key) - Unsigned(origin)))
            : -static_cast<double>(static_cast<Unsigned>(Unsigned(origin) - Unsigned(key)));
    } else {
        return static_cast<double>(key) - static_cast<double>(origin);
    }
}

// Model types for different segment characteristics
enum class ModelType : uint8_t {
    LINEAR,      // Most common: y = mx + b (1 FMA)
    QUADRATIC,   // Curved regions: y = ax^2 + bx + c (2 FMA)
    CUBIC,       // Highly curved: y = ax^3 + bx^2 + cx + d (3 FMA)
    CONSTANT     // All values same: y = c (0 computation)
};

// Model selection and search tuning constants
inline constexpr std::size_t MAX_ACCEPTABLE_LINEAR_ERROR = 2;
// Linear models with error ≤MAX_ACCEPTABLE_LINEAR_ERROR are accepted immediately.
// Keeps the local search window small.

inline constexpr double QUADRATIC_IMPROVEMENT_THRESHOLD = 0.7;
// Quadratic must be 30% better than linear (0.7 = 70% of linear error)

inline constexpr double CUBIC_IMPROVEMENT_THRESHOLD = 0.7;
// Cubic must be 30% better than quadratic (0.7 = 70% of quadratic error)

inline constexpr std::size_t MAX_ACCEPTABLE_QUADRATIC_ERROR = 6;
// Quadratic models with error ≤6 are good enough; don't try cubic

inline constexpr std::size_t MAX_CUBIC_WORTHWHILE_ERROR = 50;
// If quadratic error >50, likely a discontinuity; cubic won't help, skip computation

inline constexpr std::size_t SEARCH_RADIUS_MARGIN = 2;
// Extra margin added to max_error for the local search window

inline constexpr std::size_t MIN_SEARCH_RADIUS = 4;
// Minimum radius for the local search window

inline constexpr double UNIFORMITY_TOLERANCE = 0.30;
// Allow 30% deviation in segment spacing for uniformity detection

inline constexpr std::size_t TARGET_ERROR = 16;
// Target prediction error in elements; refinement stops at MIN_SPLIT_SIZE.

inline constexpr std::size_t MIN_SPLIT_SIZE = 64;
// Do not split a segment below this many elements. The search radius floor already
// covers a segment this small in one pass, so splitting further only adds
// segment-lookup work. Also guarantees the split recursion terminates on data no
// model can fit (long runs of duplicate keys).

// Numerical stability and tolerance constants
inline constexpr double ZERO_RANGE_THRESHOLD = std::numeric_limits<double>::epsilon();
// Threshold for detecting zero range (constant segments) in floating-point comparisons

inline constexpr double NUMERICAL_TOLERANCE = 1e-10;
// Tolerance for matrix determinant checks in polynomial fitting (singular matrix detection)

inline constexpr double MIN_DISTRIBUTION_SCALE = 1e-6;
// Minimum scale parameter for distributions to prevent division by zero

template <typename T, typename KeyExtractor>
void validate_finite_key(const T& value, const KeyExtractor& key_extract) {
    using Key = std::remove_cvref_t<std::invoke_result_t<KeyExtractor, const T&>>;
    if constexpr (std::is_floating_point_v<Key>) {
        if (!std::isfinite(std::invoke(key_extract, value))) {
            throw std::invalid_argument("JazzyIndex requires finite floating-point keys");
        }
    }
}

template <typename T, typename Compare, typename KeyExtractor>
void validate_input(const T* first, const T* last,
                    const Compare& comp, const KeyExtractor& key_extract) {
    for (const T* it = first; it != last; ++it) {
        validate_finite_key(*it, key_extract);
    }
    if (!std::is_sorted(first, last, comp)) {
        throw std::runtime_error(
            "Input data is not sorted. JazzyIndex requires sorted data. "
            "Please sort your data before building the index.");
    }
}

// Segment descriptor with optimal model
template <typename T>
struct alignas(64) Segment {  // Cache line aligned
    // Hot path: segment finding and prediction (grouped for cache locality)
    T min_val;
    T max_val;
    ModelType model_type;
    uint32_t max_error;

    // Cubic evaluation uses t = (key - min_key) * inv_scale. Linear and quadratic
    // coefficients already include their scale and evaluate on the key difference.
    float inv_scale;

    // Model parameters (hot path for predict())
    // Using float instead of double to keep struct within 64 bytes
    alignas(8) union {
        struct {
            float slope;
            float intercept;
        } linear;
        struct {
            float a;
            float b;
            float c;
        } quadratic;
        struct {
            float a;
            float b;
            float c;
            float d;
        } cubic;
        struct {
            std::size_t constant_idx;
        } constant;
    } params;

    // Warm path: clamping and bounds
    std::size_t start_idx;
    std::size_t end_idx;

    template <typename KeyExtractor = jazzy::identity>
#if defined(__clang__) || defined(__GNUC__)
    [[nodiscard]] inline __attribute__((always_inline)) std::size_t
#else
    [[nodiscard]] inline std::size_t
#endif
    predict(const T& value, KeyExtractor key_extract = KeyExtractor{}) const
        noexcept(std::is_nothrow_invocable_v<KeyExtractor, const T&>) {
        if (model_type == ModelType::CONSTANT) {
            DEBUG_LOG("predict[%zu-%zu]: CONSTANT - returning %zu",
                      start_idx, end_idx, params.constant.constant_idx);
            return params.constant.constant_idx;
        }

        // Models are expanded about the segment's own min_val, so nothing large is ever
        // subtracted and the coefficients survive narrowing to float. Integral keys
        // subtract in integer space so no low-order bits are lost before the conversion.
        //
        // LINEAR and QUADRATIC pre-fold inv_scale into their coefficients at build time,
        // keeping the multiply off the dependency chain. CUBIC cannot: folding costs it
        // inv_scale^3, which underflows float once the key range exceeds ~5.5e13, so it
        // pays for the normalization at query time. It is also rare (<3% of segments).
        const auto key = std::invoke(key_extract, value);
        const auto origin = std::invoke(key_extract, min_val);
        const double delta = key_delta(key, origin);

        const auto clamp_prediction = [this](double prediction) -> std::size_t {
            // Degenerate segment or a broken fit: rare and key-independent, so a
            // branch here predicts perfectly.
            if (end_idx <= start_idx || !std::isfinite(prediction)) {
                return start_idx;
            }
            // The bounds themselves do not depend on the key, so they compute in
            // parallel with the model evaluation. fmin/fmax lower to FMINNM/FMAXNM
            // (or MINSD/MAXSD), replacing two key-dependent branches that mispredict
            // whenever a prediction lands on a segment edge.
            const double lowest = static_cast<double>(start_idx);
            const double highest = static_cast<double>(end_idx - 1);
            // Rounding avoids a search when float coefficients predict just below an integer.
            return static_cast<std::size_t>(
                std::round(std::fmin(std::fmax(prediction, lowest), highest)));
        };

        switch (model_type) {
            case ModelType::LINEAR: {
                const double pred = std::fma(delta,
                                            params.linear.slope,
                                            params.linear.intercept);
                const std::size_t result = clamp_prediction(pred);
                DEBUG_LOG("predict[%zu-%zu]: LINEAR - delta=%.4f, pred=%.2f, result=%zu (slope=%.4g, intercept=%.4f)",
                          start_idx, end_idx, delta, pred, result, params.linear.slope, params.linear.intercept);
                return result;
            }
            case ModelType::QUADRATIC: {
                const double pred = std::fma(delta,
                                            std::fma(delta, params.quadratic.a, params.quadratic.b),
                                            params.quadratic.c);
                const std::size_t result = clamp_prediction(pred);
                DEBUG_LOG("predict[%zu-%zu]: QUADRATIC - delta=%.4f, pred=%.2f, result=%zu",
                          start_idx, end_idx, delta, pred, result);
                return result;
            }
            case ModelType::CUBIC: {
                // Horner's method: ((a*t + b)*t + c)*t + d
                const double t = delta * static_cast<double>(inv_scale);
                const double pred = std::fma(t,
                                            std::fma(t,
                                                    std::fma(t, params.cubic.a, params.cubic.b),
                                                    params.cubic.c),
                                            params.cubic.d);
                const std::size_t result = clamp_prediction(pred);
                DEBUG_LOG("predict[%zu-%zu]: CUBIC - t=%.6f, pred=%.2f, result=%zu",
                          start_idx, end_idx, t, pred, result);
                return result;
            }
            default:
                DEBUG_LOG("predict[%zu-%zu]: UNKNOWN model type, returning start_idx=%zu",
                          start_idx, end_idx, start_idx);
                return start_idx;  // Fallback
        }
    }
};

// Analyze segment to choose best model
template <typename T>
struct SegmentAnalysis {
    ModelType best_model;
    double linear_a, linear_b;
    double quad_a, quad_b, quad_c;
    double cubic_a, cubic_b, cubic_c, cubic_d;
    std::size_t max_error;
    double mean_error;
    double inv_scale;  // See Segment::inv_scale.
};

template <typename T>
struct BatchLookupState {
    const Segment<T>* segment{nullptr};
    const T* low{nullptr};
    const T* high{nullptr};
    const T* bounded_begin{nullptr};
    const T* bounded_end{nullptr};
    std::size_t left{0};
    std::size_t right{0};
    bool active{false};
};

// Fit linear model to segment: index = slope * value + intercept
template <typename T, typename Compare = std::less<>, typename KeyExtractor = jazzy::identity>
[[nodiscard]] SegmentAnalysis<T> analyze_segment(const T* data,
                                                   std::size_t start,
                                                   std::size_t end,
                                                   Compare comp = Compare{},
                                                   KeyExtractor key_extract = KeyExtractor{})
    noexcept(std::is_nothrow_invocable_v<KeyExtractor, const T&> &&
             std::is_nothrow_invocable_v<Compare, const T&, const T&>) {
    SegmentAnalysis<T> result{};

    auto make_constant = [&result, start]() {
        result.best_model = ModelType::CONSTANT;
        result.linear_b = static_cast<double>(start);
        result.max_error = 0;
        DEBUG_LOG("analyze_segment[%zu]: Selected CONSTANT model (empty or single element)",
                  start);
        return result;
    };

    if (end <= start)
        return make_constant();

    const std::size_t n = end - start;

    // For sorted data, min/max are at endpoints (extract keys for comparison)
    const auto origin = std::invoke(key_extract, data[start]);
    const double value_range = key_delta(std::invoke(key_extract, data[end - 1]), origin);

    // Check for constant segment (zero range)
    if (value_range < detail::ZERO_RANGE_THRESHOLD) {
        return make_constant();
    }
    if (!std::isfinite(value_range)) {
        result = make_constant();
        result.max_error = n - 1;
        return result;
    }

    // Runtime predictions are clamped; failed fits must never cast NaN/inf to size_t.
    const auto prediction_error = [n](double prediction, std::size_t i) {
        const double error = std::abs(prediction - static_cast<double>(i));
        return std::isfinite(error) ? std::min(error, static_cast<double>(n - 1))
                                   : static_cast<double>(n - 1);
    };

    // Linear model expanded about min_val: index = start + slope * (key - min_val).
    // The intercept is just `start`, a small exact integer. The old form stored
    // `start - slope * min_val`, which cancels away the entire fit in a float once
    // min_val is large.
    const double inv_scale = 1.0 / value_range;
    const double slope = static_cast<double>(n - 1) * inv_scale;
    const double intercept = static_cast<double>(start);

    result.linear_a = slope;
    result.linear_b = intercept;
    result.inv_scale = inv_scale;

    // Errors are measured with the float-narrowed coefficients that predict() will
    // actually use, so max_error is a bound on the real runtime error rather than on
    // an idealised double-precision one.
    const float slope_stored = static_cast<float>(slope);
    const float intercept_stored = static_cast<float>(intercept);

    // Single pass: compute linear error AND quadratic/cubic sums simultaneously
    std::size_t linear_max_error = 0;
    double linear_total_error = 0.0;

    double sum_x = 0.0, sum_x2 = 0.0, sum_x3 = 0.0, sum_x4 = 0.0, sum_x5 = 0.0, sum_x6 = 0.0;
    double sum_y = 0.0, sum_xy = 0.0, sum_x2y = 0.0, sum_x3y = 0.0;

    bool all_same = true;
    const T& first_val = data[start];

    // Helper for equality check using comparator (a == b iff !(a < b) && !(b < a))
    auto equal = [&comp](const T& a, const T& b) {
        return !comp(a, b) && !comp(b, a);
    };

    for (std::size_t i = start; i < end; ++i) {
        const T& current_val = data[i];

        // Check if all values are identical using comparator
        if (all_same && !equal(current_val, first_val)) {
            all_same = false;
        }

        const double delta = key_delta(std::invoke(key_extract, current_val), origin);
        // Normalized key for the quadratic/cubic sums (conditioning)
        const double x_normalized = delta * inv_scale;

        // Compute linear error the same way predict() will evaluate the model
        const double pred_double = std::fma(delta, slope_stored, intercept_stored);
        const double error = prediction_error(pred_double, i);
        linear_max_error = std::max(linear_max_error, static_cast<std::size_t>(std::ceil(error)));
        linear_total_error += error;

        const double y = static_cast<double>(i);
        const double x2 = x_normalized * x_normalized;
        const double x3 = x2 * x_normalized;
        const double x4 = x2 * x2;
        const double x5 = x4 * x_normalized;
        const double x6 = x3 * x3;

        sum_x += x_normalized;
        sum_x2 += x2;
        sum_x3 += x3;
        sum_x4 += x4;
        sum_x5 += x5;
        sum_x6 += x6;
        sum_y += y;
        sum_xy += x_normalized * y;
        sum_x2y += x2 * y;
        sum_x3y += x3 * y;
    }

    // If all values are identical, use constant model
    if (all_same) {
        return make_constant();
    }

    const double linear_mean_error = linear_total_error / static_cast<double>(n);

    DEBUG_LOG("analyze_segment[%zu-%zu]: n=%zu, linear_max_error=%zu, linear_mean_error=%.2f, slope=%.4f, intercept=%.4f",
              start, end, n, linear_max_error, linear_mean_error, slope, intercept);

    // If linear is good enough, use it
    if (linear_max_error <= MAX_ACCEPTABLE_LINEAR_ERROR) {
        result.best_model = ModelType::LINEAR;
        result.max_error = linear_max_error;
        result.mean_error = linear_mean_error;
        DEBUG_LOG("analyze_segment[%zu-%zu]: Selected LINEAR model (max_error=%zu <= threshold=%zu)",
                  start, end, linear_max_error, MAX_ACCEPTABLE_LINEAR_ERROR);
        return result;
    }

    DEBUG_LOG("analyze_segment[%zu-%zu]: Linear error too high (%zu > %zu), trying QUADRATIC",
              start, end, linear_max_error, MAX_ACCEPTABLE_LINEAR_ERROR);

    // Try quadratic model: we already have the sums from the single pass above

    const double n_double = static_cast<double>(n);

    // Solve normal equations using Cramer's rule for 3x3 system
    // System: [Σx⁴  Σx³  Σx²] [a]   [Σx²y]
    //         [Σx³  Σx²  Σx ] [b] = [Σxy ]
    //         [Σx²  Σx   n  ] [c]   [Σy  ]

    // Determinant of coefficient matrix
    const double det = sum_x4 * (sum_x2 * n_double - sum_x * sum_x)
                     - sum_x3 * (sum_x3 * n_double - sum_x * sum_x2)
                     + sum_x2 * (sum_x3 * sum_x - sum_x2 * sum_x2);

    if (std::abs(det) > NUMERICAL_TOLERANCE) {
        // Cramer's rule for a, b, c
        const double det_a = sum_x2y * (sum_x2 * n_double - sum_x * sum_x)
                           - sum_xy * (sum_x3 * n_double - sum_x * sum_x2)
                           + sum_y * (sum_x3 * sum_x - sum_x2 * sum_x2);

        const double det_b = sum_x4 * (sum_xy * n_double - sum_y * sum_x)
                           - sum_x3 * (sum_x2y * n_double - sum_y * sum_x2)
                           + sum_x2 * (sum_x2y * sum_x - sum_xy * sum_x2);

        const double det_c = sum_x4 * (sum_x2 * sum_y - sum_x * sum_xy)
                           - sum_x3 * (sum_x3 * sum_y - sum_x * sum_x2y)
                           + sum_x2 * (sum_x3 * sum_xy - sum_x2 * sum_x2y);

        const double a = det_a / det;
        const double b = det_b / det;
        const double c = det_c / det;

        // Fold inv_scale into the coefficients so predict() evaluates on the raw delta
        // and keeps the multiply off its dependency chain. Safe for quadratic: the
        // smallest folded term is a/range^2, which stays normal in float for any
        // uint64 key range. Cubic would need a/range^3 and is left normalized.
        const float quad_a_stored = static_cast<float>(a * inv_scale * inv_scale);
        const float quad_b_stored = static_cast<float>(b * inv_scale);
        const float quad_c_stored = static_cast<float>(c);

        // Measure with the narrowed, folded coefficients predict() will actually use
        std::size_t quad_max_error = 0;
        double quad_total_error = 0.0;

        for (std::size_t i = start; i < end; ++i) {
            const double delta = key_delta(std::invoke(key_extract, data[i]), origin);
            const double pred_double = std::fma(delta, std::fma(delta, quad_a_stored, quad_b_stored),
                                                quad_c_stored);
            const double error = prediction_error(pred_double, i);
            quad_max_error = std::max(quad_max_error, static_cast<std::size_t>(std::ceil(error)));
            quad_total_error += error;
        }

        const double quad_mean_error = quad_total_error / n_double;

        DEBUG_LOG("analyze_segment[%zu-%zu]: QUADRATIC: max_error=%zu, mean_error=%.2f, improvement_threshold=%.0f",
                  start, end, quad_max_error, quad_mean_error, linear_max_error * QUADRATIC_IMPROVEMENT_THRESHOLD);

        // Choose quadratic if it's significantly better
        if (quad_max_error < linear_max_error * QUADRATIC_IMPROVEMENT_THRESHOLD) {
            // Check monotonicity: f'(t) = 2*a*t + b must be non-negative over t in [0, 1].
            // For a search index, predictions must be monotonically INCREASING.
            // f'(t) is linear, so checking both endpoints is sufficient. Done in
            // normalized space; folding inv_scale in is a positive rescale of the
            // argument and cannot change the sign of the derivative.
            const double derivative_at_min = b;
            const double derivative_at_max = 2.0 * a + b;
            const bool is_monotonic = (derivative_at_min >= 0.0) && (derivative_at_max >= 0.0);

            DEBUG_LOG("analyze_segment[%zu-%zu]: QUADRATIC monotonicity: deriv_at_min=%.4f, deriv_at_max=%.4f, is_monotonic=%d",
                      start, end, derivative_at_min, derivative_at_max, is_monotonic);

            // Only accept quadratic if it's monotonic
            if (is_monotonic) {
                // Check if we should try cubic for even better fit
                // Only try cubic if error is in "sweet spot" (6 < error < 50)
                // High error (>50) likely indicates discontinuity where cubic won't help
                if (quad_max_error > MAX_ACCEPTABLE_QUADRATIC_ERROR &&
                    quad_max_error < MAX_CUBIC_WORTHWHILE_ERROR) {
                    DEBUG_LOG("analyze_segment[%zu-%zu]: Quad error in sweet spot (%zu), trying CUBIC",
                              start, end, quad_max_error);
                    // Try cubic model using same normalized sums
                    // System: [Σx⁶  Σx⁵  Σx⁴  Σx³] [a]   [Σx³y]
                    //         [Σx⁵  Σx⁴  Σx³  Σx²] [b] = [Σx²y]
                    //         [Σx⁴  Σx³  Σx²  Σx ] [c]   [Σxy ]
                    //         [Σx³  Σx²  Σx   n  ] [d]   [Σy  ]

                    // Compute determinant of 4x4 coefficient matrix using cofactor expansion
                    // For numerical stability and code brevity, we use a helper for 3x3 determinants
                    auto det3x3 = [](double a11, double a12, double a13,
                                    double a21, double a22, double a23,
                                    double a31, double a32, double a33) -> double {
                        return a11 * (a22 * a33 - a23 * a32)
                             - a12 * (a21 * a33 - a23 * a31)
                             + a13 * (a21 * a32 - a22 * a31);
                    };

                    // 4x4 determinant by expanding along first row
                    const double det4 =
                        sum_x6 * det3x3(sum_x4, sum_x3, sum_x2,
                                       sum_x3, sum_x2, sum_x,
                                       sum_x2, sum_x, n_double)
                        - sum_x5 * det3x3(sum_x5, sum_x3, sum_x2,
                                         sum_x4, sum_x2, sum_x,
                                         sum_x3, sum_x, n_double)
                        + sum_x4 * det3x3(sum_x5, sum_x4, sum_x2,
                                         sum_x4, sum_x3, sum_x,
                                         sum_x3, sum_x2, n_double)
                        - sum_x3 * det3x3(sum_x5, sum_x4, sum_x3,
                                         sum_x4, sum_x3, sum_x2,
                                         sum_x3, sum_x2, sum_x);

                    if (std::abs(det4) > NUMERICAL_TOLERANCE) {
                        // Compute determinants for Cramer's rule (replace each column with RHS)
                        const double det_a =
                            sum_x3y * det3x3(sum_x4, sum_x3, sum_x2,
                                           sum_x3, sum_x2, sum_x,
                                           sum_x2, sum_x, n_double)
                            - sum_x5 * det3x3(sum_x2y, sum_x3, sum_x2,
                                             sum_xy, sum_x2, sum_x,
                                             sum_y, sum_x, n_double)
                            + sum_x4 * det3x3(sum_x2y, sum_x4, sum_x2,
                                             sum_xy, sum_x3, sum_x,
                                             sum_y, sum_x2, n_double)
                            - sum_x3 * det3x3(sum_x2y, sum_x4, sum_x3,
                                             sum_xy, sum_x3, sum_x2,
                                             sum_y, sum_x2, sum_x);

                        const double det_b =
                            sum_x6 * det3x3(sum_x2y, sum_x3, sum_x2,
                                           sum_xy, sum_x2, sum_x,
                                           sum_y, sum_x, n_double)
                            - sum_x3y * det3x3(sum_x5, sum_x3, sum_x2,
                                              sum_x4, sum_x2, sum_x,
                                              sum_x3, sum_x, n_double)
                            + sum_x4 * det3x3(sum_x5, sum_x2y, sum_x2,
                                             sum_x4, sum_xy, sum_x,
                                             sum_x3, sum_y, n_double)
                            - sum_x3 * det3x3(sum_x5, sum_x2y, sum_x3,
                                             sum_x4, sum_xy, sum_x2,
                                             sum_x3, sum_y, sum_x);

                        const double det_c =
                            sum_x6 * det3x3(sum_x4, sum_x2y, sum_x2,
                                           sum_x3, sum_xy, sum_x,
                                           sum_x2, sum_y, n_double)
                            - sum_x5 * det3x3(sum_x5, sum_x2y, sum_x2,
                                             sum_x4, sum_xy, sum_x,
                                             sum_x3, sum_y, n_double)
                            + sum_x3y * det3x3(sum_x5, sum_x4, sum_x2,
                                              sum_x4, sum_x3, sum_x,
                                              sum_x3, sum_x2, n_double)
                            - sum_x3 * det3x3(sum_x5, sum_x4, sum_x2y,
                                             sum_x4, sum_x3, sum_xy,
                                             sum_x3, sum_x2, sum_y);

                        const double det_d =
                            sum_x6 * det3x3(sum_x4, sum_x3, sum_x2y,
                                           sum_x3, sum_x2, sum_xy,
                                           sum_x2, sum_x, sum_y)
                            - sum_x5 * det3x3(sum_x5, sum_x3, sum_x2y,
                                             sum_x4, sum_x2, sum_xy,
                                             sum_x3, sum_x, sum_y)
                            + sum_x4 * det3x3(sum_x5, sum_x4, sum_x2y,
                                             sum_x4, sum_x3, sum_xy,
                                             sum_x3, sum_x2, sum_y)
                            - sum_x3y * det3x3(sum_x5, sum_x4, sum_x3,
                                              sum_x4, sum_x3, sum_x2,
                                              sum_x3, sum_x2, sum_x);

                        const double cubic_a_norm = det_a / det4;
                        const double cubic_b_norm = det_b / det4;
                        const double cubic_c_norm = det_c / det4;
                        const double cubic_d_norm = det_d / det4;

                        // Measure with the narrowed coefficients and the float inv_scale
                        // that predict() will actually use, mirroring linear/quadratic:
                        // max_error must bound the runtime error, not the double-precision one.
                        const float cubic_a_stored = static_cast<float>(cubic_a_norm);
                        const float cubic_b_stored = static_cast<float>(cubic_b_norm);
                        const float cubic_c_stored = static_cast<float>(cubic_c_norm);
                        const float cubic_d_stored = static_cast<float>(cubic_d_norm);
                        const double inv_scale_runtime =
                            static_cast<double>(static_cast<float>(inv_scale));

                        std::size_t cubic_max_error = 0;
                        double cubic_total_error = 0.0;

                        for (std::size_t i = start; i < end; ++i) {
                            const double x_norm =
                                key_delta(std::invoke(key_extract, data[i]), origin) * inv_scale_runtime;
                            const double pred = std::fma(x_norm,
                                                        std::fma(x_norm,
                                                                std::fma(x_norm, cubic_a_stored, cubic_b_stored),
                                                                cubic_c_stored),
                                                        cubic_d_stored);
                            const double error = prediction_error(pred, i);
                            cubic_max_error = std::max(cubic_max_error, static_cast<std::size_t>(std::ceil(error)));
                            cubic_total_error += error;
                        }

                        DEBUG_LOG("analyze_segment[%zu-%zu]: CUBIC: max_error=%zu, improvement_threshold=%.0f",
                                  start, end, cubic_max_error, quad_max_error * CUBIC_IMPROVEMENT_THRESHOLD);

                        // Choose cubic if it's significantly better than quadratic
                        if (cubic_max_error < quad_max_error * CUBIC_IMPROVEMENT_THRESHOLD) {
                            // Cubic coefficients remain in normalized coordinates.
                            const double cubic_a_orig = cubic_a_norm;
                            const double cubic_b_orig = cubic_b_norm;
                            const double cubic_c_orig = cubic_c_norm;
                            const double cubic_d_orig = cubic_d_norm;

                            // Check monotonicity: f'(t) = 3*a*t^2 + 2*b*t + c must be >= 0 over t in [0, 1].
                            // For cubic, derivative is quadratic, so we check at critical points and endpoints
                            auto cubic_derivative = [&](double t) -> double {
                                return 3.0 * cubic_a_orig * t * t + 2.0 * cubic_b_orig * t + cubic_c_orig;
                            };

                            bool cubic_is_monotonic = true;
                            const double cubic_deriv_at_min = cubic_derivative(0.0);
                            const double cubic_deriv_at_max = cubic_derivative(1.0);

                            DEBUG_LOG("analyze_segment[%zu-%zu]: CUBIC monotonicity check: deriv_at_min=%.4f, deriv_at_max=%.4f",
                                      start, end, cubic_deriv_at_min, cubic_deriv_at_max);

                            // Check endpoints
                            if (cubic_deriv_at_min < 0.0 || cubic_deriv_at_max < 0.0) {
                                cubic_is_monotonic = false;
                                DEBUG_LOG("analyze_segment[%zu-%zu]: CUBIC failed endpoint monotonicity check", start, end);
                            }

                            // Check critical points (where f''(x) = 0)
                            // f''(x) = 6*a*x + 2*b = 0 => x = -b/(3*a)
                            if (cubic_is_monotonic && std::abs(cubic_a_orig) > NUMERICAL_TOLERANCE) {
                                const double critical_x = -cubic_b_orig / (3.0 * cubic_a_orig);
                                DEBUG_LOG("analyze_segment[%zu-%zu]: CUBIC critical point at t=%.4f (range: [0, 1])",
                                          start, end, critical_x);
                                if (critical_x >= 0.0 && critical_x <= 1.0) {
                                    const double deriv_at_critical = cubic_derivative(critical_x);
                                    DEBUG_LOG("analyze_segment[%zu-%zu]: CUBIC deriv at critical point: %.4f",
                                              start, end, deriv_at_critical);
                                    if (deriv_at_critical < 0.0) {
                                        cubic_is_monotonic = false;
                                        DEBUG_LOG("analyze_segment[%zu-%zu]: CUBIC failed critical point monotonicity", start, end);
                                    }
                                }
                            }

                            if (cubic_is_monotonic) {
                                result.best_model = ModelType::CUBIC;
                                result.cubic_a = cubic_a_orig;
                                result.cubic_b = cubic_b_orig;
                                result.cubic_c = cubic_c_orig;
                                result.cubic_d = cubic_d_orig;
                                result.max_error = cubic_max_error;
                                result.mean_error = cubic_total_error / n_double;
                                DEBUG_LOG("analyze_segment[%zu-%zu]: Selected CUBIC model (max_error=%zu, improved from quad_error=%zu)",
                                          start, end, cubic_max_error, quad_max_error);
                                return result;
                            }
                        }
                    }
                }

                // Use quadratic if cubic didn't work out
                result.best_model = ModelType::QUADRATIC;
                result.quad_a = quad_a_stored;
                result.quad_b = quad_b_stored;
                result.quad_c = quad_c_stored;
                result.max_error = quad_max_error;
                result.mean_error = quad_mean_error;
                DEBUG_LOG("analyze_segment[%zu-%zu]: Selected QUADRATIC model (max_error=%zu, improved from linear_error=%zu, cubic failed/not worthwhile)",
                          start, end, quad_max_error, linear_max_error);
                return result;
            } else {
                DEBUG_LOG("analyze_segment[%zu-%zu]: QUADRATIC not monotonic, falling back to LINEAR", start, end);
            }
            // If non-monotonic, fall through to use linear model instead
        } else {
            DEBUG_LOG("analyze_segment[%zu-%zu]: QUADRATIC not good enough (error=%zu >= threshold=%.0f), using LINEAR",
                      start, end, quad_max_error, linear_max_error * QUADRATIC_IMPROVEMENT_THRESHOLD);
        }
    } else {
        DEBUG_LOG("analyze_segment[%zu-%zu]: Quadratic matrix singular (det=%.6f), using LINEAR", start, end, det);
    }

    // Default to linear
    result.best_model = ModelType::LINEAR;
    result.max_error = linear_max_error;
    result.mean_error = linear_mean_error;
    DEBUG_LOG("analyze_segment[%zu-%zu]: Defaulted to LINEAR model (max_error=%zu)",
              start, end, linear_max_error);
    return result;
}

// Copy a fitted model into a segment. Shared by the serial and parallel builders so
// the two cannot drift apart.
template <typename T>
void apply_analysis(Segment<T>& seg, const SegmentAnalysis<T>& analysis) {
    if (analysis.max_error > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(
            "Segment prediction error exceeds uint32_t limit. "
            "Data distribution is too extreme for indexing. "
            "Consider using fewer segments or preprocessing the data.");
    }

    seg.model_type = analysis.best_model;
    seg.inv_scale = static_cast<float>(analysis.inv_scale);
    // A prediction is clamped into the segment, so it can never be off by more than
    // the segment's own length. Storing the raw fit error would inflate the search
    // radius on the short segments that refine_segments() produces.
    seg.max_error = static_cast<uint32_t>(std::min(analysis.max_error,
                                                   seg.end_idx - seg.start_idx));

    switch (analysis.best_model) {
        case ModelType::LINEAR:
            seg.params.linear.slope = static_cast<float>(analysis.linear_a);
            seg.params.linear.intercept = static_cast<float>(analysis.linear_b);
            break;
        case ModelType::QUADRATIC:
            seg.params.quadratic.a = static_cast<float>(analysis.quad_a);
            seg.params.quadratic.b = static_cast<float>(analysis.quad_b);
            seg.params.quadratic.c = static_cast<float>(analysis.quad_c);
            break;
        case ModelType::CUBIC:
            seg.params.cubic.a = static_cast<float>(analysis.cubic_a);
            seg.params.cubic.b = static_cast<float>(analysis.cubic_b);
            seg.params.cubic.c = static_cast<float>(analysis.cubic_c);
            seg.params.cubic.d = static_cast<float>(analysis.cubic_d);
            break;
        case ModelType::CONSTANT:
            seg.params.constant.constant_idx = seg.start_idx;
            break;
        default:
            break;
    }
}

template <typename T, typename Compare, typename KeyExtractor>
Segment<T> make_segment(const T* base, std::size_t start, std::size_t end,
                        const Compare& comp, const KeyExtractor& key_extract) {
    Segment<T> seg{};
    seg.min_val = base[start];
    seg.max_val = base[end - 1];
    seg.start_idx = start;
    seg.end_idx = end;
    apply_analysis(seg, analyze_segment(base, start, end, comp, key_extract));
    return seg;
}

// Split segments that missed the error budget until they meet it. Quantile boundaries
// divide the data by count, which says nothing about how well the data fits a model;
// halving the ones that fit badly turns an unbounded error into a bounded one without
// needing a different fitting algorithm.
template <typename T, typename Compare, typename KeyExtractor>
void refine_segments(std::vector<Segment<T>>& segments, const T* base,
                     const Compare& comp, const KeyExtractor& key_extract) {
    std::vector<Segment<T>> refined;
    refined.reserve(segments.size());

    std::vector<Segment<T>> pending;
    for (const auto& seg : segments) {
        pending.push_back(seg);
        while (!pending.empty()) {
            const Segment<T> current = pending.back();
            pending.pop_back();

            const std::size_t length = current.end_idx - current.start_idx;
            if (current.max_error <= TARGET_ERROR || length < 2 * MIN_SPLIT_SIZE) {
                refined.push_back(current);
                continue;
            }

            // Push right before left so the left half (and anything it splits into)
            // is emitted first, keeping `refined` sorted by start_idx.
            const std::size_t mid = current.start_idx + length / 2;
            pending.push_back(make_segment(base, mid, current.end_idx, comp, key_extract));
            pending.push_back(make_segment(base, current.start_idx, mid, comp, key_extract));
        }
    }

    segments = std::move(refined);
    // Drop push_back growth slack so size_in_bytes() (which reports capacity)
    // describes what the index actually needs.
    segments.shrink_to_fit();
}

}  // namespace detail

// Recommended segment count presets
enum class SegmentCount : std::size_t {
    SINGLE = 1,      // No segmentation: full dataset in one segment
    MINIMAL = 2,     // Binary split: extreme coarse indexing
    PICO = 4,        // Debugging tiny datasets; almost no segmentation
    NANO = 8,        // Extremely small datasets; coarse segmentation
    MICRO = 16,      // Minimal segmentation for testing larger segment sizes
    TINY = 32,       // Very small datasets or minimal memory footprint
    SMALL = 64,      // Small datasets (thousands of elements)
    MEDIUM = 128,    // Medium datasets
    LARGE = 256,     // Default: good balance for most use cases
    XLARGE = 512,    // Large datasets with complex distributions
    XXLARGE = 1024,  // Very large datasets requiring high precision
    MAX = 2048       // Maximum precision for huge datasets
};

// Helper to convert std::size_t to SegmentCount for generic code
template <std::size_t N>
inline constexpr SegmentCount to_segment_count() {
    return static_cast<SegmentCount>(N);
}

// Forward declarations for parallel build support
namespace parallel {
template <typename T, typename Compare, typename KeyExtractor>
struct BuildTask;

template <typename T, SegmentCount Segments, typename Compare, typename KeyExtractor>
class ParallelBuilder;
}  // namespace parallel

template <typename T, SegmentCount Segments = SegmentCount::LARGE, typename Compare = std::less<>, typename KeyExtractor = jazzy::identity>
class JazzyIndex {
    static constexpr std::size_t NumSegments = static_cast<std::size_t>(Segments);
    static constexpr unsigned LogRadixMantissaBits = 14;
    static constexpr bool supports_radix_routing =
        std::is_integral_v<T> && std::is_unsigned_v<T> &&
        sizeof(T) <= sizeof(std::uint64_t) &&
        (std::is_same_v<Compare, std::less<>> || std::is_same_v<Compare, std::less<T>>) &&
        std::is_same_v<KeyExtractor, jazzy::identity>;

    static_assert(NumSegments > 0 && NumSegments <= 4096,
                  "NumSegments must be in range [1, 4096]");

    // Type constraint: KeyExtractor must be callable with const T&
    static_assert(std::is_invocable_v<const KeyExtractor&, const T&>,
                  "KeyExtractor must be callable as const with const T&");

    // Type constraint: KeyExtractor must return an arithmetic type
    using KeyType = std::invoke_result_t<const KeyExtractor&, const T&>;
    using KeyTypeClean = typename std::remove_cv<typename std::remove_reference<KeyType>::type>::type;
    static_assert(std::is_arithmetic_v<KeyTypeClean>,
                  "KeyExtractor must return an arithmetic type (int, double, etc.)");

    // Type constraint: Compare must be callable with two const T& and return bool
    static_assert(std::is_invocable_r_v<bool, const Compare&, const T&, const T&>,
                  "Compare must be callable as const with (const T&, const T&) and return bool");

public:
    // Iterator type aliases
    using iterator = const T*;
    using const_iterator = const T*;
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = const T&;
    using const_reference = const T&;
    using pointer = const T*;
    using const_pointer = const T*;

    JazzyIndex() = default;
    JazzyIndex(const JazzyIndex&) = default;
    JazzyIndex& operator=(const JazzyIndex&) = default;

    // Moving the segment vector must also clear the source's non-owning state.
    JazzyIndex(JazzyIndex&& other)
        noexcept(std::is_nothrow_move_constructible_v<T> &&
                 std::is_nothrow_move_constructible_v<KeyExtractor> &&
                 std::is_nothrow_move_constructible_v<Compare>)
        : base_(std::exchange(other.base_, nullptr)),
          size_(std::exchange(other.size_, 0)),
          min_(std::move(other.min_)), max_(std::move(other.max_)),
          key_extract_(std::move(other.key_extract_)), comp_(std::move(other.comp_)),
          num_segments_(std::exchange(other.num_segments_, 0)),
          is_uniform_(std::exchange(other.is_uniform_, false)),
          built_(std::exchange(other.built_, false)),
          is_dense_(std::exchange(other.is_dense_, false)),
          segment_scale_(std::exchange(other.segment_scale_, 0)),
          radix_offsets_(std::move(other.radix_offsets_)),
          has_adjacent_duplicates_(other.has_adjacent_duplicates_),
          segments_(std::move(other.segments_)) {}

    JazzyIndex& operator=(JazzyIndex&& other)
        noexcept(std::is_nothrow_move_assignable_v<T> &&
                 std::is_nothrow_move_assignable_v<KeyExtractor> &&
                 std::is_nothrow_move_assignable_v<Compare>) {
        if (this == &other) return *this;
        min_ = std::move(other.min_);
        max_ = std::move(other.max_);
        key_extract_ = std::move(other.key_extract_);
        comp_ = std::move(other.comp_);
        segments_ = std::move(other.segments_);
        radix_offsets_ = std::move(other.radix_offsets_);
        has_adjacent_duplicates_ = other.has_adjacent_duplicates_;
        base_ = std::exchange(other.base_, nullptr);
        size_ = std::exchange(other.size_, 0);
        num_segments_ = std::exchange(other.num_segments_, 0);
        is_uniform_ = std::exchange(other.is_uniform_, false);
        built_ = std::exchange(other.built_, false);
        is_dense_ = std::exchange(other.is_dense_, false);
        segment_scale_ = std::exchange(other.segment_scale_, 0);
        return *this;
    }

    JazzyIndex(const T* first, const T* last, Compare comp = Compare{}, KeyExtractor key_extract = KeyExtractor{}) {
        build(first, last, comp, key_extract);
    }

    // Iterator-based constructor
    template <typename Iterator>
        requires std::random_access_iterator<Iterator> &&
                 std::contiguous_iterator<Iterator> &&
                 (!std::is_pointer_v<Iterator>) &&
                 std::same_as<typename std::iterator_traits<Iterator>::value_type, T>
    JazzyIndex(Iterator first, Iterator last, Compare comp = Compare{}, KeyExtractor key_extract = KeyExtractor{}) {
        build(first, last, comp, key_extract);
    }

    // The range is non-owning and must remain alive, sorted, and address-stable
    // until the index is rebuilt or destroyed.
    void build(const T* first, const T* last, Compare comp = Compare{}, KeyExtractor key_extract = KeyExtractor{}) {
        JazzyIndex candidate;
        candidate.build_in_place(first, last, comp, key_extract);
        *this = std::move(candidate);
    }

private:
    void build_in_place(const T* first, const T* last, Compare comp, KeyExtractor key_extract) {
        base_ = first;
        key_extract_ = key_extract;
        comp_ = comp;

        if (first == last) {
            DEBUG_LOG("JazzyIndex::build: Building index for 0 elements with %zu segments",
                      NumSegments);
            built_ = true;
            return;
        }
        if (first == nullptr || last == nullptr) {
            throw std::invalid_argument("JazzyIndex requires a valid contiguous range");
        }
        if (std::less<const T*>{}(last, first)) {
            throw std::invalid_argument("JazzyIndex requires a non-reversed range");
        }

        size_ = static_cast<std::size_t>(last - first);
        detail::validate_input(first, last, comp_, key_extract_);

        DEBUG_LOG("JazzyIndex::build: Building index for %zu elements with %zu segments", size_, NumSegments);

        min_ = base_[0];
        max_ = base_[size_ - 1];

        if (size_ == 1) {
            // Single element
            segments_.resize(1);
            segments_[0].min_val = min_;
            segments_[0].max_val = max_;
            segments_[0].start_idx = 0;
            segments_[0].end_idx = 1;
            segments_[0].model_type = detail::ModelType::CONSTANT;
            segments_[0].inv_scale = 0.0f;
            segments_[0].max_error = 0;
            segments_[0].params.constant.constant_idx = 0;
            num_segments_ = 1;
            built_ = true;
            return;
        }

        // Quantile split first, then split again wherever the fit missed the budget
        const std::size_t actual_segments = std::min(NumSegments, size_);
        segments_.clear();
        segments_.reserve(actual_segments);
        for (std::size_t i = 0; i < actual_segments; ++i) {
            segments_.push_back(detail::make_segment(base_,
                                                     (i * size_) / actual_segments,
                                                     ((i + 1) * size_) / actual_segments,
                                                     comp_, key_extract_));
        }
        detail::refine_segments(segments_, base_, comp_, key_extract_);
        num_segments_ = segments_.size();

        const double total_range = static_cast<double>(std::invoke(key_extract_, max_)) -
                                   static_cast<double>(std::invoke(key_extract_, min_));
        is_uniform_ = check_uniformity(total_range);

        // Compute scale factor for O(1) segment lookup if data is uniform
        if (is_uniform_ && total_range >= detail::ZERO_RANGE_THRESHOLD) {
            segment_scale_ = static_cast<double>(num_segments_) / total_range;
            DEBUG_LOG("JazzyIndex::build: Data is UNIFORM, segment_scale=%.6f", segment_scale_);
        } else {
            DEBUG_LOG("JazzyIndex::build: Data is NON-UNIFORM (is_uniform=%d, total_range=%.4f)",
                      is_uniform_, total_range);
        }
        finalize_routing();
        built_ = true;
        DEBUG_LOG("JazzyIndex::build: Build complete with %zu segments", num_segments_);
    }

public:
    // Iterator-based build method
    template <typename Iterator>
        requires std::random_access_iterator<Iterator> &&
                 std::contiguous_iterator<Iterator> &&
                 (!std::is_pointer_v<Iterator>) &&
                 std::same_as<typename std::iterator_traits<Iterator>::value_type, T>
    void build(Iterator first, Iterator last, Compare comp = Compare{}, KeyExtractor key_extract = KeyExtractor{}) {
        // Convert contiguous iterators to pointers for internal use
        build(std::to_address(first), std::to_address(last), comp, key_extract);
    }

    [[nodiscard]] const_iterator find(const T& key) const {
        DEBUG_LOG("JazzyIndex::find: Called (size=%zu, is_built=%d)", size_, is_built());

        // Return end iterator if index not built or empty
        if (!is_built() || size_ == 0) {
            DEBUG_LOG("JazzyIndex::find: Index not built or empty, returning end()");
            return end();
        }
        detail::validate_finite_key(key, key_extract_);

        // Bounds check
        const T& first_value = base_[0];
        const T& last_value = base_[size_ - 1];
        if (comp_(key, first_value) || comp_(last_value, key)) {
            DEBUG_LOG("JazzyIndex::find: Key out of bounds, returning end()");
            return end();
        }
        if constexpr (supports_radix_routing) {
            if (is_dense_) {
                return base_ + (static_cast<std::uint64_t>(key) -
                                static_cast<std::uint64_t>(min_));
            }
        }

        // Find segment using binary search
        const auto seg = find_segment(key);
        if (seg == nullptr) {
            DEBUG_LOG("JazzyIndex::find: find_segment returned nullptr, returning end()");
            return end();
        }

        // Predict index using segment's model
        std::size_t predicted = seg->predict(key, key_extract_);
        predicted = detail::clamp_value<std::size_t>(predicted, seg->start_idx,
                                                      seg->end_idx > 0 ? seg->end_idx - 1 : 0);

        DEBUG_LOG("JazzyIndex::find: Predicted index %zu for key in segment [%zu-%zu]",
                  predicted, seg->start_idx, seg->end_idx);

        const T* begin = base_;

        // Check predicted position first
        if (equal(begin[predicted], key)) {
            DEBUG_LOG("JazzyIndex::find: Found exact match at predicted index %zu", predicted);
            return begin + predicted;
        }

        // One bounded search over the window the model's error bound promises. The
        // previous exponential-doubling scheme issued log2(max_error) separate
        // lower_bound calls per lookup; with max_error routinely in the hundreds that
        // was up to eleven branchy passes where one suffices.
        const std::size_t radius = std::max<std::size_t>(
            seg->max_error + detail::SEARCH_RADIUS_MARGIN, detail::MIN_SEARCH_RADIUS);
        const std::size_t window_begin = predicted - std::min(predicted - seg->start_idx, radius);
        const std::size_t window_end = std::min(predicted + radius + 1, seg->end_idx);

        DEBUG_LOG("JazzyIndex::find: Bounded search [%zu-%zu) radius=%zu",
                  window_begin, window_end, radius);

        const T* found = std::lower_bound(begin + window_begin, begin + window_end, key, comp_);
        if (found != begin + window_end && equal(*found, key)) {
            DEBUG_LOG("JazzyIndex::find: Found match at index %zu (bounded search)",
                      static_cast<std::size_t>(found - begin));
            return found;
        }

        // Landing strictly inside the window means the insertion point is inside it, so
        // the key is absent from the whole range. Only a result pinned to an edge means
        // the model under-reported its error and the key may lie beyond that edge.
        if (found == begin + window_begin && window_begin > seg->start_idx) {
            DEBUG_LOG("JazzyIndex::find: Window undershot, searching [%zu-%zu)",
                      seg->start_idx, window_begin);
            found = std::lower_bound(begin + seg->start_idx, begin + window_begin, key, comp_);
            if (found != begin + window_begin && equal(*found, key)) {
                return found;
            }
        } else if (found == begin + window_end && window_end < seg->end_idx) {
            DEBUG_LOG("JazzyIndex::find: Window overshot, searching [%zu-%zu)",
                      window_end, seg->end_idx);
            found = std::lower_bound(begin + window_end, begin + seg->end_idx, key, comp_);
            if (found != begin + seg->end_idx && equal(*found, key)) {
                return found;
            }
        }

        DEBUG_LOG("JazzyIndex::find: Not found, returning end()");
        return end();
    }

    // Throughput-oriented lookup for independent keys. Uniform indexes retain the
    // scalar path because arithmetic segment routing is already optimal there.
    template <std::size_t N>
    void find_batch(const std::array<T, N>& keys,
                    std::array<const_iterator, N>& results) const {
        static_assert(N > 0);
        if (!is_built() || size_ == 0 || is_uniform_ || is_dense_) {
            for (std::size_t lane = 0; lane < N; ++lane) results[lane] = find(keys[lane]);
            return;
        }

        std::array<detail::BatchLookupState<T>, N> lanes{};
        for (std::size_t lane = 0; lane < N; ++lane) {
            detail::validate_finite_key(keys[lane], key_extract_);
            results[lane] = end();
            if (!comp_(keys[lane], base_[0]) && !comp_(base_[size_ - 1], keys[lane])) {
                lanes[lane].right = num_segments_;
                if constexpr (supports_radix_routing) {
                    if (!radix_offsets_.empty()) {
                        const auto key = static_cast<std::uint64_t>(keys[lane]);
                        const std::size_t bucket = radix_bucket(
                            key - static_cast<std::uint64_t>(min_));
                        lanes[lane].left = radix_offsets_[bucket];
                        const std::size_t right = std::min<std::size_t>(num_segments_,
                            static_cast<std::size_t>(radix_offsets_[bucket + 1]) + 1);
                        lanes[lane].right = std::max<std::size_t>(1, right - lanes[lane].left);
                    }
                }
                lanes[lane].active = true;
            }
        }

        std::size_t routing_length = 0;
        for (const auto& state : lanes) {
            if (state.active) routing_length = std::max(routing_length, state.right);
        }
        while (routing_length > 1) {
            for (std::size_t lane = 0; lane < N; ++lane) {
                auto& state = lanes[lane];
                if (!state.active) continue;
                const std::size_t half = state.right / 2;
                const std::size_t candidate = state.left + half;
                state.left = comp_(segments_[candidate].max_val, keys[lane])
                    ? candidate : state.left;
                state.right -= half;
            }
            routing_length -= routing_length / 2;
        }
        for (std::size_t lane = 0; lane < N; ++lane) {
            auto& state = lanes[lane];
            if (!state.active) continue;
            const std::size_t index = state.left +
                static_cast<std::size_t>(comp_(segments_[state.left].max_val, keys[lane]));
            state.segment = &segments_[std::min(index, num_segments_ - 1)];
            state.active = false;
        }

        for (std::size_t lane = 0; lane < N; ++lane) {
            auto& state = lanes[lane];
            if (state.segment == nullptr) continue;
            state.right = detail::clamp_value<std::size_t>(
                state.segment->predict(keys[lane], key_extract_), state.segment->start_idx,
                state.segment->end_idx - 1);
        }
        for (std::size_t lane = 0; lane < N; ++lane) {
            auto& state = lanes[lane];
            if (state.segment == nullptr) continue;
            const std::size_t predicted = state.right;
            if (equal(base_[predicted], keys[lane])) {
                results[lane] = base_ + predicted;
                state.segment = nullptr;
                continue;
            }
            const std::size_t radius = std::max<std::size_t>(
                state.segment->max_error + detail::SEARCH_RADIUS_MARGIN,
                detail::MIN_SEARCH_RADIUS);
            const std::size_t first = predicted - std::min(predicted - state.segment->start_idx,
                                                            radius);
            const std::size_t last = std::min(predicted + radius + 1, state.segment->end_idx);
            state.low = base_ + first;
            state.high = base_ + last;
            state.bounded_begin = state.low;
            state.bounded_end = state.high;
            state.active = true;
        }

        const auto lower_bound_stage = [&lanes, &keys, this]() {
            bool searching = true;
            while (searching) {
                searching = false;
                for (std::size_t lane = 0; lane < N; ++lane) {
                    auto& state = lanes[lane];
                    if (!state.active) continue;
                    if (state.low == state.high) {
                        state.active = false;
                        continue;
                    }
                    const T* mid = state.low + (state.high - state.low) / 2;
                    if (comp_(*mid, keys[lane])) state.low = mid + 1;
                    else state.high = mid;
                    searching = true;
                }
            }
        };
        lower_bound_stage();

        for (std::size_t lane = 0; lane < N; ++lane) {
            auto& state = lanes[lane];
            if (state.segment == nullptr) continue;
            const T* found = state.low;
            const T* segment_begin = base_ + state.segment->start_idx;
            const T* segment_end = base_ + state.segment->end_idx;
            if (found != state.bounded_end && equal(*found, keys[lane])) {
                results[lane] = found;
                state.segment = nullptr;
            } else if (found == state.bounded_begin && state.bounded_begin > segment_begin) {
                state.low = segment_begin;
                state.high = state.bounded_begin;
                state.left = 1;
                state.active = true;
            } else if (found == state.bounded_end && state.bounded_end < segment_end) {
                state.low = state.bounded_end;
                state.high = segment_end;
                state.left = 2;
                state.active = true;
            } else {
                state.segment = nullptr;
            }
        }
        lower_bound_stage();
        for (std::size_t lane = 0; lane < N; ++lane) {
            auto& state = lanes[lane];
            if (state.segment == nullptr) continue;
            const T* fallback_end = state.left == 1 ? state.bounded_begin
                                                     : base_ + state.segment->end_idx;
            if (state.low != fallback_end && equal(*state.low, keys[lane])) {
                results[lane] = state.low;
            }
        }
    }

    // Find the range of elements equal to the given value
    // Returns a pair of iterators [lower, upper) where all elements in the range are equivalent to value
    // For missing values, returns [position, position) where position is where the value would be inserted
    [[nodiscard]] std::pair<const_iterator, const_iterator> equal_range(const T& value) const {
        DEBUG_LOG("JazzyIndex::equal_range: Called for value");
        if (!is_built() || size_ == 0) return {end(), end()};
        detail::validate_finite_key(value, key_extract_);
        return std::equal_range(base_, end(), value, comp_);
    }

    // Find the first occurrence of a value (lower bound)
    [[nodiscard]] const_iterator find_lower_bound(const T& value) const {
        DEBUG_LOG("JazzyIndex::find_lower_bound: Called");
        if (!is_built() || size_ == 0) return end();
        detail::validate_finite_key(value, key_extract_);
        return std::lower_bound(base_, end(), value, comp_);
    }

    // Find one past the last occurrence of a value (upper bound)
    [[nodiscard]] const_iterator find_upper_bound(const T& value) const {
        DEBUG_LOG("JazzyIndex::find_upper_bound: Called");
        if (!is_built() || size_ == 0) return end();
        detail::validate_finite_key(value, key_extract_);
        return std::upper_bound(base_, end(), value, comp_);
    }

    [[nodiscard]] const_iterator begin() const noexcept { return base_; }
    [[nodiscard]] const_iterator end() const noexcept { return base_ + size_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t num_segments() const noexcept { return num_segments_; }

    // Segments live on the heap now, so sizeof() no longer describes the index.
    [[nodiscard]] std::size_t size_in_bytes() const noexcept {
        return segments_.capacity() * sizeof(detail::Segment<T>) +
               radix_offsets_.capacity() * sizeof(std::uint32_t);
    }
    [[nodiscard]] bool is_built() const noexcept { return built_; }

    // Parallel build API - requires #include "jazzy_index_parallel.hpp"
    // Prepare independent build tasks for custom threading model
    std::vector<parallel::BuildTask<T, Compare, KeyExtractor>>
    prepare_build_tasks(const T* first, const T* last,
                       Compare comp = Compare{},
                       KeyExtractor key_extract = KeyExtractor{});

    // Finalize build after executing tasks
    void finalize_build(const std::vector<detail::SegmentAnalysis<T>>& results);

    // Parallel build using std::async (convenience method)
    void build_parallel(const T* first, const T* last,
                       Compare comp = Compare{},
                       KeyExtractor key_extract = KeyExtractor{});

    // Iterator-based parallel build method
    template <typename Iterator>
        requires std::random_access_iterator<Iterator> &&
                 std::contiguous_iterator<Iterator> &&
                 (!std::is_pointer_v<Iterator>) &&
                 std::same_as<typename std::iterator_traits<Iterator>::value_type, T>
    void build_parallel(Iterator first, Iterator last, Compare comp = Compare{}, KeyExtractor key_extract = KeyExtractor{}) {
        // Convert contiguous iterators to pointers for internal use
        build_parallel(std::to_address(first), std::to_address(last), comp, key_extract);
    }

    // Friend declarations
    template <typename U, SegmentCount S, typename C, typename K>
    friend std::string export_index_metadata(const JazzyIndex<U, S, C, K>& index);

    template <typename U, SegmentCount S, typename C, typename K>
    friend class parallel::ParallelBuilder;

private:
    [[nodiscard]] const detail::Segment<T>* find_segment(const T& value) const {
        DEBUG_LOG("find_segment: Called with is_uniform=%d, num_segments=%zu", is_uniform_, num_segments_);

        if (num_segments_ == 0) {
            DEBUG_LOG("find_segment: No segments, returning nullptr");
            return nullptr;
        }

        // Fast path: O(1) arithmetic lookup for uniform data
        if (is_uniform_) {
            const double key_val = static_cast<double>(std::invoke(key_extract_, value));
            const double min_key = static_cast<double>(std::invoke(key_extract_, min_));
            const double offset = key_val - min_key;
            const double scaled_offset = offset * segment_scale_;
            std::size_t seg_idx = 0;
            if (scaled_offset > 0.0) {
                seg_idx = !std::isfinite(scaled_offset) ||
                          scaled_offset >= static_cast<double>(num_segments_)
                    ? num_segments_ - 1
                    : static_cast<std::size_t>(scaled_offset);
            }

            DEBUG_LOG("find_segment: UNIFORM path - key_val=%.4f, offset=%.4f, segment_scale=%.6f, seg_idx=%zu",
                      key_val, offset, segment_scale_, seg_idx);

            // Clamp to valid range
            if (seg_idx >= num_segments_) {
                seg_idx = num_segments_ - 1;
                DEBUG_LOG("find_segment: Clamped seg_idx to %zu", seg_idx);
            }

            // Verify we got the right segment (should always be true for uniform data)
            const auto& seg = segments_[seg_idx];
            if (!comp_(value, seg.min_val) && !comp_(seg.max_val, value)) {
                DEBUG_LOG("find_segment: UNIFORM succeeded, returning segment %zu [%zu-%zu]",
                          seg_idx, seg.start_idx, seg.end_idx);
                return &seg;
            }

            DEBUG_LOG("find_segment: UNIFORM verification failed, falling back to binary search");
            // Fallback to binary search if arithmetic failed (rare)
        }

        std::size_t left = 0;
        std::size_t right = num_segments_;
        if constexpr (supports_radix_routing) {
            if (!radix_offsets_.empty()) {
                const auto key = static_cast<std::uint64_t>(value);
                const std::size_t bucket = radix_bucket(
                    key - static_cast<std::uint64_t>(min_));
                left = radix_offsets_[bucket];
                right = std::min<std::size_t>(
                    num_segments_, static_cast<std::size_t>(radix_offsets_[bucket + 1]) + 1);
                if (right <= left) {
                    right = std::min(num_segments_, left + 1);
                }
                if (!has_adjacent_duplicates_) {
                    // One directory candidate still lets its local search reject in-range gaps.
                    if (right - left == 1) return &segments_[left];
                    if (right - left <= 4) {
                        auto it = segments_.begin() + static_cast<std::ptrdiff_t>(left);
                        const auto last = segments_.begin() + static_cast<std::ptrdiff_t>(right);
                        while (it != last && static_cast<std::uint64_t>(it->max_val) < key) ++it;
                        return it == last ? &segments_.back() : &*it;
                    }
                    const auto it = std::lower_bound(
                        segments_.begin() + static_cast<std::ptrdiff_t>(left),
                        segments_.begin() + static_cast<std::ptrdiff_t>(right), key,
                        [](const detail::Segment<T>& segment, std::uint64_t value) {
                            return static_cast<std::uint64_t>(segment.max_val) < value;
                        });
                    return it == segments_.begin() + static_cast<std::ptrdiff_t>(right)
                        ? &segments_.back()
                        : &*it;
                }
            }
        }

        // Duplicate keys may span segments, so keep the interval search semantics.
        DEBUG_LOG("find_segment: Using binary search (non-uniform or fallback)");
        while (left < right) {
            const std::size_t mid = left + (right - left) / 2;

            if (comp_(value, segments_[mid].min_val)) {
                right = mid;
            } else if (comp_(segments_[mid].max_val, value)) {
                left = mid + 1;
            } else {
                DEBUG_LOG("find_segment: Found segment %zu [%zu-%zu]", mid,
                          segments_[mid].start_idx, segments_[mid].end_idx);
                return &segments_[mid];
            }
        }

        // No segment contains the key: it fell in a gap between segments, or past the
        // end. Either way the neighbouring segment reports it absent.
        DEBUG_LOG("find_segment: Returning segment %zu (left=%zu)", left, left);
        return &segments_[std::min(left, num_segments_ - 1)];
    }

    // Segments are equal-key-width, so find_segment can index straight into them
    // instead of binary searching. Checked after refinement, since splitting a
    // segment makes the widths uneven.
    [[nodiscard]] bool check_uniformity(double total_range) const {
        if (num_segments_ <= 1 || total_range < detail::ZERO_RANGE_THRESHOLD) {
            return true;
        }
        const double expected_spacing = total_range / static_cast<double>(num_segments_);
        const double tolerance = expected_spacing * detail::UNIFORMITY_TOLERANCE;
        for (const auto& seg : segments_) {
            const double segment_range = static_cast<double>(std::invoke(key_extract_, seg.max_val)) -
                                         static_cast<double>(std::invoke(key_extract_, seg.min_val));
            if (std::abs(segment_range - expected_spacing) > tolerance) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool equal(const T& lhs, const T& rhs) const {
        return !comp_(lhs, rhs) && !comp_(rhs, lhs);
    }

    void finalize_routing() {
        radix_offsets_.clear();
        has_adjacent_duplicates_ = false;
        is_dense_ = false;
        if constexpr (supports_radix_routing) {
            if (size_ == 0) return;
            const std::uint64_t min_key = static_cast<std::uint64_t>(min_);
            const std::uint64_t range = static_cast<std::uint64_t>(max_) - min_key;
            if (is_uniform_ && range != size_ - 1) return;
            for (std::size_t i = 1; i < size_; ++i) {
                if (base_[i - 1] == base_[i]) {
                    has_adjacent_duplicates_ = true;
                    break;
                }
            }
            is_dense_ = !has_adjacent_duplicates_ && range == size_ - 1;
            if (is_dense_ || is_uniform_ || num_segments_ <= 1 ||
                num_segments_ > std::numeric_limits<std::uint32_t>::max()) {
                return;
            }
            const std::size_t final_bucket = radix_bucket(range);
            radix_offsets_.assign(final_bucket + 2, static_cast<std::uint32_t>(num_segments_));
            std::size_t next_bucket = 0;
            for (std::size_t i = 0; i < num_segments_; ++i) {
                const std::size_t segment_bucket = radix_bucket(
                    static_cast<std::uint64_t>(segments_[i].max_val) - min_key);
                while (next_bucket <= segment_bucket) {
                    radix_offsets_[next_bucket++] = static_cast<std::uint32_t>(i);
                }
            }
        }
    }

    // This preserves more directory precision near a dense low-key tail. At the
    // full uint64_t range it produces at most 425,985 entries (about 1.625 MiB).
    [[nodiscard]] static std::size_t radix_bucket(std::uint64_t delta) noexcept {
        constexpr std::uint64_t mask = (std::uint64_t{1} << LogRadixMantissaBits) - 1;
        const unsigned exponent = std::bit_width(delta | mask) - LogRadixMantissaBits;
        return (static_cast<std::size_t>(exponent) << (LogRadixMantissaBits - 1)) +
               static_cast<std::size_t>(delta >> exponent);
    }

    const T* base_{nullptr};
    std::size_t size_{0};
    T min_{};
    T max_{};
    KeyExtractor key_extract_{};
    Compare comp_{};
    std::size_t num_segments_{0};
    bool is_uniform_{false};
    bool built_{false};
    bool is_dense_{};
    double segment_scale_{0.0};
    std::vector<std::uint32_t> radix_offsets_{};
    bool has_adjacent_duplicates_{};
    // Heap-allocated because the final count is data-dependent: NumSegments only sets
    // the starting quantile split; refinement grows it subject to MIN_SPLIT_SIZE.
    std::vector<detail::Segment<T>> segments_{};

};

}  // namespace jazzy
