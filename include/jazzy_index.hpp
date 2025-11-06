#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "jazzy_index_utility.hpp"  // detail::clamp_value and arithmetic trait

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

// Model types for different segment characteristics
enum class ModelType : uint8_t {
    LINEAR,       // Most common: y = mx + b (1 FMA)
    QUADRATIC,    // Curved regions: y = ax^2 + bx + c (2 FMA)
    CUBIC,        // Highly curved: y = ax^3 + bx^2 + cx + d (3 FMA)
    EXPONENTIAL,  // Exponential growth: y = a * exp(b * x) + c
    LOGARITHMIC,  // Logarithmic growth: y = a * log(x + b) + c
    CONSTANT      // All values same: y = c (0 computation)
};

// Model selection and search tuning constants
inline constexpr std::size_t MAX_ACCEPTABLE_LINEAR_ERROR = 2;
// Linear models with error ≤MAX_ACCEPTABLE_LINEAR_ERROR are accepted immediately.
// Keeps exponential search efficient (~3 iterations: radius 2,4,8)

inline constexpr double QUADRATIC_IMPROVEMENT_THRESHOLD = 0.7;
// Quadratic must be 30% better than linear (0.7 = 70% of linear error)

inline constexpr double CUBIC_IMPROVEMENT_THRESHOLD = 0.7;
// Cubic must be 30% better than quadratic (0.7 = 70% of quadratic error)

inline constexpr std::size_t MAX_ACCEPTABLE_QUADRATIC_ERROR = 6;
// Quadratic models with error ≤6 are good enough; don't try cubic

inline constexpr std::size_t MAX_CUBIC_WORTHWHILE_ERROR = 50;
// If quadratic error >50, likely a discontinuity; cubic won't help, skip computation

inline constexpr std::size_t SEARCH_RADIUS_MARGIN = 2;
// Extra margin added to max_error for exponential search bounds

inline constexpr std::size_t MIN_SEARCH_RADIUS = 4;
// Minimum radius for exponential search to ensure reasonable coverage

inline constexpr std::size_t INITIAL_SEARCH_RADIUS = 2;
// Starting radius for exponential search (doubles each iteration: 2,4,8,16...)

inline constexpr double UNIFORMITY_TOLERANCE = 0.30;
// Allow 30% deviation in segment spacing for uniformity detection

// Numerical stability and tolerance constants
inline constexpr double ZERO_RANGE_THRESHOLD = std::numeric_limits<double>::epsilon();
// Threshold for detecting zero range (constant segments) in floating-point comparisons

inline constexpr double NUMERICAL_TOLERANCE = 1e-10;
// Tolerance for matrix determinant checks in polynomial fitting (singular matrix detection)

inline constexpr double MIN_DISTRIBUTION_SCALE = 1e-6;
// Minimum scale parameter for distributions to prevent division by zero

// Cost-based model selection constants
inline constexpr double COMPARISON_TO_FMA_COST_RATIO = 7.5;
// Binary search comparisons cost ~7.5× more than FMA (memory access + branch prediction)
// Used to estimate total lookup cost: prediction_fmas + log2(search_range) * this_ratio

// Prediction cost in FMA operations for each model type
inline constexpr double LINEAR_PREDICTION_COST = 1.0;   // y = mx + b (1 FMA)
inline constexpr double QUADRATIC_PREDICTION_COST = 2.0; // y = ax² + bx + c (2 FMA)

inline constexpr double LINEAR_ERROR_THRESHOLD = 1.5;
// Only consider more complex models if LINEAR error exceeds this threshold
// If LINEAR error ≤ 1.5, search range is ≤ 4 elements (trivial cost)
// For polynomial data, QUADRATIC can reduce error 3+ down to 0-1, saving many cycles

inline constexpr double COST_IMPROVEMENT_THRESHOLD = 0.95;
// More complex models must reduce total cost to < 95% of simpler model (5% savings)
// Prevents choosing complex models for negligible performance gains

inline constexpr double ERROR_RATIO_THRESHOLD = 0.5;
// More complex models must reduce error to < 50% of simpler model
// Ensures meaningful accuracy improvement justifies added complexity

inline constexpr double MIN_ERROR_REDUCTION = 3.0;
// Alternative threshold: absolute error reduction must be ≥ 3 units
// Used when error ratio threshold doesn't apply (e.g., very small errors)

// Model parameters union shared across all model types
union ModelParams {
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
        float a;  // scale factor
        float b;  // exponent coefficient
        float c;  // offset
    } exponential;
    struct {
        float a;  // scale factor
        float b;  // log argument offset
        float c;  // result offset
    } logarithmic;
    struct {
        std::size_t constant_idx;
    } constant;
};

// Common prediction logic used by both Segment and SegmentFinder
// Returns predicted index from value using the given model
inline std::size_t predict_with_model(
    double value,
    ModelType model_type,
    const ModelParams& params,
    std::size_t fallback_idx = 0) noexcept {

    switch (model_type) {
        case ModelType::LINEAR: {
            const double pred = std::fma(value, static_cast<double>(params.linear.slope), static_cast<double>(params.linear.intercept));
            return static_cast<std::size_t>(std::max(0.0, pred));
        }
        case ModelType::QUADRATIC: {
            const double pred = std::fma(value,
                                        std::fma(value, static_cast<double>(params.quadratic.a), static_cast<double>(params.quadratic.b)),
                                        static_cast<double>(params.quadratic.c));
            return static_cast<std::size_t>(std::max(0.0, pred));
        }
        case ModelType::CUBIC: {
            const double pred = std::fma(value,
                                        std::fma(value,
                                                std::fma(value, static_cast<double>(params.cubic.a), static_cast<double>(params.cubic.b)),
                                                static_cast<double>(params.cubic.c)),
                                        static_cast<double>(params.cubic.d));
            return static_cast<std::size_t>(std::max(0.0, pred));
        }
        case ModelType::EXPONENTIAL: {
            const double pred = static_cast<double>(params.exponential.a) * std::exp(static_cast<double>(params.exponential.b) * value) + static_cast<double>(params.exponential.c);
            return static_cast<std::size_t>(std::max(0.0, pred));
        }
        case ModelType::LOGARITHMIC: {
            const double arg = value + static_cast<double>(params.logarithmic.b);
            if (arg <= 0.0) {
                return 0;
            }
            const double pred = static_cast<double>(params.logarithmic.a) * std::log(arg) + static_cast<double>(params.logarithmic.c);
            return static_cast<std::size_t>(std::max(0.0, pred));
        }
        case ModelType::CONSTANT:
            return params.constant.constant_idx;
        default:
            return fallback_idx;
    }
}

// Learned model structure for segment finding
// Predicts which segment contains a value
struct SegmentFinder {
    ModelType model_type{ModelType::LINEAR};
    uint32_t max_error{0};
    ModelParams params{};

    [[nodiscard]] std::size_t predict(double value) const noexcept {
        return predict_with_model(value, model_type, params, 0);
    }
};

// Segment descriptor with optimal model
template <typename T>
struct alignas(64) Segment {  // Cache line aligned
    // Hot path: segment finding and prediction (grouped for cache locality)
    T min_val;
    T max_val;
    ModelType model_type;
    uint32_t max_error;

    // Model parameters (hot path for predict())
    // Using float instead of double to keep struct within 64 bytes
    alignas(8) ModelParams params;

    // Warm path: clamping and bounds
    std::size_t start_idx;
    std::size_t end_idx;

    // Inline prediction for maximum performance in hot path
    template <typename KeyExtractor = jazzy::identity>
    [[nodiscard]] std::size_t predict(const T& value, KeyExtractor key_extract = KeyExtractor{}) const
        noexcept(std::is_nothrow_invocable_v<KeyExtractor, const T&>) {
        const double key_val = static_cast<double>(std::invoke(key_extract, value));

        switch (model_type) {
            case ModelType::LINEAR: {
                const double pred = std::fma(key_val, params.linear.slope, params.linear.intercept);
                return static_cast<std::size_t>(std::max(0.0, pred));
            }
            case ModelType::QUADRATIC: {
                const double pred = std::fma(key_val,
                                            std::fma(key_val, params.quadratic.a, params.quadratic.b),
                                            params.quadratic.c);
                return static_cast<std::size_t>(std::max(0.0, pred));
            }
            case ModelType::CUBIC: {
                const double pred = std::fma(key_val,
                                            std::fma(key_val,
                                                    std::fma(key_val, params.cubic.a, params.cubic.b),
                                                    params.cubic.c),
                                            params.cubic.d);
                return static_cast<std::size_t>(std::max(0.0, pred));
            }
            case ModelType::EXPONENTIAL: {
                // y = a * exp(b * x) + c
                const double pred = params.exponential.a * std::exp(params.exponential.b * key_val) + params.exponential.c;
                return static_cast<std::size_t>(std::max(0.0, pred));
            }
            case ModelType::LOGARITHMIC: {
                // y = a * log(x + b) + c
                const double arg = key_val + params.logarithmic.b;
                if (arg <= 0.0) {
                    return 0;  // Handle domain error
                }
                const double pred = params.logarithmic.a * std::log(arg) + params.logarithmic.c;
                return static_cast<std::size_t>(std::max(0.0, pred));
            }
            case ModelType::CONSTANT:
                return params.constant.constant_idx;
            default:
                return start_idx;
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
        return result;
    };

    if (end <= start)
        return make_constant();

    const std::size_t n = end - start;

    // For sorted data, min/max are at endpoints (extract keys for comparison)
    const double min_val = static_cast<double>(std::invoke(key_extract, data[start]));
    const double max_val = static_cast<double>(std::invoke(key_extract, data[end - 1]));
    const double value_range = max_val - min_val;

    // Check for constant segment (zero range)
    if (value_range < detail::ZERO_RANGE_THRESHOLD) {
        return make_constant();
    }

    // Precompute linear model parameters
    const double slope = static_cast<double>(n - 1) / value_range;
    const double intercept = static_cast<double>(start) - slope * min_val;

    result.linear_a = slope;
    result.linear_b = intercept;

    // Single pass: compute linear error AND quadratic/cubic sums simultaneously
    std::size_t linear_max_error = 0;
    double linear_total_error = 0.0;

    double sum_x = 0.0, sum_x2 = 0.0, sum_x3 = 0.0, sum_x4 = 0.0, sum_x5 = 0.0, sum_x6 = 0.0;
    double sum_y = 0.0, sum_xy = 0.0, sum_x2y = 0.0, sum_x3y = 0.0;

    const double x_min = min_val;
    const double x_scale = value_range;

    bool all_same = true;
    const T first_val = data[start];

    // Helper for equality check using comparator (a == b iff !(a < b) && !(b < a))
    auto equal = [&comp](const T& a, const T& b) {
        return !comp(a, b) && !comp(b, a);
    };

    for (std::size_t i = start; i < end; ++i) {
        const T current_val = data[i];
        const double key_val = static_cast<double>(std::invoke(key_extract, current_val));

        // Check if all values are identical using comparator
        if (all_same && !equal(current_val, first_val)) {
            all_same = false;
        }

        // Compute linear error
        const double pred_double = std::fma(key_val, slope, intercept);
        const double error = std::abs(pred_double - static_cast<double>(i));
        linear_max_error = std::max(linear_max_error, static_cast<std::size_t>(std::ceil(error)));
        linear_total_error += error;

        // Accumulate quadratic/cubic sums (normalized x for numerical stability)
        const double x_normalized = (key_val - x_min) / x_scale;
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

    // If linear is good enough, use it
    if (linear_max_error <= MAX_ACCEPTABLE_LINEAR_ERROR) {
        result.best_model = ModelType::LINEAR;
        result.max_error = linear_max_error;
        result.mean_error = linear_mean_error;
        return result;
    }

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

        // Measure quadratic error using normalized x
        std::size_t quad_max_error = 0;
        double quad_total_error = 0.0;

        for (std::size_t i = start; i < end; ++i) {
            const double key_val = static_cast<double>(std::invoke(key_extract, data[i]));
            const double x_normalized = (key_val - x_min) / x_scale;
            const double pred_double = std::fma(x_normalized, std::fma(x_normalized, a, b), c);
            const double error = std::abs(pred_double - static_cast<double>(i));
            quad_max_error = std::max(quad_max_error, static_cast<std::size_t>(std::ceil(error)));
            quad_total_error += error;
        }

        const double quad_mean_error = quad_total_error / n_double;

        // Choose quadratic if it's significantly better
        if (quad_max_error < linear_max_error * QUADRATIC_IMPROVEMENT_THRESHOLD) {
            // Transform coefficients from normalized space back to original space
            // Original: index = a*x_norm^2 + b*x_norm + c, where x_norm = (x - x_min) / x_scale
            // Want: index = a'*x^2 + b'*x + c'
            const double x_scale_sq = x_scale * x_scale;
            const double quad_a_transformed = a / x_scale_sq;
            const double quad_b_transformed = b / x_scale - 2.0 * a * x_min / x_scale_sq;
            const double quad_c_transformed = a * x_min * x_min / x_scale_sq - b * x_min / x_scale + c;

            // Check monotonicity: derivative f'(x) = 2*a*x + b must be non-negative over [min_val, max_val]
            // For a search index, predictions must be monotonically INCREASING
            // Since f'(x) is linear, we just need to check both endpoints
            const double derivative_at_min = 2.0 * quad_a_transformed * min_val + quad_b_transformed;
            const double derivative_at_max = 2.0 * quad_a_transformed * max_val + quad_b_transformed;
            const bool is_monotonic = (derivative_at_min >= 0.0) && (derivative_at_max >= 0.0);

            // Only accept quadratic if it's monotonic
            if (is_monotonic) {
                // Check if we should try cubic for even better fit
                // Only try cubic if error is in "sweet spot" (6 < error < 50)
                // High error (>50) likely indicates discontinuity where cubic won't help
                if (quad_max_error > MAX_ACCEPTABLE_QUADRATIC_ERROR &&
                    quad_max_error < MAX_CUBIC_WORTHWHILE_ERROR) {
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

                        // Measure cubic error in normalized space
                        std::size_t cubic_max_error = 0;
                        double cubic_total_error = 0.0;

                        for (std::size_t i = start; i < end; ++i) {
                            const double key_value = static_cast<double>(std::invoke(key_extract, data[i]));
                            const double x_norm = (key_value - x_min) / x_scale;
                            const double pred = std::fma(x_norm,
                                                        std::fma(x_norm,
                                                                std::fma(x_norm, cubic_a_norm, cubic_b_norm),
                                                                cubic_c_norm),
                                                        cubic_d_norm);
                            const double error = std::abs(pred - static_cast<double>(i));
                            cubic_max_error = std::max(cubic_max_error, static_cast<std::size_t>(std::ceil(error)));
                            cubic_total_error += error;
                        }

                        // Choose cubic if it's significantly better than quadratic
                        if (cubic_max_error < quad_max_error * CUBIC_IMPROVEMENT_THRESHOLD) {
                            // Transform coefficients from normalized space to original space
                            // Original: y = a*x_norm^3 + b*x_norm^2 + c*x_norm + d
                            // where x_norm = (x - x_min) / x_scale
                            // Expand: y = a*((x-x_min)/s)^3 + b*((x-x_min)/s)^2 + c*((x-x_min)/s) + d

                            const double s = x_scale;
                            const double s2 = s * s;
                            const double s3 = s2 * s;
                            const double m = x_min;
                            const double m2 = m * m;
                            const double m3 = m2 * m;

                            // Coefficients in original space (after algebraic expansion)
                            const double cubic_a_orig = cubic_a_norm / s3;
                            const double cubic_b_orig = cubic_b_norm / s2 - 3.0 * cubic_a_norm * m / s3;
                            const double cubic_c_orig = cubic_c_norm / s
                                                      - 2.0 * cubic_b_norm * m / s2
                                                      + 3.0 * cubic_a_norm * m2 / s3;
                            const double cubic_d_orig = cubic_d_norm
                                                      - cubic_c_norm * m / s
                                                      + cubic_b_norm * m2 / s2
                                                      - cubic_a_norm * m3 / s3;

                            // Check monotonicity: f'(x) = 3*a*x^2 + 2*b*x + c must be >= 0 over [min_val, max_val]
                            // For cubic, derivative is quadratic, so we check at critical points and endpoints
                            auto cubic_derivative = [&](double x) -> double {
                                return 3.0 * cubic_a_orig * x * x + 2.0 * cubic_b_orig * x + cubic_c_orig;
                            };

                            bool cubic_is_monotonic = true;

                            // Check endpoints
                            if (cubic_derivative(min_val) < 0.0 || cubic_derivative(max_val) < 0.0) {
                                cubic_is_monotonic = false;
                            }

                            // Check critical points (where f''(x) = 0)
                            // f''(x) = 6*a*x + 2*b = 0 => x = -b/(3*a)
                            if (cubic_is_monotonic && std::abs(cubic_a_orig) > NUMERICAL_TOLERANCE) {
                                const double critical_x = -cubic_b_orig / (3.0 * cubic_a_orig);
                                if (critical_x >= min_val && critical_x <= max_val) {
                                    if (cubic_derivative(critical_x) < 0.0) {
                                        cubic_is_monotonic = false;
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
                                return result;
                            }
                        }
                    }
                }

                // Use quadratic if cubic didn't work out
                result.best_model = ModelType::QUADRATIC;
                result.quad_a = quad_a_transformed;
                result.quad_b = quad_b_transformed;
                result.quad_c = quad_c_transformed;
                result.max_error = quad_max_error;
                result.mean_error = quad_mean_error;
                return result;
            }
            // If non-monotonic, fall through to use linear model instead
        }
    }

    // Default to linear
    result.best_model = ModelType::LINEAR;
    result.max_error = linear_max_error;
    result.mean_error = linear_mean_error;
    return result;
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

    static_assert(NumSegments > 0 && NumSegments <= 4096,
                  "NumSegments must be in range [1, 4096]");

    // Type constraint: KeyExtractor must be callable with const T&
    static_assert(std::is_invocable_v<KeyExtractor, const T&>,
                  "KeyExtractor must be callable with const T&");

    // Type constraint: KeyExtractor must return an arithmetic type
    using KeyType = std::invoke_result_t<KeyExtractor, const T&>;
    using KeyTypeClean = typename std::remove_cv<typename std::remove_reference<KeyType>::type>::type;
    static_assert(std::is_arithmetic_v<KeyTypeClean>,
                  "KeyExtractor must return an arithmetic type (int, double, etc.)");

    // Type constraint: Compare must be callable with two const T& and return bool
    static_assert(std::is_invocable_r_v<bool, Compare, const T&, const T&>,
                  "Compare must be callable with (const T&, const T&) and return bool");

public:
    JazzyIndex() = default;

    JazzyIndex(const T* first, const T* last, Compare comp = Compare{}, KeyExtractor key_extract = KeyExtractor{}) {
        build(first, last, comp, key_extract);
    }

    void build(const T* first, const T* last, Compare comp = Compare{}, KeyExtractor key_extract = KeyExtractor{}) {
        base_ = first;
        size_ = static_cast<std::size_t>(last - first);
        key_extract_ = key_extract;
        comp_ = comp;

        if (size_ == 0) {
            return;
        }

        min_ = base_[0];
        max_ = base_[size_ - 1];

        if (size_ == 1) {
            // Single element
            segments_[0].min_val = min_;
            segments_[0].max_val = max_;
            segments_[0].start_idx = 0;
            segments_[0].end_idx = 1;
            segments_[0].model_type = detail::ModelType::CONSTANT;
            segments_[0].max_error = 0;
            segments_[0].params.constant.constant_idx = 0;
            num_segments_ = 1;
            return;
        }

        // Build quantile-based segments
        const std::size_t actual_segments = std::min(NumSegments, size_);
        num_segments_ = actual_segments;

        for (std::size_t i = 0; i < actual_segments; ++i) {
            const std::size_t start = (i * size_) / actual_segments;
            const std::size_t end = ((i + 1) * size_) / actual_segments;

            auto& seg = segments_[i];
            seg.min_val = base_[start];
            seg.max_val = base_[end - 1];
            seg.start_idx = start;
            seg.end_idx = end;

            // Verify monotonicity: check that this segment's min is >= previous segment's max
            if (i > 0 && comp_(seg.min_val, segments_[i - 1].max_val)) {
                throw std::runtime_error(
                    "Input data is not sorted. JazzyIndex requires sorted data. "
                    "Please sort your data before building the index."
                );
            }

            // Analyze segment and choose best model
            const auto analysis = detail::analyze_segment(base_, start, end, comp_, key_extract_);

            seg.model_type = analysis.best_model;

            // Check if prediction error exceeds uint32_t limit
            if (analysis.max_error > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error(
                    "Segment prediction error exceeds uint32_t limit. "
                    "Data distribution is too extreme for indexing. "
                    "Consider using fewer segments or preprocessing the data."
                );
            }
            seg.max_error = static_cast<uint32_t>(analysis.max_error);

            switch (analysis.best_model) {
                case detail::ModelType::LINEAR:
                    seg.params.linear.slope = static_cast<float>(analysis.linear_a);
                    seg.params.linear.intercept = static_cast<float>(analysis.linear_b);
                    break;
                case detail::ModelType::QUADRATIC:
                    seg.params.quadratic.a = static_cast<float>(analysis.quad_a);
                    seg.params.quadratic.b = static_cast<float>(analysis.quad_b);
                    seg.params.quadratic.c = static_cast<float>(analysis.quad_c);
                    break;
                case detail::ModelType::CUBIC:
                    seg.params.cubic.a = static_cast<float>(analysis.cubic_a);
                    seg.params.cubic.b = static_cast<float>(analysis.cubic_b);
                    seg.params.cubic.c = static_cast<float>(analysis.cubic_c);
                    seg.params.cubic.d = static_cast<float>(analysis.cubic_d);
                    break;
                case detail::ModelType::CONSTANT:
                    seg.params.constant.constant_idx = start;
                    break;
                default:
                    break;
            }
        }

        // Build model for finding segments
        build_segment_finder();
    }

    [[nodiscard]] const T* find(const T& key) const {
        // Return end iterator if index not built or empty
        if (!is_built() || size_ == 0) {
            return base_ + size_;
        }

        // Bounds check
        const T& first_value = base_[0];
        const T& last_value = base_[size_ - 1];
        if (comp_(key, first_value) || comp_(last_value, key)) {
            return base_ + size_;
        }

        // Find segment using binary search
        const auto seg = find_segment(key);
        if (seg == nullptr) {
            return base_ + size_;
        }

        // Predict index using segment's model
        std::size_t predicted = seg->predict(key, key_extract_);
        predicted = detail::clamp_value<std::size_t>(predicted, seg->start_idx,
                                                      seg->end_idx > 0 ? seg->end_idx - 1 : 0);

        const T* begin = base_;

        // Check predicted position first
        if (equal(begin[predicted], key)) {
            return begin + predicted;
        }

        // Determine search direction using one comparison
        const bool search_left = comp_(key, begin[predicted]);
        const std::size_t max_radius = std::max<std::size_t>(seg->max_error + detail::SEARCH_RADIUS_MARGIN, detail::MIN_SEARCH_RADIUS);

        if (search_left) {
            // Key is less than predicted value, search leftward
            // Track the rightmost position we've searched to avoid overlaps
            std::size_t right_boundary = predicted;  // We've checked predicted, don't search it again

            // Exponentially expand leftward: check radii 1, 2, 4, 8...
            for (std::size_t radius = 1; radius <= max_radius; radius <<= 1) {
                const std::size_t left_pos = predicted > radius ? predicted - radius : seg->start_idx;

                // If left_pos >= right_boundary, no unexplored region remains
                if (left_pos >= right_boundary) break;

                // Search the new range [left_pos, right_boundary)
                const T* found = std::lower_bound(begin + left_pos, begin + right_boundary, key, comp_);
                if (found != begin + right_boundary && equal(*found, key)) {
                    return found;
                }

                right_boundary = left_pos;  // Update boundary for next iteration
            }

            // Fallback: search any remaining unsearched left region
            if (right_boundary > seg->start_idx) {
                const T* found = std::lower_bound(begin + seg->start_idx, begin + right_boundary, key, comp_);
                if (found != begin + right_boundary && equal(*found, key)) {
                    return found;
                }
            }
        } else {
            // Key is greater than predicted value, search rightward
            // Track the leftmost position we've searched to avoid overlaps
            std::size_t left_boundary = predicted + 1;  // We've checked predicted, start after it

            // Exponentially expand rightward: check radii 1, 2, 4, 8...
            for (std::size_t radius = 1; radius <= max_radius; radius <<= 1) {
                const std::size_t right_pos = std::min<std::size_t>(predicted + radius + 1, seg->end_idx);

                // If right_pos <= left_boundary, no unexplored region remains
                if (right_pos <= left_boundary) break;

                // Search the new range [left_boundary, right_pos)
                const T* found = std::lower_bound(begin + left_boundary, begin + right_pos, key, comp_);
                if (found != begin + right_pos && equal(*found, key)) {
                    return found;
                }

                left_boundary = right_pos;  // Update boundary for next iteration
            }

            // Fallback: search any remaining unsearched right region
            if (left_boundary < seg->end_idx) {
                const T* found = std::lower_bound(begin + left_boundary, begin + seg->end_idx, key, comp_);
                if (found != begin + seg->end_idx && equal(*found, key)) {
                    return found;
                }
            }
        }

        return base_ + size_;
    }

    // Find the range of elements equal to the given value
    // Returns a pair of pointers [lower, upper) where all elements in the range are equivalent to value
    // For missing values, returns [position, position) where position is where the value would be inserted
    [[nodiscard]] std::pair<const T*, const T*> equal_range(const T& value) const {
        const T* end = base_ + size_;

        // Handle empty index
        if (size_ == 0) {
            return std::make_pair(end, end);
        }

        // Find lower and upper bounds
        const T* lower = find_lower_bound(value);
        const T* upper = find_upper_bound(value);

        // If value is not found, both lower and upper point to insertion position
        // This matches std::equal_range behavior
        if (lower == end || !are_equivalent(*lower, value)) {
            return std::make_pair(lower, lower);
        }

        return std::make_pair(lower, upper);
    }

    // Find the first occurrence of a value (lower bound)
    [[nodiscard]] const T* find_lower_bound(const T& value) const {
        const T* end = base_ + size_;

        // Handle empty case
        if (size_ == 0) {
            return end;
        }

        // Handle out-of-range values
        if (comp_(value, base_[0])) {
            return base_;  // Value less than min, return beginning
        }
        if (comp_(base_[size_ - 1], value)) {
            return end;  // Value greater than max, return end
        }

        // Use the existing prediction mechanism to get close
        const auto* seg = find_segment(value);
        if (seg == nullptr) {
            // Shouldn't happen for in-range values, but fall back to binary search
            return std::lower_bound(base_, end, value, comp_);
        }

        std::size_t predicted_index = seg->predict(value, key_extract_);

        // Clamp to segment bounds
        predicted_index = detail::clamp_value<std::size_t>(predicted_index, seg->start_idx,
                                                           seg->end_idx > 0 ? seg->end_idx - 1 : 0);

        // Now perform a local search to find the exact lower bound
        const T* ptr = base_ + predicted_index;

        // Check if we're at a matching value (using comp_ for equivalence)
        if (are_equivalent(*ptr, value)) {
            // Scan backward to find the first occurrence
            while (ptr > base_ && are_equivalent(*(ptr - 1), value)) {
                --ptr;
            }
            return ptr;
        }

        // Otherwise, use binary search in a local range
        std::size_t search_radius = seg->max_error + detail::SEARCH_RADIUS_MARGIN;
        const T* search_begin = (ptr >= base_ + search_radius) ? (ptr - search_radius) : base_;
        const T* search_end = std::min(end, ptr + search_radius + 1);

        return std::lower_bound(search_begin, search_end, value, comp_);
    }

    // Find one past the last occurrence of a value (upper bound)
    [[nodiscard]] const T* find_upper_bound(const T& value) const {
        const T* end = base_ + size_;

        // Handle empty case
        if (size_ == 0) {
            return end;
        }

        // Handle out-of-range values
        if (comp_(value, base_[0])) {
            return base_;  // Value less than min, return beginning
        }
        if (comp_(base_[size_ - 1], value)) {
            return end;  // Value greater than max, return end
        }

        // Similar to lower_bound, but finds one past the last occurrence
        const auto* seg = find_segment(value);
        if (seg == nullptr) {
            // Shouldn't happen for in-range values, but fall back to binary search
            return std::upper_bound(base_, end, value, comp_);
        }

        std::size_t predicted_index = seg->predict(value, key_extract_);

        // Clamp to segment bounds
        predicted_index = detail::clamp_value<std::size_t>(predicted_index, seg->start_idx,
                                                           seg->end_idx > 0 ? seg->end_idx - 1 : 0);

        // Perform local search for upper bound
        const T* ptr = base_ + predicted_index;

        // Check if we're at a matching value using comp_ for equivalence
        if (are_equivalent(*ptr, value)) {
            // Scan forward to find one past the last occurrence
            while (ptr < end && are_equivalent(*ptr, value)) {
                ++ptr;
            }
            return ptr;
        }

        // Otherwise, use binary search in a local range
        std::size_t search_radius = seg->max_error + detail::SEARCH_RADIUS_MARGIN;
        const T* search_begin = (ptr >= base_ + search_radius) ? (ptr - search_radius) : base_;
        const T* search_end = std::min(end, ptr + search_radius + 1);

        return std::upper_bound(search_begin, search_end, value, comp_);
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t num_segments() const noexcept { return num_segments_; }
    [[nodiscard]] bool is_built() const noexcept { return base_ != nullptr; }

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

    // Friend declarations
    template <typename U, SegmentCount S, typename C, typename K>
    friend std::string export_index_metadata(const JazzyIndex<U, S, C, K>& index);

    template <typename U, SegmentCount S, typename C, typename K>
    friend class parallel::ParallelBuilder;

private:

    // Build a model over segment boundaries to predict which segment contains a value
    void build_segment_finder() {
        if (num_segments_ <= 1) {
            // Single segment: constant model always returns segment 0
            segment_finder_.model_type = detail::ModelType::LINEAR;
            segment_finder_.max_error = 0;
            segment_finder_.params.linear.slope = 0.0f;
            segment_finder_.params.linear.intercept = 0.0f;
            return;
        }

        // Create data points: (segment.min_val -> segment_index)
        const double min_val = static_cast<double>(std::invoke(key_extract_, segments_[0].min_val));
        const double max_val = static_cast<double>(std::invoke(key_extract_, segments_[num_segments_ - 1].max_val));
        const double value_range = max_val - min_val;

        // Handle constant case (all segment boundaries have same value)
        if (value_range < detail::ZERO_RANGE_THRESHOLD) {
            segment_finder_.model_type = detail::ModelType::LINEAR;
            segment_finder_.max_error = 0;
            segment_finder_.params.linear.slope = 0.0f;
            segment_finder_.params.linear.intercept = 0.0f;
            return;
        }

        // Normalize values for numerical stability
        const double x_scale = value_range;
        const double x_min = min_val;

        // Accumulate sums for polynomial fitting in a single pass
        double sum_x = 0, sum_x2 = 0, sum_x3 = 0, sum_x4 = 0, sum_x5 = 0, sum_x6 = 0;
        double sum_y = 0, sum_xy = 0, sum_x2y = 0, sum_x3y = 0;

        // First, try linear model
        std::size_t linear_max_error = 0;
        double linear_slope = static_cast<double>(num_segments_) / value_range;
        double linear_intercept = -linear_slope * min_val;

        for (std::size_t i = 0; i < num_segments_; ++i) {
            const double seg_min = static_cast<double>(std::invoke(key_extract_, segments_[i].min_val));
            const double x_normalized = (seg_min - x_min) / x_scale;
            const double y = static_cast<double>(i);

            // Compute linear prediction error
            const double linear_pred = std::fma(seg_min, linear_slope, linear_intercept);
            const double linear_error = std::abs(linear_pred - y);
            linear_max_error = std::max(linear_max_error, static_cast<std::size_t>(std::ceil(linear_error)));

            // Accumulate sums for higher-order models
            const double x = x_normalized;
            const double x2 = x * x;
            const double x3 = x2 * x;
            const double x4 = x2 * x2;
            const double x5 = x3 * x2;
            const double x6 = x3 * x3;

            sum_x += x;
            sum_x2 += x2;
            sum_x3 += x3;
            sum_x4 += x4;
            sum_x5 += x5;
            sum_x6 += x6;
            sum_y += y;
            sum_xy += x * y;
            sum_x2y += x2 * y;
            sum_x3y += x3 * y;
        }

        const double n = static_cast<double>(num_segments_);

        // Cost estimation helper: total_cost = prediction_cost + binary_search_cost
        // Binary search within ±max_error requires log2(2*max_error+1) comparisons
        // Each comparison is ~7.5× more expensive than FMA (memory + branches)
        auto estimate_lookup_cost = [](double prediction_fmas, double max_error) -> double {
            const double search_range = std::max(1.0, 2.0 * max_error + 1.0);
            const double search_comparisons = std::log2(search_range);
            return prediction_fmas + search_comparisons * detail::COMPARISON_TO_FMA_COST_RATIO;
        };

        // Evaluate all candidate models and choose the one with lowest total cost
        detail::ModelType best_model = detail::ModelType::LINEAR;
        double best_error = linear_max_error;
        double best_cost = estimate_lookup_cost(detail::LINEAR_PREDICTION_COST, linear_max_error);

        // Storage for model parameters (will be copied to segment_finder_ at end)
        struct {
            double quad_a, quad_b, quad_c;
            double exp_a, exp_b, exp_c;
            double log_a, log_b, log_c;
        } model_params{};

        // Try quadratic model
        // System: [Σx⁴  Σx³  Σx²] [a]   [Σx²y]
        //         [Σx³  Σx²  Σx ] [b] = [Σxy ]
        //         [Σx²  Σx   n  ] [c]   [Σy  ]

        const double det = sum_x4 * (sum_x2 * n - sum_x * sum_x)
                         - sum_x3 * (sum_x3 * n - sum_x * sum_x2)
                         + sum_x2 * (sum_x3 * sum_x - sum_x2 * sum_x2);

        if (std::abs(det) > detail::NUMERICAL_TOLERANCE) {
            const double det_a = sum_x2y * (sum_x2 * n - sum_x * sum_x)
                               - sum_xy * (sum_x3 * n - sum_x * sum_x2)
                               + sum_y * (sum_x3 * sum_x - sum_x2 * sum_x2);
            const double det_b = sum_x4 * (sum_xy * n - sum_y * sum_x)
                               - sum_x3 * (sum_x2y * n - sum_y * sum_x2)
                               + sum_x2 * (sum_x2y * sum_x - sum_xy * sum_x2);
            const double det_c = sum_x4 * (sum_x2 * sum_y - sum_x * sum_xy)
                               - sum_x3 * (sum_x3 * sum_y - sum_x * sum_x2y)
                               + sum_x2 * (sum_x3 * sum_xy - sum_x2 * sum_x2y);

            const double a = det_a / det;
            const double b = det_b / det;
            const double c = det_c / det;

            // Measure quadratic error
            double quad_max_error = 0.0;
            for (std::size_t i = 0; i < num_segments_; ++i) {
                const double seg_min = static_cast<double>(std::invoke(key_extract_, segments_[i].min_val));
                const double x_normalized = (seg_min - x_min) / x_scale;
                const double pred = std::fma(x_normalized, std::fma(x_normalized, a, b), c);
                const double error = std::abs(pred - static_cast<double>(i));
                quad_max_error = std::max(quad_max_error, error);
            }

            // Transform coefficients from normalized space to original space
            const double x_scale_sq = x_scale * x_scale;
            const double quad_a = a / x_scale_sq;
            const double quad_b = b / x_scale - 2.0 * a * x_min / x_scale_sq;
            const double quad_c = a * x_min * x_min / x_scale_sq - b * x_min / x_scale + c;

            // Check monotonicity: segment finder MUST be monotonic for correctness
            // (larger values must map to larger/equal segment indices)
            // For y = ax² + bx + c, derivative is y' = 2ax + b
            // Must be non-negative across entire value range [min_val, max_val]
            const double deriv_at_min = 2.0 * quad_a * min_val + quad_b;
            const double deriv_at_max = 2.0 * quad_a * max_val + quad_b;

            if (deriv_at_min >= 0.0 && deriv_at_max >= 0.0) {
                // Quadratic is monotonic - evaluate cost-benefit
                const double quad_cost = estimate_lookup_cost(detail::QUADRATIC_PREDICTION_COST, quad_max_error);

                // Choose QUADRATIC if total lookup cost is meaningfully lower
                // AND LINEAR's fit isn't already excellent (error > 1)
                // When LINEAR error ≤ 1, search range is ≤ 3 elements (trivial)
                // so prefer simpler model as tie-breaker to avoid degenerate quadratics
                const bool linear_needs_improvement = best_error > 1.0;

                if (linear_needs_improvement && quad_cost < best_cost * detail::COST_IMPROVEMENT_THRESHOLD) {
                    best_model = detail::ModelType::QUADRATIC;
                    best_error = quad_max_error;
                    best_cost = quad_cost;
                    model_params.quad_a = quad_a;
                    model_params.quad_b = quad_b;
                    model_params.quad_c = quad_c;
                }
            }
        }

        /* DISABLED: EXPONENTIAL and LOGARITHMIC models are too expensive
         * std::exp() and std::log() cost ~100 cycles vs LINEAR's ~4 cycles (1 FMA)
         * The prediction overhead outweighs search space reduction benefits
         * Keeping code for reference but commented out
         */

        /*
        // Try EXPONENTIAL model: y = a * exp(b * x) + c
        // Good for exponentially distributed data
        double exp_a = 0.0, exp_b = 0.0, exp_c = 0.0;
        double exp_max_error = std::numeric_limits<double>::max();
        bool exp_fit_success = false;

        {
            // For exponential fitting: segment_idx = a * exp(b * value) + c
            // Use linearization: if y = a * exp(b * x) + c, then (y - c) = a * exp(b * x)
            // Taking log: log(y - c) = log(a) + b * x
            // We need to estimate c first (minimum segment index), then fit log-linear model

            // Estimate c as the minimum segment index (offset)
            double c_est = 0.0;  // Start with 0 as offset

            // Try to fit log-linear model: log(y - c) = log(a) + b * x
            // We'll use a simplified approach: assume segment indices grow exponentially
            std::vector<double> log_y;
            log_y.reserve(num_segments_);

            bool valid_for_exp = true;
            for (std::size_t i = 0; i < num_segments_; ++i) {
                double y_val = static_cast<double>(i) - c_est;
                if (y_val <= 0.0) {
                    y_val = 0.1;  // Small positive value to avoid log(0)
                }
                log_y.push_back(std::log(y_val));
            }

            // Fit linear model to (x_i, log(y_i - c))
            double sum_x_log = 0.0, sum_log = 0.0, sum_x_x_log = 0.0, sum_x_log_x = 0.0;
            for (std::size_t i = 0; i < num_segments_; ++i) {
                const double seg_min = static_cast<double>(std::invoke(key_extract_, segments_[i].min_val));
                const double x_val = seg_min;
                const double log_val = log_y[i];
                sum_x_log += x_val;
                sum_log += log_val;
                sum_x_x_log += x_val * x_val;
                sum_x_log_x += x_val * log_val;
            }

            const double denom_log = n * sum_x_x_log - sum_x_log * sum_x_log;
            if (std::abs(denom_log) > 1e-10) {
                // b coefficient (slope in log space)
                exp_b = (n * sum_x_log_x - sum_x_log * sum_log) / denom_log;
                // log(a) (intercept in log space)
                const double log_a = (sum_log - exp_b * sum_x_log) / n;
                exp_a = std::exp(log_a);
                exp_c = c_est;

                // Only accept if b is positive (exponential growth) and a is positive
                if (exp_b > 0.0 && exp_a > 0.0) {
                    // Compute max error
                    exp_max_error = 0.0;
                    for (std::size_t i = 0; i < num_segments_; ++i) {
                        const double seg_min = static_cast<double>(std::invoke(key_extract_, segments_[i].min_val));
                        const double x_val = seg_min;
                        const double predicted = exp_a * std::exp(exp_b * x_val) + exp_c;
                        const double error = std::abs(predicted - static_cast<double>(i));
                        exp_max_error = std::max(exp_max_error, error);
                    }

                    // Check if exponential is significantly better than initial baseline (20% improvement)
                    if (exp_max_error < linear_max_error * 0.8) {
                        // Verify monotonicity: derivative = a * b * exp(b * x) should be positive
                        const double deriv_coeff = exp_a * exp_b;
                        if (deriv_coeff > 0.0) {
                            // EXPONENTIAL model is expensive (~25x cost of LINEAR due to std::exp)
                            // Only use it if accuracy is exceptional or drastically better than LINEAR
                            // This ensures the prediction cost is justified by search space reduction
                            if (exp_max_error <= detail::EXPENSIVE_MODEL_MAX_ERROR_THRESHOLD ||
                                exp_max_error < linear_max_error * detail::EXPENSIVE_MODEL_IMPROVEMENT_THRESHOLD) {
                                if (exp_max_error < best_error) {
                                    best_model = detail::ModelType::EXPONENTIAL;
                                    best_error = exp_max_error;
                                    model_params.exp_a = exp_a;
                                    model_params.exp_b = exp_b;
                                    model_params.exp_c = exp_c;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Try LOGARITHMIC model: y = a * log(x + b) + c
        // Good for logarithmically distributed data
        double log_a = 0.0, log_b = 0.0, log_c = 0.0;
        double log_max_error = std::numeric_limits<double>::max();
        bool log_fit_success = false;

        {
            // For logarithmic fitting: segment_idx = a * log(value + b) + c
            // We need b > -min_val to keep log argument positive
            // Use a simple approach: set b to shift minimum value to 1
            double b_est = 1.0 - min_val;
            if (b_est < 0.0) {
                b_est = 0.0;  // If min_val >= 1, no shift needed
            }

            // Now fit linear model: y = a * log(x + b) + c
            // This is linear in log(x + b), so we can use least squares
            std::vector<double> log_x;
            log_x.reserve(num_segments_);

            bool valid_for_log = true;
            for (std::size_t i = 0; i < num_segments_; ++i) {
                const double seg_min = static_cast<double>(std::invoke(key_extract_, segments_[i].min_val));
                const double x_val = seg_min + b_est;
                if (x_val <= 0.0) {
                    valid_for_log = false;
                    break;
                }
                log_x.push_back(std::log(x_val));
            }

            if (valid_for_log) {
                // Fit linear model to (log(x_i + b), y_i)
                double sum_logx = 0.0, sum_y_log = 0.0, sum_logx_logx = 0.0, sum_logx_y = 0.0;
                for (std::size_t i = 0; i < num_segments_; ++i) {
                    const double logx_val = log_x[i];
                    const double y_val = static_cast<double>(i);
                    sum_logx += logx_val;
                    sum_y_log += y_val;
                    sum_logx_logx += logx_val * logx_val;
                    sum_logx_y += logx_val * y_val;
                }

                const double denom_logfit = n * sum_logx_logx - sum_logx * sum_logx;
                if (std::abs(denom_logfit) > 1e-10) {
                    // a coefficient (slope)
                    log_a = (n * sum_logx_y - sum_logx * sum_y_log) / denom_logfit;
                    // c (intercept)
                    log_c = (sum_y_log - log_a * sum_logx) / n;
                    log_b = b_est;

                    // Only accept if a is positive (logarithmic growth)
                    if (log_a > 0.0) {
                        // Compute max error
                        log_max_error = 0.0;
                        for (std::size_t i = 0; i < num_segments_; ++i) {
                            const double seg_min = static_cast<double>(std::invoke(key_extract_, segments_[i].min_val));
                        const double x_val = seg_min;
                            const double predicted = log_a * std::log(x_val + log_b) + log_c;
                            const double error = std::abs(predicted - static_cast<double>(i));
                            log_max_error = std::max(log_max_error, error);
                        }

                        // Check if logarithmic is significantly better than initial baseline (20% improvement)
                        if (log_max_error < linear_max_error * 0.8) {
                            // Verify monotonicity: derivative = a / (x + b) should be positive
                            // Since a > 0 and x + b > 0 (checked above), derivative is always positive
                            // LOGARITHMIC model is expensive (~25x cost of LINEAR due to std::log)
                            // Only use it if accuracy is exceptional or drastically better than LINEAR
                            // This ensures the prediction cost is justified by search space reduction
                            if (log_max_error <= detail::EXPENSIVE_MODEL_MAX_ERROR_THRESHOLD ||
                                log_max_error < linear_max_error * detail::EXPENSIVE_MODEL_IMPROVEMENT_THRESHOLD) {
                                if (log_max_error < best_error) {
                                    best_model = detail::ModelType::LOGARITHMIC;
                                    best_error = log_max_error;
                                    model_params.log_a = log_a;
                                    model_params.log_b = log_b;
                                    model_params.log_c = log_c;
                                }
                            }
                        }
                    }
                }
            }
        }
        */

        // Select the best model among all candidates
        segment_finder_.max_error = static_cast<uint32_t>(best_error);
        segment_finder_.model_type = best_model;

        switch (best_model) {
            case detail::ModelType::QUADRATIC:
                segment_finder_.params.quadratic.a = static_cast<float>(model_params.quad_a);
                segment_finder_.params.quadratic.b = static_cast<float>(model_params.quad_b);
                segment_finder_.params.quadratic.c = static_cast<float>(model_params.quad_c);
                break;
            case detail::ModelType::EXPONENTIAL:
                segment_finder_.params.exponential.a = static_cast<float>(model_params.exp_a);
                segment_finder_.params.exponential.b = static_cast<float>(model_params.exp_b);
                segment_finder_.params.exponential.c = static_cast<float>(model_params.exp_c);
                break;
            case detail::ModelType::LOGARITHMIC:
                segment_finder_.params.logarithmic.a = static_cast<float>(model_params.log_a);
                segment_finder_.params.logarithmic.b = static_cast<float>(model_params.log_b);
                segment_finder_.params.logarithmic.c = static_cast<float>(model_params.log_c);
                break;
            case detail::ModelType::LINEAR:
            default:
                segment_finder_.params.linear.slope = static_cast<float>(linear_slope);
                segment_finder_.params.linear.intercept = static_cast<float>(linear_intercept);
                break;
        }
    }

    [[nodiscard]] const detail::Segment<T>* find_segment(const T& value) const noexcept {
        if (num_segments_ == 0) {
            return nullptr;
        }

        // Single segment case
        if (num_segments_ == 1) {
            return &segments_[0];
        }

        // Use model to predict segment index
        const double key_val = static_cast<double>(std::invoke(key_extract_, value));
        std::size_t predicted_seg = segment_finder_.predict(key_val);

        // Clamp to valid range
        if (predicted_seg >= num_segments_) {
            predicted_seg = num_segments_ - 1;
        }

        // Check predicted segment first (fast path for accurate predictions)
        const auto& predicted = segments_[predicted_seg];
        if (!comp_(value, predicted.min_val) && !comp_(predicted.max_val, value)) {
            return &predicted;
        }

        // Prediction missed - use binary search within predicted range
        // For good predictions (low max_error), this narrows the search space significantly
        const std::size_t error_margin = segment_finder_.max_error + 2;
        const std::size_t search_left = (predicted_seg >= error_margin)
            ? (predicted_seg - error_margin)
            : 0;
        const std::size_t search_right = std::min(predicted_seg + error_margin + 1, num_segments_);

        // Binary search in [search_left, search_right)
        std::size_t left = search_left;
        std::size_t right = search_right;

        while (left < right) {
            const std::size_t mid = left + (right - left) / 2;
            const auto& seg = segments_[mid];

            if (comp_(value, seg.min_val)) {
                right = mid;
            } else if (comp_(seg.max_val, value)) {
                left = mid + 1;
            } else {
                // Value is within this segment
                return &seg;
            }
        }

        // If not found in predicted range (rare), do full binary search
        left = 0;
        right = num_segments_;

        while (left < right) {
            const std::size_t mid = left + (right - left) / 2;
            const auto& seg = segments_[mid];

            if (comp_(value, seg.min_val)) {
                right = mid;
            } else if (comp_(seg.max_val, value)) {
                left = mid + 1;
            } else {
                return &seg;
            }
        }

        // Value not found in any segment
        return nullptr;
    }

    [[nodiscard]] bool equal(const T& lhs, const T& rhs) const {
        return !comp_(lhs, rhs) && !comp_(rhs, lhs);
    }

    // Check if two values are equivalent according to the comparator
    [[nodiscard]] bool are_equivalent(const T& a, const T& b) const {
        return !comp_(a, b) && !comp_(b, a);
    }

    const T* base_{nullptr};
    std::size_t size_{0};
    T min_{};
    T max_{};
    KeyExtractor key_extract_{};
    Compare comp_{};
    std::size_t num_segments_{0};
    detail::SegmentFinder segment_finder_{};
    std::array<detail::Segment<T>, NumSegments> segments_{};
};

}  // namespace jazzy
