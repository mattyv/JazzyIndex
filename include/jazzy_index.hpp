#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>

#include "jazzy_index_utility.hpp"  // detail::clamp_value and arithmetic trait

namespace jazzy {

namespace detail {

// Model types for different segment characteristics
enum class ModelType : uint8_t {
    LINEAR,      // Most common: y = mx + b (1 FMA)
    QUADRATIC,   // Curved regions: y = ax^2 + bx + c (3 FMA)
    CONSTANT,    // All values same: y = c (0 computation)
    DIRECT       // Tiny dense segments: array lookup
};

// Model selection and search tuning constants
inline constexpr std::size_t MAX_ACCEPTABLE_LINEAR_ERROR = 8;
// Linear models with error ≤8 are accepted immediately.
// Keeps exponential search efficient (~3 iterations: radius 2,4,8)

inline constexpr double QUADRATIC_IMPROVEMENT_THRESHOLD = 0.7;
// Quadratic must be 30% better than linear (0.7 = 70% of linear error)

inline constexpr std::size_t SEARCH_RADIUS_MARGIN = 2;
// Extra margin added to max_error for exponential search bounds

inline constexpr std::size_t MIN_SEARCH_RADIUS = 4;
// Minimum radius for exponential search to ensure reasonable coverage

inline constexpr std::size_t INITIAL_SEARCH_RADIUS = 2;
// Starting radius for exponential search (doubles each iteration: 2,4,8,16...)

inline constexpr double UNIFORMITY_TOLERANCE = 0.30;
// Allow 30% deviation in segment spacing for uniformity detection

// Segment descriptor with optimal model
template <typename T>
struct alignas(64) Segment {  // Cache line aligned
    // Hot path: segment finding and prediction (grouped for cache locality)
    T min_val;
    T max_val;
    ModelType model_type;
    uint8_t max_error;

    // Model parameters (hot path for predict())
    alignas(8) union {
        struct {
            double slope;
            double intercept;
        } linear;
        struct {
            double a;
            double b;
            double c;
        } quadratic;
        struct {
            std::size_t constant_idx;
        } constant;
    } params;

    // Warm path: clamping and bounds
    std::size_t start_idx;
    std::size_t end_idx;

    [[nodiscard]] std::size_t predict(T value) const noexcept {
        switch (model_type) {
            case ModelType::LINEAR: {
                const double pred = std::fma(static_cast<double>(value),
                                            params.linear.slope,
                                            params.linear.intercept);
                return static_cast<std::size_t>(pred);
            }
            case ModelType::QUADRATIC: {
                const double x = static_cast<double>(value);
                const double pred = std::fma(x,
                                            std::fma(x, params.quadratic.a, params.quadratic.b),
                                            params.quadratic.c);
                return static_cast<std::size_t>(pred);
            }
            case ModelType::CONSTANT:
                return params.constant.constant_idx;
            case ModelType::DIRECT:
            default:
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
    std::size_t max_error;
    double mean_error;
};

// Fit linear model to segment: index = slope * value + intercept
template <typename T>
[[nodiscard]] SegmentAnalysis<T> analyze_segment(const T* data,
                                                   std::size_t start,
                                                   std::size_t end) noexcept {
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

    // Check if all values are identical
    const bool all_same = (std::adjacent_find(data + start, data + end, std::not_equal_to<T>{}) == data + end);

    if (all_same) 
        return make_constant();

    // Fit linear model
    const double min_val = static_cast<double>(data[start]);
    const double max_val = static_cast<double>(data[end - 1]);
    const double value_range = max_val - min_val;

    if (value_range < std::numeric_limits<double>::epsilon()) {
        return make_constant();
    }

    // Linear model: index = slope * value + intercept
    const double slope = static_cast<double>(n - 1) / value_range;
    const double intercept = static_cast<double>(start) - slope * min_val;

    result.linear_a = slope;
    result.linear_b = intercept;

    // Measure linear error
    std::size_t linear_max_error = 0;
    double linear_total_error = 0.0;

    for (std::size_t i = start; i < end; ++i) {
        const double pred_double = std::fma(static_cast<double>(data[i]), slope, intercept);
        const double error = std::abs(pred_double - static_cast<double>(i));
        linear_max_error = std::max(linear_max_error, static_cast<std::size_t>(std::ceil(error)));
        linear_total_error += error;
    }

    const double linear_mean_error = linear_total_error / static_cast<double>(n);

    // If linear is good enough, use it
    if (linear_max_error <= MAX_ACCEPTABLE_LINEAR_ERROR) {
        result.best_model = ModelType::LINEAR;
        result.max_error = linear_max_error;
        result.mean_error = linear_mean_error;
        return result;
    }

    // Try quadratic model: index = a*value^2 + b*value + c
    // Use least squares fitting
    double sum_x = 0.0, sum_x2 = 0.0, sum_x3 = 0.0, sum_x4 = 0.0;
    double sum_y = 0.0, sum_xy = 0.0, sum_x2y = 0.0;

    for (std::size_t i = start; i < end; ++i) {
        const double x = static_cast<double>(data[i]);
        const double y = static_cast<double>(i);
        const double x2 = x * x;
        const double x3 = x2 * x;
        const double x4 = x2 * x2;

        sum_x += x;
        sum_x2 += x2;
        sum_x3 += x3;
        sum_x4 += x4;
        sum_y += y;
        sum_xy += x * y;
        sum_x2y += x2 * y;
    }

    const double n_double = static_cast<double>(n);

    // Solve normal equations (simplified for speed, may not be perfect)
    const double denom = n_double * (sum_x2 * sum_x4 - sum_x3 * sum_x3) -
                        sum_x * (sum_x * sum_x4 - sum_x2 * sum_x3) +
                        sum_x2 * (sum_x * sum_x3 - sum_x2 * sum_x2);

    if (std::abs(denom) > 1e-10) {
        const double a = (n_double * (sum_x2y * sum_x2 - sum_xy * sum_x3) -
                         sum_x * (sum_xy * sum_x2 - sum_y * sum_x3) +
                         sum_x2 * (sum_xy * sum_x - sum_y * sum_x2)) / denom;

        const double b = (n_double * (sum_x4 * sum_xy - sum_x3 * sum_x2y) -
                         sum_x * (sum_x3 * sum_xy - sum_x2 * sum_x2y) +
                         sum_x2 * (sum_x2 * sum_xy - sum_x * sum_x2y)) / denom;

        const double c = (sum_y - a * sum_x2 - b * sum_x) / n_double;

        // Measure quadratic error
        std::size_t quad_max_error = 0;
        double quad_total_error = 0.0;

        for (std::size_t i = start; i < end; ++i) {
            const double x = static_cast<double>(data[i]);
            const double pred_double = std::fma(x, std::fma(x, a, b), c);
            const double error = std::abs(pred_double - static_cast<double>(i));
            quad_max_error = std::max(quad_max_error, static_cast<std::size_t>(std::ceil(error)));
            quad_total_error += error;
        }

        const double quad_mean_error = quad_total_error / n_double;

        // Choose quadratic if it's significantly better
        if (quad_max_error < linear_max_error * QUADRATIC_IMPROVEMENT_THRESHOLD) {
            result.best_model = ModelType::QUADRATIC;
            result.quad_a = a;
            result.quad_b = b;
            result.quad_c = c;
            result.max_error = quad_max_error;
            result.mean_error = quad_mean_error;
            return result;
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

template <typename T, SegmentCount Segments = SegmentCount::LARGE, typename Compare = std::less<>>
class JazzyIndex {
    static constexpr std::size_t NumSegments = static_cast<std::size_t>(Segments);

    static_assert(detail::is_strictly_arithmetic_v<T>,
                  "JazzyIndex expects arithmetic value types.");
    static_assert(NumSegments > 0 && NumSegments <= 4096,
                  "NumSegments must be in range [1, 4096]");

public:
    JazzyIndex() = default;

    JazzyIndex(const T* first, const T* last, Compare comp = Compare{}) {
        build(first, last, comp);
    }

    void build(const T* first, const T* last, Compare comp = Compare{}) {
        base_ = first;
        size_ = static_cast<std::size_t>(last - first);
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

            // Analyze segment and choose best model
            const auto analysis = detail::analyze_segment(base_, start, end);

            seg.model_type = analysis.best_model;
            seg.max_error = static_cast<uint8_t>(std::min<std::size_t>(analysis.max_error, 255));

            switch (analysis.best_model) {
                case detail::ModelType::LINEAR:
                    seg.params.linear.slope = analysis.linear_a;
                    seg.params.linear.intercept = analysis.linear_b;
                    break;
                case detail::ModelType::QUADRATIC:
                    seg.params.quadratic.a = analysis.quad_a;
                    seg.params.quadratic.b = analysis.quad_b;
                    seg.params.quadratic.c = analysis.quad_c;
                    break;
                case detail::ModelType::CONSTANT:
                    seg.params.constant.constant_idx = start;
                    break;
                default:
                    break;
            }
        }

        // Detect if data is uniformly distributed for fast O(1) segment lookup
        detect_uniformity();
    }

    [[nodiscard]] const T* find(const T& key) const {
        if (size_ == 0) {
            return base_;
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
        std::size_t predicted = seg->predict(key);
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

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t num_segments() const noexcept { return num_segments_; }

    // Friend declaration for export functionality
    template <typename U, SegmentCount S, typename C>
    friend std::string export_index_metadata(const JazzyIndex<U, S, C>& index);

private:
    void detect_uniformity() noexcept {
        if (num_segments_ <= 1) {
            is_uniform_ = true;
            return;
        }

        // Check if segments are roughly evenly spaced in value range
        // This indicates uniform distribution
        const double total_range = static_cast<double>(max_) - static_cast<double>(min_);
        if (total_range < std::numeric_limits<double>::epsilon()) {
            is_uniform_ = true;
            return;
        }

        const double expected_spacing = total_range / static_cast<double>(num_segments_);

        // Allow 30% deviation for uniformity detection
        const double tolerance = expected_spacing * detail::UNIFORMITY_TOLERANCE;

        for (std::size_t i = 0; i < num_segments_; ++i) {
            const double segment_range = static_cast<double>(segments_[i].max_val) -
                                        static_cast<double>(segments_[i].min_val);

            if (std::abs(segment_range - expected_spacing) > tolerance) {
                is_uniform_ = false;
                return;
            }
        }

        is_uniform_ = true;

        // Precompute scale factor for O(1) arithmetic lookup
        if (is_uniform_) {
            segment_scale_ = static_cast<double>(num_segments_) / total_range;
        }
    }

    [[nodiscard]] const detail::Segment<T>* find_segment(const T& value) const noexcept {
        if (num_segments_ == 0) {
            return nullptr;
        }

        // Fast path: O(1) arithmetic lookup for uniform data
        if (is_uniform_) {
            const double offset = static_cast<double>(value) - static_cast<double>(min_);
            std::size_t seg_idx = static_cast<std::size_t>(offset * segment_scale_);

            // Clamp to valid range
            if (seg_idx >= num_segments_) {
                seg_idx = num_segments_ - 1;
            }

            // Verify we got the right segment (should always be true for uniform data)
            const auto& seg = segments_[seg_idx];
            if (!comp_(value, seg.min_val) && !comp_(seg.max_val, value)) {
                return &seg;
            }

            // Fallback to binary search if arithmetic failed (rare)
        }

        // Slow path: Binary search through segments for skewed data
        std::size_t left = 0;
        std::size_t right = num_segments_;

        while (left < right) {
            const std::size_t mid = left + (right - left) / 2;

            if (comp_(value, segments_[mid].min_val)) {
                right = mid;
            } else if (comp_(segments_[mid].max_val, value)) {
                left = mid + 1;
            } else {
                // value is within this segment
                return &segments_[mid];
            }
        }

        // Edge case: if we're past all segments, return last segment
        if (left > 0 && left >= num_segments_) {
            return &segments_[num_segments_ - 1];
        }

        return left < num_segments_ ? &segments_[left] : nullptr;
    }

    [[nodiscard]] bool equal(const T& lhs, const T& rhs) const {
        return !comp_(lhs, rhs) && !comp_(rhs, lhs);
    }

    const T* base_{nullptr};
    std::size_t size_{0};
    T min_{};
    T max_{};
    Compare comp_{};
    std::size_t num_segments_{0};
    bool is_uniform_{false};
    double segment_scale_{0.0};
    std::array<detail::Segment<T>, NumSegments> segments_{};
};

}  // namespace jazzy
