#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <functional>
#include <future>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "jazzy_index.hpp"

namespace jazzy {
namespace parallel {

// Task representing a single segment build operation
template <typename T, typename Compare, typename KeyExtractor>
struct BuildTask {
    std::size_t segment_index;
    std::size_t start_idx;
    std::size_t end_idx;
    const T* data;
    Compare comp;
    KeyExtractor key_extract;

    // Execute the segment analysis
    detail::SegmentAnalysis<T> execute() const {
        return detail::analyze_segment(data, start_idx, end_idx, comp, key_extract);
    }
};

// Helper class for parallel build operations
template <typename T, SegmentCount Segments, typename Compare = std::less<>, typename KeyExtractor = jazzy::identity>
class ParallelBuilder {
    using IndexType = JazzyIndex<T, Segments, Compare, KeyExtractor>;
    static constexpr std::size_t NumSegments = static_cast<std::size_t>(Segments);

public:
    // Prepare independent build tasks for each segment
    // Users can execute these tasks in their own threading model
    static std::vector<BuildTask<T, Compare, KeyExtractor>>
    prepare_build_tasks(IndexType& index,
                       const T* first,
                       const T* last,
                       Compare comp = Compare{},
                       KeyExtractor key_extract = KeyExtractor{}) {
        // Initialize basic index state
        index.base_ = first;
        index.size_ = static_cast<std::size_t>(last - first);
        index.key_extract_ = key_extract;
        index.comp_ = comp;

        if (index.size_ == 0) {
            return {};
        }

        index.min_ = index.base_[0];
        index.max_ = index.base_[index.size_ - 1];

        // Handle single element case immediately
        if (index.size_ == 1) {
            auto& seg = index.segments_[0];
            seg.min_val = index.min_;
            seg.max_val = index.max_;
            seg.start_idx = 0;
            seg.end_idx = 1;
            seg.model_type = detail::ModelType::CONSTANT;
            seg.max_error = 0;
            seg.params.constant.constant_idx = 0;
            index.num_segments_ = 1;
            return {};
        }

        // Determine actual number of segments
        const std::size_t actual_segments = std::min(NumSegments, index.size_);
        index.num_segments_ = actual_segments;

        // Verify data is sorted and initialize segment boundaries
        std::vector<BuildTask<T, Compare, KeyExtractor>> tasks;
        tasks.reserve(actual_segments);

        for (std::size_t i = 0; i < actual_segments; ++i) {
            const std::size_t start = (i * index.size_) / actual_segments;
            const std::size_t end = ((i + 1) * index.size_) / actual_segments;

            auto& seg = index.segments_[i];
            seg.min_val = index.base_[start];
            seg.max_val = index.base_[end - 1];
            seg.start_idx = start;
            seg.end_idx = end;

            // Verify monotonicity (must be done sequentially)
            if (i > 0 && comp(seg.min_val, index.segments_[i - 1].max_val)) {
                throw std::runtime_error(
                    "Input data is not sorted. JazzyIndex requires sorted data. "
                    "Please sort your data before building the index."
                );
            }

            // Create task for this segment
            BuildTask<T, Compare, KeyExtractor> task;
            task.segment_index = i;
            task.start_idx = start;
            task.end_idx = end;
            task.data = index.base_;
            task.comp = comp;
            task.key_extract = key_extract;

            tasks.push_back(std::move(task));
        }

        return tasks;
    }

    // Finalize the index after all segment analyses are complete
    static void finalize_build(IndexType& index,
                              const std::vector<detail::SegmentAnalysis<T>>& results) {
        if (results.size() != index.num_segments_) {
            throw std::runtime_error("Result count does not match number of segments");
        }

        // Store analysis results in segments
        for (std::size_t i = 0; i < index.num_segments_; ++i) {
            const auto& analysis = results[i];
            auto& seg = index.segments_[i];

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

            // Store model coefficients
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
                    seg.params.constant.constant_idx = seg.start_idx;
                    break;
                default:
                    break;
            }
        }

        // Always compute scale factor for O(1) segment lookup hint
        const double total_range = static_cast<double>(std::invoke(index.key_extract_, index.max_)) -
                                   static_cast<double>(std::invoke(index.key_extract_, index.min_));

        // Always compute segment scale (no uniformity detection)
        // Even for non-uniform data, this gives us a good starting point
        if (total_range >= detail::ZERO_RANGE_THRESHOLD) {
            index.segment_scale_ = static_cast<double>(index.num_segments_) / total_range;
        }
    }

    // Convenience method: parallel build using std::async
    static void build_parallel(IndexType& index,
                              const T* first,
                              const T* last,
                              Compare comp = Compare{},
                              KeyExtractor key_extract = KeyExtractor{}) {
        // Prepare tasks
        auto tasks = prepare_build_tasks(index, first, last, comp, key_extract);

        // Handle trivial cases (empty or single element)
        if (tasks.empty()) {
            return;
        }

        // Launch all tasks asynchronously using std::async
        std::vector<std::future<detail::SegmentAnalysis<T>>> futures;
        futures.reserve(tasks.size());

        for (auto& task : tasks) {
            futures.push_back(std::async(std::launch::async, [task]() {
                return task.execute();
            }));
        }

        // Collect results (preserving order)
        std::vector<detail::SegmentAnalysis<T>> results;
        results.reserve(tasks.size());

        // Track first exception if any
        std::exception_ptr first_exception;

        for (auto& future : futures) {
            try {
                results.push_back(future.get());
            } catch (...) {
                // Store first exception, continue collecting to avoid deadlock
                if (!first_exception) {
                    first_exception = std::current_exception();
                }
                // Push a dummy result to maintain vector size
                results.push_back(detail::SegmentAnalysis<T>{});
            }
        }

        // Re-throw first exception if any occurred
        if (first_exception) {
            std::rethrow_exception(first_exception);
        }

        // Finalize the index
        finalize_build(index, results);
    }
};

}  // namespace parallel

// Implement JazzyIndex parallel build methods
template <typename T, SegmentCount Segments, typename Compare, typename KeyExtractor>
inline std::vector<parallel::BuildTask<T, Compare, KeyExtractor>>
JazzyIndex<T, Segments, Compare, KeyExtractor>::prepare_build_tasks(
    const T* first, const T* last,
    Compare comp, KeyExtractor key_extract) {
    return parallel::ParallelBuilder<T, Segments, Compare, KeyExtractor>::prepare_build_tasks(
        *this, first, last, comp, key_extract);
}

template <typename T, SegmentCount Segments, typename Compare, typename KeyExtractor>
inline void JazzyIndex<T, Segments, Compare, KeyExtractor>::finalize_build(
    const std::vector<detail::SegmentAnalysis<T>>& results) {
    parallel::ParallelBuilder<T, Segments, Compare, KeyExtractor>::finalize_build(*this, results);
}

template <typename T, SegmentCount Segments, typename Compare, typename KeyExtractor>
inline void JazzyIndex<T, Segments, Compare, KeyExtractor>::build_parallel(
    const T* first, const T* last,
    Compare comp, KeyExtractor key_extract) {
    parallel::ParallelBuilder<T, Segments, Compare, KeyExtractor>::build_parallel(
        *this, first, last, comp, key_extract);
}

}  // namespace jazzy
