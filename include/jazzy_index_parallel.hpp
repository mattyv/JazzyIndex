#pragma once

#include <algorithm>
#include <atomic>
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
        IndexType candidate;
        candidate.base_ = first;
        candidate.key_extract_ = key_extract;
        candidate.comp_ = comp;

        if (first == last) {
            candidate.built_ = true;
            index = std::move(candidate);
            return {};
        }
        if (first == nullptr || last == nullptr) {
            throw std::invalid_argument("JazzyIndex requires a valid contiguous range");
        }
        if (std::less<const T*>{}(last, first)) {
            throw std::invalid_argument("JazzyIndex requires a non-reversed range");
        }

        candidate.size_ = static_cast<std::size_t>(last - first);
        detail::validate_input(first, last, comp, key_extract);

        candidate.min_ = candidate.base_[0];
        candidate.max_ = candidate.base_[candidate.size_ - 1];

        // Handle single element case immediately
        if (candidate.size_ == 1) {
            candidate.segments_.resize(1);
            auto& seg = candidate.segments_[0];
            seg.min_val = candidate.min_;
            seg.max_val = candidate.max_;
            seg.start_idx = 0;
            seg.end_idx = 1;
            seg.model_type = detail::ModelType::CONSTANT;
            seg.max_error = 0;
            seg.params.constant.constant_idx = 0;
            candidate.num_segments_ = 1;
            candidate.built_ = true;
            index = std::move(candidate);
            return {};
        }

        // Determine actual number of segments
        const std::size_t actual_segments = std::min(NumSegments, candidate.size_);
        candidate.num_segments_ = actual_segments;

        // Verify data is sorted and initialize segment boundaries
        std::vector<BuildTask<T, Compare, KeyExtractor>> tasks;
        tasks.reserve(actual_segments);
        candidate.segments_.resize(actual_segments);

        for (std::size_t i = 0; i < actual_segments; ++i) {
            const std::size_t start = (i * candidate.size_) / actual_segments;
            const std::size_t end = ((i + 1) * candidate.size_) / actual_segments;

            auto& seg = candidate.segments_[i];
            seg.min_val = candidate.base_[start];
            seg.max_val = candidate.base_[end - 1];
            seg.start_idx = start;
            seg.end_idx = end;

            // Create task for this segment
            BuildTask<T, Compare, KeyExtractor> task;
            task.segment_index = i;
            task.start_idx = start;
            task.end_idx = end;
            task.data = candidate.base_;
            task.comp = comp;
            task.key_extract = key_extract;

            tasks.push_back(std::move(task));
        }

        index = std::move(candidate);
        return tasks;
    }

    // Finalize the index after all segment analyses are complete
    static void finalize_build(IndexType& index,
                              const std::vector<detail::SegmentAnalysis<T>>& results) {
        // Empty and singleton ranges are already completed by prepare_build_tasks().
        if (index.built_ && index.size_ <= 1 && results.empty()) return;
        if (results.size() != index.num_segments_) {
            throw std::runtime_error("Result count does not match number of segments");
        }

        // Store analysis results in segments
        for (std::size_t i = 0; i < index.num_segments_; ++i) {
            detail::apply_analysis(index.segments_[i], results[i]);
        }

        // Refining is serial: which segments need splitting is only known once the
        // parallel fits come back, and the split re-fits are a small tail of the work.
        detail::refine_segments(index.segments_, index.base_, index.comp_, index.key_extract_);
        index.num_segments_ = index.segments_.size();

        const double total_range = static_cast<double>(std::invoke(index.key_extract_, index.max_)) -
                                   static_cast<double>(std::invoke(index.key_extract_, index.min_));
        const bool is_uniform = index.check_uniformity(total_range);

        // Compute scale factor for O(1) segment lookup if data is uniform
        index.is_uniform_ = is_uniform;
        if (is_uniform && total_range >= detail::ZERO_RANGE_THRESHOLD) {
            index.segment_scale_ = static_cast<double>(index.num_segments_) / total_range;
        }
        index.built_ = true;
    }

    // Convenience method: parallel build using std::async
    static void build_parallel(IndexType& index,
                              const T* first,
                              const T* last,
                              Compare comp = Compare{},
                              KeyExtractor key_extract = KeyExtractor{}) {
        IndexType candidate;
        auto tasks = prepare_build_tasks(candidate, first, last, comp, key_extract);

        // Handle trivial cases (empty or single element)
        if (tasks.empty()) {
            index = std::move(candidate);
            return;
        }

        std::vector<detail::SegmentAnalysis<T>> results(tasks.size());
        std::atomic<std::size_t> next_task{0};
        const std::size_t hardware_threads =
            std::max(1u, std::thread::hardware_concurrency());
        const std::size_t worker_count = std::min(tasks.size(), hardware_threads);
        std::vector<std::future<void>> workers;
        workers.reserve(worker_count);
        for (std::size_t worker = 0; worker < worker_count; ++worker) {
            workers.push_back(std::async(std::launch::async, [&] {
                for (std::size_t i = next_task.fetch_add(1);
                     i < tasks.size();
                     i = next_task.fetch_add(1)) {
                    results[i] = tasks[i].execute();
                }
            }));
        }

        std::exception_ptr first_exception;
        for (auto& worker : workers) {
            try {
                worker.get();
            } catch (...) {
                if (!first_exception) {
                    first_exception = std::current_exception();
                }
            }
        }

        if (first_exception) {
            std::rethrow_exception(first_exception);
        }

        finalize_build(candidate, results);
        index = std::move(candidate);
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
