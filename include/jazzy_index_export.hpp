#pragma once

#include <sstream>
#include <string>
#include "jazzy_index.hpp"

namespace jazzy {

// Export JazzyIndex metadata as JSON for visualization
template <typename T, SegmentCount Segments, typename Compare, typename KeyExtractor = jazzy::identity>
std::string export_index_metadata(const JazzyIndex<T, Segments, Compare, KeyExtractor>& index) {
    std::ostringstream oss;
    oss << std::scientific;
    oss.precision(17);  // Full double precision

    const std::size_t size = index.size_;
    const std::size_t num_segments = index.num_segments_;

    oss << "{\n";
    oss << "  \"size\": " << size << ",\n";
    oss << "  \"num_segments\": " << num_segments << ",\n";
    oss << "  \"is_uniform\": " << (index.is_uniform_ ? "true" : "false") << ",\n";
    oss << "  \"segment_scale\": " << index.segment_scale_ << ",\n";
    oss << "  \"min\": " << static_cast<double>(index.min_) << ",\n";
    oss << "  \"max\": " << static_cast<double>(index.max_) << ",\n";

    // Export keys
    oss << "  \"keys\": [";
    for (std::size_t i = 0; i < size; ++i) {
        if (i > 0) oss << ", ";
        oss << static_cast<double>(std::invoke(index.key_extract_, index.base_[i]));
    }
    oss << "],\n";

    // Export segments
    oss << "  \"segments\": [\n";
    for (std::size_t i = 0; i < num_segments; ++i) {
        const auto& seg = index.segments_[i];
        if (i > 0) oss << ",\n";

        oss << "    {\n";
        oss << "      \"index\": " << i << ",\n";
        oss << "      \"start_idx\": " << seg.start_idx << ",\n";
        oss << "      \"end_idx\": " << seg.end_idx << ",\n";
        oss << "      \"min_val\": " << static_cast<double>(seg.min_val) << ",\n";
        oss << "      \"max_val\": " << static_cast<double>(seg.max_val) << ",\n";
        oss << "      \"max_error\": " << static_cast<int>(seg.max_error) << ",\n";

        // Model type
        oss << "      \"model_type\": \"";
        switch (seg.model_type) {
            case detail::ModelType::LINEAR:
                oss << "LINEAR";
                break;
            case detail::ModelType::QUADRATIC:
                oss << "QUADRATIC";
                break;
            case detail::ModelType::CUBIC:
                oss << "CUBIC";
                break;
            case detail::ModelType::CONSTANT:
                oss << "CONSTANT";
                break;
        }
        oss << "\",\n";

        // Model parameters
        oss << "      \"params\": {";
        switch (seg.model_type) {
            case detail::ModelType::LINEAR:
                oss << "\"slope\": " << seg.params.linear.slope << ", "
                    << "\"intercept\": " << seg.params.linear.intercept;
                break;
            case detail::ModelType::QUADRATIC:
                oss << "\"a\": " << seg.params.quadratic.a << ", "
                    << "\"b\": " << seg.params.quadratic.b << ", "
                    << "\"c\": " << seg.params.quadratic.c;
                break;
            case detail::ModelType::CUBIC:
                oss << "\"a\": " << seg.params.cubic.a << ", "
                    << "\"b\": " << seg.params.cubic.b << ", "
                    << "\"c\": " << seg.params.cubic.c << ", "
                    << "\"d\": " << seg.params.cubic.d;
                break;
            case detail::ModelType::CONSTANT:
                oss << "\"constant_idx\": " << seg.params.constant.constant_idx;
                break;
        }
        oss << "}\n";
        oss << "    }";
    }
    oss << "\n  ]\n";
    oss << "}\n";

    return oss.str();
}

}  // namespace jazzy
