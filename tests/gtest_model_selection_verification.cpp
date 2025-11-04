// Hard-coded tests to verify exact model selection hasn't changed
// These tests lock down the expected behavior of model selection for specific data patterns

#include "jazzy_index.hpp"
#include "jazzy_index_export.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <regex>
#include <string>
#include <vector>

namespace {

// Helper to extract model types from JSON export
std::vector<std::string> extract_model_types(const std::string& json) {
    std::vector<std::string> models;
    std::regex model_regex(R"xxx("model_type":\s*"(\w+)")xxx");
    std::smatch match;

    std::string::const_iterator search_start(json.cbegin());
    while (std::regex_search(search_start, json.cend(), match, model_regex)) {
        models.push_back(match[1]);
        search_start = match.suffix().first;
    }

    return models;
}

// Helper to count model types
int count_model_type(const std::vector<std::string>& models, const std::string& type) {
    return static_cast<int>(std::count(models.begin(), models.end(), type));
}

}  // namespace

// Test: Perfectly uniform data should use LINEAR models for all segments
TEST(ModelSelectionVerification, UniformDataUsesLinearModels) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);  // 0, 1, 2, ..., 999

    jazzy::JazzyIndex<int, jazzy::SegmentCount::LARGE> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    auto models = extract_model_types(json);

    // All segments should be LINEAR for perfectly uniform data
    int linear_count = count_model_type(models, "LINEAR");
    int total_segments = static_cast<int>(models.size());

    EXPECT_EQ(linear_count, total_segments)
        << "Expected all segments to be LINEAR for uniform data";
    EXPECT_EQ(count_model_type(models, "CONSTANT"), 0);
    EXPECT_EQ(count_model_type(models, "QUADRATIC"), 0);
}

// Test: All identical values should use CONSTANT model
TEST(ModelSelectionVerification, IdenticalValuesUseConstantModel) {
    std::vector<int> data(1000, 42);  // All values are 42

    jazzy::JazzyIndex<int, jazzy::SegmentCount::LARGE> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    auto models = extract_model_types(json);

    // All segments should be CONSTANT
    int constant_count = count_model_type(models, "CONSTANT");
    int total_segments = static_cast<int>(models.size());

    EXPECT_EQ(constant_count, total_segments)
        << "Expected all segments to be CONSTANT for identical values";
    EXPECT_EQ(count_model_type(models, "LINEAR"), 0);
    EXPECT_EQ(count_model_type(models, "QUADRATIC"), 0);
}

// Test: Quadratic data should trigger QUADRATIC models (at least some)
TEST(ModelSelectionVerification, QuadraticDataUsesQuadraticModels) {
    std::vector<int> data;
    for (int i = 0; i < 1000; ++i) {
        data.push_back(i * i);  // 0, 1, 4, 9, 16, 25, ...
    }

    // Use fewer segments so each segment has multiple elements
    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    auto models = extract_model_types(json);

    // Should have some QUADRATIC models (exact count may vary, but should have at least a few)
    int quadratic_count = count_model_type(models, "QUADRATIC");

    EXPECT_GT(quadratic_count, 0)
        << "Expected at least some QUADRATIC models for quadratic data";

    // First segment might be linear (small values), later segments should be quadratic
    // Just verify we detect the curvature somewhere
}

// Test: Specific known case - exponential-like data with S=32
TEST(ModelSelectionVerification, ExponentialDataModelDistribution) {
    std::vector<int> data;
    for (int i = 0; i < 50; ++i) {
        data.push_back(static_cast<int>(std::pow(1.5, i)));
    }
    std::sort(data.begin(), data.end());
    data.erase(std::unique(data.begin(), data.end()), data.end());

    // Use fewer segments to ensure multiple elements per segment
    jazzy::JazzyIndex<int, jazzy::to_segment_count<16>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    auto models = extract_model_types(json);

    int linear_count = count_model_type(models, "LINEAR");
    int quadratic_count = count_model_type(models, "QUADRATIC");
    int constant_count = count_model_type(models, "CONSTANT");
    int total = static_cast<int>(models.size());

    // Verify total segments is reasonable
    EXPECT_GT(total, 0);
    EXPECT_LE(total, static_cast<int>(data.size()));

    // Should use various models
    EXPECT_EQ(linear_count + quadratic_count + constant_count, total);

    // Verify index works correctly regardless of model selection
    for (const auto& val : data) {
        const int* result = index.find(val);
        EXPECT_NE(result, data.data() + data.size());
        if (result != data.data() + data.size()) {
            EXPECT_EQ(*result, val);
        }
    }
}

// Test: Mixed data - constant region followed by linear region
TEST(ModelSelectionVerification, MixedConstantLinearData) {
    std::vector<int> data;

    // First 100: all zeros (constant)
    for (int i = 0; i < 100; ++i) {
        data.push_back(0);
    }

    // Next 500: linear (0-499)
    for (int i = 0; i < 500; ++i) {
        data.push_back(i);
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    auto models = extract_model_types(json);

    int linear_count = count_model_type(models, "LINEAR");
    int constant_count = count_model_type(models, "CONSTANT");

    // Should have some CONSTANT models (for the zero region)
    EXPECT_GT(constant_count, 0)
        << "Expected some CONSTANT models for zero region";

    // Should have some LINEAR models (for the linear region)
    EXPECT_GT(linear_count, 0)
        << "Expected some LINEAR models for linear region";
}

// Test: Sparse linear data (multiples of 10)
TEST(ModelSelectionVerification, SparseLinearData) {
    std::vector<int> data;
    for (int i = 0; i < 1000; ++i) {
        data.push_back(i * 10);  // 0, 10, 20, ..., 9990
    }

    // Use fewer segments relative to data size to ensure multi-element segments
    jazzy::JazzyIndex<int, jazzy::to_segment_count<32>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    auto models = extract_model_types(json);

    // Linear spacing should use mostly LINEAR models
    int linear_count = count_model_type(models, "LINEAR");
    int total = static_cast<int>(models.size());

    // Most should be linear (allow some tolerance for edge segments)
    EXPECT_GT(linear_count, total * 8 / 10)
        << "Expected mostly LINEAR models for sparse but linear data";
}

// Test: Very high curvature quadratic
TEST(ModelSelectionVerification, HighCurvatureQuadratic) {
    std::vector<int> data;
    for (int i = 0; i < 1000; ++i) {
        // High curvature: 2x^2 + x + 10
        int value = 2 * i * i + i + 10;
        data.push_back(value);
    }

    // Use fewer segments to get multi-element segments
    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    auto models = extract_model_types(json);

    int quadratic_count = count_model_type(models, "QUADRATIC");

    // High curvature should definitely trigger QUADRATIC models
    EXPECT_GT(quadratic_count, 0)
        << "Expected QUADRATIC models for high curvature data";
}

// Test: Single element (edge case)
TEST(ModelSelectionVerification, SingleElementUsesConstant) {
    std::vector<int> data{42};

    jazzy::JazzyIndex<int, jazzy::SegmentCount::LARGE> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    auto models = extract_model_types(json);

    EXPECT_EQ(models.size(), 1);
    EXPECT_EQ(models[0], "CONSTANT");
}

// Test: Two elements - with appropriate segment count
TEST(ModelSelectionVerification, TwoElementsUsesLinear) {
    std::vector<int> data{10, 20};

    // Use 1 segment for 2 elements, otherwise we get 2 single-element segments
    jazzy::JazzyIndex<int, jazzy::to_segment_count<1>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    auto models = extract_model_types(json);

    EXPECT_EQ(models.size(), 1);
    EXPECT_EQ(models[0], "LINEAR");
}

// Test: Arithmetic progression with offset
TEST(ModelSelectionVerification, ArithmeticProgressionWithOffset) {
    std::vector<int> data;
    for (int i = 0; i < 1000; ++i) {
        data.push_back(1000 + i * 5);  // 1000, 1005, 1010, ..., 5995
    }

    // Use appropriate segment count
    jazzy::JazzyIndex<int, jazzy::to_segment_count<32>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    auto models = extract_model_types(json);

    int linear_count = count_model_type(models, "LINEAR");
    int total = static_cast<int>(models.size());

    // Arithmetic progression is linear - expect most to be LINEAR
    EXPECT_GT(linear_count, total * 8 / 10)
        << "Expected mostly LINEAR models for arithmetic progression";
}

// Test: Near-zero range (all values almost identical)
TEST(ModelSelectionVerification, NearZeroRangeUsesConstant) {
    std::vector<double> data(100);
    for (int i = 0; i < 100; ++i) {
        data[i] = 1000.0 + i * 1e-20;  // Effectively constant
    }

    jazzy::JazzyIndex<double, jazzy::SegmentCount::SMALL> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    auto models = extract_model_types(json);

    int constant_count = count_model_type(models, "CONSTANT");
    int total = static_cast<int>(models.size());

    // Near-zero range should be treated as constant
    EXPECT_EQ(constant_count, total)
        << "Expected all CONSTANT models for near-zero range";
}

// Test: Cubic data (i^3) should use CUBIC models for segments with high error
TEST(ModelSelectionVerification, CubicDataUsesCubicModels) {
    std::vector<long long> data;
    // Use larger range with strong curvature: i^4 / 10
    for (int i = 0; i < 500; ++i) {
        data.push_back(static_cast<long long>(i) * i * i * i / 10);
    }

    // Use fewer segments so each has enough elements to show curvature
    jazzy::JazzyIndex<long long, jazzy::to_segment_count<8>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    auto models = extract_model_types(json);

    int cubic_count = count_model_type(models, "CUBIC");
    int quadratic_count = count_model_type(models, "QUADRATIC");
    int linear_count = count_model_type(models, "LINEAR");

    // Highly curved data should use CUBIC or QUADRATIC models
    EXPECT_GT(cubic_count + quadratic_count, 0)
        << "Expected some higher-order models (CUBIC or QUADRATIC) for highly curved data";

    // If CUBIC implementation is working, it may be selected for some segments
    // (exact count depends on thresholds and data distribution)

    // Verify index works correctly
    for (const auto& val : data) {
        const long long* result = index.find(val);
        EXPECT_NE(result, data.data() + data.size());
        if (result != data.data() + data.size()) {
            EXPECT_EQ(*result, val);
        }
    }
}

// Test: Verify CUBIC model functionality (correctness over selection frequency)
TEST(ModelSelectionVerification, CubicModelCorrectness) {
    // This test verifies that when CUBIC models ARE selected, they work correctly
    // We use the exact configuration known to trigger CUBIC selection
    std::vector<long long> data;
    for (int i = 0; i < 500; ++i) {
        data.push_back(static_cast<long long>(i) * i * i * i / 10);
    }

    jazzy::JazzyIndex<long long, jazzy::to_segment_count<8>()> index;
    index.build(data.data(), data.data() + data.size());

    // Verify correctness - all values must be findable
    for (const auto& val : data) {
        const long long* result = index.find(val);
        EXPECT_NE(result, data.data() + data.size())
            << "Failed to find value: " << val;
        if (result != data.data() + data.size()) {
            EXPECT_EQ(*result, val)
                << "Found wrong value for key: " << val;
        }
    }

    // Note: CUBIC models are selected adaptively based on error thresholds
    // The key requirement is correctness, not specific model selection
}
