// Tests for segment finder model - the learned model that predicts which segment contains a value
// This replaces the old uniformity check with a more general adaptive approach

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

// Helper to extract segment_finder model info from JSON
std::string extract_segment_finder_model(const std::string& json) {
    std::regex model_regex(R"xxx("segment_finder":\s*\{[^}]*"model_type":\s*"(\w+)")xxx");
    std::smatch match;
    if (std::regex_search(json, match, model_regex)) {
        return match[1];
    }
    return "";
}

// Helper to extract segment_finder max_error from JSON
int extract_segment_finder_max_error(const std::string& json) {
    std::regex error_regex(R"xxx("segment_finder":\s*\{[^}]*"max_error":\s*(\d+))xxx");
    std::smatch match;
    if (std::regex_search(json, match, error_regex)) {
        return std::stoi(match[1]);
    }
    return -1;
}

}  // namespace

// Test: Uniform data should have LINEAR segment finder with low error
TEST(SegmentFinderTest, UniformDataUsesLinearModel) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);  // 0, 1, 2, ..., 999

    jazzy::JazzyIndex<int, jazzy::SegmentCount::LARGE> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    std::string model_type = extract_segment_finder_model(json);
    int max_error = extract_segment_finder_max_error(json);

    EXPECT_EQ(model_type, "LINEAR") << "Uniform data should use LINEAR segment finder";
    EXPECT_LE(max_error, 2) << "Uniform data should have very low prediction error";
}

// Test: Exponential data segment finder should handle curvature
TEST(SegmentFinderTest, ExponentialDataSegmentFinder) {
    std::vector<int> data;
    for (int i = 0; i < 1000; ++i) {
        data.push_back(static_cast<int>(std::exp(i * 0.01)));
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<256>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    std::string model_type = extract_segment_finder_model(json);
    int max_error = extract_segment_finder_max_error(json);

    // Model type could be LINEAR (current implementation)
    // Error should be reasonable even if higher than uniform
    EXPECT_FALSE(model_type.empty()) << "Should have a segment finder model";
    EXPECT_GE(max_error, 0) << "Max error should be non-negative";
}

// Test: Small segment counts should have low max_error
TEST(SegmentFinderTest, FewSegmentsLowError) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::to_segment_count<8>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    int max_error = extract_segment_finder_max_error(json);

    // With only 8 segments, predictions should be very accurate
    EXPECT_LE(max_error, 1) << "Few segments on uniform data should have near-perfect predictions";
}

// Test: Many segments on uniform data should still have low error
TEST(SegmentFinderTest, ManySegmentsUniformStillLowError) {
    std::vector<int> data(10000);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::to_segment_count<1024>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    std::string model_type = extract_segment_finder_model(json);
    int max_error = extract_segment_finder_max_error(json);

    EXPECT_EQ(model_type, "LINEAR") << "Uniform data uses LINEAR regardless of segment count";
    // Even with 1024 segments, uniform data should have low error
    EXPECT_LE(max_error, 3) << "Many segments on uniform data should still have low error";
}

// Test: Constant data (all same value) should have zero error
TEST(SegmentFinderTest, ConstantDataZeroError) {
    std::vector<int> data(1000, 42);

    jazzy::JazzyIndex<int, jazzy::SegmentCount::LARGE> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    int max_error = extract_segment_finder_max_error(json);

    // All segments have same boundaries, so prediction should be perfect
    EXPECT_EQ(max_error, 0) << "Constant data should have zero segment finder error";
}

// Test: Quadratic growth data
TEST(SegmentFinderTest, QuadraticGrowthData) {
    std::vector<int> data;
    for (int i = 0; i < 1000; ++i) {
        data.push_back(i * i);  // 0, 1, 4, 9, 16, 25, ...
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<128>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    std::string model_type = extract_segment_finder_model(json);
    int max_error = extract_segment_finder_max_error(json);

    // LINEAR model may have higher error for quadratic data
    EXPECT_FALSE(model_type.empty());
    // Error should be bounded but potentially higher than uniform
    EXPECT_GE(max_error, 0);
}

// Test: Sparse linear data
TEST(SegmentFinderTest, SparseLinearData) {
    std::vector<int> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(i * 100);  // 0, 100, 200, ...
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<64>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    std::string model_type = extract_segment_finder_model(json);
    int max_error = extract_segment_finder_max_error(json);

    EXPECT_EQ(model_type, "LINEAR") << "Sparse but linear data should use LINEAR";
    EXPECT_LE(max_error, 2) << "Linear spacing should have low error";
}

// Test: Single element index
TEST(SegmentFinderTest, SingleElementIndex) {
    std::vector<int> data{42};

    jazzy::JazzyIndex<int, jazzy::SegmentCount::LARGE> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    std::string model_type = extract_segment_finder_model(json);
    int max_error = extract_segment_finder_max_error(json);

    // Single element should have zero error (only one segment)
    EXPECT_EQ(max_error, 0) << "Single element should have zero error";
}

// Test: Two segments
TEST(SegmentFinderTest, TwoSegments) {
    std::vector<int> data(100);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::to_segment_count<2>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    std::string model_type = extract_segment_finder_model(json);
    int max_error = extract_segment_finder_max_error(json);

    EXPECT_EQ(model_type, "LINEAR");
    EXPECT_LE(max_error, 1) << "Two segments on uniform data should have perfect or near-perfect prediction";
}

// Test: Verify segment finder works correctly during lookups
TEST(SegmentFinderTest, CorrectLookupResults) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::SegmentCount::LARGE> index;
    index.build(data.data(), data.data() + data.size());

    // Verify lookups work correctly across the entire range
    for (int i = 0; i < 1000; i += 50) {
        const int* result = index.find(i);
        ASSERT_NE(result, data.data() + data.size()) << "Value " << i << " should be found";
        EXPECT_EQ(*result, i) << "Found value should match query";
    }
}

// Test: Segment finder with skewed distribution
TEST(SegmentFinderTest, SkewedDistribution) {
    std::vector<int> data;

    // Dense at beginning
    for (int i = 0; i < 500; ++i) {
        data.push_back(i);
    }

    // Sparse at end
    for (int i = 0; i < 500; ++i) {
        data.push_back(1000 + i * 10);
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<128>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    std::string model_type = extract_segment_finder_model(json);
    int max_error = extract_segment_finder_max_error(json);

    // Skewed data may have higher error
    EXPECT_FALSE(model_type.empty());
    EXPECT_GE(max_error, 0);

    // But lookups should still work correctly
    EXPECT_NE(index.find(250), data.data() + data.size());
    EXPECT_NE(index.find(2000), data.data() + data.size());
}

// Test: Segment finder handles negative numbers
TEST(SegmentFinderTest, NegativeNumbers) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), -500);  // -500 to 499

    jazzy::JazzyIndex<int, jazzy::SegmentCount::LARGE> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    std::string model_type = extract_segment_finder_model(json);
    int max_error = extract_segment_finder_max_error(json);

    EXPECT_EQ(model_type, "LINEAR");
    EXPECT_LE(max_error, 2) << "Negative numbers shouldn't affect prediction quality";

    // Verify lookups work
    EXPECT_NE(index.find(-500), data.data() + data.size());
    EXPECT_NE(index.find(0), data.data() + data.size());
    EXPECT_NE(index.find(499), data.data() + data.size());
}

// Test: Floating point data
TEST(SegmentFinderTest, FloatingPointData) {
    std::vector<double> data(1000);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<double>(i) * 0.1;
    }

    jazzy::JazzyIndex<double, jazzy::SegmentCount::LARGE> index;
    index.build(data.data(), data.data() + data.size());

    std::string json = jazzy::export_index_metadata(index);
    std::string model_type = extract_segment_finder_model(json);

    EXPECT_EQ(model_type, "LINEAR") << "Uniform floating point data should use LINEAR";

    // Verify lookups work
    EXPECT_NE(index.find(0.0), data.data() + data.size());
    EXPECT_NE(index.find(50.0), data.data() + data.size());
    EXPECT_NE(index.find(99.9), data.data() + data.size());
}
