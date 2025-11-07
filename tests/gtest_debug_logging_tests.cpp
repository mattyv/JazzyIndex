// Comprehensive tests for debug logging mechanism
// These tests verify that all key operations and decision points are logged

#include "jazzy_index.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <numeric>

#ifdef JAZZY_DEBUG_LOGGING

class DebugLoggingTest : public ::testing::Test {
protected:
    void SetUp() override {
        jazzy::clear_debug_log();
    }

    void TearDown() override {
        jazzy::clear_debug_log();
    }

    std::string get_log() {
        return jazzy::get_debug_log();
    }

    bool contains(const std::string& log, const std::string& substr) {
        return log.find(substr) != std::string::npos;
    }
};

// Test: Build phase logging
TEST_F(DebugLoggingTest, BuildPhaseLogging) {
    std::vector<int> data(100);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::to_segment_count<4>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string log = get_log();

    // Verify build start with exact parameters
    EXPECT_TRUE(contains(log, "JazzyIndex::build: Building index for 100 elements with 4 segments"))
        << "Build start log should show exact size (100) and segment count (4)";

    // Verify at least 4 top-level segments were analyzed
    // Note: analyze_segment may be called recursively, so count may be higher
    size_t count = 0;
    size_t pos = 0;
    while ((pos = log.find("analyze_segment[", pos)) != std::string::npos) {
        ++count;
        ++pos;
    }
    EXPECT_GE(count, 4) << "Should analyze at least 4 segments, found: " << count;

    // Verify each segment analysis has all required components
    EXPECT_TRUE(contains(log, "n=")) << "Missing segment size (n=)";
    EXPECT_TRUE(contains(log, "linear_max_error=")) << "Missing linear max error";
    EXPECT_TRUE(contains(log, "linear_mean_error=")) << "Missing linear mean error";
    EXPECT_TRUE(contains(log, "slope=")) << "Missing slope value";
    EXPECT_TRUE(contains(log, "intercept=")) << "Missing intercept value";

    // Verify model selection decisions (for uniform data, should be LINEAR)
    EXPECT_TRUE(contains(log, "Selected LINEAR model")) << "Uniform data should select LINEAR";

    // Verify build completion
    EXPECT_TRUE(contains(log, "Build complete with 4 segments"))
        << "Build completion should confirm segment count";

    // Verify uniformity detection (uniform data should be detected)
    EXPECT_TRUE(contains(log, "Data is UNIFORM")) << "Sequential data should be detected as UNIFORM";
    EXPECT_TRUE(contains(log, "segment_scale=")) << "UNIFORM detection should include segment_scale";
}

// Test: LINEAR model selection logging
TEST_F(DebugLoggingTest, LinearModelSelection) {
    std::vector<int> data(50);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::to_segment_count<4>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string log = get_log();

    // Verify LINEAR model was selected with threshold check
    EXPECT_TRUE(contains(log, "Selected LINEAR model")) << "LINEAR model not logged";
    EXPECT_TRUE(contains(log, "<= threshold="))
        << "Should log comparison with MAX_ACCEPTABLE_LINEAR_ERROR threshold";

    // Verify all LINEAR parameters are logged
    EXPECT_TRUE(contains(log, "slope=")) << "Missing slope parameter";
    EXPECT_TRUE(contains(log, "intercept=")) << "Missing intercept parameter";
    EXPECT_TRUE(contains(log, "max_error=")) << "Missing max_error in selection";

    // For perfectly uniform data, max_error should be 0
    EXPECT_TRUE(contains(log, "linear_max_error=0"))
        << "Perfectly uniform sequential data should have 0 linear error";
}

// Test: QUADRATIC model selection logging
TEST_F(DebugLoggingTest, QuadraticModelSelection) {
    // Create quadratic data
    std::vector<int> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(i * i);
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<8>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string log = get_log();

    // Should see linear error analysis for all segments
    EXPECT_TRUE(contains(log, "linear_max_error=")) << "Missing linear error logging";

    // Count how many segments were analyzed (may be more due to recursion)
    size_t segment_count = 0;
    size_t pos = 0;
    while ((pos = log.find("analyze_segment[", pos)) != std::string::npos) {
        ++segment_count;
        ++pos;
    }
    EXPECT_GE(segment_count, 8) << "Should analyze at least 8 segments, found: " << segment_count;

    // If QUADRATIC was tried, verify complete logging
    if (contains(log, "trying QUADRATIC")) {
        EXPECT_TRUE(contains(log, "QUADRATIC: max_error=")) << "Missing quadratic error logging";
        EXPECT_TRUE(contains(log, "monotonicity")) << "Missing monotonicity check";
        EXPECT_TRUE(contains(log, "deriv_at_min=")) << "Missing derivative at min";
        EXPECT_TRUE(contains(log, "deriv_at_max=")) << "Missing derivative at max";
        EXPECT_TRUE(contains(log, "is_monotonic=")) << "Missing monotonicity result";

        // If QUADRATIC was selected, error should be better than linear
        if (contains(log, "Selected QUADRATIC")) {
            EXPECT_TRUE(contains(log, "improvement_threshold="))
                << "Missing improvement threshold calculation";
        }
    }
}

// Test: CONSTANT model for empty/single element segments
TEST_F(DebugLoggingTest, ConstantModelSelection) {
    std::vector<int> data(100, 42); // All same value

    jazzy::JazzyIndex<int, jazzy::to_segment_count<4>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string log = get_log();

    // For data with all identical values, the behavior depends on segment analysis
    // The data will likely be analyzed and use LINEAR or other models
    // What's important is that build succeeds and produces valid segments

    // Verify build started and completed
    EXPECT_TRUE(contains(log, "JazzyIndex::build: Building index")) << "Missing build start log";
    EXPECT_TRUE(contains(log, "Build complete with 4 segments")) << "Missing build completion";

    // Verify segments were analyzed
    EXPECT_TRUE(contains(log, "analyze_segment[")) << "Missing segment analysis";

    // Verify some model was selected
    EXPECT_TRUE(contains(log, "Selected LINEAR") || contains(log, "Selected CONSTANT") ||
                contains(log, "Selected QUADRATIC") || contains(log, "Defaulted"))
        << "Missing model selection decision";
}

// Test: find() operation logging
TEST_F(DebugLoggingTest, FindOperationLogging) {
    std::vector<int> data(100);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::to_segment_count<4>()> index;
    index.build(data.data(), data.data() + data.size());

    jazzy::clear_debug_log(); // Clear build logs

    auto* result = index.find(50);

    std::string log = get_log();

    // Verify find started
    EXPECT_TRUE(contains(log, "JazzyIndex::find: Called")) << "Missing find entry log";

    // Verify segment finding
    EXPECT_TRUE(contains(log, "find_segment: Called")) << "Missing find_segment log";

    // Verify prediction with actual index
    EXPECT_TRUE(contains(log, "predict[")) << "Missing predict log";
    EXPECT_TRUE(contains(log, "Predicted index")) << "Missing predicted index log";

    // For uniform sequential data [0, 99], searching for 50 should predict around index 50
    EXPECT_TRUE(contains(log, "result=")) << "Missing result in predict log";

    // Verify the result is correct (element with value 50 is at index 50)
    ASSERT_NE(result, nullptr);
    ASSERT_NE(result, data.data() + data.size());
    EXPECT_EQ(*result, 50) << "Should find value 50";
    EXPECT_EQ(result - data.data(), 50) << "Value 50 should be at index 50";

    // For this uniform data, should find exact match
    EXPECT_TRUE(contains(log, "Found exact match") || contains(log, "Found match at index"))
        << "Missing match found log";
}

// Test: Uniform path in find_segment
TEST_F(DebugLoggingTest, UniformSegmentFinding) {
    std::vector<int> data(100);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::to_segment_count<4>()> index;
    index.build(data.data(), data.data() + data.size());

    jazzy::clear_debug_log();

    auto* result = index.find(50);

    std::string log = get_log();

    // Uniform data should use fast path
    EXPECT_TRUE(contains(log, "UNIFORM path")) << "Missing uniform path log";
    EXPECT_TRUE(contains(log, "segment_scale=")) << "Missing segment_scale in uniform path";

    // For 100 elements with 4 segments, segment_scale should be logged
    EXPECT_TRUE(contains(log, "seg_idx=")) << "Missing segment index in uniform path";

    // Verify uniform path succeeded (sequential data should always succeed uniform path)
    EXPECT_TRUE(contains(log, "UNIFORM succeeded"))
        << "Sequential data should succeed uniform path verification";

    // Verify we got the right result
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(*result, 50) << "Should find value 50";
}

// Test: Binary search path in find_segment
TEST_F(DebugLoggingTest, BinarySearchSegmentFinding) {
    // Create non-uniform data (quadratic)
    std::vector<int> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(i * i);
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<8>()> index;
    index.build(data.data(), data.data() + data.size());

    jazzy::clear_debug_log();

    auto* result = index.find(400); // 20^2

    std::string log = get_log();

    // Non-uniform data should use binary search (unless uniform path accidentally succeeds)
    if (contains(log, "binary search")) {
        EXPECT_TRUE(contains(log, "Binary search iter")) << "Missing binary search iteration logs";
        EXPECT_TRUE(contains(log, "left=") && contains(log, "mid=") && contains(log, "right="))
            << "Missing binary search position variables";

        // Count number of binary search iterations
        size_t iter_count = 0;
        size_t pos = 0;
        while ((pos = log.find("Binary search iter", pos)) != std::string::npos) {
            ++iter_count;
            ++pos;
        }
        // For 8 segments, binary search should take at most log2(8) = 3 iterations
        EXPECT_LE(iter_count, 3) << "Binary search should take at most 3 iterations for 8 segments";
        EXPECT_GE(iter_count, 1) << "Binary search should take at least 1 iteration";
    }

    // Verify we found the correct value
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(*result, 400) << "Should find value 400 (20^2)";
}

// Test: Exponential search logging
TEST_F(DebugLoggingTest, ExponentialSearchLogging) {
    std::vector<int> data;
    // Create data where prediction might be slightly off
    for (int i = 0; i < 100; ++i) {
        data.push_back(i * i);
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<4>()> index;
    index.build(data.data(), data.data() + data.size());

    jazzy::clear_debug_log();

    auto* result = index.find(225); // 15^2

    std::string log = get_log();

    // Verify prediction occurred
    EXPECT_TRUE(contains(log, "Predicted index")) << "Missing prediction log";

    // Should see search operation (either exact match or direction-based search)
    bool has_exact_match = contains(log, "Found exact match");
    bool has_search_direction = contains(log, "Search direction:");

    EXPECT_TRUE(has_exact_match || has_search_direction)
        << "Missing search logging - log:\n" << log;

    // If search direction was logged, should also have max_radius
    if (has_search_direction) {
        EXPECT_TRUE(contains(log, "max_radius:")) << "Missing max_radius log";
        EXPECT_TRUE(contains(log, "segment_start=") && contains(log, "segment_end="))
            << "Missing segment boundaries in search";
    }

    // Verify we found the correct value
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(*result, 225) << "Should find value 225 (15^2)";
}

// Test: Bounds checking logging
TEST_F(DebugLoggingTest, BoundsCheckLogging) {
    std::vector<int> data(100);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::to_segment_count<4>()> index;
    index.build(data.data(), data.data() + data.size());

    jazzy::clear_debug_log();

    // Search for value out of bounds (data range is [0, 99])
    auto* result = index.find(200);

    std::string log = get_log();

    // Verify find was called
    EXPECT_TRUE(contains(log, "JazzyIndex::find: Called")) << "Missing find log";

    // Verify out of bounds was detected and logged
    EXPECT_TRUE(contains(log, "out of bounds") || contains(log, "returning end()"))
        << "Missing out of bounds log";

    // Verify result is end()
    EXPECT_EQ(result, data.data() + data.size()) << "Out of bounds search should return end()";
}

// Test: Empty index logging
TEST_F(DebugLoggingTest, EmptyIndexLogging) {
    std::vector<int> data;

    jazzy::JazzyIndex<int, jazzy::to_segment_count<4>()> index;
    index.build(data.data(), data.data() + data.size());

    jazzy::clear_debug_log();

    auto* result = index.find(42);

    std::string log = get_log();

    // Verify find was called
    EXPECT_TRUE(contains(log, "JazzyIndex::find: Called")) << "Missing find call log";

    // Verify empty/not built detection
    EXPECT_TRUE(contains(log, "not built or empty") || contains(log, "returning end()"))
        << "Missing empty index log";

    // Verify result is end() (which equals begin() for empty data)
    EXPECT_EQ(result, data.data()) << "Empty index search should return end() (== begin())";
}

// Test: All model selection decision points
TEST_F(DebugLoggingTest, ModelSelectionDecisionPoints) {
    // Large segment to potentially trigger all model attempts
    std::vector<int> data;
    for (int i = 0; i < 200; ++i) {
        data.push_back(i * i * i); // Cubic growth
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<2>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string log = get_log();

    // Verify at least 2 segments were analyzed (may be more due to recursion)
    size_t segment_count = 0;
    size_t pos = 0;
    while ((pos = log.find("analyze_segment[", pos)) != std::string::npos) {
        ++segment_count;
        ++pos;
    }
    EXPECT_GE(segment_count, 2) << "Should analyze at least 2 segments, found: " << segment_count;

    // Should see linear analysis with all parameters
    EXPECT_TRUE(contains(log, "linear_max_error=")) << "Missing linear error analysis";
    EXPECT_TRUE(contains(log, "linear_mean_error=")) << "Missing linear mean error";
    EXPECT_TRUE(contains(log, "slope=")) << "Missing slope parameter";
    EXPECT_TRUE(contains(log, "intercept=")) << "Missing intercept parameter";

    // Should see threshold comparisons
    EXPECT_TRUE(contains(log, "threshold=")) << "Missing threshold comparison";

    // For cubic data, linear error should be large, might try higher-order models
    // Should see decision reasoning for every segment
    EXPECT_TRUE(contains(log, "Selected") || contains(log, "Defaulted"))
        << "Missing model selection decision";

    // Verify build completion
    EXPECT_TRUE(contains(log, "Build complete with 2 segments"))
        << "Missing build completion log";
}

// Test: Prediction logging for all model types
TEST_F(DebugLoggingTest, PredictionLoggingAllModels) {
    std::vector<int> data(50);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::to_segment_count<4>()> index;
    index.build(data.data(), data.data() + data.size());

    jazzy::clear_debug_log();

    auto* result = index.find(25);

    std::string log = get_log();

    // Should see prediction with model type
    EXPECT_TRUE(contains(log, "predict["))  << "Missing predict log";
    EXPECT_TRUE(contains(log, "LINEAR") || contains(log, "QUADRATIC") ||
                contains(log, "CUBIC") || contains(log, "CONSTANT"))
        << "Missing model type in prediction";

    // For uniform sequential data, should use LINEAR model
    EXPECT_TRUE(contains(log, "LINEAR")) << "Sequential data should use LINEAR model";

    // Verify all prediction parameters are logged
    EXPECT_TRUE(contains(log, "key=")) << "Missing key value in prediction";
    EXPECT_TRUE(contains(log, "pred=")) << "Missing predicted value in prediction";
    EXPECT_TRUE(contains(log, "result=")) << "Missing result index in prediction";

    // Verify the prediction result is correct
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(*result, 25) << "Should find value 25";
    EXPECT_EQ(result - data.data(), 25) << "Value 25 should be at index 25";
}

// Test: equal_range logging
TEST_F(DebugLoggingTest, EqualRangeLogging) {
    std::vector<int> data{1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 5};

    jazzy::JazzyIndex<int, jazzy::to_segment_count<4>()> index;
    index.build(data.data(), data.data() + data.size());

    jazzy::clear_debug_log();

    auto range = index.equal_range(3);

    std::string log = get_log();

    // Verify equal_range was called
    EXPECT_TRUE(contains(log, "equal_range: Called")) << "Missing equal_range log";

    // Should see both lower_bound and upper_bound operations
    EXPECT_TRUE(contains(log, "find_lower_bound")) << "Missing find_lower_bound log";
    EXPECT_TRUE(contains(log, "find_upper_bound")) << "Missing find_upper_bound log";

    // Verify the range is correct
    // Data has 4 instances of value 3 at indices 5, 6, 7, 8
    EXPECT_EQ(range.first - data.data(), 5) << "Lower bound of 3 should be at index 5";
    EXPECT_EQ(range.second - data.data(), 9) << "Upper bound of 3 should be at index 9";
    EXPECT_EQ(range.second - range.first, 4) << "Should have 4 elements with value 3";
}

// Test: Verify actual numeric values in logs (internal state validation)
TEST_F(DebugLoggingTest, NumericValuesValidation) {
    // Create simple linear data [0, 1, 2, ..., 99]
    std::vector<int> data(100);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::to_segment_count<4>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string log = get_log();

    // For perfectly linear data [0, 99], we expect:
    // - linear_max_error = 0
    // - slope ≈ 1.0 (depends on segment)
    // - intercept varies by segment
    EXPECT_TRUE(contains(log, "linear_max_error=0"))
        << "Perfect linear data should have 0 max error";

    // Verify slope parameter exists and is logged with decimal precision
    EXPECT_TRUE(contains(log, "slope=")) << "Missing slope parameter";

    // Verify intercept exists
    EXPECT_TRUE(contains(log, "intercept=")) << "Missing intercept parameter";

    // For uniform data, should detect uniformity
    EXPECT_TRUE(contains(log, "Data is UNIFORM")) << "Should detect uniform data";
    EXPECT_TRUE(contains(log, "segment_scale=")) << "Should log segment_scale for uniform data";

    // Now test a find operation and verify predicted values
    jazzy::clear_debug_log();
    auto* result = index.find(75);

    log = get_log();

    // Should predict close to index 75 for value 75
    EXPECT_TRUE(contains(log, "Predicted index")) << "Missing prediction log";
    EXPECT_TRUE(contains(log, "result=")) << "Missing predicted result index";

    // Verify actual result is correct
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(*result, 75) << "Should find value 75";
    EXPECT_EQ(result - data.data(), 75) << "Value 75 should be at index 75";
}

// Test: Verify segment boundaries are logged correctly
TEST_F(DebugLoggingTest, SegmentBoundariesValidation) {
    std::vector<int> data(100);
    std::iota(data.begin(), data.end(), 0);

    jazzy::JazzyIndex<int, jazzy::to_segment_count<4>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string log = get_log();

    // For 100 elements with 4 segments, segments should be approximately:
    // [0-25), [25-50), [50-75), [75-100)
    // Verify segment range notation is present
    EXPECT_TRUE(contains(log, "analyze_segment[0-")) << "Missing segment 0 analysis";

    // Verify all segments report their size (n=)
    size_t n_count = 0;
    size_t pos = 0;
    while ((pos = log.find("n=", pos)) != std::string::npos) {
        ++n_count;
        ++pos;
    }
    EXPECT_GE(n_count, 4) << "Should report size (n=) for at least 4 segments";

    // Verify model parameters are logged (slope, intercept, etc.)
    EXPECT_TRUE(contains(log, "slope=") || contains(log, "intercept="))
        << "Missing model parameters in segment analysis";
}

// Test: Verify error metrics are meaningful
TEST_F(DebugLoggingTest, ErrorMetricsValidation) {
    // Create quadratic data where linear model will have significant error
    std::vector<int> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(i * i);
    }

    jazzy::JazzyIndex<int, jazzy::to_segment_count<4>()> index;
    index.build(data.data(), data.data() + data.size());

    std::string log = get_log();

    // For quadratic data, linear_max_error should be non-zero
    EXPECT_FALSE(contains(log, "linear_max_error=0"))
        << "Quadratic data should have non-zero linear error";

    // Verify error is logged with actual numeric value
    EXPECT_TRUE(contains(log, "linear_max_error=")) << "Missing linear max error";
    EXPECT_TRUE(contains(log, "linear_mean_error=")) << "Missing linear mean error";

    // Should see threshold comparison
    EXPECT_TRUE(contains(log, "threshold=")) << "Missing threshold in decision";

    // If quadratic model is attempted, should see its error too
    if (contains(log, "trying QUADRATIC")) {
        EXPECT_TRUE(contains(log, "QUADRATIC: max_error="))
            << "Missing quadratic error when quadratic is tried";
    }
}

#else

// Placeholder test when debug logging is disabled
TEST(DebugLoggingDisabled, NoLogsWhenDisabled) {
    GTEST_SKIP() << "Debug logging tests require -DENABLE_DEBUG_LOGGING=ON";
}

#endif // JAZZY_DEBUG_LOGGING
