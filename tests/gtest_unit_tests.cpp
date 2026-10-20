// Google Test migration of original unit tests from unit_tests.cpp
// Migrated tests now use Google Test framework for better organization,
// parameterization, and integration with modern tooling.

#include "jazzy_index.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace {

// Standalone helper functions that work with any segment size
template <typename IndexType, typename T>
bool expect_found(const IndexType& index,
                  const std::vector<T>& data,
                  const T& value) {
    const T* result = index.find(value);
    if (result == data.data() + data.size()) {
        return false;
    }
    return *result == value;
}

template <typename IndexType, typename T>
bool expect_missing(const IndexType& index,
                    const std::vector<T>& data,
                    const T& value) {
    const T* result = index.find(value);
    return result == data.data() + data.size();
}

// Test fixture for basic JazzyIndex operations
template <typename T, std::size_t Segments = 256>
class JazzyIndexTest : public ::testing::Test {
protected:
    jazzy::JazzyIndex<T, jazzy::to_segment_count<Segments>()> build_index(const std::vector<T>& data) {
        jazzy::JazzyIndex<T, jazzy::to_segment_count<Segments>()> index;
        index.build(data.data(), data.data() + data.size());
        return index;
    }
};

// Type aliases for common test configurations
using IntIndex = JazzyIndexTest<int, 256>;
using UInt64Index = JazzyIndexTest<std::uint64_t, 64>;

}  // namespace

// Test: Empty index returns end iterator
TEST_F(IntIndex, EmptyIndexReturnsEndIterator) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data;
    jazzy::JazzyIndex<int> index;
    index.build(data.data(), data.data());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build"), std::string::npos)
            << "Should log build even for empty data";
    }
    jazzy::clear_debug_log();
#endif

    EXPECT_TRUE(expect_missing(index, data, 42));
    EXPECT_EQ(index.size(), 0);

#ifdef JAZZY_DEBUG_LOGGING
    std::string find_log = jazzy::get_debug_log();
    if (!find_log.empty()) {
        EXPECT_NE(find_log.find("not built or empty"), std::string::npos)
            << "Should detect empty index";
    }
#endif
}

// Test: Single element hit and miss
TEST_F(IntIndex, SingleElementHitAndMiss) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data{42};
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build: Building index for 1 elements"),
                 std::string::npos) << "Should log single element build";
    }
    jazzy::clear_debug_log();
#endif

    EXPECT_TRUE(expect_found(index, data, 42));
    EXPECT_TRUE(expect_missing(index, data, 43));
    EXPECT_TRUE(expect_missing(index, data, 41));
    EXPECT_EQ(index.size(), 1);
}

// Test: Uniform sequence lookups
TEST_F(IntIndex, UniformSequenceLookups) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    // Verify that build was logged with correct parameters
    EXPECT_NE(build_log.find("JazzyIndex::build: Building index for 1000 elements"), std::string::npos)
        << "Missing build start log with element count";

    // Verify that segments were analyzed
    EXPECT_NE(build_log.find("analyze_segment"), std::string::npos) << "Missing segment analysis";

    // For uniform sequential data, should have 0 linear error
    EXPECT_NE(build_log.find("linear_max_error=0"), std::string::npos)
        << "Expected 0 linear error for uniform sequential data";

    // With 256 segments for 1000 elements, uniformity detection may or may not trigger
    // (depends on segment size thresholds), but we should at least see LINEAR model
    EXPECT_NE(build_log.find("Selected LINEAR model"), std::string::npos)
        << "Expected LINEAR model for uniform sequential data";

    jazzy::clear_debug_log();
#endif

    // Test various points in the sequence
    EXPECT_TRUE(expect_found(index, data, 0));
    EXPECT_TRUE(expect_found(index, data, 10));
    EXPECT_TRUE(expect_found(index, data, 500));
    EXPECT_TRUE(expect_found(index, data, 999));

#ifdef JAZZY_DEBUG_LOGGING
    std::string find_log = jazzy::get_debug_log();
    // Verify that find operations were logged (should have 4 finds)
    size_t find_count = 0;
    size_t pos = 0;
    while ((pos = find_log.find("JazzyIndex::find: Called", pos)) != std::string::npos) {
        ++find_count;
        ++pos;
    }
    EXPECT_EQ(find_count, 4) << "Should have 4 find operations logged";

    // Verify that predictions were logged
    EXPECT_NE(find_log.find("Predicted index"), std::string::npos) << "Missing prediction logs";

    // Verify segment finding was logged
    EXPECT_NE(find_log.find("find_segment: Called"), std::string::npos)
        << "Missing find_segment logs";

    // Should use either UNIFORM path or binary search (depends on whether uniform was detected)
    bool has_uniform = find_log.find("UNIFORM path") != std::string::npos;
    bool has_binary = find_log.find("binary search") != std::string::npos;
    EXPECT_TRUE(has_uniform || has_binary) << "Expected either UNIFORM path or binary search";
#endif

    // Test missing values
    EXPECT_TRUE(expect_missing(index, data, -1));
    EXPECT_TRUE(expect_missing(index, data, 1500));
}

// Test: Duplicate values
TEST_F(IntIndex, DuplicateValues) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data{1, 1, 1, 2, 2, 5, 5, 9};
    jazzy::JazzyIndex<int, jazzy::to_segment_count<128>()> index;
    index.build(data.data(), data.data() + data.size());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build: Building index for 8 elements"),
                 std::string::npos) << "Should log 8 element build";
    }
    jazzy::clear_debug_log();
#endif

    EXPECT_TRUE(expect_found(index, data, 1));
    EXPECT_TRUE(expect_found(index, data, 2));
    EXPECT_TRUE(expect_found(index, data, 5));
    EXPECT_TRUE(expect_found(index, data, 9));

    EXPECT_TRUE(expect_missing(index, data, 3));
    EXPECT_TRUE(expect_missing(index, data, 0));
    EXPECT_TRUE(expect_missing(index, data, 10));
}

// Test: uint64_t values including large numbers
TEST_F(UInt64Index, LargeUInt64Values) {
    std::vector<std::uint64_t> data{
        0ull,
        42ull,
        1'000ull,
        1'000'000ull,
        (1ull << 32),
        (1ull << 40)
    };
    auto index = build_index(data);

    EXPECT_TRUE(expect_found(index, data, uint64_t{0}));
    EXPECT_TRUE(expect_found(index, data, uint64_t{42}));
    EXPECT_TRUE(expect_found(index, data, uint64_t{1ull << 40}));

    EXPECT_TRUE(expect_missing(index, data, uint64_t{(1ull << 40) + 1}));
    EXPECT_TRUE(expect_missing(index, data, uint64_t{100}));
}

// Parameterized test for different segment configurations
class SegmentSizeTest : public ::testing::TestWithParam<std::size_t> {
protected:
    template <std::size_t Segments>
    bool test_with_segments(std::size_t segment_size) {
        std::vector<int> data(1000);
        std::iota(data.begin(), data.end(), 0);

        jazzy::JazzyIndex<int, jazzy::to_segment_count<Segments>()> index;
        index.build(data.data(), data.data() + data.size());

        // Test various lookups
        for (int value : {0, 250, 500, 750, 999}) {
            const int* result = index.find(value);
            if (result == data.data() + data.size() || *result != value) {
                return false;
            }
        }
        return true;
    }
};

TEST_P(SegmentSizeTest, UniformDataWithVariousSegmentSizes) {
    std::size_t segment_size = GetParam();

    bool result = false;
    switch (segment_size) {
        case 64:
            result = test_with_segments<64>(segment_size);
            break;
        case 128:
            result = test_with_segments<128>(segment_size);
            break;
        case 256:
            result = test_with_segments<256>(segment_size);
            break;
        case 512:
            result = test_with_segments<512>(segment_size);
            break;
        default:
            FAIL() << "Unsupported segment size: " << segment_size;
    }

    EXPECT_TRUE(result) << "Failed with segment size: " << segment_size;
}

INSTANTIATE_TEST_SUITE_P(
    DifferentSegmentCounts,
    SegmentSizeTest,
    ::testing::Values(64, 128, 256, 512)
);

// Test: Very small datasets
TEST_F(IntIndex, TwoElements) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data{1, 2};
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build: Building index for 2 elements"),
                 std::string::npos) << "Should log 2 element build";
    }
#endif

    EXPECT_TRUE(expect_found(index, data, 1));
    EXPECT_TRUE(expect_found(index, data, 2));
    EXPECT_TRUE(expect_missing(index, data, 0));
    EXPECT_TRUE(expect_missing(index, data, 3));
}

TEST_F(IntIndex, ThreeElements) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data{10, 20, 30};
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build: Building index for 3 elements"),
                 std::string::npos) << "Should log 3 element build";
    }
#endif

    EXPECT_TRUE(expect_found(index, data, 10));
    EXPECT_TRUE(expect_found(index, data, 20));
    EXPECT_TRUE(expect_found(index, data, 30));
    EXPECT_TRUE(expect_missing(index, data, 15));
    EXPECT_TRUE(expect_missing(index, data, 25));
}

// Test: Rebuild index
TEST_F(IntIndex, RebuildIndex) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data1{1, 2, 3, 4, 5};
    std::vector<int> data2{10, 20, 30, 40, 50};

    jazzy::JazzyIndex<int, jazzy::to_segment_count<256>()> index;

    // Build with first dataset
    index.build(data1.data(), data1.data() + data1.size());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log1 = jazzy::get_debug_log();
    if (!build_log1.empty()) {
        EXPECT_NE(build_log1.find("JazzyIndex::build: Building index for 5 elements"),
                 std::string::npos) << "Should log first build with 5 elements";
    }
    jazzy::clear_debug_log();
#endif

    EXPECT_TRUE(expect_found(index, data1, 3));
    EXPECT_EQ(index.size(), 5);

    // Rebuild with second dataset
    index.build(data2.data(), data2.data() + data2.size());

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log2 = jazzy::get_debug_log();
    if (!build_log2.empty()) {
        EXPECT_NE(build_log2.find("JazzyIndex::build: Building index for 5 elements"),
                 std::string::npos) << "Should log rebuild with 5 elements";
    }
#endif

    EXPECT_TRUE(expect_found(index, data2, 30));
    EXPECT_TRUE(expect_missing(index, data2, 3));  // Value from old dataset
    EXPECT_EQ(index.size(), 5);
}

// Test: Negative numbers
TEST_F(IntIndex, NegativeNumbers) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    std::vector<int> data{-100, -50, -10, 0, 10, 50, 100};
    auto index = build_index(data);

#ifdef JAZZY_DEBUG_LOGGING
    std::string build_log = jazzy::get_debug_log();
    if (!build_log.empty()) {
        EXPECT_NE(build_log.find("JazzyIndex::build: Building index for 7 elements"),
                 std::string::npos) << "Should log 7 element build";
    }
#endif

    EXPECT_TRUE(expect_found(index, data, -100));
    EXPECT_TRUE(expect_found(index, data, -50));
    EXPECT_TRUE(expect_found(index, data, 0));
    EXPECT_TRUE(expect_found(index, data, 50));
    EXPECT_TRUE(expect_found(index, data, 100));

    EXPECT_TRUE(expect_missing(index, data, -101));
    EXPECT_TRUE(expect_missing(index, data, 101));
    EXPECT_TRUE(expect_missing(index, data, -25));
}

TEST(JazzyIndexBuildTest, RejectsInternalInversionAndReversedRange) {
    std::vector<int> data{0, 2, 1, 3};
    jazzy::JazzyIndex<int, jazzy::SegmentCount::SINGLE> index;

    EXPECT_THROW(index.build(data.data(), data.data() + data.size()), std::runtime_error);
    EXPECT_THROW(index.build(data.data() + 3, data.data() + 1), std::invalid_argument);
    EXPECT_FALSE(index.is_built());
}

TEST(JazzyIndexBuildTest, EmptyRebuildClearsState) {
    std::vector<int> data{1, 2, 3};
    std::vector<int> empty;
    jazzy::JazzyIndex<int> index(data.data(), data.data() + data.size());

    index.build(empty.data(), empty.data());

    EXPECT_TRUE(index.is_built());
    EXPECT_EQ(index.size(), 0u);
    EXPECT_EQ(index.num_segments(), 0u);
    EXPECT_EQ(index.find(2), index.end());
    EXPECT_EQ(index.find_lower_bound(2), index.end());
    EXPECT_EQ(index.find_upper_bound(2), index.end());
    EXPECT_EQ(index.equal_range(2), std::make_pair(index.end(), index.end()));
}

TEST(JazzyIndexBuildTest, FailedRebuildPreservesPreviousIndex) {
    std::vector<int> good{1, 2, 3, 4};
    std::vector<int> bad{5, 7, 6, 8};
    jazzy::JazzyIndex<int> index(good.data(), good.data() + good.size());

    EXPECT_THROW(index.build(bad.data(), bad.data() + bad.size()), std::runtime_error);

    EXPECT_TRUE(index.is_built());
    EXPECT_EQ(index.size(), good.size());
    EXPECT_EQ(index.find(3), good.data() + 2);
    EXPECT_EQ(index.find(7), index.end());
}

// Regression: refine_segments must split segments whose fit misses the error
// budget. A single linear model over piecewise data with a huge jump has an
// enormous max_error, so a SINGLE-segment index must end up with more than one
// segment and still answer every query correctly.
TEST(JazzyIndexRefineTest, SplitsSegmentsThatMissErrorBudget) {
    std::vector<std::int64_t> data;
    data.reserve(4096);
    for (std::int64_t i = 0; i < 4096; ++i) {
        data.push_back(i < 2048 ? i : 100000000LL + i * 7);
    }

    jazzy::JazzyIndex<std::int64_t, jazzy::SegmentCount::SINGLE> index;
    index.build(data.data(), data.data() + data.size());

    EXPECT_GT(index.num_segments(), 1u) << "badly-fit segment was not split";

    for (const auto& value : data) {
        const auto* it = index.find(value);
        ASSERT_NE(it, index.end());
        EXPECT_EQ(*it, value);
    }
    EXPECT_EQ(index.find(50000000LL), index.end());  // in the jump gap
}

// Refinement must stop splitting below MIN_SPLIT_SIZE even when no model can
// fit the data (duplicate runs make every model's error irreducible).
TEST(JazzyIndexRefineTest, SplitTerminatesOnUnfittableData) {
    std::vector<int> data;
    for (int v = 0; v < 64; ++v) {
        for (int r = 0; r < 200; ++r) {
            data.push_back(v * v * v);
        }
    }

    jazzy::JazzyIndex<int, jazzy::SegmentCount::SINGLE> index;
    index.build(data.data(), data.data() + data.size());

    for (int v = 0; v < 64; ++v) {
        const auto* it = index.find(v * v * v);
        ASSERT_NE(it, index.end());
        EXPECT_EQ(*it, v * v * v);
    }
}
