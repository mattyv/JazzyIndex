#include "jazzy_index_parallel.hpp"
#include "jazzy_index_export.hpp"
#include "../benchmarks/fixtures.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

TEST(ReviewRegression, SparseUniformKeysPredictExactPositions) {
    std::vector<std::uint64_t> data(4096);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = (std::uint64_t{1} << 63) + i * 1'000'001;
    }
    const auto segment = jazzy::detail::make_segment(
        data.data(), 137, data.size(), std::less<>{}, jazzy::identity{});
    ASSERT_EQ(segment.model_type, jazzy::detail::ModelType::LINEAR);
    for (std::size_t i = segment.start_idx; i < segment.end_idx; ++i) {
        ASSERT_EQ(segment.predict(data[i]), i) << "key " << data[i];
    }
}

TEST(ReviewRegression, LargeIntegerModelsUseTheSameCoordinatesAsLookup) {
    const auto check = []<typename T>(T origin) {
        std::vector<T> data;
        for (int i = 0; i < 512; ++i) data.push_back(origin + T(i * 3));
        const auto segment = jazzy::detail::make_segment(
            data.data(), 0, data.size(), std::less<>{}, jazzy::identity{});
        ASSERT_EQ(segment.model_type, jazzy::detail::ModelType::LINEAR);
        EXPECT_LE(segment.max_error, 1u);
        for (std::size_t i = 0; i < data.size(); ++i) {
            const auto predicted = segment.predict(data[i]);
            EXPECT_LE(predicted > i ? predicted - i : i - predicted, segment.max_error);
        }
    };
    check(std::uint64_t{1} << 63);
    check(std::numeric_limits<std::int64_t>::min());
    check(std::int16_t{-500});
    EXPECT_EQ(jazzy::detail::key_delta(std::numeric_limits<std::int64_t>::max(),
                                     std::numeric_limits<std::int64_t>::min()),
              static_cast<double>(std::numeric_limits<std::uint64_t>::max()));
}

TEST(ReviewRegression, CubicTurningPointIsCheckedInNormalizedCoordinates) {
    // Deterministic sorted fixture whose least-squares cubic turns backwards near t=.49.
    std::mt19937 random(12);
    std::vector<double> data;
    for (int trial = 0; trial <= 10722; ++trial) {
        const int count = 100 + random() % 500;
        if (trial != 10722) {
            random.discard(count);
            continue;
        }
        double value = 100;
        for (int i = 0; i < count; ++i) {
            value += std::pow(10.0, double(random() % 60) / 10 - 3);
            data.push_back(value);
        }
    }
    const auto analysis = jazzy::detail::analyze_segment(data.data(), 0, data.size());
    EXPECT_NE(analysis.best_model, jazzy::detail::ModelType::CUBIC);
}

TEST(ReviewRegression, ExtremeFiniteKeysRemainSearchable) {
    const double max = std::numeric_limits<double>::max();
    for (const auto& data : {std::vector<double>{-max, 0, max},
                             std::vector<double>{0, 1e-15, 2e-15, 1e-14}}) {
        jazzy::JazzyIndex<double, jazzy::SegmentCount::SINGLE> index;
        for (bool parallel : {false, true}) {
            if (parallel) index.build_parallel(data.begin(), data.end());
            else index.build(data.begin(), data.end());
            for (double key : data) {
                ASSERT_NE(index.find(key), index.end());
                EXPECT_EQ(*index.find(key), key);
            }
            EXPECT_EQ(index.find(1.0), index.end());
        }
    }
}

TEST(ReviewRegression, SingletonCustomBuildCanAlwaysBeFinalized) {
    std::vector<int> data{42};
    jazzy::JazzyIndex<int> index;
    const auto tasks = index.prepare_build_tasks(data.data(), data.data() + data.size());
    std::vector<jazzy::detail::SegmentAnalysis<int>> results;
    for (const auto& task : tasks) results.push_back(task.execute());
    ASSERT_NO_THROW(index.finalize_build(results));
    EXPECT_EQ(index.find(42), data.data());
}

TEST(ReviewRegression, MovingAnIndexLeavesTheSourceSafeToQueryAndRebuild) {
    const std::vector<int> data{1, 3, 5};
    jazzy::JazzyIndex<int> source(data.begin(), data.end());
    auto destination = std::move(source);
    EXPECT_EQ(destination.find(3), data.data() + 1);
    ASSERT_FALSE(source.is_built());
    EXPECT_EQ(source.find(3), source.end());
    EXPECT_EQ(source.num_segments(), 0u);
    source.build(data.begin(), data.end());
    destination = std::move(source);
    ASSERT_FALSE(source.is_built());
    EXPECT_EQ(source.find(3), source.end());
    EXPECT_EQ(destination.find(5), data.data() + 2);
    const auto copy = destination;
    EXPECT_EQ(copy.find(1), data.data());
}

TEST(ReviewRegression, EmptyMetadataDoesNotExtractADefaultValue) {
    struct Value { int key = -1; };
    struct Extract {
        int operator()(const Value& value) const {
            if (value.key == -1) throw std::runtime_error("not a stored value");
            return value.key;
        }
    };
    struct Compare {
        bool operator()(const Value& a, const Value& b) const { return a.key < b.key; }
    };
    jazzy::JazzyIndex<Value, jazzy::SegmentCount::SINGLE, Compare, Extract> index;
    EXPECT_NO_THROW((void)jazzy::export_index_metadata(index));
}

TEST(ReviewRegression, BenchmarkGeneratorsHandleSmallAndFullWidthRanges) {
    using namespace dataset;
    const auto max = std::numeric_limits<std::uint64_t>::max();
    EXPECT_TRUE(generate_clustered(0, 5, .02, 42, 0, max).empty());
    EXPECT_TRUE(generate_mixed(0, .5, 1, 5, .02, 42, 0, max).empty());
    for (auto generate : {generate_quadratic, generate_extreme_polynomial,
                          generate_inverse_polynomial}) {
        const auto single = generate(1, 42, 0, max);
        ASSERT_EQ(single.size(), 1u);
        EXPECT_EQ(single[0], 0u);
        const auto data = generate(1000, 42, 0, max);
        EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
    }
    const auto data = finalize_samples({0., 1e19, static_cast<double>(max)}, 0, max);
    ASSERT_EQ(data.size(), 3u);
    EXPECT_EQ(data[1], 10000000000000000000ULL);
    EXPECT_EQ(data[2], max);
}

TEST(ReviewRegression, LearnedSearchMatchesStandardLibraryForHitsAndMisses) {
    std::mt19937_64 random(90210);
    for (const auto origin : {std::uint64_t{0}, std::uint64_t{1} << 63}) {
        std::vector<std::uint64_t> data;
        for (int i = 0; i < 2048; ++i) data.push_back(origin + (random() % 1500) * 3);
        std::sort(data.begin(), data.end());
        jazzy::JazzyIndex<std::uint64_t, jazzy::SegmentCount::NANO> index;
        for (bool parallel : {false, true}) {
            if (parallel) index.build_parallel(data.begin(), data.end());
            else index.build(data.begin(), data.end());
            for (std::uint64_t delta = 0; delta < 4501; ++delta) {
                const auto key = origin + delta;
                const auto expected = std::lower_bound(data.data(), data.data() + data.size(), key);
                const auto actual = index.find(key);
                const bool present = expected != data.data() + data.size() && *expected == key;
                ASSERT_EQ(actual != index.end(), present);
                if (present) EXPECT_EQ(*actual, key);
                EXPECT_EQ(index.find_lower_bound(key), expected);
                EXPECT_EQ(index.find_upper_bound(key),
                          std::upper_bound(data.data(), data.data() + data.size(), key));
            }
        }
    }
}
