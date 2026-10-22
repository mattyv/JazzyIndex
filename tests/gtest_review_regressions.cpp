#include "jazzy_index_parallel.hpp"
#include "jazzy_index_export.hpp"
#include "../benchmarks/fixtures.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
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

namespace {

constexpr std::size_t kFullRangeLogRadixOffsets = 50 * 8192 + 16383 + 2;

template <typename UInt>
std::vector<UInt> radix_data(bool duplicates = true) {
    constexpr std::uint64_t factor = std::is_same_v<UInt, std::uint64_t>
        ? 1'000'000'000ULL
        : 31ULL;
    std::vector<UInt> data{0};
    for (std::uint64_t i = 1; i < 900; ++i) {
        data.push_back(static_cast<UInt>(i * i * factor));
    }
    data.insert(data.end(), duplicates ? 320 : 1,
                static_cast<UInt>(900 * 900 * factor));
    for (std::uint64_t i = 901; i <= 3000; ++i) {
        data.push_back(static_cast<UInt>(i * i * factor));
    }
    data.push_back(std::numeric_limits<UInt>::max());
    return data;
}

template <typename UInt>
std::vector<UInt> radix_boundary_data() {
    std::vector<UInt> data{0};
    constexpr unsigned digits = std::numeric_limits<UInt>::digits;
    const auto max = std::numeric_limits<UInt>::max();
    for (unsigned bit = 1; bit < digits; ++bit) {
        const std::uint64_t pivot = std::uint64_t{1} << bit;
        for (const std::uint64_t value : {pivot - 1, pivot, pivot + 1}) {
            if (value <= max) data.push_back(static_cast<UInt>(value));
        }
    }
    data.push_back(max);
    std::sort(data.begin(), data.end());
    data.erase(std::unique(data.begin(), data.end()), data.end());
    return data;
}

template <typename UInt>
std::vector<UInt> radix_high_origin_data() {
    const UInt origin = std::numeric_limits<UInt>::max() - UInt{1'000'000};
    std::vector<UInt> data;
    data.reserve(1001);
    for (std::uint64_t i = 0; i <= 1000; ++i) {
        data.push_back(origin + static_cast<UInt>(i * i));
    }
    return data;
}

template <typename UInt, typename Compare = std::less<>>
void expect_routed_lookup(const std::vector<UInt>& data, Compare compare = Compare{}) {
    jazzy::JazzyIndex<UInt, jazzy::SegmentCount::NANO, Compare> index;
    const auto check = [&](const auto& target) {
        for (std::size_t i = 0; i < data.size(); ++i) {
            const auto expected = std::lower_bound(
                data.data(), data.data() + data.size(), data[i], compare);
            const auto actual = target.find(data[i]);
            ASSERT_NE(actual, target.end()) << data[i];
            EXPECT_EQ(*actual, data[i]);
            EXPECT_EQ(target.find_lower_bound(data[i]), expected);
            EXPECT_EQ(target.find_upper_bound(data[i]),
                      std::upper_bound(data.data(), data.data() + data.size(), data[i], compare));
            if (i + 1 < data.size() && data[i] + 1 < data[i + 1]) {
                EXPECT_EQ(target.find(data[i] + 1), target.end());
            }
        }
    };

    index.build(data.begin(), data.end(), compare);
    ASSERT_GT(index.size_in_bytes(),
              index.num_segments() * sizeof(jazzy::detail::Segment<UInt>));
    check(index);

    auto copy = index;
    check(copy);
    auto moved = std::move(index);
    check(moved);
    jazzy::JazzyIndex<UInt, jazzy::SegmentCount::NANO, Compare> assigned;
    assigned = copy;
    check(assigned);
    assigned = std::move(moved);
    check(assigned);

    assigned.build_parallel(data.begin(), data.end(), compare);
    check(assigned);
    assigned.build(data.begin(), data.begin());
    EXPECT_EQ(assigned.size_in_bytes(), 0u);
    EXPECT_EQ(assigned.find(data.front()), assigned.end());
    assigned.build(data.begin(), data.begin() + 1);
    EXPECT_EQ(assigned.size_in_bytes(), sizeof(jazzy::detail::Segment<UInt>));
    EXPECT_EQ(assigned.find(data.front()), assigned.begin());
    std::vector<UInt> uniform(256, data.front());
    assigned.build(uniform.begin(), uniform.end());
    EXPECT_EQ(assigned.size_in_bytes(),
              assigned.num_segments() * sizeof(jazzy::detail::Segment<UInt>));
    EXPECT_NE(assigned.find(uniform.front()), assigned.end());
}

struct CustomLess {
    bool operator()(std::uint64_t left, std::uint64_t right) const { return left < right; }
};

struct Keyed {
    std::uint64_t key;
};

struct KeyedLess {
    bool operator()(const Keyed& left, const Keyed& right) const { return left.key < right.key; }
};

}  // namespace

TEST(ReviewRegression, RadixRoutingCoversUnsignedBoundsDuplicatesAndStateChanges) {
    expect_routed_lookup(radix_data<std::uint64_t>());
    expect_routed_lookup(radix_data<std::uint64_t>(false));
    expect_routed_lookup<std::uint32_t, std::less<std::uint32_t>>(radix_data<std::uint32_t>());
    expect_routed_lookup<std::uint32_t, std::less<std::uint32_t>>(radix_data<std::uint32_t>(false));
}

TEST(ReviewRegression, LogRadixRoutingCoversPowerBoundaries) {
    expect_routed_lookup(radix_boundary_data<std::uint32_t>());
    expect_routed_lookup(radix_boundary_data<std::uint64_t>());
}

TEST(ReviewRegression, LogRadixRoutingHandlesHighUnsignedOrigins) {
    expect_routed_lookup(radix_high_origin_data<std::uint32_t>());
    expect_routed_lookup(radix_high_origin_data<std::uint64_t>());
}

TEST(ReviewRegression, RadixSingleCandidateHandlesGapsAndFinalBucket) {
    const auto data = radix_high_origin_data<std::uint64_t>();
    jazzy::JazzyIndex<std::uint64_t, jazzy::SegmentCount::NANO> index(
        data.begin(), data.end());
    ASSERT_GT(index.size_in_bytes(),
              index.num_segments() * sizeof(jazzy::detail::Segment<std::uint64_t>));

    const auto origin = data.front();
    EXPECT_EQ(index.find(origin), data.data());
    EXPECT_EQ(index.find(origin + 2), index.end());
    EXPECT_EQ(index.find(std::numeric_limits<std::uint64_t>::max() - 1), index.end());
    EXPECT_EQ(index.find(std::numeric_limits<std::uint64_t>::max()), data.data() + data.size() - 1);
}

TEST(ReviewRegression, RadixSmallCandidateScanHandlesSharedHighBucket) {
    std::vector<std::uint64_t> data;
    data.reserve(512);
    for (std::uint64_t i = 0; i < 128; ++i) data.push_back(i * 2);
    const std::uint64_t high = std::uint64_t{1} << 63;
    for (std::uint64_t i = 0; i < 384; ++i) data.push_back(high + i * 2);

    jazzy::JazzyIndex<std::uint64_t, jazzy::SegmentCount::PICO> index(
        data.begin(), data.end());
    ASSERT_GT(index.size_in_bytes(),
              index.num_segments() * sizeof(jazzy::detail::Segment<std::uint64_t>));
    for (std::size_t i = 128; i < data.size(); ++i) {
        EXPECT_EQ(index.find(data[i]), data.data() + i);
        EXPECT_EQ(index.find(data[i] + 1), index.end());
    }
    EXPECT_EQ(index.find(data.back()), data.data() + data.size() - 1);
}

TEST(ReviewRegression, RadixRoutingSkipsUnsupportedComparatorAndExtractor) {
    const auto data = radix_data<std::uint64_t>();
    jazzy::JazzyIndex<std::uint64_t, jazzy::SegmentCount::NANO> routed(data.begin(), data.end());
    jazzy::JazzyIndex<std::uint64_t, jazzy::SegmentCount::NANO, CustomLess> custom(data.begin(), data.end());
    EXPECT_EQ(custom.size_in_bytes() +
                  kFullRangeLogRadixOffsets * sizeof(std::uint32_t),
              routed.size_in_bytes());
    for (auto key : {data.front(), data[900], data.back(), data[900] + 1}) {
        EXPECT_EQ(custom.find(key) != custom.end(),
                  std::binary_search(data.begin(), data.end(), key, CustomLess{}));
    }

    std::vector<Keyed> keyed;
    keyed.reserve(data.size());
    for (const auto key : data) keyed.push_back({key});
    jazzy::JazzyIndex<Keyed, jazzy::SegmentCount::NANO, KeyedLess,
                      decltype(&Keyed::key)> extracted(keyed.begin(), keyed.end(),
                                                       KeyedLess{}, &Keyed::key);
    EXPECT_EQ(extracted.size_in_bytes(),
              extracted.num_segments() * sizeof(jazzy::detail::Segment<Keyed>));
    EXPECT_NE(extracted.find({data[900]}), extracted.end());
    EXPECT_EQ(extracted.find({data[900] + 1}), extracted.end());
}

namespace {

template <typename Index, typename T, std::size_t N, typename Compare>
void expect_batch_matches_lower_bound(const Index& index, const std::vector<T>& data,
                                      const std::array<T, N>& keys, Compare compare) {
    std::array<const T*, N> results{};
    index.find_batch(keys, results);
    for (std::size_t lane = 0; lane < keys.size(); ++lane) {
        const auto expected = std::lower_bound(data.begin(), data.end(), keys[lane], compare);
        const bool hit = expected != data.end() &&
            !compare(*expected, keys[lane]) && !compare(keys[lane], *expected);
        const T* actual = results[lane];
        ASSERT_TRUE(actual == index.end() || (actual >= data.data() && actual < index.end()))
            << "lane " << lane;
        ASSERT_EQ(actual != index.end(), hit) << "lane " << lane;
        if (hit) EXPECT_FALSE(compare(*actual, keys[lane]) || compare(keys[lane], *actual));
        EXPECT_EQ(actual != index.end(), index.find(keys[lane]) != index.end()) << "lane " << lane;
    }
}

struct BatchRecord {
    std::uint64_t key;
};

struct BatchRecordLess {
    bool operator()(const BatchRecord& left, const BatchRecord& right) const {
        return left.key < right.key;
    }
};

template <typename UInt>
void expect_dense_lookup(UInt origin) {
    constexpr std::size_t count = 512;
    std::vector<UInt> data(count);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = origin + static_cast<UInt>(i);

    using Index = jazzy::JazzyIndex<UInt, jazzy::SegmentCount::NANO>;
    const auto check = [&](const Index& index) {
        for (std::size_t i = 0; i < data.size(); ++i) {
            EXPECT_EQ(index.find(data[i]), data.data() + i);
        }
        if (origin > 0) EXPECT_EQ(index.find(origin - 1), index.end());
        if (data.back() < std::numeric_limits<UInt>::max()) {
            EXPECT_EQ(index.find(data.back() + 1), index.end());
        }
        expect_batch_matches_lower_bound(index, data,
                                         std::array<UInt, 4>{data.front(), data[1], data[255], data.back()},
                                         std::less<>{});
    };

    Index index(data.begin(), data.end());
    check(index);
    const auto copy = index;
    check(copy);
    auto moved = std::move(index);
    check(moved);
    EXPECT_EQ(index.find(data.front()), index.end());

    Index assigned;
    assigned = copy;
    check(assigned);
    assigned = std::move(moved);
    check(assigned);
    assigned.build_parallel(data.begin(), data.end());
    check(assigned);
    assigned.build(data.begin(), data.begin());
    EXPECT_EQ(assigned.find(data.front()), assigned.end());
}

}  // namespace

TEST(ReviewRegression, BatchLookupHandlesUnbuiltEmptyDuplicatesAndMisses) {
    using Index = jazzy::JazzyIndex<std::uint64_t, jazzy::SegmentCount::NANO>;
    const std::array<std::uint64_t, 4> keys{0, 1, 8, 21};
    std::array<const std::uint64_t*, 4> results{};
    Index index;
    index.find_batch(keys, results);
    for (const auto* result : results) EXPECT_EQ(result, index.end());

    const std::vector<std::uint64_t> data{1, 1, 1, 8, 8, 20};
    index.build(data.begin(), data.end());
    expect_batch_matches_lower_bound(index, data, keys, std::less<>{});
    expect_batch_matches_lower_bound(index, data, std::array<std::uint64_t, 4>{1, 2, 20, 21},
                                     std::less<>{});

    Index empty;
    empty.build(data.begin(), data.begin());
    empty.find_batch(keys, results);
    for (const auto* result : results) EXPECT_EQ(result, empty.end());
}

TEST(ReviewRegression, DenseUnsignedRoutingHandlesOriginsDuplicatesAndBatch) {
    expect_dense_lookup<std::uint32_t>(0);
    expect_dense_lookup<std::uint32_t>(std::numeric_limits<std::uint32_t>::max() - 511);
    expect_dense_lookup<std::uint64_t>(0);
    expect_dense_lookup<std::uint64_t>(std::numeric_limits<std::uint64_t>::max() - 511);

    const std::vector<std::uint64_t> duplicates{0, 1, 1, 3};
    jazzy::JazzyIndex<std::uint64_t, jazzy::SegmentCount::NANO> index(
        duplicates.begin(), duplicates.end());
    EXPECT_EQ(index.find(2), index.end());
    expect_batch_matches_lower_bound(index, duplicates,
                                     std::array<std::uint64_t, 4>{0, 1, 2, 3}, std::less<>{});
    index.build_parallel(duplicates.begin(), duplicates.end());
    EXPECT_EQ(index.find(2), index.end());
}

TEST(ReviewRegression, BatchLookupPreservesComparatorAndExtractorSemantics) {
    std::vector<int> descending;
    descending.reserve(512);
    for (int value = 512; value > 0; --value) {
        descending.push_back(value * value + (value % 7 == 0 ? 1'000 : 0));
    }
    std::sort(descending.begin(), descending.end(), std::greater<>{});
    jazzy::JazzyIndex<int, jazzy::SegmentCount::NANO, std::greater<>> descending_index(
        descending.begin(), descending.end(), std::greater<>{});
    expect_batch_matches_lower_bound(descending_index, descending,
                                     std::array<int, 4>{300'000, descending[0], descending[200], 0},
                                     std::greater<>{});

    std::vector<BatchRecord> records;
    records.reserve(512);
    std::uint64_t key = 0;
    for (std::size_t i = 0; i < 512; ++i) {
        key += i % 7 == 0 ? 10'000 + i * i : 1 + i % 3;
        records.push_back({key});
    }
    jazzy::JazzyIndex<BatchRecord, jazzy::SegmentCount::NANO, BatchRecordLess,
                      decltype(&BatchRecord::key)> extracted(records.begin(), records.end(),
                                                              BatchRecordLess{}, &BatchRecord::key);
    expect_batch_matches_lower_bound(extracted, records,
                                     std::array<BatchRecord, 4>{{{0}, records[1], records[200], {999'999}}},
                                     BatchRecordLess{});
}

TEST(ReviewRegression, BatchLookupRejectsNaNLikeScalarFind) {
    const std::vector<double> data{0.0, 0.5, 3.0, 10.0, 100.0};
    jazzy::JazzyIndex<double, jazzy::SegmentCount::NANO> index(data.begin(), data.end());
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)index.find(nan), std::invalid_argument);
    std::array<const double*, 4> results{};
    EXPECT_THROW(index.find_batch(std::array<double, 4>{0.0, nan, 3.0, 10.0}, results),
                 std::invalid_argument);
    expect_batch_matches_lower_bound(index, data, std::array<double, 4>{0.0, 0.25, 3.0, 100.0},
                                     std::less<>{});
}

TEST(ReviewRegression, BatchLookupMatchesScalarOnAdversarialNonUniformData) {
    std::vector<std::uint64_t> data;
    data.reserve(2048);
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < 2048; ++i) {
        value += i % 7 == 0 ? 10'000 + i * i : 1 + (i % 3);
        data.push_back(value);
    }
    jazzy::JazzyIndex<std::uint64_t, jazzy::SegmentCount::NANO> index(data.begin(), data.end());
    for (std::size_t i = 0; i + 3 < data.size(); i += 4) {
        expect_batch_matches_lower_bound(index, data,
                                         std::array<std::uint64_t, 4>{
                                             data[i], data[i] + 1, data[i + 2], data[i + 3] + 1},
                                         std::less<>{});
    }
    expect_batch_matches_lower_bound(index, data, std::array<std::uint64_t, 1>{data[10]},
                                     std::less<>{});
    expect_batch_matches_lower_bound(index, data,
                                     std::array<std::uint64_t, 3>{data[20], data[21] + 1, data[22]},
                                     std::less<>{});
    expect_batch_matches_lower_bound(index, data,
                                     std::array<std::uint64_t, 8>{
                                         data[30], data[31] + 1, data[32], data[33] + 1,
                                         data[34], data[35] + 1, data[36], data[37] + 1},
                                     std::less<>{});
}

TEST(ReviewRegression, BatchLookupReturnsEndForAllOutOfRangeLanes) {
    const auto data = radix_high_origin_data<std::uint64_t>();
    jazzy::JazzyIndex<std::uint64_t, jazzy::SegmentCount::NANO> index(data.begin(), data.end());
    const auto origin = data.front();
    expect_batch_matches_lower_bound(index, data,
                                     std::array<std::uint64_t, 4>{0, 1, origin - 2, origin - 1},
                                     std::less<>{});
}

TEST(ReviewRegression, BatchLookupCoversLogRadixRangeWidthsAndBoundaries) {
    const auto data = radix_data<std::uint64_t>(false);
    jazzy::JazzyIndex<std::uint64_t, jazzy::SegmentCount::NANO> index(data.begin(), data.end());
    expect_batch_matches_lower_bound(index, data,
                                     std::array<std::uint64_t, 4>{
                                         data.front(), data[1], data[data.size() / 2], data.back()},
                                     std::less<>{});

    const auto boundaries = radix_boundary_data<std::uint64_t>();
    jazzy::JazzyIndex<std::uint64_t, jazzy::SegmentCount::NANO> boundary_index(
        boundaries.begin(), boundaries.end());
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t i = 0; i + 3 < boundaries.size(); i += 4) {
        expect_batch_matches_lower_bound(boundary_index, boundaries,
                                         std::array<std::uint64_t, 4>{
                                             boundaries[i], boundaries[i + 1], boundaries[i + 2],
                                             i + 4 == boundaries.size() ? maximum : boundaries[i + 3]},
                                         std::less<>{});
    }
    expect_batch_matches_lower_bound(boundary_index, boundaries,
                                     std::array<std::uint64_t, 4>{
                                         boundaries.back(), maximum, maximum - 1, 0},
                                     std::less<>{});
}
