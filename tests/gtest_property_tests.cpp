// Expanded property-based tests using RapidCheck with Google Test
// These tests verify invariants and properties that should hold for all inputs.

#include "jazzy_index.hpp"

#include <gtest/gtest.h>
#include <rapidcheck.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

using value_type = std::uint64_t;
using qi_index = jazzy::JazzyIndex<value_type, jazzy::to_segment_count<256>()>;

// Property: All inserted values can be found
TEST(PropertyTests, FindsAllInsertedIntegers) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    const bool result = rc::check("Finds all inserted integers", [](std::vector<value_type> values) {
        RC_PRE(!values.empty());
        std::sort(values.begin(), values.end());

        qi_index index;
        index.build(values.data(), values.data() + values.size());

#ifdef JAZZY_DEBUG_LOGGING
        std::string build_log = jazzy::get_debug_log();
        if (!build_log.empty()) {
            RC_ASSERT(build_log.find("JazzyIndex::build") != std::string::npos);
        }
#endif

        for (value_type value : values) {
            const value_type* result = index.find(value);
            RC_ASSERT(result != values.data() + values.size());
            RC_ASSERT(*result == value);
        }
    });
    EXPECT_TRUE(result);
}

// Property: Values outside range are rejected
TEST(PropertyTests, RejectsValuesOutsideRange) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    const bool result = rc::check("Rejects values outside range", [](std::vector<value_type> values) {
        RC_PRE(!values.empty());
        std::sort(values.begin(), values.end());

        qi_index index;
        index.build(values.data(), values.data() + values.size());

#ifdef JAZZY_DEBUG_LOGGING
        std::string build_log = jazzy::get_debug_log();
        if (!build_log.empty()) {
            RC_ASSERT(build_log.find("JazzyIndex::build") != std::string::npos);
        }
#endif

        const value_type* end = values.data() + values.size();

        if (values.front() > 0) {
            RC_ASSERT(index.find(values.front() - 1) == end);
        }
        if (values.back() < std::numeric_limits<value_type>::max()) {
            RC_ASSERT(index.find(values.back() + 1) == end);
        }
    });
    EXPECT_TRUE(result);
}

// Property: find() returns pointer within data range or end
TEST(PropertyTests, FindReturnsValidPointer) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    const bool result = rc::check("Find returns valid pointer", [](std::vector<value_type> values, value_type query) {
        RC_PRE(!values.empty());
        std::sort(values.begin(), values.end());

        qi_index index;
        index.build(values.data(), values.data() + values.size());

#ifdef JAZZY_DEBUG_LOGGING
        std::string build_log = jazzy::get_debug_log();
        if (!build_log.empty()) {
            RC_ASSERT(build_log.find("JazzyIndex::build") != std::string::npos);
        }
#endif

        const value_type* result = index.find(query);
        const value_type* begin = values.data();
        const value_type* end = begin + values.size();

        RC_ASSERT(result >= begin && result <= end);
    });
    EXPECT_TRUE(result);
}

// Property: Monotonicity - queries in sorted order return monotonic pointers
TEST(PropertyTests, MonotonicPointers) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    const bool result = rc::check("Monotonic pointers", [](std::vector<value_type> values) {
        RC_PRE(values.size() >= 2);
        std::sort(values.begin(), values.end());

        qi_index index;
        index.build(values.data(), values.data() + values.size());

#ifdef JAZZY_DEBUG_LOGGING
        std::string build_log = jazzy::get_debug_log();
        if (!build_log.empty()) {
            RC_ASSERT(build_log.find("JazzyIndex::build") != std::string::npos);
        }
#endif

        const value_type* prev = nullptr;
        for (value_type v : values) {
            const value_type* curr = index.find(v);
            if (prev != nullptr) {
                RC_ASSERT(curr >= prev);
            }
            prev = curr;
        }
    });
    EXPECT_TRUE(result);
}

// Property: Duplicate values are found
TEST(PropertyTests, FindsDuplicates) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    const bool result = rc::check("Finds duplicates", [](std::vector<value_type> values) {
        RC_PRE(!values.empty());
        std::sort(values.begin(), values.end());

        qi_index index;
        index.build(values.data(), values.data() + values.size());

#ifdef JAZZY_DEBUG_LOGGING
        std::string build_log = jazzy::get_debug_log();
        if (!build_log.empty()) {
            RC_ASSERT(build_log.find("JazzyIndex::build") != std::string::npos);
        }
#endif

        auto unique_vals = values;
        unique_vals.erase(std::unique(unique_vals.begin(), unique_vals.end()), unique_vals.end());

        for (value_type v : unique_vals) {
            const value_type* result = index.find(v);
            RC_ASSERT(result != values.data() + values.size());
            RC_ASSERT(*result == v);
        }
    });
    EXPECT_TRUE(result);
}

// Property: Size is preserved
TEST(PropertyTests, SizePreserved) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    const bool result = rc::check("Size preserved", [](std::vector<value_type> values) {
        std::sort(values.begin(), values.end());  // Sort data before building index
        qi_index index;
        index.build(values.data(), values.data() + values.size());

#ifdef JAZZY_DEBUG_LOGGING
        std::string build_log = jazzy::get_debug_log();
        if (!build_log.empty()) {
            RC_ASSERT(build_log.find("JazzyIndex::build") != std::string::npos);
        }
#endif

        RC_ASSERT(index.size() == values.size());
    });
    EXPECT_TRUE(result);
}

// Property: Empty index always returns end
TEST(PropertyTests, EmptyIndexReturnsEnd) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    const bool result = rc::check("Empty index returns end", [](value_type query) {
        std::vector<value_type> empty;
        qi_index index;
        index.build(empty.data(), empty.data());

#ifdef JAZZY_DEBUG_LOGGING
        std::string build_log = jazzy::get_debug_log();
        if (!build_log.empty()) {
            RC_ASSERT(build_log.find("JazzyIndex::build") != std::string::npos);
        }
#endif

        RC_ASSERT(index.find(query) == empty.data());
    });
    EXPECT_TRUE(result);
}

// Property: Single element behavior
TEST(PropertyTests, SingleElementBehavior) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    const bool result = rc::check("Single element behavior", [](value_type val, value_type query) {
        std::vector<value_type> data{val};
        qi_index index;
        index.build(data.data(), data.data() + data.size());

#ifdef JAZZY_DEBUG_LOGGING
        std::string build_log = jazzy::get_debug_log();
        if (!build_log.empty()) {
            RC_ASSERT(build_log.find("JazzyIndex::build") != std::string::npos);
        }
#endif

        const value_type* result = index.find(query);
        if (query == val) {
            RC_ASSERT(result != data.data() + data.size());
            RC_ASSERT(*result == val);
        } else {
            RC_ASSERT(result == data.data() + data.size());
        }
    });
    EXPECT_TRUE(result);
}

// Property: Min/max values are always findable
TEST(PropertyTests, MinMaxAlwaysFindable) {
#ifdef JAZZY_DEBUG_LOGGING
    jazzy::clear_debug_log();
#endif

    const bool result = rc::check("Min/max always findable", [](std::vector<value_type> values) {
        RC_PRE(!values.empty());
        std::sort(values.begin(), values.end());

        qi_index index;
        index.build(values.data(), values.data() + values.size());

#ifdef JAZZY_DEBUG_LOGGING
        std::string build_log = jazzy::get_debug_log();
        if (!build_log.empty()) {
            RC_ASSERT(build_log.find("JazzyIndex::build") != std::string::npos);
        }
#endif

        const value_type* min_result = index.find(values.front());
        RC_ASSERT(min_result != values.data() + values.size());
        RC_ASSERT(*min_result == values.front());

        const value_type* max_result = index.find(values.back());
        RC_ASSERT(max_result != values.data() + values.size());
        RC_ASSERT(*max_result == values.back());
    });
    EXPECT_TRUE(result);
}
