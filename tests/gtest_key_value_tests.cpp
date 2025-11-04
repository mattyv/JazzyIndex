#include <gtest/gtest.h>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "jazzy_index.hpp"

// Simple key-value struct for testing
struct KeyValue {
    int key;
    std::string value;

    bool operator<(const KeyValue& other) const {
        return key < other.key;
    }

    bool operator>(const KeyValue& other) const {
        return key > other.key;
    }

    bool operator==(const KeyValue& other) const {
        return key == other.key;
    }

    bool operator!=(const KeyValue& other) const {
        return !(*this == other);
    }
};

// Test with key-value struct
TEST(KeyValueTests, KeyValueStruct) {
    std::vector<KeyValue> data{
        {1, "apple"},
        {2, "banana"},
        {5, "cherry"},
        {8, "date"},
        {12, "elderberry"},
        {15, "fig"},
        {20, "grape"},
        {25, "honeydew"}
    };

    // Build index with key extractor
    jazzy::JazzyIndex<KeyValue, jazzy::SegmentCount::SMALL, std::less<>, decltype(&KeyValue::key)> index;
    index.build(data.data(), data.data() + data.size(), std::less<>{}, &KeyValue::key);

    EXPECT_EQ(index.size(), data.size());
    EXPECT_GT(index.num_segments(), 0u);

    // Test finding by key
    auto* result = index.find({2, ""});
    ASSERT_NE(result, data.data() + data.size());
    EXPECT_EQ(result->key, 2);
    EXPECT_EQ(result->value, "banana");

    result = index.find({15, ""});
    ASSERT_NE(result, data.data() + data.size());
    EXPECT_EQ(result->key, 15);
    EXPECT_EQ(result->value, "fig");

    // Test not found
    result = index.find({3, ""});
    EXPECT_EQ(result, data.data() + data.size());
}

// Test with custom struct
struct Person {
    int id;
    std::string name;
    int age;

    bool operator<(const Person& other) const {
        return id < other.id;
    }

    bool operator==(const Person& other) const {
        return id == other.id;
    }

    bool operator!=(const Person& other) const {
        return !(*this == other);
    }
};

TEST(KeyValueTests, CustomStruct) {
    std::vector<Person> people{
        {101, "Alice", 30},
        {102, "Bob", 25},
        {103, "Charlie", 35},
        {104, "David", 28},
        {105, "Eve", 32}
    };

    // Build index with member pointer to extract id
    jazzy::JazzyIndex<Person, jazzy::SegmentCount::TINY, std::less<>, decltype(&Person::id)> index;
    index.build(people.data(), people.data() + people.size(), std::less<>{}, &Person::id);

    EXPECT_EQ(index.size(), people.size());

    // Find by id
    auto* result = index.find(Person{102, "", 0});
    ASSERT_NE(result, people.data() + people.size());
    EXPECT_EQ(result->id, 102);
    EXPECT_EQ(result->name, "Bob");
    EXPECT_EQ(result->age, 25);

    result = index.find(Person{105, "", 0});
    ASSERT_NE(result, people.data() + people.size());
    EXPECT_EQ(result->name, "Eve");
}

// Test with member function pointer
struct Record {
    int key;
    double value;

    int get_key() const { return key; }

    bool operator<(const Record& other) const {
        return key < other.key;
    }

    bool operator==(const Record& other) const {
        return key == other.key && value == other.value;
    }

    bool operator!=(const Record& other) const {
        return !(*this == other);
    }
};

TEST(KeyValueTests, MemberFunctionPointer) {
    std::vector<Record> records{
        {10, 1.5},
        {20, 2.5},
        {30, 3.5},
        {40, 4.5},
        {50, 5.5}
    };

    // Using member data pointer
    jazzy::JazzyIndex<Record, jazzy::SegmentCount::TINY, std::less<>, decltype(&Record::key)> index;
    index.build(records.data(), records.data() + records.size(), std::less<>{}, &Record::key);

    auto* result = index.find(Record{30, 0.0});
    ASSERT_NE(result, records.data() + records.size());
    EXPECT_EQ(result->key, 30);
    EXPECT_DOUBLE_EQ(result->value, 3.5);
}

// Test backward compatibility - plain types should still work
TEST(KeyValueTests, BackwardCompatibilityPlainTypes) {
    std::vector<int> data{1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50};

    // Old API - no key extractor needed
    jazzy::JazzyIndex<int, jazzy::SegmentCount::SMALL> index;
    index.build(data.data(), data.data() + data.size());

    EXPECT_EQ(index.size(), data.size());

    auto* result = index.find(20);
    ASSERT_NE(result, data.data() + data.size());
    EXPECT_EQ(*result, 20);
}

// Test with large dataset
TEST(KeyValueTests, LargeDatasetKeyValue) {
    std::vector<KeyValue> data;
    for (int i = 0; i < 10000; ++i) {
        data.push_back({i, "value_" + std::to_string(i)});
    }

    jazzy::JazzyIndex<KeyValue, jazzy::SegmentCount::LARGE, std::less<>, decltype(&KeyValue::key)> index;
    index.build(data.data(), data.data() + data.size(), std::less<>{}, &KeyValue::key);

    EXPECT_EQ(index.size(), data.size());

    // Test several lookups
    for (int i : {0, 100, 500, 1000, 5000, 9999}) {
        auto* result = index.find({i, ""});
        ASSERT_NE(result, data.data() + data.size()) << "Failed to find key: " << i;
        EXPECT_EQ(result->key, i);
        EXPECT_EQ(result->value, "value_" + std::to_string(i));
    }
}

// Test with reverse order (std::greater)
TEST(KeyValueTests, ReverseOrderKeyValue) {
    std::vector<KeyValue> data{
        {50, "fifty"},
        {40, "forty"},
        {30, "thirty"},
        {20, "twenty"},
        {10, "ten"}
    };

    jazzy::JazzyIndex<KeyValue, jazzy::SegmentCount::TINY, std::greater<>, decltype(&KeyValue::key)> index;
    index.build(data.data(), data.data() + data.size(), std::greater<>{}, &KeyValue::key);

    auto* result = index.find({30, ""});
    ASSERT_NE(result, data.data() + data.size());
    EXPECT_EQ(result->key, 30);
    EXPECT_EQ(result->value, "thirty");
}

// Test edge cases
TEST(KeyValueTests, EdgeCasesSingleElement) {
    std::vector<KeyValue> data{{42, "answer"}};

    jazzy::JazzyIndex<KeyValue, jazzy::SegmentCount::TINY, std::less<>, decltype(&KeyValue::key)> index;
    index.build(data.data(), data.data() + data.size(), std::less<>{}, &KeyValue::key);

    auto* result = index.find({42, ""});
    ASSERT_NE(result, data.data() + data.size());
    EXPECT_EQ(result->key, 42);
    EXPECT_EQ(result->value, "answer");

    result = index.find({99, ""});
    EXPECT_EQ(result, data.data() + data.size());
}

// Test with duplicate keys (should find first occurrence)
TEST(KeyValueTests, DuplicateKeys) {
    std::vector<KeyValue> data{
        {1, "first_1"},
        {2, "first_2"},
        {2, "second_2"},
        {3, "first_3"},
        {3, "second_3"},
        {3, "third_3"}
    };

    jazzy::JazzyIndex<KeyValue, jazzy::SegmentCount::TINY, std::less<>, decltype(&KeyValue::key)> index;
    index.build(data.data(), data.data() + data.size(), std::less<>{}, &KeyValue::key);

    auto* result = index.find({2, ""});
    ASSERT_NE(result, data.data() + data.size());
    EXPECT_EQ(result->key, 2);
    // Should find one of the "2" entries (implementation-dependent which one)
    EXPECT_TRUE(result->value == "first_2" || result->value == "second_2");
}

// Test correctness against std::lower_bound
TEST(KeyValueTests, CorrectnessAgainstStdLowerBound) {
    std::vector<KeyValue> data;
    for (int i = 0; i < 1000; ++i) {
        data.push_back({i * 3, "value_" + std::to_string(i * 3)});  // 0, 3, 6, 9, ...
    }

    jazzy::JazzyIndex<KeyValue, jazzy::SegmentCount::MEDIUM, std::less<>, decltype(&KeyValue::key)> index;
    index.build(data.data(), data.data() + data.size(), std::less<>{}, &KeyValue::key);

    // Test lookups and compare with std::lower_bound
    for (int test_key : {0, 3, 6, 99, 150, 300, 999, 2997}) {
        auto* jazzy_result = index.find({test_key, ""});

        // Manual lower_bound with custom comparator
        auto lb = std::lower_bound(data.begin(), data.end(), test_key,
            [](const KeyValue& kv, int key) {
                return kv.key < key;
            });

        if (lb != data.end() && lb->key == test_key) {
            // Should be found
            ASSERT_NE(jazzy_result, data.data() + data.size()) << "JazzyIndex failed to find key: " << test_key;
            EXPECT_EQ(jazzy_result->key, lb->key);
            EXPECT_EQ(jazzy_result->value, lb->value);
        } else {
            // Should not be found
            EXPECT_EQ(jazzy_result, data.data() + data.size()) << "JazzyIndex incorrectly found key: " << test_key;
        }
    }
}
