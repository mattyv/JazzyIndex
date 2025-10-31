#include "quantile_index.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

void run_property_tests();

namespace {

template <typename T, std::size_t Segments = 256>
bucket_index::QuantileIndex<T, Segments> build_index(const std::vector<T>& data) {
    bucket_index::QuantileIndex<T, Segments> index;
    index.build(data.data(), data.data() + data.size());
    return index;
}

template <typename T, std::size_t Segments = 256>
bool expect_found(const bucket_index::QuantileIndex<T, Segments>& index,
                  const std::vector<T>& data,
                  const T& value) {
    const T* result = index.find(value);
    if (result == data.data() + data.size()) {
        return false;
    }
    return *result == value;
}

template <typename T, std::size_t Segments = 256>
bool expect_missing(const bucket_index::QuantileIndex<T, Segments>& index,
                    const std::vector<T>& data,
                    const T& value) {
    const T* result = index.find(value);
    return result == data.data() + data.size();
}

bool test_empty_index() {
    std::vector<int> data;
    bucket_index::QuantileIndex<int> index;
    index.build(data.data(), data.data());
    return expect_missing(index, data, 42);
}

bool test_single_element() {
    std::vector<int> data{42};
    auto index = build_index(data);
    return expect_found(index, data, 42) && expect_missing(index, data, 43);
}

bool test_uniform_sequence() {
    std::vector<int> data(1'000);
    std::iota(data.begin(), data.end(), 0);
    auto index = build_index<int, 256>(data);

    for (int value : {0, 10, 500, 999}) {
        if (!expect_found(index, data, value)) {
            return false;
        }
    }
    return expect_missing(index, data, -1) && expect_missing(index, data, 1'500);
}

bool test_duplicate_values() {
    std::vector<int> data{1, 1, 1, 2, 2, 5, 5, 9};
    auto index = build_index<int, 128>(data);
    for (int value : {1, 2, 5, 9}) {
        if (!expect_found(index, data, value)) {
            return false;
        }
    }
    return expect_missing(index, data, 3);
}

bool test_float_values() {
    std::vector<float> data{0.5f, 0.75f, 1.0f, 2.5f, 2.5f, 5.0f};
    auto index = build_index<float, 64>(data);
    return expect_found(index, data, 0.5f) &&
           expect_found(index, data, 2.5f) &&
           expect_missing(index, data, 3.0f);
}

struct test_case {
    std::string name;
    bool (*fn)();
};

}  // namespace

int main() {
    const test_case tests[] = {
        {"empty index returns end iterator", test_empty_index},
        {"single element hit/miss", test_single_element},
        {"uniform sequence lookups", test_uniform_sequence},
        {"duplicate values", test_duplicate_values},
        {"float values", test_float_values},
    };

    for (const auto& t : tests) {
        if (!t.fn()) {
            std::cerr << "[FAIL] " << t.name << '\n';
            return 1;
        }
        std::cout << "[PASS] " << t.name << '\n';
    }

    run_property_tests();

    std::cout << "All unit and property tests passed\n";
    return 0;
}
