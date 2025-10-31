#include "quantile_index.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <rapidcheck.h>

namespace {

using qi_index = bucket_index::QuantileIndex<int, 256>;

}  // namespace

void run_property_tests() {
    const bool hits = rc::check("QuantileIndex finds all inserted integers",
                                [](std::vector<int> values) {
                                    RC_PRE(!values.empty());
                                    std::sort(values.begin(), values.end());

                                    qi_index index;
                                    index.build(values.data(), values.data() + values.size());

                                    for (int value : values) {
                                        const int* result = index.find(value);
                                        RC_ASSERT(result != values.data() + values.size());
                                        RC_ASSERT(*result == value);
                                    }
                                });

    if (!hits) {
        std::cerr << "[FAIL] property: QuantileIndex finds all inserted integers\n";
        std::exit(1);
    }
    std::cout << "[PASS] property: QuantileIndex finds all inserted integers\n";

    const bool misses =
        rc::check("QuantileIndex rejects values outside data range",
                  [](std::vector<int> values) {
                      RC_PRE(!values.empty());
                      std::sort(values.begin(), values.end());

                      qi_index index;
                      index.build(values.data(), values.data() + values.size());

                      const int* begin = values.data();
                      const int* end = begin + values.size();
                      RC_ASSERT(index.find(values.front() - 1) == end);
                      RC_ASSERT(index.find(values.back() + 1) == end);
                  });

    if (!misses) {
        std::cerr << "[FAIL] property: QuantileIndex rejects values outside data range\n";
        std::exit(1);
    }
    std::cout << "[PASS] property: QuantileIndex rejects values outside data range\n";
}

