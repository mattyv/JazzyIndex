#include "jazzy_index.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include <rapidcheck.h>

namespace {

using value_type = std::uint64_t;
using qi_index = jazzy::JazzyIndex<value_type, 256>;

}  // namespace

void run_property_tests() {
    const bool hits = rc::check("JazzyIndex finds all inserted integers",
                                [](std::vector<value_type> values) {
                                    RC_PRE(!values.empty());
                                    std::sort(values.begin(), values.end());

                                    qi_index index;
                                    index.build(values.data(), values.data() + values.size());

                                    for (value_type value : values) {
                                        const value_type* result = index.find(value);
                                        RC_ASSERT(result != values.data() + values.size());
                                        RC_ASSERT(*result == value);
                                    }
                                });

    if (!hits) {
        std::cerr << "[FAIL] property: JazzyIndex finds all inserted integers\n";
        std::exit(1);
    }
    std::cout << "[PASS] property: JazzyIndex finds all inserted integers\n";

    const bool misses =
        rc::check("JazzyIndex rejects values outside data range",
                  [](std::vector<value_type> values) {
                      RC_PRE(!values.empty());
                      std::sort(values.begin(), values.end());

                      qi_index index;
                      index.build(values.data(), values.data() + values.size());

                      const value_type* begin = values.data();
                      const value_type* end = begin + values.size();

                      if (values.front() > 0) {
                          RC_ASSERT(index.find(values.front() - 1) == end);
                      }
                      if (values.back() < std::numeric_limits<value_type>::max()) {
                          RC_ASSERT(index.find(values.back() + 1) == end);
                      }
                  });

    if (!misses) {
        std::cerr << "[FAIL] property: JazzyIndex rejects values outside data range\n";
        std::exit(1);
    }
    std::cout << "[PASS] property: JazzyIndex rejects values outside data range\n";
}
