#include "jazzy_index.hpp"

#include <iostream>
#include <vector>

int main() {
    std::vector<int> values{1, 3, 5, 7, 9};
    jazzy::JazzyIndex<int, jazzy::SegmentCount::SMALL> index(
        values.data(), values.data() + values.size());

    if (const int* result = index.find(7); result != index.end()) {
        std::cout << "found " << *result << '\n';
    }
}
