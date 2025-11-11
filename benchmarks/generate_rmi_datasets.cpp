// Helper program to generate synthetic datasets in RMI binary format
// This allows RMI to be benchmarked against the same synthetic distributions as JazzyIndex

#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define mkdir(dir, mode) _mkdir(dir)
#endif

#include "fixtures.hpp"
#include "dataset_generators.hpp"

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::size_t size = 10000;  // Default size
    if (argc > 1) {
        size = std::stoull(argv[1]);
    }

    std::cout << "Generating synthetic datasets for RMI (N=" << size << ")..." << std::endl;

    // Create output directory
    mkdir("benchmarks/datasets", 0755);

    struct Distribution {
        std::string name;
        std::function<std::vector<std::uint64_t>(std::size_t)> generator;
    };

    std::vector<Distribution> distributions = {
        {"uniform", [](std::size_t s) { return qi::bench::make_uniform_values(s); }},
        {"exponential", [](std::size_t s) { return qi::bench::make_exponential_values(s); }},
        {"clustered", [](std::size_t s) { return qi::bench::make_clustered_values(s); }},
        {"lognormal", [](std::size_t s) { return qi::bench::make_lognormal_values(s); }},
        {"zipf", [](std::size_t s) { return qi::bench::make_zipf_values(s); }},
        {"mixed", [](std::size_t s) { return qi::bench::make_mixed_values(s); }},
        {"quadratic", [](std::size_t s) { return qi::bench::make_quadratic_values(s); }},
        {"extreme_poly", [](std::size_t s) { return qi::bench::make_extreme_polynomial_values(s); }},
        {"inverse_poly", [](std::size_t s) { return qi::bench::make_inverse_polynomial_values(s); }}
    };

    for (const auto& [name, gen] : distributions) {
        std::string filename = "benchmarks/datasets/" + name + "_" + std::to_string(size) + "_uint64";
        std::cout << "  Generating " << filename << "..." << std::flush;

        auto data = gen(size);
        if (qi::bench::write_rmi_dataset(filename, data)) {
            std::cout << " OK (" << data.size() << " elements)" << std::endl;
        } else {
            std::cerr << " FAILED" << std::endl;
        }
    }

    std::cout << "\nGenerated datasets can be used with RMI compiler:" << std::endl;
    std::cout << "  cd external/RMI" << std::endl;
    std::cout << "  cargo run --release -- ../../benchmarks/datasets/uniform_" << size
              << "_uint64 uniform_rmi linear,linear 100" << std::endl;
    std::cout << "\nThen move generated files to benchmarks/rmi_generated/" << std::endl;

    return 0;
}
