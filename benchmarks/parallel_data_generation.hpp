#pragma once

#include <future>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <iostream>

namespace qi::bench {

// Cache for pre-generated datasets
class DatasetCache {
public:
    using Dataset = std::vector<std::uint64_t>;
    using DatasetPtr = std::shared_ptr<Dataset>;

    // Singleton access
    static DatasetCache& instance() {
        static DatasetCache cache;
        return cache;
    }

    // Get or generate dataset
    template <typename Generator>
    DatasetPtr get(const std::string& name, std::size_t size, Generator&& gen) {
        const std::string key = name + "_" + std::to_string(size);

        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }

        // Generate and cache
        auto data = std::make_shared<Dataset>(gen(size));
        cache_[key] = data;
        return data;
    }

    // Pre-generate all datasets in parallel
    template <typename... Generators>
    void pre_generate_parallel(const std::vector<std::size_t>& sizes,
                               const std::vector<std::string>& names,
                               Generators&&... generators) {
        std::vector<std::future<void>> futures;

        // Launch parallel generation for each distribution/size combination
        size_t gen_idx = 0;
        auto gen_tuple = std::make_tuple(std::forward<Generators>(generators)...);

        ((void)([&](auto&& gen, const std::string& name) {
            for (std::size_t size : sizes) {
                futures.push_back(std::async(std::launch::async, [this, name, size, &gen]() {
                    const std::string key = name + "_" + std::to_string(size);
                    if (cache_.find(key) == cache_.end()) {
                        std::cout << "Generating " << name << " dataset (N=" << size << ")..." << std::endl;
                        auto data = std::make_shared<Dataset>(gen(size));
                        cache_[key] = data;
                    }
                }));
            }
        }(std::get<gen_idx++>(gen_tuple), names[gen_idx - 1])), ...);

        // Wait for all generations to complete
        std::cout << "Waiting for " << futures.size() << " dataset generations..." << std::endl;
        for (auto& f : futures) {
            f.wait();
        }
        std::cout << "Dataset generation complete!" << std::endl;
    }

    void clear() {
        cache_.clear();
    }

    size_t size() const {
        return cache_.size();
    }

private:
    DatasetCache() = default;
    std::unordered_map<std::string, DatasetPtr> cache_;
};

} // namespace qi::bench
