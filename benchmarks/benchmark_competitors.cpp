#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <unordered_set>
#include <vector>

#include <benchmark/benchmark.h>
#include <pgm/pgm_index.hpp>
#include <rs/builder.h>

#include "fixtures.hpp"

namespace {

using Key = std::uint64_t;
using Data = std::vector<Key>;
constexpr std::size_t kQueryCount = 262'144;

bool use_full_benchmarks = false;

struct AllocationCounter {
    std::size_t live_bytes = 0;
};

template <class T>
struct CountingAllocator {
    using value_type = T;

    std::shared_ptr<AllocationCounter> counter = std::make_shared<AllocationCounter>();

    CountingAllocator() = default;
    explicit CountingAllocator(std::shared_ptr<AllocationCounter> value)
        : counter(std::move(value)) {}
    template <class U>
    CountingAllocator(const CountingAllocator<U>& other) : counter(other.counter) {}

    T* allocate(std::size_t count) {
        T* result = std::allocator<T>{}.allocate(count);
        counter->live_bytes += count * sizeof(T);
        return result;
    }
    void deallocate(T* pointer, std::size_t count) noexcept {
        counter->live_bytes -= count * sizeof(T);
        std::allocator<T>{}.deallocate(pointer, count);
    }
    template <class U>
    bool operator==(const CountingAllocator<U>& other) const noexcept {
        return counter == other.counter;
    }
    template <class U>
    bool operator!=(const CountingAllocator<U>& other) const noexcept {
        return !(*this == other);
    }
    template <class>
    friend struct CountingAllocator;
};

void verify_counting_allocator() {
    auto counter = std::make_shared<AllocationCounter>();
    {
        std::vector<Key, CountingAllocator<Key>> values{CountingAllocator<Key>{counter}};
        values.resize(3);
        if (counter->live_bytes != values.capacity() * sizeof(Key)) {
            throw std::logic_error("counting allocator did not report live allocation bytes");
        }
    }
    if (counter->live_bytes != 0) {
        throw std::logic_error("counting allocator retained deallocated bytes");
    }
}

Data make_unique_keys(Data values) {
    if (values.size() >= std::numeric_limits<Key>::max()) {
        throw std::overflow_error("dataset is too large for unique-key encoding");
    }
    const Key stride = static_cast<Key>(values.size()) + 1;
    Key previous = 0;
    Key occurrence = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const Key original = values[i];
        if (i != 0 && original < previous) {
            throw std::invalid_argument("unique-key encoding requires sorted input");
        }
        occurrence = i != 0 && original == previous ? occurrence + 1 : 0;
        if (original > (std::numeric_limits<Key>::max() - occurrence) / stride) {
            throw std::overflow_error("unique-key encoding exceeds uint64_t");
        }
        values[i] = original * stride + occurrence;
        previous = original;
    }
    return values;
}

std::size_t exact_lower_bound(const Data& data, Key key,
                              std::size_t first, std::size_t last) {
    if (first > last || last > data.size()) {
        throw std::out_of_range("learned index returned an invalid search bound");
    }
    const auto it = std::lower_bound(data.begin() + static_cast<std::ptrdiff_t>(first),
                                     data.begin() + static_cast<std::ptrdiff_t>(last), key);
    return it != data.begin() + static_cast<std::ptrdiff_t>(last) && *it == key
        ? static_cast<std::size_t>(it - data.begin())
        : data.size();
}

std::size_t exact_lower_bound(const Data& data, Key key) {
    return exact_lower_bound(data, key, 0, data.size());
}

template <typename Lookup>
void register_lookup(const std::string& implementation, const std::string& configuration,
                     const std::string& distribution, std::shared_ptr<const Data> data,
                     std::shared_ptr<const Data> queries, std::size_t index_bytes,
                     std::size_t backing_bytes,
                     Lookup lookup) {
    for (const Key query : *queries) {
        const std::size_t expected = exact_lower_bound(*data, query);
        const std::size_t actual = lookup(query);
        if (expected != actual) {
            throw std::logic_error(implementation + "/" + configuration +
                                   " returned an incorrect exact result for key " +
                                   std::to_string(query));
        }
    }

    std::string name = "Lookup/" + implementation;
    if (!configuration.empty()) {
        name += "/" + configuration;
    }
    name += "/" + distribution + "/N" + std::to_string(data->size()) + "/RandomMixed";
    const std::size_t size = data->size();

    benchmark::RegisterBenchmark(
        name.c_str(),
        [queries = std::move(queries), index_bytes, backing_bytes, size,
         lookup = std::move(lookup)](benchmark::State& state) {
            std::size_t checksum = 0, cursor = 0;
            for (auto _ : state) {
                checksum += lookup((*queries)[cursor]);
                if (++cursor == queries->size()) cursor = 0;
            }
            benchmark::DoNotOptimize(checksum);
            state.counters["index_bytes"] = static_cast<double>(index_bytes);
            state.counters["backing_bytes"] = static_cast<double>(backing_bytes);
            state.counters["total_bytes"] = static_cast<double>(index_bytes + backing_bytes);
            state.counters["size"] = static_cast<double>(size);
        })->Unit(benchmark::kNanosecond);
}

template <typename Contains>
void register_contains(const std::string& implementation, const std::string& distribution,
                       std::shared_ptr<const Data> data, std::shared_ptr<const Data> queries,
                       std::size_t index_bytes, Contains contains) {
    for (const Key query : *queries) {
        const bool expected = exact_lower_bound(*data, query) != data->size();
        if (expected != contains(query)) {
            throw std::logic_error(implementation + " returned incorrect membership for key " +
                                   std::to_string(query));
        }
    }
    const std::string name = "Membership/" + implementation + "/" + distribution + "/N" +
                             std::to_string(data->size()) + "/RandomMixed";
    const std::size_t size = data->size();
    benchmark::RegisterBenchmark(
        name.c_str(),
        [queries = std::move(queries), index_bytes, size,
         contains = std::move(contains)](benchmark::State& state) {
            std::size_t checksum = 0, cursor = 0;
            for (auto _ : state) {
                checksum += contains((*queries)[cursor]);
                if (++cursor == queries->size()) cursor = 0;
            }
            benchmark::DoNotOptimize(checksum);
            state.counters["index_bytes"] = static_cast<double>(index_bytes);
            state.counters["backing_bytes"] = 0;
            state.counters["total_bytes"] = static_cast<double>(index_bytes);
            state.counters["size"] = static_cast<double>(size);
        })->Unit(benchmark::kNanosecond);
}

template <typename Build>
void register_build(const std::string& implementation, const std::string& configuration,
                    const std::string& distribution, std::shared_ptr<const Data> data,
                    Build build) {
    const std::string name = "Build/" + implementation + "/" + configuration + "/" +
                             distribution + "/N" + std::to_string(data->size());
    benchmark::RegisterBenchmark(
        name.c_str(),
        [data = std::move(data), build = std::move(build)](benchmark::State& state) {
            for (auto _ : state) {
                {
                    auto index = build(*data);
                    benchmark::DoNotOptimize(index);
                    state.PauseTiming();
                }
                state.ResumeTiming();
            }
            state.counters["size"] = static_cast<double>(data->size());
        })->Unit(benchmark::kMicrosecond);
}

template <std::size_t Segments>
void register_jazzy(const std::string& distribution, std::shared_ptr<const Data> data,
                    std::shared_ptr<const Data> queries) {
    using Index = jazzy::JazzyIndex<Key, jazzy::to_segment_count<Segments>()>;
    auto index = std::make_shared<Index>();
    index->build(data->data(), data->data() + data->size());
    const std::string config = "S" + std::to_string(Segments);

    register_lookup("JazzyIndex", config, distribution, data, queries,
                    sizeof(Index) + index->size_in_bytes(), data->capacity() * sizeof(Key),
                    [index, data](Key key) {
                        const auto* result = index->find(key);
                        return result == index->end()
                            ? data->size()
                            : static_cast<std::size_t>(result - data->data());
                    });
    if constexpr (Segments == 256) {
        constexpr std::size_t kBatchWidth = 4;
        static_assert(kQueryCount % kBatchWidth == 0,
                      "Batch4 requires the deterministic query fixture to divide evenly");
        using BatchKeys = std::array<Key, kBatchWidth>;
        using BatchResults = std::array<typename Index::const_iterator, kBatchWidth>;
        for (std::size_t offset = 0; offset < queries->size(); offset += kBatchWidth) {
            BatchKeys keys{};
            BatchResults results{};
            for (std::size_t lane = 0; lane < kBatchWidth; ++lane) keys[lane] = (*queries)[offset + lane];
            index->find_batch(keys, results);
            for (std::size_t lane = 0; lane < kBatchWidth; ++lane) {
                const std::size_t expected = exact_lower_bound(*data, keys[lane]);
                const std::size_t actual = results[lane] == index->end()
                    ? data->size() : static_cast<std::size_t>(results[lane] - data->data());
                if (expected != actual) {
                    throw std::logic_error("JazzyIndex/S256/Batch4 returned an incorrect exact result");
                }
            }
        }
        const std::string batch_name = "Lookup/JazzyIndex/S256/Batch4/" + distribution + "/N" +
                                       std::to_string(data->size()) + "/RandomMixed";
        const std::size_t index_bytes = sizeof(Index) + index->size_in_bytes();
        const std::size_t backing_bytes = data->capacity() * sizeof(Key);
        benchmark::RegisterBenchmark(batch_name.c_str(),
            [index, data, queries, index_bytes, backing_bytes](benchmark::State& state) {
                std::size_t cursor = 0, checksum = 0;
                BatchKeys keys{};
                BatchResults results{};
                for (auto _ : state) {
                    for (std::size_t lane = 0; lane < kBatchWidth; ++lane) {
                        keys[lane] = (*queries)[cursor + lane];
                    }
                    index->find_batch(keys, results);
                    for (const auto result : results) {
                        checksum += result == index->end()
                            ? data->size() : static_cast<std::size_t>(result - data->data());
                    }
                    cursor += kBatchWidth;
                    if (cursor == queries->size()) cursor = 0;
                }
                benchmark::DoNotOptimize(checksum);
                state.SetItemsProcessed(state.iterations() * kBatchWidth);
                state.counters["batch_width"] = static_cast<double>(kBatchWidth);
                state.counters["index_bytes"] = static_cast<double>(index_bytes);
                state.counters["backing_bytes"] = static_cast<double>(backing_bytes);
                state.counters["total_bytes"] = static_cast<double>(index_bytes + backing_bytes);
                state.counters["size"] = static_cast<double>(data->size());
            })->Unit(benchmark::kNanosecond);
    }
    register_build("JazzyIndex", config, distribution, std::move(data),
                   [](const Data& values) {
                       Index result;
                       result.build(values.data(), values.data() + values.size());
                       return result;
                   });
}

template <std::size_t Epsilon>
void register_pgm(const std::string& distribution, std::shared_ptr<const Data> data,
                  std::shared_ptr<const Data> queries) {
    using Index = pgm::PGMIndex<Key, Epsilon>;
    auto index = std::make_shared<Index>(data->begin(), data->end());
    const std::string config = "E" + std::to_string(Epsilon);

    register_lookup("PGM", config, distribution, data, queries,
                    sizeof(Index) + index->size_in_bytes(), data->capacity() * sizeof(Key),
                    [index, data](Key key) {
                        const auto range = index->search(key);
                        return exact_lower_bound(*data, key, range.lo, range.hi);
                    });
    register_build("PGM", config, distribution, std::move(data),
                   [](const Data& values) {
                       return Index(values.begin(), values.end());
                   });
}

template <std::size_t RadixBits, std::size_t MaxError>
void register_radix_spline(const std::string& distribution,
                           std::shared_ptr<const Data> data,
                           std::shared_ptr<const Data> queries) {
    auto build = [](const Data& values) {
        rs::Builder<Key> builder(values.front(), values.back(), RadixBits, MaxError);
        for (const Key key : values) {
            builder.AddKey(key);
        }
        return builder.Finalize();
    };

    auto index = std::make_shared<rs::RadixSpline<Key>>(build(*data));
    const std::string config = "R" + std::to_string(RadixBits) +
                               "/E" + std::to_string(MaxError);
    register_lookup("RadixSpline", config, distribution, data, queries, index->GetSize(),
                    data->capacity() * sizeof(Key),
                    [index, data](Key key) {
                        const auto range = index->GetSearchBound(key);
                        return exact_lower_bound(*data, key, range.begin, range.end);
                    });
    register_build("RadixSpline", config, distribution, std::move(data), std::move(build));
}

void register_dataset(const std::string& distribution, Data values) {
    auto data = std::make_shared<const Data>(make_unique_keys(std::move(values)));
    auto queries = std::make_shared<const Data>(
        qi::bench::make_random_queries(*data, kQueryCount));

    register_lookup("LowerBound", "", distribution, data, queries, 0,
                    data->capacity() * sizeof(Key),
                    [data](Key key) { return exact_lower_bound(*data, key); });

    using Alloc = CountingAllocator<Key>;
    auto tree_counter = std::make_shared<AllocationCounter>();
    auto tree = std::make_shared<std::set<Key, std::less<Key>, Alloc>>(
        std::less<Key>{}, Alloc{tree_counter});
    tree->insert(data->begin(), data->end());
    register_contains("stdset", distribution, data, queries,
                      sizeof(*tree) + tree_counter->live_bytes,
                      [tree](Key key) { return tree->contains(key); });

    auto hash_counter = std::make_shared<AllocationCounter>();
    auto hash = std::make_shared<std::unordered_set<Key, std::hash<Key>, std::equal_to<Key>, Alloc>>(
        0, std::hash<Key>{}, std::equal_to<Key>{}, Alloc{hash_counter});
    hash->reserve(data->size());
    hash->insert(data->begin(), data->end());
    register_contains("stdunordered_set", distribution, data, queries,
                      sizeof(*hash) + hash_counter->live_bytes,
                      [hash](Key key) { return hash->contains(key); });

    register_jazzy<64>(distribution, data, queries);
    register_jazzy<256>(distribution, data, queries);
    register_jazzy<512>(distribution, data, queries);

    register_pgm<16>(distribution, data, queries);
    register_pgm<64>(distribution, data, queries);
    register_pgm<256>(distribution, data, queries);

    register_radix_spline<12, 16>(distribution, data, queries);
    register_radix_spline<12, 64>(distribution, data, queries);
    register_radix_spline<12, 256>(distribution, data, queries);
    register_radix_spline<18, 16>(distribution, data, queries);
    register_radix_spline<18, 64>(distribution, data, queries);
    register_radix_spline<18, 256>(distribution, data, queries);
}

void register_suites() {
    std::vector<std::size_t> sizes{10'000};
    if (use_full_benchmarks) {
        sizes.push_back(100'000);
        sizes.push_back(1'000'000);
    }

    for (const std::size_t size : sizes) {
        register_dataset("Uniform", qi::bench::make_uniform_values(size));
        register_dataset("Exponential", qi::bench::make_exponential_values(size));
        register_dataset("Clustered", qi::bench::make_clustered_values(size));
        register_dataset("Lognormal", qi::bench::make_lognormal_values(size));
        register_dataset("Zipf", qi::bench::make_zipf_values(size));
        register_dataset("Mixed", qi::bench::make_mixed_values(size));
        register_dataset("Quadratic", qi::bench::make_quadratic_values(size));
        register_dataset("ExtremePoly", qi::bench::make_extreme_polynomial_values(size));
        register_dataset("InversePoly", qi::bench::make_inverse_polynomial_values(size));
    }
}

void consume_custom_flags(int& argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) != "--full-benchmarks") {
            continue;
        }
        use_full_benchmarks = true;
        for (int j = i; j + 1 < argc; ++j) {
            argv[j] = argv[j + 1];
        }
        --argc;
        argv[argc] = nullptr;
        --i;
    }
}

}  // namespace

int main(int argc, char** argv) {
    consume_custom_flags(argc, argv);
    verify_counting_allocator();
    register_suites();
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
