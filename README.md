# Quantile Index

A standalone extraction of the `QuantileIndex` adaptive learned index first prototyped in the
`early-exit-find` project. The implementation partitions a sorted array into quantile-sized segments
and fits an optimal model per segment (constant, linear, or quadratic) to deliver fast lookup
predictions with graceful fallback to binary search for skewed data.

## Features

- Header-only library (`include/quantile_index.hpp`) with minimal dependencies.
- Comprehensive unit tests plus RapidCheck property tests for correctness.
- Benchmarks covering uniform and skewed input distributions (exponential, clustered, lognormal, Zipf, mixed).
- Dataset generators extracted from the original project for repeatable experiments.

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Targets are optional and can be toggled with `-DBUILD_TESTS=ON|OFF` and `-DBUILD_BENCHMARKS=ON|OFF`.
Release builds default to `-O3 -march=native -DNDEBUG`.

## Running Tests

```bash
cmake --build build --target quantile_index_tests
ctest --test-dir build
```

The test binary prints the status of individual unit tests as well as RapidCheck property checks.

## Running Benchmarks

```bash
cmake --build build --target quantile_index_benchmarks
./build/quantile_index_benchmarks --benchmark_format=console
```

Benchmarks are organised by dataset type (uniform, exponential, etc.), segment count, and query
pattern (found, miss, random).

## Project Layout

```
include/
  quantile_index.hpp            # Core index
  quantile_index_utility.hpp    # Arithmetic trait & clamp helper
  dataset_generators.hpp        # Distribution generators for tests & benchmarks
benchmarks/
  fixtures.hpp                  # Data builders shared across benchmarks
  benchmark_main.cpp            # Google Benchmark suite
tests/
  unit_tests.cpp                # Deterministic correctness checks
  property_tests.cpp            # RapidCheck property-based tests
```
