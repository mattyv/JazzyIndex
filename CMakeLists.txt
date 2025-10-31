cmake_minimum_required(VERSION 3.20)
project(quantile_index LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

add_library(quantile_index INTERFACE)
target_include_directories(quantile_index INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

option(BUILD_TESTS "Build QuantileIndex unit and property tests" ON)
option(BUILD_BENCHMARKS "Build QuantileIndex benchmark suite" ON)

if(BUILD_TESTS OR BUILD_BENCHMARKS)
    include(FetchContent)
endif()

if(BUILD_TESTS)
    FetchContent_Declare(
        rapidcheck
        GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(rapidcheck)

    add_executable(quantile_index_tests
        tests/unit_tests.cpp
        tests/property_tests.cpp
    )
    target_link_libraries(quantile_index_tests PRIVATE quantile_index rapidcheck)
    target_include_directories(quantile_index_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )

    enable_testing()
    add_test(NAME quantile_index_tests COMMAND quantile_index_tests)
endif()

if(BUILD_BENCHMARKS)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)

    add_executable(quantile_index_benchmarks
        benchmarks/benchmark_main.cpp
    )
    target_link_libraries(quantile_index_benchmarks PRIVATE quantile_index benchmark::benchmark)
    target_include_directories(quantile_index_benchmarks PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
    )
endif()
