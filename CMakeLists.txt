cmake_minimum_required(VERSION 3.20)
project(jazzy_index LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
endif()

add_library(jazzy_index INTERFACE)
target_include_directories(jazzy_index INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

option(BUILD_TESTS "Build JazzyIndex unit and property tests" ON)
option(BUILD_BENCHMARKS "Build JazzyIndex benchmark suite" ON)
option(ENABLE_COVERAGE "Enable coverage instrumentation for tests" OFF)
option(ENABLE_DEBUG_LOGGING "Enable debug logging for tests" OFF)

# Benchmark configuration
set(BENCHMARK_REPETITIONS "1" CACHE STRING "Number of times to repeat each benchmark (default: 1)")
set(BENCHMARK_THREADS "0" CACHE STRING "Number of threads per benchmark (0 = single-threaded, default: 0)")

if(BUILD_TESTS OR BUILD_BENCHMARKS)
    include(FetchContent)
endif()

if(BUILD_TESTS)
    # Fetch Google Test
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # Work around libstdc++-14 incompatibilities with std::format in gtest
    # These workarounds are only needed on Linux with libstdc++ (gcc's standard library)
    # Windows uses Microsoft's STL which doesn't have these issues
    if(UNIX AND NOT APPLE)
        # On Linux, disable C++20 ranges/format features in GoogleTest
        # to avoid libstdc++-14 bugs with ranges::subrange and std::format
        set(_GTEST_COMPAT_DEFS
            GTEST_INTERNAL_HAS_STD_FORMAT=0
            GTEST_HAS_STD_WSTRING=0
            GTEST_INTERNAL_HAS_WSTRING=0
            GTEST_INTERNAL_HAS_RANGES=0
            GTEST_INTERNAL_HAS_STRING_VIEW=0
        )

        foreach(_gtest_target IN ITEMS gtest gtest_main gmock gmock_main)
            if(TARGET ${_gtest_target})
                target_compile_definitions(${_gtest_target} PUBLIC ${_GTEST_COMPAT_DEFS})
            endif()
        endforeach()
        # Ensure our tests see the same compatibility defines when including gtest headers
        set(_APPLY_GTEST_COMPAT_TO_TESTS ON)
    else()
        set(_APPLY_GTEST_COMPAT_TO_TESTS OFF)
    endif()

    # Ensure all gtest targets use C++20
    foreach(_gtest_target IN ITEMS gtest gtest_main gmock gmock_main)
        if(TARGET ${_gtest_target})
            set_property(TARGET ${_gtest_target} PROPERTY CXX_STANDARD 20)
            set_property(TARGET ${_gtest_target} PROPERTY CXX_STANDARD_REQUIRED ON)
        endif()
    endforeach()

    # Fetch RapidCheck for property-based testing
    FetchContent_Declare(
        rapidcheck
        GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(rapidcheck)

    # Fix RapidCheck compilation on Windows with sanitizers - add /EHsc for exception handling
    if(MSVC OR (WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
        if(TARGET rapidcheck)
            target_compile_options(rapidcheck PRIVATE /EHsc)
        endif()
    endif()

    # Google Test suite with comprehensive coverage
    add_executable(jazzy_index_tests
        tests/gtest_unit_tests.cpp
        tests/gtest_model_tests.cpp
        tests/gtest_model_selection_verification.cpp
        tests/gtest_boundary_tests.cpp
        tests/gtest_floating_point_tests.cpp
        tests/gtest_comparator_tests.cpp
        tests/gtest_error_recovery_tests.cpp
        tests/gtest_uniformity_tests.cpp
        tests/gtest_property_tests.cpp
        tests/gtest_key_value_tests.cpp
        tests/gtest_binary_search_tests.cpp
        tests/gtest_parallel_tests.cpp
        tests/gtest_equal_range_tests.cpp
        tests/gtest_iterator_tests.cpp
        tests/gtest_debug_logging_tests.cpp
    )
    target_link_libraries(jazzy_index_tests PRIVATE
        jazzy_index
        GTest::gtest
        GTest::gtest_main
        rapidcheck
    )
    # Only apply GoogleTest compatibility defines on Linux where libstdc++ bugs exist
    if(_APPLY_GTEST_COMPAT_TO_TESTS)
        target_compile_definitions(jazzy_index_tests PRIVATE ${_GTEST_COMPAT_DEFS})
    endif()
    set_property(TARGET jazzy_index_tests PROPERTY CXX_STANDARD 20)
    set_property(TARGET jazzy_index_tests PROPERTY CXX_STANDARD_REQUIRED ON)
    target_include_directories(jazzy_index_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )

    # Enable debug logging if requested
    if(ENABLE_DEBUG_LOGGING)
        target_compile_definitions(jazzy_index_tests PRIVATE JAZZY_DEBUG_LOGGING)
    endif()

    # Create a separate target for tests with debug logging always enabled
    # Usage: cmake --build build --target jazzy_index_tests_debug && ./build/jazzy_index_tests_debug
    # This target validates internal logging and provides visibility into JazzyIndex operations
    add_executable(jazzy_index_tests_debug EXCLUDE_FROM_ALL
        tests/gtest_unit_tests.cpp
        tests/gtest_model_tests.cpp
        tests/gtest_boundary_tests.cpp
        tests/gtest_floating_point_tests.cpp
        tests/gtest_comparator_tests.cpp
        tests/gtest_error_recovery_tests.cpp
        tests/gtest_uniformity_tests.cpp
        tests/gtest_property_tests.cpp
        tests/gtest_key_value_tests.cpp
        tests/gtest_binary_search_tests.cpp
        tests/gtest_parallel_tests.cpp
        tests/gtest_equal_range_tests.cpp
        tests/gtest_iterator_tests.cpp
        tests/gtest_debug_logging_tests.cpp
    )
    target_link_libraries(jazzy_index_tests_debug PRIVATE
        jazzy_index
        GTest::gtest
        GTest::gtest_main
        rapidcheck
    )
    if(DEFINED _GTEST_COMPAT_DEFS)
        target_compile_definitions(jazzy_index_tests_debug PRIVATE ${_GTEST_COMPAT_DEFS})
    endif()
    set_property(TARGET jazzy_index_tests_debug PROPERTY CXX_STANDARD 20)
    set_property(TARGET jazzy_index_tests_debug PROPERTY CXX_STANDARD_REQUIRED ON)
    target_include_directories(jazzy_index_tests_debug PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    # Always enable debug logging for this target
    target_compile_definitions(jazzy_index_tests_debug PRIVATE JAZZY_DEBUG_LOGGING)

    enable_testing()
    add_test(NAME jazzy_index_tests COMMAND jazzy_index_tests)

    # Coverage configuration (optional, enabled with -DCMAKE_BUILD_TYPE=Coverage)
    if(ENABLE_COVERAGE OR CMAKE_BUILD_TYPE STREQUAL "Coverage")
        target_compile_options(jazzy_index_tests PRIVATE --coverage -O0 -g)
        target_link_options(jazzy_index_tests PRIVATE --coverage)

        # Coverage report generation
        find_program(LCOV lcov)
        find_program(GENHTML genhtml)

        if(LCOV AND GENHTML)
            add_custom_target(coverage
                COMMAND ${CMAKE_COMMAND} -E make_directory coverage_html
                COMMAND ${LCOV} --capture --directory . --output-file coverage.info --ignore-errors mismatch
                COMMAND ${LCOV} --remove coverage.info '/usr/*' '*/googletest/*' '*/rapidcheck/*' '*/tests/*'
                        --output-file coverage_filtered.info --ignore-errors mismatch
                COMMAND ${GENHTML} coverage_filtered.info --output-directory coverage_html
                COMMAND ${CMAKE_COMMAND} -E echo "Coverage report generated: file://${CMAKE_BINARY_DIR}/coverage_html/index.html"
                COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/docs/coverage
                COMMAND ${CMAKE_COMMAND} -E copy_directory coverage_html ${CMAKE_SOURCE_DIR}/docs/coverage
                COMMAND ${CMAKE_COMMAND} -E echo "Coverage report copied to docs/coverage/index.html"
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                DEPENDS jazzy_index_tests
                COMMENT "Generating code coverage report"
            )
        else()
            message(STATUS "lcov and/or genhtml not found; coverage target is disabled.")
        endif()
    endif()
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

    # Workaround for Google Benchmark v1.8.3 switch warning on Windows
    # The benchmark library has an unhandled enum value in sysinfo.cc
    if(TARGET benchmark)
        target_compile_options(benchmark PRIVATE -Wno-error=switch)
    endif()
    if(TARGET benchmark_main)
        target_compile_options(benchmark_main PRIVATE -Wno-error=switch)
    endif()

    add_executable(jazzy_index_benchmarks
        benchmarks/benchmark_main.cpp
    )
    target_link_libraries(jazzy_index_benchmarks PRIVATE jazzy_index benchmark::benchmark)
    target_include_directories(jazzy_index_benchmarks PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
    )

    # Worst-case key-value benchmark
    add_executable(benchmark_key_value_worst_case
        benchmarks/benchmark_key_value_worst_case.cpp
    )
    target_link_libraries(benchmark_key_value_worst_case PRIVATE jazzy_index benchmark::benchmark)
    target_include_directories(benchmark_key_value_worst_case PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
    )

    # Range functions benchmark (equal_range, lower_bound, upper_bound)
    add_executable(benchmark_range_functions
        benchmarks/benchmark_range_functions.cpp
    )
    target_link_libraries(benchmark_range_functions PRIVATE jazzy_index benchmark::benchmark)
    target_include_directories(benchmark_range_functions PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
    )

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        set(JAZZY_VENV_DIR ${CMAKE_BINARY_DIR}/jazzy_venv)
        set(JAZZY_REQUIREMENTS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/requirements.txt)
        set(JAZZY_PLOT_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/scripts/plot_benchmarks.py)
        if(WIN32)
            set(JAZZY_VENV_PYTHON ${JAZZY_VENV_DIR}/Scripts/python.exe)
        else()
            set(JAZZY_VENV_PYTHON ${JAZZY_VENV_DIR}/bin/python)
        endif()

        add_custom_command(
            OUTPUT ${JAZZY_VENV_PYTHON}
            COMMAND ${Python3_EXECUTABLE} -m venv ${JAZZY_VENV_DIR}
            COMMAND ${JAZZY_VENV_PYTHON} -m pip install --upgrade pip
            COMMAND ${JAZZY_VENV_PYTHON} -m pip install -r ${JAZZY_REQUIREMENTS}
            DEPENDS ${JAZZY_REQUIREMENTS}
            COMMENT "Creating Python virtual environment for benchmark plotting"
            VERBATIM
        )
        add_custom_target(jazzy_python_env DEPENDS ${JAZZY_VENV_PYTHON})

        set(JAZZY_BENCHMARK_JSON ${CMAKE_BINARY_DIR}/jazzy_benchmarks.json)

        # Build benchmark command with optional repetitions and threads
        set(BENCH_CMD $<TARGET_FILE:jazzy_index_benchmarks> --benchmark_format=json --benchmark_out=${JAZZY_BENCHMARK_JSON})
        if(BENCHMARK_REPETITIONS GREATER 1)
            list(APPEND BENCH_CMD --benchmark_repetitions=${BENCHMARK_REPETITIONS} --benchmark_report_aggregates_only=true)
        endif()
        if(BENCHMARK_THREADS GREATER 0)
            list(APPEND BENCH_CMD --benchmark_threads=${BENCHMARK_THREADS})
        endif()

        add_custom_command(
            OUTPUT ${JAZZY_BENCHMARK_JSON}
            COMMAND ${BENCH_CMD}
            DEPENDS jazzy_index_benchmarks
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running JazzyIndex benchmarks (JSON output)"
            VERBATIM
        )
        add_custom_target(jazzy_benchmark_json DEPENDS ${JAZZY_BENCHMARK_JSON})

        set(JAZZY_DOCS_BENCHMARKS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/docs/images/benchmarks)
        set(JAZZY_BENCHMARK_PNG ${JAZZY_DOCS_BENCHMARKS_DIR}/jazzy_benchmarks.png)
        add_custom_command(
            OUTPUT ${JAZZY_BENCHMARK_PNG}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${JAZZY_DOCS_BENCHMARKS_DIR}
            COMMAND ${JAZZY_VENV_PYTHON} ${JAZZY_PLOT_SCRIPT} --input ${JAZZY_BENCHMARK_JSON} --output ${JAZZY_BENCHMARK_PNG}
            DEPENDS ${JAZZY_BENCHMARK_JSON} ${JAZZY_PLOT_SCRIPT} ${JAZZY_VENV_PYTHON}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating JazzyIndex benchmark plots in docs/images/benchmarks/"
            VERBATIM
        )
        add_custom_target(plot_benchmarks DEPENDS ${JAZZY_BENCHMARK_PNG})
        add_dependencies(plot_benchmarks jazzy_python_env jazzy_benchmark_json)

        # Full benchmark suite (up to 1M keys)
        set(JAZZY_BENCHMARK_FULL_JSON ${CMAKE_BINARY_DIR}/jazzy_benchmarks_full.json)

        # Build full benchmark command with optional repetitions and threads
        set(BENCH_FULL_CMD $<TARGET_FILE:jazzy_index_benchmarks> --full-benchmarks --benchmark_format=json --benchmark_out=${JAZZY_BENCHMARK_FULL_JSON})
        if(BENCHMARK_REPETITIONS GREATER 1)
            list(APPEND BENCH_FULL_CMD --benchmark_repetitions=${BENCHMARK_REPETITIONS} --benchmark_report_aggregates_only=true)
        endif()
        if(BENCHMARK_THREADS GREATER 0)
            list(APPEND BENCH_FULL_CMD --benchmark_threads=${BENCHMARK_THREADS})
        endif()

        add_custom_command(
            OUTPUT ${JAZZY_BENCHMARK_FULL_JSON}
            COMMAND ${BENCH_FULL_CMD}
            DEPENDS jazzy_index_benchmarks
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running JazzyIndex full benchmarks (up to 1M keys, JSON output)"
            VERBATIM
        )
        add_custom_target(jazzy_benchmark_full_json DEPENDS ${JAZZY_BENCHMARK_FULL_JSON})

        set(JAZZY_BENCHMARK_FULL_PNG ${JAZZY_DOCS_BENCHMARKS_DIR}/jazzy_benchmarks_full.png)
        add_custom_command(
            OUTPUT ${JAZZY_BENCHMARK_FULL_PNG}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${JAZZY_DOCS_BENCHMARKS_DIR}
            COMMAND ${JAZZY_VENV_PYTHON} ${JAZZY_PLOT_SCRIPT} --input ${JAZZY_BENCHMARK_FULL_JSON} --output ${JAZZY_BENCHMARK_FULL_PNG}
            DEPENDS ${JAZZY_BENCHMARK_FULL_JSON} ${JAZZY_PLOT_SCRIPT} ${JAZZY_VENV_PYTHON}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating JazzyIndex full benchmark plots in docs/images/benchmarks/"
            VERBATIM
        )
        add_custom_target(plot_benchmarks_full DEPENDS ${JAZZY_BENCHMARK_FULL_PNG})
        add_dependencies(plot_benchmarks_full jazzy_python_env jazzy_benchmark_full_json)

        # Range functions benchmarks (equal_range, lower_bound, upper_bound)
        set(JAZZY_RANGE_BENCHMARK_JSON ${CMAKE_BINARY_DIR}/jazzy_range_benchmarks.json)

        # Build range benchmark command with optional repetitions and threads
        set(BENCH_RANGE_CMD $<TARGET_FILE:benchmark_range_functions> --benchmark_format=json --benchmark_out=${JAZZY_RANGE_BENCHMARK_JSON})
        if(BENCHMARK_REPETITIONS GREATER 1)
            list(APPEND BENCH_RANGE_CMD --benchmark_repetitions=${BENCHMARK_REPETITIONS} --benchmark_report_aggregates_only=true)
        endif()
        if(BENCHMARK_THREADS GREATER 0)
            list(APPEND BENCH_RANGE_CMD --benchmark_threads=${BENCHMARK_THREADS})
        endif()

        add_custom_command(
            OUTPUT ${JAZZY_RANGE_BENCHMARK_JSON}
            COMMAND ${BENCH_RANGE_CMD}
            DEPENDS benchmark_range_functions
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running JazzyIndex range functions benchmarks (JSON output)"
            VERBATIM
        )
        add_custom_target(jazzy_range_benchmark_json DEPENDS ${JAZZY_RANGE_BENCHMARK_JSON})

        set(JAZZY_RANGE_BENCHMARK_PNG ${JAZZY_DOCS_BENCHMARKS_DIR}/jazzy_range_benchmarks.png)
        add_custom_command(
            OUTPUT ${JAZZY_RANGE_BENCHMARK_PNG}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${JAZZY_DOCS_BENCHMARKS_DIR}
            COMMAND ${JAZZY_VENV_PYTHON} ${JAZZY_PLOT_SCRIPT} --input ${JAZZY_RANGE_BENCHMARK_JSON} --output ${JAZZY_RANGE_BENCHMARK_PNG}
            DEPENDS ${JAZZY_RANGE_BENCHMARK_JSON} ${JAZZY_PLOT_SCRIPT} ${JAZZY_VENV_PYTHON}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating JazzyIndex range functions benchmark plots in docs/images/benchmarks/"
            VERBATIM
        )
        add_custom_target(plot_range_benchmarks DEPENDS ${JAZZY_RANGE_BENCHMARK_PNG})
        add_dependencies(plot_range_benchmarks jazzy_python_env jazzy_range_benchmark_json)

        # Index visualization target (outputs to committed docs/images/index_data/)
        set(JAZZY_INDEX_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/docs/images/index_data)
        set(JAZZY_PLOT_INDEX_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/scripts/plot_index_structure.py)
        add_custom_target(visualize_index
            COMMAND ${CMAKE_COMMAND} -E make_directory ${JAZZY_INDEX_DATA_DIR}
            COMMAND $<TARGET_FILE:jazzy_index_benchmarks> --visualize-index
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/index_data ${JAZZY_INDEX_DATA_DIR}
            COMMAND ${JAZZY_VENV_PYTHON} ${JAZZY_PLOT_INDEX_SCRIPT} ${JAZZY_INDEX_DATA_DIR}
            DEPENDS jazzy_index_benchmarks ${JAZZY_VENV_PYTHON} ${JAZZY_PLOT_INDEX_SCRIPT}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating index structure visualizations in docs/images/index_data/"
            VERBATIM
        )
        add_dependencies(visualize_index jazzy_python_env)

        # Documentation plots target (generates plots into committed docs/images/ directory)
        set(JAZZY_DOCS_PLOT_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_docs_plots.sh)
        add_custom_target(generate_docs_plots
            COMMAND ${JAZZY_DOCS_PLOT_SCRIPT}
            DEPENDS jazzy_index_benchmarks
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Generating documentation plots into docs/images/"
            VERBATIM
        )
        add_dependencies(generate_docs_plots jazzy_python_env)
    else()
        message(STATUS "Python interpreter not found; benchmark plotting is disabled.")
    endif()
endif()
