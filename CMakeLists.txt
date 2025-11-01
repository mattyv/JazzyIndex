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
    # Additional workarounds needed for clang-15 with gcc-14's libstdc++
    set(_GTEST_COMPAT_DEFS
        GTEST_INTERNAL_HAS_STD_FORMAT=0
        GTEST_HAS_STD_WSTRING=0
    )

    # Disable problematic features when using older clang with newer libstdc++
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "16.0")
        list(APPEND _GTEST_COMPAT_DEFS
            GTEST_INTERNAL_HAS_WSTRING=0
            GTEST_INTERNAL_HAS_RANGES=0
        )
    endif()

    foreach(_gtest_target IN ITEMS gtest gtest_main gmock gmock_main)
        if(TARGET ${_gtest_target})
            target_compile_definitions(${_gtest_target} PUBLIC ${_GTEST_COMPAT_DEFS})
            set_property(TARGET ${_gtest_target} PROPERTY CXX_STANDARD 17)
            set_property(TARGET ${_gtest_target} PROPERTY CXX_STANDARD_REQUIRED ON)
        endif()
    endforeach()
    # Ensure our tests see the same compatibility defines when including gtest headers

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
        tests/gtest_boundary_tests.cpp
        tests/gtest_floating_point_tests.cpp
        tests/gtest_comparator_tests.cpp
        tests/gtest_error_recovery_tests.cpp
        tests/gtest_uniformity_tests.cpp
        tests/gtest_property_tests.cpp
    )
    target_link_libraries(jazzy_index_tests PRIVATE
        jazzy_index
        GTest::gtest
        GTest::gtest_main
        rapidcheck
    )
    target_compile_definitions(jazzy_index_tests PRIVATE ${_GTEST_COMPAT_DEFS})
    set_property(TARGET jazzy_index_tests PROPERTY CXX_STANDARD 17)
    set_property(TARGET jazzy_index_tests PROPERTY CXX_STANDARD_REQUIRED ON)
    target_include_directories(jazzy_index_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )

    enable_testing()
    add_test(NAME jazzy_index_tests COMMAND jazzy_index_tests)

    # Coverage configuration (optional, enabled with -DCMAKE_BUILD_TYPE=Coverage)
    if(ENABLE_COVERAGE OR CMAKE_BUILD_TYPE STREQUAL "Coverage")
        target_compile_options(jazzy_index_tests PRIVATE --coverage -O0 -g)
        target_link_options(jazzy_index_tests PRIVATE --coverage)
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

    add_executable(jazzy_index_benchmarks
        benchmarks/benchmark_main.cpp
    )
    target_link_libraries(jazzy_index_benchmarks PRIVATE jazzy_index benchmark::benchmark)
    target_include_directories(jazzy_index_benchmarks PRIVATE
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
        add_custom_command(
            OUTPUT ${JAZZY_BENCHMARK_JSON}
            COMMAND $<TARGET_FILE:jazzy_index_benchmarks> --benchmark_format=json --benchmark_out=${JAZZY_BENCHMARK_JSON}
            DEPENDS jazzy_index_benchmarks
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running JazzyIndex benchmarks (JSON output)"
            VERBATIM
        )
        add_custom_target(jazzy_benchmark_json DEPENDS ${JAZZY_BENCHMARK_JSON})

        set(JAZZY_BENCHMARK_PNG ${CMAKE_BINARY_DIR}/jazzy_benchmarks.png)
        add_custom_command(
            OUTPUT ${JAZZY_BENCHMARK_PNG}
            COMMAND ${JAZZY_VENV_PYTHON} ${JAZZY_PLOT_SCRIPT} --input ${JAZZY_BENCHMARK_JSON} --output ${JAZZY_BENCHMARK_PNG}
            DEPENDS ${JAZZY_BENCHMARK_JSON} ${JAZZY_PLOT_SCRIPT} ${JAZZY_VENV_PYTHON}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating JazzyIndex benchmark plots"
            VERBATIM
        )
        add_custom_target(plot_benchmarks DEPENDS ${JAZZY_BENCHMARK_PNG})
        add_dependencies(plot_benchmarks jazzy_python_env jazzy_benchmark_json)
    else()
        message(STATUS "Python interpreter not found; benchmark plotting is disabled.")
    endif()
endif()
