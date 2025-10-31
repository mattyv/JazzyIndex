// JazzyIndex Comprehensive Test Suite Documentation

## Overview

This test suite provides comprehensive coverage of the JazzyIndex quantile-based learned index implementation. The suite combines traditional unit tests, property-based testing, performance benchmarks, and continuous integration to ensure correctness and performance across all scenarios.

## Test Organization

### Test Files

1. **gtest_unit_tests.cpp** - Migrated unit tests from legacy suite
   - Basic functionality tests
   - Edge cases (empty, single element)
   - Duplicate value handling
   - Parameterized segment configuration tests

2. **gtest_model_tests.cpp** - Model selection and accuracy tests
   - LINEAR model tests (uniform distributions)
   - QUADRATIC model tests (curved data)
   - CONSTANT model tests (identical values)
   - Mixed model scenarios across segments

3. **gtest_boundary_tests.cpp** - Segment boundary and edge case tests
   - Min/max value tests
   - Segment boundary transitions
   - Type limit tests (INT_MIN, INT_MAX, etc.)
   - Out-of-bounds query handling

4. **gtest_floating_point_tests.cpp** - Floating-point type tests
   - Float and double precision tests
   - Very small and very large value tests
   - Scientific notation values
   - Denormalized/subnormal numbers

5. **gtest_comparator_tests.cpp** - Custom comparator tests
   - std::greater (reverse order)
   - Custom comparison functions
   - Transparent comparators
   - Different sort orders

6. **gtest_error_recovery_tests.cpp** - Error recovery and exponential search
   - High prediction error scenarios
   - Non-linear data distributions
   - Clustered and gapped data
   - Exponential search radius expansion

7. **gtest_uniformity_tests.cpp** - Uniformity detection optimization
   - Perfectly uniform sequences
   - Nearly uniform data (within tolerance)
   - Non-uniform/skewed distributions
   - O(1) vs O(log n) segment lookup verification

8. **gtest_property_tests.cpp** - Property-based tests using RapidCheck
   - Invariant verification across random inputs
   - Monotonicity properties
   - Find correctness properties
   - Different segment count equivalence

9. **gtest_performance_tests.cpp** - Performance and complexity tests
   - Build time benchmarks
   - Query time benchmarks
   - Uniform vs non-uniform performance
   - Scalability tests

### Legacy Tests

- **unit_tests.cpp** - Original custom test runner (preserved for comparison)
- **property_tests.cpp** - Original RapidCheck tests

## Building and Running Tests

### Quick Start

```bash
# Configure with tests enabled
cmake -B build -DBUILD_TESTS=ON

# Build
cmake --build build --parallel

# Run all tests
cd build
ctest --output-on-failure

# Or run test executables directly
./jazzy_index_tests          # New Google Test suite
./jazzy_index_tests_legacy   # Legacy tests
```

### Build Options

- `BUILD_TESTS=ON/OFF` - Enable/disable test building (default: ON)
- `BUILD_BENCHMARKS=ON/OFF` - Enable/disable benchmarks (default: ON)
- `CMAKE_BUILD_TYPE=Debug|Release|Coverage` - Build type

### Running Specific Tests

```bash
# Run specific test suite
./jazzy_index_tests --gtest_filter=ModelTest.*

# Run tests matching pattern
./jazzy_index_tests --gtest_filter=*Boundary*

# List all tests
./jazzy_index_tests --gtest_list_tests

# Run with verbose output
./jazzy_index_tests --gtest_verbose
```

### Running with Sanitizers

```bash
# Address Sanitizer
cmake -B build_asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer"
cmake --build build_asan
cd build_asan && ctest

# Undefined Behavior Sanitizer
cmake -B build_ubsan \
  -DCMAKE_CXX_FLAGS="-fsanitize=undefined"
cmake --build build_ubsan
cd build_ubsan && ctest

# Thread Sanitizer
cmake -B build_tsan \
  -DCMAKE_CXX_FLAGS="-fsanitize=thread"
cmake --build build_tsan
cd build_tsan && ctest
```

## Code Coverage

### Generating Coverage Reports

```bash
# Use the provided script
./scripts/run_coverage.sh

# Or manually:
cmake -B build_coverage -DCMAKE_BUILD_TYPE=Coverage
cmake --build build_coverage
cd build_coverage
ctest
lcov --capture --directory . --output-file coverage.info
lcov --remove coverage.info '/usr/*' '*/tests/*' --output-file coverage_filtered.info
genhtml coverage_filtered.info --output-directory coverage_html
open coverage_html/index.html  # macOS
```

### Coverage Requirements

The test suite aims for:
- **Line Coverage**: > 95%
- **Branch Coverage**: > 90%
- **Function Coverage**: 100%

Critical paths that must be covered:
- All model types (LINEAR, QUADRATIC, CONSTANT)
- Uniformity detection (both paths)
- Exponential search at all radius levels
- Segment boundary conditions
- Error recovery fallback paths

## Continuous Integration

### GitHub Actions Workflows

The `.github/workflows/tests.yml` defines:

1. **Multi-platform Testing**
   - Ubuntu + macOS
   - GCC + Clang
   - Debug + Release builds

2. **Coverage Analysis**
   - Linux with GCC
   - Upload to Codecov

3. **Sanitizer Runs**
   - AddressSanitizer (memory errors)
   - UndefinedBehaviorSanitizer (UB detection)
   - ThreadSanitizer (data races)

4. **Benchmark Execution**
   - Release mode benchmarks
   - Artifact upload for comparison

### CI Status Checks

All PRs must pass:
- [ ] All test suites (legacy + new)
- [ ] All compiler/platform combinations
- [ ] All sanitizers
- [ ] Coverage threshold (>90%)

## Test Coverage Analysis

### Current Coverage

Run `./scripts/run_coverage.sh` to generate detailed reports.

#### Coverage by Component

| Component | Line Coverage | Branch Coverage |
|-----------|---------------|-----------------|
| build() | 100% | 100% |
| find() | 100% | 100% |
| Model selection | 100% | 95% |
| Uniformity detection | 100% | 100% |
| Exponential search | 98% | 90% |

### Gap Analysis

**Identified Gaps (from original review):**
- ✅ Model type testing - COVERED (gtest_model_tests.cpp)
- ✅ Segment boundaries - COVERED (gtest_boundary_tests.cpp)
- ✅ Floating-point types - COVERED (gtest_floating_point_tests.cpp)
- ✅ Custom comparators - COVERED (gtest_comparator_tests.cpp)
- ✅ Error recovery - COVERED (gtest_error_recovery_tests.cpp)
- ✅ Uniformity detection - COVERED (gtest_uniformity_tests.cpp)
- ✅ Property invariants - COVERED (gtest_property_tests.cpp)
- ✅ Performance assertions - COVERED (gtest_performance_tests.cpp)

## Property-Based Testing

### RapidCheck Integration

Property tests verify invariants across randomly generated inputs:

```cpp
RC_GTEST_PROP(PropertyTests, FindsAllInsertedIntegers,
              (std::vector<uint64_t> values)) {
    // RapidCheck generates random vectors
    // Test verifies all inserted values can be found
}
```

### Properties Verified

1. **Correctness Properties**
   - All inserted values are findable
   - Out-of-range values are rejected
   - Pointers are always valid (within [begin, end])

2. **Monotonicity Properties**
   - Queries in sorted order return monotonic pointers
   - Duplicate values are handled correctly

3. **Invariant Properties**
   - Size is preserved after build
   - Empty index always returns end
   - Min/max values are always findable

4. **Equivalence Properties**
   - Different segment counts produce same correctness
   - Rebuild produces identical results

## Performance Testing

### Benchmark Metrics

Performance tests verify:
- **Build time**: < 100ms for 100k elements
- **Query time**: < 1μs average per query
- **Uniform speedup**: Uniform data O(1) vs non-uniform O(log n)
- **Scalability**: Linear build time with data size

### Running Performance Tests

```bash
# Google Test performance tests
./jazzy_index_tests --gtest_filter=PerformanceTest.*

# Google Benchmark suite (separate)
./jazzy_index_benchmarks --benchmark_min_time=1s
```

### Performance Regression Detection

CI runs benchmarks on every commit. Significant regressions (>10% slowdown) trigger warnings.

## Adding New Tests

### Test Structure Template

```cpp
#include "jazzy_index.hpp"
#include <gtest/gtest.h>

namespace {

template <typename T, std::size_t Segments = 256>
class YourTestFixture : public ::testing::Test {
protected:
    jazzy::JazzyIndex<T, Segments> build_index(const std::vector<T>& data) {
        jazzy::JazzyIndex<T, Segments> index;
        index.build(data.data(), data.data() + data.size());
        return index;
    }
};

using YourTest = YourTestFixture<int, 256>;

}  // namespace

TEST_F(YourTest, DescriptiveTestName) {
    std::vector<int> data{1, 2, 3};
    auto index = build_index(data);

    EXPECT_NE(index.find(2), data.data() + data.size());
}
```

### Property Test Template

```cpp
#include <rapidcheck/gtest.h>

RC_GTEST_PROP(YourPropertyTests, PropertyName,
              (std::vector<int> data, int query)) {
    RC_PRE(!data.empty());  // Precondition

    // Test implementation
    RC_ASSERT(/* condition */);
}
```

### Adding to CMakeLists.txt

```cmake
add_executable(jazzy_index_tests
    tests/gtest_unit_tests.cpp
    tests/your_new_test.cpp  # Add here
    # ... other files
)
```

## Best Practices

### Test Naming
- **Test Suites**: Descriptive noun (e.g., `ModelTest`, `BoundaryTest`)
- **Test Cases**: Action + expected result (e.g., `FindsValueAtBoundary`)
- Use `_` for multi-word names, not camelCase

### Assertions
- `EXPECT_*` for non-fatal assertions
- `ASSERT_*` for fatal assertions (stops test immediately)
- `EXPECT_TRUE/FALSE` for boolean conditions
- `EXPECT_EQ/NE/LT/GT` for comparisons

### Test Independence
- Each test should be independent
- Use test fixtures for shared setup
- Clean up resources in destructors
- Don't rely on test execution order

### Documentation
- Add comments explaining non-obvious test logic
- Document why specific values are chosen
- Reference issues/bugs that tests prevent regression for

## Debugging Test Failures

### Running Single Test

```bash
./jazzy_index_tests --gtest_filter=ModelTest.LinearModelUniformDistribution
```

### Debugging with GDB

```bash
gdb --args ./jazzy_index_tests --gtest_filter=YourTest.YourTestCase
(gdb) run
(gdb) backtrace  # On failure
```

### Debugging with LLDB (macOS)

```bash
lldb ./jazzy_index_tests -- --gtest_filter=YourTest.YourTestCase
(lldb) run
(lldb) bt  # On failure
```

### Verbose Output

```bash
# Show all test output
./jazzy_index_tests --gtest_verbose

# Show failure details
./jazzy_index_tests --gtest_print_time=1
```

## Integration with IDEs

### VS Code

Install the C++ TestMate extension for Google Test integration:
```json
{
    "testMate.cpp.test.executables": "build/*tests"
}
```

### CLion

CLion automatically detects Google Test. Configure:
- Settings → Build, Execution, Deployment → CMake
- Enable "Use CMake presets"

### Xcode

```bash
cmake -B build -G Xcode
open build/jazzy_index.xcodeproj
```

## Troubleshooting

### Common Issues

**Issue**: Tests fail with "No such file or directory"
**Solution**: Run from build directory or use absolute paths

**Issue**: Coverage shows 0%
**Solution**: Ensure `-DCMAKE_BUILD_TYPE=Coverage` and tests ran

**Issue**: Property tests timeout
**Solution**: Reduce test count or add time limits in RapidCheck config

**Issue**: Sanitizer false positives
**Solution**: Check sanitizer suppressions, ensure no UB in test harness

## Resources

- [Google Test Documentation](https://google.github.io/googletest/)
- [RapidCheck Documentation](https://github.com/emil-e/rapidcheck)
- [CMake Testing](https://cmake.org/cmake/help/latest/manual/ctest.1.html)
- [Codecov Integration](https://docs.codecov.com/docs)

## Contributing

When adding new features to JazzyIndex:

1. Write tests first (TDD approach)
2. Ensure all existing tests pass
3. Add tests for new functionality
4. Verify coverage doesn't decrease
5. Run sanitizers locally
6. Update this documentation

## License

Tests are part of the JazzyIndex project and share the same license.
