#!/bin/bash
# Script to build tests with coverage instrumentation and generate coverage reports

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}JazzyIndex Code Coverage Script${NC}"
echo "=================================="

# Check for required tools
command -v gcov >/dev/null 2>&1 || { echo -e "${RED}Error: gcov is required but not installed.${NC}" >&2; exit 1; }
command -v lcov >/dev/null 2>&1 || { echo -e "${YELLOW}Warning: lcov not found. Install with: brew install lcov (macOS) or apt-get install lcov (Linux)${NC}"; }

# Create coverage build directory
COVERAGE_DIR="build_coverage"
echo -e "${GREEN}Creating coverage build directory: ${COVERAGE_DIR}${NC}"
mkdir -p "${COVERAGE_DIR}"
cd "${COVERAGE_DIR}"

# Configure with coverage flags
echo -e "${GREEN}Configuring with coverage instrumentation...${NC}"
cmake -DCMAKE_BUILD_TYPE=Coverage ..

# Build tests
echo -e "${GREEN}Building tests with coverage...${NC}"
cmake --build . --parallel

# Run tests
echo -e "${GREEN}Running tests to collect coverage data...${NC}"
ctest --output-on-failure

# Generate coverage report with lcov (if available)
if command -v lcov >/dev/null 2>&1; then
    echo -e "${GREEN}Generating coverage report with lcov...${NC}"

    # Capture coverage info
    lcov --capture --directory . --output-file coverage.info

    # Remove external dependencies and test files from coverage
    lcov --remove coverage.info \
        '/usr/*' \
        '*/build_coverage/_deps/*' \
        '*/tests/*' \
        --output-file coverage_filtered.info

    # Generate HTML report
    if command -v genhtml >/dev/null 2>&1; then
        echo -e "${GREEN}Generating HTML coverage report...${NC}"
        genhtml coverage_filtered.info --output-directory coverage_html

        echo -e "${GREEN}Coverage report generated in: ${COVERAGE_DIR}/coverage_html/index.html${NC}"

        # Try to open in browser (macOS)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            open coverage_html/index.html 2>/dev/null || true
        fi
    fi

    # Print summary
    echo -e "${GREEN}Coverage Summary:${NC}"
    lcov --summary coverage_filtered.info
else
    echo -e "${YELLOW}lcov not available. Raw .gcda files in: ${COVERAGE_DIR}${NC}"
    echo -e "${YELLOW}Use 'gcov' manually to analyze coverage.${NC}"
fi

echo -e "${GREEN}Coverage analysis complete!${NC}"
