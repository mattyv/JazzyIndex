#!/bin/bash
#
# Generate all documentation plots into committed locations
#
# This script runs benchmarks and generates visualizations, placing
# the output in docs/images/ for commit to the repository.
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
DOCS_IMAGES="$PROJECT_ROOT/docs/images"

echo "========================================"
echo "Generating JazzyIndex Documentation Plots"
echo "========================================"
echo

# Ensure build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found at $BUILD_DIR"
    echo "Please run: cmake -S . -B build"
    exit 1
fi

# Ensure executables exist
if [ ! -f "$BUILD_DIR/jazzy_index_benchmarks" ]; then
    echo "Error: jazzy_index_benchmarks not found"
    echo "Please run: cmake --build build"
    exit 1
fi

# Create output directories
mkdir -p "$DOCS_IMAGES/benchmarks"
mkdir -p "$DOCS_IMAGES/visualizations"
mkdir -p "$DOCS_IMAGES/index_data"

echo "Step 1/3: Running performance benchmarks..."
"$BUILD_DIR/jazzy_index_benchmarks" \
    --benchmark_format=json \
    --benchmark_out="$BUILD_DIR/jazzy_benchmarks.json"

echo
echo "Step 2/3: Generating performance plots..."
python3 "$SCRIPT_DIR/plot_benchmarks.py" \
    --input "$BUILD_DIR/jazzy_benchmarks.json" \
    --output "$DOCS_IMAGES/benchmarks/jazzy_benchmarks.png"

echo
echo "Step 3/3: Generating index structure visualizations..."
"$BUILD_DIR/jazzy_index_benchmarks" --visualize-index

# Set up Python venv if needed
if [ ! -d "$BUILD_DIR/jazzy_venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$BUILD_DIR/jazzy_venv"
    "$BUILD_DIR/jazzy_venv/bin/pip" install --upgrade pip
    "$BUILD_DIR/jazzy_venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
fi

# Copy JSON files from build to docs
echo "Copying visualization data to docs..."
cp -r "$BUILD_DIR/index_data"/* "$DOCS_IMAGES/index_data/"

# Generate all visualization plots
"$BUILD_DIR/jazzy_venv/bin/python" "$SCRIPT_DIR/plot_index_structure.py" "$DOCS_IMAGES/index_data"

# Copy select showcase visualizations to docs/images/visualizations/
echo
echo "Copying select showcase visualizations..."
cp "$DOCS_IMAGES/index_data/index_Uniform_N1000_S256.png" "$DOCS_IMAGES/visualizations/"
cp "$DOCS_IMAGES/index_data/index_Quadratic_N1000_S256.png" "$DOCS_IMAGES/visualizations/"
cp "$DOCS_IMAGES/index_data/index_ExtremePoly_N1000_S8.png" "$DOCS_IMAGES/visualizations/"
cp "$DOCS_IMAGES/index_data/index_Zipf_N10000_S256.png" "$DOCS_IMAGES/visualizations/"
cp "$DOCS_IMAGES/index_data/index_InversePoly_N1000_S64.png" "$DOCS_IMAGES/visualizations/"

echo
echo "========================================"
echo "Documentation Plots Generated Successfully!"
echo "========================================"
echo
echo "Performance plots:"
echo "  - $DOCS_IMAGES/benchmarks/jazzy_benchmarks_low.png"
echo "  - $DOCS_IMAGES/benchmarks/jazzy_benchmarks_medium.png"
echo "  - $DOCS_IMAGES/benchmarks/jazzy_benchmarks_high.png"
echo
echo "Showcase visualization plots:"
ls -1 "$DOCS_IMAGES/visualizations/"
echo
echo "All index structure visualizations (270+ plots):"
echo "  $DOCS_IMAGES/index_data/"
echo
echo "These plots are ready to commit to the repository."
