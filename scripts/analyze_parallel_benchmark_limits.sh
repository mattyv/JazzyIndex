#!/bin/bash
# Script to test parallel benchmark execution limits
# Tests if running multiple benchmark processes simultaneously affects timing accuracy

set -e

BENCHMARK_BIN="./build/jazzy_index_benchmarks"
OUTPUT_DIR="benchmark_parallelism_test"

echo "=== Parallel Benchmark Execution Analysis ==="
echo ""

# Check if benchmark binary exists
if [ ! -f "$BENCHMARK_BIN" ]; then
    echo "Error: Benchmark binary not found at $BENCHMARK_BIN"
    echo "Please build it first: cmake --build build --target jazzy_index_benchmarks"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Get CPU info
echo "System Information:"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS detected"
    sysctl -n machdep.cpu.brand_string
    echo "Total cores: $(sysctl -n hw.ncpu)"
    echo "Physical cores: $(sysctl -n hw.physicalcpu)"

    # Try to detect P/E cores (Apple Silicon)
    if sysctl -n hw.perflevel0.physicalcpu 2>/dev/null; then
        P_CORES=$(sysctl -n hw.perflevel0.physicalcpu)
        E_CORES=$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo "0")
        echo "Performance cores: $P_CORES"
        echo "Efficiency cores: $E_CORES"
    fi
else
    echo "Linux detected"
    lscpu | grep -E "Model name|Core\(s\)|Thread\(s\)"
fi

echo ""
echo "=== Test 1: Sequential Baseline (Single Process) ==="
echo "Running a subset of benchmarks sequentially..."

# Run a quick subset as baseline
$BENCHMARK_BIN \
    --benchmark_filter=".*Uniform/N1000/Found.*" \
    --benchmark_format=json \
    --benchmark_out="$OUTPUT_DIR/sequential_baseline.json" \
    --benchmark_repetitions=3 \
    --benchmark_report_aggregates_only=true

echo "Sequential baseline complete."
echo ""

echo "=== Test 2: Parallel Execution (2 Processes) ==="
echo "Running 2 benchmark processes simultaneously..."

# Split benchmarks into 2 groups
$BENCHMARK_BIN \
    --benchmark_filter=".*Uniform/N1000/FoundMiddle" \
    --benchmark_format=json \
    --benchmark_out="$OUTPUT_DIR/parallel_2_group1.json" \
    --benchmark_repetitions=3 \
    --benchmark_report_aggregates_only=true &
PID1=$!

$BENCHMARK_BIN \
    --benchmark_filter=".*Uniform/N1000/FoundEnd" \
    --benchmark_format=json \
    --benchmark_out="$OUTPUT_DIR/parallel_2_group2.json" \
    --benchmark_repetitions=3 \
    --benchmark_report_aggregates_only=true &
PID2=$!

wait $PID1
wait $PID2

echo "2 parallel processes complete."
echo ""

echo "=== Test 3: Parallel Execution (4 Processes) ==="
echo "Running 4 benchmark processes simultaneously..."

$BENCHMARK_BIN \
    --benchmark_filter=".*Uniform/N1000/FoundMiddle" \
    --benchmark_format=json \
    --benchmark_out="$OUTPUT_DIR/parallel_4_group1.json" \
    --benchmark_repetitions=3 \
    --benchmark_report_aggregates_only=true &
PID1=$!

$BENCHMARK_BIN \
    --benchmark_filter=".*Exponential/N1000/FoundMiddle" \
    --benchmark_format=json \
    --benchmark_out="$OUTPUT_DIR/parallel_4_group2.json" \
    --benchmark_repetitions=3 \
    --benchmark_report_aggregates_only=true &
PID2=$!

$BENCHMARK_BIN \
    --benchmark_filter=".*Clustered/N1000/FoundMiddle" \
    --benchmark_format=json \
    --benchmark_out="$OUTPUT_DIR/parallel_4_group3.json" \
    --benchmark_repetitions=3 \
    --benchmark_report_aggregates_only=true &
PID3=$!

$BENCHMARK_BIN \
    --benchmark_filter=".*Lognormal/N1000/FoundMiddle" \
    --benchmark_format=json \
    --benchmark_out="$OUTPUT_DIR/parallel_4_group4.json" \
    --benchmark_repetitions=3 \
    --benchmark_report_aggregates_only=true &
PID4=$!

wait $PID1 $PID2 $PID3 $PID4

echo "4 parallel processes complete."
echo ""

echo "=== Test 4: Maximum Parallelism (8 Processes) ==="
echo "Running 8 benchmark processes simultaneously..."

# Run 8 different small benchmarks in parallel
$BENCHMARK_BIN --benchmark_filter=".*Uniform/N100/.*" --benchmark_format=json --benchmark_out="$OUTPUT_DIR/parallel_8_group1.json" --benchmark_repetitions=1 &
$BENCHMARK_BIN --benchmark_filter=".*Exponential/N100/.*" --benchmark_format=json --benchmark_out="$OUTPUT_DIR/parallel_8_group2.json" --benchmark_repetitions=1 &
$BENCHMARK_BIN --benchmark_filter=".*Clustered/N100/.*" --benchmark_format=json --benchmark_out="$OUTPUT_DIR/parallel_8_group3.json" --benchmark_repetitions=1 &
$BENCHMARK_BIN --benchmark_filter=".*Lognormal/N100/.*" --benchmark_format=json --benchmark_out="$OUTPUT_DIR/parallel_8_group4.json" --benchmark_repetitions=1 &
$BENCHMARK_BIN --benchmark_filter=".*Zipf/N100/.*" --benchmark_format=json --benchmark_out="$OUTPUT_DIR/parallel_8_group5.json" --benchmark_repetitions=1 &
$BENCHMARK_BIN --benchmark_filter=".*Mixed/N100/.*" --benchmark_format=json --benchmark_out="$OUTPUT_DIR/parallel_8_group6.json" --benchmark_repetitions=1 &
$BENCHMARK_BIN --benchmark_filter=".*Quadratic/N100/.*" --benchmark_format=json --benchmark_out="$OUTPUT_DIR/parallel_8_group7.json" --benchmark_repetitions=1 &
$BENCHMARK_BIN --benchmark_filter=".*ExtremePoly/N100/.*" --benchmark_format=json --benchmark_out="$OUTPUT_DIR/parallel_8_group8.json" --benchmark_repetitions=1 &

wait

echo "8 parallel processes complete."
echo ""

echo "=== Analysis ==="
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "To compare results, check if cpu_time values are consistent between:"
echo "  - sequential_baseline.json"
echo "  - parallel_2_group*.json"
echo "  - parallel_4_group*.json"
echo ""
echo "If cpu_time degrades significantly with more parallel processes,"
echo "you've hit cache contention or CPU scheduling limits."
