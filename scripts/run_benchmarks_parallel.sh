#!/bin/bash
# Parallel Benchmark Runner for JazzyIndex
# Runs different segment counts in parallel processes to speed up benchmark execution

set -e

# Configuration
MAX_PARALLEL=${1:-4}  # Number of parallel processes (default: 4)
DATASET="Books"
BENCHMARK_ARGS="--200m-benchmarks"

echo "Parallel Benchmark Runner"
echo "========================="
echo "Dataset: $DATASET"
echo "Max Parallel: $MAX_PARALLEL"
echo "Additional Args: $BENCHMARK_ARGS"
echo ""

# Segment counts to benchmark
SEGMENT_COUNTS=(1 2 4 8 16 32 64 128 256 512)

# Create output directory for results
OUTPUT_DIR="benchmark_results"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Function to run benchmark for a specific segment count
run_benchmark() {
    local segment=$1
    local filter="${DATASET}/S${segment}"
    local output_file="${OUTPUT_DIR}/${DATASET}_S${segment}_${TIMESTAMP}.txt"

    echo "Starting: $filter"
    local start_time=$(date +%s)

    # Run the benchmark
    ./build/jazzy_index_benchmarks $BENCHMARK_ARGS --benchmark_filter="$filter" 2>&1 | tee "$output_file"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo "Completed: $filter (${duration}s)"
    echo "$segment:$output_file:$duration"
}

# Export function for parallel execution
export -f run_benchmark
export DATASET BENCHMARK_ARGS OUTPUT_DIR TIMESTAMP

# Run benchmarks in parallel batches
echo "Running benchmarks in batches of $MAX_PARALLEL..."
echo ""

RESULTS_FILE="${OUTPUT_DIR}/results_${TIMESTAMP}.tmp"
> "$RESULTS_FILE"

TOTAL_START=$(date +%s)

# Process segments in batches
for ((i=0; i<${#SEGMENT_COUNTS[@]}; i+=MAX_PARALLEL)); do
    batch=("${SEGMENT_COUNTS[@]:i:MAX_PARALLEL}")
    echo "Batch: S${batch[*]}"

    # Run batch in parallel
    printf "%s\n" "${batch[@]}" | xargs -P "$MAX_PARALLEL" -I {} bash -c 'run_benchmark {}' >> "$RESULTS_FILE"

    echo ""
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

# Print summary
echo ""
echo "============================================"
echo "Benchmark Summary"
echo "============================================"
echo ""

# Sort and display results
sort -n -t: -k1 "$RESULTS_FILE" | while IFS=: read -r segment output_file duration; do
    printf "S%-3d: %4ds - %s\n" "$segment" "$duration" "$output_file"
done

echo ""
echo "Total execution time: ${TOTAL_DURATION}s"
echo ""

# Combine all results into a single file
COMBINED_FILE="${OUTPUT_DIR}/${DATASET}_combined_${TIMESTAMP}.txt"
echo "Combining results into: $COMBINED_FILE"

> "$COMBINED_FILE"

sort -n -t: -k1 "$RESULTS_FILE" | while IFS=: read -r segment output_file duration; do
    echo "==========================================" >> "$COMBINED_FILE"
    echo "Segment Count: $segment" >> "$COMBINED_FILE"
    echo "==========================================" >> "$COMBINED_FILE"
    cat "$output_file" >> "$COMBINED_FILE"
    echo "" >> "$COMBINED_FILE"
done

# Cleanup temp file
rm "$RESULTS_FILE"

echo ""
echo "Done! Results saved to $OUTPUT_DIR/"
echo "Combined results: $COMBINED_FILE"
