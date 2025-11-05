#!/bin/bash
# Script to generate separate benchmark plots for each range function
# (equal_range, lower_bound, upper_bound)

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

INPUT_JSON="${1:-$PROJECT_ROOT/build/jazzy_range_benchmarks.json}"
OUTPUT_DIR="${2:-$PROJECT_ROOT/docs/images/benchmarks}"

# Check if input file exists
if [ ! -f "$INPUT_JSON" ]; then
    echo "Error: Input file not found: $INPUT_JSON"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get Python from venv if available
if [ -f "$PROJECT_ROOT/build/jazzy_venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/build/jazzy_venv/bin/python"
else
    PYTHON="python3"
fi

echo "Generating range function benchmark plots..."
echo "Input: $INPUT_JSON"
echo "Output directory: $OUTPUT_DIR"

# Generate plots for each function type by filtering the JSON
for func in "EqualRange" "LowerBound" "UpperBound"; do
    echo ""
    echo "Processing $func benchmarks..."

    # Create a filtered JSON with only this function's benchmarks
    TEMP_JSON="/tmp/jazzy_${func}_benchmarks.json"

    # Filter the JSON to include only benchmarks for this function
    jq --arg func "$func" '{
        context: .context,
        benchmarks: [
            .benchmarks[] | select(
                .name | contains("_" + $func + "_") or contains("Std_" + $func)
            )
        ]
    }' "$INPUT_JSON" > "$TEMP_JSON"

    # Check if we have any benchmarks for this function
    BENCH_COUNT=$(jq '.benchmarks | length' "$TEMP_JSON")
    if [ "$BENCH_COUNT" -eq 0 ]; then
        echo "  No benchmarks found for $func, skipping..."
        rm -f "$TEMP_JSON"
        continue
    fi

    echo "  Found $BENCH_COUNT benchmarks for $func"

    # Generate plots using the main plotting script
    # Convert function name to lowercase
    func_lower=$(echo "$func" | tr '[:upper:]' '[:lower:]')
    FUNC_OUTPUT="$OUTPUT_DIR/jazzy_${func_lower}_benchmarks.png"
    "$PYTHON" "$SCRIPT_DIR/plot_benchmarks.py" \
        --input "$TEMP_JSON" \
        --output "$FUNC_OUTPUT"

    if [ -f "${FUNC_OUTPUT%.*}_low.png" ]; then
        # Rename the output files to include function name
        for suffix in "low" "medium" "high"; do
            if [ -f "${FUNC_OUTPUT%.*}_${suffix}.png" ]; then
                echo "  Generated: jazzy_${func_lower}_benchmarks_${suffix}.png"
            fi
        done
    fi

    # Clean up temp file
    rm -f "$TEMP_JSON"
done

echo ""
echo "Done! Benchmark plots generated in: $OUTPUT_DIR"
