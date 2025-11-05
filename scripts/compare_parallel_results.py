#!/usr/bin/env python3
"""
Compare benchmark results from sequential vs parallel execution to determine
if running multiple benchmark processes simultaneously affects timing accuracy.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def load_benchmark_results(json_path: Path) -> Dict:
    """Load benchmark results from JSON file."""
    try:
        with open(json_path, 'r') as f:
            content = f.read().strip()
            if not content:
                return {'benchmarks': []}
            return json.loads(content)
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse {json_path}, skipping")
        return {'benchmarks': []}

def extract_timings(results: Dict) -> Dict[str, float]:
    """Extract benchmark name -> cpu_time mapping."""
    timings = {}
    for bench in results.get('benchmarks', []):
        name = bench['name']
        # Only keep _mean entries (most relevant for comparison)
        # Skip _median, _stddev, _cv, and individual runs
        if '_mean' in name:
            # Remove the _mean suffix for cleaner comparison
            clean_name = name.replace('_mean', '')
            timings[clean_name] = bench['cpu_time']
        elif '_median' in name or '_stddev' in name or '_cv' in name:
            continue
        elif 'aggregate' not in bench.get('run_type', ''):
            # Also include non-aggregated single runs (when no repetitions)
            timings[name] = bench['cpu_time']
    return timings

def compare_timings(baseline: Dict[str, float],
                   parallel: Dict[str, float],
                   label: str) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Compare baseline vs parallel timings.
    Returns (max_degradation_percent, list of (name, degradation_percent)).
    """
    differences = []

    for name, baseline_time in baseline.items():
        if name not in parallel:
            continue

        parallel_time = parallel[name]
        degradation_pct = ((parallel_time - baseline_time) / baseline_time) * 100
        differences.append((name, degradation_pct, baseline_time, parallel_time))

    if not differences:
        return 0.0, []

    max_degradation = max(abs(d[1]) for d in differences)

    print(f"\n=== {label} ===")
    print(f"{'Benchmark':<50} {'Baseline (ns)':<15} {'Parallel (ns)':<15} {'Change %':<10}")
    print("-" * 95)

    for name, deg_pct, base_time, par_time in sorted(differences, key=lambda x: abs(x[1]), reverse=True):
        status = "⚠️" if abs(deg_pct) > 10 else "✓"
        print(f"{status} {name:<48} {base_time:>13.2f} {par_time:>13.2f} {deg_pct:>8.1f}%")

    return max_degradation, [(d[0], d[1]) for d in differences]

def main():
    test_dir = Path("benchmark_parallelism_test")

    if not test_dir.exists():
        print(f"Error: Test directory '{test_dir}' not found.")
        print("Please run analyze_parallel_benchmark_limits.sh first.")
        sys.exit(1)

    # Load baseline
    baseline_path = test_dir / "sequential_baseline.json"
    if not baseline_path.exists():
        print(f"Error: Baseline results not found at {baseline_path}")
        sys.exit(1)

    baseline_results = load_benchmark_results(baseline_path)
    baseline_timings = extract_timings(baseline_results)

    print("=" * 95)
    print("PARALLEL BENCHMARK EXECUTION ANALYSIS")
    print("=" * 95)
    print(f"\nBaseline (sequential): {len(baseline_timings)} benchmarks")

    # Compare 2 parallel processes
    parallel_2_files = list(test_dir.glob("parallel_2_group*.json"))
    if parallel_2_files:
        all_parallel_2 = {}
        for f in parallel_2_files:
            all_parallel_2.update(extract_timings(load_benchmark_results(f)))

        max_deg_2, _ = compare_timings(baseline_timings, all_parallel_2,
                                        "2 Parallel Processes")
        print(f"\nMax timing degradation: {max_deg_2:.1f}%")

    # Compare 4 parallel processes
    parallel_4_files = list(test_dir.glob("parallel_4_group*.json"))
    if parallel_4_files:
        all_parallel_4 = {}
        for f in parallel_4_files:
            all_parallel_4.update(extract_timings(load_benchmark_results(f)))

        max_deg_4, _ = compare_timings(baseline_timings, all_parallel_4,
                                        "4 Parallel Processes")
        print(f"\nMax timing degradation: {max_deg_4:.1f}%")

    # Summary
    print("\n" + "=" * 95)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 95)

    if parallel_2_files:
        if max_deg_2 < 5:
            print("✓ 2 parallel processes: SAFE (<5% degradation)")
        elif max_deg_2 < 10:
            print("⚠️  2 parallel processes: ACCEPTABLE (5-10% degradation)")
        else:
            print("❌ 2 parallel processes: NOT RECOMMENDED (>10% degradation)")

    if parallel_4_files:
        if max_deg_4 < 5:
            print("✓ 4 parallel processes: SAFE (<5% degradation)")
        elif max_deg_4 < 10:
            print("⚠️  4 parallel processes: ACCEPTABLE (5-10% degradation)")
        else:
            print("❌ 4 parallel processes: NOT RECOMMENDED (>10% degradation)")

    print("\nNote: On Apple Silicon Macs with P/E cores:")
    print("  - Use 'taskset' or 'numactl' to pin to Performance cores only")
    print("  - Avoid Efficiency cores for benchmarking (much slower)")
    print("  - Check cache topology to avoid cores sharing L2/L3 cache")
    print("  - Consider process affinity to minimize cache contention")

if __name__ == '__main__':
    main()
