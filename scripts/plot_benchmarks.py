#!/usr/bin/env python3
"""
Plot JazzyIndex benchmark results.

The script expects a JSON file produced by Google Benchmark
(`--benchmark_format=json --benchmark_out=<file>`). It generates a PNG
containing per-distribution lookup cost over dataset sizes for each
segment configuration.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator


# Key format: (implementation, distribution, scenario, segments/None, size)
BenchmarkKey = Tuple[str, str, str, int, int]

SEGMENT_ORDER = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Segment groups for split plotting
SEGMENT_GROUPS = {
    'low': [1, 2, 4, 8],
    'medium': [16, 32, 64, 128],
    'high': [256, 512]
}

SEGMENT_COLORS = {
    1: "#9467bd",    # purple
    2: "#8c564b",    # brown
    4: "#e377c2",    # pink
    8: "#7f7f7f",    # gray
    16: "#bcbd22",   # olive
    32: "#17becf",   # cyan
    64: "#1f77b4",   # blue
    128: "#ff7f0e",  # orange
    256: "#2ca02c",  # green
    512: "#d62728",  # red
}
SEGMENT_MARKERS = {
    1: "v",
    2: "<",
    4: ">",
    8: "p",
    16: "*",
    32: "h",
    64: "o",
    128: "s",
    256: "^",
    512: "D",
}

# std::lower_bound styling
LOWER_BOUND_COLOR = "#000000"  # black
LOWER_BOUND_MARKER = "x"
LOWER_BOUND_LINEWIDTH = 2.5

SCENARIO_ORDER = ["Found", "FoundMiddle", "FoundEnd", "NotFound"]
SCENARIO_STYLES = {
    "Found": "-",           # Solid line
    "FoundMiddle": "-.",    # Dash-dot line (matches main benchmarks)
    "FoundEnd": ":",        # Dotted line
    "NotFound": "--",       # Dashed line (matches main benchmarks)
}
SCENARIO_LINE_WIDTHS = {
    "Found": 2.5,
    "FoundMiddle": 2.5,
    "FoundEnd": 2.5,
    "NotFound": 2.5,
}
SCENARIO_LABELS = {
    "Found": "Found",
    "FoundMiddle": "Found (mid)",
    "FoundEnd": "Found (end)",
    "NotFound": "Not found",
}


def get_cpu_name() -> str:
    """Get the CPU name for the current platform."""
    system = platform.system()
    try:
        if system == "Darwin":
            # macOS
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        elif system == "Linux":
            # Linux
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        elif system == "Windows":
            # Windows
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True,
                text=True,
                check=True,
            )
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                return lines[1].strip()
    except Exception:
        pass
    # Fallback to platform info
    return f"{platform.system()} {platform.machine()}"


def parse_benchmark_name(name: str) -> BenchmarkKey:
    """
    Parse benchmark names of the form:
        JazzyIndex/<Distribution>/S<Segments>/N<Size>/<Scenario>[/threads:N]
        LowerBound/<Distribution>/N<Size>/<Scenario>[/threads:N]
        BM_JazzyIndex_<Function>_<Segments>_<Distribution>/<Scenario>/<Size>[/threads:N]
        BM_Std_<Function>/<Distribution>/<Scenario>/<Size>[/threads:N]

    Returns: (implementation, distribution, scenario, segments, size)
    where implementation is "JazzyIndex" or "LowerBound" or function name, segments is None for std:: baselines

    Note: The optional /threads:N suffix (added by Google Benchmark when using
    multithreading) is stripped before parsing.
    """
    parts = name.split("/")

    # Strip optional threads:N suffix
    if parts and parts[-1].startswith("threads:"):
        parts = parts[:-1]

    # Parse JazzyIndex format (old style)
    if parts[0] == "JazzyIndex":
        if len(parts) != 5:
            raise ValueError(f"Unexpected JazzyIndex benchmark format: {name}")
        _, distribution, segments_part, size_part, scenario = parts
        if not segments_part.startswith("S") or not size_part.startswith("N"):
            raise ValueError(f"Unexpected encoding of segments/size in: {name}")
        segments = int(segments_part[1:])
        size = int(size_part[1:])
        return "JazzyIndex", distribution, scenario, segments, size

    # Parse LowerBound format (old style)
    elif parts[0] == "LowerBound":
        if len(parts) != 4:
            raise ValueError(f"Unexpected LowerBound benchmark format: {name}")
        _, distribution, size_part, scenario = parts
        if not size_part.startswith("N"):
            raise ValueError(f"Unexpected encoding of size in: {name}")
        size = int(size_part[1:])
        return "LowerBound", distribution, scenario, None, size

    # Parse new-style range function benchmarks: BM_JazzyIndex_<Function>_<Segments>_<Distribution>/<Scenario>/<Size>
    elif parts[0].startswith("BM_JazzyIndex_"):
        # Extract function, segments, and distribution from benchmark name
        # Example: BM_JazzyIndex_EqualRange_64_Uniform/FoundMiddle/100
        benchmark_parts = parts[0].split("_")
        if len(benchmark_parts) < 5:  # BM, JazzyIndex, Function, Segments, Distribution
            raise ValueError(f"Unexpected BM_JazzyIndex benchmark format: {name}")

        function = benchmark_parts[2]  # EqualRange, LowerBound, or UpperBound
        segments = int(benchmark_parts[3])  # 64, 256, 512
        distribution = benchmark_parts[4]  # Uniform, Clustered, etc.

        if len(parts) < 3:
            raise ValueError(f"Missing scenario or size in benchmark: {name}")

        scenario = parts[1]  # FoundMiddle, FoundEnd, or NotFound
        size = int(parts[2])

        return "JazzyIndex", distribution, scenario, segments, size

    # Parse std:: baseline benchmarks: BM_Std_<Function>/<Distribution>/<Scenario>/<Size>
    elif parts[0].startswith("BM_Std_"):
        # Example: BM_Std_EqualRange/Uniform/FoundMiddle/100
        function = parts[0].split("_")[2]  # EqualRange, LowerBound, or UpperBound

        if len(parts) < 4:
            raise ValueError(f"Missing distribution, scenario, or size in benchmark: {name}")

        distribution = parts[1]
        scenario = parts[2]  # FoundMiddle, FoundEnd, or NotFound
        size = int(parts[3])

        return f"Std{function}", distribution, scenario, None, size

    else:
        raise ValueError(f"Unknown benchmark type: {parts[0]}")


def load_segment_finder_models(index_data_dir: Path) -> Dict[Tuple[str, int, int], str]:
    """
    Load segment finder model types from index data JSON files.
    Returns: dict mapping (distribution, N, S) -> model_type
    """
    models = {}
    if not index_data_dir.exists():
        return models

    for json_file in index_data_dir.glob("index_*_N*_S*.json"):
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # Parse filename: index_<Distribution>_N<size>_S<segments>.json
            filename = json_file.stem  # Remove .json
            parts = filename.split("_")
            if len(parts) >= 4:
                distribution = parts[1]
                n_str = parts[2]  # N10000
                s_str = parts[3]  # S256

                if n_str.startswith("N") and s_str.startswith("S"):
                    n = int(n_str[1:])
                    s = int(s_str[1:])
                    model_type = data.get("segment_finder", {}).get("model_type", "UNKNOWN")
                    models[(distribution, n, s)] = model_type
        except Exception:
            # Skip files that can't be parsed
            continue

    return models


def load_benchmark_data(path: Path, index_data_dir: Path = None):
    """
    Load the JSON file and aggregate results as:
      data[distribution][scenario][(implementation, segments)] -> list[(size, time_ns)]
    where implementation is "JazzyIndex" or "LowerBound"
    and segments is the segment count for JazzyIndex or "baseline" for LowerBound

    Also returns segment finder model information if index_data_dir is provided.
    """
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    runs = payload.get("benchmarks", [])
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for run in runs:
        name = run.get("name")
        if not name:
            continue

        try:
            impl, distribution, scenario, segments, size = parse_benchmark_name(name)
        except ValueError:
            # Skip unrelated benchmarks (e.g., warmup lines)
            continue

        time_ns = float(run.get("real_time", 0.0))

        # Use a key that identifies implementation + segments
        if impl == "LowerBound":
            impl_key = ("LowerBound", "baseline")
        elif impl.startswith("Std"):
            # Preserve std:: function names (StdEqualRange, StdLowerBound, StdUpperBound)
            impl_key = (impl, None)
        else:
            # JazzyIndex benchmarks
            impl_key = ("JazzyIndex", segments)

        grouped[distribution][scenario][impl_key].append((size, time_ns))

    # Load segment finder models if directory provided
    models = {}
    if index_data_dir:
        models = load_segment_finder_models(index_data_dir)

    return grouped, models


def scenario_sort_key(scenario: str) -> Tuple[int, str]:
    try:
        return (SCENARIO_ORDER.index(scenario), scenario)
    except ValueError:
        return (len(SCENARIO_ORDER), scenario)


def plot_segment_group(grouped, output: Path, segment_list: List[int], group_name: str, models: Dict[Tuple[str, int, int], str] = None) -> None:
    """Plot benchmarks for a specific group of segment counts."""
    if models is None:
        models = {}
    distributions = sorted(grouped)
    if not distributions:
        raise ValueError("No benchmark results were found in the input file.")

    cols = 2
    rows = ceil(len(distributions) / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 6.2, rows * 4.2),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for ax_idx, (ax, distribution) in enumerate(zip(axes.flatten(), distributions)):
        scenarios = grouped[distribution]
        for scenario, by_impl in sorted(scenarios.items(), key=lambda item: scenario_sort_key(item[0])):
            linestyle = SCENARIO_STYLES.get(scenario, "-")
            scenario_label = SCENARIO_LABELS.get(scenario, scenario)

            # Sort with custom key to handle None values (baselines)
            def sort_key(item):
                impl_key, _ = item
                impl, seg_or_baseline = impl_key
                # Put baselines first, then sort by segment count
                if seg_or_baseline is None:
                    return (0, 0, impl)  # Baselines first, sorted by name
                elif isinstance(seg_or_baseline, str):
                    return (0, 1, impl)  # String baselines (e.g., "baseline") after None
                else:
                    return (1, seg_or_baseline, "")  # Then by segment count (numeric)

            for impl_key, points in sorted(by_impl.items(), key=sort_key):
                if not points:
                    continue

                impl, seg_or_baseline = impl_key

                # Filter: only plot if segment is in our group (or it's a baseline)
                if impl == "JazzyIndex" and seg_or_baseline not in segment_list:
                    continue
                # Also skip baselines that aren't for this function/scenario
                if seg_or_baseline is None and not (impl.startswith("Std") or impl == "LowerBound"):
                    continue

                points.sort(key=lambda item: item[0])
                sizes = [size for size, _ in points]
                times = [time for _, time in points]

                if impl == "LowerBound" or impl.startswith("Std"):
                    # Plot std:: baselines as thick black line
                    # Map function names to display labels
                    baseline_label = impl
                    if impl == "LowerBound":
                        baseline_label = "std::lower_bound"
                    elif impl == "StdEqualRange":
                        baseline_label = "std::equal_range"
                    elif impl == "StdLowerBound":
                        baseline_label = "std::lower_bound"
                    elif impl == "StdUpperBound":
                        baseline_label = "std::upper_bound"

                    ax.plot(
                        sizes,
                        times,
                        marker=LOWER_BOUND_MARKER,
                        linewidth=LOWER_BOUND_LINEWIDTH,
                        markersize=7,
                        color=LOWER_BOUND_COLOR,
                        linestyle=linestyle,
                        alpha=0.8,
                        zorder=10,  # Draw on top
                        label=baseline_label if scenario == list(scenarios.keys())[0] else None
                    )
                else:
                    # Plot JazzyIndex with segment-specific colors
                    segments = seg_or_baseline
                    color = SEGMENT_COLORS.get(segments)
                    if color is None:
                        color = plt.get_cmap("tab10")(SEGMENT_ORDER.index(segments) % 10)
                    marker = SEGMENT_MARKERS.get(segments, "o")
                    linewidth = SCENARIO_LINE_WIDTHS.get(scenario, 2.0)
                    ax.plot(
                        sizes,
                        times,
                        marker=marker,
                        linewidth=linewidth,
                        markersize=5.5,
                        color=color,
                        linestyle=linestyle,
                        alpha=0.9,
                    )

        # Find model type for this distribution (use largest segment/size if available)
        model_info = ""
        for seg in reversed(segment_list):  # Check largest segment first
            for size in [10000, 1000, 100]:  # Check common sizes
                model_type = models.get((distribution, size, seg))
                if model_type:
                    model_info = f" ({model_type})"
                    break
            if model_info:
                break

        ax.set_title(f"{distribution}{model_info}", fontsize=12, fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=15))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{int(val)}" if val >= 1 else f"{val:.1f}"))
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.tick_params(which="both", labelsize=9, labelleft=True)

    # Axis labels for outer edges only
    for ax in axes[-1, :]:
        ax.set_xlabel("Dataset size (elements)", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("Nanoseconds per lookup", fontsize=11)

    # Hide unused axes if the grid is larger than the number of distributions
    for unused_ax in axes.flatten()[len(distributions) :]:
        unused_ax.axis("off")

    # Create legend handles (only for segments in this group)
    segment_handles = [
        Line2D(
            [0],
            [0],
            color=SEGMENT_COLORS.get(seg, "gray"),
            marker=SEGMENT_MARKERS.get(seg, "o"),
            linestyle="-",
            linewidth=2.0,
            markersize=6,
            label=f"JI S={seg}",
        )
        for seg in segment_list
    ]

    # Detect which std:: function is being benchmarked by checking impl types
    std_function_name = "std::lower_bound"  # default
    found_function = False
    for distribution in grouped:
        for scenario in grouped[distribution]:
            for impl_key in grouped[distribution][scenario]:
                impl, _ = impl_key
                if impl == "StdEqualRange":
                    std_function_name = "std::equal_range"
                    found_function = True
                    break
                elif impl == "StdLowerBound":
                    std_function_name = "std::lower_bound"
                    found_function = True
                    break
                elif impl == "StdUpperBound":
                    std_function_name = "std::upper_bound"
                    found_function = True
                    break
                elif impl == "LowerBound":
                    std_function_name = "std::lower_bound"
                    found_function = True
                    break
            if found_function:
                break
        if found_function:
            break

    # Add std:: baseline handle
    lower_bound_handle = Line2D(
        [0],
        [0],
        color=LOWER_BOUND_COLOR,
        marker=LOWER_BOUND_MARKER,
        linestyle="-",
        linewidth=LOWER_BOUND_LINEWIDTH,
        markersize=7,
        label=std_function_name,
    )
    segment_handles.append(lower_bound_handle)

    scenario_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=SCENARIO_STYLES.get(scenario, "-"),
            linewidth=SCENARIO_LINE_WIDTHS.get(scenario, 2.0),
            label=SCENARIO_LABELS.get(scenario, scenario),
        )
        for scenario in SCENARIO_ORDER
        if any(scenario in grouped[d] for d in grouped)
    ]

    cpu_name = get_cpu_name()
    title = f"JazzyIndex vs {std_function_name} Performance - {group_name.title()} Segments"
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.text(0.99, 0.95, cpu_name, ha="right", va="top", fontsize=9, color="gray", transform=fig.transFigure)
    fig.subplots_adjust(bottom=0.18, top=0.90)
    fig.legend(
        handles=segment_handles,
        labels=[handle.get_label() for handle in segment_handles],
        title="Implementation",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=len(segment_handles),
        fontsize=9,
        title_fontsize=10,
    )
    if scenario_handles:
        fig.legend(
            handles=scenario_handles,
            labels=[handle.get_label() for handle in scenario_handles],
            title="Scenarios",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.10),
            ncol=max(1, len(scenario_handles)),
            fontsize=9,
            title_fontsize=10,
        )

    fig.tight_layout(rect=(0.02, 0.18, 0.98, 0.88))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot(grouped, output: Path, models: Dict[Tuple[str, int, int], str] = None) -> None:
    """Generate plots split by segment count groups."""
    # Generate output paths for each segment group
    output_stem = output.stem
    output_dir = output.parent
    output_suffix = output.suffix

    for group_name, segment_list in SEGMENT_GROUPS.items():
        group_output = output_dir / f"{output_stem}_{group_name}{output_suffix}"
        plot_segment_group(grouped, group_output, segment_list, group_name, models)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot JazzyIndex benchmark results.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to benchmark JSON produced by Google Benchmark.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/jazzy_benchmarks.png"),
        help="Destination PNG (will be overwritten). Default: benchmarks/jazzy_benchmarks.png",
    )
    parser.add_argument(
        "--index-data-dir",
        type=Path,
        default=Path("docs/images/index_data"),
        help="Directory containing index data JSON files with model information. Default: docs/images/index_data",
    )
    args = parser.parse_args()

    grouped, models = load_benchmark_data(args.input, args.index_data_dir)
    plot(grouped, args.output, models)


if __name__ == "__main__":
    main()
