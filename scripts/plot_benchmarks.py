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


BenchmarkKey = Tuple[str, str, int, int]

SEGMENT_ORDER = [64, 128, 256, 512]
SEGMENT_COLORS = {
    64: "#1f77b4",   # blue
    128: "#ff7f0e",  # orange
    256: "#2ca02c",  # green
    512: "#d62728",  # red
}
SEGMENT_MARKERS = {
    64: "o",
    128: "s",
    256: "^",
    512: "D",
}

SCENARIO_ORDER = ["Found", "FoundMiddle", "FoundEnd", "NotFound"]
SCENARIO_STYLES = {
    "Found": "-",
    "FoundMiddle": "--",
    "FoundEnd": "-.",
    "NotFound": ":",
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
        JazzyIndex/<Distribution>/S<Segments>/N<Size>/<Scenario>
    """
    parts = name.split("/")
    if len(parts) != 5 or parts[0] != "JazzyIndex":
        raise ValueError(f"Unexpected benchmark name format: {name}")

    _, distribution, segments_part, size_part, scenario = parts
    if not segments_part.startswith("S") or not size_part.startswith("N"):
        raise ValueError(f"Unexpected encoding of segments/size in: {name}")

    segments = int(segments_part[1:])
    size = int(size_part[1:])
    return distribution, scenario, segments, size


def load_benchmark_data(path: Path) -> Dict[str, Dict[str, Dict[int, List[Tuple[int, float]]]]]:
    """
    Load the JSON file and aggregate results as:
      data[distribution][scenario][segments] -> list[(size, time_ns)]
    """
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    runs = payload.get("benchmarks", [])
    grouped: Dict[str, Dict[str, Dict[int, List[Tuple[int, float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for run in runs:
        name = run.get("name")
        if not name:
            continue

        try:
            distribution, scenario, segments, size = parse_benchmark_name(name)
        except ValueError:
            # Skip unrelated benchmarks (e.g., warmup lines)
            continue

        time_ns = float(run.get("real_time", 0.0))
        grouped[distribution][scenario][segments].append((size, time_ns))

    return grouped


def scenario_sort_key(scenario: str) -> Tuple[int, str]:
    try:
        return (SCENARIO_ORDER.index(scenario), scenario)
    except ValueError:
        return (len(SCENARIO_ORDER), scenario)


def plot(grouped: Dict[str, Dict[str, Dict[int, List[Tuple[int, float]]]]], output: Path) -> None:
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
        for scenario, by_segments in sorted(scenarios.items(), key=lambda item: scenario_sort_key(item[0])):
            linestyle = SCENARIO_STYLES.get(scenario, "-")
            scenario_label = SCENARIO_LABELS.get(scenario, scenario)
            for segments, points in sorted(by_segments.items()):
                if not points:
                    continue
                points.sort(key=lambda item: item[0])
                sizes = [size for size, _ in points]
                times = [time for _, time in points]
                color = SEGMENT_COLORS.get(segments)
                if color is None:
                    color = plt.get_cmap("tab10")(SEGMENT_ORDER.index(segments) % 10)
                marker = SEGMENT_MARKERS.get(segments, "o")
                ax.plot(
                    sizes,
                    times,
                    marker=marker,
                    linewidth=2.0,
                    markersize=5.5,
                    color=color,
                    linestyle=linestyle,
                    alpha=0.9,
                )

        ax.set_title(distribution, fontsize=12, fontweight="bold")
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

    segment_handles = [
        Line2D(
            [0],
            [0],
            color=SEGMENT_COLORS.get(seg, "gray"),
            marker=SEGMENT_MARKERS.get(seg, "o"),
            linestyle="-",
            linewidth=2.0,
            markersize=6,
            label=f"S={seg}",
        )
        for seg in SEGMENT_ORDER
    ]
    scenario_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=SCENARIO_STYLES.get(scenario, "-"),
            linewidth=2.0,
            label=SCENARIO_LABELS.get(scenario, scenario),
        )
        for scenario in SCENARIO_ORDER
        if any(scenario in grouped[d] for d in grouped)
    ]

    cpu_name = get_cpu_name()
    fig.suptitle("JazzyIndex Lookup Performance", fontsize=16, fontweight="bold")
    fig.text(0.99, 0.95, cpu_name, ha="right", va="top", fontsize=9, color="gray", transform=fig.transFigure)
    fig.subplots_adjust(bottom=0.18, top=0.90)
    fig.legend(
        handles=segment_handles,
        labels=[handle.get_label() for handle in segment_handles],
        title="Segments",
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
    args = parser.parse_args()

    grouped = load_benchmark_data(args.input)
    plot(grouped, args.output)


if __name__ == "__main__":
    main()
