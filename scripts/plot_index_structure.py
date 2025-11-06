#!/usr/bin/env python3
"""
Visualize JazzyIndex structure: data distribution, segment boundaries, and model predictions.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_index_data(json_file: Path) -> Dict[str, Any]:
    """Load index metadata from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def plot_segment_model(ax, segment: Dict[str, Any], keys: np.ndarray, alpha: float = 0.3):
    """Plot the prediction curve for a single segment."""
    model_type = segment['model_type']
    start_idx = segment['start_idx']
    end_idx = segment['end_idx']

    if end_idx <= start_idx:
        return

    # Get value range for this segment
    min_val = segment['min_val']
    max_val = segment['max_val']

    # Create dense sample of values in this segment's range
    if max_val > min_val:
        values = np.linspace(min_val, max_val, 100)
    else:
        values = np.array([min_val])

    # Compute predicted indices based on model type
    params = segment['params']

    if model_type == 'LINEAR':
        slope = params['slope']
        intercept = params['intercept']
        predictions = slope * values + intercept
        color = 'red'
        label = 'L'
    elif model_type == 'QUADRATIC':
        a = params['a']
        b = params['b']
        c = params['c']
        predictions = a * values * values + b * values + c
        color = 'blue'
        label = 'Q'
    elif model_type == 'CUBIC':
        a = params['a']
        b = params['b']
        c = params['c']
        d = params['d']
        predictions = a * values * values * values + b * values * values + c * values + d
        color = 'orange'
        label = 'Cu'
    elif model_type == 'CONSTANT':
        predictions = np.full_like(values, params['constant_idx'])
        color = 'green'
        label = 'C'
    else:  # DIRECT
        predictions = np.full_like(values, start_idx)
        color = 'purple'
        label = 'D'

    # Plot the prediction curve
    ax.plot(predictions, values, color=color, linewidth=2, alpha=alpha, zorder=2)

    # Add model type label at the midpoint
    if len(predictions) > 0:
        mid_idx = len(predictions) // 2
        ax.text(predictions[mid_idx], values[mid_idx], label,
                fontsize=10, fontweight='bold', color=color,
                bbox=dict(boxstyle='circle,pad=0.1', facecolor='white', edgecolor=color, alpha=0.8),
                ha='center', va='center', zorder=4)

    return color, label


def plot_segment_finder_model(ax, segment_finder: Dict[str, Any], keys: np.ndarray, num_segments: int):
    """Plot the segment finder model as an overlay showing segment_index = f(value).

    Returns the (min_y, max_y) range covered by the visualization, or None if nothing plotted.
    """
    if not segment_finder or len(keys) == 0:
        return None

    model_type = segment_finder['model_type']
    params = segment_finder['params']
    max_error = segment_finder['max_error']

    # Sample values across the full key range
    min_key = np.min(keys)
    max_key = np.max(keys)

    if max_key <= min_key:
        return None

    values = np.linspace(min_key, max_key, 200)

    # Compute predicted segment indices
    if model_type == 'LINEAR':
        slope = params['slope']
        intercept = params['intercept']
        seg_predictions = slope * values + intercept
    elif model_type == 'QUADRATIC':
        a = params['a']
        b = params['b']
        c = params['c']
        seg_predictions = a * values * values + b * values + c
    elif model_type == 'CUBIC':
        a = params['a']
        b = params['b']
        c = params['c']
        d = params['d']
        seg_predictions = a * values * values * values + b * values * values + c * values + d
    elif model_type == 'EXPONENTIAL':
        a = params['a']
        b = params['b']
        c = params['c']
        seg_predictions = a * np.exp(b * values) + c
    elif model_type == 'LOGARITHMIC':
        a = params['a']
        b = params['b']
        c = params['c']
        # Handle domain: log argument must be positive
        arg = values + b
        seg_predictions = np.where(arg > 0, a * np.log(arg) + c, 0.0)
    else:
        seg_predictions = np.zeros_like(values)

    # Map segment indices to actual index positions (approximate)
    # For visualization, approximate each segment's midpoint position
    total_size = len(keys)

    # Only plot where predictions are within valid range to avoid clamping artifacts
    valid_mask = (seg_predictions >= 0) & (seg_predictions < num_segments)

    if np.any(valid_mask):
        # Use raw (unclamped) predictions for valid points only
        segment_positions_valid = seg_predictions[valid_mask] * (total_size / num_segments)
        values_valid = values[valid_mask]

        ax.plot(segment_positions_valid, values_valid, 'magenta',
                linewidth=2.5, linestyle='--',
                alpha=0.7, zorder=5, label=f'Segment Finder ({model_type}, err={max_error})')

        # Plot error band if there's any error (only for valid prediction range)
        if max_error > 0:
            error_predictions_low = seg_predictions[valid_mask] - max_error
            error_predictions_high = seg_predictions[valid_mask] + max_error

            # Clamp the error band predictions to valid segment range
            error_predictions_low = np.clip(error_predictions_low, 0, num_segments - 1)
            error_predictions_high = np.clip(error_predictions_high, 0, num_segments - 1)

            error_positions_low = error_predictions_low * (total_size / num_segments)
            error_positions_high = error_predictions_high * (total_size / num_segments)

            ax.fill_betweenx(values_valid, error_positions_low, error_positions_high,
                            color='magenta', alpha=0.15, zorder=1)

        # Return the y-range covered by the visualization
        return (np.min(values_valid), np.max(values_valid))
    else:
        # If all predictions are out of range, show a note in the legend
        # Plot a small dummy line for the legend entry
        ax.plot([], [], 'magenta', linewidth=2.5, linestyle='--',
                alpha=0.4, label=f'Segment Finder ({model_type}, err={max_error}, BAD FIT)')
        return None


def plot_error_bands(ax, segment: Dict[str, Any], keys: np.ndarray):
    """Plot error bands along the entire segment model prediction."""
    start_idx = segment['start_idx']
    end_idx = segment['end_idx']
    max_error = segment['max_error']

    if max_error == 0 or end_idx <= start_idx:
        return

    # Get value range for this segment
    min_val = segment['min_val']
    max_val = segment['max_val']

    # Create dense sample of values in this segment's range
    if max_val > min_val:
        values = np.linspace(min_val, max_val, 100)
    else:
        values = np.array([min_val])

    # Compute predicted indices based on model type
    model_type = segment['model_type']
    params = segment['params']

    if model_type == 'LINEAR':
        slope = params['slope']
        intercept = params['intercept']
        predictions = slope * values + intercept
    elif model_type == 'QUADRATIC':
        a = params['a']
        b = params['b']
        c = params['c']
        predictions = a * values * values + b * values + c
    elif model_type == 'CUBIC':
        a = params['a']
        b = params['b']
        c = params['c']
        d = params['d']
        predictions = a * values * values * values + b * values * values + c * values + d
    elif model_type == 'CONSTANT':
        predictions = np.full_like(values, params['constant_idx'])
    else:  # DIRECT
        predictions = np.full_like(values, start_idx)

    # Draw error band as a shaded region around the prediction line
    # fill_betweenx shades horizontally (along x-axis = index positions)
    ax.fill_betweenx(values,
                     predictions - max_error,
                     predictions + max_error,
                     color='tan', alpha=0.2, zorder=1)


def plot_index_structure(data: Dict[str, Any], output_file: Path):
    """Create visualization of index structure."""
    keys = np.array(data['keys'])
    segments = data['segments']
    size = data['size']
    num_segments = data['num_segments']
    segment_finder = data.get('segment_finder', None)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot actual keys as scatter plot
    indices = np.arange(len(keys))
    ax.scatter(indices, keys, c='black', s=10, alpha=0.6, zorder=3, label='Keys')

    # Plot segment boundaries
    for i, segment in enumerate(segments):
        start_idx = segment['start_idx']
        end_idx = segment['end_idx']

        # Vertical lines for segment boundaries
        if i == 0:
            ax.axvline(start_idx, color='gray', linestyle='--', linewidth=1, alpha=0.5,
                      label='Segment boundaries', zorder=1)
        else:
            ax.axvline(start_idx, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)

        if i == len(segments) - 1:
            ax.axvline(end_idx, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)

    # Plot error bands for segments
    for segment in segments:
        plot_error_bands(ax, segment, keys)

    # Plot segment finder model overlay (NEW!)
    sf_y_range = None
    if segment_finder:
        sf_y_range = plot_segment_finder_model(ax, segment_finder, keys, num_segments)

    # Plot model prediction curves for each segment
    model_colors = {'L': 'red', 'Q': 'blue', 'Cu': 'orange', 'C': 'green', 'D': 'purple'}
    model_counts = {'LINEAR': 0, 'QUADRATIC': 0, 'CUBIC': 0, 'CONSTANT': 0, 'DIRECT': 0}
    total_error = 0
    max_max_error = 0

    for segment in segments:
        plot_segment_model(ax, segment, keys, alpha=0.6)
        model_counts[segment['model_type']] += 1
        total_error += segment['max_error']
        max_max_error = max(max_max_error, segment['max_error'])

    avg_error = total_error / num_segments if num_segments > 0 else 0

    # Add labels and title
    ax.set_xlabel('Index Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Key Value', fontsize=12, fontweight='bold')

    # Extract metadata from filename for title
    title_parts = output_file.stem.replace('index_structure_', '').split('_')
    base_title = f"JazzyIndex Structure: {' '.join(title_parts)}"

    # Add segment finder model type to title if available
    if segment_finder:
        sf_model = segment_finder['model_type']
        sf_error = segment_finder['max_error']
        title = f"{base_title} - Segment Finder: {sf_model} (max_error={sf_error})"
    else:
        title = base_title

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Create legend
    legend_elements = [
        mpatches.Patch(color='black', label='Keys (actual data)'),
        mpatches.Patch(color='magenta', label='Segment Finder (2-level learned index)'),
        mpatches.Patch(color='red', label=f'LINEAR models ({model_counts["LINEAR"]})'),
        mpatches.Patch(color='blue', label=f'QUADRATIC models ({model_counts["QUADRATIC"]})'),
        mpatches.Patch(color='orange', label=f'CUBIC models ({model_counts["CUBIC"]})'),
        mpatches.Patch(color='green', label=f'CONSTANT models ({model_counts["CONSTANT"]})'),
        mpatches.Patch(color='tan', alpha=0.2, label='Error bands (±max_error)'),
        mpatches.Patch(color='gray', label='Segment boundaries')
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # Add statistics text box
    sf_info = ""
    if segment_finder:
        sf_type = segment_finder['model_type']
        sf_error = segment_finder['max_error']
        sf_info = f"Seg Finder: {sf_type} (err={sf_error})\n"

    stats_text = f"""Statistics:
Size: {size:,}
Segments: {num_segments}
{sf_info}Avg Seg Error: {avg_error:.1f}
Max Seg Error: {max_max_error}
L: {model_counts['LINEAR']} | Q: {model_counts['QUADRATIC']} | Cu: {model_counts['CUBIC']} | C: {model_counts['CONSTANT']}"""

    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Adjust y-axis limits to ensure segment finder overlay and error bands are fully visible
    if sf_y_range is not None:
        current_ylim = ax.get_ylim()
        new_ymin = min(current_ylim[0], sf_y_range[0])
        new_ymax = max(current_ylim[1], sf_y_range[1])
        ax.set_ylim(new_ymin, new_ymax)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Created: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize JazzyIndex structure from exported JSON data'
    )
    parser.add_argument(
        'input_dir',
        type=Path,
        nargs='?',
        default=Path('index_data'),
        help='Directory containing index JSON files (default: index_data)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for plots (default: same as input_dir)'
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir if args.output_dir else input_dir

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist", file=sys.stderr)
        return 1

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = list(input_dir.glob('index_*.json'))

    if not json_files:
        print(f"Error: No index JSON files found in '{input_dir}'", file=sys.stderr)
        return 1

    print(f"Found {len(json_files)} index files to visualize")
    print(f"Output directory: {output_dir}")
    print()

    # Process each file
    success_count = 0
    error_count = 0

    for json_file in sorted(json_files):
        try:
            data = load_index_data(json_file)
            output_file = output_dir / json_file.name.replace('.json', '.png')
            plot_index_structure(data, output_file)
            success_count += 1
        except Exception as e:
            print(f"  Error processing {json_file}: {e}", file=sys.stderr)
            error_count += 1

    print()
    print(f"Visualization complete: {success_count} plots created", end='')
    if error_count > 0:
        print(f" ({error_count} errors)")
    else:
        print()

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
