#!/usr/bin/env python3
"""
Download and prepare SOSD (Searching on Sorted Data) benchmark datasets.

This script downloads real-world datasets used in learned index benchmarks,
subsamples them to a manageable size (200M elements by default), and saves
them in binary format for JazzyIndex benchmarks.

Datasets are cached in benchmarks/datasets/ (gitignored).
"""

import argparse
import hashlib
import os
import struct
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

# SOSD dataset metadata
# See: https://github.com/learnedsystems/SOSD
DATASETS = {
    "books": {
        "name": "Books (Amazon book popularity)",
        "url": "https://dataverse.harvard.edu/api/access/datafile/3407241",
        "size_mb": 762,
        "num_elements": 100000000,  # 100M uint64_t values
        "description": "Sorted book IDs from Amazon sales rankings (mostly uniform)",
    },
    "osm": {
        "name": "OpenStreetMap (Cell IDs)",
        "url": "https://dataverse.harvard.edu/api/access/datafile/3407243",
        "size_mb": 4577,
        "num_elements": 800000000,  # 800M uint64_t values
        "description": "Cell IDs from OpenStreetMap (highly skewed, Hilbert curve ordering)",
    },
    "wiki": {
        "name": "Wikipedia (Revision IDs)",
        "url": "https://dataverse.harvard.edu/api/access/datafile/3407244",
        "size_mb": 3052,
        "num_elements": 200000000,  # 200M uint64_t values
        "description": "Wikipedia revision IDs (time-series with temporal clustering)",
    },
    "fb": {
        "name": "Facebook (User IDs)",
        "url": "https://dataverse.harvard.edu/api/access/datafile/3407242",
        "size_mb": 7629,
        "num_elements": 200000000,  # 200M uint64_t values
        "description": "Facebook user IDs (clustered, power-law distribution)",
    }
}

CACHE_DIR = Path(__file__).parent.parent / "benchmarks" / "datasets"
CHUNK_SIZE = 8192


def download_file(url: str, dest_path: Path, expected_size_mb: Optional[float] = None) -> bool:
    """Download a file with progress reporting."""
    print(f"Downloading from: {url}")
    print(f"Destination: {dest_path}")

    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))

            if expected_size_mb:
                expected_bytes = int(expected_size_mb * 1024 * 1024)
                if total_size > 0 and abs(total_size - expected_bytes) > expected_bytes * 0.1:
                    print(f"Warning: Expected ~{expected_size_mb}MB but server reports {total_size / 1024 / 1024:.1f}MB")

            downloaded = 0
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_downloaded = downloaded / 1024 / 1024
                        mb_total = total_size / 1024 / 1024
                        print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f} / {mb_total:.1f} MB)", end='', flush=True)

            print()  # New line after progress
            print(f"Download complete: {downloaded / 1024 / 1024:.1f} MB")
            return True

    except Exception as e:
        print(f"\nError downloading file: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def load_binary_uint64(file_path: Path, max_elements: Optional[int] = None) -> List[int]:
    """Load uint64_t values from binary file."""
    print(f"Loading data from {file_path}...")

    file_size = file_path.stat().st_size
    num_elements = file_size // 8  # 8 bytes per uint64_t

    if max_elements and num_elements > max_elements:
        num_elements = max_elements

    values = []
    with open(file_path, 'rb') as f:
        for i in range(num_elements):
            data = f.read(8)
            if len(data) < 8:
                break
            value = struct.unpack('<Q', data)[0]  # Little-endian uint64
            values.append(value)

            if (i + 1) % 10_000_000 == 0:
                print(f"  Loaded {i + 1:,} elements...")

    print(f"Loaded {len(values):,} elements")
    return values


def subsample_dataset(values: List[int], target_size: int) -> List[int]:
    """Subsample dataset to target size (uniform sampling)."""
    if len(values) <= target_size:
        return values

    print(f"Subsampling from {len(values):,} to {target_size:,} elements...")

    # Uniform sampling - take every Nth element
    step = len(values) / target_size
    sampled = [values[int(i * step)] for i in range(target_size)]

    # Ensure still sorted
    sampled.sort()

    return sampled


def save_binary_uint64(values: List[int], file_path: Path) -> None:
    """Save uint64_t values to binary file."""
    print(f"Saving {len(values):,} elements to {file_path}...")

    with open(file_path, 'wb') as f:
        for value in values:
            f.write(struct.pack('<Q', value))  # Little-endian uint64

    file_size_mb = file_path.stat().st_size / 1024 / 1024
    print(f"Saved: {file_size_mb:.1f} MB")


def prepare_dataset(dataset_name: str, target_size: int = 20_000_000, force: bool = False) -> Optional[Path]:
    """Download and prepare a dataset."""

    if dataset_name not in DATASETS:
        print(f"Error: Unknown dataset '{dataset_name}'")
        print(f"Available datasets: {', '.join(DATASETS.keys())}")
        return None

    dataset_info = DATASETS[dataset_name]

    # Create cache directory
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # File paths
    raw_file = CACHE_DIR / f"{dataset_name}_raw.bin"
    processed_file = CACHE_DIR / f"{dataset_name}_{target_size // 1_000_000}M.bin"

    # Check if processed file already exists
    if processed_file.exists() and not force:
        file_size_mb = processed_file.stat().st_size / 1024 / 1024
        print(f"Dataset already prepared: {processed_file} ({file_size_mb:.1f} MB)")
        print("Use --force to re-download and re-process")
        return processed_file

    # Download raw dataset if needed
    if not raw_file.exists() or force:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_info['name']}")
        print(f"Description: {dataset_info['description']}")
        print(f"Download size: ~{dataset_info['size_mb']} MB")
        print(f"Elements: {dataset_info['num_elements']:,}")
        print(f"{'='*70}\n")

        if not download_file(dataset_info['url'], raw_file, dataset_info['size_mb']):
            return None
    else:
        print(f"Using cached raw data: {raw_file}")

    # Load and subsample
    print(f"\nProcessing dataset...")
    values = load_binary_uint64(raw_file, dataset_info['num_elements'])

    if len(values) == 0:
        print("Error: No data loaded")
        return None

    # Verify sorted
    if values != sorted(values):
        print("Warning: Data not sorted, sorting now...")
        values.sort()

    # Subsample to target size
    if len(values) > target_size:
        values = subsample_dataset(values, target_size)

    # Save processed dataset
    save_binary_uint64(values, processed_file)

    print(f"\n✓ Dataset ready: {processed_file}")
    print(f"  Elements: {len(values):,}")
    print(f"  Range: [{min(values):,} ... {max(values):,}]")
    print(f"  Size: {processed_file.stat().st_size / 1024 / 1024:.1f} MB")

    return processed_file


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare SOSD benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and prepare Wikipedia dataset (200M elements)
  %(prog)s wiki

  # Download OpenStreetMap dataset (200M subsampled from 800M)
  %(prog)s osm

  # Prepare smaller 50M element subset
  %(prog)s wiki --size 50000000

  # Force re-download and re-process
  %(prog)s wiki --force

Available datasets:
""" + "\n".join(f"  {name}: {info['name']}" for name, info in DATASETS.items())
    )

    parser.add_argument(
        "dataset",
        choices=list(DATASETS.keys()),
        help="Dataset to download"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=200_000_000,
        help="Target number of elements (default: 200M)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-process even if cached"
    )

    args = parser.parse_args()

    result = prepare_dataset(args.dataset, args.size, args.force)

    if result:
        print(f"\n{'='*70}")
        print("SUCCESS! Dataset is ready for benchmarking.")
        print(f"{'='*70}")
        return 0
    else:
        print(f"\n{'='*70}")
        print("FAILED to prepare dataset.")
        print(f"{'='*70}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
