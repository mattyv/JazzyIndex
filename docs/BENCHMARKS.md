# JazzyIndex Performance Benchmarks

Performance analysis of JazzyIndex, `std::lower_bound`, PGM-index, and
RadixSpline across multiple data distributions and index configurations.

## Table of Contents
- [Overview](#overview)
- [Measured small-bucket routing results](#measured-small-bucket-routing-results-2026-10-22)
- [Measured routing and batch results](#measured-routing-and-batch-results-2026-10-20)
- [Benchmark Setup](#benchmark-setup)
- [Performance by Segment Group](#performance-by-segment-group)
- [Distribution Analysis](#distribution-analysis)
- [Segment Count Selection Guide](#segment-count-selection-guide)
- [Range Function Benchmarks](#range-function-benchmarks)
- [Running Benchmarks](#running-benchmarks)
- [Key Takeaways](#key-takeaways)

## Overview

JazzyIndex is benchmarked against `std::lower_bound` (binary search) across 9 different data distributions to evaluate:
- **Query performance:** Nanoseconds per lookup
- **Segment count impact:** How 1-512 segments affect performance
- **Distribution sensitivity:** Which data patterns benefit most
- **Query scenarios:** Found (middle/end) vs. Not Found cases

All benchmarks use Google Benchmark on sorted `uint64_t` arrays. The historical
plots cover 100 to 10,000 elements; the competitive suite extends to one million.

The committed plots are historical, hardware-specific results and were not regenerated for the current correctness release. Re-run the suite on an otherwise idle target machine before using these figures for capacity planning or release claims.

### Competitive container suite

`benchmark_competitors` runs identical deterministic query streams generated
with 262,144 queries and a 90% target hit probability against:

- JazzyIndex with 64, 256, and 512 segments
- PGM-index with epsilon 16, 64, and 256
- RadixSpline with maximum error 16, 64, and 256, using both 12- and 18-bit radix tables
- `std::lower_bound`
- `std::set` and `std::unordered_set` membership
- Four independent JazzyIndex queries through `find_batch` (256 initial segments)
- Four independent RadixSpline queries with interleaved bounded correction (R12/E16 and R18/E16)

Each lookup reports `index_bytes`, `backing_bytes`, and their sum, `total_bytes`.
JazzyIndex counts its object and allocated metadata capacity. PGM and RadixSpline
use their library-reported metadata sizes, which may exclude spare capacity.
Sorted-array implementations count the backing vector's allocated capacity;
the standard containers own their keys and report live node/bucket allocations
with zero separate backing bytes. Query storage and allocator bookkeeping are
excluded. Construction is benchmarked separately for the learned indexes, with
destruction outside the timed region.

Batch4 timing is **per batch of four queries**: divide its nanoseconds by four
for throughput per query. It does not measure individual query latency.
Every implementation validates its results against `std::lower_bound` before
timing; batch validation checks every lane.
PGM-index and RadixSpline are fetched only when `BUILD_BENCHMARKS=ON` and are
pinned to exact upstream revisions.
Build measurements are serial for all three indexes; PGM-index's optional
OpenMP construction path is intentionally not enabled in this target.

For some synthetic long duplicate runs, preflight found that the upstream
RadixSpline search window did not cover the first occurrence needed by an exact
lower-bound lookup. The competitive suite therefore maps each occurrence to an
order-preserving unique key
(`key * (N + 1) + occurrence`) for every implementation, rejecting an input if
that checked transform would overflow `uint64_t`. The primary JazzyIndex suite
continues to test the original duplicate-heavy data.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON
cmake --build build --target benchmark_competitors

# 10K rows across all distributions
./build/benchmark_competitors

# 10K, 100K, and 1M rows
./build/benchmark_competitors --full-benchmarks
```

These synthetic distributions make implementations directly comparable but do
not replace validation on real application data or SOSD datasets.
The unique-key transform can also interact poorly with integer hashing: on the
tested libc++ implementation, the one-million-key Zipf fixture causes heavy
bucket collisions. Its hash-table result should not be generalized to other
key encodings or hash functions.

## Measured small-bucket routing results (2026-10-22)

RadixSpline's small-range scan suggested a narrower optimization for JazzyIndex:
return the segment immediately when the directory identifies just one candidate,
and scan intervals of two to four candidates. Larger intervals retain binary
search. This applies to the existing unsigned, identity-key, duplicate-free
routing path; the selected segment still verifies the key and rejects gaps.

The change adds no index fields or allocations. Retained index storage and backing
array capacity are identical to commit `f74d418` on every measured fixture. The
default `uint64_t` index object remains 120 bytes.

Final measurements use the same M2, compiler, optimization flags, deterministic
query streams, and seven randomized repetitions described below. The broad
comparison covers 10K and 1M elements, all nine distributions, and both original
and uniquely encoded keys. The competitive comparison uses one million uniquely
encoded keys and now includes **RadixSpline Batch4**, with all four search bounds
computed before interleaving their exact corrections. Batch results are throughput
per query, not isolated lookup latency.

The initial broad run included noisy repetitions (up to 47% CV), so the entire
72-row comparison was repeated with the same settings. Both runs are preserved
separately in the summary. The confirmation run had at most 7.1% CV and measured
these scalar results at one million uniquely encoded keys:

| Distribution | Previous ns | New ns | Lookup time reduction |
|---|---:|---:|---:|
| Uniform | 12.50 | 12.58 | -0.6% |
| Exponential | 14.72 | 12.59 | 14.5% |
| Clustered | 39.55 | 36.39 | 8.0% |
| Lognormal | 15.21 | 13.33 | 12.4% |
| Zipf | 32.76 | 29.65 | 9.5% |
| Mixed | 14.14 | 11.86 | 16.1% |
| Quadratic | 37.18 | 33.82 | 9.0% |
| ExtremePoly | 38.16 | 33.82 | 11.4% |
| InversePoly | 40.89 | 37.64 | 8.0% |

The nonuniform unique fixtures improve by roughly 8–16%; Uniform is effectively
unchanged within run variation. The broad and competitor runs use different
executables and ran separately, so compare implementations within each table.

The competitive run measured the following median CPU nanoseconds per query.
JazzyIndex uses 256 initial segments. RadixSpline columns select the fastest
tested configuration independently from six scalar and two batch settings.

| Distribution | Jazzy scalar | RadixSpline scalar | Jazzy Batch4 | RadixSpline Batch4 |
|---|---:|---:|---:|---:|
| Uniform | 12.94 | 15.37 | 12.30 | 10.89 |
| Exponential | 12.74 | 17.92 | 12.98 | 11.37 |
| Clustered | 35.93 | 25.57 | 36.19 | 18.51 |
| Lognormal | 13.66 | 18.47 | 13.21 | 11.85 |
| Zipf | 30.38 | 26.84 | 26.84 | 23.89 |
| Mixed | 12.05 | 17.67 | 12.09 | 11.39 |
| Quadratic | 34.30 | 25.83 | 26.84 | 19.53 |
| ExtremePoly | 34.45 | 25.90 | 29.86 | 20.38 |
| InversePoly | 38.53 | 28.21 | 32.67 | 20.52 |

JazzyIndex wins four scalar comparisons; RadixSpline wins all nine batch
comparisons. The new optimization changes scalar routing only. The separate
JazzyIndex batch router remains an optimization target, and batching does not
help every distribution. Clustered is the largest remaining batch gap here.
These are comparisons of the tested implementations, not upper bounds on what
either index can achieve with further tuning.

Both winning RadixSpline columns use R12/E16 for Uniform, Exponential, and Mixed,
and R18/E16 elsewhere. Their total reported storage, including the key array,
compares as follows (decimal MB):

| Distribution | JazzyIndex | Winning RadixSpline |
|---|---:|---:|
| Uniform | 8.017 | 8.015 |
| Exponential | 8.398 | 8.013 |
| Clustered | 9.028 | 9.000 |
| Lognormal | 8.468 | 8.860 |
| Zipf | 9.055 | 9.021 |
| Mixed | 8.393 | 8.012 |
| Quadratic | 8.934 | 9.029 |
| ExtremePoly | 9.011 | 9.043 |
| InversePoly | 9.011 | 9.043 |

JazzyIndex scalar also beats the tested PGM, binary search, tree set, and hash
set on all nine fixtures. The fastest PGM settings report 8.00–8.03 MB total;
the hash and tree sets retain 32 and 40 MB. PGM and RadixSpline report logical
storage that may omit spare capacity, while JazzyIndex counts allocated capacity.
These figures exclude peak construction memory and process RSS. The Zipf hash
result retains the encoding/collision limitation described above.

All seven samples, coefficients of variation, and memory counters are retained
in the [machine-readable summary](benchmark-summary-2026-10-22.json). Some
competitor cases had noticeable variation, including 12% CV for the winning
Clustered RadixSpline batch; close timing differences should not be treated as
conclusive.

All results are checked against exact lookup before timing. Normal tests passed
265/265, debug tests 280/280, and ASan/UBSan/float-cast-overflow tests 265/265.
New regressions cover high-origin gaps, the final unsigned bucket, and several
segment boundaries sharing one high-key bucket.

The experiment also tested packed routing keys, skipping the initial prediction
probe, cached endpoint checks, and cached spline slopes. None was promoted. The
packed-key variant added memory and was slower; cached endpoint checks did not
show a consistent benefit over the selected routing change. Noisy prediction and
spline runs were excluded from fine-grained comparisons. The retained change was
supported by separate routing and combined screens before final measurement.

## Measured routing and batch results (2026-10-20)

This earlier snapshot predates the small-bucket optimization and the RadixSpline
batch comparator. Its batch-versus-scalar comparisons do not establish a lead
over batched RadixSpline.

These measurements use an Apple M2 with 8 GiB RAM, macOS 27.0.1, Apple Clang
21.0.0, the Xcode 26.5 SDK, C++20, `-O3`, and `-DNDEBUG`. Each case uses
262,144 deterministic queries with a 90% target hit probability, seven randomly
interleaved repetitions, and a minimum 0.1-second timing interval. Numbers are
median CPU nanoseconds per query. Builds, tests, and profiling ran separately
from timing. CPU affinity was unavailable; the OS scheduled execution.

The changes add a logarithmic radix directory for eligible unsigned keys,
direct offsets for contiguous unique integers, and a four-query throughput API.
Model evaluation stays inline. Against commit `40e0e9c`, the scalar path improved
34 of 36 measured cases across nine distributions, original and uniquely encoded
keys, and 10K and 1M elements. The two regressions were noncontiguous uniform
keys: approximately 1% at 10K and 2% at 1M.

All repetition samples, medians, variation, and memory measurements are retained
in the [machine-readable summary](benchmark-summary-2026-10-20.json).

At one million elements, the original key distributions measured:

| Distribution | Previous scalar ns | New scalar ns | Lookup time reduction |
|---|---:|---:|---:|
| Uniform, contiguous | 8.00 | 3.41 | 57% |
| Exponential | 11.34 | 6.48 | 43% |
| Clustered | 58.14 | 35.95 | 38% |
| Lognormal | 14.89 | 8.53 | 43% |
| Zipf | 36.44 | 20.18 | 45% |
| Mixed | 10.59 | 6.40 | 40% |
| Quadratic | 39.27 | 14.95 | 62% |
| ExtremePoly | 40.86 | 16.71 | 59% |
| InversePoly | 39.85 | 20.85 | 48% |

### Comparison with other containers

The table below uses one million **uniquely encoded** keys and identical query
streams. JazzyIndex uses the default 256 initial segments. PGM and RadixSpline
columns show their fastest tested configuration for each distribution, chosen
from all three PGM settings and all six RadixSpline settings. Batch4 reports
throughput per query, not the latency of an isolated lookup. Standard containers
answer membership; sorted-array implementations return an exact position.

| Distribution | Jazzy scalar | Jazzy Batch4 / query | Best PGM | Best RadixSpline | Binary search | Hash set | Tree set |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform | 12.78 | 12.07 | 18.92 | 15.24 | 53.08 | 14.60 | 153.28 |
| Exponential | 14.75 | 12.56 | 30.32 | 17.60 | 50.42 | 30.78 | 177.59 |
| Clustered | 40.07 | 35.89 | 42.58 | 25.23 | 53.10 | 47.00 | 152.90 |
| Lognormal | 15.28 | 12.93 | 30.74 | 18.35 | 51.24 | 45.21 | 192.26 |
| Zipf | 33.31 | 26.71 | 43.15 | 26.47 | 52.86 | 311.70 | 162.57 |
| Mixed | 13.92 | 11.90 | 29.27 | 17.71 | 50.77 | 25.15 | 185.64 |
| Quadratic | 37.44 | 26.03 | 47.46 | 25.41 | 52.54 | 39.98 | 152.74 |
| ExtremePoly | 38.02 | 29.29 | 44.63 | 25.35 | 53.15 | 100.35 | 155.85 |
| InversePoly | 40.45 | 32.01 | 48.36 | 27.79 | 53.00 | 160.72 | 152.23 |

PGM's winner is E16 except for Clustered, where E256 wins. RadixSpline's winner
is R12/E16 for Uniform, Exponential, and Mixed, and R18/E16 elsewhere.
JazzyIndex scalar measured faster than PGM, binary search, and both standard
containers on all nine fixtures, and faster than RadixSpline on four. RadixSpline
retains a substantial Clustered lead. Batch4 reduces JazzyIndex time per query by
approximately 6–30%, bringing Zipf and Quadratic within about 1% and 2.5% of
RadixSpline respectively; those close results do not establish a win.

Counting the backing array, default JazzyIndex retains 8.02–9.05 MB here, the
winning PGM settings 8.00–8.03 MB, and the winning RadixSpline settings
8.01–9.04 MB. The hash set retains 32.00 MB and the tree set 40.00 MB.
PGM and RadixSpline figures use upstream logical size reports; JazzyIndex and
the standard containers count allocated capacity. These are retained storage
figures, not peak construction memory or process RSS. The uniquely encoded Zipf
fixture causes hash collisions on this libc++ implementation; its hash timing
does not establish general superiority over hashing.

Batching is optional. In a separate scalar-versus-batch run on original,
duplicate-heavy Zipf keys, Batch4 was about 7% slower per query (21.10 versus
19.77 ns). It should be selected using representative measurements.

Reproduce the competitive lookup run with:

```bash
./build/benchmark_competitors --full-benchmarks \
  --benchmark_filter='^(Lookup|Membership)/.*/N1000000/RandomMixed$' \
  --benchmark_min_time=0.1s --benchmark_repetitions=7 \
  --benchmark_enable_random_interleaving=true \
  --benchmark_display_aggregates_only=true \
  --benchmark_out=competitors.json --benchmark_out_format=json
```

### Hardware-counter evidence

Instruments CPU Counters profiled the scalar lookup loop on one million uniquely
encoded Clustered keys. Both profiles exclude the first two seconds and aggregate
main-thread samples weighted by cycle count.

| Fraction of sustainable instruction bandwidth | Previous | New |
|---|---:|---:|
| Discarded | 57.09% | 36.67% |
| Instruction delivery | 22.67% | 2.46% |
| Instruction processing | 1.34% | 37.60% |
| Useful | 18.90% | 23.26% |

These categories are **not branch-miss rates or wall-time shares**. The new
profile contains 23,201 complete samples; one Efficiency-core sample with a
missing delivery counter was excluded as a whole. The changed balance supports
investigating local search and dependent loads next, particularly on Clustered
data where RadixSpline still leads. The timing results, rather than these
fractions alone, establish the lookup improvement.

Larger flat directories, prefetching, and more aggressive segment refinement were
also tried. They did not provide enough consistent speed benefit for their memory
or complexity costs. The retained batch API exposes independent lookups so the
processor can overlap their work without adding persistent index memory.

These are repeated-stream lookup measurements on synthetic fixtures, not a
cold-cache test or a guarantee for other hardware. The unique-key encoding used
by the competitive suite changes spacing and duplicates, so its timings must
be compared within that suite.

## Benchmark Setup

### Test Distributions

1. **Uniform** - Evenly spaced values (ideal case)
2. **Exponential** - Exponential decay distribution
3. **Clustered** - 5 Gaussian clusters
4. **Lognormal** - Log-normal distribution (realistic workload)
5. **Zipf** - Power-law distribution (web traffic, word frequencies)
6. **Mixed** - 50% exponential + 50% clustered
7. **Quadratic** - S-curve using tanh (smooth sigmoid)
8. **ExtremePoly** - x⁵ polynomial (explosive end curvature)
9. **InversePoly** - 1-(1-x)⁵ (steep start curvature)

### Query Scenarios

Each distribution is tested with 4 query patterns:
- **FoundMiddle (`--` dashed):** Search for value at middle position
- **FoundEnd (`:` dotted):** Search for value at end position
- **NotFound (`-.` dash-dot):** Search for value not in dataset
- **RandomMixed:** A deterministic rotation of 90% hits and 10% interior/out-of-range misses

### Segment Counts Tested

10 configurations: **1, 2, 4, 8, 16, 32, 64, 128, 256, 512**

For clarity, performance plots are split into three groups:

## Performance by Segment Group

### Low Segment Counts (1, 2, 4, 8)

![Low Segments Performance](images/benchmarks/jazzy_benchmarks_low.png)

**Characteristics:**
- Fewer segments = less memory overhead
- Larger error bounds per segment
- More queries fall back to exponential search
- **Best for:** Very small datasets (< 1K elements) or memory-constrained systems

**Key Observations:**
- Segment count 1 (purple) has the broadest fallback search window and is mainly useful for tiny datasets
- Segments 4-8 show reasonable performance for simple distributions (Uniform, Exponential)
- Complex distributions (Zipf, Lognormal) struggle with low segment counts

---

### Medium Segment Counts (16, 32, 64, 128)

![Medium Segments Performance](images/benchmarks/jazzy_benchmarks_medium.png)

**Characteristics:**
- **Sweet spot** for most workloads
- Good balance of accuracy vs. overhead
- Consistently beats `std::lower_bound` (black X line)
- Quadratic models provide measurable benefits

**Key Observations:**
- **Segments 64-128** are often a useful low-overhead range; the API default remains 256
- All JazzyIndex configurations beat binary search on most distributions
- NotFound queries (dash-dot lines) often perform best due to early termination
- Uniform distribution shows minimal difference between segment counts (already optimal)

**Performance Wins:**
- **Exponential:** 2-3x faster than `std::lower_bound`
- **Zipf:** 2-4x faster
- **Quadratic/Poly:** 1.5-2x faster (quadratic models shine here)
- **Uniform:** Comparable (both are already near-optimal)

---

### High Segment Counts (256, 512)

![High Segments Performance](images/benchmarks/jazzy_benchmarks_high.png)

**Characteristics:**
- Very tight error bounds
- Minimal exponential search iterations
- Higher memory cost
- Segment lookup overhead becomes noticeable

**Key Observations:**
- **Segments 256 (green)** vs **512 (red):** Marginal gains, not worth 2x memory for most cases
- For datasets > 10K elements, high segment counts become more attractive
- Diminishing returns: 512 segments rarely beats 256 significantly

---

## Distribution Analysis

### Uniform Distribution
- **Best Case for JazzyIndex**
- Perfect linear fit, minimal error bounds
- All segment counts perform well
- Uses arithmetic segment selection and a small local verification search in the common case

**Recommendation:** Use any segment count 32+

---

### Exponential Distribution
- **Excellent Fit**
- JazzyIndex **2-3x faster** than binary search
- Quadratic models selected for steep initial curve
- Performance improves with segment count

**Recommendation:** Segments 64-128 for best results

---

### Clustered Distribution
- **Moderate Challenge**
- 5 distinct value clusters create discontinuities
- Linear models within clusters, larger errors at boundaries
- Still beats `std::lower_bound` by ~30-50%

**Recommendation:** Segments 128-256 to handle cluster boundaries

---

### Lognormal Distribution
- **Realistic Workload**
- Common in real-world systems (file sizes, response times)
- JazzyIndex **1.5-2x faster**
- Quadratic models help with right tail

**Recommendation:** Segments 64-128

---

### Zipf Distribution
- **Heavy-Tailed Challenge**
- Power-law distribution (α=1.2) common in web traffic
- Steep initial curve, long tail
- JazzyIndex **2-4x faster** due to quadratic model selection

**Recommendation:** Segments 128-256 for optimal performance

---

### Mixed Distribution
- **Complex Hybrid**
- Combines exponential decay + clustered regions
- Tests adaptability of model selection
- JazzyIndex **2x faster** on average

**Recommendation:** Segments 128-256

---

### Quadratic (S-Curve)
- **Designed for Quadratic Models**
- Smooth sigmoid curve using `tanh`
- Nearly 100% quadratic model selection at low segment counts
- JazzyIndex **1.5-2x faster**

**Recommendation:** Segments 64-128 (higher counts force more linear models)

---

### ExtremePoly (x⁵)
- **Explosive End Curve**
- Flat beginning, extreme curvature at end
- High segment counts needed to handle right side
- Quadratic models essential

**Recommendation:** Segments 128-256

---

### InversePoly (1-(1-x)⁵)
- **Steep Start Curve**
- Mirror of ExtremePoly
- Quadratic models at beginning, linear at end
- Similar performance characteristics

**Recommendation:** Segments 128-256

---

## Segment Count Selection Guide

| Dataset Size | Memory Priority | Latency Priority | Recommended Segments |
|--------------|----------------|------------------|---------------------|
| < 1K         | High           | Low              | 32-64               |
| 1K - 10K     | High           | Low              | 64-128              |
| 1K - 10K     | Low            | High             | 128-256             |
| 10K - 100K   | High           | Low              | 128-256             |
| 10K - 100K   | Low            | High             | 256-512             |
| 100K - 1M    | High           | Low              | 256-512             |
| 100K - 1M    | Low            | High             | 512-2048            |
| > 1M         | -              | -                | 512-2048            |

### Rule of Thumb

**Default:** Use **256 segments** unless you have specific constraints.

- **Reduce** to 64-128 if memory is tight
- **Increase** to 512-1024 for latency-critical applications with large datasets
- **Minimize** to 32 for tiny datasets (< 500 elements)

### Memory Footprint

Use `sizeof(index) + index.size_in_bytes()` for retained index storage, then add
the backing array's allocated capacity. The template segment count is an initial
split: refinement can increase the count, and vector capacity can exceed it.
On this build, a `uint64_t` segment occupies 64 bytes and the default index object
occupies 120 bytes, up from 88.

Eligible nonuniform unsigned indexes also retain a routing directory whose size
depends on key span. It uses at most 1,703,940 bytes for the full `uint64_t` span.
Fresh uniform indexes and indexes using custom comparators or extractors allocate
no directory. `find_batch<4>` adds 256 bytes of lane state on this target's stack,
plus its input/output arrays; it adds no persistent index storage.

For the measured one-million-key original fixtures, total retained index storage
ranges from 16,504 to 402,460 bytes, plus the 8,000,000-byte backing array.
Unique encoding widens the key span: those indexes use 16,504 to 1,054,860 bytes
plus the same backing array. This is a deliberate speed-for-memory tradeoff.
At 10K uniquely encoded keys, index storage reaches 491,328 bytes for an
80,000-byte array, compared with 16,472 bytes before this change.

Directory construction also adds a duplicate scan and table initialization to
eligible builds. Lookup gains do not imply faster construction.

---

## Range Function Benchmarks

JazzyIndex supports `equal_range`, `find_lower_bound`, and `find_upper_bound`. The current implementations use the corresponding standard-library algorithms over the complete indexed range, providing O(log N) bounds and STL-equivalent behavior.

### Range Functions Tested

- **`equal_range(value)`** - Returns `[lower, upper)` pair for all elements equal to value
- **`find_lower_bound(value)`** - Returns pointer to first element ≥ value
- **`find_upper_bound(value)`** - Returns pointer to first element > value

### Benchmark Coverage

Range function benchmarks test the same distributions and scenarios as the main benchmarks:
- **Distributions:** All 9 distributions (Uniform, Exponential, Clustered, etc.)
- **Segment counts:** All 10 counts (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
- **Scenarios:** FoundMiddle, FoundEnd, NotFound
- **Dataset sizes:** 100, 1,000, 10,000 elements

This results in **2,673 total benchmark cases** (compared to 891 for the main `find()` benchmarks).

### Running Range Benchmarks

```bash
# Build the range benchmark executable
cmake --build build --target benchmark_range_functions

# Run with multithreading (4 threads per benchmark)
./build/benchmark_range_functions \
  --benchmark_format=json \
  --benchmark_out=build/jazzy_range_benchmarks.json \
  --benchmark_threads=4

# Generate separate plots for each function
./scripts/plot_range_benchmarks.sh build/jazzy_range_benchmarks.json
```

**Output:**
- `docs/images/benchmarks/jazzy_equalrange_benchmarks_{low,medium,high}.png`
- `docs/images/benchmarks/jazzy_lowerbound_benchmarks_{low,medium,high}.png`
- `docs/images/benchmarks/jazzy_upperbound_benchmarks_{low,medium,high}.png`

Learned acceleration is deferred until representative profiles justify the extra search logic. See [DEFERRED_WORK.md](DEFERRED_WORK.md#2-learned-acceleration-for-range-queries) for its entry criteria and correctness requirements.

---

## Running Benchmarks

### Quick Benchmark (up to 10K elements)
```bash
cmake --build build --target plot_benchmarks
```

Generates:
- `build/jazzy_benchmarks.json` (raw data)
- `docs/images/benchmarks/jazzy_benchmarks_low.png`
- `docs/images/benchmarks/jazzy_benchmarks_medium.png`
- `docs/images/benchmarks/jazzy_benchmarks_high.png`

### Full Benchmark (up to 1M elements)
```bash
cmake --build build --target plot_benchmarks_full
```

⚠️ **Warning:** Full benchmarks take 15-30 minutes to run.

### Custom Benchmark
```bash
# Run specific distributions
./build/jazzy_index_benchmarks --benchmark_filter="Zipf"

# Run specific segment counts
./build/jazzy_index_benchmarks --benchmark_filter="S64"

# JSON output
./build/jazzy_index_benchmarks \
    --benchmark_format=json \
    --benchmark_out=results.json
```

### Generate Documentation Plots
```bash
./scripts/generate_docs_plots.sh
```

This script:
1. Runs performance benchmarks
2. Generates all three segment group plots
3. Creates index structure visualizations
4. Copies everything to `docs/images/` for commit

---

## Key Takeaways

1. The historical fixed-query runs show that JazzyIndex can outperform `std::lower_bound`; they are not a universal guarantee.
2. Segment count is workload-dependent and should be selected with representative measurements.
3. Curved distributions can benefit from quadratic models.
4. Randomized hit/miss queries should accompany fixed-position cases in new reports.
5. Segment count trades index memory for narrower local searches.

**Next Steps:**
- See [VISUALIZATIONS.md](VISUALIZATIONS.md) to understand how model selection works
- See [MODEL_SELECTION.md](MODEL_SELECTION.md) for mathematical details
- See [README.md](../README.md) for usage guide and API documentation
