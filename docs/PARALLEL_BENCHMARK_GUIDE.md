# Parallel Benchmark Execution Guide

This guide explains how to safely parallelize benchmark execution to speed up benchmark runs without affecting timing accuracy.

## The Problem

Running benchmarks is time-consuming. On systems with many cores (especially Apple Silicon Macs with P and E cores), you want to:
1. **Run multiple benchmark processes in parallel** to speed things up
2. **Pin to Performance cores only** (avoid E cores)
3. **Avoid cache contention** between parallel processes

## Quick Start

### Linux

```bash
# Run the parallel benchmark test
./scripts/analyze_parallel_benchmark_limits.sh

# Analyze results
python3 scripts/compare_parallel_results.py
```

### macOS

```bash
# Same scripts work on Mac
./scripts/analyze_parallel_benchmark_limits.sh
python3 scripts/compare_parallel_results.py
```

## Understanding Your CPU Topology

### macOS (Apple Silicon)

```bash
# Total cores
sysctl hw.ncpu

# Physical cores
sysctl hw.physicalcpu

# P cores (Performance)
sysctl hw.perflevel0.physicalcpu
sysctl hw.perflevel0.logicalcpu

# E cores (Efficiency)
sysctl hw.perflevel1.physicalcpu
sysctl hw.perflevel1.logicalcpu

# Example output for M1 Max:
# hw.ncpu: 10
# hw.physicalcpu: 10
# hw.perflevel0.physicalcpu: 8   ← 8 P cores
# hw.perflevel1.physicalcpu: 2   ← 2 E cores
```

### Linux

```bash
# CPU info
lscpu

# Cache topology
lscpu --all --extended
# or
cat /sys/devices/system/cpu/cpu*/cache/index*/shared_cpu_list

# Cores sharing L2 cache
grep . /sys/devices/system/cpu/cpu*/cache/index2/shared_cpu_list | sort -u

# Cores sharing L3 cache
grep . /sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list | sort -u
```

## CPU Affinity (Pinning to Specific Cores)

### macOS - No Native Support ❌

Unfortunately, macOS does not provide a user-space API for CPU affinity:
- **No `taskset`** equivalent
- **No `pthread_setaffinity_np()`**
- **No direct way to pin to P cores**

**Workarounds:**
1. **QoS (Quality of Service)** hints (limited control):
   ```bash
   # Run with user-interactive QoS (prefers P cores)
   nice -n -20 ./build/jazzy_index_benchmarks
   ```

2. **Disable E cores entirely** (requires SIP disable - NOT RECOMMENDED):
   This is invasive and affects the entire system.

3. **Just run fewer parallel processes**:
   Since you can't pin to P cores, running ≤ P_core_count processes should naturally prefer P cores.

**Best Practice for Mac:**
- On M1 Max (8P + 2E): Run **maximum 4-6 parallel processes**
- On M2 Pro (8P + 4E): Run **maximum 4-6 parallel processes**
- Monitor with Activity Monitor to verify P core usage

### Linux - Full Support ✅

Linux provides excellent CPU affinity control:

```bash
# Pin to specific cores (0, 1, 2, 3)
taskset -c 0-3 ./build/jazzy_index_benchmarks

# Pin to cores 0, 2, 4, 6 (avoid hyperthreads if needed)
taskset -c 0,2,4,6 ./build/jazzy_index_benchmarks

# Run parallel benchmarks on different cores
taskset -c 0-1 ./benchmark_part1 &
taskset -c 2-3 ./benchmark_part2 &
taskset -c 4-5 ./benchmark_part3 &
wait
```

**Recommended Strategy:**
1. Identify cores that DON'T share L2 cache (avoid contention)
2. Pin each benchmark process to separate core clusters
3. Leave some cores free for OS tasks

Example for 16-core system with 8 core pairs sharing L2:
```bash
# Each process gets 2 cores from different L2 clusters
taskset -c 0-1   ./benchmark_group1 &  # Cluster 0
taskset -c 4-5   ./benchmark_group2 &  # Cluster 2
taskset -c 8-9   ./benchmark_group3 &  # Cluster 4
taskset -c 12-13 ./benchmark_group4 &  # Cluster 6
wait
```

## Cache Contention

### Why It Matters

Benchmarks are **highly cache-sensitive**. If two benchmark processes share L2/L3 cache and work with similar data sizes, they can evict each other's cache lines, causing timing variations.

### How to Avoid

**Strategy 1: Separate by dataset size**
```bash
# Process 1: Small datasets (fit in L1)
./benchmark --filter="N100"

# Process 2: Medium datasets (fit in L2)
./benchmark --filter="N10000"

# Process 3: Large datasets (exceed L2)
./benchmark --filter="N1000000"
```

**Strategy 2: Separate by distribution**
```bash
# Each distribution has different access patterns
./benchmark --filter="Uniform" &
./benchmark --filter="Clustered" &
./benchmark --filter="Exponential" &
```

**Strategy 3: Pin to non-sharing cores** (Linux only)
```bash
# Check which cores share L2 cache
lscpu --all --extended

# Pin to cores that DON'T share caches
taskset -c 0,4,8,12 ...  # Different L2 clusters
```

## Testing Parallelism Limits

### Step 1: Run the Test Script

```bash
./scripts/analyze_parallel_benchmark_limits.sh
```

This runs the same benchmarks:
- Sequentially (baseline)
- With 2 parallel processes
- With 4 parallel processes
- With 8 parallel processes

### Step 2: Analyze Results

```bash
python3 scripts/compare_parallel_results.py
```

This compares timing results and reports:
- ✓ **<5% degradation**: Safe to parallelize
- ⚠️ **5-10% degradation**: Acceptable (slight cache contention)
- ❌ **>10% degradation**: Not recommended (significant interference)

### Step 3: Interpret Results

**Example output:**
```
=== 2 Parallel Processes ===
✓ JazzyIndex/Uniform/N1000/S1/FoundMiddle    10.23 ns    10.45 ns    +2.1%
✓ JazzyIndex/Uniform/N1000/S16/FoundMiddle   11.12 ns    11.34 ns    +2.0%
✓ JazzyIndex/Uniform/N1000/S64/FoundMiddle   12.56 ns    12.89 ns    +2.6%

Max timing degradation: 2.6%
✓ 2 parallel processes: SAFE (<5% degradation)
```

## Practical Recommendations

### Apple Silicon Mac (e.g., M1 Max with 8P + 2E cores)

**Goal: Run at least 3 parallel processes**

```bash
# Split benchmarks into 3-4 groups
./build/jazzy_index_benchmarks --filter="Uniform" &
./build/jazzy_index_benchmarks --filter="Exponential" &
./build/jazzy_index_benchmarks --filter="Clustered" &
wait
```

**Why this works:**
- 3-4 processes < 8 P cores (no E core spillover)
- OS scheduler naturally prefers P cores for high-CPU tasks
- Different distributions minimize cache contention

**Alternative: Split by size**
```bash
./build/jazzy_index_benchmarks --filter="N(100|1000)" &      # Small
./build/jazzy_index_benchmarks --filter="N10000" &           # Medium
./build/jazzy_index_benchmarks --filter="N(100000|1000000)" & # Large
wait
```

### Linux (e.g., 16-core Xeon)

```bash
# Check cache topology first
lscpu --all --extended

# Pin to separate core clusters (example for 4-core clusters)
taskset -c 0-3   ./benchmark --filter="Uniform" &
taskset -c 4-7   ./benchmark --filter="Exponential" &
taskset -c 8-11  ./benchmark --filter="Clustered" &
taskset -c 12-15 ./benchmark --filter="Lognormal" &
wait
```

## Monitoring During Execution

### macOS

```bash
# Watch core usage in real-time
# Open Activity Monitor → Window → CPU History
# Verify benchmarks run on P cores (cores 0-7 for M1 Max)
```

Or use command line:
```bash
# Install powermetrics (requires sudo)
sudo powermetrics --samplers cpu_power -i 1000

# Watch for "E-Cluster" vs "P-Cluster" activity
```

### Linux

```bash
# Watch CPU usage per core
htop  # Press 't' to see per-core usage

# Or use mpstat
mpstat -P ALL 1  # Update every 1 second
```

## Data Generation Parallelization

The benchmark setup can also be parallelized. For example, in `benchmark_main.cpp`:

### Current (Sequential)

```cpp
auto uniform = qi::bench::make_uniform_values(10000);
auto exponential = qi::bench::make_exponential_values(10000);
auto clustered = qi::bench::make_clustered_values(10000);
// ... takes 3x time
```

### Parallelized (Future Improvement)

```cpp
#include <future>

// Generate all distributions in parallel
auto uniform_future = std::async(std::launch::async,
    []{ return qi::bench::make_uniform_values(10000); });
auto exponential_future = std::async(std::launch::async,
    []{ return qi::bench::make_exponential_values(10000); });
auto clustered_future = std::async(std::launch::async,
    []{ return qi::bench::make_clustered_values(10000); });

// Wait for all to complete
auto uniform = uniform_future.get();
auto exponential = exponential_future.get();
auto clustered = clustered_future.get();
// ... takes ~1x time (3x speedup)
```

**Tradeoff:** This is only helpful if data generation is a significant portion of benchmark time (it usually isn't for small datasets).

## Summary Decision Tree

```
Do you have Apple Silicon (P/E cores)?
├─ YES → Can't pin to cores, run ≤ P_core_count/2 processes
│         For M1 Max (8P+2E): Use 3-4 parallel processes ✓
│
└─ NO (Linux) → Can pin to cores
    ├─ Check cache topology
    ├─ Pin to non-sharing cores
    └─ Can safely run more parallel processes

Does cache contention test show >10% degradation?
├─ YES → Reduce parallel processes OR separate by size/distribution
└─ NO → Current parallelism is safe ✓
```

## Next Steps

1. Run `./scripts/analyze_parallel_benchmark_limits.sh` on YOUR Mac
2. Check results with `python3 scripts/compare_parallel_results.py`
3. Based on results, create a parallelized benchmark runner script
4. Enjoy 3-4x faster benchmark execution! 🚀

## References

- [Google Benchmark Threading Documentation](https://github.com/google/benchmark#multithreaded-benchmarks)
- [Apple Developer: QoS Classes](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/EnergyGuide-iOS/PrioritizeWorkWithQoS.html)
- [Linux CPU Affinity: `taskset` man page](https://man7.org/linux/man-pages/man1/taskset.1.html)
