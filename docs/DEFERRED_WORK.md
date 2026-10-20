# Deferred work

This ledger records product work intentionally left out of the 0.1 correctness and packaging pass. These are not hidden release blockers: each item has a safe current behavior, a reason to wait, a trigger for reopening it, and completion criteria.

## 1. Built-in custom executor or persistent thread pool

### Current behavior

`build_parallel()` uses a bounded set of `std::async` workers: no more than the number of segment tasks or `std::thread::hardware_concurrency()`, whichever is smaller. Applications that already own a scheduler can call `prepare_build_tasks()`, execute those independent tasks themselves, and pass the ordered results to `finalize_build()`.

### Why it is deferred

A library-owned persistent pool adds shutdown rules, queueing policy, cancellation, exception propagation, oversubscription choices, and another public configuration surface. The existing bounded convenience path fixes the unbounded-thread bug, while the task API covers advanced integrations without making JazzyIndex own application-wide scheduling policy.

### Reopen when

- Profiles show thread creation is a material part of build latency on supported large datasets.
- Multiple real consumers request integration with an executor they cannot express through the task API.
- Repeated builds are a supported primary workload rather than an occasional preprocessing step.

### Completion criteria

- Start from measured workloads and publish the benchmark method, not only a synthetic best case.
- Accept an application-owned executor or reuse a persistent pool without adding a mandatory dependency.
- Preserve the current worker bound, strong exception guarantee, deterministic result ordering, and empty/single-element behavior.
- Add concurrency, exception, shutdown, ThreadSanitizer, and repeated-build tests.
- Document ownership and lifetime rules for the executor.

## 2. Learned acceleration for range queries

### Current behavior

`find_lower_bound()`, `find_upper_bound()`, and `equal_range()` delegate to `std::lower_bound`, `std::upper_bound`, and `std::equal_range` over the full indexed range. This is intentionally boring O(log N) behavior with standard-library edge-case handling.

### Why it is deferred

The previous learned-window implementation was wrong for duplicate runs crossing segment boundaries, out-of-range queries, and large integer keys that lose precision when converted to `double`. Restoring STL behavior is smaller and safer. A second optimized search path is not warranted until range queries are shown to be a significant bottleneck.

### Reopen when

- Representative production profiles show range queries materially affect end-to-end latency.
- A controlled benchmark demonstrates a meaningful win over the STL implementation across realistic hit/miss and duplicate distributions.

### Completion criteria

- Keep an STL fallback whenever a learned window cannot be proven to contain the answer.
- Differentially test all three operations against the STL for randomized data, custom comparators, key extractors, empty input, missing values, long duplicate runs, segment-boundary duplicates, and 64-bit keys above `2^53`.
- Validate complexity and latency for duplicate-heavy and adversarial distributions, not only one fixed hit.
- Preserve the current non-finite query rejection and unbuilt/empty behavior.

## 3. Repository history cleanup

### Current behavior

The checkout contains large committed generated reports and images, and the existing Git object database is much larger than the current source tree. The product pass leaves history untouched; install rules already exclude reports, tests, and benchmark assets from the package.

At the time this item was recorded, tracked files occupied roughly 68 MiB and the local Git pack roughly 589 MiB. Re-measure before acting because both numbers can change.

### Why it is deferred

Removing large objects from history rewrites commit IDs and requires a coordinated force-push, fresh clones, and migration instructions for every contributor and open branch. Doing that inside a correctness patch would be disruptive and hard to reverse.

### Reopen when

- Before a broadly announced stable release, if clone size remains excessive.
- Hosting, CI checkout time, or contributor onboarding is measurably harmed.
- Maintainers can schedule a coordinated history rewrite window.

### Completion criteria

- Inventory large current files and historical blobs; decide which generated artifacts should move to release assets or external hosting.
- Back up all refs and publish the cutoff/migration plan before rewriting.
- Use a standard history-filtering tool, force-push all affected refs, and expire obsolete server-side objects where supported.
- Verify a fresh clone, tags, submodules, CI, release automation, documentation links, and package contents.
- Record old-to-new tag/commit guidance and require contributors to reclone or carefully rebase.

## 4. Additional model and configuration abstractions

### Current behavior

The initial segment count is a compile-time `SegmentCount` parameter; refinement grows the segment vector when fits are poor, model selection uses the built-in constant/linear/quadratic/cubic choices, and tuning thresholds remain implementation constants. Comparators and numeric key extractors are already customizable.

### Why it is deferred

There is no demonstrated need for runtime model factories, pluggable model interfaces, configuration objects, or alternative storage policies. Those abstractions would enlarge a header-only API, complicate optimization, and make compatibility promises before real callers establish the required variation.

### Reopen when

- At least two concrete consumers require the same runtime setting or model extension.
- Benchmarks show a built-in policy cannot meet a supported workload.
- Compile-time configuration creates a measured binary-size or build-time problem.

### Completion criteria

- Begin with the smallest proven extension point; do not introduce a factory or inheritance hierarchy for one implementation.
- Specify ownership, exception, thread-safety, and serialization behavior.
- Preserve the simple zero-dependency header-only path.
- Benchmark compile time, binary size, index size, build time, and query latency against the current design.
- Add compatibility tests for defaults so existing callers retain identical behavior.

## 5. SOSD and application-dataset ingestion

### Current behavior

`benchmark_competitors` compares JazzyIndex, PGM-index, RadixSpline, and
`std::lower_bound` on the repository's nine deterministic synthetic
distributions. It measures mixed-query latency, serial construction time, and
logical index bytes (excluding the shared source data and allocator slack)
across multiple configurations. For some generated long duplicate runs,
preflight found that the upstream RadixSpline search window did not cover the
first occurrence needed by exact lower-bound lookup. The suite therefore
applies the same documented order-preserving unique-key encoding to every
implementation. The primary JazzyIndex benchmark retains the original
duplicate-heavy data.

### Why it is deferred

Bundling or automatically downloading multi-gigabyte SOSD datasets would add
storage, licensing, checksum, cache, and CI-runtime policy to a header-only
library. An ad hoc binary-file parser would also be a new input surface that is
not needed to establish the initial competitor harness. Synthetic results are
useful for controlled comparisons but are not sufficient for broad performance
claims.

### Reopen when

- Competitive performance is being prepared for publication or a stable release.
- A real consumer can provide a representative, redistributable key trace.
- CI or release infrastructure has an explicit large-dataset cache and retention policy.

### Completion criteria

- Consume the standard SOSD `uint64` format without silently changing key order,
  cardinality, or duplicates.
- Pin dataset URLs and checksums outside the default build; never download large
  files merely by configuring CMake.
- Report dataset provenance, duplicate count, key count, query mix, index bytes,
  construction time, and lookup distributions—not only a mean.
- Define how implementations that cannot satisfy bounds on duplicate-heavy data
  are reported; do not silently remove or widen their failures.
- Run repeated measurements on an otherwise idle, identified machine and retain
  raw JSON plus the exact dependency revisions and compiler flags.
- Produce latency/space Pareto comparisons and avoid a single cherry-picked
  configuration per implementation.
