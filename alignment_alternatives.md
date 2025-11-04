# Segment Struct Alignment Alternatives

## Current Problem
With cubic model support, the Segment struct exceeds 64 bytes when T is double/int64_t:
- Cubic params: 4 × double = 32 bytes
- Plus other fields: ~40 bytes
- Total: ~72 bytes
- With alignas(64): padded to 128 bytes

## Solution Comparison

### Option 1: Float Parameters (Current Solution)
```cpp
struct Segment {
    T min_val, max_val;
    ModelType model_type;
    uint8_t max_error;
    union {
        struct { float slope, intercept; } linear;
        struct { float a, b, c; } quadratic;
        struct { float a, b, c, d; } cubic;
        struct { size_t constant_idx; } constant;
    } params;
    size_t start_idx, end_idx;
};
alignas(64) // Total: 64 bytes
```

**Pros:**
- ✅ Fits in single 64-byte cache line
- ✅ Optimal memory efficiency
- ✅ Best cache performance
- ✅ Tests show <2 index error even for extreme cases

**Cons:**
- ⚠️ ~7 decimal digit precision for parameters
- ⚠️ Potential issues with very large int64_t ranges
- ⚠️ May accumulate errors for huge datasets (>10M elements per segment)

**Risk Assessment:** LOW for most use cases
- Float precision: ~1.2e-7 relative error
- Prediction uses double-precision FMA (promoted automatically)
- Acceptable for typical learned index scenarios

---

### Option 2: alignas(128) with Double Parameters
```cpp
alignas(128) struct Segment { /* all double */ };
```

**Pros:**
- ✅ Full double precision (~15 digits)
- ✅ No precision concerns
- ✅ Still cache-aligned

**Cons:**
- ❌ 2× memory usage (128 bytes vs 64 bytes)
- ❌ Wastes 50+ bytes per segment
- ❌ Spans 2 cache lines
- ❌ Worse cache utilization
- ❌ For 1000 segments: 125KB vs 62.5KB

**Use Case:** High-precision requirements, memory not a concern

---

### Option 3: No Forced Alignment (Natural Alignment)
```cpp
struct Segment { /* all double, no alignas */ };
```

**Pros:**
- ✅ Full double precision
- ✅ More memory efficient than Option 2 (~72-80 bytes)
- ✅ ~12.5% more memory than Option 1

**Cons:**
- ❌ Not cache-line aligned
- ❌ May span cache lines (boundary crossing)
- ❌ Unpredictable performance (depends on allocation)
- ❌ Potential false sharing in multithreaded scenarios

**Risk Assessment:** MEDIUM
- Performance depends on runtime memory layout
- Some segments will cross cache line boundaries

---

### Option 4: Separate Storage for Cubic Models
```cpp
struct Segment {
    // ... base fields ...
    union {
        struct { double slope, intercept; } linear;      // 16 bytes
        struct { double a, b, c; } quadratic;            // 24 bytes
        size_t cubic_model_index;                        // 8 bytes -> points to external storage
        size_t constant_idx;
    } params;
};
std::vector<CubicParams> cubic_storage;  // Separate array
```

**Pros:**
- ✅ Fits in 64 bytes
- ✅ Full precision for all models
- ✅ Most segments use linear/quadratic (common case fast)

**Cons:**
- ❌ Extra indirection for cubic models (pointer chasing)
- ❌ Cache miss for cubic predictions
- ❌ Complexity: need to manage external storage
- ❌ Makes code more complex

**Use Case:** If cubic models are rare (<5% of segments)

---

### Option 5: Hybrid Precision
```cpp
union {
    struct { float slope, intercept; } linear;      // 8 bytes
    struct { float a, b, c; } quadratic;            // 12 bytes
    struct { double a, b, c, d; } cubic;            // 32 bytes - highest precision where needed
};
```

**Pros:**
- ✅ Float for simple models (usually sufficient)
- ✅ Double for cubic (most complex/error-prone)
- ❌ Still exceeds 64 bytes (→ 128 bytes with alignas)

**Verdict:** Doesn't solve the size problem

---

### Option 6: Template Parameter for Alignment
```cpp
template<typename T, SegmentCount S, typename Compare, std::size_t Alignment = 64>
class JazzyIndex { /* ... */ };

template<typename T>
struct alignas(Alignment) Segment { /* ... */ };
```

**Pros:**
- ✅ User choice: performance vs memory vs precision
- ✅ Flexibility for different use cases
- ✅ Can use 64/128/unaligned as needed

**Cons:**
- ❌ More template parameters
- ❌ Complexity in API
- ❌ Users need to understand tradeoffs

---

### Option 7: Compressed Parameters (Advanced)
```cpp
struct Segment {
    // Store normalized parameters in smaller range
    float16 slope;  // Half precision (if available)
    int32_t start_idx, end_idx;  // Limit to 4B elements
    // ... clever packing ...
};
```

**Pros:**
- ✅ Can fit in 64 bytes with double equivalent precision via normalization
- ✅ Optimal for specific use cases

**Cons:**
- ❌ Very complex
- ❌ Not portable (float16 support varies)
- ❌ Normalization overhead
- ❌ Hard to maintain

---

## Recommendation

### For General Use: **Option 1 (Float - Current Solution)** ✅

**Rationale:**
1. Test results show <2 index error even for extreme 94.6 petasecond timestamp ranges
2. Float precision (1e-7) is orders of magnitude better than typical max_error (2-8 indices)
3. The exponential search handles prediction errors gracefully
4. 2× better memory efficiency matters for large datasets
5. Single cache line = better performance

### When to Consider Alternatives:

- **Option 2 (alignas(128))**: If you need absolute precision and memory is abundant
- **Option 3 (Natural alignment)**: If you're okay with unpredictable performance
- **Option 4 (Separate storage)**: If profiling shows <5% cubic usage and you need precision
- **Option 6 (Template param)**: For a library supporting diverse use cases

### Hybrid Approach (Recommended for Library):

Provide a compile-time flag:
```cpp
#ifdef JAZZY_HIGH_PRECISION
    using ParamType = double;
    static constexpr std::size_t SegmentAlignment = 128;
#else
    using ParamType = float;
    static constexpr std::size_t SegmentAlignment = 64;
#endif
```

This gives users control without API complexity.

## Benchmarking Recommendation

Test real-world scenarios:
1. Large int64_t keys (timestamps, IDs)
2. High-precision doubles (scientific data)
3. Memory-constrained environments (embedded systems)
4. Cache-sensitive workloads (millions of lookups)

Measure:
- Prediction accuracy (avg error, max error)
- Lookup performance (ns per query)
- Memory usage
- Cache miss rates
