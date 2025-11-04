# Model Selection and Fitting Algorithms

Technical deep-dive into how JazzyIndex fits and selects prediction models (constant, linear, quadratic).

## Table of Contents
- [Overview](#overview)
- [Model Types](#model-types)
- [Selection Thresholds](#selection-thresholds)
- [Linear Model Fitting](#linear-model-fitting)
- [Quadratic Model Fitting](#quadratic-model-fitting)
- [Input Normalization](#input-normalization)
- [Error Measurement](#error-measurement)

## Overview

JazzyIndex uses learned models to predict array indices from key values. For each segment, it fits three candidate models and selects the most cost-effective one based on prediction accuracy.

The core trade-off:
- **More complex models** = Better prediction accuracy = Fewer exponential search iterations
- **Simpler models** = Faster prediction = Lower overhead per query

The algorithm adaptively chooses the optimal balance for each segment's local data distribution.

## Model Types

### CONSTANT Model
```cpp
predicted_index = c
```

**When used:** All values in the segment are identical
**Cost:** 0 FMA instructions
**Use case:** Duplicate keys, plateau regions

**Implementation:**
```cpp
if (max_value == min_value) {
    model.type = CONSTANT;
    model.constant_value = start_index;
    max_error = 0;  // Perfect prediction
}
```

---

### LINEAR Model
```cpp
predicted_index = m × value + b
```

**When used:** Default choice for non-constant data
**Cost:** 1 FMA (fused multiply-add) instruction
**Use case:** Most segments in practice

**Implementation:**
```cpp
double value_range = max_value - min_value;
double index_range = end_index - start_index;

m = index_range / value_range;  // slope
b = start_index - m × min_value; // intercept
```

**Error calculation:** Measure max deviation across all segment elements

---

### QUADRATIC Model
```cpp
predicted_index = a × value² + b × value + c
```

**When used:** Curved data where quadratic reduces error significantly
**Cost:** 2 FMA instructions (using Horner's method)
**Use case:** Exponential, power-law, S-curve distributions

---

### CUBIC Model
```cpp
predicted_index = a × value³ + b × value² + c × value + d
```

**When used:** Highly curved data where cubic significantly improves over quadratic
**Cost:** 3 FMA instructions (using Horner's method)
**Use case:** Strong power-law, high-degree polynomial, compound growth distributions

**Selection criteria (Quadratic):**
1. Linear max_error > `MAX_ACCEPTABLE_LINEAR_ERROR` (2 elements)
2. Quadratic max_error ≤ `QUADRATIC_IMPROVEMENT_THRESHOLD` × linear_error (≤70%)
3. **Monotonicity constraint:** Derivative must be non-negative over segment range

**Selection criteria (Cubic):**
1. Quadratic max_error > `MAX_ACCEPTABLE_QUADRATIC_ERROR` (6 elements)
2. Cubic max_error ≤ `CUBIC_IMPROVEMENT_THRESHOLD` × quadratic_error (≤70%)
3. **Monotonicity constraint:** Derivative f'(x) = 3ax² + 2bx + c ≥ 0 over segment range

**Monotonicity requirement:**
For a search index, the prediction function `f(key) = index` must be **monotonically increasing**.
This is validated by checking that the derivative `f'(x) = 2ax + b ≥ 0` at both segment endpoints:
```cpp
derivative_at_min = 2*a*min_val + b >= 0
derivative_at_max = 2*a*max_val + b >= 0
```
If either check fails, the quadratic is non-monotonic and rejected (falls back to linear).

**Why this matters:**
- Non-monotonic quadratics can produce backwards-looping curves
- This violates the fundamental property: `key1 < key2 ⟹ index1 ≤ index2`
- Such models would give incorrect search results
- Trade-off: Some segments use linear (higher error) but maintain correctness

**Why 30% improvement?**
- Quadratic costs 3× more than linear (3 FMA vs. 1 FMA)
- But can reduce exponential search from ~10 iterations to ~3 iterations
- 30% error reduction typically saves 20-50 CPU cycles, worth the extra 2 FMA

---

## Selection Thresholds

These constants control model selection (defined in `jazzy_index.hpp`):

```cpp
// If linear error ≤ 2, accept it immediately
static constexpr double MAX_ACCEPTABLE_LINEAR_ERROR = 2.0;

// Quadratic must reduce error to ≤70% of linear error
static constexpr double QUADRATIC_IMPROVEMENT_THRESHOLD = 0.7;

// Add margin to search radius
static constexpr std::uint64_t SEARCH_RADIUS_MARGIN = 2;

// Minimum search coverage
static constexpr std::uint64_t MIN_SEARCH_RADIUS = 4;

// Exponential search starts at radius 2
static constexpr std::uint64_t INITIAL_SEARCH_RADIUS = 2;
```

### Why MAX_ACCEPTABLE_LINEAR_ERROR = 2?

**Historical context:** Originally set to 8, but analysis showed:
- Error of 2 → Exponential search typically completes in 2-3 iterations (radii: 2, 4)
- Error of 8 → Exponential search needs 4-5 iterations (radii: 2, 4, 8, 16)
- The extra iterations cost more than fitting a quadratic model

**Result:** Changing from 8→2 improved query performance by ~15-25% on curved distributions.

---

## Linear Model Fitting

Linear least-squares fit using the **closed-form solution**:

```cpp
// Given: sorted key-value pairs (value[i], index[i])
// Find: slope m and intercept b minimizing Σ(predicted_index - actual_index)²

double sum_x = 0.0, sum_y = 0.0;
double sum_xy = 0.0, sum_x2 = 0.0;
int n = segment_size;

for (int i = 0; i < n; ++i) {
    double x = static_cast<double>(value[i]);
    double y = static_cast<double>(index[i]);

    sum_x += x;
    sum_y += y;
    sum_xy += x * y;
    sum_x2 += x * x;
}

// Slope: m = (n×Σxy - Σx×Σy) / (n×Σx² - (Σx)²)
double denominator = n * sum_x2 - sum_x * sum_x;
double m = (n * sum_xy - sum_x * sum_y) / denominator;

// Intercept: b = (Σy - m×Σx) / n
double b = (sum_y - m * sum_x) / n;
```

**Numerical stability:** Division-by-zero handled by constant model fallback.

**Maximum error:**
```cpp
double max_error = 0.0;
for (int i = 0; i < n; ++i) {
    double predicted = m * value[i] + b;
    double actual = index[i];
    max_error = std::max(max_error, std::abs(predicted - actual));
}
```

---

## Quadratic Model Fitting

Quadratic least-squares fit using **Cramer's Rule** on a **normalized** [0,1] input range.

### Why Normalization?

**Problem:** Direct quadratic fitting on raw uint64_t values causes:
- **Overflow:** x⁴ terms for large values exceed double precision
- **Underflow:** x⁴ terms for small values lose significance
- **Ill-conditioning:** Matrix becomes near-singular

**Solution:** Normalize input to [0,1]:
```cpp
x_normalized = (x - x_min) / x_range
```

Where:
- `x_min` = minimum value in segment
- `x_range` = max_value - min_value

This ensures all x² and x⁴ terms stay in [0,1], preventing numerical issues.

### Algorithm

**Step 1: Normalize inputs**
```cpp
const double x_min = min_value;
const double x_scale = (max_value - min_value) > 0
                        ? (max_value - min_value)
                        : 1.0;

for (int i = 0; i < n; ++i) {
    x_norm[i] = (value[i] - x_min) / x_scale;
    y[i] = index[i];
}
```

**Step 2: Accumulate sums**
```cpp
// Compute: Σx, Σx², Σx³, Σx⁴, Σy, Σxy, Σx²y
double sum_x = 0.0, sum_x2 = 0.0, sum_x3 = 0.0, sum_x4 = 0.0;
double sum_y = 0.0, sum_xy = 0.0, sum_x2y = 0.0;

for (int i = 0; i < n; ++i) {
    double x = x_norm[i];
    double x2 = x * x;
    double x3 = x2 * x;
    double x4 = x2 * x2;

    sum_x += x;
    sum_x2 += x2;
    sum_x3 += x3;
    sum_x4 += x4;
    sum_y += y[i];
    sum_xy += x * y[i];
    sum_x2y += x2 * y[i];
}
```

**Step 3: Solve 3×3 system using Cramer's Rule**

We need to solve:
```
[ Σx⁴  Σx³  Σx² ] [ a ]   [ Σx²y ]
[ Σx³  Σx²  Σx  ] [ b ] = [ Σxy  ]
[ Σx²  Σx   n   ] [ c ]   [ Σy   ]
```

**Determinant:**
```cpp
double det = sum_x4 * (sum_x2 * n - sum_x * sum_x)
           - sum_x3 * (sum_x3 * n - sum_x * sum_x2)
           + sum_x2 * (sum_x3 * sum_x - sum_x2 * sum_x2);

if (std::abs(det) < 1e-10) {
    // Matrix singular - fall back to linear
}
```

**Coefficient a:**
```cpp
double det_a = sum_x2y * (sum_x2 * n - sum_x * sum_x)
             - sum_xy * (sum_x3 * n - sum_x * sum_x2)
             + sum_y * (sum_x3 * sum_x - sum_x2 * sum_x2);

double a = det_a / det;
```

*Similarly for b and c using Cramer's rule...*

**Step 4: Transform coefficients back to original space**

The coefficients (a, b, c) are for the normalized equation:
```
y = a×x_norm² + b×x_norm + c
```

To convert to original value space:
```cpp
// Substitute: x_norm = (x - x_min) / x_scale
// Expand: y = a×((x-x_min)/x_scale)² + b×((x-x_min)/x_scale) + c
// Simplify to: y = a'×x² + b'×x + c'

double x_scale_sq = x_scale * x_scale;

a_final = a / x_scale_sq;
b_final = b / x_scale - 2.0 * a * x_min / x_scale_sq;
c_final = a * x_min * x_min / x_scale_sq
        - b * x_min / x_scale
        + c;
```

**Mathematical derivation:**
```
y = a×((x - x_min) / x_scale)² + b×((x - x_min) / x_scale) + c

Expand (x - x_min)²:
y = a×(x² - 2x×x_min + x_min²) / x_scale²
  + b×(x - x_min) / x_scale
  + c

Distribute:
y = (a / x_scale²)×x²
  + (b/x_scale - 2a×x_min/x_scale²)×x
  + (a×x_min²/x_scale² - b×x_min/x_scale + c)
```

---

## Input Normalization

### The Problem

Consider extreme polynomial distribution with values [0, 1000]:
- `x⁴` for x=1000 → 1,000,000,000,000 (1 trillion)
- `x⁴` for x=1 → 1
- **Dynamic range:** 12 orders of magnitude!

Double precision (53-bit mantissa) cannot represent both accurately.

### The Solution

Map all inputs to [0,1]:
```cpp
x_normalized = (x - min_value) / (max_value - min_value)
```

Now:
- `x_norm⁴` for x_norm=1.0 → 1.0
- `x_norm⁴` for x_norm=0.001 → 1e-12
- **Dynamic range:** Manageable within double precision

### Additional Benefits

1. **Matrix conditioning:** Normalized system has condition number ~10² vs ~10¹² for raw values
2. **Faster convergence:** Cramer's rule arithmetic stays in reasonable ranges
3. **Numerical stability:** Subtraction cancellation errors minimized

---

## Error Measurement

After fitting each model, compute maximum prediction error:

```cpp
double max_error = 0.0;

for (int i = start_index; i < end_index; ++i) {
    double predicted_index = evaluate_model(model, data[i]);
    double actual_index = static_cast<double>(i);
    double error = std::abs(predicted_index - actual_index);

    max_error = std::max(max_error, error);
}
```

**Usage in exponential search:**
```cpp
// Start with radius based on max_error
uint64_t radius = std::max(max_error + SEARCH_RADIUS_MARGIN,
                            MIN_SEARCH_RADIUS);

// Expand: 2, 4, 8, 16, ...
while (true) {
    if (check_range(predicted - radius, predicted + radius)) {
        break;
    }
    radius = std::min(radius * 2, MAX_RADIUS);
}
```

---

## Summary

JazzyIndex's model selection is driven by empirical cost-benefit analysis:
- **Constant models:** Free prediction, perfect for duplicates
- **Linear models:** 1 FMA, good for ~90% of segments
- **Quadratic models:** 3 FMA, essential for curved regions

The key insight: **Don't use quadratic unless it saves more than it costs.**

The recent fix made quadratic models actually work, transforming JazzyIndex from "fast on uniform data" to "fast on all real-world distributions."

---

**See also:**
- [BENCHMARKS.md](BENCHMARKS.md) - Performance results demonstrating the impact
- [VISUALIZATIONS.md](VISUALIZATIONS.md) - Visual examples of model selection in action
- [README.md](../README.md) - Usage guide and high-level overview

