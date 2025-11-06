# Index Structure Visualizations

This directory contains **270+ visualization plots** showing the internal structure of JazzyIndex across all test distributions and segment configurations.

## What's Here

Each plot visualizes how JazzyIndex segments the data and selects models (CONSTANT, LINEAR, QUADRATIC, or CUBIC) for each segment.

**Coverage:**
- **9 distributions**: Uniform, Exponential, Clustered, Lognormal, Zipf, Mixed, Quadratic, ExtremePoly, InversePoly
- **10 segment counts**: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
- **Multiple sizes**: 100, 1000, 10000 elements

## File Naming Convention

Format: `index_{Distribution}_N{Size}_S{Segments}.png`

Examples:
- `index_Uniform_N1000_S256.png` - Uniform distribution, 1K elements, 256 segments
- `index_ExtremePoly_N1000_S8.png` - Extreme polynomial (x⁵), 1K elements, 8 segments
- `index_Zipf_N10000_S256.png` - Zipf distribution, 10K elements, 256 segments

## How to Read These Plots

Each visualization shows:
- **Black dots**: Actual data points (key-value pairs)
- **Red lines**: LINEAR models (cost: 1 FMA)
- **Blue lines**: QUADRATIC models (cost: 2 FMA)
- **Purple lines**: CUBIC models (cost: 3 FMA)
- **Green lines**: CONSTANT models (cost: 0)
- **Tan bands**: Prediction error tolerance zones (±max_error) shown along the entire model line. The band width shows how far off the model's predictions can be from actual index positions. Narrower bands mean more accurate predictions and faster queries.
- **Gray dashed lines**: Segment boundaries

For a detailed guide, see [VISUALIZATIONS.md](../../VISUALIZATIONS.md).

## Generating These Plots

Run:
```bash
cmake --build build --target visualize_index
```

This will regenerate all plots in this directory.

## Showcase Plots

Key plots highlighted in the documentation:
- `index_Uniform_N1000_S256.png` - Perfect linear fit
- `index_Quadratic_N1000_S256.png` - Adaptive quadratic selection
- `index_ExtremePoly_N1000_S8.png` - Model type diversity
- `index_Zipf_N10000_S256.png` - Real-world heavy-tailed distribution
- `index_InversePoly_N1000_S64.png` - Inverse curvature handling
