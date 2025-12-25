# Binlocal k-NN Filtering Proposal

## Problem

Current binlocal k-NN timing breakdown (100k points, k=24):

```
Gather:       0.56 ms (  0.6%)
SIMD:        13.29 ms ( 13.5%)
Pairs:        7.37 ms (  7.5%)
Select:      52.80 ms ( 53.7%)  <-- bottleneck
Sort:        21.30 ms ( 21.6%)  <-- bottleneck
Write:        3.09 ms (  3.1%)
Total:       98.41 ms
```

**75% of time is spent in select+sort** on ~200 candidates per query. The SIMD dot computation is only 13.5%. We need to reduce the candidate count before selection.

## Key Concept: Security Thresholds

A point is **certified** if no unsearched point could beat it. We define two security thresholds based on neighborhood boundaries:

```
┌─────────────────────────┐
│  5x5 neighborhood       │
│  ┌─────────────────┐    │
│  │  3x3 neighborhood│   │
│  │  ┌─────────┐    │    │
│  │  │   1x1   │    │    │
│  │  │ (center)│    │    │
│  │  └─────────┘    │    │
│  │                 │    │
│  └────────A────────┘    │  A = 3x3 outer boundary = security_1x1
│                         │
└────────────B────────────┘  B = 5x5 outer boundary = security_3x3
```

- **security_1x1** = max dot from query to boundary A (3x3 inner edge)
  - 1x1 points with dot > security_1x1 are certified (no 3x3 point can beat them)

- **security_3x3** = max dot from query to boundary B (5x5 inner edge)
  - 3x3 points with dot > security_3x3 are certified (no 5x5 point can beat them)

### Efficient Boundary Representation

The 3x3 neighborhood forms a single ~square region on the sphere. Its outer boundary is just **4 great circle arcs** connecting 4 corners. Same for 5x5.

**Precompute per cell C:**
- `corners_1x1[C]`: 4 outer corners of cell C itself
- `corners_3x3[C]`: 4 outer corners of C's 3x3 neighborhood

**Per-query security threshold:**
```rust
fn security_threshold(q: Vec3, corners: &[Vec3; 4]) -> f32 {
    let mut max_dot = f32::NEG_INFINITY;
    for i in 0..4 {
        let closest = closest_point_on_arc(q, corners[i], corners[(i + 1) % 4]);
        max_dot = max_dot.max(q.dot(closest));
    }
    max_dot
}

fn closest_point_on_arc(q: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let n = a.cross(b).normalize();           // great circle normal
    let q_proj = (q - n * q.dot(n)).normalize(); // project onto great circle

    // Check if projection lies on the arc between a and b
    let cross_a = a.cross(q_proj);
    let cross_b = q_proj.cross(b);

    if cross_a.dot(n) >= 0.0 && cross_b.dot(n) >= 0.0 {
        q_proj  // on the arc
    } else if q.dot(a) > q.dot(b) {
        a
    } else {
        b
    }
}
```

## Proposed Solutions

### Idea 1: Geometric Security Filter (5x5 boundary)

Filter 3x3 candidates whose dot < security_3x3. These points are "uncertified" - they could be beaten by unsearched 5x5 points. Since we accept this risk by only searching 3x3, filtering them doesn't make correctness worse.

**Properties:**
- Purely geometric threshold, robust to non-uniform distributions
- Always applicable (doesn't depend on 1x1 having enough points)
- Conservative baseline that Ideas 2 and 3 can tighten

### Idea 2: Threshold-Based Filtering (replaces arc-based certification)

Use the actual 1x1 dots as thresholds - no geometric arc computation needed.

**Key insight:** If 1x1 has m points, the mth-best dot is a valid filter threshold. Any 3x3 point that can't beat the worst 1x1 point is either:
- Definitely not in top-k (if m ≥ k), or
- At best filling the remaining (k-m) slots

**Algorithm:**
```rust
let m = center_dots.len();
let need = k.saturating_sub(m);  // how many more we need from 3x3

// Threshold = worst dot in 1x1 (or -∞ if empty)
let center_threshold = if m > 0 {
    *center_dots.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
} else {
    f32::NEG_INFINITY
};

// Combined threshold: must beat both 1x1's worst AND security_3x3
let threshold = f32::max(security_3x3, center_threshold);
let filtered_ring = simd_filter(ring_dots, threshold);

if need == 0 {
    // 1x1 has ≥k points, just merge filtered ring and select
    let pool = merge(center_dots, filtered_ring);
    return select_top_k(pool, k);
} else if filtered_ring.len() >= need {
    // Enough 3x3 candidates passed filter
    let pool = merge(center_dots, filtered_ring);
    return select_top_k(pool, k);
} else {
    // Fallback: not enough passed filter, take best (k-m) from unfiltered 3x3
    let top_from_ring = select_top_n(ring_dots, need);
    return merge_sorted(center_dots, top_from_ring);
}
```

**Properties:**
- No arc computation for security_1x1 (saves ~126 ns/query)
- Uses actual dot products as thresholds (tight, adaptive)
- Fallback path handles sparse cases correctly
- Still uses precomputed security_3x3 for geometric filtering

## Combined Algorithm

```
1. Precompute per cell:
   - security_3x3[cell]: precomputed via ring caps (see Bounds Test Results)

2. For each query Q in cell C:
   a. Compute dots for 1x1 (center cell), get m points
   b. need = max(0, k - m)

   c. Compute dots for 3x3 ring (SIMD)

   d. threshold = max(security_3x3[C], worst_dot_in_1x1)
   e. filtered_ring = SIMD filter by threshold

   f. If need == 0 OR filtered_ring.len() >= need:
      → pool = 1x1 + filtered_ring
      → select_top_k(pool, k)

   g. Else (fallback - filtered ring too small):
      → take best `need` from unfiltered 3x3
      → merge with all of 1x1
```

## Expected Impact

With ~24 points per cell average:
- Center cell: ~24 candidates
- Ring cells: ~192 candidates (8 cells)
- After filtering: expect ~30-50 ring candidates (those that beat thresholds)

**Projected speedup:**
- Select+sort on 50 vs 200 candidates: ~4x faster
- Small-m fast path when certified_1x1 ≈ k: eliminates select entirely
- Combined: potentially 3-5x reduction in select+sort time

If select+sort drops from 74ms to ~20ms, total time drops from 98ms to ~44ms, achieving ~2.5x speedup over current binlocal (and ~5.7x over individual queries).

## Implementation Order

1. **Precompute security_3x3** - ring caps method per cell (add to CubeMapGrid)
2. **Implement combined filtering** - threshold = max(security_3x3, worst_1x1_dot)
3. **Add fallback path** - for sparse filtered results, take best (k-m) from unfiltered

## Bounds Test Results

Tested in `bounds_test.rs` with 10k points, res=8 (~24 points/cell).

### Method Comparison

| Method | security_1x1 (1x1→3x3) | security_3x3 (3x3→5x5) | Cost |
|--------|------------------------|------------------------|------|
| Arc (exact) | Works ✓ | Works ✓ | ~126 ns |
| Ring caps | **Broken** (100% useless) | Works ✓ | ~102 ns (or precomputed) |

### Why Ring Caps Fails for 1x1→3x3

For `security_1x1`, we check distance to the 3x3 ring (8 neighbor cells). But queries in the center cell are almost always **inside** at least one neighbor's spherical cap, giving distance=0 and threshold≈1.0. This makes the bound useless.

### security_3x3: Ring Caps is Viable

For `security_3x3`, the 5x5 ring (16 cells) is far enough that only 30% of queries fall inside a ring cell's cap. The remaining 70% get meaningful thresholds.

**Trade-off:** Ring caps overestimates by ~6.7% (requires that many more expansions vs exact arc). But ring caps can be **precomputed per cell** (≈0 ns lookup) vs arc at ~126 ns per query.

### Recommendation

- **security_1x1**: Not needed! Use threshold-based filtering with 1x1's worst dot instead (see Idea 2)
- **security_3x3**: Precompute ring caps per cell (add `security_3x3: Vec<f32>` to CubeMapGrid)

## Open Questions

- Is there a faster SIMD compaction method than scalar mask iteration?
- How often does the fallback path trigger? (filtered_ring.len() < need)
