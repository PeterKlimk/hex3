# Staged k-NN for Half-Plane Clipping

## Overview

Replace flat 3x3 neighborhood search with staged expansion:
1. Stage 1: 1x1 (same cell only) with symmetric dot computation
2. Stage 2: 3x3 for unsatisfied points only (outer ring candidates)
3. Stage 3+: 5x5, 7x7, etc. if needed (very rare)

Most points are satisfied at Stage 1, avoiding expensive wide searches.

## Key Insight: k is Arbitrary

For half-plane clipping (Voronoi cell construction), we don't need exactly k neighbors.
We need "enough" neighbors to prove the cell is closed. This means:

- Don't expand just because you're 1-2 neighbors short
- Use two thresholds: `k_track` (how many to keep) vs `k_satisfy` (when to expand)
- Pass security radius to clipping for early termination even with fewer neighbors

## Algorithm

### Two-k Thresholds

- `k_track = 24`: Maximum neighbors to track in top-k buffer
- `k_satisfy = 20`: Minimum confirmed neighbors before marking satisfied

A point is "satisfied" at Stage N if:
- It has ≥ k_satisfy neighbors, AND
- The k_satisfy-th neighbor's dot > security threshold for Stage N

### Batch All KNN Before Clipping

To avoid cache thrashing between KNN and clipping data structures:

```
Stage 1: all points, symmetric dots within each cell
    ↓ check satisfaction per point
    ↓ satisfied (80%) → have their neighbors
    ↓ unsatisfied (20%) → need outer ring

Stage 2: only unsatisfied points × outer ring candidates
    ↓ now everyone has enough neighbors (conservative k)

Clip ALL points (single pass, good cache locality)
    ↓ rare failures → request more neighbors → Stage 3+
```

### Stage 1: 1x1 Symmetric Search

For each cell C with points P[0..n]:

```
// Exploit symmetry: dot(A, B) == dot(B, A)
// Only compute upper triangle: n*(n-1)/2 dot products instead of n*n

for i in 0..n:
    for j in i+1..n:
        dot = P[i] · P[j]
        // Update top-k for both points
        update_topk(i, dot, j)
        update_topk(j, dot, i)

// Check satisfaction for each point
for i in 0..n:
    if count_above_threshold(topk[i], security_dot_1x1[C]) >= k_satisfy:
        mark_satisfied(P[i])
    else:
        mark_unsatisfied(P[i])
```

### Stage 2: Expand Unsatisfied to 3x3

Only unsatisfied points compute dots against the outer ring (8 neighbor cells).

```
For each cell C:
    unsatisfied = get_unsatisfied_points(C)
    if unsatisfied.empty(): continue

    // Gather candidates from outer ring only (~175 points)
    outer_candidates = gather_outer_ring(C)

    // Compute dots: unsatisfied queries × outer candidates
    // (Recompute from scratch - simpler than merging Stage 1 results)
    for query in unsatisfied:
        all_candidates = cell_points(C) + outer_candidates

        for candidate in all_candidates:
            dot = query · candidate
            update_topk(query, dot, candidate)

        // Check satisfaction against 3x3 security threshold
        if count_above_threshold(topk[query], security_dot_3x3[C]) >= k_satisfy:
            mark_satisfied(query)
        else:
            mark_unsatisfied(query)  // needs 5x5 (very rare)
```

Note: We recompute all dots for unsatisfied points rather than merging Stage 1
results. This is simpler and the unsatisfied points are only ~20% of total.

### Stage 3+: Further Expansion (Rare)

Same pattern with 5x5 ring, 7x7, etc. In practice, <1% of points reach Stage 3.
Use fixed rings (not "closest cell first") for simplicity - the rare cases
don't justify the complexity of smarter expansion.

### Clipping Integration

After all KNN stages complete:

```
For each point:
    neighbors = topk[point]  // up to k_track neighbors
    security_bound = security_dot[point]  // from last stage searched

    result = try_clip(point, neighbors, security_bound)

    if result == NeedMore:
        // Rare: clipping couldn't prove termination
        // Expand to next stage and retry
        expand_and_reclip(point)
```

The security_bound tells clipping: "any neighbor I haven't seen is at least this
far away." Clipping can often prove termination using this bound even with
fewer than k_track neighbors.

**Clip before expand**: If Stage 1 gives you 22 neighbors but you're unsatisfied
(22nd is outside security radius), try clipping first. Clipping might succeed,
avoiding the expansion cost entirely.

## Work Estimates

### Current Approach (flat 3x3)
- All points: ~200 candidates each
- Dot products: 100k × 200 = 20M
- Select + sort: O(200) per point

### Staged Approach

For 100k points, k_track=24, k_satisfy=20:

| Stage | Points | Candidates | Dot Products |
|-------|--------|------------|--------------|
| Stage 1 | 100k | 25 (symmetric) | 100k × 12 = 1.2M |
| Stage 2 | 20k | 175 (outer ring) | 20k × 175 = 3.5M |
| Stage 3 | <1k | 350 | negligible |
| **Total** | | | **~4.7M** |

**~4x reduction in dot product work**, plus:
- Stage 1 symmetry halves the work
- Smaller candidate sets → better cache behavior
- Selection/sorting on smaller sets

## Security Radius

### Concept

For a point in cell C searching an NxN neighborhood:
- **Security radius** = maximum dot product to any point OUTSIDE the NxN region
- If k_satisfy-th neighbor dot > security radius → search is complete
- If k_satisfy-th neighbor dot ≤ security radius → must expand

### Precomputation

```rust
/// Maximum possible dot product between any point in cell A and any point in cell B.
/// Higher dot = closer on unit sphere.
fn max_dot_between_cells(
    a_center: Vec3, a_cos_r: f32, a_sin_r: f32,
    b_center: Vec3, b_cos_r: f32, b_sin_r: f32,
) -> f32 {
    let cos_d = a_center.dot(b_center).clamp(-1.0, 1.0);
    let d = cos_d.acos();  // center-to-center angle

    let a_radius = a_cos_r.acos();
    let b_radius = b_cos_r.acos();

    // Minimum angular distance = center distance - both radii
    let min_angle = (d - a_radius - b_radius).max(0.0);

    // Maximum dot = cos(minimum angle)
    min_angle.cos()
}

/// Security threshold for 1x1: max dot to nearest neighbor cell.
fn compute_security_dot_1x1(grid: &CubeMapGrid, cell: usize) -> f32 {
    let mut max_dot = -1.0f32;
    for &neighbor in grid.cell_neighbors(cell) {
        if neighbor == u32::MAX || neighbor == cell as u32 {
            continue;
        }
        let dot = max_dot_between_cells(cell, neighbor);
        max_dot = max_dot.max(dot);
    }
    max_dot
}

/// Security threshold for 3x3: max dot to nearest non-3x3 cell.
/// Only check 5x5 ring (16 cells), not all cells.
fn compute_security_dot_3x3(grid: &CubeMapGrid, cell: usize) -> f32 {
    let ring_5x5 = get_5x5_ring(grid, cell);
    let mut max_dot = -1.0f32;
    for &ring_cell in &ring_5x5 {
        let dot = max_dot_between_cells(cell, ring_cell);
        max_dot = max_dot.max(dot);
    }
    max_dot
}
```

Add to CubeMapGrid:
```rust
pub(super) security_dot_1x1: Vec<f32>,
pub(super) security_dot_3x3: Vec<f32>,
```

Computed once at grid construction.

## Implementation Plan

### Phase 1: Simple Staged Implementation
- [ ] Implement security threshold precomputation
- [ ] Stage 1 with symmetric dot computation
- [ ] Stage 2 with recompute (no merge complexity)
- [ ] Benchmark against current flat 3x3
- [ ] Verify correctness

### Phase 2: Clipping Integration
- [ ] Pass security_bound to clipping algorithm
- [ ] Clipping uses bound for early termination
- [ ] "Clip before expand" path for edge cases
- [ ] Benchmark end-to-end (KNN + clipping)

### Phase 3: Optimization (if needed)
- [ ] Stage 1 → Stage 2 result merging (avoid recompute)
- [ ] SIMD satisfaction check (bitmask + popcount)
- [ ] Tune k_track / k_satisfy thresholds
- [ ] Profile and optimize hot paths

## Open Questions

1. **Optimal k_satisfy**: How much buffer between k_track and k_satisfy?
   - Too small → too many unsatisfied, defeats the purpose
   - Too large → clipping gets fewer neighbors, more "need more" failures
   - Start with k_track=24, k_satisfy=20 and tune

2. **Symmetry storage**: Best layout for symmetric dot matrix in Stage 1?
   - Triangular array (memory efficient, index math)
   - Streaming update (compute once, update both endpoints immediately)

3. **Stage 2 batching**: How to batch unsatisfied points efficiently?
   - Group by cell (shared outer ring)
   - Process all unsatisfied from one cell together

4. **Skip satisfaction check**: If cell has < k_satisfy points, all are
   automatically unsatisfied. Skip the check.
