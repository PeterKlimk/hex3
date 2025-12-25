# binlocal_knn Optimization Ideas

## Current Performance Breakdown (100k points, k=24)

After no-gather optimization (~103ms total, down from ~117ms):

| Phase | Time | % | Description |
|-------|------|---|-------------|
| Gather | ~5ms | 5% | Collect candidate indices only (no coords) |
| SIMD | ~20ms | 19% | Compute dot products via f32x8 from grid SoA |
| Sort | ~73ms | 71% | Build pairs, select_nth, sort top-k |
| Write | ~5ms | 5% | Write results to output buffer |
| **Total** | ~103ms | 100% | |

*Note: SIMD % increased because gather % decreased significantly.*

## Ideas (Prioritized)

### 1. Select Swap Less (HIGH PRIORITY)

**Target:** Sort phase (68%)
**Complexity:** Medium
**Expected Impact:** High

Current flow builds `(f32, u32)` tuples and sorts them:
```rust
pairs.extend(cell_dots[..].iter().zip(candidate_indices.iter()).map(...));
select_nth_unstable_by(&mut pairs, k-1, ...);
sort_unstable_by(&mut pairs[..k], ...);
```

Proposed: Sort indices only, compare via indirection:
```rust
// pos: [u16] = [0, 1, 2, ..., num_candidates-1]
// Reuse across queries (just rebuild 0..n each time)
select_nth_unstable_by(&mut pos, k-1, |&a, &b| dots[b].total_cmp(&dots[a]));
sort_unstable_by(&mut pos[..k], |&a, &b| dots[b].total_cmp(&dots[a]));
// Map: candidate_indices[pos[i]]
```

Benefits:
- 4x less data movement (u16 vs 8-byte tuple)
- Dots stay in place, better cache utilization
- No per-query tuple construction

### 2. Remove Per-Query Allocations (MEDIUM PRIORITY)

**Target:** Sort phase (alloc overhead)
**Complexity:** Low
**Expected Impact:** Medium

Currently rebuilding `pairs` vector each query. Could:
- Reuse scratch buffers across queries
- Avoid materializing tuples entirely (see #1)
- Use fixed-capacity arrays for small k

### 3. Drop Candidate Gather - IMPLEMENTED ✓

**Target:** Gather phase (18%)
**Result:** 1.14x speedup (now the default implementation)

Instead of copying coordinates into scratch buffers, iterate grid SoA directly:
```rust
// Track cell ranges, iterate directly
for &(soa_start, soa_end) in &cell_ranges {
    let xs = &grid.cell_points_x[soa_start..soa_end];
    // SIMD directly on grid's storage
}
```

Implementation notes:
- Still gathers `candidate_indices` (just u32s) for output mapping
- Processes each neighbor cell's range with SIMD + scalar tail
- Handles cell boundaries by processing each cell range separately

### 4. Hoist SIMD Setup (LOW PRIORITY)

**Target:** SIMD phase (9%)
**Complexity:** Low
**Expected Impact:** Minor

Precompute query splats once per cell instead of per chunk:
```rust
// Before chunk loop
let qx_splats: Vec<f32x8> = query_points.iter()
    .map(|&qi| f32x8::splat(points[qi].x))
    .collect();

// In chunk loop
let qx = qx_splats[qi];  // instead of f32x8::splat(...)
```

### 5. Simplify Hot Loop (LOW PRIORITY)

**Target:** SIMD phase (9%)
**Complexity:** Low
**Expected Impact:** Minor

Use SIMD for full chunks, scalar for remainder:
```rust
// Full SIMD chunks
for chunk_start in (0..full_chunks * 8).step_by(8) {
    // f32x8 processing
}
// Scalar tail
for i in (full_chunks * 8)..num_candidates {
    // scalar dot product
}
```

Avoids padding and masking overhead.

---

## Already Tried (Not Worth Revisiting)

### Streaming Top-K (Array-Based)

Maintain fixed-size `[f32; MAX_K]` arrays during SIMD, update as we go.

**Result:** ~1.6x slower than binlocal
**Why:** O(k) rescan overhead after each replacement dominates

### Streaming Top-K (Heap-Based)

Use `BinaryHeap` for O(log k) insertions.

**Result:** ~1.85x slower than binlocal
**Why:** Heap operation overhead exceeds select_nth benefit for k=24, ~200 candidates

### Swap-Less (u16 Index Sorting)

Sort `Vec<u16>` indices instead of `Vec<(f32, u32)>` tuples, compare via indirection.

**Result:** ~1.46x slower than binlocal
**Why:** Indirection overhead (memory lookups during comparison) exceeds benefit of smaller swaps

### Small-K Selection Strategies

For k=24 and candidates~200, `select_nth_unstable` is already near-optimal.
Rust's implementation is highly optimized (introselect with fallbacks).

---

## Already Optimal

### Self Exclusion

Current implementation is already O(Q) per cell:
```rust
for qi in 0..num_queries {
    cell_dots[qi * padded_len + qi] = f32::NEG_INFINITY;
}
```

Since current cell is gathered first in order, query qi's self is at position qi.

---

## Implementation Order

1. **Select Swap Less** - Highest impact, directly targets 68% bottleneck
2. **Remove Per-Query Allocs** - Can combine with #1
3. **Drop Candidate Gather** - If #1 successful, tackle gather phase next

## Notes

- All timings from `bench_binlocal_vs_individual_100k` benchmark
- Run with: `cargo test bench_binlocal --release -- --ignored --nocapture`
- Target: 100k points, k=24, ~200 candidates per cell average
