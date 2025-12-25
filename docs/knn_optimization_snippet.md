# k-NN Optimization Problem

## Context

Batched k-NN on unit sphere. For each of 100k query points, find k=24 nearest neighbors from ~200 candidates (points in same + neighboring grid cells). Distance = dot product (higher = closer).

## Current Timing (100k points, k=24)

```
Phase        None      Combined   Notes
─────────────────────────────────────────────────
SIMD dots    17ms      19ms       Compute all ~200 dots per query
Pairs        47ms      60ms       Filter + build (dot, idx) vector
Select       57ms      30ms       select_nth_unstable to find k-th
Sort         26ms      30ms       Sort top k
─────────────────────────────────────────────────
Total       ~150ms    ~150ms      Filtering helps Select but adds Pairs overhead
```

- **None**: No filtering, ~200 candidates go to Select
- **Combined**: Filter to ~80 candidates using threshold

## The Core Loop (what we want to optimize)

```rust
// After SIMD: cell_dots[qi * padded_len + i] = dot product for query qi, candidate i
// ~200 candidates per query, want top k=24 sorted by dot (descending)

let mut pairs: Vec<(f32, u32)> = Vec::with_capacity(512);

for (qi, &query_idx) in query_points.iter().enumerate() {
    let dot_start = qi * padded_len;

    // === PAIRS PHASE (47-60ms total) ===
    pairs.clear();

    // Option A: No filtering (None mode)
    for i in 0..num_candidates {  // ~200 candidates
        let d = cell_dots[dot_start + i];
        if d > threshold {  // threshold = -inf for None, geometric bound for Combined
            pairs.push((d, candidate_indices[i]));
        }
    }
    // After filtering: ~80-200 pairs depending on mode

    // === SELECT + SORT PHASE (57-87ms total) ===
    let k_actual = k.min(pairs.len());

    if pairs.len() <= 2 * k {
        // Small enough: just sort
        pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    } else {
        // Large: select k-th, then sort top k
        pairs.select_nth_unstable_by(k_actual - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
        pairs[..k_actual].sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    }

    // Write results
    for (i, &(_, idx)) in pairs.iter().take(k_actual).enumerate() {
        neighbors[query_idx * k + i] = idx;
    }
}
```

## What We've Tried (all slower or breakeven)

1. **Fused filter+select+sort** - maintain sorted top-k array on the fly, insert as we iterate
2. **Binary heap** - maintain min-heap of size k
3. **Two-bucket approach** - collect "definitely in" vs "maybe in" buckets in single pass
4. **Parallel arrays** - separate dots/indices vectors instead of tuple
5. **Indices-only** - store just indices, look up dots during select/sort (indirection kills perf)
6. **SIMD filtering** - the compaction step (mask → dense output) is expensive

## The Question

Any ideas to speed up the Pairs → Select → Sort pipeline?

Constraints:
- ~80-200 candidates after filtering
- Want top k=24 sorted
- Dots already computed in contiguous array
- Must handle f32 (no Ord, need partial_cmp)
- Runs 100k times, so per-query overhead matters

## Additional Context

The "Combined" mode does extra work to compute a tighter threshold:
```rust
// Center cell: track worst (min) dot
let mut min_center_dot = f32::INFINITY;
for i in 0..center_count {  // ~24 candidates
    let d = cell_dots[dot_start + i];
    if d > security_threshold {
        pairs.push((d, candidate_indices[i]));
        min_center_dot = min_center_dot.min(d);
    }
}

// Ring cells: filter by max(security_threshold, worst_center - eps)
let threshold = security_threshold.max(min_center_dot - 1e-6);
for i in center_count..num_candidates {  // ~176 candidates
    let d = cell_dots[dot_start + i];
    if d > threshold {
        pairs.push((d, candidate_indices[i]));
    }
}
```

This filters more aggressively (69% vs 15%) and helps Select, but the extra loop overhead + tracking min_center_dot costs time in Pairs phase. Net result is ~breakeven.
