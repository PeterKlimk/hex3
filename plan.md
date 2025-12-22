# Live Dedup with Sharded Ownership

## Goal
Replace post-hoc dedup with live dedup during cell building. Each thread/shard
owns a spatial region of generators and can safely dedup triplets where the
lowest generator index belongs to its region.

## V1 Scope (Simplify First)
For an initial implementation, keep the design correct and measurable before
maximizing parallelism:
- Do **parallel cell building** per bin/shard (Phase 2).
- Do **overflow flush single-threaded** (Phase 3) to avoid cross-thread
  write-back complexity in v1.
- Keep **per-cell duplicate index removal** as a dedicated post-pass after all
  deferred indices have been resolved (Phase 4).

## Key Data Structures

### Pre-computed (once per run)
```rust
generator_bin: Vec<u8>        // bin (0-11) for each generator, ~2.5MB for 2.5M
points
global_to_local: Vec<u32>     // global gen idx → local idx within its bin
bin_ranges: [(usize, usize); 12]  // (start, count) for each bin's generators
```

### Per-shard (owned, no sharing)
```rust
heads: Vec<u32>               // size = generators in this shard
nodes: Vec<TripletNode>       // linked list nodes
vertices: Vec<Vec3>           // thread-local vertex storage
// Overflow entries must include where to patch the originating cell index.
// `source_slot` is the index into this shard's `cell_indices` to overwrite later.
triplet_overflow: [Vec<(u32,u32,u32,Vec3,u32)>; 12] // (a,b,c,pos,source_slot)
support_overflow: [Vec<(Vec<u32>, Vec3, u32)>; 12]  // (support_set,pos,source_slot)
cell_indices: Vec<u32>         // local vertex indices per cell vertex (DEFERRED placeholders allowed)
```

### Overflow Matrix (12×12)
```rust
// overflow_matrix[source_bin][target_bin]
// - Source thread owns its row (no contention during write)
// - Flush can parallelize by target column
// - Each cell is a Vec of (a, b, c, pos, source_bin, source_slot) where
//   generator_bin[a] == target_bin
```

## Algorithm

### Phase 1: Bin Assignment (~1-3ms for 2.5M points)
```rust
fn point_to_bin_12(p: Vec3) -> u8  // 6 faces × 2 = 12 bins
```
- Single pass over generators
- Build `generator_bin` and `global_to_local`
- Count generators per bin for pre-allocation

**Caveat**: `point_to_bin_12()` needs explicit, deterministic tie-breaking.
Consider reusing cube-face selection logic (like `cube_grid::point_to_face_uv`)
to avoid “seam surprises”. Also consider making `NUM_BINS` configurable:
coarser bins typically improve locality (reduce overflow), but can hurt load
balance.

### Phase 2: Parallel Cell Building (main work)
Each shard processes generators in its bin:
```rust
for cell_idx in my_generators {
    // ... build cell, get vertices with triplet keys [a,b,c] ...

    for (a, b, c, pos) in cell_vertices {
        if generator_bin[a] == my_bin {
            // I own this triplet
            let local_a = global_to_local[a];
            let idx = local_dedup(&mut heads[local_a], &mut nodes, a, b, c,
pos);
            // store idx (thread_id, local_vertex_idx)
        } else {
            // Not mine → defer
            // Must record where to write back into my `cell_indices`.
            triplet_overflow[target_bin].push((a, b, c, pos, source_slot));
        }
    }
}
```

**Caveat**: do not perform per-cell index deduplication/compaction yet, because
it will shift `source_slot` positions. Treat `cell_indices` as a stable stream
until all overflow entries have been resolved and patched.

**Support sets** (`VertexKey::Support`) should follow the same ownership rule
as triplets:
- Canonical owner is `min(support_set)`.
- If not owned, defer to the target bin for that owner and record `source_slot`.

### Phase 3: Flush Overflow (parallel by target bin)
```rust
// Can parallelize: each target bin T processes overflow_matrix[*][T]
for target_bin in 0..12 {  // parallel
    for source_bin in 0..12 {
        for (a, b, c, pos) in overflow_matrix[source_bin][target_bin].drain(..)
{
            // Dedup into target_bin's structures
            // Returns (target_bin, local_idx)
        }
    }
}
```
- Seam vertices are ~5-10% of total
- No contention: each target bin's dedup structures are independent

**V1 note**: implement this flush **single-threaded** initially:
- It avoids the need for cross-thread write-back into source shards.
- It keeps correctness simpler while you validate locality and performance.

**If parallelizing later**: do not directly mutate source shards from target
workers. Instead, have the flush produce a list of fixups:
`(source_bin, source_slot, target_bin, target_local_idx)` and apply fixups in a
separate pass grouped by source bin/shard.

### Phase 4: Assemble Final Output
```rust
// Compute vertex offsets per shard
let shard_offsets = prefix_sum(shard_vertex_counts);  // [0, n0, n0+n1, ...]

// Concatenate vertices (simple memcpy per shard)
let all_vertices = concatenate(shard_vertices);

// Remap cell indices: local_idx → shard_offsets[shard_id] + local_idx
// Each cell knows which shard produced it, so remapping is O(total_indices)
```

**Additional required step**: per-cell duplicate vertex index removal (similar
to current `deduplicate_cell_indices()`), performed only after:
1) all deferred indices have been patched, and
2) global remapping is complete (or remap after dedup; either is fine as long as
it is consistent).

## Files to Modify

1. **src/geometry/gpu_voronoi/mod.rs** (major changes)
- Add `point_to_bin_12()` function (inline, ~15 lines)
- Add `BinAssignment` struct: `generator_bin`, `global_to_local`,
`bin_generators`
- Replace `build_cells_data_flat()` parallel loop: iterate by bin instead of
index range
- Remove `FlatChunk`/`FlatCellsData` (replaced by `ShardOutput`)
- Update `compute_voronoi_gpu_style_core()` to use new flow

2. **src/geometry/gpu_voronoi/dedup.rs** (major rewrite)
- Remove `dedup_vertices_hash_flat()`
- Add `ShardState` struct with live dedup methods:
    - `dedup_triplet(local_a, b, c, pos) -> u32`
    - `dedup_support(support, pos) -> u32`
- Add `ShardOutput` struct (vertices, cell_indices, overflow)
- Add `flush_overflow()` for Phase 3
- Add `assemble_final()` for Phase 4

3. **src/geometry/gpu_voronoi/timing.rs** (minor)
- Add `bin_assignment` and `overflow_flush` sub-phases to `PhaseTimings`

4. **src/geometry/gpu_voronoi/cell_builder.rs** (no change)
- `F64CellBuilder` and `to_vertex_data()` unchanged

## Benefits
- No atomics or unsafe in hot path
- ~90-95% of triplets handled locally without contention
- Thread-local vertices avoid allocation contention
- Simple ownership model Rust can verify

## Resolved Design Decisions
1. **Overflow**: 12×12 matrix `[source_bin][target_bin]` - source owns row,
flush parallelizes by column
2. **Vertex indices**: Local indices only, remap during assembly using
prefix-sum offsets
3. **Ownership check**: `generator_bin: Vec<u8>` lookup, O(1) per triplet

## Issues Identified from Code Exploration

### 1. Support Sets (VertexKey::Support)
Current code has two key types:
- `Triplet([u32; 3])` - common case, ~95%+, bins by `a` (lowest index)
- `Support { start, len }` - degenerate case, 4+ generators, uses
`FxHashMap<Vec<u32>, usize>`

**Solution**: Use `min(support_set)` as owner, same as triplet. Support sets are
rare (~1-5%), so even if they all overflow, it's fine.

### 2. Current Chunk Structure Mismatch
Current: Chunks are contiguous index ranges `[start..end]`
Proposed: Bins are spatial (non-contiguous indices)

**Solution**: Restructure parallel loop to iterate by bin:
```rust
// Current: par_iter over index ranges
ranges.par_iter().map(|(start, end)| process_cells(start..end))

// New: par_iter over bins
(0..12).into_par_iter().map(|bin| {
    let my_generators = &bin_generators[bin];  // indices in this bin
    process_cells_for_bin(bin, my_generators)
})
```

### 3. FlatChunk Changes
Current FlatChunk stores raw `Vec<VertexData>` (undeduped).
With live dedup, each shard directly produces deduped vertices.

**New per-shard output:**
```rust
struct ShardOutput {
    vertices: Vec<Vec3>,           // deduped vertices for this shard
    cell_indices: Vec<u32>,        // local vertex indices per cell
    cell_counts: Vec<u8>,          // vertices per cell
    cell_generator_indices: Vec<usize>,  // which generator each cell is for
    overflow: [Vec<(u32,u32,u32,Vec3)>; 12],  // cross-bin triplets
    support_overflow: Vec<(Vec<u32>, Vec3)>,  // cross-bin support sets
}
```

## Final Approach

### Phase 1: Bin Assignment (new, ~1-3ms)
```rust
// Single pass to compute bin assignments
let generator_bin: Vec<u8> = points.iter().map(point_to_bin_12).collect();

// Build per-bin generator lists
let mut bin_generators: [Vec<usize>; 12] = Default::default();
for (i, &bin) in generator_bin.iter().enumerate() {
    bin_generators[bin as usize].push(i);
}

// Build global_to_local mapping
let mut global_to_local: Vec<u32> = vec![0; n];
for generators in &bin_generators {
    for (local_idx, &global_idx) in generators.iter().enumerate() {
        global_to_local[global_idx] = local_idx as u32;
    }
}
```

### Phase 2: Parallel Cell Building with Live Dedup
```rust
let shard_outputs: Vec<ShardOutput> = (0..12).into_par_iter().map(|bin| {
    let my_generators = &bin_generators[bin];
    let mut shard = ShardState::new(my_generators.len());

    for &cell_idx in my_generators {
        // Build cell (existing F64CellBuilder logic)
        let vertices = build_cell(cell_idx, &points, &knn);

        for (key, pos) in vertices {
            match key {
                VertexKey::Triplet([a, b, c]) => {
                    if generator_bin[a as usize] == bin as u8 {
                        // I own this → local dedup
                        let local_a = global_to_local[a as usize];
                        let idx = shard.dedup_triplet(local_a, b, c, pos);
                        shard.cell_indices.push(idx);
                    } else {
                        // Not mine → overflow to target bin
                        let target = generator_bin[a as usize] as usize;
                        let source_slot = shard.cell_indices.len() as u32;
                        shard.triplet_overflow[target].push((a, b, c, pos, source_slot));
                        shard.cell_indices.push(DEFERRED);
                    }
                }
                VertexKey::Support { start, len } => {
                    let support = &support_data[start..start+len];
                    let owner = support.iter().min().unwrap();
                    if generator_bin[*owner as usize] == bin as u8 {
                        let idx = shard.dedup_support(support, pos);
                        shard.cell_indices.push(idx);
                    } else {
                        let target = generator_bin[*owner as usize] as usize;
                        let source_slot = shard.cell_indices.len() as u32;
                        shard.support_overflow[target].push((support.to_vec(), pos, source_slot));
                        shard.cell_indices.push(DEFERRED);
                    }
                }
            }
        }
        shard.finish_cell(cell_idx);
    }
    shard.into_output()
}).collect();
```

### Phase 3: Flush Overflow (parallel by target)
```rust
// Collect overflow by target bin
let overflow_by_target: [Vec<_>; 12] = collect_overflow(&shard_outputs);

// Parallel flush - each target processes its column
(0..12).into_par_iter().for_each(|target| {
    let shard = &mut shard_outputs[target];
    for (a, b, c, pos) in overflow_by_target[target].drain(..) {
        let local_a = global_to_local[a as usize];
        shard.dedup_triplet(local_a, b, c, pos);
    }
    // Similarly for support overflow
});
```

**Correction**: overflow entries must also carry `(source_bin, source_slot)` and
the flush must emit fixups so the originating shard can patch its `DEFERRED`
entries. In v1, do this sequentially to avoid parallel write-back.

### Phase 4: Assemble Final Output
```rust
// Compute vertex offsets
let shard_offsets = prefix_sum(shard_outputs.iter().map(|s| s.vertices.len()));

// Concatenate vertices
let all_vertices: Vec<Vec3> = shard_outputs.iter()
    .flat_map(|s| s.vertices.iter().copied())
    .collect();

// Remap cell indices and build final cells
// Each cell knows its shard (from generator_bin[cell_idx])
// local_idx → shard_offsets[shard] + local_idx
```
