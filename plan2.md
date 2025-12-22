# Live Dedup During Cell Build (Concise Plan)

## Goal
Replace post-hoc global dedup with **live dedup during parallel cell building**, using a simple ownership rule so the hot path stays lock-free.

## V1 Scope
- Phase 2 (cell building) runs **in parallel** by shard/bin.
- Phase 3 (overflow flush) runs **single-threaded** for correctness simplicity.
- Per-cell duplicate index removal runs **after** all deferred indices are patched.

## Key Idea: Ownership
- `VertexKey::Triplet([a,b,c])`: owner is `a` (the minimum support index in current code).
- `VertexKey::Support{..}`: owner is `min(support_set)`.
- A shard “owns” a key if the owner generator falls in that shard’s bin.

## Partitioning (`point_to_bin`)
- Must be deterministic with explicit tie-breaking.
- Prefer a cube-face based partition (aligned with existing cube projection logic).
- Make `NUM_BINS` configurable; coarser bins usually improve locality (less overflow), but can worsen load balance.

## Data Structures (V1)
Per run:
- `generator_bin: Vec<u8>`
- `bin_generators: Vec<Vec<usize>>` (global generator indices per bin)
- `global_to_local: Vec<u32>` (global idx → local idx within its bin)

Per shard/bin:
- `vertices: Vec<Vec3>`
- `triplet_heads: Vec<u32>` sized to generators in this bin (indexed by `local_a`)
- `triplet_nodes: Vec<TripletNode>` for linked-list lookup by `(b,c)`
- `support_map: FxHashMap<Vec<u32>, u32>` (degenerate keys)
- `cell_indices: Vec<u32>` (local vertex indices; `DEFERRED` placeholders allowed)
- `cell_counts: Vec<u8>` (vertices per cell)
- `triplet_overflow[target_bin]: Vec<(a,b,c,pos,source_slot:u32)>`
- `support_overflow[target_bin]: Vec<(support_set,pos,source_slot:u32)>`

## Phases

### Phase 1: Bin assignment
Compute `generator_bin`, build `bin_generators`, and `global_to_local`.

### Phase 2: Parallel cell building + local live dedup
For each bin in parallel:
1. For each generator in `bin_generators[bin]`, build the cell (existing `F64CellBuilder` flow).
2. For each produced `(VertexKey, pos)`:
   - If owned: dedup into shard-local structures, push local index into `cell_indices`.
   - Else: push `DEFERRED` into `cell_indices`, and enqueue overflow with `(key,pos,source_slot)`.
3. Record `cell_counts` for later assembly.

**Important**: do not compact or dedup per-cell indices yet; `source_slot` must remain valid.

### Phase 3: Overflow flush (single-threaded in V1)
For each `target_bin`:
- Drain all `(…, source_slot)` entries targeting `target_bin`.
- Dedup into target shard; get `target_local_idx`.
- Patch the originating shard’s `cell_indices[source_slot] = (target_bin, target_local_idx)` representation.

Implementation detail: in v1, simplest is to store deferred entries as a tagged pair
or temporarily store `(target_bin, target_local_idx)` in side tables and patch later.

### Phase 4: Assemble output
1. Compute per-shard vertex offsets and concatenate `vertices`.
2. Remap `(bin, local_idx)` → global vertex index using prefix sums.
3. Run per-cell duplicate index removal (equivalent to current `deduplicate_cell_indices()` semantics).
4. Build final `SphericalVoronoi` output in original generator order.

## Correctness/Integration Notes
- Cells must be emitted in original generator index order (even if built by bins).
- Support sets must be canonicalized (sorted, deduped) and ownership must match everywhere.
- Keep old `FlatCellsData` path for stats/tests until the new path matches behavior.

## V2 (Optional Improvements)
- Parallelize Phase 3 using fixup records (avoid target threads mutating source shards).
- Consider more bins + work-stealing within bins for better CPU utilization without exploding overflow.

