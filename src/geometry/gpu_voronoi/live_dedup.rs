//! Live vertex deduplication during cell construction using sharded ownership.
//!
//! V1 design:
//! - Parallel cell building by spatial bin
//! - Single-threaded overflow flush (simplifies correctness)
//! - Per-cell duplicate index removal after overflow resolution

use glam::Vec3;
#[cfg(not(feature = "single-threaded"))]
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::Arc;

use super::cell_builder::F64CellBuilder;
use super::timing::{DedupSubPhases, Timer};
use super::{TerminationConfig, VertexKey};
use crate::geometry::VoronoiCell;
use crate::geometry::cube_grid::packed_knn::{
    packed_knn_cell_stream, PackedKnnCellScratch, PackedKnnCellStatus, PackedV4Edges,
};

const NIL: u32 = u32::MAX;
const DEFERRED: u64 = u64::MAX;

#[inline]
fn pack_ref(bin: u32, local: u32) -> u64 {
    ((bin as u64) << 32) | (local as u64)
}

#[inline]
fn unpack_ref(packed: u64) -> (u32, u32) {
    ((packed >> 32) as u32, (packed & 0xFFFF_FFFF) as u32)
}

#[inline]
fn pack_bc(b: u32, c: u32) -> u64 {
    (b as u64) | ((c as u64) << 32)
}

#[repr(C)]
struct TripletNode {
    bc: u64,
    idx: u32,
    next: u32,
}

#[derive(Clone, Copy)]
struct TripletOverflow {
    source_bin: u32,
    target_bin: u32,
    source_slot: u32,
    a: u32,
    b: u32,
    c: u32,
    pos: Vec3,
}

struct SupportOverflow {
    source_bin: u32,
    target_bin: u32,
    source_slot: u32,
    support: Vec<u32>,
    pos: Vec3,
}

struct BinAssignment {
    generator_bin: Vec<u32>,
    global_to_local: Vec<u32>,
    bin_generators: Vec<Vec<usize>>,
    num_bins: usize,
}

struct BinQuery {
    cell: u32,
    global: usize,
    local: u32,
}

struct PackedSeed<'a> {
    neighbors: &'a [u32],
    count: usize,
    security: f32,
    k: usize,
}

#[inline]
fn point_to_face_uv(p: Vec3) -> (usize, f32, f32) {
    let (x, y, z) = (p.x, p.y, p.z);
    let (ax, ay, az) = (x.abs(), y.abs(), z.abs());

    if ax >= ay && ax >= az {
        if x >= 0.0 {
            (0, -z / ax, y / ax)
        } else {
            (1, z / ax, y / ax)
        }
    } else if ay >= ax && ay >= az {
        if y >= 0.0 {
            (2, x / ay, -z / ay)
        } else {
            (3, x / ay, z / ay)
        }
    } else if z >= 0.0 {
        (4, x / az, y / az)
    } else {
        (5, -x / az, y / az)
    }
}

#[inline]
fn point_to_bin(p: Vec3, tile_res: usize) -> usize {
    debug_assert!(tile_res >= 1);
    let (face, u, v) = point_to_face_uv(p);
    let fu = ((u + 1.0) * 0.5) * (tile_res as f32);
    let fv = ((v + 1.0) * 0.5) * (tile_res as f32);
    let iu = (fu as usize).min(tile_res - 1);
    let iv = (fv as usize).min(tile_res - 1);
    face * tile_res * tile_res + iv * tile_res + iu
}

fn choose_tile_res() -> usize {
    let threads = rayon::current_num_threads().max(1);
    let target_bins = (threads * 2).clamp(6, 96);
    let target_per_face = (target_bins as f64 / 6.0).max(1.0);
    (target_per_face.sqrt().ceil() as usize).clamp(1, 8)
}

fn assign_bins(points: &[Vec3]) -> BinAssignment {
    let n = points.len();
    let tile_res = choose_tile_res();
    let num_bins = 6 * tile_res * tile_res;

    let mut generator_bin: Vec<u32> = Vec::with_capacity(n);
    let mut counts: Vec<usize> = vec![0; num_bins];
    for &p in points {
        let b = point_to_bin(p, tile_res);
        generator_bin.push(b as u32);
        counts[b] += 1;
    }

    let mut bin_generators: Vec<Vec<usize>> = (0..num_bins)
        .map(|b| Vec::with_capacity(counts[b]))
        .collect();
    for (i, &b) in generator_bin.iter().enumerate() {
        bin_generators[b as usize].push(i);
    }

    let mut global_to_local: Vec<u32> = vec![0; n];
    for generators in &bin_generators {
        for (local_idx, &global_idx) in generators.iter().enumerate() {
            global_to_local[global_idx] = local_idx as u32;
        }
    }

    BinAssignment {
        generator_bin,
        global_to_local,
        bin_generators,
        num_bins,
    }
}

struct ShardState {
    vertices: Vec<Vec3>,
    heads: Vec<u32>,
    nodes: Vec<TripletNode>,
    support_map: FxHashMap<Vec<u32>, u32>,
    support_data: Vec<u32>,
    cell_indices: Vec<u64>,
    cell_starts: Vec<u32>,
    cell_counts: Vec<u8>,
    triplet_overflow: Vec<TripletOverflow>,
    support_overflow: Vec<SupportOverflow>,
    triplet_keys: u64,
    support_keys: u64,
}

impl ShardState {
    fn new(num_local_generators: usize) -> Self {
        Self {
            vertices: Vec::new(),
            heads: vec![NIL; num_local_generators],
            nodes: Vec::new(),
            support_map: FxHashMap::default(),
            support_data: Vec::new(),
            cell_indices: Vec::new(),
            cell_starts: vec![0; num_local_generators],
            cell_counts: vec![0; num_local_generators],
            triplet_overflow: Vec::new(),
            support_overflow: Vec::new(),
            triplet_keys: 0,
            support_keys: 0,
        }
    }

    #[inline(always)]
    fn dedup_triplet(&mut self, local_a: u32, b: u32, c: u32, pos: Vec3) -> u32 {
        let bc = pack_bc(b, c);
        let mut node_id = self.heads[local_a as usize];
        while node_id != NIL {
            let node = &self.nodes[node_id as usize];
            if node.bc == bc {
                return node.idx;
            }
            node_id = node.next;
        }

        let idx = self.vertices.len() as u32;
        self.vertices.push(pos);
        let new_id = self.nodes.len() as u32;
        self.nodes.push(TripletNode {
            bc,
            idx,
            next: self.heads[local_a as usize],
        });
        self.heads[local_a as usize] = new_id;
        idx
    }

    #[inline(always)]
    fn dedup_support_owned(&mut self, support: Vec<u32>, pos: Vec3) -> u32 {
        if let Some(&idx) = self.support_map.get(support.as_slice()) {
            return idx;
        }
        let idx = self.vertices.len() as u32;
        self.vertices.push(pos);
        self.support_map.insert(support, idx);
        idx
    }
}

pub(super) struct ShardedCellsData {
    assignment: BinAssignment,
    shards: Vec<ShardState>,
    pub(super) cell_sub: super::timing::CellSubAccum,
}

fn with_two_mut<T>(v: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    assert!(i != j);
    if i < j {
        let (a, b) = v.split_at_mut(j);
        (&mut a[i], &mut b[0])
    } else {
        let (a, b) = v.split_at_mut(i);
        (&mut b[0], &mut a[j])
    }
}

pub(super) fn build_cells_sharded_live_dedup(
    points: &[Vec3],
    knn: &super::CubeMapGridKnn,
    termination: TerminationConfig,
) -> ShardedCellsData {
    let assignment = assign_bins(points);
    let num_bins = assignment.num_bins;
    let packed_k = super::KNN_RESUME_KS[0].min(points.len().saturating_sub(1));
    let packed_edges = Arc::new(PackedV4Edges::new(knn.grid().res()));

    #[cfg(not(feature = "single-threaded"))]
    let iter = (0..num_bins).into_par_iter();
    #[cfg(feature = "single-threaded")]
    let iter = 0..num_bins;

    let per_bin: Vec<(ShardState, super::timing::CellSubAccum)> = iter
        .map(|bin_usize| {
            use super::timing::{CellSubAccum, KnnCellStage, Timer};

            let bin = bin_usize as u32;
            let my_generators = &assignment.bin_generators[bin_usize];
            let mut shard = ShardState::new(my_generators.len());

            let mut scratch = knn.make_scratch();
            let mut builder = F64CellBuilder::new(0, Vec3::ZERO);
            let mut sub_accum = CellSubAccum::new();

            shard
                .vertices
                .reserve(my_generators.len().saturating_mul(4));
            shard.nodes.reserve(my_generators.len().saturating_mul(4));
            shard
                .cell_indices
                .reserve(my_generators.len().saturating_mul(6));
            shard
                .support_data
                .reserve(my_generators.len().saturating_mul(2));

            let grid = knn.grid();
            let packed_edges = packed_edges.clone();
            let mut packed_scratch = PackedKnnCellScratch::new();
            let mut packed_queries: Vec<u32> = Vec::new();

            let mut bin_queries: Vec<BinQuery> = Vec::with_capacity(my_generators.len());
            for &i in my_generators {
                let cell = grid.point_index_to_cell(i) as u32;
                let local = assignment.global_to_local[i];
                bin_queries.push(BinQuery {
                    cell,
                    global: i,
                    local,
                });
            }
            bin_queries.sort_unstable_by_key(|q| q.cell);

            let mut process_cell =
                |cell_sub: &mut super::timing::CellSubAccum,
                 i: usize,
                 local: u32,
                 packed: Option<PackedSeed>| {
                builder.reset(i, points[i]);

                let cell_start = shard.cell_indices.len() as u32;
                shard.cell_starts[local as usize] = cell_start;

                let mut cell_neighbors_processed = 0usize;
                let mut terminated = false;
                let mut knn_exhausted = false;
                let mut full_scan_done = false;
                let mut did_full_scan_fallback = false;
                let mut did_full_scan_recovery = false;
                let mut used_knn = false;

                let mut worst_cos = 1.0f32;
                let max_neighbors = points.len().saturating_sub(1);
                let mut neighbors: Vec<usize> = Vec::with_capacity(super::KNN_RESTART_MAX);
                let mut processed = 0usize;
                let mut max_k_requested = 0usize;
                let mut knn_stage = KnnCellStage::Resume(super::KNN_RESUME_KS[0]);
                let mut did_reach_knn_limit = false;

                let mut did_packed = false;
                let mut packed_count = 0usize;
                let mut packed_security = 0.0f32;
                let mut packed_k_local = 0usize;

                if let Some(seed) = packed {
                    did_packed = true;
                    packed_count = seed.count;
                    packed_security = seed.security;
                    packed_k_local = seed.k;

                    if packed_count > 0 {
                        let t_clip = Timer::start();
                        for &neighbor_idx in seed.neighbors {
                            let neighbor_idx = neighbor_idx as usize;
                            if neighbor_idx == i {
                                continue;
                            }
                            let neighbor = points[neighbor_idx];
                            if builder.clip(neighbor_idx, neighbor).is_err() {
                                break;
                            }
                            cell_neighbors_processed += 1;
                            let dot = points[i].dot(neighbor);
                            worst_cos = worst_cos.min(dot);

                            if termination.enabled && builder.vertex_count() >= 3 {
                                if termination.should_check(cell_neighbors_processed)
                                    && builder.can_terminate(worst_cos)
                                {
                                    terminated = true;
                                    break;
                                }
                            }
                        }
                        cell_sub.add_clip(t_clip.elapsed());
                    }

                    if termination.enabled && !terminated && builder.vertex_count() >= 3 {
                        let bound = if packed_count == packed_k_local {
                            worst_cos
                        } else {
                            packed_security
                        };
                        if builder.can_terminate(bound) {
                            terminated = true;
                        }
                    }
                }

                let resume_track_limit = (*super::KNN_RESUME_KS.last().unwrap()).min(max_neighbors);
                for (stage_idx, &k_stage) in super::KNN_RESUME_KS.iter().enumerate() {
                    if terminated || knn_exhausted || builder.is_failed() {
                        break;
                    }
                    let k = k_stage.min(resume_track_limit);
                    if k == 0 || k <= processed {
                        continue;
                    }
                    used_knn = true;
                    max_k_requested = max_k_requested.max(k);
                    let t_knn = Timer::start();
                    let status = if stage_idx == 0 {
                        knn.knn_resumable_into(
                            points[i],
                            i,
                            k,
                            resume_track_limit,
                            &mut scratch,
                            &mut neighbors,
                        )
                    } else {
                        knn.knn_resume_append_into(
                            points[i],
                            i,
                            processed,
                            k,
                            &mut scratch,
                            &mut neighbors,
                        )
                    };
                    cell_sub.add_knn(t_knn.elapsed());

                    // Track which resume stage we're at
                    knn_stage = KnnCellStage::Resume(k_stage);

                    let t_clip = Timer::start();
                    for &neighbor_idx in &neighbors[processed..] {
                        if did_packed && builder.has_neighbor(neighbor_idx) {
                            continue;
                        }
                        let neighbor = points[neighbor_idx];
                        if builder.clip(neighbor_idx, neighbor).is_err() {
                            break;
                        }
                        cell_neighbors_processed += 1;
                        let dot = points[i].dot(neighbor);
                        worst_cos = worst_cos.min(dot);

                        if termination.enabled && builder.vertex_count() >= 3 {
                            if termination.should_check(cell_neighbors_processed)
                                && builder.can_terminate(worst_cos)
                            {
                                terminated = true;
                                break;
                            }
                        }
                    }
                    cell_sub.add_clip(t_clip.elapsed());
                    processed = neighbors.len();

                    if terminated {
                        knn_exhausted = status == crate::geometry::cube_grid::KnnStatus::Exhausted;
                        break;
                    }

                    if status == crate::geometry::cube_grid::KnnStatus::Exhausted {
                        knn_exhausted = true;
                        break;
                    }
                }

                if !terminated && !knn_exhausted && !builder.is_failed() {
                    for &k_stage in super::KNN_RESTART_KS.iter() {
                        let k = k_stage.min(max_neighbors);
                        if k == 0 || k <= max_k_requested {
                            continue;
                        }
                        used_knn = true;
                        max_k_requested = k;
                        neighbors.clear();

                        let t_knn = Timer::start();
                        let status = knn.knn_resumable_into(
                            points[i],
                            i,
                            k,
                            k,
                            &mut scratch,
                            &mut neighbors,
                        );
                        cell_sub.add_knn(t_knn.elapsed());

                        // Track which restart stage we're at
                        knn_stage = KnnCellStage::Restart(k_stage);

                        let t_clip = Timer::start();
                        for &neighbor_idx in &neighbors {
                            if builder.has_neighbor(neighbor_idx) {
                                continue;
                            }
                            let neighbor = points[neighbor_idx];
                            if builder.clip(neighbor_idx, neighbor).is_err() {
                                break;
                            }
                            cell_neighbors_processed += 1;
                            let dot = points[i].dot(neighbor);
                            worst_cos = worst_cos.min(dot);

                            if termination.enabled && builder.vertex_count() >= 3 {
                                if termination.should_check(cell_neighbors_processed)
                                    && builder.can_terminate(worst_cos)
                                {
                                    terminated = true;
                                    break;
                                }
                            }
                        }
                        cell_sub.add_clip(t_clip.elapsed());

                        if terminated {
                            break;
                        }

                        knn_exhausted =
                            status == crate::geometry::cube_grid::KnnStatus::Exhausted;
                        if knn_exhausted {
                            break;
                        }
                    }
                }

                let max_knn_target = super::KNN_RESTART_MAX.min(max_neighbors);
                if !terminated && !knn_exhausted && max_k_requested >= max_knn_target {
                    did_reach_knn_limit = true;
                }

                // Final termination check at the end
                if termination.enabled && !terminated && builder.vertex_count() >= 3 {
                    let bound = if used_knn {
                        worst_cos
                    } else if did_packed && packed_count < packed_k_local {
                        packed_security
                    } else {
                        worst_cos
                    };
                    if builder.can_terminate(bound) {
                        terminated = true;
                    }
                }

                // Full scan fallback if KNN is exhausted or we hit the schedule limit and still not terminated
                if termination.enabled
                    && !terminated
                    && (knn_exhausted || did_reach_knn_limit)
                    && !builder.is_failed()
                    && builder.vertex_count() >= 3
                {
                    did_full_scan_fallback = true;
                    let already_clipped: rustc_hash::FxHashSet<usize> =
                        builder.neighbor_indices_iter().collect();
                    for (p_idx, &p) in points.iter().enumerate() {
                        if p_idx == i || already_clipped.contains(&p_idx) {
                            continue;
                        }
                        if builder.clip(p_idx, p).is_err() {
                            break;
                        }
                        cell_neighbors_processed += 1;
                    }
                    full_scan_done = true;
                }

                // Failed cell recovery
                if builder.is_failed() {
                    let recovered = builder.try_reseed_best();
                    if !recovered {
                        if !full_scan_done {
                            did_full_scan_recovery = true;
                            builder.reset(i, points[i]);
                            for (p_idx, &p) in points.iter().enumerate() {
                                if p_idx == i {
                                    continue;
                                }
                                if builder.clip(p_idx, p).is_err() {
                                    break;
                                }
                            }
                            full_scan_done = true;
                        }
                        let recovered = if builder.is_failed() {
                            builder.try_reseed_best()
                        } else {
                            builder.vertex_count() >= 3
                        };
                        if !recovered {
                            panic!(
                                "TODO: reseed/full-scan recovery failed for cell {} (planes={})",
                                i,
                                builder.planes_count()
                            );
                        }
                    }
                }

                let knn_stage = if did_full_scan_recovery {
                    KnnCellStage::FullScanRecovery
                } else if did_full_scan_fallback {
                    KnnCellStage::FullScanFallback
                } else {
                    knn_stage
                };
                cell_sub.add_cell_stage(knn_stage, knn_exhausted, cell_neighbors_processed);

                // Phase 4: Extract vertices with certified keys
                let t_cert = Timer::start();
                if builder.is_failed() || builder.vertex_count() < 3 {
                    panic!(
                        "Cell {} construction failed: failure={:?}, vertex_count={}",
                        i,
                        builder.failure(),
                        builder.vertex_count()
                    );
                }
                let cell_vertices = builder.to_vertex_data(points, &mut shard.support_data)
                    .unwrap_or_else(|e| panic!("Cell {} certification failed: {:?}", i, e));
                cell_sub.add_cert(t_cert.elapsed());

                let count = cell_vertices.len();
                shard.cell_counts[local as usize] =
                    u8::try_from(count).expect("cell vertex count exceeds u8 capacity");

                let t_keys = Timer::start();
                for (key, pos) in cell_vertices {
                    match key {
                        VertexKey::Triplet([a, b, c]) => {
                            shard.triplet_keys += 1;
                            let a_usize = a as usize;
                            let owner_bin = assignment.generator_bin[a_usize];
                            if owner_bin == bin {
                                let local_a = assignment.global_to_local[a_usize];
                                let idx = shard.dedup_triplet(local_a, b, c, pos);
                                shard.cell_indices.push(pack_ref(bin, idx));
                            } else {
                                let source_slot = shard.cell_indices.len() as u32;
                                shard.cell_indices.push(DEFERRED);
                                shard.triplet_overflow.push(TripletOverflow {
                                    source_bin: bin,
                                    target_bin: owner_bin,
                                    source_slot,
                                    a,
                                    b,
                                    c,
                                    pos,
                                });
                            }
                        }
                        VertexKey::Support { start, len } => {
                            shard.support_keys += 1;
                            let start = start as usize;
                            let len = len as usize;
                            let support: Vec<u32> = shard.support_data[start..start + len].to_vec();
                            let owner =
                                *support.iter().min().expect("support set must be non-empty");
                            let owner_bin = assignment.generator_bin[owner as usize];
                            if owner_bin == bin {
                                let idx = shard.dedup_support_owned(support, pos);
                                shard.cell_indices.push(pack_ref(bin, idx));
                            } else {
                                let source_slot = shard.cell_indices.len() as u32;
                                shard.cell_indices.push(DEFERRED);
                                shard.support_overflow.push(SupportOverflow {
                                    source_bin: bin,
                                    target_bin: owner_bin,
                                    source_slot,
                                    support,
                                    pos,
                                });
                            }
                        }
                    }
                }
                cell_sub.add_key_dedup(t_keys.elapsed());

                debug_assert_eq!(
                    shard.cell_indices.len() as u32 - cell_start,
                    count as u32,
                    "cell index stream mismatch"
                );

                let _ = (full_scan_done, cell_neighbors_processed);
            };

            let mut cursor = 0usize;
            while cursor < bin_queries.len() {
                let cell = bin_queries[cursor].cell;
                let start = cursor;
                while cursor < bin_queries.len() && bin_queries[cursor].cell == cell {
                    cursor += 1;
                }
                let group = &bin_queries[start..cursor];

                if packed_k > 0 {
                    packed_queries.clear();
                    if packed_queries.capacity() < group.len() {
                        packed_queries.reserve(group.len().saturating_sub(packed_queries.len()));
                    }
                    for q in group {
                        packed_queries.push(q.global as u32);
                    }

                    let t_packed = Timer::start();
                    let status = packed_knn_cell_stream(
                        grid,
                        points,
                        cell as usize,
                        &packed_queries,
                        packed_k,
                        packed_edges.as_ref(),
                        &mut packed_scratch,
                        |qi, query_idx, neighbors, count, security| {
                            let local = group[qi].local;
                            let seed = PackedSeed {
                                neighbors,
                                count,
                                security,
                                k: packed_k,
                            };
                            process_cell(&mut sub_accum, query_idx as usize, local, Some(seed));
                        },
                    );
                    let packed_elapsed = t_packed.elapsed();

                    if status == PackedKnnCellStatus::SlowPath {
                        for q in group {
                            process_cell(&mut sub_accum, q.global, q.local, None);
                        }
                    }
                    sub_accum.add_packed_knn(packed_elapsed);
                } else {
                    for q in group {
                        process_cell(&mut sub_accum, q.global, q.local, None);
                    }
                }
            }

            (shard, sub_accum)
        })
        .collect();

    let mut shards: Vec<ShardState> = Vec::with_capacity(num_bins);
    let mut merged_sub = super::timing::CellSubAccum::new();
    for (shard, sub) in per_bin {
        merged_sub.merge(&sub);
        shards.push(shard);
    }

    ShardedCellsData {
        assignment,
        shards,
        cell_sub: merged_sub,
    }
}

pub(super) fn assemble_sharded_live_dedup(
    mut data: ShardedCellsData,
) -> (Vec<Vec3>, Vec<VoronoiCell>, Vec<usize>, DedupSubPhases) {
    let t0 = Timer::start();

    let num_bins = data.assignment.num_bins;

    // Phase 3: collect overflow by target bin
    let mut triplet_by_target: Vec<Vec<TripletOverflow>> = vec![Vec::new(); num_bins];
    let mut support_by_target: Vec<Vec<SupportOverflow>> =
        (0..num_bins).map(|_| Vec::new()).collect();
    for shard in &mut data.shards {
        for entry in shard.triplet_overflow.drain(..) {
            triplet_by_target[entry.target_bin as usize].push(entry);
        }
        for entry in shard.support_overflow.drain(..) {
            support_by_target[entry.target_bin as usize].push(entry);
        }
    }

    #[allow(unused_variables)]
    let overflow_collect_time = t0.elapsed();
    let t1 = Timer::start();

    // Phase 3: overflow flush (V1: single-threaded)
    for target in 0..num_bins {
        // Triplets
        for entry in triplet_by_target[target].drain(..) {
            let source = entry.source_bin as usize;
            let target = target;
            debug_assert_ne!(source, target, "overflow should not target same bin");
            let (source_shard, target_shard) = with_two_mut(&mut data.shards, source, target);

            let local_a = data.assignment.global_to_local[entry.a as usize];
            let idx = target_shard.dedup_triplet(local_a, entry.b, entry.c, entry.pos);
            source_shard.cell_indices[entry.source_slot as usize] = pack_ref(target as u32, idx);
        }

        // Support sets
        for entry in support_by_target[target].drain(..) {
            let source = entry.source_bin as usize;
            let target = target;
            debug_assert_ne!(source, target, "overflow should not target same bin");
            let (source_shard, target_shard) = with_two_mut(&mut data.shards, source, target);

            let idx = target_shard.dedup_support_owned(entry.support, entry.pos);
            source_shard.cell_indices[entry.source_slot as usize] = pack_ref(target as u32, idx);
        }
    }

    #[cfg(debug_assertions)]
    for shard in &data.shards {
        debug_assert!(
            !shard.cell_indices.iter().any(|&x| x == DEFERRED),
            "unresolved deferred indices remain after overflow flush"
        );
    }

    #[allow(unused_variables)]
    let overflow_flush_time = t1.elapsed();
    let t2 = Timer::start();

    // Phase 4: concatenate vertices
    let mut vertex_offsets: Vec<usize> = vec![0; num_bins];
    let mut total_vertices = 0usize;
    for (bin, shard) in data.shards.iter().enumerate() {
        vertex_offsets[bin] = total_vertices;
        total_vertices += shard.vertices.len();
    }

    let mut all_vertices: Vec<Vec3> = Vec::with_capacity(total_vertices);
    for shard in &data.shards {
        all_vertices.extend_from_slice(&shard.vertices);
    }

    let num_cells = data.assignment.generator_bin.len();
    #[allow(unused_variables)]
    let concat_vertices_time = t2.elapsed();
    let t3 = Timer::start();

    // Phase 4: emit cells in generator index order.
    let mut cells: Vec<VoronoiCell> = Vec::with_capacity(num_cells);
    let mut cell_indices: Vec<usize> =
        Vec::with_capacity(data.shards.iter().map(|s| s.cell_indices.len()).sum());
    // Cells are small (<= MAX_VERTICES), so a compact linear "seen" set is often faster than
    // hashing or random-access marks arrays.
    let mut seen: [usize; super::MAX_VERTICES] = [0usize; super::MAX_VERTICES];
    #[allow(unused_mut)]
    let mut dupes_removed = 0u64;
    for gen_idx in 0..num_cells {
        let bin = data.assignment.generator_bin[gen_idx] as usize;
        let local = data.assignment.global_to_local[gen_idx] as usize;
        let shard = &data.shards[bin];
        let start = shard.cell_starts[local] as usize;
        let count = shard.cell_counts[local] as usize;

        let base = cell_indices.len();
        let mut seen_len = 0usize;
        for &packed in &shard.cell_indices[start..start + count] {
            debug_assert_ne!(packed, DEFERRED, "deferred index leaked to assembly");
            let (vbin, local) = unpack_ref(packed);
            let global = vertex_offsets[vbin as usize] + local as usize;
            let mut duplicate = false;
            for &idx in &seen[..seen_len] {
                if idx == global {
                    duplicate = true;
                    break;
                }
            }
            if !duplicate {
                debug_assert!(seen_len < super::MAX_VERTICES);
                seen[seen_len] = global;
                seen_len += 1;
                cell_indices.push(global);
            } else {
                dupes_removed += 1;
            }
        }
        let new_count = cell_indices.len() - base;
        cells.push(VoronoiCell::new(gen_idx, base, new_count));
    }
    #[allow(unused_variables)]
    let emit_cells_time = t3.elapsed();

    #[cfg(feature = "timing")]
    let sub_phases = DedupSubPhases {
        setup: overflow_collect_time + concat_vertices_time,
        lookup: overflow_flush_time,
        cell_dedup: emit_cells_time,
        overflow_collect: overflow_collect_time,
        overflow_flush: overflow_flush_time,
        concat_vertices: concat_vertices_time,
        emit_cells: emit_cells_time,
        triplet_keys: data.shards.iter().map(|s| s.triplet_keys).sum(),
        support_keys: data.shards.iter().map(|s| s.support_keys).sum(),
        cell_dupes_removed: dupes_removed,
    };
    #[cfg(not(feature = "timing"))]
    let sub_phases = DedupSubPhases;

    (all_vertices, cells, cell_indices, sub_phases)
}
