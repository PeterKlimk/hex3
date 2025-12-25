//! Batched k-NN using PackedV4 filtering for unit vectors on a cube-map grid.
//!
//! Dominant mode: PackedV4 (batched, cell-local, SIMD dot products).

use super::{cell_to_face_ij, face_uv_to_3d, st_to_uv, CubeMapGrid};
use glam::Vec3;
use std::mem::MaybeUninit;
use std::simd::f32x8;
use std::simd::{cmp::SimdPartialOrd, Mask};

const MAX_CANDIDATES_FAST: usize = 512;

/// Result of packed k-NN: for each point, the indices of its k nearest neighbors.
/// Layout: [p0_n0, p0_n1, ..., p0_n(k-1), p1_n0, ...]
pub struct PackedKnnResult {
    pub neighbors: Vec<u32>,
    pub k: usize,
}

impl PackedKnnResult {
    #[inline]
    pub fn get(&self, point_idx: usize) -> &[u32] {
        let start = point_idx * self.k;
        &self.neighbors[start..start + self.k]
    }

    /// Returns the valid prefix of `get()` (stops at the first `u32::MAX` sentinel).
    ///
    /// Packed mode may return fewer than `k` neighbors for some points.
    #[inline]
    pub fn get_valid(&self, point_idx: usize) -> &[u32] {
        let slice = self.get(point_idx);
        let len = slice
            .iter()
            .position(|&idx| idx == u32::MAX)
            .unwrap_or(self.k);
        &slice[..len]
    }
}

/// Statistics from PackedV4 batched k-NN.
#[derive(Clone, Debug, Default)]
pub struct PackedKnnStats {
    pub total_candidates: u64,
    pub filtered_out: u64,
    pub fallback_queries: u64,
    pub slow_path_cells: u64,
    pub under_k_count: u64,
    pub num_queries: u64,
}

impl PackedKnnStats {
    pub fn filter_rate(&self) -> f64 {
        if self.total_candidates == 0 {
            0.0
        } else {
            self.filtered_out as f64 / self.total_candidates as f64
        }
    }
}

/// Precomputed plane-edge data for PackedV4 security thresholds.
pub struct PackedV4Edges {
    edges: Vec<[EdgeData; 4]>,
}

impl PackedV4Edges {
    pub fn new(res: usize) -> Self {
        Self {
            edges: precompute_edges_all_cells(res),
        }
    }

    #[inline]
    fn get(&self, cell: usize) -> &[EdgeData; 4] {
        &self.edges[cell]
    }
}

/// Reusable scratch buffers for packed per-cell streaming queries.
pub struct PackedKnnCellScratch {
    candidate_indices: Vec<u32>,
    cell_ranges: Vec<(usize, usize)>,
    keys_slab: Vec<MaybeUninit<u64>>,
    lens: Vec<usize>,
    center_lens: Vec<usize>,
    min_center_dot: Vec<f32>,
    security_thresholds: Vec<f32>,
    thresholds: Vec<f32>,
    query_x: Vec<f32>,
    query_y: Vec<f32>,
    query_z: Vec<f32>,
    neighbors: Vec<u32>,
}

impl PackedKnnCellScratch {
    pub fn new() -> Self {
        Self {
            candidate_indices: Vec::with_capacity(MAX_CANDIDATES_FAST),
            cell_ranges: Vec::with_capacity(9),
            keys_slab: Vec::new(),
            lens: Vec::new(),
            center_lens: Vec::new(),
            min_center_dot: Vec::new(),
            security_thresholds: Vec::new(),
            thresholds: Vec::new(),
            query_x: Vec::new(),
            query_y: Vec::new(),
            query_z: Vec::new(),
            neighbors: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedKnnCellStatus {
    Ok,
    SlowPath,
}

/// PackedV4 per-cell k-NN for a subset of queries, streaming results to a callback.
///
/// Queries are assumed to lie in the center cell (same as the full packed path),
/// but may be a strict subset of that cell's points.
pub fn packed_knn_cell_stream(
    grid: &CubeMapGrid,
    points: &[Vec3],
    cell: usize,
    queries: &[u32],
    k: usize,
    edges: &PackedV4Edges,
    scratch: &mut PackedKnnCellScratch,
    mut on_query: impl FnMut(usize, u32, &[u32], usize, f32),
) -> PackedKnnCellStatus {
    let num_queries = queries.len();
    if num_queries == 0 || k == 0 {
        return PackedKnnCellStatus::Ok;
    }

    let num_cells = 6 * grid.res * grid.res;
    if cell >= num_cells {
        return PackedKnnCellStatus::Ok;
    }

    scratch.candidate_indices.clear();
    scratch.cell_ranges.clear();

    let q_start = grid.cell_offsets[cell] as usize;
    let q_end = grid.cell_offsets[cell + 1] as usize;
    scratch
        .candidate_indices
        .extend_from_slice(&grid.point_indices[q_start..q_end]);
    scratch.cell_ranges.push((q_start, q_end));

    for &ncell in grid.cell_neighbors(cell) {
        if ncell == u32::MAX || ncell == cell as u32 {
            continue;
        }
        let nc = ncell as usize;
        let n_start = grid.cell_offsets[nc] as usize;
        let n_end = grid.cell_offsets[nc + 1] as usize;
        if n_start < n_end {
            scratch
                .candidate_indices
                .extend_from_slice(&grid.point_indices[n_start..n_end]);
            scratch.cell_ranges.push((n_start, n_end));
        }
    }

    let num_candidates = scratch.candidate_indices.len();
    if num_candidates == 0 {
        return PackedKnnCellStatus::Ok;
    }
    if num_candidates > MAX_CANDIDATES_FAST {
        return PackedKnnCellStatus::SlowPath;
    }

    let edges = edges.get(cell);

    let stride = num_candidates;
    let slab_size = num_queries * stride;
    scratch.keys_slab.clear();
    if scratch.keys_slab.capacity() < slab_size {
        scratch
            .keys_slab
            .reserve(slab_size.saturating_sub(scratch.keys_slab.len()));
    }
    unsafe { scratch.keys_slab.set_len(slab_size) };
    scratch.lens.resize(num_queries, 0);
    scratch.lens.fill(0);
    scratch.min_center_dot.resize(num_queries, f32::INFINITY);
    scratch.min_center_dot.fill(f32::INFINITY);

    scratch.query_x.resize(num_queries, 0.0);
    scratch.query_y.resize(num_queries, 0.0);
    scratch.query_z.resize(num_queries, 0.0);
    for (qi, &query_idx) in queries.iter().enumerate() {
        let q = points[query_idx as usize];
        scratch.query_x[qi] = q.x;
        scratch.query_y[qi] = q.y;
        scratch.query_z[qi] = q.z;
    }

    scratch.security_thresholds.clear();
    scratch.security_thresholds.reserve(num_queries);
    for qi in 0..num_queries {
        let q = Vec3::new(scratch.query_x[qi], scratch.query_y[qi], scratch.query_z[qi]);
        scratch.security_thresholds.push(security_planes(q, edges));
    }

    let (center_soa_start, center_soa_end) = scratch.cell_ranges[0];
    let center_len = center_soa_end - center_soa_start;
    let xs = &grid.cell_points_x[center_soa_start..center_soa_end];
    let ys = &grid.cell_points_y[center_soa_start..center_soa_end];
    let zs = &grid.cell_points_z[center_soa_start..center_soa_end];

    let full_chunks = center_len / 8;
    for chunk in 0..full_chunks {
        let i = chunk * 8;
        let cx = f32x8::from_slice(&xs[i..]);
        let cy = f32x8::from_slice(&ys[i..]);
        let cz = f32x8::from_slice(&zs[i..]);

        for qi in 0..num_queries {
            let qx = f32x8::splat(scratch.query_x[qi]);
            let qy = f32x8::splat(scratch.query_y[qi]);
            let qz = f32x8::splat(scratch.query_z[qi]);
            let dots = cx * qx + cy * qy + cz * qz;

            let thresh_vec = f32x8::splat(scratch.security_thresholds[qi]);
            let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
            let mut mask_bits = mask.to_bitmask() as u32;

            if mask_bits != 0 {
                let dots_arr: [f32; 8] = dots.into();
                let query_idx = queries[qi];
                while mask_bits != 0 {
                    let lane = mask_bits.trailing_zeros() as usize;
                    let cand_idx = i + lane;
                    let cand_global = scratch.candidate_indices[cand_idx];
                    if cand_global != query_idx {
                        let dot = dots_arr[lane];
                        let slab_idx = qi * stride + scratch.lens[qi];
                        scratch.keys_slab[slab_idx].write(make_desc_key(dot, cand_global));
                        scratch.lens[qi] += 1;
                        scratch.min_center_dot[qi] = scratch.min_center_dot[qi].min(dot);
                    }
                    mask_bits &= mask_bits - 1;
                }
            }
        }
    }

    let tail_start = full_chunks * 8;
    for i in tail_start..center_len {
        let cx = xs[i];
        let cy = ys[i];
        let cz = zs[i];
        let cand_global = scratch.candidate_indices[i];
        for qi in 0..num_queries {
            if cand_global == queries[qi] {
                continue;
            }
            let dot = cx * scratch.query_x[qi] + cy * scratch.query_y[qi] + cz * scratch.query_z[qi];
            if dot > scratch.security_thresholds[qi] {
                let slab_idx = qi * stride + scratch.lens[qi];
                scratch.keys_slab[slab_idx].write(make_desc_key(dot, cand_global));
                scratch.lens[qi] += 1;
                scratch.min_center_dot[qi] = scratch.min_center_dot[qi].min(dot);
            }
        }
    }

    scratch.center_lens.resize(num_queries, 0);
    scratch.center_lens.copy_from_slice(&scratch.lens);

    scratch.thresholds.clear();
    scratch.thresholds.reserve(num_queries);
    for qi in 0..num_queries {
        let threshold = if scratch.center_lens[qi] > 0 {
            scratch.security_thresholds[qi].max(scratch.min_center_dot[qi] - 1e-6)
        } else {
            scratch.security_thresholds[qi]
        };
        scratch.thresholds.push(threshold);
    }

    let mut cand_offset = center_len;
    for &(soa_start, soa_end) in &scratch.cell_ranges[1..] {
        let range_len = soa_end - soa_start;
        let xs = &grid.cell_points_x[soa_start..soa_end];
        let ys = &grid.cell_points_y[soa_start..soa_end];
        let zs = &grid.cell_points_z[soa_start..soa_end];

        let full_chunks = range_len / 8;
        for chunk in 0..full_chunks {
            let i = chunk * 8;
            let cx = f32x8::from_slice(&xs[i..]);
            let cy = f32x8::from_slice(&ys[i..]);
            let cz = f32x8::from_slice(&zs[i..]);

            for qi in 0..num_queries {
                let qx = f32x8::splat(scratch.query_x[qi]);
                let qy = f32x8::splat(scratch.query_y[qi]);
                let qz = f32x8::splat(scratch.query_z[qi]);
                let dots = cx * qx + cy * qy + cz * qz;

                let thresh_vec = f32x8::splat(scratch.thresholds[qi]);
                let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
                let mut mask_bits = mask.to_bitmask() as u32;

                if mask_bits != 0 {
                    let dots_arr: [f32; 8] = dots.into();
                    let query_idx = queries[qi];
                    while mask_bits != 0 {
                        let lane = mask_bits.trailing_zeros() as usize;
                        let cand_idx = cand_offset + i + lane;
                        let cand_global = scratch.candidate_indices[cand_idx];
                        if cand_global != query_idx {
                            let dot = dots_arr[lane];
                            let slab_idx = qi * stride + scratch.lens[qi];
                            scratch
                                .keys_slab[slab_idx]
                                .write(make_desc_key(dot, cand_global));
                            scratch.lens[qi] += 1;
                        }
                        mask_bits &= mask_bits - 1;
                    }
                }
            }
        }

        let tail_start = full_chunks * 8;
        for i in tail_start..range_len {
            let cx = xs[i];
            let cy = ys[i];
            let cz = zs[i];
            let cand_global = scratch.candidate_indices[cand_offset + i];
            for qi in 0..num_queries {
                if cand_global == queries[qi] {
                    continue;
                }
                let dot =
                    cx * scratch.query_x[qi] + cy * scratch.query_y[qi] + cz * scratch.query_z[qi];
                if dot > scratch.thresholds[qi] {
                    let slab_idx = qi * stride + scratch.lens[qi];
                    scratch.keys_slab[slab_idx].write(make_desc_key(dot, cand_global));
                    scratch.lens[qi] += 1;
                }
            }
        }

        cand_offset += range_len;
    }

    for qi in 0..num_queries {
        let ring_added = scratch.lens[qi] - scratch.center_lens[qi];
        let need = k.saturating_sub(scratch.center_lens[qi]);
        if ring_added < need {
            scratch.lens[qi] = scratch.center_lens[qi];
            let mut cand_offset = center_len;
            for &(soa_start, soa_end) in &scratch.cell_ranges[1..] {
                let range_len = soa_end - soa_start;
                let xs = &grid.cell_points_x[soa_start..soa_end];
                let ys = &grid.cell_points_y[soa_start..soa_end];
                let zs = &grid.cell_points_z[soa_start..soa_end];

                for i in 0..range_len {
                    let cand_global = scratch.candidate_indices[cand_offset + i];
                    if cand_global == queries[qi] {
                        continue;
                    }
                    let dot = xs[i] * scratch.query_x[qi]
                        + ys[i] * scratch.query_y[qi]
                        + zs[i] * scratch.query_z[qi];
                    if dot > scratch.security_thresholds[qi] {
                        let slab_idx = qi * stride + scratch.lens[qi];
                        scratch
                            .keys_slab[slab_idx]
                            .write(make_desc_key(dot, cand_global));
                        scratch.lens[qi] += 1;
                    }
                }
                cand_offset += range_len;
            }
        }
    }

    scratch.neighbors.resize(num_queries * k, u32::MAX);
    for (qi, &query_idx) in queries.iter().enumerate() {
        let m = scratch.lens[qi];
        let keys_uninit = &mut scratch.keys_slab[qi * stride..qi * stride + m];
        let keys_slice = unsafe {
            std::slice::from_raw_parts_mut(keys_uninit.as_mut_ptr() as *mut u64, m)
        };

        let k_actual = k.min(m);
        if k_actual > 0 {
            if m > k_actual {
                keys_slice.select_nth_unstable(k_actual - 1);
            }
            keys_slice[..k_actual].sort_unstable();

            let out_start = qi * k;
            for i in 0..k_actual {
                scratch.neighbors[out_start + i] = key_to_idx(keys_slice[i]);
            }
        }

        let out_start = qi * k;
        let out_end = out_start + k_actual;
        on_query(
            qi,
            query_idx,
            &scratch.neighbors[out_start..out_end],
            k_actual,
            scratch.security_thresholds[qi],
        );
    }

    PackedKnnCellStatus::Ok
}

/// PackedV4 batched k-NN (fast path, no stats instrumentation).
pub fn packed_knn(grid: &CubeMapGrid, points: &[Vec3], k: usize) -> PackedKnnResult {
    packed_knn_impl(grid, points, k, None)
}

/// PackedV4 batched k-NN with stats.
pub fn packed_knn_stats(
    grid: &CubeMapGrid,
    points: &[Vec3],
    k: usize,
) -> (PackedKnnResult, PackedKnnStats) {
    let mut stats = PackedKnnStats::default();
    let result = packed_knn_impl(grid, points, k, Some(&mut stats));
    (result, stats)
}

fn packed_knn_impl(
    grid: &CubeMapGrid,
    points: &[Vec3],
    k: usize,
    mut stats: Option<&mut PackedKnnStats>,
) -> PackedKnnResult {
    let n = points.len();
    let num_cells = 6 * grid.res * grid.res;
    let mut neighbors = vec![u32::MAX; n * k];

    let mut candidate_indices: Vec<u32> = Vec::with_capacity(512);
    let mut cell_ranges: Vec<(usize, usize)> = Vec::with_capacity(9);
    let all_edges = precompute_edges_all_cells(grid.res);

    let mut keys_slab: Vec<MaybeUninit<u64>> = Vec::new();
    let mut lens: Vec<usize> = Vec::new();
    let mut center_lens: Vec<usize> = Vec::new();
    let mut min_center_dot: Vec<f32> = Vec::new();
    let mut security_thresholds: Vec<f32> = Vec::new();
    let mut thresholds: Vec<f32> = Vec::new();
    let mut query_x: Vec<f32> = Vec::new();
    let mut query_y: Vec<f32> = Vec::new();
    let mut query_z: Vec<f32> = Vec::new();

    for cell in 0..num_cells {
        let query_points = grid.cell_points(cell);
        let num_queries = query_points.len();
        if num_queries == 0 {
            continue;
        }

        // Gather indices and track SoA ranges.
        candidate_indices.clear();
        cell_ranges.clear();

        let q_start = grid.cell_offsets[cell] as usize;
        let q_end = grid.cell_offsets[cell + 1] as usize;
        candidate_indices.extend_from_slice(&grid.point_indices[q_start..q_end]);
        cell_ranges.push((q_start, q_end));

        for &ncell in grid.cell_neighbors(cell) {
            if ncell == u32::MAX || ncell == cell as u32 {
                continue;
            }
            let nc = ncell as usize;
            let n_start = grid.cell_offsets[nc] as usize;
            let n_end = grid.cell_offsets[nc + 1] as usize;
            if n_start < n_end {
                candidate_indices.extend_from_slice(&grid.point_indices[n_start..n_end]);
                cell_ranges.push((n_start, n_end));
            }
        }

        let num_candidates = candidate_indices.len();
        if num_candidates == 0 {
            continue;
        }

        let edges = &all_edges[cell];

        // Worst-case: fall back to an unbounded (slow) path rather than silently truncating.
        if num_candidates > MAX_CANDIDATES_FAST {
            if let Some(stats) = stats.as_deref_mut() {
                stats.slow_path_cells += 1;
            }
            packed_knn_fallback_cell(
                grid,
                points,
                k,
                query_points,
                &cell_ranges,
                &candidate_indices,
                edges,
                &mut neighbors,
                stats.as_deref_mut(),
            );
            continue;
        }

        // Per-query state.
        let stride = num_candidates;
        let slab_size = num_queries * stride;
        keys_slab.clear();
        if keys_slab.capacity() < slab_size {
            keys_slab.reserve(slab_size.saturating_sub(keys_slab.len()));
        }
        unsafe { keys_slab.set_len(slab_size) };
        lens.resize(num_queries, 0);
        lens.fill(0);
        min_center_dot.resize(num_queries, f32::INFINITY);
        min_center_dot.fill(f32::INFINITY);

        let (center_soa_start, center_soa_end) = cell_ranges[0];
        let center_len = center_soa_end - center_soa_start;
        let xs = &grid.cell_points_x[center_soa_start..center_soa_end];
        let ys = &grid.cell_points_y[center_soa_start..center_soa_end];
        let zs = &grid.cell_points_z[center_soa_start..center_soa_end];

        // Cache query vectors (queries are exactly the center cell points).
        query_x.resize(num_queries, 0.0);
        query_y.resize(num_queries, 0.0);
        query_z.resize(num_queries, 0.0);
        for qi in 0..num_queries {
            query_x[qi] = xs[qi];
            query_y[qi] = ys[qi];
            query_z[qi] = zs[qi];
        }

        // Precompute security_3x3 per query.
        security_thresholds.clear();
        security_thresholds.reserve(num_queries);
        for qi in 0..num_queries {
            let q = Vec3::new(query_x[qi], query_y[qi], query_z[qi]);
            security_thresholds.push(security_planes(q, edges));
        }

        // Pass A: center range only - compute dots, filter, track min.
        let full_chunks = center_len / 8;
        for chunk in 0..full_chunks {
            let i = chunk * 8;
            let cx = f32x8::from_slice(&xs[i..]);
            let cy = f32x8::from_slice(&ys[i..]);
            let cz = f32x8::from_slice(&zs[i..]);

            for qi in 0..num_queries {
                let qx = f32x8::splat(query_x[qi]);
                let qy = f32x8::splat(query_y[qi]);
                let qz = f32x8::splat(query_z[qi]);
                let dots = cx * qx + cy * qy + cz * qz;

                let thresh_vec = f32x8::splat(security_thresholds[qi]);
                let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
                let mut mask_bits = mask.to_bitmask() as u32;

                // Clear self bit if in this chunk.
                let base = i;
                if base <= qi && qi < base + 8 {
                    mask_bits &= !(1u32 << (qi - base));
                }

                if mask_bits != 0 {
                    let dots_arr: [f32; 8] = dots.into();
                    while mask_bits != 0 {
                        let lane = mask_bits.trailing_zeros() as usize;
                        let cand_idx = i + lane;
                        let dot = dots_arr[lane];
                        let slab_idx = qi * stride + lens[qi];
                        keys_slab[slab_idx].write(make_desc_key(dot, candidate_indices[cand_idx]));
                        lens[qi] += 1;
                        min_center_dot[qi] = min_center_dot[qi].min(dot);
                        mask_bits &= mask_bits - 1;
                    }
                }
            }
        }

        // Center tail.
        let tail_start = full_chunks * 8;
        for i in tail_start..center_len {
            let cx = xs[i];
            let cy = ys[i];
            let cz = zs[i];
            for qi in 0..num_queries {
                if i == qi {
                    continue;
                }
                let dot = cx * query_x[qi] + cy * query_y[qi] + cz * query_z[qi];
                if dot > security_thresholds[qi] {
                    let slab_idx = qi * stride + lens[qi];
                    keys_slab[slab_idx].write(make_desc_key(dot, candidate_indices[i]));
                    lens[qi] += 1;
                    min_center_dot[qi] = min_center_dot[qi].min(dot);
                }
            }
        }

        center_lens.resize(num_queries, 0);
        center_lens.copy_from_slice(&lens);

        // Compute per-query thresholds for ring pass (PackedV4).
        thresholds.clear();
        thresholds.reserve(num_queries);
        for qi in 0..num_queries {
            let threshold = if center_lens[qi] > 0 {
                security_thresholds[qi].max(min_center_dot[qi] - 1e-6)
            } else {
                security_thresholds[qi]
            };
            thresholds.push(threshold);
        }

        // Pass B: ring ranges - compute dots, filter by threshold.
        let mut cand_offset = center_len;
        for &(soa_start, soa_end) in &cell_ranges[1..] {
            let range_len = soa_end - soa_start;
            let xs = &grid.cell_points_x[soa_start..soa_end];
            let ys = &grid.cell_points_y[soa_start..soa_end];
            let zs = &grid.cell_points_z[soa_start..soa_end];

            let full_chunks = range_len / 8;
            for chunk in 0..full_chunks {
                let i = chunk * 8;
                let cx = f32x8::from_slice(&xs[i..]);
                let cy = f32x8::from_slice(&ys[i..]);
                let cz = f32x8::from_slice(&zs[i..]);

                for qi in 0..num_queries {
                    let qx = f32x8::splat(query_x[qi]);
                    let qy = f32x8::splat(query_y[qi]);
                    let qz = f32x8::splat(query_z[qi]);
                    let dots = cx * qx + cy * qy + cz * qz;

                    let thresh_vec = f32x8::splat(thresholds[qi]);
                    let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
                    let mut mask_bits = mask.to_bitmask() as u32;

                    if mask_bits != 0 {
                        let dots_arr: [f32; 8] = dots.into();
                        while mask_bits != 0 {
                            let lane = mask_bits.trailing_zeros() as usize;
                            let cand_idx = cand_offset + i + lane;
                            let dot = dots_arr[lane];
                            let slab_idx = qi * stride + lens[qi];
                            keys_slab[slab_idx].write(make_desc_key(dot, candidate_indices[cand_idx]));
                            lens[qi] += 1;
                            mask_bits &= mask_bits - 1;
                        }
                    }
                }
            }

            // Ring tail.
            let tail_start = full_chunks * 8;
            for i in tail_start..range_len {
                let cx = xs[i];
                let cy = ys[i];
                let cz = zs[i];
                for qi in 0..num_queries {
                    let dot = cx * query_x[qi] + cy * query_y[qi] + cz * query_z[qi];
                    if dot > thresholds[qi] {
                        let slab_idx = qi * stride + lens[qi];
                        keys_slab[slab_idx].write(make_desc_key(dot, candidate_indices[cand_offset + i]));
                        lens[qi] += 1;
                    }
                }
            }

            cand_offset += range_len;
        }

        // Fallback: re-run ring with security threshold if we didn't get enough.
        for qi in 0..num_queries {
            let ring_added = lens[qi] - center_lens[qi];
            let need = k.saturating_sub(center_lens[qi]);
            if ring_added < need {
                if let Some(stats) = stats.as_deref_mut() {
                    stats.fallback_queries += 1;
                }
                lens[qi] = center_lens[qi];
                let mut cand_offset = center_len;
                for &(soa_start, soa_end) in &cell_ranges[1..] {
                    let range_len = soa_end - soa_start;
                    let xs = &grid.cell_points_x[soa_start..soa_end];
                    let ys = &grid.cell_points_y[soa_start..soa_end];
                    let zs = &grid.cell_points_z[soa_start..soa_end];

                    for i in 0..range_len {
                        let dot = xs[i] * query_x[qi] + ys[i] * query_y[qi] + zs[i] * query_z[qi];
                        if dot > security_thresholds[qi] {
                            let slab_idx = qi * stride + lens[qi];
                            keys_slab[slab_idx].write(make_desc_key(dot, candidate_indices[cand_offset + i]));
                            lens[qi] += 1;
                        }
                    }
                    cand_offset += range_len;
                }
            }
        }

        // Select+sort per query.
        for (qi, &query_idx) in query_points.iter().enumerate() {
            let m = lens[qi];
            if let Some(stats) = stats.as_deref_mut() {
                stats.total_candidates += num_candidates as u64;
                stats.filtered_out += (num_candidates - m) as u64;
                stats.num_queries += 1;
            }

            let keys_uninit = &mut keys_slab[qi * stride..qi * stride + m];
            let keys_slice = unsafe {
                std::slice::from_raw_parts_mut(keys_uninit.as_mut_ptr() as *mut u64, m)
            };

            let k_actual = k.min(m);
            if k_actual < k {
                if let Some(stats) = stats.as_deref_mut() {
                    stats.under_k_count += 1;
                }
            }
            if k_actual > 0 {
                if m > k_actual {
                    keys_slice.select_nth_unstable(k_actual - 1);
                }
                keys_slice[..k_actual].sort_unstable();

                let out_start = query_idx as usize * k;
                for i in 0..k_actual {
                    neighbors[out_start + i] = key_to_idx(keys_slice[i]);
                }
            }
        }
    }

    PackedKnnResult { neighbors, k }
}

/// Worst-case fallback when the per-cell candidate list is too large for the fixed-size fast path.
fn packed_knn_fallback_cell(
    grid: &CubeMapGrid,
    points: &[Vec3],
    k: usize,
    query_points: &[u32],
    cell_ranges: &[(usize, usize)],
    candidate_indices: &[u32],
    edges: &[EdgeData; 4],
    neighbors: &mut [u32],
    mut stats: Option<&mut PackedKnnStats>,
) {
    let (center_soa_start, center_soa_end) = cell_ranges[0];
    let center_len = center_soa_end - center_soa_start;

    let mut keys: Vec<u64> = Vec::new();

    for (qi, &query_idx) in query_points.iter().enumerate() {
        let q = points[query_idx as usize];
        let security = security_planes(q, edges);

        keys.clear();
        keys.reserve(k.min(candidate_indices.len()));

        let center_xs = &grid.cell_points_x[center_soa_start..center_soa_end];
        let center_ys = &grid.cell_points_y[center_soa_start..center_soa_end];
        let center_zs = &grid.cell_points_z[center_soa_start..center_soa_end];

        let mut min_center_dot = f32::INFINITY;
        let mut center_added = 0usize;

        for i in 0..center_len {
            if i == qi {
                continue;
            }
            let dot = center_xs[i] * q.x + center_ys[i] * q.y + center_zs[i] * q.z;
            if dot > security {
                keys.push(make_desc_key(dot, candidate_indices[i]));
                min_center_dot = min_center_dot.min(dot);
                center_added += 1;
            }
        }

        let threshold = if center_added > 0 {
            security.max(min_center_dot - 1e-6)
        } else {
            security
        };

        // Ring pass.
        let center_keys_len = keys.len();
        let mut cand_offset = center_len;
        for &(soa_start, soa_end) in &cell_ranges[1..] {
            let xs = &grid.cell_points_x[soa_start..soa_end];
            let ys = &grid.cell_points_y[soa_start..soa_end];
            let zs = &grid.cell_points_z[soa_start..soa_end];
            let range_len = soa_end - soa_start;

            for i in 0..range_len {
                let dot = xs[i] * q.x + ys[i] * q.y + zs[i] * q.z;
                if dot > threshold {
                    keys.push(make_desc_key(dot, candidate_indices[cand_offset + i]));
                }
            }
            cand_offset += range_len;
        }

        // Fallback to security threshold if we didn't get enough ring candidates.
        let ring_added = keys.len() - center_keys_len;
        let need = k.saturating_sub(center_added);
        if ring_added < need {
            if let Some(stats) = stats.as_deref_mut() {
                stats.fallback_queries += 1;
            }
            keys.truncate(center_keys_len);
            let mut cand_offset = center_len;
            for &(soa_start, soa_end) in &cell_ranges[1..] {
                let xs = &grid.cell_points_x[soa_start..soa_end];
                let ys = &grid.cell_points_y[soa_start..soa_end];
                let zs = &grid.cell_points_z[soa_start..soa_end];
                let range_len = soa_end - soa_start;

                for i in 0..range_len {
                    let dot = xs[i] * q.x + ys[i] * q.y + zs[i] * q.z;
                    if dot > security {
                        keys.push(make_desc_key(dot, candidate_indices[cand_offset + i]));
                    }
                }
                cand_offset += range_len;
            }
        }

        let m = keys.len();
        if let Some(stats) = stats.as_deref_mut() {
            stats.total_candidates += candidate_indices.len() as u64;
            stats.filtered_out += (candidate_indices.len().saturating_sub(m)) as u64;
            stats.num_queries += 1;
        }

        let k_actual = k.min(m);
        if k_actual < k {
            if let Some(stats) = stats.as_deref_mut() {
                stats.under_k_count += 1;
            }
        }
        if k_actual > 0 {
            if m > k_actual {
                keys.select_nth_unstable(k_actual - 1);
            }
            keys[..k_actual].sort_unstable();
            let out_start = query_idx as usize * k;
            for i in 0..k_actual {
                neighbors[out_start + i] = key_to_idx(keys[i]);
            }
        }
    }
}

#[inline(always)]
fn f32_to_ordered_u32(val: f32) -> u32 {
    let b = val.to_bits();
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b ^ 0x8000_0000
    }
}

#[inline(always)]
fn make_desc_key(dot: f32, idx: u32) -> u64 {
    // Bigger dot = smaller key, so ascending sort gives descending dot.
    let ord = f32_to_ordered_u32(dot);
    let desc = !ord;
    ((desc as u64) << 32) | (idx as u64)
}

#[inline(always)]
fn key_to_idx(key: u64) -> u32 {
    (key & 0xFFFF_FFFF) as u32
}

/// Precomputed edge data for planes-based security threshold.
struct EdgeData {
    n: Vec3, // normalized normal of great circle plane
}

fn precompute_edges(corners: &[Vec3; 4]) -> [EdgeData; 4] {
    std::array::from_fn(|i| {
        let a = corners[i];
        let b = corners[(i + 1) % 4];
        let n = a.cross(b).normalize_or_zero();
        EdgeData { n }
    })
}

fn precompute_edges_all_cells(res: usize) -> Vec<[EdgeData; 4]> {
    let num_cells = 6 * res * res;
    let mut all_edges: Vec<[EdgeData; 4]> = Vec::with_capacity(num_cells);
    for cell in 0..num_cells {
        let corners = neighborhood_corners_5x5(cell, res);
        all_edges.push(precompute_edges(&corners));
    }
    all_edges
}

/// Cheap exact bound: max dot to boundary via great circle projections.
fn security_planes(q: Vec3, edges: &[EdgeData; 4]) -> f32 {
    let mut max_dot = f32::NEG_INFINITY;
    for edge in edges {
        let dn = q.dot(edge.n).abs();
        let dot_to_plane = (1.0 - dn * dn).max(0.0).sqrt();
        max_dot = max_dot.max(dot_to_plane);
    }
    max_dot
}

/// Get 4 outer corners of a cell's 5x5 neighborhood (for security_3x3 threshold).
fn neighborhood_corners_5x5(cell: usize, res: usize) -> [Vec3; 4] {
    let (face, iu, iv) = cell_to_face_ij(cell, res);

    // 5x5 neighborhood bounds: center ± 2 cells (clamped to face).
    let i0 = iu.saturating_sub(2);
    let i1 = (iu + 3).min(res);
    let j0 = iv.saturating_sub(2);
    let j1 = (iv + 3).min(res);

    let (u0, u1) = (st_to_uv(i0 as f32 / res as f32), st_to_uv(i1 as f32 / res as f32));
    let (v0, v1) = (st_to_uv(j0 as f32 / res as f32), st_to_uv(j1 as f32 / res as f32));

    [
        face_uv_to_3d(face, u0, v0),
        face_uv_to_3d(face, u1, v0),
        face_uv_to_3d(face, u1, v1),
        face_uv_to_3d(face, u0, v1),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::fibonacci_sphere_points_with_rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn gen_fibonacci(n: usize, seed: u64) -> Vec<Vec3> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
        fibonacci_sphere_points_with_rng(n, spacing * 0.1, &mut rng)
    }

    fn res_for_target(n: usize, target: f64) -> usize {
        ((n as f64 / (6.0 * target)).sqrt() as usize).max(4)
    }

    #[test]
    fn test_packed_knn_basic() {
        let n = 10_000;
        let k = 24;
        let points = gen_fibonacci(n, 12345);
        let grid = CubeMapGrid::new(&points, res_for_target(n, 24.0));

        let result = packed_knn(&grid, &points, k);

        assert_eq!(result.neighbors.len(), n * k);

        for qi in [0, 100, 5000, 9999] {
            let neighbors = result.get_valid(qi);
            assert!(neighbors.len() <= k);
            assert!(!neighbors.contains(&(qi as u32)));
            for &idx in neighbors {
                assert!((idx as usize) < n);
            }
        }
    }
}
