//! Batched k-NN using SIMD, processing entire bins at once.
//!
//! For each bin, iterates grid SoA directly and computes distances to all
//! query points using SIMD.

use super::{cell_to_face_ij, face_uv_to_3d, st_to_uv, CubeMapGrid};
use glam::Vec3;
use std::simd::f32x8;
use std::simd::num::SimdFloat;

/// Result of batched k-NN: for each point, the indices of its k nearest neighbors.
/// Layout: [p0_n0, p0_n1, ..., p0_n(k-1), p1_n0, ...]
pub struct BatchedKnnResult {
    pub neighbors: Vec<u32>,
    pub k: usize,
}

impl BatchedKnnResult {
    #[inline]
    pub fn get(&self, point_idx: usize) -> &[u32] {
        let start = point_idx * self.k;
        &self.neighbors[start..start + self.k]
    }
}

/// Compute k-NN for all points using bin-local SIMD approach.
///
/// For each grid cell, iterates candidates from 3x3 neighborhood directly
/// from grid SoA storage and computes k nearest neighbors using SIMD dot products.
///
/// Optimizations:
/// - No coordinate gather - iterates grid SoA directly
/// - Flat contiguous buffer for dot products
/// - Candidate-major iteration (coords stay in L1 cache)
/// - Dot products directly (higher = closer for unit vectors)
/// - Reused pairs buffer (no allocation per query)
/// - O(1) self-marking via ordered gather
pub fn binlocal_knn(grid: &CubeMapGrid, points: &[Vec3], k: usize) -> BatchedKnnResult {
    binlocal_knn_inner(grid, points, k, false)
}

/// Same as binlocal_knn but prints timing breakdown.
pub fn binlocal_knn_timed(grid: &CubeMapGrid, points: &[Vec3], k: usize) -> BatchedKnnResult {
    binlocal_knn_inner(grid, points, k, true)
}

fn binlocal_knn_inner(
    grid: &CubeMapGrid,
    points: &[Vec3],
    k: usize,
    print_timing: bool,
) -> BatchedKnnResult {
    if print_timing {
        binlocal_knn_timed_impl(grid, points, k)
    } else {
        binlocal_knn_fast_impl(grid, points, k)
    }
}

/// Fast path without timing overhead.
fn binlocal_knn_fast_impl(grid: &CubeMapGrid, points: &[Vec3], k: usize) -> BatchedKnnResult {
    let n = points.len();
    let num_cells = 6 * grid.res * grid.res;

    let mut neighbors = vec![u32::MAX; n * k];
    let mut candidate_indices: Vec<u32> = Vec::with_capacity(512);
    let mut cell_ranges: Vec<(usize, usize)> = Vec::with_capacity(9);
    let mut cell_dots: Vec<f32> = Vec::new();
    let mut pairs: Vec<(f32, u32)> = Vec::with_capacity(512);

    for cell in 0..num_cells {
        let query_points = grid.cell_points(cell);
        let num_queries = query_points.len();
        if num_queries == 0 {
            continue;
        }

        // Gather indices and track SoA ranges
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

        // SIMD dot computation
        let padded_len = (num_candidates + 7) & !7;
        cell_dots.clear();
        cell_dots.resize(num_queries * padded_len, 0.0);

        let mut cand_offset = 0usize;
        for &(soa_start, soa_end) in &cell_ranges {
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

                for (qi, &query_idx) in query_points.iter().enumerate() {
                    let q = points[query_idx as usize];
                    let qx = f32x8::splat(q.x);
                    let qy = f32x8::splat(q.y);
                    let qz = f32x8::splat(q.z);
                    let dots = cx * qx + cy * qy + cz * qz;
                    let offset = qi * padded_len + cand_offset + i;
                    dots.copy_to_slice(&mut cell_dots[offset..offset + 8]);
                }
            }

            let tail_start = full_chunks * 8;
            for i in tail_start..range_len {
                let cx = xs[i];
                let cy = ys[i];
                let cz = zs[i];
                for (qi, &query_idx) in query_points.iter().enumerate() {
                    let q = points[query_idx as usize];
                    let dot = cx * q.x + cy * q.y + cz * q.z;
                    cell_dots[qi * padded_len + cand_offset + i] = dot;
                }
            }

            cand_offset += range_len;
        }

        // Mark self as -inf
        for qi in 0..num_queries {
            cell_dots[qi * padded_len + qi] = f32::NEG_INFINITY;
        }

        // Select and sort for each query
        for (qi, &query_idx) in query_points.iter().enumerate() {
            pairs.clear();
            let dot_start = qi * padded_len;
            pairs.extend(
                cell_dots[dot_start..dot_start + num_candidates]
                    .iter()
                    .zip(candidate_indices.iter())
                    .map(|(&d, &idx)| (d, idx)),
            );

            let k_actual = k.min(pairs.len());
            if k_actual > 0 {
                // Skip select if candidates <= 2k, just sort everything
                if pairs.len() <= 2 * k {
                    pairs.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                } else {
                    pairs.select_nth_unstable_by(k_actual - 1, |a, b| b.0.total_cmp(&a.0));
                    pairs[..k_actual].sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                }
            }

            let out_start = query_idx as usize * k;
            for (i, &(_, idx)) in pairs.iter().take(k_actual).enumerate() {
                neighbors[out_start + i] = idx;
            }
        }
    }

    BatchedKnnResult { neighbors, k }
}

/// Timed path with per-phase breakdown.
fn binlocal_knn_timed_impl(grid: &CubeMapGrid, points: &[Vec3], k: usize) -> BatchedKnnResult {
    use std::time::Instant;

    let n = points.len();
    let num_cells = 6 * grid.res * grid.res;

    let mut neighbors = vec![u32::MAX; n * k];
    let mut candidate_indices: Vec<u32> = Vec::with_capacity(512);
    let mut cell_ranges: Vec<(usize, usize)> = Vec::with_capacity(9);
    let mut cell_dots: Vec<f32> = Vec::new();
    let mut pairs: Vec<(f32, u32)> = Vec::with_capacity(512);

    let mut time_gather_ns = 0u64;
    let mut time_simd_ns = 0u64;
    let mut time_pairs_ns = 0u64;
    let mut time_select_ns = 0u64;
    let mut time_sort_ns = 0u64;
    let mut time_write_ns = 0u64;

    for cell in 0..num_cells {
        let query_points = grid.cell_points(cell);
        let num_queries = query_points.len();
        if num_queries == 0 {
            continue;
        }

        let t0 = Instant::now();
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
        time_gather_ns += t0.elapsed().as_nanos() as u64;

        let t0 = Instant::now();
        let padded_len = (num_candidates + 7) & !7;
        cell_dots.clear();
        cell_dots.resize(num_queries * padded_len, 0.0);

        let mut cand_offset = 0usize;
        for &(soa_start, soa_end) in &cell_ranges {
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

                for (qi, &query_idx) in query_points.iter().enumerate() {
                    let q = points[query_idx as usize];
                    let qx = f32x8::splat(q.x);
                    let qy = f32x8::splat(q.y);
                    let qz = f32x8::splat(q.z);
                    let dots = cx * qx + cy * qy + cz * qz;
                    let offset = qi * padded_len + cand_offset + i;
                    dots.copy_to_slice(&mut cell_dots[offset..offset + 8]);
                }
            }

            let tail_start = full_chunks * 8;
            for i in tail_start..range_len {
                let cx = xs[i];
                let cy = ys[i];
                let cz = zs[i];
                for (qi, &query_idx) in query_points.iter().enumerate() {
                    let q = points[query_idx as usize];
                    let dot = cx * q.x + cy * q.y + cz * q.z;
                    cell_dots[qi * padded_len + cand_offset + i] = dot;
                }
            }

            cand_offset += range_len;
        }

        for qi in 0..num_queries {
            cell_dots[qi * padded_len + qi] = f32::NEG_INFINITY;
        }
        time_simd_ns += t0.elapsed().as_nanos() as u64;

        for (qi, &query_idx) in query_points.iter().enumerate() {
            let t0 = Instant::now();
            pairs.clear();
            let dot_start = qi * padded_len;
            pairs.extend(
                cell_dots[dot_start..dot_start + num_candidates]
                    .iter()
                    .zip(candidate_indices.iter())
                    .map(|(&d, &idx)| (d, idx)),
            );
            time_pairs_ns += t0.elapsed().as_nanos() as u64;

            let k_actual = k.min(pairs.len());
            if k_actual > 0 {
                // Skip select if candidates <= 2k, just sort everything
                if pairs.len() <= 2 * k {
                    let t0 = Instant::now();
                    pairs.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                    time_sort_ns += t0.elapsed().as_nanos() as u64;
                } else {
                    let t0 = Instant::now();
                    pairs.select_nth_unstable_by(k_actual - 1, |a, b| b.0.total_cmp(&a.0));
                    time_select_ns += t0.elapsed().as_nanos() as u64;

                    let t0 = Instant::now();
                    pairs[..k_actual].sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                    time_sort_ns += t0.elapsed().as_nanos() as u64;
                }
            }

            let t0 = Instant::now();
            let out_start = query_idx as usize * k;
            for (i, &(_, idx)) in pairs.iter().take(k_actual).enumerate() {
                neighbors[out_start + i] = idx;
            }
            time_write_ns += t0.elapsed().as_nanos() as u64;
        }
    }

    let total_ns =
        time_gather_ns + time_simd_ns + time_pairs_ns + time_select_ns + time_sort_ns + time_write_ns;
    println!("\n=== Binlocal k-NN Timing Breakdown ===");
    println!("Total points: {n}");
    println!();
    println!(
        "Gather:      {:>8.2} ms ({:>5.1}%) - {:>5.0} ns/point",
        time_gather_ns as f64 / 1e6,
        time_gather_ns as f64 / total_ns as f64 * 100.0,
        time_gather_ns as f64 / n as f64
    );
    println!(
        "SIMD:        {:>8.2} ms ({:>5.1}%) - {:>5.0} ns/point",
        time_simd_ns as f64 / 1e6,
        time_simd_ns as f64 / total_ns as f64 * 100.0,
        time_simd_ns as f64 / n as f64
    );
    println!(
        "Pairs:       {:>8.2} ms ({:>5.1}%) - {:>5.0} ns/point",
        time_pairs_ns as f64 / 1e6,
        time_pairs_ns as f64 / total_ns as f64 * 100.0,
        time_pairs_ns as f64 / n as f64
    );
    println!(
        "Select:      {:>8.2} ms ({:>5.1}%) - {:>5.0} ns/point",
        time_select_ns as f64 / 1e6,
        time_select_ns as f64 / total_ns as f64 * 100.0,
        time_select_ns as f64 / n as f64
    );
    println!(
        "Sort:        {:>8.2} ms ({:>5.1}%) - {:>5.0} ns/point",
        time_sort_ns as f64 / 1e6,
        time_sort_ns as f64 / total_ns as f64 * 100.0,
        time_sort_ns as f64 / n as f64
    );
    println!(
        "Write:       {:>8.2} ms ({:>5.1}%) - {:>5.0} ns/point",
        time_write_ns as f64 / 1e6,
        time_write_ns as f64 / total_ns as f64 * 100.0,
        time_write_ns as f64 / n as f64
    );
    println!(
        "Total:       {:>8.2} ms         - {:>5.0} ns/point",
        total_ns as f64 / 1e6,
        total_ns as f64 / n as f64
    );

    BatchedKnnResult { neighbors, k }
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
    fn test_binlocal_knn_basic() {
        let n = 10_000;
        let k = 24;
        let points = gen_fibonacci(n, 12345);
        let grid = CubeMapGrid::new(&points, res_for_target(n, 24.0));

        let result = binlocal_knn(&grid, &points, k);

        // Check that we got results for all points
        assert_eq!(result.neighbors.len(), n * k);

        // Check a few points have valid neighbors
        for qi in [0, 100, 5000, 9999] {
            let neighbors = result.get(qi);
            assert_eq!(neighbors.len(), k);
            // No self-references
            assert!(!neighbors.contains(&(qi as u32)));
            // All valid indices
            for &idx in neighbors {
                assert!((idx as usize) < n);
            }
        }
    }
}

// ============================================================================
// Filtered binlocal variants (Idea 1: security_3x3 filtering)
// ============================================================================

/// Get 4 outer corners of a cell's 5x5 neighborhood (for security_3x3 threshold).
/// The 5x5 boundary is what we compare against to certify 3x3 candidates.
pub fn neighborhood_corners_5x5(cell: usize, res: usize) -> [Vec3; 4] {
    let (face, iu, iv) = cell_to_face_ij(cell, res);

    // 5x5 neighborhood bounds: center ± 2 cells (clamped to face)
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

/// Compute security_3x3 threshold using arc method (exact, per-query).
pub fn security_arc(q: Vec3, corners: &[Vec3; 4]) -> f32 {
    (0..4)
        .map(|i| {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            q.dot(closest_point_on_arc(q, a, b))
        })
        .fold(f32::NEG_INFINITY, f32::max)
}

fn closest_point_on_arc(q: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let n = a.cross(b);
    let n_len_sq = n.length_squared();
    if n_len_sq < 1e-10 {
        return if q.dot(a) > q.dot(b) { a } else { b };
    }
    let n = n / n_len_sq.sqrt();
    let q_proj = (q - n * q.dot(n)).normalize_or_zero();
    if q_proj == Vec3::ZERO {
        return a;
    }

    let on_arc = a.cross(q_proj).dot(n) >= -1e-6 && q_proj.cross(b).dot(n) >= -1e-6;
    if on_arc { q_proj } else if q.dot(a) > q.dot(b) { a } else { b }
}

/// Filtering mode for binlocal_knn_filtered.
#[derive(Clone, Copy, Debug)]
pub enum FilterMode {
    /// No filtering (baseline). Returns all 3x3 candidates including uncertified.
    None,
    /// Planes-based security_3x3 filtering. Returns only certified neighbors.
    Planes,
    /// Combined: max(security_3x3, worst_1x1_dot). Fast + certified.
    Combined,
    /// Packed u64 keys: encode (dot, idx) as u64 for fast integer sort.
    Packed,
    /// Packed v2: always select+sort, no cutoff.
    PackedV2,
    /// Packed v3: SIMD filter + compact (branch once per 8 candidates).
    PackedV3,
    /// Packed v4: Fused dot compute + key building (no cell_dots buffer).
    PackedV4,
}

// === Packed u64 key helpers ===

#[inline(always)]
pub fn f32_to_ordered_u32(x: f32) -> u32 {
    let b = x.to_bits();
    // Map IEEE754 bits to monotonically increasing order
    if (b & 0x8000_0000) != 0 {
        !b
    } else {
        b ^ 0x8000_0000
    }
}

#[inline(always)]
pub fn make_desc_key(dot: f32, idx: u32) -> u64 {
    // Bigger dot = smaller key, so ascending sort gives descending dot
    let ord = f32_to_ordered_u32(dot);
    let desc = !ord;
    ((desc as u64) << 32) | (idx as u64)
}

#[inline(always)]
pub fn key_to_idx(key: u64) -> u32 {
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

/// Binlocal k-NN with security_3x3 filtering.
/// Returns (result, stats) where stats contains filtering metrics.
pub fn binlocal_knn_filtered(
    grid: &CubeMapGrid,
    points: &[Vec3],
    k: usize,
    mode: FilterMode,
) -> (BatchedKnnResult, FilterStats) {
    use std::time::Instant;

    let n = points.len();
    let num_cells = 6 * grid.res * grid.res;

    let mut neighbors = vec![u32::MAX; n * k];
    let mut candidate_indices: Vec<u32> = Vec::with_capacity(512);
    let mut cell_ranges: Vec<(usize, usize)> = Vec::with_capacity(9);
    let mut cell_dots: Vec<f32> = Vec::new();
    let mut pairs: Vec<(f32, u32)> = Vec::with_capacity(512);

    // Stats
    let mut total_candidates = 0u64;
    let mut filtered_candidates = 0u64;
    let mut time_filter_ns = 0u64;
    let mut fallback_count = 0u64;
    let mut under_k_count = 0u64;
    let mut total_certified = 0u64;
    let mut query_count = 0u64;
    // Phase timing
    let mut gather_ns = 0u64;
    let mut simd_ns = 0u64;
    let mut pairs_ns = 0u64;
    let mut select_ns = 0u64;
    let mut sort_ns = 0u64;
    let mut write_ns = 0u64;

    for cell in 0..num_cells {
        let query_points = grid.cell_points(cell);
        let num_queries = query_points.len();
        if num_queries == 0 {
            continue;
        }

        // Precompute data for filtering (planes-based security_3x3)
        let edges = match mode {
            FilterMode::Planes | FilterMode::Combined | FilterMode::Packed | FilterMode::PackedV2 | FilterMode::PackedV3 | FilterMode::PackedV4 => {
                let corners = neighborhood_corners_5x5(cell, grid.res);
                Some(precompute_edges(&corners))
            }
            FilterMode::None => None,
        };

        // Gather indices and track SoA ranges
        let t_gather = Instant::now();
        candidate_indices.clear();
        cell_ranges.clear();

        let q_start = grid.cell_offsets[cell] as usize;
        let q_end = grid.cell_offsets[cell + 1] as usize;
        let center_count = q_end - q_start;
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
        gather_ns += t_gather.elapsed().as_nanos() as u64;

        // PackedV4: Fused dot compute + key building (no cell_dots buffer)
        if matches!(mode, FilterMode::PackedV4) {
            use std::mem::MaybeUninit;
            use std::simd::{cmp::SimdPartialOrd, Mask};

            let t_simd = Instant::now();

            // Per-query state
            let stride = num_candidates.min(512);
            let slab_size = num_queries * stride;
            let mut keys_slab: Vec<MaybeUninit<u64>> = Vec::with_capacity(slab_size);
            unsafe { keys_slab.set_len(slab_size); }
            let mut lens = vec![0usize; num_queries];
            let mut min_center_dot = vec![f32::INFINITY; num_queries];

            // Precompute security_3x3 per query (needed for center pass)
            let mut security_thresholds = Vec::with_capacity(num_queries);
            for &query_idx in query_points.iter() {
                let q = points[query_idx as usize];
                security_thresholds.push(security_planes(q, edges.as_ref().unwrap()));
            }

            // Pass A: Center range only - compute dots, filter, track min
            let (center_soa_start, center_soa_end) = cell_ranges[0];
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

                for (qi, &query_idx) in query_points.iter().enumerate() {
                    let q = points[query_idx as usize];
                    let qx = f32x8::splat(q.x);
                    let qy = f32x8::splat(q.y);
                    let qz = f32x8::splat(q.z);
                    let dots = cx * qx + cy * qy + cz * qz;

                    let thresh_vec = f32x8::splat(security_thresholds[qi]);
                    let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
                    let mut mask_bits = mask.to_bitmask() as u32;

                    // Clear self bit if in this chunk
                    let base = i;
                    if base <= qi && qi < base + 8 {
                        mask_bits &= !(1u32 << (qi - base));
                    }

                    // Extract passing candidates and track min
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

            // Handle center tail
            let tail_start = full_chunks * 8;
            for i in tail_start..center_len {
                let cx = xs[i];
                let cy = ys[i];
                let cz = zs[i];
                for (qi, &query_idx) in query_points.iter().enumerate() {
                    if i == qi { continue; } // skip self
                    let q = points[query_idx as usize];
                    let dot = cx * q.x + cy * q.y + cz * q.z;
                    if dot > security_thresholds[qi] {
                        let slab_idx = qi * stride + lens[qi];
                        keys_slab[slab_idx].write(make_desc_key(dot, candidate_indices[i]));
                        lens[qi] += 1;
                        min_center_dot[qi] = min_center_dot[qi].min(dot);
                    }
                }
            }

            // Track center counts for fallback detection
            let center_lens: Vec<usize> = lens.clone();

            // Compute per-query thresholds for ring pass
            // Only use min_center_dot if we found center candidates; else use security only
            let mut thresholds = Vec::with_capacity(num_queries);
            for qi in 0..num_queries {
                let threshold = if center_lens[qi] > 0 {
                    security_thresholds[qi].max(min_center_dot[qi] - 1e-6)
                } else {
                    security_thresholds[qi]
                };
                thresholds.push(threshold);
            }
            let ring_start_lens: Vec<usize> = lens.clone();

            // Pass B: Ring ranges - compute dots, filter by threshold
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

                    for (qi, &query_idx) in query_points.iter().enumerate() {
                        let q = points[query_idx as usize];
                        let qx = f32x8::splat(q.x);
                        let qy = f32x8::splat(q.y);
                        let qz = f32x8::splat(q.z);
                        let dots = cx * qx + cy * qy + cz * qz;

                        let thresh_vec = f32x8::splat(thresholds[qi]);
                        let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
                        let mask_bits = mask.to_bitmask() as u32;

                        if mask_bits != 0 {
                            let dots_arr: [f32; 8] = dots.into();
                            let mut bits = mask_bits;
                            while bits != 0 {
                                let lane = bits.trailing_zeros() as usize;
                                let cand_idx = cand_offset + i + lane;
                                let dot = dots_arr[lane];
                                let slab_idx = qi * stride + lens[qi];
                                if slab_idx < slab_size {
                                    keys_slab[slab_idx].write(make_desc_key(dot, candidate_indices[cand_idx]));
                                    lens[qi] += 1;
                                }
                                bits &= bits - 1;
                            }
                        }
                    }
                }

                // Handle ring tail
                let tail_start = full_chunks * 8;
                for i in tail_start..range_len {
                    let cx = xs[i];
                    let cy = ys[i];
                    let cz = zs[i];
                    for (qi, &query_idx) in query_points.iter().enumerate() {
                        let q = points[query_idx as usize];
                        let dot = cx * q.x + cy * q.y + cz * q.z;
                        if dot > thresholds[qi] {
                            let slab_idx = qi * stride + lens[qi];
                            if slab_idx < slab_size {
                                keys_slab[slab_idx].write(make_desc_key(dot, candidate_indices[cand_offset + i]));
                                lens[qi] += 1;
                            }
                        }
                    }
                }

                cand_offset += range_len;
            }

            simd_ns += t_simd.elapsed().as_nanos() as u64;

            // Check for fallback and re-run ring if needed
            let t_pairs = Instant::now();
            for qi in 0..num_queries {
                let ring_added = lens[qi] - ring_start_lens[qi];
                let need = k.saturating_sub(center_lens[qi]);
                if ring_added < need {
                    fallback_count += 1;
                    // Reset ring portion and re-filter with security_3x3 only
                    lens[qi] = ring_start_lens[qi];
                    let mut cand_offset = center_len;
                    for &(soa_start, soa_end) in &cell_ranges[1..] {
                        let range_len = soa_end - soa_start;
                        let xs = &grid.cell_points_x[soa_start..soa_end];
                        let ys = &grid.cell_points_y[soa_start..soa_end];
                        let zs = &grid.cell_points_z[soa_start..soa_end];
                        let query_idx = query_points[qi];
                        let q = points[query_idx as usize];
                        for i in 0..range_len {
                            let dot = xs[i] * q.x + ys[i] * q.y + zs[i] * q.z;
                            if dot > security_thresholds[qi] {
                                let slab_idx = qi * stride + lens[qi];
                                if slab_idx < slab_size {
                                    keys_slab[slab_idx].write(make_desc_key(dot, candidate_indices[cand_offset + i]));
                                    lens[qi] += 1;
                                }
                            }
                        }
                        cand_offset += range_len;
                    }
                }
            }
            pairs_ns += t_pairs.elapsed().as_nanos() as u64;

            // Select+sort per query
            for (qi, &query_idx) in query_points.iter().enumerate() {
                let m = lens[qi];
                total_candidates += num_candidates as u64;
                filtered_candidates += (num_candidates - m) as u64;
                total_certified += m as u64;
                query_count += 1;

                let keys_ptr = keys_slab[qi * stride..].as_mut_ptr() as *mut u64;
                let keys_slice = unsafe { std::slice::from_raw_parts_mut(keys_ptr, m) };

                let k_actual = k.min(m);
                if k_actual < k {
                    under_k_count += 1;
                }
                if k_actual > 0 {
                    let t_select = Instant::now();
                    keys_slice.select_nth_unstable(k_actual - 1);
                    select_ns += t_select.elapsed().as_nanos() as u64;

                    let t_sort = Instant::now();
                    keys_slice[..k_actual].sort_unstable();
                    sort_ns += t_sort.elapsed().as_nanos() as u64;
                }

                let t_write = Instant::now();
                let out_start = query_idx as usize * k;
                for i in 0..k_actual {
                    neighbors[out_start + i] = key_to_idx(keys_slice[i]);
                }
                write_ns += t_write.elapsed().as_nanos() as u64;
            }

            continue; // Skip the normal cell_dots path
        }

        // SIMD dot computation
        let t_simd = Instant::now();
        let padded_len = (num_candidates + 7) & !7;
        cell_dots.clear();
        cell_dots.resize(num_queries * padded_len, 0.0);

        let mut cand_offset = 0usize;
        for &(soa_start, soa_end) in &cell_ranges {
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

                for (qi, &query_idx) in query_points.iter().enumerate() {
                    let q = points[query_idx as usize];
                    let qx = f32x8::splat(q.x);
                    let qy = f32x8::splat(q.y);
                    let qz = f32x8::splat(q.z);
                    let dots = cx * qx + cy * qy + cz * qz;
                    let offset = qi * padded_len + cand_offset + i;
                    dots.copy_to_slice(&mut cell_dots[offset..offset + 8]);
                }
            }

            let tail_start = full_chunks * 8;
            for i in tail_start..range_len {
                let cx = xs[i];
                let cy = ys[i];
                let cz = zs[i];
                for (qi, &query_idx) in query_points.iter().enumerate() {
                    let q = points[query_idx as usize];
                    let dot = cx * q.x + cy * q.y + cz * q.z;
                    cell_dots[qi * padded_len + cand_offset + i] = dot;
                }
            }

            cand_offset += range_len;
        }

        // Mark self as -inf
        for qi in 0..num_queries {
            cell_dots[qi * padded_len + qi] = f32::NEG_INFINITY;
        }
        simd_ns += t_simd.elapsed().as_nanos() as u64;

        // Select and sort for each query (with filtering)
        for (qi, &query_idx) in query_points.iter().enumerate() {
            let q = points[query_idx as usize];
            let dot_start = qi * padded_len;

            // Compute security_3x3 threshold (planes method)
            let t0 = Instant::now();
            let security_3x3 = match mode {
                FilterMode::Planes | FilterMode::Combined | FilterMode::Packed | FilterMode::PackedV2 | FilterMode::PackedV3 | FilterMode::PackedV4 => {
                    security_planes(q, edges.as_ref().unwrap())
                }
                FilterMode::None => f32::NEG_INFINITY,
            };
            time_filter_ns += t0.elapsed().as_nanos() as u64;

            // Packed modes: use u64 keys with uninitialized stack buffer
            if matches!(mode, FilterMode::Packed | FilterMode::PackedV2) {
                use std::mem::MaybeUninit;

                let t_pairs = Instant::now();
                let mut keys: [MaybeUninit<u64>; 512] = unsafe { MaybeUninit::uninit().assume_init() };
                let mut m = 0usize;

                // Combined filtering logic with packed keys
                let mut min_center_dot = f32::INFINITY;
                for i in 0..center_count {
                    if i != qi {
                        let d = cell_dots[dot_start + i];
                        if d > security_3x3 {
                            keys[m].write(make_desc_key(d, candidate_indices[i]));
                            m += 1;
                            min_center_dot = min_center_dot.min(d);
                        }
                    }
                }
                let center_added = m;

                let worst_1x1 = min_center_dot - 1e-6;
                let threshold = security_3x3.max(worst_1x1);
                let need = k.saturating_sub(center_added);
                let ring_start = m;

                for i in center_count..num_candidates {
                    let d = cell_dots[dot_start + i];
                    if d > threshold {
                        keys[m].write(make_desc_key(d, candidate_indices[i]));
                        m += 1;
                    }
                }

                // Fallback if needed
                if m - ring_start < need {
                    fallback_count += 1;
                    m = ring_start;
                    for i in center_count..num_candidates {
                        let d = cell_dots[dot_start + i];
                        if d > security_3x3 {
                            keys[m].write(make_desc_key(d, candidate_indices[i]));
                            m += 1;
                        }
                    }
                }

                total_candidates += num_candidates as u64;
                filtered_candidates += (num_candidates - m) as u64;
                total_certified += m as u64;
                query_count += 1;
                pairs_ns += t_pairs.elapsed().as_nanos() as u64;

                // Get initialized slice
                let keys_init = unsafe {
                    std::slice::from_raw_parts_mut(keys.as_mut_ptr() as *mut u64, m)
                };

                let k_actual = k.min(m);
                if k_actual < k {
                    under_k_count += 1;
                }
                if k_actual > 0 {
                    // PackedV2: always select+sort (no cutoff)
                    // Packed: use cutoff of 4*k
                    let use_select = matches!(mode, FilterMode::PackedV2) || m > 4 * k;
                    if use_select {
                        let t_select = Instant::now();
                        keys_init.select_nth_unstable(k_actual - 1);
                        select_ns += t_select.elapsed().as_nanos() as u64;

                        let t_sort = Instant::now();
                        keys_init[..k_actual].sort_unstable();
                        sort_ns += t_sort.elapsed().as_nanos() as u64;
                    } else {
                        let t_sort = Instant::now();
                        keys_init.sort_unstable();
                        sort_ns += t_sort.elapsed().as_nanos() as u64;
                    }
                }

                let t_write = Instant::now();
                let out_start = query_idx as usize * k;
                for i in 0..k_actual {
                    neighbors[out_start + i] = key_to_idx(keys_init[i]);
                }
                write_ns += t_write.elapsed().as_nanos() as u64;
                continue;
            }

            // PackedV3: SIMD filter + compact
            if matches!(mode, FilterMode::PackedV3) {
                use std::mem::MaybeUninit;
                use std::simd::{cmp::SimdPartialOrd, Mask};

                let t_pairs = Instant::now();
                let mut keys: [MaybeUninit<u64>; 512] = unsafe { MaybeUninit::uninit().assume_init() };
                let mut m = 0usize;

                // First pass: center cells (need to track min for threshold)
                let mut min_center_dot = f32::INFINITY;
                for i in 0..center_count {
                    if i != qi {
                        let d = cell_dots[dot_start + i];
                        if d > security_3x3 {
                            keys[m].write(make_desc_key(d, candidate_indices[i]));
                            m += 1;
                            min_center_dot = min_center_dot.min(d);
                        }
                    }
                }
                let center_added = m;

                let worst_1x1 = min_center_dot - 1e-6;
                let threshold = security_3x3.max(worst_1x1);
                let need = k.saturating_sub(center_added);
                let ring_start = m;

                // SIMD filter for ring candidates
                let ring_dots = &cell_dots[dot_start + center_count..dot_start + num_candidates];
                let ring_indices = &candidate_indices[center_count..num_candidates];
                let thresh_vec = f32x8::splat(threshold);

                let chunks = ring_dots.len() / 8;
                for chunk in 0..chunks {
                    let base = chunk * 8;
                    let dots = f32x8::from_slice(&ring_dots[base..]);
                    let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
                    let mask_bits = mask.to_bitmask() as u32;

                    if mask_bits != 0 {
                        let mut bits = mask_bits;
                        while bits != 0 {
                            let lane = bits.trailing_zeros() as usize;
                            let idx = base + lane;
                            keys[m].write(make_desc_key(ring_dots[idx], ring_indices[idx]));
                            m += 1;
                            bits &= bits - 1; // clear lowest set bit
                        }
                    }
                }

                // Handle tail (remaining candidates after last full chunk)
                let tail_start = chunks * 8;
                for i in tail_start..ring_dots.len() {
                    let d = ring_dots[i];
                    if d > threshold {
                        keys[m].write(make_desc_key(d, ring_indices[i]));
                        m += 1;
                    }
                }

                // Fallback if needed
                if m - ring_start < need {
                    fallback_count += 1;
                    m = ring_start;
                    let sec_vec = f32x8::splat(security_3x3);

                    for chunk in 0..chunks {
                        let base = chunk * 8;
                        let dots = f32x8::from_slice(&ring_dots[base..]);
                        let mask: Mask<i32, 8> = dots.simd_gt(sec_vec);
                        let mask_bits = mask.to_bitmask() as u32;

                        if mask_bits != 0 {
                            let mut bits = mask_bits;
                            while bits != 0 {
                                let lane = bits.trailing_zeros() as usize;
                                let idx = base + lane;
                                keys[m].write(make_desc_key(ring_dots[idx], ring_indices[idx]));
                                m += 1;
                                bits &= bits - 1;
                            }
                        }
                    }

                    for i in tail_start..ring_dots.len() {
                        let d = ring_dots[i];
                        if d > security_3x3 {
                            keys[m].write(make_desc_key(d, ring_indices[i]));
                            m += 1;
                        }
                    }
                }

                total_candidates += num_candidates as u64;
                filtered_candidates += (num_candidates - m) as u64;
                total_certified += m as u64;
                query_count += 1;
                pairs_ns += t_pairs.elapsed().as_nanos() as u64;

                // Get initialized slice
                let keys_init = unsafe {
                    std::slice::from_raw_parts_mut(keys.as_mut_ptr() as *mut u64, m)
                };

                let k_actual = k.min(m);
                if k_actual < k {
                    under_k_count += 1;
                }
                if k_actual > 0 {
                    let t_select = Instant::now();
                    keys_init.select_nth_unstable(k_actual - 1);
                    select_ns += t_select.elapsed().as_nanos() as u64;

                    let t_sort = Instant::now();
                    keys_init[..k_actual].sort_unstable();
                    sort_ns += t_sort.elapsed().as_nanos() as u64;
                }

                let t_write = Instant::now();
                let out_start = query_idx as usize * k;
                for i in 0..k_actual {
                    neighbors[out_start + i] = key_to_idx(keys_init[i]);
                }
                write_ns += t_write.elapsed().as_nanos() as u64;
                continue;
            }

            // Build pairs with filtering
            let t_pairs = Instant::now();
            pairs.clear();

            let before_filter = num_candidates;
            if matches!(mode, FilterMode::Combined) {
                // Combined: center loop (track min), then ring (filter by max threshold)
                let mut min_center_dot = f32::INFINITY;
                let mut center_added = 0;
                for i in 0..center_count {
                    if i != qi {
                        let d = cell_dots[dot_start + i];
                        if d > security_3x3 {
                            pairs.push((d, candidate_indices[i]));
                            min_center_dot = min_center_dot.min(d);
                            center_added += 1;
                        }
                    }
                }

                // threshold = max(security_3x3, worst_1x1 - eps)
                let worst_1x1 = min_center_dot - 1e-6;
                let threshold = security_3x3.max(worst_1x1);

                let need = k.saturating_sub(center_added);
                let ring_start = pairs.len();

                for i in center_count..num_candidates {
                    let d = cell_dots[dot_start + i];
                    if d > threshold {
                        pairs.push((d, candidate_indices[i]));
                    }
                }

                // Fallback: relax to security_3x3 only if not enough passed
                if pairs.len() - ring_start < need {
                    fallback_count += 1;
                    pairs.truncate(ring_start);
                    for i in center_count..num_candidates {
                        let d = cell_dots[dot_start + i];
                        if d > security_3x3 {
                            pairs.push((d, candidate_indices[i]));
                        }
                    }
                }
            } else {
                // None/Planes: filter all candidates by security_3x3
                for i in 0..num_candidates {
                    let d = cell_dots[dot_start + i];
                    if d > security_3x3 {
                        pairs.push((d, candidate_indices[i]));
                    }
                }
            }
            let after_filter = pairs.len();

            total_candidates += before_filter as u64;
            filtered_candidates += (before_filter - after_filter) as u64;
            total_certified += after_filter as u64;
            query_count += 1;
            pairs_ns += t_pairs.elapsed().as_nanos() as u64;

            let k_actual = k.min(pairs.len());
            if k_actual < k {
                under_k_count += 1;
            }
            if k_actual > 0 {
                if pairs.len() <= 2 * k {
                    let t_sort = Instant::now();
                    pairs.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                    sort_ns += t_sort.elapsed().as_nanos() as u64;
                } else {
                    let t_select = Instant::now();
                    pairs.select_nth_unstable_by(k_actual - 1, |a, b| b.0.total_cmp(&a.0));
                    select_ns += t_select.elapsed().as_nanos() as u64;

                    let t_sort = Instant::now();
                    pairs[..k_actual].sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                    sort_ns += t_sort.elapsed().as_nanos() as u64;
                }
            }

            let t_write = Instant::now();
            let out_start = query_idx as usize * k;
            for (i, &(_, idx)) in pairs.iter().take(k_actual).enumerate() {
                neighbors[out_start + i] = idx;
            }
            write_ns += t_write.elapsed().as_nanos() as u64;
        }
    }

    let stats = FilterStats {
        total_ring_candidates: total_candidates,
        filtered_out: filtered_candidates,
        filter_time_ns: time_filter_ns,
        fallback_count,
        under_k_count,
        total_certified,
        num_queries: query_count,
        gather_ns,
        simd_ns,
        pairs_ns,
        select_ns,
        sort_ns,
        write_ns,
    };

    (BatchedKnnResult { neighbors, k }, stats)
}

/// Statistics from filtered binlocal k-NN.
#[derive(Clone, Debug)]
pub struct FilterStats {
    pub total_ring_candidates: u64,
    pub filtered_out: u64,
    pub filter_time_ns: u64,
    pub fallback_count: u64, // queries that triggered fallback
    pub under_k_count: u64,  // queries that got fewer than k neighbors
    pub total_certified: u64, // total certified candidates across all queries
    pub num_queries: u64,
    // Phase timing (ns)
    pub gather_ns: u64,
    pub simd_ns: u64,
    pub pairs_ns: u64,
    pub select_ns: u64,
    pub sort_ns: u64,
    pub write_ns: u64,
}

impl FilterStats {
    pub fn filter_rate(&self) -> f64 {
        if self.total_ring_candidates == 0 {
            0.0
        } else {
            self.filtered_out as f64 / self.total_ring_candidates as f64
        }
    }
}
