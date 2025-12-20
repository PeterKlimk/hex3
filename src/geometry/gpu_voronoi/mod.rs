//! GPU-friendly spherical Voronoi computation via half-space (great circle) clipping.
//!
//! This module implements a "meshless" approach where each Voronoi cell is computed
//! independently from its k nearest neighbors. This enables massive parallelism on GPU.

mod cell_builder;
pub mod dedup;
mod knn;

#[cfg(test)]
mod tests;

use glam::Vec3;
use kiddo::ImmutableKdTree;

// Re-exports
pub use cell_builder::{
    GreatCircle, IncrementalCellBuilder, DEFAULT_K, MAX_PLANES, MAX_VERTICES,
    VertexData, VertexList,
    geodesic_distance, order_vertices_ccw_indices,
};
pub use dedup::dedup_vertices_hash_flat;
pub use knn::{CubeMapGridKnn, KnnProvider};

/// Minimum distance between generator and neighbor for a valid bisector.
/// Points closer than this are merged before Voronoi computation to prevent:
/// 1. Numerically unstable bisector planes
/// 2. Multiple close neighbors collectively killing cells
/// 3. Orphan edges from inconsistent cell topologies
pub const MIN_BISECTOR_DISTANCE: f32 = 1e-3;
/// Fraction of mean generator spacing used for vertex welding.
pub const VERTEX_WELD_FRACTION: f32 = 0.03;

/// Approximate mean chord length between uniformly-distributed generators.
/// Used to scale tolerances; assumes roughly even generator spacing.
pub(crate) fn mean_generator_spacing_chord(num_points: usize) -> f32 {
    if num_points == 0 {
        return 0.0;
    }
    let mean_angle = (4.0 * std::f32::consts::PI / num_points as f32).sqrt();
    2.0 * (0.5 * mean_angle).sin()
}

#[derive(Debug)]
pub(crate) struct FlatChunk {
    pub(crate) vertices: Vec<VertexData>,
    pub(crate) counts: Vec<u8>,
    /// Total neighbors processed in this chunk (for stats).
    pub(crate) total_neighbors_processed: usize,
    /// Number of cells that terminated early in this chunk.
    pub(crate) terminated_cells: usize,
    /// Number of cells that used fallback but still did not terminate.
    pub(crate) fallback_unterminated: usize,
}

#[derive(Debug)]
pub struct FlatCellsData {
    pub(crate) chunks: Vec<FlatChunk>,
    num_cells: usize,
}

impl FlatCellsData {
    #[inline]
    pub fn num_cells(&self) -> usize {
        self.num_cells
    }

    #[inline]
    pub fn total_vertices(&self) -> usize {
        self.chunks.iter().map(|c| c.vertices.len()).sum()
    }

    /// Compute VoronoiStats from chunk data.
    pub fn stats(&self) -> VoronoiStats {
        let total_neighbors: usize = self.chunks.iter().map(|c| c.total_neighbors_processed).sum();
        let terminated: usize = self.chunks.iter().map(|c| c.terminated_cells).sum();
        let fallback_unterminated: usize = self.chunks.iter().map(|c| c.fallback_unterminated).sum();
        let n = self.num_cells.max(1) as f64;
        VoronoiStats {
            avg_neighbors_processed: total_neighbors as f64 / n,
            termination_rate: terminated as f64 / n,
            fallback_unterminated,
        }
    }

    /// Iterate over cells, yielding vertex slices for each cell.
    /// Useful for tests that need to inspect per-cell data.
    pub fn iter_cells(&self) -> impl Iterator<Item = &[VertexData]> {
        self.chunks.iter().flat_map(|chunk| {
            let mut offset = 0usize;
            chunk.counts.iter().map(move |&count| {
                let count = count as usize;
                let start = offset;
                offset += count;
                &chunk.vertices[start..start + count]
            })
        })
    }
}

/// Build a k-d tree from sphere points for efficient k-NN queries.
pub fn build_kdtree(points: &[Vec3]) -> (ImmutableKdTree<f32, 3>, Vec<[f32; 3]>) {
    let entries: Vec<[f32; 3]> = points.iter().map(|p| [p.x, p.y, p.z]).collect();
    let tree = ImmutableKdTree::new_from_slice(&entries);
    (tree, entries)
}

/// Find k nearest neighbors using k-d tree.
pub fn find_k_nearest(
    tree: &ImmutableKdTree<f32, 3>,
    _entries: &[[f32; 3]],
    query: Vec3,
    query_idx: usize,
    k: usize,
) -> Vec<usize> {
    use kiddo::SquaredEuclidean;
    let results = tree.nearest_n::<SquaredEuclidean>(&[query.x, query.y, query.z], k + 1);
    results
        .into_iter()
        .map(|n| n.item as usize)
        .filter(|&idx| idx != query_idx)
        .take(k)
        .collect()
}

/// Statistics about the Voronoi computation.
#[derive(Debug, Clone)]
pub struct VoronoiStats {
    /// Average number of neighbors processed per cell before termination.
    pub avg_neighbors_processed: f64,
    /// Fraction of cells that terminated early (0.0 to 1.0).
    pub termination_rate: f64,
    /// Number of cells that used fallback but still did not terminate.
    pub fallback_unterminated: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct TerminationConfig {
    pub enabled: bool,
    pub check_start: usize,
    pub check_step: usize,
}

impl TerminationConfig {
    #[inline]
    pub fn should_check(&self, neighbors_processed: usize) -> bool {
        self.enabled
            && self.check_step > 0
            && neighbors_processed >= self.check_start
            && (neighbors_processed - self.check_start) % self.check_step == 0
    }
}

#[derive(Debug, Clone, Copy)]
struct PhaseTimingsMs {
    total: f64,
    kdtree: f64,
    knn: f64,
    cell_construction: f64,
    dedup: f64,
    assemble: f64,
}

/// Build cells into flat buffers with chunked parallelism.
/// Returns (vertices, counts).
/// - vertices: flat buffer of all vertex data
/// - counts: vertex count per cell (counts[i] = number of vertices for cell i)
/// Configuration for adaptive k-NN fetching.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveKConfig {
    /// Initial number of neighbors to fetch.
    pub initial_k: usize,
    /// Step size when resuming to fetch more neighbors.
    pub step_k: usize,
    /// Track limit for resumable queries (bounds how far we can resume).
    /// If we need more than this, falls back to fresh query with fallback_k.
    pub track_limit: usize,
    /// Fallback k for fresh query when track_limit isn't enough.
    /// Set to 0 to disable fallback (cells may be incomplete).
    pub fallback_k: usize,
}

impl Default for AdaptiveKConfig {
    fn default() -> Self {
        Self {
            initial_k: 12,
            step_k: 12,
            track_limit: 24,
            fallback_k: 48,
        }
    }
}

impl AdaptiveKConfig {
    /// Non-adaptive config: always fetch exactly k neighbors.
    pub fn fixed(k: usize) -> Self {
        Self {
            initial_k: k,
            step_k: 0,
            track_limit: k,
            fallback_k: 0,
        }
    }

    /// Adaptive config with track_limit and optional fallback.
    pub fn adaptive(initial_k: usize, step_k: usize, track_limit: usize, fallback_k: usize) -> Self {
        Self { initial_k, step_k, track_limit, fallback_k }
    }

    /// Fast preset for Lloyd-relaxed points (well-behaved, uniform spacing).
    /// Most cells terminate early; track_limit=24 is sufficient with k=48 fallback.
    pub fn fast() -> Self {
        Self {
            initial_k: 12,
            step_k: 12,
            track_limit: 24,
            fallback_k: 48,
        }
    }

    /// Robust preset for jittery or irregular point distributions.
    /// Higher track_limit handles most cells without fallback.
    pub fn robust() -> Self {
        Self {
            initial_k: 12,
            step_k: 10,
            track_limit: 32,
            fallback_k: 48,
        }
    }
}

pub fn build_cells_data_flat(
    points: &[Vec3],
    knn: &impl KnnProvider,
    k: usize,
    termination: TerminationConfig,
) -> FlatCellsData {
    // Use adaptive-k by default: fast path with track_limit=k, fallback to 48 if needed
    let adaptive = AdaptiveKConfig {
        initial_k: 12.min(k),
        step_k: 12,
        track_limit: k,
        fallback_k: if k < 48 { 48 } else { 0 },
    };
    build_cells_data_flat_adaptive(points, knn, adaptive, termination)
}

pub fn build_cells_data_flat_adaptive(
    points: &[Vec3],
    knn: &impl KnnProvider,
    adaptive: AdaptiveKConfig,
    termination: TerminationConfig,
) -> FlatCellsData {
    use rayon::prelude::*;

    let n = points.len();
    if n == 0 {
        return FlatCellsData {
            chunks: Vec::new(),
            num_cells: 0,
        };
    }

    let threads = rayon::current_num_threads().max(1);
    let chunk_size = (n / (threads * 8)).clamp(256, 4096).max(1);
    let mut ranges = Vec::with_capacity((n + chunk_size - 1) / chunk_size);
    let mut start = 0;
    while start < n {
        let end = (start + chunk_size).min(n);
        ranges.push((start, end));
        start = end;
    }

    // Check if adaptive-k is actually needed (step_k > 0 and track_limit > initial_k)
    let use_resumable = adaptive.step_k > 0 && adaptive.track_limit > adaptive.initial_k;
    let max_capacity = adaptive.track_limit.max(adaptive.fallback_k);

    let chunks: Vec<FlatChunk> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let mut scratch = knn.make_scratch();
            let mut neighbors = Vec::with_capacity(max_capacity);
            let mut builder = IncrementalCellBuilder::new(0, Vec3::ZERO);

            let estimated_vertices = (end - start) * 6;
            let mut vertices = Vec::with_capacity(estimated_vertices);
            let mut counts = Vec::with_capacity(end - start);
            let mut total_neighbors_processed = 0usize;
            let mut terminated_cells = 0usize;
            let mut fallback_unterminated = 0usize;

            for i in start..end {
                builder.reset(i, points[i]);

                let mut cell_neighbors_processed = 0usize;
                let mut terminated = false;
                let mut current_k = adaptive.initial_k;
                let mut used_fallback = false;
                let mut full_scan_done = false;

                if use_resumable {
                    // Use resumable queries for adaptive-k
                    let track_limit = adaptive.track_limit;
                    neighbors.clear();
                    let mut status = knn.knn_resumable_into(
                        points[i], i, current_k, track_limit, &mut scratch, &mut neighbors
                    );

                    loop {
                        // Process neighbors starting from where we left off
                        for idx in cell_neighbors_processed..neighbors.len() {
                            let neighbor_idx = neighbors[idx];
                            let neighbor = points[neighbor_idx];
                            builder.clip(neighbor_idx, neighbor);
                            cell_neighbors_processed = idx + 1;

                            if termination.should_check(cell_neighbors_processed) && builder.vertex_count() >= 3 {
                                let neighbor_cos = points[i].dot(neighbor).clamp(-1.0, 1.0);
                                if builder.can_terminate(neighbor_cos) {
                                    terminated = true;
                                    break;
                                }
                            }
                        }

                        if terminated {
                            break;
                        }

                        // Need more neighbors?
                        if status.is_exhausted() || current_k >= track_limit {
                            // Hit track_limit - check if we should fallback
                            if !terminated && adaptive.fallback_k > track_limit && !used_fallback {
                                // Fallback: fresh query with higher k
                                // IMPORTANT: The fresh query returns ALL k neighbors sorted by distance.
                                // We must process from index 0, not cell_neighbors_processed, because
                                // the new neighbors list is completely different from the resumable one.
                                // However, we've already clipped some neighbors into the builder.
                                // The builder tracks clipped neighbor indices, so we skip those.
                                used_fallback = true;
                                neighbors.clear();
                                knn.knn_into(points[i], i, adaptive.fallback_k, &mut scratch, &mut neighbors);

                                // Process ALL neighbors from the fresh query, skipping already-clipped ones
                                for idx in 0..neighbors.len() {
                                    let neighbor_idx = neighbors[idx];
                                    // Skip if we already clipped this neighbor
                                    if builder.has_neighbor(neighbor_idx) {
                                        continue;
                                    }
                                    let neighbor = points[neighbor_idx];
                                    builder.clip(neighbor_idx, neighbor);
                                    cell_neighbors_processed += 1;

                                    if termination.should_check(cell_neighbors_processed) && builder.vertex_count() >= 3 {
                                        let neighbor_cos = points[i].dot(neighbor).clamp(-1.0, 1.0);
                                        if builder.can_terminate(neighbor_cos) {
                                            terminated = true;
                                            break;
                                        }
                                    }
                                }
                                if termination.enabled
                                    && !terminated
                                    && adaptive.fallback_k < points.len().saturating_sub(1)
                                {
                                    // Correctness fallback: clip against all generators.
                                    // This is rare and expensive but guarantees completeness.
                                    for (p_idx, &p) in points.iter().enumerate() {
                                        if p_idx == i || builder.has_neighbor(p_idx) {
                                            continue;
                                        }
                                        builder.clip(p_idx, p);
                                        cell_neighbors_processed += 1;
                                        if builder.is_dead() {
                                            break;
                                        }
                                    }
                                    full_scan_done = true;
                                }
                                break; // Fallback processed, we're done
                            }
                            break; // Can't fetch more
                        }

                        // Resume with larger k
                        current_k = (current_k + adaptive.step_k).min(track_limit);
                        status = knn.knn_resume_into(
                            points[i], i, current_k, &mut scratch, &mut neighbors
                        );
                    }
                } else {
                    // Non-adaptive: single fetch
                    neighbors.clear();
                    knn.knn_into(points[i], i, current_k, &mut scratch, &mut neighbors);

                    for idx in 0..neighbors.len() {
                        let neighbor_idx = neighbors[idx];
                        let neighbor = points[neighbor_idx];
                        builder.clip(neighbor_idx, neighbor);
                        cell_neighbors_processed = idx + 1;

                        if termination.should_check(cell_neighbors_processed) && builder.vertex_count() >= 3 {
                            let neighbor_cos = points[i].dot(neighbor).clamp(-1.0, 1.0);
                            if builder.can_terminate(neighbor_cos) {
                                terminated = true;
                                break;
                            }
                        }
                    }
                }

                total_neighbors_processed += cell_neighbors_processed;
                terminated_cells += terminated as usize;
                if termination.enabled && used_fallback && !terminated {
                    fallback_unterminated += 1;
                }

                if builder.is_dead() {
                    let recovered = builder.try_reseed_best();
                    if !recovered {
                        if !full_scan_done {
                            builder.reset(i, points[i]);
                            for (p_idx, &p) in points.iter().enumerate() {
                                if p_idx == i {
                                    continue;
                                }
                                builder.clip(p_idx, p);
                            }
                        }
                        let recovered = if builder.is_dead() {
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

                let count = builder.get_vertices_into(&mut vertices);
                debug_assert!(count <= u8::MAX as usize, "vertex count exceeds u8");
                counts.push(count as u8);
            }

            FlatChunk {
                vertices,
                counts,
                total_neighbors_processed,
                terminated_cells,
                fallback_unterminated,
            }
        })
        .collect();

    FlatCellsData {
        chunks,
        num_cells: n,
    }
}

struct CoreResult {
    voronoi: crate::geometry::SphericalVoronoi,
    stats: Option<VoronoiStats>,
    timings_ms: PhaseTimingsMs,
}

/// Result of merging close points before Voronoi computation.
pub struct MergeResult {
    /// Points to use for Voronoi (representatives only, or all if no merges).
    pub effective_points: Vec<Vec3>,
    /// Maps original point index -> representative index in effective_points.
    /// If no merges occurred, this is just identity (0, 1, 2, ...).
    pub original_to_effective: Vec<usize>,
    /// Number of points that were merged (removed).
    pub num_merged: usize,
}

/// Simple disjoint-set (union-find) for point merging.
struct SimpleDsu {
    parent: Vec<usize>,
}

impl SimpleDsu {
    fn new(n: usize) -> Self {
        Self { parent: (0..n).collect() }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

/// Find and merge points that are too close together.
/// Uses strict radius-based merging to ensure no remaining pair is within threshold.
/// Returns effective points (representatives) and a mapping from original to effective indices.
///
/// NOTE: A potentially more efficient approach for the future would be to detect close
/// neighbors during cell construction and "borrow" the cell from the close neighbor
/// when a cell dies. This would avoid the preprocessing pass but requires more complex
/// recovery logic in the cell builder.
pub fn merge_close_points(points: &[Vec3], threshold: f32, knn: &impl KnnProvider) -> MergeResult {
    let n = points.len();
    if n == 0 {
        return MergeResult {
            effective_points: Vec::new(),
            original_to_effective: Vec::new(),
            num_merged: 0,
        };
    }

    let threshold_sq = threshold * threshold;
    let mut dsu = SimpleDsu::new(n);
    let _ = knn; // Keep signature for now; strict merge doesn't use k-NN.

    // Strict radius-based merge: union all pairs within threshold.
    // Guarantees no remaining pair is closer than threshold (within precision).
    let cell_size = threshold;
    let inv_cell_size = 1.0 / cell_size;

    #[inline]
    fn grid_cell(p: Vec3, inv_cell_size: f32) -> (i32, i32, i32) {
        (
            (p.x * inv_cell_size).floor() as i32,
            (p.y * inv_cell_size).floor() as i32,
            (p.z * inv_cell_size).floor() as i32,
        )
    }

    let mut grid: rustc_hash::FxHashMap<(i32, i32, i32), Vec<usize>> =
        rustc_hash::FxHashMap::default();

    for (i, &p) in points.iter().enumerate() {
        let (cx, cy, cz) = grid_cell(p, inv_cell_size);

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in indices {
                            let dist_sq = (points[j] - p).length_squared();
                            if dist_sq < threshold_sq {
                                dsu.union(i, j);
                            }
                        }
                    }
                }
            }
        }

        grid.entry((cx, cy, cz))
            .or_insert_with(Vec::new)
            .push(i);
    }

    // Count how many unique representatives we have
    let mut rep_to_effective: Vec<Option<usize>> = vec![None; n];
    let mut effective_points = Vec::new();
    let mut original_to_effective = vec![0usize; n];

    for i in 0..n {
        let rep = dsu.find(i);
        if rep_to_effective[rep].is_none() {
            rep_to_effective[rep] = Some(effective_points.len());
            effective_points.push(points[rep]);
        }
        original_to_effective[i] = rep_to_effective[rep].unwrap();
    }

    let num_merged = n - effective_points.len();

    MergeResult {
        effective_points,
        original_to_effective,
        num_merged,
    }
}

fn compute_voronoi_gpu_style_core(
    points: &[Vec3],
    k: usize,
    termination: TerminationConfig,
    collect_stats: bool,
) -> CoreResult {
    use std::time::Instant;

    let t0 = Instant::now();

    // Build initial KNN on original points (used for merge detection)
    let knn = CubeMapGridKnn::new(points);

    // Find and merge close points to prevent orphan edges
    let merge_result = merge_close_points(points, MIN_BISECTOR_DISTANCE, &knn);
    let t1 = Instant::now();

    // Use effective points for Voronoi computation
    let (effective_points, effective_knn);
    let needs_remap = merge_result.num_merged > 0;

    if needs_remap {
        // Some points were merged - need new KNN on effective points
        effective_points = merge_result.effective_points;
        effective_knn = Some(CubeMapGridKnn::new(&effective_points));
    } else {
        // No merges - use original points directly
        effective_points = points.to_vec();
        effective_knn = None;
    }

    let knn_ref: &CubeMapGridKnn = effective_knn.as_ref().unwrap_or(&knn);

    let flat_data = build_cells_data_flat(&effective_points, knn_ref, k, termination);
    let stats = if collect_stats { Some(flat_data.stats()) } else { None };
    let t2 = Instant::now();

    let (all_vertices, eff_cells, eff_cell_indices) =
        dedup::dedup_vertices_hash_flat(flat_data, &effective_points, false);
    let t3 = Instant::now();

    // Remap cells back to original point indices if we merged
    let (cells, cell_indices) = if needs_remap {
        use crate::geometry::VoronoiCell;

        // Each original point maps to an effective point's cell
        let mut new_cells = Vec::with_capacity(points.len());
        let mut new_cell_indices = Vec::new();

        for orig_idx in 0..points.len() {
            let eff_idx = merge_result.original_to_effective[orig_idx];
            let eff_cell = &eff_cells[eff_idx];

            let start = new_cell_indices.len();
            let eff_start = eff_cell.vertex_start();
            let eff_end = eff_start + eff_cell.vertex_count();
            new_cell_indices.extend_from_slice(&eff_cell_indices[eff_start..eff_end]);

            new_cells.push(VoronoiCell::new(orig_idx, start, eff_cell.vertex_count()));
        }
        (new_cells, new_cell_indices)
    } else {
        (eff_cells, eff_cell_indices)
    };

    let voronoi = crate::geometry::SphericalVoronoi::from_raw_parts(
        points.to_vec(),
        all_vertices,
        cells,
        cell_indices,
    );
    let t4 = Instant::now();

    let timings_ms = PhaseTimingsMs {
        total: (t4 - t0).as_secs_f64() * 1000.0,
        kdtree: (t1 - t0).as_secs_f64() * 1000.0,
        knn: 0.0,
        cell_construction: (t2 - t1).as_secs_f64() * 1000.0,
        dedup: (t3 - t2).as_secs_f64() * 1000.0,
        assemble: (t4 - t3).as_secs_f64() * 1000.0,
    };

    CoreResult { voronoi, stats, timings_ms }
}

/// Compute spherical Voronoi diagram using the GPU-style algorithm.
pub fn compute_voronoi_gpu_style(points: &[Vec3], k: usize) -> crate::geometry::SphericalVoronoi {
    compute_voronoi_gpu_style_timed(points, k, false)
}

/// Compute spherical Voronoi with statistics.
pub fn compute_voronoi_gpu_style_with_stats(
    points: &[Vec3],
    k: usize,
) -> (crate::geometry::SphericalVoronoi, VoronoiStats) {
    compute_voronoi_gpu_style_with_stats_and_termination_params(points, k, true, 10, 6)
}

/// Compute spherical Voronoi with statistics, optionally disabling early termination.
pub fn compute_voronoi_gpu_style_with_stats_and_termination(
    points: &[Vec3],
    k: usize,
    enable_termination: bool,
) -> (crate::geometry::SphericalVoronoi, VoronoiStats) {
    compute_voronoi_gpu_style_with_stats_and_termination_params(points, k, enable_termination, 10, 6)
}

/// Compute spherical Voronoi with statistics and configurable termination.
pub fn compute_voronoi_gpu_style_with_stats_and_termination_params(
    points: &[Vec3],
    k: usize,
    enable_termination: bool,
    termination_check_start: usize,
    termination_check_step: usize,
) -> (crate::geometry::SphericalVoronoi, VoronoiStats) {
    let termination = TerminationConfig {
        enabled: enable_termination,
        check_start: termination_check_start,
        check_step: termination_check_step,
    };
    let result = compute_voronoi_gpu_style_core(points, k, termination, true);
    (result.voronoi, result.stats.expect("stats requested"))
}

/// Compute spherical Voronoi with optional timing output.
pub fn compute_voronoi_gpu_style_timed(
    points: &[Vec3],
    k: usize,
    print_timing: bool,
) -> crate::geometry::SphericalVoronoi {
    compute_voronoi_gpu_style_timed_with_termination(points, k, print_timing, true)
}

/// Compute spherical Voronoi with optional timing output, optionally disabling early termination.
pub fn compute_voronoi_gpu_style_timed_with_termination(
    points: &[Vec3],
    k: usize,
    print_timing: bool,
    enable_termination: bool,
) -> crate::geometry::SphericalVoronoi {
    compute_voronoi_gpu_style_timed_with_termination_params(points, k, print_timing, enable_termination, 10, 6)
}

/// Compute spherical Voronoi with optional timing output and configurable termination.
pub fn compute_voronoi_gpu_style_timed_with_termination_params(
    points: &[Vec3],
    k: usize,
    print_timing: bool,
    enable_termination: bool,
    termination_check_start: usize,
    termination_check_step: usize,
) -> crate::geometry::SphericalVoronoi {
    let termination = TerminationConfig {
        enabled: enable_termination,
        check_start: termination_check_start,
        check_step: termination_check_step,
    };
    let result = compute_voronoi_gpu_style_core(points, k, termination, false);

    if print_timing {
        let t = result.timings_ms;
        let total = t.total.max(1e-9);
        println!("GPU-style Voronoi breakdown (n={}, k={}):", points.len(), k);
        println!("  k-d tree build:    {:6.1} ms ({:4.1}%)", t.kdtree, t.kdtree / total * 100.0);
        if t.knn > 0.0 {
            println!("  k-NN queries:      {:6.1} ms ({:4.1}%)", t.knn, t.knn / total * 100.0);
        } else {
            println!("  k-NN queries:      (computed on-demand in cell construction)");
        }
        println!("  Cell construction: {:6.1} ms ({:4.1}%)", t.cell_construction, t.cell_construction / total * 100.0);
        println!("  Dedup (hash):      {:6.1} ms ({:4.1}%) [FxHashMap]", t.dedup, t.dedup / total * 100.0);
        println!("  Assemble struct:   {:6.1} ms ({:4.1}%) [{} verts]", t.assemble, t.assemble / total * 100.0, result.voronoi.vertices.len());
        println!("  Total:             {:6.1} ms", t.total);
    }

    result.voronoi
}

/// Benchmark result for comparing Voronoi methods.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub method: String,
    pub num_points: usize,
    pub time_ms: f64,
    pub cells_with_vertices: usize,
    pub avg_vertices_per_cell: f64,
}

/// Run a benchmark comparing the two Voronoi methods.
pub fn benchmark_voronoi(
    num_points: usize,
    k: usize,
    iterations: usize,
) -> (BenchmarkResult, BenchmarkResult) {
    use crate::geometry::{random_sphere_points, SphericalVoronoi};
    use std::time::Instant;

    let points = random_sphere_points(num_points);

    // Warm up
    let _ = SphericalVoronoi::compute(&points);
    let _ = compute_voronoi_gpu_style(&points, k);

    // Benchmark convex hull method
    let start = Instant::now();
    let mut hull_result = None;
    for _ in 0..iterations {
        hull_result = Some(SphericalVoronoi::compute(&points));
    }
    let hull_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    let hull_voronoi = hull_result.unwrap();

    let hull_cells_with_verts = hull_voronoi.iter_cells().filter(|c| !c.is_empty()).count();
    let hull_avg_verts = hull_voronoi.iter_cells().map(|c| c.len()).sum::<usize>() as f64
        / hull_voronoi.num_cells() as f64;

    // Benchmark GPU-style method
    let start = Instant::now();
    let mut gpu_result = None;
    for _ in 0..iterations {
        gpu_result = Some(compute_voronoi_gpu_style(&points, k));
    }
    let gpu_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    let gpu_voronoi = gpu_result.unwrap();

    let gpu_cells_with_verts = gpu_voronoi.iter_cells().filter(|c| !c.is_empty()).count();
    let gpu_avg_verts = gpu_voronoi.iter_cells().map(|c| c.len()).sum::<usize>() as f64
        / gpu_voronoi.num_cells() as f64;

    (
        BenchmarkResult {
            method: "Convex Hull".to_string(),
            num_points,
            time_ms: hull_time,
            cells_with_vertices: hull_cells_with_verts,
            avg_vertices_per_cell: hull_avg_verts,
        },
        BenchmarkResult {
            method: format!("GPU-style (k={})", k),
            num_points,
            time_ms: gpu_time,
            cells_with_vertices: gpu_cells_with_verts,
            avg_vertices_per_cell: gpu_avg_verts,
        },
    )
}
