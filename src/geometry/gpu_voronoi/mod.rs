//! GPU-friendly spherical Voronoi computation via half-space (great circle) clipping.
//!
//! This module implements a "meshless" approach where each Voronoi cell is computed
//! independently from its k nearest neighbors. This enables massive parallelism on GPU.

mod cell_builder;
mod constants;
pub mod dedup;
mod live_dedup;
mod knn;
mod timing;

#[cfg(test)]
mod tests;

use glam::Vec3;
use kiddo::ImmutableKdTree;
use rustc_hash::FxHashSet;

// Re-exports
pub use cell_builder::{
    geodesic_distance, order_vertices_ccw_indices, F64CellBuilder, GreatCircle, VertexData,
    VertexKey, VertexList, MAX_PLANES, MAX_VERTICES,
};
pub use constants::{
    support_cluster_drift_dot, MIN_BISECTOR_DISTANCE, SUPPORT_CERT_MARGIN_ABS,
    SUPPORT_CLUSTER_RADIUS_ANGLE, SUPPORT_EPS_ABS, SUPPORT_VERTEX_ANGLE_EPS, VERTEX_WELD_FRACTION,
};
pub use dedup::dedup_vertices_hash_flat;
pub use knn::{CubeMapGridKnn, KnnProvider};

/// Approximate mean chord length between uniformly-distributed generators.
/// Used to scale tolerances; assumes roughly even generator spacing.
#[allow(dead_code)]
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
    pub(crate) support_data: Vec<u32>,
    /// Per-cell flag: whether the cell terminated early.
    #[allow(dead_code)]
    pub(crate) cell_terminated: Vec<u8>,
    /// Per-cell flag: whether the cell used the fallback k-NN query.
    #[allow(dead_code)]
    pub(crate) cell_used_fallback: Vec<u8>,
    /// Per-cell flag: whether the cell performed a full scan over all generators.
    #[allow(dead_code)]
    pub(crate) cell_full_scan_done: Vec<u8>,
    /// Per-cell flag: whether the neighbor candidate set is treated as complete
    /// (i.e., support certification is eligible to run for this cell).
    #[allow(dead_code)]
    pub(crate) cell_candidates_complete: Vec<u8>,
    /// Total neighbors processed in this chunk (for stats).
    pub(crate) total_neighbors_processed: usize,
    /// Number of cells that terminated early in this chunk.
    pub(crate) terminated_cells: usize,
    /// Number of cells that used fallback but still did not terminate.
    pub(crate) fallback_unterminated: usize,
    /// Number of cells where support certification was evaluated.
    pub(crate) cert_checked_cells: usize,
    /// Number of cells where support certification failed.
    pub(crate) cert_failed_cells: usize,
    /// Number of vertices where support certification was evaluated.
    pub(crate) cert_checked_vertices: usize,
    /// Number of vertices where support certification failed.
    pub(crate) cert_failed_vertices: usize,
    /// Number of vertices that failed due to ill-conditioned intersections.
    pub(crate) cert_failed_ill_vertices: usize,
    /// Number of vertices that failed due to a small gap to the next best generator.
    pub(crate) cert_failed_gap_vertices: usize,
    /// Number of support gap samples recorded.
    pub(crate) cert_gap_count: usize,
    /// Minimum observed support gap.
    pub(crate) cert_gap_min: f64,
    /// Reservoir samples of support gaps.
    pub(crate) cert_gap_samples: Vec<f64>,
    /// Failed vertex indices with reason codes (1=ill_cond, 2=gap, 3=both).
    /// Index is global within the chunk's vertex array.
    pub(crate) cert_failed_vertex_indices: Vec<(u32, u8)>,
    /// Starting global vertex index for this chunk (set during assembly).
    pub(crate) vertex_offset: u32,
    /// Number of cells that used f64 fallback.
    pub(crate) f64_fallback_cells: usize,
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
        let total_neighbors: usize = self
            .chunks
            .iter()
            .map(|c| c.total_neighbors_processed)
            .sum();
        let terminated: usize = self.chunks.iter().map(|c| c.terminated_cells).sum();
        let fallback_unterminated: usize =
            self.chunks.iter().map(|c| c.fallback_unterminated).sum();
        let cert_checked_cells: usize = self.chunks.iter().map(|c| c.cert_checked_cells).sum();
        let cert_failed_cells: usize = self.chunks.iter().map(|c| c.cert_failed_cells).sum();
        let cert_checked_vertices: usize =
            self.chunks.iter().map(|c| c.cert_checked_vertices).sum();
        let cert_failed_vertices: usize = self.chunks.iter().map(|c| c.cert_failed_vertices).sum();
        let cert_failed_ill_vertices: usize =
            self.chunks.iter().map(|c| c.cert_failed_ill_vertices).sum();
        let cert_failed_gap_vertices: usize =
            self.chunks.iter().map(|c| c.cert_failed_gap_vertices).sum();
        let cert_gap_count: usize = self.chunks.iter().map(|c| c.cert_gap_count).sum();
        let cert_gap_min = self
            .chunks
            .iter()
            .filter(|c| c.cert_gap_count > 0)
            .map(|c| c.cert_gap_min)
            .fold(f64::INFINITY, f64::min);
        let mut cert_gap_samples: Vec<(f64, f64)> = Vec::new();
        cert_gap_samples.reserve(self.chunks.len() * 8);
        for chunk in &self.chunks {
            if chunk.cert_gap_count == 0 || chunk.cert_gap_samples.is_empty() {
                continue;
            }
            let weight = chunk.cert_gap_count as f64 / chunk.cert_gap_samples.len() as f64;
            cert_gap_samples.extend(chunk.cert_gap_samples.iter().map(|&v| (v, weight)));
        }
        let cert_gap_median = if cert_gap_count == 0 || cert_gap_samples.is_empty() {
            f64::NAN
        } else {
            cert_gap_samples
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let target = cert_gap_count as f64 * 0.5;
            let mut accum = 0.0f64;
            let mut median = cert_gap_samples.last().map(|v| v.0).unwrap_or(f64::NAN);
            for (value, weight) in cert_gap_samples {
                accum += weight;
                if accum >= target {
                    median = value;
                    break;
                }
            }
            median
        };
        let f64_fallback_cells: usize = self.chunks.iter().map(|c| c.f64_fallback_cells).sum();
        // Aggregate failed vertex indices with global offsets
        let mut cert_failed_vertex_indices: Vec<(u32, u8)> = Vec::new();
        for chunk in &self.chunks {
            let offset = chunk.vertex_offset;
            cert_failed_vertex_indices.extend(
                chunk
                    .cert_failed_vertex_indices
                    .iter()
                    .map(|&(idx, reason)| (idx + offset, reason)),
            );
        }
        let n = self.num_cells.max(1) as f64;
        VoronoiStats {
            avg_neighbors_processed: total_neighbors as f64 / n,
            termination_rate: terminated as f64 / n,
            fallback_unterminated,
            support_cert_checked: cert_checked_cells,
            support_cert_failed: cert_failed_cells,
            support_cert_checked_vertices: cert_checked_vertices,
            support_cert_failed_vertices: cert_failed_vertices,
            support_cert_failed_ill_vertices: cert_failed_ill_vertices,
            support_cert_failed_gap_vertices: cert_failed_gap_vertices,
            support_cert_gap_count: cert_gap_count,
            support_cert_gap_min: cert_gap_min,
            support_cert_gap_median: cert_gap_median,
            support_cert_failed_vertex_indices: cert_failed_vertex_indices,
            f64_fallback_cells,
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
    /// Number of cells where support certification was evaluated.
    pub support_cert_checked: usize,
    /// Number of cells where support certification failed.
    pub support_cert_failed: usize,
    /// Number of vertices where support certification was evaluated.
    pub support_cert_checked_vertices: usize,
    /// Number of vertices where support certification failed.
    pub support_cert_failed_vertices: usize,
    /// Number of vertices that failed due to ill-conditioned intersections.
    pub support_cert_failed_ill_vertices: usize,
    /// Number of vertices that failed due to a small gap to the next best generator.
    pub support_cert_failed_gap_vertices: usize,
    /// Number of gap samples recorded.
    pub support_cert_gap_count: usize,
    /// Minimum observed support gap.
    pub support_cert_gap_min: f64,
    /// Median observed support gap (approximate).
    pub support_cert_gap_median: f64,
    /// Failed vertex indices with reason codes (1=ill_cond, 2=gap, 3=both).
    /// Global vertex indices across all chunks.
    pub support_cert_failed_vertex_indices: Vec<(u32, u8)>,
    /// Number of cells that used f64 fallback.
    pub f64_fallback_cells: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct TerminationConfig {
    /// Enables adaptive k-NN + early termination checks.
    ///
    /// When disabled, the builder will still run the initial k-NN pass but will not
    /// attempt to terminate early or expand to larger k (so this should generally
    /// remain enabled for correct results).
    pub enabled: bool,
    pub check_start: usize,
    pub check_step: usize,
}

// Keep the adaptive-k schedule and the default termination cadence in one place.
// If you change the k schedule, the default termination values should update with it.
pub(super) const ADAPTIVE_K_INITIAL: usize = 12;
pub(super) const ADAPTIVE_K_RESUME: usize = 24;
pub(super) const ADAPTIVE_K_RARE: usize = 48;

// Default termination cadence:
// - start near the end of the initial k pass
// - then check roughly twice per initial-k window
const DEFAULT_TERMINATION_CHECK_START: usize = 6;
const DEFAULT_TERMINATION_CHECK_STEP: usize = 1;

impl Default for TerminationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_start: DEFAULT_TERMINATION_CHECK_START,
            check_step: DEFAULT_TERMINATION_CHECK_STEP,
        }
    }
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

/// Build cells into flat buffers with chunked parallelism.
/// Returns (vertices, counts).
/// - vertices: flat buffer of all vertex data
/// - counts: vertex count per cell (counts[i] = number of vertices for cell i)
///
/// Uses adaptive k-NN with early termination:
/// 1. Fetch k=12 neighbors, clip, termination checks
/// 2. If needed, re-query at k=24, then k=48
/// 3. If still not terminated, full scan
pub fn build_cells_data_flat(
    points: &[Vec3],
    knn: &impl KnnProvider,
    termination: TerminationConfig,
) -> (FlatCellsData, timing::CellSubAccum) {
    use rayon::prelude::*;

    let n = points.len();
    if n == 0 {
        return (
            FlatCellsData {
                chunks: Vec::new(),
                num_cells: 0,
            },
            timing::CellSubAccum::new(),
        );
    }

    let _support_eps = SUPPORT_EPS_ABS; // Kept for potential future use

    let threads = rayon::current_num_threads().max(1);
    let chunk_size = (n / (threads * 8)).clamp(256, 4096).max(1);
    let mut ranges = Vec::with_capacity((n + chunk_size - 1) / chunk_size);
    let mut start = 0;
    while start < n {
        let end = (start + chunk_size).min(n);
        ranges.push((start, end));
        start = end;
    }

    let chunks: Vec<(FlatChunk, timing::CellSubAccum)> = ranges
        .par_iter()
        .map(|&(start, end)| {
            use cell_builder::F64CellBuilder;
            use timing::{Timer, CellSubAccum};

            let mut scratch = knn.make_scratch();
            let mut neighbors = Vec::with_capacity(64);
            let mut builder = F64CellBuilder::new(0, Vec3::ZERO);

            let estimated_vertices = (end - start) * 6;
            let mut vertices = Vec::with_capacity(estimated_vertices);
            let mut counts = Vec::with_capacity(end - start);
            let mut support_data: Vec<u32> = Vec::with_capacity(estimated_vertices);
            let mut cell_terminated: Vec<u8> = Vec::with_capacity(end - start);
            let mut cell_used_fallback: Vec<u8> = Vec::with_capacity(end - start);
            let mut cell_full_scan_done: Vec<u8> = Vec::with_capacity(end - start);
            let mut cell_candidates_complete: Vec<u8> = Vec::with_capacity(end - start);
            let mut total_neighbors_processed = 0usize;
            let mut terminated_cells = 0usize;
            let mut fallback_unterminated = 0usize;
            let cert_checked_cells = 0usize;
            let cert_failed_cells = 0usize;
            let cert_checked_vertices = 0usize;
            let cert_failed_vertices = 0usize;
            let cert_failed_ill_vertices = 0usize;
            let cert_failed_gap_vertices = 0usize;
            let cert_failed_vertex_indices: Vec<(u32, u8)> = Vec::new();
            let f64_fallback_cells = 0usize;

            // Sub-phase timing accumulators
            let mut sub_accum = CellSubAccum::new();

            for i in start..end {
                builder.reset(i, points[i]);

                let mut cell_neighbors_processed = 0usize;
                let mut terminated = false;
                let mut used_fallback = false;
                let mut full_scan_done = false;
                let mut knn_exhausted_any = false;
                let mut did_k24 = false;
                let mut did_k48 = false;
                let mut did_full_scan_fallback = false;
                let mut did_full_scan_recovery = false;

                // Phase 1: Initial k-NN query (k=12)
                let t_knn = Timer::start();
                neighbors.clear();
                knn.knn_into(
                    points[i],
                    i,
                    ADAPTIVE_K_INITIAL,
                    &mut scratch,
                    &mut neighbors,
                );
                let mut knn_exhausted = neighbors.len() < ADAPTIVE_K_INITIAL;
                knn_exhausted_any |= knn_exhausted;
                sub_accum.add_knn(t_knn.elapsed());

                let t_clip = Timer::start();

                // Process initial neighbors
                let mut last_neighbor_cos: Option<f32> = None;
                for &neighbor_idx in &neighbors {
                    let neighbor = points[neighbor_idx];
                    builder.clip(neighbor_idx, neighbor);
                    cell_neighbors_processed += 1;

                    if termination.enabled && builder.vertex_count() >= 3 {
                        let neighbor_cos = points[i].dot(neighbor).clamp(-1.0, 1.0);
                        last_neighbor_cos = Some(neighbor_cos);
                        if termination.should_check(cell_neighbors_processed)
                            && builder.can_terminate(neighbor_cos)
                        {
                            terminated = true;
                            break;
                        }
                    }
                }
                // Always check once at the end of the stage so we don't "miss" termination
                // due to a coarse check cadence.
                if termination.enabled && !terminated && builder.vertex_count() >= 3 {
                    if let Some(cos) = last_neighbor_cos {
                        if builder.can_terminate(cos) {
                            terminated = true;
                        }
                    }
                }

                // Phase 2: Re-query at k=24 if needed
                if termination.enabled && !terminated && !knn_exhausted {
                    used_fallback = true;
                    did_k24 = true;
                    let t_knn2 = Timer::start();
                    knn.knn_into(points[i], i, ADAPTIVE_K_RESUME, &mut scratch, &mut neighbors);
                    knn_exhausted = neighbors.len() < ADAPTIVE_K_RESUME;
                    knn_exhausted_any |= knn_exhausted;
                    sub_accum.add_knn(t_knn2.elapsed());

                    // Process (potentially overlapping) neighbors; skip ones we've already clipped.
                    let mut last_neighbor_cos: Option<f32> = None;
                    for &neighbor_idx in &neighbors {
                        if builder.has_neighbor(neighbor_idx) {
                            continue;
                        }
                        let neighbor = points[neighbor_idx];
                        builder.clip(neighbor_idx, neighbor);
                        cell_neighbors_processed += 1;

                        if builder.vertex_count() >= 3 {
                            let neighbor_cos = points[i].dot(neighbor).clamp(-1.0, 1.0);
                            last_neighbor_cos = Some(neighbor_cos);
                            if termination.should_check(cell_neighbors_processed)
                                && builder.can_terminate(neighbor_cos)
                            {
                                terminated = true;
                                break;
                            }
                        }
                    }
                    if !terminated && builder.vertex_count() >= 3 {
                        if let Some(cos) = last_neighbor_cos {
                            if builder.can_terminate(cos) {
                                terminated = true;
                            }
                        }
                    }
                }

                // Phase 3: Fresh query at k=48 if still needed (rare)
                if termination.enabled && !terminated && !knn_exhausted {
                    used_fallback = true;
                    did_k48 = true;
                    let t_knn3 = Timer::start();
                    knn.knn_into(
                        points[i],
                        i,
                        ADAPTIVE_K_RARE,
                        &mut scratch,
                        &mut neighbors,
                    );
                    knn_exhausted = neighbors.len() < ADAPTIVE_K_RARE;
                    knn_exhausted_any |= knn_exhausted;
                    sub_accum.add_knn(t_knn3.elapsed());

                    let mut last_neighbor_cos: Option<f32> = None;
                    for &neighbor_idx in &neighbors {
                        if builder.has_neighbor(neighbor_idx) {
                            continue;
                        }
                        let neighbor = points[neighbor_idx];
                        builder.clip(neighbor_idx, neighbor);
                        cell_neighbors_processed += 1;

                        if builder.vertex_count() >= 3 {
                            let neighbor_cos = points[i].dot(neighbor).clamp(-1.0, 1.0);
                            last_neighbor_cos = Some(neighbor_cos);
                            if termination.should_check(cell_neighbors_processed)
                                && builder.can_terminate(neighbor_cos)
                            {
                                terminated = true;
                                break;
                            }
                        }
                    }
                    if !terminated && builder.vertex_count() >= 3 {
                        if let Some(cos) = last_neighbor_cos {
                            if builder.can_terminate(cos) {
                                terminated = true;
                            }
                        }
                    }
                }
                sub_accum.add_clip(t_clip.elapsed());

                // Final fallback: full scan if knn exhausted and still not terminated
                if termination.enabled && !terminated && knn_exhausted && !builder.is_dead() && builder.vertex_count() >= 3 {
                    did_full_scan_fallback = true;
                    if std::env::var("VORONOI_DEBUG_FALLBACK").is_ok() {
                        eprintln!("[FALLBACK] cell {} entering full-scan fallback ({} neighbors so far)",
                            i, builder.planes_count());
                    }
                    let fallback_start = std::time::Instant::now();
                    let already_clipped: FxHashSet<usize> = builder.neighbor_indices_iter().collect();
                    for (p_idx, &p) in points.iter().enumerate() {
                        if p_idx == i || already_clipped.contains(&p_idx) {
                            continue;
                        }
                        builder.clip(p_idx, p);
                        cell_neighbors_processed += 1;
                        if builder.is_dead() {
                            break;
                        }
                    }
                    if std::env::var("VORONOI_DEBUG_FALLBACK").is_ok() {
                        let elapsed = fallback_start.elapsed().as_secs_f64() * 1000.0;
                        eprintln!("[FALLBACK] cell {} full-scan took {:.1}ms", i, elapsed);
                    }
                    full_scan_done = true;
                }

                total_neighbors_processed += cell_neighbors_processed;
                terminated_cells += terminated as usize;
                if termination.enabled && used_fallback && !terminated {
                    fallback_unterminated += 1;
                }

                // Dead cell recovery
                if builder.is_dead() {
                    let recovered = builder.try_reseed_best();
                    if !recovered {
                        if !full_scan_done {
                            did_full_scan_recovery = true;
                            if std::env::var("VORONOI_DEBUG_FALLBACK").is_ok() {
                                eprintln!("[RECOVERY] cell {} entering full-scan recovery (was dead after {} planes)",
                                    i, builder.planes_count());
                            }
                            let recovery_start = std::time::Instant::now();
                            builder.reset(i, points[i]);
                            for (p_idx, &p) in points.iter().enumerate() {
                                if p_idx == i {
                                    continue;
                                }
                                builder.clip(p_idx, p);
                            }
                            if std::env::var("VORONOI_DEBUG_FALLBACK").is_ok() {
                                let elapsed = recovery_start.elapsed().as_secs_f64() * 1000.0;
                                eprintln!("[RECOVERY] cell {} full-scan recovery took {:.1}ms", i, elapsed);
                            }
                            full_scan_done = true;
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

                let knn_stage = if did_full_scan_recovery {
                    timing::KnnCellStage::FullScanRecovery
                } else if did_full_scan_fallback {
                    timing::KnnCellStage::FullScanFallback
                } else if did_k48 {
                    timing::KnnCellStage::K48
                } else if did_k24 {
                    timing::KnnCellStage::K24
                } else {
                    timing::KnnCellStage::K12
                };
                sub_accum.add_cell_stage(knn_stage, knn_exhausted_any);

                let candidates_complete = full_scan_done || (termination.enabled && terminated);
                cell_terminated.push(terminated as u8);
                cell_used_fallback.push(used_fallback as u8);
                cell_full_scan_done.push(full_scan_done as u8);
                cell_candidates_complete.push(candidates_complete as u8);

                // Extract vertices with certified keys
                let t_cert = Timer::start();
                if builder.is_dead() || builder.vertex_count() < 3 {
                    panic!(
                        "Cell {} construction failed: is_dead={}, vertex_count={}",
                        i,
                        builder.is_dead(),
                        builder.vertex_count()
                    );
                }
                let cell_vertices = builder.to_vertex_data(points, &mut support_data);
                let count = cell_vertices.len();
                vertices.extend(cell_vertices);
                sub_accum.add_cert(t_cert.elapsed());
                counts.push(count as u8);
            }

            (FlatChunk {
                vertices,
                counts,
                support_data,
                cell_terminated,
                cell_used_fallback,
                cell_full_scan_done,
                cell_candidates_complete,
                total_neighbors_processed,
                terminated_cells,
                fallback_unterminated,
                cert_checked_cells,
                cert_failed_cells,
                cert_checked_vertices,
                cert_failed_vertices,
                cert_failed_ill_vertices,
                cert_failed_gap_vertices,
                cert_gap_count: 0,
                cert_gap_min: f64::INFINITY,
                cert_gap_samples: Vec::new(),
                cert_failed_vertex_indices,
                vertex_offset: 0, // Computed below
                f64_fallback_cells,
            }, sub_accum)
        })
        .collect();

    // Separate chunks and timing, merge timing from all chunks
    let mut merged_timing = timing::CellSubAccum::new();
    let mut flat_chunks: Vec<FlatChunk> = Vec::with_capacity(chunks.len());
    for (chunk, sub_timing) in chunks {
        merged_timing.merge(&sub_timing);
        flat_chunks.push(chunk);
    }

    // Compute vertex offsets for global indexing
    let mut offset = 0u32;
    for chunk in &mut flat_chunks {
        chunk.vertex_offset = offset;
        offset += chunk.vertices.len() as u32;
    }

    (
        FlatCellsData {
            chunks: flat_chunks,
            num_cells: n,
        },
        merged_timing,
    )
}

struct CoreResult {
    voronoi: crate::geometry::SphericalVoronoi,
    stats: Option<VoronoiStats>,
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
        Self {
            parent: (0..n).collect(),
        }
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

        grid.entry((cx, cy, cz)).or_insert_with(Vec::new).push(i);
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
    termination: TerminationConfig,
    collect_stats: bool,
    skip_preprocess: bool,
) -> CoreResult {
    use timing::{Timer, TimingBuilder};

    let mut tb = TimingBuilder::new();

    // Preprocessing: merge close points (not timed - separate from Voronoi computation)
    let (effective_points, merge_result) = if skip_preprocess {
        (points.to_vec(), None)
    } else {
        let knn_for_merge = CubeMapGridKnn::new(points);
        let result = merge_close_points(points, MIN_BISECTOR_DISTANCE, &knn_for_merge);
        let pts = if result.num_merged > 0 {
            result.effective_points.clone()
        } else {
            points.to_vec()
        };
        (pts, Some(result))
    };
    let needs_remap = merge_result.as_ref().map_or(false, |r| r.num_merged > 0);

    // Build KNN on effective points (this is the timed grid build)
    let t = Timer::start();
    let knn = CubeMapGridKnn::new(&effective_points);
    tb.set_knn_build(t.elapsed());

    let (all_vertices, eff_cells, eff_cell_indices, stats) = if collect_stats {
        let t = Timer::start();
        let (flat_data, cell_sub_timing) =
            build_cells_data_flat(&effective_points, &knn, termination);
        let stats = Some(flat_data.stats());
        tb.set_cell_construction(t.elapsed(), cell_sub_timing.into_sub_phases());

        let t = Timer::start();
        let (all_vertices, eff_cells, eff_cell_indices, dedup_sub) =
            dedup::dedup_vertices_hash_flat(flat_data, false);
        tb.set_dedup(t.elapsed(), dedup_sub);

        (all_vertices, eff_cells, eff_cell_indices, stats)
    } else {
        let t = Timer::start();
        let sharded =
            live_dedup::build_cells_sharded_live_dedup(&effective_points, &knn, termination);
        tb.set_cell_construction(t.elapsed(), sharded.cell_sub.clone().into_sub_phases());

        let t = Timer::start();
        let (all_vertices, eff_cells, eff_cell_indices, dedup_sub) =
            live_dedup::assemble_sharded_live_dedup(sharded);
        tb.set_dedup(t.elapsed(), dedup_sub);

        (all_vertices, eff_cells, eff_cell_indices, None)
    };

    // Remap cells back to original point indices if we merged
    let t = Timer::start();
    let (cells, cell_indices) = if needs_remap {
        use crate::geometry::VoronoiCell;
        let merge_result = merge_result.as_ref().unwrap();

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
    tb.set_assemble(t.elapsed());

    // Report timing if feature enabled
    let timings = tb.finish();
    timings.report(points.len());

    CoreResult { voronoi, stats }
}

/// Compute spherical Voronoi diagram using the GPU-style algorithm.
///
/// Uses adaptive k-NN (12→24→48→full) with early termination.
///
/// Timing output is controlled by the `timing` feature flag:
/// ```bash
/// cargo run --release --features timing
/// ```
pub fn compute_voronoi_gpu_style(points: &[Vec3]) -> crate::geometry::SphericalVoronoi {
    let termination = TerminationConfig::default();
    compute_voronoi_gpu_style_core(points, termination, false, false).voronoi
}

/// Compute spherical Voronoi with statistics.
pub fn compute_voronoi_gpu_style_with_stats(
    points: &[Vec3],
) -> (crate::geometry::SphericalVoronoi, VoronoiStats) {
    let termination = TerminationConfig::default();
    let result = compute_voronoi_gpu_style_core(points, termination, true, false);
    (result.voronoi, result.stats.expect("stats requested"))
}

/// Compute spherical Voronoi with custom termination config (for benchmarks).
pub fn compute_voronoi_gpu_style_with_termination(
    points: &[Vec3],
    termination: TerminationConfig,
) -> crate::geometry::SphericalVoronoi {
    compute_voronoi_gpu_style_core(points, termination, false, false).voronoi
}

/// Compute spherical Voronoi WITHOUT preprocessing (merge close points).
/// For benchmarking only - assumes points are already well-spaced.
pub fn compute_voronoi_gpu_style_no_preprocess(points: &[Vec3]) -> crate::geometry::SphericalVoronoi {
    let termination = TerminationConfig::default();
    compute_voronoi_gpu_style_core(points, termination, false, true).voronoi
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
    iterations: usize,
) -> (BenchmarkResult, BenchmarkResult) {
    use crate::geometry::{random_sphere_points, SphericalVoronoi};
    use std::time::Instant;

    let points = random_sphere_points(num_points);

    // Warm up
    let _ = SphericalVoronoi::compute(&points);
    let _ = compute_voronoi_gpu_style(&points);

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
        gpu_result = Some(compute_voronoi_gpu_style(&points));
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
            method: "GPU-style".to_string(),
            num_points,
            time_ms: gpu_time,
            cells_with_vertices: gpu_cells_with_verts,
            avg_vertices_per_cell: gpu_avg_verts,
        },
    )
}
