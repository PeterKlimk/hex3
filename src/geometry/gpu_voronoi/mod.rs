//! GPU-friendly spherical Voronoi computation via half-space (great circle) clipping.
//!
//! This module implements a "meshless" approach where each Voronoi cell is computed
//! independently from its k nearest neighbors. This enables massive parallelism on GPU.

mod cell_builder;
mod constants;
mod knn;
mod live_dedup;
mod timing;

#[cfg(test)]
mod tests;

use glam::Vec3;
use kiddo::ImmutableKdTree;

// Re-exports
pub use cell_builder::{
    geodesic_distance, order_vertices_ccw_indices, F64CellBuilder, GreatCircle, VertexData,
    VertexKey, VertexList, MAX_PLANES, MAX_VERTICES,
};
pub use constants::{
    support_cluster_drift_dot, MIN_BISECTOR_DISTANCE, SUPPORT_CERT_MARGIN_ABS,
    SUPPORT_CLUSTER_RADIUS_ANGLE, SUPPORT_EPS_ABS, SUPPORT_VERTEX_ANGLE_EPS, VERTEX_WELD_FRACTION,
};
pub use knn::CubeMapGridKnn;

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

#[derive(Debug, Clone, Copy)]
pub struct TerminationConfig {
    /// Enables adaptive k-NN + early termination checks.
    ///
    /// When disabled, the builder will still run the initial k-NN pass but will not
    /// attempt to terminate early; the k-NN schedule still runs to ensure
    /// correctness (so this should generally remain enabled for performance).
    pub enabled: bool,
    pub check_start: usize,
    pub check_step: usize,
}

// Keep the k-NN schedule and the default termination cadence in one place.
pub(super) const KNN_RESUME_KS: [usize; 1] = [20];
pub(super) const KNN_RESTART_MAX: usize = 48;
pub(super) const KNN_RESTART_KS: [usize; 2] = [32, KNN_RESTART_MAX];

// Default termination cadence:
// - start near the end of the initial k pass
// - then check roughly twice per initial-k window
const DEFAULT_TERMINATION_CHECK_START: usize = 8;
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
pub fn merge_close_points(points: &[Vec3], threshold: f32) -> MergeResult {
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
    skip_preprocess: bool,
) -> crate::geometry::SphericalVoronoi {
    use timing::{Timer, TimingBuilder};

    let mut tb = TimingBuilder::new();

    // Preprocessing: merge close points (not timed - separate from Voronoi computation)
    let (effective_points, merge_result) = if skip_preprocess {
        (points.to_vec(), None)
    } else {
        let result = merge_close_points(points, MIN_BISECTOR_DISTANCE);
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

    // Build cells using sharded live dedup
    let t = Timer::start();
    let sharded =
        live_dedup::build_cells_sharded_live_dedup(&effective_points, &knn, termination);
    tb.set_cell_construction(t.elapsed(), sharded.cell_sub.clone().into_sub_phases());

    let t = Timer::start();
    let (all_vertices, eff_cells, eff_cell_indices, dedup_sub) =
        live_dedup::assemble_sharded_live_dedup(sharded);
    tb.set_dedup(t.elapsed(), dedup_sub);

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

    voronoi
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
    compute_voronoi_gpu_style_core(points, termination, false)
}

/// Compute spherical Voronoi with custom termination config (for benchmarks).
pub fn compute_voronoi_gpu_style_with_termination(
    points: &[Vec3],
    termination: TerminationConfig,
) -> crate::geometry::SphericalVoronoi {
    compute_voronoi_gpu_style_core(points, termination, false)
}

/// Compute spherical Voronoi WITHOUT preprocessing (merge close points).
/// For benchmarking only - assumes points are already well-spaced.
pub fn compute_voronoi_gpu_style_no_preprocess(
    points: &[Vec3],
) -> crate::geometry::SphericalVoronoi {
    let termination = TerminationConfig::default();
    compute_voronoi_gpu_style_core(points, termination, true)
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
