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
    geodesic_distance, order_vertices_ccw_indices,
};
pub use dedup::dedup_vertices_hash;
pub use dedup::dedup_vertices_hash_with_degeneracy_edges;
pub use knn::{CubeMapGridKnn, KnnProvider};

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
}

#[derive(Debug, Clone, Copy)]
pub struct TerminationConfig {
    pub enabled: bool,
    pub check_start: usize,
    pub check_step: usize,
}

impl TerminationConfig {
    #[inline]
    fn should_check(&self, neighbors_processed: usize) -> bool {
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

#[derive(Debug, Default, Clone, Copy)]
struct CellBuildStats {
    total_neighbors_processed: usize,
    terminated_cells: usize,
}

fn build_cells_data<K: KnnProvider>(
    points: &[Vec3],
    knn: &K,
    k: usize,
    termination: TerminationConfig,
    collect_stats: bool,
) -> (
    Vec<Vec<([usize; 3], Vec3)>>,
    Option<CellBuildStats>,
    Vec<([usize; 3], [usize; 3])>,
) {
    if !collect_stats {
        let (cells_data, degenerate_edges) =
            build_cells_data_incremental(points, knn, k, termination);
        return (cells_data, None, degenerate_edges);
    }

    // Stats collection path - uses same IncrementalCellBuilder but tracks stats
    use rayon::prelude::*;

    #[derive(Debug)]
    struct CellResult {
        verts: Vec<([usize; 3], Vec3)>,
        degenerate: Vec<([usize; 3], [usize; 3])>,
        neighbors_processed: usize,
        terminated: bool,
    }

    let results: Vec<CellResult> = (0..points.len())
        .into_par_iter()
        .map_init(
            || (
                knn.make_scratch(),
                Vec::with_capacity(k),
                IncrementalCellBuilder::new(0, Vec3::ZERO),
            ),
            |(scratch, neighbors, builder), i| {
                neighbors.clear();
                knn.knn_into(points[i], i, k, scratch, neighbors);

                builder.reset(i, points[i]);
                let mut neighbors_processed = 0usize;
                let mut terminated = false;

                for (count, &neighbor_idx) in neighbors.iter().enumerate() {
                    let neighbor = points[neighbor_idx];
                    builder.clip(neighbor_idx, neighbor);
                    neighbors_processed = count + 1;

                    if termination.should_check(neighbors_processed) && builder.vertex_count() >= 3 {
                        let neighbor_cos = points[i].dot(neighbor).clamp(-1.0, 1.0);
                        if builder.can_terminate(neighbor_cos) {
                            terminated = true;
                            break;
                        }
                    }
                }

                CellResult {
                    verts: builder.get_vertices_with_keys(),
                    degenerate: builder.degenerate_edges().to_vec(),
                    neighbors_processed,
                    terminated,
                }
            },
        )
        .collect();

    let mut stats = CellBuildStats::default();
    let mut cells_data: Vec<Vec<([usize; 3], Vec3)>> = Vec::with_capacity(results.len());
    let mut all_degenerate: Vec<([usize; 3], [usize; 3])> = Vec::new();
    for r in results {
        stats.total_neighbors_processed += r.neighbors_processed;
        stats.terminated_cells += r.terminated as usize;
        cells_data.push(r.verts);
        all_degenerate.extend(r.degenerate);
    }

    (cells_data, Some(stats), all_degenerate)
}

/// Build cells using the incremental O(n) polygon clipping algorithm.
/// Returns (cells_data, degenerate_edges).
/// If degenerate_edges is empty, no degeneracy unification pass is needed.
pub fn build_cells_data_incremental(
    points: &[Vec3],
    knn: &impl KnnProvider,
    k: usize,
    termination: TerminationConfig,
) -> (
    Vec<Vec<([usize; 3], Vec3)>>,
    Vec<([usize; 3], [usize; 3])>,
) {
    use rayon::prelude::*;

    let results: Vec<_> = (0..points.len())
        .into_par_iter()
        .map_init(
            || (
                knn.make_scratch(),
                Vec::with_capacity(k),
                IncrementalCellBuilder::new(0, Vec3::ZERO),
            ),
            |(scratch, neighbors, builder), i| {
                neighbors.clear();
                knn.knn_into(points[i], i, k, scratch, neighbors);

                builder.reset(i, points[i]);

                for (count, &neighbor_idx) in neighbors.iter().enumerate() {
                    let neighbor = points[neighbor_idx];
                    builder.clip(neighbor_idx, neighbor);

                    let neighbors_processed = count + 1;
                    if termination.should_check(neighbors_processed) && builder.vertex_count() >= 3 {
                        let neighbor_cos = points[i].dot(neighbor).clamp(-1.0, 1.0);
                        if builder.can_terminate(neighbor_cos) {
                            break;
                        }
                    }
                }

                let verts = builder.get_vertices_with_keys();
                let degen = builder.degenerate_edges().to_vec();
                (verts, degen)
            },
        )
        .collect();

    // Separate cells_data and aggregate degenerate triplets
    let mut cells_data = Vec::with_capacity(results.len());
    let mut all_degenerate: Vec<([usize; 3], [usize; 3])> = Vec::new();
    for (verts, degen) in results {
        cells_data.push(verts);
        all_degenerate.extend(degen);
    }

    (cells_data, all_degenerate)
}

struct CoreResult {
    voronoi: crate::geometry::SphericalVoronoi,
    stats: Option<VoronoiStats>,
    timings_ms: PhaseTimingsMs,
}

fn compute_voronoi_gpu_style_core(
    points: &[Vec3],
    k: usize,
    termination: TerminationConfig,
    collect_stats: bool,
) -> CoreResult {
    use std::time::Instant;

    let t0 = Instant::now();
    let knn = CubeMapGridKnn::new(points);
    let t1 = Instant::now();
    let (cells_data, cell_stats, degenerate_edges) =
        build_cells_data(points, &knn, k, termination, collect_stats);
    let t2 = Instant::now();

    // IncrementalCellBuilder maintains CCW winding, so no separate ordering pass needed.
    // Unify degenerate vertices by unioning equivalent triplet keys identified during cell building.
    let (all_vertices, cells, cell_indices) = dedup::dedup_vertices_hash_with_degeneracy_edges(
        points.len(),
        cells_data,
        &degenerate_edges,
    );
    let t3 = Instant::now();

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

    let stats = cell_stats.map(|s| {
        let n = points.len().max(1) as f64;
        VoronoiStats {
            avg_neighbors_processed: s.total_neighbors_processed as f64 / n,
            termination_rate: s.terminated_cells as f64 / n,
        }
    });

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
