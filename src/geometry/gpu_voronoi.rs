//! GPU-friendly spherical Voronoi computation via half-space (great circle) clipping.
//!
//! This module implements a "meshless" approach where each Voronoi cell is computed
//! independently from its k nearest neighbors. This enables massive parallelism on GPU.
//!
//! Algorithm based on: "Parallel Voronoi Computation on GPU" adapted for spherical geometry.
//!
//! Key adaptations for spherical Voronoi:
//! - Bisector planes become great circles (planes through origin)
//! - Half-spaces become hemispheres
//! - Distance is geodesic (Euclidean works as proxy for k-NN on unit sphere)
//! - Voronoi vertices are intersections of 3 great circles, projected to sphere

use glam::Vec3;
use kiddo::{ImmutableKdTree, SquaredEuclidean};

#[inline]
fn bits_for_indices(max_index: usize) -> u32 {
    // Need at least 1 bit even when max_index == 0.
    let bits = usize::BITS - max_index.leading_zeros();
    bits.max(1)
}

#[inline]
fn pack_triplet_u128(triplet: [usize; 3], bits: u32) -> u128 {
    // Triplets are already sorted canonical keys.
    // Pack as: a | (b<<bits) | (c<<(2*bits))
    (triplet[0] as u128) | ((triplet[1] as u128) << bits) | ((triplet[2] as u128) << (2 * bits))
}

/// Maximum number of neighbors to consider per cell.
/// For well-distributed points (Lloyd relaxed), k=20 is sufficient.
/// We use 24 for a small safety margin. Higher k is only needed for
/// poorly-distributed (random) point sets.
pub const DEFAULT_K: usize = 24;

/// Maximum number of planes (great circle boundaries) per cell.
pub const MAX_PLANES: usize = 24;

/// Maximum number of vertices (plane triplet intersections) per cell.
pub const MAX_VERTICES: usize = 32;

/// A great circle on the unit sphere, represented by its normal vector.
/// The great circle is the set of points P where N·P = 0.
/// The "positive" hemisphere is where N·P > 0.
#[derive(Debug, Clone, Copy)]
pub struct GreatCircle {
    /// Normal vector (unit length, defines the plane through origin)
    pub normal: Vec3,
}

impl GreatCircle {
    /// Create the bisector great circle between two points on the unit sphere.
    /// Points on the returned circle are equidistant from both points.
    /// The positive hemisphere contains point `a`.
    #[inline]
    pub fn bisector(a: Vec3, b: Vec3) -> Self {
        // The bisector plane has normal in direction (a - b), normalized
        // The positive side contains `a`
        let normal = (a - b).normalize();
        GreatCircle { normal }
    }

    /// Check if a point is in the positive hemisphere (same side as the cell generator).
    #[inline]
    pub fn contains(&self, point: Vec3) -> bool {
        self.normal.dot(point) >= -1e-10 // Small epsilon for numerical stability
    }

    /// Signed distance from point to the great circle plane.
    /// Positive = same side as generator, negative = opposite side.
    #[inline]
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        self.normal.dot(point)
    }
}

/// A spherical Voronoi cell builder using the intersection approach.
///
/// For spherical Voronoi, we use a simpler algorithm:
/// 1. Collect bisector planes for all neighbors
/// 2. Find all candidate vertices (intersections of plane pairs on sphere)
/// 3. Filter to vertices inside all half-spaces
/// 4. Order vertices to form the cell polygon
#[derive(Debug, Clone)]
pub struct CellBuilder {
    /// The index of this cell's generator
    pub generator_idx: usize,
    /// The generator point for this cell
    pub generator: Vec3,
    /// Great circle boundaries (bisectors with neighbors)
    pub planes: Vec<GreatCircle>,
    /// The neighbor index for each plane (for combinatorial vertex identification)
    pub neighbor_indices: Vec<usize>,
    /// The neighbor point for each plane (for canonical vertex computation)
    pub neighbors: Vec<Vec3>,
    /// Current bounding radius (for security radius early termination)
    pub bounding_radius: f32,
}

impl CellBuilder {
    /// Initialize a cell for the given generator.
    pub fn new(generator_idx: usize, generator: Vec3) -> Self {
        CellBuilder {
            generator_idx,
            generator,
            planes: Vec::with_capacity(MAX_PLANES),
            neighbor_indices: Vec::with_capacity(MAX_PLANES),
            neighbors: Vec::with_capacity(MAX_PLANES),
            bounding_radius: std::f32::consts::PI,
        }
    }

    /// Add a neighbor's bisector plane.
    pub fn clip(&mut self, neighbor_idx: usize, neighbor: Vec3) -> bool {
        let bisector = GreatCircle::bisector(self.generator, neighbor);
        self.planes.push(bisector);
        self.neighbor_indices.push(neighbor_idx);
        self.neighbors.push(neighbor);
        true
    }

    /// Check if we can early-terminate based on security radius.
    ///
    /// `next_neighbor_cos` is the cosine of the angular distance from the generator to the next
    /// neighbor (i.e. `generator.dot(neighbor)`), avoiding expensive `acos` in the hot loop.
    pub fn can_terminate(&self, next_neighbor_cos: f32) -> bool {
        // Need at least 3 planes to have any vertices
        if self.planes.len() < 3 {
            return false;
        }

        // Rebuild vertices (keyed) and compute θ_max from them.
        // This avoids the extra Vec allocation in `get_vertices()`.
        let keyed = self.get_vertices_with_keys();
        if keyed.len() < 3 {
            return false;
        }

        // Let θ_max be the maximum angular distance from generator to any current vertex.
        // Since cos is monotone decreasing on [0, π], θ_max corresponds to the minimum dot.
        let min_cos = keyed
            .iter()
            .map(|(_, v)| self.generator.dot(*v).clamp(-1.0, 1.0))
            .fold(1.0f32, |a, b| a.min(b));

        // If the cell spans beyond a hemisphere relative to the generator, avoid early termination.
        // (Also avoids cases where 2*θ_max would exceed π and the cosine comparison loses monotonicity.)
        if min_cos <= 0.0 {
            return false;
        }

        // Condition: θ_next > 2*θ_max
        // Equivalent: cos(θ_next) < cos(2*θ_max) = 2*cos(θ_max)^2 - 1
        let cos_2max = 2.0 * min_cos * min_cos - 1.0;
        next_neighbor_cos < cos_2max - 1e-6
    }

    fn termination_vertices_with_stats(
        &self,
        next_neighbor_cos: f32,
    ) -> Option<(Vec<([usize; 3], Vec3)>, usize)> {
        if self.planes.len() < 3 {
            return None;
        }

        let (keyed, secondary_hits) = self.get_vertices_with_keys_and_stats();
        if keyed.len() < 3 {
            return None;
        }

        let min_cos = keyed
            .iter()
            .map(|(_, v)| self.generator.dot(*v).clamp(-1.0, 1.0))
            .fold(1.0f32, |a, b| a.min(b));

        if min_cos <= 0.0 {
            return None;
        }

        let cos_2max = 2.0 * min_cos * min_cos - 1.0;
        if next_neighbor_cos < cos_2max - 1e-6 {
            Some((keyed, secondary_hits))
        } else {
            None
        }
    }

    /// Compute both antipodal intersection points of two bisector planes on the unit sphere.
    ///
    /// The intersection of two great-circle planes through the origin is a line through the origin,
    /// which meets the unit sphere at two antipodal points. Which of the two is a Voronoi vertex
    /// depends on the other half-space constraints, so callers must consider both.
    ///
    /// Returns the candidates ordered by descending dot with the generator (i.e., closer first).
    fn plane_intersection_candidates(&self, i: usize, j: usize) -> Option<(Vec3, Vec3)> {
        let n_i = self.planes[i].normal;
        let n_j = self.planes[j].normal;

        // Cross product gives direction of intersection line
        let cross = n_i.cross(n_j);
        let len = cross.length();
        if len < 1e-10 {
            return None; // Parallel planes
        }

        // Two antipodal points on the sphere
        let v1 = cross / len;
        let v2 = -v1;

        // Order candidates so we usually test the likely one first.
        if v1.dot(self.generator) >= v2.dot(self.generator) {
            Some((v1, v2))
        } else {
            Some((v2, v1))
        }
    }

    /// Check if a point is inside all half-spaces (on generator's side of all bisectors).
    #[allow(dead_code)]
    fn is_inside_all(&self, point: Vec3) -> bool {
        self.planes
            .iter()
            .all(|plane| plane.signed_distance(point) >= -1e-6)
    }

    /// Extract the final vertices with their combinatorial keys, plus diagnostic counters.
    ///
    /// Each vertex is identified by the sorted triplet of generator indices that define it.
    /// Returns (vertices, secondary_candidate_hits) where `secondary_candidate_hits` counts
    /// how often the nearer antipodal intersection failed but the opposite one succeeded.
    pub fn get_vertices_with_keys_and_stats(&self) -> (Vec<([usize; 3], Vec3)>, usize) {
        let n = self.planes.len();
        if n < 3 {
            return (Vec::new(), 0);
        }

        // Fast sort for exactly 3 elements
        #[inline]
        fn sort3(mut a: [usize; 3]) -> [usize; 3] {
            if a[0] > a[1] {
                a.swap(0, 1);
            }
            if a[1] > a[2] {
                a.swap(1, 2);
            }
            if a[0] > a[1] {
                a.swap(0, 1);
            }
            a
        }

        // Each plane-pair (i, j) yields exactly one triplet key (generator, neighbor[i], neighbor[j]),
        // and k-NN neighbor indices are unique, so we don't need per-cell hash-based de-duplication.
        let mut vertices = Vec::with_capacity(n); // Expect ~n vertices
        let mut secondary_candidate_hits = 0usize;

        // Try all pairs of planes
        for i in 0..n {
            for j in (i + 1)..n {
                if let Some((v_primary, v_secondary)) = self.plane_intersection_candidates(i, j) {
                    // Exactly one of the two antipodal candidates can be inside the intersection
                    // of hemispheres (except for fully-degenerate cases). Try the closer one first
                    // for performance, but fall back to the other to avoid missing valid vertices.
                    let mut chosen: Option<Vec3> = None;
                    for (idx, v) in [v_primary, v_secondary].into_iter().enumerate() {
                        let inside_all = self.planes.iter().enumerate().all(|(k, plane)| {
                            k == i || k == j || plane.signed_distance(v) >= -1e-6
                        });
                        if inside_all {
                            chosen = Some(v);
                            if idx == 1 {
                                secondary_candidate_hits += 1;
                            }
                            break;
                        }
                    }

                    if let Some(v) = chosen {
                        // The vertex is defined by generators: self, neighbor[i], neighbor[j]
                        let triplet = sort3([
                            self.generator_idx,
                            self.neighbor_indices[i],
                            self.neighbor_indices[j],
                        ]);
                        vertices.push((triplet, v));
                    }
                }
            }
        }

        (vertices, secondary_candidate_hits)
    }

    /// Extract the final vertices with their combinatorial keys.
    /// Each vertex is identified by the sorted triplet of generator indices that define it.
    /// Returns (canonical_triplet_key, vertex_position) pairs.
    pub fn get_vertices_with_keys(&self) -> Vec<([usize; 3], Vec3)> {
        self.get_vertices_with_keys_and_stats().0
    }

    /// Extract just the vertex positions (for compatibility with can_terminate).
    pub fn get_vertices(&self) -> Vec<Vec3> {
        self.get_vertices_with_keys()
            .into_iter()
            .map(|(_, v)| v)
            .collect()
    }
}

/// Compute geodesic distance between two points on unit sphere.
#[inline]
pub fn geodesic_distance(a: Vec3, b: Vec3) -> f32 {
    let dot = a.dot(b).clamp(-1.0, 1.0);
    dot.acos()
}

/// Order vertices counter-clockwise around a generator when viewed from outside.
/// Returns the indices into the original array in CCW order.
fn order_vertices_ccw_indices(generator: Vec3, vertices: &[Vec3]) -> Vec<usize> {
    if vertices.len() <= 2 {
        return (0..vertices.len()).collect();
    }

    let up = if generator.y.abs() < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let tangent_x = generator.cross(up).normalize();
    let tangent_y = generator.cross(tangent_x).normalize();

    let mut indexed: Vec<(usize, f32)> = vertices
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let to_point = v - generator * generator.dot(v);
            let x = to_point.dot(tangent_x);
            let y = to_point.dot(tangent_y);
            (i, y.atan2(x))
        })
        .collect();

    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    indexed.into_iter().map(|(i, _)| i).collect()
}

/// Build a k-d tree from sphere points for efficient k-NN queries.
/// Returns the tree and the entries array (needed for index lookup).
pub fn build_kdtree(points: &[Vec3]) -> (ImmutableKdTree<f32, 3>, Vec<[f32; 3]>) {
    let entries: Vec<[f32; 3]> = points.iter().map(|p| [p.x, p.y, p.z]).collect();
    let tree = ImmutableKdTree::new_from_slice(&entries);
    (tree, entries)
}

/// Find k nearest neighbors using k-d tree.
/// Returns indices sorted by distance (excluding the query point itself).
pub fn find_k_nearest(
    tree: &ImmutableKdTree<f32, 3>,
    _entries: &[[f32; 3]],
    query: Vec3,
    query_idx: usize,
    k: usize,
) -> Vec<usize> {
    // Query k+1 to account for the point itself potentially being in results
    let results = tree.nearest_n::<SquaredEuclidean>(&[query.x, query.y, query.z], k + 1);

    results
        .into_iter()
        .map(|n| n.item as usize)
        .filter(|&idx| idx != query_idx)
        .take(k)
        .collect()
}

/// Compute spherical Voronoi diagram using the GPU-style algorithm (CPU reference).
pub fn compute_voronoi_gpu_style(points: &[Vec3], k: usize) -> super::SphericalVoronoi {
    compute_voronoi_gpu_style_timed(points, k, false)
}

/// Statistics about the Voronoi computation.
#[derive(Debug, Clone)]
pub struct VoronoiStats {
    /// Average number of neighbors processed per cell before termination.
    pub avg_neighbors_processed: f64,
    /// Fraction of cells that terminated early (0.0 to 1.0).
    pub termination_rate: f64,
    /// Average number of times per cell that the "nearer" antipodal candidate failed
    /// but the opposite candidate satisfied all half-space constraints.
    pub avg_secondary_candidate_hits: f64,
}

#[derive(Debug, Clone, Copy)]
struct TerminationConfig {
    enabled: bool,
    check_start: usize,
    check_step: usize,
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
    ccw_order: f64,
    dedup: f64,
    assemble: f64,
}

#[derive(Debug, Default, Clone, Copy)]
struct CellBuildStats {
    total_neighbors_processed: usize,
    terminated_cells: usize,
    secondary_candidate_hits: usize,
}

fn build_cells_data(
    points: &[Vec3],
    knn: &[Vec<usize>],
    termination: TerminationConfig,
    collect_stats: bool,
) -> (Vec<Vec<([usize; 3], Vec3)>>, Option<CellBuildStats>) {
    use rayon::prelude::*;

    if !collect_stats {
        let cells_data: Vec<Vec<([usize; 3], Vec3)>> = (0..points.len())
            .into_par_iter()
            .map(|i| {
                let mut builder = CellBuilder::new(i, points[i]);
                let mut terminated_keyed: Option<Vec<([usize; 3], Vec3)>> = None;

                for (count, &neighbor_idx) in knn[i].iter().enumerate() {
                    let neighbor = points[neighbor_idx];
                    builder.clip(neighbor_idx, neighbor);

                    let neighbors_processed = count + 1;
                    if termination.should_check(neighbors_processed) {
                        let neighbor_cos = points[i].dot(neighbor).clamp(-1.0, 1.0);
                        if let Some((keyed, _hits)) =
                            builder.termination_vertices_with_stats(neighbor_cos)
                        {
                            terminated_keyed = Some(keyed);
                            break;
                        }
                    }
                }

                terminated_keyed.unwrap_or_else(|| builder.get_vertices_with_keys())
            })
            .collect();

        return (cells_data, None);
    }

    #[derive(Debug)]
    struct CellResult {
        verts: Vec<([usize; 3], Vec3)>,
        neighbors_processed: usize,
        terminated: bool,
        secondary_hits: usize,
    }

    let results: Vec<CellResult> = (0..points.len())
        .into_par_iter()
        .map(|i| {
            let mut builder = CellBuilder::new(i, points[i]);
            let mut neighbors_processed = 0usize;
            let mut terminated = false;
            let mut terminated_keyed: Option<Vec<([usize; 3], Vec3)>> = None;
            let mut terminated_secondary_hits: usize = 0;

            for (count, &neighbor_idx) in knn[i].iter().enumerate() {
                let neighbor = points[neighbor_idx];
                builder.clip(neighbor_idx, neighbor);
                neighbors_processed = count + 1;

                if termination.should_check(neighbors_processed) {
                    let neighbor_cos = points[i].dot(neighbor).clamp(-1.0, 1.0);
                    if let Some((keyed, hits)) =
                        builder.termination_vertices_with_stats(neighbor_cos)
                    {
                        terminated = true;
                        terminated_keyed = Some(keyed);
                        terminated_secondary_hits = hits;
                        break;
                    }
                }
            }

            if let Some(keyed) = terminated_keyed {
                CellResult {
                    verts: keyed,
                    neighbors_processed,
                    terminated,
                    secondary_hits: terminated_secondary_hits,
                }
            } else {
                let (verts, hits) = builder.get_vertices_with_keys_and_stats();
                CellResult {
                    verts,
                    neighbors_processed,
                    terminated,
                    secondary_hits: hits,
                }
            }
        })
        .collect();

    let mut stats = CellBuildStats::default();
    let mut cells_data: Vec<Vec<([usize; 3], Vec3)>> = Vec::with_capacity(results.len());
    for r in results {
        stats.total_neighbors_processed += r.neighbors_processed;
        stats.terminated_cells += r.terminated as usize;
        stats.secondary_candidate_hits += r.secondary_hits;
        cells_data.push(r.verts);
    }

    (cells_data, Some(stats))
}

fn order_cells_ccw(
    points: &[Vec3],
    cells_data: Vec<Vec<([usize; 3], Vec3)>>,
) -> Vec<Vec<([usize; 3], Vec3)>> {
    use rayon::prelude::*;

    cells_data
        .into_par_iter()
        .enumerate()
        .map(|(i, keyed_verts)| {
            let verts: Vec<Vec3> = keyed_verts.iter().map(|(_, v)| *v).collect();
            let ordered_indices = order_vertices_ccw_indices(points[i], &verts);
            ordered_indices
                .into_iter()
                .map(|idx| keyed_verts[idx])
                .collect()
        })
        .collect()
}

fn dedup_vertices_hash(
    num_points: usize,
    ordered_cells: Vec<Vec<([usize; 3], Vec3)>>,
) -> (Vec<Vec3>, Vec<super::VoronoiCell>, Vec<usize>) {
    use rustc_hash::FxHashMap;

    let total_indices: usize = ordered_cells.iter().map(|v| v.len()).sum();

    let mut cell_starts: Vec<usize> = Vec::with_capacity(num_points + 1);
    cell_starts.push(0);
    for c in &ordered_cells {
        cell_starts.push(cell_starts.last().copied().unwrap() + c.len());
    }

    let bits = bits_for_indices(num_points.saturating_sub(1));
    let expected_vertices = num_points * 2;
    let mut all_vertices: Vec<Vec3> = Vec::with_capacity(expected_vertices);
    let mut triplet_to_index: FxHashMap<u128, usize> =
        FxHashMap::with_capacity_and_hasher(expected_vertices, Default::default());

    let mut cells: Vec<super::VoronoiCell> = Vec::with_capacity(num_points);
    for generator_index in 0..num_points {
        let vertex_start = cell_starts[generator_index];
        let vertex_count = cell_starts[generator_index + 1] - vertex_start;
        cells.push(super::VoronoiCell::new(
            generator_index,
            vertex_start,
            vertex_count,
        ));
    }

    let mut cell_indices: Vec<usize> = vec![0usize; total_indices];
    for (cell_idx, ordered_keyed_verts) in ordered_cells.into_iter().enumerate() {
        let base = cell_starts[cell_idx];
        for (local_i, (triplet, pos)) in ordered_keyed_verts.into_iter().enumerate() {
            let key = pack_triplet_u128(triplet, bits);
            let idx = *triplet_to_index.entry(key).or_insert_with(|| {
                let idx = all_vertices.len();
                all_vertices.push(pos);
                idx
            });
            cell_indices[base + local_i] = idx;
        }
    }

    (all_vertices, cells, cell_indices)
}

struct CoreResult {
    voronoi: super::SphericalVoronoi,
    stats: Option<VoronoiStats>,
    timings_ms: PhaseTimingsMs,
}

fn compute_voronoi_gpu_style_core(
    points: &[Vec3],
    k: usize,
    termination: TerminationConfig,
    collect_stats: bool,
) -> CoreResult {
    use rayon::prelude::*;
    use std::time::Instant;

    let t0 = Instant::now();
    let (tree, entries) = build_kdtree(points);
    let t1 = Instant::now();
    let knn: Vec<Vec<usize>> = (0..points.len())
        .into_par_iter()
        .map(|i| find_k_nearest(&tree, &entries, points[i], i, k))
        .collect();
    let t2 = Instant::now();

    let (cells_data, cell_stats) = build_cells_data(points, &knn, termination, collect_stats);
    let t3 = Instant::now();

    let ordered_cells = order_cells_ccw(points, cells_data);
    let t3a = Instant::now();

    let (all_vertices, cells, cell_indices) = dedup_vertices_hash(points.len(), ordered_cells);
    let t4 = Instant::now();

    let voronoi =
        super::SphericalVoronoi::from_raw_parts(points.to_vec(), all_vertices, cells, cell_indices);
    let t5 = Instant::now();

    let total = (t5 - t0).as_secs_f64() * 1000.0;
    let kdtree_ms = (t1 - t0).as_secs_f64() * 1000.0;
    let knn_ms = (t2 - t1).as_secs_f64() * 1000.0;
    let cells_build = (t3 - t2).as_secs_f64() * 1000.0;
    let ccw_order = (t3a - t3).as_secs_f64() * 1000.0;
    let dedup = (t4 - t3a).as_secs_f64() * 1000.0;
    let assemble = (t5 - t4).as_secs_f64() * 1000.0;

    let timings_ms = PhaseTimingsMs {
        total,
        kdtree: kdtree_ms,
        knn: knn_ms,
        cell_construction: cells_build,
        ccw_order,
        dedup,
        assemble,
    };

    let stats = cell_stats.map(|s| {
        let n = points.len().max(1) as f64;
        VoronoiStats {
            avg_neighbors_processed: s.total_neighbors_processed as f64 / n,
            termination_rate: s.terminated_cells as f64 / n,
            avg_secondary_candidate_hits: s.secondary_candidate_hits as f64 / n,
        }
    });

    CoreResult {
        voronoi,
        stats,
        timings_ms,
    }
}

/// Compute spherical Voronoi with statistics about termination behavior.
pub fn compute_voronoi_gpu_style_with_stats(
    points: &[Vec3],
    k: usize,
) -> (super::SphericalVoronoi, VoronoiStats) {
    // For Lloyd-relaxed points, a check at 6 neighbors is often wasted (rarely terminates),
    // so start checks at 8 and then every 2 neighbors.
    compute_voronoi_gpu_style_with_stats_and_termination_params(points, k, true, 8, 2)
}

/// Compute spherical Voronoi with statistics, optionally disabling early termination.
pub fn compute_voronoi_gpu_style_with_stats_and_termination(
    points: &[Vec3],
    k: usize,
    enable_termination: bool,
) -> (super::SphericalVoronoi, VoronoiStats) {
    compute_voronoi_gpu_style_with_stats_and_termination_params(points, k, enable_termination, 8, 2)
}

/// Compute spherical Voronoi with statistics and a configurable termination schedule.
pub fn compute_voronoi_gpu_style_with_stats_and_termination_params(
    points: &[Vec3],
    k: usize,
    enable_termination: bool,
    termination_check_start: usize,
    termination_check_step: usize,
) -> (super::SphericalVoronoi, VoronoiStats) {
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
) -> super::SphericalVoronoi {
    compute_voronoi_gpu_style_timed_with_termination(points, k, print_timing, true)
}

/// Compute spherical Voronoi with optional timing output, optionally disabling early termination.
pub fn compute_voronoi_gpu_style_timed_with_termination(
    points: &[Vec3],
    k: usize,
    print_timing: bool,
    enable_termination: bool,
) -> super::SphericalVoronoi {
    compute_voronoi_gpu_style_timed_with_termination_params(
        points,
        k,
        print_timing,
        enable_termination,
        8,
        2,
    )
}

/// Compute spherical Voronoi with optional timing output, with configurable termination schedule.
///
/// Termination checks run after processing `termination_check_start` neighbors, then every
/// `termination_check_step` neighbors thereafter. Set `termination_check_step` to 0 to never check.
pub fn compute_voronoi_gpu_style_timed_with_termination_params(
    points: &[Vec3],
    k: usize,
    print_timing: bool,
    enable_termination: bool,
    termination_check_start: usize,
    termination_check_step: usize,
) -> super::SphericalVoronoi {
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
        println!(
            "  k-d tree build:    {:6.1} ms ({:4.1}%)",
            t.kdtree,
            t.kdtree / total * 100.0
        );
        println!(
            "  k-NN queries:      {:6.1} ms ({:4.1}%)",
            t.knn,
            t.knn / total * 100.0
        );
        println!(
            "  Cell construction: {:6.1} ms ({:4.1}%)",
            t.cell_construction,
            t.cell_construction / total * 100.0
        );
        println!(
            "  CCW ordering:      {:6.1} ms ({:4.1}%) [parallel]",
            t.ccw_order,
            t.ccw_order / total * 100.0
        );
        println!(
            "  Dedup (hash):      {:6.1} ms ({:4.1}%) [FxHashMap]",
            t.dedup,
            t.dedup / total * 100.0
        );
        println!(
            "  Assemble struct:   {:6.1} ms ({:4.1}%) [{} verts]",
            t.assemble,
            t.assemble / total * 100.0,
            result.voronoi.vertices.len()
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::random_sphere_points;

    #[test]
    fn test_great_circle_bisector() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);

        let gc = GreatCircle::bisector(a, b);

        assert!(gc.contains(a));
        assert!(!gc.contains(b));

        let mid = (a + b).normalize();
        assert!(gc.signed_distance(mid).abs() < 1e-6);
    }

    #[test]
    fn test_gpu_voronoi_basic() {
        let points = random_sphere_points(50);
        let voronoi = compute_voronoi_gpu_style(&points, DEFAULT_K);

        assert_eq!(voronoi.num_cells(), 50);
        assert_eq!(voronoi.generators.len(), 50);

        let cells_with_verts = voronoi.iter_cells().filter(|c| !c.is_empty()).count();

        println!(
            "Cells with vertices: {}/{}",
            cells_with_verts,
            voronoi.num_cells()
        );
        assert!(
            cells_with_verts > 40,
            "Too few cells with vertices: {}",
            cells_with_verts
        );
    }

    #[test]
    fn test_kdtree_knn() {
        let points = random_sphere_points(100);
        let (tree, entries) = build_kdtree(&points);

        let neighbors = find_k_nearest(&tree, &entries, points[0], 0, 5);
        assert_eq!(neighbors.len(), 5);
        assert!(!neighbors.contains(&0)); // Shouldn't include self
    }

    #[test]
    fn test_compare_with_hull_voronoi() {
        use crate::geometry::SphericalVoronoi;

        let points = random_sphere_points(100);

        let hull_voronoi = SphericalVoronoi::compute(&points);
        let gpu_voronoi = compute_voronoi_gpu_style(&points, DEFAULT_K);

        assert_eq!(hull_voronoi.num_cells(), gpu_voronoi.num_cells());

        // Compare vertex counts
        let mut matching = 0;
        let mut close = 0;
        let mut different = 0;

        for i in 0..points.len() {
            let hull_count = hull_voronoi.cell(i).len();
            let gpu_count = gpu_voronoi.cell(i).len();

            if hull_count == gpu_count {
                matching += 1;
            } else if (hull_count as i32 - gpu_count as i32).abs() <= 1 {
                close += 1;
            } else {
                different += 1;
                println!(
                    "Cell {}: hull={} vs gpu={} vertices",
                    i, hull_count, gpu_count
                );
            }
        }

        println!(
            "Vertex count comparison: {} matching, {} close (±1), {} different",
            matching, close, different
        );

        // Allow some variation due to numerical precision
        assert!(
            matching + close > 90,
            "Too many cells with different vertex counts"
        );
    }

    #[test]
    #[ignore] // Run with: cargo test accuracy_audit -- --ignored --nocapture
    fn accuracy_audit() {
        use crate::geometry::SphericalVoronoi;

        println!("\n=== GPU Voronoi Accuracy Audit ===\n");

        for &n in &[100, 1000, 10000] {
            println!("--- n = {} ---", n);
            let points = random_sphere_points(n);

            let hull = SphericalVoronoi::compute(&points);
            let gpu = compute_voronoi_gpu_style(&points, DEFAULT_K);

            // Detailed comparison
            let mut exact_match = 0;
            let mut vertex_count_match = 0;
            let mut missing_vertices = 0;
            let mut extra_vertices = 0;
            let mut total_hull_verts = 0;
            let mut total_gpu_verts = 0;
            let mut worst_cells: Vec<(usize, i32, usize, usize)> = Vec::new();

            for i in 0..n {
                let hull_cell = hull.cell(i);
                let gpu_cell = gpu.cell(i);
                let hull_verts: Vec<Vec3> = hull_cell
                    .vertex_indices
                    .iter()
                    .map(|&vi| hull.vertices[vi])
                    .collect();
                let gpu_verts: Vec<Vec3> = gpu_cell
                    .vertex_indices
                    .iter()
                    .map(|&vi| gpu.vertices[vi])
                    .collect();

                total_hull_verts += hull_verts.len();
                total_gpu_verts += gpu_verts.len();

                // Check if vertices match (within tolerance)
                let mut matched_hull = vec![false; hull_verts.len()];
                let mut matched_gpu = vec![false; gpu_verts.len()];

                for (gi, gv) in gpu_verts.iter().enumerate() {
                    for (hi, hv) in hull_verts.iter().enumerate() {
                        if !matched_hull[hi] && (*gv - *hv).length() < 0.01 {
                            matched_hull[hi] = true;
                            matched_gpu[gi] = true;
                            break;
                        }
                    }
                }

                let hull_unmatched = matched_hull.iter().filter(|&&x| !x).count();
                let gpu_unmatched = matched_gpu.iter().filter(|&&x| !x).count();

                if hull_unmatched == 0 && gpu_unmatched == 0 {
                    exact_match += 1;
                }
                if hull_verts.len() == gpu_verts.len() {
                    vertex_count_match += 1;
                }
                missing_vertices += hull_unmatched;
                extra_vertices += gpu_unmatched;

                let diff = gpu_verts.len() as i32 - hull_verts.len() as i32;
                if diff.abs() > 1 {
                    worst_cells.push((i, diff, hull_verts.len(), gpu_verts.len()));
                }
            }

            println!(
                "  Exact vertex match: {}/{} cells ({:.1}%)",
                exact_match,
                n,
                exact_match as f64 / n as f64 * 100.0
            );
            println!(
                "  Vertex count match: {}/{} cells ({:.1}%)",
                vertex_count_match,
                n,
                vertex_count_match as f64 / n as f64 * 100.0
            );
            println!(
                "  Total vertices: hull={}, gpu={} (diff={})",
                total_hull_verts,
                total_gpu_verts,
                total_gpu_verts as i32 - total_hull_verts as i32
            );
            println!(
                "  Missing vertices (in hull, not gpu): {}",
                missing_vertices
            );
            println!("  Extra vertices (in gpu, not hull): {}", extra_vertices);

            if !worst_cells.is_empty() {
                println!("  Worst cells (|diff| > 1):");
                worst_cells.sort_by_key(|x| -x.1.abs());
                for (cell, diff, hull_n, gpu_n) in worst_cells.iter().take(10) {
                    let generator = points[*cell];
                    println!(
                        "    Cell {}: hull={} gpu={} (diff={:+}) @ ({:.2},{:.2},{:.2})",
                        cell, hull_n, gpu_n, diff, generator.x, generator.y, generator.z
                    );
                }
            }
            println!();
        }

        // Analyze a specific bad cell in detail
        println!("--- Detailed analysis of problematic cells ---");
        let points = random_sphere_points(1000);
        let hull = SphericalVoronoi::compute(&points);
        let gpu = compute_voronoi_gpu_style(&points, DEFAULT_K);

        // Find cell with biggest discrepancy
        let mut worst_idx = 0;
        let mut worst_diff = 0i32;
        for i in 0..1000 {
            let diff = (gpu.cell(i).len() as i32 - hull.cell(i).len() as i32).abs();
            if diff > worst_diff {
                worst_diff = diff;
                worst_idx = i;
            }
        }

        if worst_diff > 0 {
            let hull_cell = hull.cell(worst_idx);
            let gpu_cell = gpu.cell(worst_idx);
            println!("Worst cell: {}", worst_idx);
            println!("  Generator: {:?}", points[worst_idx]);
            println!("  Hull vertices ({}):", hull_cell.len());
            for &vi in hull_cell.vertex_indices {
                println!("    {:?}", hull.vertices[vi]);
            }
            println!("  GPU vertices ({}):", gpu_cell.len());
            for &vi in gpu_cell.vertex_indices {
                println!("    {:?}", gpu.vertices[vi]);
            }

            // Check how many neighbors this cell has
            let (tree, entries) = build_kdtree(&points);
            let neighbors = find_k_nearest(&tree, &entries, points[worst_idx], worst_idx, 64);
            println!("  64-NN neighbor distances:");
            for (i, &ni) in neighbors.iter().enumerate().take(20) {
                let dist = geodesic_distance(points[worst_idx], points[ni]);
                println!("    {}: idx={} dist={:.4}", i, ni, dist);
            }
        }
    }

    #[test]
    #[ignore] // Run with: cargo test accuracy_lloyd -- --ignored --nocapture
    fn accuracy_lloyd() {
        use crate::geometry::{
            fibonacci_sphere_points_with_rng, lloyd_relax_kmeans, SphericalVoronoi,
        };
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        println!("\n=== Accuracy: Lloyd-relaxed vs Random points ===\n");

        for &n in &[1000, 5000, 10000] {
            let mut rng = ChaCha8Rng::seed_from_u64(12345);

            // Random points
            let random_points = random_sphere_points(n);
            let hull_random = SphericalVoronoi::compute(&random_points);
            let gpu_random = compute_voronoi_gpu_style(&random_points, DEFAULT_K);

            let random_exact = (0..n)
                .filter(|&i| hull_random.cell(i).len() == gpu_random.cell(i).len())
                .count();

            // Lloyd-relaxed points (like actual world generation)
            let mean_spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
            let jitter = mean_spacing * 0.25;
            let mut lloyd_points = fibonacci_sphere_points_with_rng(n, jitter, &mut rng);
            lloyd_relax_kmeans(&mut lloyd_points, 2, 20, &mut rng);

            let hull_lloyd = SphericalVoronoi::compute(&lloyd_points);
            let gpu_lloyd = compute_voronoi_gpu_style(&lloyd_points, DEFAULT_K);

            let lloyd_exact = (0..n)
                .filter(|&i| hull_lloyd.cell(i).len() == gpu_lloyd.cell(i).len())
                .count();

            println!("n = {}:", n);
            println!(
                "  Random points:  {}/{} exact ({:.1}%)",
                random_exact,
                n,
                random_exact as f64 / n as f64 * 100.0
            );
            println!(
                "  Lloyd-relaxed:  {}/{} exact ({:.1}%)",
                lloyd_exact,
                n,
                lloyd_exact as f64 / n as f64 * 100.0
            );

            // For Lloyd, also check total vertex difference
            let hull_total: usize = hull_lloyd.iter_cells().map(|c| c.len()).sum();
            let gpu_total: usize = gpu_lloyd.iter_cells().map(|c| c.len()).sum();
            println!(
                "  Lloyd total verts: hull={}, gpu={} (diff={})",
                hull_total,
                gpu_total,
                gpu_total as i32 - hull_total as i32
            );

            // Check with higher k
            let gpu_k64 = compute_voronoi_gpu_style(&lloyd_points, 64);
            let k64_exact = (0..n)
                .filter(|&i| hull_lloyd.cell(i).len() == gpu_k64.cell(i).len())
                .count();
            let gpu_k64_total: usize = gpu_k64.iter_cells().map(|c| c.len()).sum();
            println!(
                "  Lloyd k=64:     {}/{} exact ({:.1}%), total verts={}",
                k64_exact,
                n,
                k64_exact as f64 / n as f64 * 100.0,
                gpu_k64_total
            );
            println!();
        }
    }

    #[test]
    #[ignore] // Run with: cargo test benchmark_methods -- --ignored --nocapture
    fn benchmark_methods() {
        println!("\n=== Voronoi Method Benchmark ===\n");

        for &n in &[100, 500, 1000, 5000, 10000] {
            let iterations = if n <= 1000 { 10 } else { 3 };
            let (hull, gpu) = benchmark_voronoi(n, DEFAULT_K, iterations);

            println!("n = {}:", n);
            println!(
                "  {:20} {:8.2} ms  (avg {:.1} verts/cell, {} cells ok)",
                hull.method, hull.time_ms, hull.avg_vertices_per_cell, hull.cells_with_vertices
            );
            println!(
                "  {:20} {:8.2} ms  (avg {:.1} verts/cell, {} cells ok)",
                gpu.method, gpu.time_ms, gpu.avg_vertices_per_cell, gpu.cells_with_vertices
            );
            println!("  Speedup: {:.2}x", hull.time_ms / gpu.time_ms);
            println!();
        }
    }

    #[test]
    #[ignore] // Run with: cargo test benchmark_extended -- --ignored --nocapture
    fn benchmark_extended() {
        println!("\n=== Extended Voronoi Benchmark ===\n");

        // Test larger scales
        println!("--- Large Scale ---");
        for &n in &[20000, 50000, 100000] {
            let iterations = 1;
            let (hull, gpu) = benchmark_voronoi(n, DEFAULT_K, iterations);

            println!("n = {}:", n);
            println!(
                "  {:20} {:8.2} ms  (avg {:.1} verts/cell)",
                hull.method, hull.time_ms, hull.avg_vertices_per_cell
            );
            println!(
                "  {:20} {:8.2} ms  (avg {:.1} verts/cell)",
                gpu.method, gpu.time_ms, gpu.avg_vertices_per_cell
            );
            println!("  Speedup: {:.2}x", hull.time_ms / gpu.time_ms);
            println!();
        }

        // Test effect of k parameter
        println!("--- Effect of k (n=10000) ---");
        for &k in &[16, 24, 32, 48, 64] {
            let (hull, gpu) = benchmark_voronoi(10000, k, 3);
            println!(
                "k = {:2}: GPU-style {:6.2} ms (avg {:.1} verts), hull {:6.2} ms, speedup {:.2}x",
                k,
                gpu.time_ms,
                gpu.avg_vertices_per_cell,
                hull.time_ms,
                hull.time_ms / gpu.time_ms
            );
        }
    }

    #[test]
    #[ignore] // Run with: cargo test profile_80k -- --ignored --nocapture
    fn profile_80k() {
        use crate::geometry::{fibonacci_sphere_points_with_rng, lloyd_relax_kmeans};
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        println!("\n=== GPU-style Voronoi Profiling (80k Lloyd-relaxed) ===\n");

        let n = 80000;
        let mut rng = ChaCha8Rng::seed_from_u64(12345);

        let mean_spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
        let jitter = mean_spacing * 0.25;
        let mut points = fibonacci_sphere_points_with_rng(n, jitter, &mut rng);
        lloyd_relax_kmeans(&mut points, 2, 20, &mut rng);

        // Run with timing
        let _ = compute_voronoi_gpu_style_timed(&points, DEFAULT_K, true);
    }

    #[test]
    #[ignore] // Run with: cargo test benchmark_lloyd_80k -- --ignored --nocapture
    fn benchmark_lloyd_80k() {
        use crate::geometry::{
            fibonacci_sphere_points_with_rng, lloyd_relax_kmeans, SphericalVoronoi,
        };
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use std::time::Instant;

        println!("\n=== Lloyd-Relaxed 80k Benchmark (Production Use Case) ===\n");

        let n = 80000;
        let mut rng = ChaCha8Rng::seed_from_u64(12345);

        // Generate Lloyd-relaxed points (same as world generation)
        let mean_spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
        let jitter = mean_spacing * 0.25;
        let mut points = fibonacci_sphere_points_with_rng(n, jitter, &mut rng);
        lloyd_relax_kmeans(&mut points, 2, 20, &mut rng);

        println!("Generated {} Lloyd-relaxed points\n", n);

        // Benchmark convex hull method
        let start = Instant::now();
        let hull_voronoi = SphericalVoronoi::compute(&points);
        let hull_time = start.elapsed().as_secs_f64() * 1000.0;
        println!("Convex Hull: {:.1} ms", hull_time);

        // Benchmark GPU-style with varying k
        println!("\nGPU-style with varying k:");
        for &k in &[16, 20, 24, 28, 32, 40, 48] {
            let start = Instant::now();
            let gpu_voronoi = compute_voronoi_gpu_style(&points, k);
            let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

            // Check accuracy (cells with correct vertex count)
            let correct = (0..n)
                .filter(|&i| hull_voronoi.cell(i).len() == gpu_voronoi.cell(i).len())
                .count();

            // Check for problem cells
            let bad_cells = gpu_voronoi.iter_cells().filter(|c| c.len() < 3).count();

            let speedup = hull_time / gpu_time;
            println!(
                "  k={:2}: {:6.1} ms ({:.2}x) | accuracy: {:.1}% | bad cells: {}",
                k,
                gpu_time,
                speedup,
                correct as f64 / n as f64 * 100.0,
                bad_cells
            );
        }
    }
}
