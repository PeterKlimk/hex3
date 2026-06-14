//! Spherical tessellation - Voronoi cells and adjacency graph.

use std::collections::HashMap;
use std::ops::Index;

use glam::Vec3;
use rand::Rng;

use crate::geometry::{fibonacci_sphere_points_with_rng, lloyd_relax_kmeans, SphericalVoronoi};

/// A spherical tessellation with Voronoi cells and cell adjacency.
pub struct Tessellation {
    /// The underlying Voronoi diagram.
    pub voronoi: SphericalVoronoi,

    /// Immutable cell adjacency graph.
    pub adjacency: CellAdjacency,
}

/// Compact immutable cell adjacency graph.
///
/// Neighbor lists are stored in one flat buffer with per-cell offsets. This
/// matches the topology shape better than `Vec<Vec<_>>`: Voronoi valence is
/// small, the graph is immutable after tessellation, and hot loops mostly
/// iterate neighbors.
pub struct CellAdjacency {
    offsets: Vec<usize>,
    neighbors: Vec<usize>,
}

impl CellAdjacency {
    pub fn from_vecs(adjacency: Vec<Vec<usize>>) -> Self {
        let mut offsets = Vec::with_capacity(adjacency.len() + 1);
        let total_neighbors = adjacency.iter().map(Vec::len).sum();
        let mut neighbors = Vec::with_capacity(total_neighbors);
        offsets.push(0);
        for cell_neighbors in adjacency {
            neighbors.extend(cell_neighbors);
            offsets.push(neighbors.len());
        }
        Self { offsets, neighbors }
    }

    pub fn from_slices<I, S>(neighbor_lists: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<[usize]>,
    {
        let iter = neighbor_lists.into_iter();
        let (lower_bound, _) = iter.size_hint();
        let mut offsets = Vec::with_capacity(lower_bound + 1);
        let mut neighbors = Vec::new();
        offsets.push(0);
        for cell_neighbors in iter {
            neighbors.extend_from_slice(cell_neighbors.as_ref());
            offsets.push(neighbors.len());
        }
        Self { offsets, neighbors }
    }

    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn neighbors(&self, cell_idx: usize) -> &[usize] {
        let start = self.offsets[cell_idx];
        let end = self.offsets[cell_idx + 1];
        &self.neighbors[start..end]
    }

    pub fn iter(&self) -> CellAdjacencyIter<'_> {
        CellAdjacencyIter {
            adjacency: self,
            next_cell: 0,
        }
    }

    pub fn total_neighbor_entries(&self) -> usize {
        self.neighbors.len()
    }
}

impl Index<usize> for CellAdjacency {
    type Output = [usize];

    fn index(&self, index: usize) -> &Self::Output {
        self.neighbors(index)
    }
}

pub struct CellAdjacencyIter<'a> {
    adjacency: &'a CellAdjacency,
    next_cell: usize,
}

impl<'a> Iterator for CellAdjacencyIter<'a> {
    type Item = &'a [usize];

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_cell >= self.adjacency.len() {
            return None;
        }
        let cell = self.next_cell;
        self.next_cell += 1;
        Some(self.adjacency.neighbors(cell))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.adjacency.len().saturating_sub(self.next_cell);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for CellAdjacencyIter<'_> {
    fn len(&self) -> usize {
        self.adjacency.len().saturating_sub(self.next_cell)
    }
}

/// Default jitter amount for Fibonacci lattice (as fraction of mean cell spacing).
/// Higher values = more organic but less uniform. Combined with Lloyd relaxation
/// to recover uniformity while preserving organic appearance.
const FIBONACCI_JITTER: f32 = 0.25;

/// Number of Lloyd relaxation iterations to regularize cell areas.
const LLOYD_ITERATIONS: usize = 2;

/// Samples per site for k-means Lloyd approximation.
/// Higher = better approximation but slower. 20 is a good balance.
const LLOYD_SAMPLES_PER_SITE: usize = 20;

impl Tessellation {
    /// Generate a tessellation with the given number of cells.
    ///
    /// Uses Fibonacci lattice with jitter for organic distribution, then
    /// k-means Lloyd relaxation to recover uniform cell areas.
    /// The `_lloyd_iterations` parameter is kept for API compatibility but ignored.
    pub fn generate<R: Rng>(num_cells: usize, _lloyd_iterations: usize, rng: &mut R) -> Self {
        use crate::util::Timed;

        // Mean angular spacing between cells: sqrt(4π / n)
        let mean_spacing = (4.0 * std::f32::consts::PI / num_cells as f32).sqrt();
        let jitter = mean_spacing * FIBONACCI_JITTER;

        let mut points = {
            let _t = Timed::debug("  Fibonacci points");
            fibonacci_sphere_points_with_rng(num_cells, jitter, rng)
        };

        // K-means Lloyd relaxation improves cell area uniformity
        if LLOYD_ITERATIONS > 0 {
            let _t = Timed::debug("  Lloyd relaxation");
            lloyd_relax_kmeans(&mut points, LLOYD_ITERATIONS, LLOYD_SAMPLES_PER_SITE, rng);
        }

        let voronoi = {
            let _t = Timed::debug("  Voronoi computation");
            SphericalVoronoi::compute(&points)
        };

        let adjacency = {
            let _t = Timed::debug("  Build adjacency");
            build_adjacency(&voronoi)
        };

        Self { voronoi, adjacency }
    }

    /// Generate a tessellation using the kNN clipping Voronoi algorithm.
    ///
    /// Same point distribution as `generate`, but uses the half-space clipping
    /// algorithm (via s2-voronoi crate) instead of convex hull duality.
    pub fn generate_knn_clipping<R: Rng>(
        num_cells: usize,
        _lloyd_iterations: usize,
        rng: &mut R,
    ) -> Self {
        let mean_spacing = (4.0 * std::f32::consts::PI / num_cells as f32).sqrt();
        let jitter = mean_spacing * FIBONACCI_JITTER;

        let mut points = {
            let _t = crate::util::Timed::debug("  Fibonacci points");
            fibonacci_sphere_points_with_rng(num_cells, jitter, rng)
        };

        if LLOYD_ITERATIONS > 0 {
            let _t = crate::util::Timed::debug("  Lloyd relaxation");
            lloyd_relax_kmeans(&mut points, LLOYD_ITERATIONS, LLOYD_SAMPLES_PER_SITE, rng);
        }

        Self::from_points_knn_clipping(points)
    }

    /// Build a tessellation from pre-sampled unit-sphere points using the
    /// s2-voronoi half-space clipping backend.
    pub fn from_points_knn_clipping(points: Vec<Vec3>) -> Self {
        use crate::geometry::VoronoiCell;
        use crate::util::Timed;

        // Adaptive fine meshes intentionally skip Lloyd relaxation; callers
        // that need relaxation should generate points through
        // `generate_knn_clipping`.

        // Pass plain arrays so hex3's glam version stays independent of the crate's
        let raw_points: Vec<[f32; 3]> = points.iter().map(|p| p.to_array()).collect();
        let diagram = {
            let _t = Timed::debug("  s2-voronoi computation");
            s2_voronoi::compute(&raw_points).expect("Voronoi computation failed")
        };

        // Use s2-voronoi's native adjacency (one sorted pass over boundary
        // edges) rather than rebuilding it with a hash map: ~11x faster at
        // millions of cells. Defective edges carry NO_NEIGHBOR; drop them.
        let adjacency = {
            let _t = Timed::debug("  Build adjacency (native)");
            let native = diagram.build_adjacency();
            let mut offsets = Vec::with_capacity(diagram.num_cells() + 1);
            let mut neighbors = Vec::with_capacity(diagram.num_cells() * 6);
            offsets.push(0);
            for i in 0..diagram.num_cells() {
                neighbors.extend(
                    native
                        .neighbors_of(i)
                        .iter()
                        .filter(|&&n| n != s2_voronoi::adjacency::NO_NEIGHBOR)
                        .map(|&n| n as usize),
                );
                offsets.push(neighbors.len());
            }
            CellAdjacency { offsets, neighbors }
        };

        let voronoi = {
            let _t = Timed::debug("  Convert diagram");

            let generators: Vec<Vec3> = diagram
                .generators()
                .iter()
                .map(|u| Vec3::new(u.x, u.y, u.z))
                .collect();
            let vertices: Vec<Vec3> = diagram
                .vertices()
                .iter()
                .map(|u| Vec3::new(u.x, u.y, u.z))
                .collect();

            // Build cells from s2-voronoi's cell views
            let mut cells = Vec::with_capacity(diagram.num_cells());
            let mut cell_indices: Vec<u32> = Vec::new();
            for cell_view in diagram.iter_cells() {
                let start = cell_indices.len() as u32;
                cell_indices.extend_from_slice(cell_view.vertex_indices);
                let count = cell_view.vertex_indices.len() as u16;
                cells.push(VoronoiCell::new(start, count));
            }

            SphericalVoronoi::from_raw_parts(generators, vertices, cells, cell_indices)
        };

        // Diagnostic: count orphan cells (no neighbors)
        let orphan_count = adjacency.iter().filter(|a| a.is_empty()).count();
        if orphan_count > 0 {
            log::warn!(
                "s2-voronoi: {} cells have no neighbors (orphans)",
                orphan_count
            );
        }

        Self { voronoi, adjacency }
    }

    /// Number of cells in this tessellation.
    pub fn num_cells(&self) -> usize {
        self.voronoi.num_cells()
    }

    /// Get the center point (generator) of a cell.
    pub fn cell_center(&self, cell_idx: usize) -> Vec3 {
        self.voronoi.generators[cell_idx]
    }

    /// Get the neighbors of a cell.
    pub fn neighbors(&self, cell_idx: usize) -> &[usize] {
        self.adjacency.neighbors(cell_idx)
    }

    /// Compute the solid angle (spherical area) of each Voronoi cell.
    ///
    /// Returns areas in steradians (total sphere = 4π).
    pub fn cell_areas(&self) -> Vec<f32> {
        let mut areas = vec![0.0f32; self.num_cells()];

        for cell_idx in 0..self.num_cells() {
            let cell = self.voronoi.cell(cell_idx);
            let center = self.cell_center(cell_idx);
            let verts: Vec<Vec3> = cell
                .vertex_indices
                .iter()
                .map(|&vi| self.voronoi.vertices[vi as usize])
                .collect();

            let n = verts.len();
            if n < 3 {
                continue;
            }

            // Sum spherical triangle areas from center to each edge
            let mut area = 0.0f32;
            for i in 0..n {
                let v1 = verts[i];
                let v2 = verts[(i + 1) % n];
                area += spherical_triangle_area(center, v1, v2);
            }
            areas[cell_idx] = area;
        }

        areas
    }

    /// Get the mean cell area (4π / num_cells).
    ///
    /// This is the theoretical mean for a uniform distribution on the sphere.
    pub fn mean_cell_area(&self) -> f32 {
        4.0 * std::f32::consts::PI / self.num_cells() as f32
    }

    /// Moran's I spatial autocorrelation of a per-cell field over the
    /// adjacency graph. ~1.0 for smooth fields, ~0.0 for spatially random
    /// noise, negative for checkerboards. Use to detect cell-scale speckle
    /// in fields that should be smooth at synoptic scales (e.g. rain).
    pub fn morans_i(&self, values: &[f32]) -> f32 {
        let n = self.num_cells();
        if n < 2 || values.len() != n {
            return 0.0;
        }
        let mean = values.iter().sum::<f32>() / n as f32;
        let variance: f32 = values.iter().map(|v| (v - mean).powi(2)).sum();
        if variance < 1e-12 {
            return 1.0; // constant field is trivially smooth
        }
        let mut cross = 0.0f64;
        let mut weight_count = 0usize;
        for i in 0..n {
            let vi = values[i] - mean;
            for &j in self.neighbors(i) {
                cross += (vi * (values[j] - mean)) as f64;
                weight_count += 1;
            }
        }
        ((n as f64 / weight_count.max(1) as f64) * cross / variance as f64) as f32
    }

    /// Length of the shared boundary edge of two adjacent cells.
    ///
    /// Returned as the CHORD between the two shared vertices, not the great-circle
    /// arc: `acos(dot)` collapses to 0 in f32 for separations below ~3 km
    /// (cos rounds to 1.0), which corrupts the fine mesh where edges are ~km
    /// scale. Chord is computed by direct subtraction (accurate in f32) and is
    /// identical to the arc to ~1e-8 at these scales. Coarse-mesh consumers
    /// (atmosphere/moisture/boundary) are unaffected.
    pub fn shared_edge_length(&self, cell_a: usize, cell_b: usize) -> f32 {
        // Voronoi cells have ~6 vertices, so a nested scan finds the two shared
        // ones without the HashSet allocations the old version paid per call
        // (this runs millions of times building fine-mesh geometry).
        let verts_a = self.voronoi.cell(cell_a).vertex_indices;
        let verts_b = self.voronoi.cell(cell_b).vertex_indices;
        let mut shared = [0u32; 2];
        let mut shared_len = 0usize;
        for &v in verts_a {
            if verts_b.contains(&v) {
                if shared_len < 2 {
                    shared[shared_len] = v;
                }
                shared_len += 1;
            }
        }

        if shared_len == 2 {
            let v0 = self.voronoi.vertices[shared[0] as usize];
            let v1 = self.voronoi.vertices[shared[1] as usize];
            (v0 - v1).length()
        } else {
            debug_assert_eq!(
                shared_len, 2,
                "adjacent cells {cell_a} and {cell_b} share {shared_len} vertices"
            );
            // Fallback: approximate the boundary edge length from the cell-center separation.
            // This should be unreachable for valid Voronoi topology.
            let pos_a = self.cell_center(cell_a);
            let pos_b = self.cell_center(cell_b);
            0.5 * (pos_a - pos_b).length()
        }
    }
}

/// Build adjacency list: for each cell, list of neighboring cell indices.
///
/// Two cells are adjacent if they share an edge (two consecutive Voronoi vertices).
fn build_adjacency(voronoi: &SphericalVoronoi) -> CellAdjacency {
    // Map from edge (as canonical vertex pair) to list of cells containing that edge
    let mut edge_to_cells: HashMap<(u32, u32), Vec<usize>> = HashMap::new();

    for cell_idx in 0..voronoi.num_cells() {
        let cell = voronoi.cell(cell_idx);
        let verts = cell.vertex_indices;
        let n = verts.len();

        for i in 0..n {
            let a = verts[i];
            let b = verts[(i + 1) % n];
            // Canonical ordering: smaller index first
            let edge = if a < b { (a, b) } else { (b, a) };
            edge_to_cells.entry(edge).or_default().push(cell_idx);
        }
    }

    // Build adjacency from edges shared by exactly 2 cells
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); voronoi.num_cells()];

    for cells in edge_to_cells.values() {
        if cells.len() == 2 {
            let c0 = cells[0];
            let c1 = cells[1];
            adjacency[c0].push(c1);
            adjacency[c1].push(c0);
        }
    }

    CellAdjacency::from_vecs(adjacency)
}

/// Compute the area of a spherical triangle on the unit sphere.
///
/// Uses the Van Oosterom–Strackee formula for the spherical excess (= area on
/// the unit sphere): `area = 2·atan2(|a·(b×c)|, 1 + a·b + b·c + c·a)`.
///
/// The older `Σ(dihedral angle) − π` form cancels catastrophically in f32 for
/// small triangles — every angle is ≈ π/3 and their sum sits a hair above π, so
/// `sum − π` is lost to rounding and the cell collapses to area 0. This form has
/// no such subtraction; it's evaluated in f64 to stay accurate down to tiny
/// cells (matching `validation::spherical_polygon_area`). All three points must
/// be on the unit sphere.
fn spherical_triangle_area(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let a = a.as_dvec3();
    let b = b.as_dvec3();
    let c = c.as_dvec3();
    let numerator = a.dot(b.cross(c)).abs();
    let denominator = 1.0 + a.dot(b) + b.dot(c) + c.dot(a);
    (2.0 * numerator.atan2(denominator)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{
        fibonacci_sphere_points_with_rng, lloyd_relax_kmeans, lloyd_relax_voronoi,
    };
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn measure_regularity(label: &str, points: &[Vec3]) -> (f32, f32, f32, usize) {
        let voronoi = SphericalVoronoi::compute(points);
        let adjacency = build_adjacency(&voronoi);
        let n = points.len();

        let orphans = adjacency.iter().filter(|nb| nb.is_empty()).count();

        // Cell areas
        let mean_area = 4.0 * std::f32::consts::PI / n as f32;
        let mut areas: Vec<f32> = Vec::with_capacity(n);
        for cell_idx in 0..voronoi.num_cells() {
            let cell = voronoi.cell(cell_idx);
            let center = points[cell_idx];
            let verts: Vec<Vec3> = cell
                .vertex_indices
                .iter()
                .map(|&vi| voronoi.vertices[vi as usize])
                .collect();
            let nv = verts.len();
            if nv < 3 {
                areas.push(0.0);
                continue;
            }
            let mut area = 0.0f32;
            for i in 0..nv {
                area += spherical_triangle_area(center, verts[i], verts[(i + 1) % nv]);
            }
            areas.push(area);
        }

        let actual_mean: f32 = areas.iter().sum::<f32>() / n as f32;
        let variance: f32 = areas.iter().map(|a| (a - actual_mean).powi(2)).sum::<f32>() / n as f32;
        let cv = variance.sqrt() / actual_mean;
        let min_ratio = areas.iter().cloned().fold(f32::INFINITY, f32::min) / mean_area;
        let max_ratio = areas.iter().cloned().fold(0.0f32, f32::max) / mean_area;

        println!(
            "{}: CV={:.3}, min/mean={:.3}, max/mean={:.3}, orphans={}",
            label, cv, min_ratio, max_ratio, orphans
        );

        (cv, min_ratio, max_ratio, orphans)
    }

    #[test]
    fn test_lloyd_comparison() {
        use std::time::Instant;

        let num_cells = 80000; // Production size
        let mean_spacing = (4.0 * std::f32::consts::PI / num_cells as f32).sqrt();

        println!("\n=== Comparing Lloyd methods at {} cells ===\n", num_cells);

        // Baseline - no Lloyd
        {
            let mut rng = ChaCha8Rng::seed_from_u64(99);
            let start = Instant::now();
            let points = fibonacci_sphere_points_with_rng(num_cells, mean_spacing * 0.25, &mut rng);
            let elapsed = start.elapsed();
            let (cv, _min_r, _max_r, _orphans) =
                measure_regularity("Fib j=0.25 (no Lloyd)", &points);
            println!("  Time: {:?}, CV={:.3}\n", elapsed, cv);
        }

        // Voronoi Lloyd 1 iter
        {
            let mut rng = ChaCha8Rng::seed_from_u64(99);
            let start = Instant::now();
            let mut points =
                fibonacci_sphere_points_with_rng(num_cells, mean_spacing * 0.25, &mut rng);
            lloyd_relax_voronoi(&mut points, 1);
            let elapsed = start.elapsed();
            let (cv, _, _, _) = measure_regularity("Fib j=0.25 + Voronoi Lloyd 1 iter", &points);
            println!("  Time: {:?}, CV={:.3}\n", elapsed, cv);
        }

        // K-means Lloyd 1 iter, 20 samples
        {
            let mut rng = ChaCha8Rng::seed_from_u64(99);
            let start = Instant::now();
            let mut points =
                fibonacci_sphere_points_with_rng(num_cells, mean_spacing * 0.25, &mut rng);
            lloyd_relax_kmeans(&mut points, 1, 20, &mut rng);
            let elapsed = start.elapsed();
            let (cv, _, _, _) = measure_regularity("Fib j=0.25 + K-means 1 iter (20samp)", &points);
            println!("  Time: {:?}, CV={:.3}\n", elapsed, cv);
        }

        // K-means Lloyd 2 iters, 20 samples
        {
            let mut rng = ChaCha8Rng::seed_from_u64(99);
            let start = Instant::now();
            let mut points =
                fibonacci_sphere_points_with_rng(num_cells, mean_spacing * 0.25, &mut rng);
            lloyd_relax_kmeans(&mut points, 2, 20, &mut rng);
            let elapsed = start.elapsed();
            let (cv, _, _, _) =
                measure_regularity("Fib j=0.25 + K-means 2 iters (20samp)", &points);
            println!("  Time: {:?}, CV={:.3}\n", elapsed, cv);
        }
    }

    #[test]
    fn test_cell_regularity() {
        // Test current configuration (j=0.25 + k-means Lloyd) produces good results
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let tess = Tessellation::generate(10000, 5, &mut rng);

        let orphans = tess.adjacency.iter().filter(|n| n.is_empty()).count();
        assert_eq!(orphans, 0, "Should have no orphan cells");

        let areas = tess.cell_areas();
        let mean_area = tess.mean_cell_area();
        let variance: f32 =
            areas.iter().map(|a| (a - mean_area).powi(2)).sum::<f32>() / areas.len() as f32;
        let cv = variance.sqrt() / mean_area;

        // With j=0.25 + k-means Lloyd, CV is typically 0.10-0.16
        assert!(cv < 0.18, "Cell area CV should be < 0.18, got {:.3}", cv);

        let min_area = areas.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_area = areas.iter().cloned().fold(0.0f32, f32::max);

        // Cells should be within reasonable range of mean
        assert!(
            min_area / mean_area > 0.4,
            "Smallest cell too small: {:.3}",
            min_area / mean_area
        );
        assert!(
            max_area / mean_area < 2.0,
            "Largest cell too large: {:.3}",
            max_area / mean_area
        );

        println!(
            "Cell regularity (j={}, {} k-means Lloyd, {}samp):",
            FIBONACCI_JITTER, LLOYD_ITERATIONS, LLOYD_SAMPLES_PER_SITE
        );
        println!(
            "  CV: {:.3}, min/mean: {:.3}, max/mean: {:.3}, orphans: {}",
            cv,
            min_area / mean_area,
            max_area / mean_area,
            orphans
        );
    }
}
