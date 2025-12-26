//! Spherical tessellation - Voronoi cells and adjacency graph.

use std::collections::HashMap;

use glam::Vec3;
use rand::Rng;

use crate::geometry::{fibonacci_sphere_points_with_rng, lloyd_relax_kmeans, SphericalVoronoi};

/// A spherical tessellation with Voronoi cells and cell adjacency.
pub struct Tessellation {
    /// The underlying Voronoi diagram.
    pub voronoi: SphericalVoronoi,

    /// Adjacency list: for each cell, indices of neighboring cells.
    pub adjacency: Vec<Vec<usize>>,
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

    /// Generate a tessellation using the GPU-style Voronoi algorithm.
    ///
    /// Same point distribution as `generate`, but uses the half-space clipping
    /// algorithm (via s2-voronoi crate) instead of convex hull duality.
    pub fn generate_gpu_style<R: Rng>(
        num_cells: usize,
        _lloyd_iterations: usize,
        rng: &mut R,
    ) -> Self {
        use crate::geometry::VoronoiCell;
        use crate::util::Timed;

        let mean_spacing = (4.0 * std::f32::consts::PI / num_cells as f32).sqrt();
        let jitter = mean_spacing * FIBONACCI_JITTER;

        let mut points = {
            let _t = Timed::debug("  Fibonacci points");
            fibonacci_sphere_points_with_rng(num_cells, jitter, rng)
        };

        if LLOYD_ITERATIONS > 0 {
            let _t = Timed::debug("  Lloyd relaxation");
            lloyd_relax_kmeans(&mut points, LLOYD_ITERATIONS, LLOYD_SAMPLES_PER_SITE, rng);
        }

        let voronoi = {
            let _t = Timed::debug("  s2-voronoi computation");

            // Use s2-voronoi crate for computation
            let output = s2_voronoi::compute(&points).expect("Voronoi computation failed");

            // Log diagnostics from s2-voronoi
            if !output.diagnostics.bad_cells.is_empty() {
                log::warn!(
                    "s2-voronoi: {} cells have < 3 vertices: {:?}",
                    output.diagnostics.bad_cells.len(),
                    &output.diagnostics.bad_cells[..output.diagnostics.bad_cells.len().min(20)]
                );
            }
            if !output.diagnostics.degenerate_cells.is_empty() {
                log::warn!(
                    "s2-voronoi: {} degenerate cells: {:?}",
                    output.diagnostics.degenerate_cells.len(),
                    &output.diagnostics.degenerate_cells
                        [..output.diagnostics.degenerate_cells.len().min(20)]
                );
            }

            // Convert s2-voronoi output to hex3's SphericalVoronoi
            let generators: Vec<Vec3> = output
                .diagram
                .generators
                .iter()
                .map(|u| Vec3::new(u.x, u.y, u.z))
                .collect();
            let vertices: Vec<Vec3> = output
                .diagram
                .vertices
                .iter()
                .map(|u| Vec3::new(u.x, u.y, u.z))
                .collect();

            // Build cells from s2-voronoi's cell views
            let mut cells = Vec::with_capacity(output.diagram.num_cells());
            let mut cell_indices: Vec<u32> = Vec::new();
            for i in 0..output.diagram.num_cells() {
                let cell_view = output.diagram.cell(i);
                let start = cell_indices.len() as u32;
                cell_indices.extend_from_slice(cell_view.vertex_indices);
                let count = cell_view.vertex_indices.len() as u16;
                cells.push(VoronoiCell::new(start, count));
            }

            SphericalVoronoi::from_raw_parts(generators, vertices, cells, cell_indices)
        };

        let adjacency = {
            let _t = Timed::debug("  Build adjacency");
            build_adjacency(&voronoi)
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
        &self.adjacency[cell_idx]
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
}

/// Build adjacency list: for each cell, list of neighboring cell indices.
///
/// Two cells are adjacent if they share an edge (two consecutive Voronoi vertices).
fn build_adjacency(voronoi: &SphericalVoronoi) -> Vec<Vec<usize>> {
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

    adjacency
}

/// Compute the area of a spherical triangle on the unit sphere.
///
/// Uses the spherical excess formula: area = (sum of angles) - π.
/// All three points must be on the unit sphere.
fn spherical_triangle_area(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    // Compute great-circle normals for each edge
    let ab = a.cross(b);
    let bc = b.cross(c);
    let ca = c.cross(a);

    // Handle degenerate triangles
    let len_ab = ab.length();
    let len_bc = bc.length();
    let len_ca = ca.length();
    if len_ab < 1e-10 || len_bc < 1e-10 || len_ca < 1e-10 {
        return 0.0;
    }

    let ab = ab / len_ab;
    let bc = bc / len_bc;
    let ca = ca / len_ca;

    // Dihedral angles at each vertex
    // Angle at a: between planes (a,b) and (c,a)
    let angle_a = (-ab).dot(ca).clamp(-1.0, 1.0).acos();
    // Angle at b: between planes (b,c) and (a,b)
    let angle_b = (-bc).dot(ab).clamp(-1.0, 1.0).acos();
    // Angle at c: between planes (c,a) and (b,c)
    let angle_c = (-ca).dot(bc).clamp(-1.0, 1.0).acos();

    // Spherical excess = sum of angles - π
    let excess = angle_a + angle_b + angle_c - std::f32::consts::PI;

    // Area on unit sphere equals the excess (in steradians)
    excess.max(0.0)
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
            let (cv, min_r, max_r, orphans) = measure_regularity("Fib j=0.25 (no Lloyd)", &points);
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
