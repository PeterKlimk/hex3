//! Validation utilities for spherical Voronoi diagrams.
//!
//! These functions check for grid validity issues that could cause problems
//! in downstream processing (adjacency, rivers, rendering, etc.)

use super::SphericalVoronoi;
use glam::Vec3;
use std::collections::{HashMap, HashSet};

/// Compute the signed area of a spherical polygon on a unit sphere.
/// Returns positive for counterclockwise winding, negative for clockwise.
fn spherical_polygon_area(vertices: &[Vec3]) -> f64 {
    let n = vertices.len();
    if n < 3 {
        return 0.0;
    }

    // Use the signed solid angle formula
    // Sum over triangles formed with first vertex (works for convex and simple polygons)
    let mut total = 0.0f64;
    let v0 = glam::DVec3::new(
        vertices[0].x as f64,
        vertices[0].y as f64,
        vertices[0].z as f64,
    );
    for i in 1..(n - 1) {
        let v1 = glam::DVec3::new(
            vertices[i].x as f64,
            vertices[i].y as f64,
            vertices[i].z as f64,
        );
        let v2 = glam::DVec3::new(
            vertices[i + 1].x as f64,
            vertices[i + 1].y as f64,
            vertices[i + 1].z as f64,
        );
        // Signed area of spherical triangle using the formula:
        // tan(E/2) = v0·(v1×v2) / (1 + v0·v1 + v1·v2 + v2·v0)
        // where E is the spherical excess (= area for unit sphere)
        let triple = v0.dot(v1.cross(v2));
        let denom = 1.0 + v0.dot(v1) + v1.dot(v2) + v2.dot(v0);
        let half_excess = triple.atan2(denom);
        total += 2.0 * half_excess;
    }
    total
}

/// Results of validating a spherical Voronoi diagram.
#[derive(Debug, Clone, Default)]
pub struct ValidationResult {
    /// Total number of cells
    pub num_cells: usize,
    /// Cells with fewer than 3 vertices (invalid polygons)
    pub degenerate_cells: Vec<usize>,
    /// Cells with duplicate vertex indices (same index appears twice)
    pub duplicate_index_cells: Vec<usize>,
    /// Cells with near-duplicate vertex positions (different indices, same position)
    pub near_duplicate_cells: Vec<(usize, Vec<(usize, usize, f32)>)>, // cell_idx, [(v1, v2, dist)]
    /// Edges that appear in only one cell (boundary errors)
    pub orphan_edges: Vec<(u32, u32)>, // (vertex_idx_a, vertex_idx_b)
    /// Edges that appear in more than 2 cells (topology errors)
    pub overcounted_edges: Vec<(u32, u32, usize)>, // (vertex_idx_a, vertex_idx_b, count)
    /// Cells that don't contain their generator in the cell interior
    pub generator_outside_cells: Vec<usize>,
    /// Cells with wrong winding order (clockwise instead of counterclockwise)
    pub wrong_winding_cells: Vec<usize>,
    /// Cells with negative or zero area
    pub negative_area_cells: Vec<usize>,
    /// Total surface area (should be 4π for a complete sphere)
    pub total_area: f64,
    /// Euler characteristic components (V - E + F should equal 2)
    pub euler_v: usize,
    pub euler_e: usize,
    pub euler_f: usize,
}

impl ValidationResult {
    /// Check if the grid is valid (no hard errors)
    pub fn is_valid(&self) -> bool {
        self.degenerate_cells.is_empty()
            && self.duplicate_index_cells.is_empty()
            && self.orphan_edges.is_empty()
            && self.overcounted_edges.is_empty()
            && self.generator_outside_cells.is_empty()
            && self.wrong_winding_cells.is_empty()
            && self.negative_area_cells.is_empty()
            && self.euler_check()
    }

    /// Check Euler characteristic: V - E + F = 2 for a sphere
    pub fn euler_check(&self) -> bool {
        (self.euler_v as i64) - (self.euler_e as i64) + (self.euler_f as i64) == 2
    }

    /// Check if total area is close to 4π (within 1%)
    pub fn area_check(&self) -> bool {
        let expected = 4.0 * std::f64::consts::PI;
        (self.total_area - expected).abs() / expected < 0.01
    }

    /// Total number of hard issues found
    pub fn issue_count(&self) -> usize {
        self.degenerate_cells.len()
            + self.duplicate_index_cells.len()
            + self.orphan_edges.len()
            + self.overcounted_edges.len()
            + self.generator_outside_cells.len()
            + self.wrong_winding_cells.len()
            + self.negative_area_cells.len()
            + if self.euler_check() { 0 } else { 1 }
    }

    /// Print a summary of validation results
    pub fn print_summary(&self) {
        println!("Voronoi Validation Results:");
        println!("  Total cells: {}", self.num_cells);
        println!(
            "  Euler: V={} E={} F={} (V-E+F={})",
            self.euler_v,
            self.euler_e,
            self.euler_f,
            (self.euler_v as i64) - (self.euler_e as i64) + (self.euler_f as i64)
        );
        println!(
            "  Total area: {:.6} (expected {:.6}, error {:.2}%)",
            self.total_area,
            4.0 * std::f64::consts::PI,
            100.0 * (self.total_area - 4.0 * std::f64::consts::PI).abs()
                / (4.0 * std::f64::consts::PI)
        );

        if self.is_valid() {
            println!("  Status: VALID");
            return;
        }

        println!("  Status: INVALID");
        if !self.euler_check() {
            println!("  Euler characteristic FAILED (expected V-E+F=2)");
        }
        if !self.degenerate_cells.is_empty() {
            println!(
                "  Degenerate cells (<3 vertices): {}",
                self.degenerate_cells.len()
            );
        }
        if !self.duplicate_index_cells.is_empty() {
            println!(
                "  Cells with duplicate indices: {}",
                self.duplicate_index_cells.len()
            );
        }
        if !self.near_duplicate_cells.is_empty() {
            println!(
                "  (info) Cells with near-duplicate vertices: {}",
                self.near_duplicate_cells.len()
            );
        }
        if !self.orphan_edges.is_empty() {
            println!(
                "  Orphan edges (in only 1 cell): {}",
                self.orphan_edges.len()
            );
        }
        if !self.overcounted_edges.is_empty() {
            println!(
                "  Overcounted edges (in >2 cells): {}",
                self.overcounted_edges.len()
            );
        }
        if !self.generator_outside_cells.is_empty() {
            println!(
                "  Generator outside cell: {}",
                self.generator_outside_cells.len()
            );
        }
        if !self.wrong_winding_cells.is_empty() {
            println!("  Wrong winding cells: {}", self.wrong_winding_cells.len());
        }
        if !self.negative_area_cells.is_empty() {
            println!(
                "  Negative/zero area cells: {}",
                self.negative_area_cells.len()
            );
        }
    }
}

/// Validate a spherical Voronoi diagram for grid consistency issues.
///
/// `near_duplicate_threshold` is the distance below which two vertices
/// are considered duplicates (e.g., 1e-6 for strict, 1e-4 for lenient).
pub fn validate_voronoi(
    voronoi: &SphericalVoronoi,
    near_duplicate_threshold: f32,
) -> ValidationResult {
    let mut result = ValidationResult {
        num_cells: voronoi.num_cells(),
        ..Default::default()
    };

    // Track edges for consistency checking
    // Key: (min_vertex_idx, max_vertex_idx), Value: list of cell indices
    let mut edge_to_cells: HashMap<(u32, u32), Vec<usize>> = HashMap::new();

    for cell_idx in 0..voronoi.num_cells() {
        let cell = voronoi.cell(cell_idx);
        let vertex_indices = cell.vertex_indices;

        // Check 1: Degenerate cells
        if vertex_indices.len() < 3 {
            result.degenerate_cells.push(cell_idx);
            continue; // Skip further checks for degenerate cells
        }

        // Check 2: Duplicate indices (same index appears twice)
        let mut seen_indices: Vec<u32> = Vec::with_capacity(vertex_indices.len());
        let mut has_dup_index = false;
        for &vi in vertex_indices {
            if seen_indices.contains(&vi) {
                has_dup_index = true;
                break;
            }
            seen_indices.push(vi);
        }
        if has_dup_index {
            result.duplicate_index_cells.push(cell_idx);
        }

        // Check 3: Near-duplicate positions (different indices, same position)
        let positions: Vec<Vec3> = vertex_indices
            .iter()
            .map(|&vi| voronoi.vertices[vi as usize])
            .collect();
        let mut near_dups: Vec<(usize, usize, f32)> = Vec::new();
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let dist = (positions[i] - positions[j]).length();
                if dist < near_duplicate_threshold {
                    near_dups.push((i, j, dist));
                }
            }
        }
        if !near_dups.is_empty() {
            result.near_duplicate_cells.push((cell_idx, near_dups));
        }

        // Track edges for this cell
        for i in 0..vertex_indices.len() {
            let v1 = vertex_indices[i];
            let v2 = vertex_indices[(i + 1) % vertex_indices.len()];
            let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
            edge_to_cells.entry(edge).or_default().push(cell_idx);
        }

        // Check 4: Generator should be "inside" the cell
        // For a spherical polygon, a point is inside if it's on the correct side of all edges.
        // Each edge defines a great circle; the generator should be on the interior side.
        let generator = voronoi.generators[cell.generator_index];
        let mut generator_inside = true;
        for i in 0..vertex_indices.len() {
            let v1 = voronoi.vertices[vertex_indices[i] as usize];
            let v2 = voronoi.vertices[vertex_indices[(i + 1) % vertex_indices.len()] as usize];
            // Edge plane normal: cross product of the two vertices (points on unit sphere)
            // This gives the normal to the great circle containing the edge
            let edge_normal = v1.cross(v2);
            if edge_normal.length_squared() < 1e-12 {
                continue; // Degenerate edge, skip
            }
            // Generator should be on the positive side (same side as the interior)
            // The winding should be such that the interior is on the left of the edge
            if generator.dot(edge_normal) < -1e-6 {
                generator_inside = false;
                break;
            }
        }
        if !generator_inside {
            result.generator_outside_cells.push(cell_idx);
        }

        // Check 5: Cell area (should be positive for counterclockwise winding)
        let area = spherical_polygon_area(&positions);
        result.total_area += area;
        if area <= 0.0 {
            // Negative or zero area indicates wrong winding or degenerate polygon
            if area < -1e-10 {
                result.wrong_winding_cells.push(cell_idx);
            } else {
                result.negative_area_cells.push(cell_idx);
            }
        }
    }

    // Check edge consistency and compute Euler characteristic
    let mut unique_vertices: HashSet<u32> = HashSet::new();
    for cell_idx in 0..voronoi.num_cells() {
        let cell = voronoi.cell(cell_idx);
        for &vi in cell.vertex_indices {
            unique_vertices.insert(vi);
        }
    }
    result.euler_v = unique_vertices.len();
    result.euler_e = edge_to_cells.len();
    result.euler_f = voronoi.num_cells();

    for (edge, cells) in &edge_to_cells {
        match cells.len() {
            1 => result.orphan_edges.push(*edge),
            2 => {} // Correct: each edge should be shared by exactly 2 cells
            n => result.overcounted_edges.push((edge.0, edge.1, n)),
        }
    }

    result
}

/// Result of random point sampling validation
#[derive(Debug, Clone, Default)]
pub struct PointSampleResult {
    /// Number of samples tested
    pub num_samples: usize,
    /// Number of samples where the nearest generator matched the cell
    pub correct: usize,
    /// Number of samples where the nearest generator didn't match (within tolerance)
    pub incorrect: usize,
    /// Maximum distance error (how much closer a wrong generator was)
    pub max_error: f64,
    /// Samples where point wasn't inside any cell (mesh has gaps)
    pub outside_all_cells: usize,
}

impl PointSampleResult {
    pub fn is_valid(&self) -> bool {
        self.incorrect == 0 && self.outside_all_cells == 0
    }

    pub fn accuracy(&self) -> f64 {
        if self.num_samples == 0 {
            1.0
        } else {
            self.correct as f64 / self.num_samples as f64
        }
    }
}

/// Validate Voronoi diagram by random point sampling.
/// For each random point on the sphere, find which cell contains it,
/// then verify that cell's generator is the closest generator to the point.
pub fn validate_voronoi_sampling<R: rand::Rng>(
    voronoi: &SphericalVoronoi,
    num_samples: usize,
    tolerance: f64,
    rng: &mut R,
) -> PointSampleResult {
    use rand::distributions::Distribution;

    let mut result = PointSampleResult {
        num_samples,
        ..Default::default()
    };

    // Build a simple spatial lookup for finding which cell contains a point
    // For each sample point, we'll check against all cells (brute force but correct)

    for _ in 0..num_samples {
        // Generate random point on unit sphere
        let normal = rand_distr::StandardNormal;
        let x: f64 = normal.sample(rng);
        let y: f64 = normal.sample(rng);
        let z: f64 = normal.sample(rng);
        let len = (x * x + y * y + z * z).sqrt();
        let point = Vec3::new((x / len) as f32, (y / len) as f32, (z / len) as f32);

        // Find the nearest generator
        let mut nearest_gen = 0;
        let mut nearest_dist = f64::MAX;
        for (gi, &gen) in voronoi.generators.iter().enumerate() {
            let dist = (point - gen).length() as f64;
            if dist < nearest_dist {
                nearest_dist = dist;
                nearest_gen = gi;
            }
        }

        // Find which cell contains this point
        let mut containing_cell: Option<usize> = None;
        for cell_idx in 0..voronoi.num_cells() {
            let cell = voronoi.cell(cell_idx);
            if cell.vertex_indices.len() < 3 {
                continue;
            }

            // Check if point is inside this cell's spherical polygon
            let mut inside = true;
            for i in 0..cell.vertex_indices.len() {
                let v1 = voronoi.vertices[cell.vertex_indices[i] as usize];
                let v2 = voronoi.vertices
                    [cell.vertex_indices[(i + 1) % cell.vertex_indices.len()] as usize];
                let edge_normal = v1.cross(v2);
                if edge_normal.length_squared() < 1e-12 {
                    continue;
                }
                if point.dot(edge_normal) < -1e-6 {
                    inside = false;
                    break;
                }
            }

            if inside {
                containing_cell = Some(cell_idx);
                break;
            }
        }

        match containing_cell {
            Some(cell_idx) => {
                let cell = voronoi.cell(cell_idx);
                if cell.generator_index == nearest_gen {
                    result.correct += 1;
                } else {
                    // Check if this is within tolerance
                    let cell_gen_dist =
                        (point - voronoi.generators[cell.generator_index]).length() as f64;
                    let error = cell_gen_dist - nearest_dist;
                    if error > tolerance {
                        result.incorrect += 1;
                        result.max_error = result.max_error.max(error);
                    } else {
                        result.correct += 1; // Within tolerance, count as correct
                    }
                }
            }
            None => {
                result.outside_all_cells += 1;
            }
        }
    }

    result
}

/// Validate and return simple pass/fail with optional details
pub fn validate_voronoi_quick(voronoi: &SphericalVoronoi) -> (bool, String) {
    let result = validate_voronoi(voronoi, 1e-6);
    if result.is_valid() {
        (true, "Valid".to_string())
    } else {
        let mut issues = Vec::new();
        if !result.degenerate_cells.is_empty() {
            issues.push(format!("{} degenerate", result.degenerate_cells.len()));
        }
        if !result.duplicate_index_cells.is_empty() {
            issues.push(format!("{} dup-index", result.duplicate_index_cells.len()));
        }
        if !result.near_duplicate_cells.is_empty() {
            issues.push(format!("{} near-dup", result.near_duplicate_cells.len()));
        }
        if !result.orphan_edges.is_empty() {
            issues.push(format!("{} orphan-edge", result.orphan_edges.len()));
        }
        if !result.overcounted_edges.is_empty() {
            issues.push(format!("{} over-edge", result.overcounted_edges.len()));
        }
        (false, issues.join(", "))
    }
}

/// Count unique vertices in a cell (for deduplication analysis)
pub fn count_unique_vertices(voronoi: &SphericalVoronoi, cell_idx: usize, tolerance: f32) -> usize {
    let cell = voronoi.cell(cell_idx);
    let positions: Vec<Vec3> = cell
        .vertex_indices
        .iter()
        .map(|&vi| voronoi.vertices[vi as usize])
        .collect();

    let mut unique: Vec<Vec3> = Vec::new();
    for p in &positions {
        if !unique.iter().any(|u| (*p - *u).length() < tolerance) {
            unique.push(*p);
        }
    }
    unique.len()
}

/// Create a deduplicated version of a Voronoi diagram.
/// Returns a new SphericalVoronoi with duplicate vertices merged within each cell.
pub fn deduplicate_voronoi(voronoi: &SphericalVoronoi, tolerance: f32) -> SphericalVoronoi {
    let mut new_cell_data: Vec<Vec<u32>> = Vec::with_capacity(voronoi.num_cells());

    for cell_idx in 0..voronoi.num_cells() {
        let cell = voronoi.cell(cell_idx);
        let positions: Vec<Vec3> = cell
            .vertex_indices
            .iter()
            .map(|&vi| voronoi.vertices[vi as usize])
            .collect();

        // Build list of unique vertex indices
        let mut unique_indices: Vec<u32> = Vec::new();
        let mut unique_positions: Vec<Vec3> = Vec::new();

        for (local_idx, &global_idx) in cell.vertex_indices.iter().enumerate() {
            let pos = positions[local_idx];
            let is_dup = unique_positions
                .iter()
                .any(|u| (*u - pos).length() < tolerance);
            if !is_dup {
                unique_indices.push(global_idx);
                unique_positions.push(pos);
            }
        }

        new_cell_data.push(unique_indices);
    }

    SphericalVoronoi::new(
        voronoi.generators.clone(),
        voronoi.vertices.clone(),
        new_cell_data,
    )
}

/// Analyze orphan edges to understand their cause
pub fn analyze_orphan_edges(voronoi: &SphericalVoronoi, tolerance: f32) -> OrphanEdgeAnalysis {
    let result = validate_voronoi(voronoi, tolerance);

    let mut analysis = OrphanEdgeAnalysis {
        total_orphan_edges: result.orphan_edges.len(),
        ..Default::default()
    };

    if result.orphan_edges.is_empty() {
        return analysis;
    }

    // For each orphan edge, find which cell it belongs to and check if there's
    // a "missing" adjacent cell that should share this edge
    for &(v1, v2) in &result.orphan_edges {
        let pos1 = voronoi.vertices[v1 as usize];
        let pos2 = voronoi.vertices[v2 as usize];
        let _edge_midpoint = ((pos1 + pos2) / 2.0).normalize();

        // Find cells that contain either v1 or v2
        let mut cells_with_v1: Vec<usize> = Vec::new();
        let mut cells_with_v2: Vec<usize> = Vec::new();

        for cell_idx in 0..voronoi.num_cells() {
            let cell = voronoi.cell(cell_idx);
            if cell.vertex_indices.contains(&v1) {
                cells_with_v1.push(cell_idx);
            }
            if cell.vertex_indices.contains(&v2) {
                cells_with_v2.push(cell_idx);
            }
        }

        // The orphan edge should be in exactly one cell
        // Check if there's a vertex position match in another cell that uses different index
        let mut found_positional_match = false;
        for cell_idx in 0..voronoi.num_cells() {
            let cell = voronoi.cell(cell_idx);
            let cell_has_edge = cell
                .vertex_indices
                .windows(2)
                .any(|w| (w[0] == v1 && w[1] == v2) || (w[0] == v2 && w[1] == v1))
                || (cell.vertex_indices.first() == Some(&v1)
                    && cell.vertex_indices.last() == Some(&v2))
                || (cell.vertex_indices.first() == Some(&v2)
                    && cell.vertex_indices.last() == Some(&v1));

            if cell_has_edge {
                continue;
            }

            // Check if cell has vertices at same positions but different indices
            let positions: Vec<Vec3> = cell
                .vertex_indices
                .iter()
                .map(|&vi| voronoi.vertices[vi as usize])
                .collect();

            let has_pos1 = positions.iter().any(|p| (*p - pos1).length() < tolerance);
            let has_pos2 = positions.iter().any(|p| (*p - pos2).length() < tolerance);

            if has_pos1 && has_pos2 {
                found_positional_match = true;
                analysis.position_mismatch_edges += 1;
                break;
            }
        }

        if !found_positional_match {
            // This edge truly has no adjacent cell
            analysis.true_orphan_edges += 1;
        }
    }

    analysis
}

/// Analysis of orphan edge causes
#[derive(Debug, Clone, Default)]
pub struct OrphanEdgeAnalysis {
    pub total_orphan_edges: usize,
    /// Edges where adjacent cell exists but uses different vertex indices for same position
    pub position_mismatch_edges: usize,
    /// Edges that truly have no adjacent cell (geometry error)
    pub true_orphan_edges: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{fibonacci_sphere_points_with_rng, SphericalVoronoi};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn generate_test_points(n: usize, seed: u64) -> Vec<Vec3> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mean_spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
        let jitter = mean_spacing * 0.25;
        fibonacci_sphere_points_with_rng(n, jitter, &mut rng)
    }

    #[test]
    fn test_hull_voronoi_valid() {
        let points = generate_test_points(1000, 12345);
        let voronoi = SphericalVoronoi::compute(&points);
        let result = validate_voronoi(&voronoi, 1e-6);

        result.print_summary();
        // Hull method should produce valid grids (may have some near-duplicates at degeneracies)
        assert!(
            result.degenerate_cells.is_empty(),
            "Hull should not produce degenerate cells"
        );
        assert!(
            result.duplicate_index_cells.is_empty(),
            "Hull should not have duplicate indices"
        );
    }

    #[test]
    fn test_s2_voronoi_validity() {
        let points = generate_test_points(1000, 12345);
        let output = s2_voronoi::compute(&points).expect("s2-voronoi should succeed");

        // Convert to hex3's SphericalVoronoi for validation
        let cell_data: Vec<Vec<u32>> = output
            .diagram
            .iter_cells()
            .map(|c| c.vertex_indices.to_vec())
            .collect();
        let generators: Vec<Vec3> = output
            .diagram
            .generators
            .iter()
            .map(|g| Vec3::new(g.x, g.y, g.z))
            .collect();
        let vertices: Vec<Vec3> = output
            .diagram
            .vertices
            .iter()
            .map(|v| Vec3::new(v.x, v.y, v.z))
            .collect();
        let voronoi = SphericalVoronoi::new(generators, vertices, cell_data);

        let result = validate_voronoi(&voronoi, 1e-6);
        result.print_summary();

        assert!(
            result.degenerate_cells.is_empty(),
            "s2-voronoi should not produce degenerate cells"
        );
    }

    #[test]
    fn test_deduplicate_voronoi() {
        let points = generate_test_points(1000, 12345);
        let output = s2_voronoi::compute(&points).expect("s2-voronoi should succeed");

        // Convert to hex3's SphericalVoronoi
        let cell_data: Vec<Vec<u32>> = output
            .diagram
            .iter_cells()
            .map(|c| c.vertex_indices.to_vec())
            .collect();
        let generators: Vec<Vec3> = output
            .diagram
            .generators
            .iter()
            .map(|g| Vec3::new(g.x, g.y, g.z))
            .collect();
        let vertices: Vec<Vec3> = output
            .diagram
            .vertices
            .iter()
            .map(|v| Vec3::new(v.x, v.y, v.z))
            .collect();
        let voronoi = SphericalVoronoi::new(generators, vertices, cell_data);

        let before = validate_voronoi(&voronoi, 1e-4);
        let deduped = deduplicate_voronoi(&voronoi, 1e-4);
        let after = validate_voronoi(&deduped, 1e-4);

        println!(
            "Before dedup: {} near-dup cells",
            before.near_duplicate_cells.len()
        );
        println!(
            "After dedup: {} near-dup cells",
            after.near_duplicate_cells.len()
        );

        assert!(
            after.near_duplicate_cells.len() <= before.near_duplicate_cells.len(),
            "Deduplication should reduce or maintain near-duplicate count"
        );
    }

    #[test]
    fn test_edge_consistency() {
        let points = generate_test_points(500, 12345);
        let voronoi = SphericalVoronoi::compute(&points);
        let result = validate_voronoi(&voronoi, 1e-6);

        // A valid spherical Voronoi should have no orphan or overcounted edges
        // (it's a closed surface, every edge is shared by exactly 2 cells)
        assert!(
            result.orphan_edges.is_empty(),
            "Should have no orphan edges, found {}",
            result.orphan_edges.len()
        );
        assert!(
            result.overcounted_edges.is_empty(),
            "Should have no overcounted edges, found {}",
            result.overcounted_edges.len()
        );
    }
}
