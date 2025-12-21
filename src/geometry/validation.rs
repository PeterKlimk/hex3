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
    let v0 = glam::DVec3::new(vertices[0].x as f64, vertices[0].y as f64, vertices[0].z as f64);
    for i in 1..(n - 1) {
        let v1 = glam::DVec3::new(vertices[i].x as f64, vertices[i].y as f64, vertices[i].z as f64);
        let v2 = glam::DVec3::new(vertices[i + 1].x as f64, vertices[i + 1].y as f64, vertices[i + 1].z as f64);
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
    pub orphan_edges: Vec<(usize, usize)>, // (vertex_idx_a, vertex_idx_b)
    /// Edges that appear in more than 2 cells (topology errors)
    pub overcounted_edges: Vec<(usize, usize, usize)>, // (vertex_idx_a, vertex_idx_b, count)
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
            println!("  Orphan edges (in only 1 cell): {}", self.orphan_edges.len());
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
            println!(
                "  Wrong winding cells: {}",
                self.wrong_winding_cells.len()
            );
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
pub fn validate_voronoi(voronoi: &SphericalVoronoi, near_duplicate_threshold: f32) -> ValidationResult {
    let mut result = ValidationResult {
        num_cells: voronoi.num_cells(),
        ..Default::default()
    };

    // Track edges for consistency checking
    // Key: (min_vertex_idx, max_vertex_idx), Value: list of cell indices
    let mut edge_to_cells: HashMap<(usize, usize), Vec<usize>> = HashMap::new();

    for cell_idx in 0..voronoi.num_cells() {
        let cell = voronoi.cell(cell_idx);
        let vertex_indices = cell.vertex_indices;

        // Check 1: Degenerate cells
        if vertex_indices.len() < 3 {
            result.degenerate_cells.push(cell_idx);
            continue; // Skip further checks for degenerate cells
        }

        // Check 2: Duplicate indices (same index appears twice)
        let mut seen_indices: Vec<usize> = Vec::with_capacity(vertex_indices.len());
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
            .map(|&vi| voronoi.vertices[vi])
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
            let v1 = voronoi.vertices[vertex_indices[i]];
            let v2 = voronoi.vertices[vertex_indices[(i + 1) % vertex_indices.len()]];
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
    let mut unique_vertices: HashSet<usize> = HashSet::new();
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
                let v1 = voronoi.vertices[cell.vertex_indices[i]];
                let v2 = voronoi.vertices[cell.vertex_indices[(i + 1) % cell.vertex_indices.len()]];
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
                    let cell_gen_dist = (point - voronoi.generators[cell.generator_index]).length() as f64;
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

/// Strict validation with support-set unification (debug use; O(V * N)).
#[derive(Debug, Clone, Default)]
pub struct StrictValidationResult {
    pub num_cells: usize,
    pub num_vertices: usize,
    pub num_effective_generators: usize,
    pub num_merged_generators: usize,
    pub merge_threshold: f64,
    pub degenerate_cells: Vec<usize>,
    pub orphan_edges: Vec<(usize, usize)>,
    pub overcounted_edges: Vec<(usize, usize, usize)>,
    pub duplicate_support_vertices: Vec<(usize, usize)>,
    pub support_lt3: Vec<usize>,
    pub generator_mismatch_vertices: Vec<usize>,
    pub ambiguous_vertices: usize,
    pub collapsed_edges: usize,
    pub missing_key_edges: usize,
    pub max_generator_delta: f64,
    pub max_edge_bisector_error: f64,
    pub edge_samples: usize,
    pub euler_v: usize,
    pub euler_e: usize,
    pub euler_f: usize,
    pub euler_ok: bool,
}

impl StrictValidationResult {
    /// Check if the mesh is valid (hard errors only).
    /// Degeneracy-related metrics (support_lt3, gen_mismatch, dup_support, collapsed_edges)
    /// are informational.
    pub fn is_valid(&self) -> bool {
        self.degenerate_cells.is_empty()
            && self.orphan_edges.is_empty()
            && self.overcounted_edges.is_empty()
            && self.euler_ok
    }

    /// Check if there are any degeneracy-related issues (informational, not errors)
    pub fn has_degeneracy_issues(&self) -> bool {
        !self.support_lt3.is_empty()
            || !self.generator_mismatch_vertices.is_empty()
            || !self.duplicate_support_vertices.is_empty()
            || self.collapsed_edges > 0
            || self.missing_key_edges > 0
    }

    pub fn print_summary(&self) {
        let ambiguous_rate = if self.num_vertices > 0 {
            self.ambiguous_vertices as f64 / self.num_vertices as f64
        } else {
            0.0
        };
        println!("Strict Validation Results:");
        println!(
            "  merged_generators={} (effective={}), merge_threshold={:.2e}",
            self.num_merged_generators,
            self.num_effective_generators,
            self.merge_threshold
        );
        println!(
            "  V/E/F = {}/{}/{} (Euler ok: {})",
            self.euler_v, self.euler_e, self.euler_f, self.euler_ok
        );
        println!(
            "  [hard] degenerate_cells={}, orphan_edges={}, overcounted_edges={}",
            self.degenerate_cells.len(),
            self.orphan_edges.len(),
            self.overcounted_edges.len(),
        );
        println!(
            "  [info] support_lt3={}, gen_mismatch={}, dup_support={}, collapsed_edges={}, missing_key_edges={}",
            self.support_lt3.len(),
            self.generator_mismatch_vertices.len(),
            self.duplicate_support_vertices.len(),
            self.collapsed_edges,
            self.missing_key_edges
        );
        println!(
            "  ambiguous_vertices={} ({:.2}%), max_gen_delta={:.2e}, max_edge_err={:.2e} (samples={})",
            self.ambiguous_vertices,
            ambiguous_rate * 100.0,
            self.max_generator_delta,
            self.max_edge_bisector_error,
            self.edge_samples
        );
    }
}

/// Validate with support sets S(v) = { i | max_dot - v·g_i <= eps }.
/// Uses eps_lo for strict membership and eps_hi for ambiguity detection.
pub fn validate_voronoi_strict(
    voronoi: &SphericalVoronoi,
    eps_lo: f64,
    eps_hi: f64,
    merge_threshold: Option<f32>,
) -> StrictValidationResult {
    use crate::geometry::gpu_voronoi::{CubeMapGridKnn, KnnProvider, merge_close_points};

    let num_cells = voronoi.num_cells();
    let num_vertices = voronoi.vertices.len();
    let mut original_to_effective: Vec<usize> = (0..voronoi.generators.len()).collect();
    let mut effective_generators: Vec<Vec3> = voronoi.generators.clone();
    let mut num_merged_generators = 0usize;
    let merge_threshold = merge_threshold.unwrap_or(0.0);

    if merge_threshold > 0.0 {
        let knn = CubeMapGridKnn::new(&voronoi.generators);
        let merge_result = merge_close_points(&voronoi.generators, merge_threshold, &knn);
        if merge_result.num_merged > 0 {
            effective_generators = merge_result.effective_points;
            original_to_effective = merge_result.original_to_effective;
            num_merged_generators = merge_result.num_merged;
        }
    }

    let num_effective_generators = effective_generators.len();
    let effective_gens_d: Vec<glam::DVec3> = effective_generators
        .iter()
        .map(|g| glam::DVec3::new(g.x as f64, g.y as f64, g.z as f64))
        .collect();
    let mut result = StrictValidationResult {
        num_cells,
        num_vertices,
        num_effective_generators,
        num_merged_generators,
        merge_threshold: merge_threshold as f64,
        ..Default::default()
    };

    let mut edge_to_cells: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut vertex_to_cells: Vec<Vec<usize>> = vec![Vec::new(); num_vertices];

    for cell_idx in 0..num_cells {
        let cell = voronoi.cell(cell_idx);
        let vertex_indices = cell.vertex_indices;
        let eff_cell_idx = original_to_effective[cell.generator_index];
        if vertex_indices.len() < 3 {
            result.degenerate_cells.push(cell_idx);
        }

        for &vi in vertex_indices {
            if vi < num_vertices {
                if !vertex_to_cells[vi].contains(&eff_cell_idx) {
                    vertex_to_cells[vi].push(eff_cell_idx);
                }
            }
        }

        for i in 0..vertex_indices.len() {
            let v1 = vertex_indices[i];
            let v2 = vertex_indices[(i + 1) % vertex_indices.len()];
            if v1 == v2 {
                continue;
            }
            let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
            let cells = edge_to_cells.entry(edge).or_default();
            if !cells.contains(&eff_cell_idx) {
                cells.push(eff_cell_idx);
            }
        }
    }

    for (edge, cells) in &edge_to_cells {
        if cells.len() == 1 {
            result.orphan_edges.push(*edge);
        } else if cells.len() > 2 {
            result.overcounted_edges.push((edge.0, edge.1, cells.len()));
        }
    }

    let mut support_key_to_id: HashMap<Vec<usize>, (usize, usize)> = HashMap::new();
    let mut support_keys: Vec<Vec<usize>> = Vec::new();
    let mut vertex_key_id: Vec<usize> = vec![usize::MAX; num_vertices];

    for (v_idx, &v) in voronoi.vertices.iter().enumerate() {
        if vertex_to_cells[v_idx].is_empty() {
            continue;
        }
        let v64 = glam::DVec3::new(v.x as f64, v.y as f64, v.z as f64).normalize();
        let mut max_dot = f64::NEG_INFINITY;
        for g64 in &effective_gens_d {
            let d = v64.dot(*g64);
            if d > max_dot {
                max_dot = d;
            }
        }

        let mut support_lo: Vec<usize> = Vec::new();
        let mut support_hi: Vec<usize> = Vec::new();
        for (gi, g64) in effective_gens_d.iter().enumerate() {
            let d = v64.dot(*g64);
            let delta = max_dot - d;
            if delta <= eps_hi {
                support_hi.push(gi);
                if delta <= eps_lo {
                    support_lo.push(gi);
                }
            }
        }

        let mut vertex_ambiguous = false;
        if support_lo.len() < 3 {
            result.support_lt3.push(v_idx);
            if support_hi.len() >= 3 {
                vertex_ambiguous = true;
            }
        }

        if support_hi.len() > support_lo.len() {
            vertex_ambiguous = true;
        }

        let support_key: &[usize] = if support_lo.len() >= 3 {
            &support_lo
        } else {
            &support_hi
        };

        if support_key.len() >= 3 {
            if let Some(&(id, rep_vertex)) = support_key_to_id.get(support_key) {
                result.duplicate_support_vertices.push((rep_vertex, v_idx));
                vertex_key_id[v_idx] = id;
            } else {
                let id = support_key_to_id.len();
                let key_vec = support_key.to_vec();
                support_keys.push(key_vec.clone());
                support_key_to_id.insert(key_vec, (id, v_idx));
                vertex_key_id[v_idx] = id;
            }
        }

        if !vertex_to_cells[v_idx].is_empty() {
            let mut mismatch = false;
            for &eff_idx in &vertex_to_cells[v_idx] {
                let g = effective_gens_d[eff_idx];
                let delta = max_dot - v64.dot(g);
                if delta > result.max_generator_delta {
                    result.max_generator_delta = delta;
                }
                if support_hi.binary_search(&eff_idx).is_err() {
                    mismatch = true;
                    break;
                }
                if support_lo.binary_search(&eff_idx).is_err() {
                    vertex_ambiguous = true;
                }
            }
            if mismatch {
                result.generator_mismatch_vertices.push(v_idx);
            }
        }

        if vertex_ambiguous {
            result.ambiguous_vertices += 1;
        }
    }

    let mut rep_cell_for_eff: Vec<Option<usize>> = vec![None; num_effective_generators];
    for cell_idx in 0..num_cells {
        let eff_idx = original_to_effective[voronoi.cell(cell_idx).generator_index];
        if rep_cell_for_eff[eff_idx].is_none() {
            rep_cell_for_eff[eff_idx] = Some(cell_idx);
        }
    }

    let mut edges: HashSet<(usize, usize)> = HashSet::new();
    let mut collapsed_samples: Vec<(usize, usize, usize)> = Vec::new();
    let mut missing_key_samples: Vec<(usize, usize)> = Vec::new();
    for rep in rep_cell_for_eff.iter().flatten() {
        let cell = voronoi.cell(*rep);
        let vertex_indices = cell.vertex_indices;
        if vertex_indices.len() < 3 {
            continue;
        }
        for i in 0..vertex_indices.len() {
            let v1 = vertex_indices[i];
            let v2 = vertex_indices[(i + 1) % vertex_indices.len()];
            let k1 = vertex_key_id.get(v1).copied().unwrap_or(usize::MAX);
            let k2 = vertex_key_id.get(v2).copied().unwrap_or(usize::MAX);
            if k1 == usize::MAX || k2 == usize::MAX {
                result.missing_key_edges += 1;
                if missing_key_samples.len() < 5 {
                    missing_key_samples.push((v1, v2));
                }
                continue;
            }
            if k1 == k2 {
                result.collapsed_edges += 1;
                if collapsed_samples.len() < 5 {
                    collapsed_samples.push((v1, v2, k1));
                }
                continue;
            }
            let edge = if k1 < k2 { (k1, k2) } else { (k2, k1) };
            edges.insert(edge);
        }
    }

    result.euler_v = support_key_to_id.len();
    result.euler_e = edges.len();
    result.euler_f = rep_cell_for_eff.iter().flatten().count();
    let euler_lhs = result.euler_v as i64 - result.euler_e as i64 + result.euler_f as i64;
    result.euler_ok = euler_lhs == 2;

    // Soft geometry check: edge bisector error for edges shared by exactly 2 cells.
    for (edge, cells) in edge_to_cells {
        if cells.len() != 2 {
            continue;
        }
        let (v1, v2) = edge;
        let p1 = voronoi.vertices[v1];
        let p2 = voronoi.vertices[v2];
        let p1 = glam::DVec3::new(p1.x as f64, p1.y as f64, p1.z as f64).normalize();
        let p2 = glam::DVec3::new(p2.x as f64, p2.y as f64, p2.z as f64).normalize();
        let mid = (p1 + p2).normalize();
        let a = (p1 * 2.0 + p2).normalize();
        let b = (p1 + p2 * 2.0).normalize();

        let g1 = effective_gens_d[cells[0]];
        let g2 = effective_gens_d[cells[1]];
        let n = g1 - g2;

        for s in [p1, a, mid, b, p2] {
            let err = n.dot(s).abs();
            if err > result.max_edge_bisector_error {
                result.max_edge_bisector_error = err;
            }
            result.edge_samples += 1;
        }
    }

    if std::env::var("STRICT_VALIDATE_DEBUG").is_ok()
        && (!result.is_valid() || result.ambiguous_vertices > 0)
    {
        use crate::geometry::gpu_voronoi::DEFAULT_K;

        eprintln!(
            "[strict] dup_support={}, collapsed_edges={}, missing_key_edges={}",
            result.duplicate_support_vertices.len(),
            result.collapsed_edges,
            result.missing_key_edges
        );
        for &(rep, dup) in result.duplicate_support_vertices.iter().take(5) {
            let key_id = vertex_key_id.get(rep).copied().unwrap_or(usize::MAX);
            let key = if key_id != usize::MAX {
                support_keys.get(key_id)
            } else {
                None
            };
            let pos_rep = voronoi.vertices[rep];
            let pos_dup = voronoi.vertices[dup];
            let dist = (pos_rep - pos_dup).length();
            eprintln!(
                "[strict] dup_support rep=v{} dup=v{} dist={:.2e} key={:?} rep_cells={:?} dup_cells={:?}",
                rep,
                dup,
                dist,
                key,
                vertex_to_cells.get(rep),
                vertex_to_cells.get(dup)
            );
            if let Some(key) = key {
                let rep64 = glam::DVec3::new(pos_rep.x as f64, pos_rep.y as f64, pos_rep.z as f64).normalize();
                let dup64 = glam::DVec3::new(pos_dup.x as f64, pos_dup.y as f64, pos_dup.z as f64).normalize();
                let mut rep_max = f64::NEG_INFINITY;
                let mut dup_max = f64::NEG_INFINITY;
                for g64 in &effective_gens_d {
                    rep_max = rep_max.max(rep64.dot(*g64));
                    dup_max = dup_max.max(dup64.dot(*g64));
                }
                eprintln!("  rep deltas:");
                for &gi in key {
                    let delta = rep_max - rep64.dot(effective_gens_d[gi]);
                    eprintln!("    g{} delta={:.2e}", gi, delta);
                }
                eprintln!("  dup deltas:");
                for &gi in key {
                    let delta = dup_max - dup64.dot(effective_gens_d[gi]);
                    eprintln!("    g{} delta={:.2e}", gi, delta);
                }
            }
        }
        for (v1, v2, key) in &collapsed_samples {
            eprintln!("[strict] collapsed_edge v{}-v{} key={}", v1, v2, key);
        }
        for (v1, v2) in &missing_key_samples {
            eprintln!("[strict] missing_key_edge v{}-v{}", v1, v2);
        }

        if !result.duplicate_support_vertices.is_empty() {
            let knn = CubeMapGridKnn::new(&effective_generators);
            let mut scratch = knn.make_scratch();
            let mut neighbors: Vec<usize> = Vec::with_capacity(DEFAULT_K);

            for &(rep, _dup) in result.duplicate_support_vertices.iter().take(3) {
                let key_id = vertex_key_id.get(rep).copied().unwrap_or(usize::MAX);
                let Some(key) = support_keys.get(key_id) else { continue };
                eprintln!("[strict] support key {:?} neighbor coverage (k={})", key, DEFAULT_K);

                for &gi in key {
                    neighbors.clear();
                    knn.knn_into(
                        effective_generators[gi],
                        gi,
                        DEFAULT_K,
                        &mut scratch,
                        &mut neighbors,
                    );
                    for &gj in key {
                        if gi == gj {
                            continue;
                        }
                        if !neighbors.contains(&gj) {
                            eprintln!("  missing: g{} -> g{}", gi, gj);
                        }
                    }
                }
            }
        }

        if !result.generator_mismatch_vertices.is_empty() {
            eprintln!(
                "[strict] generator_mismatch (eps_lo={:.2e}, eps_hi={:.2e})",
                eps_lo, eps_hi
            );
            for &v_idx in result.generator_mismatch_vertices.iter().take(3) {
                let v = voronoi.vertices[v_idx];
                let v64 = glam::DVec3::new(v.x as f64, v.y as f64, v.z as f64).normalize();
                let mut max_dot = f64::NEG_INFINITY;
                for g64 in &effective_gens_d {
                    let d = v64.dot(*g64);
                    if d > max_dot {
                        max_dot = d;
                    }
                }

                let mut support_lo = 0usize;
                let mut support_hi = 0usize;
                for g64 in &effective_gens_d {
                    let delta = max_dot - v64.dot(*g64);
                    if delta <= eps_hi {
                        support_hi += 1;
                        if delta <= eps_lo {
                            support_lo += 1;
                        }
                    }
                }

                eprintln!(
                    "[strict] v{} max_dot={:.2e} support_lo/hi={}/{} cells={:?}",
                    v_idx,
                    max_dot,
                    support_lo,
                    support_hi,
                    vertex_to_cells.get(v_idx),
                );
                if let Some(cells) = vertex_to_cells.get(v_idx) {
                    for &eff_idx in cells.iter().take(6) {
                        let delta = max_dot - v64.dot(effective_gens_d[eff_idx]);
                        eprintln!("  cell g{} delta={:.2e}", eff_idx, delta);
                    }
                }
            }
        }
    }

    result
}

/// Count unique vertices in a cell (for deduplication analysis)
pub fn count_unique_vertices(voronoi: &SphericalVoronoi, cell_idx: usize, tolerance: f32) -> usize {
    let cell = voronoi.cell(cell_idx);
    let positions: Vec<Vec3> = cell
        .vertex_indices
        .iter()
        .map(|&vi| voronoi.vertices[vi])
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
    let mut new_cell_data: Vec<(usize, Vec<usize>)> = Vec::with_capacity(voronoi.num_cells());

    for cell_idx in 0..voronoi.num_cells() {
        let cell = voronoi.cell(cell_idx);
        let positions: Vec<Vec3> = cell
            .vertex_indices
            .iter()
            .map(|&vi| voronoi.vertices[vi])
            .collect();

        // Build list of unique vertex indices
        let mut unique_indices: Vec<usize> = Vec::new();
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

        new_cell_data.push((cell.generator_index, unique_indices));
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
        let pos1 = voronoi.vertices[v1];
        let pos2 = voronoi.vertices[v2];
        let edge_midpoint = ((pos1 + pos2) / 2.0).normalize();

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
            let cell_has_edge = cell.vertex_indices.windows(2)
                .any(|w| (w[0] == v1 && w[1] == v2) || (w[0] == v2 && w[1] == v1))
                || (cell.vertex_indices.first() == Some(&v1) && cell.vertex_indices.last() == Some(&v2))
                || (cell.vertex_indices.first() == Some(&v2) && cell.vertex_indices.last() == Some(&v1));

            if cell_has_edge {
                continue;
            }

            // Check if cell has vertices at same positions but different indices
            let positions: Vec<Vec3> = cell.vertex_indices.iter()
                .map(|&vi| voronoi.vertices[vi]).collect();

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

/// Analyze why k=16 fails for specific orphan edges.
/// Returns details about which generators are missing from k-NN.
pub fn analyze_knn_coverage(
    points: &[glam::Vec3],
    k: usize,
) -> KnnCoverageAnalysis {
    use super::gpu_voronoi::{build_kdtree, find_k_nearest, compute_voronoi_gpu_style};

    let voronoi = compute_voronoi_gpu_style(points, k);
    let result = validate_voronoi(&voronoi, 1e-6);

    if result.orphan_edges.is_empty() {
        return KnnCoverageAnalysis {
            k,
            num_points: points.len(),
            orphan_edges: 0,
            missing_second_order_neighbors: 0,
            examples: Vec::new(),
        };
    }

    let (tree, entries) = build_kdtree(points);

    // For each orphan edge, find out why it's orphan
    let mut missing_second_order = 0;
    let mut examples: Vec<OrphanEdgeExample> = Vec::new();

    // Build edge-to-cells map
    let mut edge_to_cells: std::collections::HashMap<(usize, usize), Vec<usize>> = std::collections::HashMap::new();
    for cell_idx in 0..voronoi.num_cells() {
        let cell = voronoi.cell(cell_idx);
        let n = cell.vertex_indices.len();
        for i in 0..n {
            let v1 = cell.vertex_indices[i];
            let v2 = cell.vertex_indices[(i + 1) % n];
            let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
            edge_to_cells.entry(edge).or_default().push(cell_idx);
        }
    }

    for &(v1, v2) in result.orphan_edges.iter().take(20) {
        // Find the single cell that has this edge
        let cells = edge_to_cells.get(&(v1, v2)).cloned().unwrap_or_default();
        if cells.len() != 1 {
            continue;
        }
        let cell_a = cells[0];

        // Find the "missing" adjacent cell - which cell SHOULD share this edge?
        // The edge between cells A and B is defined by vertices at circumcenters of
        // triangles that share A and B. We need to find B.

        // Get k-NN for cell A
        let knn_a = find_k_nearest(&tree, &entries, points[cell_a], cell_a, k);

        // The edge vertex positions
        let pos1 = voronoi.vertices[v1];
        let pos2 = voronoi.vertices[v2];

        // Find which generator is closest to the edge midpoint (that's likely cell B)
        let edge_mid = ((pos1 + pos2) / 2.0).normalize();
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for i in 0..points.len() {
            if i == cell_a {
                continue;
            }
            let dist = (points[i] - edge_mid).length();
            candidates.push((i, dist));
        }
        candidates.sort_by(|a, b| a.1.total_cmp(&b.1));

        // The second closest to edge midpoint (after cell_a's generator) is likely the missing cell B
        let cell_b = candidates[0].0;

        // Check: is B in A's k-NN?
        let b_in_a_knn = knn_a.contains(&cell_b);

        // Get k-NN for cell B
        let knn_b = find_k_nearest(&tree, &entries, points[cell_b], cell_b, k);

        // Check: is A in B's k-NN?
        let a_in_b_knn = knn_b.contains(&cell_a);

        // For the edge A-B, we need the two vertex triplets.
        // Find which third generators complete the triplets.
        // These should be the next closest to the edge midpoint after A and B.
        let third_gens: Vec<usize> = candidates.iter()
            .filter(|(i, _)| *i != cell_a && *i != cell_b)
            .take(4)
            .map(|(i, _)| *i)
            .collect();

        // Check if these third generators are in both k-NN lists
        let mut missing_from_a: Vec<usize> = Vec::new();
        let mut missing_from_b: Vec<usize> = Vec::new();

        for &g in &third_gens {
            if !knn_a.contains(&g) {
                missing_from_a.push(g);
            }
            if !knn_b.contains(&g) {
                missing_from_b.push(g);
            }
        }

        if !missing_from_a.is_empty() || !missing_from_b.is_empty() {
            missing_second_order += 1;
        }

        if examples.len() < 5 {
            examples.push(OrphanEdgeExample {
                cell_a,
                cell_b,
                b_in_a_knn,
                a_in_b_knn,
                third_generators: third_gens,
                missing_from_a,
                missing_from_b,
            });
        }
    }

    KnnCoverageAnalysis {
        k,
        num_points: points.len(),
        orphan_edges: result.orphan_edges.len(),
        missing_second_order_neighbors: missing_second_order,
        examples,
    }
}

/// Analysis of k-NN coverage and why it causes orphan edges
#[derive(Debug, Clone)]
pub struct KnnCoverageAnalysis {
    pub k: usize,
    pub num_points: usize,
    pub orphan_edges: usize,
    /// Count of orphan edges caused by missing second-order neighbors
    pub missing_second_order_neighbors: usize,
    /// Example orphan edges with details
    pub examples: Vec<OrphanEdgeExample>,
}

/// Details about a specific orphan edge
#[derive(Debug, Clone)]
pub struct OrphanEdgeExample {
    pub cell_a: usize,
    pub cell_b: usize,
    pub b_in_a_knn: bool,
    pub a_in_b_knn: bool,
    pub third_generators: Vec<usize>,
    pub missing_from_a: Vec<usize>,
    pub missing_from_b: Vec<usize>,
}

/// Configuration for large-count validation.
#[derive(Debug, Clone)]
pub struct LargeValidationConfig {
    /// Number of neighbors to check for support set validation (default: 48).
    /// Higher values catch more issues but are slower.
    pub knn_k: usize,
    /// Fraction of vertices to sample for detailed checks (0.0-1.0).
    /// 1.0 = check all vertices, 0.01 = check 1% of vertices.
    pub vertex_sample_rate: f64,
    /// Absolute epsilon for support set membership in dot space.
    pub eps_abs: f64,
    /// Random seed for reproducible sampling.
    pub seed: u64,
}

impl Default for LargeValidationConfig {
    fn default() -> Self {
        use crate::geometry::gpu_voronoi::SUPPORT_EPS_ABS;
        Self {
            knn_k: 48,
            vertex_sample_rate: 1.0, // Check all by default
            eps_abs: SUPPORT_EPS_ABS,
            seed: 12345,
        }
    }
}

impl LargeValidationConfig {
    /// Create config for fast validation (1% sampling).
    pub fn fast() -> Self {
        Self {
            knn_k: 32,
            vertex_sample_rate: 0.01,
            ..Default::default()
        }
    }

    /// Create config for thorough validation (10% sampling).
    pub fn thorough() -> Self {
        Self {
            knn_k: 48,
            vertex_sample_rate: 0.10,
            ..Default::default()
        }
    }

    /// Create config for exhaustive validation (all vertices, high k).
    pub fn exhaustive() -> Self {
        Self {
            knn_k: 64,
            vertex_sample_rate: 1.0,
            ..Default::default()
        }
    }
}

/// Result of large-count validation.
#[derive(Debug, Clone, Default)]
pub struct LargeValidationResult {
    // Topology checks (fast, always complete)
    pub num_cells: usize,
    pub num_vertices: usize,
    pub euler_v: usize,
    pub euler_e: usize,
    pub euler_f: usize,
    pub euler_ok: bool,
    pub degenerate_cells: usize,
    pub orphan_edges: usize,
    pub overcounted_edges: usize,
    pub total_area: f64,
    pub area_error_pct: f64,

    // Sampled geometric checks
    pub vertices_sampled: usize,
    pub sample_rate: f64,

    /// Vertices where support set has < 3 generators within k-NN.
    pub support_lt3: usize,
    /// Vertices where assigned cell's generator is not in support set.
    pub generator_mismatch: usize,
    /// Maximum delta (max_dot - cell_gen_dot) observed.
    pub max_generator_delta: f64,
    /// Vertices flagged as ambiguous (support set boundary cases).
    pub ambiguous_vertices: usize,

    // Timing
    pub topology_time_ms: f64,
    pub sampling_time_ms: f64,
}

impl LargeValidationResult {
    /// Check if topological structure is valid.
    pub fn topology_valid(&self) -> bool {
        self.euler_ok && self.degenerate_cells == 0 && self.orphan_edges == 0 && self.overcounted_edges == 0
    }

    /// Check if geometric accuracy is acceptable.
    /// Returns true if no generator mismatches and support_lt3 rate is low.
    pub fn geometry_valid(&self, support_lt3_threshold: f64) -> bool {
        if self.vertices_sampled == 0 {
            return true;
        }
        let support_lt3_rate = self.support_lt3 as f64 / self.vertices_sampled as f64;
        self.generator_mismatch == 0 && support_lt3_rate <= support_lt3_threshold
    }

    /// Overall validity check.
    pub fn is_valid(&self) -> bool {
        self.topology_valid() && self.geometry_valid(0.05)
    }

    pub fn print_summary(&self) {
        println!("Large-Count Validation Results:");
        println!("  Cells: {}, Vertices: {}", self.num_cells, self.num_vertices);
        println!(
            "  Euler: V={} E={} F={} (V-E+F={}, ok={})",
            self.euler_v,
            self.euler_e,
            self.euler_f,
            self.euler_v as i64 - self.euler_e as i64 + self.euler_f as i64,
            self.euler_ok
        );
        println!(
            "  Area: {:.4} (error {:.2}%)",
            self.total_area, self.area_error_pct
        );
        println!(
            "  Topology: degenerate={}, orphan={}, overcounted={}",
            self.degenerate_cells, self.orphan_edges, self.overcounted_edges
        );
        println!(
            "  Sampled: {} vertices ({:.1}%)",
            self.vertices_sampled,
            self.sample_rate * 100.0
        );
        if self.vertices_sampled > 0 {
            let support_lt3_rate = self.support_lt3 as f64 / self.vertices_sampled as f64 * 100.0;
            println!(
                "  Geometry: support_lt3={} ({:.2}%), gen_mismatch={}, ambiguous={}, max_delta={:.2e}",
                self.support_lt3, support_lt3_rate, self.generator_mismatch, self.ambiguous_vertices, self.max_generator_delta
            );
        }
        println!(
            "  Time: topology={:.1}ms, sampling={:.1}ms",
            self.topology_time_ms, self.sampling_time_ms
        );
        println!(
            "  Status: {}",
            if self.is_valid() { "VALID" } else { "INVALID" }
        );
    }
}

/// Validate a Voronoi diagram for large point counts (100k+).
///
/// This function is optimized for large counts by:
/// 1. Using fast O(V+E+F) topological checks (Euler, edge consistency)
/// 2. Using k-NN to accelerate support set validation (O(V*k) instead of O(V*N))
/// 3. Supporting vertex sampling to reduce validation time
///
/// For small counts (<10k), use `validate_voronoi_strict` instead.
pub fn validate_voronoi_large(
    voronoi: &SphericalVoronoi,
    config: &LargeValidationConfig,
) -> LargeValidationResult {
    use crate::geometry::gpu_voronoi::{CubeMapGridKnn, KnnProvider};
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::time::Instant;

    let mut result = LargeValidationResult {
        num_cells: voronoi.num_cells(),
        num_vertices: voronoi.vertices.len(),
        sample_rate: config.vertex_sample_rate,
        ..Default::default()
    };

    // Phase 1: Fast topology checks
    let t0 = Instant::now();

    let mut edge_to_cells: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut unique_vertices: HashSet<usize> = HashSet::new();

    for cell_idx in 0..voronoi.num_cells() {
        let cell = voronoi.cell(cell_idx);
        let vertex_indices = cell.vertex_indices;

        // Check for degenerate cells
        if vertex_indices.len() < 3 {
            result.degenerate_cells += 1;
            continue;
        }

        // Track vertices and edges
        for &vi in vertex_indices {
            unique_vertices.insert(vi);
        }

        for i in 0..vertex_indices.len() {
            let v1 = vertex_indices[i];
            let v2 = vertex_indices[(i + 1) % vertex_indices.len()];
            if v1 == v2 {
                continue;
            }
            let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
            edge_to_cells.entry(edge).or_default().push(cell_idx);
        }

        // Compute cell area
        let positions: Vec<Vec3> = vertex_indices
            .iter()
            .map(|&vi| voronoi.vertices[vi])
            .collect();
        result.total_area += spherical_polygon_area(&positions);
    }

    // Euler characteristic
    result.euler_v = unique_vertices.len();
    result.euler_e = edge_to_cells.len();
    result.euler_f = voronoi.num_cells();
    let euler_sum = result.euler_v as i64 - result.euler_e as i64 + result.euler_f as i64;
    result.euler_ok = euler_sum == 2;

    // Edge consistency
    for (_edge, cells) in &edge_to_cells {
        match cells.len() {
            1 => result.orphan_edges += 1,
            2 => {}
            _ => result.overcounted_edges += 1,
        }
    }

    // Area error
    let expected_area = 4.0 * std::f64::consts::PI;
    result.area_error_pct = (result.total_area - expected_area).abs() / expected_area * 100.0;

    result.topology_time_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Phase 2: Sampled geometric checks with k-NN acceleration
    let t1 = Instant::now();

    // Build vertex -> cells mapping
    let mut vertex_to_cells: Vec<Vec<usize>> = vec![Vec::new(); voronoi.vertices.len()];
    for cell_idx in 0..voronoi.num_cells() {
        let cell = voronoi.cell(cell_idx);
        for &vi in cell.vertex_indices {
            if vi < vertex_to_cells.len() && !vertex_to_cells[vi].contains(&cell_idx) {
                vertex_to_cells[vi].push(cell_idx);
            }
        }
    }

    // Collect referenced vertices
    let referenced_vertices: Vec<usize> = unique_vertices.into_iter().collect();

    // Sample vertices
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let sample_size = ((referenced_vertices.len() as f64 * config.vertex_sample_rate).ceil() as usize)
        .max(1)
        .min(referenced_vertices.len());

    let mut sampled_indices = referenced_vertices.clone();
    if sample_size < referenced_vertices.len() {
        sampled_indices.shuffle(&mut rng);
        sampled_indices.truncate(sample_size);
    }
    result.vertices_sampled = sampled_indices.len();

    // Build k-NN for generator lookup
    let knn = CubeMapGridKnn::new(&voronoi.generators);
    let mut scratch = knn.make_scratch();
    let mut neighbors: Vec<usize> = Vec::with_capacity(config.knn_k);

    // Use an absolute epsilon in dot space (based on numeric precision).
    let eps = config.eps_abs;

    // Convert generators to f64 for precision
    let generators_d: Vec<glam::DVec3> = voronoi
        .generators
        .iter()
        .map(|g| glam::DVec3::new(g.x as f64, g.y as f64, g.z as f64).normalize())
        .collect();

    for &v_idx in &sampled_indices {
        let v = voronoi.vertices[v_idx];
        let v64 = glam::DVec3::new(v.x as f64, v.y as f64, v.z as f64).normalize();

        // Get k-NN generators for this vertex position
        // Use any nearby cell's generator as query point (vertex should be near its cells' generators)
        let query_gen = if !vertex_to_cells[v_idx].is_empty() {
            voronoi.generators[vertex_to_cells[v_idx][0]]
        } else {
            v // Fallback to vertex position
        };

        neighbors.clear();
        knn.knn_into(query_gen, usize::MAX, config.knn_k, &mut scratch, &mut neighbors);

        // Find max dot product among k-NN neighbors
        let mut max_dot = f64::NEG_INFINITY;
        for &ni in &neighbors {
            let d = v64.dot(generators_d[ni]);
            if d > max_dot {
                max_dot = d;
            }
        }

        // Also check actual cell generators (they should be in k-NN, but verify)
        for &cell_idx in &vertex_to_cells[v_idx] {
            let d = v64.dot(generators_d[cell_idx]);
            if d > max_dot {
                max_dot = d;
            }
        }

        // Build support set from k-NN
        let mut support_count = 0;
        let mut cell_gen_in_support = true;
        let mut has_ambiguous = false;

        for &ni in &neighbors {
            let delta = max_dot - v64.dot(generators_d[ni]);
            if delta <= eps {
                support_count += 1;
            } else if delta <= eps * 2.0 {
                has_ambiguous = true;
            }
        }

        // Check if cell generators are in support set
        for &cell_idx in &vertex_to_cells[v_idx] {
            let delta = max_dot - v64.dot(generators_d[cell_idx]);
            if delta > result.max_generator_delta {
                result.max_generator_delta = delta;
            }
            if delta > eps {
                cell_gen_in_support = false;
            }
        }

        if support_count < 3 {
            result.support_lt3 += 1;
        }
        if !cell_gen_in_support {
            result.generator_mismatch += 1;
        }
        if has_ambiguous {
            result.ambiguous_vertices += 1;
        }
    }

    result.sampling_time_ms = t1.elapsed().as_secs_f64() * 1000.0;

    result
}

/// Convenience function for quick validation of large diagrams.
pub fn validate_voronoi_large_quick(voronoi: &SphericalVoronoi) -> (bool, String) {
    let config = if voronoi.num_cells() > 100_000 {
        LargeValidationConfig::fast()
    } else {
        LargeValidationConfig::thorough()
    };
    let result = validate_voronoi_large(voronoi, &config);

    if result.is_valid() {
        (true, format!("Valid (sampled {:.1}%)", result.sample_rate * 100.0))
    } else {
        let mut issues = Vec::new();
        if !result.euler_ok {
            issues.push("euler_fail".to_string());
        }
        if result.degenerate_cells > 0 {
            issues.push(format!("{} degenerate", result.degenerate_cells));
        }
        if result.orphan_edges > 0 {
            issues.push(format!("{} orphan_edges", result.orphan_edges));
        }
        if result.generator_mismatch > 0 {
            issues.push(format!("{} gen_mismatch", result.generator_mismatch));
        }
        let support_lt3_rate = if result.vertices_sampled > 0 {
            result.support_lt3 as f64 / result.vertices_sampled as f64
        } else {
            0.0
        };
        if support_lt3_rate > 0.05 {
            issues.push(format!("support_lt3 {:.1}%", support_lt3_rate * 100.0));
        }
        (false, issues.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{
        compute_voronoi_gpu_style, fibonacci_sphere_points_with_rng, SphericalVoronoi,
    };
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
    fn test_gpu_voronoi_k24_validity() {
        let points = generate_test_points(1000, 12345);
        let voronoi = compute_voronoi_gpu_style(&points, 24);
        let result = validate_voronoi(&voronoi, 1e-6);

        result.print_summary();
        assert!(
            result.degenerate_cells.is_empty(),
            "GPU k=24 should not produce degenerate cells"
        );
    }

    #[test]
    fn test_gpu_voronoi_k12_validity() {
        let points = generate_test_points(1000, 12345);
        let voronoi = compute_voronoi_gpu_style(&points, 12);
        let result = validate_voronoi(&voronoi, 1e-6);

        result.print_summary();
        // k=12 might have some issues at this scale
        println!(
            "k=12 validity: {} degenerate, {} near-dup cells",
            result.degenerate_cells.len(),
            result.near_duplicate_cells.len()
        );
    }

    #[test]
    fn test_deduplicate_voronoi() {
        let points = generate_test_points(1000, 12345);
        let voronoi = compute_voronoi_gpu_style(&points, 24);

        let before = validate_voronoi(&voronoi, 1e-4);
        let deduped = deduplicate_voronoi(&voronoi, 1e-4);
        let after = validate_voronoi(&deduped, 1e-4);

        println!("Before dedup: {} near-dup cells", before.near_duplicate_cells.len());
        println!("After dedup: {} near-dup cells", after.near_duplicate_cells.len());

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

    #[test]
    #[ignore] // Run with: cargo test validation_at_scale -- --ignored --nocapture
    fn validation_at_scale() {
        println!("\n=== Voronoi Validation at Scale ===\n");

        for &n in &[1000, 10000, 50000] {
            println!("--- n = {} ---", n);
            let points = generate_test_points(n, 12345);

            // Test hull method
            let hull = SphericalVoronoi::compute(&points);
            let hull_result = validate_voronoi(&hull, 1e-6);
            println!("Hull method:");
            hull_result.print_summary();

            // Test GPU method at various k
            for k in [12, 16, 24, 32] {
                let gpu = compute_voronoi_gpu_style(&points, k);
                let gpu_result = validate_voronoi(&gpu, 1e-6);
                let (valid, msg) = validate_voronoi_quick(&gpu);
                println!("GPU k={}: {} - {}", k, if valid { "VALID" } else { "INVALID" }, msg);
            }
            println!();
        }
    }

    #[test]
    #[ignore] // Run with: cargo test analyze_orphan_edge_causes -- --ignored --nocapture
    fn analyze_orphan_edge_causes() {
        println!("\n=== Orphan Edge Analysis ===\n");

        for &n in &[1000, 10000] {
            println!("--- n = {} ---", n);
            let points = generate_test_points(n, 12345);

            for k in [16, 24, 32] {
                let gpu = compute_voronoi_gpu_style(&points, k);
                let analysis = analyze_orphan_edges(&gpu, 1e-4);

                if analysis.total_orphan_edges > 0 {
                    println!("GPU k={}:", k);
                    println!("  Total orphan edges: {}", analysis.total_orphan_edges);
                    println!("  Position mismatch (same pos, diff idx): {}", analysis.position_mismatch_edges);
                    println!("  True orphans (no adjacent cell): {}", analysis.true_orphan_edges);
                } else {
                    println!("GPU k={}: No orphan edges", k);
                }
            }
            println!();
        }
    }

    #[test]
    #[ignore] // Run with: cargo test investigate_over_edges -- --ignored --nocapture
    fn investigate_over_edges() {
        println!("\n=== Over-edge Investigation ===\n");

        let n = 50000;
        let points = generate_test_points(n, 12345);
        let gpu = compute_voronoi_gpu_style(&points, 24);
        let result = validate_voronoi(&gpu, 1e-6);

        println!("Orphan edges: {}", result.orphan_edges.len());
        println!("Overcounted edges: {}", result.overcounted_edges.len());

        // Examine overcounted edges
        for &(v1, v2, count) in &result.overcounted_edges {
            let pos1 = gpu.vertices[v1];
            let pos2 = gpu.vertices[v2];
            println!("\nEdge ({}, {}) appears in {} cells:", v1, v2, count);
            println!("  v1: ({:.6}, {:.6}, {:.6})", pos1.x, pos1.y, pos1.z);
            println!("  v2: ({:.6}, {:.6}, {:.6})", pos2.x, pos2.y, pos2.z);

            // Find which cells contain this edge
            for cell_idx in 0..gpu.num_cells() {
                let cell = gpu.cell(cell_idx);
                let indices = cell.vertex_indices;
                let n = indices.len();
                for i in 0..n {
                    let a = indices[i];
                    let b = indices[(i + 1) % n];
                    if (a == v1 && b == v2) || (a == v2 && b == v1) {
                        println!("  Cell {} has this edge (gen: {:?})", cell_idx, gpu.generators[cell_idx]);
                    }
                }
            }
        }
    }

    #[test]
    #[ignore] // Run with: cargo test termination_effectiveness -- --ignored --nocapture
    fn termination_effectiveness() {
        use crate::geometry::compute_voronoi_gpu_style_with_stats;

        println!("\n=== Early Termination Effectiveness ===\n");

        let n = 50000;
        let points = generate_test_points(n, 12345);

        for k in [16, 24, 32, 48] {
            let (voronoi, stats) = compute_voronoi_gpu_style_with_stats(&points, k);
            let (valid, msg) = validate_voronoi_quick(&voronoi);

            println!("k={:2}: avg neighbors processed: {:.1}, termination rate: {:.1}%, valid: {} ({})",
                k,
                stats.avg_neighbors_processed,
                stats.termination_rate * 100.0,
                valid,
                msg);
        }

        println!("\nConclusion: With early termination, higher k has diminishing cost impact.");
    }

    #[test]
    #[ignore] // Run with: cargo test bench_termination_frequency -- --ignored --nocapture
    fn bench_termination_frequency() {
        use crate::geometry::gpu_voronoi::compute_voronoi_gpu_style_timed_with_termination_params;
        use std::time::Instant;

        println!("\n=== Termination Check Frequency Benchmark ===\n");

        let n = 500_000;
        let points = generate_test_points(n, 12345);
        let k = 24;
        let iterations = 3;

        println!("n={}, k={}, {} iterations each\n", n, k, iterations);

        // Compare old default (8,2) with new (10,6)
        for (start, step) in [(8, 2), (10, 6)] {
            let mut times = Vec::new();
            for _ in 0..iterations {
                let t0 = Instant::now();
                let _ = compute_voronoi_gpu_style_timed_with_termination_params(
                    &points, k, false, step > 0, start, step
                );
                times.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            let avg = times.iter().sum::<f64>() / times.len() as f64;
            let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
            println!("({:2}, {:1}): avg={:.0}ms, min={:.0}ms, runs={:?}",
                start, step, avg, min, times.iter().map(|t| *t as u32).collect::<Vec<_>>());
        }
    }

    #[test]
    #[ignore] // Run with: cargo test test_incremental_voronoi_validation -- --ignored --nocapture
    fn test_incremental_voronoi_validation() {
        use crate::geometry::gpu_voronoi::{
            build_cells_data_flat, CubeMapGridKnn, TerminationConfig,
            dedup::dedup_vertices_hash_flat,
        };

        println!("\n=== Incremental Voronoi Full Validation ===\n");

        let n = 10_000;
        let k = 24;
        let points = generate_test_points(n, 12345);
        let knn = CubeMapGridKnn::new(&points);

        let termination = TerminationConfig {
            enabled: true,
            check_start: 10,
            check_step: 6,
        };

        // Build with flat buffers (already produces CCW-ordered vertices)
        let flat_data = build_cells_data_flat(&points, &knn, k, termination);

        // Dedup vertices
        let (vertices, cells, cell_indices) = dedup_vertices_hash_flat(flat_data, false);

        // Build SphericalVoronoi
        let voronoi = crate::geometry::SphericalVoronoi::from_raw_parts(
            points.to_vec(),
            vertices,
            cells,
            cell_indices,
        );

        // Validate
        let result = validate_voronoi(&voronoi, 1e-5);
        println!("Validation result: {}", if result.is_valid() { "VALID" } else { "INVALID" });
        println!("  Degenerate cells: {}", result.degenerate_cells.len());
        println!("  Orphan edges: {}", result.orphan_edges.len());
        println!("  Duplicate index cells: {}", result.duplicate_index_cells.len());
        println!("  Near-duplicate cells: {}", result.near_duplicate_cells.len());
    }

    #[test]
    #[ignore] // Run with: cargo test why_k16_fails -- --ignored --nocapture
    fn why_k16_fails() {
        println!("\n=== Why k=16 Fails: Second-Order Neighbor Analysis ===\n");

        let n = 50000;
        let points = generate_test_points(n, 12345);

        for k in [12, 16, 20, 24] {
            let analysis = analyze_knn_coverage(&points, k);
            println!("k={}: {} orphan edges", k, analysis.orphan_edges);

            if analysis.orphan_edges > 0 {
                println!("  Missing second-order neighbors: {}/{} analyzed",
                    analysis.missing_second_order_neighbors,
                    analysis.orphan_edges.min(20));

                for (i, ex) in analysis.examples.iter().enumerate() {
                    println!("\n  Example {}:", i + 1);
                    println!("    Cell A={}, Cell B={}", ex.cell_a, ex.cell_b);
                    println!("    B in A's k-NN: {}, A in B's k-NN: {}", ex.b_in_a_knn, ex.a_in_b_knn);
                    println!("    Third generators (edge vertices): {:?}", ex.third_generators);
                    println!("    Missing from A's k-NN: {:?}", ex.missing_from_a);
                    println!("    Missing from B's k-NN: {:?}", ex.missing_from_b);
                }
            }
            println!();
        }
    }

    #[test]
    #[ignore] // Run with: cargo test compare_hull_gpu_edges -- --ignored --nocapture
    fn compare_hull_gpu_edges() {
        println!("\n=== Hull vs GPU Edge Comparison ===\n");

        let n = 1000;
        let points = generate_test_points(n, 12345);

        let hull = SphericalVoronoi::compute(&points);
        let gpu = compute_voronoi_gpu_style(&points, 24);

        // Count edges in each
        let count_edges = |voronoi: &SphericalVoronoi| -> usize {
            let mut edges = std::collections::HashSet::new();
            for cell_idx in 0..voronoi.num_cells() {
                let cell = voronoi.cell(cell_idx);
                let n = cell.vertex_indices.len();
                for i in 0..n {
                    let v1 = cell.vertex_indices[i];
                    let v2 = cell.vertex_indices[(i + 1) % n];
                    let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
                    edges.insert(edge);
                }
            }
            edges.len()
        };

        println!("Hull: {} unique edges, {} vertices", count_edges(&hull), hull.vertices.len());
        println!("GPU:  {} unique edges, {} vertices", count_edges(&gpu), gpu.vertices.len());

        // The issue might be that GPU creates duplicate vertices at the same position
        // with different indices, so edges aren't properly shared
        let hull_result = validate_voronoi(&hull, 1e-6);
        let gpu_result = validate_voronoi(&gpu, 1e-6);

        println!("\nHull validation:");
        hull_result.print_summary();
        println!("\nGPU validation:");
        gpu_result.print_summary();
    }
}
