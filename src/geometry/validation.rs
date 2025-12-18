//! Validation utilities for spherical Voronoi diagrams.
//!
//! These functions check for grid validity issues that could cause problems
//! in downstream processing (adjacency, rivers, rendering, etc.)

use super::SphericalVoronoi;
use glam::Vec3;
use std::collections::HashMap;

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
}

impl ValidationResult {
    /// Check if the grid is valid (no errors)
    pub fn is_valid(&self) -> bool {
        self.degenerate_cells.is_empty()
            && self.duplicate_index_cells.is_empty()
            && self.near_duplicate_cells.is_empty()
            && self.orphan_edges.is_empty()
            && self.overcounted_edges.is_empty()
            && self.generator_outside_cells.is_empty()
    }

    /// Total number of issues found
    pub fn issue_count(&self) -> usize {
        self.degenerate_cells.len()
            + self.duplicate_index_cells.len()
            + self.near_duplicate_cells.len()
            + self.orphan_edges.len()
            + self.overcounted_edges.len()
            + self.generator_outside_cells.len()
    }

    /// Print a summary of validation results
    pub fn print_summary(&self) {
        println!("Voronoi Validation Results:");
        println!("  Total cells: {}", self.num_cells);

        if self.is_valid() {
            println!("  Status: VALID ✓");
            return;
        }

        println!("  Status: INVALID");
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
                "  Cells with near-duplicate vertices: {}",
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
        // (For spherical Voronoi, generator should be closer to cell center than to any vertex)
        // This is a weak check - just verify generator is closer to cell centroid than any neighbor
        // Skip for now as it's expensive and less critical
    }

    // Check edge consistency
    for (edge, cells) in &edge_to_cells {
        match cells.len() {
            1 => result.orphan_edges.push(*edge),
            2 => {} // Correct: each edge should be shared by exactly 2 cells
            n => result.overcounted_edges.push((edge.0, edge.1, n)),
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
    use glam::Vec3;

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
            build_cells_data_incremental, CubeMapGridKnn, TerminationConfig,
            dedup_vertices_hash,
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

        // Build with incremental (already produces CCW-ordered vertices)
        let (cells_data, _degenerate_triplets) = build_cells_data_incremental(&points, &knn, k, termination);

        // Dedup vertices
        let (vertices, cells, cell_indices) = dedup_vertices_hash(n, cells_data);

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
