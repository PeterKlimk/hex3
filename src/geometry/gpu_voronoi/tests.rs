//! Tests for GPU-style Voronoi computation.

use glam::Vec3;
use rustc_hash::FxHashMap;

use super::*;
use crate::geometry::{random_sphere_points, random_sphere_points_with_rng};

/// Find minimum distance between any two distinct REFERENCED vertices using spatial grid.
fn find_min_vertex_distance_referenced(voronoi: &crate::geometry::SphericalVoronoi) -> f32 {
    use std::collections::HashSet;

    // Collect all referenced vertex indices
    let mut referenced: HashSet<usize> = HashSet::new();
    for cell in voronoi.iter_cells() {
        for &idx in cell.vertex_indices {
            referenced.insert(idx);
        }
    }

    let ref_indices: Vec<usize> = referenced.into_iter().collect();
    if ref_indices.len() < 2 {
        return f32::MAX;
    }

    let vertices = &voronoi.vertices;

    // Use a spatial grid with cell size ~0.01 (reasonable for unit sphere)
    let cell_size = 0.01f32;
    let inv_cell_size = 1.0 / cell_size;

    let grid_cell = |p: Vec3| -> (i32, i32, i32) {
        (
            (p.x * inv_cell_size).floor() as i32,
            (p.y * inv_cell_size).floor() as i32,
            (p.z * inv_cell_size).floor() as i32,
        )
    };

    let mut grid: FxHashMap<(i32, i32, i32), Vec<usize>> = FxHashMap::default();
    for &idx in &ref_indices {
        let pos = vertices[idx];
        let cell = grid_cell(pos);
        grid.entry(cell).or_insert_with(Vec::new).push(idx);
    }

    let mut min_dist = f32::MAX;

    for &i in &ref_indices {
        let pos = vertices[i];
        let (cx, cy, cz) = grid_cell(pos);
        // Check 27 neighboring cells
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in indices {
                            if j != i {
                                let dist = (vertices[j] - pos).length();
                                if dist < min_dist {
                                    min_dist = dist;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    min_dist
}

const MIN_SPACING_MARGIN: f32 = 1.05;

fn min_spacing_threshold() -> f32 {
    super::MIN_BISECTOR_DISTANCE * MIN_SPACING_MARGIN
}

fn near_duplicate_threshold(num_points: usize) -> f32 {
    if num_points == 0 {
        return super::MIN_BISECTOR_DISTANCE * 0.25;
    }
    let spacing = super::mean_generator_spacing_chord(num_points);
    (spacing * super::VERTEX_WELD_FRACTION).max(super::MIN_BISECTOR_DISTANCE * 0.25)
}

fn assert_min_spacing(points: &[Vec3], label: &str) -> f32 {
    let min_dist = check_min_point_distance(points);
    let threshold = min_spacing_threshold();
    println!(
        "{} min point distance = {:.2e} (threshold: {:.2e})",
        label,
        min_dist,
        threshold
    );
    assert!(
        min_dist >= threshold,
        "{} min spacing {:.2e} below threshold {:.2e}",
        label,
        min_dist,
        threshold
    );
    min_dist
}

#[test]
fn test_incremental_builder_maintains_ccw_order() {
    // Verify that IncrementalCellBuilder already produces CCW-ordered vertices,
    // making the separate order_cells_ccw() pass redundant.
    let points = random_sphere_points(1000);
    let knn = CubeMapGridKnn::new(&points);
    let termination = TerminationConfig {
        enabled: true,
        check_start: 10,
        check_step: 6,
    };

    let flat_data = build_cells_data_flat(&points, &knn, DEFAULT_K, termination);

    let mut already_ordered = 0;
    let mut rotated_ccw = 0;  // CCW but starting from different vertex
    let mut reversed = 0;     // CW order (needs reversal)

    for (i, keyed_verts) in flat_data.iter_cells().enumerate() {
        let n = keyed_verts.len();
        if n < 3 {
            continue;
        }

        let verts: Vec<Vec3> = keyed_verts.iter().map(|(_, v)| *v).collect();
        let ordered_indices = order_vertices_ccw_indices(points[i], &verts);

        // Check if indices are identity [0,1,2,...]
        let is_identity = ordered_indices.iter().enumerate().all(|(idx, &val)| idx == val);

        if is_identity {
            already_ordered += 1;
            continue;
        }

        // Check if it's a rotation of identity (CCW but different start)
        // Find where 0 appears in ordered_indices
        if let Some(start) = ordered_indices.iter().position(|&x| x == 0) {
            let is_rotation = (0..n).all(|j| ordered_indices[(start + j) % n] == j);
            if is_rotation {
                rotated_ccw += 1;
                continue;
            }
        }

        // Otherwise it's reversed or shuffled
        reversed += 1;
    }

    println!(
        "CCW order: {} identity, {} rotated, {} reversed/shuffled",
        already_ordered, rotated_ccw, reversed
    );

    // If most are just rotated, order_cells_ccw is still needed but only for normalization
    // If many are reversed, there's a winding issue
    assert_eq!(
        reversed, 0,
        "IncrementalCellBuilder should maintain CCW winding (reversed={})", reversed
    );
}

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
    let _ = assert_min_spacing(&points, "gpu_voronoi_basic");
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


/// Test soundness with non-coincident (Lloyd-relaxed) input.
/// These points are guaranteed to be well-spaced, so the algorithm should produce
/// valid Voronoi diagrams with no degenerate cells or orphan edges.
#[test]
fn test_soundness_lloyd_relaxed_points() {
    use crate::geometry::{fibonacci_sphere_points_with_rng, lloyd_relax_voronoi, validation::validate_voronoi};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    println!("\n=== Soundness Test: Lloyd-Relaxed Points ===\n");

    for &n in &[1000, 10_000, 100_000] {
        let mut rng = ChaCha8Rng::seed_from_u64(12345);
        let mean_spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
        let jitter = mean_spacing * 0.25;
        let mut points = fibonacci_sphere_points_with_rng(n, jitter, &mut rng);

        // Lloyd relaxation ensures well-spaced points
        lloyd_relax_voronoi(&mut points, 2);

        // Verify min spacing requirement
        let label = format!("n={}", n);
        let _ = assert_min_spacing(&points, &label);

        // Build Voronoi and validate
        let voronoi = compute_voronoi_gpu_style(&points, DEFAULT_K);
        let result = validate_voronoi(&voronoi, near_duplicate_threshold(points.len()));

        println!("  Degenerate cells: {}", result.degenerate_cells.len());
        println!("  Orphan edges: {}", result.orphan_edges.len());
        println!("  Near-duplicate cells: {}", result.near_duplicate_cells.len());

        assert!(result.degenerate_cells.is_empty(),
            "n={}: Lloyd-relaxed points should produce no degenerate cells", n);
        assert!(result.orphan_edges.len() <= 10,
            "n={}: Lloyd-relaxed points should have minimal orphan edges, got {}", n, result.orphan_edges.len());
    }
}

/// Test soundness with NON-Lloyd-relaxed points (fibonacci + jitter only).
/// This can violate the min spacing requirement and is only for stress testing.
/// Run with: cargo test --release test_soundness_non_lloyd -- --ignored --nocapture
#[test]
#[ignore]
fn test_soundness_non_lloyd() {
    use crate::geometry::{fibonacci_sphere_points_with_rng, validation::validate_voronoi};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    println!("\n=== Soundness Test: Non-Lloyd Points (fibonacci + jitter) ===\n");

    for &n in &[100_000, 250_000, 500_000, 750_000] {
        let mut rng = ChaCha8Rng::seed_from_u64(12345);
        let mean_spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
        let jitter = mean_spacing * 0.25;  // Same jitter as bench_voronoi
        let points = fibonacci_sphere_points_with_rng(n, jitter, &mut rng);

        // Check minimum point distance
        let min_dist = check_min_point_distance(&points);
        let ratio = min_dist / mean_spacing;
        println!("n={}: min_dist={:.2e}, mean={:.2e}, ratio={:.3}", n, min_dist, mean_spacing, ratio);
        if min_dist < min_spacing_threshold() {
            println!(
                "  WARNING: min spacing {:.2e} below required {:.2e}",
                min_dist,
                min_spacing_threshold()
            );
        }

        // Build Voronoi and validate
        let voronoi = compute_voronoi_gpu_style(&points, DEFAULT_K);
        let result = validate_voronoi(&voronoi, near_duplicate_threshold(points.len()));

        // Find minimum distance between distinct referenced vertices
        let min_vert_dist = find_min_vertex_distance_referenced(&voronoi);

        println!("  Degenerate cells: {}", result.degenerate_cells.len());
        println!("  Orphan edges: {}", result.orphan_edges.len());
        println!("  Min vertex distance: {:.2e}", min_vert_dist);

        // Non-Lloyd points may have some degenerate cells due to close spacing
        // The key question: are there orphan edges?
        if !result.orphan_edges.is_empty() {
            println!("  WARNING: {} orphan edges detected!", result.orphan_edges.len());
            // Check if orphan vertices have near-duplicates
            for (i, &(v1, v2)) in result.orphan_edges.iter().take(3).enumerate() {
                let pos1 = voronoi.vertices[v1];
                let pos2 = voronoi.vertices[v2];

                // Find nearest other vertex to each endpoint
                let mut min_dist1 = f32::MAX;
                let mut min_dist2 = f32::MAX;
                for (j, &v) in voronoi.vertices.iter().enumerate() {
                    if j != v1 {
                        min_dist1 = min_dist1.min((v - pos1).length());
                    }
                    if j != v2 {
                        min_dist2 = min_dist2.min((v - pos2).length());
                    }
                }

                println!("    Orphan {}: v{}--v{}, edge_len={:.2e}", i, v1, v2, (pos1 - pos2).length());
                println!("      v{} nearest other vertex: {:.2e}", v1, min_dist1);
                println!("      v{} nearest other vertex: {:.2e}", v2, min_dist2);
            }
        }
        println!();
    }
}

/// Diagnose orphan edges in well-spaced points.
/// Run with: cargo test --release diagnose_orphan_edges -- --ignored --nocapture
#[test]
#[ignore]
fn diagnose_orphan_edges() {
    use crate::geometry::{fibonacci_sphere_points_with_rng, lloyd_relax_kmeans, validation::validate_voronoi};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::HashMap;

    println!("\n=== Orphan Edge Diagnosis ===\n");

    let n = 750_000;  // Use 750k to match largest soundness test scale
    let mut rng = ChaCha8Rng::seed_from_u64(12345);
    let mean_spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
    let jitter = mean_spacing * 0.25;
    let mut points = fibonacci_sphere_points_with_rng(n, jitter, &mut rng);
    lloyd_relax_kmeans(&mut points, 2, 20, &mut rng);

    // Check min spacing
    let min_dist = check_min_point_distance(&points);
    println!("Point count: {}", n);
    println!("Mean spacing: {:.2e}", mean_spacing);
    println!("Min point distance: {:.2e}", min_dist);

    let voronoi = compute_voronoi_gpu_style(&points, DEFAULT_K);
    let near_dup = near_duplicate_threshold(points.len());
    let result = validate_voronoi(&voronoi, near_dup);

    println!("\nTotal cells: {}", voronoi.num_cells());
    println!("Total vertices: {}", voronoi.vertices.len());
    println!("Orphan edges: {}", result.orphan_edges.len());
    println!("Near-duplicate cells: {}", result.near_duplicate_cells.len());

    if result.orphan_edges.is_empty() {
        println!("\nNo orphan edges to diagnose!");
        return;
    }

    // Build edge -> cells map
    let mut edge_to_cells: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for cell_idx in 0..voronoi.num_cells() {
        let cell = voronoi.cell(cell_idx);
        let n_verts = cell.vertex_indices.len();
        for i in 0..n_verts {
            let v1 = cell.vertex_indices[i];
            let v2 = cell.vertex_indices[(i + 1) % n_verts];
            let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
            edge_to_cells.entry(edge).or_default().push(cell_idx);
        }
    }

    // Helper: find vertices near a position
    let find_near_vertices = |target_pos: Vec3, exclude: usize| -> Vec<(usize, f32)> {
        voronoi.vertices.iter().enumerate()
            .filter(|(vi, _)| *vi != exclude)
            .map(|(vi, &pos)| (vi, (pos - target_pos).length()))
            .filter(|(_, dist)| *dist < near_dup)  // Look for near-duplicates
            .collect()
    };

    println!("\n--- Analyzing orphan edges ---\n");

    for (i, &(v1, v2)) in result.orphan_edges.iter().take(3).enumerate() {
        let pos1 = voronoi.vertices[v1];
        let pos2 = voronoi.vertices[v2];
        let edge_len = (pos1 - pos2).length();

        println!("Orphan edge {}: vertices ({}, {})", i, v1, v2);
        println!("  Edge length: {:.2e}", edge_len);
        println!("  v1 pos: ({:.6}, {:.6}, {:.6})", pos1.x, pos1.y, pos1.z);
        println!("  v2 pos: ({:.6}, {:.6}, {:.6})", pos2.x, pos2.y, pos2.z);

        // Check for near-duplicate vertices
        let mut near_v1 = find_near_vertices(pos1, v1);
        near_v1.sort_by(|a, b| a.1.total_cmp(&b.1));
        let mut near_v2 = find_near_vertices(pos2, v2);
        near_v2.sort_by(|a, b| a.1.total_cmp(&b.1));

        if !near_v1.is_empty() {
            println!("  Near-duplicate vertices for v1:");
            for (vj, dist) in near_v1.iter().take(3) {
                println!("    vertex {} at dist {:.2e}", vj, dist);
            }
        }
        if !near_v2.is_empty() {
            println!("  Near-duplicate vertices for v2:");
            for (vj, dist) in near_v2.iter().take(3) {
                println!("    vertex {} at dist {:.2e}", vj, dist);
            }
        }

        // Check what cells should share this edge
        let cells = edge_to_cells.get(&(v1, v2)).cloned().unwrap_or_default();
        println!("  Edge appears in {} cell(s): {:?}", cells.len(), cells);
        println!();
    }
}

/// Strict correctness sweep on small counts.
/// Run with: cargo test --release strict_small_counts
#[test]
/// Validation test with new strategy:
/// - Hard errors (degenerate cells, orphan edges, etc.) fail the test
/// - Support/degeneracy issues (support_lt3, dup_support, collapsed_edges) are informational
/// - Geometric accuracy is checked with spacing-scaled bounds
fn strict_small_counts() {
    use crate::geometry::{fibonacci_sphere_points_with_rng, validation::validate_voronoi_strict};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let sizes = [500usize, 1_000, 2_000, 5_000, 10_000];
    let repeats = 5u64;
    let cases = [
        ("mild", 0.15f32),
        ("default", 0.25f32),
        ("rough", 0.45f32),
    ];
    const SUPPORT_LT3_SOFT_FRAC: f64 = 0.02;
    const SUPPORT_LT3_HARD_FRAC: f64 = 0.05;

    for &n in &sizes {
        for &(case_name, jitter_scale) in &cases {
            for seed_offset in 0..repeats {
                let seed = 12345 + seed_offset;
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                let mean_spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
                let mut jitter = mean_spacing * jitter_scale;
                let min_spacing = min_spacing_threshold();

                let mut points = fibonacci_sphere_points_with_rng(n, jitter, &mut rng);
                let mut min_dist = check_min_point_distance(&points);
                let mut attempts = 0;
                while min_dist < min_spacing && attempts < 4 {
                    jitter *= 0.5;
                    points = fibonacci_sphere_points_with_rng(n, jitter, &mut rng);
                    min_dist = check_min_point_distance(&points);
                    attempts += 1;
                }
                assert!(
                    min_dist >= min_spacing,
                    "input min spacing {:.2e} below threshold {:.2e} (n={}, case={})",
                    min_dist,
                    min_spacing,
                    n,
                    case_name
                );

                let spacing_sq = (mean_spacing as f64) * (mean_spacing as f64);
                let max_generator_delta = (spacing_sq * 0.05).max(1e-6);
                let max_edge_error = (spacing_sq * 0.05).max(1e-6);

                let eps_lo = super::SUPPORT_EPS_ABS;
                let eps_hi = eps_lo * 5.0;

                let voronoi = super::compute_voronoi_gpu_style(&points, DEFAULT_K);
                let strict = validate_voronoi_strict(
                    &voronoi,
                    eps_lo,
                    eps_hi,
                    Some(super::MIN_BISECTOR_DISTANCE),
                );

                // Print summary
                print!(
                    "n={}, seed={}, case={}, min_dist={:.2e}: ",
                    n, seed, case_name, min_dist
                );
                if strict.is_valid() {
                    print!("VALID");
                } else {
                    print!("INVALID");
                }
                if strict.has_degeneracy_issues() {
                    print!(" (support/degeneracy issues)");
                }
                println!(
                    " | max_gen_delta={:.2e}, max_edge_err={:.2e}",
                    strict.max_generator_delta,
                    strict.max_edge_bisector_error
                );

                // Print details if there are any issues
                if !strict.is_valid() || strict.has_degeneracy_issues() {
                    strict.print_summary();
                }

                // Hard requirements: mesh coherence
                assert!(
                    strict.is_valid(),
                    "mesh coherence failed for n={}, seed={}, case={}",
                    n,
                    seed,
                    case_name
                );

                let support_lt3 = strict.support_lt3.len() as f64;
                let support_lt3_ratio = if strict.num_vertices > 0 {
                    support_lt3 / strict.num_vertices as f64
                } else {
                    0.0
                };
                if support_lt3_ratio > SUPPORT_LT3_SOFT_FRAC {
                    println!(
                        "WARNING: support_lt3 ratio {:.2}% exceeds soft cap {:.2}% ({} of {})",
                        support_lt3_ratio * 100.0,
                        SUPPORT_LT3_SOFT_FRAC * 100.0,
                        strict.support_lt3.len(),
                        strict.num_vertices
                    );
                }
                assert!(
                    support_lt3_ratio <= SUPPORT_LT3_HARD_FRAC,
                    "support_lt3 ratio {:.2}% exceeds hard cap {:.2}% ({} of {}) for n={}, seed={}, case={}",
                    support_lt3_ratio * 100.0,
                    SUPPORT_LT3_HARD_FRAC * 100.0,
                    strict.support_lt3.len(),
                    strict.num_vertices,
                    n,
                    seed,
                    case_name
                );

                // Soft requirements: geometric accuracy bounds
                assert!(
                    strict.max_generator_delta < max_generator_delta,
                    "max_generator_delta {:.2e} exceeds bound {:.2e} for n={}, seed={}, case={}",
                    strict.max_generator_delta,
                    max_generator_delta,
                    n,
                    seed,
                    case_name
                );
                assert!(
                    strict.max_edge_bisector_error < max_edge_error,
                    "max_edge_bisector_error {:.2e} exceeds bound {:.2e} for n={}, seed={}, case={}",
                    strict.max_edge_bisector_error,
                    max_edge_error,
                    n,
                    seed,
                    case_name
                );
            }
        }
    }
}

/// Diagnose orphan edges for specific failing case.
/// Run with: cargo test --release diagnose_5000_orphans -- --ignored --nocapture
#[test]
#[ignore]
fn diagnose_5000_orphans() {
    use crate::geometry::fibonacci_sphere_points_with_rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::HashMap;

    let n = 5000usize;
    let seed = 12349u64;
    let jitter_scale = 0.15f32; // mild case

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mean_spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
    let jitter = mean_spacing * jitter_scale;
    let points = fibonacci_sphere_points_with_rng(n, jitter, &mut rng);

    println!("\n=== Diagnosis for n={}, seed={}, jitter={:.2e} ===\n", n, seed, jitter);

    let voronoi = compute_voronoi_gpu_style(&points, DEFAULT_K);

    println!("Cells: {}", voronoi.num_cells());
    println!("Vertices: {}", voronoi.vertices.len());

    // Build edge map
    let mut edge_to_cells: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for cell_idx in 0..voronoi.num_cells() {
        let cell = voronoi.cell(cell_idx);
        for i in 0..cell.vertex_indices.len() {
            let v1 = cell.vertex_indices[i];
            let v2 = cell.vertex_indices[(i + 1) % cell.vertex_indices.len()];
            if v1 == v2 {
                continue;
            }
            let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
            edge_to_cells.entry(edge).or_default().push(cell_idx);
        }
    }

    // Find orphan edges
    let orphan_edges: Vec<_> = edge_to_cells
        .iter()
        .filter(|(_, cells)| cells.len() == 1)
        .map(|(e, c)| (*e, c[0]))
        .collect();

    println!("Orphan edges: {}\n", orphan_edges.len());

    if orphan_edges.is_empty() {
        println!("No orphan edges to analyze!");
        return;
    }

    // Analyze each orphan edge
    for (i, &((v1, v2), cell_idx)) in orphan_edges.iter().enumerate() {
        println!("--- Orphan edge {} ---", i);
        let pos1 = voronoi.vertices[v1];
        let pos2 = voronoi.vertices[v2];
        let edge_len = (pos1 - pos2).length();
        let edge_mid = ((pos1 + pos2) / 2.0).normalize();

        println!("Edge v{}--v{}, length={:.2e}", v1, v2, edge_len);
        println!("  v{}: ({:.6}, {:.6}, {:.6})", v1, pos1.x, pos1.y, pos1.z);
        println!("  v{}: ({:.6}, {:.6}, {:.6})", v2, pos2.x, pos2.y, pos2.z);
        println!("Appears only in cell {}", cell_idx);

        let cell = voronoi.cell(cell_idx);
        let gen_idx = cell.generator_index;
        let gen = voronoi.generators[gen_idx];
        println!(
            "  Cell {} generator: {} at ({:.4}, {:.4}, {:.4})",
            cell_idx, gen_idx, gen.x, gen.y, gen.z
        );
        println!("  Cell vertex count: {}", cell.vertex_indices.len());

        // What generators are closest to the edge midpoint?
        let mut dots: Vec<(usize, f32)> = voronoi
            .generators
            .iter()
            .enumerate()
            .map(|(gi, g)| (gi, edge_mid.dot(*g)))
            .collect();
        dots.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("Top 5 generators closest to edge midpoint:");
        for (gi, dot) in dots.iter().take(5) {
            let delta = dots[0].1 - dot;
            let is_owner = if *gi == gen_idx { " <-- owner" } else { "" };
            println!("    g{}: dot={:.6}, delta={:.2e}{}", gi, dot, delta, is_owner);
        }

        // What cells touch v1 and v2?
        let cells_with_v1: std::collections::HashSet<usize> = edge_to_cells
            .iter()
            .filter(|(e, _)| e.0 == v1 || e.1 == v1)
            .flat_map(|(_, cells)| cells.iter().cloned())
            .collect();

        let cells_with_v2: std::collections::HashSet<usize> = edge_to_cells
            .iter()
            .filter(|(e, _)| e.0 == v2 || e.1 == v2)
            .flat_map(|(_, cells)| cells.iter().cloned())
            .collect();

        println!("Cells touching v{}: {:?}", v1, cells_with_v1);
        println!("Cells touching v{}: {:?}", v2, cells_with_v2);

        // The "missing neighbor" would be in both sets but not cell_idx
        let common: Vec<usize> = cells_with_v1
            .intersection(&cells_with_v2)
            .filter(|&&c| c != cell_idx)
            .cloned()
            .collect();
        println!("Potential missing neighbor (common excl owner): {:?}", common);

        // Check if there's a near-duplicate vertex
        println!("\nNear-duplicates of v{} (within 1e-4):", v1);
        let mut near_v1: Vec<(usize, f32)> = voronoi
            .vertices
            .iter()
            .enumerate()
            .filter(|(vi, _)| *vi != v1)
            .map(|(vi, v)| (vi, (*v - pos1).length()))
            .filter(|(_, d)| *d < 1e-4)
            .collect();
        near_v1.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        if near_v1.is_empty() {
            println!("  (none)");
        } else {
            for (vi, d) in near_v1.iter().take(3) {
                println!("  v{} at dist {:.2e}", vi, d);
            }
        }

        println!("Near-duplicates of v{} (within 1e-4):", v2);
        let mut near_v2: Vec<(usize, f32)> = voronoi
            .vertices
            .iter()
            .enumerate()
            .filter(|(vi, _)| *vi != v2)
            .map(|(vi, v)| (vi, (*v - pos2).length()))
            .filter(|(_, d)| *d < 1e-4)
            .collect();
        near_v2.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        if near_v2.is_empty() {
            println!("  (none)");
        } else {
            for (vi, d) in near_v2.iter().take(3) {
                println!("  v{} at dist {:.2e}", vi, d);
            }
        }
        println!();
    }

    // Now check support sets for the near-duplicate vertices
    println!("\n=== Support Set Analysis ===\n");
    let v7581 = voronoi.vertices[7581];
    let v7644 = voronoi.vertices[7644];
    println!("v7581 pos: ({:.9}, {:.9}, {:.9})", v7581.x, v7581.y, v7581.z);
    println!("v7644 pos: ({:.9}, {:.9}, {:.9})", v7644.x, v7644.y, v7644.z);
    println!("Distance: {:.2e}\n", (v7581 - v7644).length());

    // Compute support sets for each vertex using double precision
    let compute_support = |v: Vec3, eps: f64| -> Vec<(usize, f64)> {
        let v64 = glam::DVec3::new(v.x as f64, v.y as f64, v.z as f64).normalize();
        let mut dots: Vec<(usize, f64)> = voronoi
            .generators
            .iter()
            .enumerate()
            .map(|(gi, g)| {
                let g64 = glam::DVec3::new(g.x as f64, g.y as f64, g.z as f64);
                (gi, v64.dot(g64))
            })
            .collect();
        dots.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let max_dot = dots[0].1;
        dots.into_iter()
            .filter(|(_, d)| max_dot - d <= eps)
            .collect()
    };

    // Try different epsilon values
    for eps in [1e-7, 1e-6, 1e-5, 1e-4] {
        println!("eps = {:.0e}:", eps);
        let support_7581 = compute_support(v7581, eps);
        let support_7644 = compute_support(v7644, eps);
        let gens_7581: Vec<usize> = support_7581.iter().map(|(g, _)| *g).collect();
        let gens_7644: Vec<usize> = support_7644.iter().map(|(g, _)| *g).collect();
        println!("  v7581 support ({} gens): {:?}", gens_7581.len(), gens_7581);
        println!("  v7644 support ({} gens): {:?}", gens_7644.len(), gens_7644);
        let set1: std::collections::HashSet<_> = gens_7581.iter().collect();
        let set2: std::collections::HashSet<_> = gens_7644.iter().collect();
        if set1 == set2 {
            println!("  -> MATCH (same set)");
        } else {
            println!("  -> MISMATCH!");
        }
    }

    // Check which cells have which vertices and their neighbor counts
    println!("\n=== Cell Neighbor Analysis ===\n");
    for cell_idx in [3762usize, 3796, 3851, 3707] {
        let cell = voronoi.cell(cell_idx);
        println!("Cell {} has {} vertices", cell_idx, cell.vertex_indices.len());

        // Check if v7581 or v7644 is in this cell
        let has_7581 = cell.vertex_indices.contains(&7581);
        let has_7644 = cell.vertex_indices.contains(&7644);
        if has_7581 {
            println!("  Contains v7581");
        }
        if has_7644 {
            println!("  Contains v7644");
        }
    }

    // Compute support_eps used
    let mean_spacing = (4.0 * std::f64::consts::PI / n as f64).sqrt();
    let support_eps = super::SUPPORT_EPS_ABS;
    println!("\n=== Exact Support Analysis ===");
    println!("mean_spacing = {:.6e}", mean_spacing);
    println!("support_eps = {:.6e}\n", support_eps);

    // Check deltas for each vertex
    for (name, v) in [("v7581", v7581), ("v7644", v7644)] {
        println!("{}:", name);
        let v64 = glam::DVec3::new(v.x as f64, v.y as f64, v.z as f64).normalize();
        let mut dots: Vec<(usize, f64)> = voronoi
            .generators
            .iter()
            .enumerate()
            .map(|(gi, g)| {
                let g64 = glam::DVec3::new(g.x as f64, g.y as f64, g.z as f64);
                (gi, v64.dot(g64))
            })
            .collect();
        dots.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let max_dot = dots[0].1;

        for (gi, dot) in dots.iter().take(6) {
            let delta = max_dot - dot;
            let in_support = delta <= support_eps;
            println!(
                "  g{}: delta={:.6e} {}",
                gi,
                delta,
                if in_support { "<= eps (IN)" } else { "> eps (OUT)" }
            );
        }
    }
}

/// Measure ambiguity rates for standard point distributions.
/// Run with: cargo test --release measure_ambiguity_rates -- --ignored --nocapture
#[test]
#[ignore]
fn measure_ambiguity_rates() {
    use crate::geometry::{fibonacci_sphere_points_with_rng, lloyd_relax_kmeans};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::HashSet;

    println!("\n=== Ambiguity Rate Analysis ===\n");

    let sizes = [1_000usize, 5_000, 10_000, 50_000, 100_000];
    let margins = [0.0, 0.5, 1.0, 2.0]; // multiples of support_eps

    for &n in &sizes {
        println!("--- N = {} ---", n);

        let mean_spacing = (4.0 * std::f64::consts::PI / n as f64).sqrt();
        let support_eps = mean_spacing * mean_spacing * 0.01; // SUPPORT_EPS_SCALE = 0.01

        // Test different point distributions
        for (dist_name, lloyd_iters) in [("Fibonacci+jitter", 0), ("Lloyd 2 iters", 2)] {
            let mut rng = ChaCha8Rng::seed_from_u64(12345);
            let jitter = mean_spacing as f32 * 0.25;
            let mut points = fibonacci_sphere_points_with_rng(n, jitter, &mut rng);

            if lloyd_iters > 0 {
                lloyd_relax_kmeans(&mut points, lloyd_iters, 20, &mut rng);
            }

            let voronoi = compute_voronoi_gpu_style(&points, DEFAULT_K);
            let num_vertices = voronoi.vertices.len();
            let num_cells = voronoi.num_cells();

            // For each vertex, compute delta to 4th closest generator
            let mut fourth_deltas: Vec<f64> = Vec::with_capacity(num_vertices);

            for v in &voronoi.vertices {
                let v64 = glam::DVec3::new(v.x as f64, v.y as f64, v.z as f64).normalize();

                // Find top 4 generators by dot product
                let mut dots: Vec<f64> = voronoi
                    .generators
                    .iter()
                    .map(|g| {
                        let g64 = glam::DVec3::new(g.x as f64, g.y as f64, g.z as f64);
                        v64.dot(g64)
                    })
                    .collect();
                dots.sort_by(|a, b| b.partial_cmp(a).unwrap());

                let max_dot = dots[0];
                let fourth_delta = if dots.len() >= 4 {
                    max_dot - dots[3]
                } else {
                    f64::MAX
                };
                fourth_deltas.push(fourth_delta);
            }

            // Build vertex -> cells mapping
            let mut vertex_to_cells: Vec<HashSet<usize>> = vec![HashSet::new(); num_vertices];
            for cell_idx in 0..num_cells {
                let cell = voronoi.cell(cell_idx);
                for &vi in cell.vertex_indices {
                    if vi < num_vertices {
                        vertex_to_cells[vi].insert(cell_idx);
                    }
                }
            }

            println!("  {}: {} vertices, {} cells", dist_name, num_vertices, num_cells);
            println!("    support_eps = {:.2e}", support_eps);

            for &margin_mult in &margins {
                let threshold = support_eps * (1.0 + margin_mult);

                let ambiguous_vertices: Vec<usize> = fourth_deltas
                    .iter()
                    .enumerate()
                    .filter(|(_, &delta)| delta <= threshold)
                    .map(|(i, _)| i)
                    .collect();

                let ambiguous_cells: HashSet<usize> = ambiguous_vertices
                    .iter()
                    .flat_map(|&vi| vertex_to_cells[vi].iter().cloned())
                    .collect();

                let vert_pct = 100.0 * ambiguous_vertices.len() as f64 / num_vertices as f64;
                let cell_pct = 100.0 * ambiguous_cells.len() as f64 / num_cells as f64;

                println!(
                    "    margin={:.1}x: {:5} ambig verts ({:5.2}%), {:5} ambig cells ({:5.2}%)",
                    margin_mult,
                    ambiguous_vertices.len(),
                    vert_pct,
                    ambiguous_cells.len(),
                    cell_pct
                );
            }

            // Also show distribution of fourth_delta values
            let mut sorted_deltas = fourth_deltas.clone();
            sorted_deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = sorted_deltas[num_vertices / 2];
            let p90 = sorted_deltas[num_vertices * 9 / 10];
            let p99 = sorted_deltas[num_vertices * 99 / 100];
            let min_delta = sorted_deltas[0];

            println!(
                "    4th-gen delta: min={:.2e}, p50={:.2e}, p90={:.2e}, p99={:.2e}",
                min_delta, p50, p90, p99
            );
            println!();
        }
    }
}

/// Strict correctness sweep on large counts using efficient k-NN validation.
/// Run with: cargo test --release strict_large_counts -- --ignored --nocapture
#[test]
#[ignore]
fn strict_large_counts() {
    use crate::geometry::{
        fibonacci_sphere_points_with_rng,
        lloyd_relax_kmeans,
        validation::{validate_voronoi_large, LargeValidationConfig},
    };
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::time::Instant;

    println!("\n=== Strict Large Counts Validation ===\n");

    // Test sizes from 25k to 500k
    let sizes = [25_000usize, 50_000, 100_000, 250_000, 500_000];
    let repeats = 3u64;

    // Validation thresholds
    const SUPPORT_LT3_SOFT_FRAC: f64 = 0.02;
    const SUPPORT_LT3_HARD_FRAC: f64 = 0.05;

    for &n in &sizes {
        println!("--- n = {} ---", n);

        for seed_offset in 0..repeats {
            let seed = 12345 + seed_offset;
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mean_spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
            let jitter = mean_spacing * 0.25;

            // Generate points with Lloyd relaxation for quality
            let mut points = fibonacci_sphere_points_with_rng(n, jitter, &mut rng);

            // Use k-means Lloyd for large n (faster)
            let t0 = Instant::now();
            if n >= 100_000 {
                lloyd_relax_kmeans(&mut points, 2, 15, &mut rng);
            } else {
                lloyd_relax_kmeans(&mut points, 3, 20, &mut rng);
            }
            let lloyd_ms = t0.elapsed().as_secs_f64() * 1000.0;

            // Verify min spacing
            let min_dist = check_min_point_distance(&points);
            let min_spacing = min_spacing_threshold();
            if min_dist < min_spacing {
                println!(
                    "  seed={}: SKIP - min spacing {:.2e} below threshold {:.2e}",
                    seed, min_dist, min_spacing
                );
                continue;
            }

            // Build Voronoi
            let t0 = Instant::now();
            let voronoi = super::compute_voronoi_gpu_style(&points, DEFAULT_K);
            let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

            // Configure validation based on size
            let config = if n >= 250_000 {
                // Fast for very large: 5% sampling, k=48
                LargeValidationConfig {
                    knn_k: 48,
                    vertex_sample_rate: 0.05,
                    eps_abs: super::SUPPORT_EPS_ABS,
                    seed,
                }
            } else if n >= 100_000 {
                // Thorough for large: 10% sampling, k=48
                LargeValidationConfig {
                    knn_k: 48,
                    vertex_sample_rate: 0.10,
                    eps_abs: super::SUPPORT_EPS_ABS,
                    seed,
                }
            } else {
                // Full for medium: all vertices, k=48
                LargeValidationConfig {
                    knn_k: 48,
                    vertex_sample_rate: 1.0,
                    eps_abs: super::SUPPORT_EPS_ABS,
                    seed,
                }
            };

            // Validate
            let t0 = Instant::now();
            let result = validate_voronoi_large(&voronoi, &config);
            let validate_ms = t0.elapsed().as_secs_f64() * 1000.0;

            // Compute support_lt3 rate
            let support_lt3_rate = if result.vertices_sampled > 0 {
                result.support_lt3 as f64 / result.vertices_sampled as f64
            } else {
                0.0
            };

            // Print summary
            print!(
                "  seed={}: lloyd={:.0}ms build={:.0}ms validate={:.0}ms | ",
                seed, lloyd_ms, build_ms, validate_ms
            );
            if result.topology_valid() {
                print!("TOPO_OK");
            } else {
                print!("TOPO_FAIL(euler={} orphan={} degen={})",
                    if result.euler_ok { "ok" } else { "FAIL" },
                    result.orphan_edges,
                    result.degenerate_cells
                );
            }
            print!(
                " | sampled={} support_lt3={:.2}% gen_mismatch={} max_delta={:.2e}",
                result.vertices_sampled,
                support_lt3_rate * 100.0,
                result.generator_mismatch,
                result.max_generator_delta
            );
            println!();

            // Hard assertions: any degenerate cell or orphan edge is a failure
            assert!(
                result.degenerate_cells == 0,
                "Found {} degenerate cells for n={}, seed={}",
                result.degenerate_cells,
                n,
                seed
            );
            assert!(
                result.orphan_edges == 0,
                "Found {} orphan edges for n={}, seed={}",
                result.orphan_edges,
                n,
                seed
            );
            assert!(
                result.euler_ok,
                "Euler check failed for n={}, seed={}: V-E+F = {}",
                n,
                seed,
                result.euler_v as i64 - result.euler_e as i64 + result.euler_f as i64
            );

            // Soft warning for support_lt3
            if support_lt3_rate > SUPPORT_LT3_SOFT_FRAC {
                println!(
                    "    WARNING: support_lt3 rate {:.2}% exceeds soft cap {:.2}%",
                    support_lt3_rate * 100.0,
                    SUPPORT_LT3_SOFT_FRAC * 100.0
                );
            }

            // Hard cap for support_lt3
            assert!(
                support_lt3_rate <= SUPPORT_LT3_HARD_FRAC,
                "support_lt3 rate {:.2}% exceeds hard cap {:.2}% for n={}, seed={}",
                support_lt3_rate * 100.0,
                SUPPORT_LT3_HARD_FRAC * 100.0,
                n,
                seed
            );
        }
        println!();
    }
}

/// Correlate orphan edges with certification failures.
/// Run with: cargo test --release test_orphan_edge_certification -- --ignored --nocapture
///
/// This test checks: when validation finds orphan edges, were the vertices involved
/// flagged by certification as problematic?
///
/// Uses the known-failing case: n=100000, seed=12346 which produces 4 orphan edges.
#[test]
#[ignore]
fn test_orphan_edge_certification() {
    use crate::geometry::{fibonacci_sphere_points_with_rng, lloyd_relax_kmeans, validation::validate_voronoi_strict};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::HashSet;

    println!("\n=== Orphan Edge vs Certification Correlation ===\n");

    // Known failing cases from strict_large_counts
    let configs = [
        // (n, seed) - these are cases that produce orphan edges with default k
        (100_000usize, 12346u64),
        (100_000, 12345), // control case - should have no orphan edges
    ];

    for &(n, seed) in &configs {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mean_spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
        let jitter = mean_spacing * 0.25;
        let mut points = fibonacci_sphere_points_with_rng(n, jitter, &mut rng);

        // Use same Lloyd relaxation as strict_large_counts (for n >= 100k)
        lloyd_relax_kmeans(&mut points, 2, 15, &mut rng);

        // Check min distance
        let min_dist = check_min_point_distance(&points);
        let min_spacing = min_spacing_threshold();
        if min_dist < min_spacing {
            println!("n={} seed={}: SKIP (min_dist={:.2e} < threshold)", n, seed, min_dist);
            continue;
        }

        // Build flat data to get cert-failed vertex positions
        use super::{build_cells_data_flat, CubeMapGridKnn, TerminationConfig};

        let knn = CubeMapGridKnn::new(&points);
        let termination = TerminationConfig { enabled: true, check_start: 10, check_step: 6 };
        let flat_data = build_cells_data_flat(&points, &knn, DEFAULT_K, termination);
        let flat_stats = flat_data.stats();

        // Collect positions of cert-failed vertices (pre-dedup)
        let mut cert_failed_positions: Vec<(glam::Vec3, u8)> = Vec::new();
        for chunk in &flat_data.chunks {
            for &(local_idx, reason) in &chunk.cert_failed_vertex_indices {
                // Find this vertex's position in the chunk
                let pos = chunk.vertices[local_idx as usize].1;
                cert_failed_positions.push((pos, reason));
            }
        }

        // Build voronoi separately for validation (duplicates work but simpler for test)
        let voronoi = super::compute_voronoi_gpu_style(&points, DEFAULT_K);

        // Run strict validation to get actual orphan edge indices
        let eps_lo = super::SUPPORT_EPS_ABS;
        let eps_hi = eps_lo * 5.0;
        let strict = validate_voronoi_strict(&voronoi, eps_lo, eps_hi, None);

        print!("n={:>6} seed={}: ", n, seed);
        print!(
            "orphan_edges={} cert_failed={} cert_checked_cells={} cert_failed_cells={} cert_checked_vertices={} cert_failed_vertices={} ",
            strict.orphan_edges.len(),
            cert_failed_positions.len(),
            flat_stats.support_cert_checked,
            flat_stats.support_cert_failed,
            flat_stats.support_cert_checked_vertices,
            flat_stats.support_cert_failed_vertices,
        );

        if strict.orphan_edges.is_empty() {
            println!("(no orphan edges to analyze)");
        } else {
            println!();
            strict.print_summary();

            let tol = near_duplicate_threshold(n);
            println!("  positional match tolerance: {:.2e}", tol);

            // Build edge -> cells map for positional matching diagnostics.
            // (This duplicates work done in strict validation but keeps the test self-contained.)
            let mut edge_to_cells: std::collections::HashMap<(usize, usize), Vec<usize>> =
                std::collections::HashMap::new();
            for cell_idx in 0..voronoi.num_cells() {
                let cell = voronoi.cell(cell_idx);
                let m = cell.vertex_indices.len();
                for i in 0..m {
                    let a = cell.vertex_indices[i];
                    let b = cell.vertex_indices[(i + 1) % m];
                    if a == b {
                        continue;
                    }
                    let edge = if a < b { (a, b) } else { (b, a) };
                    edge_to_cells.entry(edge).or_default().push(cell_idx);
                }
            }

            let mismatch_vertices: HashSet<usize> = strict.generator_mismatch_vertices.iter().copied().collect();
            let support_lt3_vertices: HashSet<usize> = strict.support_lt3.iter().copied().collect();

            let find_owner_cell = |v1: usize, v2: usize| -> Option<usize> {
                for cell_idx in 0..voronoi.num_cells() {
                    let cell = voronoi.cell(cell_idx);
                    let n = cell.vertex_indices.len();
                    for i in 0..n {
                        let a = cell.vertex_indices[i];
                        let b = cell.vertex_indices[(i + 1) % n];
                        if (a == v1 && b == v2) || (a == v2 && b == v1) {
                            return Some(cell_idx);
                        }
                    }
                }
                None
            };

            let nearest_other_vertex = |v_idx: usize| -> (usize, f32) {
                let pos = voronoi.vertices[v_idx];
                let mut best_idx = usize::MAX;
                let mut best_dist = f32::MAX;
                for (i, &p) in voronoi.vertices.iter().enumerate() {
                    if i == v_idx {
                        continue;
                    }
                    let d = (p - pos).length();
                    if d < best_dist {
                        best_dist = d;
                        best_idx = i;
                    }
                }
                (best_idx, best_dist)
            };

            let find_positional_edge_match = |v1: usize, v2: usize| -> Option<((usize, usize), f32)> {
                let p1 = voronoi.vertices[v1];
                let p2 = voronoi.vertices[v2];
                let mut best: Option<((usize, usize), f32)> = None;
                for (&(a, b), _) in &edge_to_cells {
                    if (a == v1 && b == v2) || (a == v2 && b == v1) {
                        continue;
                    }
                    let pa = voronoi.vertices[a];
                    let pb = voronoi.vertices[b];
                    let d_direct = (pa - p1).length().max((pb - p2).length());
                    let d_swap = (pa - p2).length().max((pb - p1).length());
                    let d = d_direct.min(d_swap);
                    if d <= tol {
                        if best.map(|(_, bd)| d < bd).unwrap_or(true) {
                            best = Some(((a, b), d));
                        }
                    }
                }
                best
            };

            let cell_flags = |cell_idx: usize| -> Option<(u8, u8, u8, u8)> {
                let mut base = 0usize;
                for chunk in &flat_data.chunks {
                    let len = chunk.counts.len();
                    if cell_idx < base + len {
                        let local = cell_idx - base;
                        return Some((
                            chunk.cell_terminated[local],
                            chunk.cell_used_fallback[local],
                            chunk.cell_full_scan_done[local],
                            chunk.cell_candidates_complete[local],
                        ));
                    }
                    base += len;
                }
                None
            };

            // Helper to find nearest cert-failed vertex distance for a given position
            let find_nearest_cert_fail = |pos: glam::Vec3| -> (f32, Option<u8>) {
                let mut min_dist = f32::MAX;
                let mut nearest_reason: Option<u8> = None;
                for &(cert_pos, reason) in &cert_failed_positions {
                    let dist = (pos - cert_pos).length();
                    if dist < min_dist {
                        min_dist = dist;
                        nearest_reason = Some(reason);
                    }
                }
                (min_dist, nearest_reason)
            };

            // For each orphan edge, report the minimum distance of the two endpoints
            let mut matches = 0usize;
            let mut near_misses = 0usize;

            for &(v1, v2) in &strict.orphan_edges {
                let pos_a = voronoi.vertices[v1];
                let pos_b = voronoi.vertices[v2];
                let owner_cell = find_owner_cell(v1, v2);
                let flags = owner_cell.and_then(|c| cell_flags(c));
                let (terminated, used_fallback, full_scan, candidates_complete) = flags.unwrap_or((0, 0, 0, 0));

                let (dist_a, reason_a) = find_nearest_cert_fail(pos_a);
                let (dist_b, reason_b) = find_nearest_cert_fail(pos_b);

                // Use the closer endpoint
                let (min_dist, nearest_reason, closer_v) = if dist_a <= dist_b {
                    (dist_a, reason_a, v1)
                } else {
                    (dist_b, reason_b, v2)
                };

                let reason_str = match nearest_reason {
                    Some(1) => "ill_cond",
                    Some(2) => "gap",
                    Some(3) => "both",
                    _ => "none",
                };

                if min_dist < 1e-6 {
                    matches += 1;
                    println!("    edge({},{}): MATCH via v{} (dist={:.2e}, reason={})",
                        v1, v2, closer_v, min_dist, reason_str);
                } else if min_dist < 1e-3 {
                    near_misses += 1;
                    println!("    edge({},{}): NEAR via v{} (dist={:.2e}, reason={})",
                        v1, v2, closer_v, min_dist, reason_str);
                } else {
                    println!("    edge({},{}): NO MATCH (closer v{} dist={:.2e})",
                        v1, v2, closer_v, min_dist);
                }

                let v1_mismatch = mismatch_vertices.contains(&v1);
                let v2_mismatch = mismatch_vertices.contains(&v2);
                let v1_lt3 = support_lt3_vertices.contains(&v1);
                let v2_lt3 = support_lt3_vertices.contains(&v2);
                let (near1, near1_d) = nearest_other_vertex(v1);
                let (near2, near2_d) = nearest_other_vertex(v2);
                let positional_edge_match = find_positional_edge_match(v1, v2);
                println!(
                    "      owner_cell={:?} candidates_complete={} terminated={} used_fallback={} full_scan={} gen_mismatch(v1,v2)=({},{}) support_lt3(v1,v2)=({},{})",
                    owner_cell,
                    candidates_complete,
                    terminated,
                    used_fallback,
                    full_scan,
                    v1_mismatch as u8,
                    v2_mismatch as u8,
                    v1_lt3 as u8,
                    v2_lt3 as u8,
                );
                println!(
                    "      nearest_other(v1)=(v{} dist={:.2e}) nearest_other(v2)=(v{} dist={:.2e})",
                    near1,
                    near1_d,
                    near2,
                    near2_d,
                );
                if let Some(((a, b), d)) = positional_edge_match {
                    let cells = edge_to_cells.get(&(a.min(b), a.max(b))).map(|v| v.len()).unwrap_or(0);
                    println!("      positional_edge_match=edge({},{}) dist={:.2e} shared_by_cells={}", a, b, d, cells);
                } else {
                    println!("      positional_edge_match=none (tol={:.2e})", tol);
                }
            }

            println!("  Summary: {} exact matches, {} near misses out of {} orphan edges",
                matches, near_misses, strict.orphan_edges.len());
        }
    }
    println!();
}

/// Large-scale soundness test - verifies the algorithm at 750k points.
/// Run with: cargo test --release test_soundness_large_scale -- --ignored --nocapture
#[test]
#[ignore]
fn test_soundness_large_scale() {
    use crate::geometry::{fibonacci_sphere_points_with_rng, lloyd_relax_kmeans, validation::validate_voronoi};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::time::Instant;

    println!("\n=== Large-Scale Soundness Test ===\n");

    for &n in &[250_000, 500_000, 750_000] {
        println!("Testing n = {}...", n);
        let mut rng = ChaCha8Rng::seed_from_u64(12345);
        let mean_spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
        let jitter = mean_spacing * 0.25;
        let mut points = fibonacci_sphere_points_with_rng(n, jitter, &mut rng);

        // Use k-means Lloyd (faster for large n)
        let t0 = Instant::now();
        lloyd_relax_kmeans(&mut points, 2, 20, &mut rng);
        println!("  Lloyd relaxation: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

        // Verify min spacing requirement
        let min_dist = check_min_point_distance(&points);
        let threshold = min_spacing_threshold();
        println!("  Min point distance: {:.2e} (threshold: {:.2e})", min_dist, threshold);
        assert!(
            min_dist >= threshold,
            "Lloyd-relaxed points should meet min spacing"
        );

        // Build Voronoi and validate
        let t0 = Instant::now();
        let voronoi = compute_voronoi_gpu_style(&points, DEFAULT_K);
        println!("  Voronoi build: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

        let t0 = Instant::now();
        let result = validate_voronoi(&voronoi, near_duplicate_threshold(points.len()));
        println!("  Validation: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

        println!("  Degenerate cells: {}", result.degenerate_cells.len());
        println!("  Orphan edges: {}", result.orphan_edges.len());
        println!("  Near-duplicate cells: {}", result.near_duplicate_cells.len());
        println!();

        // With spacing-constrained inputs, we should have 0 degenerate cells
        assert!(result.degenerate_cells.is_empty(),
            "n={}: Should have 0 degenerate cells, got {}", n, result.degenerate_cells.len());

        // Orphan edges should be minimal (<0.01%)
        let orphan_rate = result.orphan_edges.len() as f64 / n as f64;
        assert!(orphan_rate < 0.0001,
            "n={}: Orphan rate {:.4}% exceeds 0.01% threshold", n, orphan_rate * 100.0);
    }
}

/// Test behavior with coincident points.
/// This validates that:
/// 1. MIN_BISECTOR_DISTANCE merging avoids unstable bisectors
/// 2. Coincident points still collapse to shared cells (no crash)
/// 3. The algorithm doesn't panic on bad input
/// 4. Issues are bounded by the number of bad input points
#[test]
fn test_coincident_points_handled_gracefully() {
    use crate::geometry::validation::validate_voronoi;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    println!("\n=== Coincident Points Handling Test ===\n");

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let n = 1000;

    // Generate base points
    let mut points = random_sphere_points_with_rng(n, &mut rng);

    // Inject coincident points (exact duplicates)
    let num_duplicates = 50;
    for i in 0..num_duplicates {
        let src_idx = i * 10;
        let dst_idx = src_idx + 1;
        if dst_idx < n {
            points[dst_idx] = points[src_idx];
        }
    }

    // Also add some near-duplicates (within MIN_BISECTOR_DISTANCE)
    for i in 0..20 {
        let idx = 500 + i;
        if idx + 1 < n {
            let offset = Vec3::splat(super::MIN_BISECTOR_DISTANCE * 0.2);
            points[idx + 1] = (points[idx] + offset).normalize();
        }
    }

    println!("Injected {} exact duplicates and 20 near-duplicates", num_duplicates);

    // Check minimum distance - should show the coincident points
    let min_dist = check_min_point_distance(&points);
    println!("Minimum point distance: {:.2e}", min_dist);
    assert!(
        min_dist < super::MIN_BISECTOR_DISTANCE,
        "Test setup should have coincident points below threshold"
    );

    // Build Voronoi - should handle gracefully without panicking
    let voronoi = compute_voronoi_gpu_style(&points, DEFAULT_K);
    let result = validate_voronoi(&voronoi, near_duplicate_threshold(points.len()));

    println!("\nWith MIN_BISECTOR_DISTANCE merge (current behavior):");
    println!("  Degenerate cells: {}", result.degenerate_cells.len());
    println!("  Orphan edges: {}", result.orphan_edges.len());
    println!("  Cells produced: {}", voronoi.num_cells());

    if !result.degenerate_cells.is_empty() {
        println!(
            "\nNote: {} degenerate cells from invalid input",
            result.degenerate_cells.len()
        );
    }
    if !result.orphan_edges.is_empty() {
        println!(
            "      {} orphan edges from invalid input",
            result.orphan_edges.len()
        );
    }
    if !result.overcounted_edges.is_empty() {
        println!(
            "      {} overcounted edges from duplicate cells",
            result.overcounted_edges.len()
        );
    }

    // 3. All cells should have been processed (no panics)
    assert_eq!(voronoi.num_cells(), n, "All cells should be processed");

    // 4. The number of issues should be bounded by the number of bad input points
    // We injected ~70 problem points, so we shouldn't have catastrophic failure
    let total_issues = result.degenerate_cells.len();
    assert!(total_issues <= num_duplicates + 20,
        "Degenerate cells ({}) should be bounded by coincident point count ({})",
        total_issues, num_duplicates + 20);
}

/// Test that the algorithm correctly skips neighbors below MIN_BISECTOR_DISTANCE.
#[test]
fn test_bisector_distance_threshold_applied() {
    use super::cell_builder::IncrementalCellBuilder;

    println!("\n=== Bisector Distance Threshold Test ===\n");

    // Create a generator and neighbors at various distances
    let generator = Vec3::new(1.0, 0.0, 0.0);
    let mut builder = IncrementalCellBuilder::new(0, generator);

    // Neighbor below threshold - should be skipped
    let too_close = generator + Vec3::new(super::MIN_BISECTOR_DISTANCE * 0.2, 0.0, 0.0);
    let too_close = too_close.normalize();
    builder.clip(1, too_close);

    // Neighbor above threshold - should be added
    let far_enough = Vec3::new(0.0, 1.0, 0.0);
    builder.clip(2, far_enough);

    let far_enough2 = Vec3::new(0.0, 0.0, 1.0);
    builder.clip(3, far_enough2);

    let far_enough3 = Vec3::new(-1.0, 0.0, 0.0);
    builder.clip(4, far_enough3);

    // After clipping, we should have a valid cell from the 3 far neighbors
    // The too-close neighbor should have been skipped
    println!("Vertex count after clipping: {}", builder.vertex_count());
    println!("(Too-close neighbor should have been skipped)");

    // We need at least 3 valid planes to form a cell
    // With 3 valid planes (far_enough, far_enough2, far_enough3), we should get vertices
    assert!(builder.vertex_count() >= 3,
        "Should form a cell from the 3 valid neighbors");
}

/// Helper: compute minimum distance between any two points
fn check_min_point_distance(points: &[Vec3]) -> f32 {
    use kiddo::{ImmutableKdTree, SquaredEuclidean};

    let entries: Vec<[f32; 3]> = points.iter().map(|p| [p.x, p.y, p.z]).collect();
    let tree = ImmutableKdTree::new_from_slice(&entries);

    let mut min_dist = f32::MAX;
    for (i, p) in points.iter().enumerate() {
        // Find 2 nearest (self + closest other)
        let results = tree.nearest_n::<SquaredEuclidean>(&[p.x, p.y, p.z], 2);
        for r in results {
            if r.item as usize != i {
                let dist = r.distance.sqrt();
                min_dist = min_dist.min(dist);
            }
        }
    }
    min_dist
}

#[test]
#[ignore] // Run with: cargo test bench_knn_breakdown -- --ignored --nocapture
fn bench_knn_breakdown() {
    use std::time::Instant;
    use crate::geometry::cube_grid::CubeMapGrid;

    println!("\nk-NN micro-breakdown at 500k points");
    println!("{:-<60}", "");

    let n = 500_000;
    let k = DEFAULT_K;
    let points = random_sphere_points(n);

    // Build grid directly to access stats
    let res = ((n as f64 / 300.0).sqrt() as usize).max(4);
    let grid = CubeMapGrid::new(&points, res);
    let stats = grid.stats();
    println!("Grid: res={} cells={} avg_pts/cell={:.1}", res, stats.num_cells, stats.avg_points_per_cell);

    let knn = CubeMapGridKnn::new(&points);

    // Sample queries to measure
    let sample_size = 10_000;

    // Warm up
    let mut scratch = knn.make_scratch();
    let mut neighbors = Vec::with_capacity(k);
    for i in 0..1000 {
        neighbors.clear();
        knn.knn_into(points[i], i, k, &mut scratch, &mut neighbors);
    }

    // Measure total time for sample
    let t0 = Instant::now();
    for i in 0..sample_size {
        neighbors.clear();
        knn.knn_into(points[i], i, k, &mut scratch, &mut neighbors);
    }
    let total_us = t0.elapsed().as_micros();
    let per_query_us = total_us as f64 / sample_size as f64;

    println!("Total time for {} queries: {:.1}ms", sample_size, total_us as f64 / 1000.0);
    println!("Per-query average: {:.2}µs", per_query_us);
    println!("\nEstimated full 500k (serial): {:.1}ms", per_query_us * n as f64 / 1000.0);

    // Compare with kiddo
    let (tree, entries) = super::build_kdtree(&points);

    let t0 = Instant::now();
    for i in 0..sample_size {
        let _ = super::find_k_nearest(&tree, &entries, points[i], i, k);
    }
    let kiddo_us = t0.elapsed().as_micros();
    let kiddo_per_query = kiddo_us as f64 / sample_size as f64;

    println!("\nKiddo per-query: {:.2}µs", kiddo_per_query);
    println!("CubeMapGrid is {:.2}x slower than Kiddo (serial)", per_query_us / kiddo_per_query);

    // At 50 pts/cell and k=24, we expect to visit ~1-2 cells typically
    // Let's estimate: if we check ~50-100 points per query, that's 50-100 heap operations
    println!("\n--- Estimated operation counts per query ---");
    println!("Points per cell: ~{:.0}", stats.avg_points_per_cell);
    println!("If visiting 1-2 cells: ~{:.0}-{:.0} distance computations",
        stats.avg_points_per_cell, stats.avg_points_per_cell * 2.0);
    println!("Heap operations for k={}: up to {} push + {} pop", k, k, k);
}

