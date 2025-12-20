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
/// Run with: cargo test --release strict_small_counts -- --ignored --nocapture
#[test]
/// Validation test with new strategy:
/// - Hard errors (degenerate cells, orphan edges, etc.) fail the test
/// - Support/degeneracy issues (support_lt3, dup_support, collapsed_edges) are informational
/// - Geometric accuracy is checked with spacing-scaled bounds
#[ignore]
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

                let eps_lo = (spacing_sq * 0.005).max(1e-7);
                let eps_hi = (eps_lo * 5.0).max(5e-7);

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

