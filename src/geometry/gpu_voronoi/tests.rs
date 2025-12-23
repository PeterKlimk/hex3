//! Tests for GPU-style Voronoi computation.

use glam::Vec3;

use super::*;
use crate::geometry::{random_sphere_points, random_sphere_points_with_rng};

const MIN_SPACING_MARGIN: f32 = 1.05;

fn min_spacing_threshold() -> f32 {
    super::MIN_BISECTOR_DISTANCE * MIN_SPACING_MARGIN
}

/// Approximate mean chord length between uniformly-distributed generators.
fn mean_generator_spacing_chord(num_points: usize) -> f32 {
    if num_points == 0 {
        return 0.0;
    }
    let mean_angle = (4.0 * std::f32::consts::PI / num_points as f32).sqrt();
    2.0 * (0.5 * mean_angle).sin()
}

fn near_duplicate_threshold(num_points: usize) -> f32 {
    if num_points == 0 {
        return super::MIN_BISECTOR_DISTANCE * 0.25;
    }
    let spacing = mean_generator_spacing_chord(num_points);
    (spacing * super::VERTEX_WELD_FRACTION).max(super::MIN_BISECTOR_DISTANCE * 0.25)
}

fn assert_min_spacing(points: &[Vec3], label: &str) -> f32 {
    let min_dist = check_min_point_distance(points);
    let threshold = min_spacing_threshold();
    println!(
        "{} min point distance = {:.2e} (threshold: {:.2e})",
        label, min_dist, threshold
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
fn test_great_circle_bisector() {
    let a = Vec3::new(1.0, 0.0, 0.0);
    let b = Vec3::new(0.0, 1.0, 0.0);

    // Bisector plane normal: (a - b).normalize()
    let normal = (a - b).normalize();

    // a should be on positive side, b on negative
    assert!(normal.dot(a) > 0.0);
    assert!(normal.dot(b) < 0.0);

    // Midpoint should be on the plane
    let mid = (a + b).normalize();
    assert!(normal.dot(mid).abs() < 1e-6);
}

#[test]
fn test_gpu_voronoi_basic() {
    let points = random_sphere_points(50);
    let _ = assert_min_spacing(&points, "gpu_voronoi_basic");
    let voronoi = compute_voronoi_gpu_style(&points);

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
    use crate::geometry::{
        fibonacci_sphere_points_with_rng, lloyd_relax_voronoi, validation::validate_voronoi,
    };
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
        let voronoi = compute_voronoi_gpu_style(&points);
        let result = validate_voronoi(&voronoi, near_duplicate_threshold(points.len()));

        println!("  Degenerate cells: {}", result.degenerate_cells.len());
        println!("  Orphan edges: {}", result.orphan_edges.len());
        println!(
            "  Near-duplicate cells: {}",
            result.near_duplicate_cells.len()
        );

        assert!(
            result.degenerate_cells.is_empty(),
            "n={}: Lloyd-relaxed points should produce no degenerate cells",
            n
        );
        assert!(
            result.orphan_edges.len() <= 10,
            "n={}: Lloyd-relaxed points should have minimal orphan edges, got {}",
            n,
            result.orphan_edges.len()
        );
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
    let cases = [("mild", 0.15f32), ("default", 0.25f32), ("rough", 0.45f32)];
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

                // Use f32-appropriate tolerance for validation (vertices are stored as f32)
                // SUPPORT_EPS_ABS is for f64 internal computation, not f32 validation
                let eps_lo = (f32::EPSILON * 64.0) as f64; // ~7.6e-6
                let eps_hi = eps_lo * 5.0;

                let voronoi = super::compute_voronoi_gpu_style(&points);
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
                    strict.max_generator_delta, strict.max_edge_bisector_error
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

/// Strict correctness sweep on large counts using efficient k-NN validation.
/// Run with: cargo test --release strict_large_counts -- --ignored --nocapture
#[test]
#[ignore]
fn strict_large_counts() {
    use crate::geometry::{
        fibonacci_sphere_points_with_rng, lloyd_relax_kmeans,
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
            let voronoi = super::compute_voronoi_gpu_style(&points);
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
                print!(
                    "TOPO_FAIL(euler={} orphan={} degen={})",
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


/// Large-scale soundness test - verifies the algorithm at 750k points.
/// Run with: cargo test --release test_soundness_large_scale -- --ignored --nocapture
#[test]
#[ignore]
fn test_soundness_large_scale() {
    use crate::geometry::{
        fibonacci_sphere_points_with_rng, lloyd_relax_kmeans, validation::validate_voronoi,
    };
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
        println!(
            "  Lloyd relaxation: {:.1}ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );

        // Verify min spacing requirement
        let min_dist = check_min_point_distance(&points);
        let threshold = min_spacing_threshold();
        println!(
            "  Min point distance: {:.2e} (threshold: {:.2e})",
            min_dist, threshold
        );
        assert!(
            min_dist >= threshold,
            "Lloyd-relaxed points should meet min spacing"
        );

        // Build Voronoi and validate
        let t0 = Instant::now();
        let voronoi = compute_voronoi_gpu_style(&points);
        println!(
            "  Voronoi build: {:.1}ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );

        let t0 = Instant::now();
        let result = validate_voronoi(&voronoi, near_duplicate_threshold(points.len()));
        println!("  Validation: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

        println!("  Degenerate cells: {}", result.degenerate_cells.len());
        println!("  Orphan edges: {}", result.orphan_edges.len());
        println!(
            "  Near-duplicate cells: {}",
            result.near_duplicate_cells.len()
        );
        println!();

        // With spacing-constrained inputs, we should have 0 degenerate cells
        assert!(
            result.degenerate_cells.is_empty(),
            "n={}: Should have 0 degenerate cells, got {}",
            n,
            result.degenerate_cells.len()
        );

        // Orphan edges should be minimal (<0.01%)
        let orphan_rate = result.orphan_edges.len() as f64 / n as f64;
        assert!(
            orphan_rate < 0.0001,
            "n={}: Orphan rate {:.4}% exceeds 0.01% threshold",
            n,
            orphan_rate * 100.0
        );
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

    println!(
        "Injected {} exact duplicates and 20 near-duplicates",
        num_duplicates
    );

    // Check minimum distance - should show the coincident points
    let min_dist = check_min_point_distance(&points);
    println!("Minimum point distance: {:.2e}", min_dist);
    assert!(
        min_dist < super::MIN_BISECTOR_DISTANCE,
        "Test setup should have coincident points below threshold"
    );

    // Build Voronoi - should handle gracefully without panicking
    let voronoi = compute_voronoi_gpu_style(&points);
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
    assert!(
        total_issues <= num_duplicates + 20,
        "Degenerate cells ({}) should be bounded by coincident point count ({})",
        total_issues,
        num_duplicates + 20
    );
}

/// Test that the algorithm correctly skips neighbors below MIN_BISECTOR_DISTANCE.
#[test]
fn test_bisector_distance_threshold_applied() {
    use super::cell_builder::F64CellBuilder;

    println!("\n=== Bisector Distance Threshold Test ===\n");

    // Create a generator and neighbors at various distances
    let generator = Vec3::new(1.0, 0.0, 0.0);
    let mut builder = F64CellBuilder::new(0, generator);

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
    assert!(
        builder.vertex_count() >= 3,
        "Should form a cell from the 3 valid neighbors"
    );
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

/// Stress test for f64 slack-based certification.
/// Tests that f64 can certify vertices across various point distributions.
#[test]
#[ignore] // Run with: cargo test test_f64_slack_stress -- --ignored --nocapture
fn test_f64_slack_stress() {
    use super::cell_builder::F64CellBuilder;
    use crate::geometry::fibonacci_sphere_points_with_rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    println!("\n=== F64 Slack-Based Certification Stress Test ===\n");

    // Test configurations: (size, jitter_scale, description)
    let configs: &[(usize, f32, &str)] = &[
        (5_000, 0.15, "mild jitter"),
        (5_000, 0.35, "moderate jitter"),
        (5_000, 0.50, "heavy jitter"),
        (10_000, 0.25, "default jitter"),
        (10_000, 0.45, "rough jitter"),
        (25_000, 0.25, "large default"),
        (25_000, 0.40, "large rough"),
        (50_000, 0.25, "very large"),
    ];

    let seeds: Vec<u64> = (0..10).map(|i| 12345 + i * 7919).collect();

    let mut total_tests = 0usize;
    let mut total_cells = 0usize;
    let mut total_vertices = 0usize;

    for &(n, jitter_scale, desc) in configs {
        let mean_spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
        let jitter = mean_spacing * jitter_scale;

        for &seed in &seeds {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let points = fibonacci_sphere_points_with_rng(n, jitter, &mut rng);
            let knn = CubeMapGridKnn::new(&points);

            let mut cells_built = 0usize;
            let mut vertices_certified = 0usize;

            for cell_idx in 0..n {
                // Build f64 cell
                let mut f64_builder = F64CellBuilder::new(cell_idx, points[cell_idx]);
                let mut scratch = knn.make_scratch();
                let mut neighbors = Vec::with_capacity(24);
                knn.knn_into(points[cell_idx], cell_idx, 24, &mut scratch, &mut neighbors);

                for &neighbor_idx in &neighbors {
                    f64_builder.clip(neighbor_idx, points[neighbor_idx]);
                }

                if f64_builder.vertex_count() < 3 {
                    continue;
                }

                cells_built += 1;

                // This will panic if certification fails
                let mut support_data: Vec<u32> = Vec::new();
                let vertex_data = f64_builder.to_vertex_data(&points, &mut support_data);
                vertices_certified += vertex_data.len();
            }

            total_tests += 1;
            total_cells += cells_built;
            total_vertices += vertices_certified;

            println!(
                "[ok] n={:5}, seed={:5}, {}: cells={:5}, vertices={}",
                n, seed, desc, cells_built, vertices_certified
            );
        }
    }

    println!("\n=== Summary ===");
    println!("Total test cases: {}", total_tests);
    println!("Total cells built: {}", total_cells);
    println!("Total vertices certified: {}", total_vertices);
    println!("\nStress test passed!");
}

/// Verify our error calculations using arbitrary precision arithmetic.
/// This test reproduces the failing certification case and computes the "true"
/// vertex position and gaps using 256-bit precision to check if our f64 error
/// model is correct.
#[test]
#[ignore] // Run with: cargo test test_verify_error_with_arb -- --ignored --nocapture
fn test_verify_error_with_arb() {
    use super::cell_builder::F64CellBuilder;
    use super::constants::{SUPPORT_EPS_ABS, SUPPORT_VERTEX_ANGLE_EPS};
    use crate::geometry::fibonacci_sphere_points_with_rng;
    use astro_float::{BigFloat, RoundingMode};
    use glam::DVec3;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    const PREC: usize = 256;
    let rm = RoundingMode::None;

    // Helper to convert BigFloat to f64 via string (inefficient but works)
    fn bf_to_f64(bf: &BigFloat) -> f64 {
        format!("{}", bf).parse().unwrap_or(f64::NAN)
    }

    println!("\n=== Verify Error Model with Arbitrary Precision ===\n");

    // Reproduce the failing case: n=5000, seed=12345, jitter=0.15, cell=41
    let n = 5000usize;
    let seed = 12345u64;
    let jitter_scale = 0.15f32;
    let target_cell = 41usize;

    let mean_spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
    let jitter = mean_spacing * jitter_scale;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let points = fibonacci_sphere_points_with_rng(n, jitter, &mut rng);
    let knn = CubeMapGridKnn::new(&points);

    // Build f64 cell for target
    let mut f64_builder = F64CellBuilder::new(target_cell, points[target_cell]);
    let mut scratch = knn.make_scratch();
    let mut neighbors = Vec::with_capacity(24);
    knn.knn_into(
        points[target_cell],
        target_cell,
        24,
        &mut scratch,
        &mut neighbors,
    );

    for &neighbor_idx in &neighbors {
        f64_builder.clip(neighbor_idx, points[neighbor_idx]);
    }

    println!(
        "Cell {} has {} vertices",
        target_cell,
        f64_builder.vertex_count()
    );
    println!("Neighbors: {:?}", neighbors);

    // Get generator position
    let gen_f32 = points[target_cell];
    let gen_f64 = DVec3::new(gen_f32.x as f64, gen_f32.y as f64, gen_f32.z as f64).normalize();

    // For each vertex, compute gaps using f64 and arbitrary precision, compare
    for (v_idx, v_pos_raw, plane_a, plane_b) in f64_builder.vertices_iter() {
        let v_f64 = v_pos_raw.normalize();
        let na_idx = f64_builder.neighbor_index(plane_a);
        let nb_idx = f64_builder.neighbor_index(plane_b);

        println!(
            "\n--- Vertex {} (defining: {}, {}, {}) ---",
            v_idx, target_cell, na_idx, nb_idx
        );
        println!(
            "f64 vertex pos: ({:.15}, {:.15}, {:.15})",
            v_f64.x, v_f64.y, v_f64.z
        );

        let g = points[target_cell];
        let na = points[na_idx];
        let nb = points[nb_idx];

        // Helper to normalize a BigFloat vector
        let normalize_bf =
            |x: &BigFloat, y: &BigFloat, z: &BigFloat| -> (BigFloat, BigFloat, BigFloat) {
                let len_sq = x.mul(x, PREC, rm).add(&y.mul(y, PREC, rm), PREC, rm).add(
                    &z.mul(z, PREC, rm),
                    PREC,
                    rm,
                );
                let len = len_sq.sqrt(PREC, rm);
                (
                    x.div(&len, PREC, rm),
                    y.div(&len, PREC, rm),
                    z.div(&len, PREC, rm),
                )
            };

        // Convert to BigFloat and NORMALIZE (matching F64CellBuilder)
        let gx = BigFloat::from_f32(g.x, PREC);
        let gy = BigFloat::from_f32(g.y, PREC);
        let gz = BigFloat::from_f32(g.z, PREC);
        let (gx, gy, gz) = normalize_bf(&gx, &gy, &gz);

        let nax = BigFloat::from_f32(na.x, PREC);
        let nay = BigFloat::from_f32(na.y, PREC);
        let naz = BigFloat::from_f32(na.z, PREC);
        let (nax, nay, naz) = normalize_bf(&nax, &nay, &naz);

        let nbx = BigFloat::from_f32(nb.x, PREC);
        let nby = BigFloat::from_f32(nb.y, PREC);
        let nbz = BigFloat::from_f32(nb.z, PREC);
        let (nbx, nby, nbz) = normalize_bf(&nbx, &nby, &nbz);

        // Bisector plane directions: da = g - na, db = g - nb
        let da_x = gx.sub(&nax, PREC, rm);
        let da_y = gy.sub(&nay, PREC, rm);
        let da_z = gz.sub(&naz, PREC, rm);
        // Normalize to get plane normal (matching F64GreatCircle::bisector)
        let (pa_x, pa_y, pa_z) = normalize_bf(&da_x, &da_y, &da_z);

        let db_x = gx.sub(&nbx, PREC, rm);
        let db_y = gy.sub(&nby, PREC, rm);
        let db_z = gz.sub(&nbz, PREC, rm);
        let (pb_x, pb_y, pb_z) = normalize_bf(&db_x, &db_y, &db_z);

        // Cross product: c = pa × pb (now using normalized plane normals)
        let cx = pa_y
            .mul(&pb_z, PREC, rm)
            .sub(&pa_z.mul(&pb_y, PREC, rm), PREC, rm);
        let cy = pa_z
            .mul(&pb_x, PREC, rm)
            .sub(&pa_x.mul(&pb_z, PREC, rm), PREC, rm);
        let cz = pa_x
            .mul(&pb_y, PREC, rm)
            .sub(&pa_y.mul(&pb_x, PREC, rm), PREC, rm);

        // Length: |c|
        let c_len_sq = cx
            .mul(&cx, PREC, rm)
            .add(&cy.mul(&cy, PREC, rm), PREC, rm)
            .add(&cz.mul(&cz, PREC, rm), PREC, rm);
        let c_len = c_len_sq.sqrt(PREC, rm);

        // Normalize: v = c / |c|
        let vx_arb = cx.div(&c_len, PREC, rm);
        let vy_arb = cy.div(&c_len, PREC, rm);
        let vz_arb = cz.div(&c_len, PREC, rm);

        // Check sign: pick the vertex that matches f64 direction (closer distance)
        let vx_f_test = bf_to_f64(&vx_arb);
        let vy_f_test = bf_to_f64(&vy_arb);
        let vz_f_test = bf_to_f64(&vz_arb);

        let dist_pos = (vx_f_test - v_f64.x).powi(2)
            + (vy_f_test - v_f64.y).powi(2)
            + (vz_f_test - v_f64.z).powi(2);
        let dist_neg = (vx_f_test + v_f64.x).powi(2)
            + (vy_f_test + v_f64.y).powi(2)
            + (vz_f_test + v_f64.z).powi(2);

        let (vx_arb, vy_arb, vz_arb) = if dist_neg < dist_pos {
            (vx_arb.neg(), vy_arb.neg(), vz_arb.neg())
        } else {
            (vx_arb, vy_arb, vz_arb)
        };

        let vx_f = bf_to_f64(&vx_arb);
        let vy_f = bf_to_f64(&vy_arb);
        let vz_f = bf_to_f64(&vz_arb);

        println!("ARB vertex pos: ({:.15}, {:.15}, {:.15})", vx_f, vy_f, vz_f);

        // Position error
        let pos_err =
            ((vx_f - v_f64.x).powi(2) + (vy_f - v_f64.y).powi(2) + (vz_f - v_f64.z).powi(2)).sqrt();
        println!("Position error (f64 vs ARB): {:.2e}", pos_err);

        // Normalize generator
        let g_len_sq = gx
            .mul(&gx, PREC, rm)
            .add(&gy.mul(&gy, PREC, rm), PREC, rm)
            .add(&gz.mul(&gz, PREC, rm), PREC, rm);
        let g_len = g_len_sq.sqrt(PREC, rm);
        let gx_n = gx.div(&g_len, PREC, rm);
        let gy_n = gy.div(&g_len, PREC, rm);
        let gz_n = gz.div(&g_len, PREC, rm);

        let dot_g_f64 = v_f64.dot(gen_f64);
        let dot_g_arb = vx_arb
            .mul(&gx_n, PREC, rm)
            .add(&vy_arb.mul(&gy_n, PREC, rm), PREC, rm)
            .add(&vz_arb.mul(&gz_n, PREC, rm), PREC, rm);

        println!("dot(V, G) f64: {:.15}", dot_g_f64);
        println!("dot(V, G) ARB: {:.15}", bf_to_f64(&dot_g_arb));

        // Analyze gaps with error model: gap_error ≤ pos_error * |G - C|
        //
        // For certification we need:
        // - OUT generators: true_gap > SUPPORT_EPS_ABS (can't enter support set)
        // - IN generators: true_gap ≤ SUPPORT_EPS_ABS (won't leave support set)
        //
        // Since true_gap ∈ [f64_gap - gap_error_bound, f64_gap + gap_error_bound]
        // we check worst case for each direction.

        println!("\nGap analysis (gap_error_bound = pos_err * |G-C|):");
        println!("  pos_err = {:.2e}", pos_err);

        let mut min_out_margin = f64::MAX; // smallest (true_gap - EPS) for OUT generators
        let mut max_in_gap = f64::MIN; // largest true_gap for IN generators

        for &neighbor_idx in &neighbors {
            if neighbor_idx == na_idx || neighbor_idx == nb_idx {
                continue;
            }

            let c = points[neighbor_idx];
            let c_f64 = DVec3::new(c.x as f64, c.y as f64, c.z as f64).normalize();

            // |G - C| (distance between normalized generators)
            let g_minus_c = (gen_f64 - c_f64).length();

            // Predicted gap error bound
            let gap_error_bound = pos_err * g_minus_c;

            // f64 gap
            let dot_c_f64 = v_f64.dot(c_f64);
            let gap_f64 = dot_g_f64 - dot_c_f64;

            // ARB gap (true gap)
            let cx_raw = BigFloat::from_f32(c.x, PREC);
            let cy_raw = BigFloat::from_f32(c.y, PREC);
            let cz_raw = BigFloat::from_f32(c.z, PREC);
            let c_len_sq = cx_raw
                .mul(&cx_raw, PREC, rm)
                .add(&cy_raw.mul(&cy_raw, PREC, rm), PREC, rm)
                .add(&cz_raw.mul(&cz_raw, PREC, rm), PREC, rm);
            let c_len = c_len_sq.sqrt(PREC, rm);
            let cx_n = cx_raw.div(&c_len, PREC, rm);
            let cy_n = cy_raw.div(&c_len, PREC, rm);
            let cz_n = cz_raw.div(&c_len, PREC, rm);

            let dot_c_arb = vx_arb
                .mul(&cx_n, PREC, rm)
                .add(&vy_arb.mul(&cy_n, PREC, rm), PREC, rm)
                .add(&vz_arb.mul(&cz_n, PREC, rm), PREC, rm);

            let gap_arb = dot_g_arb.sub(&dot_c_arb, PREC, rm);
            let gap_arb_f = bf_to_f64(&gap_arb);

            let actual_gap_err = (gap_f64 - gap_arb_f).abs();

            // Classification based on f64 gap
            let in_support = gap_f64 <= SUPPORT_EPS_ABS;

            // Worst-case true gap for membership decision
            // If OUT: could it enter? Check if gap_f64 - gap_error_bound <= EPS
            // If IN: could it leave? Check if gap_f64 + gap_error_bound > EPS
            let could_change = if in_support {
                gap_f64 + gap_error_bound > SUPPORT_EPS_ABS
            } else {
                gap_f64 - gap_error_bound <= SUPPORT_EPS_ABS
            };

            // Track margins
            if !in_support {
                let margin = gap_arb_f - SUPPORT_EPS_ABS;
                if margin < min_out_margin {
                    min_out_margin = margin;
                }
            } else {
                if gap_arb_f > max_in_gap {
                    max_in_gap = gap_arb_f;
                }
            }

            // Print interesting cases (small gaps or potential membership changes)
            if gap_f64.abs() < 2e-5 || could_change {
                println!(
                    "  neighbor {:3}: |G-C|={:.4e} gap_f64={:.6e} gap_true={:.6e} err_bound={:.2e} actual_err={:.2e} {} {}",
                    neighbor_idx, g_minus_c, gap_f64, gap_arb_f, gap_error_bound, actual_gap_err,
                    if in_support { "[IN]" } else { "[OUT]" },
                    if could_change { "*** COULD CHANGE ***" } else { "" }
                );
            }
        }

        println!("\nCertification check:");
        println!("  SUPPORT_EPS_ABS = {:.6e}", SUPPORT_EPS_ABS);
        println!(
            "  Min margin for OUT generators (true_gap - EPS): {:.6e}",
            min_out_margin
        );
        if max_in_gap > f64::MIN {
            println!("  Max gap for IN generators: {:.6e}", max_in_gap);
        }

        let could_certify = min_out_margin > 0.0;
        println!(
            "  Could certify (all OUT generators have true_gap > EPS): {}",
            could_certify
        );

        println!("\nOur current threshold model:");
        let our_margin = 2.0 * (SUPPORT_VERTEX_ANGLE_EPS + f64::EPSILON * 16.0).sin();
        println!("  f64_error_margin (2*sin(eps_angle)) = {:.6e}", our_margin);
        println!(
            "  cert_threshold = EPS + margin = {:.6e}",
            SUPPORT_EPS_ABS + our_margin
        );
    }
}

#[test]
fn test_gpu_voronoi_10k_timing() {
    use crate::geometry::random_sphere_points;
    use std::time::Instant;

    let points = random_sphere_points(10_000);

    println!("\nTesting 10k cells with GPU voronoi...");
    let t0 = Instant::now();
    let voronoi = compute_voronoi_gpu_style(&points);
    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

    println!("Time: {:.1}ms", elapsed);
    println!("Vertices: {}", voronoi.vertices.len());
    println!("Cells: {}", voronoi.num_cells());
}

#[test]
#[ignore]
fn test_gpu_voronoi_50k_timing() {
    use crate::geometry::random_sphere_points;
    use std::time::Instant;

    let points = random_sphere_points(50_000);

    println!("\nTesting 50k cells with GPU voronoi...");
    let t0 = Instant::now();
    let voronoi = compute_voronoi_gpu_style(&points);
    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

    println!("Time: {:.1}ms", elapsed);
    println!("Vertices: {}", voronoi.vertices.len());
    println!("Cells: {}", voronoi.num_cells());
}

#[test]
#[ignore]
fn test_gpu_voronoi_100k_validation() {
    use crate::geometry::random_sphere_points;
    use crate::geometry::validation::validate_voronoi;
    use std::time::Instant;

    let points = random_sphere_points(100_000);

    println!("\nTesting 100k cells with GPU voronoi...");
    let t0 = Instant::now();
    let voronoi = compute_voronoi_gpu_style(&points);
    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

    println!("Time: {:.1}ms", elapsed);
    println!("Vertices: {}", voronoi.vertices.len());
    println!("Cells: {}", voronoi.num_cells());

    // Validate
    let result = validate_voronoi(&voronoi, 1e-6);
    result.print_summary();
}
