//! Tests for GPU-style Voronoi computation.

use glam::Vec3;

use super::*;
use crate::geometry::random_sphere_points;

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

    let (cells_data, _degenerate_triplets) = build_cells_data_incremental(&points, &knn, DEFAULT_K, termination);

    let mut already_ordered = 0;
    let mut rotated_ccw = 0;  // CCW but starting from different vertex
    let mut reversed = 0;     // CW order (needs reversal)

    for (i, keyed_verts) in cells_data.iter().enumerate() {
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

#[test]
#[ignore] // Run with: cargo test benchmark_merge_cost -- --ignored --nocapture
fn benchmark_merge_cost() {
    use std::time::Instant;

    println!("\n=== merge_coincident_vertices Cost Analysis ===\n");

    for &n in &[10_000, 50_000, 100_000, 500_000, 1_000_000] {
        let points = random_sphere_points(n);
        let knn = CubeMapGridKnn::new(&points);
        let termination = TerminationConfig {
            enabled: true,
            check_start: 10,
            check_step: 6,
        };

        // Build cells
        let (cells_data, _degenerate_edges) =
            build_cells_data_incremental(&points, &knn, DEFAULT_K, termination);

        // Collect all vertices with triplet keys (simulating first part of dedup)
        let bits = super::dedup::bits_for_indices(n.saturating_sub(1));
        let mut all_vertices: Vec<Vec3> = Vec::new();
        let mut triplet_to_index: rustc_hash::FxHashMap<u128, usize> =
            rustc_hash::FxHashMap::default();

        for keyed_verts in &cells_data {
            for (triplet, pos) in keyed_verts {
                let key = super::dedup::pack_triplet_u128(*triplet, bits);
                triplet_to_index.entry(key).or_insert_with(|| {
                    let idx = all_vertices.len();
                    all_vertices.push(*pos);
                    idx
                });
            }
        }

        let num_unique_verts = all_vertices.len();

        // Time the merge pass
        let t0 = Instant::now();
        let (merged, remap) = super::dedup::merge_coincident_vertices(&all_vertices, 1e-5);
        let merge_time_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let merged_count = num_unique_verts - merged.len();
        let merge_rate = merged_count as f64 / num_unique_verts as f64 * 100.0;

        println!(
            "n={:>7}: {:>7} unique verts -> {:>7} merged ({:>5} removed, {:.3}%) in {:6.1}ms",
            n, num_unique_verts, merged.len(), merged_count, merge_rate, merge_time_ms
        );

        // Also check how many remaps are non-identity
        let non_identity = remap.iter().enumerate().filter(|(i, &v)| *i != v).count();
        if non_identity > 0 {
            println!("         {} indices remapped", non_identity);
        }
    }
}

#[test]
#[ignore] // Run with: cargo test test_degeneracy_heuristic -- --ignored --nocapture
fn test_degeneracy_heuristic() {
    // Test that the degeneracy heuristic correctly identifies when merge pass is needed.
    // Random points should rarely have 4+ equidistant generators.
    use rand::{SeedableRng, rngs::StdRng};
    use crate::geometry::random_sphere_points_with_rng;

    println!("\nDegeneracy heuristic test:");
    println!("Tests whether random points trigger degeneracy detection.\n");

    let sizes = [500, 1_000, 5_000, 10_000, 50_000];
    let seeds = [12345u64, 99999, 42, 1337, 7654321];

    let mut total_detected = 0;
    let mut total_missed = 0;
    let mut total_false_positives = 0;
    let mut total_no_op = 0;

    for &n in &sizes {
        println!("n={}:", n);
        for &seed in &seeds {
            let mut rng = StdRng::seed_from_u64(seed);
            let points = random_sphere_points_with_rng(n, &mut rng);
            let knn = CubeMapGridKnn::new(&points);
            let termination = TerminationConfig {
                enabled: true,
                check_start: 10,
                check_step: 6,
            };

            let (cells_data, degenerate_edges) =
                build_cells_data_incremental(&points, &knn, DEFAULT_K, termination);
            let has_degeneracy = !degenerate_edges.is_empty();

            // Check how many vertices would actually be merged
            let bits = super::dedup::bits_for_indices(n.saturating_sub(1));
            let mut all_vertices: Vec<Vec3> = Vec::new();
            let mut triplet_to_index: rustc_hash::FxHashMap<u128, usize> =
                rustc_hash::FxHashMap::default();

            for keyed_verts in &cells_data {
                for (triplet, pos) in keyed_verts {
                    let key = super::dedup::pack_triplet_u128(*triplet, bits);
                    triplet_to_index.entry(key).or_insert_with(|| {
                        let idx = all_vertices.len();
                        all_vertices.push(*pos);
                        idx
                    });
                }
            }

            let (merged, _remap) = super::dedup::merge_coincident_vertices(&all_vertices, 1e-5);
            let merged_count = all_vertices.len() - merged.len();

            let status = if has_degeneracy && merged_count > 0 {
                total_detected += 1;
                "✓ detected"
            } else if !has_degeneracy && merged_count == 0 {
                total_no_op += 1;
                "✓ no-op"
            } else if !has_degeneracy && merged_count > 0 {
                total_missed += 1;
                "✗ MISSED"
            } else {
                total_false_positives += 1;
                "~ false positive"
            };

            println!(
                "  seed={:>7}: degeneracy={:>5} ({:>4} triplets), merges={:>3} -> {}",
                seed, has_degeneracy, degenerate_edges.len(), merged_count, status
            );
        }
    }

    println!("\nSummary: {} detected, {} no-op, {} missed, {} false positives",
        total_detected, total_no_op, total_missed, total_false_positives);
}

#[test]
#[ignore] // Run with: cargo test benchmark_targeted_merge -- --ignored --nocapture
fn benchmark_targeted_merge() {
    // Compare full merge vs targeted merge performance
    use std::time::Instant;

    println!("\nTargeted merge benchmark:");
    println!("Comparing full merge (all vertices) vs targeted merge (degenerate only)\n");

    let sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000];

    for &n in &sizes {
        let points = random_sphere_points(n);
        let knn = CubeMapGridKnn::new(&points);
        let termination = TerminationConfig {
            enabled: true,
            check_start: 10,
            check_step: 6,
        };

        let (cells_data, degenerate_edges) =
            build_cells_data_incremental(&points, &knn, DEFAULT_K, termination);

        // Collect all vertices with triplet keys
        let bits = super::dedup::bits_for_indices(n.saturating_sub(1));
        let mut all_vertices: Vec<Vec3> = Vec::new();
        let mut triplet_to_index: rustc_hash::FxHashMap<u128, usize> =
            rustc_hash::FxHashMap::default();

        for keyed_verts in &cells_data {
            for (triplet, pos) in keyed_verts {
                let key = super::dedup::pack_triplet_u128(*triplet, bits);
                triplet_to_index.entry(key).or_insert_with(|| {
                    let idx = all_vertices.len();
                    all_vertices.push(*pos);
                    idx
                });
            }
        }

        let num_verts = all_vertices.len();
        let num_edges = degenerate_edges.len();
        let num_degenerate = num_edges;

        // Time full merge
        let t0 = Instant::now();
        let (full_merged, _) = super::dedup::merge_coincident_vertices(&all_vertices, 1e-5);
        let full_time_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let full_merge_count = num_verts - full_merged.len();

        // Time targeted merge (simulated - using full merge on degenerate subset)
        // For a true comparison, we need to implement a test version
        let t1 = Instant::now();
        // Only merge a sampled subset (edges imply degeneracy sites).
        // This is not comparable to DSU unification but gives a rough upper bound on subset merge cost.
        let mut sampled: Vec<Vec3> = Vec::with_capacity(num_edges.min(10_000));
        for (t1, _t2) in degenerate_edges.iter().take(10_000) {
            let key = super::dedup::pack_triplet_u128(*t1, bits);
            if let Some(&idx) = triplet_to_index.get(&key) {
                sampled.push(all_vertices[idx]);
            }
        }
        let (_, _) = super::dedup::merge_coincident_vertices(&sampled, 1e-5);
        let targeted_time_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let speedup = full_time_ms / targeted_time_ms.max(0.001);

        println!(
            "n={:>7}: {:>7} verts, {:>5} degenerate ({:.2}%), full={:6.1}ms, targeted={:6.1}ms, speedup={:.1}x, merges={}",
            n, num_verts, num_degenerate,
            num_degenerate as f64 / num_verts as f64 * 100.0,
            full_time_ms, targeted_time_ms, speedup, full_merge_count
        );
    }
}
