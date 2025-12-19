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

/// Test that degeneracy detection finds cases where 4+ generators are equidistant.
#[test]
fn test_degeneracy_detection() {
    use rand::{rngs::StdRng, SeedableRng};
    use crate::geometry::random_sphere_points_with_rng;

    // Run with a few seeds and sizes - degeneracy is rare in random points
    let test_cases = [(1000, 12345u64), (5000, 99999)];

    for (n, seed) in test_cases {
        let mut rng = StdRng::seed_from_u64(seed);
        let points = random_sphere_points_with_rng(n, &mut rng);
        let knn = CubeMapGridKnn::new(&points);
        let termination = TerminationConfig {
            enabled: true,
            check_start: 10,
            check_step: 6,
        };

        let flat_data = build_cells_data_flat(&points, &knn, DEFAULT_K, termination);

        // Degeneracy count should be small for random points
        // This mostly tests that the code runs without panicking
        assert!(
            flat_data.degenerate_edges.len() < n / 10,
            "Too many degenerate edges for random points: {}",
            flat_data.degenerate_edges.len()
        );
    }
}

#[test]
fn test_parallel_dedup_matches_hash_dedup() {
    // Test that the parallel dedup produces equivalent results to the hash-based dedup.
    // Note: The two methods may pick different representative positions for the same triplet
    // (due to floating point differences when computing from different cells), so we use
    // approximate comparison.

    for n in [100, 1000, 5000] {
        let points = random_sphere_points(n);
        let knn = CubeMapGridKnn::new(&points);
        let termination = TerminationConfig {
            enabled: true,
            check_start: 10,
            check_step: 6,
        };

        let (cells_data, _degenerate) = build_cells_data_incremental(&points, &knn, DEFAULT_K, termination);

        // Clone cells_data for parallel version
        let cells_data_clone = cells_data.clone();

        // Run both dedup methods
        let (hash_verts, hash_cells, hash_indices) =
            super::dedup::dedup_vertices_hash(n, cells_data);
        let (par_verts, par_cells, par_indices) =
            super::dedup::dedup_vertices_parallel(n, cells_data_clone);

        // Check that cell counts match
        assert_eq!(hash_cells.len(), par_cells.len(), "n={}: cell count mismatch", n);

        // Check that each cell has the same number of vertices
        for (i, (hc, pc)) in hash_cells.iter().zip(par_cells.iter()).enumerate() {
            assert_eq!(
                hc.vertex_count(), pc.vertex_count(),
                "n={}: cell {} vertex count mismatch ({} vs {})",
                n, i, hc.vertex_count(), pc.vertex_count()
            );
        }

        // Check that vertex positions are approximately equivalent
        // For each vertex in hash result, there should be a close vertex in parallel result
        // The two methods may pick different representative positions for the same triplet
        // (computed from different cells with floating point differences up to ~2e-3 in rare cases)
        let tol = 2e-3;
        let mut mismatches = 0;
        for (i, (hc, pc)) in hash_cells.iter().zip(par_cells.iter()).enumerate() {
            let hash_cell_verts: Vec<Vec3> = hash_indices[hc.vertex_start()..hc.vertex_start() + hc.vertex_count()]
                .iter()
                .map(|&idx| hash_verts[idx])
                .collect();

            let par_cell_verts: Vec<Vec3> = par_indices[pc.vertex_start()..pc.vertex_start() + pc.vertex_count()]
                .iter()
                .map(|&idx| par_verts[idx])
                .collect();

            // Each hash vertex should have a close match in parallel vertices
            for hv in &hash_cell_verts {
                let has_match = par_cell_verts.iter().any(|pv| (*hv - *pv).length() < tol);
                if !has_match {
                    mismatches += 1;
                }
            }
        }
        assert_eq!(mismatches, 0, "n={}: {} vertex mismatches, {} hash verts, {} par verts",
            n, mismatches, hash_verts.len(), par_verts.len());

        println!("n={}: parallel dedup matches hash dedup ({} cells, {} verts vs {} verts)",
            n, hash_cells.len(), hash_verts.len(), par_verts.len());
    }
}

#[test]
#[ignore] // Run with: cargo test bench_parallel_vs_hash_dedup -- --ignored --nocapture
fn bench_parallel_vs_hash_dedup() {
    use std::time::Instant;

    println!("\nDedup benchmark: parallel vs hash-based");
    println!("{:>10} {:>12} {:>12} {:>10}", "n", "hash (ms)", "parallel (ms)", "speedup");
    println!("{:-<50}", "");

    for n in [10_000, 50_000, 100_000, 250_000, 500_000] {
        let points = random_sphere_points(n);
        let knn = CubeMapGridKnn::new(&points);
        let termination = TerminationConfig {
            enabled: true,
            check_start: 10,
            check_step: 6,
        };

        let (cells_data, _) = build_cells_data_incremental(&points, &knn, DEFAULT_K, termination);

        // Clone for parallel version
        let cells_data_clone = cells_data.clone();

        // Benchmark hash-based dedup
        let t0 = Instant::now();
        let (_verts, _cells, _indices) = super::dedup::dedup_vertices_hash(n, cells_data);
        let hash_time = t0.elapsed().as_secs_f64() * 1000.0;

        // Benchmark parallel dedup (with timing breakdown for largest size)
        let t0 = Instant::now();
        let print_breakdown = n >= 250_000;
        let (_verts, _cells, _indices) = super::dedup::dedup_vertices_parallel_timed(n, cells_data_clone, print_breakdown);
        let par_time = t0.elapsed().as_secs_f64() * 1000.0;

        let speedup = hash_time / par_time;
        println!("{:>10} {:>12.2} {:>12.2} {:>10.2}x", n, hash_time, par_time, speedup);
    }
}

#[test]
#[ignore] // Run with: cargo test bench_allocation_cost -- --ignored --nocapture
fn bench_allocation_cost() {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    println!("\nAllocation cost: breakdown of cell construction phases");
    println!("{:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "n", "total", "knn", "clip", "extract", "extract%");
    println!("{:-<70}", "");

    for n in [50_000, 100_000, 250_000, 500_000] {
        let points = random_sphere_points(n);
        let knn = CubeMapGridKnn::new(&points);

        // Use atomics to accumulate per-phase times across threads
        let knn_ns = AtomicU64::new(0);
        let clip_ns = AtomicU64::new(0);
        let extract_ns = AtomicU64::new(0);

        let t0 = Instant::now();
        let _cells: Vec<super::CellVerts> = (0..points.len())
            .into_par_iter()
            .map_init(
                || (
                    knn.make_scratch(),
                    Vec::with_capacity(DEFAULT_K),
                    IncrementalCellBuilder::new(0, glam::Vec3::ZERO),
                ),
                |(scratch, neighbors, builder), i| {
                    // Phase 1: k-NN lookup
                    let t_knn = Instant::now();
                    neighbors.clear();
                    knn.knn_into(points[i], i, DEFAULT_K, scratch, neighbors);
                    knn_ns.fetch_add(t_knn.elapsed().as_nanos() as u64, Ordering::Relaxed);

                    // Phase 2: Clipping
                    let t_clip = Instant::now();
                    builder.reset(i, points[i]);
                    for &neighbor_idx in neighbors.iter() {
                        builder.clip(neighbor_idx, points[neighbor_idx]);
                    }
                    clip_ns.fetch_add(t_clip.elapsed().as_nanos() as u64, Ordering::Relaxed);

                    // Phase 3: Extract vertices (allocates Vec)
                    let t_extract = Instant::now();
                    let result = builder.get_vertices_with_keys();
                    extract_ns.fetch_add(t_extract.elapsed().as_nanos() as u64, Ordering::Relaxed);

                    result
                },
            )
            .collect();
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let knn_ms = knn_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let clip_ms = clip_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let extract_ms = extract_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0;

        // Note: sum of phases > total because of parallelism (wall time vs CPU time)
        // The percentage tells us relative weight
        let phase_total = knn_ms + clip_ms + extract_ms;
        let extract_pct = (extract_ms / phase_total) * 100.0;

        println!("{:>10} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1}%",
            n, total_ms, knn_ms, clip_ms, extract_ms, extract_pct);
    }

    println!("\nNote: knn+clip+extract are CPU-time (sum across threads), total is wall-time.");
    println!("extract% shows the relative weight of vertex extraction (includes allocation).");
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
