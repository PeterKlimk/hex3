use super::*;
use crate::geometry::fibonacci_sphere_points_with_rng;
use crate::geometry::gpu_voronoi::{build_kdtree, find_k_nearest as kiddo_find_k_nearest};
use glam::Vec3;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const TARGET_POINTS_PER_CELL: f64 = 24.0;

fn default_res(n: usize) -> usize {
    ((n as f64 / (6.0 * TARGET_POINTS_PER_CELL)).sqrt() as usize).max(4)
}

fn generate_test_points(n: usize, seed: u64) -> Vec<Vec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mean_spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
    let jitter = mean_spacing * 0.25;
    fibonacci_sphere_points_with_rng(n, jitter, &mut rng)
}

#[test]
fn test_cube_grid_basic() {
    let points = generate_test_points(10_000, 12345);
    let grid = CubeMapGrid::new(&points, 20);

    let stats = grid.stats();
    println!("Grid stats: {:?}", stats);

    // Check that all points are accounted for
    assert_eq!(stats.num_points, 10_000);

    // Check reasonable distribution
    assert!(stats.avg_points_per_cell > 1.0);
    assert!(stats.max_points_per_cell < 500); // Not too uneven
}

#[test]
fn test_cube_grid_cell_bounds_are_conservative() {
    // Sanity check: random points inside some cells should be within the precomputed cap.
    // This doesn't prove correctness but guards against gross underestimation.
    use rand::Rng;

    let points = generate_test_points(10_000, 12345);
    let grid = CubeMapGrid::new(&points, 32);
    let mut rng = rand::thread_rng();

    let num_cells = 6 * grid.res * grid.res;
    for _ in 0..100 {
        let cell = rng.gen_range(0..num_cells);
        let (face, iu, iv) = cell_to_face_ij(cell, grid.res);
        let u0 = (iu as f32) / grid.res as f32 * 2.0 - 1.0;
        let u1 = ((iu + 1) as f32) / grid.res as f32 * 2.0 - 1.0;
        let v0 = (iv as f32) / grid.res as f32 * 2.0 - 1.0;
        let v1 = ((iv + 1) as f32) / grid.res as f32 * 2.0 - 1.0;

        let center = grid.cell_centers[cell];
        let cap_angle = grid.cell_cos_radius[cell].clamp(-1.0, 1.0).acos() + 1e-3;

        for _ in 0..100 {
            let u = rng.gen_range(u0..u1);
            let v = rng.gen_range(v0..v1);
            let p = face_uv_to_3d(face, u, v);
            let ang = center.dot(p).clamp(-1.0, 1.0).acos();
            assert!(
                ang <= cap_angle,
                "cell cap underestimates (ang={ang}, cap={cap_angle})"
            );
        }
    }
}

#[test]
#[ignore] // Run with: cargo test test_cube_grid_exhaustive -- --ignored --nocapture
fn test_cube_grid_exhaustive() {
    use std::collections::HashSet;

    println!("\n=== Exhaustive CubeMapGrid Correctness Test ===\n");

    for &n in &[10_000usize, 50_000] {
        // Skip small n where resolution is problematic
        let points = generate_test_points(n, 12345);
        let k = 24;
        let res = default_res(n);

        let grid = CubeMapGrid::new(&points, res);
        let mut scratch = grid.make_scratch();
        let (tree, entries) = build_kdtree(&points);

        let mut exact_match = 0;
        let mut missing_neighbors = 0;
        let mut wrong_neighbors = 0;
        let mut worst_overlap = k;

        let mut first_failure_printed = false;
        for i in 0..n {
            let grid_knn = grid.find_k_nearest_with_scratch(&points, points[i], i, k, &mut scratch);
            let kiddo_knn = kiddo_find_k_nearest(&tree, &entries, points[i], i, k);

            let grid_set: HashSet<_> = grid_knn.iter().copied().collect();
            let kiddo_set: HashSet<_> = kiddo_knn.iter().copied().collect();

            let overlap = grid_set.intersection(&kiddo_set).count();
            worst_overlap = worst_overlap.min(overlap);

            if grid_set == kiddo_set {
                exact_match += 1;
            } else {
                // Check if grid is missing valid neighbors
                let missing: Vec<_> = kiddo_set.difference(&grid_set).copied().collect();
                let extra: Vec<_> = grid_set.difference(&kiddo_set).copied().collect();

                if !missing.is_empty() {
                    missing_neighbors += 1;
                    // Verify the missing ones are actually closer
                    let query = points[i];
                    for &m in &missing {
                        let missing_dist = (points[m] - query).length_squared();
                        // Check if any extra neighbor is farther
                        for &e in &extra {
                            let extra_dist = (points[e] - query).length_squared();
                            if extra_dist > missing_dist + 1e-9 {
                                wrong_neighbors += 1;
                            }
                        }
                    }

                    // Print first failure for debugging
                    if !first_failure_printed {
                        first_failure_printed = true;
                        let cell = grid.point_to_cell(query);
                        let neighbors = grid.cell_neighbors(cell);
                        let valid_neighbors: Vec<_> =
                            neighbors.iter().filter(|&&c| c != u32::MAX).collect();
                        let total_candidates: usize = valid_neighbors
                            .iter()
                            .map(|&&c| grid.cell_points(c as usize).len())
                            .sum();
                        println!(
                            "  First mismatch at i={}: cell={} neighbors={:?} candidates={}",
                            i,
                            cell,
                            valid_neighbors.len(),
                            total_candidates
                        );
                        println!(
                            "    grid_knn.len()={} grid_set.len()={} (DUPLICATES={})",
                            grid_knn.len(),
                            grid_set.len(),
                            grid_knn.len() - grid_set.len()
                        );
                        println!(
                            "    overlap={}, missing={}, extra={}",
                            overlap,
                            missing.len(),
                            extra.len()
                        );
                        // Print the neighbor cells to check for duplicates
                        println!("    neighbor_cells: {:?}", neighbors);
                    }
                }
            }
        }

        let match_pct = exact_match as f64 / n as f64 * 100.0;
        println!(
            "n={:>5} res={:>2}: exact={}/{} ({:.1}%) missing={} wrong={} worst_overlap={}/{}",
            n, res, exact_match, n, match_pct, missing_neighbors, wrong_neighbors, worst_overlap, k
        );

        // Strict correctness requirement
        assert_eq!(
            wrong_neighbors, 0,
            "Grid returned farther neighbors than kiddo!"
        );
    }

    println!("\nAll tests passed - no incorrect neighbors returned.");
}

#[test]
#[ignore] // Run with: cargo test bench_candidate_structures -- --ignored --nocapture
fn bench_candidate_structures() {
    use rand::Rng;
    use smallvec::SmallVec;
    use std::collections::BinaryHeap;
    use std::time::Instant;

    println!("\n=== Candidate Structure Microbench ===\n");
    println!("Measures per-insert maintenance cost for keeping the best k distances.\n");

    let mut rng = rand::thread_rng();
    let ops = 200_000usize;
    let ks = [8usize, 12, 24, 48, 96, 192];

    // Generate (dist, idx) samples once so all strategies see the same inputs.
    let mut samples: Vec<(f32, u32)> = Vec::with_capacity(ops);
    for i in 0..ops {
        // Distances in [0, 4] (unit-chord squared range is [0,4]).
        let d = rng.gen::<f32>() * 4.0;
        samples.push((d, i as u32));
    }

    fn vec_insert_best_k(buf: &mut Vec<(f32, u32)>, d: f32, idx: u32, k: usize) {
        if k == 0 {
            return;
        }
        if buf.len() == k && d >= buf[k - 1].0 {
            return;
        }
        let pos = buf.partition_point(|&(bd, _)| bd < d);
        buf.insert(pos, (d, idx));
        if buf.len() > k {
            buf.pop();
        }
    }

    fn smallvec_insert_best_k(buf: &mut SmallVec<[(f32, u32); 48]>, d: f32, idx: u32, k: usize) {
        if k == 0 {
            return;
        }
        if buf.len() == k && d >= buf[k - 1].0 {
            return;
        }
        let pos = buf.partition_point(|&(bd, _)| bd < d);
        buf.insert(pos, (d, idx));
        if buf.len() > k {
            buf.pop();
        }
    }

    fn heap_insert_best_k(heap: &mut BinaryHeap<(OrdF32, u32)>, d: f32, idx: u32, k: usize) {
        if k == 0 || d.is_nan() {
            return;
        }
        let d = OrdF32::new(d);
        if heap.len() < k {
            heap.push((d, idx));
            return;
        }
        if let Some((worst_d, _)) = heap.peek() {
            if d >= *worst_d {
                return;
            }
        }
        heap.pop();
        heap.push((d, idx));
    }

    for &k in &ks {
        println!("\n-- k={} --", k);

        // Vec
        let mut v: Vec<(f32, u32)> = Vec::with_capacity(k.min(256));
        let t0 = Instant::now();
        for &(d, idx) in &samples {
            vec_insert_best_k(&mut v, d, idx, k);
        }
        let dt = t0.elapsed();
        println!(
            "Vec sorted insert:   {:>8.2} ns/op",
            dt.as_secs_f64() * 1e9 / ops as f64
        );

        // SmallVec (inline up to 48, spills for larger)
        let mut sv: SmallVec<[(f32, u32); 48]> = SmallVec::new();
        sv.reserve(k.min(256));
        let t0 = Instant::now();
        for &(d, idx) in &samples {
            smallvec_insert_best_k(&mut sv, d, idx, k);
        }
        let dt = t0.elapsed();
        println!(
            "SmallVec sorted ins: {:>8.2} ns/op",
            dt.as_secs_f64() * 1e9 / ops as f64
        );

        // Heap (max-heap keeps best k)
        let mut heap: BinaryHeap<(OrdF32, u32)> = BinaryHeap::new();
        let t0 = Instant::now();
        for &(d, idx) in &samples {
            heap_insert_best_k(&mut heap, d, idx, k);
        }
        let dt = t0.elapsed();
        println!(
            "BinaryHeap top-k:    {:>8.2} ns/op",
            dt.as_secs_f64() * 1e9 / ops as f64
        );
    }
}

#[test]
#[ignore] // Run with: cargo test bench_cube_grid_knn -- --ignored --nocapture
fn bench_cube_grid_knn() {
    use std::time::Instant;

    println!("\n=== CubeMapGrid k-NN Benchmark ===\n");

    let n = 200_000;
    let points = generate_test_points(n, 12345);
    let res = default_res(n);
    let grid = CubeMapGrid::new(&points, res);
    let mut scratch = grid.make_scratch();

    let query_count = 20_000usize;
    let mut out = Vec::new();

    // (k, track_limit) pairs to exercise small/large modes.
    let configs: &[(usize, usize, &str)] = &[
        (12, 12, "small (k=12)"),
        (24, 24, "small (k=24)"),
        (48, 48, "small (k=48)"),
        (96, 96, "large (k=96)"),
        (192, 192, "large (k=192)"),
        (12, 256, "large track (k=12, track=256)"),
    ];

    for &(k, track, label) in configs {
        let t0 = Instant::now();
        let mut checksum = 0usize;

        for i in 0..query_count {
            let qi = (i * 97) % n;
            let q = points[qi];
            grid.find_k_nearest_resumable_into(&points, q, qi, k, track, &mut scratch, &mut out);
            // Touch output to prevent optimizer eliminating the call.
            checksum ^= out.get(0).copied().unwrap_or(0);
        }

        let dt = t0.elapsed();
        let ns_per = dt.as_secs_f64() * 1e9 / query_count as f64;
        println!(
            "{:<22} {:>8.2} ns/query (checksum={})",
            label, ns_per, checksum
        );
    }
}

#[test]
#[ignore] // Run with: cargo test test_cube_grid_vs_kiddo -- --ignored --nocapture
fn test_cube_grid_vs_kiddo() {
    use std::time::Instant;

    println!("\n=== CubeMapGrid vs Kiddo k-NN Comparison ===\n");

    for &n in &[10_000usize, 100_000, 500_000, 1_000_000, 2_000_000] {
        let points = generate_test_points(n, 12345);
        let k = 24;

        // Target ~24 points per cell
        let res = default_res(n);

        // Build cube grid
        let t0 = Instant::now();
        let grid = CubeMapGrid::new(&points, res);
        let grid_build = t0.elapsed().as_secs_f64() * 1000.0;

        // Build kiddo tree
        let t0 = Instant::now();
        let (tree, entries) = build_kdtree(&points);
        let kiddo_build = t0.elapsed().as_secs_f64() * 1000.0;

        // Query timing - sample 1000 points
        let sample_size = 1000.min(n);

        let t0 = Instant::now();
        let mut scratch = grid.make_scratch();
        let mut grid_results: Vec<Vec<usize>> = Vec::with_capacity(sample_size);
        for i in 0..sample_size {
            grid_results.push(grid.find_k_nearest_with_scratch(
                &points,
                points[i],
                i,
                k,
                &mut scratch,
            ));
        }
        let grid_query = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let kiddo_results: Vec<Vec<usize>> = (0..sample_size)
            .map(|i| kiddo_find_k_nearest(&tree, &entries, points[i], i, k))
            .collect();
        let kiddo_query = t0.elapsed().as_secs_f64() * 1000.0;

        // Compare results
        let mut exact_matches = 0;
        let mut partial_matches = 0;
        let mut total_overlap = 0usize;

        for (grid_knn, kiddo_knn) in grid_results.iter().zip(kiddo_results.iter()) {
            let grid_set: std::collections::HashSet<_> = grid_knn.iter().collect();
            let kiddo_set: std::collections::HashSet<_> = kiddo_knn.iter().collect();
            let overlap = grid_set.intersection(&kiddo_set).count();

            total_overlap += overlap;
            if overlap == k {
                exact_matches += 1;
            } else if overlap >= k - 2 {
                partial_matches += 1;
            }
        }

        let stats = grid.stats();

        println!(
            "n={:>7} res={:>3} ({:.1} pts/cell, {} empty)",
            n, res, stats.avg_points_per_cell, stats.empty_cells
        );
        println!(
            "  Build:  Grid={:>8.1}ms  Kiddo={:>8.1}ms  ({:.1}x)",
            grid_build,
            kiddo_build,
            kiddo_build / grid_build
        );
        println!(
            "  Query:  Grid={:>8.1}ms  Kiddo={:>8.1}ms  ({:.1}x) [{} samples]",
            grid_query,
            kiddo_query,
            kiddo_query / grid_query,
            sample_size
        );
        // Extrapolated total time for all n queries
        let grid_total_est = grid_build + (grid_query / sample_size as f64) * n as f64;
        let kiddo_total_est = kiddo_build + (kiddo_query / sample_size as f64) * n as f64;

        println!(
            "  Match:  exact={}/{} partial={} avg_overlap={:.1}/{}",
            exact_matches,
            sample_size,
            partial_matches,
            total_overlap as f64 / sample_size as f64,
            k
        );
        println!(
            "  Est total (build + all queries): Grid={:.0}ms  Kiddo={:.0}ms",
            grid_total_est, kiddo_total_est
        );
        println!();
    }
}

#[test]
#[ignore] // Run with: cargo test test_cube_grid_parallel -- --ignored --nocapture
fn test_cube_grid_parallel() {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    println!("\n=== Parallel k-NN: CubeMapGrid vs Kiddo ===\n");

    // Test one size at a time to reduce memory pressure
    for &n in &[100_000usize, 500_000, 1_000_000, 2_000_000] {
        let points = generate_test_points(n, 12345);
        let k = 24;
        let res = default_res(n);

        // Grid: build + query (don't store results, just count)
        let t0 = Instant::now();
        let grid = CubeMapGrid::new(&points, res);
        let grid_build = t0.elapsed().as_secs_f64() * 1000.0;

        let count = AtomicUsize::new(0);
        let t0 = Instant::now();
        (0..n).into_par_iter().for_each_init(
            || grid.make_scratch(),
            |scratch, i| {
                let knn = grid.find_k_nearest_with_scratch(&points, points[i], i, k, scratch);
                count.fetch_add(knn.len(), Ordering::Relaxed);
            },
        );
        let grid_query = t0.elapsed().as_secs_f64() * 1000.0;
        let _ = count.load(Ordering::Relaxed); // prevent optimization

        drop(grid); // Free memory before kiddo

        // Kiddo: build + query
        let t0 = Instant::now();
        let (tree, entries) = build_kdtree(&points);
        let kiddo_build = t0.elapsed().as_secs_f64() * 1000.0;

        let count = AtomicUsize::new(0);
        let t0 = Instant::now();
        (0..n).into_par_iter().for_each(|i| {
            let knn = kiddo_find_k_nearest(&tree, &entries, points[i], i, k);
            count.fetch_add(knn.len(), Ordering::Relaxed);
        });
        let kiddo_query = t0.elapsed().as_secs_f64() * 1000.0;
        let _ = count.load(Ordering::Relaxed);

        let grid_total = grid_build + grid_query;
        let kiddo_total = kiddo_build + kiddo_query;

        println!("n={:>7}:", n);
        println!(
            "  Grid:  build={:>7.1}ms  query={:>7.1}ms  total={:>7.1}ms",
            grid_build, grid_query, grid_total
        );
        println!(
            "  Kiddo: build={:>7.1}ms  query={:>7.1}ms  total={:>7.1}ms",
            kiddo_build, kiddo_query, kiddo_total
        );
        println!("  Speedup: {:.2}x", kiddo_total / grid_total);
        println!();
    }
}
