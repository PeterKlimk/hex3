//! Benchmarks for cube_grid k-NN queries.
//!
//! Run with: cargo test bench_knn --release -- --ignored --nocapture

use super::batched_knn::{binlocal_knn_filtered, FilterMode, FilterStats};
use super::*;
use crate::geometry::fibonacci_sphere_points_with_rng;
use glam::Vec3;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;
use std::time::Instant;

fn gen_points(n: usize, seed: u64) -> Vec<Vec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
    fibonacci_sphere_points_with_rng(n, spacing * 0.1, &mut rng)
}

fn res_for_k(n: usize, k: f64) -> usize {
    ((n as f64 / (6.0 * k)).sqrt() as usize).max(4)
}

/// Main benchmark: compare None vs Planes vs Combined.
#[test]
#[ignore]
fn bench_knn_modes() {
    const N: usize = 100_000;
    const K: usize = 24;
    const SEED: u64 = 12345;
    const RUNS: usize = 5;

    println!("\n=== k-NN Filtering Benchmark ===");
    println!("Points: {N}, k: {K}, runs: {RUNS}\n");

    let points = gen_points(N, SEED);
    let res = res_for_k(N, K as f64);
    let grid = CubeMapGrid::new(&points, res);

    let stats = grid.stats();
    println!(
        "Grid: res={}, cells={}, avg pts/cell={:.1}\n",
        res, stats.num_cells, stats.avg_points_per_cell
    );

    // Warmup
    for _ in 0..3 {
        let _ = binlocal_knn_filtered(&grid, &points, K, FilterMode::None);
        let _ = binlocal_knn_filtered(&grid, &points, K, FilterMode::PackedV4);
    }

    // Run all modes multiple times
    let (none_result, none_stats, none_time) = run_mode_avg(&grid, &points, K, FilterMode::None, RUNS);
    let (v4_result, v4_stats, v4_time) = run_mode_avg(&grid, &points, K, FilterMode::PackedV4, RUNS);

    // Print results
    println!("=== Timing ===");
    print_timing("None", &none_stats, none_time, N, none_time);
    print_timing("PackedV4", &v4_stats, v4_time, N, none_time);

    println!("\n=== Filtering ===");
    print_filtering("None", &none_stats, N, K);
    print_filtering("PackedV4", &v4_stats, N, K);

    println!("\n=== Phase Breakdown (ms) ===");
    println!("{:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "", "Gather", "SIMD", "Thresh", "Pairs", "Select", "Sort", "Write");
    print_phases("None", &none_stats, N);
    print_phases("PackedV4", &v4_stats, N);

    // Quality check
    let diff_v4 = count_differences(&v4_result, &none_result, &points, N);
    println!("\n=== Quality (vs None) ===");
    println!("PackedV4: {} differ ({} ties)", diff_v4.queries_differ, diff_v4.ties);

    println!("\n=== Speedup vs None ===");
    println!("PackedV4: {:.2}x", none_time.as_secs_f64() / v4_time.as_secs_f64());
}

/// Compare per-point queries vs batched PackedV2.
#[test]
#[ignore]
fn bench_perpoint_vs_batched() {
    use glam::Vec3A;

    const N: usize = 100_000;
    const K: usize = 24;
    const SEED: u64 = 12345;

    println!("\n=== Per-Point vs Batched Benchmark ===");
    println!("Points: {N}, k: {K}\n");

    let points = gen_points(N, SEED);
    let points_a: Vec<Vec3A> = points.iter().map(|&p| Vec3A::from(p)).collect();
    let res = res_for_k(N, K as f64);
    let grid = CubeMapGrid::new(&points, res);

    // Warmup
    let num_cells = 6 * res * res;
    let mut scratch = CubeMapGridScratch::new(num_cells);
    let mut out_indices = Vec::new();
    for i in 0..100 {
        grid.find_k_nearest_with_scratch_into_dot_topk(
            &points_a, points_a[i], i, K, &mut scratch, &mut out_indices
        );
    }
    let _ = binlocal_knn_filtered(&grid, &points, K, FilterMode::PackedV2);

    // Per-point queries
    let t0 = Instant::now();
    for i in 0..N {
        grid.find_k_nearest_with_scratch_into_dot_topk(
            &points_a, points_a[i], i, K, &mut scratch, &mut out_indices
        );
    }
    let perpoint_time = t0.elapsed();

    // Batched PackedV2
    let t0 = Instant::now();
    let (_, _) = binlocal_knn_filtered(&grid, &points, K, FilterMode::PackedV2);
    let v2_time = t0.elapsed();

    // Batched PackedV3
    let t0 = Instant::now();
    let (_, _) = binlocal_knn_filtered(&grid, &points, K, FilterMode::PackedV3);
    let v3_time = t0.elapsed();

    // Batched PackedV4
    let t0 = Instant::now();
    let (_, _) = binlocal_knn_filtered(&grid, &points, K, FilterMode::PackedV4);
    let v4_time = t0.elapsed();

    println!("Per-point:  {:>7.2} ms ({:>5.0} ns/pt)",
        perpoint_time.as_secs_f64() * 1000.0,
        perpoint_time.as_nanos() as f64 / N as f64);
    println!("PackedV2:   {:>7.2} ms ({:>5.0} ns/pt)",
        v2_time.as_secs_f64() * 1000.0,
        v2_time.as_nanos() as f64 / N as f64);
    println!("PackedV3:   {:>7.2} ms ({:>5.0} ns/pt)",
        v3_time.as_secs_f64() * 1000.0,
        v3_time.as_nanos() as f64 / N as f64);
    println!("PackedV4:   {:>7.2} ms ({:>5.0} ns/pt)",
        v4_time.as_secs_f64() * 1000.0,
        v4_time.as_nanos() as f64 / N as f64);
    println!("\nPackedV2 speedup: {:.2}x", perpoint_time.as_secs_f64() / v2_time.as_secs_f64());
    println!("PackedV3 speedup: {:.2}x", perpoint_time.as_secs_f64() / v3_time.as_secs_f64());
    println!("PackedV4 speedup: {:.2}x", perpoint_time.as_secs_f64() / v4_time.as_secs_f64());
}

fn run_mode(
    grid: &CubeMapGrid,
    points: &[Vec3],
    k: usize,
    mode: FilterMode,
) -> (super::batched_knn::BatchedKnnResult, FilterStats, std::time::Duration) {
    let t0 = Instant::now();
    let (result, stats) = binlocal_knn_filtered(grid, points, k, mode);
    let time = t0.elapsed();
    (result, stats, time)
}

fn run_mode_avg(
    grid: &CubeMapGrid,
    points: &[Vec3],
    k: usize,
    mode: FilterMode,
    runs: usize,
) -> (super::batched_knn::BatchedKnnResult, FilterStats, std::time::Duration) {
    let mut total_time = std::time::Duration::ZERO;
    let mut result = None;
    let mut stats = None;
    for _ in 0..runs {
        let t0 = Instant::now();
        let (r, s) = binlocal_knn_filtered(grid, points, k, mode);
        total_time += t0.elapsed();
        result = Some(r);
        stats = Some(s);
    }
    (result.unwrap(), stats.unwrap(), total_time / runs as u32)
}

fn print_timing(name: &str, _stats: &FilterStats, time: std::time::Duration, n: usize, baseline: std::time::Duration) {
    let delta = (time.as_secs_f64() / baseline.as_secs_f64() - 1.0) * 100.0;
    println!(
        "{:>10}: {:>7.2} ms ({:>5.0} ns/pt) {:>+6.1}%",
        name,
        time.as_secs_f64() * 1000.0,
        time.as_nanos() as f64 / n as f64,
        delta
    );
}

fn print_filtering(name: &str, stats: &FilterStats, n: usize, k: usize) {
    let avg_certified = if stats.num_queries > 0 {
        stats.total_certified as f64 / stats.num_queries as f64
    } else {
        0.0
    };
    println!(
        "{:>10}: filtered {:>5.1}%, fallback {:>5.2}%, under_k {:>5.2}%, avg_cert {:.1}",
        name,
        stats.filter_rate() * 100.0,
        stats.fallback_count as f64 / n as f64 * 100.0,
        stats.under_k_count as f64 / n as f64 * 100.0,
        avg_certified
    );
}

fn print_phases(name: &str, stats: &FilterStats, _n: usize) {
    println!(
        "{:>10} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
        name,
        stats.gather_ns as f64 / 1e6,
        stats.simd_ns as f64 / 1e6,
        stats.filter_time_ns as f64 / 1e6,
        stats.pairs_ns as f64 / 1e6,
        stats.select_ns as f64 / 1e6,
        stats.sort_ns as f64 / 1e6,
        stats.write_ns as f64 / 1e6,
    );
}

struct DiffStats {
    queries_differ: usize,
    neighbors_differ: usize,
    ties: usize,
}

fn count_differences(
    reference: &super::batched_knn::BatchedKnnResult,
    result: &super::batched_knn::BatchedKnnResult,
    points: &[Vec3],
    n: usize,
) -> DiffStats {
    let mut queries_differ = 0;
    let mut neighbors_differ = 0;
    let mut ties = 0;

    for qi in 0..n {
        let ref_set: HashSet<_> = reference.get(qi).iter().copied().collect();
        let res_set: HashSet<_> = result.get(qi).iter().copied().collect();

        if ref_set != res_set {
            queries_differ += 1;

            let q = points[qi];
            let missing: Vec<_> = ref_set.difference(&res_set).copied().collect();
            let extra: Vec<_> = res_set.difference(&ref_set).copied().collect();

            if missing.len() == extra.len() {
                let missing_min_dot = missing.iter()
                    .filter(|&&idx| idx != u32::MAX)
                    .map(|&idx| q.dot(points[idx as usize]))
                    .fold(f32::INFINITY, f32::min);
                let extra_min_dot = extra.iter()
                    .filter(|&&idx| idx != u32::MAX)
                    .map(|&idx| q.dot(points[idx as usize]))
                    .fold(f32::INFINITY, f32::min);

                if (missing_min_dot - extra_min_dot).abs() < 1e-5 {
                    ties += 1;
                } else {
                    neighbors_differ += missing.len();
                }
            } else {
                neighbors_differ += missing.len();
            }
        }
    }

    DiffStats { queries_differ, neighbors_differ, ties }
}

/// Benchmark different select cutoffs for packed mode.
#[test]
#[ignore]
fn bench_select_cutoffs() {
    use super::*;
    use std::time::Instant;

    const N: usize = 100_000;
    const K: usize = 24;
    const SEED: u64 = 12345;
    const RUNS: usize = 3;

    println!("\n=== Select Cutoff Benchmark ===");
    println!("Points: {N}, k: {K}, runs: {RUNS}\n");

    let points = gen_points(N, SEED);
    let res = res_for_k(N, K as f64);
    let grid = CubeMapGrid::new(&points, res);

    // Test cutoffs: 0 (always select), 2k, 3k, 4k, 5k, 6k, 8k, 1000 (never select)
    let cutoffs = [0, 2, 3, 4, 5, 6, 8, 1000];

    println!("{:>8} {:>10} {:>10} {:>10}", "Cutoff", "Time (ms)", "ns/pt", "Speedup");
    println!("{}", "-".repeat(45));

    let mut baseline_time = None;

    for &mult in &cutoffs {
        let cutoff = mult * K;
        let mut total_time = std::time::Duration::ZERO;

        for _ in 0..RUNS {
            let t0 = Instant::now();
            let _ = packed_with_cutoff(&grid, &points, K, cutoff);
            total_time += t0.elapsed();
        }

        let avg_time = total_time / RUNS as u32;
        if baseline_time.is_none() {
            baseline_time = Some(avg_time);
        }

        let speedup = baseline_time.unwrap().as_secs_f64() / avg_time.as_secs_f64();
        let label = if mult == 0 { "0 (always)".to_string() }
                    else if mult == 1000 { "∞ (never)".to_string() }
                    else { format!("{}*k={}", mult, cutoff) };

        println!(
            "{:>8} {:>10.2} {:>10.0} {:>10.2}x",
            label,
            avg_time.as_secs_f64() * 1000.0,
            avg_time.as_nanos() as f64 / N as f64,
            speedup
        );
    }
}

/// Packed mode with configurable select cutoff.
fn packed_with_cutoff(
    grid: &CubeMapGrid,
    points: &[Vec3],
    k: usize,
    cutoff: usize,  // if m <= cutoff, just sort; else select+sort
) -> Vec<u32> {
    use super::batched_knn::{make_desc_key, key_to_idx, neighborhood_corners_5x5};

    let n = points.len();
    let num_cells = 6 * grid.res * grid.res;
    let mut neighbors = vec![u32::MAX; n * k];

    // Precompute edge normals for security_3x3
    fn precompute_edges(corners: &[Vec3; 4]) -> [Vec3; 4] {
        std::array::from_fn(|i| {
            let a = corners[i];
            let b = corners[(i + 1) % 4];
            a.cross(b).normalize_or_zero()
        })
    }

    fn security_planes(q: Vec3, normals: &[Vec3; 4]) -> f32 {
        let mut max_dot = f32::NEG_INFINITY;
        for n in normals {
            let dn = q.dot(*n).abs();
            let dot_to_plane = (1.0 - dn * dn).max(0.0).sqrt();
            max_dot = max_dot.max(dot_to_plane);
        }
        max_dot
    }

    let mut candidate_indices: Vec<u32> = Vec::with_capacity(512);
    let mut cell_dots: Vec<f32> = Vec::new();

    for cell in 0..num_cells {
        let query_points = grid.cell_points(cell);
        if query_points.is_empty() {
            continue;
        }

        let corners = neighborhood_corners_5x5(cell, grid.res);
        let normals = precompute_edges(&corners);

        // Gather candidates
        candidate_indices.clear();
        let q_start = grid.cell_offsets[cell] as usize;
        let q_end = grid.cell_offsets[cell + 1] as usize;
        let center_count = q_end - q_start;
        candidate_indices.extend_from_slice(&grid.point_indices[q_start..q_end]);

        for &ncell in grid.cell_neighbors(cell) {
            if ncell != u32::MAX && ncell != cell as u32 {
                let nc = ncell as usize;
                let n_start = grid.cell_offsets[nc] as usize;
                let n_end = grid.cell_offsets[nc + 1] as usize;
                if n_start < n_end {
                    candidate_indices.extend_from_slice(&grid.point_indices[n_start..n_end]);
                }
            }
        }

        let num_candidates = candidate_indices.len();
        if num_candidates == 0 {
            continue;
        }

        // SIMD dot computation
        let num_queries = query_points.len();
        let padded_len = (num_candidates + 7) & !7;
        cell_dots.clear();
        cell_dots.resize(num_queries * padded_len, 0.0);

        // Compute dots (simplified - not full SIMD for brevity)
        for (qi, &query_idx) in query_points.iter().enumerate() {
            let q = points[query_idx as usize];
            for (ci, &cand_idx) in candidate_indices.iter().enumerate() {
                let c = points[cand_idx as usize];
                cell_dots[qi * padded_len + ci] = q.dot(c);
            }
            cell_dots[qi * padded_len + qi] = f32::NEG_INFINITY; // self
        }

        // Process each query
        for (qi, &query_idx) in query_points.iter().enumerate() {
            let q = points[query_idx as usize];
            let dot_start = qi * padded_len;
            let security_3x3 = security_planes(q, &normals);

            let mut keys: [u64; 512] = [0; 512];
            let mut m = 0usize;

            // Combined filtering with packed keys
            let mut min_center_dot = f32::INFINITY;
            for i in 0..center_count {
                if i != qi {
                    let d = cell_dots[dot_start + i];
                    if d > security_3x3 {
                        keys[m] = make_desc_key(d, candidate_indices[i]);
                        m += 1;
                        min_center_dot = min_center_dot.min(d);
                    }
                }
            }
            let center_added = m;

            let worst_1x1 = min_center_dot - 1e-6;
            let threshold = security_3x3.max(worst_1x1);
            let need = k.saturating_sub(center_added);
            let ring_start = m;

            for i in center_count..num_candidates {
                let d = cell_dots[dot_start + i];
                if d > threshold {
                    keys[m] = make_desc_key(d, candidate_indices[i]);
                    m += 1;
                }
            }

            // Fallback
            if m - ring_start < need {
                m = ring_start;
                for i in center_count..num_candidates {
                    let d = cell_dots[dot_start + i];
                    if d > security_3x3 {
                        keys[m] = make_desc_key(d, candidate_indices[i]);
                        m += 1;
                    }
                }
            }

            let k_actual = k.min(m);
            if k_actual > 0 {
                if m <= cutoff {
                    keys[..m].sort_unstable();
                } else {
                    keys[..m].select_nth_unstable(k_actual - 1);
                    keys[..k_actual].sort_unstable();
                }
            }

            let out_start = query_idx as usize * k;
            for i in 0..k_actual {
                neighbors[out_start + i] = key_to_idx(keys[i]);
            }
        }
    }

    neighbors
}
