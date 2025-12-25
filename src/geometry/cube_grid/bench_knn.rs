//! Benchmarks for cube_grid packed k-NN.
//!
//! Run with: cargo test bench_knn --release -- --ignored --nocapture

use super::packed_knn::{
    packed_knn, packed_knn_cell_stream, packed_knn_stats, PackedKnnCellScratch,
    PackedKnnCellStatus, PackedKnnStats, PackedV4Edges,
};
use super::*;
use crate::geometry::fibonacci_sphere_points_with_rng;
use glam::{Vec3, Vec3A};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

fn gen_points(n: usize, seed: u64) -> Vec<Vec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
    fibonacci_sphere_points_with_rng(n, spacing * 0.1, &mut rng)
}

fn res_for_k(n: usize, k: f64) -> usize {
    ((n as f64 / (6.0 * k)).sqrt() as usize).max(4)
}

#[test]
#[ignore]
fn bench_packed_knn() {
    const N: usize = 100_000;
    const K: usize = 24;
    const SEED: u64 = 12345;
    const RUNS: usize = 5;

    println!("\n=== Packed k-NN Benchmark ===");
    println!("Points: {N}, k: {K}, runs: {RUNS}\n");

    let points = gen_points(N, SEED);
    let res = res_for_k(N, K as f64);
    let grid = CubeMapGrid::new(&points, res);

    // Warmup
    for _ in 0..3 {
        let _ = packed_knn(&grid, &points, K);
    }

    let mut total_time = std::time::Duration::ZERO;
    let mut last_result = None;
    for _ in 0..RUNS {
        let t0 = Instant::now();
        let r = packed_knn(&grid, &points, K);
        total_time += t0.elapsed();
        last_result = Some(r);
    }
    let avg_time = total_time / RUNS as u32;

    let (_result, stats) = packed_knn_stats(&grid, &points, K);

    print_timing("PackedV4", avg_time, N);
    let num_cells = 6 * res * res;
    print_stats(&stats, N, num_cells);
    drop(last_result);
}

#[test]
#[ignore]
fn bench_perpoint_vs_packed() {
    const N: usize = 100_000;
    const K: usize = 24;
    const SEED: u64 = 12345;

    println!("\n=== Per-Point vs Packed Benchmark ===");
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
            &points_a,
            points_a[i],
            i,
            K,
            &mut scratch,
            &mut out_indices,
        );
    }
    let _ = packed_knn(&grid, &points, K);

    // Per-point queries
    let t0 = Instant::now();
    for i in 0..N {
        grid.find_k_nearest_with_scratch_into_dot_topk(
            &points_a,
            points_a[i],
            i,
            K,
            &mut scratch,
            &mut out_indices,
        );
    }
    let perpoint_time = t0.elapsed();

    // Packed
    let t0 = Instant::now();
    let _ = packed_knn(&grid, &points, K);
    let packed_time = t0.elapsed();

    println!(
        "Per-point:  {:>7.2} ms ({:>5.0} ns/pt)",
        perpoint_time.as_secs_f64() * 1000.0,
        perpoint_time.as_nanos() as f64 / N as f64
    );
    println!(
        "PackedV4:   {:>7.2} ms ({:>5.0} ns/pt)",
        packed_time.as_secs_f64() * 1000.0,
        packed_time.as_nanos() as f64 / N as f64
    );
    println!(
        "PackedV4 speedup: {:.2}x",
        perpoint_time.as_secs_f64() / packed_time.as_secs_f64()
    );
}

#[test]
#[ignore]
fn bench_stream_vs_perpoint() {
    const N: usize = 100_000;
    const K: usize = 24;
    const SEED: u64 = 12345;

    println!("\n=== Packed Stream vs Per-Point Benchmark ===");
    println!("Points: {N}, k: {K}\n");

    let points = gen_points(N, SEED);
    let points_a: Vec<Vec3A> = points.iter().map(|&p| Vec3A::from(p)).collect();
    let res = res_for_k(N, K as f64);
    let grid = CubeMapGrid::new(&points, res);
    let edges = PackedV4Edges::new(res);

    let num_cells = 6 * res * res;
    let mut scratch = CubeMapGridScratch::new(num_cells);
    let mut out_indices = Vec::new();

    // Warmup
    for i in 0..100 {
        grid.find_k_nearest_with_scratch_into_dot_topk(
            &points_a,
            points_a[i],
            i,
            K,
            &mut scratch,
            &mut out_indices,
        );
    }
    let mut packed_scratch = PackedKnnCellScratch::new();
    for cell in 0..num_cells {
        let queries = grid.cell_points(cell);
        let _ = packed_knn_cell_stream(
            &grid,
            cell,
            queries,
            K,
            &edges,
            &mut packed_scratch,
            |_qi, _query_idx, _neighbors, _count, _security| {},
        );
    }

    // Per-point queries (baseline)
    let t0 = Instant::now();
    for i in 0..N {
        grid.find_k_nearest_with_scratch_into_dot_topk(
            &points_a,
            points_a[i],
            i,
            K,
            &mut scratch,
            &mut out_indices,
        );
    }
    let perpoint_time = t0.elapsed();

    // Packed stream per cell with slow-path fallback to per-point.
    let t0 = Instant::now();
    for cell in 0..num_cells {
        let queries = grid.cell_points(cell);
        let status = packed_knn_cell_stream(
            &grid,
            cell,
            queries,
            K,
            &edges,
            &mut packed_scratch,
            |_qi, _query_idx, _neighbors, _count, _security| {},
        );
        if status == PackedKnnCellStatus::SlowPath {
            for &query_idx in queries {
                let i = query_idx as usize;
                grid.find_k_nearest_with_scratch_into_dot_topk(
                    &points_a,
                    points_a[i],
                    i,
                    K,
                    &mut scratch,
                    &mut out_indices,
                );
            }
        }
    }
    let packed_time = t0.elapsed();

    println!(
        "Per-point:  {:>7.2} ms ({:>5.0} ns/pt)",
        perpoint_time.as_secs_f64() * 1000.0,
        perpoint_time.as_nanos() as f64 / N as f64
    );
    println!(
        "PackedStream: {:>7.2} ms ({:>5.0} ns/pt)",
        packed_time.as_secs_f64() * 1000.0,
        packed_time.as_nanos() as f64 / N as f64
    );
    println!(
        "PackedStream speedup: {:.2}x",
        perpoint_time.as_secs_f64() / packed_time.as_secs_f64()
    );
}

fn print_timing(name: &str, time: std::time::Duration, n: usize) {
    println!(
        "{:>12}: {:>7.2} ms ({:>5.0} ns/pt)",
        name,
        time.as_secs_f64() * 1000.0,
        time.as_nanos() as f64 / n as f64
    );
}

fn print_stats(stats: &PackedKnnStats, n: usize, num_cells: usize) {
    println!(
        "{:>12}: filtered {:>5.1}%, slow_cells {:>5.2}%, fallback {:>5.2}%, under_k {:>5.2}%",
        "PackedV4",
        stats.filter_rate() * 100.0,
        stats.slow_path_cells as f64 / num_cells as f64 * 100.0,
        stats.fallback_queries as f64 / n as f64 * 100.0,
        stats.under_k_count as f64 / n as f64 * 100.0
    );
}
