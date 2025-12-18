//! Benchmark GPU-style Voronoi at large scales.
//!
//! Run with: cargo run --release --bin bench_voronoi
//!
//! Usage:
//!   bench_voronoi              Run all sizes (100k to 10M)
//!   bench_voronoi 100000       Run specific size
//!   bench_voronoi 100k 500k 1m Run multiple specific sizes
//!
//! Tests performance at various cell counts with fibonacci + jitter distribution.

use clap::Parser;
use glam::Vec3;
use hex3::geometry::{fibonacci_sphere_points_with_rng, gpu_voronoi::DEFAULT_K};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "bench_voronoi")]
#[command(about = "Benchmark GPU-style Voronoi at various scales")]
struct Args {
    /// Cell counts to benchmark (e.g., 100000, 100k, 1m, 10M)
    /// If none provided, runs all sizes: 100k, 500k, 1M, 2M, 5M, 10M
    #[arg(value_parser = parse_count)]
    sizes: Vec<usize>,

    /// Value of k (number of neighbors) to use
    #[arg(short, long, default_value_t = DEFAULT_K)]
    k: usize,

    /// Random seed
    #[arg(short, long, default_value_t = 12345)]
    seed: u64,

    /// Also test different k values (16, 24, 32, 48) for first size
    #[arg(long)]
    vary_k: bool,

    /// Analyze actual neighbor distribution and test k accuracy (slower, uses convex hull)
    #[arg(long)]
    analyze: bool,

    /// Deep analysis of vertex validity for disagreements
    #[arg(long)]
    deep: bool,

    /// Print detailed dedup timing breakdown
    #[arg(long)]
    dedup_timing: bool,
}

fn parse_count(s: &str) -> Result<usize, String> {
    let s = s.to_lowercase();
    let (num_str, multiplier) = if s.ends_with('m') {
        (&s[..s.len() - 1], 1_000_000)
    } else if s.ends_with('k') {
        (&s[..s.len() - 1], 1_000)
    } else {
        (s.as_str(), 1)
    };

    num_str
        .parse::<f64>()
        .map(|n| (n * multiplier as f64) as usize)
        .map_err(|e| format!("Invalid number '{}': {}", s, e))
}

fn generate_points(n: usize, seed: u64) -> Vec<Vec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mean_spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
    let jitter = mean_spacing * 0.25;
    fibonacci_sphere_points_with_rng(n, jitter, &mut rng)
}

/// Detailed timing breakdown for each phase.
#[derive(Debug, Clone, Copy)]
struct PhaseTimings {
    grid_ms: f64,
    knn_ms: f64,
    cell_construction_ms: f64,
    ccw_order_ms: f64,
    dedup_ms: f64,
    assemble_ms: f64,
    total_ms: f64,
}

/// Run the voronoi computation with detailed phase timing.
/// Uses the optimized code path with union-find degeneracy unification.
fn benchmark_voronoi_phases(points: &[Vec3], k: usize) -> (PhaseTimings, usize, usize) {
    benchmark_voronoi_phases_inner(points, k, false)
}

fn benchmark_voronoi_phases_inner(points: &[Vec3], k: usize, print_dedup_timing: bool) -> (PhaseTimings, usize, usize) {
    use hex3::geometry::gpu_voronoi::{
        CubeMapGridKnn, TerminationConfig,
        build_cells_data_incremental, dedup::dedup_vertices_hash_with_degeneracy_edges_timed,
    };

    let n = points.len();

    // Phase 1: CubeMapGridKnn construction
    let t0 = Instant::now();
    let knn = CubeMapGridKnn::new(points);
    let t1 = Instant::now();

    // Phase 2+3: Cell construction with k-NN (interleaved, parallel)
    // This now collects degenerate triplets for targeted merge
    let termination = TerminationConfig {
        enabled: true,
        check_start: 10,
        check_step: 6,
    };
    let (cells_data, degenerate_edges) =
        build_cells_data_incremental(points, &knn, k, termination);
    let t2 = Instant::now();

    // Phase 4: CCW ordering is now a no-op (IncrementalCellBuilder maintains CCW order)
    let t3 = t2;

    // Phase 5: Vertex deduplication with union-find degeneracy unification
    let (all_vertices, _cells, cell_indices) = dedup_vertices_hash_with_degeneracy_edges_timed(
        n,
        cells_data,
        &degenerate_edges,
        print_dedup_timing,
    );
    let t4 = Instant::now();

    // Phase 6: Final assembly (just counting)
    let num_unique_vertices = all_vertices.len();
    let total_vertex_refs = cell_indices.len();
    let t5 = Instant::now();

    let timings = PhaseTimings {
        grid_ms: (t1 - t0).as_secs_f64() * 1000.0,
        knn_ms: 0.0, // Now interleaved with cell construction
        cell_construction_ms: (t2 - t1).as_secs_f64() * 1000.0,
        ccw_order_ms: 0.0, // No longer needed
        dedup_ms: (t4 - t3).as_secs_f64() * 1000.0,
        assemble_ms: (t5 - t4).as_secs_f64() * 1000.0,
        total_ms: (t5 - t0).as_secs_f64() * 1000.0,
    };

    (timings, num_unique_vertices, total_vertex_refs)
}

fn format_rate(count: usize, ms: f64) -> String {
    let per_sec = count as f64 / (ms / 1000.0);
    if per_sec >= 1_000_000.0 {
        format!("{:.2}M/s", per_sec / 1_000_000.0)
    } else if per_sec >= 1_000.0 {
        format!("{:.1}k/s", per_sec / 1000.0)
    } else {
        format!("{:.0}/s", per_sec)
    }
}

fn print_results(n: usize, k: usize, timings: PhaseTimings, unique_verts: usize, total_refs: usize) {
    let total = timings.total_ms.max(0.001);

    println!("\n{}", "=".repeat(70));
    println!(
        " GPU-style Voronoi: n = {}, k = {}",
        format_num(n),
        k
    );
    println!("{}", "=".repeat(70));
    println!();

    println!("Phase Breakdown:");
    println!(
        "  Grid build:          {:>9.1} ms  ({:>5.1}%)  {}",
        timings.grid_ms,
        timings.grid_ms / total * 100.0,
        format_rate(n, timings.grid_ms)
    );
    println!(
        "  K-NN queries:        {:>9.1} ms  ({:>5.1}%)  {}",
        timings.knn_ms,
        timings.knn_ms / total * 100.0,
        format_rate(n, timings.knn_ms)
    );
    println!(
        "  Cell construction:   {:>9.1} ms  ({:>5.1}%)  {}",
        timings.cell_construction_ms,
        timings.cell_construction_ms / total * 100.0,
        format_rate(n, timings.cell_construction_ms)
    );
    println!(
        "  CCW ordering:        {:>9.1} ms  ({:>5.1}%)  {}",
        timings.ccw_order_ms,
        timings.ccw_order_ms / total * 100.0,
        format_rate(n, timings.ccw_order_ms)
    );
    println!(
        "  Vertex dedup:        {:>9.1} ms  ({:>5.1}%)  {} refs",
        timings.dedup_ms,
        timings.dedup_ms / total * 100.0,
        format_rate(total_refs, timings.dedup_ms)
    );
    println!(
        "  Final assembly:      {:>9.1} ms  ({:>5.1}%)",
        timings.assemble_ms,
        timings.assemble_ms / total * 100.0
    );
    println!("  ─────────────────────────────────────────────");
    println!("  TOTAL:               {:>9.1} ms", timings.total_ms);
    println!();

    println!("Statistics:");
    println!("  Unique vertices:     {}", format_num(unique_verts));
    println!("  Vertex references:   {}", format_num(total_refs));
    println!(
        "  Avg verts/cell:      {:.2}",
        total_refs as f64 / n as f64
    );
    println!(
        "  Throughput:          {} cells",
        format_rate(n, timings.total_ms)
    );

    // Memory estimate
    let point_mem = n * std::mem::size_of::<Vec3>();
    let vertex_mem = unique_verts * std::mem::size_of::<Vec3>();
    let index_mem = total_refs * std::mem::size_of::<usize>();
    let knn_mem = n * k * std::mem::size_of::<usize>();
    let total_mem = point_mem + vertex_mem + index_mem + knn_mem;
    println!();
    println!("Memory (estimated):");
    println!("  Points:              {} MB", point_mem / (1024 * 1024));
    println!("  Vertices:            {} MB", vertex_mem / (1024 * 1024));
    println!("  Indices:             {} MB", index_mem / (1024 * 1024));
    println!("  K-NN cache:          {} MB", knn_mem / (1024 * 1024));
    println!("  Total:               {} MB", total_mem / (1024 * 1024));
}

fn format_num(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{}k", n / 1_000)
    } else {
        format!("{}", n)
    }
}

/// Summary row for final comparison table
struct SummaryRow {
    n: usize,
    timings: PhaseTimings,
}

/// Analyze the actual neighbor count distribution using ground-truth Voronoi
fn analyze_neighbor_distribution(points: &[Vec3]) -> (usize, usize, f64, Vec<usize>) {
    use hex3::geometry::SphericalVoronoi;

    // Compute ground truth Voronoi via convex hull
    let voronoi = SphericalVoronoi::compute(points);

    // Count vertices per cell (vertices = neighbors for Voronoi)
    let mut counts: Vec<usize> = Vec::with_capacity(points.len());
    for i in 0..voronoi.num_cells() {
        counts.push(voronoi.cell(i).len());
    }

    let min = *counts.iter().min().unwrap_or(&0);
    let max = *counts.iter().max().unwrap_or(&0);
    let avg = counts.iter().sum::<usize>() as f64 / counts.len().max(1) as f64;

    // Build histogram
    let mut histogram = vec![0usize; max + 1];
    for &c in &counts {
        histogram[c] += 1;
    }

    (min, max, avg, histogram)
}

/// Test accuracy at different k values against ground truth
fn test_accuracy_at_k(points: &[Vec3], k: usize) -> (usize, usize, f64) {
    use hex3::geometry::{gpu_voronoi::compute_voronoi_gpu_style, SphericalVoronoi};

    let hull = SphericalVoronoi::compute(points);
    let gpu = compute_voronoi_gpu_style(points, k);

    let mut exact_match = 0usize;
    let mut vertex_diff_sum = 0i64;
    let mut bad_cells = 0usize; // cells with <3 vertices

    for i in 0..points.len() {
        let hull_count = hull.cell(i).len();
        let gpu_count = gpu.cell(i).len();

        if hull_count == gpu_count {
            exact_match += 1;
        }
        vertex_diff_sum += (gpu_count as i64 - hull_count as i64).abs();

        if gpu_count < 3 {
            bad_cells += 1;
        }
    }

    (exact_match, bad_cells, vertex_diff_sum as f64 / points.len() as f64)
}

/// Detailed mismatch analysis
fn analyze_mismatches(points: &[Vec3], k: usize) {
    use hex3::geometry::{gpu_voronoi::compute_voronoi_gpu_style, SphericalVoronoi};

    let hull = SphericalVoronoi::compute(points);
    let gpu = compute_voronoi_gpu_style(points, k);

    println!("\nMismatch analysis (k={}):", k);

    let mut mismatch_by_diff: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    let mut mismatch_by_hull_count: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

    for i in 0..points.len() {
        let hull_count = hull.cell(i).len();
        let gpu_count = gpu.cell(i).len();
        let diff = gpu_count as i32 - hull_count as i32;

        if diff != 0 {
            *mismatch_by_diff.entry(diff).or_insert(0) += 1;
            *mismatch_by_hull_count.entry(hull_count).or_insert(0) += 1;
        }
    }

    println!("  By vertex difference (gpu - hull):");
    let mut diffs: Vec<_> = mismatch_by_diff.into_iter().collect();
    diffs.sort_by_key(|&(d, _)| d);
    for (diff, count) in diffs {
        println!("    {:+2}: {} cells", diff, count);
    }

    println!("  By hull vertex count:");
    let mut counts: Vec<_> = mismatch_by_hull_count.into_iter().collect();
    counts.sort_by_key(|&(c, _)| c);
    for (hull_count, mismatch_count) in counts {
        println!("    {} verts: {} mismatches", hull_count, mismatch_count);
    }
}

/// Deep analysis of disagreement between methods
fn analyze_vertex_validity(points: &[Vec3], k: usize, num_samples: usize) {
    use hex3::geometry::{gpu_voronoi::compute_voronoi_gpu_style, SphericalVoronoi};

    let hull = SphericalVoronoi::compute(points);
    let gpu = compute_voronoi_gpu_style(points, k);

    println!("\n========== Vertex Validity Analysis (k={}) ==========", k);

    // Find cells where methods disagree
    let mut disagreements: Vec<(usize, i32)> = Vec::new();
    for i in 0..points.len() {
        let hull_count = hull.cell(i).len();
        let gpu_count = gpu.cell(i).len();
        let diff = gpu_count as i32 - hull_count as i32;
        if diff != 0 {
            disagreements.push((i, diff));
        }
    }

    println!("Total disagreements: {}", disagreements.len());

    // Sample some disagreements for detailed analysis
    let samples: Vec<_> = disagreements.iter().take(num_samples).collect();

    for &(cell_idx, diff) in &samples {
        let generator = points[*cell_idx];
        let hull_cell = hull.cell(*cell_idx);
        let gpu_cell = gpu.cell(*cell_idx);

        let hull_verts: Vec<Vec3> = hull_cell.vertex_indices.iter()
            .map(|&vi| hull.vertices[vi])
            .collect();
        let gpu_verts: Vec<Vec3> = gpu_cell.vertex_indices.iter()
            .map(|&vi| gpu.vertices[vi])
            .collect();

        println!("\n--- Cell {} (diff={:+}) ---", cell_idx, diff);
        println!("  Generator: ({:.6}, {:.6}, {:.6})", generator.x, generator.y, generator.z);
        println!("  Hull: {} verts, GPU: {} verts", hull_verts.len(), gpu_verts.len());

        // Print all vertices from both methods
        println!("  Hull vertices:");
        for (i, v) in hull_verts.iter().enumerate() {
            println!("    {}: ({:.6}, {:.6}, {:.6})", i, v.x, v.y, v.z);
        }
        println!("  GPU vertices:");
        for (i, v) in gpu_verts.iter().enumerate() {
            println!("    {}: ({:.6}, {:.6}, {:.6})", i, v.x, v.y, v.z);
        }

        // Check for near-duplicates within each method
        let check_dups = |verts: &[Vec3], name: &str| {
            for i in 0..verts.len() {
                for j in (i+1)..verts.len() {
                    let dist = (verts[i] - verts[j]).length();
                    if dist < 0.01 {
                        println!("  {} near-duplicate: v{} and v{} dist={:.6}", name, i, j, dist);
                    }
                }
            }
        };
        check_dups(&hull_verts, "Hull");
        check_dups(&gpu_verts, "GPU");

        // Find vertices that are in one but not the other
        let tolerance = 0.001; // Position matching tolerance

        // Vertices in GPU but not in Hull (extra vertices)
        let mut extra_in_gpu: Vec<(Vec3, f32)> = Vec::new();
        for gv in &gpu_verts {
            let in_hull = hull_verts.iter().any(|hv| (*gv - *hv).length() < tolerance);
            if !in_hull {
                // Check how valid this vertex is as a Voronoi vertex
                // A valid vertex should be equidistant from the generator and its 2 nearest other generators
                let dist_to_gen = geodesic_distance(*gv, generator);
                extra_in_gpu.push((*gv, dist_to_gen));
            }
        }

        // Vertices in Hull but not in GPU (missing vertices)
        let mut missing_from_gpu: Vec<(Vec3, f32)> = Vec::new();
        for hv in &hull_verts {
            let in_gpu = gpu_verts.iter().any(|gv| (*hv - *gv).length() < tolerance);
            if !in_gpu {
                let dist_to_gen = geodesic_distance(*hv, generator);
                missing_from_gpu.push((*hv, dist_to_gen));
            }
        }

        if !extra_in_gpu.is_empty() {
            println!("  Extra in GPU ({}):", extra_in_gpu.len());
            for (v, dist) in &extra_in_gpu {
                // Find the 3 nearest generators to this vertex
                let mut gen_dists: Vec<(usize, f32)> = points.iter()
                    .enumerate()
                    .map(|(i, p)| (i, geodesic_distance(*v, *p)))
                    .collect();
                gen_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                let d0 = gen_dists[0].1;
                let d1 = gen_dists[1].1;
                let d2 = gen_dists[2].1;
                let d3 = gen_dists[3].1;

                // For a valid Voronoi vertex, d0 ≈ d1 ≈ d2 (equidistant from 3 generators)
                let spread_3 = (d2 - d0).abs();
                let gap_to_4th = d3 - d2;

                println!("    v=({:.4},{:.4},{:.4}) dist={:.6}", v.x, v.y, v.z, dist);
                println!("      3 nearest gens: {}, {}, {} (this cell={})",
                    gen_dists[0].0, gen_dists[1].0, gen_dists[2].0, cell_idx);
                println!("      distances: {:.6}, {:.6}, {:.6}, {:.6}",
                    d0, d1, d2, d3);
                println!("      spread(d0-d2): {:.2e}, gap to 4th: {:.6}",
                    spread_3, gap_to_4th);

                if spread_3 < 1e-5 {
                    println!("      -> VALID vertex (spread < 1e-5)");
                } else if spread_3 < 1e-4 {
                    println!("      -> MARGINAL vertex (spread < 1e-4)");
                } else {
                    println!("      -> INVALID vertex (spread >= 1e-4)");
                }
            }
        }

        if !missing_from_gpu.is_empty() {
            println!("  Missing from GPU ({}):", missing_from_gpu.len());
            for (v, dist) in &missing_from_gpu {
                let mut gen_dists: Vec<(usize, f32)> = points.iter()
                    .enumerate()
                    .map(|(i, p)| (i, geodesic_distance(*v, *p)))
                    .collect();
                gen_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                let d0 = gen_dists[0].1;
                let d1 = gen_dists[1].1;
                let d2 = gen_dists[2].1;
                let d3 = gen_dists[3].1;

                let spread_3 = (d2 - d0).abs();
                let gap_to_4th = d3 - d2;

                println!("    v=({:.4},{:.4},{:.4}) dist={:.6}", v.x, v.y, v.z, dist);
                println!("      3 nearest gens: {}, {}, {} (this cell={})",
                    gen_dists[0].0, gen_dists[1].0, gen_dists[2].0, cell_idx);
                println!("      distances: {:.6}, {:.6}, {:.6}, {:.6}",
                    d0, d1, d2, d3);
                println!("      spread(d0-d2): {:.2e}, gap to 4th: {:.6}",
                    spread_3, gap_to_4th);

                if spread_3 < 1e-5 {
                    println!("      -> VALID vertex (spread < 1e-5)");
                } else if spread_3 < 1e-4 {
                    println!("      -> MARGINAL vertex (spread < 1e-4)");
                } else {
                    println!("      -> INVALID vertex (spread >= 1e-4)");
                }
            }
        }
    }

    // Check for degenerate vertices (4+ equidistant generators)
    println!("\n--- Degeneracy analysis ---");
    let mut total_degenerate_verts = 0usize;
    let mut cells_with_degeneracy = 0usize;

    for (cell_idx, _diff) in &disagreements {
        let gpu_cell = gpu.cell(*cell_idx);
        let gpu_verts: Vec<Vec3> = gpu_cell.vertex_indices.iter()
            .map(|&vi| gpu.vertices[vi])
            .collect();

        // Count unique vertices (within tolerance)
        let mut unique_verts: Vec<Vec3> = Vec::new();
        for v in &gpu_verts {
            let is_dup = unique_verts.iter().any(|u| (*v - *u).length() < 0.0001);
            if !is_dup {
                unique_verts.push(*v);
            }
        }

        let dup_count = gpu_verts.len() - unique_verts.len();
        if dup_count > 0 {
            cells_with_degeneracy += 1;
            total_degenerate_verts += dup_count;
        }
    }

    println!("Cells with degenerate vertices: {} / {}", cells_with_degeneracy, disagreements.len());
    println!("Total duplicate vertices in GPU: {}", total_degenerate_verts);

    // Now check if hull also has degeneracies
    let mut hull_degen_verts = 0usize;
    for (cell_idx, _diff) in &disagreements {
        let hull_cell = hull.cell(*cell_idx);
        let hull_verts: Vec<Vec3> = hull_cell.vertex_indices.iter()
            .map(|&vi| hull.vertices[vi])
            .collect();

        let mut unique_verts: Vec<Vec3> = Vec::new();
        for v in &hull_verts {
            let is_dup = unique_verts.iter().any(|u| (*v - *u).length() < 0.0001);
            if !is_dup {
                unique_verts.push(*v);
            }
        }
        hull_degen_verts += hull_verts.len() - unique_verts.len();
    }
    println!("Total duplicate vertices in Hull: {}", hull_degen_verts);

    // Compare unique vertex counts
    println!("\nUnique vertex comparison:");
    let mut unique_match = 0usize;
    let mut unique_mismatch = 0usize;
    for (cell_idx, _diff) in &disagreements {
        let hull_cell = hull.cell(*cell_idx);
        let gpu_cell = gpu.cell(*cell_idx);

        let count_unique = |verts: &[Vec3]| -> usize {
            let mut unique: Vec<Vec3> = Vec::new();
            for v in verts {
                if !unique.iter().any(|u| (*v - *u).length() < 0.0001) {
                    unique.push(*v);
                }
            }
            unique.len()
        };

        let hull_verts: Vec<Vec3> = hull_cell.vertex_indices.iter()
            .map(|&vi| hull.vertices[vi]).collect();
        let gpu_verts: Vec<Vec3> = gpu_cell.vertex_indices.iter()
            .map(|&vi| gpu.vertices[vi]).collect();

        let hull_unique = count_unique(&hull_verts);
        let gpu_unique = count_unique(&gpu_verts);

        if hull_unique == gpu_unique {
            unique_match += 1;
        } else {
            unique_mismatch += 1;
            println!("  Cell {}: hull_unique={}, gpu_unique={}", cell_idx, hull_unique, gpu_unique);
        }
    }
    println!("Cells where unique vertex counts match: {} / {}", unique_match, disagreements.len());
    println!("Cells with true geometric difference: {}", unique_mismatch);

    // Aggregate statistics
    println!("\n--- Aggregate vertex validity ---");
    let mut total_extra = 0usize;
    let mut total_missing = 0usize;
    let mut extra_valid = 0usize;
    let mut extra_marginal = 0usize;
    let mut extra_invalid = 0usize;
    let mut missing_valid = 0usize;
    let mut missing_marginal = 0usize;
    let mut missing_invalid = 0usize;

    for (cell_idx, _diff) in &disagreements {
        let hull_cell = hull.cell(*cell_idx);
        let gpu_cell = gpu.cell(*cell_idx);

        let hull_verts: Vec<Vec3> = hull_cell.vertex_indices.iter()
            .map(|&vi| hull.vertices[vi])
            .collect();
        let gpu_verts: Vec<Vec3> = gpu_cell.vertex_indices.iter()
            .map(|&vi| gpu.vertices[vi])
            .collect();

        let tolerance = 0.001;

        // Extra in GPU
        for gv in &gpu_verts {
            let in_hull = hull_verts.iter().any(|hv| (*gv - *hv).length() < tolerance);
            if !in_hull {
                total_extra += 1;
                let spread = compute_vertex_spread(*gv, points);
                if spread < 1e-5 {
                    extra_valid += 1;
                } else if spread < 1e-4 {
                    extra_marginal += 1;
                } else {
                    extra_invalid += 1;
                }
            }
        }

        // Missing from GPU
        for hv in &hull_verts {
            let in_gpu = gpu_verts.iter().any(|gv| (*hv - *gv).length() < tolerance);
            if !in_gpu {
                total_missing += 1;
                let spread = compute_vertex_spread(*hv, points);
                if spread < 1e-5 {
                    missing_valid += 1;
                } else if spread < 1e-4 {
                    missing_marginal += 1;
                } else {
                    missing_invalid += 1;
                }
            }
        }
    }

    println!("Extra vertices in GPU (not in Hull): {}", total_extra);
    println!("  Valid (spread < 1e-5):    {}", extra_valid);
    println!("  Marginal (< 1e-4):        {}", extra_marginal);
    println!("  Invalid (>= 1e-4):        {}", extra_invalid);

    println!("Missing vertices from GPU (in Hull): {}", total_missing);
    println!("  Valid (spread < 1e-5):    {}", missing_valid);
    println!("  Marginal (< 1e-4):        {}", missing_marginal);
    println!("  Invalid (>= 1e-4):        {}", missing_invalid);
}

/// Compute the spread of distances from a vertex to its 3 nearest generators
fn compute_vertex_spread(vertex: Vec3, points: &[Vec3]) -> f32 {
    let mut gen_dists: Vec<f32> = points.iter()
        .map(|p| geodesic_distance(vertex, *p))
        .collect();
    gen_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (gen_dists[2] - gen_dists[0]).abs()
}

/// Geodesic distance on unit sphere
fn geodesic_distance(a: Vec3, b: Vec3) -> f32 {
    a.dot(b).clamp(-1.0, 1.0).acos()
}

fn main() {
    let args = Args::parse();

    println!("GPU-style Voronoi Benchmark");
    println!("===========================\n");

    let seed = args.seed;
    let k = args.k;

    // Determine sizes to run
    let sizes: Vec<usize> = if args.sizes.is_empty() {
        vec![
            100_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000,
        ]
    } else {
        args.sizes
    };

    println!("Configuration:");
    println!("  k = {} neighbors", k);
    println!("  seed = {}", seed);
    println!(
        "  sizes = {:?}",
        sizes.iter().map(|&n| format_num(n)).collect::<Vec<_>>()
    );

    // If analyze is set, analyze neighbor distribution and test k accuracy
    if args.analyze && !sizes.is_empty() {
        let n = sizes[0].min(100_000); // Cap at 100k for convex hull (it's slow)
        println!("\n\n========== Neighbor Distribution Analysis (n = {}) ==========", format_num(n));

        let points = generate_points(n, seed);

        println!("\nComputing ground-truth Voronoi via convex hull...");
        let t0 = Instant::now();
        let (min, max, avg, histogram) = analyze_neighbor_distribution(&points);
        let hull_time = t0.elapsed().as_secs_f64() * 1000.0;
        println!("  Convex hull time: {:.1} ms", hull_time);

        println!("\nNeighbor count distribution:");
        println!("  Min: {}", min);
        println!("  Max: {}", max);
        println!("  Avg: {:.2}", avg);
        println!();
        println!("  Histogram (vertices per cell):");
        for (count, &freq) in histogram.iter().enumerate() {
            if freq > 0 {
                let pct = freq as f64 / n as f64 * 100.0;
                let bar_len = (pct * 0.5) as usize;
                println!("    {:>2} vertices: {:>6} cells ({:>5.2}%) {}",
                    count, freq, pct, "#".repeat(bar_len));
            }
        }

        println!("\n\nAccuracy at different k values:");
        println!("{:>4} | {:>12} | {:>10} | {:>12} | {:>10}",
            "k", "Exact Match", "Match %", "Bad Cells", "Avg Diff");
        println!("{:-<4}-+-{:-<12}-+-{:-<10}-+-{:-<12}-+-{:-<10}", "", "", "", "", "");

        // Test k values from conservative to aggressive
        for test_k in [8, 10, 12, 14, 16, 18, 20, 24] {
            let (exact, bad, avg_diff) = test_accuracy_at_k(&points, test_k);
            let pct = exact as f64 / n as f64 * 100.0;
            println!("{:>4} | {:>12} | {:>9.2}% | {:>12} | {:>10.4}",
                test_k, exact, pct, bad, avg_diff);
        }

        // Detailed mismatch analysis at k=16 and k=24
        analyze_mismatches(&points, 16);
        analyze_mismatches(&points, 24);

        // Deep vertex validity analysis if requested
        if args.deep {
            for test_k in [16, 24, 32, 48] {
                analyze_vertex_validity(&points, test_k, 0); // 0 samples = just aggregate
            }
        }

        // Recommendation
        println!();
        println!("Recommendation:");
        println!("  For max neighbors = {}, minimum safe k = {} (with small margin)", max, max + 2);
        println!("  Using k = {} would be conservative", max + 4);

        drop(points);
    }

    // If vary_k is set, first test different k values on the first size
    if args.vary_k && !sizes.is_empty() {
        let n = sizes[0];
        println!("\n\n========== K-value scaling (n = {}) ==========", format_num(n));

        let points = generate_points(n, seed);

        println!("\n{:>4} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
            "k", "Total(ms)", "KNN(ms)", "Cells(ms)", "CCW(ms)", "Dedup(ms)");
        println!("{:-<4}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}", "", "", "", "", "", "");

        for test_k in [12, 16, 20, 24, 28, 32, 40, 48] {
            let (timings, _, _) = benchmark_voronoi_phases(&points, test_k);
            println!("{:>4} | {:>10.1} | {:>10.1} | {:>10.1} | {:>10.1} | {:>10.1}",
                test_k,
                timings.total_ms,
                timings.knn_ms,
                timings.cell_construction_ms,
                timings.ccw_order_ms,
                timings.dedup_ms);
        }

        println!("\nCell construction complexity is O(k³) per cell (k² pairs × k half-space checks)");
        drop(points);
    }

    let mut summary: Vec<SummaryRow> = Vec::new();

    for n in sizes {
        println!("\nGenerating {} fibonacci+jitter points...", format_num(n));
        let t_gen = Instant::now();
        let points = generate_points(n, seed);
        let gen_time = t_gen.elapsed().as_secs_f64() * 1000.0;
        println!("  Point generation: {:.1} ms", gen_time);

        println!("Running benchmark...");
        let (timings, unique_verts, total_refs) = benchmark_voronoi_phases_inner(&points, k, args.dedup_timing);
        print_results(n, k, timings, unique_verts, total_refs);

        summary.push(SummaryRow { n, timings });

        // Memory cleanup hint
        drop(points);
    }

    // Print summary table
    if summary.len() > 1 {
        println!("\n\n{}", "=".repeat(90));
        println!(" SUMMARY (k={})", k);
        println!("{}", "=".repeat(90));
        println!();
        println!("{:>8} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>12}",
            "n", "Total", "Grid", "KNN", "Cells", "Dedup", "Throughput");
        println!("{:-<8}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<12}",
            "", "", "", "", "", "", "");

        for row in &summary {
            let t = &row.timings;
            println!("{:>8} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>12}",
                format_num(row.n),
                format_ms(t.total_ms),
                format_ms_pct(t.grid_ms, t.total_ms),
                format_ms_pct(t.knn_ms, t.total_ms),
                format_ms_pct(t.cell_construction_ms, t.total_ms),
                format_ms_pct(t.dedup_ms, t.total_ms),
                format_rate(row.n, t.total_ms));
        }

        println!();
        println!("Bottleneck Analysis:");
        for row in &summary {
            let t = &row.timings;
            let total = t.total_ms.max(0.001);
            let phases = [
                ("Grid", t.grid_ms),
                ("KNN", t.knn_ms),
                ("Cells", t.cell_construction_ms),
                ("CCW", t.ccw_order_ms),
                ("Dedup", t.dedup_ms),
            ];
            let (bottleneck, _) = phases.iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            let cells_pct = t.cell_construction_ms / total * 100.0;
            println!("  {:>8}: {} bottleneck ({:.0}% cells)",
                format_num(row.n), bottleneck, cells_pct);
        }
    }

    println!("\n\nBenchmark complete.");
}

fn format_ms(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else {
        format!("{:.0}ms", ms)
    }
}

fn format_ms_pct(ms: f64, total: f64) -> String {
    let pct = ms / total.max(0.001) * 100.0;
    if ms >= 1000.0 {
        format!("{:.1}s({:.0}%)", ms / 1000.0, pct)
    } else {
        format!("{:.0}ms({:.0}%)", ms, pct)
    }
}
