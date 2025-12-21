//! Micro-benchmark for cell construction phase breakdown.
//!
//! Measures: KNN lookup, clip() total, get_vertices_into, termination checks
//!
//! Usage:
//!   cargo run --release --bin bench_cell_phases [distribution]
//!
//! Distributions:
//!   behaved  - Lloyd-relaxed points (like production)
//!   jittery  - Fibonacci with jitter, no Lloyd (default)
//!   evil     - Highly degenerate: clustered + high jitter
//!   all      - Run all distributions

mod bench_common;

use clap::{Parser, ValueEnum};
use glam::Vec3;
use hex3::geometry::{lloyd_relax_kmeans, gpu_voronoi::*};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

/// Point distribution tiers for benchmarking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointDistribution {
    /// Lloyd-relaxed points - uniform, well-behaved (like production)
    Behaved,
    /// Fibonacci with 0.25 spacing jitter - mild irregularity
    Jittery,
    /// Highly degenerate: clustered points + high jitter (violates min spacing)
    Evil,
}

impl PointDistribution {
    fn name(&self) -> &'static str {
        match self {
            Self::Behaved => "behaved",
            Self::Jittery => "jittery",
            Self::Evil => "evil",
        }
    }

    fn description(&self) -> &'static str {
        match self {
            Self::Behaved => "Lloyd-relaxed (production-like)",
            Self::Jittery => "Fibonacci + 0.25 jitter",
            Self::Evil => "Clustered + high jitter (invalid spacing)",
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DistributionArg {
    Behaved,
    Jittery,
    Evil,
    All,
}

impl DistributionArg {
    fn to_distribution(self) -> Option<PointDistribution> {
        match self {
            Self::Behaved => Some(PointDistribution::Behaved),
            Self::Jittery => Some(PointDistribution::Jittery),
            Self::Evil => Some(PointDistribution::Evil),
            Self::All => None,
        }
    }
}

#[derive(Parser)]
#[command(name = "bench_cell_phases")]
#[command(about = "Micro-benchmark for cell construction phase breakdown")]
struct Args {
    /// Point distribution tier
    #[arg(value_enum)]
    distribution: Option<DistributionArg>,

    /// Sizes to benchmark (e.g., 100k, 1m, 10m)
    #[arg(value_parser = bench_common::parse_count)]
    sizes: Vec<usize>,

    /// Run all distributions
    #[arg(long)]
    all: bool,

    /// Only run 100k points
    #[arg(long)]
    quick: bool,

    /// Show detailed histograms
    #[arg(long)]
    verbose: bool,

    /// Run tier comparison benchmarks
    #[arg(long)]
    tiers: bool,

    /// Run adaptive-k tuning benchmark for behaved points
    #[arg(long)]
    tuning: bool,

    /// Run raw KNN query benchmark (for measuring scratch changes)
    #[arg(long)]
    knn: bool,
}

/// Generate points with the specified distribution
fn generate_points(n: usize, distribution: PointDistribution, seed: u64) -> Vec<Vec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    match distribution {
        PointDistribution::Behaved => {
            // Production-like: fibonacci + low jitter + Lloyd relaxation
            let mut points = bench_common::fibonacci_points_with_jitter(n, 0.1, &mut rng);
            lloyd_relax_kmeans(&mut points, 2, 20, &mut rng);
            points
        }
        PointDistribution::Jittery => {
            // Moderate irregularity: fibonacci + 0.25 spacing jitter
            bench_common::fibonacci_points_with_jitter(n, 0.25, &mut rng)
        }
        PointDistribution::Evil => {
            // Highly degenerate: clusters + high jitter + very close points (invalid spacing)
            let mut points = bench_common::fibonacci_points_with_jitter(n, 0.5, &mut rng);

            // Create ~5% clustered regions by pulling points toward random attractors
            let num_attractors = (n / 100).max(5);
            let attractor_strength = 0.3;

            for _ in 0..num_attractors {
                let attractor_idx = rng.gen_range(0..n);
                let attractor = points[attractor_idx];

                // Pull ~2% of points toward each attractor
                let cluster_size = n / 50;
                for _ in 0..cluster_size {
                    let victim_idx = rng.gen_range(0..n);
                    if victim_idx != attractor_idx {
                        let victim = points[victim_idx];
                        // Interpolate toward attractor
                        let pulled = Vec3::new(
                            victim.x + (attractor.x - victim.x) * attractor_strength,
                            victim.y + (attractor.y - victim.y) * attractor_strength,
                            victim.z + (attractor.z - victim.z) * attractor_strength,
                        ).normalize();
                        points[victim_idx] = pulled;
                    }
                }
            }

            // Add some nearly-coincident points (~0.1%)
            let num_near_coincident = (n / 1000).max(10);
            for _ in 0..num_near_coincident {
                let src = rng.gen_range(0..n);
                let dst = rng.gen_range(0..n);
                if src != dst {
                    // Make dst almost exactly at src position
                    let tiny_offset = Vec3::new(
                        rng.gen_range(-1e-5..1e-5),
                        rng.gen_range(-1e-5..1e-5),
                        rng.gen_range(-1e-5..1e-5),
                    );
                    points[dst] = (points[src] + tiny_offset).normalize();
                }
            }

            points
        }
    }
}

/// Instrumented version of cell building to measure phase times.
fn build_cells_instrumented(
    points: &[Vec3],
    knn: &impl KnnProvider,
    k: usize,
    termination: TerminationConfig,
) -> CellPhaseTimings {
    let n = points.len();

    let mut total_knn_ns = 0u64;
    let mut total_clip_ns = 0u64;
    let mut total_terminate_ns = 0u64;
    let mut total_output_ns = 0u64;

    let mut total_clips = 0usize;
    let mut total_terminate_checks = 0usize;
    let mut total_vertices_written = 0usize;

    let mut scratch = knn.make_scratch();
    let mut neighbors = Vec::with_capacity(k);
    let mut builder = IncrementalCellBuilder::new(0, Vec3::ZERO);
    let mut vertices_out: Vec<VertexData> = Vec::with_capacity(n * 6);
    let mut support_data: Vec<u32> = Vec::new();

    for i in 0..n {
        // Phase 1: KNN lookup
        let t0 = Instant::now();
        neighbors.clear();
        knn.knn_into(points[i], i, k, &mut scratch, &mut neighbors);
        total_knn_ns += t0.elapsed().as_nanos() as u64;

        // Phase 2: Clipping
        builder.reset(i, points[i]);

        let mut terminated = false;
        let mut cell_terminate_ns = 0u64;
        for (count, &neighbor_idx) in neighbors.iter().enumerate() {
            let neighbor = points[neighbor_idx];

            let t_clip = Instant::now();
            builder.clip(neighbor_idx, neighbor);
            total_clip_ns += t_clip.elapsed().as_nanos() as u64;
            total_clips += 1;

            // Phase 3: Termination check
            if termination.should_check(count + 1) && builder.vertex_count() >= 3 {
                let t_term = Instant::now();
                let neighbor_cos = points[i].dot(neighbor).clamp(-1.0, 1.0);
                let can_term = builder.can_terminate(neighbor_cos);
                cell_terminate_ns += t_term.elapsed().as_nanos() as u64;
                total_terminate_checks += 1;
                if can_term {
                    terminated = true;
                    break;
                }
            }
        }
        total_terminate_ns += cell_terminate_ns;
        let _ = terminated;

        // Phase 4: Output vertices
        let t3 = Instant::now();
        let before = vertices_out.len();
        let mut cert_checked = false;
        let mut cert_failed = false;
        let mut cert_checked_vertices = 0usize;
        let mut cert_failed_vertices = 0usize;
        let mut cert_failed_ill_vertices = 0usize;
        let mut cert_failed_gap_vertices = 0usize;
        let mut cert_failed_vertex_indices: Vec<(u32, u8)> = Vec::new();
        let mut gap_sampler = hex3::geometry::gpu_voronoi::GapSampler::new(0, 1);
        builder.get_vertices_into(
            points,
            0.0,
            false,
            &mut support_data,
            &mut vertices_out,
            &mut cert_checked,
            &mut cert_failed,
            &mut cert_checked_vertices,
            &mut cert_failed_vertices,
            &mut cert_failed_ill_vertices,
            &mut cert_failed_gap_vertices,
            &mut gap_sampler,
            &mut cert_failed_vertex_indices,
        );
        total_vertices_written += vertices_out.len() - before;
        total_output_ns += t3.elapsed().as_nanos() as u64;
    }

    CellPhaseTimings {
        num_cells: n,
        knn_ms: total_knn_ns as f64 / 1_000_000.0,
        clip_ms: total_clip_ns as f64 / 1_000_000.0,
        terminate_ms: total_terminate_ns as f64 / 1_000_000.0,
        output_ms: total_output_ns as f64 / 1_000_000.0,
        total_clips,
        total_terminate_checks,
        total_vertices_written,
    }
}

struct CellPhaseTimings {
    num_cells: usize,
    knn_ms: f64,
    clip_ms: f64,
    terminate_ms: f64,
    output_ms: f64,
    total_clips: usize,
    total_terminate_checks: usize,
    total_vertices_written: usize,
}

impl CellPhaseTimings {
    fn total_ms(&self) -> f64 {
        self.knn_ms + self.clip_ms + self.terminate_ms + self.output_ms
    }

    fn print(&self) {
        let total = self.total_ms();
        let avg_clips = self.total_clips as f64 / self.num_cells as f64;
        let avg_verts = self.total_vertices_written as f64 / self.num_cells as f64;

        println!("  KNN lookup:    {:>7.1}ms ({:>5.1}%)",
            self.knn_ms, self.knn_ms / total * 100.0);
        println!("  Clip total:    {:>7.1}ms ({:>5.1}%)  [{:.1} clips/cell]",
            self.clip_ms, self.clip_ms / total * 100.0, avg_clips);
        println!("  Termination:   {:>7.1}ms ({:>5.1}%)  [{} checks]",
            self.terminate_ms, self.terminate_ms / total * 100.0, self.total_terminate_checks);
        println!("  Output verts:  {:>7.1}ms ({:>5.1}%)  [{:.1} verts/cell]",
            self.output_ms, self.output_ms / total * 100.0, avg_verts);
        println!("  ─────────────────────────────────");
        println!("  Total:         {:>7.1}ms", total);
    }
}

/// Collect distribution of neighbors needed before termination
fn benchmark_termination_distribution(points: &[Vec3], knn: &impl KnnProvider, k: usize) {
    println!("\n--- Termination Distribution ---\n");

    let n = points.len();
    let termination = TerminationConfig {
        enabled: true,
        check_start: 10,
        check_step: 6,
    };

    // Count how many cells terminate at each neighbor count
    let mut terminate_at: Vec<usize> = vec![0; k + 1]; // index = neighbor count when terminated
    let mut no_terminate = 0usize;

    let mut scratch = knn.make_scratch();
    let mut neighbors = Vec::with_capacity(k);
    let mut builder = IncrementalCellBuilder::new(0, Vec3::ZERO);

    for i in 0..n {
        neighbors.clear();
        knn.knn_into(points[i], i, k, &mut scratch, &mut neighbors);
        builder.reset(i, points[i]);

        let mut terminated_at: Option<usize> = None;
        for (count, &neighbor_idx) in neighbors.iter().enumerate() {
            let neighbor = points[neighbor_idx];
            builder.clip(neighbor_idx, neighbor);

            if termination.should_check(count + 1) && builder.vertex_count() >= 3 {
                let neighbor_cos = points[i].dot(neighbor).clamp(-1.0, 1.0);
                if builder.can_terminate(neighbor_cos) {
                    terminated_at = Some(count + 1);
                    break;
                }
            }
        }

        match terminated_at {
            Some(at) => terminate_at[at] += 1,
            None => no_terminate += 1,
        }
    }

    // Print distribution
    let mut cumulative = 0usize;
    println!("  Neighbors | Terminated |  Cumul % | Histogram");
    println!("  ----------+------------+----------+----------");

    for i in 1..=k {
        if terminate_at[i] > 0 || i <= 16 {
            cumulative += terminate_at[i];
            let pct = cumulative as f64 / n as f64 * 100.0;
            let bar_len = (terminate_at[i] as f64 / n as f64 * 50.0) as usize;
            let bar: String = "█".repeat(bar_len);
            println!("  {:>9} | {:>10} | {:>7.1}% | {}",
                i, terminate_at[i], pct, bar);
        }
    }

    if no_terminate > 0 {
        let pct = (n - no_terminate) as f64 / n as f64 * 100.0;
        println!("  {:>9} | {:>10} | {:>7.1}% | (used all k)",
            "no term", no_terminate, pct);
    }

    // Summary stats
    let terminated_count = n - no_terminate;
    let avg_neighbors: f64 = (1..=k)
        .map(|i| i * terminate_at[i])
        .sum::<usize>() as f64 / terminated_count.max(1) as f64;

    println!();
    println!("  Terminated: {}/{} ({:.1}%)",
        terminated_count, n, terminated_count as f64 / n as f64 * 100.0);
    println!("  Avg neighbors (when terminated): {:.1}", avg_neighbors);
    if no_terminate > 0 {
        println!("  No termination (used all k={}): {} ({:.1}%)",
            k, no_terminate, no_terminate as f64 / n as f64 * 100.0);
    }
}

/// Compare adaptive-k against a fixed high-k baseline
fn test_adaptive_k_correctness(points: &[Vec3], knn: &impl KnnProvider) {
    use hex3::geometry::gpu_voronoi::AdaptiveKConfig;

    println!("\n--- Adaptive-K Correctness Test ---\n");

    let termination = TerminationConfig {
        enabled: true,
        check_start: 10,
        check_step: 6,
    };

    // Build with adaptive-k (what production uses)
    let adaptive = AdaptiveKConfig::default(); // initial=12, step=12, max=48
    let flat_adaptive = hex3::geometry::gpu_voronoi::build_cells_data_flat_adaptive(
        points, knn, adaptive, termination
    );

    // Build with fixed high-k (ground truth)
    let fixed_high = AdaptiveKConfig::fixed(64);
    let flat_truth = hex3::geometry::gpu_voronoi::build_cells_data_flat_adaptive(
        points, knn, fixed_high, termination
    );

    // Compare vertex counts per cell
    let adaptive_counts: Vec<usize> = flat_adaptive.iter_cells()
        .map(|verts| verts.len())
        .collect();
    let truth_counts: Vec<usize> = flat_truth.iter_cells()
        .map(|verts| verts.len())
        .collect();

    let mut mismatches = 0usize;
    for (i, (&a, &t)) in adaptive_counts.iter().zip(truth_counts.iter()).enumerate() {
        if a != t {
            mismatches += 1;
            if mismatches <= 5 {
                println!("  Cell {}: adaptive={} verts, truth={} verts", i, a, t);
            }
        }
    }

    if mismatches == 0 {
        println!("  ✓ All {} cells match! Adaptive-k is correct.", points.len());
    } else {
        println!("  ✗ {} cells have mismatched vertex counts", mismatches);
    }

    println!("  Adaptive stats: {} total vertices", flat_adaptive.total_vertices());
    println!("  Truth stats: {} total vertices", flat_truth.total_vertices());
}

/// Performance comparison: fixed-k vs adaptive-k
fn benchmark_adaptive_vs_fixed(points: &[Vec3], knn: &impl KnnProvider) {
    use hex3::geometry::gpu_voronoi::AdaptiveKConfig;

    println!("\n--- Adaptive-K vs Fixed-K Performance ---\n");

    let termination = TerminationConfig {
        enabled: true,
        check_start: 10,
        check_step: 6,
    };

    let configs = [
        ("Fixed k=24", AdaptiveKConfig::fixed(24)),
        ("Fixed k=48", AdaptiveKConfig::fixed(48)),
        ("Adaptive 12/24/48", AdaptiveKConfig::adaptive(12, 12, 24, 48)),
    ];

    for (name, config) in configs {
        // Warmup
        let _ = hex3::geometry::gpu_voronoi::build_cells_data_flat_adaptive(
            points, knn, config, termination
        );

        // Measure
        let t0 = Instant::now();
        let result = hex3::geometry::gpu_voronoi::build_cells_data_flat_adaptive(
            points, knn, config, termination
        );
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(&result);

        println!("  {:<20}: {:>6.1}ms  ({} verts)", name, elapsed, result.total_vertices());
    }
}

/// Benchmark a single distribution tier
fn benchmark_distribution(distribution: PointDistribution, n: usize, seed: u64, k: usize) -> DistributionStats {
    let termination = TerminationConfig {
        enabled: true,
        check_start: 10,
        check_step: 6,
    };

    let points = generate_points(n, distribution, seed);
    let knn = CubeMapGridKnn::new(&points);
    let mut scratch = knn.make_scratch();
    let mut neighbors = Vec::with_capacity(k);
    let mut builder = IncrementalCellBuilder::new(0, Vec3::ZERO);

    let mut no_terminate = 0usize;
    let mut max_neighbors_needed = 0usize;
    let mut neighbors_histogram = vec![0usize; k + 1];

    for i in 0..n {
        neighbors.clear();
        knn.knn_into(points[i], i, k, &mut scratch, &mut neighbors);
        builder.reset(i, points[i]);

        let mut terminated = false;
        for (count, &neighbor_idx) in neighbors.iter().enumerate() {
            builder.clip(neighbor_idx, points[neighbor_idx]);
            if termination.should_check(count + 1) && builder.vertex_count() >= 3 {
                let neighbor_cos = points[i].dot(points[neighbor_idx]).clamp(-1.0, 1.0);
                if builder.can_terminate(neighbor_cos) {
                    terminated = true;
                    max_neighbors_needed = max_neighbors_needed.max(count + 1);
                    neighbors_histogram[count + 1] += 1;
                    break;
                }
            }
        }
        if !terminated {
            no_terminate += 1;
        }
    }

    DistributionStats {
        distribution,
        n,
        no_terminate,
        max_neighbors_needed,
        neighbors_histogram,
    }
}

struct DistributionStats {
    distribution: PointDistribution,
    n: usize,
    no_terminate: usize,
    max_neighbors_needed: usize,
    neighbors_histogram: Vec<usize>,
}

impl DistributionStats {
    fn print_summary(&self) {
        let pct = self.no_terminate as f64 / self.n as f64 * 100.0;
        println!("  {:<12} ({:<30}): no_term={:<5} ({:>6.3}%), max_k={}",
            self.distribution.name(),
            self.distribution.description(),
            self.no_terminate,
            pct,
            self.max_neighbors_needed);
    }

    fn print_histogram(&self) {
        println!("\n  Termination histogram for {} ({}):",
            self.distribution.name(), self.distribution.description());
        println!("  Neighbors | Count    |  Cumul % | Histogram");
        println!("  ----------+----------+----------+----------");

        let mut cumulative = 0usize;
        for (i, &count) in self.neighbors_histogram.iter().enumerate() {
            if count > 0 || (i >= 10 && i <= 16) {
                cumulative += count;
                let pct = cumulative as f64 / self.n as f64 * 100.0;
                let bar_len = (count as f64 / self.n as f64 * 50.0) as usize;
                let bar: String = "█".repeat(bar_len);
                println!("  {:>9} | {:>8} | {:>7.1}% | {}",
                    i, count, pct, bar);
            }
        }
        if self.no_terminate > 0 {
            println!("  {:>9} | {:>8} | {:>7.1}% | (used all k)",
                "no term", self.no_terminate,
                (self.n - self.no_terminate) as f64 / self.n as f64 * 100.0);
        }
    }
}

/// Full benchmark comparing all distribution tiers
fn benchmark_all_distributions(n: usize, seed: u64, k: usize, verbose: bool) {
    println!("\n=== Point Distribution Comparison (n={}, k={}) ===\n", n, k);

    let distributions = [
        PointDistribution::Behaved,
        PointDistribution::Jittery,
        PointDistribution::Evil,
    ];

    let stats: Vec<_> = distributions.iter()
        .map(|&d| benchmark_distribution(d, n, seed, k))
        .collect();

    for s in &stats {
        s.print_summary();
    }

    if verbose {
        for s in &stats {
            s.print_histogram();
        }
    }
}

/// Test adaptive-k correctness across all distribution tiers
fn test_adaptive_correctness_all_tiers(n: usize, seed: u64) {
    use hex3::geometry::gpu_voronoi::AdaptiveKConfig;

    println!("\n=== Adaptive-K Correctness Across Tiers (n={}) ===\n", n);

    let termination = TerminationConfig {
        enabled: true,
        check_start: 10,
        check_step: 6,
    };

    let distributions = [
        PointDistribution::Behaved,
        PointDistribution::Jittery,
        PointDistribution::Evil,
    ];

    for dist in distributions {
        let points = generate_points(n, dist, seed);
        let knn = CubeMapGridKnn::new(&points);

        // Adaptive-k
        let adaptive = AdaptiveKConfig::default();
        let flat_adaptive = hex3::geometry::gpu_voronoi::build_cells_data_flat_adaptive(
            &points, &knn, adaptive, termination
        );

        // Ground truth (fixed high-k)
        let fixed_high = AdaptiveKConfig::fixed(64);
        let flat_truth = hex3::geometry::gpu_voronoi::build_cells_data_flat_adaptive(
            &points, &knn, fixed_high, termination
        );

        let adaptive_counts: Vec<usize> = flat_adaptive.iter_cells()
            .map(|verts| verts.len())
            .collect();
        let truth_counts: Vec<usize> = flat_truth.iter_cells()
            .map(|verts| verts.len())
            .collect();

        let mismatches: usize = adaptive_counts.iter()
            .zip(truth_counts.iter())
            .filter(|(a, t)| a != t)
            .count();

        let status = if mismatches == 0 { "✓" } else { "✗" };
        println!("  {} {:<12}: {} cells, {} mismatches, {} total verts",
            status, dist.name(), n, mismatches, flat_adaptive.total_vertices());
    }
}

/// Performance comparison across tiers
fn benchmark_performance_all_tiers(n: usize, seed: u64) {
    use hex3::geometry::gpu_voronoi::AdaptiveKConfig;

    println!("\n=== Performance Across Tiers (n={}) ===\n", n);

    let termination = TerminationConfig {
        enabled: true,
        check_start: 10,
        check_step: 6,
    };

    let distributions = [
        PointDistribution::Behaved,
        PointDistribution::Jittery,
        PointDistribution::Evil,
    ];

    let configs = [
        ("Fixed k=24", AdaptiveKConfig::fixed(24)),
        ("Adaptive", AdaptiveKConfig::adaptive(12, 12, 24, 48)),
    ];

    println!("  {:12} | {:>12} | {:>12}", "Distribution", "Fixed k=24", "Adaptive");
    println!("  {:─<12}-+-{:─>12}-+-{:─>12}", "", "", "");

    for dist in distributions {
        let points = generate_points(n, dist, seed);
        let knn = CubeMapGridKnn::new(&points);

        let mut times = Vec::new();
        for (_, config) in &configs {
            // Warmup
            let _ = hex3::geometry::gpu_voronoi::build_cells_data_flat_adaptive(
                &points, &knn, *config, termination
            );

            // Measure
            let t0 = Instant::now();
            let result = hex3::geometry::gpu_voronoi::build_cells_data_flat_adaptive(
                &points, &knn, *config, termination
            );
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            std::hint::black_box(&result);
            times.push(elapsed);
        }

        println!("  {:12} | {:>10.1}ms | {:>10.1}ms",
            dist.name(), times[0], times[1]);
    }
}

/// Benchmark various adaptive-k configurations across all distributions
fn benchmark_adaptive_tuning_behaved(n: usize, seed: u64) {
    use hex3::geometry::gpu_voronoi::AdaptiveKConfig;

    println!("\n=== Adaptive-K Tuning Across Distributions (n={}) ===\n", n);
    println!("  Config format: initial/step/max (track_limit = max)");
    println!();

    let termination = TerminationConfig {
        enabled: true,
        check_start: 10,
        check_step: 6,
    };

    // Configurations to test
    // Format: adaptive(initial_k, step_k, track_limit, fallback_k)
    let configs: Vec<(&str, AdaptiveKConfig)> = vec![
        // Baselines (no fallback)
        ("Fixed k=24", AdaptiveKConfig::fixed(24)),
        ("Fixed k=48", AdaptiveKConfig::fixed(48)),

        // track_limit=24, no fallback (fast, behaved only)
        ("12/12/24/0", AdaptiveKConfig::adaptive(12, 12, 24, 0)),

        // track_limit=24 with fallback=48 (fast path + safety net)
        ("12/12/24/48", AdaptiveKConfig::adaptive(12, 12, 24, 48)),
        ("10/7/24/48", AdaptiveKConfig::adaptive(10, 7, 24, 48)),
        ("10/14/24/48", AdaptiveKConfig::adaptive(10, 14, 24, 48)),

        // track_limit=32 with fallback=48
        ("12/10/32/48", AdaptiveKConfig::adaptive(12, 10, 32, 48)),
        ("10/11/32/48", AdaptiveKConfig::adaptive(10, 11, 32, 48)),

        // Default config
        ("default", AdaptiveKConfig::default()),
    ];

    let distributions = [
        PointDistribution::Behaved,
        PointDistribution::Jittery,
        PointDistribution::Evil,
    ];

    // Header
    println!("  {:14} | {:^12} | {:^12} | {:^12}",
        "Config", "Behaved", "Jittery", "Evil");
    println!("  {:─<14}-+-{:─^12}-+-{:─^12}-+-{:─^12}", "", "", "", "");

    for (name, config) in &configs {
        let mut times = Vec::new();
        let mut all_correct = true;

        for dist in &distributions {
            let points = generate_points(n, *dist, seed);
            let knn = CubeMapGridKnn::new(&points);

            // Ground truth
            let truth = hex3::geometry::gpu_voronoi::build_cells_data_flat_adaptive(
                &points, &knn, AdaptiveKConfig::fixed(64), termination
            );
            let truth_verts = truth.total_vertices();

            // Warmup
            let _ = hex3::geometry::gpu_voronoi::build_cells_data_flat_adaptive(
                &points, &knn, *config, termination
            );

            // Measure
            let t0 = Instant::now();
            let result = hex3::geometry::gpu_voronoi::build_cells_data_flat_adaptive(
                &points, &knn, *config, termination
            );
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            std::hint::black_box(&result);

            let correct = result.total_vertices() == truth_verts;
            all_correct = all_correct && correct;
            times.push((elapsed, correct));
        }

        let fmt = |t: (f64, bool)| -> String {
            if t.1 { format!("{:>6.1}ms", t.0) } else { format!("{:>6.1}ms ✗", t.0) }
        };

        println!("  {:14} | {:>12} | {:>12} | {:>12}",
            name, fmt(times[0]), fmt(times[1]), fmt(times[2]));
    }
}

/// Benchmark raw KNN query performance (for measuring scratch/resumable changes)
fn benchmark_knn_raw(n: usize, seed: u64) {
    use hex3::geometry::cube_grid::CubeMapGrid;

    println!("\n=== Raw KNN Query Performance (n={}) ===\n", n);

    for dist in [PointDistribution::Behaved, PointDistribution::Evil] {
        let points = generate_points(n, dist, seed);
        let res = ((n as f64 / 300.0).sqrt() as usize).max(4);
        let grid = CubeMapGrid::new(&points, res);

        println!("  {} points:", dist.name());

        // Test different k values
        for k in [12, 24, 48] {
            let mut scratch = grid.make_scratch();
            let mut neighbors = Vec::with_capacity(k);

            // Warmup
            for i in 0..1000 {
                grid.find_k_nearest_with_scratch_into(&points, points[i], i, k, &mut scratch, &mut neighbors);
            }

            // Measure
            let t0 = Instant::now();
            for i in 0..n {
                grid.find_k_nearest_with_scratch_into(&points, points[i], i, k, &mut scratch, &mut neighbors);
            }
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

            println!("    k={:2}: {:>7.1}ms ({:.2}µs/query)", k, elapsed, elapsed * 1000.0 / n as f64);
        }

        // Test resumable: k=12 then resume to k=24
        {
            let mut scratch = grid.make_scratch();
            let mut neighbors = Vec::with_capacity(48);
            let track_limit = 24; // Track 24 candidates (like-for-like with fixed k=24)

            // Warmup
            for i in 0..1000 {
                grid.find_k_nearest_resumable_into(&points, points[i], i, 12, track_limit, &mut scratch, &mut neighbors);
                grid.resume_k_nearest_into(&points, points[i], i, 24, &mut scratch, &mut neighbors);
            }

            // Measure
            let t0 = Instant::now();
            for i in 0..n {
                let _exhausted = grid.find_k_nearest_resumable_into(&points, points[i], i, 12, track_limit, &mut scratch, &mut neighbors);
                grid.resume_k_nearest_into(&points, points[i], i, 24, &mut scratch, &mut neighbors);
            }
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

            println!("    12→24 resume:  {:>7.1}ms ({:.2}µs/query)", elapsed, elapsed * 1000.0 / n as f64);
        }

        // Test re-query: k=12 then fresh k=24 (baseline)
        {
            let mut scratch = grid.make_scratch();
            let mut neighbors = Vec::with_capacity(24);

            // Warmup
            for i in 0..1000 {
                grid.find_k_nearest_with_scratch_into(&points, points[i], i, 12, &mut scratch, &mut neighbors);
                grid.find_k_nearest_with_scratch_into(&points, points[i], i, 24, &mut scratch, &mut neighbors);
            }

            // Measure
            let t0 = Instant::now();
            for i in 0..n {
                grid.find_k_nearest_with_scratch_into(&points, points[i], i, 12, &mut scratch, &mut neighbors);
                grid.find_k_nearest_with_scratch_into(&points, points[i], i, 24, &mut scratch, &mut neighbors);
            }
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

            println!("    12→24 requery: {:>7.1}ms ({:.2}µs/query)", elapsed, elapsed * 1000.0 / n as f64);
        }

        // Compare track_limit=24 vs track_limit=48
        {
            let mut scratch = grid.make_scratch();
            let mut neighbors = Vec::with_capacity(48);

            // track_limit=24
            let t0 = Instant::now();
            for i in 0..n {
                grid.find_k_nearest_resumable_into(&points, points[i], i, 12, 24, &mut scratch, &mut neighbors);
                grid.resume_k_nearest_into(&points, points[i], i, 24, &mut scratch, &mut neighbors);
            }
            let time_24 = t0.elapsed().as_secs_f64() * 1000.0;

            // track_limit=48
            let t0 = Instant::now();
            for i in 0..n {
                grid.find_k_nearest_resumable_into(&points, points[i], i, 12, 48, &mut scratch, &mut neighbors);
                grid.resume_k_nearest_into(&points, points[i], i, 24, &mut scratch, &mut neighbors);
            }
            let time_48 = t0.elapsed().as_secs_f64() * 1000.0;

            let diff_pct = (time_48 - time_24) / time_24 * 100.0;
            println!("    track_limit comparison: 24={:.1}ms, 48={:.1}ms ({:+.1}%)",
                time_24, time_48, diff_pct);
        }

        println!();
    }
}

fn main() {
    let args = Args::parse();

    let mut run_all = args.all;
    let distribution_arg = args.distribution.unwrap_or(DistributionArg::Jittery);
    if matches!(distribution_arg, DistributionArg::All) {
        run_all = true;
    }
    let distribution = distribution_arg
        .to_distribution()
        .unwrap_or(PointDistribution::Jittery);

    let seed = 12345u64;
    let k = 24;
    let sizes = if !args.sizes.is_empty() {
        args.sizes
    } else if args.quick {
        vec![100_000]
    } else {
        vec![100_000, 500_000, 1_000_000]
    };

    if args.knn {
        // Run raw KNN benchmark
        let n = 100_000;
        benchmark_knn_raw(n, seed);
        return;
    }

    if args.tuning {
        // Run adaptive-k tuning benchmark
        let n = 100_000;
        benchmark_adaptive_tuning_behaved(n, seed);
        return;
    }

    if args.tiers || run_all {
        // Run tier comparison benchmarks
        let n = 100_000;
        benchmark_all_distributions(n, seed, k, args.verbose);
        test_adaptive_correctness_all_tiers(n, seed);
        benchmark_performance_all_tiers(n, seed);
        return;
    }

    let termination = TerminationConfig {
        enabled: true,
        check_start: 10,
        check_step: 6,
    };

    println!("Cell Construction Phase Breakdown");
    println!("==================================");
    println!("Distribution: {} ({})\n", distribution.name(), distribution.description());

    for &n in &sizes {
        println!("n = {}", n);
        println!("-----------------------------------");

        let points = generate_points(n, distribution, seed);

        // For large scales (>500k), use production parallel code instead of instrumented
        if n > 500_000 {
            // Production benchmark - parallel, no instrumentation overhead
            use hex3::geometry::gpu_voronoi::compute_voronoi_gpu_style_timed;

            // Warmup
            let _ = compute_voronoi_gpu_style_timed(&points, k, false);

            // Measure
            let t0 = Instant::now();
            let voronoi = compute_voronoi_gpu_style_timed(&points, k, true);
            let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let total_verts: usize = voronoi.iter_cells().map(|c| c.len()).sum();
            println!("  Total:         {:>7.1}ms", total_ms);
            println!("  Cells: {}, Vertices: {}, Avg: {:.1} verts/cell",
                voronoi.num_cells(), total_verts, total_verts as f64 / n as f64);
        } else {
            // Detailed instrumented benchmark for smaller scales
            let knn = CubeMapGridKnn::new(&points);

            // Warmup
            let _ = build_cells_instrumented(&points, &knn, k, termination);

            // Measure
            let timings = build_cells_instrumented(&points, &knn, k, termination);
            timings.print();

            // For 100k, also do detailed breakdowns
            if n == 100_000 {
                benchmark_termination_distribution(&points, &knn, k);
                test_adaptive_k_correctness(&points, &knn);
                benchmark_adaptive_vs_fixed(&points, &knn);
            }
        }

        println!();
    }
}
