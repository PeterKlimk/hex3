//! Benchmark GPU-style Voronoi at large scales.
//!
//! Run with: cargo run --release --bin bench_voronoi
//!
//! Usage:
//!   bench_voronoi              Run default size (100k)
//!   bench_voronoi 100k 500k 1m Run multiple sizes
//!   bench_voronoi --lloyd      Use Lloyd-relaxed points
//!   bench_voronoi -n 10        Run 10 iterations (for profiling)
//!
//! For detailed sub-phase timing, build with: cargo run --release --features timing --bin bench_voronoi

use clap::Parser;
use glam::Vec3;
use hex3::geometry::gpu_voronoi::{compute_voronoi_gpu_style, compute_voronoi_gpu_style_bench};
use hex3::geometry::{fibonacci_sphere_points_with_rng, lloyd_relax_kmeans};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::io::{self, Write};
use std::time::Instant;

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

fn mean_spacing(num_points: usize) -> f32 {
    if num_points == 0 {
        return 0.0;
    }
    (4.0 * std::f32::consts::PI / num_points as f32).sqrt()
}

#[derive(Parser)]
#[command(name = "bench_voronoi")]
#[command(about = "Benchmark GPU-style Voronoi at various scales")]
struct Args {
    /// Cell counts to benchmark (e.g., 100k, 1m, 10M)
    #[arg(value_parser = parse_count)]
    sizes: Vec<usize>,

    /// Random seed
    #[arg(short, long, default_value_t = 12345)]
    seed: u64,

    /// Use Lloyd-relaxed points (well-behaved, like production)
    #[arg(long)]
    lloyd: bool,

    /// Compare against convex hull ground truth (slow, max 100k)
    #[arg(long)]
    validate: bool,

    /// Skip preprocessing (merge close points) - for benchmarking
    #[arg(long)]
    no_preprocess: bool,

    /// Use old hash-based post-dedup instead of live sharded dedup
    #[arg(long)]
    old_dedup: bool,

    /// Compare both dedup methods side-by-side
    #[arg(long)]
    compare_dedup: bool,

    /// Number of iterations to run (useful for profiling)
    #[arg(short = 'n', long, default_value_t = 1)]
    repeat: usize,
}

fn generate_points(n: usize, seed: u64, lloyd: bool) -> Vec<Vec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let jitter_scale = if lloyd { 0.1 } else { 0.25 };
    let jitter = mean_spacing(n) * jitter_scale;
    let mut points = fibonacci_sphere_points_with_rng(n, jitter, &mut rng);
    if lloyd {
        lloyd_relax_kmeans(&mut points, 2, 20, &mut rng);
    }
    points
}

fn format_rate(count: usize, ms: f64) -> String {
    if ms <= 0.0 {
        return "N/A".to_string();
    }
    let per_sec = count as f64 / (ms / 1000.0);
    if per_sec >= 1_000_000.0 {
        format!("{:.2}M/s", per_sec / 1_000_000.0)
    } else if per_sec >= 1_000.0 {
        format!("{:.1}k/s", per_sec / 1000.0)
    } else {
        format!("{:.0}/s", per_sec)
    }
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

fn validate_against_hull(points: &[Vec3]) {
    use hex3::geometry::SphericalVoronoi;

    println!("\nValidating against convex hull ground truth...");

    let t0 = Instant::now();
    let hull = SphericalVoronoi::compute(points);
    let hull_time = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let gpu = compute_voronoi_gpu_style(points);
    let gpu_time = t1.elapsed().as_secs_f64() * 1000.0;

    let mut exact_match = 0usize;
    let mut bad_cells = 0usize;

    for i in 0..points.len() {
        let hull_count = hull.cell(i).len();
        let gpu_count = gpu.cell(i).len();

        if hull_count == gpu_count {
            exact_match += 1;
        }
        if gpu_count < 3 {
            bad_cells += 1;
        }
    }

    let match_pct = exact_match as f64 / points.len() as f64 * 100.0;

    println!("  Convex hull time: {:>8.1}ms", hull_time);
    println!(
        "  GPU Voronoi time: {:>8.1}ms ({:.1}x faster)",
        gpu_time,
        hull_time / gpu_time
    );
    println!(
        "  Exact matches:    {:>8} / {} ({:.2}%)",
        exact_match,
        points.len(),
        match_pct
    );
    if bad_cells > 0 {
        println!("  Invalid cells:    {:>8} (< 3 vertices)", bad_cells);
    }
}

struct BenchResult {
    n: usize,
    time_ms: f64,
    num_vertices: usize,
    num_cells: usize,
    method: &'static str,
}

fn run_benchmark(points: &[Vec3], use_live_dedup: bool) -> BenchResult {
    let n = points.len();
    let method = if use_live_dedup {
        "live_dedup"
    } else {
        "old_dedup"
    };

    let t0 = Instant::now();
    let voronoi = compute_voronoi_gpu_style_bench(points, use_live_dedup);
    let time_ms = t0.elapsed().as_secs_f64() * 1000.0;

    BenchResult {
        n,
        time_ms,
        num_vertices: voronoi.vertices.len(),
        num_cells: voronoi.num_cells(),
        method,
    }
}

fn main() {
    let args = Args::parse();

    println!("GPU-style Voronoi Benchmark");
    println!("===========================\n");

    let sizes: Vec<usize> = if args.sizes.is_empty() {
        vec![100_000]
    } else {
        args.sizes
    };

    let point_type = if args.lloyd {
        "Lloyd-relaxed"
    } else {
        "fibonacci+jitter"
    };

    let dedup_mode = if args.compare_dedup {
        "compare (both)"
    } else if args.old_dedup {
        "old (hash-based post-dedup)"
    } else {
        "live (sharded live dedup)"
    };

    println!("Configuration:");
    println!("  seed = {}", args.seed);
    println!("  point type = {}", point_type);
    println!("  dedup = {}", dedup_mode);
    println!(
        "  sizes = {:?}",
        sizes.iter().map(|&n| format_num(n)).collect::<Vec<_>>()
    );
    if args.repeat > 1 {
        println!("  repeat = {}", args.repeat);
    }

    #[cfg(feature = "timing")]
    println!("  timing = enabled (detailed sub-phase timing will be printed)");

    let mut results: Vec<BenchResult> = Vec::new();

    for n in &sizes {
        println!("\n{}", "=".repeat(60));
        println!("Benchmarking n = {}", format_num(*n));
        println!("{}", "=".repeat(60));

        let t_gen = Instant::now();
        let points = generate_points(*n, args.seed, args.lloyd);
        let gen_time = t_gen.elapsed().as_secs_f64() * 1000.0;
        println!("Point generation: {:.1}ms", gen_time);

        if args.compare_dedup {
            // Warmup run (discard results) to eliminate first-run bias
            print!("Warmup run... ");
            io::stdout().flush().unwrap();
            let _ = run_benchmark(&points, true);
            println!("done");

            // Run both methods and compare
            println!("\n--- LIVE DEDUP (sharded) ---");
            let live_result = run_benchmark(&points, true);
            println!("  Total time:    {:>8.1}ms", live_result.time_ms);
            println!(
                "  Throughput:    {:>8}",
                format_rate(live_result.n, live_result.time_ms)
            );
            println!(
                "  Vertices:      {:>8}",
                format_num(live_result.num_vertices)
            );

            println!("\n--- OLD DEDUP (hash-based) ---");
            let old_result = run_benchmark(&points, false);
            println!("  Total time:    {:>8.1}ms", old_result.time_ms);
            println!(
                "  Throughput:    {:>8}",
                format_rate(old_result.n, old_result.time_ms)
            );
            println!(
                "  Vertices:      {:>8}",
                format_num(old_result.num_vertices)
            );

            // Comparison
            let speedup = old_result.time_ms / live_result.time_ms;
            let diff_ms = old_result.time_ms - live_result.time_ms;
            println!("\n--- COMPARISON ---");
            if diff_ms > 0.0 {
                println!(
                    "  Live dedup is {:.1}ms FASTER ({:.2}x speedup)",
                    diff_ms, speedup
                );
            } else {
                println!(
                    "  Live dedup is {:.1}ms SLOWER ({:.2}x slowdown)",
                    -diff_ms,
                    1.0 / speedup
                );
            }

            if old_result.num_vertices != live_result.num_vertices {
                println!(
                    "  WARNING: vertex count differs! old={} live={}",
                    old_result.num_vertices, live_result.num_vertices
                );
            }

            results.push(live_result);
            results.push(old_result);
        } else {
            let use_live = !args.old_dedup;

            // Run benchmark (with optional repeats for profiling)
            let mut times: Vec<f64> = Vec::with_capacity(args.repeat);
            let mut last_result: Option<BenchResult> = None;

            for iter in 0..args.repeat {
                if args.repeat > 1 {
                    print!("  Iteration {}/{}... ", iter + 1, args.repeat);
                    io::stdout().flush().unwrap();
                }

                let result = run_benchmark(&points, use_live);
                times.push(result.time_ms);

                if args.repeat > 1 {
                    println!("{:.1}ms", result.time_ms);
                }

                last_result = Some(result);
            }

            let result = last_result.unwrap();

            println!("\nResults ({}):", result.method);
            if args.repeat > 1 {
                let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let avg = times.iter().sum::<f64>() / times.len() as f64;
                println!("  Min time:      {:>8.1}ms", min);
                println!("  Max time:      {:>8.1}ms", max);
                println!("  Avg time:      {:>8.1}ms", avg);
                println!(
                    "  Throughput:    {:>8} (avg)",
                    format_rate(result.n, avg)
                );
            } else {
                println!("  Total time:    {:>8.1}ms", result.time_ms);
                println!(
                    "  Throughput:    {:>8}",
                    format_rate(result.n, result.time_ms)
                );
            }
            println!("  Vertices:      {:>8}", format_num(result.num_vertices));
            println!("  Cells:         {:>8}", format_num(result.num_cells));
            println!(
                "  Avg verts/cell:{:>8.2}",
                result.num_vertices as f64 * 3.0 / result.num_cells as f64
            );

            if args.validate && *n <= 100_000 {
                validate_against_hull(&points);
            } else if args.validate && *n > 100_000 {
                println!("\n  (skipping validation for n > 100k - convex hull is slow)");
            }

            results.push(result);
        }
    }

    // Summary table if multiple sizes (and not compare mode)
    if results.len() > 1 && !args.compare_dedup {
        println!("\n\n{}", "=".repeat(60));
        println!("SUMMARY");
        println!("{}", "=".repeat(60));
        println!(
            "{:>10} | {:>10} | {:>12} | {:>10}",
            "n", "time", "throughput", "verts"
        );
        println!("{:-<10}-+-{:-<10}-+-{:-<12}-+-{:-<10}", "", "", "", "");

        for r in &results {
            println!(
                "{:>10} | {:>9.1}ms | {:>12} | {:>10}",
                format_num(r.n),
                r.time_ms,
                format_rate(r.n, r.time_ms),
                format_num(r.num_vertices)
            );
        }
    }

    println!("\nBenchmark complete.");
}
