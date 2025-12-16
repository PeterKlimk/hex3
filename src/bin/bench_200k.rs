// Benchmark at 200k cells - with timing breakdown and accuracy check
use hex3::geometry::{fibonacci_sphere_points_with_rng, lloyd_relax_kmeans, SphericalVoronoi};
use hex3::geometry::gpu_voronoi::{
    compute_voronoi_gpu_style_timed_with_termination_params,
    compute_voronoi_gpu_style_with_stats_and_termination_params,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

fn main() {
    let mut skip_hull = false;
    let mut enable_termination = true;
    let mut termination_check_start: usize = 8;
    let mut termination_check_step: usize = 2;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--skip-hull" => skip_hull = true,
            "--no-termination" => enable_termination = false,
            "--term-start" => {
                termination_check_start = args
                    .next()
                    .expect("--term-start requires a value")
                    .parse()
                    .expect("invalid --term-start value");
            }
            "--term-step" => {
                termination_check_step = args
                    .next()
                    .expect("--term-step requires a value")
                    .parse()
                    .expect("invalid --term-step value");
            }
            _ => {}
        }
    }
    let n = 200_000;
    let mut rng = ChaCha8Rng::seed_from_u64(12345);

    // Generate Lloyd-relaxed points
    let mean_spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
    let jitter = mean_spacing * 0.25;

    println!("Generating {} Lloyd-relaxed points...", n);
    let start = Instant::now();
    let mut points = fibonacci_sphere_points_with_rng(n, jitter, &mut rng);
    lloyd_relax_kmeans(&mut points, 2, 20, &mut rng);
    println!("  Points: {:.1}ms\n", start.elapsed().as_secs_f64() * 1000.0);

    // Run with timing breakdown
    println!(
        "GPU-style Voronoi (k=20, termination={}, start={}, step={}) timing breakdown:\n",
        if enable_termination { "on" } else { "off" },
        termination_check_start,
        termination_check_step
    );
    let gpu = compute_voronoi_gpu_style_timed_with_termination_params(
        &points,
        20,
        true,
        enable_termination,
        termination_check_start,
        termination_check_step,
    );

    // Verify accuracy
    if !skip_hull {
        println!("\nVerifying accuracy against convex hull...");
        let hull = SphericalVoronoi::compute(&points);
        let exact = (0..n)
            .filter(|&i| hull.cell(i).len() == gpu.cell(i).len())
            .count();
        let bad = gpu.iter_cells().filter(|c| c.len() < 3).count();
        println!(
            "  Accuracy: {:.1}% | Bad cells: {}",
            exact as f64 / n as f64 * 100.0,
            bad
        );
    } else {
        let bad = gpu.iter_cells().filter(|c| c.len() < 3).count();
        println!("\nSkipped convex-hull verification (--skip-hull). Bad cells: {}", bad);
    }

    // Stats
    println!("\nWith stats (k=20):");
    let (_, stats) = compute_voronoi_gpu_style_with_stats_and_termination_params(
        &points,
        20,
        enable_termination,
        termination_check_start,
        termination_check_step,
    );
    println!(
        "  Avg neighbors: {:.1} | Termination rate: {:.1}% | Avg secondary hits: {:.2}",
        stats.avg_neighbors_processed,
        stats.termination_rate * 100.0,
        stats.avg_secondary_candidate_hits
    );
}
