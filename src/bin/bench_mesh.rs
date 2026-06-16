//! Benchmark the fine-mesh pipeline for the erosion project: tessellation
//! at millions of points via the knn-clipping backend, plus the per-iteration
//! cost of an erosion-style pass (steepest-descent routing + flow
//! accumulation) on the resulting mesh.
//!
//! Usage:
//!     RUST_LOG=debug cargo run --release --bin bench_mesh -- --sizes 100000,1000000,2500000

use std::time::Instant;

use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use hex3::world::Tessellation;

#[derive(Parser)]
struct Cli {
    /// Comma-separated point counts to benchmark.
    #[arg(long, default_value = "100000,500000,1000000,2500000,5000000")]
    sizes: String,

    #[arg(long, default_value_t = 12345)]
    seed: u64,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
        .format_timestamp(None)
        .init();

    let cli = Cli::parse();
    let sizes: Vec<usize> = cli
        .sizes
        .split(',')
        .map(|s| s.trim().parse().expect("bad size"))
        .collect();

    for &n in &sizes {
        println!("\n================ {} points ================", n);
        let mut rng = ChaCha8Rng::seed_from_u64(cli.seed);

        let t0 = Instant::now();
        let tess = Tessellation::generate_knn_clipping(n, 0, &mut rng);
        let t_tess = t0.elapsed();
        println!(
            "tessellation total (points + lloyd + voronoi + adjacency): {:.2?}",
            t_tess
        );

        // How fast is s2-voronoi's own adjacency, vs hex3's build_adjacency?
        {
            use rand::Rng;
            let mut rng2 = ChaCha8Rng::seed_from_u64(cli.seed ^ 0xa5a5);
            let pts: Vec<[f32; 3]> = (0..n)
                .map(|_| {
                    let v: [f32; 3] = [
                        rng2.gen::<f32>() - 0.5,
                        rng2.gen::<f32>() - 0.5,
                        rng2.gen::<f32>() - 0.5,
                    ];
                    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-9);
                    [v[0] / len, v[1] / len, v[2] / len]
                })
                .collect();
            let t0 = Instant::now();
            let diagram = s2_voronoi::compute(&pts).expect("voronoi failed");
            let t_vor = t0.elapsed();
            let t0 = Instant::now();
            let adj = diagram.build_adjacency();
            let t_adj = t0.elapsed();
            println!(
                "s2-voronoi native: compute {:.2?} + build_adjacency {:.2?} (complete: {})",
                t_vor,
                t_adj,
                adj.is_complete()
            );
        }

        let t0 = Instant::now();
        let areas = tess.cell_areas();
        println!(
            "cell areas: {:.2?} (sum {:.4})",
            t0.elapsed(),
            areas.iter().sum::<f32>()
        );

        // Synthetic elevation: smooth function of position + small hash noise,
        // enough structure for non-trivial drainage.
        let elev: Vec<f32> = (0..n)
            .map(|i| {
                let p = tess.cell_center(i);
                (p.x * 3.0).sin() * (p.y * 2.0).cos() + 0.3 * (p.z * 5.0).sin()
            })
            .collect();

        // One erosion-style iteration, the loop body that runs 50-200 times:
        // (a) steepest-descent receiver per cell, (b) flow accumulation via
        // topological order, (c) a slope-dependent update touching each edge.
        let t0 = Instant::now();
        let mut receiver: Vec<Option<usize>> = vec![None; n];
        for i in 0..n {
            let mut best = elev[i];
            for &nb in tess.neighbors(i) {
                if elev[nb] < best {
                    best = elev[nb];
                    receiver[i] = Some(nb);
                }
            }
        }
        let t_route = t0.elapsed();

        let t0 = Instant::now();
        let mut indegree = vec![0u32; n];
        for r in receiver.iter().flatten() {
            indegree[*r] += 1;
        }
        let mut flow = vec![1.0f32; n];
        let mut stack: Vec<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
        let mut processed = 0usize;
        while let Some(c) = stack.pop() {
            processed += 1;
            if let Some(r) = receiver[c] {
                flow[r] += flow[c];
                indegree[r] -= 1;
                if indegree[r] == 0 {
                    stack.push(r);
                }
            }
        }
        let t_accum = t0.elapsed();

        let t0 = Instant::now();
        let mut incision = vec![0.0f32; n];
        for i in 0..n {
            if let Some(r) = receiver[i] {
                let slope = (elev[i] - elev[r]).max(0.0);
                incision[i] = flow[i].sqrt() * slope;
            }
        }
        let t_incise = t0.elapsed();

        println!(
            "erosion iteration: route {:.2?} + accumulate {:.2?} + incise {:.2?} = {:.2?}  (processed {} cells, max flow {:.0})",
            t_route,
            t_accum,
            t_incise,
            t_route + t_accum + t_incise,
            processed,
            flow.iter().cloned().fold(0.0f32, f32::max)
        );
        println!(
            "  -> 100 iterations ~= {:.1?}",
            (t_route + t_accum + t_incise) * 100
        );
    }
}
