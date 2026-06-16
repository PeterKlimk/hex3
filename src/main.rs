mod app;

use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use winit::event_loop::{ControlFlow, EventLoop};

use app::world::{
    advance_to_stage_2, advance_to_stage_3, advance_to_stage_4, create_world_with_options,
};
use hex3::world::{FineCacheMode, VoronoiBackend};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliVoronoiBackend {
    #[value(name = "convex-hull")]
    ConvexHull,
    #[value(name = "knn-clipping")]
    KnnClipping,
}

impl From<CliVoronoiBackend> for VoronoiBackend {
    fn from(value: CliVoronoiBackend) -> Self {
        match value {
            CliVoronoiBackend::ConvexHull => VoronoiBackend::ConvexHull,
            CliVoronoiBackend::KnnClipping => VoronoiBackend::KnnClipping,
        }
    }
}

/// Hex3 - Spherical Voronoi planet generator
#[derive(Parser, Debug)]
#[command(name = "hex3", version, about)]
struct Cli {
    /// Run in headless mode (no window, generate and quit)
    #[arg(long)]
    headless: bool,

    /// Target stage (1-4: Lithosphere, Atmosphere, Hydrosphere, Erosion).
    /// Interactive defaults to 1, headless defaults to max.
    #[arg(long)]
    stage: Option<u32>,

    /// Random seed for world generation
    #[arg(long)]
    seed: Option<u64>,

    /// Coarse Voronoi cell count (default 100000). Sweep this to test
    /// resolution independence of stages 1-2 (lithosphere/atmosphere).
    #[arg(long, default_value_t = 100_000)]
    cells: usize,

    /// Uniform multiplier on the fine-mesh cell-size targets (plains/mountain/
    /// ocean km). >1 coarsens the fine mesh, <1 refines it. Sweep this to test
    /// resolution independence of stages 3-4 (erosion/fine hydrology). A value
    /// other than 1.0 disables the fine-base disk cache.
    #[arg(long, default_value_t = 1.0)]
    fine_scale: f32,

    /// Export world data to file (supports .json and .json.gz)
    #[arg(long, value_name = "FILE")]
    export: Option<PathBuf>,

    /// Voronoi backend to use (convex-hull or knn-clipping)
    #[arg(long, value_enum, default_value_t = CliVoronoiBackend::ConvexHull)]
    voronoi_backend: CliVoronoiBackend,

    /// Disable the fine-mesh base disk cache (always regenerate stage 3a)
    #[arg(long)]
    no_fine_cache: bool,

    /// Force-rebuild and overwrite the fine-mesh base disk cache
    #[arg(long)]
    rebuild_fine_cache: bool,

    /// Legacy flag: equivalent to --stage 2
    #[arg(long, hide = true)]
    stage2: bool,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    // Determine target stage
    let target_stage = if let Some(s) = cli.stage {
        s
    } else if cli.stage2 {
        2 // Legacy flag
    } else if cli.headless {
        4 // Headless defaults to max stage (currently 4 = Erosion)
    } else {
        1 // Interactive defaults to stage 1
    };

    let backend = VoronoiBackend::from(cli.voronoi_backend);
    let fine_cache = if cli.no_fine_cache {
        FineCacheMode::Disabled
    } else if cli.rebuild_fine_cache {
        FineCacheMode::Rebuild
    } else {
        FineCacheMode::Enabled
    };

    if cli.headless {
        run_headless(
            cli.seed,
            cli.cells,
            cli.fine_scale,
            target_stage,
            cli.export,
            backend,
            fine_cache,
        );
    } else {
        run_interactive(cli.seed, target_stage, cli.export, backend, fine_cache);
    }
}

#[allow(clippy::too_many_arguments)]
fn run_headless(
    seed: Option<u64>,
    num_cells: usize,
    fine_scale: f32,
    target_stage: u32,
    export_path: Option<PathBuf>,
    voronoi_backend: VoronoiBackend,
    fine_cache: FineCacheMode,
) {
    let seed = seed.unwrap_or_else(rand::random);
    println!(
        "Headless mode: seed={}, cells={}, fine_scale={}, target_stage={}, voronoi_backend={}",
        seed, num_cells, fine_scale, target_stage, voronoi_backend
    );

    // Generate world
    print!("Generating world... ");
    let start = std::time::Instant::now();
    let mut world = create_world_with_options(seed, num_cells, voronoi_backend, fine_cache);
    // Apply the fine-mesh resolution multiplier. A non-default scale changes the
    // sampled mesh, so the disk cache (keyed on the density params) would miss
    // anyway; disable it to avoid writing a base per scale.
    if (fine_scale - 1.0).abs() > f32::EPSILON {
        let dp = &mut world.fine_density_params;
        dp.plains_km *= fine_scale;
        dp.mountain_km *= fine_scale;
        dp.ocean_km *= fine_scale;
        world.fine_cache = FineCacheMode::Disabled;
    }
    println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

    // Advance to target stage
    if target_stage >= 2 {
        print!("Advancing to stage 2 (Climate)... ");
        let start = std::time::Instant::now();
        advance_to_stage_2(&mut world);
        println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);
    }
    if target_stage >= 3 {
        print!("Advancing to stage 3 (Hydrosphere, pre-erosion)... ");
        let start = std::time::Instant::now();
        advance_to_stage_3(&mut world);
        println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);
    }
    if target_stage >= 4 {
        print!("Advancing to stage 4 (Erosion)... ");
        let start = std::time::Instant::now();
        advance_to_stage_4(&mut world);
        println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);
    }

    println!(
        "World complete: {} cells, stage {}",
        world.num_cells(),
        world.current_stage()
    );

    // Export if requested
    if let Some(path) = export_path {
        app::export::export_world(&mut world, seed, &path);
    }
}

fn run_interactive(
    seed: Option<u64>,
    target_stage: u32,
    export_path: Option<PathBuf>,
    voronoi_backend: VoronoiBackend,
    fine_cache: FineCacheMode,
) {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Wait);

    if export_path.is_some() {
        eprintln!("Note: --export is ignored in interactive mode; press D to export instead.");
    }

    let config = app::AppConfig {
        seed,
        target_stage,
        voronoi_backend,
        fine_cache,
    };

    let mut app = app::App::new(config);
    event_loop
        .run_app(&mut app)
        .expect("Failed to run application");
}
