mod app;

use std::path::PathBuf;

use clap::Parser;
use winit::event_loop::{ControlFlow, EventLoop};

use app::world::{advance_to_stage_2, advance_to_stage_3, create_world_with_options};

/// Hex3 - Spherical Voronoi planet generator
#[derive(Parser, Debug)]
#[command(name = "hex3", version, about)]
struct Cli {
    /// Run in headless mode (no window, generate and quit)
    #[arg(long)]
    headless: bool,

    /// Target stage (1-3). Interactive defaults to 1, headless defaults to max.
    #[arg(long)]
    stage: Option<u32>,

    /// Random seed for world generation
    #[arg(long)]
    seed: Option<u64>,

    /// Export world data to file (supports .json and .json.gz)
    #[arg(long, value_name = "FILE")]
    export: Option<PathBuf>,

    /// Use GPU-style Voronoi algorithm (experimental)
    #[arg(long)]
    gpu_voronoi: bool,

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
        3 // Headless defaults to max stage (currently 3)
    } else {
        1 // Interactive defaults to stage 1
    };

    if cli.headless {
        run_headless(cli.seed, target_stage, cli.export, cli.gpu_voronoi);
    } else {
        run_interactive(cli.seed, target_stage, cli.export, cli.gpu_voronoi);
    }
}

fn run_headless(
    seed: Option<u64>,
    target_stage: u32,
    export_path: Option<PathBuf>,
    gpu_voronoi: bool,
) {
    let seed = seed.unwrap_or_else(rand::random);
    println!(
        "Headless mode: seed={}, target_stage={}, gpu_voronoi={}",
        seed, target_stage, gpu_voronoi
    );

    // Generate world
    print!("Generating world... ");
    let start = std::time::Instant::now();
    let mut world = create_world_with_options(seed, gpu_voronoi);
    println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

    // Advance to target stage
    if target_stage >= 2 {
        print!("Advancing to stage 2 (Climate)... ");
        let start = std::time::Instant::now();
        advance_to_stage_2(&mut world);
        println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);
    }
    if target_stage >= 3 {
        print!("Advancing to stage 3 (Hydrology)... ");
        let start = std::time::Instant::now();
        advance_to_stage_3(&mut world);
        println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);
    }

    println!(
        "World complete: {} cells, stage {}",
        world.tessellation.num_cells(),
        world.current_stage()
    );

    // Export if requested
    if let Some(path) = export_path {
        app::export::export_world(&world, seed, &path);
    }
}

fn run_interactive(
    seed: Option<u64>,
    target_stage: u32,
    export_path: Option<PathBuf>,
    gpu_voronoi: bool,
) {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Wait);

    let config = app::AppConfig {
        seed,
        target_stage,
        export_path,
        gpu_voronoi,
    };

    let mut app = app::App::new(config);
    event_loop
        .run_app(&mut app)
        .expect("Failed to run application");
}
