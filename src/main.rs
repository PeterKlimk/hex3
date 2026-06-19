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

    /// MFD erosion exponent (routing ladder Rung 2/3). <0 = EROSION_MFD_EXPONENT
    /// default (off); 0 = single-flow; ~1 dispersive .. high ≈ single-flow. Set
    /// >0 to visually A/B MFD incision. See docs/specs/erosion-routing-ladder.md.
    #[arg(long, default_value_t = -1.0)]
    erosion_mfd_exponent: f32,

    /// Barnes convergent flat resolution (Rung 1). -1 = default (on); 0 = off
    /// (old flood_parent wavefront); 1 = on. A/B the spiral-on-flats fix.
    #[arg(long, default_value_t = -1)]
    erosion_flat_resolution: i8,

    /// Plains alluvial regime gate: channel slope (elev/km) at/above which
    /// incision is full; gentler channels fade to alluvial (floodplains, not
    /// ditches). <0 = EROSION_CONFINEMENT_SLOPE default (off). See
    /// docs/specs/erosion-valleys-not-channels.md.
    #[arg(long, default_value_t = -1.0)]
    erosion_confinement_slope: f32,

    /// Erosion erodibility K (incision strength; default 4e-2). <0 = default.
    /// LOWER = gentler dissection (less sharp). Visual dissection-texture lever.
    #[arg(long, default_value_t = -1.0)]
    erosion_k: f32,

    /// Hillslope diffusivity (smoothing; default 2e-8). <0 = default. HIGHER =
    /// smoother (rounds the sharp dissection). Visual dissection-texture lever.
    #[arg(long, default_value_t = -1.0)]
    erosion_diffusivity: f32,

    /// Channel-initiation support area (km² at mean land wetness; default 30).
    /// <0 = default. HIGHER = channels start later = LOWER drainage density
    /// (fewer/broader valleys, less "busy"). The primary density lever.
    #[arg(long, default_value_t = -1.0)]
    erosion_channel_support: f32,

    /// Uplift-FORCING smoothing length (km; escalation #1). Smooths the per-step
    /// tectonic uplift source over a sub-grid orogenic width to remove mountain-top
    /// cell-scale "swiss cheese" without flattening orogens. <0 = default; 0 = off.
    /// See docs/specs/erosion-uplift-smoothing.md.
    #[arg(long, default_value_t = -1.0)]
    erosion_uplift_smooth: f32,

    /// Roering nonlinear-hillslope critical slope S_c (escalation #2; Δelev/radian,
    /// ~grade·637). Diffusivity blows up toward S_c -> planar slopes + crisp
    /// ridges (vs linear-creep mush). <0 = default; 0 = off. Visual de-prickle
    /// lever; sweep ~150-300. See docs/specs/erosion-escalations.md.
    #[arg(long, default_value_t = -1.0)]
    erosion_hillslope_crit: f32,

    /// Hillslope-diffusion Jacobi sweeps per step (default 6). 0 = default. LOWER =
    /// cheaper diffuse phase; 6->3 measured metric-equivalent (incision percentiles
    /// byte-identical) — a near-free speedup, visual-confirm it.
    #[arg(long, default_value_t = 0)]
    erosion_diffusion_iters: usize,

    /// Steps between drainage re-routings (default 6). 0 = default; 1 = re-route
    /// every step. HIGHER = cheaper route phase; 6->12 measured near-equivalent
    /// (~3-5% incision drift). Visual-A/B the speedup.
    #[arg(long, default_value_t = 0)]
    erosion_reroute_interval: usize,

    /// Fluvial step count per precip pass (default 60). 0 = default. FEWER = cheaper
    /// AND less-mature / less-dissected terrain (erosion is still evolving at 60, not
    /// converged) — a perf x "less busy" fidelity dial, not free.
    #[arg(long, default_value_t = 0)]
    erosion_steps: usize,

    /// Coupled erode<->precip feedback passes (default 2). 0 = default; 1 = no
    /// feedback (halves total erosion, but +16% incision / pricklier — the 2nd pass
    /// does anti-prickle healing). Fidelity dial, not free.
    #[arg(long, default_value_t = 0)]
    erosion_precip_iters: usize,

    /// Tectonic uplift scale ("Hold & carve"). <0 = EROSION_UPLIFT_SCALE default;
    /// 0 = relaxation only.
    #[arg(long, default_value_t = -1.0)]
    erosion_uplift_scale: f32,

    /// Depositional repose slope (fans/floodplains/deltas). <0 = default; 0 =
    /// sink-fill only.
    #[arg(long, default_value_t = -1.0)]
    erosion_deposition_slope: f32,

    /// Lithologic erodibility contrast sigma. <0 = default; 0 = uniform K.
    #[arg(long, default_value_t = -1.0)]
    erosion_litho_sigma: f32,

    /// Structural-grain erodibility strength (fold-belt ridge-and-valley). <0 =
    /// default; 0 = off.
    #[arg(long, default_value_t = -1.0)]
    erosion_litho_grain: f32,

    /// Orographic precip modulation strength (windward wetter, lee drier). <0 =
    /// default; 0 = coarse precip.
    #[arg(long, default_value_t = -1.0)]
    erosion_orographic_strength: f32,

    /// Lakes-as-evaporation precip boost. <0 = default; 0 = off.
    #[arg(long, default_value_t = -1.0)]
    erosion_lake_evap: f32,

    /// Glacial abrasion coefficient (ice-flux over-deepening). <0 = default; 0 =
    /// no glacial pass.
    #[arg(long, default_value_t = -1.0)]
    glacial_k: f32,

    /// Fault range-front scarp relief. <0 = default; 0 = off (smooth fronts).
    #[arg(long, default_value_t = -1.0)]
    fault_scarp: f32,

    /// Fine interior structural relief amplitude (P1a: mid-band fault/fold grain
    /// that breaks the flat orogen summit). <0 = default; 0 = off (pure interpolant).
    #[arg(long, default_value_t = -1.0)]
    interior_relief: f32,

    /// Sweep mode: erosion knob to vary across columns (enables a headless
    /// render-to-PNG sweep). Knobs: k, diffusivity, channel_support,
    /// hillslope_crit, confinement_slope, uplift_smooth, mfd_exponent,
    /// diffusion_iters, reroute_interval, steps, precip_iters, flat_resolution.
    #[arg(long)]
    sweep: Option<String>,

    /// Comma-separated values for --sweep (e.g. "10,30,60,120").
    #[arg(long, default_value = "")]
    sweep_values: String,

    /// Optional second knob varied across rows (2-D grid).
    #[arg(long)]
    sweep2: Option<String>,

    /// Comma-separated values for --sweep2.
    #[arg(long, default_value = "")]
    sweep2_values: String,

    /// Output directory for sweep PNG tiles + montage.png.
    #[arg(long, default_value = "sweep_out")]
    out_dir: PathBuf,

    /// Sweep tile width in pixels.
    #[arg(long, default_value_t = 1024)]
    sweep_width: u32,

    /// Sweep tile height in pixels.
    #[arg(long, default_value_t = 1024)]
    sweep_height: u32,

    /// Sweep camera yaw in degrees (globe orbit).
    #[arg(long, default_value_t = 30.0)]
    sweep_yaw: f32,

    /// Sweep camera pitch in degrees (globe orbit).
    #[arg(long, default_value_t = 25.0)]
    sweep_pitch: f32,

    /// Sweep camera distance from globe center (overview).
    #[arg(long, default_value_t = 2.2)]
    sweep_distance: f32,

    /// Zoomed close-up views per tile (auto-aimed at the highest land), in
    /// addition to the globe overview. 0 = overview only.
    #[arg(long, default_value_t = 3)]
    sweep_zoom_views: usize,

    /// Close-up camera altitude above target (smaller = tighter zoom).
    #[arg(long, default_value_t = 0.3)]
    sweep_zoom_alt: f32,

    /// Rivers in sweep tiles: off, major, or all.
    #[arg(long, default_value = "major")]
    sweep_rivers: String,

    /// Legacy flag: equivalent to --stage 2
    #[arg(long, hide = true)]
    stage2: bool,
}

/// Parse a comma-separated list of f64 values, ignoring empty entries.
fn parse_values(s: &str) -> Vec<f64> {
    s.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<f64>()
                .unwrap_or_else(|_| panic!("invalid sweep value: '{t}'"))
        })
        .collect()
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
    let erosion = app::world::ErosionOverrides {
        mfd_exponent: (cli.erosion_mfd_exponent >= 0.0).then_some(cli.erosion_mfd_exponent),
        flat_resolution: (cli.erosion_flat_resolution >= 0)
            .then_some(cli.erosion_flat_resolution != 0),
        confinement_slope: (cli.erosion_confinement_slope >= 0.0)
            .then_some(cli.erosion_confinement_slope),
        k: (cli.erosion_k >= 0.0).then_some(cli.erosion_k),
        diffusivity: (cli.erosion_diffusivity >= 0.0).then_some(cli.erosion_diffusivity),
        channel_support_km2: (cli.erosion_channel_support >= 0.0)
            .then_some(cli.erosion_channel_support),
        uplift_smooth_km: (cli.erosion_uplift_smooth >= 0.0).then_some(cli.erosion_uplift_smooth),
        hillslope_critical_slope: (cli.erosion_hillslope_crit >= 0.0)
            .then_some(cli.erosion_hillslope_crit),
        diffusion_iters: (cli.erosion_diffusion_iters > 0).then_some(cli.erosion_diffusion_iters),
        reroute_interval: (cli.erosion_reroute_interval > 0)
            .then_some(cli.erosion_reroute_interval),
        steps: (cli.erosion_steps > 0).then_some(cli.erosion_steps),
        precip_outer_iters: (cli.erosion_precip_iters > 0).then_some(cli.erosion_precip_iters),
        uplift_scale: (cli.erosion_uplift_scale >= 0.0).then_some(cli.erosion_uplift_scale),
        deposition_slope: (cli.erosion_deposition_slope >= 0.0)
            .then_some(cli.erosion_deposition_slope),
        litho_sigma: (cli.erosion_litho_sigma >= 0.0).then_some(cli.erosion_litho_sigma),
        litho_grain_strength: (cli.erosion_litho_grain >= 0.0).then_some(cli.erosion_litho_grain),
        orographic_precip_strength: (cli.erosion_orographic_strength >= 0.0)
            .then_some(cli.erosion_orographic_strength),
        lake_evap_strength: (cli.erosion_lake_evap >= 0.0).then_some(cli.erosion_lake_evap),
        glacial_k: (cli.glacial_k >= 0.0).then_some(cli.glacial_k),
        fault_scarp_height: (cli.fault_scarp >= 0.0).then_some(cli.fault_scarp),
        interior_relief: (cli.interior_relief >= 0.0).then_some(cli.interior_relief),
    };
    let fine_cache = if cli.no_fine_cache {
        FineCacheMode::Disabled
    } else if cli.rebuild_fine_cache {
        FineCacheMode::Rebuild
    } else {
        FineCacheMode::Enabled
    };

    if let Some(knob1) = cli.sweep.clone() {
        let river_mode = match cli.sweep_rivers.as_str() {
            "off" => app::RiverMode::Off,
            "major" => app::RiverMode::Major,
            "all" => app::RiverMode::All,
            other => panic!("invalid --sweep-rivers '{other}'; use off, major, or all"),
        };
        let values1 = parse_values(&cli.sweep_values);
        if values1.is_empty() {
            panic!("--sweep requires --sweep-values (e.g. --sweep-values 10,30,60)");
        }
        let opts = app::sweep::SweepOptions {
            seed: cli.seed.unwrap_or_else(rand::random),
            cells: cli.cells,
            fine_scale: cli.fine_scale,
            // Sweeps are about erosion, so default to the erosion stage.
            target_stage: cli.stage.unwrap_or(4),
            voronoi_backend: backend,
            fine_cache,
            base_erosion: erosion,
            knob1,
            values1,
            knob2: cli.sweep2.clone(),
            values2: parse_values(&cli.sweep2_values),
            out_dir: cli.out_dir,
            width: cli.sweep_width,
            height: cli.sweep_height,
            yaw_deg: cli.sweep_yaw,
            pitch_deg: cli.sweep_pitch,
            distance: cli.sweep_distance,
            zoom_views: cli.sweep_zoom_views,
            zoom_alt: cli.sweep_zoom_alt,
            river_mode,
        };
        app::sweep::run_sweep(opts);
    } else if cli.headless {
        run_headless(
            cli.seed,
            cli.cells,
            cli.fine_scale,
            target_stage,
            cli.export,
            backend,
            fine_cache,
            erosion,
        );
    } else {
        run_interactive(
            cli.seed,
            target_stage,
            cli.export,
            backend,
            fine_cache,
            erosion,
        );
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
    erosion: app::world::ErosionOverrides,
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
    erosion.apply(&mut world);
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
    erosion: app::world::ErosionOverrides,
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
        erosion,
    };

    let mut app = app::App::new(config);
    event_loop
        .run_app(&mut app)
        .expect("Failed to run application");
}
