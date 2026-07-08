//! World diagnostics: generate a world and measure its features numerically.
//!
//! The primary instrument for judging physical plausibility of generated
//! artifacts — sizes in km/km² with Earth references alongside. Catches what
//! aggregate stats and rendered images both miss.
//!
//!     cargo run --release --bin diagnose -- --seed 12345
//!     cargo run --release --bin diagnose -- --seed 12345 --cells 40000

use clap::Parser;
use hex3::world::diagnostics::{
    distance_from_mask, measure_components, ComponentStats, EARTH_RADIUS_KM,
};
use hex3::world::{CellWaterState, World};

#[derive(Parser, Debug)]
#[command(name = "diagnose", about = "Measure generated world features")]
struct Cli {
    #[arg(long, default_value_t = 12345)]
    seed: u64,
    #[arg(long, default_value_t = 100_000)]
    cells: usize,
    /// Max components listed per section.
    #[arg(long, default_value_t = 8)]
    top: usize,
    /// Run the COARSE-vs-FINE drainage audit (endorheic land fraction, lake
    /// fraction, basin counts) and exit. Compares the freshly-computed coarse
    /// hydrology against the fine (eroded stage-4) hydrology.
    #[arg(long, default_value_t = false)]
    drainage_audit: bool,
    /// Run the per-lake FEATURE audit and exit: object-level lake statistics
    /// (size spectrum, shape, placement, outlet plumbing) with Earth references,
    /// plus the climate-dial response curve, for coarse + fine hydrology.
    /// A/B against the pre-integration world with HEX3_NO_DRAINAGE_INTEGRATION=1.
    #[arg(long, default_value_t = false)]
    lake_audit: bool,
    /// Run the per-range MOUNTAIN audit and exit: range objects on the fine
    /// eroded surface (size/elongation, crest-pass spectrum, cross-strike
    /// asymmetry, sampled local relief) with Earth references.
    #[arg(long, default_value_t = false)]
    mountain_audit: bool,
    /// Run the RIVER NETWORK audit and exit: the rendered river network's
    /// Strahler/Horton structure, drainage density, mouths, and per-trunk
    /// length/sinuosity/profile table with Earth references.
    #[arg(long, default_value_t = false)]
    river_audit: bool,
    /// Fine-mesh cell cap (the emergent count is coarsened to fit). Lower it to
    /// iterate faster on erosion/roughness probes. 0 = use the FINE_MAX_CELLS default.
    #[arg(long, default_value_t = 0)]
    fine_max: usize,
    /// Override erosion erodibility K. <0 = use the EROSION_K default. Lets you
    /// sweep erosion strength without a recompile (FineBase is cached).
    #[arg(long, default_value_t = -1.0)]
    erosion_k: f32,
    /// Override stream-power SLOPE exponent n (E=K·A^m·S^n). <0 = EROSION_N default (1);
    /// >1 (≈1.5–2) = sharper valleys/divides. Newton-solved.
    #[arg(long, default_value_t = -1.0)]
    erosion_n: f32,
    /// Override erosion step count. 0 = use the EROSION_STEPS default.
    #[arg(long, default_value_t = 0)]
    erosion_steps: usize,
    /// Override hillslope diffusivity. <0 = use the EROSION_DIFFUSIVITY default.
    #[arg(long, default_value_t = -1.0)]
    erosion_diffusivity: f32,
    /// Override channel-initiation support area (km² at mean land wetness).
    /// <0 = use the EROSION_CHANNEL_SUPPORT_KM2 default; 0 = disable the
    /// threshold (incise wherever downhill).
    #[arg(long, default_value_t = -1.0)]
    erosion_channel_support: f32,
    /// Override Jacobi sweeps per implicit diffusion solve. 0 = use the
    /// EROSION_DIFFUSION_ITERS default. Sweep to check the finest cells aren't
    /// under-converged (speckle).
    #[arg(long, default_value_t = 0)]
    erosion_diffusion_iters: usize,
    /// Override the drainage re-route interval (steps between re-routings).
    /// 0 = use the EROSION_REROUTE_INTERVAL default; 1 = re-route every step.
    /// Set to 1 to test whether stale routing is driving the spiral/perforation
    /// artifacts (see docs/specs/erosion.md "Roughness counters").
    #[arg(long, default_value_t = 0)]
    erosion_reroute_interval: usize,
    /// Barnes convergent flat resolution (Rung 1). -1 = use the
    /// EROSION_FLAT_RESOLUTION default; 0 = off (old flood_parent wavefront);
    /// 1 = on. A/B the spiral-on-flats fix (docs/specs/erosion-routing-ladder.md).
    #[arg(long, default_value_t = -1)]
    erosion_flat_resolution: i8,
    /// MFD drainage-area exponent (Rung 2). <0 = use EROSION_MFD_EXPONENT; 0 =
    /// single-flow (SFD); higher = sharper (p→∞ ≈ SFD), lower ≈ 1 = dispersive.
    /// Sweep to A/B multi-flow vs single-flow discharge.
    #[arg(long, default_value_t = -1.0)]
    erosion_mfd_exponent: f32,
    /// Plains alluvial regime gate: channel slope (elev/km) at/above which
    /// incision is full; gentler channels fade to alluvial. <0 = use
    /// EROSION_CONFINEMENT_SLOPE; 0 = off. See docs/specs/erosion-valleys-not-channels.md.
    #[arg(long, default_value_t = -1.0)]
    erosion_confinement_slope: f32,
    /// Override lithologic erodibility contrast (exp-amplitude sigma). <0 = use
    /// the EROSION_LITHO_SIGMA default; 0 = uniform K (no lithology).
    #[arg(long, default_value_t = -1.0)]
    erosion_litho_sigma: f32,
    /// Override tectonic uplift scale (Hold & carve: uplift ~balances erosion to
    /// hold orogens while valleys grade). <0 = use the EROSION_UPLIFT_SCALE
    /// default; 0 = relaxation only (no uplift).
    #[arg(long, default_value_t = -1.0)]
    erosion_uplift_scale: f32,
    /// Emergent builder over-rebuild gain (relief-spectrum candidate B): >1 builds
    /// more orogen volume than the coarse target so erosion carves the excess into
    /// relief. <0 = EMERGENT_REBUILD_GAIN default (1.2). Pair with --erosion-k /
    /// --erosion-hillslope-crit for the joint high-relief regime.
    #[arg(long, default_value_t = -1.0)]
    rebuild_gain: f32,
    /// Override uplift-FORCING smoothing length (km). Escalation #1: smooths the
    /// per-step uplift source over a sub-grid orogenic width to kill mountain-top
    /// cell-scale chatter without flattening orogens. <0 = use
    /// EROSION_UPLIFT_SMOOTH_KM default; 0 = off. See
    /// docs/specs/erosion-uplift-smoothing.md.
    #[arg(long, default_value_t = -1.0)]
    erosion_uplift_smooth: f32,
    /// Override Roering nonlinear-hillslope critical slope S_c (escalation #2;
    /// Δelev/radian, ~grade·637). Diffusivity blows up toward S_c -> planar
    /// slopes + crisp ridges. <0 = use EROSION_HILLSLOPE_CRITICAL_SLOPE default;
    /// 0 = off (linear creep). Read curv-rms/peak% as you sweep ~150-300.
    #[arg(long, default_value_t = -1.0)]
    erosion_hillslope_crit: f32,
    /// Override orographic precip modulation strength (climate↔erosion: windward
    /// wetter, lee drier). <0 = use OROGRAPHIC_PRECIP_STRENGTH; 0 = coarse precip.
    #[arg(long, default_value_t = -1.0)]
    erosion_orographic_strength: f32,
    /// Override downwind rain-shadow strength (lee dry-anomaly propagated downwind).
    /// <0 = use DOWNWIND_SHADOW_STRENGTH (0=off).
    #[arg(long, default_value_t = -1.0)]
    erosion_downwind_shadow: f32,
    /// Override coupled erode↔precip feedback passes. 0 = use
    /// EROSION_PRECIP_OUTER_ITERS default; 1 = no erosion feedback.
    #[arg(long, default_value_t = 0)]
    erosion_precip_iters: usize,
    /// Override lakes-as-evaporation precip boost strength. <0 = use
    /// LAKE_EVAP_STRENGTH default; 0 = off.
    #[arg(long, default_value_t = -1.0)]
    erosion_lake_evap: f32,
    /// Override depositional repose slope (en-route aggradation: fans/floodplains/
    /// deltas). <0 = use EROSION_DEPOSITION_SLOPE default; 0 = sink-fill only.
    /// Read the mass ledger (lost-to-ocean should drop) as you raise it.
    #[arg(long, default_value_t = -1.0)]
    erosion_deposition_slope: f32,
    /// Override glacial abrasion coefficient (ice-flux over-deepening). <0 = use
    /// GLACIAL_K default; 0 = no glacial pass. Read the logged glaciated coverage
    /// and abraded volume as you sweep it.
    #[arg(long, default_value_t = -1.0)]
    glacial_k: f32,
    /// Override glacial over-deepening max (reverse-gradient/tarn depth a cell may
    /// carve below its receiver). <0 = use GLACIAL_OVERDEEPEN_MAX default; 0 =
    /// no over-deepening (no closed rock basins). Isolates over-deepening from
    /// SFD ice abrasion as the curv-rms/pit source.
    #[arg(long, default_value_t = -1.0)]
    glacial_overdeepen_max: f32,
    /// Override structural-grain erodibility strength (fold-belt ridge-and-valley).
    /// <0 = use EROSION_LITHO_GRAIN_STRENGTH default; 0 = no grain. Experimental.
    #[arg(long, default_value_t = -1.0)]
    litho_grain_strength: f32,
    /// Override fault range-front scarp relief (sharpen active orogen margins).
    /// <0 = use FAULT_SCARP_HEIGHT default; 0 = off (smooth fronts).
    #[arg(long, default_value_t = -1.0)]
    fault_scarp: f32,
    /// Override fine interior structural relief amplitude (P1a: mid-band fault/fold
    /// grain that breaks the flat orogen summit). <0 = use FINE_INTERIOR_RELIEF
    /// default; 0 = off (pure interpolant). Regenerates the fine base.
    #[arg(long, default_value_t = -1.0)]
    interior_relief: f32,
    /// Override fine strike-band weight (P1b: how much of the interior grain aligns
    /// to the nearest orogen front vs isotropic). <0 = use FINE_FRONT_STRIKE_WEIGHT
    /// default; 0 = isotropic (P1a). Regenerates the fine base.
    #[arg(long, default_value_t = -1.0)]
    front_strike_weight: f32,
    /// Override fine margin contrast (P1c: sharpen relief on active/convergent coasts,
    /// damp on passive ones). <0 = use FINE_MARGIN_CONTRAST default; 0 = off (P1b).
    /// Regenerates the fine base.
    #[arg(long, default_value_t = -1.0)]
    margin_contrast: f32,
    /// Emergent-orogens demotion fraction (erosion-v3): demote λ·(arc+collision) from
    /// the base and rebuild it by active uplift. <0 = default (0=off); 0.25-0.5 = test.
    /// Pair with a raised --erosion-uplift-scale (~λ/(steps·dt)). NOTE: the painted P1
    /// relief still runs unless you also zero it (--interior-relief 0.005 --fault-scarp 0
    /// etc.) — the `--sweep-stack v3` preset does this for a clean A/B. Regenerates base.
    #[arg(long, default_value_t = -1.0)]
    emergent_lambda: f32,
    /// O0 structured emergent uplift (orogen-structure): blend of asymmetric+segmented
    /// uplift shape vs uniform rebuild. <0 = default (0=off); 1 = fully structured.
    /// Needs --emergent-lambda >0 and (for the decisive test) --erosion-n ~2.
    #[arg(long, default_value_t = -1.0)]
    emergent_structured: f32,
    /// Candidate A meso uplift-shape modulation depth. <0 = default (0=off);
    /// 0 = off; 0.3-0.9 = test. Regenerates the fine base.
    #[arg(long, default_value_t = -1.0)]
    meso_relief: f32,
    /// Candidate A' meso base-elevation relief amplitude. <0 = default (0=off);
    /// elevation units: 0.01 is about 100 m. Regenerates the fine base.
    #[arg(long, default_value_t = -1.0)]
    meso_base_relief: f32,
    /// Candidate A meso fold-train wavelength in km. <0 = default (25 km).
    /// Regenerates the fine base.
    #[arg(long, default_value_t = -1.0)]
    meso_wavelength_km: f32,
    /// Fine-mesh density knobs (cell-size targets in km / blend). <0 = use the
    /// FINE_* default. Setting any forces fine-base regeneration (no cache). Use
    /// to sweep the ocean/plains/mountain budget: e.g. --fine-plains-km 20.
    #[arg(long, default_value_t = -1.0)]
    fine_plains_km: f32,
    #[arg(long, default_value_t = -1.0)]
    fine_mountain_km: f32,
    #[arg(long, default_value_t = -1.0)]
    fine_ocean_km: f32,
    #[arg(long, default_value_t = -1.0)]
    fine_density_exponent: f32,
    #[arg(long, default_value_t = -1.0)]
    fine_slope_weight: f32,
    #[arg(long, default_value_t = -1.0)]
    fine_flow_weight: f32,
    #[arg(long, default_value_t = -1.0)]
    fine_activity_weight: f32,
    /// Uniform multiplier on the fine-mesh cell-size targets (plains/mountain/
    /// ocean km) — one knob to sweep fine-mesh resolution. >1 coarsens, <1
    /// refines. Forces fine-base regeneration (no cache). Use with the
    /// "Fine-scale local relief" convergence probe.
    #[arg(long, default_value_t = 1.0)]
    fine_scale: f32,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    eprintln!(
        "Generating world (seed={}, cells={})...",
        cli.seed, cli.cells
    );
    let mut world = World::new(cli.seed, cli.cells, 1);
    world.generate_plates(hex3::world::NUM_PLATES_DEFAULT);
    world.generate_crust();
    world.generate_dynamics();
    world.generate_features();
    world.generate_elevation();
    world.generate_atmosphere();
    if cli.erosion_k >= 0.0 {
        world.erosion_params.k = cli.erosion_k;
    }
    if cli.erosion_n >= 0.0 {
        world.erosion_params.n = cli.erosion_n;
    }
    if cli.erosion_steps > 0 {
        world.erosion_params.steps = cli.erosion_steps;
    }
    if cli.erosion_diffusivity >= 0.0 {
        world.erosion_params.diffusivity = cli.erosion_diffusivity;
    }
    if cli.erosion_channel_support >= 0.0 {
        world.erosion_params.channel_support_km2 = cli.erosion_channel_support;
    }
    if cli.erosion_diffusion_iters > 0 {
        world.erosion_params.diffusion_iters = cli.erosion_diffusion_iters;
    }
    if cli.erosion_reroute_interval > 0 {
        world.erosion_params.reroute_interval = cli.erosion_reroute_interval;
    }
    if cli.erosion_flat_resolution >= 0 {
        world.erosion_params.flat_resolution = cli.erosion_flat_resolution != 0;
    }
    if cli.erosion_mfd_exponent >= 0.0 {
        world.erosion_params.mfd_exponent = cli.erosion_mfd_exponent;
    }
    if cli.erosion_confinement_slope >= 0.0 {
        world.erosion_params.confinement_slope = cli.erosion_confinement_slope;
    }
    if cli.erosion_litho_sigma >= 0.0 {
        world.erosion_params.litho_sigma = cli.erosion_litho_sigma;
    }
    if cli.erosion_uplift_scale >= 0.0 {
        world.erosion_params.uplift_scale = cli.erosion_uplift_scale;
    }
    if cli.rebuild_gain >= 0.0 {
        world.erosion_params.rebuild_gain = cli.rebuild_gain;
    }
    if cli.erosion_uplift_smooth >= 0.0 {
        world.erosion_params.uplift_smooth_km = cli.erosion_uplift_smooth;
    }
    if cli.erosion_hillslope_crit >= 0.0 {
        world.erosion_params.hillslope_critical_slope = cli.erosion_hillslope_crit;
    }
    if cli.erosion_orographic_strength >= 0.0 {
        world.erosion_params.orographic_precip_strength = cli.erosion_orographic_strength;
    }
    if cli.erosion_downwind_shadow >= 0.0 {
        world.erosion_params.downwind_shadow_strength = cli.erosion_downwind_shadow;
    }
    if cli.erosion_precip_iters > 0 {
        world.erosion_params.precip_outer_iters = cli.erosion_precip_iters;
    }
    if cli.erosion_lake_evap >= 0.0 {
        world.erosion_params.lake_evap_strength = cli.erosion_lake_evap;
    }
    if cli.erosion_deposition_slope >= 0.0 {
        world.erosion_params.deposition_slope = cli.erosion_deposition_slope;
    }
    if cli.glacial_overdeepen_max >= 0.0 {
        world.erosion_params.glacial_overdeepen_max = cli.glacial_overdeepen_max;
    }
    if cli.glacial_k >= 0.0 {
        world.erosion_params.glacial_k = cli.glacial_k;
    }
    if cli.litho_grain_strength >= 0.0 {
        world.erosion_params.litho_grain_strength = cli.litho_grain_strength;
    }
    if cli.fault_scarp >= 0.0 {
        world.fine_structure_params.fault_scarp_height = cli.fault_scarp;
    }
    if cli.interior_relief >= 0.0 {
        world.fine_structure_params.interior_relief = cli.interior_relief;
    }
    if cli.front_strike_weight >= 0.0 {
        world.fine_structure_params.front_strike_weight = cli.front_strike_weight;
    }
    if cli.margin_contrast >= 0.0 {
        world.fine_structure_params.margin_contrast = cli.margin_contrast;
    }
    if cli.emergent_lambda >= 0.0 {
        world.fine_structure_params.emergent_lambda = cli.emergent_lambda;
    }
    if cli.emergent_structured >= 0.0 {
        world.fine_structure_params.emergent_structured = cli.emergent_structured;
    }
    if cli.meso_relief >= 0.0 {
        world.fine_structure_params.meso_relief = cli.meso_relief;
    }
    if cli.meso_base_relief >= 0.0 {
        world.fine_structure_params.meso_base_relief = cli.meso_base_relief;
    }
    if cli.meso_wavelength_km >= 0.0 {
        world.fine_structure_params.meso_wavelength_km = cli.meso_wavelength_km;
    }
    // Fine-mesh density overrides (force regeneration so the cache can't serve a
    // base built with the default knobs).
    let mut dp = world.fine_density_params;
    let overrides = [
        (&mut dp.plains_km, cli.fine_plains_km),
        (&mut dp.mountain_km, cli.fine_mountain_km),
        (&mut dp.ocean_km, cli.fine_ocean_km),
        (&mut dp.exponent, cli.fine_density_exponent),
        (&mut dp.slope_weight, cli.fine_slope_weight),
        (&mut dp.flow_weight, cli.fine_flow_weight),
        (&mut dp.activity_weight, cli.fine_activity_weight),
    ];
    let mut density_overridden = false;
    for (target, v) in overrides {
        if v >= 0.0 {
            *target = v;
            density_overridden = true;
        }
    }
    // Uniform fine-resolution multiplier, applied after any explicit per-class
    // overrides so it scales whatever budget is in effect.
    if (cli.fine_scale - 1.0).abs() > f32::EPSILON {
        dp.plains_km *= cli.fine_scale;
        dp.mountain_km *= cli.fine_scale;
        dp.ocean_km *= cli.fine_scale;
        density_overridden = true;
    }
    if density_overridden {
        world.fine_density_params = dp;
        world.fine_cache = hex3::world::FineCacheMode::Disabled;
        eprintln!(
            "fine density override: plains={:.1} mountain={:.1} ocean={:.1} km, exp={:.1}, weights slope/flow/activity={:.1}/{:.1}/{:.1}",
            dp.plains_km, dp.mountain_km, dp.ocean_km, dp.exponent, dp.slope_weight, dp.flow_weight, dp.activity_weight
        );
    }

    if cli.fine_max > 0 {
        world.generate_hydrology_with_fine_cap(cli.fine_max);
    } else {
        world.generate_hydrology();
    }

    // Feature audits are combinable (one world generation, several panels).
    let mut audited = false;
    if cli.drainage_audit {
        run_drainage_audit(&world, cli.seed);
        audited = true;
    }
    if cli.lake_audit {
        run_lake_audit(&mut world, cli.seed, cli.top);
        audited = true;
    }
    if cli.mountain_audit {
        run_mountain_audit(&world, cli.seed, cli.top);
        audited = true;
    }
    if cli.river_audit {
        run_river_audit(&world, cli.seed, cli.top);
        audited = true;
    }
    if audited {
        return;
    }

    let tess = world.active_tessellation();
    let n = tess.num_cells();
    let crust = world.crust.as_ref().unwrap();
    let features = world.features.as_ref().unwrap();
    let elevation = &world.active_elevation().unwrap().values;
    let hydrology = world.active_hydrology().unwrap();
    let temperature = world.active_temperature().unwrap();
    let precipitation = world.active_precipitation().unwrap();
    let uplift = world.active_uplift().unwrap();
    let coarse_of = |i: usize| -> usize {
        world
            .fine
            .as_ref()
            .map(|fine| fine.coarse_cell()[i])
            .unwrap_or(i)
    };
    let cont_mask: Vec<bool> = (0..n)
        .map(|i| {
            world
                .fine
                .as_ref()
                .map(|fine| fine.fields().elevation_fields.continentality[i] >= 0.5)
                .unwrap_or_else(|| crust.is_continental(i))
        })
        .collect();
    let margin_distance: Vec<f32> = (0..n)
        .map(|i| {
            world
                .fine
                .as_ref()
                .map(|fine| {
                    let cont = fine.fields().elevation_fields.continentality[i];
                    (cont - 0.5).abs() * 0.1
                })
                .unwrap_or_else(|| crust.margin_distance(i))
        })
        .collect();
    let feature_divergent: Vec<f32> = (0..n).map(|i| features.divergent[coarse_of(i)]).collect();
    let feature_collision: Vec<f32> = (0..n).map(|i| features.collision[coarse_of(i)]).collect();
    let feature_arc: Vec<f32> = (0..n).map(|i| features.arc[coarse_of(i)]).collect();
    let feature_trench: Vec<f32> = world
        .fine
        .as_ref()
        .map(|fine| fine.fields().elevation_fields.trench.clone())
        .unwrap_or_else(|| features.trench.clone());
    let feature_ridge_age: Vec<f32> = world
        .fine
        .as_ref()
        .map(|fine| fine.fields().elevation_fields.ridge_age_distance.clone())
        .unwrap_or_else(|| features.ridge_age_distance.clone());

    let cell_km2 = tess.mean_cell_area() * EARTH_RADIUS_KM * EARTH_RADIUS_KM;
    println!(
        "\n================ WORLD DIAGNOSTICS seed={} cells={} ================",
        cli.seed, n
    );
    println!(
        "resolution: {:.0} km²/cell (~{:.0} km spacing) — features below ~{:.0} km² are unresolvable",
        cell_km2,
        (tess.mean_cell_area()).sqrt() * EARTH_RADIUS_KM,
        cell_km2
    );
    if let Some(fine) = &world.fine {
        println!(
            "fine mesh: coarse {} -> fine {} cells | density ratio {:.1}:1",
            world.tessellation.num_cells(),
            fine.tessellation().num_cells(),
            fine.achieved_density_ratio()
        );
        // ---- Erosion roughness counters (artifact vs. genuine dissection) ----
        // pit% is the swiss-cheese meter (a drained surface is ~0); checkerboard%
        // the SFD-groove banding; aspect R/entropy catch GLOBAL anisotropy / mesh-
        // axis locking only (a local spiral with balanced azimuths stays hidden —
        // judge spirals on the map). Pre-erosion (stage 3) vs eroded (stage 4) on
        // the SAME fine mesh. NOTE: the land mask is recomputed per surface, so the
        // `land` column shifts if erosion moves cells across sea level — a falling
        // pit% with a falling `land` may be submergence, not artifact removal.
        // See docs/specs/erosion.md.
        let ftess = fine.tessellation();
        let pre = hex3::world::roughness_counters(ftess, &fine.surface_for(3).elevation.values);
        let ero = hex3::world::roughness_counters(ftess, &fine.surface_for(4).elevation.values);
        println!(
            "\n-- Erosion roughness counters (reroute-interval {})  [pit% is the swiss-cheese meter; lower better] --",
            world.erosion_params.reroute_interval
        );
        println!(
            "             {:>9} {:>10} {:>10} {:>10} {:>12} {:>9} {:>9}",
            "land", "pit%", "peak%", "checker%", "curv-rms", "aspectR", "entropy"
        );
        let row = |label: &str, c: &hex3::world::RoughnessCounters| {
            println!(
                "  {:<8}   {:>9} {:>10.3} {:>10.3} {:>10.2} {:>12.3e} {:>9.3} {:>9.3}",
                label,
                c.land,
                c.pit_pct,
                c.peak_pct,
                c.checkerboard_pct,
                c.curv_rms,
                c.aspect_r,
                c.aspect_entropy
            );
        };
        row("pre", &pre);
        row("eroded", &ero);
        println!(
            "  {:<8}   {:>+9} {:>+10.3} {:>+10.3} {:>+10.2} {:>12} {:>+9.3} {:>+9.3}",
            "delta",
            ero.land as i64 - pre.land as i64,
            ero.pit_pct - pre.pit_pct,
            ero.peak_pct - pre.peak_pct,
            ero.checkerboard_pct - pre.checkerboard_pct,
            "",
            ero.aspect_r - pre.aspect_r,
            ero.aspect_entropy - pre.aspect_entropy,
        );

        // ---- Mountain-top plateau probe (the localized "cottage cheese on flat
        // summits" artifact). Global counters are blind to a summit-only pattern,
        // so restrict to the highest land (top elevation decile) and compare
        // pre-erosion vs eroded THERE. Also report summit max-downhill slope: low
        // slope = genuinely flat-topped (plateau). If the texture is already in
        // `pre`, it's the fine-base synthesis (interp + noise), not erosion/routing.
        {
            let pre_e = &fine.surface_for(3).elevation.values;
            let ero_e = &fine.surface_for(4).elevation.values;
            let mut land_el: Vec<f32> = ero_e.iter().copied().filter(|&e| e >= 0.0).collect();
            land_el.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let thr = land_el[((land_el.len() as f32 * 0.90) as usize).min(land_el.len() - 1)];
            let mask = |src: &[f32]| -> Vec<f32> {
                (0..ftess.num_cells())
                    .map(|i| if ero_e[i] >= thr { src[i] } else { -1.0 })
                    .collect()
            };
            let pre_top = hex3::world::roughness_counters(ftess, &mask(pre_e));
            let ero_top = hex3::world::roughness_counters(ftess, &mask(ero_e));
            // Summit max-downhill slope (elev/km), pre and eroded, for flatness.
            let summit_slopes = |elev: &[f32]| -> Vec<f32> {
                let mut s = Vec::new();
                for i in 0..ftess.num_cells() {
                    if ero_e[i] < thr {
                        continue;
                    }
                    let ci = ftess.cell_center(i);
                    let mut g = 0.0f32;
                    for &nb in ftess.neighbors(i) {
                        let d = (ci - ftess.cell_center(nb)).length().max(1e-9) * EARTH_RADIUS_KM;
                        g = g.max((elev[i] - elev[nb]) / d);
                    }
                    s.push(g);
                }
                s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                s
            };
            let pct = |v: &[f32], p: f32| v[(((v.len() as f32) * p) as usize).min(v.len() - 1)];
            let (sp, se) = (summit_slopes(pre_e), summit_slopes(ero_e));
            println!(
                "\n-- Mountain-top plateau probe (top elev decile, elev>={:.3}) --",
                thr
            );
            row("top pre", &pre_top);
            row("top eroded", &ero_top);
            println!(
                "  summit max-downhill slope (elev/km): pre p50 {:.3e} p90 {:.3e} | eroded p50 {:.3e} p90 {:.3e}  (global land slope p50 ~3.8e-4; << = flat plateau)",
                pct(&sp, 0.5),
                pct(&sp, 0.9),
                pct(&se, 0.5),
                pct(&se, 0.9),
            );
        }

        // ---- Carved dissection density (the "too busy" calibration target) ----
        // Incision = pre-erosion base minus eroded (meters). A "carved" cell is
        // incised past a depth threshold (a real valley, not surface noise); the
        // density of carved cells (channel km / km² land) and spacing ~1/Dd index
        // how busy the dissection is — and UNLIKE the hydrology network this
        // responds to channel-support / K. Resolution + threshold dependent, so
        // read vs Earth as a ballpark + a knob-response signal, not a hard number.
        const M_PER_UNIT: f32 = 10_000.0; // ~10 km vertical per elev-unit
        let base = &fine.surface_for(3).elevation.values;
        let eroded = &fine.surface_for(4).elevation.values;
        let fareas = ftess.cell_areas();
        let mut land_area_km2 = 0.0f64;
        let mut depths: Vec<f32> = Vec::new(); // land incision depth (m, clamped >=0)
        let mut carved: Vec<(f32, f32)> = Vec::new(); // (depth_m, width_km) for incised
        for i in 0..ftess.num_cells() {
            if eroded[i] < 0.0 {
                continue; // ocean
            }
            land_area_km2 += (fareas[i] * EARTH_RADIUS_KM * EARTH_RADIUS_KM) as f64;
            let inc = (base[i] - eroded[i]) * M_PER_UNIT;
            depths.push(inc.max(0.0));
            if inc > 0.0 {
                carved.push((inc, fareas[i].sqrt() * EARTH_RADIUS_KM));
            }
        }
        depths.sort_by(f32::total_cmp);
        let nland = depths.len().max(1);
        let dq = |p: f32| depths[(((nland - 1) as f32) * p) as usize];
        println!(
            "\n-- Carved dissection (incision = base - eroded)  [Earth land Dd ~0.5-5 km/km², spacing ~0.2-2 km] --"
        );
        println!(
            "  incision depth (land): p50 {:>4.0} m | p90 {:>4.0} m | p99 {:>5.0} m | max {:>5.0} m",
            dq(0.50),
            dq(0.90),
            dq(0.99),
            dq(1.0),
        );
        for thr in [30.0f32, 100.0, 300.0] {
            let len_km: f64 = carved
                .iter()
                .filter(|&&(d, _)| d >= thr)
                .map(|&(_, w)| w as f64)
                .sum();
            let cnt = carved.iter().filter(|&&(d, _)| d >= thr).count();
            let dd = if land_area_km2 > 0.0 {
                len_km / land_area_km2
            } else {
                0.0
            };
            let spacing = if dd > 0.0 { 1.0 / dd } else { f64::INFINITY };
            println!(
                "  carved >{:>3.0} m: {:>5.1}% of land | Dd ~{:.3} km/km² | spacing ~{:.2} km",
                thr,
                100.0 * cnt as f32 / nland as f32,
                dd,
                spacing,
            );
        }
    }

    // ---- Global elevation structure ----
    let land: Vec<bool> = elevation.iter().map(|&e| e >= 0.0).collect();
    let land_frac = land.iter().filter(|&&l| l).count() as f32 / n as f32;
    let mut sorted: Vec<f32> = elevation.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f32| sorted[((p * (n - 1) as f32) as usize).min(n - 1)];
    println!("\n-- Global --");
    println!(
        "land {:.1}% | elevation p5 {:+.3} p50 {:+.3} p95 {:+.3} max {:+.3}",
        100.0 * land_frac,
        pct(0.05),
        pct(0.50),
        pct(0.95),
        sorted[n - 1]
    );
    println!(
        "field smoothness (Moran's I): elevation {:.3}, precipitation {:.3}, uplift {:.3}",
        tess.morans_i(elevation),
        tess.morans_i(precipitation),
        tess.morans_i(uplift)
    );
    let land_temps: Vec<f32> = (0..n)
        .filter(|&i| elevation[i] >= 0.0)
        .map(|i| temperature[i])
        .collect();
    let ocean_temps: Vec<f32> = (0..n)
        .filter(|&i| elevation[i] < 0.0)
        .map(|i| temperature[i])
        .collect();
    let mean = |values: &[f32]| {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f32>() / values.len() as f32
        }
    };
    let land_temp_mean = mean(&land_temps);
    let ocean_temp_mean = mean(&ocean_temps);
    println!(
        "thermal contrast: land mean {:+.3}, ocean mean {:+.3}, delta {:+.3}",
        land_temp_mean,
        ocean_temp_mean,
        land_temp_mean - ocean_temp_mean
    );

    // ---- Continents (connected continental crust) ----
    let continents = measure_components(tess, &cont_mask);
    println!("\n-- Continents (connected continental crust) --   [Earth: Afro-Eurasia 85M km², Americas 42M, Australia 7.7M, Greenland 2.1M]");
    for c in continents.iter().take(cli.top) {
        let submerged = c.fraction_where(|i| elevation[i] < 0.0);
        println!(
            "  {:>10.0} km²  extent {:>5.0} km  submerged {:>4.1}%",
            c.area_km2,
            c.length_km,
            100.0 * submerged
        );
    }
    println!("  ({} total)", continents.len());

    // ---- Interior continental water (rift seaways, inland seas, lakes) ----
    let interior_water: Vec<bool> = (0..n)
        .map(|i| {
            cont_mask[i]
                && margin_distance[i] > 0.03
                && matches!(
                    hydrology.water_state(i),
                    CellWaterState::Ocean | CellWaterState::LakeWater
                )
        })
        .collect();
    let waters = measure_components(tess, &interior_water);
    println!("\n-- Interior continental water (>~190km from margin) --   [Earth: Caspian 371k km², Black Sea 436k, Red Sea 438k (2250x355 km), Tanganyika 33k (670x50), Baikal 32k]");
    for w in waters.iter().take(cli.top) {
        let connected_to_ocean =
            w.fraction_where(|i| hydrology.water_state(i) == CellWaterState::Ocean);
        let mean_div = w.mean_of(&feature_divergent);
        let (min_e, _) = w.range_of(elevation);
        println!(
            "  {:>9.0} km²  {:>5.0} x {:>4.0} km (elong {:>4.1})  depth_max {:>5.2}  divergent {:.2}  {}",
            w.area_km2,
            w.length_km,
            w.width_km,
            w.elongation(),
            -min_e,
            mean_div,
            if connected_to_ocean > 0.5 { "SEAWAY (ocean-connected)" } else { "landlocked" }
        );
    }
    let total_water_km2: f32 = waters.iter().map(|w| w.area_km2).sum();
    println!(
        "  ({} bodies, {:.0} km² total = {:.1}% of continental crust)",
        waters.len(),
        total_water_km2,
        100.0 * total_water_km2 / (continents.iter().map(|c| c.area_km2).sum::<f32>()).max(1.0)
    );

    // ---- Islands (oceanic-crust land) ----
    let island_mask: Vec<bool> = (0..n)
        .map(|i| !cont_mask[i] && elevation[i] >= 0.0)
        .collect();
    let islands = measure_components(tess, &island_mask);
    println!("\n-- Islands (oceanic-crust land) --   [Earth: Greenland 2.1M km² (continental), Honshu 228k, Iceland 103k, Hawaii Big Island 10k]");
    for isl in islands.iter().take(cli.top.min(5)) {
        let (_, max_e) = isl.range_of(elevation);
        println!(
            "  {:>8.0} km²  extent {:>5.0} km  peak {:+.2}",
            isl.area_km2, isl.length_km, max_e
        );
    }
    let single_cell = islands.iter().filter(|i| i.cells.len() == 1).count();
    println!(
        "  ({} islands, {} of them single-cell)",
        islands.len(),
        single_cell
    );

    // ---- Mountain ranges (land above threshold) ----
    const RANGE_ELEV: f32 = 0.15;
    let range_mask: Vec<bool> = (0..n).map(|i| elevation[i] >= RANGE_ELEV).collect();
    let ranges = measure_components(tess, &range_mask);
    println!("\n-- Mountain ranges (elevation > {RANGE_ELEV}) --   [Earth: Andes 7000x300 km, Himalaya+Tibet 2400x1000, Alps 1200x200]");
    for r in ranges.iter().take(cli.top) {
        let (_, peak) = r.range_of(elevation);
        let collision_frac = r.fraction_where(|i| feature_collision[i] > 0.02);
        let arc_frac = r.fraction_where(|i| feature_arc[i] > 0.02);
        println!(
            "  {:>8.0} km²  {:>5.0} x {:>4.0} km  peak {:+.2}  driver: collision {:>3.0}% arc {:>3.0}%",
            r.area_km2,
            r.length_km,
            r.width_km,
            peak,
            100.0 * collision_frac,
            100.0 * arc_frac
        );
    }
    println!("  ({} ranges)", ranges.len());

    // ---- Arc-trench gap ----
    let trench_peak = feature_trench.iter().cloned().fold(0.0f32, f32::max);
    let arc_peak = feature_arc.iter().cloned().fold(0.0f32, f32::max);
    if trench_peak > 0.0 && arc_peak > 0.0 {
        let trench_mask: Vec<bool> = feature_trench
            .iter()
            .map(|&t| t > 0.3 * trench_peak)
            .collect();
        let dist = distance_from_mask(tess, &trench_mask);
        let mut gaps: Vec<f32> = (0..n)
            .filter(|&i| feature_arc[i] > 0.5 * arc_peak && dist[i].is_finite())
            .map(|i| dist[i] * EARTH_RADIUS_KM)
            .collect();
        if !gaps.is_empty() {
            gaps.sort_by(|a, b| a.total_cmp(b));
            println!(
                "\n-- Arc-trench gap (arc crest cells to nearest trench) --   [Earth: 100-250 km]"
            );
            println!(
                "  p25 {:>4.0} km  p50 {:>4.0} km  p75 {:>4.0} km",
                gaps[gaps.len() / 4],
                gaps[gaps.len() / 2],
                gaps[3 * gaps.len() / 4]
            );
        }
    }

    // ---- Flexure profile ----
    let deepest_deflection = feature_trench.iter().cloned().fold(0.0f32, f32::max);
    let strongest_outer_rise = -feature_trench.iter().cloned().fold(0.0f32, f32::min);
    let flexure_ratio = if deepest_deflection > 0.0 {
        strongest_outer_rise / deepest_deflection
    } else {
        0.0
    };
    let outer_rise_cells = feature_trench.iter().filter(|&&t| t < 0.0).count();
    println!(
        "\n-- Flexure profile --   [Earth: outer rise ~200-500 m vs trenches 2-8 km -> ~0.05]"
    );
    println!(
        "  deepest deflection {:.3} | strongest outer rise {:.3} | ratio {:.3} | outer-rise cells {}",
        deepest_deflection, strongest_outer_rise, flexure_ratio, outer_rise_cells
    );
    let mut ridge_age: Vec<f32> = feature_ridge_age
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .collect();
    if !ridge_age.is_empty() {
        ridge_age.sort_by(|a, b| a.total_cmp(b));
        println!(
            "  ridge age-distance rad: min {:.3} | median {:.3} | max {:.3}",
            ridge_age[0],
            ridge_age[ridge_age.len() / 2],
            ridge_age[ridge_age.len() - 1]
        );
    }

    // ---- Rivers ----
    let max_flow = hydrology
        .flow_accumulation
        .iter()
        .cloned()
        .fold(0.0f32, f32::max);
    let big_rivers = hydrology
        .flow_accumulation
        .iter()
        .zip(land.iter())
        .filter(|(&f, &l)| l && f > 0.01 * max_flow)
        .count();
    println!("\n-- Rivers --");
    println!(
        "  max flow {:.0} cell-equivalents | land cells carrying >1% of max: {}",
        max_flow, big_rivers
    );

    // ---- Climate ----
    // Aridity index AI = P / PET (precip ÷ evaporative demand), the Earth-standard
    // way to define arid — NOT raw precip. PET rises with temperature (warm air
    // evaporates more), so a cold low-precip region (poles) is NOT arid the way a
    // hot low-precip one (subtropics) is. PET proxy: temperature in ~[0,1] mapped
    // to a 0.2..1.0 demand (same shape as the moisture model's carrying_capacity).
    // AI normalized to land-mean 1 so the threshold is a relative water-stress
    // cutoff; report the ZONAL AI pattern too (that's the threshold-robust signal).
    let pet = |i: usize| 0.2 + 0.8 * temperature[i].clamp(0.0, 1.0);
    let ai_raw: Vec<f32> = (0..n)
        .map(|i| precipitation[i] / pet(i).max(1e-6))
        .collect();
    let land_idx: Vec<usize> = (0..n).filter(|&i| land[i]).collect();
    let ai_mean = land_idx.iter().map(|&i| ai_raw[i]).sum::<f32>() / land_idx.len().max(1) as f32;
    let ai: Vec<f32> = ai_raw.iter().map(|&v| v / ai_mean.max(1e-9)).collect();
    let frac = |pred: &dyn Fn(usize) -> bool| -> f32 {
        100.0 * land_idx.iter().filter(|&&i| pred(i)).count() as f32 / land_idx.len().max(1) as f32
    };
    // Raw-precip arid (the OLD metric) for comparison, then the AI-based one.
    let arid_precip = frac(&|i| precipitation[i] < 0.35);
    let arid_ai = frac(&|i| ai[i] < 0.4); // hot+dry water stress (relative threshold)
    let humid_ai = frac(&|i| ai[i] > 2.0);
    println!(
        "\n-- Climate --   [arid by P/PET (aridity index), not raw P — Earth arid+semiarid ~33%]"
    );
    println!(
        "  aridity index (P/PET, land-mean 1): arid(AI<0.4) {:.0}%  humid(AI>2) {:.0}%  | raw-precip<0.35 {:.0}% (old, temp-blind) | lakes {:.2}%",
        arid_ai,
        humid_ai,
        arid_precip,
        100.0
            * hydrology
                .water_bodies
                .iter()
                .map(|wb| wb.cells.len())
                .sum::<usize>() as f32
            / n as f32
    );
    // Zonal aridity index (mean AI per 15° band): deserts are LOW-AI bands; on a
    // physical planet they sit in the hot subtropics, not the cold poles.
    {
        const BW: f32 = 15.0;
        let nb = (180.0 / BW) as usize;
        let (mut s, mut c) = (vec![0.0f64; nb], vec![0u32; nb]);
        for &i in &land_idx {
            let lat = tess.cell_center(i).y.clamp(-1.0, 1.0).asin().to_degrees();
            let b = (((lat + 90.0) / BW) as usize).min(nb - 1);
            s[b] += ai[i] as f64;
            c[b] += 1;
        }
        let mut line = String::new();
        for b in (0..nb).rev() {
            if c[b] == 0 {
                continue;
            }
            let lo = -90.0 + b as f32 * BW;
            line.push_str(&format!(
                "  [{:+.0}] {:.2}",
                lo + BW * 0.5,
                s[b] / c[b] as f64
            ));
        }
        println!("  zonal aridity index (AI, mid-band lat):{line}");
    }
    let mean_evap = if hydrology.basins.is_empty() {
        1.0
    } else {
        hydrology
            .basins
            .iter()
            .map(|b| b.evaporation_factor)
            .sum::<f32>()
            / hydrology.basins.len() as f32
    };
    println!(
        "  basin evaporation: mean factor {:.2} across {} basins",
        mean_evap,
        hydrology.basins.len()
    );

    // ---- Zonal precipitation profile (latitude bands) ----
    // Aggregate arid% hides WHERE the dryness is. A physical climate has wet
    // bands at the equator (ITCZ ascent) and mid-latitudes, and DRY bands in the
    // subtropics (~15-35°, Hadley descent). Mean LAND precip per 15° band shows
    // whether subtropical dry bands exist (the H9 subsidence-suppression test).
    {
        const BW: f32 = 15.0; // band width, degrees
        let nb = (180.0 / BW) as usize;
        let mut sum = vec![0.0f64; nb];
        let mut cnt = vec![0u32; nb];
        for i in 0..n {
            if !land[i] {
                continue;
            }
            let lat = tess.cell_center(i).y.clamp(-1.0, 1.0).asin().to_degrees();
            let b = (((lat + 90.0) / BW) as usize).min(nb - 1);
            sum[b] += precipitation[i] as f64;
            cnt[b] += 1;
        }
        println!("\n-- Zonal land precip (mean per 15° lat band)  [wet: eq + midlat; dry: subtropics ~15-35°] --");
        let mut line = String::new();
        for b in (0..nb).rev() {
            if cnt[b] == 0 {
                continue;
            }
            let lo = -90.0 + b as f32 * BW;
            let mean = (sum[b] / cnt[b] as f64) as f32;
            line.push_str(&format!("  [{:+.0}..{:+.0}] {:.2}", lo, lo + BW, mean));
        }
        println!("{line}");
    }

    // ---- River concavity (population slope-area) — UNRELIABLE, see note ----
    // WARNING (2026-06-15): this population slope-area theta is NOT trustworthy on
    // this adaptive mesh. It returns flat/negative theta (~0) even when the river
    // long profiles are demonstrably concave-up/graded, and swings wildly (+0.8 to
    // -0.2) under small parameter changes. Causes: flow_accumulation routes through
    // filled/non-overflowing lakes (artificial drainage area), single-edge drop/dist
    // is noisy on irregular Voronoi cells, and the channel population mixes regimes.
    // It sent a whole erosion investigation chasing a phantom "under-graded" bug
    // that did not exist (the aggregate long-profile probe below showed rivers ARE
    // graded). KEPT only as a documented negative result; judge grading by the
    // "River grading (aggregate ...)" probe, not this theta. See
    // docs/erosion-validation-2026-06-15.md.
    // Detachment-limited stream power at steady state gives S ~ A^(-theta),
    // theta = m/n (~0.5). Population method: over channel cells (flow > support,
    // draining downhill) take S = drop/dist, A = flow, bin by ln(A), regress
    // median ln(S) per bin -> theta = -slope.
    {
        // Count-equivalent flow (hydrology stores physical precip×area discharge),
        // so the absolute channel-support threshold stays in upstream-cell units.
        let flow: Vec<f32> = (0..n).map(|i| hydrology.flow_count_equiv(i)).collect();
        let drainage = &hydrology.drainage_dir;
        let channel_thresh = 50.0f32;
        // Exclude cells whose flow_accumulation is contaminated by lake routing:
        // lake-water cells, and cells inside a non-overflowing (terminal) basin
        // whose drainage is captured by the lake rather than reaching the ocean.
        // flow_accumulation routes through filled basins, so without this the
        // slope-area population mixes real channels with lake-spill artifacts.
        let in_terminal_basin = |c: usize| -> bool {
            match hydrology.basin_id[c] {
                Some(b) => !hydrology.basins[b].is_overflowing(),
                None => false,
            }
        };
        let is_lake = |c: usize| hydrology.water_state(c) == CellWaterState::LakeWater;
        let mut pts: Vec<(f32, f32)> = Vec::new(); // (ln A, ln S)
        let mut skipped_lake = 0usize;
        for c in 0..n {
            if !land[c] || flow[c] < channel_thresh {
                continue;
            }
            if is_lake(c) || in_terminal_basin(c) {
                skipped_lake += 1;
                continue;
            }
            let Some(d) = drainage[c] else { continue };
            // Receiver in a lake: the "slope" is to a lake surface, not a graded
            // channel — drop it.
            if is_lake(d) {
                skipped_lake += 1;
                continue;
            }
            let dz = elevation[c] - elevation[d];
            let dx = (tess.cell_center(c) - tess.cell_center(d)).length() * EARTH_RADIUS_KM;
            if dz <= 0.0 || dx <= 0.0 {
                continue;
            }
            pts.push((flow[c].ln(), (dz / dx).ln()));
        }
        if skipped_lake > 0 {
            println!(
                "  (excluded {} lake/terminal-basin channel cells from slope-area)",
                skipped_lake
            );
        }
        println!("\n-- River concavity (slope-area)  [stream-power theta=m/n ~0.5, concave-up] --");
        if pts.len() < 50 {
            println!("  too few channel cells ({}) to fit", pts.len());
        } else {
            let (lo, hi) = pts
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &(x, _)| {
                    (lo.min(x), hi.max(x))
                });
            const NB: usize = 12;
            let mut bins: Vec<Vec<f32>> = vec![Vec::new(); NB];
            for &(x, y) in &pts {
                let t = ((x - lo) / (hi - lo) * NB as f32).floor() as usize;
                bins[t.min(NB - 1)].push(y);
            }
            // Median ln(S) per bin -> (bin-center ln A, median ln S).
            let (mut sx, mut sy, mut sxx, mut sxy, mut k) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0u32);
            for (bi, b) in bins.iter_mut().enumerate() {
                if b.len() < 5 {
                    continue;
                }
                b.sort_by(f32::total_cmp);
                let med = b[b.len() / 2] as f64;
                let xc = (lo + (bi as f32 + 0.5) / NB as f32 * (hi - lo)) as f64;
                sx += xc;
                sy += med;
                sxx += xc * xc;
                sxy += xc * med;
                k += 1;
            }
            let theta = if k >= 3 {
                -((k as f64 * sxy - sx * sy) / (k as f64 * sxx - sx * sx)) as f32
            } else {
                f32::NAN
            };
            println!(
                "  channel cells {} | bins fitted {} | theta = {:+.2}  (UNRELIABLE on this mesh — judge grading by the aggregate long-profile probe below, not this)",
                pts.len(),
                k,
                theta,
            );
        }
    }

    // ---- River grading: aggregate long-profile concavity (ground truth) ----
    // A SINGLE river's profile is too sensitive to which head gets picked (and the
    // population slope-area theta is noisy on this mesh), so aggregate over the N
    // largest mountain rivers. Trace each (highest unused channel head -> sea),
    // measure its concavity, and report the MEDIAN: a graded landscape has rivers
    // that are concave-up (steep source, gentle mouth) — negative normalized bow,
    // source/mouth slope ratio > 1. This cancels single-river selection noise.
    {
        let cflow: Vec<f32> = (0..n).map(|i| hydrology.flow_count_equiv(i)).collect();
        let mut heads: Vec<usize> = (0..n).filter(|&i| land[i] && cflow[i] > 100.0).collect();
        heads.sort_by(|&a, &b| elevation[b].total_cmp(&elevation[a])); // highest first
        let mut visited = vec![false; n];
        let mut bows: Vec<f32> = Vec::new(); // normalized bow (bow/drop): <0 = concave-up
        let mut ratios: Vec<f32> = Vec::new(); // source-third / mouth-third slope
        let (mut lake_sum, mut stem_sum) = (0usize, 0usize);
        const N_RIVERS: usize = 50;
        const MIN_KM: f32 = 200.0;
        for &head in &heads {
            if bows.len() >= N_RIVERS {
                break;
            }
            if visited[head] {
                continue;
            }
            let mut stem = vec![head];
            visited[head] = true;
            let mut cur = head;
            while let Some(d) = hydrology.drainage_dir[cur] {
                if hydrology.water_state(d) == CellWaterState::Ocean || visited[d] || stem.len() > n
                {
                    break;
                }
                visited[d] = true;
                stem.push(d);
                cur = d;
            }
            if stem.len() < 8 {
                continue;
            }
            let mut dist = vec![0.0f32; stem.len()];
            for k in 1..stem.len() {
                dist[k] = dist[k - 1]
                    + (tess.cell_center(stem[k]) - tess.cell_center(stem[k - 1])).length()
                        * EARTH_RADIUS_KM;
            }
            let total_len = dist[stem.len() - 1];
            let drop = elevation[head] - elevation[*stem.last().unwrap()];
            if total_len < MIN_KM || drop <= 1e-4 {
                continue;
            }
            let mid = total_len * 0.5;
            let mk = (0..stem.len())
                .min_by(|&a, &b| (dist[a] - mid).abs().total_cmp(&(dist[b] - mid).abs()))
                .unwrap();
            let chord =
                elevation[head] + (elevation[*stem.last().unwrap()] - elevation[head]) * 0.5;
            bows.push((elevation[stem[mk]] - chord) / drop);
            let seg = |lo: f32, hi: f32| -> f32 {
                let (mut dz, mut dx) = (0.0f32, 0.0f32);
                for k in 1..stem.len() {
                    if dist[k] > lo * total_len && dist[k] <= hi * total_len {
                        dz += elevation[stem[k - 1]] - elevation[stem[k]];
                        dx += dist[k] - dist[k - 1];
                    }
                }
                if dx > 0.0 {
                    dz / dx
                } else {
                    0.0
                }
            };
            let (ss, sm) = (seg(0.0, 0.333), seg(0.667, 1.0));
            if sm > 1e-9 {
                ratios.push(ss / sm);
            }
            lake_sum += stem
                .iter()
                .filter(|&&c| hydrology.water_state(c) == CellWaterState::LakeWater)
                .count();
            stem_sum += stem.len();
        }
        println!("\n-- River grading (aggregate over largest mountain rivers, source -> sea) --   [graded = concave-up]");
        if bows.is_empty() {
            println!("  (no qualifying rivers)");
        } else {
            bows.sort_by(f32::total_cmp);
            ratios.sort_by(f32::total_cmp);
            let med_bow = bows[bows.len() / 2];
            let pct_concave =
                100.0 * bows.iter().filter(|&&b| b < 0.0).count() as f32 / bows.len() as f32;
            let med_ratio = ratios.get(ratios.len() / 2).copied().unwrap_or(f32::NAN);
            println!(
                "  rivers {} | median norm-bow {:+.3} ({}) | concave {:.0}% | median source/mouth slope ratio {:.2} (graded >1) | lake reaches {:.0}%",
                bows.len(),
                med_bow,
                if med_bow < -0.02 {
                    "concave-up / graded"
                } else if med_bow > 0.02 {
                    "convex (NOT graded)"
                } else {
                    "~uniform"
                },
                pct_concave,
                med_ratio,
                100.0 * lake_sum as f32 / stem_sum.max(1) as f32,
            );
        }
    }

    // ---- Drainage density (wet vs arid) ----
    // Area-weighted accumulation should make wet regions carry a finer channel
    // network than arid ones. Channel = flow above a support threshold; density
    // = channel length / land area (1/km). NOTE: flow is precip-weighted, so wet
    // land has more flow trivially -- read the ratio as directional (wet should
    // be denser), and the upland-restricted line controls somewhat for uplift.
    {
        // Count-equivalent flow (see river-concavity probe) for the absolute
        // channel-support threshold below.
        let flow: Vec<f32> = (0..n).map(|i| hydrology.flow_count_equiv(i)).collect();
        let areas = tess.cell_areas();
        let mut lp: Vec<f32> = (0..n)
            .filter(|&i| land[i])
            .map(|i| precipitation[i])
            .collect();
        lp.sort_by(f32::total_cmp);
        let mut le: Vec<f32> = (0..n).filter(|&i| land[i]).map(|i| elevation[i]).collect();
        le.sort_by(f32::total_cmp);
        let p_med = lp.get(lp.len() / 2).copied().unwrap_or(0.0);
        let e_med = le.get(le.len() / 2).copied().unwrap_or(0.0);
        let channel_thresh = 50.0f32;

        let density = |upland_only: bool| -> (f32, f32) {
            let (mut lw, mut aw, mut la, mut aa) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for i in 0..n {
                if !land[i] || (upland_only && elevation[i] < e_med) {
                    continue;
                }
                let a_km2 = areas[i] * EARTH_RADIUS_KM * EARTH_RADIUS_KM;
                let l_km = areas[i].sqrt() * EARTH_RADIUS_KM;
                let wet = precipitation[i] >= p_med;
                if wet {
                    aw += a_km2;
                } else {
                    aa += a_km2;
                }
                if flow[i] >= channel_thresh {
                    if wet {
                        lw += l_km;
                    } else {
                        la += l_km;
                    }
                }
            }
            (
                if aw > 0.0 { lw / aw } else { 0.0 },
                if aa > 0.0 { la / aa } else { 0.0 },
            )
        };

        let (d_wet, d_arid) = density(false);
        let (u_wet, u_arid) = density(true);
        println!(
            "\n-- Drainage density (channel km/km², channel=flow>={})  [wet should dissect more] --",
            channel_thresh
        );
        println!(
            "  all land  : wet(precip>={:.2}) D={:.4} | arid D={:.4} | ratio {:.2}",
            p_med,
            d_wet,
            d_arid,
            d_wet / d_arid.max(1e-9),
        );
        println!(
            "  uplands   : wet D={:.4} | arid D={:.4} | ratio {:.2}  (elev>={:.3}, controls for uplift)",
            u_wet,
            u_arid,
            u_wet / u_arid.max(1e-9),
            e_med,
        );
    }

    // ---- Density allocation audit (fine mesh only) ----
    // Is the fine-cell budget spent where the terrain needs it? The monitor is
    // relief-per-cell Δh_i = max_neighbour |elev_i - elev_j| — the resolved
    // elevation step across a cell. An optimally adaptive mesh EQUIDISTRIBUTES
    // this: classes with tiny Δh/cell (ocean, plains) are over-resolved (cells
    // smaller than their relief needs — wasted budget); classes with large
    // Δh/cell (mountains) carry the representation error. We report, per terrain
    // class: cell budget (count, %), area (%), achieved vs intended cell size,
    // and the Δh/cell distribution — so "coarsen ocean/plains?" and "are
    // mountains starved?" are answered numerically. (Δh in elevation units;
    // cross-class RATIOS are what matter and are unit-free.)
    if let Some(fine) = &world.fine {
        let areas = tess.cell_areas();
        let density = fine.density(); // intended areal density g (cells/steradian)
                                      // Δh per cell: max elevation step to a neighbour.
        let relief: Vec<f32> = (0..n)
            .map(|i| {
                let e = elevation[i];
                tess.neighbors(i)
                    .iter()
                    .map(|&j| (e - elevation[j]).abs())
                    .fold(0.0f32, f32::max)
            })
            .collect();

        // Classes by eroded elevation (independent of the relief monitor, so the
        // equidistribution comparison isn't circular). MOUNTAIN matches the
        // range threshold used above.
        let class_of = |i: usize| -> usize {
            let e = elevation[i];
            if e < 0.0 {
                0 // ocean
            } else if e < 0.03 {
                1 // lowland / plains
            } else if e < RANGE_ELEV {
                2 // upland / hills
            } else {
                3 // mountain
            }
        };
        let names = ["ocean", "lowland", "upland", "mountain"];
        let r2km2 = EARTH_RADIUS_KM * EARTH_RADIUS_KM;
        let _ = density; // intended-size cross-check dropped (cap-scaled + coastal outliers)

        println!("\n-- Density allocation audit (monitor: relief Δh per cell = max neighbour elev step) --");
        println!("   (cell km is cap-dependent: --fine-max coarsens uniformly; relief RATIOS and %cells are cap-robust)");
        println!(
            "{:<9} {:>9} {:>6} {:>9} {:>6} {:>8} {:>9} {:>9} {:>10}",
            "class",
            "cells",
            "%cell",
            "area Mkm²",
            "%area",
            "cell km",
            "Δh/cell p90",
            "awΔh p90",
            "slope p90"
        );
        let total_area: f32 = areas.iter().sum();
        let mut p90_by_class = [0.0f32; 4]; // area-weighted Δh p90 (the steerable monitor)
        let mut slope_p90_by_class = [0.0f32; 4];
        for c in 0..4 {
            let idx: Vec<usize> = (0..n).filter(|&i| class_of(i) == c).collect();
            if idx.is_empty() {
                continue;
            }
            let cnt = idx.len();
            let area_sr: f32 = idx.iter().map(|&i| areas[i]).sum();
            // Median achieved cell size (robust to a few coastal slivers).
            let mut sizes: Vec<f32> = idx
                .iter()
                .map(|&i| areas[i].max(0.0).sqrt() * EARTH_RADIUS_KM)
                .collect();
            sizes.sort_by(f32::total_cmp);
            let cell_km = sizes[cnt / 2];
            // Cell-count Δh p90.
            let mut r: Vec<f32> = idx.iter().map(|&i| relief[i]).collect();
            r.sort_by(f32::total_cmp);
            let cc_p90 = r[(((cnt - 1) as f32) * 0.90) as usize];
            // Area-weighted Δh p90 (each cell weighted by its area, so big cells
            // count proportionally — avoids small mountain cells dominating).
            let mut rw: Vec<(f32, f32)> = idx.iter().map(|&i| (relief[i], areas[i])).collect();
            rw.sort_by(|a, b| a.0.total_cmp(&b.0));
            let aw_p90 = {
                let target = 0.90 * area_sr;
                let mut acc = 0.0f32;
                let mut v = rw.last().map(|x| x.0).unwrap_or(0.0);
                for &(val, a) in &rw {
                    acc += a;
                    if acc >= target {
                        v = val;
                        break;
                    }
                }
                v
            };
            // Slope p90 = Δh per km (separates steepness from cell size: ocean's
            // big Δh is big cells, not steep terrain — this exposes that).
            let mut slope: Vec<f32> = idx
                .iter()
                .map(|&i| relief[i] / (areas[i].max(1e-20).sqrt() * EARTH_RADIUS_KM))
                .collect();
            slope.sort_by(f32::total_cmp);
            let slope_p90 = slope[(((cnt - 1) as f32) * 0.90) as usize];
            p90_by_class[c] = aw_p90;
            slope_p90_by_class[c] = slope_p90;
            println!(
                "{:<9} {:>9} {:>5.1}% {:>9.1} {:>5.1}% {:>8.1} {:>11.4} {:>9.4} {:>10.5}",
                names[c],
                cnt,
                100.0 * cnt as f32 / n as f32,
                area_sr * r2km2 / 1e6,
                100.0 * area_sr / total_area,
                cell_km,
                cc_p90,
                aw_p90,
                slope_p90,
            );
        }
        // Equidistribution ratios on the AREA-WEIGHTED Δh (the quantity the prior
        // should flatten over land). Slope ratios show how much is steepness vs
        // cell size: if the slope ratio ≈ 1 but Δh ratio ≫ 1, it's pure cell-size
        // (over-resolution); if slope ratio is also ≫ 1, the terrain is genuinely
        // steeper there.
        let safe = |x: f32| if x > 1e-9 { x } else { 1e-9 };
        println!(
            "  equidistribution (area-wtd Δh p90): mtn/ocean {:.1}x | mtn/lowland {:.1}x | mtn/upland {:.1}x  (1.0 = balanced)",
            p90_by_class[3] / safe(p90_by_class[0]),
            p90_by_class[3] / safe(p90_by_class[1]),
            p90_by_class[3] / safe(p90_by_class[2]),
        );
        println!(
            "  slope p90 ratio (Δh/km): mtn/ocean {:.1}x | mtn/lowland {:.1}x | mtn/upland {:.1}x  (steepness, cell-size removed)",
            slope_p90_by_class[3] / safe(slope_p90_by_class[0]),
            slope_p90_by_class[3] / safe(slope_p90_by_class[1]),
            slope_p90_by_class[3] / safe(slope_p90_by_class[2]),
        );
        let ocean_cells = (0..n).filter(|&i| class_of(i) == 0).count();
        let ocean_area: f32 = (0..n).filter(|&i| class_of(i) == 0).map(|i| areas[i]).sum();
        println!(
            "  budget: ocean = {:.1}% of cells for {:.1}% of surface; coarsening ocean kx shrinks ocean cells k²-fold",
            100.0 * ocean_cells as f32 / n as f32,
            100.0 * ocean_area / total_area,
        );

        // ---- Fine-scale local relief (scale-controlled convergence probe) ----
        // Fixed PHYSICAL-radius local relief (max-min eroded elevation within R km).
        // Unlike per-cell Δh (which shrinks trivially with cell size), this is
        // independent of resolution, so across a FINE_MOUNTAIN_CELL_KM sweep it
        // reveals whether finer cells keep uncovering dissection (rising ->
        // under-resolved) or it has converged (flat -> dense enough). Sampled over
        // mountain-class cells (the relief-bearing terrain).
        // Mutable KdTree (not ImmutableKdTree): the immutable optimal-layout
        // build panics with "mid > len" on the near-coincident points that
        // appear at multi-million-cell fine meshes (see fine.rs:1167).
        use kiddo::{KdTree, SquaredEuclidean};
        let entries: Vec<[f32; 3]> = (0..n).map(|i| tess.cell_center(i).to_array()).collect();
        let mut tree = KdTree::<f32, 3>::with_capacity(entries.len());
        for (i, e) in entries.iter().enumerate() {
            tree.add(e, i as u64);
        }
        let mtn: Vec<usize> = (0..n).filter(|&i| elevation[i] >= RANGE_ELEV).collect();
        let stride = (mtn.len() / 20_000).max(1);
        let sample: Vec<usize> = mtn.iter().copied().step_by(stride).collect();
        // ~10 km vertical per elev-unit (canonical scale) -> meters for Earth refs.
        const M_PER_UNIT: f32 = 10_000.0;
        println!(
            "\n-- Fine-scale local relief (fixed-radius max-min, meters, {} mountain samples)  [Earth mountains: ~1-2 km within 25 km; hills ~0.3-1 km] --",
            sample.len()
        );
        println!("   ('sharp' = high relief at short scale; scale-controlled: rising p90 across a density sweep => under-resolved, flat => converged)");
        for &rk in &[10.0f32, 25.0f32] {
            let rad = rk / EARTH_RADIUS_KM; // small-angle: chord ≈ arc
            let rsq = rad * rad;
            let mut relief_r: Vec<f32> = sample
                .iter()
                .map(|&i| {
                    let mut lo = f32::INFINITY;
                    let mut hi = f32::NEG_INFINITY;
                    for nb in tree.within_unsorted::<SquaredEuclidean>(&entries[i], rsq) {
                        let e = elevation[nb.item as usize];
                        lo = lo.min(e);
                        hi = hi.max(e);
                    }
                    (hi - lo).max(0.0)
                })
                .collect();
            relief_r.sort_by(f32::total_cmp);
            let m = relief_r.len().max(1);
            let q = |p: f32| relief_r[(((m - 1) as f32) * p) as usize] * M_PER_UNIT;
            println!(
                "  R={:>2.0} km: local relief p50 {:>5.0} m | p90 {:>5.0} m | p99 {:>5.0} m",
                rk,
                q(0.50),
                q(0.90),
                q(0.99)
            );
        }
    }
}

/// Result of auditing one hydrology surface (coarse or fine).
struct DrainageAudit {
    label: String,
    num_cells: usize,
    land_cells: usize,
    land_area: f64,
    endorheic_land_area: f64,
    lake_area: f64,
    num_basins: usize,
    num_endorheic_basins: usize,
    // Lake-capability breakdown (static / climate-independent unless noted):
    lake_capable_basins: usize, // spill - bottom >= MIN_LAKE_DEPTH
    lake_capable_area: f64,     // sum of those basins' total_area (steradians)
    has_water_basins: usize,    // water_level > bottom (at current climate)
    overflowing_basins: usize,  // water_level >= spill (at current climate)
    is_lake_bodies: usize,      // water_bodies with is_lake == true
}

/// Classify a basin chain as sea-reaching or endorheic by walking the
/// `overflow_target` chain. A basin's water escapes to the sea only if every
/// basin along the chain is currently overflowing (full to its spill point) and
/// the chain terminates at a basin whose `overflow_target == None` (drains to
/// ocean). Stops at the first non-overflowing basin (endorheic terminal sink).
/// Cycles (mutually-overflowing closed groups with no external escape) are
/// treated as endorheic. Returns true if sea-reaching.
fn basin_reaches_sea(basins: &[hex3::world::Basin], start: usize) -> bool {
    let mut cur = start;
    let mut visited = std::collections::HashSet::new();
    let cap = basins.len() + 1;
    for _ in 0..cap {
        if !visited.insert(cur) {
            return false; // cycle: closed group, no escape -> endorheic
        }
        let b = &basins[cur];
        if !b.is_overflowing() {
            return false; // water pools here below spill -> endorheic terminal
        }
        match b.overflow_target {
            None => return true, // overflowing AND drains to ocean -> sea-reaching
            Some(next) => cur = next,
        }
    }
    false // ran past the cap without resolving -> treat as endorheic
}

/// Audit one hydrology surface over LAND cells (area-weighted).
fn audit_hydrology(
    label: &str,
    tess: &hex3::world::Tessellation,
    hydro: &hex3::world::Hydrology,
) -> DrainageAudit {
    let n = tess.num_cells();
    let areas = tess.cell_areas();

    // Pre-classify every basin once.
    let basin_sea: Vec<bool> = (0..hydro.basins.len())
        .map(|b| basin_reaches_sea(&hydro.basins, b))
        .collect();

    let mut land_cells = 0usize;
    let mut land_area = 0.0f64;
    let mut endorheic_land_area = 0.0f64;
    let mut lake_area = 0.0f64;

    for i in 0..n {
        if hydro.is_ocean(i) {
            continue;
        }
        // LAND = anything not ocean. Lake/dry-basin cells sit on land surface.
        let a = areas[i] as f64;
        land_cells += 1;
        land_area += a;

        // Lake water on land counts toward the lake fraction.
        if matches!(hydro.water_state(i), hex3::world::CellWaterState::LakeWater) {
            lake_area += a;
        }

        // Trace this cell's drainage to its first basin (capture point), then
        // classify via that basin's overflow chain. If the drainage path reaches
        // ocean without entering any basin, it is sea-reaching.
        let endorheic = {
            let mut cell = i;
            let mut hops = 0usize;
            let cap = n + 1;
            loop {
                if hydro.is_ocean(cell) {
                    break false; // reached the sea directly
                }
                if let Some(bid) = hydro.basin_id[cell] {
                    // First basin entered: its chain decides the fate.
                    break !basin_sea[bid];
                }
                hops += 1;
                if hops > cap {
                    break true; // drainage cycle off-basin: cannot reach sea
                }
                match hydro.downstream(cell) {
                    Some(next) => cell = next,
                    None => break true, // dead-end not in a basin -> inland sink
                }
            }
        };
        if endorheic {
            endorheic_land_area += a;
        }
    }

    let num_endorheic_basins = hydro.basins.iter().filter(|b| !b.is_overflowing()).count();

    // Lake-capability breakdown. lake_capable is static topology (independent of
    // climate); has_water/overflowing/is_lake reflect the CURRENT climate ratio.
    let min_lake_depth = hex3::world::MIN_LAKE_DEPTH;
    let mut lake_capable_basins = 0usize;
    let mut lake_capable_area = 0.0f64;
    let mut has_water_basins = 0usize;
    let mut overflowing_basins = 0usize;
    for b in &hydro.basins {
        if (b.spill_elevation - b.bottom_elevation) >= min_lake_depth {
            lake_capable_basins += 1;
            lake_capable_area += b.total_area as f64;
        }
        if b.has_water() {
            has_water_basins += 1;
        }
        if b.is_overflowing() {
            overflowing_basins += 1;
        }
    }
    let is_lake_bodies = hydro.water_bodies.iter().filter(|w| w.is_lake).count();

    DrainageAudit {
        label: label.to_string(),
        num_cells: n,
        land_cells,
        land_area,
        endorheic_land_area,
        lake_area,
        num_basins: hydro.basins.len(),
        num_endorheic_basins,
        lake_capable_basins,
        lake_capable_area,
        has_water_basins,
        overflowing_basins,
        is_lake_bodies,
    }
}

fn print_audit(a: &DrainageAudit) {
    let land = a.land_area.max(1e-30);
    println!(
        "  [{}] cells={} land_cells={} land_area={:.4} sr\n     endorheic_land_fraction = {:.1}%   lake_fraction_of_land = {:.2}%\n     basins: {} total, {} endorheic (non-overflowing) = {:.1}%",
        a.label,
        a.num_cells,
        a.land_cells,
        a.land_area,
        100.0 * a.endorheic_land_area / land,
        100.0 * a.lake_area / land,
        a.num_basins,
        a.num_endorheic_basins,
        if a.num_basins > 0 {
            100.0 * a.num_endorheic_basins as f64 / a.num_basins as f64
        } else {
            0.0
        },
    );
    println!(
        "     lake-capable basins (depth>=MIN_LAKE_DEPTH): {} ({:.2}% of land area)\n     of those, currently: {} have water, {} overflowing | {} water-bodies classified is_lake",
        a.lake_capable_basins,
        100.0 * a.lake_capable_area / land,
        a.has_water_basins,
        a.overflowing_basins,
        a.is_lake_bodies,
    );
}

/// COARSE-vs-FINE drainage audit. Computes endorheic land fraction, lake
/// fraction, and basin counts for (A) the fine eroded hydrology and (B) a
/// freshly-computed coarse hydrology, using the SAME generator + default
/// climate ratio.
fn run_drainage_audit(world: &World, seed: u64) {
    use hex3::world::{Hydrology, DEFAULT_CLIMATE_RATIO};

    println!(
        "\n================ DRAINAGE AUDIT seed={} (climate_ratio={}) ================",
        seed, DEFAULT_CLIMATE_RATIO
    );
    println!("(land = non-ocean cells; all fractions AREA-weighted in steradians)");

    // (B) COARSE hydrology: same call fine.rs uses for its preview, on the
    // coarse tessellation + coarse elevation + coarse crust/atmosphere fields.
    let crust = world.crust.as_ref().expect("crust");
    let coarse_elev = world.elevation.as_ref().expect("coarse elevation");
    let atmos = world.atmosphere.as_ref().expect("atmosphere");
    let coarse_hydro = Hydrology::generate(
        &world.tessellation,
        crust,
        coarse_elev,
        &atmos.precipitation,
        &atmos.temperature,
    );
    let coarse = audit_hydrology("COARSE", &world.tessellation, &coarse_hydro);

    // (A) FINE hydrology: the eroded stage-4 surface (default view_stage = MAX).
    let fine_audit = match world.fine.as_ref() {
        Some(fine) => {
            let surf = fine.surface_for(u32::MAX); // eroded stage-4 surface
            Some(audit_hydrology(
                "FINE (eroded)",
                fine.tessellation(),
                &surf.hydrology,
            ))
        }
        None => {
            println!("  [FINE] no fine surface present (run without disabling fine mesh)");
            None
        }
    };

    print_audit(&coarse);
    if let Some(f) = &fine_audit {
        print_audit(f);
    }

    println!("\n  VERDICT INPUTS:");
    println!(
        "    COARSE endorheic-land = {:.1}%   FINE endorheic-land = {}",
        100.0 * coarse.endorheic_land_area / coarse.land_area.max(1e-30),
        fine_audit
            .as_ref()
            .map(|f| format!(
                "{:.1}%",
                100.0 * f.endorheic_land_area / f.land_area.max(1e-30)
            ))
            .unwrap_or_else(|| "n/a".into())
    );
}

// ======================== LAKE AUDIT ========================
//
// Object-level lake statistics: each lake is measured as a FEATURE (size,
// shape, depth, placement, plumbing) rather than a per-cell fraction. This is
// the instrument for judging lake QUALITY numerically — aggregate fractions
// and rendered images both miss the failure modes that read as "not great"
// (speckle-sized lakes, perched lakes, lakes without catchments, a dead
// climate dial).

/// One lake (an `is_lake` water body) with object-level measurements.
struct LakeRecord {
    area_km2: f32,
    cells: usize,
    max_depth: f32,
    length_km: f32,
    elongation: f32,
    /// Basin overflows: the lake has an outlet river (exorheic).
    has_outlet: bool,
    /// Direct (first-capture) catchment area / lake area. Earth: ~10-100.
    catchment_ratio: f32,
    /// Area-weighted mean precipitation over the direct catchment.
    mean_catchment_precip: f32,
    /// Land-hypsometry percentile of the lake surface (0 = lowest land).
    hypsometric_pct: f32,
}

struct LakePanel {
    label: String,
    land_area_km2: f64,
    /// Lakes, sorted largest first.
    records: Vec<LakeRecord>,
    /// Water bodies below the lake depth threshold (not counted as lakes).
    ponds: usize,
    climate_ratio: f32,
}

/// Attribute every land cell to the FIRST basin its drainage path enters
/// (path-compressed walk). Cells already inside a depression belong to that
/// basin directly (handled by the caller via `basin_id`); cells whose path
/// reaches the ocean, dead-ends, or cycles off-basin belong to none.
fn first_capture_basin(hydro: &hex3::world::Hydrology, n: usize) -> Vec<Option<usize>> {
    let mut capture: Vec<Option<usize>> = vec![None; n];
    let mut state = vec![0u8; n]; // 0 unvisited, 1 on current path, 2 resolved
    for start in 0..n {
        if hydro.is_ocean(start) || hydro.basin_id[start].is_some() || state[start] == 2 {
            continue;
        }
        let mut path = Vec::new();
        let mut cell = start;
        let result = loop {
            if hydro.is_ocean(cell) {
                break None;
            }
            if let Some(b) = hydro.basin_id[cell] {
                break Some(b);
            }
            match state[cell] {
                2 => break capture[cell],
                1 => break None, // drainage cycle off-basin: no capture
                _ => {}
            }
            state[cell] = 1;
            path.push(cell);
            match hydro.downstream(cell) {
                Some(next) => cell = next,
                None => break None,
            }
        };
        for c in path {
            capture[c] = result;
            state[c] = 2;
        }
    }
    capture
}

fn measure_lakes(
    label: &str,
    tess: &hex3::world::Tessellation,
    hydro: &hex3::world::Hydrology,
    precipitation: &[f32],
) -> LakePanel {
    let n = tess.num_cells();
    let areas = tess.cell_areas();
    let r2 = EARTH_RADIUS_KM * EARTH_RADIUS_KM;

    // Land hypsometry (sorted elevation + prefix areas) for placement percentiles.
    let mut land: Vec<(f32, f32)> = (0..n)
        .filter(|&i| !hydro.is_ocean(i))
        .map(|i| (hydro.elevation[i], areas[i]))
        .collect();
    land.sort_by(|a, b| a.0.total_cmp(&b.0));
    let land_area_sr: f64 = land.iter().map(|&(_, a)| a as f64).sum();
    let mut prefix = Vec::with_capacity(land.len());
    let mut acc = 0.0f64;
    for &(_, a) in &land {
        acc += a as f64;
        prefix.push(acc);
    }
    let hyps_pct = |elev: f32| -> f32 {
        let idx = land.partition_point(|&(e, _)| e < elev);
        if idx == 0 {
            0.0
        } else {
            (prefix[idx - 1] / land_area_sr.max(1e-30)) as f32 * 100.0
        }
    };

    // Direct catchment (first-capture) per basin: area + mean precipitation.
    let capture = first_capture_basin(hydro, n);
    let mut catch_area = vec![0.0f64; hydro.basins.len()];
    let mut catch_precip = vec![0.0f64; hydro.basins.len()];
    for i in 0..n {
        if hydro.is_ocean(i) {
            continue;
        }
        if let Some(b) = hydro.basin_id[i].or(capture[i]) {
            catch_area[b] += areas[i] as f64;
            catch_precip[b] += (precipitation[i] * areas[i]) as f64;
        }
    }

    // Lake objects: connected components of lake water, mapped back to water
    // bodies for depth + basin plumbing. measure_components sorts largest-first.
    let mask: Vec<bool> = (0..n).map(|i| hydro.is_lake_water(i)).collect();
    let comps = measure_components(tess, &mask);
    let mut records = Vec::with_capacity(comps.len());
    for comp in &comps {
        let Some(wb_id) = comp.cells.iter().find_map(|&c| hydro.cell_water_body[c]) else {
            continue;
        };
        let wb = &hydro.water_bodies[wb_id];
        let basin = &hydro.basins[wb.basin_id];
        let lake_area_sr = (comp.area_km2 / r2) as f64;
        records.push(LakeRecord {
            area_km2: comp.area_km2,
            cells: comp.cells.len(),
            max_depth: wb.max_depth,
            length_km: comp.length_km,
            elongation: comp.elongation(),
            has_outlet: basin.is_overflowing(),
            catchment_ratio: (catch_area[wb.basin_id] / lake_area_sr.max(1e-30)) as f32,
            mean_catchment_precip: (catch_precip[wb.basin_id] / catch_area[wb.basin_id].max(1e-30))
                as f32,
            hypsometric_pct: hyps_pct(basin.water_level),
        });
    }
    let ponds = hydro.water_bodies.iter().filter(|w| !w.is_lake).count();

    LakePanel {
        label: label.to_string(),
        land_area_km2: land_area_sr * r2 as f64,
        records,
        ponds,
        climate_ratio: hydro.climate_ratio(),
    }
}

fn print_lake_panel(p: &LakePanel, top: usize) {
    println!("\n  [{}] climate_ratio={:.2}", p.label, p.climate_ratio);
    if p.records.is_empty() {
        println!(
            "     NO LAKES ({} sub-threshold ponds). Earth ref: lakes ≈ 1.8% of land.",
            p.ponds
        );
        return;
    }
    let total_km2: f64 = p.records.iter().map(|r| r.area_km2 as f64).sum();
    let land_pct = 100.0 * total_km2 / p.land_area_km2.max(1e-30);

    // Size spectrum (records are sorted largest-first).
    let largest = &p.records[0];
    let largest_share = 100.0 * largest.area_km2 as f64 / total_km2.max(1e-30);
    let area_at = |q: f64| -> f32 {
        // records sorted DESC; take from the ascending view
        let k = ((p.records.len() - 1) as f64 * (1.0 - q)) as usize;
        p.records[k].area_km2
    };
    let speckle: Vec<&LakeRecord> = p.records.iter().filter(|r| r.cells <= 2).collect();
    let speckle_area: f64 = speckle.iter().map(|r| r.area_km2 as f64).sum();

    println!(
        "     lakes: {} (+{} sub-threshold ponds)   total {:.0} km² = {:.2}% of land (Earth ≈ 1.8%)",
        p.records.len(),
        p.ponds,
        total_km2,
        land_pct
    );
    println!(
        "     size spectrum: largest {:.0} km² = {:.0}% of all lake area (Earth: Caspian ≈ 30%) | p50 {:.0} km², p90 {:.0} km² (Earth is HEAVY-tailed: many small, area in the few big)",
        largest.area_km2,
        largest_share,
        area_at(0.5),
        area_at(0.9)
    );
    println!(
        "     speckle: {} lakes ≤2 cells = {:.0}% of count, {:.1}% of lake area (high count-share + low area-share is fine; high AREA-share reads as noise)",
        speckle.len(),
        100.0 * speckle.len() as f64 / p.records.len() as f64,
        100.0 * speckle_area / total_km2.max(1e-30)
    );

    // Plumbing + placement.
    let outlets = p.records.iter().filter(|r| r.has_outlet).count();
    let mean_precip = |sel: &dyn Fn(&&LakeRecord) -> bool| -> Option<f32> {
        let v: Vec<f32> = p
            .records
            .iter()
            .filter(sel)
            .map(|r| r.mean_catchment_precip)
            .collect();
        if v.is_empty() {
            None
        } else {
            Some(v.iter().sum::<f32>() / v.len() as f32)
        }
    };
    let fmt_opt = |o: Option<f32>| o.map(|v| format!("{v:.2}")).unwrap_or_else(|| "-".into());
    println!(
        "     outlets: {}/{} lakes overflow (have an outlet river) | catchment precip: outlet-lakes {} vs terminal {} (Earth: terminal lakes sit in DRY basins)",
        outlets,
        p.records.len(),
        fmt_opt(mean_precip(&|r| r.has_outlet)),
        fmt_opt(mean_precip(&|r| !r.has_outlet))
    );
    let mut hyps: Vec<f32> = p.records.iter().map(|r| r.hypsometric_pct).collect();
    hyps.sort_by(f32::total_cmp);
    let mut catch: Vec<f32> = p.records.iter().map(|r| r.catchment_ratio).collect();
    catch.sort_by(f32::total_cmp);
    let q = |v: &[f32], f: f64| v[((v.len() - 1) as f64 * f) as usize];
    println!(
        "     placement: lake-surface land-hypsometry pct p10/p50/p90 = {:.0}/{:.0}/{:.0} (Earth: most lake AREA sits low) | catchment/lake ratio p10/p50/p90 = {:.0}/{:.0}/{:.0}× (Earth ≈ 10-100×; ~1× = a puddle with no watershed)",
        q(&hyps, 0.1), q(&hyps, 0.5), q(&hyps, 0.9),
        q(&catch, 0.1), q(&catch, 0.5), q(&catch, 0.9)
    );

    println!(
        "     top lakes:   area_km²  cells  depth  len_km  elong  outlet  catch×  precip  hyps%"
    );
    for (i, r) in p.records.iter().take(top).enumerate() {
        println!(
            "       {:>2}. {:>10.0}  {:>5}  {:>5.3}  {:>6.0}  {:>5.1}  {:>6} {:>6.0}×  {:>6.2}  {:>4.0}",
            i + 1,
            r.area_km2,
            r.cells,
            r.max_depth,
            r.length_km,
            r.elongation,
            if r.has_outlet { "yes" } else { "TERM" },
            r.catchment_ratio,
            r.mean_catchment_precip,
            r.hypsometric_pct
        );
    }
}

/// Lake count + lake fraction of land at the CURRENT climate ratio.
fn lake_dial_point(
    tess: &hex3::world::Tessellation,
    hydro: &hex3::world::Hydrology,
) -> (usize, f64) {
    let areas = tess.cell_areas();
    let n = tess.num_cells();
    let mut land = 0.0f64;
    let mut lake = 0.0f64;
    for i in 0..n {
        if hydro.is_ocean(i) {
            continue;
        }
        land += areas[i] as f64;
        if hydro.is_lake_water(i) {
            lake += areas[i] as f64;
        }
    }
    let n_lakes = hydro.water_bodies.iter().filter(|w| w.is_lake).count();
    (n_lakes, 100.0 * lake / land.max(1e-30))
}

fn run_lake_audit(world: &mut World, seed: u64, top: usize) {
    use hex3::world::Hydrology;

    println!(
        "\n================ LAKE AUDIT seed={} ================",
        seed
    );
    println!("(object-level lake features, Earth refs inline; all fractions AREA-weighted)");
    println!("(pre-integration baseline: rerun with HEX3_NO_DRAINAGE_INTEGRATION=1)");

    // COARSE: freshly generated on the coarse tessellation (same call the fine
    // preview uses), OWNED so the dial sweep below can mutate it.
    let crust = world.crust.as_ref().expect("crust");
    let coarse_elev = world.elevation.as_ref().expect("coarse elevation");
    let atmos = world.atmosphere.as_ref().expect("atmosphere");
    let mut coarse_hydro = Hydrology::generate(
        &world.tessellation,
        crust,
        coarse_elev,
        &atmos.precipitation,
        &atmos.temperature,
    );
    let coarse_panel = measure_lakes(
        "COARSE",
        &world.tessellation,
        &coarse_hydro,
        &atmos.precipitation,
    );
    print_lake_panel(&coarse_panel, top);

    // FINE: the eroded stage-4 surface (falls back to pre-erosion if absent).
    match world.fine.as_ref() {
        Some(fine) => {
            let surf = fine.surface_for(u32::MAX);
            let panel = measure_lakes(
                "FINE (eroded)",
                fine.tessellation(),
                &surf.hydrology,
                &surf.precipitation,
            );
            print_lake_panel(&panel, top);
        }
        None => println!("  [FINE] no fine surface present (run without disabling fine mesh)"),
    }

    // Climate-dial response: the lake system's transfer function. Healthy =
    // smooth + monotonic; a step function = degenerate fill criterion.
    let ratios = [0.05f32, 0.10, 0.15, 0.20, 0.30, 0.50];
    let fine_orig = world
        .fine
        .as_ref()
        .map(|f| f.surface_for(u32::MAX).hydrology.climate_ratio());
    println!("\n  CLIMATE-DIAL RESPONSE (lakes | lake%-of-land):");
    println!("    ratio        COARSE               FINE");
    for &r in &ratios {
        coarse_hydro.set_climate_ratio(&world.tessellation, r);
        let (cn, cf) = lake_dial_point(&world.tessellation, &coarse_hydro);
        let fine_str = match world.fine.as_mut() {
            Some(fine) => {
                fine.set_climate_ratio(4, r);
                let surf = fine.surface_for(u32::MAX);
                let (fnum, ff) = lake_dial_point(fine.tessellation(), &surf.hydrology);
                format!("{fnum:>5} | {ff:5.2}%")
            }
            None => "n/a".into(),
        };
        println!("    {:>5.2}   {:>5} | {:5.2}%      {}", r, cn, cf, fine_str);
    }
    // Restore the world's fine climate ratio (the audit must not mutate state
    // a later consumer would see).
    if let (Some(fine), Some(orig)) = (world.fine.as_mut(), fine_orig) {
        fine.set_climate_ratio(4, orig);
    }
}

// ======================== MOUNTAIN AUDIT ========================
//
// Object-level RANGE statistics on the fine eroded surface. Each range is a
// connected component of high land, measured as a feature: size/shape, crest
// continuity (the pass/gap spectrum along strike), cross-strike asymmetry,
// and sampled local relief. This is where "the mountains look wrong" gets
// localized: blob-shaped ranges (low elongation), unbroken walls (no passes),
// symmetric profiles (lost O0 asymmetry), or low local relief all have
// distinct signatures here. Earth refs inline.

const AUDIT_RANGE_ELEV: f32 = 0.15;
const AUDIT_M_PER_UNIT: f32 = 10_000.0; // ~10 km vertical per elevation unit

/// The (sampled) farthest-apart pair of cells in a component — the range's
/// principal axis endpoints.
fn audit_extremal_pair(tess: &hex3::world::Tessellation, cells: &[usize]) -> (usize, usize) {
    const MAX_SAMPLE: usize = 200;
    let stride = (cells.len() / MAX_SAMPLE).max(1);
    let sample: Vec<usize> = cells.iter().step_by(stride).copied().collect();
    let (mut best, mut pair) = (-1.0f32, (cells[0], cells[0]));
    for (i, &a) in sample.iter().enumerate() {
        let pa = tess.cell_center(a);
        for &b in &sample[i + 1..] {
            let d = (pa - tess.cell_center(b)).length_squared();
            if d > best {
                best = d;
                pair = (a, b);
            }
        }
    }
    pair
}

struct RangeAudit {
    area_km2: f32,
    length_km: f32,
    width_km: f32,
    peak_m: f32,
    /// Crest bins along the principal axis that contain range cells.
    crest_bins: usize,
    /// Distinct passes: contiguous crest-profile lows ≥100 m below both
    /// flanking crest maxima.
    passes: usize,
    /// Median pass depth (m below the lower flanking crest max).
    median_pass_depth_m: f32,
    /// Lowest crest-profile point (m) — the range's deepest through-gap.
    crest_floor_m: f32,
    /// Crest offset across strike: 0 = crest centered, ±1 = crest hugs an
    /// edge (asymmetric range profile, the O0 signature).
    crest_offset: f32,
}

fn audit_range(
    tess: &hex3::world::Tessellation,
    comp: &ComponentStats,
    elev: &[f32],
) -> RangeAudit {
    let (pa, pb) = audit_extremal_pair(tess, &comp.cells);
    let a = tess.cell_center(pa);
    let b = tess.cell_center(pb);
    let nrm = a.cross(b).normalize_or_zero();
    let c_axis = nrm.cross(a).normalize_or_zero();
    let arc = a.dot(b).clamp(-1.0, 1.0).acos();

    // Along-strike parameter t (radians along the great-circle axis) and
    // signed cross-strike parameter s (radians off the axis) per cell.
    let param = |i: usize| -> (f32, f32) {
        let p = tess.cell_center(i);
        let t = p.dot(c_axis).atan2(p.dot(a));
        let s = p.dot(nrm).clamp(-1.0, 1.0).asin();
        (t, s)
    };

    // Crest profile: max elevation per ~50 km bin along strike.
    let bin_w = 50.0 / EARTH_RADIUS_KM;
    let nbins = ((arc / bin_w).ceil() as usize).clamp(1, 4096);
    let mut crest = vec![f32::NEG_INFINITY; nbins];
    let mut ss: Vec<f32> = Vec::with_capacity(comp.cells.len());
    let mut peak = f32::NEG_INFINITY;
    for &i in &comp.cells {
        let (t, s) = param(i);
        let bin = ((t / arc.max(1e-9)) * nbins as f32) as usize;
        let bin = bin.min(nbins - 1);
        crest[bin] = crest[bin].max(elev[i]);
        ss.push(s);
        peak = peak.max(elev[i]);
    }
    let filled: Vec<(usize, f32)> = crest
        .iter()
        .enumerate()
        .filter(|(_, &e)| e.is_finite())
        .map(|(b, &e)| (b, e))
        .collect();

    // Pass detection on the filled crest profile: prefix/suffix maxima.
    let m = filled.len();
    let mut passes = 0usize;
    let mut pass_depths: Vec<f32> = Vec::new();
    let mut crest_floor = peak;
    if m >= 3 {
        let mut pre = vec![f32::NEG_INFINITY; m];
        let mut suf = vec![f32::NEG_INFINITY; m];
        for k in 1..m {
            pre[k] = pre[k - 1].max(filled[k - 1].1);
        }
        for k in (0..m - 1).rev() {
            suf[k] = suf[k + 1].max(filled[k + 1].1);
        }
        const PASS_MIN_DEPTH: f32 = 0.01; // 100 m
        let mut in_pass = false;
        let mut cur_depth = 0.0f32;
        for k in 1..m - 1 {
            let depth = pre[k].min(suf[k]) - filled[k].1;
            crest_floor = crest_floor.min(filled[k].1);
            if depth >= PASS_MIN_DEPTH {
                in_pass = true;
                cur_depth = cur_depth.max(depth);
            } else if in_pass {
                passes += 1;
                pass_depths.push(cur_depth);
                in_pass = false;
                cur_depth = 0.0;
            }
        }
        if in_pass {
            passes += 1;
            pass_depths.push(cur_depth);
        }
    }
    pass_depths.sort_by(f32::total_cmp);
    let median_pass_depth = pass_depths
        .get(pass_depths.len() / 2)
        .copied()
        .unwrap_or(0.0);

    // Cross-strike asymmetry: where does the crest sit inside the range's
    // s-envelope? Crest cells = top 20% of the elevation band.
    ss.sort_by(f32::total_cmp);
    let sq = |q: f32| ss[(((ss.len() - 1) as f32) * q) as usize];
    let (s_lo, s_hi) = (sq(0.05), sq(0.95));
    let half = ((s_hi - s_lo) * 0.5).max(1e-9);
    let center = (s_hi + s_lo) * 0.5;
    let crest_thr = peak - 0.2 * (peak - AUDIT_RANGE_ELEV);
    let mut s_crest: Vec<f32> = comp
        .cells
        .iter()
        .filter(|&&i| elev[i] >= crest_thr)
        .map(|&i| param(i).1)
        .collect();
    s_crest.sort_by(f32::total_cmp);
    let s_crest_med = s_crest.get(s_crest.len() / 2).copied().unwrap_or(center);
    let crest_offset = ((s_crest_med - center) / half).clamp(-1.5, 1.5);

    RangeAudit {
        area_km2: comp.area_km2,
        length_km: comp.length_km,
        width_km: comp.width_km,
        peak_m: peak * AUDIT_M_PER_UNIT,
        crest_bins: m,
        passes,
        median_pass_depth_m: median_pass_depth * AUDIT_M_PER_UNIT,
        crest_floor_m: crest_floor * AUDIT_M_PER_UNIT,
        crest_offset,
    }
}

fn run_mountain_audit(world: &World, seed: u64, top: usize) {
    let Some(fine) = world.fine.as_ref() else {
        println!("\n[MOUNTAIN AUDIT] no fine surface present");
        return;
    };
    let tess = fine.tessellation();
    let surf = fine.surface_for(u32::MAX);
    let elev = &surf.elevation.values;
    let n = tess.num_cells();
    let areas = tess.cell_areas();

    println!(
        "\n================ MOUNTAIN AUDIT seed={} (fine eroded, elev>={}) ================",
        seed, AUDIT_RANGE_ELEV
    );

    let mask: Vec<bool> = (0..n).map(|i| elev[i] >= AUDIT_RANGE_ELEV).collect();
    let comps = measure_components(tess, &mask);
    let land_km2: f64 = (0..n)
        .filter(|&i| elev[i] >= 0.0)
        .map(|i| (areas[i] * EARTH_RADIUS_KM * EARTH_RADIUS_KM) as f64)
        .sum();
    let mtn_km2: f64 = comps.iter().map(|c| c.area_km2 as f64).sum();
    let significant: Vec<&ComponentStats> =
        comps.iter().filter(|c| c.area_km2 >= 20_000.0).collect();
    println!(
        "  mountain land: {:.1}% ({} components, {} significant >=20k km²)  [Earth high-mountain land ~10-12%]",
        100.0 * mtn_km2 / land_km2.max(1e-30),
        comps.len(),
        significant.len()
    );
    if significant.is_empty() {
        println!("  NO significant ranges — nothing to audit at this threshold");
        return;
    }
    let mut elong: Vec<f32> = significant.iter().map(|c| c.elongation()).collect();
    elong.sort_by(f32::total_cmp);
    let eq = |q: f32| elong[(((elong.len() - 1) as f32) * q) as usize];
    println!(
        "  elongation (significant ranges) p10/p50/p90 = {:.1}/{:.1}/{:.1}  [Earth belts 5-20x; <2 = blob]",
        eq(0.1),
        eq(0.5),
        eq(0.9)
    );

    println!(
        "  per-range:   area_km²  len×wid km   peak_m  crest: bins passes med_pass_m floor_m  offset"
    );
    println!(
        "               [Andes 7000x300, Himalaya 2400x1000, Alps 1200x200 | passes: real belts are BREACHED — Alps ~10 major; floor<<peak | offset: 0 = symmetric, ±1 = one-sided (O0)]"
    );
    for (k, comp) in significant.iter().take(top).enumerate() {
        let r = audit_range(tess, comp, elev);
        println!(
            "    {:>2}. {:>9.0}  {:>5.0}x{:>4.0}  {:>7.0}   {:>4} {:>6} {:>10.0} {:>7.0}  {:>+.2}",
            k + 1,
            r.area_km2,
            r.length_km,
            r.width_km,
            r.peak_m,
            r.crest_bins,
            r.passes,
            r.median_pass_depth_m,
            r.crest_floor_m,
            r.crest_offset
        );
    }

    // RELIEF SPECTRUM: local relief (max-min elevation) at nested window sizes
    // over mountain cells, on the PRE-erosion base and the ERODED surface. This
    // is the wavelength-ownership instrument: which synthesis layer produces
    // relief at which scale, and what erosion adds/removes per band. Chord
    // distances (f32 acos collapses below ~3 km on the fine mesh). One BFS per
    // sample at the largest radius; inner windows read from the same expansion.
    const WINDOWS_KM: [f32; 5] = [5.0, 10.0, 25.0, 50.0, 100.0]; // diameters
    let pre_elev = &fine.surface_for(3).elevation.values;
    let mtn_cells: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
    let stride = (mtn_cells.len() / 4_000).max(1);
    let max_radius = WINDOWS_KM[WINDOWS_KM.len() - 1] * 0.5 / EARTH_RADIUS_KM;
    let radii: Vec<f32> = WINDOWS_KM
        .iter()
        .map(|w| w * 0.5 / EARTH_RADIUS_KM)
        .collect();
    // reliefs[metric][surface][window] -> samples. metric 0 = max-min (reference),
    // metric 1 = p95-p05 within the window (robust: one spike/pit can't inflate it
    // — the B-regime GATE metric per the relief-spectrum spec).
    let mut reliefs = vec![vec![vec![Vec::<f32>::new(); WINDOWS_KM.len()]; 2]; 2];
    let mut visited_mark = vec![u32::MAX; n];
    let mut window_cells: Vec<(f32, f32, f32)> = Vec::new(); // (dist, pre, eroded)
    for (si, &start) in mtn_cells.iter().step_by(stride).enumerate() {
        let mark = si as u32;
        let c0 = tess.cell_center(start);
        window_cells.clear();
        let mut queue = std::collections::VecDeque::new();
        visited_mark[start] = mark;
        queue.push_back((start, 0.0f32));
        while let Some((cell, dist)) = queue.pop_front() {
            window_cells.push((dist, pre_elev[cell], elev[cell]));
            for &nb in tess.neighbors(cell) {
                if visited_mark[nb] != mark {
                    let d = (tess.cell_center(nb) - c0).length();
                    if d <= max_radius {
                        visited_mark[nb] = mark;
                        queue.push_back((nb, d));
                    }
                }
            }
        }
        for (w, &r) in radii.iter().enumerate() {
            for s in 0..2 {
                let mut vals: Vec<f32> = window_cells
                    .iter()
                    .filter(|&&(d, _, _)| d <= r)
                    .map(|&(_, pre, ero)| if s == 0 { pre } else { ero })
                    .collect();
                if vals.is_empty() {
                    continue;
                }
                vals.sort_by(f32::total_cmp);
                let q = |f: f32| vals[(((vals.len() - 1) as f32) * f) as usize];
                reliefs[0][s][w].push((q(1.0) - q(0.0)) * AUDIT_M_PER_UNIT);
                reliefs[1][s][w].push((q(0.95) - q(0.05)) * AUDIT_M_PER_UNIT);
            }
        }
    }
    if !reliefs[0][1][0].is_empty() {
        println!(
            "  relief spectrum ({} samples; p50 m, [p90]):  [Earth alpine ballpark: 10 km ~1000-1800, 25 km ~1500-3000, 50 km ~2000-3500, then saturates. p95-p05 is the GATE metric; max-min is reference]",
            reliefs[0][1][0].len()
        );
        for (mlabel, m) in [("max-min", 0usize), ("p95-p05", 1usize)] {
            for (label, s) in [("pre", 0usize), ("eroded", 1usize)] {
                let mut line = format!("    {mlabel:<8} {label:<7}");
                for (w, wkm) in WINDOWS_KM.iter().enumerate() {
                    let v = &mut reliefs[m][s][w];
                    if v.is_empty() {
                        continue;
                    }
                    v.sort_by(f32::total_cmp);
                    let q = |f: f32| v[(((v.len() - 1) as f32) * f) as usize];
                    line.push_str(&format!(
                        "  {:>3.0}km {:>4.0} [{:>4.0}]",
                        wkm,
                        q(0.5),
                        q(0.9)
                    ));
                }
                println!("{line}");
            }
        }
    }
}

// ======================== RIVER AUDIT ========================
//
// The RENDERED river network as an object: same thresholds as the renderer
// (All = 0.00005·N count-equivalents; Major = 0.004/0.0006·N outlet/branch),
// so these numbers describe the rivers the user sees. Structure (Strahler/
// Horton), coverage (drainage density, mouth census), and the top trunks
// (length, sinuosity, long-profile shape). Earth refs inline.

fn run_river_audit(world: &World, seed: u64, top: usize) {
    use hex3::world::Hydrology;
    let Some(fine) = world.fine.as_ref() else {
        println!("\n[RIVER AUDIT] no fine surface present");
        return;
    };
    let tess = fine.tessellation();
    let surf = fine.surface_for(u32::MAX);
    let hydro = &surf.hydrology;
    let elev = &surf.elevation.values;
    let n = tess.num_cells();
    let areas = tess.cell_areas();
    let r_km = EARTH_RADIUS_KM;

    println!(
        "\n================ RIVER AUDIT seed={} (fine eroded) ================",
        seed
    );

    let land_km2: f64 = (0..n)
        .filter(|&i| !hydro.is_submerged(i) && elev[i] >= 0.0)
        .map(|i| (areas[i] * r_km * r_km) as f64)
        .sum();
    // Continent scale reference for river lengths (computed once).
    let land_mask: Vec<bool> = (0..n)
        .map(|i| !hydro.is_submerged(i) && elev[i] >= 0.0)
        .collect();
    let continents = measure_components(tess, &land_mask);
    let cont_len = continents.first().map(|c| c.length_km).unwrap_or(0.0);

    // A/B the two renderer calibrations: LEGACY count-equivalent thresholds
    // (coarse-tuned; stub network on the adaptive fine mesh) vs the PHYSICAL
    // catchment-area thresholds (--river-min-catchment-km2, default 2000).
    let min_catchment = 2000.0f32;
    let legacy_all = (n as f32 * 0.00005).max(1.0) * hydro.mean_cell_discharge;
    let per_count = hydro.mean_cell_discharge.max(1e-12);
    let modes = [
        (
            "LEGACY (count-equivalent)".to_string(),
            legacy_all,
            (n as f32 * 0.004).max(1.0),
            (n as f32 * 0.0006).max(1.0),
        ),
        (
            format!("CATCHMENT ({min_catchment:.0} km² min)"),
            Hydrology::flow_for_catchment_km2(min_catchment),
            Hydrology::flow_for_catchment_km2(75.0 * min_catchment) / per_count,
            Hydrology::flow_for_catchment_km2(12.5 * min_catchment) / per_count,
        ),
    ];
    for (label, thr_all, outlet_count, branch_count) in modes {
        river_mode_panel(
            &label,
            tess,
            hydro,
            elev,
            &areas,
            land_km2,
            cont_len,
            thr_all,
            outlet_count,
            branch_count,
            top,
        );
    }
}

/// One river-network panel (density, Strahler/Horton, mouths, top trunks) for
/// a given 'All' flow threshold + Major outlet/branch count-equivalents.
#[allow(clippy::too_many_arguments)]
fn river_mode_panel(
    label: &str,
    tess: &hex3::world::Tessellation,
    hydro: &hex3::world::Hydrology,
    elev: &[f32],
    areas: &[f32],
    land_km2: f64,
    cont_len: f32,
    thr_all: f32,
    outlet_count: f32,
    branch_count: f32,
    top: usize,
) {
    let n = tess.num_cells();
    let r_km = EARTH_RADIUS_KM;
    let is_channel: Vec<bool> = (0..n)
        .map(|i| hydro.flow_accumulation[i] >= thr_all && !hydro.is_submerged(i))
        .collect();
    let is_major = hydro.compute_major_river_cells(outlet_count, branch_count);
    let net_len = |m: &dyn Fn(usize) -> bool| -> f64 {
        (0..n)
            .filter(|&i| m(i))
            .map(|i| (areas[i].sqrt() * r_km) as f64)
            .sum()
    };
    let all_len = net_len(&|i| is_channel[i]);
    let major_len = net_len(&|i| is_major[i]);
    println!("\n  [{label}]");
    println!(
        "    network ('All'):   {:>7} cells, ~{:>7.0} km total, Dd {:.4} km/km²   [Earth perennial-river Dd at map scale ~0.01-0.1]",
        is_channel.iter().filter(|&&c| c).count(),
        all_len,
        all_len / land_km2.max(1e-30)
    );
    println!(
        "    network ('Major'): {:>7} cells, ~{:>7.0} km total, Dd {:.4} km/km²",
        is_major.iter().filter(|&&c| c).count(),
        major_len,
        major_len / land_km2.max(1e-30)
    );

    // Upstream adjacency over the channel network + a topological order
    // (ascending flow accumulation is a valid topo order on the SFD tree).
    let mut ups: Vec<Vec<u32>> = vec![Vec::new(); n];
    for c in 0..n {
        if !is_channel[c] {
            continue;
        }
        if let Some(d) = hydro.downstream(c) {
            if is_channel[d] {
                ups[d].push(c as u32);
            }
        }
    }
    let mut chan: Vec<u32> = (0..n as u32).filter(|&i| is_channel[i as usize]).collect();
    chan.sort_by(|&a, &b| {
        hydro.flow_accumulation[a as usize].total_cmp(&hydro.flow_accumulation[b as usize])
    });

    // Strahler orders + Horton bifurcation ratio.
    let mut order = vec![0u8; n];
    for &c in &chan {
        let c = c as usize;
        let (mut best, mut ties) = (0u8, 0u32);
        for &u in &ups[c] {
            let o = order[u as usize];
            if o > best {
                best = o;
                ties = 1;
            } else if o == best {
                ties += 1;
            }
        }
        order[c] = if best == 0 {
            1
        } else if ties >= 2 {
            best + 1
        } else {
            best
        };
    }
    let max_order = chan.iter().map(|&c| order[c as usize]).max().unwrap_or(0);
    let mut streams = vec![0u32; max_order as usize + 1];
    for &c in &chan {
        let c = c as usize;
        let o = order[c];
        // A stream of order o starts where no upstream child has order o.
        if !ups[c].iter().any(|&u| order[u as usize] == o) {
            streams[o as usize] += 1;
        }
    }
    let mut rbs: Vec<f64> = Vec::new();
    for o in 1..max_order as usize {
        if streams[o + 1] > 0 {
            rbs.push(streams[o] as f64 / streams[o + 1] as f64);
        }
    }
    let rb = if rbs.is_empty() {
        0.0
    } else {
        (rbs.iter().map(|r| r.ln()).sum::<f64>() / rbs.len() as f64).exp()
    };
    let stream_counts: Vec<String> = (1..=max_order as usize)
        .map(|o| format!("N{}={}", o, streams[o]))
        .collect();
    println!(
        "    Strahler: max order {} | {} | Horton Rb = {:.1}   [Earth Rb 3-5]",
        max_order,
        stream_counts.join(" "),
        rb
    );

    // Mouth census: where do rivers end?
    let (mut ocean_m, mut lake_m, mut inland_m) = (0u32, 0u32, 0u32);
    let mut mouths: Vec<u32> = Vec::new();
    for &c in &chan {
        let c = c as usize;
        match hydro.downstream(c) {
            Some(d) if !is_channel[d] && hydro.is_submerged(d) => {
                if hydro.is_ocean(d) {
                    ocean_m += 1;
                } else {
                    lake_m += 1;
                }
                mouths.push(c as u32);
            }
            None => {
                inland_m += 1;
                mouths.push(c as u32);
            }
            _ => {}
        }
    }
    println!(
        "    mouths: {} to ocean, {} to lakes, {} inland dead-ends   [post-integration a major river should NOT dead-end inland]",
        ocean_m, lake_m, inland_m
    );

    // Top trunks by mouth discharge: walk upstream along the max-flow child.
    mouths.sort_by(|&a, &b| {
        hydro.flow_accumulation[b as usize].total_cmp(&hydro.flow_accumulation[a as usize])
    });
    println!(
        "    top rivers:    len_km  straight  sinuos  basin_km²  src_m  %drop-upper-half  mouth"
    );
    println!(
        "                   [largest continent extent {:.0} km; Earth: longest ~0.5-0.9x continent; graded profile drops 60-85% in the upper half]",
        cont_len
    );
    for (k, &mouth) in mouths.iter().take(top).enumerate() {
        let mouth = mouth as usize;
        // Trace the trunk upstream.
        let mut path = vec![mouth];
        let mut cur = mouth;
        loop {
            let next = ups[cur].iter().copied().max_by(|&a, &b| {
                hydro.flow_accumulation[a as usize].total_cmp(&hydro.flow_accumulation[b as usize])
            });
            match next {
                Some(u) => {
                    cur = u as usize;
                    path.push(cur);
                }
                None => break,
            }
        }
        // Chord-sum length (arc ≈ chord at cell scale; f32 acos is unusable here).
        let mut len_km = 0.0f64;
        for w in path.windows(2) {
            len_km += ((tess.cell_center(w[0]) - tess.cell_center(w[1])).length() * r_km) as f64;
        }
        let straight_km =
            (tess.cell_center(mouth) - tess.cell_center(*path.last().unwrap())).length() * r_km;
        let sinuosity = len_km / (straight_km as f64).max(1e-9);
        // Basin area: BFS the full upstream tree (all cells, not just channels).
        let mut basin_km2 = 0.0f64;
        {
            let mut stack = vec![mouth as u32];
            let mut seen = std::collections::HashSet::new();
            seen.insert(mouth as u32);
            while let Some(c) = stack.pop() {
                let c = c as usize;
                basin_km2 += (areas[c] * r_km * r_km) as f64;
                for &nb in tess.neighbors(c) {
                    if hydro.downstream(nb) == Some(c) && seen.insert(nb as u32) {
                        stack.push(nb as u32);
                    }
                }
            }
        }
        // Long-profile shape: % of total drop achieved in the upper half of length.
        let src = *path.last().unwrap();
        let drop = elev[src] - elev[mouth];
        let mut acc = 0.0f64;
        let mut mid_elev = elev[mouth];
        for w in path.windows(2) {
            acc += ((tess.cell_center(w[0]) - tess.cell_center(w[1])).length() * r_km) as f64;
            if acc >= len_km * 0.5 {
                mid_elev = elev[w[1]];
                break;
            }
        }
        // Meaningless on stub trunks (<2 cells) or near-zero total drop (<50 m).
        let upper_frac = if path.len() < 3 || drop < 0.005 {
            f32::NAN
        } else {
            100.0 * (elev[src] - mid_elev) / drop
        };
        let mouth_kind = match hydro.downstream(mouth) {
            Some(d) if hydro.is_ocean(d) => "ocean",
            Some(_) => "lake",
            None => "INLAND",
        };
        let upper_str = if upper_frac.is_nan() {
            "     -".to_string()
        } else {
            format!("{upper_frac:>5.0}%")
        };
        println!(
            "      {:>2}. {:>8.0}  {:>8.0}  {:>6.2}  {:>9.0}  {:>5.0}  {:>16}  {}",
            k + 1,
            len_km,
            straight_km,
            sinuosity,
            basin_km2,
            elev[src] * AUDIT_M_PER_UNIT,
            upper_str,
            mouth_kind
        );
    }
}
