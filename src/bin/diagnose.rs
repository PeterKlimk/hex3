//! World diagnostics: generate a world and measure its features numerically.
//!
//! The primary instrument for judging physical plausibility of generated
//! artifacts — sizes in km/km² with Earth references alongside. Catches what
//! aggregate stats and rendered images both miss.
//!
//!     cargo run --release --bin diagnose -- --seed 12345
//!     cargo run --release --bin diagnose -- --seed 12345 --cells 40000

use clap::Parser;
use hex3::world::diagnostics::{distance_from_mask, measure_components, EARTH_RADIUS_KM};
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
    /// Fine-mesh cell cap (the emergent count is coarsened to fit). Lower it to
    /// iterate faster on erosion/roughness probes. 0 = use the FINE_MAX_CELLS default.
    #[arg(long, default_value_t = 0)]
    fine_max: usize,
    /// Override erosion erodibility K. <0 = use the EROSION_K default. Lets you
    /// sweep erosion strength without a recompile (FineBase is cached).
    #[arg(long, default_value_t = -1.0)]
    erosion_k: f32,
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
    if cli.erosion_uplift_smooth >= 0.0 {
        world.erosion_params.uplift_smooth_km = cli.erosion_uplift_smooth;
    }
    if cli.erosion_hillslope_crit >= 0.0 {
        world.erosion_params.hillslope_critical_slope = cli.erosion_hillslope_crit;
    }
    if cli.erosion_orographic_strength >= 0.0 {
        world.erosion_params.orographic_precip_strength = cli.erosion_orographic_strength;
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
