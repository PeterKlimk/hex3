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
    /// Override lithologic erodibility contrast (exp-amplitude sigma). <0 = use
    /// the EROSION_LITHO_SIGMA default; 0 = uniform K (no lithology).
    #[arg(long, default_value_t = -1.0)]
    erosion_litho_sigma: f32,
    /// Override tectonic uplift scale (Hold & carve: uplift ~balances erosion to
    /// hold orogens while valleys grade). <0 = use the EROSION_UPLIFT_SCALE
    /// default; 0 = relaxation only (no uplift).
    #[arg(long, default_value_t = -1.0)]
    erosion_uplift_scale: f32,
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
    /// Override structural-grain erodibility strength (fold-belt ridge-and-valley).
    /// <0 = use EROSION_LITHO_GRAIN_STRENGTH default; 0 = no grain. Experimental.
    #[arg(long, default_value_t = -1.0)]
    litho_grain_strength: f32,
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
    if cli.erosion_litho_sigma >= 0.0 {
        world.erosion_params.litho_sigma = cli.erosion_litho_sigma;
    }
    if cli.erosion_uplift_scale >= 0.0 {
        world.erosion_params.uplift_scale = cli.erosion_uplift_scale;
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
    if cli.glacial_k >= 0.0 {
        world.erosion_params.glacial_k = cli.glacial_k;
    }
    if cli.litho_grain_strength >= 0.0 {
        world.erosion_params.litho_grain_strength = cli.litho_grain_strength;
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
    let land_precip: Vec<f32> = (0..n)
        .filter(|&i| land[i])
        .map(|i| precipitation[i])
        .collect();
    let arid =
        land_precip.iter().filter(|&&p| p < 0.35).count() as f32 / land_precip.len().max(1) as f32;
    let humid =
        land_precip.iter().filter(|&&p| p > 1.5).count() as f32 / land_precip.len().max(1) as f32;
    println!("\n-- Climate --   [Earth: ~33% of land arid/semi-arid]");
    println!(
        "  land precip: arid {:.0}%  humid {:.0}%  | lakes {:.2}% of surface",
        100.0 * arid,
        100.0 * humid,
        100.0
            * hydrology
                .water_bodies
                .iter()
                .map(|wb| wb.cells.len())
                .sum::<usize>() as f32
            / n as f32
    );
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

    // ---- River concavity (population slope-area) ----
    // Detachment-limited stream power at steady state gives channel slope
    // S ~ A^(-theta), theta = m/n (~0.5 here): concave-up graded rivers. Rather
    // than trace stems (which lakes truncate), use the standard population
    // method: over every channel cell (flow above a support threshold, draining
    // downhill) take S = drop/dist to its receiver and A = flow, bin by ln(A),
    // take the median ln(S) per bin, and regress bin medians -> theta = -slope.
    // Median bins are robust to lakes/outliers. theta<=0 (convex/flat) = rivers
    // not graded (erosion too weak / K too low).
    {
        // Count-equivalent flow (hydrology stores physical precip×area discharge),
        // so the absolute channel-support threshold stays in upstream-cell units.
        let flow: Vec<f32> = (0..n).map(|i| hydrology.flow_count_equiv(i)).collect();
        let drainage = &hydrology.drainage_dir;
        let channel_thresh = 50.0f32;
        let mut pts: Vec<(f32, f32)> = Vec::new(); // (ln A, ln S)
        for c in 0..n {
            if !land[c] || flow[c] < channel_thresh {
                continue;
            }
            let Some(d) = drainage[c] else { continue };
            let dz = elevation[c] - elevation[d];
            let dx = (tess.cell_center(c) - tess.cell_center(d)).length() * EARTH_RADIUS_KM;
            if dz <= 0.0 || dx <= 0.0 {
                continue;
            }
            pts.push((flow[c].ln(), (dz / dx).ln()));
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
                "  channel cells {} | bins fitted {} | theta = {:+.2}  ({})",
                pts.len(),
                k,
                theta,
                if theta > 0.15 {
                    "concave-up / graded"
                } else if theta > -0.15 {
                    "~flat (under-graded)"
                } else {
                    "convex (not graded)"
                },
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
}
