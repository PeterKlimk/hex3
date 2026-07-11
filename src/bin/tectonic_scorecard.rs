//! Cross-seed/resolution promotion scorecard for moving-carrier tectonics.
//!
//! This is intentionally coarse-only: it asks whether the upstream geological
//! state is stable and attributable before fine erosion or rendering can hide it.

use std::time::Instant;

use clap::Parser;
use hex3::world::diagnostics::{measure_components, EARTH_RADIUS_KM};
use hex3::world::{
    CarrierOperatorAudit, OrogenModel, TectonicCarrierConfig, World, NUM_PLATES_DEFAULT,
};

const RANGE_ELEV: f32 = 0.15;
const SIGNIFICANT_RANGE_KM2: f32 = 20_000.0;

#[derive(Parser, Debug)]
#[command(
    name = "tectonic-scorecard",
    about = "Cross-seed and carrier-resolution audit for moving tectonics"
)]
struct Cli {
    /// Present terrain-mesh cell count. Carrier resolution is swept separately.
    #[arg(long, default_value_t = 100_000)]
    cells: usize,

    /// Seeds to audit, comma separated.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "12345,777,4242,9001,314159,271828,8675309,20260711,42,1001"
    )]
    seeds: Vec<u64>,

    /// Fixed-carrier resolutions to compare, comma separated.
    #[arg(long, value_delimiter = ',', default_value = "4096,8192,16384")]
    carrier_cells: Vec<usize>,

    /// Geological snapshot interval in Myr.
    #[arg(long, default_value_t = 2.0)]
    carrier_step_myr: f32,

    /// Duration for which the generated Euler motions remain coherent.
    /// The product/default clock is 100 Myr; this override is scorecard-only.
    #[arg(long, default_value_t = 100.0)]
    lookback_myr: f32,

    /// Skip the legacy reset baseline rows.
    #[arg(long, default_value_t = false)]
    no_legacy: bool,

    /// Experimental coarse denudation capacity in km/Myr (numerically equal
    /// to mm/yr). Zero preserves the tectonic-only carrier exactly.
    #[arg(long, default_value_t = 0.0)]
    denudation_rate_km_per_myr: f32,

    /// Skip the expensive frozen/operator attribution ladder.
    #[arg(long, default_value_t = false)]
    no_operator_audit: bool,

    /// Mean plate-motion coherence time in Myr. Zero keeps the present Euler
    /// vectors constant through history (the existing/default behavior).
    #[arg(long, default_value_t = 0.0)]
    motion_coherence_myr: f32,

    /// Run the forward plate/crust lifecycle instead of the backward-history
    /// carrier evolution. Identity-gated; legacy rows remain unchanged.
    #[arg(long, default_value_t = false)]
    lifecycle: bool,
}

#[derive(Clone, Debug)]
struct Metrics {
    seed: u64,
    model: &'static str,
    carrier_cells: usize,
    lookback_myr: f32,
    motion_coherence_myr: f32,
    mean_reorganizations_per_plate: f32,
    mean_plate_speed_fraction: f32,
    mean_integrated_rotation_rad: f32,
    wall_s: f32,
    carrier_s: f32,
    evolution_s: f32,
    peak_km: f32,
    land_pct: f32,
    mountain_land_pct: f32,
    land_p50_km: f32,
    land_p90_km: f32,
    land_p99_km: f32,
    ranges: usize,
    width_p50_km: f32,
    elong_p50: f32,
    cap500_p50_km2: f32,
    flat_cap_p50_pct: f32,
    positive_work: f64,
    negative_work: f64,
    arc_share_positive_pct: f32,
    inherited_outside_active_pct: f32,
    inherited_active_cosine: f32,
    moving_forcing_pct: f32,
    gap_pct: f32,
    overlap_pct: f32,
    mass_relative_residual: f64,
    denudation_rate_km_per_myr: f32,
    material_removed: f64,
    operator: Option<CarrierOperatorAudit>,
    lifecycle_created_ocean: f64,
    lifecycle_consumed_ocean: f64,
    lifecycle_underthrust: f64,
    lifecycle_foundered: f64,
    lifecycle_underthrust_footprint: f64,
    lifecycle_max_layer_fraction: f32,
    lifecycle_magma: f64,
    lifecycle_sutures: usize,
    lifecycle_merges: usize,
    lifecycle_final_plates: usize,
    lifecycle_max_underthrust: f32,
    lifecycle_max_magma: f32,
    lifecycle_max_remap: f32,
}

fn main() {
    let cli = Cli::parse();
    validate(&cli);
    let started = Instant::now();
    let mut rows = Vec::new();

    for &seed in &cli.seeds {
        if !cli.no_legacy {
            rows.push(run_world(seed, cli.cells, None, None, false));
        }
        for &carrier_cells in &cli.carrier_cells {
            rows.push(run_world(
                seed,
                cli.cells,
                Some(TectonicCarrierConfig {
                    cells: carrier_cells,
                    step_myr: cli.carrier_step_myr,
                    operator_audit: !cli.no_operator_audit,
                    denudation_rate_km_per_myr: cli.denudation_rate_km_per_myr,
                    motion_coherence_myr: cli.motion_coherence_myr,
                }),
                Some(cli.lookback_myr),
                cli.lifecycle,
            ));
        }
    }

    print_rows(&rows);
    print_operator_ladder(&rows, &cli.carrier_cells);
    print_convergence(&rows, &cli.carrier_cells);
    print_lifecycle_summary(&rows);
    println!(
        "\nscorecard wall time: {:.2}s ({} worlds; rendering/fine erosion excluded)",
        started.elapsed().as_secs_f32(),
        rows.len()
    );
}

fn validate(cli: &Cli) {
    assert!(!cli.seeds.is_empty(), "at least one seed is required");
    assert!(
        !cli.carrier_cells.is_empty(),
        "at least one carrier resolution is required"
    );
    assert!(cli.carrier_step_myr > 0.0);
    assert!(cli.lookback_myr > 0.0);
    assert!(cli.denudation_rate_km_per_myr >= 0.0);
    assert!(cli.motion_coherence_myr >= 0.0);
    for &cells in &cli.carrier_cells {
        assert!(
            (64..=u16::MAX as usize).contains(&cells),
            "carrier cell count must fit the compact u16 replay: {cells}"
        );
    }
}

fn run_world(
    seed: u64,
    cells: usize,
    carrier: Option<TectonicCarrierConfig>,
    lookback_myr: Option<f32>,
    lifecycle: bool,
) -> Metrics {
    let started = Instant::now();
    let mut world = World::new(seed, cells, 1);
    world.orogen_model = if carrier.is_some() {
        if lifecycle {
            OrogenModel::HistoryCarrierLifecycle
        } else {
            OrogenModel::HistoryCarrierEvolved
        }
    } else {
        OrogenModel::Legacy
    };
    if let Some(config) = carrier {
        world.tectonic_carrier_config = config;
    }
    world.generate_plates(NUM_PLATES_DEFAULT);
    world.generate_crust();
    world.generate_dynamics();
    if let Some(lookback_myr) = lookback_myr {
        world
            .dynamics
            .as_mut()
            .expect("dynamics generated above")
            .clock
            .lookback_myr = lookback_myr;
    }
    world.generate_features();
    world.generate_elevation();
    measure_world(&world, started.elapsed().as_secs_f32())
}

fn measure_world(world: &World, wall_s: f32) -> Metrics {
    let tess = &world.tessellation;
    let areas = tess.cell_areas();
    let elevation = &world.elevation.as_ref().unwrap().values;
    let features = world.features.as_ref().unwrap();
    let r2 = EARTH_RADIUS_KM * EARTH_RADIUS_KM;

    let total_area: f64 = areas.iter().map(|&area| area as f64).sum();
    let land_area: f64 = (0..elevation.len())
        .filter(|&i| elevation[i] >= 0.0)
        .map(|i| areas[i] as f64)
        .sum();
    let mountain_area: f64 = (0..elevation.len())
        .filter(|&i| elevation[i] >= RANGE_ELEV)
        .map(|i| areas[i] as f64)
        .sum();
    let peak_km = elevation.iter().copied().fold(f32::NEG_INFINITY, f32::max) * 10.0;

    let land_samples: Vec<_> = (0..elevation.len())
        .filter(|&i| elevation[i] >= 0.0)
        .map(|i| (elevation[i] * 10.0, areas[i]))
        .collect();
    let land_p50_km = weighted_quantile(&land_samples, 0.50);
    let land_p90_km = weighted_quantile(&land_samples, 0.90);
    let land_p99_km = weighted_quantile(&land_samples, 0.99);

    let range_mask: Vec<bool> = elevation.iter().map(|&value| value >= RANGE_ELEV).collect();
    let components = measure_components(tess, &range_mask);
    let significant: Vec<_> = components
        .iter()
        .filter(|component| component.area_km2 >= SIGNIFICANT_RANGE_KM2)
        .collect();
    let mut widths: Vec<_> = significant
        .iter()
        .map(|component| component.width_km)
        .collect();
    let mut elongations: Vec<_> = significant
        .iter()
        .map(|component| component.elongation())
        .collect();
    let mut cap500 = Vec::new();
    let mut flat_caps = Vec::new();
    for component in &significant {
        let component_peak = component
            .cells
            .iter()
            .map(|&cell| elevation[cell])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut cap_area = 0.0f64;
        let mut flat_area = 0.0f64;
        for &cell in &component.cells {
            if elevation[cell] < component_peak - 0.05 {
                continue;
            }
            let area_km2 = areas[cell] as f64 * r2 as f64;
            cap_area += area_km2;
            let center = tess.cell_center(cell);
            let max_downhill = tess
                .neighbors(cell)
                .iter()
                .map(|&next| {
                    let distance =
                        (center - tess.cell_center(next)).length().max(1e-9) * EARTH_RADIUS_KM;
                    (elevation[cell] - elevation[next]).max(0.0) / distance
                })
                .fold(0.0f32, f32::max);
            if max_downhill < 1.0e-3 {
                flat_area += area_km2;
            }
        }
        cap500.push(cap_area as f32);
        flat_caps.push(if cap_area > 0.0 {
            (100.0 * flat_area / cap_area) as f32
        } else {
            0.0
        });
    }
    for values in [&mut widths, &mut elongations, &mut cap500, &mut flat_caps] {
        values.sort_by(f32::total_cmp);
    }

    let thickening = &features.thin_sheet_thickness_delta;
    let uplift = &features.tectonic_uplift_rate;
    let positive_work: f64 = thickening
        .iter()
        .zip(areas.iter())
        .map(|(&delta, &area)| delta.max(0.0) as f64 * area as f64)
        .sum();
    let negative_work: f64 = thickening
        .iter()
        .zip(areas.iter())
        .map(|(&delta, &area)| (-delta).max(0.0) as f64 * area as f64)
        .sum();
    let arc_share_positive_pct = if positive_work > 0.0 {
        (100.0 * features.thin_sheet_material_added / positive_work) as f32
    } else {
        0.0
    };
    let active_support = top_positive_support(uplift, &areas, 0.90);
    let inherited_outside: f64 = (0..thickening.len())
        .filter(|&i| !active_support[i])
        .map(|i| thickening[i].max(0.0) as f64 * areas[i] as f64)
        .sum();
    let inherited_outside_active_pct = if positive_work > 0.0 {
        (100.0 * inherited_outside / positive_work) as f32
    } else {
        0.0
    };
    let inherited_active_cosine = area_weighted_positive_cosine(thickening, uplift, &areas);

    let (
        carrier_cells,
        carrier_s,
        gap_pct,
        overlap_pct,
        motion_coherence_myr,
        mean_reorganizations_per_plate,
        mean_plate_speed_fraction,
        mean_integrated_rotation_rad,
    ) = world
        .tectonic_history
        .as_ref()
        .and_then(|history| history.carrier_replay.as_ref())
        .map(|replay| {
            let snapshots = replay.snapshots.len().max(1) as f64;
            let cells = replay.num_cells.max(1) as f64;
            let gaps: f64 = replay
                .snapshots
                .iter()
                .map(|snapshot| snapshot.gap_cells as f64)
                .sum();
            let overlaps: f64 = replay
                .snapshots
                .iter()
                .map(|snapshot| snapshot.overlap_excess as f64)
                .sum();
            (
                replay.num_cells,
                replay.build_seconds,
                (100.0 * gaps / (snapshots * cells)) as f32,
                (100.0 * overlaps / (snapshots * cells)) as f32,
                replay.motion_coherence_myr,
                replay.mean_reorganizations_per_plate,
                replay.mean_plate_speed_fraction,
                replay.mean_integrated_rotation_rad,
            )
        })
        .unwrap_or((0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    let residual_denominator = (features.thin_sheet_material_added
        + features.thin_sheet_material_removed)
        .abs()
        .max(1e-30);

    Metrics {
        seed: world.seed,
        model: if world.orogen_model == OrogenModel::Legacy {
            "legacy"
        } else if world.orogen_model == OrogenModel::HistoryCarrierLifecycle {
            "lifecycle"
        } else {
            "evolved"
        },
        carrier_cells,
        lookback_myr: world
            .dynamics
            .as_ref()
            .map(|dynamics| dynamics.clock.lookback_myr)
            .unwrap_or(0.0),
        motion_coherence_myr,
        mean_reorganizations_per_plate,
        mean_plate_speed_fraction,
        mean_integrated_rotation_rad,
        wall_s,
        carrier_s,
        evolution_s: features.carrier_evolution_seconds,
        peak_km,
        land_pct: (100.0 * land_area / total_area.max(1e-30)) as f32,
        mountain_land_pct: (100.0 * mountain_area / land_area.max(1e-30)) as f32,
        land_p50_km,
        land_p90_km,
        land_p99_km,
        ranges: significant.len(),
        width_p50_km: median(&widths),
        elong_p50: median(&elongations),
        cap500_p50_km2: median(&cap500),
        flat_cap_p50_pct: median(&flat_caps),
        positive_work,
        negative_work,
        arc_share_positive_pct,
        inherited_outside_active_pct,
        inherited_active_cosine,
        moving_forcing_pct: 100.0 * features.carrier_moving_forcing_fraction,
        gap_pct,
        overlap_pct,
        mass_relative_residual: features.thin_sheet_material_residual.abs() / residual_denominator,
        denudation_rate_km_per_myr: world
            .tectonic_history
            .as_ref()
            .and_then(|history| history.carrier_replay.as_ref())
            .map(|replay| replay.denudation_rate_km_per_myr)
            .unwrap_or(0.0),
        material_removed: features.thin_sheet_material_removed,
        operator: features.carrier_operator_audit,
        lifecycle_created_ocean: features
            .lifecycle_audit
            .as_ref()
            .map(|audit| audit.created_ocean_volume)
            .unwrap_or(0.0),
        lifecycle_consumed_ocean: features
            .lifecycle_audit
            .as_ref()
            .map(|audit| audit.consumed_ocean_volume)
            .unwrap_or(0.0),
        lifecycle_underthrust: features
            .lifecycle_audit
            .as_ref()
            .map(|audit| audit.continental_underthrust_volume)
            .unwrap_or(0.0),
        lifecycle_foundered: features
            .lifecycle_audit
            .as_ref()
            .map(|audit| audit.foundered_continental_volume)
            .unwrap_or(0.0),
        lifecycle_underthrust_footprint: features
            .lifecycle_audit
            .as_ref()
            .map(|audit| audit.underthrust_footprint_area_sr)
            .unwrap_or(0.0),
        lifecycle_max_layer_fraction: features
            .lifecycle_audit
            .as_ref()
            .map(|audit| audit.max_underthrust_layer_fraction)
            .unwrap_or(0.0),
        lifecycle_magma: features
            .lifecycle_audit
            .as_ref()
            .map(|audit| audit.magmatic_added_volume)
            .unwrap_or(0.0),
        lifecycle_sutures: features
            .lifecycle_audit
            .as_ref()
            .map(|audit| audit.active_sutures)
            .unwrap_or(0),
        lifecycle_merges: features
            .lifecycle_audit
            .as_ref()
            .map(|audit| audit.plate_merges)
            .unwrap_or(0),
        lifecycle_final_plates: features
            .lifecycle_audit
            .as_ref()
            .map(|audit| audit.final_plate_count)
            .unwrap_or(0),
        lifecycle_max_underthrust: features
            .lifecycle_audit
            .as_ref()
            .map(|audit| audit.max_underthrust_thickness)
            .unwrap_or(0.0),
        lifecycle_max_magma: features
            .lifecycle_audit
            .as_ref()
            .map(|audit| audit.max_magma_thickness)
            .unwrap_or(0.0),
        lifecycle_max_remap: features
            .lifecycle_audit
            .as_ref()
            .map(|audit| audit.max_remap_thickness)
            .unwrap_or(0.0),
    }
}

fn print_lifecycle_summary(rows: &[Metrics]) {
    let lifecycle: Vec<_> = rows.iter().filter(|row| row.model == "lifecycle").collect();
    if lifecycle.is_empty() {
        return;
    }
    println!("\n## Forward lifecycle ledgers");
    println!("\n| seed | created ocean | consumed ocean | underthrust/foundered | sheet area | max layer | magma | sutures | merges | final plates | max H underthrust/magma/remap | evolve s |");
    println!("|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|:---|---:|");
    for row in lifecycle {
        println!(
            "| {} | {:.3e} | {:.3e} | {:.3e}/{:.3e} | {:.3e} | {:.3} | {:.3e} | {} | {} | {} | {:.3}/{:.3}/{:.3} | {:.2} |",
            row.seed,
            row.lifecycle_created_ocean,
            row.lifecycle_consumed_ocean,
            row.lifecycle_underthrust,
            row.lifecycle_foundered,
            row.lifecycle_underthrust_footprint,
            row.lifecycle_max_layer_fraction,
            row.lifecycle_magma,
            row.lifecycle_sutures,
            row.lifecycle_merges,
            row.lifecycle_final_plates,
            row.lifecycle_max_underthrust,
            row.lifecycle_max_magma,
            row.lifecycle_max_remap,
            row.evolution_s,
        );
    }
}

fn print_rows(rows: &[Metrics]) {
    println!("# Tectonic promotion scorecard\n");
    println!("Absolute gates: peak >14 km = FAIL, >12 km = WARN; mass residual >1e-4 = FAIL.");
    println!("Magma/+work may exceed 100% when denudation leaves net negative inherited work.");
    println!("Motion τ=0 denotes the constant-motion/infinite-coherence identity path.");
    println!("Active support is the smallest cell set carrying 90% of positive present uplift.\n");
    println!("| seed | model | carrier | history Myr | motion τ/reorg/speed/path | denude km/Myr | peak km | land % | mountain-land % | land p50/p90/p99 km | ranges | width p50 km | elong p50 | cap500 p50 km² | flat-cap p50 % | +work/-work | +work/Myr | removed | magma/+work % | inherited outside active % | inherited·active cos | moved forcing % | gap/overlap % | mass residual | carrier/evolve/wall s | gate |");
    println!("|---:|:---|---:|---:|:---|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|:---|---:|:---|:---|");
    for row in rows {
        let gate = if row.mass_relative_residual > 1e-4 || row.peak_km > 14.0 {
            "FAIL"
        } else if row.peak_km > 12.0 {
            "WARN"
        } else {
            "PASS"
        };
        println!(
            "| {} | {} | {} | {:.0} | {:.0}/{:.2}/{:.2}/{:.3} | {:.3} | {:.2} | {:.1} | {:.1} | {:.2}/{:.2}/{:.2} | {} | {:.0} | {:.1} | {:.0} | {:.1} | {:.3e}/{:.3e} | {:.3e} | {:.3e} | {:.1} | {:.1} | {:.3} | {:.1} | {:.1}/{:.1} | {:.2e} | {:.2}/{:.2}/{:.2} | {} |",
            row.seed,
            row.model,
            if row.carrier_cells == 0 { "-".to_string() } else { row.carrier_cells.to_string() },
            row.lookback_myr,
            row.motion_coherence_myr,
            row.mean_reorganizations_per_plate,
            row.mean_plate_speed_fraction,
            row.mean_integrated_rotation_rad,
            row.denudation_rate_km_per_myr,
            row.peak_km,
            row.land_pct,
            row.mountain_land_pct,
            row.land_p50_km,
            row.land_p90_km,
            row.land_p99_km,
            row.ranges,
            row.width_p50_km,
            row.elong_p50,
            row.cap500_p50_km2,
            row.flat_cap_p50_pct,
            row.positive_work,
            row.negative_work,
            row.positive_work / row.lookback_myr.max(f32::EPSILON) as f64,
            row.material_removed,
            row.arc_share_positive_pct,
            row.inherited_outside_active_pct,
            row.inherited_active_cosine,
            row.moving_forcing_pct,
            row.gap_pct,
            row.overlap_pct,
            row.mass_relative_residual,
            row.carrier_s,
            row.evolution_s,
            row.wall_s,
            gate,
        );
    }
}

fn print_operator_ladder(rows: &[Metrics], requested_resolutions: &[usize]) {
    println!("\n## Operator isolation ladder\n");
    println!("Volumes are thickness×steradian; maxima are thickness units before isostasy. Projection residual compares projected with carrier-native net volume.\n");
    println!("| seed | carrier | boundary km | swept km²/Myr | support % | target L1 | arc rate | one-step total/transport/magma max | one-step +/- | frozen-100 max/+/- | moving-native max/+/- | projected max/+/- | projection net residual |");
    println!("|---:|---:|---:|---:|---:|---:|---:|:---|:---|:---|:---|:---|---:|");
    for row in rows {
        let Some(audit) = row.operator else {
            continue;
        };
        println!(
            "| {} | {} | {:.0} | {:.0} | {:.1} | {:.3e} | {:.3e} | {:.3}/{:.3}/{:.3} | {:.3e}/{:.3e} | {:.3}/{:.3e}/{:.3e} | {:.3}/{:.3e}/{:.3e} | {:.3}/{:.3e}/{:.3e} | {:+.3e} |",
            row.seed,
            row.carrier_cells,
            audit.mean_boundary_length_km,
            audit.mean_convergent_swept_area_km2_per_myr,
            audit.mean_boundary_support_pct,
            audit.mean_target_l1,
            audit.mean_arc_addition_rate,
            audit.one_step_max,
            audit.one_step_transport_max,
            audit.one_step_magma_max,
            audit.one_step_positive,
            audit.one_step_negative,
            audit.frozen_max,
            audit.frozen_positive,
            audit.frozen_negative,
            audit.moving_max,
            audit.moving_positive,
            audit.moving_negative,
            audit.projected_max,
            audit.projected_positive,
            audit.projected_negative,
            audit.projection_net_residual,
        );
    }

    println!("\n### Maximum-thickness span by rung\n");
    println!("| seed | resolutions | one step | frozen 100 Myr | moving native | projected | first divergent rung |");
    println!("|---:|:---|---:|---:|---:|---:|:---|");
    let mut seeds: Vec<_> = rows.iter().map(|row| row.seed).collect();
    seeds.sort_unstable();
    seeds.dedup();
    for seed in seeds {
        let audits: Vec<_> = rows
            .iter()
            .filter(|row| row.seed == seed)
            .filter_map(|row| row.operator)
            .collect();
        if audits.len() < 2 {
            continue;
        }
        let one = span(audits.iter().map(|audit| audit.one_step_max));
        let frozen = span(audits.iter().map(|audit| audit.frozen_max));
        let moving = span(audits.iter().map(|audit| audit.moving_max));
        let projected = span(audits.iter().map(|audit| audit.projected_max));
        let first = if one > 0.10 {
            "boundary/one-step"
        } else if frozen > 0.10 {
            "time integration"
        } else if moving > 0.10 {
            "parcel remap"
        } else if projected > 0.10 {
            "projection"
        } else {
            "none >0.1"
        };
        println!(
            "| {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {} |",
            seed,
            requested_resolutions
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("/"),
            one,
            frozen,
            moving,
            projected,
            first,
        );
    }
}

fn print_convergence(rows: &[Metrics], requested_resolutions: &[usize]) {
    println!("\n## Carrier-resolution convergence\n");
    println!(
        "FAIL: peak drift >2 km or land drift >2 percentage points. WARN at >1 km / >1 point.\n"
    );
    println!("| seed | resolutions | peak span km | land span points | mountain-land span points | inherited span points | verdict |");
    println!("|---:|:---|---:|---:|---:|---:|:---|");
    let mut seeds: Vec<_> = rows.iter().map(|row| row.seed).collect();
    seeds.sort_unstable();
    seeds.dedup();
    for seed in seeds {
        let evolved: Vec<_> = rows
            .iter()
            .filter(|row| row.seed == seed && row.model == "evolved")
            .collect();
        if evolved.len() < 2 {
            continue;
        }
        let peak_span = span(evolved.iter().map(|row| row.peak_km));
        let land_span = span(evolved.iter().map(|row| row.land_pct));
        let mountain_span = span(evolved.iter().map(|row| row.mountain_land_pct));
        let inherited_span = span(evolved.iter().map(|row| row.inherited_outside_active_pct));
        let verdict = if peak_span > 2.0 || land_span > 2.0 {
            "FAIL"
        } else if peak_span > 1.0 || land_span > 1.0 {
            "WARN"
        } else {
            "PASS"
        };
        println!(
            "| {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {} |",
            seed,
            requested_resolutions
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("/"),
            peak_span,
            land_span,
            mountain_span,
            inherited_span,
            verdict,
        );
    }
}

fn weighted_quantile(samples: &[(f32, f32)], quantile: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = sorted.iter().map(|&(_, weight)| weight as f64).sum();
    let target = total * quantile.clamp(0.0, 1.0) as f64;
    let mut cumulative = 0.0f64;
    for (value, weight) in sorted {
        cumulative += weight as f64;
        if cumulative >= target {
            return value;
        }
    }
    samples.last().unwrap().0
}

fn median(values: &[f32]) -> f32 {
    values
        .get(values.len().saturating_sub(1) / 2)
        .copied()
        .unwrap_or(0.0)
}

fn top_positive_support(field: &[f32], areas: &[f32], fraction: f64) -> Vec<bool> {
    let mut contributions: Vec<_> = field
        .iter()
        .zip(areas.iter())
        .enumerate()
        .filter_map(|(i, (&value, &area))| (value > 0.0).then_some((i, value as f64 * area as f64)))
        .collect();
    contributions.sort_by(|a, b| b.1.total_cmp(&a.1));
    let total: f64 = contributions.iter().map(|&(_, value)| value).sum();
    let mut support = vec![false; field.len()];
    let mut accumulated = 0.0f64;
    for (cell, value) in contributions {
        if accumulated >= fraction * total {
            break;
        }
        support[cell] = true;
        accumulated += value;
    }
    support
}

fn area_weighted_positive_cosine(a: &[f32], b: &[f32], areas: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut aa = 0.0f64;
    let mut bb = 0.0f64;
    for ((&a, &b), &area) in a.iter().zip(b.iter()).zip(areas.iter()) {
        let a = a.max(0.0) as f64;
        let b = b.max(0.0) as f64;
        let weight = area as f64;
        dot += weight * a * b;
        aa += weight * a * a;
        bb += weight * b * b;
    }
    if aa > 0.0 && bb > 0.0 {
        (dot / (aa * bb).sqrt()) as f32
    } else {
        0.0
    }
}

fn span(values: impl Iterator<Item = f32>) -> f32 {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for value in values {
        min = min.min(value);
        max = max.max(value);
    }
    if min.is_finite() && max.is_finite() {
        max - min
    } else {
        0.0
    }
}
