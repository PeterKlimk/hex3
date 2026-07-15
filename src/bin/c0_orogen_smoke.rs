//! Minimal CPU-only C0 U/L research smoke run.
//!
//! This executable deliberately freezes a provisional, dimensioned response
//! regime without changing the zero-denudation C0 library default. It is not a
//! rendering path or a promoted product model.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, ValueEnum};
use serde::Serialize;

use hex3::world::landscape::{
    linked_low_relief_initial_surface, linked_scenario, uniform_scenario, C0DischargeSupport,
    C0DischargeSupportArm, C0LandscapeParams, C0LandscapeSolver, C0LandscapeState,
    C0TimestepLimiter, ConservativeHillslopeParams, EffectiveArealDenudationParams, LandscapeMesh,
};

const WIDTH_KM: f64 = 960.0;
const HEIGHT_KM: f64 = 640.0;
const INITIAL_SEED: u64 = 12_345;
const TRUTH_LIMIT: &str = "Research smoke only: elevation is a finite-volume cell mean and fluvial lowering is coarse-grained effective areal denudation. Channels, channel beds and widths, gorges, sediment, lateral erosion, and valley geometry are unresolved; K is provisional and the C0 library/product default remains zero.";

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
enum CaseArg {
    U,
    L,
}

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
enum DischargeSupportArg {
    Unfiltered,
    Q16,
}

impl DischargeSupportArg {
    fn solver_support(self) -> C0DischargeSupport {
        match self {
            Self::Unfiltered => C0DischargeSupport::Unfiltered,
            Self::Q16 => C0DischargeSupport::fixed_helmholtz(16.0),
        }
    }
}

impl CaseArg {
    fn id(self) -> &'static str {
        match self {
            Self::U => "U",
            Self::L => "L",
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "c0_orogen_smoke",
    about = "Run an isolated CPU-only C0 U/L research smoke"
)]
struct Cli {
    /// Causal forcing case: U (uniform block) or L (linked segments).
    #[arg(long = "case", value_enum, ignore_case = true, default_value = "L")]
    case: CaseArg,

    /// Approximate full-hex center spacing in kilometres.
    #[arg(long, default_value_t = 8.0)]
    spacing_km: f64,

    /// Simulated end time in millions of years.
    #[arg(long, default_value_t = 1.0)]
    end_myr: f64,

    /// Requested maximum integration step in millions of years.
    #[arg(long, default_value_t = 0.01)]
    requested_dt_myr: f64,

    /// Denudation-intensity representation: raw C0-V or preregistered C0-Q16.
    #[arg(long, value_enum, default_value = "unfiltered")]
    discharge_support: DischargeSupportArg,

    /// Optional path for the complete JSON summary.
    #[arg(long)]
    output_json: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct FrozenParameters {
    runoff_depth_rate_km_myr: f64,
    discharge_exponent_m: f64,
    slope_exponent_n: f64,
    reference_specific_discharge_q0_km2_myr: f64,
    reference_slope_s0: f64,
    reference_denudation_e0_km_myr: f64,
    derived_k_per_km: f64,
    hillslope_diffusivity_km2_myr: f64,
    hillslope_critical_slope: f64,
}

impl FrozenParameters {
    fn values() -> Self {
        Self {
            runoff_depth_rate_km_myr: 500.0,
            discharge_exponent_m: 1.0,
            slope_exponent_n: 1.0,
            reference_specific_discharge_q0_km2_myr: 50_000.0,
            reference_slope_s0: 0.02,
            reference_denudation_e0_km_myr: 0.1,
            derived_k_per_km: 1.0e-4,
            hillslope_diffusivity_km2_myr: 0.1,
            hillslope_critical_slope: 0.7,
        }
    }

    fn solver_params(&self, discharge_support: C0DischargeSupport) -> C0LandscapeParams {
        C0LandscapeParams {
            effective_areal_denudation: EffectiveArealDenudationParams {
                k: self.derived_k_per_km,
                discharge_exponent_m: self.discharge_exponent_m,
                slope_exponent_n: self.slope_exponent_n,
            },
            runoff_depth_rate_km_myr: self.runoff_depth_rate_km_myr,
            discharge_support,
            hillslope: ConservativeHillslopeParams {
                diffusivity_km2_myr: self.hillslope_diffusivity_km2_myr,
                critical_slope_grade: self.hillslope_critical_slope,
                ..ConservativeHillslopeParams::default()
            },
            ..C0LandscapeParams::default()
        }
    }
}

#[derive(Debug, Default, Serialize)]
struct LimiterCounts {
    requested: u64,
    uplift_accuracy: u64,
    effective_denudation_accuracy: u64,
    effective_denudation_slope_courant: u64,
    hillslope_stability: u64,
}

#[derive(Debug, Serialize)]
struct SmokeSummary {
    schema: &'static str,
    case: &'static str,
    nominal_width_km: f64,
    nominal_height_km: f64,
    spacing_km: f64,
    cell_count: usize,
    actual_domain_area_km2: f64,
    initial_seed: u64,
    final_time_myr: f64,
    requested_dt_myr: f64,
    accepted_steps: u64,
    accepted_dt_min_myr: f64,
    accepted_dt_max_myr: f64,
    total_adaptive_attempts: u64,
    maximum_attempts_for_one_step: u32,
    limiter_counts: LimiterCounts,
    minimum_elevation_km: f64,
    maximum_elevation_km: f64,
    relief_km: f64,
    initial_elevation_volume_moment_km3: f64,
    rock_uplift_moment_km3: f64,
    effective_areal_denudation_export_km3: f64,
    hillslope_portal_transfer_km3: f64,
    final_elevation_volume_moment_km3: f64,
    ledger_closure_error_km3: f64,
    final_portal_water_outflow_km3_myr: f64,
    final_sink_water_storage_km3_myr: f64,
    integrated_portal_water_outflow_km3: f64,
    integrated_sink_water_storage_km3: f64,
    final_unresolved_specific_discharge_cells: usize,
    maximum_unresolved_specific_discharge_cells: usize,
    maximum_effective_denudation_rate_km_myr: f64,
    discharge_support: SmokeDischargeSupportDiagnostics,
    maximum_hillslope_slope_ratio: f64,
    runtime_seconds: f64,
    frozen_research_parameters: FrozenParameters,
    truth_limit: &'static str,
}

#[derive(Debug, Serialize)]
struct SmokeDischargeSupportDiagnostics {
    arm: C0DischargeSupportArm,
    alpha_km: Option<f64>,
    final_raw_maximum_km2_myr: f64,
    final_effective_maximum_km2_myr: f64,
    maximum_raw_intensity_over_run_km2_myr: f64,
    maximum_effective_intensity_over_run_km2_myr: f64,
    final_raw_area_weighted_integral_km4_myr: f64,
    final_effective_area_weighted_integral_km4_myr: f64,
    accepted_filter_solves: u64,
    total_filter_iterations: u64,
    maximum_filter_iterations: usize,
    total_filter_true_residual_restarts: u64,
    maximum_filter_true_residual_restarts: usize,
    maximum_filter_final_residual_l2: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    validate_cli(&cli)?;
    let summary = run(&cli)?;
    print_human_summary(&summary);
    if let Some(path) = &cli.output_json {
        write_json(path, &summary)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

fn validate_cli(cli: &Cli) -> Result<(), String> {
    for (name, value) in [
        ("spacing-km", cli.spacing_km),
        ("end-myr", cli.end_myr),
        ("requested-dt-myr", cli.requested_dt_myr),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("--{name} must be finite and greater than zero"));
        }
    }
    Ok(())
}

fn run(cli: &Cli) -> Result<SmokeSummary, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let mesh = LandscapeMesh::uniform_planar_hex(WIDTH_KM, HEIGHT_KM, cli.spacing_km)?;
    let scenario = match cli.case {
        CaseArg::U => uniform_scenario(),
        CaseArg::L => linked_scenario(),
    };
    let evaluator = scenario.compile(&mesh)?;
    let frozen = FrozenParameters::values();
    debug_assert_eq!(
        frozen.reference_denudation_e0_km_myr
            / (frozen.reference_specific_discharge_q0_km2_myr * frozen.reference_slope_s0),
        frozen.derived_k_per_km
    );
    let solver =
        C0LandscapeSolver::new(frozen.solver_params(cli.discharge_support.solver_support()))?;
    let mut state = C0LandscapeState::new(
        &mesh,
        linked_low_relief_initial_surface(&mesh, INITIAL_SEED),
    )?;

    let mut steps = 0_u64;
    let mut minimum_dt = f64::INFINITY;
    let mut maximum_dt: f64 = 0.0;
    let mut total_attempts = 0_u64;
    let mut maximum_attempts = 0_u32;
    let mut limiter_counts = LimiterCounts::default();
    let mut integrated_portal = 0.0;
    let mut integrated_sink = 0.0;
    let mut final_portal = 0.0;
    let mut final_sink = 0.0;
    let mut final_unresolved = 0;
    let mut maximum_unresolved = 0;
    let mut maximum_denudation_rate: f64 = 0.0;
    let mut maximum_slope_ratio: f64 = 0.0;
    let expected_arm = match cli.discharge_support {
        DischargeSupportArg::Unfiltered => C0DischargeSupportArm::Unfiltered,
        DischargeSupportArg::Q16 => C0DischargeSupportArm::FixedHelmholtz,
    };
    let mut support_summary = SmokeDischargeSupportDiagnostics {
        arm: expected_arm,
        alpha_km: match cli.discharge_support {
            DischargeSupportArg::Unfiltered => None,
            DischargeSupportArg::Q16 => Some(16.0),
        },
        final_raw_maximum_km2_myr: 0.0,
        final_effective_maximum_km2_myr: 0.0,
        maximum_raw_intensity_over_run_km2_myr: 0.0,
        maximum_effective_intensity_over_run_km2_myr: 0.0,
        final_raw_area_weighted_integral_km4_myr: 0.0,
        final_effective_area_weighted_integral_km4_myr: 0.0,
        accepted_filter_solves: 0,
        total_filter_iterations: 0,
        maximum_filter_iterations: 0,
        total_filter_true_residual_restarts: 0,
        maximum_filter_true_residual_restarts: 0,
        maximum_filter_final_residual_l2: 0.0,
    };

    while state.time_myr < cli.end_myr - 16.0 * f64::EPSILON {
        let requested = cli.requested_dt_myr.min(cli.end_myr - state.time_myr);
        let diagnostics = solver.step_with_forcing(&mesh, requested, &mut state, |midpoint| {
            evaluator.evaluate(midpoint)
        })?;
        let accepted = diagnostics.operator_limits.accepted_dt_myr;
        steps += 1;
        minimum_dt = minimum_dt.min(accepted);
        maximum_dt = maximum_dt.max(accepted);
        total_attempts += u64::from(diagnostics.operator_limits.attempted_steps);
        maximum_attempts = maximum_attempts.max(diagnostics.operator_limits.attempted_steps);
        match diagnostics.operator_limits.limiting_operator {
            C0TimestepLimiter::Requested => limiter_counts.requested += 1,
            C0TimestepLimiter::UpliftAccuracy => limiter_counts.uplift_accuracy += 1,
            C0TimestepLimiter::EffectiveDenudationAccuracy => {
                limiter_counts.effective_denudation_accuracy += 1
            }
            C0TimestepLimiter::EffectiveDenudationSlopeCourant => {
                limiter_counts.effective_denudation_slope_courant += 1
            }
            C0TimestepLimiter::HillslopeStability => limiter_counts.hillslope_stability += 1,
        }
        final_portal = diagnostics.water.total_portal_outflow_km3_myr;
        final_sink = diagnostics.water.total_sink_storage_km3_myr;
        integrated_portal += final_portal * accepted;
        integrated_sink += final_sink * accepted;
        final_unresolved = diagnostics.water.unresolved_specific_discharge_cells;
        maximum_unresolved = maximum_unresolved.max(final_unresolved);
        maximum_denudation_rate =
            maximum_denudation_rate.max(diagnostics.maximum_effective_denudation_rate_km_myr);
        maximum_slope_ratio = maximum_slope_ratio.max(diagnostics.maximum_hillslope_slope_ratio);
        let support = diagnostics.discharge_support;
        debug_assert_eq!(support.arm, support_summary.arm);
        debug_assert_eq!(support.alpha_km, support_summary.alpha_km);
        support_summary.final_raw_maximum_km2_myr = support.raw_maximum_km2_myr;
        support_summary.final_effective_maximum_km2_myr = support.effective_maximum_km2_myr;
        support_summary.maximum_raw_intensity_over_run_km2_myr = support_summary
            .maximum_raw_intensity_over_run_km2_myr
            .max(support.raw_maximum_km2_myr);
        support_summary.maximum_effective_intensity_over_run_km2_myr = support_summary
            .maximum_effective_intensity_over_run_km2_myr
            .max(support.effective_maximum_km2_myr);
        support_summary.final_raw_area_weighted_integral_km4_myr =
            support.raw_area_weighted_integral_km4_myr;
        support_summary.final_effective_area_weighted_integral_km4_myr =
            support.effective_area_weighted_integral_km4_myr;
        if let Some(audit) = support.filter_audit {
            support_summary.accepted_filter_solves += 1;
            support_summary.total_filter_iterations += audit.iterations as u64;
            support_summary.maximum_filter_iterations = support_summary
                .maximum_filter_iterations
                .max(audit.iterations);
            support_summary.total_filter_true_residual_restarts +=
                audit.true_residual_restarts as u64;
            support_summary.maximum_filter_true_residual_restarts = support_summary
                .maximum_filter_true_residual_restarts
                .max(audit.true_residual_restarts);
            support_summary.maximum_filter_final_residual_l2 = support_summary
                .maximum_filter_final_residual_l2
                .max(audit.final_residual_l2);
        }
    }

    let (minimum, maximum) = state
        .mean_bedrock_elevation_km
        .iter()
        .copied()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), z| {
            (low.min(z), high.max(z))
        });
    let ledger = state.elevation_volume_moment_ledger;
    Ok(SmokeSummary {
        schema: "hex3.c0-orogen-smoke.v2",
        case: cli.case.id(),
        nominal_width_km: WIDTH_KM,
        nominal_height_km: HEIGHT_KM,
        spacing_km: cli.spacing_km,
        cell_count: mesh.cell_count(),
        actual_domain_area_km2: mesh.actual_domain_area_km2(),
        initial_seed: INITIAL_SEED,
        final_time_myr: state.time_myr,
        requested_dt_myr: cli.requested_dt_myr,
        accepted_steps: steps,
        accepted_dt_min_myr: minimum_dt,
        accepted_dt_max_myr: maximum_dt,
        total_adaptive_attempts: total_attempts,
        maximum_attempts_for_one_step: maximum_attempts,
        limiter_counts,
        minimum_elevation_km: minimum,
        maximum_elevation_km: maximum,
        relief_km: maximum - minimum,
        initial_elevation_volume_moment_km3: ledger.initial_elevation_volume_moment_km3,
        rock_uplift_moment_km3: ledger.rock_uplift_moment_km3,
        effective_areal_denudation_export_km3: ledger.effective_areal_denudation_export_km3,
        hillslope_portal_transfer_km3: ledger.hillslope_portal_transfer_km3,
        final_elevation_volume_moment_km3: ledger.final_elevation_volume_moment_km3,
        ledger_closure_error_km3: ledger.closure_error_km3,
        final_portal_water_outflow_km3_myr: final_portal,
        final_sink_water_storage_km3_myr: final_sink,
        integrated_portal_water_outflow_km3: integrated_portal,
        integrated_sink_water_storage_km3: integrated_sink,
        final_unresolved_specific_discharge_cells: final_unresolved,
        maximum_unresolved_specific_discharge_cells: maximum_unresolved,
        maximum_effective_denudation_rate_km_myr: maximum_denudation_rate,
        discharge_support: support_summary,
        maximum_hillslope_slope_ratio: maximum_slope_ratio,
        runtime_seconds: started.elapsed().as_secs_f64(),
        frozen_research_parameters: frozen,
        truth_limit: TRUTH_LIMIT,
    })
}

fn print_human_summary(summary: &SmokeSummary) {
    println!(
        "C0 {} smoke: {} cells, {:.3} km2 actual area, {:.6} Myr in {} accepted steps ({:.3}s)",
        summary.case,
        summary.cell_count,
        summary.actual_domain_area_km2,
        summary.final_time_myr,
        summary.accepted_steps,
        summary.runtime_seconds
    );
    println!(
        "surface: min {:.6} km, max {:.6} km, relief {:.6} km; ledger residual {:.3e} km3",
        summary.minimum_elevation_km,
        summary.maximum_elevation_km,
        summary.relief_km,
        summary.ledger_closure_error_km3
    );
    println!(
        "water: portal {:.6e} km3/Myr, sinks {:.6e} km3/Myr; unresolved q cells {} (max {})",
        summary.final_portal_water_outflow_km3_myr,
        summary.final_sink_water_storage_km3_myr,
        summary.final_unresolved_specific_discharge_cells,
        summary.maximum_unresolved_specific_discharge_cells
    );
    println!(
        "accepted dt: {:.6e}..{:.6e} Myr; {} adaptive attempts (max {})",
        summary.accepted_dt_min_myr,
        summary.accepted_dt_max_myr,
        summary.total_adaptive_attempts,
        summary.maximum_attempts_for_one_step
    );
    println!(
        "discharge support: {:?} alpha={:?} km; final raw/effective q max {:.6e}/{:.6e} km2/Myr",
        summary.discharge_support.arm,
        summary.discharge_support.alpha_km,
        summary.discharge_support.final_raw_maximum_km2_myr,
        summary.discharge_support.final_effective_maximum_km2_myr
    );
    println!("truth limit: {}", summary.truth_limit);
}

fn write_json(path: &Path, summary: &SmokeSummary) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = serde_json::to_vec_pretty(summary)?;
    bytes.push(b'\n');
    fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_nonphysical_cli_values() {
        let cli = Cli {
            case: CaseArg::U,
            spacing_km: 0.0,
            end_myr: 0.01,
            requested_dt_myr: 0.001,
            discharge_support: DischargeSupportArg::Unfiltered,
            output_json: None,
        };
        assert!(validate_cli(&cli).is_err());
    }

    #[test]
    fn cli_selects_unfiltered_and_preregistered_q16_cleanly() {
        let omitted = Cli::try_parse_from(["c0_orogen_smoke", "--end-myr", "0.001"]).unwrap();
        assert!(matches!(
            omitted.discharge_support.solver_support(),
            C0DischargeSupport::Unfiltered
        ));

        let q16 = Cli::try_parse_from([
            "c0_orogen_smoke",
            "--end-myr",
            "0.001",
            "--discharge-support",
            "q16",
        ])
        .unwrap();
        assert_eq!(
            q16.discharge_support.solver_support(),
            C0DischargeSupport::fixed_helmholtz(16.0)
        );
    }

    #[test]
    fn initial_surface_is_deterministic_and_refinement_continuous() {
        let mesh = LandscapeMesh::uniform_planar_hex(48.0, 32.0, 4.0).unwrap();
        let first = linked_low_relief_initial_surface(&mesh, INITIAL_SEED);
        let second = linked_low_relief_initial_surface(&mesh, INITIAL_SEED);
        assert_eq!(first, second);
        assert!(first.iter().all(|z| z.is_finite() && *z >= 0.0));
    }

    #[test]
    fn frozen_reference_values_derive_k() {
        let p = FrozenParameters::values();
        let derived = p.reference_denudation_e0_km_myr
            / (p.reference_specific_discharge_q0_km2_myr * p.reference_slope_s0);
        assert!((derived - p.derived_k_per_km).abs() < f64::EPSILON);
        assert_eq!(
            C0LandscapeParams::default().effective_areal_denudation.k,
            0.0
        );
    }
}
