//! Run the CPU-only bounded orogen organization testbed.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use serde::Serialize;

use hex3::world::landscape::{
    linked_scenario, uniform_scenario, BoundaryCondition, DeformationFrame, LandscapeLedger,
    LandscapeMesh, LandscapeParams, LandscapeSolver, LandscapeState, StepDiagnostics,
};

const ARTIFACT_SCHEMA: &str = "hex3.orogen-testbed.v1";
const METRIC_SCHEMA: &str = "hex3.orogen-testbed.metrics.v1";
const DEFAULT_OUTPUT_ROOT: &str = "artifacts/orogen-testbed";

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
enum CaseArg {
    /// Smooth, finite-width uplift block (unstructured physical null).
    U,
    /// Two linked, en-echelon deformation segments.
    L,
}

impl CaseArg {
    fn id(self) -> &'static str {
        match self {
            Self::U => "U",
            Self::L => "L",
        }
    }
}

#[derive(Debug, Parser, Serialize)]
#[command(
    name = "orogen_testbed",
    about = "Run the CPU-only bounded orogen organization testbed"
)]
struct Cli {
    /// Causal case: U (uniform block) or L (linked segments).
    #[arg(long = "case", value_enum, ignore_case = true, default_value = "L")]
    case: CaseArg,

    /// Approximate planar hex-cell spacing in kilometres.
    #[arg(long, default_value_t = 4.0)]
    spacing_km: f64,

    /// Simulated duration in millions of years.
    #[arg(long, default_value_t = 10.0)]
    duration_myr: f64,

    /// Requested maximum integration step in millions of years.
    #[arg(long, default_value_t = 0.01)]
    max_dt_myr: f64,

    /// Interval between scalar metric records in millions of years.
    #[arg(long, default_value_t = 0.25)]
    metric_interval_myr: f64,

    /// Deterministic initial-condition seed.
    #[arg(long, default_value_t = 12345)]
    seed: u64,

    /// Parent directory for atomically published run directories.
    #[arg(long, default_value = DEFAULT_OUTPUT_ROOT)]
    output_root: PathBuf,
}

#[derive(Debug, Serialize)]
struct Units {
    horizontal_distance: &'static str,
    area: &'static str,
    elevation: &'static str,
    time: &'static str,
    vertical_velocity: &'static str,
    runoff_depth_rate: &'static str,
    discharge: &'static str,
    diffusivity: &'static str,
    slope: &'static str,
    volume: &'static str,
}

impl Default for Units {
    fn default() -> Self {
        Self {
            horizontal_distance: "km",
            area: "km^2",
            elevation: "km",
            time: "Myr",
            vertical_velocity: "km/Myr",
            runoff_depth_rate: "km/Myr",
            discharge: "km^3/Myr",
            diffusivity: "km^2/Myr",
            slope: "grade",
            volume: "km^3",
        }
    }
}

#[derive(Debug, Serialize)]
struct RunIdentity<'a> {
    artifact_schema: &'a str,
    case: &'a str,
    arm: &'a str,
    seed: u64,
    spacing_km: f64,
    duration_myr: f64,
    max_dt_myr: f64,
    metric_interval_myr: f64,
}

#[derive(Debug, Serialize)]
struct HashRecord {
    algorithm: &'static str,
    mesh: String,
    scenario: String,
    solver_config: String,
    initial_state: String,
    final_state: String,
}

#[derive(Debug, Serialize)]
struct Manifest<'a, S: Serialize, P: Serialize> {
    artifact_schema: &'static str,
    metric_schema: &'static str,
    run_id: &'a str,
    case: &'a str,
    arm: &'static str,
    seed: u64,
    run: &'a RunIdentity<'a>,
    mesh: MeshSummary,
    source: SourceRevision,
    units: Units,
    scenario: &'a S,
    solver: &'a P,
    hashes: HashRecord,
    thread_count: usize,
    wall_seconds: f64,
}

#[derive(Debug, Serialize)]
struct MeshSummary {
    kind: &'static str,
    width_km: f64,
    height_km: f64,
    spacing_km: f64,
    cell_count: usize,
}

#[derive(Clone, Debug, Serialize)]
struct SourceRevision {
    git_commit: Option<String>,
    git_dirty: Option<bool>,
    executable_hash_fnv1a64: Option<String>,
}

fn source_revision() -> SourceRevision {
    let git_commit = git_output(&["rev-parse", "HEAD"]);
    let git_dirty = git_output(&["status", "--porcelain"]).map(|value| !value.is_empty());
    let executable_hash_fnv1a64 = std::env::current_exe()
        .ok()
        .and_then(|path| fs::read(path).ok())
        .map(|bytes| format!("{:016x}", fnv1a64(&bytes)));
    SourceRevision {
        git_commit,
        git_dirty,
        executable_hash_fnv1a64,
    }
}

fn git_output(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

#[derive(Debug, Serialize)]
struct MetricRecord {
    schema: &'static str,
    time_myr: f64,
    revision: u64,
    step_count: u64,
    actual_dt_myr: f64,
    requested_dt_myr: f64,
    uplift_limit_myr: Option<f64>,
    incision_limit_myr: Option<f64>,
    hillslope_limit_myr: Option<f64>,
    limiting_operator: &'static str,
    cell_count: usize,
    min_elevation_km: f64,
    mean_elevation_km: f64,
    max_elevation_km: f64,
    relief_km: f64,
    mean_uplift_rate_km_myr: f64,
    active_uplift_area_km2: f64,
    total_surface_volume_km3: f64,
    cumulative_uplift_km3: f64,
    cumulative_incision_export_km3: f64,
    cumulative_hillslope_export_km3: f64,
    ledger_closure_error_km3: f64,
    max_discharge_km3_myr: f64,
    outlet_count: usize,
    sink_count: usize,
    maximum_slope_ratio: f64,
    nonlinear_regularized_faces: usize,
    state_hash_fnv1a64: String,
}

#[derive(Debug, Serialize)]
struct Summary<'a> {
    artifact_schema: &'static str,
    run_id: &'a str,
    completed: bool,
    final_time_myr: f64,
    step_count: u64,
    wall_seconds: f64,
    initial_state_hash_fnv1a64: &'a str,
    final_state_hash_fnv1a64: &'a str,
    ledger_closure_error_km3: f64,
    ledger_relative_error: f64,
    initial_bedrock_volume_km3: f64,
    cumulative_uplift_km3: f64,
    cumulative_incision_export_km3: f64,
    cumulative_hillslope_export_km3: f64,
    cumulative_base_level_adjustment_km3: f64,
    final_bedrock_volume_km3: f64,
    peak_relief_km: f64,
    peak_discharge_km3_myr: f64,
}

#[derive(Debug, Serialize)]
struct ScalarCheckpoint<'a> {
    schema: &'static str,
    time_myr: f64,
    revision: u64,
    bedrock_elevation_km: &'a [f64],
    bedrock_change_km: &'a [f64],
    cumulative_rock_uplift_km: &'a [f64],
    rock_uplift_rate_km_myr: &'a [f32],
    receiver: &'a [Option<usize>],
    discharge_km3_myr: &'a [f64],
    ledger: LandscapeLedger,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    validate_cli(&cli)?;
    run(cli)
}

fn validate_cli(cli: &Cli) -> Result<(), String> {
    for (name, value) in [
        ("spacing-km", cli.spacing_km),
        ("duration-myr", cli.duration_myr),
        ("max-dt-myr", cli.max_dt_myr),
        ("metric-interval-myr", cli.metric_interval_myr),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("--{name} must be finite and greater than zero"));
        }
    }
    Ok(())
}

/// Common neutral datum: a 20 m central swell draining toward both open
/// north/south edges, plus one band-limited continuous perturbation below 10 m.
/// The field is a function of physical coordinates rather than cell identity,
/// so convergence meshes sample the same initial landscape.
fn neutral_initial_bedrock(mesh: &LandscapeMesh, seed: u64) -> Vec<f64> {
    let max_abs_y = mesh
        .cell_center_km
        .iter()
        .map(|center| center.y.abs())
        .fold(0.0, f64::max)
        .max(f64::EPSILON);
    let phase = |stream: u64| {
        let random = splitmix64(seed ^ stream.wrapping_mul(0x9e3779b97f4a7c15));
        let unit = (random >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
        unit * std::f64::consts::TAU
    };
    let phases = [phase(1), phase(2), phase(3)];
    mesh.cell_center_km
        .iter()
        .enumerate()
        .map(|(cell, center)| {
            if matches!(mesh.boundary[cell], BoundaryCondition::OpenBaseLevel { .. }) {
                return 0.0;
            }
            let taper = (1.0 - center.y.abs() / max_abs_y).clamp(0.0, 1.0);
            let perturbation = 0.0020 * (0.071 * center.x + 0.043 * center.y + phases[0]).sin()
                + 0.0015 * (-0.038 * center.x + 0.063 * center.y + phases[1]).sin()
                + 0.0010 * (0.027 * center.x - 0.052 * center.y + phases[2]).sin();
            taper * (0.020 + perturbation)
        })
        .collect()
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let mesh = LandscapeMesh::uniform_planar_hex(960.0, 640.0, cli.spacing_km)?;
    let scenario = match cli.case {
        CaseArg::U => uniform_scenario(),
        CaseArg::L => linked_scenario(),
    };
    let evaluator = scenario.compile(&mesh)?;
    let params = LandscapeParams::default();
    let solver = LandscapeSolver::new(params)?;
    let mut state = LandscapeState::new(&mesh, neutral_initial_bedrock(&mesh, cli.seed))?;
    solver.refresh_drainage(&mesh, &mut state)?;

    let identity = RunIdentity {
        artifact_schema: ARTIFACT_SCHEMA,
        case: cli.case.id(),
        arm: "C",
        seed: cli.seed,
        spacing_km: cli.spacing_km,
        duration_myr: cli.duration_myr,
        max_dt_myr: cli.max_dt_myr,
        metric_interval_myr: cli.metric_interval_myr,
    };
    let mesh_hash = hash_serialized(&mesh)?;
    let scenario_hash = hash_serialized(&scenario)?;
    let solver_hash = hash_serialized(&params)?;
    let initial_hash = hash_serialized(&state)?;
    let source = source_revision();
    let run_id = format!(
        "run-{}",
        hash_serialized(&(
            &identity,
            &mesh_hash,
            &scenario_hash,
            &solver_hash,
            &initial_hash,
            &source
        ))?
    );
    fs::create_dir_all(&cli.output_root)?;
    let final_dir = cli.output_root.join(&run_id);
    let temp_dir = cli
        .output_root
        .join(format!(".{run_id}.tmp-{}", std::process::id()));
    if temp_dir.exists() || final_dir.exists() {
        return Err(format!("artifact path already exists for {run_id}").into());
    }
    fs::create_dir(&temp_dir)?;

    let initial_bedrock = state.bedrock_elevation_km.clone();
    let mut cumulative_rock_uplift_km = vec![0.0; mesh.cell_count()];
    let checkpoints_dir = temp_dir.join("checkpoints");
    fs::create_dir(&checkpoints_dir)?;
    let initial_frame = evaluator.evaluate(state.time_myr);
    write_checkpoint(
        &checkpoints_dir,
        &state,
        &initial_bedrock,
        &cumulative_rock_uplift_km,
        &initial_frame,
    )?;
    let mut metrics = vec![metric_record(&mesh, &state, &initial_frame, 0, None)?];
    let mut step_count = 0_u64;
    let mut next_metric = cli.metric_interval_myr.min(cli.duration_myr);
    let epsilon = 1.0e-12;

    while state.time_myr + epsilon < cli.duration_myr {
        let to_end = cli.duration_myr - state.time_myr;
        let to_metric = (next_metric - state.time_myr).max(epsilon);
        let requested = cli.max_dt_myr.min(to_end).min(to_metric);
        let diagnostics = solver.step_with_forcing(&mesh, requested, &mut state, |time| {
            evaluator.evaluate(time)
        })?;
        let applied_frame = evaluator
            .evaluate(diagnostics.time_start_myr + 0.5 * diagnostics.timestep.accepted_dt_myr);
        for (cumulative, rate) in cumulative_rock_uplift_km
            .iter_mut()
            .zip(&applied_frame.rock_vertical_rate_km_myr)
        {
            *cumulative += f64::from(*rate) * diagnostics.timestep.accepted_dt_myr;
        }
        step_count += 1;
        if state.time_myr + epsilon >= next_metric || state.time_myr + epsilon >= cli.duration_myr {
            let current_frame = evaluator.evaluate(state.time_myr);
            metrics.push(metric_record(
                &mesh,
                &state,
                &current_frame,
                step_count,
                Some(&diagnostics),
            )?);
            write_checkpoint(
                &checkpoints_dir,
                &state,
                &initial_bedrock,
                &cumulative_rock_uplift_km,
                &current_frame,
            )?;
            while next_metric <= state.time_myr + epsilon {
                next_metric += cli.metric_interval_myr;
            }
            next_metric = next_metric.min(cli.duration_myr);
        }
    }

    let wall_seconds = started.elapsed().as_secs_f64();
    let final_hash = hash_serialized(&state)?;
    let peak_relief = metrics.iter().map(|m| m.relief_km).fold(0.0, f64::max);
    let peak_discharge = metrics
        .iter()
        .map(|m| m.max_discharge_km3_myr)
        .fold(0.0, f64::max);
    let relative_error =
        state.ledger.closure_error_km3.abs() / state.ledger.rock_uplift_km3.abs().max(f64::EPSILON);
    let hashes = HashRecord {
        algorithm: "fnv1a64-over-serde-json-v1",
        mesh: mesh_hash,
        scenario: scenario_hash,
        solver_config: solver_hash,
        initial_state: initial_hash.clone(),
        final_state: final_hash.clone(),
    };
    let manifest = Manifest {
        artifact_schema: ARTIFACT_SCHEMA,
        metric_schema: METRIC_SCHEMA,
        run_id: &run_id,
        case: cli.case.id(),
        arm: "C",
        seed: cli.seed,
        run: &identity,
        mesh: MeshSummary {
            kind: "uniform-planar-hex",
            width_km: 960.0,
            height_km: 640.0,
            spacing_km: cli.spacing_km,
            cell_count: mesh.cell_count(),
        },
        source,
        units: Units::default(),
        scenario: &scenario,
        solver: &params,
        hashes,
        thread_count: rayon::current_num_threads(),
        wall_seconds,
    };
    let summary = Summary {
        artifact_schema: ARTIFACT_SCHEMA,
        run_id: &run_id,
        completed: true,
        final_time_myr: state.time_myr,
        step_count,
        wall_seconds,
        initial_state_hash_fnv1a64: &initial_hash,
        final_state_hash_fnv1a64: &final_hash,
        ledger_closure_error_km3: state.ledger.closure_error_km3,
        ledger_relative_error: relative_error,
        initial_bedrock_volume_km3: state.ledger.initial_bedrock_volume_km3,
        cumulative_uplift_km3: state.ledger.rock_uplift_km3,
        cumulative_incision_export_km3: state.ledger.incision_export_km3,
        cumulative_hillslope_export_km3: state.ledger.hillslope_boundary_export_km3,
        cumulative_base_level_adjustment_km3: state.ledger.base_level_adjustment_km3,
        final_bedrock_volume_km3: state.ledger.final_bedrock_volume_km3,
        peak_relief_km: peak_relief,
        peak_discharge_km3_myr: peak_discharge,
    };
    let final_frame = evaluator.evaluate(state.time_myr);
    let bedrock_change: Vec<_> = state
        .bedrock_elevation_km
        .iter()
        .zip(&initial_bedrock)
        .map(|(final_value, initial_value)| final_value - initial_value)
        .collect();
    let checkpoint = ScalarCheckpoint {
        schema: "hex3.orogen-testbed.scalar-checkpoint.v1",
        time_myr: state.time_myr,
        revision: state.revision,
        bedrock_elevation_km: &state.bedrock_elevation_km,
        bedrock_change_km: &bedrock_change,
        cumulative_rock_uplift_km: &cumulative_rock_uplift_km,
        rock_uplift_rate_km_myr: &final_frame.rock_vertical_rate_km_myr,
        receiver: &state.drainage.receiver,
        discharge_km3_myr: &state.drainage.discharge_km3_myr,
        ledger: state.ledger,
    };

    write_json(&temp_dir.join("manifest.json"), &manifest)?;
    write_ndjson(&temp_dir.join("metrics.ndjson"), &metrics)?;
    write_json(&temp_dir.join("summary.json"), &summary)?;
    fs::write(
        temp_dir.join("checkpoint-final.bin"),
        bincode::serialize(&checkpoint)?,
    )?;
    publish_directory(&temp_dir, &final_dir)?;
    println!("wrote {}", final_dir.display());
    Ok(())
}

fn write_checkpoint(
    directory: &Path,
    state: &LandscapeState,
    initial_bedrock: &[f64],
    cumulative_rock_uplift_km: &[f64],
    frame: &DeformationFrame,
) -> Result<(), Box<dyn std::error::Error>> {
    let bedrock_change: Vec<_> = state
        .bedrock_elevation_km
        .iter()
        .zip(initial_bedrock)
        .map(|(current, initial)| current - initial)
        .collect();
    let checkpoint = ScalarCheckpoint {
        schema: "hex3.orogen-testbed.scalar-checkpoint.v1",
        time_myr: state.time_myr,
        revision: state.revision,
        bedrock_elevation_km: &state.bedrock_elevation_km,
        bedrock_change_km: &bedrock_change,
        cumulative_rock_uplift_km,
        rock_uplift_rate_km_myr: &frame.rock_vertical_rate_km_myr,
        receiver: &state.drainage.receiver,
        discharge_km3_myr: &state.drainage.discharge_km3_myr,
        ledger: state.ledger,
    };
    let micro_myr = (state.time_myr * 1_000_000.0).round() as u64;
    fs::write(
        directory.join(format!("t-{micro_myr:012}-micro-myr.bin")),
        bincode::serialize(&checkpoint)?,
    )?;
    Ok(())
}

fn finite_limit(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn metric_record(
    mesh: &LandscapeMesh,
    state: &LandscapeState,
    frame: &DeformationFrame,
    step_count: u64,
    diagnostics: Option<&StepDiagnostics>,
) -> Result<MetricRecord, serde_json::Error> {
    let total_area: f64 = mesh.cell_area_km2.iter().sum();
    let (minimum, maximum) = state
        .bedrock_elevation_km
        .iter()
        .copied()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), value| {
            (low.min(value), high.max(value))
        });
    let mean = state
        .bedrock_elevation_km
        .iter()
        .zip(&mesh.cell_area_km2)
        .map(|(elevation, area)| elevation * area)
        .sum::<f64>()
        / total_area;
    let uplift_volume_rate: f64 = frame
        .rock_vertical_rate_km_myr
        .iter()
        .zip(&mesh.cell_area_km2)
        .map(|(rate, area)| f64::from(*rate) * area)
        .sum();
    let active_uplift_area = frame
        .rock_vertical_rate_km_myr
        .iter()
        .zip(&mesh.cell_area_km2)
        .filter(|(rate, _)| **rate > 0.0)
        .map(|(_, area)| area)
        .sum();
    let outlet_count = state
        .drainage
        .receiver
        .iter()
        .zip(&state.drainage.outlet_by_cell)
        .filter(|(receiver, outlet)| receiver.is_none() && outlet.is_some())
        .count();
    let sink_count = state
        .drainage
        .receiver
        .iter()
        .zip(&state.drainage.outlet_by_cell)
        .filter(|(receiver, outlet)| receiver.is_none() && outlet.is_none())
        .count();
    let timestep = diagnostics.map(|value| value.timestep);
    Ok(MetricRecord {
        schema: METRIC_SCHEMA,
        time_myr: state.time_myr,
        revision: state.revision,
        step_count,
        actual_dt_myr: timestep.map_or(0.0, |value| value.accepted_dt_myr),
        requested_dt_myr: timestep.map_or(0.0, |value| value.requested_dt_myr),
        uplift_limit_myr: timestep.and_then(|value| finite_limit(value.uplift_limit_myr)),
        incision_limit_myr: timestep.and_then(|value| finite_limit(value.incision_limit_myr)),
        hillslope_limit_myr: timestep.and_then(|value| finite_limit(value.hillslope_limit_myr)),
        limiting_operator: timestep.map_or("initial", |value| match value.limiting_operator {
            hex3::world::landscape::TimestepLimiter::Requested => "requested",
            hex3::world::landscape::TimestepLimiter::Uplift => "uplift",
            hex3::world::landscape::TimestepLimiter::Incision => "incision",
            hex3::world::landscape::TimestepLimiter::Hillslope => "hillslope",
        }),
        cell_count: mesh.cell_count(),
        min_elevation_km: minimum,
        mean_elevation_km: mean,
        max_elevation_km: maximum,
        relief_km: maximum - minimum,
        mean_uplift_rate_km_myr: uplift_volume_rate / total_area,
        active_uplift_area_km2: active_uplift_area,
        total_surface_volume_km3: state.surface_volume_km3(mesh),
        cumulative_uplift_km3: state.ledger.rock_uplift_km3,
        cumulative_incision_export_km3: state.ledger.incision_export_km3,
        cumulative_hillslope_export_km3: state.ledger.hillslope_boundary_export_km3,
        ledger_closure_error_km3: state.ledger.closure_error_km3,
        max_discharge_km3_myr: state
            .drainage
            .discharge_km3_myr
            .iter()
            .copied()
            .fold(0.0, f64::max),
        outlet_count,
        sink_count,
        maximum_slope_ratio: diagnostics.map_or(0.0, |value| value.maximum_slope_ratio),
        nonlinear_regularized_faces: diagnostics
            .map_or(0, |value| value.nonlinear_regularized_faces),
        state_hash_fnv1a64: hash_serialized(state)?,
    })
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn hash_serialized(value: &impl Serialize) -> Result<String, serde_json::Error> {
    Ok(format!("{:016x}", fnv1a64(&serde_json::to_vec(value)?)))
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn write_ndjson(
    path: &Path,
    values: impl IntoIterator<Item = impl Serialize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = BufWriter::new(File::create(path)?);
    for value in values {
        serde_json::to_writer(&mut writer, &value)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn publish_directory(temp: &Path, final_path: &Path) -> std::io::Result<()> {
    if final_path.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("artifact already exists: {}", final_path.display()),
        ));
    }
    fs::rename(temp, final_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_hash_is_stable() {
        assert_eq!(format!("{:016x}", fnv1a64(b"hex3")), "0a704fcc5f7fc31b");
    }

    #[test]
    fn rejects_nonphysical_cli_values() {
        let cli = Cli {
            case: CaseArg::L,
            spacing_km: 0.0,
            duration_myr: 10.0,
            max_dt_myr: 0.01,
            metric_interval_myr: 0.25,
            seed: 1,
            output_root: PathBuf::from("unused"),
        };
        assert!(validate_cli(&cli).is_err());
    }

    #[test]
    fn neutral_initial_condition_is_deterministic_and_pins_outlets() {
        let mesh = LandscapeMesh::uniform_planar_hex(48.0, 32.0, 4.0).unwrap();
        let first = neutral_initial_bedrock(&mesh, 17);
        let second = neutral_initial_bedrock(&mesh, 17);
        assert_eq!(first, second);
        for (cell, boundary) in mesh.boundary.iter().enumerate() {
            if matches!(boundary, BoundaryCondition::OpenBaseLevel { .. }) {
                assert_eq!(first[cell], 0.0);
            }
        }
    }
}
