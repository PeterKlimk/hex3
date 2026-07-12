//! Resumable, provenance-complete evaluation corpus runner.
//!
//! This intentionally starts with a small stable metric adapter. Detailed
//! `diagnose` panels remain human-oriented until their definitions are promoted
//! into the metric registry.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use hex3::world::{
    elevation_to_km, BiomeKind, EcologySemantics, FineCacheMode, OrogenModel, RiverSelection,
    RiverThresholdPolicy, RunManifest, VoronoiBackend, World, NUM_PLATES_DEFAULT,
};
use serde::{Deserialize, Serialize};

const ARTIFACT_SCHEMA_VERSION: u32 = 1;
const METRIC_REGISTRY_VERSION: u32 = 1;

#[derive(Parser, Debug)]
#[command(name = "corpus", about = "Run a declarative Hex3 evaluation corpus")]
struct Cli {
    /// JSON corpus specification.
    #[arg(long)]
    spec: PathBuf,
    /// Artifact root. Defaults to artifacts/evaluation/<corpus-id>.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Re-run completed artifacts instead of resuming around them.
    #[arg(long, default_value_t = false)]
    force: bool,
    /// Validate and list stable run IDs without generating worlds.
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CorpusSpec {
    schema_version: u32,
    corpus_id: String,
    description: String,
    runs: Vec<RunSpec>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RunSpec {
    label: String,
    seed: u64,
    coarse_cells: usize,
    #[serde(default = "default_lloyd_iterations")]
    lloyd_iterations: usize,
    #[serde(default = "default_stage")]
    stage: u32,
    #[serde(default)]
    fine_max_cells: usize,
    #[serde(default = "default_fine_scale")]
    fine_scale: f32,
    #[serde(default)]
    backend: VoronoiBackend,
    #[serde(default)]
    orogen_model: OrogenModel,
    #[serde(default)]
    fine_cache: FineCacheMode,
}

fn default_lloyd_iterations() -> usize {
    1
}

fn default_stage() -> u32 {
    4
}

fn default_fine_scale() -> f32 {
    1.0
}

#[derive(Debug, Serialize)]
struct ArtifactMetadata<'a> {
    artifact_schema_version: u32,
    metric_registry_version: u32,
    corpus_id: &'a str,
    run_id: &'a str,
    run: &'a RunSpec,
    manifest: RunManifest,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum RunState {
    Completed,
    Failed,
}

#[derive(Debug, Serialize)]
struct StatusRecord<'a> {
    artifact_schema_version: u32,
    corpus_id: &'a str,
    run_id: &'a str,
    state: RunState,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct MetricRecord {
    id: &'static str,
    value: f64,
    unit: &'static str,
    weighting: &'static str,
    aggregation: &'static str,
    stage: u32,
}

#[derive(Debug, Serialize)]
struct TimingRecord {
    stage: &'static str,
    seconds: f64,
}

#[derive(Debug, Serialize)]
struct CorpusIndex<'a> {
    artifact_schema_version: u32,
    metric_registry_version: u32,
    corpus_id: &'a str,
    description: &'a str,
    runs: Vec<IndexEntry>,
}

#[derive(Debug, Serialize)]
struct IndexEntry {
    run_id: String,
    label: String,
    state: &'static str,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();
    let spec: CorpusSpec = read_json(&cli.spec).unwrap_or_else(|error| {
        panic!("could not read corpus spec {}: {error}", cli.spec.display())
    });
    validate_spec(&spec);
    let root = cli
        .out
        .unwrap_or_else(|| PathBuf::from("artifacts/evaluation").join(&spec.corpus_id));

    let mut seen = BTreeMap::new();
    let mut failures = 0usize;
    for run in &spec.runs {
        let run_id = stable_run_id(run);
        if let Some(previous) = seen.insert(run_id.clone(), run.label.clone()) {
            panic!(
                "runs {previous:?} and {:?} have identical identity",
                run.label
            );
        }
        println!("{}  {}", run_id, run.label);
        if cli.dry_run {
            continue;
        }
        if !run_one(&root, &spec.corpus_id, &run_id, run, cli.force) {
            failures += 1;
        }
    }
    write_corpus_index(&root, &spec).expect("write corpus index");
    if failures > 0 {
        eprintln!("corpus completed with {failures} failed run(s)");
        std::process::exit(1);
    }
}

fn write_corpus_index(root: &Path, spec: &CorpusSpec) -> Result<(), String> {
    let runs = spec
        .runs
        .iter()
        .map(|run| {
            let run_id = stable_run_id(run);
            let state = if root.join(&run_id).join("status.json").is_file() {
                "completed"
            } else if root
                .join(format!("{run_id}.failed"))
                .join("status.json")
                .is_file()
            {
                "failed"
            } else {
                "pending"
            };
            IndexEntry {
                run_id,
                label: run.label.clone(),
                state,
            }
        })
        .collect();
    let index = CorpusIndex {
        artifact_schema_version: ARTIFACT_SCHEMA_VERSION,
        metric_registry_version: METRIC_REGISTRY_VERSION,
        corpus_id: &spec.corpus_id,
        description: &spec.description,
        runs,
    };
    let temp = root.join(format!(".index.tmp-{}", std::process::id()));
    write_json(&temp, &index)?;
    fs::rename(temp, root.join("index.json")).map_err(|error| error.to_string())
}

fn validate_spec(spec: &CorpusSpec) {
    assert_eq!(spec.schema_version, 1, "unsupported corpus schema version");
    assert!(
        !spec.corpus_id.trim().is_empty(),
        "corpus_id must not be empty"
    );
    assert!(
        !spec.runs.is_empty(),
        "corpus must contain at least one run"
    );
    for run in &spec.runs {
        assert!(!run.label.trim().is_empty(), "run label must not be empty");
        assert!(run.coarse_cells >= 12, "coarse_cells must be at least 12");
        assert!((1..=4).contains(&run.stage), "stage must be 1..=4");
        assert!(run.fine_scale.is_finite() && run.fine_scale > 0.0);
        if run.stage >= 3 {
            assert!(
                run.fine_max_cells > 0,
                "stage 3/4 requires explicit fine_max_cells"
            );
        }
    }
}

fn stable_run_id(run: &RunSpec) -> String {
    let mut identity = serde_json::to_value(run).expect("serialize run identity");
    identity
        .as_object_mut()
        .expect("run identity object")
        .remove("label");
    let bytes = serde_json::to_vec(&(1u32, METRIC_REGISTRY_VERSION, identity))
        .expect("serialize versioned run identity");
    format!("run-{:016x}", fnv1a64(&bytes))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn run_one(root: &Path, corpus_id: &str, run_id: &str, run: &RunSpec, force: bool) -> bool {
    fs::create_dir_all(root).expect("create corpus artifact root");
    let final_dir = root.join(run_id);
    let failed_dir = root.join(format!("{run_id}.failed"));
    if final_dir.join("status.json").is_file() && !force {
        println!("  resume: completed artifact exists, skipping");
        return true;
    }
    if force && final_dir.exists() {
        fs::remove_dir_all(&final_dir).expect("remove forced prior artifact");
    }
    if failed_dir.exists() {
        fs::remove_dir_all(&failed_dir).expect("remove prior failed artifact before retry");
    }

    let temp_dir = root.join(format!(".{run_id}.tmp-{}", std::process::id()));
    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir).expect("remove stale temporary artifact");
    }
    fs::create_dir_all(&temp_dir).expect("create temporary artifact");
    write_json(&temp_dir.join("run-spec.json"), run).expect("write run spec");

    let result = std::panic::catch_unwind(|| generate_run(run));
    match result {
        Ok((world, metrics, timings)) => {
            let metadata = ArtifactMetadata {
                artifact_schema_version: ARTIFACT_SCHEMA_VERSION,
                metric_registry_version: METRIC_REGISTRY_VERSION,
                corpus_id,
                run_id,
                run,
                manifest: world.manifest(),
            };
            write_json(&temp_dir.join("manifest.json"), &metadata).expect("write manifest");
            write_json(&temp_dir.join("metrics.json"), &metrics).expect("write metrics");
            write_json(&temp_dir.join("timings.json"), &timings).expect("write timings");
            write_json(
                &temp_dir.join("status.json"),
                &StatusRecord {
                    artifact_schema_version: ARTIFACT_SCHEMA_VERSION,
                    corpus_id,
                    run_id,
                    state: RunState::Completed,
                    message: None,
                },
            )
            .expect("write completion status");
            fs::rename(&temp_dir, &final_dir).expect("publish atomic artifact directory");
            println!("  completed -> {}", final_dir.display());
            true
        }
        Err(payload) => {
            let message = panic_message(payload);
            write_json(
                &temp_dir.join("status.json"),
                &StatusRecord {
                    artifact_schema_version: ARTIFACT_SCHEMA_VERSION,
                    corpus_id,
                    run_id,
                    state: RunState::Failed,
                    message: Some(message.clone()),
                },
            )
            .expect("write failure status");
            fs::rename(&temp_dir, &failed_dir).expect("publish failure artifact");
            eprintln!("  failed: {message}");
            false
        }
    }
}

fn generate_run(run: &RunSpec) -> (World, Vec<MetricRecord>, Vec<TimingRecord>) {
    let total = Instant::now();
    let start = Instant::now();
    let mut world = World::new_with_options(
        run.seed,
        run.coarse_cells,
        run.lloyd_iterations,
        run.backend,
    );
    world.orogen_model = run.orogen_model;
    world.fine_cache = run.fine_cache;
    if (run.fine_scale - 1.0).abs() > f32::EPSILON {
        world.fine_density_params.plains_km *= run.fine_scale;
        world.fine_density_params.mountain_km *= run.fine_scale;
        world.fine_density_params.ocean_km *= run.fine_scale;
        world.fine_cache = FineCacheMode::Disabled;
    }
    world.generate_all(NUM_PLATES_DEFAULT);
    let mut timings = vec![TimingRecord {
        stage: "lithosphere",
        seconds: start.elapsed().as_secs_f64(),
    }];

    if run.stage >= 2 {
        let start = Instant::now();
        world.generate_atmosphere();
        timings.push(TimingRecord {
            stage: "atmosphere",
            seconds: start.elapsed().as_secs_f64(),
        });
    }
    if run.stage >= 3 {
        let start = Instant::now();
        world.generate_fine_pre_with_cap(run.fine_max_cells);
        timings.push(TimingRecord {
            stage: "fine-pre-hydrology",
            seconds: start.elapsed().as_secs_f64(),
        });
    }
    if run.stage >= 4 {
        let start = Instant::now();
        world.generate_fine_eroded();
        timings.push(TimingRecord {
            stage: "erosion",
            seconds: start.elapsed().as_secs_f64(),
        });
    }
    timings.push(TimingRecord {
        stage: "total",
        seconds: total.elapsed().as_secs_f64(),
    });

    let metrics = collect_metrics(&world);
    (world, metrics, timings)
}

fn collect_metrics(world: &World) -> Vec<MetricRecord> {
    let stage = world.current_stage();
    let tessellation = world.active_tessellation();
    let areas = tessellation.cell_areas();
    let elevation = &world
        .active_elevation()
        .expect("stage has elevation")
        .values;
    let submerged: Vec<bool> = (0..tessellation.num_cells())
        .map(|cell| {
            world
                .active_hydrology()
                .map(|hydrology| hydrology.is_submerged(cell))
                .unwrap_or(elevation[cell] < 0.0)
        })
        .collect();
    let total_area: f64 = areas.iter().map(|&area| area as f64).sum();
    let land_area: f64 = areas
        .iter()
        .enumerate()
        .filter(|(cell, _)| !submerged[*cell])
        .map(|(_, &area)| area as f64)
        .sum();
    let max_elevation = elevation.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut metrics = vec![
        metric(
            "mesh.active_cells.v1",
            tessellation.num_cells() as f64,
            "count",
            "none",
            "world",
            stage,
        ),
        metric(
            "terrain.land_area_fraction.v1",
            land_area / total_area.max(1e-30),
            "fraction",
            "area",
            "world",
            stage,
        ),
        metric(
            "terrain.elevation.max_km.v1",
            elevation_to_km(max_elevation) as f64,
            "km",
            "none",
            "world-maximum",
            stage,
        ),
    ];
    for (id, quantile) in [
        ("terrain.land_elevation_area_p50_km.v1", 0.50),
        ("terrain.land_elevation_area_p90_km.v1", 0.90),
        ("terrain.land_elevation_area_p99_km.v1", 0.99),
    ] {
        metrics.push(metric(
            id,
            elevation_to_km(weighted_quantile(elevation, &areas, &submerged, quantile)) as f64,
            "km",
            "land-area",
            "world-quantile",
            stage,
        ));
    }

    if let Some(hydrology) = world.active_hydrology() {
        let rivers = RiverSelection::build(hydrology, RiverThresholdPolicy::default());
        metrics.push(metric(
            "hydrology.semantic_river_cells.v1",
            rivers.all_cells.iter().filter(|&&v| v).count() as f64,
            "count",
            "cells",
            "world",
            stage,
        ));
        metrics.push(metric(
            "hydrology.lake_area_fraction_of_terrestrial.v1",
            lake_area_fraction(hydrology, &areas),
            "fraction",
            "area",
            "world",
            stage,
        ));
    }

    if let (Some(temperature), Some(precipitation)) =
        (world.active_temperature(), world.active_precipitation())
    {
        let ecology = EcologySemantics::build(
            tessellation,
            elevation,
            temperature,
            precipitation,
            world.active_hydrology(),
        );
        let transition_area: f64 = ecology
            .cells
            .iter()
            .enumerate()
            .filter(|(_, cell)| {
                !matches!(cell.biome, BiomeKind::Ocean | BiomeKind::Lake)
                    && cell.classification_confidence < 0.20
            })
            .map(|(cell, _)| areas[cell] as f64)
            .sum();
        metrics.push(metric(
            "climate.relative_aridity_land_mean_raw.v1",
            ecology.land_mean_raw_aridity as f64,
            "ratio",
            "land-area",
            "world-mean",
            stage,
        ));
        metrics.push(metric(
            "ecology.biome_transition_area.v1",
            transition_area / land_area.max(1e-30),
            "fraction",
            "land-area",
            "world",
            stage,
        ));
    }
    metrics
}

fn metric(
    id: &'static str,
    value: f64,
    unit: &'static str,
    weighting: &'static str,
    aggregation: &'static str,
    stage: u32,
) -> MetricRecord {
    MetricRecord {
        id,
        value,
        unit,
        weighting,
        aggregation,
        stage,
    }
}

fn weighted_quantile(values: &[f32], weights: &[f32], excluded: &[bool], quantile: f64) -> f32 {
    let mut pairs: Vec<(f32, f64)> = values
        .iter()
        .zip(weights)
        .zip(excluded)
        .filter(|(_, excluded)| !**excluded)
        .map(|((&value, &weight), _)| (value, weight.max(0.0) as f64))
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = pairs.iter().map(|pair| pair.1).sum();
    let target = quantile.clamp(0.0, 1.0) * total;
    let mut cumulative = 0.0;
    for (value, weight) in &pairs {
        cumulative += weight;
        if cumulative >= target {
            return *value;
        }
    }
    pairs.last().map(|pair| pair.0).unwrap_or(0.0)
}

fn lake_area_fraction(hydrology: &hex3::world::Hydrology, areas: &[f32]) -> f64 {
    let lake_area: f64 = (0..areas.len())
        .filter(|&cell| hydrology.is_lake_water(cell))
        .map(|cell| areas[cell] as f64)
        .sum();
    let terrestrial_area: f64 = (0..areas.len())
        .filter(|&cell| !hydrology.is_ocean(cell))
        .map(|cell| areas[cell] as f64)
        .sum();
    (lake_area / terrestrial_area.max(1e-30)).max(0.0)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let file = File::open(path).map_err(|error| error.to_string())?;
    serde_json::from_reader(BufReader::new(file)).map_err(|error| error.to_string())
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let file = File::create(path).map_err(|error| error.to_string())?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value).map_err(|error| error.to_string())?;
    writer.write_all(b"\n").map_err(|error| error.to_string())?;
    writer.flush().map_err(|error| error.to_string())
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|value| (*value).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run() -> RunSpec {
        RunSpec {
            label: "smoke".into(),
            seed: 42,
            coarse_cells: 1200,
            lloyd_iterations: 1,
            stage: 4,
            fine_max_cells: 4000,
            fine_scale: 1.0,
            backend: VoronoiBackend::ConvexHull,
            orogen_model: OrogenModel::Legacy,
            fine_cache: FineCacheMode::Disabled,
        }
    }

    #[test]
    fn run_identity_is_stable_and_configuration_sensitive() {
        let original = run();
        assert_eq!(stable_run_id(&original), stable_run_id(&original));
        let mut changed = run();
        changed.seed += 1;
        assert_ne!(stable_run_id(&original), stable_run_id(&changed));
        let mut relabeled = run();
        relabeled.label = "renamed".into();
        assert_eq!(stable_run_id(&original), stable_run_id(&relabeled));
    }

    #[test]
    fn weighted_quantile_uses_area_and_exclusion() {
        let values = [1.0, 2.0, 3.0];
        let weights = [1.0, 8.0, 1.0];
        assert_eq!(weighted_quantile(&values, &weights, &[false; 3], 0.5), 2.0);
        assert_eq!(
            weighted_quantile(&values, &weights, &[false, true, false], 0.5),
            1.0
        );
    }
}
