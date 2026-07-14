//! Frozen one-seed product G0/S0 ancestry observation.
//!
//! Contract: `docs/research/landform-product-g0s0-observation-2026-07-14.md`.
//! This binary intentionally exposes only the artifact path. Every scientific
//! input and extraction setting is compiled from the preregistered contract.

use std::collections::BTreeSet;
use std::env;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use hex3::world::landforms::{
    adapt_product_tessellation_graph_v0, build_surface_hierarchy_v0, EvaluationDomainV0,
    EvaluationSurfaceGraphV0, HighlandFeatureV0, HighlandMeasurementsV0, LocalReliefSummaryV0,
    PeakBranchV0, SaddleNodeV0, SphericalFootprintGeometryV0, SummitCapSummaryV0,
    SurfaceHierarchyConfigV0, SurfaceHierarchyV0,
};
use hex3::world::{
    FineCacheMode, OrogenModel, RunManifest, VoronoiBackend, World, ELEVATION_UNIT_KM,
    NUM_PLATES_DEFAULT,
};
use serde::Serialize;

const ARTIFACT_SCHEMA_VERSION: &str = "landform-product-g0s0-observation-v0";
const OBSERVATION_ID: &str = "legacy-product-ancestry-seed-12345-250k-v0";
const CONTRACT_PATH: &str = "docs/research/landform-product-g0s0-observation-2026-07-14.md";
const SEED: u64 = 12_345;
const COARSE_REQUESTED_CELLS: usize = 100_000;
const LLOYD_METADATA: usize = 1;
const FINE_MAX_CELLS: usize = 250_000;
const COMPUTED_STAGE: u32 = 4;
const ELEVATION_HASH_VERSION: &str = "fnv1a64-f64-le-v0";

#[derive(Debug, Parser)]
#[command(
    name = "landform_baseline",
    about = "Run the preregistered product G0/S0 ancestry observation"
)]
struct Cli {
    /// Atomic JSON artifact destination. This is the only runtime degree of freedom.
    output: PathBuf,
}

#[derive(Serialize)]
struct Artifact {
    schema_version: &'static str,
    observation_id: &'static str,
    contract: &'static str,
    interpretation: InterpretationGuard,
    scored_policy: ScoredPolicy,
    frozen_run: FrozenRun,
    hierarchy_config: SurfaceHierarchyConfigV0,
    manifest: RunManifest,
    command: CommandRecord,
    platform: PlatformRecord,
    memory: MemoryRecord,
    timings_seconds: TimingRecord,
    fine_base_elevations_byte_identical: bool,
    graphs: Vec<GraphArtifact>,
    surfaces: Vec<SurfaceArtifact>,
}

#[derive(Serialize)]
struct InterpretationGuard {
    observation_kind: &'static str,
    has_drainage_d0: bool,
    has_relationships_o0: bool,
    highland_feature_semantics: &'static str,
    broad_cap_semantics: &'static str,
}

#[derive(Serialize)]
struct ScoredPolicy {
    scored_domain: &'static str,
    active_test: &'static str,
    elevation_conversion: &'static str,
    datum: &'static str,
}

#[derive(Serialize)]
struct FrozenRun {
    seed: u64,
    coarse_requested_cells: usize,
    coarse_backend: VoronoiBackend,
    lloyd_metadata: usize,
    plates: usize,
    orogen_model: OrogenModel,
    fine_cap: usize,
    fine_cache: FineCacheMode,
    computed_stage: u32,
    terrain_climate_hydrology_erosion_controls_product_defaults: bool,
}

#[derive(Serialize)]
struct CommandRecord {
    invoked_argv: Vec<String>,
    required_guarded_invocation: &'static str,
}

#[derive(Serialize)]
struct PlatformRecord {
    os: &'static str,
    family: &'static str,
    architecture: &'static str,
    kernel_release: Option<String>,
    kernel_version: Option<String>,
    wsl_distribution: Option<String>,
    release_build_required: bool,
    debug_assertions_enabled: bool,
}

#[derive(Serialize)]
struct MemoryRecord {
    whole_process_peak_rss_capture: &'static str,
    proc_self_vmhwm_kib_before_publish: Option<u64>,
    retained_memory: Option<u64>,
    retained_memory_status: &'static str,
}

#[derive(Default, Serialize)]
struct TimingRecord {
    lithosphere: f64,
    atmosphere: f64,
    fine_pre: f64,
    erosion: f64,
    product_generation_total: f64,
    artifact_serialization_measurement: f64,
}

#[derive(Serialize)]
struct GraphArtifact {
    id: &'static str,
    domain: EvaluationDomainV0,
    cells: usize,
    directed_edges: usize,
    physical_area_km2: f64,
    g0_adaptation_seconds: f64,
}

#[derive(Serialize)]
struct SurfaceArtifact {
    id: &'static str,
    graph_id: &'static str,
    elevation_source: &'static str,
    elevation_hash_version: &'static str,
    elevation_hash_hex: String,
    active: ActiveDomainSummary,
    structural_counts: StructuralCounts,
    population_counts: PopulationCounts,
    derived_evidence_hash_hex: String,
    s0_morphology_seconds: f64,
    reference_counts: ReferenceEvidenceCounts,
    reference_footprint_union_area_km2: f64,
    object_summaries: ObjectSummaries,
    largest_reference_objects: Vec<RankedObject>,
    most_persistent_reference_objects: Vec<RankedObject>,
    reference_highlands: Vec<ReferenceHighlandRecord>,
}

#[derive(Serialize)]
struct ActiveDomainSummary {
    cell_count: usize,
    cell_fraction: f64,
    area_km2: f64,
    area_fraction: f64,
}

#[derive(Serialize)]
struct StructuralCounts {
    raw_peaks: usize,
    raw_saddles: usize,
    roots: usize,
    child_peaks: usize,
    flat_maximum_nodes: usize,
    flat_maximum_cells: usize,
    flat_saddle_nodes: usize,
    flat_saddle_cells: usize,
    equal_elder_ambiguous_peaks: usize,
    equal_elder_ambiguous_saddles: usize,
}

#[derive(Serialize)]
struct PopulationCounts {
    reference: usize,
    persistence_low: usize,
    persistence_high: usize,
    footprint_low: usize,
    footprint_high: usize,
}

#[derive(Default, Serialize)]
struct ReferenceEvidenceCounts {
    root_features: usize,
    child_features: usize,
    local_geometry: usize,
    nonlocal_geometry: usize,
    spherical_nonlocal_warnings: usize,
    orientation_ambiguous: usize,
    rank_deficient_grade_objects: usize,
    rank_deficient_grade_cells: usize,
    relief_truncated_objects: usize,
    relief_truncated_summaries: usize,
    relief_truncated_member_occurrences: usize,
    cap_merge_censored_objects: usize,
    cap_merge_censored_summaries: usize,
}

#[derive(Serialize)]
struct ReferenceHighlandRecord {
    feature: HighlandFeatureV0,
    peak: PeakBranchV0,
    key_saddle: Option<SaddleNodeV0>,
}

#[derive(Serialize)]
struct RankedObject {
    peak_id: u32,
    value: f64,
}

#[derive(Serialize)]
struct ObjectSummaries {
    persistence_km: OptionalFiniteSummary,
    footprint_area_km2: OptionalFiniteSummary,
    two_sweep_extent_km: OptionalFiniteSummary,
    mean_width_km: OptionalFiniteSummary,
    equivalent_ellipse_length_km: OptionalFiniteSummary,
    equivalent_ellipse_width_km: OptionalFiniteSummary,
    anisotropy: OptionalFiniteSummary,
    local_relief: Vec<ReliefObjectSummary>,
    summit_caps: Vec<CapObjectSummary>,
}

#[derive(Serialize)]
struct ReliefObjectSummary {
    radius_km: f64,
    area_weighted_p50_km: OptionalFiniteSummary,
    area_weighted_p90_km: OptionalFiniteSummary,
}

#[derive(Serialize)]
struct CapObjectSummary {
    depth_km: f64,
    area_km2: OptionalFiniteSummary,
    fraction: OptionalFiniteSummary,
    gentle_fractions: Vec<GentleObjectSummary>,
}

#[derive(Serialize)]
struct GentleObjectSummary {
    grade_threshold: f64,
    fraction: OptionalFiniteSummary,
}

#[derive(Serialize)]
struct OptionalFiniteSummary {
    present_count: usize,
    missing_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    values: Option<FiniteSummary>,
}

#[derive(Serialize)]
struct FiniteSummary {
    min: f64,
    p10: f64,
    p50: f64,
    p90: f64,
    max: f64,
}

struct SurfaceInput<'a> {
    id: &'static str,
    graph_id: &'static str,
    elevation_source: &'static str,
    graph: &'a EvaluationSurfaceGraphV0,
    native_elevation: &'a [f32],
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    validate_runtime(&cli.output)?;

    let generation_start = Instant::now();
    let stage_start = Instant::now();
    let mut world = World::new_with_options(
        SEED,
        COARSE_REQUESTED_CELLS,
        LLOYD_METADATA,
        VoronoiBackend::ConvexHull,
    );
    world.orogen_model = OrogenModel::Legacy;
    world.fine_cache = FineCacheMode::Disabled;
    world.generate_all(NUM_PLATES_DEFAULT);
    let lithosphere = stage_start.elapsed().as_secs_f64();

    let stage_start = Instant::now();
    world.generate_atmosphere();
    let atmosphere = stage_start.elapsed().as_secs_f64();

    let stage_start = Instant::now();
    world.generate_fine_pre_with_cap(FINE_MAX_CELLS);
    let fine_pre = stage_start.elapsed().as_secs_f64();

    let stage_start = Instant::now();
    world.generate_fine_eroded();
    let erosion = stage_start.elapsed().as_secs_f64();
    let product_generation_total = generation_start.elapsed().as_secs_f64();

    let manifest = world.manifest();
    validate_manifest(&manifest)?;
    let config = SurfaceHierarchyConfigV0::default();

    let g0_start = Instant::now();
    let coarse_graph = adapt_product_tessellation_graph_v0(&world.tessellation, &config)?;
    let coarse_g0_seconds = g0_start.elapsed().as_secs_f64();

    let fine = world
        .fine
        .as_ref()
        .ok_or("frozen run did not retain the fine world")?;
    let eroded = fine
        .eroded
        .as_ref()
        .ok_or("frozen run did not retain the stage-4 surface")?;
    let g0_start = Instant::now();
    let fine_graph = adapt_product_tessellation_graph_v0(&fine.base.tessellation, &config)?;
    let fine_g0_seconds = g0_start.elapsed().as_secs_f64();

    let raw_stage4: Vec<f32> = (0..fine.base.tessellation.num_cells())
        .map(|cell| eroded.hydrology.pre_integration_elevation(cell))
        .collect();
    let fine_base_elevations_byte_identical =
        f32_slices_byte_identical(&fine.base.coarse_base_elevation, &fine.base.base_elevation);

    let inputs = [
        SurfaceInput {
            id: "coarse-stage1",
            graph_id: "coarse-g0",
            elevation_source: "World.elevation.values",
            graph: &coarse_graph,
            native_elevation: &world
                .elevation
                .as_ref()
                .ok_or("stage-1 elevation unavailable")?
                .values,
        },
        SurfaceInput {
            id: "fine-base-raw",
            graph_id: "fine-g0",
            elevation_source: "FineBase.base_elevation",
            graph: &fine_graph,
            native_elevation: &fine.base.base_elevation,
        },
        SurfaceInput {
            id: "fine-stage3-final",
            graph_id: "fine-g0",
            elevation_source: "FineWorld.pre.elevation.values",
            graph: &fine_graph,
            native_elevation: &fine.pre.elevation.values,
        },
        SurfaceInput {
            id: "fine-stage4-raw",
            graph_id: "fine-g0",
            elevation_source: "FineWorld.eroded.hydrology.pre_integration_elevation(i)",
            graph: &fine_graph,
            native_elevation: &raw_stage4,
        },
        SurfaceInput {
            id: "fine-stage4-final",
            graph_id: "fine-g0",
            elevation_source: "FineWorld.eroded.elevation.values",
            graph: &fine_graph,
            native_elevation: &eroded.elevation.values,
        },
    ];
    let surfaces = inputs
        .iter()
        .map(|input| observe_surface(input, config))
        .collect::<Result<Vec<_>, _>>()?;

    let graphs = vec![
        summarize_graph("coarse-g0", &coarse_graph, coarse_g0_seconds),
        summarize_graph("fine-g0", &fine_graph, fine_g0_seconds),
    ];
    let mut artifact = Artifact {
        schema_version: ARTIFACT_SCHEMA_VERSION,
        observation_id: OBSERVATION_ID,
        contract: CONTRACT_PATH,
        interpretation: InterpretationGuard {
            observation_kind: "bounded noncompetitive descriptive product ancestry observation",
            has_drainage_d0: false,
            has_relationships_o0: false,
            highland_feature_semantics: "operational retained split-tree branch; not a promoted range or massif",
            broad_cap_semantics: "continuous summit-cap, local-relief, and gentle-fraction evidence; no plateau classifier",
        },
        scored_policy: ScoredPolicy {
            scored_domain: "every product cell",
            active_test: "elevation_km > 0.0 (strict)",
            elevation_conversion: "f64(native_elevation) * f64(ELEVATION_UNIT_KM), exactly once",
            datum: "product coarse sea-level datum; no hydrologic water level or normalization",
        },
        frozen_run: FrozenRun {
            seed: SEED,
            coarse_requested_cells: COARSE_REQUESTED_CELLS,
            coarse_backend: VoronoiBackend::ConvexHull,
            lloyd_metadata: LLOYD_METADATA,
            plates: NUM_PLATES_DEFAULT,
            orogen_model: OrogenModel::Legacy,
            fine_cap: FINE_MAX_CELLS,
            fine_cache: FineCacheMode::Disabled,
            computed_stage: COMPUTED_STAGE,
            terrain_climate_hydrology_erosion_controls_product_defaults: true,
        },
        hierarchy_config: config,
        manifest,
        command: CommandRecord {
            invoked_argv: env::args().collect(),
            required_guarded_invocation: "cargo build --release --bin landform_baseline; /usr/bin/time -v timeout 20m target/release/landform_baseline <output.json>",
        },
        platform: platform_record(),
        memory: MemoryRecord {
            whole_process_peak_rss_capture: "authoritative value is external /usr/bin/time -v Maximum resident set size; preserve its stderr with the audit",
            proc_self_vmhwm_kib_before_publish: proc_status_kib("VmHWM"),
            retained_memory: None,
            retained_memory_status: "unavailable",
        },
        timings_seconds: TimingRecord {
            lithosphere,
            atmosphere,
            fine_pre,
            erosion,
            product_generation_total,
            ..TimingRecord::default()
        },
        fine_base_elevations_byte_identical,
        graphs,
        surfaces,
    };

    // JSON cannot contain the duration of its own final serialization. Measure
    // an otherwise identical compact serialization pass, store that duration,
    // then perform the single atomic publication serialization.
    let serialization_start = Instant::now();
    serde_json::to_writer(std::io::sink(), &artifact)?;
    artifact.timings_seconds.artifact_serialization_measurement =
        serialization_start.elapsed().as_secs_f64();
    atomic_write_json(&cli.output, &artifact)?;
    Ok(())
}

fn validate_runtime(output: &Path) -> Result<(), Box<dyn Error>> {
    if cfg!(debug_assertions) {
        return Err("the frozen observation must be run with a release build".into());
    }
    if env::consts::OS != "linux" {
        return Err("the frozen observation must be run on Linux under WSL2".into());
    }
    let release = fs::read_to_string("/proc/sys/kernel/osrelease").unwrap_or_default();
    if !release.to_ascii_lowercase().contains("wsl2") {
        return Err("the frozen observation requires the preregistered WSL2 CPU platform".into());
    }
    let parent = output_parent(output);
    if !parent.is_dir() {
        return Err(format!("output parent does not exist: {}", parent.display()).into());
    }
    Ok(())
}

fn validate_manifest(manifest: &RunManifest) -> Result<(), Box<dyn Error>> {
    if manifest.seed != SEED
        || manifest.voronoi_backend != VoronoiBackend::ConvexHull
        || manifest.lloyd_iterations != LLOYD_METADATA
        || manifest.plate_count != Some(NUM_PLATES_DEFAULT)
        || manifest.orogen_model != OrogenModel::Legacy
        || manifest.fine_cache_mode != FineCacheMode::Disabled
        || manifest.computed_stage != COMPUTED_STAGE
    {
        return Err("generated manifest diverged from the frozen run".into());
    }
    let cache = manifest
        .fine_cache
        .as_ref()
        .ok_or("fine cache record unavailable")?;
    if cache.max_cells != FINE_MAX_CELLS || cache.actual_cells != manifest.active_cells {
        return Err("fine cache record diverged from the frozen fine run".into());
    }
    Ok(())
}

fn summarize_graph(
    id: &'static str,
    graph: &EvaluationSurfaceGraphV0,
    g0_adaptation_seconds: f64,
) -> GraphArtifact {
    GraphArtifact {
        id,
        domain: graph.domain,
        cells: graph.cell_count(),
        directed_edges: graph.edge_neighbor.len(),
        physical_area_km2: graph.cell_area_km2.iter().sum(),
        g0_adaptation_seconds,
    }
}

fn observe_surface(
    input: &SurfaceInput<'_>,
    config: SurfaceHierarchyConfigV0,
) -> Result<SurfaceArtifact, Box<dyn Error>> {
    if input.native_elevation.len() != input.graph.cell_count() {
        return Err(format!("{} elevation length does not match G0", input.id).into());
    }
    let elevation_km: Vec<f64> = input
        .native_elevation
        .iter()
        .map(|&value| f64::from(value) * f64::from(ELEVATION_UNIT_KM))
        .collect();
    let scored = vec![true; elevation_km.len()];
    let active: Vec<bool> = elevation_km.iter().map(|&value| value > 0.0).collect();
    let active_cell_count = active.iter().filter(|&&value| value).count();
    let total_area: f64 = input.graph.cell_area_km2.iter().sum();
    let active_area: f64 = active
        .iter()
        .zip(&input.graph.cell_area_km2)
        .filter_map(|(&is_active, &area)| is_active.then_some(area))
        .sum();

    let extraction_start = Instant::now();
    let hierarchy = build_surface_hierarchy_v0(input.graph, &elevation_km, &scored, config)?;
    let s0_morphology_seconds = extraction_start.elapsed().as_secs_f64();
    let measured_peak_ids: Vec<u32> = hierarchy
        .reference_highlands
        .iter()
        .map(|feature| feature.peak_id)
        .collect();
    if measured_peak_ids != hierarchy.populations.reference {
        return Err(format!(
            "{} reference records diverged from the population",
            input.id
        )
        .into());
    }

    let reference_counts = reference_counts(&hierarchy);
    let reference_footprint_union_area_km2 = reference_union_area(input.graph, &hierarchy);
    let object_summaries = object_summaries(&hierarchy, &config);
    let largest_reference_objects = ranked_objects(&hierarchy, |peak| peak.footprint_area_km2);
    let most_persistent_reference_objects = ranked_objects(&hierarchy, |peak| peak.persistence_km);
    let reference_highlands = reference_records(&hierarchy)?;

    Ok(SurfaceArtifact {
        id: input.id,
        graph_id: input.graph_id,
        elevation_source: input.elevation_source,
        elevation_hash_version: ELEVATION_HASH_VERSION,
        elevation_hash_hex: format!("{:016x}", hash_f64_slice(&elevation_km)),
        active: ActiveDomainSummary {
            cell_count: active_cell_count,
            cell_fraction: active_cell_count as f64 / elevation_km.len() as f64,
            area_km2: active_area,
            area_fraction: active_area / total_area,
        },
        structural_counts: structural_counts(&hierarchy),
        population_counts: PopulationCounts {
            reference: hierarchy.populations.reference.len(),
            persistence_low: hierarchy.populations.persistence_low.len(),
            persistence_high: hierarchy.populations.persistence_high.len(),
            footprint_low: hierarchy.populations.footprint_low.len(),
            footprint_high: hierarchy.populations.footprint_high.len(),
        },
        derived_evidence_hash_hex: format!("{:016x}", hierarchy.derived_evidence_hash),
        s0_morphology_seconds,
        reference_counts,
        reference_footprint_union_area_km2,
        object_summaries,
        largest_reference_objects,
        most_persistent_reference_objects,
        reference_highlands,
    })
}

fn structural_counts(hierarchy: &SurfaceHierarchyV0) -> StructuralCounts {
    StructuralCounts {
        raw_peaks: hierarchy.peaks.len(),
        raw_saddles: hierarchy.saddles.len(),
        roots: hierarchy.roots.len(),
        child_peaks: hierarchy
            .peaks
            .iter()
            .filter(|peak| peak.parent_peak.is_some())
            .count(),
        flat_maximum_nodes: hierarchy
            .peaks
            .iter()
            .filter(|peak| peak.flat_maximum_cells.len() > 1)
            .count(),
        flat_maximum_cells: hierarchy
            .peaks
            .iter()
            .filter(|peak| peak.flat_maximum_cells.len() > 1)
            .map(|peak| peak.flat_maximum_cells.len())
            .sum(),
        flat_saddle_nodes: hierarchy
            .saddles
            .iter()
            .filter(|saddle| saddle.flat_saddle_cells.len() > 1)
            .count(),
        flat_saddle_cells: hierarchy
            .saddles
            .iter()
            .filter(|saddle| saddle.flat_saddle_cells.len() > 1)
            .map(|saddle| saddle.flat_saddle_cells.len())
            .sum(),
        equal_elder_ambiguous_peaks: hierarchy
            .peaks
            .iter()
            .filter(|peak| peak.equal_elder_ambiguous)
            .count(),
        equal_elder_ambiguous_saddles: hierarchy
            .saddles
            .iter()
            .filter(|saddle| saddle.equal_elder_ambiguous)
            .count(),
    }
}

fn reference_counts(hierarchy: &SurfaceHierarchyV0) -> ReferenceEvidenceCounts {
    let mut counts = ReferenceEvidenceCounts::default();
    for feature in &hierarchy.reference_highlands {
        let peak = &hierarchy.peaks[feature.peak_id as usize];
        if peak.parent_peak.is_some() {
            counts.child_features += 1;
        } else {
            counts.root_features += 1;
        }
        let measurements = match &feature.measurements {
            HighlandMeasurementsV0::Spherical(measurements) => measurements,
            HighlandMeasurementsV0::Planar(_) => continue,
        };
        match &measurements.footprint_geometry {
            SphericalFootprintGeometryV0::Local(geometry) => {
                counts.local_geometry += 1;
                counts.spherical_nonlocal_warnings +=
                    usize::from(geometry.spherical_nonlocal_warning);
                counts.orientation_ambiguous += usize::from(geometry.orientation_ambiguous);
            }
            SphericalFootprintGeometryV0::NonLocalGeometry => counts.nonlocal_geometry += 1,
        }
        if !measurements.rank_deficient_grade_cells.is_empty() {
            counts.rank_deficient_grade_objects += 1;
            counts.rank_deficient_grade_cells += measurements.rank_deficient_grade_cells.len();
        }
        let truncated: Vec<&LocalReliefSummaryV0> = measurements
            .local_relief
            .iter()
            .filter(|summary| !summary.truncated_member_cells.is_empty())
            .collect();
        if !truncated.is_empty() {
            counts.relief_truncated_objects += 1;
            counts.relief_truncated_summaries += truncated.len();
            counts.relief_truncated_member_occurrences += truncated
                .iter()
                .map(|summary| summary.truncated_member_cells.len())
                .sum::<usize>();
        }
        let censored = measurements
            .summit_caps
            .iter()
            .filter(|cap| cap.cap_merge_censored)
            .count();
        if censored > 0 {
            counts.cap_merge_censored_objects += 1;
            counts.cap_merge_censored_summaries += censored;
        }
    }
    counts
}

fn reference_union_area(graph: &EvaluationSurfaceGraphV0, hierarchy: &SurfaceHierarchyV0) -> f64 {
    let cells: BTreeSet<u32> = hierarchy
        .populations
        .reference
        .iter()
        .flat_map(|&peak_id| {
            hierarchy.peaks[peak_id as usize]
                .footprint_members
                .iter()
                .copied()
        })
        .collect();
    cells
        .into_iter()
        .map(|cell| graph.cell_area_km2[cell as usize])
        .sum()
}

fn reference_records(
    hierarchy: &SurfaceHierarchyV0,
) -> Result<Vec<ReferenceHighlandRecord>, Box<dyn Error>> {
    hierarchy
        .reference_highlands
        .iter()
        .map(|feature| {
            let peak = hierarchy
                .peaks
                .get(feature.peak_id as usize)
                .ok_or("reference feature peak ID is invalid")?;
            let key_saddle = peak
                .key_saddle
                .map(|id| {
                    hierarchy
                        .saddles
                        .get(id as usize)
                        .cloned()
                        .ok_or("reference feature saddle ID is invalid")
                })
                .transpose()?;
            Ok(ReferenceHighlandRecord {
                feature: feature.clone(),
                peak: peak.clone(),
                key_saddle,
            })
        })
        .collect()
}

fn ranked_objects(
    hierarchy: &SurfaceHierarchyV0,
    value: impl Fn(&PeakBranchV0) -> f64,
) -> Vec<RankedObject> {
    let mut ranked: Vec<RankedObject> = hierarchy
        .populations
        .reference
        .iter()
        .map(|&peak_id| RankedObject {
            peak_id,
            value: value(&hierarchy.peaks[peak_id as usize]),
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.value
            .total_cmp(&a.value)
            .then_with(|| a.peak_id.cmp(&b.peak_id))
    });
    ranked.truncate(5);
    ranked
}

fn object_summaries(
    hierarchy: &SurfaceHierarchyV0,
    config: &SurfaceHierarchyConfigV0,
) -> ObjectSummaries {
    let count = hierarchy.reference_highlands.len();
    let peaks: Vec<&PeakBranchV0> = hierarchy
        .reference_highlands
        .iter()
        .map(|feature| &hierarchy.peaks[feature.peak_id as usize])
        .collect();

    let two_sweep = hierarchy.reference_highlands.iter().map(|feature| {
        spherical_measurements(feature).and_then(|measurements| measurements.two_sweep_extent_km)
    });
    let mean_width = hierarchy.reference_highlands.iter().map(|feature| {
        spherical_measurements(feature).and_then(|measurements| measurements.mean_width_km)
    });
    let geometry_value =
        |feature: &HighlandFeatureV0,
         select: fn(&hex3::world::landforms::SphericalLocalFootprintGeometryV0) -> f64| {
            spherical_measurements(feature).and_then(|measurements| {
                match &measurements.footprint_geometry {
                    SphericalFootprintGeometryV0::Local(geometry) => Some(select(geometry)),
                    SphericalFootprintGeometryV0::NonLocalGeometry => None,
                }
            })
        };

    let local_relief = config
        .local_relief_radii_km
        .iter()
        .enumerate()
        .map(|(index, &radius_km)| ReliefObjectSummary {
            radius_km,
            area_weighted_p50_km: summarize_optional(
                hierarchy.reference_highlands.iter().map(|feature| {
                    relief_at(feature, index, radius_km).map(|summary| summary.area_weighted_p50_km)
                }),
                count,
            ),
            area_weighted_p90_km: summarize_optional(
                hierarchy.reference_highlands.iter().map(|feature| {
                    relief_at(feature, index, radius_km).map(|summary| summary.area_weighted_p90_km)
                }),
                count,
            ),
        })
        .collect();

    let summit_caps =
        config
            .summit_cap_depths_km
            .iter()
            .enumerate()
            .map(|(cap_index, &depth_km)| CapObjectSummary {
                depth_km,
                area_km2: summarize_optional(
                    hierarchy.reference_highlands.iter().map(|feature| {
                        cap_at(feature, cap_index, depth_km).map(|cap| cap.area_km2)
                    }),
                    count,
                ),
                fraction: summarize_optional(
                    hierarchy.reference_highlands.iter().map(|feature| {
                        cap_at(feature, cap_index, depth_km).map(|cap| cap.fraction)
                    }),
                    count,
                ),
                gentle_fractions: config
                    .gentle_grade_thresholds
                    .iter()
                    .enumerate()
                    .map(|(grade_index, &grade_threshold)| GentleObjectSummary {
                        grade_threshold,
                        fraction: summarize_optional(
                            hierarchy.reference_highlands.iter().map(|feature| {
                                cap_at(feature, cap_index, depth_km).and_then(|cap| {
                                    cap.gentle_fractions.get(grade_index).and_then(|gentle| {
                                        same_f64(gentle.grade_threshold, grade_threshold)
                                            .then_some(gentle.fraction)
                                    })
                                })
                            }),
                            count,
                        ),
                    })
                    .collect(),
            })
            .collect();

    ObjectSummaries {
        persistence_km: summarize_optional(
            peaks.iter().map(|peak| Some(peak.persistence_km)),
            count,
        ),
        footprint_area_km2: summarize_optional(
            peaks.iter().map(|peak| Some(peak.footprint_area_km2)),
            count,
        ),
        two_sweep_extent_km: summarize_optional(two_sweep, count),
        mean_width_km: summarize_optional(mean_width, count),
        equivalent_ellipse_length_km: summarize_optional(
            hierarchy.reference_highlands.iter().map(|feature| {
                geometry_value(feature, |geometry| geometry.equivalent_ellipse_length_km)
            }),
            count,
        ),
        equivalent_ellipse_width_km: summarize_optional(
            hierarchy.reference_highlands.iter().map(|feature| {
                geometry_value(feature, |geometry| geometry.equivalent_ellipse_width_km)
            }),
            count,
        ),
        anisotropy: summarize_optional(
            hierarchy
                .reference_highlands
                .iter()
                .map(|feature| geometry_value(feature, |geometry| geometry.anisotropy)),
            count,
        ),
        local_relief,
        summit_caps,
    }
}

fn spherical_measurements(
    feature: &HighlandFeatureV0,
) -> Option<&hex3::world::landforms::SphericalHighlandMeasurementsV0> {
    match &feature.measurements {
        HighlandMeasurementsV0::Spherical(measurements) => Some(measurements),
        HighlandMeasurementsV0::Planar(_) => None,
    }
}

fn relief_at(
    feature: &HighlandFeatureV0,
    index: usize,
    expected_radius: f64,
) -> Option<&LocalReliefSummaryV0> {
    spherical_measurements(feature)?
        .local_relief
        .get(index)
        .filter(|summary| same_f64(summary.radius_km, expected_radius))
}

fn cap_at(
    feature: &HighlandFeatureV0,
    index: usize,
    expected_depth: f64,
) -> Option<&SummitCapSummaryV0> {
    spherical_measurements(feature)?
        .summit_caps
        .get(index)
        .filter(|cap| same_f64(cap.depth_km, expected_depth))
}

fn same_f64(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

fn summarize_optional(
    values: impl IntoIterator<Item = Option<f64>>,
    expected_count: usize,
) -> OptionalFiniteSummary {
    let mut finite: Vec<f64> = values
        .into_iter()
        .flatten()
        .filter(|value| value.is_finite())
        .collect();
    finite.sort_by(f64::total_cmp);
    let present_count = finite.len();
    OptionalFiniteSummary {
        present_count,
        missing_count: expected_count.saturating_sub(present_count),
        values: (!finite.is_empty()).then(|| FiniteSummary {
            min: finite[0],
            p10: nearest_rank(&finite, 0.10),
            p50: nearest_rank(&finite, 0.50),
            p90: nearest_rank(&finite, 0.90),
            max: finite[finite.len() - 1],
        }),
    }
}

fn nearest_rank(sorted: &[f64], quantile: f64) -> f64 {
    let index = ((quantile * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

fn hash_f64_slice(values: &[f64]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

fn f32_slices_byte_identical(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(&left, &right)| left.to_bits() == right.to_bits())
}

fn platform_record() -> PlatformRecord {
    PlatformRecord {
        os: env::consts::OS,
        family: env::consts::FAMILY,
        architecture: env::consts::ARCH,
        kernel_release: read_trimmed("/proc/sys/kernel/osrelease"),
        kernel_version: read_trimmed("/proc/version"),
        wsl_distribution: env::var("WSL_DISTRO_NAME").ok(),
        release_build_required: true,
        debug_assertions_enabled: cfg!(debug_assertions),
    }
}

fn read_trimmed(path: &str) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn proc_status_kib(field: &str) -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        (name == field)
            .then(|| value.split_whitespace().next()?.parse().ok())
            .flatten()
    })
}

fn atomic_write_json(path: &Path, artifact: &Artifact) -> Result<(), Box<dyn Error>> {
    let file_name = path
        .file_name()
        .ok_or("output path must name a JSON artifact")?
        .to_string_lossy();
    let temporary = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));
    let result = (|| -> Result<(), Box<dyn Error>> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        serde_json::to_writer(&mut file, artifact)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        OpenOptions::new()
            .read(true)
            .open(output_parent(path))?
            .sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn output_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_and_missing_values_follow_contract() {
        let summary =
            summarize_optional([Some(5.0), None, Some(1.0), Some(4.0), Some(f64::NAN)], 5);
        assert_eq!(summary.present_count, 3);
        assert_eq!(summary.missing_count, 2);
        let values = summary.values.unwrap();
        assert_eq!(values.min, 1.0);
        assert_eq!(values.p10, 1.0);
        assert_eq!(values.p50, 4.0);
        assert_eq!(values.p90, 5.0);
        assert_eq!(values.max, 5.0);
    }

    #[test]
    fn native_identity_and_converted_hash_are_bit_sensitive() {
        assert!(f32_slices_byte_identical(&[0.0, 1.0], &[0.0, 1.0]));
        assert!(!f32_slices_byte_identical(&[0.0], &[-0.0]));
        assert_ne!(hash_f64_slice(&[0.0]), hash_f64_slice(&[-0.0]));
    }
}
