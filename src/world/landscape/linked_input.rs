//! Immutable, arm-neutral inputs for the preregistered linked-orogen case.
//!
//! This module intentionally stops before terrain ownership. It materializes
//! only the mesh, forcing inputs, initial scalar field, local runoff, material
//! membership, and candidate evaluation geometry.

use super::{
    linked_scenario, BoundaryFaceCondition, BoundarySide, DeformationEpisode, EpisodeId,
    LandscapeMesh, LandscapeScenario, OutletPortalId, SegmentId, SupportStencil,
    REFERENCE_ROCK_VOLUME_RATE_KM3_MYR,
};
use bincode::Options;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;

pub const LINKED_INPUT_SCHEMA_VERSION_V0: &str = "orogen-linked-shared-input-v0";
pub const LINKED_INPUT_HASH_VERSION_V0: &str = "fnv1a64-bincode-fixint-le-v0";
pub const LINKED_INPUT_BUNDLE_HASH_V0: u64 = 0x0d6a_4ab7_aec2_4e68;
const MANIFEST_SCHEMA_VERSION_V0: &str = "orogen-linked-shared-input-manifest-json-v0";
const WIDTH_KM: f64 = 960.0;
const HEIGHT_KM: f64 = 640.0;
const SPACINGS_KM: [f64; 3] = [8.0, 4.0, 2.0];
const INITIAL_SEED: u64 = 12_345;
const RUNOFF_DEPTH_RATE_KM_MYR: f64 = 500.0;
const ACTIVITY_INTEGRAL_MYR: f64 = 5.75;
const TOTAL_WORK_KM3: f64 = 100_625.0;
const ORACLE_TIMES_MYR: [f64; 9] = [0.0, 0.125, 0.25, 3.0, 5.75, 5.875, 6.0, 8.0, 10.0];
const ORACLE_ACTIVITIES: [f64; 9] = [0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0];
const MAX_ENCODED_BUNDLE_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkedSharedInputBundleV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub declaration: LinkedInputDeclarationV0,
    pub resolutions: Vec<LinkedResolutionInputV0>,
    pub derived_bundle_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkedInputDeclarationV0 {
    pub requested_width_km: f64,
    pub requested_height_km: f64,
    pub spacings_km: Vec<f64>,
    pub case_horizon_myr: f64,
    pub mesh_constructor_id: String,
    pub scenario: LandscapeScenario,
    pub forcing_compiler_id: String,
    pub forcing_compiler_semantics: ForcingCompilerSemanticsV0,
    pub activity_policy_id: String,
    pub forcing_oracle_times_myr: Vec<f64>,
    pub analytic_activity_integral_myr: f64,
    pub analytic_rock_volume_km3: f64,
    pub work_ledgers: Vec<DeclaredWorkLedgerV0>,
    pub initial_surface: InitialSurfaceDeclarationV0,
    pub runoff: RunoffDeclarationV0,
    pub material_id: String,
    pub candidate_geometry_id: String,
    pub units: LinkedInputUnitsV0,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkedResolutionInputV0 {
    pub nominal_spacing_km: f64,
    pub mesh: LandscapeMesh,
    pub initial_elevation_km: Vec<f64>,
    pub local_runoff_supply_km3_myr: Vec<f64>,
    pub base_material_present: Vec<bool>,
    pub whole_graph_candidate: Vec<bool>,
    pub central_window_candidate: Vec<bool>,
    pub compiled_stencils: Vec<SupportStencil>,
    pub cumulative_rock_displacement_km: Vec<f64>,
    pub frame_witnesses: Vec<ForcingFrameWitnessV0>,
    pub summary: LinkedResolutionSummaryV0,
    pub component_hashes: LinkedInputComponentHashesV0,
    pub derived_resolution_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InitialSurfaceDeclarationV0 {
    pub generator_id: String,
    pub seed: u64,
    pub phase_streams: [u64; 3],
    pub base_elevation_km: f64,
    pub amplitudes_km: [f64; 3],
    pub wave_vectors_per_km: [[f64; 2]; 3],
    pub taper_full_height_km: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunoffDeclarationV0 {
    pub generator_id: String,
    pub depth_rate_km_myr: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForcingCompilerSemanticsV0 {
    pub field_consumed: Vec<String>,
    pub identity_or_output_only: Vec<String>,
    pub retained_but_ignored: Vec<String>,
    pub horizontal_velocity_policy: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeclaredWorkLedgerV0 {
    pub episode_id: EpisodeId,
    pub segment_id: Option<SegmentId>,
    pub activity_integral_myr: f64,
    pub share: f64,
    pub positive_rock_volume_km3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkedInputUnitsV0 {
    pub coordinate: String,
    pub elevation: String,
    pub area: String,
    pub time: String,
    pub vertical_rate: String,
    pub runoff_depth_rate: String,
    pub runoff_supply: String,
    pub volume: String,
    pub support: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForcingFrameWitnessV0 {
    pub time_myr: f64,
    pub expected_activity: f64,
    pub vertical_rate_hash: u64,
    pub horizontal_velocity_hash: u64,
    pub dominant_episode_hash: u64,
    pub integrated_rate_km3_myr: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkedResolutionSummaryV0 {
    pub cell_count: u64,
    pub directed_edge_count: u64,
    pub raw_exposed_face_count: u64,
    pub split_boundary_record_count: u64,
    pub actual_domain_area_km2: f64,
    pub physical_boundary_arc_km: f64,
    pub center_bounds_km: [f64; 4],
    pub portal_summaries: Vec<LinkedPortalSummaryV0>,
    pub initial_min_km: f64,
    pub initial_max_km: f64,
    pub local_runoff_total_km3_myr: f64,
    pub base_material_present_count: u64,
    pub whole_graph_count: u64,
    pub central_window_count: u64,
    pub central_window_area_km2: f64,
    pub cumulative_rock_volume_km3: f64,
    pub stencil_summaries: Vec<LinkedStencilSummaryV0>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkedPortalSummaryV0 {
    pub portal_id: OutletPortalId,
    pub face_record_count: u64,
    pub owner_cell_count: u64,
    pub projected_coverage_km: f64,
    pub physical_open_arc_km: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkedStencilSummaryV0 {
    pub segment_id: SegmentId,
    pub support_cell_count: u64,
    pub minimum_weight_per_km2: f64,
    pub maximum_weight_per_km2: f64,
    pub area_integral: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkedInputComponentHashesV0 {
    pub mesh_hash: u64,
    pub initial_elevation_hash: u64,
    pub local_runoff_hash: u64,
    pub base_material_present_hash: u64,
    pub whole_graph_candidate_hash: u64,
    pub central_window_candidate_hash: u64,
    pub compiled_stencils_hash: u64,
    pub cumulative_rock_displacement_hash: u64,
    pub frame_witnesses_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkedSharedInputManifestJsonV0 {
    pub schema_version: String,
    pub semantic_schema_version: String,
    pub hash_version: String,
    pub derived_bundle_hash_hex: String,
    pub declaration: LinkedInputDeclarationV0,
    pub resolutions: Vec<LinkedResolutionManifestJsonV0>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkedResolutionManifestJsonV0 {
    pub nominal_spacing_km: f64,
    pub summary: LinkedResolutionSummaryV0,
    pub component_hashes_hex: Vec<String>,
    pub derived_resolution_hash_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkedInputErrorV0(pub String);

impl fmt::Display for LinkedInputErrorV0 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LinkedInputErrorV0 {}

/// The single library authority for the linked-case low-relief initial field.
pub fn linked_low_relief_initial_surface(mesh: &LandscapeMesh, seed: u64) -> Vec<f64> {
    let phases = [phase(seed, 1), phase(seed, 2), phase(seed, 3)];
    mesh.cell_center_km
        .iter()
        .map(|center| {
            let yhat = (2.0 * center.y / HEIGHT_KM).clamp(-1.0, 1.0);
            let taper = (1.0 - yhat * yhat).max(0.0);
            let perturbation = 0.0020 * (0.071 * center.x + 0.043 * center.y + phases[0]).sin()
                + 0.0015 * (-0.038 * center.x + 0.063 * center.y + phases[1]).sin()
                + 0.0010 * (0.027 * center.x - 0.052 * center.y + phases[2]).sin();
            taper * (0.020 + perturbation)
        })
        .collect()
}

fn phase(seed: u64, stream: u64) -> f64 {
    let random = splitmix64(seed ^ stream.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let unit = (random >> 11) as f64 / ((1_u64 << 53) as f64);
    unit * std::f64::consts::TAU
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub fn build_linked_shared_input_bundle_v0() -> Result<LinkedSharedInputBundleV0, LinkedInputErrorV0>
{
    let bundle = assemble_bundle()?;
    validate_linked_shared_input_bundle_v0(&bundle)?;
    Ok(bundle)
}

fn assemble_bundle() -> Result<LinkedSharedInputBundleV0, LinkedInputErrorV0> {
    let schema_version = LINKED_INPUT_SCHEMA_VERSION_V0.to_owned();
    let hash_version = LINKED_INPUT_HASH_VERSION_V0.to_owned();
    let declaration = canonical_declaration();
    validate_scenario(&declaration.scenario)?;
    let mut resolutions = Vec::with_capacity(SPACINGS_KM.len());
    for spacing in SPACINGS_KM {
        resolutions.push(assemble_resolution(
            &schema_version,
            &hash_version,
            &declaration.scenario,
            spacing,
        )?);
    }
    let mut bundle = LinkedSharedInputBundleV0 {
        schema_version,
        hash_version,
        declaration,
        resolutions,
        derived_bundle_hash: 0,
    };
    bundle.derived_bundle_hash = bundle_hash(&bundle)?;
    Ok(bundle)
}

fn canonical_declaration() -> LinkedInputDeclarationV0 {
    let strings = |values: &[&str]| values.iter().map(|value| (*value).to_owned()).collect();
    LinkedInputDeclarationV0 {
        requested_width_km: WIDTH_KM,
        requested_height_km: HEIGHT_KM,
        spacings_km: SPACINGS_KM.to_vec(),
        case_horizon_myr: 10.0,
        mesh_constructor_id: "centered-pointy-row-full-hex-v0".into(),
        scenario: linked_scenario(),
        forcing_compiler_id: "linked-cosine-support-area-normalized-v0".into(),
        forcing_compiler_semantics: ForcingCompilerSemanticsV0 {
            field_consumed: strings(&[
                "segment.id",
                "segment.geometry",
                "segment.width_km",
                "segment.along_strike_taper",
                "episode.active_myr",
                "episode.ramp_myr",
                "episode.rock_volume_rate_km3_myr",
                "episode.segment_shares.segment_id",
                "episode.segment_shares.share",
            ]),
            identity_or_output_only: strings(&["episode.id"]),
            retained_but_ignored: strings(&["scenario.id", "segment.vergence", "segment.links"]),
            horizontal_velocity_policy: "identically-zero-v0".into(),
        },
        activity_policy_id: "smoothstep-episode-ends-v0".into(),
        forcing_oracle_times_myr: ORACLE_TIMES_MYR.to_vec(),
        analytic_activity_integral_myr: ACTIVITY_INTEGRAL_MYR,
        analytic_rock_volume_km3: TOTAL_WORK_KM3,
        work_ledgers: vec![
            DeclaredWorkLedgerV0 {
                episode_id: EpisodeId(0),
                segment_id: None,
                activity_integral_myr: ACTIVITY_INTEGRAL_MYR,
                share: 1.0,
                positive_rock_volume_km3: TOTAL_WORK_KM3,
            },
            DeclaredWorkLedgerV0 {
                episode_id: EpisodeId(0),
                segment_id: Some(SegmentId(0)),
                activity_integral_myr: ACTIVITY_INTEGRAL_MYR,
                share: 0.5,
                positive_rock_volume_km3: 50_312.5,
            },
            DeclaredWorkLedgerV0 {
                episode_id: EpisodeId(0),
                segment_id: Some(SegmentId(1)),
                activity_integral_myr: ACTIVITY_INTEGRAL_MYR,
                share: 0.5,
                positive_rock_volume_km3: 50_312.5,
            },
        ],
        initial_surface: InitialSurfaceDeclarationV0 {
            generator_id: "linked-low-relief-parabolic-taper-v0".into(),
            seed: INITIAL_SEED,
            phase_streams: [1, 2, 3],
            base_elevation_km: 0.020,
            amplitudes_km: [0.0020, 0.0015, 0.0010],
            wave_vectors_per_km: [[0.071, 0.043], [-0.038, 0.063], [0.027, -0.052]],
            taper_full_height_km: HEIGHT_KM,
        },
        runoff: RunoffDeclarationV0 {
            generator_id: "uniform-depth-local-supply-v0".into(),
            depth_rate_km_myr: RUNOFF_DEPTH_RATE_KM_MYR,
        },
        material_id: "uniform-present-base-material-v0".into(),
        candidate_geometry_id: "whole-and-central-cell-centre-candidates-v0".into(),
        units: LinkedInputUnitsV0 {
            coordinate: "km".into(),
            elevation: "km".into(),
            area: "km2".into(),
            time: "Myr".into(),
            vertical_rate: "km/Myr".into(),
            runoff_depth_rate: "km/Myr".into(),
            runoff_supply: "km3/Myr".into(),
            volume: "km3".into(),
            support: "km^-2".into(),
        },
    }
}

fn assemble_resolution(
    schema: &str,
    hash_version: &str,
    scenario: &LandscapeScenario,
    spacing: f64,
) -> Result<LinkedResolutionInputV0, LinkedInputErrorV0> {
    let mesh = LandscapeMesh::uniform_planar_hex(WIDTH_KM, HEIGHT_KM, spacing)
        .map_err(|error| LinkedInputErrorV0(error.to_string()))?;
    mesh.validate()
        .map_err(|error| LinkedInputErrorV0(error.to_string()))?;
    validate_mesh_geometry(&mesh, spacing)?;
    let evaluator = scenario
        .compile(&mesh)
        .map_err(|error| LinkedInputErrorV0(error.to_string()))?;
    let compiled_stencils = evaluator.support_stencils().to_vec();
    let n = mesh.cell_count();
    let initial_elevation_km = linked_low_relief_initial_surface(&mesh, INITIAL_SEED);
    let local_runoff_supply_km3_myr = mesh
        .cell_area_km2
        .iter()
        .map(|area| RUNOFF_DEPTH_RATE_KM_MYR * area)
        .collect::<Vec<_>>();
    let base_material_present = vec![true; n];
    let whole_graph_candidate = vec![true; n];
    let central_window_candidate = mesh
        .cell_center_km
        .iter()
        .map(|center| center.x.abs() <= 320.0 && center.y.abs() <= 160.0)
        .collect::<Vec<_>>();
    let cumulative_rock_displacement_km = (0..n)
        .map(|i| {
            compiled_stencils
                .iter()
                .map(|stencil| 50_312.5 * stencil.weight_per_km2[i])
                .sum()
        })
        .collect::<Vec<_>>();

    let mut frame_witnesses = Vec::with_capacity(ORACLE_TIMES_MYR.len());
    for (&time_myr, &expected_activity) in ORACLE_TIMES_MYR.iter().zip(&ORACLE_ACTIVITIES) {
        let frame = evaluator.evaluate(time_myr);
        if frame.rock_vertical_rate_km_myr.iter().any(|value| {
            !value.is_finite() || *value < 0.0 || (*value == 0.0 && value.to_bits() != 0)
        }) {
            return Err(LinkedInputErrorV0(
                "forcing compiler produced invalid or negative-zero vertical rate".into(),
            ));
        }
        if frame.horizontal_velocity_km_myr.iter().any(|value| {
            value
                .to_array()
                .into_iter()
                .any(|component| component.to_bits() != 0)
        }) {
            return Err(LinkedInputErrorV0(
                "forcing compiler produced nonzero horizontal velocity".into(),
            ));
        }
        frame_witnesses.push(ForcingFrameWitnessV0 {
            time_myr,
            expected_activity,
            vertical_rate_hash: frame_array_hash(
                "orogen-linked-input-v0/frame-vertical-rate",
                schema,
                hash_version,
                spacing,
                time_myr,
                &frame.rock_vertical_rate_km_myr,
            )?,
            horizontal_velocity_hash: frame_array_hash(
                "orogen-linked-input-v0/frame-horizontal-velocity",
                schema,
                hash_version,
                spacing,
                time_myr,
                &frame.horizontal_velocity_km_myr,
            )?,
            dominant_episode_hash: frame_array_hash(
                "orogen-linked-input-v0/frame-dominant-episode",
                schema,
                hash_version,
                spacing,
                time_myr,
                &frame.dominant_episode,
            )?,
            integrated_rate_km3_myr: evaluator.integrated_rate_km3_myr(&frame),
        });
    }

    let summary = resolution_summary(
        &mesh,
        &initial_elevation_km,
        &local_runoff_supply_km3_myr,
        &base_material_present,
        &whole_graph_candidate,
        &central_window_candidate,
        &compiled_stencils,
        &cumulative_rock_displacement_km,
    )?;
    validate_numerical_gates(spacing, &mesh, &frame_witnesses, &summary)?;

    let component_hashes = LinkedInputComponentHashesV0 {
        mesh_hash: component_hash(
            "orogen-linked-input-v0/mesh",
            schema,
            hash_version,
            spacing,
            &mesh,
        )?,
        initial_elevation_hash: component_hash(
            "orogen-linked-input-v0/initial-elevation",
            schema,
            hash_version,
            spacing,
            &initial_elevation_km,
        )?,
        local_runoff_hash: component_hash(
            "orogen-linked-input-v0/local-runoff",
            schema,
            hash_version,
            spacing,
            &local_runoff_supply_km3_myr,
        )?,
        base_material_present_hash: component_hash(
            "orogen-linked-input-v0/base-material-present",
            schema,
            hash_version,
            spacing,
            &base_material_present,
        )?,
        whole_graph_candidate_hash: component_hash(
            "orogen-linked-input-v0/whole-graph-candidate",
            schema,
            hash_version,
            spacing,
            &whole_graph_candidate,
        )?,
        central_window_candidate_hash: component_hash(
            "orogen-linked-input-v0/central-window-candidate",
            schema,
            hash_version,
            spacing,
            &central_window_candidate,
        )?,
        compiled_stencils_hash: component_hash(
            "orogen-linked-input-v0/compiled-stencils",
            schema,
            hash_version,
            spacing,
            &compiled_stencils,
        )?,
        cumulative_rock_displacement_hash: component_hash(
            "orogen-linked-input-v0/cumulative-rock-displacement",
            schema,
            hash_version,
            spacing,
            &cumulative_rock_displacement_km,
        )?,
        frame_witnesses_hash: component_hash(
            "orogen-linked-input-v0/frame-witnesses",
            schema,
            hash_version,
            spacing,
            &frame_witnesses,
        )?,
    };

    let mut resolution = LinkedResolutionInputV0 {
        nominal_spacing_km: spacing,
        mesh,
        initial_elevation_km,
        local_runoff_supply_km3_myr,
        base_material_present,
        whole_graph_candidate,
        central_window_candidate,
        compiled_stencils,
        cumulative_rock_displacement_km,
        frame_witnesses,
        summary,
        component_hashes,
        derived_resolution_hash: 0,
    };
    resolution.derived_resolution_hash = resolution_hash(schema, hash_version, &resolution)?;
    Ok(resolution)
}

#[allow(clippy::too_many_arguments)]
fn resolution_summary(
    mesh: &LandscapeMesh,
    initial: &[f64],
    runoff: &[f64],
    material: &[bool],
    whole: &[bool],
    central: &[bool],
    stencils: &[SupportStencil],
    cumulative: &[f64],
) -> Result<LinkedResolutionSummaryV0, LinkedInputErrorV0> {
    let mut bounds = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    for center in &mesh.cell_center_km {
        bounds[0] = bounds[0].min(center.x);
        bounds[1] = bounds[1].max(center.x);
        bounds[2] = bounds[2].min(center.y);
        bounds[3] = bounds[3].max(center.y);
    }
    let mut portal_summaries = Vec::with_capacity(mesh.outlet_portals.len());
    for portal in &mesh.outlet_portals {
        let faces = mesh.boundary_faces.iter().filter(|face| {
            matches!(face.condition, BoundaryFaceCondition::OpenBaseLevel { portal_id, .. } if portal_id == portal.id)
        });
        let records = faces.clone().collect::<Vec<_>>();
        let owners = records
            .iter()
            .map(|face| face.cell)
            .collect::<BTreeSet<_>>();
        portal_summaries.push(LinkedPortalSummaryV0 {
            portal_id: portal.id,
            face_record_count: records.len() as u64,
            owner_cell_count: owners.len() as u64,
            projected_coverage_km: records.iter().map(|face| face.projected_width_km()).sum(),
            physical_open_arc_km: records.iter().map(|face| face.width_km).sum(),
        });
    }
    let stencil_summaries = stencils
        .iter()
        .map(|stencil| {
            let positive = stencil
                .weight_per_km2
                .iter()
                .copied()
                .filter(|weight| *weight > 0.0)
                .collect::<Vec<_>>();
            if positive.is_empty() {
                return Err(LinkedInputErrorV0(format!(
                    "empty stencil {:?}",
                    stencil.segment_id
                )));
            }
            let minimum = stencil
                .weight_per_km2
                .iter()
                .copied()
                .reduce(f64::min)
                .ok_or_else(|| {
                    LinkedInputErrorV0(format!("empty stencil {:?}", stencil.segment_id))
                })?;
            Ok(LinkedStencilSummaryV0 {
                segment_id: stencil.segment_id,
                support_cell_count: positive.len() as u64,
                minimum_weight_per_km2: minimum,
                maximum_weight_per_km2: positive.into_iter().reduce(f64::max).unwrap(),
                area_integral: stencil
                    .weight_per_km2
                    .iter()
                    .zip(&mesh.cell_area_km2)
                    .map(|(weight, area)| weight * area)
                    .sum(),
            })
        })
        .collect::<Result<Vec<_>, LinkedInputErrorV0>>()?;
    let raw_exposed_face_count = (0..mesh.cell_count())
        .map(|i| 6 - (mesh.edge_offsets[i + 1] - mesh.edge_offsets[i]) as u64)
        .sum();
    Ok(LinkedResolutionSummaryV0 {
        cell_count: mesh.cell_count() as u64,
        directed_edge_count: mesh.edge_neighbor.len() as u64,
        raw_exposed_face_count,
        split_boundary_record_count: mesh.boundary_faces.len() as u64,
        actual_domain_area_km2: mesh.cell_area_km2.iter().sum(),
        physical_boundary_arc_km: mesh.boundary_faces.iter().map(|face| face.width_km).sum(),
        center_bounds_km: bounds,
        portal_summaries,
        initial_min_km: initial.iter().copied().reduce(f64::min).unwrap(),
        initial_max_km: initial.iter().copied().reduce(f64::max).unwrap(),
        local_runoff_total_km3_myr: runoff.iter().sum(),
        base_material_present_count: material.iter().filter(|value| **value).count() as u64,
        whole_graph_count: whole.iter().filter(|value| **value).count() as u64,
        central_window_count: central.iter().filter(|value| **value).count() as u64,
        central_window_area_km2: central
            .iter()
            .zip(&mesh.cell_area_km2)
            .filter_map(|(included, area)| included.then_some(*area))
            .sum(),
        cumulative_rock_volume_km3: cumulative
            .iter()
            .zip(&mesh.cell_area_km2)
            .map(|(height, area)| height * area)
            .sum(),
        stencil_summaries,
    })
}

fn validate_numerical_gates(
    spacing: f64,
    mesh: &LandscapeMesh,
    frames: &[ForcingFrameWitnessV0],
    summary: &LinkedResolutionSummaryV0,
) -> Result<(), LinkedInputErrorV0> {
    let expected = match spacing as u32 {
        8 => (11_040, 65_394, 846, 848, 3_680, 480, 120),
        4 => (44_400, 264_702, 1_698, 1_700, 14_880, 960, 240),
        2 => (177_600, 1_062_202, 3_398, 3_400, 58_880, 1_920, 480),
        _ => return Err(LinkedInputErrorV0("unregistered spacing".into())),
    };
    let observed = (
        summary.cell_count,
        summary.directed_edge_count,
        summary.raw_exposed_face_count,
        summary.split_boundary_record_count,
        summary.central_window_count,
        summary
            .portal_summaries
            .iter()
            .map(|portal| portal.face_record_count)
            .sum(),
        summary.portal_summaries[0].owner_cell_count,
    );
    if observed != expected {
        return Err(LinkedInputErrorV0(format!(
            "integer topology mismatch at {spacing} km: {observed:?} != {expected:?}"
        )));
    }
    if summary.portal_summaries.iter().any(|portal| {
        portal.owner_cell_count != expected.6
            || !close(
                portal.projected_coverage_km,
                match spacing as u32 {
                    8 => 958.0,
                    4 => 959.0,
                    2 => 959.5,
                    _ => unreachable!(),
                },
                1e-10,
                0.0,
            )
    }) {
        return Err(LinkedInputErrorV0("portal coverage mismatch".into()));
    }
    if summary.stencil_summaries.iter().any(|stencil| {
        stencil.support_cell_count == 0
            || !close(stencil.area_integral, 1.0, 1e-12, 1e-12)
            || !stencil.minimum_weight_per_km2.is_finite()
            || stencil.minimum_weight_per_km2 < 0.0
    }) {
        return Err(LinkedInputErrorV0("stencil normalization mismatch".into()));
    }
    if !close(
        summary.cumulative_rock_volume_km3,
        TOTAL_WORK_KM3,
        1e-8,
        5e-12,
    ) {
        return Err(LinkedInputErrorV0(
            "cumulative work ledger does not close".into(),
        ));
    }
    if !close(
        summary.local_runoff_total_km3_myr,
        RUNOFF_DEPTH_RATE_KM_MYR * summary.actual_domain_area_km2,
        1e-6,
        5e-12,
    ) {
        return Err(LinkedInputErrorV0("runoff ledger does not close".into()));
    }
    if summary.initial_min_km < 0.0 || !summary.initial_max_km.is_finite() {
        return Err(LinkedInputErrorV0("invalid initial elevation range".into()));
    }
    for ((frame, &time), &activity) in frames.iter().zip(&ORACLE_TIMES_MYR).zip(&ORACLE_ACTIVITIES)
    {
        if frame.time_myr != time || frame.expected_activity != activity {
            return Err(LinkedInputErrorV0("forcing witness order mismatch".into()));
        }
        let expected_rate = activity * REFERENCE_ROCK_VOLUME_RATE_KM3_MYR;
        if activity == 0.0 {
            if frame.integrated_rate_km3_myr.to_bits() != 0 {
                return Err(LinkedInputErrorV0("inactive forcing is nonzero".into()));
            }
        } else if !close(frame.integrated_rate_km3_myr, expected_rate, 0.0, 2e-7) {
            return Err(LinkedInputErrorV0(
                "evaluated f32 rate does not match activity oracle".into(),
            ));
        }
    }
    if mesh.cell_area_km2.iter().any(|value| !nonnegative(*value)) {
        return Err(LinkedInputErrorV0("invalid cell area".into()));
    }
    Ok(())
}

fn validate_mesh_geometry(mesh: &LandscapeMesh, spacing: f64) -> Result<(), LinkedInputErrorV0> {
    let unsplit = LandscapeMesh::uniform_planar_hex_with_portals(WIDTH_KM, HEIGHT_KM, spacing, &[])
        .map_err(|error| LinkedInputErrorV0(error.to_string()))?;
    let analytic_cell_area = 0.5 * 3.0_f64.sqrt() * spacing * spacing;
    for cell in 0..mesh.cell_count() {
        let mut gauss = glam::DVec3::ZERO;
        let mut area_moment = 0.0;
        for edge in mesh.edge_offsets[cell] as usize..mesh.edge_offsets[cell + 1] as usize {
            let neighbor = mesh.edge_neighbor[edge] as usize;
            let reverse = (mesh.edge_offsets[neighbor] as usize
                ..mesh.edge_offsets[neighbor + 1] as usize)
                .find(|&candidate| mesh.edge_neighbor[candidate] as usize == cell)
                .ok_or_else(|| LinkedInputErrorV0("nonreciprocal mesh edge".into()))?;
            if mesh.edge_distance_km[edge].to_bits() != mesh.edge_distance_km[reverse].to_bits()
                || mesh.edge_face_width_km[edge].to_bits()
                    != mesh.edge_face_width_km[reverse].to_bits()
                || mesh.edge_outward_tangent[edge] != -mesh.edge_outward_tangent[reverse]
            {
                return Err(LinkedInputErrorV0(
                    "reciprocal mesh face values disagree".into(),
                ));
            }
            let normal = mesh.edge_outward_tangent[edge].as_dvec3();
            let width = f64::from(mesh.edge_face_width_km[edge]);
            gauss += width * normal;
            area_moment += width * 0.5 * f64::from(mesh.edge_distance_km[edge]);
        }
        for face in mesh
            .boundary_faces
            .iter()
            .filter(|face| face.cell as usize == cell)
        {
            let projected_distance = (face.center_km - mesh.cell_center_km[cell])
                .dot(face.outward_normal)
                .abs();
            if (face.center_distance_km - projected_distance).abs() >= 1e-6
                || (face.outward_normal.length() - 1.0).abs() >= 1e-12
                || !nonnegative(face.width_km)
                || face.width_km == 0.0
            {
                return Err(LinkedInputErrorV0(
                    "invalid independent boundary-face geometry".into(),
                ));
            }
            gauss += face.width_km * face.outward_normal;
            area_moment += face.width_km
                * (face.center_km - mesh.cell_center_km[cell]).dot(face.outward_normal);
        }
        if gauss.length() >= 2e-6 {
            return Err(LinkedInputErrorV0(format!(
                "cell {cell} vector Gauss residual exceeds gate"
            )));
        }
        if (0.5 * area_moment - analytic_cell_area).abs() >= 2e-6 {
            return Err(LinkedInputErrorV0(format!(
                "cell {cell} area-moment residual exceeds gate"
            )));
        }
    }

    let actual_area: f64 = mesh.cell_area_km2.iter().sum();
    let boundary_area_moment: f64 = mesh
        .boundary_faces
        .iter()
        .map(|face| face.width_km * face.center_km.dot(face.outward_normal))
        .sum();
    if (0.5 * boundary_area_moment - actual_area).abs() >= 1e-5
        || (actual_area - mesh.cell_count() as f64 * analytic_cell_area).abs() >= 1e-5
    {
        return Err(LinkedInputErrorV0(
            "global mesh area closure exceeds gate".into(),
        ));
    }
    let raw_physical_arc: f64 = unsplit
        .boundary_faces
        .iter()
        .map(|face| face.width_km)
        .sum();
    let split_physical_arc: f64 = mesh.boundary_faces.iter().map(|face| face.width_km).sum();
    if (raw_physical_arc - split_physical_arc).abs() >= 1e-8 {
        return Err(LinkedInputErrorV0(
            "portal splitting changed physical boundary arc".into(),
        ));
    }

    for face in &mesh.boundary_faces {
        let matching = mesh.outlet_portals.iter().find(|portal| {
            portal.side == face.side
                && face.projected_span_start_km < portal.span_end_km
                && portal.span_start_km < face.projected_span_end_km
        });
        match (matching, face.condition) {
            (
                Some(portal),
                BoundaryFaceCondition::OpenBaseLevel {
                    portal_id,
                    elevation_km,
                },
            ) if portal.id == portal_id
                && portal.base_level_km.to_bits() == elevation_km.to_bits() =>
            {
                if face.projected_span_start_km < portal.span_start_km - 1e-12
                    || face.projected_span_end_km > portal.span_end_km + 1e-12
                {
                    return Err(LinkedInputErrorV0(
                        "open face extends beyond portal intersection".into(),
                    ));
                }
            }
            (None, BoundaryFaceCondition::Closed) => {}
            _ => {
                return Err(LinkedInputErrorV0(
                    "boundary-face portal intersection semantics mismatch".into(),
                ))
            }
        }
    }
    for portal in &mesh.outlet_portals {
        let independent_projected_intersection: f64 = unsplit
            .boundary_faces
            .iter()
            .filter(|face| face.side == portal.side)
            .map(|face| {
                (face.projected_span_end_km.min(portal.span_end_km)
                    - face.projected_span_start_km.max(portal.span_start_km))
                .max(0.0)
            })
            .sum();
        if (independent_projected_intersection - mesh.portal_projected_coverage_km(portal.id)).abs()
            >= 1e-10
        {
            return Err(LinkedInputErrorV0(
                "portal projected intersection closure exceeds gate".into(),
            ));
        }
    }
    for side in [BoundarySide::South, BoundarySide::North] {
        let closed = mesh
            .boundary_faces
            .iter()
            .filter(|face| face.side == side && face.condition == BoundaryFaceCondition::Closed)
            .count();
        if closed != 1 {
            return Err(LinkedInputErrorV0(format!(
                "expected one closed outboard portal sliver on {side:?}, got {closed}"
            )));
        }
    }
    if mesh.boundary_faces.iter().any(|face| {
        matches!(face.side, BoundarySide::East | BoundarySide::West)
            && face.condition != BoundaryFaceCondition::Closed
    }) {
        return Err(LinkedInputErrorV0(
            "east/west boundary is not closed".into(),
        ));
    }
    Ok(())
}

fn validate_scenario(scenario: &LandscapeScenario) -> Result<(), LinkedInputErrorV0> {
    if scenario.id != "L" {
        return Err(LinkedInputErrorV0("wrong linked scenario ID".into()));
    }
    let mut segment_ids = BTreeSet::new();
    for segment in &scenario.segments {
        if !segment_ids.insert(segment.id.0)
            || !segment.width_km.is_finite()
            || segment.width_km <= 0.0
            || !segment.geometry.start_km.is_finite()
            || !segment.geometry.end_km.is_finite()
            || !segment.vergence.is_finite()
        {
            return Err(LinkedInputErrorV0("invalid or duplicate segment".into()));
        }
    }
    for segment in &scenario.segments {
        if segment
            .links
            .iter()
            .any(|link| !segment_ids.contains(&link.other.0))
        {
            return Err(LinkedInputErrorV0("unresolved segment link".into()));
        }
    }
    let mut episode_ids = BTreeSet::new();
    for episode in &scenario.episodes {
        if !episode_ids.insert(episode.id.0)
            || !episode.active_myr.start.is_finite()
            || !episode.active_myr.end.is_finite()
            || episode.active_myr.start >= episode.active_myr.end
            || !nonnegative(episode.ramp_myr)
            || !nonnegative(episode.rock_volume_rate_km3_myr)
            || episode.segment_shares.iter().any(|share| {
                !segment_ids.contains(&share.segment_id.0) || !nonnegative(share.share)
            })
            || !close(
                episode.segment_shares.iter().map(|share| share.share).sum(),
                1.0,
                1e-12,
                1e-12,
            )
        {
            return Err(LinkedInputErrorV0("invalid or duplicate episode".into()));
        }
    }
    Ok(())
}

fn episode_activity(episode: &DeformationEpisode, time: f64) -> f64 {
    if time < episode.active_myr.start || time > episode.active_myr.end {
        return 0.0;
    }
    let ramp = episode
        .ramp_myr
        .max(0.0)
        .min(0.5 * (episode.active_myr.end - episode.active_myr.start));
    if ramp == 0.0 {
        return 1.0;
    }
    let edge = ((time - episode.active_myr.start) / ramp)
        .min((episode.active_myr.end - time) / ramp)
        .clamp(0.0, 1.0);
    edge * edge * (3.0 - 2.0 * edge)
}

pub fn validate_linked_shared_input_bundle_v0(
    bundle: &LinkedSharedInputBundleV0,
) -> Result<(), LinkedInputErrorV0> {
    let expected = assemble_bundle()?;
    validate_against_expected(bundle, &expected)
}

fn validate_against_expected(
    bundle: &LinkedSharedInputBundleV0,
    expected: &LinkedSharedInputBundleV0,
) -> Result<(), LinkedInputErrorV0> {
    if expected.derived_bundle_hash != LINKED_INPUT_BUNDLE_HASH_V0 {
        return Err(LinkedInputErrorV0(format!(
            "linked-input V0 implementation drifted from frozen bundle hash: {:016x} != {:016x}",
            expected.derived_bundle_hash, LINKED_INPUT_BUNDLE_HASH_V0
        )));
    }
    if bundle.schema_version != LINKED_INPUT_SCHEMA_VERSION_V0
        || bundle.hash_version != LINKED_INPUT_HASH_VERSION_V0
    {
        return Err(LinkedInputErrorV0(
            "wrong linked-input schema or hash version".into(),
        ));
    }
    if fixed_bytes(&bundle.declaration)? != fixed_bytes(&canonical_declaration())? {
        return Err(LinkedInputErrorV0(
            "noncanonical linked-input declaration".into(),
        ));
    }
    validate_scenario(&bundle.declaration.scenario)?;
    if bundle.declaration.scenario.episodes.iter().any(|episode| {
        ORACLE_TIMES_MYR
            .iter()
            .zip(&ORACLE_ACTIVITIES)
            .any(|(&time, &expected)| episode_activity(episode, time) != expected)
    }) {
        return Err(LinkedInputErrorV0("activity oracle mismatch".into()));
    }
    if bundle.derived_bundle_hash != LINKED_INPUT_BUNDLE_HASH_V0
        || bundle.derived_bundle_hash != bundle_hash(bundle)?
    {
        return Err(LinkedInputErrorV0("bundle hash mismatch".into()));
    }
    // Rust float equality aliases +/-0.0; canonical replay is a byte identity
    // and therefore also enforces every registered sign-bit rule.
    if fixed_bytes(bundle)? != fixed_bytes(expected)? {
        return Err(LinkedInputErrorV0(
            "bundle does not exactly replay the frozen linked inputs".into(),
        ));
    }
    Ok(())
}

pub fn encode_linked_shared_input_bundle_v0(
    bundle: &LinkedSharedInputBundleV0,
) -> Result<Vec<u8>, LinkedInputErrorV0> {
    validate_linked_shared_input_bundle_v0(bundle)?;
    fixed_bytes(bundle)
}

pub fn decode_linked_shared_input_bundle_v0(
    bytes: &[u8],
) -> Result<LinkedSharedInputBundleV0, LinkedInputErrorV0> {
    let bundle = decode_linked_shared_input_bundle_stored_v0(bytes)?;
    validate_linked_shared_input_bundle_v0(&bundle)?;
    Ok(bundle)
}

/// Decode the stored semantic identity without requiring platform-local
/// transcendental/compiler replay. Consumers should compare the published
/// bundle hash before trusting an externally supplied artifact; the exact
/// replay decoder above remains the generation/audit trust boundary.
pub fn decode_linked_shared_input_bundle_stored_v0(
    bytes: &[u8],
) -> Result<LinkedSharedInputBundleV0, LinkedInputErrorV0> {
    if bytes.len() as u64 > MAX_ENCODED_BUNDLE_BYTES {
        return Err(LinkedInputErrorV0(format!(
            "linked-input artifact exceeds {} byte decoder limit",
            MAX_ENCODED_BUNDLE_BYTES
        )));
    }
    let bundle: LinkedSharedInputBundleV0 = bincode_options()
        .with_limit(MAX_ENCODED_BUNDLE_BYTES)
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|error| LinkedInputErrorV0(format!("linked-input decode failed: {error}")))?;
    validate_stored_bundle_identity(&bundle)?;
    Ok(bundle)
}

fn validate_stored_bundle_identity(
    bundle: &LinkedSharedInputBundleV0,
) -> Result<(), LinkedInputErrorV0> {
    if bundle.schema_version != LINKED_INPUT_SCHEMA_VERSION_V0
        || bundle.hash_version != LINKED_INPUT_HASH_VERSION_V0
        || fixed_bytes(&bundle.declaration)? != fixed_bytes(&canonical_declaration())?
        || bundle.resolutions.len() != SPACINGS_KM.len()
    {
        return Err(LinkedInputErrorV0(
            "stored linked-input declaration or shape is noncanonical".into(),
        ));
    }
    validate_scenario(&bundle.declaration.scenario)?;
    for (resolution, &spacing) in bundle.resolutions.iter().zip(&SPACINGS_KM) {
        if resolution.nominal_spacing_km.to_bits() != spacing.to_bits() {
            return Err(LinkedInputErrorV0(
                "stored linked-input resolution order is noncanonical".into(),
            ));
        }
        resolution
            .mesh
            .validate()
            .map_err(|error| LinkedInputErrorV0(error.to_string()))?;
        let n = resolution.mesh.cell_count();
        if resolution.initial_elevation_km.len() != n
            || resolution.local_runoff_supply_km3_myr.len() != n
            || resolution.base_material_present.len() != n
            || resolution.whole_graph_candidate.len() != n
            || resolution.central_window_candidate.len() != n
            || resolution.cumulative_rock_displacement_km.len() != n
            || resolution.compiled_stencils.len() != 2
            || resolution
                .compiled_stencils
                .iter()
                .any(|stencil| stencil.weight_per_km2.len() != n)
            || resolution.frame_witnesses.len() != ORACLE_TIMES_MYR.len()
        {
            return Err(LinkedInputErrorV0(
                "stored linked-input array length is invalid".into(),
            ));
        }
        let actual_components = recompute_component_hashes_v0(
            &bundle.schema_version,
            &bundle.hash_version,
            resolution,
        )?;
        if fixed_bytes(&actual_components)? != fixed_bytes(&resolution.component_hashes)?
            || resolution.derived_resolution_hash
                != resolution_hash(&bundle.schema_version, &bundle.hash_version, resolution)?
        {
            return Err(LinkedInputErrorV0(
                "stored linked-input component or resolution hash mismatch".into(),
            ));
        }
    }
    if bundle.derived_bundle_hash != LINKED_INPUT_BUNDLE_HASH_V0
        || bundle.derived_bundle_hash != bundle_hash(bundle)?
    {
        return Err(LinkedInputErrorV0("stored bundle hash mismatch".into()));
    }
    Ok(())
}

impl LinkedSharedInputManifestJsonV0 {
    pub fn from_bundle(bundle: &LinkedSharedInputBundleV0) -> Self {
        Self {
            schema_version: MANIFEST_SCHEMA_VERSION_V0.into(),
            semantic_schema_version: bundle.schema_version.clone(),
            hash_version: bundle.hash_version.clone(),
            derived_bundle_hash_hex: hex_hash(bundle.derived_bundle_hash),
            declaration: bundle.declaration.clone(),
            resolutions: bundle
                .resolutions
                .iter()
                .map(|resolution| LinkedResolutionManifestJsonV0 {
                    nominal_spacing_km: resolution.nominal_spacing_km,
                    summary: resolution.summary.clone(),
                    component_hashes_hex: component_hash_values(&resolution.component_hashes)
                        .into_iter()
                        .map(hex_hash)
                        .collect(),
                    derived_resolution_hash_hex: hex_hash(resolution.derived_resolution_hash),
                })
                .collect(),
        }
    }

    pub fn validate_against(
        &self,
        bundle: &LinkedSharedInputBundleV0,
    ) -> Result<(), LinkedInputErrorV0> {
        let expected = Self::from_bundle(bundle);
        if self != &expected {
            return Err(LinkedInputErrorV0(
                "manifest JSON does not project the semantic bundle".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct RunEnvelopeV0 {
    schema_version: String,
    source_revision: Option<String>,
    source_dirty: Option<bool>,
    source_manifest_dir: String,
    invocation_cwd: Option<String>,
    executable: String,
    executable_length_bytes: Option<u64>,
    executable_hash_fnv1a64: Option<String>,
    rust_toolchain: Option<String>,
    os: String,
    kernel: Option<String>,
    wsl_distribution: Option<String>,
    cpu: Option<String>,
    materializer_thread_count: usize,
    available_parallelism: usize,
    command: Vec<String>,
    elapsed_to_prepublication_validation_seconds: f64,
    proc_self_vmhwm_kib_before_publication: Option<u64>,
    final_measurement_authority: String,
    artifacts: Vec<EnvelopeArtifactV0>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct EnvelopeArtifactV0 {
    name: String,
    length_bytes: u64,
    hash_fnv1a64: String,
}

/// Build, validate and atomically publish the registered three-file directory.
pub fn materialize_linked_shared_input_v0(output_dir: &Path) -> Result<(), LinkedInputErrorV0> {
    let started = std::time::Instant::now();
    // Provenance is captured before even the temporary publication tree exists,
    // so the materializer cannot mark its own source tree dirty.
    let source_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let revision = command_output_in(source_root, "git", &["rev-parse", "HEAD"]);
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(source_root)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| !output.stdout.is_empty());
    ensure_path_missing(output_dir, "output target")?;
    let parent = output_dir
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(LinkedInputErrorV0(format!(
            "output parent is not a directory: {}",
            parent.display()
        )));
    }
    let file_name = output_dir
        .file_name()
        .ok_or_else(|| LinkedInputErrorV0("output directory needs a final component".into()))?;
    let temp_dir = parent.join(format!(
        ".{}.tmp-{}",
        file_name.to_string_lossy(),
        std::process::id()
    ));
    let lock_path = parent.join(format!(".{}.publish-lock", file_name.to_string_lossy()));
    ensure_path_missing(&temp_dir, "temporary target")?;
    let _publication_lock = PublicationLock::acquire(&lock_path)?;
    ensure_path_missing(output_dir, "output target")?;
    fs::create_dir(&temp_dir).map_err(io_error)?;
    let result = (|| {
        let bundle = build_linked_shared_input_bundle_v0()?;
        let binary = encode_linked_shared_input_bundle_v0(&bundle)?;
        let decoded = decode_linked_shared_input_bundle_v0(&binary)?;
        if decoded != bundle {
            return Err(LinkedInputErrorV0(
                "binary round trip changed bundle".into(),
            ));
        }
        let manifest = LinkedSharedInputManifestJsonV0::from_bundle(&bundle);
        manifest.validate_against(&decoded)?;
        let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|error| LinkedInputErrorV0(error.to_string()))?;
        manifest_bytes.push(b'\n');
        write_synced_new(&temp_dir.join("shared-input.bin"), &binary)?;
        write_synced_new(&temp_dir.join("manifest.json"), &manifest_bytes)?;

        // Reread and validate the semantic artifacts before taking the
        // prepublication time/RSS witnesses recorded in the envelope.
        let disk_binary = fs::read(temp_dir.join("shared-input.bin")).map_err(io_error)?;
        if disk_binary != binary {
            return Err(LinkedInputErrorV0(
                "on-disk binary bytes differ from canonical bytes".into(),
            ));
        }
        let disk_bundle = decode_linked_shared_input_bundle_v0(&disk_binary)?;
        let disk_manifest_bytes = fs::read(temp_dir.join("manifest.json")).map_err(io_error)?;
        if disk_manifest_bytes != manifest_bytes {
            return Err(LinkedInputErrorV0(
                "on-disk manifest bytes differ from canonical bytes".into(),
            ));
        }
        let disk_manifest: LinkedSharedInputManifestJsonV0 =
            serde_json::from_slice(&disk_manifest_bytes)
                .map_err(|error| LinkedInputErrorV0(error.to_string()))?;
        disk_manifest.validate_against(&disk_bundle)?;

        let executable = std::env::current_exe().map_err(io_error)?;
        let executable_bytes = fs::read(&executable).ok();
        let elapsed_to_prepublication_validation = started.elapsed().as_secs_f64();
        let prepublication_peak_rss_kib = peak_rss_kib();
        let envelope = RunEnvelopeV0 {
            schema_version: "orogen-linked-shared-input-run-envelope-v0".into(),
            source_revision: revision,
            source_dirty: dirty,
            source_manifest_dir: source_root.display().to_string(),
            invocation_cwd: std::env::current_dir()
                .ok()
                .map(|path| path.display().to_string()),
            executable: executable.display().to_string(),
            executable_length_bytes: executable_bytes.as_ref().map(|bytes| bytes.len() as u64),
            executable_hash_fnv1a64: executable_bytes
                .as_deref()
                .map(fnv1a64)
                .map(hex_hash),
            rust_toolchain: command_output("rustc", &["-Vv"]),
            os: std::env::consts::OS.into(),
            kernel: command_output("uname", &["-a"]),
            wsl_distribution: std::env::var("WSL_DISTRO_NAME").ok(),
            cpu: cpu_model(),
            materializer_thread_count: 1,
            available_parallelism: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            command: std::env::args_os()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect(),
            elapsed_to_prepublication_validation_seconds: elapsed_to_prepublication_validation,
            proc_self_vmhwm_kib_before_publication: prepublication_peak_rss_kib,
            final_measurement_authority: "/usr/bin/time -v; record final wall time and whole-process peak RSS in the dated audit".into(),
            artifacts: vec![
                EnvelopeArtifactV0 {
                    name: "shared-input.bin".into(),
                    length_bytes: binary.len() as u64,
                    hash_fnv1a64: hex_hash(fnv1a64(&binary)),
                },
                EnvelopeArtifactV0 {
                    name: "manifest.json".into(),
                    length_bytes: manifest_bytes.len() as u64,
                    hash_fnv1a64: hex_hash(fnv1a64(&manifest_bytes)),
                },
            ],
        };
        let mut envelope_bytes = serde_json::to_vec_pretty(&envelope)
            .map_err(|error| LinkedInputErrorV0(error.to_string()))?;
        envelope_bytes.push(b'\n');
        let envelope_path = temp_dir.join("run-envelope.json");
        write_synced_new(&envelope_path, &envelope_bytes)?;
        let disk_envelope_bytes = fs::read(&envelope_path).map_err(io_error)?;
        if disk_envelope_bytes != envelope_bytes {
            return Err(LinkedInputErrorV0(
                "on-disk envelope bytes differ from canonical bytes".into(),
            ));
        }
        let disk_envelope: RunEnvelopeV0 = serde_json::from_slice(&disk_envelope_bytes)
            .map_err(|error| LinkedInputErrorV0(error.to_string()))?;
        if disk_envelope != envelope
            || disk_envelope.artifacts
                != vec![
                    EnvelopeArtifactV0 {
                        name: "shared-input.bin".into(),
                        length_bytes: disk_binary.len() as u64,
                        hash_fnv1a64: hex_hash(fnv1a64(&disk_binary)),
                    },
                    EnvelopeArtifactV0 {
                        name: "manifest.json".into(),
                        length_bytes: disk_manifest_bytes.len() as u64,
                        hash_fnv1a64: hex_hash(fnv1a64(&disk_manifest_bytes)),
                    },
                ]
        {
            return Err(LinkedInputErrorV0(
                "run envelope does not bind the on-disk semantic files".into(),
            ));
        }
        sync_directory(&temp_dir)?;
        ensure_path_missing(output_dir, "output target")?;
        fs::rename(&temp_dir, output_dir).map_err(io_error)?;
        sync_directory(parent)?;
        Ok(())
    })();
    if result.is_err() && temp_dir.exists() {
        let _ = fs::remove_dir_all(&temp_dir);
    }
    result
}

struct PublicationLock {
    path: std::path::PathBuf,
    file: Option<File>,
}

impl PublicationLock {
    fn acquire(path: &Path) -> Result<Self, LinkedInputErrorV0> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| {
                LinkedInputErrorV0(format!(
                    "cannot reserve publication lock {}: {error}",
                    path.display()
                ))
            })?;
        let mut lock = Self {
            path: path.to_owned(),
            file: Some(file),
        };
        let file = lock.file.as_mut().expect("publication lock file");
        writeln!(file, "{}", std::process::id()).map_err(io_error)?;
        file.sync_all().map_err(io_error)?;
        Ok(lock)
    }
}

impl Drop for PublicationLock {
    fn drop(&mut self) {
        self.file.take();
        let _ = fs::remove_file(&self.path);
    }
}

fn ensure_path_missing(path: &Path, role: &str) -> Result<(), LinkedInputErrorV0> {
    match fs::symlink_metadata(path) {
        Ok(_) => Err(LinkedInputErrorV0(format!(
            "{role} already exists: {}",
            path.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(io_error(error)),
    }
}

fn write_synced_new(path: &Path, bytes: &[u8]) -> Result<(), LinkedInputErrorV0> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(io_error)?;
    file.write_all(bytes).map_err(io_error)?;
    file.sync_all().map_err(io_error)
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<(), LinkedInputErrorV0> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(io_error)
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<(), LinkedInputErrorV0> {
    Ok(())
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_owned())
}

fn command_output_in(directory: &Path, program: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(program)
        .args(args)
        .current_dir(directory)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_owned())
}

fn cpu_model() -> Option<String> {
    fs::read_to_string("/proc/cpuinfo").ok().and_then(|text| {
        text.lines()
            .find_map(|line| line.strip_prefix("model name\t: "))
            .map(str::to_owned)
    })
}

fn peak_rss_kib() -> Option<u64> {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                line.strip_prefix("VmHWM:")?
                    .split_whitespace()
                    .next()?
                    .parse()
                    .ok()
            })
        })
}

fn io_error(error: std::io::Error) -> LinkedInputErrorV0 {
    LinkedInputErrorV0(error.to_string())
}

fn close(actual: f64, expected: f64, abs_tol: f64, rel_tol: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs() <= abs_tol + rel_tol * actual.abs().max(expected.abs())
}

fn nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0 && !(value == 0.0 && value.is_sign_negative())
}

#[derive(Serialize)]
struct ComponentPreimage<'a, T> {
    domain: &'a str,
    schema_version: &'a str,
    hash_version: &'a str,
    nominal_spacing_km: f64,
    payload: &'a T,
}

fn component_hash<T: Serialize>(
    domain: &str,
    schema: &str,
    hash_version: &str,
    spacing: f64,
    payload: &T,
) -> Result<u64, LinkedInputErrorV0> {
    Ok(fnv1a64(&fixed_bytes(&ComponentPreimage {
        domain,
        schema_version: schema,
        hash_version,
        nominal_spacing_km: spacing,
        payload,
    })?))
}

#[derive(Serialize)]
struct FrameArrayPreimage<'a, T> {
    domain: &'a str,
    schema_version: &'a str,
    hash_version: &'a str,
    nominal_spacing_km: f64,
    time_myr: f64,
    payload: &'a T,
}

fn frame_array_hash<T: Serialize>(
    domain: &str,
    schema: &str,
    hash_version: &str,
    spacing: f64,
    time: f64,
    payload: &T,
) -> Result<u64, LinkedInputErrorV0> {
    Ok(fnv1a64(&fixed_bytes(&FrameArrayPreimage {
        domain,
        schema_version: schema,
        hash_version,
        nominal_spacing_km: spacing,
        time_myr: time,
        payload,
    })?))
}

#[derive(Serialize)]
struct ResolutionPreimage<'a> {
    domain: &'static str,
    schema_version: &'a str,
    hash_version: &'a str,
    nominal_spacing_km: f64,
    mesh: &'a LandscapeMesh,
    initial_elevation_km: &'a Vec<f64>,
    local_runoff_supply_km3_myr: &'a Vec<f64>,
    base_material_present: &'a Vec<bool>,
    whole_graph_candidate: &'a Vec<bool>,
    central_window_candidate: &'a Vec<bool>,
    compiled_stencils: &'a Vec<SupportStencil>,
    cumulative_rock_displacement_km: &'a Vec<f64>,
    frame_witnesses: &'a Vec<ForcingFrameWitnessV0>,
    summary: &'a LinkedResolutionSummaryV0,
    component_hashes: &'a LinkedInputComponentHashesV0,
}

fn resolution_hash(
    schema: &str,
    hash_version: &str,
    value: &LinkedResolutionInputV0,
) -> Result<u64, LinkedInputErrorV0> {
    Ok(fnv1a64(&fixed_bytes(&ResolutionPreimage {
        domain: "orogen-linked-input-v0/resolution",
        schema_version: schema,
        hash_version,
        nominal_spacing_km: value.nominal_spacing_km,
        mesh: &value.mesh,
        initial_elevation_km: &value.initial_elevation_km,
        local_runoff_supply_km3_myr: &value.local_runoff_supply_km3_myr,
        base_material_present: &value.base_material_present,
        whole_graph_candidate: &value.whole_graph_candidate,
        central_window_candidate: &value.central_window_candidate,
        compiled_stencils: &value.compiled_stencils,
        cumulative_rock_displacement_km: &value.cumulative_rock_displacement_km,
        frame_witnesses: &value.frame_witnesses,
        summary: &value.summary,
        component_hashes: &value.component_hashes,
    })?))
}

#[derive(Serialize)]
struct BundlePreimage<'a> {
    domain: &'static str,
    schema_version: &'a str,
    hash_version: &'a str,
    declaration: &'a LinkedInputDeclarationV0,
    resolutions: &'a Vec<LinkedResolutionInputV0>,
}

fn bundle_hash(value: &LinkedSharedInputBundleV0) -> Result<u64, LinkedInputErrorV0> {
    Ok(fnv1a64(&fixed_bytes(&BundlePreimage {
        domain: "orogen-linked-input-v0/bundle",
        schema_version: &value.schema_version,
        hash_version: &value.hash_version,
        declaration: &value.declaration,
        resolutions: &value.resolutions,
    })?))
}

fn component_hash_values(value: &LinkedInputComponentHashesV0) -> [u64; 9] {
    [
        value.mesh_hash,
        value.initial_elevation_hash,
        value.local_runoff_hash,
        value.base_material_present_hash,
        value.whole_graph_candidate_hash,
        value.central_window_candidate_hash,
        value.compiled_stencils_hash,
        value.cumulative_rock_displacement_hash,
        value.frame_witnesses_hash,
    ]
}

fn recompute_component_hashes_v0(
    schema: &str,
    hash_version: &str,
    value: &LinkedResolutionInputV0,
) -> Result<LinkedInputComponentHashesV0, LinkedInputErrorV0> {
    let spacing = value.nominal_spacing_km;
    Ok(LinkedInputComponentHashesV0 {
        mesh_hash: component_hash(
            "orogen-linked-input-v0/mesh",
            schema,
            hash_version,
            spacing,
            &value.mesh,
        )?,
        initial_elevation_hash: component_hash(
            "orogen-linked-input-v0/initial-elevation",
            schema,
            hash_version,
            spacing,
            &value.initial_elevation_km,
        )?,
        local_runoff_hash: component_hash(
            "orogen-linked-input-v0/local-runoff",
            schema,
            hash_version,
            spacing,
            &value.local_runoff_supply_km3_myr,
        )?,
        base_material_present_hash: component_hash(
            "orogen-linked-input-v0/base-material-present",
            schema,
            hash_version,
            spacing,
            &value.base_material_present,
        )?,
        whole_graph_candidate_hash: component_hash(
            "orogen-linked-input-v0/whole-graph-candidate",
            schema,
            hash_version,
            spacing,
            &value.whole_graph_candidate,
        )?,
        central_window_candidate_hash: component_hash(
            "orogen-linked-input-v0/central-window-candidate",
            schema,
            hash_version,
            spacing,
            &value.central_window_candidate,
        )?,
        compiled_stencils_hash: component_hash(
            "orogen-linked-input-v0/compiled-stencils",
            schema,
            hash_version,
            spacing,
            &value.compiled_stencils,
        )?,
        cumulative_rock_displacement_hash: component_hash(
            "orogen-linked-input-v0/cumulative-rock-displacement",
            schema,
            hash_version,
            spacing,
            &value.cumulative_rock_displacement_km,
        )?,
        frame_witnesses_hash: component_hash(
            "orogen-linked-input-v0/frame-witnesses",
            schema,
            hash_version,
            spacing,
            &value.frame_witnesses,
        )?,
    })
}

fn fixed_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, LinkedInputErrorV0> {
    bincode_options()
        .serialize(value)
        .map_err(|error| LinkedInputErrorV0(error.to_string()))
}

fn bincode_options() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn hex_hash(hash: u64) -> String {
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_surface_is_deterministic_nonnegative_and_has_registered_phases() {
        let mesh = LandscapeMesh::uniform_planar_hex(48.0, 32.0, 4.0).unwrap();
        let first = linked_low_relief_initial_surface(&mesh, INITIAL_SEED);
        assert_eq!(
            first,
            linked_low_relief_initial_surface(&mesh, INITIAL_SEED)
        );
        assert!(first.iter().all(|value| nonnegative(*value)));
        assert!(mesh.boundary_faces.iter().any(|face| {
            matches!(face.condition, BoundaryFaceCondition::OpenBaseLevel { .. })
                && first[face.cell as usize] > 0.0
        }));
        let expected = [4.4037979935143605, 1.4440520298135509, 2.055503350315996];
        for (stream, expected) in [1, 2, 3].into_iter().zip(expected) {
            assert!((phase(INITIAL_SEED, stream) - expected).abs() < 1e-15);
        }
        assert!(decode_linked_shared_input_bundle_stored_v0(&u64::MAX.to_le_bytes()).is_err());
    }

    #[test]
    #[ignore = "full atomic publication boundary; run explicitly in release mode"]
    fn materializer_publishes_exact_files_and_preserves_existing_target() {
        let target = std::env::temp_dir().join(format!(
            "hex3-linked-input-publication-test-{}",
            std::process::id()
        ));
        let file_name = target.file_name().unwrap().to_string_lossy();
        let temp = target
            .parent()
            .unwrap()
            .join(format!(".{file_name}.tmp-{}", std::process::id()));
        let lock = target
            .parent()
            .unwrap()
            .join(format!(".{file_name}.publish-lock"));
        let _ = fs::remove_dir_all(&target);
        let _ = fs::remove_dir_all(&temp);
        let _ = fs::remove_file(&lock);

        materialize_linked_shared_input_v0(&target).unwrap();
        let names = fs::read_dir(&target)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            names,
            ["manifest.json", "run-envelope.json", "shared-input.bin"]
                .into_iter()
                .map(str::to_owned)
                .collect()
        );
        let binary = fs::read(target.join("shared-input.bin")).unwrap();
        let bundle = decode_linked_shared_input_bundle_v0(&binary).unwrap();
        let manifest_bytes = fs::read(target.join("manifest.json")).unwrap();
        assert_eq!(manifest_bytes.last(), Some(&b'\n'));
        let manifest: LinkedSharedInputManifestJsonV0 =
            serde_json::from_slice(&manifest_bytes).unwrap();
        manifest.validate_against(&bundle).unwrap();
        let envelope: RunEnvelopeV0 =
            serde_json::from_slice(&fs::read(target.join("run-envelope.json")).unwrap()).unwrap();
        assert_eq!(
            envelope.artifacts,
            vec![
                EnvelopeArtifactV0 {
                    name: "shared-input.bin".into(),
                    length_bytes: binary.len() as u64,
                    hash_fnv1a64: hex_hash(fnv1a64(&binary)),
                },
                EnvelopeArtifactV0 {
                    name: "manifest.json".into(),
                    length_bytes: manifest_bytes.len() as u64,
                    hash_fnv1a64: hex_hash(fnv1a64(&manifest_bytes)),
                },
            ]
        );

        let sentinel = target.join("sentinel");
        fs::write(&sentinel, b"preserve").unwrap();
        assert!(materialize_linked_shared_input_v0(&target).is_err());
        assert_eq!(fs::read(&sentinel).unwrap(), b"preserve");
        assert!(!temp.exists());
        assert!(!lock.exists());
        fs::remove_dir_all(&target).unwrap();
    }

    #[test]
    #[ignore = "full 8/4/2 executable contract; run explicitly in release mode"]
    fn full_bundle_is_deterministic_canonical_and_round_trips() {
        let first = assemble_bundle().unwrap();
        let second = assemble_bundle().unwrap();
        assert_eq!(first, second);
        assert_eq!(fixed_bytes(&first).unwrap(), fixed_bytes(&second).unwrap());
        let manifest_bytes = |bundle: &LinkedSharedInputBundleV0| {
            let mut bytes =
                serde_json::to_vec_pretty(&LinkedSharedInputManifestJsonV0::from_bundle(bundle))
                    .unwrap();
            bytes.push(b'\n');
            bytes
        };
        assert_eq!(manifest_bytes(&first), manifest_bytes(&second));
        let bytes = fixed_bytes(&first).unwrap();
        let decoded: LinkedSharedInputBundleV0 = bincode_options()
            .reject_trailing_bytes()
            .deserialize(&bytes)
            .unwrap();
        validate_against_expected(&decoded, &first).unwrap();
        assert_eq!(decoded, first);
        let json = LinkedSharedInputManifestJsonV0::from_bundle(&first);
        json.validate_against(&first).unwrap();
    }

    #[test]
    #[ignore = "full 8/4/2 repaired-mutation matrix; run explicitly in release mode"]
    fn decode_rejects_trailing_bytes_and_repaired_semantic_mutations() {
        let canonical = assemble_bundle().unwrap();
        let mut trailing = fixed_bytes(&canonical).unwrap();
        trailing.push(0);
        assert!(decode_linked_shared_input_bundle_v0(&trailing).is_err());

        let reject = |value: &LinkedSharedInputBundleV0| {
            assert!(validate_against_expected(value, &canonical).is_err());
        };
        let mutate_and_repair =
            |mut value: LinkedSharedInputBundleV0, mutation: fn(&mut LinkedResolutionInputV0)| {
                mutation(&mut value.resolutions[1]);
                repair_resolution_and_bundle(&mut value, 1);
                reject(&value);
            };

        let mut declaration = canonical.clone();
        let endpoint = &mut declaration.declaration.scenario.segments[0]
            .geometry
            .end_km
            .x;
        *endpoint = next_up(*endpoint);
        declaration.derived_bundle_hash = bundle_hash(&declaration).unwrap();
        reject(&declaration);

        mutate_and_repair(canonical.clone(), |resolution| {
            let i = (0..resolution.mesh.cell_count())
                .find(|&i| {
                    resolution.mesh.edge_offsets[i + 1] - resolution.mesh.edge_offsets[i] == 6
                })
                .unwrap();
            resolution.mesh.cell_center_km[i].x = next_up(resolution.mesh.cell_center_km[i].x);
        });
        mutate_and_repair(canonical.clone(), |resolution| {
            let i = resolution.mesh.cell_count() / 2;
            resolution.initial_elevation_km[i] = next_up(resolution.initial_elevation_km[i]);
        });
        mutate_and_repair(canonical.clone(), |resolution| {
            let i = resolution.mesh.cell_count() / 2;
            resolution.local_runoff_supply_km3_myr[i] =
                next_up(resolution.local_runoff_supply_km3_myr[i]);
        });
        mutate_and_repair(canonical.clone(), |resolution| {
            resolution.base_material_present[0] = false;
        });
        mutate_and_repair(canonical.clone(), |resolution| {
            resolution.whole_graph_candidate[0] = false;
        });
        mutate_and_repair(canonical.clone(), |resolution| {
            let i = resolution
                .mesh
                .cell_center_km
                .iter()
                .position(|center| center.x.abs() > 320.0 || center.y.abs() > 160.0)
                .unwrap();
            assert!(!resolution.central_window_candidate[i]);
            resolution.central_window_candidate[i] = true;
        });
        mutate_and_repair(canonical.clone(), |resolution| {
            let i = resolution.compiled_stencils[0]
                .weight_per_km2
                .iter()
                .position(|weight| *weight > 0.0)
                .unwrap();
            resolution.compiled_stencils[0].weight_per_km2[i] =
                next_up(resolution.compiled_stencils[0].weight_per_km2[i]);
        });
        mutate_and_repair(canonical.clone(), |resolution| {
            let i = resolution
                .cumulative_rock_displacement_km
                .iter()
                .position(|value| *value > 0.0)
                .unwrap();
            resolution.cumulative_rock_displacement_km[i] =
                next_up(resolution.cumulative_rock_displacement_km[i]);
        });
        mutate_and_repair(canonical.clone(), |resolution| {
            let frame = resolution
                .frame_witnesses
                .iter_mut()
                .find(|frame| frame.expected_activity == 1.0)
                .unwrap();
            frame.integrated_rate_km3_myr = next_up(frame.integrated_rate_km3_myr);
        });
        mutate_and_repair(canonical.clone(), |resolution| {
            resolution.summary.central_window_count += 1;
        });

        let mut swapped_frames = canonical.clone();
        swapped_frames.resolutions[1].frame_witnesses.swap(2, 3);
        repair_resolution_and_bundle(&mut swapped_frames, 1);
        reject(&swapped_frames);

        let mut mutated_frame_hash = canonical.clone();
        let resolution = &mut mutated_frame_hash.resolutions[1];
        let time = resolution.frame_witnesses[2].time_myr;
        let evaluator = mutated_frame_hash
            .declaration
            .scenario
            .compile(&resolution.mesh)
            .unwrap();
        let mut vertical = evaluator.evaluate(time).rock_vertical_rate_km_myr;
        let i = vertical.iter().position(|value| *value > 0.0).unwrap();
        vertical[i] = f32::from_bits(vertical[i].to_bits() + 1);
        resolution.frame_witnesses[2].vertical_rate_hash = frame_array_hash(
            "orogen-linked-input-v0/frame-vertical-rate",
            &mutated_frame_hash.schema_version,
            &mutated_frame_hash.hash_version,
            resolution.nominal_spacing_km,
            time,
            &vertical,
        )
        .unwrap();
        repair_resolution_and_bundle(&mut mutated_frame_hash, 1);
        reject(&mutated_frame_hash);

        let mut reversed = canonical.clone();
        reversed.resolutions.reverse();
        reversed.derived_bundle_hash = bundle_hash(&reversed).unwrap();
        reject(&reversed);

        // One fully repaired mutation crosses the public byte decoder as a
        // representative of the matrix; every matrix row above crosses the
        // identical semantic validator without repeatedly serializing ~70 MB.
        let mut decoded_witness = canonical.clone();
        let i = decoded_witness.resolutions[1].mesh.cell_count() / 2;
        decoded_witness.resolutions[1].initial_elevation_km[i] =
            next_up(decoded_witness.resolutions[1].initial_elevation_km[i]);
        repair_resolution_and_bundle(&mut decoded_witness, 1);
        let bytes = fixed_bytes(&decoded_witness).unwrap();
        assert!(decode_linked_shared_input_bundle_v0(&bytes).is_err());
    }

    fn repair_resolution_and_bundle(value: &mut LinkedSharedInputBundleV0, index: usize) {
        let schema = value.schema_version.clone();
        let hash = value.hash_version.clone();
        let resolution = &mut value.resolutions[index];
        resolution.component_hashes = recompute_component_hashes(&schema, &hash, resolution);
        resolution.derived_resolution_hash = resolution_hash(&schema, &hash, resolution).unwrap();
        value.derived_bundle_hash = bundle_hash(value).unwrap();
    }

    fn recompute_component_hashes(
        schema: &str,
        hash_version: &str,
        value: &LinkedResolutionInputV0,
    ) -> LinkedInputComponentHashesV0 {
        let spacing = value.nominal_spacing_km;
        LinkedInputComponentHashesV0 {
            mesh_hash: component_hash(
                "orogen-linked-input-v0/mesh",
                schema,
                hash_version,
                spacing,
                &value.mesh,
            )
            .unwrap(),
            initial_elevation_hash: component_hash(
                "orogen-linked-input-v0/initial-elevation",
                schema,
                hash_version,
                spacing,
                &value.initial_elevation_km,
            )
            .unwrap(),
            local_runoff_hash: component_hash(
                "orogen-linked-input-v0/local-runoff",
                schema,
                hash_version,
                spacing,
                &value.local_runoff_supply_km3_myr,
            )
            .unwrap(),
            base_material_present_hash: component_hash(
                "orogen-linked-input-v0/base-material-present",
                schema,
                hash_version,
                spacing,
                &value.base_material_present,
            )
            .unwrap(),
            whole_graph_candidate_hash: component_hash(
                "orogen-linked-input-v0/whole-graph-candidate",
                schema,
                hash_version,
                spacing,
                &value.whole_graph_candidate,
            )
            .unwrap(),
            central_window_candidate_hash: component_hash(
                "orogen-linked-input-v0/central-window-candidate",
                schema,
                hash_version,
                spacing,
                &value.central_window_candidate,
            )
            .unwrap(),
            compiled_stencils_hash: component_hash(
                "orogen-linked-input-v0/compiled-stencils",
                schema,
                hash_version,
                spacing,
                &value.compiled_stencils,
            )
            .unwrap(),
            cumulative_rock_displacement_hash: component_hash(
                "orogen-linked-input-v0/cumulative-rock-displacement",
                schema,
                hash_version,
                spacing,
                &value.cumulative_rock_displacement_km,
            )
            .unwrap(),
            frame_witnesses_hash: component_hash(
                "orogen-linked-input-v0/frame-witnesses",
                schema,
                hash_version,
                spacing,
                &value.frame_witnesses,
            )
            .unwrap(),
        }
    }

    fn next_up(value: f64) -> f64 {
        f64::from_bits(value.to_bits() + 1)
    }
}
