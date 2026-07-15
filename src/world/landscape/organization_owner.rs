//! Thin, non-authoritative engineering probes for organization owners.
//!
//! These results deliberately are not campaign artifacts. They exist only to
//! exercise the frozen owner algorithms end to end before investing in the
//! promotion-grade artifact, evidence, publication, and review machinery.

use super::organization_artifact::{
    organization_arm_config_hash_v0, GAccumulationOrderV0, GAmplitudeAuthorityV0,
    GCalibrationSolveConfigWireV0, GConfigWireV0, GForestPolicyV0, GPlanningPolicyV0,
    GQueueOrderV0, GReconstructionPolicyV0, OrganizationArmConfigPayloadV0,
    OrganizationArmConfigV0, OrganizationArmV0, OrganizationArtifactIdentityV0,
    OrganizationPredecessorsV0, OrganizationRunPurposeV0,
    ORGANIZATION_ARM_CONFIG_SCHEMA_VERSION_V0, ORGANIZATION_ARTIFACT_HASH_VERSION_V0,
    ORGANIZATION_G_CONFIG_SCHEMA_VERSION_V0,
};
use super::{
    validate_linked_shared_input_bundle_v0, BoundaryFaceCondition, LandscapeMesh,
    LinkedResolutionInputV0, LinkedSharedInputBundleV0,
};
use bincode::Options;
use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BinaryHeap};
use std::fmt;

pub const THIN_G_4KM_SCHEMA_VERSION_V0: &str = "orogen-owner-thin-g-4km-probe-v0";
pub const THIN_OWNER_PROFILE_V0: &str = "non-authoritative-4km-engineering-probe";

const ELEVATION_ARRAY_DOMAIN_V0: &str = "orogen-organization-v0/elevation-array";
const THIN_G_FOREST_DOMAIN_V0: &str = "orogen-owner-thin-v0/g-forest";
const Q_REFERENCE_KM3_MYR: f64 = 500_000.0;
const TARGET_SPACING_KM: f64 = 4.0;
const INITIAL_UPPER_A_KM_INVERSE: f64 = 0.001;
const BRACKET_GROWTH_FACTOR: f64 = 2.0;
const MAXIMUM_BRACKET_EXPANSIONS: u32 = 64;
const MAXIMUM_ITERATIONS: u32 = 128;
const VOLUME_ABSOLUTE_TOLERANCE_KM3: f64 = 1e-8;
const VOLUME_RELATIVE_TOLERANCE: f64 = 5e-12;
const RUNOFF_ABSOLUTE_TOLERANCE_KM3_MYR: f64 = 1e-6;
const AREA_ABSOLUTE_TOLERANCE_KM2: f64 = 1e-8;
const SUPPORT_THRESHOLDS_KM2: [f64; 3] = [1_000.0, 2_000.0, 4_000.0];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinOwnerErrorV0(pub String);

impl fmt::Display for ThinOwnerErrorV0 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ThinOwnerErrorV0 {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThinGReceiverV0 {
    Portal(u32),
    Cell(u32),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinGQueueCountersV0 {
    pub portal_seed_count: u64,
    pub push_count: u64,
    pub pop_count: u64,
    pub stale_pop_count: u64,
    pub relaxation_count: u64,
    pub tie_replacement_count: u64,
    pub maximum_queue_length: u64,
    pub finalized_count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinGPortalAccumulationV0 {
    pub portal_id: u32,
    pub accumulated_runoff_km3_myr: f64,
    pub accumulated_area_km2: f64,
    pub owned_cell_count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinGCalibrationAuditV0 {
    pub amplitude_a_g_km_inverse: f64,
    pub enclosing_lower_a_km_inverse: f64,
    pub enclosing_upper_a_km_inverse: f64,
    pub signed_volume_residual_km3: f64,
    pub bracket_expansion_count: u32,
    pub iteration_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinGStrahlerSupportV0 {
    pub cell_order: Vec<u32>,
    pub maximum_order: u32,
    pub thresholds_km2: Vec<f64>,
    pub cell_counts: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinGReconstructionLedgerV0 {
    pub initial_elevation_volume_moment_km3: f64,
    pub positive_added_volume_km3: f64,
    pub negative_added_volume_km3: f64,
    pub final_elevation_volume_moment_km3: f64,
    pub moment_identity_error_km3: f64,
    pub declared_work_volume_km3: f64,
    pub work_volume_residual_km3: f64,
    pub next_up_only_positive_volume_km3: f64,
    pub total_local_runoff_km3_myr: f64,
    pub total_portal_runoff_km3_myr: f64,
    pub runoff_balance_error_km3_myr: f64,
    pub total_cell_area_km2: f64,
    pub total_portal_area_km2: f64,
    pub area_balance_error_km2: f64,
    pub portals: Vec<ThinGPortalAccumulationV0>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinG4KmObservationV0 {
    pub schema_version: String,
    pub profile: String,
    pub identity: OrganizationArtifactIdentityV0,
    pub config_hash: u64,
    pub diagnostic_forest_hash: u64,
    pub final_elevation_component_hash: u64,
    pub cell_count: u64,
    pub final_min_elevation_km: f64,
    pub final_max_elevation_km: f64,
    pub strahler_support: ThinGStrahlerSupportV0,
    pub calibration: ThinGCalibrationAuditV0,
    pub ledger: ThinGReconstructionLedgerV0,
    pub queue: ThinGQueueCountersV0,
    pub final_elevation_km: Vec<f64>,
}

#[derive(Debug, Clone)]
struct GForestV0 {
    receiver: Vec<ThinGReceiverV0>,
    inherited_portal: Vec<u32>,
    path_maximum: Vec<f64>,
    finalization_order: Vec<u32>,
    rank: Vec<u32>,
    counters: ThinGQueueCountersV0,
}

#[derive(Debug, Clone)]
struct GAccumulationV0 {
    raw_edge_term_km2: Vec<f64>,
    accumulated_area_km2: Vec<f64>,
    portals: Vec<ThinGPortalAccumulationV0>,
    total_local_runoff: f64,
    total_portal_runoff: f64,
    total_cell_area: f64,
    total_portal_area: f64,
}

#[derive(Debug, Clone, Copy)]
struct VolumeReductionV0 {
    positive: f64,
    negative: f64,
}

#[derive(Debug, Clone, Copy)]
struct QueueKeyV0 {
    path_maximum: f64,
    portal_id: u32,
    candidate_cell: u32,
    receiver_kind: u32,
    receiver_index: u32,
}

impl QueueKeyV0 {
    fn bit_equal(self, other: Self) -> bool {
        self.path_maximum.to_bits() == other.path_maximum.to_bits()
            && self.portal_id == other.portal_id
            && self.candidate_cell == other.candidate_cell
            && self.receiver_kind == other.receiver_kind
            && self.receiver_index == other.receiver_index
    }

    fn receiver(self) -> ThinGReceiverV0 {
        if self.receiver_kind == 0 {
            ThinGReceiverV0::Portal(self.receiver_index)
        } else {
            ThinGReceiverV0::Cell(self.receiver_index)
        }
    }
}

impl PartialEq for QueueKeyV0 {
    fn eq(&self, other: &Self) -> bool {
        self.bit_equal(*other)
    }
}

impl Eq for QueueKeyV0 {}

impl PartialOrd for QueueKeyV0 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueKeyV0 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.path_maximum
            .total_cmp(&other.path_maximum)
            .then_with(|| self.portal_id.cmp(&other.portal_id))
            .then_with(|| self.candidate_cell.cmp(&other.candidate_cell))
            .then_with(|| self.receiver_kind.cmp(&other.receiver_kind))
            .then_with(|| self.receiver_index.cmp(&other.receiver_index))
    }
}

#[derive(Serialize)]
struct ElevationArrayPreimageV0<'a> {
    domain: &'static str,
    identity: &'a OrganizationArtifactIdentityV0,
    elevation_km: &'a Vec<f64>,
}

#[derive(Serialize)]
struct ThinForestPreimageV0<'a> {
    domain: &'static str,
    identity: &'a OrganizationArtifactIdentityV0,
    config_hash: u64,
    receiver: &'a Vec<ThinGReceiverV0>,
    inherited_portal: &'a Vec<u32>,
    path_maximum: &'a Vec<f64>,
    finalization_order: &'a Vec<u32>,
    strahler_order: &'a Vec<u32>,
    support_thresholds_km2: &'a Vec<f64>,
    support_cell_counts: &'a Vec<u64>,
}

/// Execute the exact registered G base algorithm at 4 km without constructing
/// a promotion-grade arm result or publication tree.
pub fn run_thin_g_4km_v0(
    bundle: &LinkedSharedInputBundleV0,
) -> Result<ThinG4KmObservationV0, ThinOwnerErrorV0> {
    validate_linked_shared_input_bundle_v0(bundle)
        .map_err(|error| fail(format!("linked input validation failed: {error}")))?;
    run_validated_thin_g_4km_v0(bundle)
}

/// Validate the accepted input once, execute the frozen probe twice, and reject
/// any bit-level difference before returning an observation.
pub fn run_repeated_thin_g_4km_v0(
    bundle: &LinkedSharedInputBundleV0,
) -> Result<ThinG4KmObservationV0, ThinOwnerErrorV0> {
    validate_linked_shared_input_bundle_v0(bundle)
        .map_err(|error| fail(format!("linked input validation failed: {error}")))?;
    let first = run_validated_thin_g_4km_v0(bundle)?;
    let second = run_validated_thin_g_4km_v0(bundle)?;
    require(
        fixed_bytes(&first)? == fixed_bytes(&second)?,
        "repeated thin G probe differs at bit-level comparison",
    )?;
    Ok(first)
}

fn run_validated_thin_g_4km_v0(
    bundle: &LinkedSharedInputBundleV0,
) -> Result<ThinG4KmObservationV0, ThinOwnerErrorV0> {
    let input = bundle
        .resolutions
        .iter()
        .find(|input| input.nominal_spacing_km.to_bits() == TARGET_SPACING_KM.to_bits())
        .ok_or_else(|| fail("accepted bundle has no exact 4 km resolution"))?;
    validate_g_input(input)?;

    let identity = OrganizationArtifactIdentityV0 {
        input_bundle_hash: bundle.derived_bundle_hash,
        input_resolution_hash: input.derived_resolution_hash,
        nominal_spacing_km: TARGET_SPACING_KM,
        arm: OrganizationArmV0::G,
        purpose: OrganizationRunPurposeV0::Base,
    };
    let config_hash = registered_g_config_hash(input, &identity)?;
    let planning = build_planning(input)?;
    let forest = build_g_forest(&input.mesh, &planning)?;
    validate_g_forest(&input.mesh, &planning, &forest)?;
    let accumulation = accumulate_g(input, &forest)?;
    let strahler_support = derive_g_strahler_support(
        &forest,
        &accumulation.accumulated_area_km2,
        &SUPPORT_THRESHOLDS_KM2,
    )?;

    let target_volume = bundle.declaration.analytic_rock_volume_km3;
    let calibration = solve_g_amplitude(
        input,
        &forest,
        &accumulation.raw_edge_term_km2,
        target_volume,
    )?;
    let next_up_surface = reconstruct_g(input, &forest, &accumulation.raw_edge_term_km2, 0.0)?;
    let next_up_volume = reduce_added_volume(input, &next_up_surface).positive;
    let final_elevation = reconstruct_g(
        input,
        &forest,
        &accumulation.raw_edge_term_km2,
        calibration.amplitude_a_g_km_inverse,
    )?;
    let volume = reduce_added_volume(input, &final_elevation);
    let initial_moment = elevation_moment(&input.mesh, &input.initial_elevation_km);
    let final_moment = elevation_moment(&input.mesh, &final_elevation);
    let actual_change = final_moment - initial_moment;
    let expected_change = volume.positive - volume.negative;
    let moment_error = actual_change - expected_change;
    require_close(
        final_moment,
        initial_moment + expected_change,
        VOLUME_ABSOLUTE_TOLERANCE_KM3,
        VOLUME_RELATIVE_TOLERANCE,
        "G reconstruction moment",
    )?;
    require_close(
        volume.positive,
        target_volume,
        VOLUME_ABSOLUTE_TOLERANCE_KM3,
        VOLUME_RELATIVE_TOLERANCE,
        "G declared work",
    )?;

    let final_elevation_component_hash = fnv1a64(&fixed_bytes(&ElevationArrayPreimageV0 {
        domain: ELEVATION_ARRAY_DOMAIN_V0,
        identity: &identity,
        elevation_km: &final_elevation,
    })?);
    let diagnostic_forest_hash = fnv1a64(&fixed_bytes(&ThinForestPreimageV0 {
        domain: THIN_G_FOREST_DOMAIN_V0,
        identity: &identity,
        config_hash,
        receiver: &forest.receiver,
        inherited_portal: &forest.inherited_portal,
        path_maximum: &forest.path_maximum,
        finalization_order: &forest.finalization_order,
        strahler_order: &strahler_support.cell_order,
        support_thresholds_km2: &strahler_support.thresholds_km2,
        support_cell_counts: &strahler_support.cell_counts,
    })?);
    let (final_min_elevation_km, final_max_elevation_km) = finite_min_max(&final_elevation)?;

    Ok(ThinG4KmObservationV0 {
        schema_version: THIN_G_4KM_SCHEMA_VERSION_V0.into(),
        profile: THIN_OWNER_PROFILE_V0.into(),
        identity,
        config_hash,
        diagnostic_forest_hash,
        final_elevation_component_hash,
        cell_count: checked_u64(final_elevation.len(), "cell count")?,
        final_min_elevation_km,
        final_max_elevation_km,
        strahler_support,
        calibration,
        ledger: ThinGReconstructionLedgerV0 {
            initial_elevation_volume_moment_km3: initial_moment,
            positive_added_volume_km3: volume.positive,
            negative_added_volume_km3: volume.negative,
            final_elevation_volume_moment_km3: final_moment,
            moment_identity_error_km3: moment_error,
            declared_work_volume_km3: target_volume,
            work_volume_residual_km3: volume.positive - target_volume,
            next_up_only_positive_volume_km3: next_up_volume,
            total_local_runoff_km3_myr: accumulation.total_local_runoff,
            total_portal_runoff_km3_myr: accumulation.total_portal_runoff,
            runoff_balance_error_km3_myr: accumulation.total_local_runoff
                - accumulation.total_portal_runoff,
            total_cell_area_km2: accumulation.total_cell_area,
            total_portal_area_km2: accumulation.total_portal_area,
            area_balance_error_km2: accumulation.total_cell_area - accumulation.total_portal_area,
            portals: accumulation.portals,
        },
        queue: forest.counters,
        final_elevation_km: final_elevation,
    })
}

fn registered_g_config_hash(
    input: &LinkedResolutionInputV0,
    identity: &OrganizationArtifactIdentityV0,
) -> Result<u64, ThinOwnerErrorV0> {
    let config = OrganizationArmConfigV0 {
        schema_version: ORGANIZATION_ARM_CONFIG_SCHEMA_VERSION_V0.into(),
        hash_version: ORGANIZATION_ARTIFACT_HASH_VERSION_V0.into(),
        identity: identity.clone(),
        predecessors: OrganizationPredecessorsV0 {
            opportunity_control_result_hash: None,
            g_reference_4km: None,
        },
        payload: OrganizationArmConfigPayloadV0::G(GConfigWireV0 {
            config_schema: ORGANIZATION_G_CONFIG_SCHEMA_VERSION_V0.into(),
            planning_policy: GPlanningPolicyV0::InitialPlusCumulativeDisplacement,
            forest_policy: GForestPolicyV0::MultiSourceMinimaxPortalForest,
            queue_order: GQueueOrderV0::PathMaximumTotalCmpPortalCellReceiverKindReceiver,
            accumulation_order: GAccumulationOrderV0::ReverseFinalization,
            reconstruction_policy:
                GReconstructionPolicyV0::ReceiverRecursiveNextUpRunoffConditionedRise,
            cumulative_displacement_component_hash: input
                .component_hashes
                .cumulative_rock_displacement_hash,
            runoff_component_hash: input.component_hashes.local_runoff_hash,
            q_reference_km3_myr: Q_REFERENCE_KM3_MYR,
            support_thresholds_km2: SUPPORT_THRESHOLDS_KM2.to_vec(),
            amplitude_authority: GAmplitudeAuthorityV0::SolveAtThis4Km(
                GCalibrationSolveConfigWireV0 {
                    initial_upper_a_km_inverse: INITIAL_UPPER_A_KM_INVERSE,
                    bracket_growth_factor: BRACKET_GROWTH_FACTOR,
                    maximum_bracket_expansions: MAXIMUM_BRACKET_EXPANSIONS,
                    maximum_iterations: MAXIMUM_ITERATIONS,
                    volume_absolute_tolerance_km3: VOLUME_ABSOLUTE_TOLERANCE_KM3,
                    volume_relative_tolerance: VOLUME_RELATIVE_TOLERANCE,
                },
            ),
        }),
        derived_config_hash: 0,
    };
    organization_arm_config_hash_v0(&config)
        .map_err(|error| fail(format!("registered G configuration rejected: {error}")))
}

fn validate_g_input(input: &LinkedResolutionInputV0) -> Result<(), ThinOwnerErrorV0> {
    input
        .mesh
        .validate()
        .map_err(|error| fail(format!("invalid stored mesh: {error}")))?;
    let n = input.mesh.cell_count();
    require(
        input.nominal_spacing_km.to_bits() == TARGET_SPACING_KM.to_bits(),
        "G probe requires exact 4 km input",
    )?;
    require(
        input.initial_elevation_km.len() == n
            && input.local_runoff_supply_km3_myr.len() == n
            && input.cumulative_rock_displacement_km.len() == n,
        "G input field length mismatch",
    )?;
    for cell in 0..n {
        require_finite(input.initial_elevation_km[cell], "initial elevation")?;
        require_positive(
            input.local_runoff_supply_km3_myr[cell],
            "local runoff supply",
        )?;
        require_nonnegative(
            input.cumulative_rock_displacement_km[cell],
            "cumulative displacement",
        )?;
    }
    Ok(())
}

fn build_planning(input: &LinkedResolutionInputV0) -> Result<Vec<f64>, ThinOwnerErrorV0> {
    input
        .initial_elevation_km
        .iter()
        .zip(&input.cumulative_rock_displacement_km)
        .map(|(&initial, &displacement)| {
            let value = initial + displacement;
            require_finite(value, "G planning value")?;
            Ok(value)
        })
        .collect()
}

fn build_g_forest(mesh: &LandscapeMesh, planning: &[f64]) -> Result<GForestV0, ThinOwnerErrorV0> {
    let n = mesh.cell_count();
    require(planning.len() == n, "planning length does not match mesh")?;
    let portal_levels = portal_levels(mesh)?;
    let mut queue = BinaryHeap::<Reverse<QueueKeyV0>>::new();
    let mut best = vec![None::<QueueKeyV0>; n];
    let mut finalized = vec![false; n];
    let mut receiver = vec![ThinGReceiverV0::Portal(u32::MAX); n];
    let mut inherited_portal = vec![u32::MAX; n];
    let mut path_maximum = vec![f64::NAN; n];
    let mut finalization_order = Vec::with_capacity(n);
    let mut counters = ThinGQueueCountersV0 {
        portal_seed_count: 0,
        push_count: 0,
        pop_count: 0,
        stale_pop_count: 0,
        relaxation_count: 0,
        tie_replacement_count: 0,
        maximum_queue_length: 0,
        finalized_count: 0,
    };

    for (&portal_id, &base_level) in &portal_levels {
        for face in &mesh.boundary_faces {
            let BoundaryFaceCondition::OpenBaseLevel {
                portal_id: face_portal,
                elevation_km,
            } = face.condition
            else {
                continue;
            };
            if face_portal.0 != portal_id {
                continue;
            }
            require(
                f64::from(elevation_km).to_bits() == base_level.to_bits(),
                "portal face and portal declaration base levels disagree",
            )?;
            counters.portal_seed_count = checked_add(counters.portal_seed_count, 1, "seed count")?;
            let cell = face.cell as usize;
            let key = QueueKeyV0 {
                path_maximum: base_level.max(planning[cell]),
                portal_id,
                candidate_cell: face.cell,
                receiver_kind: 0,
                receiver_index: portal_id,
            };
            maybe_push_candidate(key, &finalized, &mut best, &mut queue, &mut counters)?;
        }
    }
    require(
        counters.portal_seed_count > 0,
        "G forest has no portal seeds",
    )?;

    while finalization_order.len() < n {
        let Some(Reverse(key)) = queue.pop() else {
            return Err(fail("G forest queue emptied before all cells finalized"));
        };
        counters.pop_count = checked_add(counters.pop_count, 1, "pop count")?;
        let cell = key.candidate_cell as usize;
        if finalized[cell] || !best[cell].is_some_and(|current| current.bit_equal(key)) {
            counters.stale_pop_count = checked_add(counters.stale_pop_count, 1, "stale pop count")?;
            continue;
        }

        finalized[cell] = true;
        receiver[cell] = key.receiver();
        inherited_portal[cell] = key.portal_id;
        path_maximum[cell] = key.path_maximum;
        finalization_order.push(key.candidate_cell);
        counters.finalized_count = checked_add(counters.finalized_count, 1, "finalized count")?;

        let start = mesh.edge_offsets[cell] as usize;
        let end = mesh.edge_offsets[cell + 1] as usize;
        let mut neighbors = mesh.edge_neighbor[start..end].to_vec();
        neighbors.sort_unstable();
        for neighbor in neighbors {
            let candidate_cell = neighbor as usize;
            if finalized[candidate_cell] {
                continue;
            }
            let candidate = QueueKeyV0 {
                path_maximum: key.path_maximum.max(planning[candidate_cell]),
                portal_id: key.portal_id,
                candidate_cell: neighbor,
                receiver_kind: 1,
                receiver_index: key.candidate_cell,
            };
            maybe_push_candidate(candidate, &finalized, &mut best, &mut queue, &mut counters)?;
        }
    }

    while let Some(Reverse(_)) = queue.pop() {
        counters.pop_count = checked_add(counters.pop_count, 1, "drain pop count")?;
        counters.stale_pop_count = checked_add(counters.stale_pop_count, 1, "drain stale count")?;
    }
    require(
        counters.pop_count == counters.push_count,
        "G queue did not drain exactly",
    )?;
    require(
        counters.stale_pop_count == counters.pop_count - counters.finalized_count,
        "G stale-pop identity failed",
    )?;
    let structural_bound = checked_add(
        counters.portal_seed_count,
        checked_u64(mesh.edge_neighbor.len(), "directed edge count")?,
        "queue structural bound",
    )?;
    require(
        counters.push_count <= structural_bound
            && counters.relaxation_count == counters.push_count
            && counters.tie_replacement_count <= counters.relaxation_count
            && counters.maximum_queue_length <= counters.push_count,
        "G queue structural bound failed",
    )?;

    let mut rank = vec![u32::MAX; n];
    for (ordinal, &cell) in finalization_order.iter().enumerate() {
        rank[cell as usize] =
            u32::try_from(ordinal).map_err(|_| fail("G finalization rank exceeds u32"))?;
    }
    Ok(GForestV0 {
        receiver,
        inherited_portal,
        path_maximum,
        finalization_order,
        rank,
        counters,
    })
}

fn maybe_push_candidate(
    key: QueueKeyV0,
    finalized: &[bool],
    best: &mut [Option<QueueKeyV0>],
    queue: &mut BinaryHeap<Reverse<QueueKeyV0>>,
    counters: &mut ThinGQueueCountersV0,
) -> Result<(), ThinOwnerErrorV0> {
    require_finite(key.path_maximum, "G queue path maximum")?;
    let cell = key.candidate_cell as usize;
    if finalized[cell] {
        return Ok(());
    }
    let improves = best[cell].is_none_or(|current| key < current);
    if !improves {
        return Ok(());
    }
    if best[cell]
        .is_some_and(|current| key.path_maximum.total_cmp(&current.path_maximum) == Ordering::Equal)
    {
        counters.tie_replacement_count =
            checked_add(counters.tie_replacement_count, 1, "tie replacement count")?;
    }
    best[cell] = Some(key);
    queue.push(Reverse(key));
    counters.push_count = checked_add(counters.push_count, 1, "push count")?;
    counters.relaxation_count = checked_add(counters.relaxation_count, 1, "relaxation count")?;
    counters.maximum_queue_length = counters
        .maximum_queue_length
        .max(checked_u64(queue.len(), "queue length")?);
    Ok(())
}

fn validate_g_forest(
    mesh: &LandscapeMesh,
    planning: &[f64],
    forest: &GForestV0,
) -> Result<(), ThinOwnerErrorV0> {
    let n = mesh.cell_count();
    require(
        forest.receiver.len() == n
            && forest.inherited_portal.len() == n
            && forest.path_maximum.len() == n
            && forest.finalization_order.len() == n
            && forest.rank.len() == n,
        "G forest array length mismatch",
    )?;
    let portal_levels = portal_levels(mesh)?;
    let mut seen = vec![false; n];
    for (ordinal, &cell_u32) in forest.finalization_order.iter().enumerate() {
        let cell = cell_u32 as usize;
        require(
            cell < n && !seen[cell],
            "duplicate or invalid G finalization cell",
        )?;
        seen[cell] = true;
        require(
            forest.rank[cell] as usize == ordinal,
            "G finalization rank mismatch",
        )?;
        match forest.receiver[cell] {
            ThinGReceiverV0::Portal(portal_id) => {
                let base = *portal_levels
                    .get(&portal_id)
                    .ok_or_else(|| fail("G receiver names an unknown portal"))?;
                require(
                    has_portal_face(mesh, cell, portal_id),
                    "G portal receiver lacks an owned open face",
                )?;
                require(
                    forest.inherited_portal[cell] == portal_id
                        && forest.path_maximum[cell].to_bits()
                            == base.max(planning[cell]).to_bits(),
                    "G portal path recurrence failed",
                )?;
            }
            ThinGReceiverV0::Cell(receiver) => {
                let receiver = receiver as usize;
                require(
                    receiver < n
                        && forest.rank[receiver] < forest.rank[cell]
                        && has_directed_edge(mesh, cell, receiver),
                    "G cell receiver is not an earlier adjacent cell",
                )?;
                require(
                    forest.inherited_portal[cell] == forest.inherited_portal[receiver]
                        && forest.path_maximum[cell].to_bits()
                            == forest.path_maximum[receiver].max(planning[cell]).to_bits(),
                    "G internal path recurrence failed",
                )?;
            }
        }
    }
    require(seen.into_iter().all(|value| value), "G forest omits a cell")
}

fn accumulate_g(
    input: &LinkedResolutionInputV0,
    forest: &GForestV0,
) -> Result<GAccumulationV0, ThinOwnerErrorV0> {
    let mesh = &input.mesh;
    let n = mesh.cell_count();
    let mut runoff = input.local_runoff_supply_km3_myr.clone();
    let mut area = mesh.cell_area_km2.clone();
    let mut count = vec![1u64; n];

    for &cell_u32 in forest.finalization_order.iter().rev() {
        let cell = cell_u32 as usize;
        if let ThinGReceiverV0::Cell(receiver) = forest.receiver[cell] {
            let receiver = receiver as usize;
            runoff[receiver] += runoff[cell];
            area[receiver] += area[cell];
            count[receiver] = checked_add(count[receiver], count[cell], "accumulated cell count")?;
            require_positive(runoff[receiver], "accumulated runoff")?;
            require_positive(area[receiver], "accumulated area")?;
        }
    }

    let portal_levels = portal_levels(mesh)?;
    let mut portals = Vec::with_capacity(portal_levels.len());
    for &portal_id in portal_levels.keys() {
        let mut portal_runoff = 0.0;
        let mut portal_area = 0.0;
        let mut portal_count = 0u64;
        for cell in 0..n {
            if forest.receiver[cell] == ThinGReceiverV0::Portal(portal_id) {
                portal_runoff += runoff[cell];
                portal_area += area[cell];
                portal_count = checked_add(portal_count, count[cell], "portal owned count")?;
            }
        }
        portals.push(ThinGPortalAccumulationV0 {
            portal_id,
            accumulated_runoff_km3_myr: portal_runoff,
            accumulated_area_km2: portal_area,
            owned_cell_count: portal_count,
        });
    }
    let total_local_runoff = stored_sum(&input.local_runoff_supply_km3_myr);
    let total_portal_runoff = portals
        .iter()
        .fold(0.0, |sum, portal| sum + portal.accumulated_runoff_km3_myr);
    let total_cell_area = stored_sum(&mesh.cell_area_km2);
    let total_portal_area = portals
        .iter()
        .fold(0.0, |sum, portal| sum + portal.accumulated_area_km2);
    let total_owned_count = portals.iter().try_fold(0u64, |sum, portal| {
        checked_add(sum, portal.owned_cell_count, "total portal owner count")
    })?;
    require(
        total_owned_count == n as u64,
        "portal owner counts do not partition cells",
    )?;
    require_close(
        total_portal_runoff,
        total_local_runoff,
        RUNOFF_ABSOLUTE_TOLERANCE_KM3_MYR,
        VOLUME_RELATIVE_TOLERANCE,
        "G runoff balance",
    )?;
    require_close(
        total_portal_area,
        total_cell_area,
        AREA_ABSOLUTE_TOLERANCE_KM2,
        VOLUME_RELATIVE_TOLERANCE,
        "G area balance",
    )?;

    let mut raw_edge_term_km2 = Vec::with_capacity(n);
    for (cell, &cell_runoff) in runoff.iter().enumerate() {
        let length = match forest.receiver[cell] {
            ThinGReceiverV0::Portal(portal_id) => portal_edge_length(mesh, cell, portal_id)?,
            ThinGReceiverV0::Cell(receiver) => directed_edge_length(mesh, cell, receiver as usize)?,
        };
        require_positive(length, "G receiver distance")?;
        let ratio = cell_runoff / Q_REFERENCE_KM3_MYR;
        require_positive(ratio, "G runoff ratio")?;
        let root = ratio.sqrt();
        require_positive(root, "G runoff root")?;
        let numerator = input.cumulative_rock_displacement_km[cell] * length;
        require_nonnegative(numerator, "G edge numerator")?;
        let raw = numerator / root;
        require_nonnegative(raw, "G raw edge term")?;
        raw_edge_term_km2.push(raw);
    }
    Ok(GAccumulationV0 {
        raw_edge_term_km2,
        accumulated_area_km2: area,
        portals,
        total_local_runoff,
        total_portal_runoff,
        total_cell_area,
        total_portal_area,
    })
}

fn derive_g_strahler_support(
    forest: &GForestV0,
    accumulated_area_km2: &[f64],
    thresholds_km2: &[f64],
) -> Result<ThinGStrahlerSupportV0, ThinOwnerErrorV0> {
    let n = forest.receiver.len();
    require(
        accumulated_area_km2.len() == n,
        "G accumulated-area length does not match forest",
    )?;
    let mut children = vec![Vec::<u32>::new(); n];
    for donor in 0..n {
        if let ThinGReceiverV0::Cell(receiver) = forest.receiver[donor] {
            let receiver = receiver as usize;
            require(receiver < n, "G Strahler receiver is out of range")?;
            children[receiver]
                .push(u32::try_from(donor).map_err(|_| fail("G Strahler donor exceeds u32"))?);
        }
    }

    let mut cell_order = vec![0u32; n];
    for &cell_u32 in forest.finalization_order.iter().rev() {
        let cell = cell_u32 as usize;
        require(cell < n, "G Strahler finalization cell is out of range")?;
        let order = if children[cell].is_empty() {
            1
        } else {
            let maximum = children[cell]
                .iter()
                .map(|&child| cell_order[child as usize])
                .max()
                .ok_or_else(|| fail("G Strahler child maximum is absent"))?;
            require(maximum > 0, "G Strahler child was not evaluated first")?;
            let maximum_count = children[cell]
                .iter()
                .filter(|&&child| cell_order[child as usize] == maximum)
                .count();
            if maximum_count >= 2 {
                maximum
                    .checked_add(1)
                    .ok_or_else(|| fail("G Strahler order overflowed"))?
            } else {
                maximum
            }
        };
        cell_order[cell] = order;
    }

    for cell in 0..n {
        require(cell_order[cell] > 0, "G Strahler order is zero")?;
        require(
            children[cell].windows(2).all(|pair| pair[0] < pair[1]),
            "G Strahler children are not in ascending cell order",
        )?;
        let expected = if children[cell].is_empty() {
            1
        } else {
            let maximum = children[cell]
                .iter()
                .map(|&child| cell_order[child as usize])
                .max()
                .ok_or_else(|| fail("G Strahler validation has no child"))?;
            let maximum_count = children[cell]
                .iter()
                .filter(|&&child| cell_order[child as usize] == maximum)
                .count();
            if maximum_count >= 2 {
                maximum
                    .checked_add(1)
                    .ok_or_else(|| fail("G Strahler validation overflowed"))?
            } else {
                maximum
            }
        };
        require(
            cell_order[cell] == expected,
            "G Strahler recurrence validation failed",
        )?;
        require_positive(accumulated_area_km2[cell], "G accumulated support area")?;
    }
    let maximum_order = cell_order.iter().copied().max().unwrap_or(0);
    require(maximum_order > 0, "G Strahler maximum order is zero")?;

    let mut cell_counts = Vec::with_capacity(thresholds_km2.len());
    for (index, &threshold) in thresholds_km2.iter().enumerate() {
        require_positive(threshold, "G support threshold")?;
        if index > 0 {
            require(
                threshold > thresholds_km2[index - 1],
                "G support thresholds are not strictly increasing",
            )?;
        }
        let count = accumulated_area_km2
            .iter()
            .filter(|&&area| area >= threshold)
            .count();
        cell_counts.push(checked_u64(count, "G support cell count")?);
    }

    Ok(ThinGStrahlerSupportV0 {
        cell_order,
        maximum_order,
        thresholds_km2: thresholds_km2.to_vec(),
        cell_counts,
    })
}

fn solve_g_amplitude(
    input: &LinkedResolutionInputV0,
    forest: &GForestV0,
    raw_edge_term_km2: &[f64],
    target_volume_km3: f64,
) -> Result<ThinGCalibrationAuditV0, ThinOwnerErrorV0> {
    require_positive(target_volume_km3, "G target work volume")?;
    let mut lo = 0.0;
    let mut f_lo = evaluate_g_volume(input, forest, raw_edge_term_km2, lo)?;
    require(
        f_lo < target_volume_km3,
        "G zero-amplitude volume reaches target",
    )?;
    let mut hi = INITIAL_UPPER_A_KM_INVERSE;
    let mut expansions = 0u32;
    let mut f_hi = evaluate_g_volume(input, forest, raw_edge_term_km2, hi)?;
    while f_hi < target_volume_km3 {
        require(
            expansions < MAXIMUM_BRACKET_EXPANSIONS,
            "G amplitude bracket expansion limit reached",
        )?;
        lo = hi;
        f_lo = f_hi;
        hi *= BRACKET_GROWTH_FACTOR;
        expansions += 1;
        require_positive(hi, "G amplitude upper bracket")?;
        f_hi = evaluate_g_volume(input, forest, raw_edge_term_km2, hi)?;
    }
    require(
        f_lo < target_volume_km3 && f_hi >= target_volume_km3,
        "G amplitude bracket does not enclose target",
    )?;

    for iteration in 1..=MAXIMUM_ITERATIONS {
        let lower_before = lo;
        let upper_before = hi;
        let mid = lo + (0.5 * (hi - lo));
        require(
            mid != lo && mid != hi,
            "G amplitude midpoint no longer advances",
        )?;
        let f_mid = evaluate_g_volume(input, forest, raw_edge_term_km2, mid)?;
        require(
            f_lo <= f_mid && f_mid <= f_hi,
            "G calibration volume is not monotone",
        )?;
        let residual = f_mid - target_volume_km3;
        if close(
            f_mid,
            target_volume_km3,
            VOLUME_ABSOLUTE_TOLERANCE_KM3,
            VOLUME_RELATIVE_TOLERANCE,
        )? {
            require_positive(mid, "G solved amplitude")?;
            return Ok(ThinGCalibrationAuditV0 {
                amplitude_a_g_km_inverse: mid,
                enclosing_lower_a_km_inverse: lower_before,
                enclosing_upper_a_km_inverse: upper_before,
                signed_volume_residual_km3: residual,
                bracket_expansion_count: expansions,
                iteration_count: iteration,
            });
        }
        if residual < 0.0 {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
            f_hi = f_mid;
        }
        require(
            f_lo < target_volume_km3 && target_volume_km3 <= f_hi && lo < hi,
            "G calibration bracket invariant failed",
        )?;
    }
    Err(fail("G amplitude iteration limit reached"))
}

fn evaluate_g_volume(
    input: &LinkedResolutionInputV0,
    forest: &GForestV0,
    raw_edge_term_km2: &[f64],
    amplitude: f64,
) -> Result<f64, ThinOwnerErrorV0> {
    let elevation = reconstruct_g(input, forest, raw_edge_term_km2, amplitude)?;
    Ok(reduce_added_volume(input, &elevation).positive)
}

fn reconstruct_g(
    input: &LinkedResolutionInputV0,
    forest: &GForestV0,
    raw_edge_term_km2: &[f64],
    amplitude: f64,
) -> Result<Vec<f64>, ThinOwnerErrorV0> {
    require_nonnegative(amplitude, "G reconstruction amplitude")?;
    let n = input.mesh.cell_count();
    require(
        raw_edge_term_km2.len() == n,
        "G raw edge term length mismatch",
    )?;
    let portal_levels = portal_levels(&input.mesh)?;
    let mut elevation = vec![f64::NAN; n];
    for &cell_u32 in &forest.finalization_order {
        let cell = cell_u32 as usize;
        let receiver_value = match forest.receiver[cell] {
            ThinGReceiverV0::Portal(portal_id) => *portal_levels
                .get(&portal_id)
                .ok_or_else(|| fail("G reconstruction names an unknown portal"))?,
            ThinGReceiverV0::Cell(receiver) => elevation[receiver as usize],
        };
        let receiver_floor = next_up_v0(receiver_value)?;
        let floor = input.initial_elevation_km[cell].max(receiver_floor);
        let rise = amplitude * raw_edge_term_km2[cell];
        let value = floor + rise;
        require_finite(value, "G reconstructed elevation")?;
        require(
            value >= input.initial_elevation_km[cell],
            "G reconstruction fell below initial elevation",
        )?;
        match forest.receiver[cell] {
            ThinGReceiverV0::Portal(_) => require(
                value > receiver_value,
                "G portal receiver is not strictly downhill",
            )?,
            ThinGReceiverV0::Cell(_) => require(
                value > receiver_value,
                "G internal receiver is not strictly downhill",
            )?,
        }
        elevation[cell] = value;
    }
    require(
        elevation.iter().all(|value| value.is_finite()),
        "G reconstruction omitted a cell",
    )?;
    Ok(elevation)
}

fn next_up_v0(value: f64) -> Result<f64, ThinOwnerErrorV0> {
    require_finite(value, "next_up input")?;
    require(
        value != 0.0 || value.to_bits() == 0.0f64.to_bits(),
        "next_up rejects negative zero",
    )?;
    let bits = if value >= 0.0 {
        value.to_bits().checked_add(1)
    } else {
        value.to_bits().checked_sub(1)
    }
    .ok_or_else(|| fail("next_up bit arithmetic overflowed"))?;
    let result = if bits == (-0.0f64).to_bits() {
        0.0
    } else {
        f64::from_bits(bits)
    };
    require_finite(result, "next_up result")?;
    require(result > value, "next_up did not increase its input")?;
    Ok(result)
}

fn reduce_added_volume(input: &LinkedResolutionInputV0, elevation: &[f64]) -> VolumeReductionV0 {
    let mut positive = 0.0;
    let mut negative = 0.0;
    for (cell, &value) in elevation.iter().enumerate() {
        let difference = value - input.initial_elevation_km[cell];
        if difference > 0.0 {
            positive += difference * input.mesh.cell_area_km2[cell];
        } else if difference < 0.0 {
            negative += (-difference) * input.mesh.cell_area_km2[cell];
        }
    }
    VolumeReductionV0 { positive, negative }
}

fn elevation_moment(mesh: &LandscapeMesh, elevation: &[f64]) -> f64 {
    elevation
        .iter()
        .zip(&mesh.cell_area_km2)
        .fold(0.0, |sum, (&value, &area)| sum + (value * area))
}

fn portal_levels(mesh: &LandscapeMesh) -> Result<BTreeMap<u32, f64>, ThinOwnerErrorV0> {
    let mut levels = BTreeMap::new();
    for portal in &mesh.outlet_portals {
        let level = f64::from(portal.base_level_km);
        require_finite(level, "portal base level")?;
        require(
            levels.insert(portal.id.0, level).is_none(),
            "duplicate portal ID",
        )?;
    }
    require(!levels.is_empty(), "mesh has no outlet portals")?;
    Ok(levels)
}

fn has_portal_face(mesh: &LandscapeMesh, cell: usize, portal_id: u32) -> bool {
    mesh.boundary_faces.iter().any(|face| {
        face.cell as usize == cell
            && matches!(
                face.condition,
                BoundaryFaceCondition::OpenBaseLevel { portal_id: id, .. } if id.0 == portal_id
            )
    })
}

fn portal_edge_length(
    mesh: &LandscapeMesh,
    cell: usize,
    portal_id: u32,
) -> Result<f64, ThinOwnerErrorV0> {
    let mut minimum = None::<f64>;
    for face in &mesh.boundary_faces {
        if face.cell as usize != cell {
            continue;
        }
        if !matches!(
            face.condition,
            BoundaryFaceCondition::OpenBaseLevel { portal_id: id, .. } if id.0 == portal_id
        ) {
            continue;
        }
        require_positive(face.center_distance_km, "portal face distance")?;
        if minimum.is_none_or(|current| face.center_distance_km < current) {
            minimum = Some(face.center_distance_km);
        }
    }
    minimum.ok_or_else(|| fail("G portal receiver has no matching boundary face"))
}

fn has_directed_edge(mesh: &LandscapeMesh, donor: usize, receiver: usize) -> bool {
    let start = mesh.edge_offsets[donor] as usize;
    let end = mesh.edge_offsets[donor + 1] as usize;
    mesh.edge_neighbor[start..end]
        .iter()
        .filter(|&&neighbor| neighbor as usize == receiver)
        .count()
        == 1
}

fn directed_edge_length(
    mesh: &LandscapeMesh,
    donor: usize,
    receiver: usize,
) -> Result<f64, ThinOwnerErrorV0> {
    let start = mesh.edge_offsets[donor] as usize;
    let end = mesh.edge_offsets[donor + 1] as usize;
    let mut found = None;
    for edge in start..end {
        if mesh.edge_neighbor[edge] as usize == receiver {
            require(found.is_none(), "duplicate donor-owned receiver edge")?;
            found = Some(f64::from(mesh.edge_distance_km[edge]));
        }
    }
    found.ok_or_else(|| fail("missing donor-owned receiver edge"))
}

fn fixed_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, ThinOwnerErrorV0> {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .serialize(value)
        .map_err(|error| fail(error.to_string()))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn stored_sum(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |sum, value| sum + value)
}

fn finite_min_max(values: &[f64]) -> Result<(f64, f64), ThinOwnerErrorV0> {
    let (&first, rest) = values
        .split_first()
        .ok_or_else(|| fail("cannot reduce empty elevation"))?;
    require_finite(first, "elevation minimum/maximum")?;
    let mut minimum = first;
    let mut maximum = first;
    for &value in rest {
        require_finite(value, "elevation minimum/maximum")?;
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    Ok((minimum, maximum))
}

fn require_close(
    actual: f64,
    expected: f64,
    absolute: f64,
    relative: f64,
    name: &str,
) -> Result<(), ThinOwnerErrorV0> {
    require(
        close(actual, expected, absolute, relative)?,
        &format!("{name} does not close: actual={actual:.17e} expected={expected:.17e}"),
    )
}

fn close(
    actual: f64,
    expected: f64,
    absolute: f64,
    relative: f64,
) -> Result<bool, ThinOwnerErrorV0> {
    require_finite(actual, "close actual")?;
    require_finite(expected, "close expected")?;
    require_nonnegative(absolute, "close absolute tolerance")?;
    require_nonnegative(relative, "close relative tolerance")?;
    let difference = (actual - expected).abs();
    let scale = actual.abs().max(expected.abs());
    let limit = absolute + (relative * scale);
    Ok(difference <= limit)
}

fn checked_u64(value: usize, name: &str) -> Result<u64, ThinOwnerErrorV0> {
    u64::try_from(value).map_err(|_| fail(format!("{name} exceeds u64")))
}

fn checked_add(left: u64, right: u64, name: &str) -> Result<u64, ThinOwnerErrorV0> {
    left.checked_add(right)
        .ok_or_else(|| fail(format!("{name} overflowed")))
}

fn require_finite(value: f64, name: &str) -> Result<(), ThinOwnerErrorV0> {
    require(value.is_finite(), &format!("{name} is nonfinite"))
}

fn require_nonnegative(value: f64, name: &str) -> Result<(), ThinOwnerErrorV0> {
    require_finite(value, name)?;
    require(value >= 0.0, &format!("{name} is negative"))?;
    require(
        value != 0.0 || value.to_bits() == 0.0f64.to_bits(),
        &format!("{name} is negative zero"),
    )
}

fn require_positive(value: f64, name: &str) -> Result<(), ThinOwnerErrorV0> {
    require_finite(value, name)?;
    require(value > 0.0, &format!("{name} is not positive"))
}

fn require(condition: bool, message: &str) -> Result<(), ThinOwnerErrorV0> {
    if condition {
        Ok(())
    } else {
        Err(fail(message))
    }
}

fn fail(message: impl Into<String>) -> ThinOwnerErrorV0 {
    ThinOwnerErrorV0(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_up_is_exact_across_zero_and_signs() {
        assert_eq!(next_up_v0(0.0).unwrap().to_bits(), 1);
        assert_eq!(next_up_v0(1.0).unwrap().to_bits(), 1.0f64.to_bits() + 1);
        assert_eq!(next_up_v0(-1.0).unwrap().to_bits(), (-1.0f64).to_bits() - 1);
        assert_eq!(
            next_up_v0(-f64::from_bits(1)).unwrap().to_bits(),
            0.0f64.to_bits()
        );
        assert!(next_up_v0(-0.0).is_err());
        assert!(next_up_v0(f64::MAX).is_err());
    }

    #[test]
    fn flat_two_portal_forest_uses_complete_tie_key_and_drains_queue() {
        let mesh = LandscapeMesh::uniform_planar_hex(32.0, 24.0, 4.0).unwrap();
        let planning = vec![0.0; mesh.cell_count()];
        let first = build_g_forest(&mesh, &planning).unwrap();
        let second = build_g_forest(&mesh, &planning).unwrap();

        assert_eq!(first.receiver, second.receiver);
        assert_eq!(first.inherited_portal, second.inherited_portal);
        assert_eq!(first.path_maximum, second.path_maximum);
        assert_eq!(first.finalization_order, second.finalization_order);
        assert_eq!(first.counters, second.counters);
        assert!(first.inherited_portal.iter().all(|&portal| portal == 0));
        assert_eq!(first.counters.pop_count, first.counters.push_count);
        assert_eq!(
            first.counters.stale_pop_count,
            first.counters.pop_count - mesh.cell_count() as u64
        );
        validate_g_forest(&mesh, &planning, &first).unwrap();
    }

    #[test]
    fn forest_receiver_rank_is_strict_and_mutation_rejects() {
        let mesh = LandscapeMesh::uniform_planar_hex(40.0, 32.0, 4.0).unwrap();
        let planning: Vec<f64> = mesh
            .cell_center_km
            .iter()
            .map(|center| 1.0 + 0.001 * (center.x * center.x + center.y * center.y))
            .collect();
        let forest = build_g_forest(&mesh, &planning).unwrap();
        validate_g_forest(&mesh, &planning, &forest).unwrap();
        for cell in 0..mesh.cell_count() {
            if let ThinGReceiverV0::Cell(receiver) = forest.receiver[cell] {
                assert!(forest.rank[receiver as usize] < forest.rank[cell]);
            }
        }

        let mut broken = forest.clone();
        let donor = broken
            .receiver
            .iter()
            .position(|receiver| matches!(receiver, ThinGReceiverV0::Cell(_)))
            .unwrap();
        broken.receiver[donor] = ThinGReceiverV0::Cell(donor as u32);
        assert!(validate_g_forest(&mesh, &planning, &broken).is_err());
    }

    #[test]
    fn queue_key_orders_path_portal_cell_kind_receiver() {
        let base = QueueKeyV0 {
            path_maximum: 1.0,
            portal_id: 0,
            candidate_cell: 3,
            receiver_kind: 0,
            receiver_index: 0,
        };
        let mut values = [
            QueueKeyV0 {
                path_maximum: 2.0,
                ..base
            },
            QueueKeyV0 {
                portal_id: 1,
                ..base
            },
            QueueKeyV0 {
                candidate_cell: 4,
                ..base
            },
            QueueKeyV0 {
                receiver_kind: 1,
                receiver_index: 2,
                ..base
            },
            base,
        ];
        values.sort();
        assert!(values[0].bit_equal(base));
        assert_eq!(values[1].receiver_kind, 1);
        assert_eq!(values[2].candidate_cell, 4);
        assert_eq!(values[3].portal_id, 1);
        assert_eq!(values[4].path_maximum, 2.0);
    }

    #[test]
    fn strahler_and_support_follow_the_authored_forest() {
        let forest = GForestV0 {
            receiver: vec![
                ThinGReceiverV0::Portal(0),
                ThinGReceiverV0::Cell(0),
                ThinGReceiverV0::Cell(0),
                ThinGReceiverV0::Cell(1),
                ThinGReceiverV0::Cell(1),
            ],
            inherited_portal: vec![0; 5],
            path_maximum: vec![0.0; 5],
            finalization_order: vec![0, 1, 2, 3, 4],
            rank: vec![0, 1, 2, 3, 4],
            counters: ThinGQueueCountersV0 {
                portal_seed_count: 1,
                push_count: 5,
                pop_count: 5,
                stale_pop_count: 0,
                relaxation_count: 5,
                tie_replacement_count: 0,
                maximum_queue_length: 2,
                finalized_count: 5,
            },
        };
        let result = derive_g_strahler_support(
            &forest,
            &[7_000.0, 3_000.0, 1_000.0, 500.0, 500.0],
            &SUPPORT_THRESHOLDS_KM2,
        )
        .unwrap();

        assert_eq!(result.cell_order, vec![2, 2, 1, 1, 1]);
        assert_eq!(result.maximum_order, 2);
        assert_eq!(result.thresholds_km2, SUPPORT_THRESHOLDS_KM2);
        assert_eq!(result.cell_counts, vec![3, 2, 1]);
    }

    #[test]
    #[ignore = "builds and validates the complete linked 8/4/2 bundle before two 4 km G runs"]
    fn accepted_linked_input_g_probe_is_bit_deterministic() {
        let bundle = crate::world::landscape::build_linked_shared_input_bundle_v0().unwrap();
        let first = run_repeated_thin_g_4km_v0(&bundle).unwrap();
        assert_eq!(
            first.identity.nominal_spacing_km.to_bits(),
            4.0f64.to_bits()
        );
        assert_eq!(first.ledger.declared_work_volume_km3, 100_625.0);
        assert_eq!(first.queue.finalized_count, first.cell_count);
        assert_eq!(first.queue.pop_count, first.queue.push_count);
    }
}
