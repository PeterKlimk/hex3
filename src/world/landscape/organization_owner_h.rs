//! Thin, non-authoritative 4 km H organization-owner probe.
//!
//! This module executes the frozen target-only control and hold-and-carve base
//! algorithm without constructing promotion-grade result, trace, checkpoint or
//! publication artifacts.  In particular, the control binding below is a
//! diagnostic hash and is not a canonical campaign predecessor.

use super::organization_artifact::{
    organization_arm_config_hash_v0, ActiveProcessAccuracyConfigWireV0, ActiveProcessConfigWireV0,
    AdaptiveIntegrationConfigWireV0, DischargeSupportPolicyV0, EffectiveDenudationConfigWireV0,
    FlowPartitionPolicyV0, HConfigWireV0, HEndpointPolicyV0, HProcessModeV0, HSchedulePolicyV0,
    HillslopeBoundaryPolicyV0, LinearHillslopeConfigWireV0, OrganizationArmConfigPayloadV0,
    OrganizationArmConfigV0, OrganizationArmV0, OrganizationArtifactIdentityV0,
    OrganizationPredecessorsV0, OrganizationRunPurposeV0, RoutingConfigWireV0,
    RoutingDepressionPolicyV0, SplitOrderV0, ORGANIZATION_ARM_CONFIG_SCHEMA_VERSION_V0,
    ORGANIZATION_ARTIFACT_HASH_VERSION_V0, ORGANIZATION_H_CONFIG_SCHEMA_VERSION_V0,
};
use super::organization_process::{
    attempt_active_process_v0, elevation_moment_v0, fresh_routing_diagnostics_v0, ProcessAttemptV0,
    ProcessLimiterV0, ProcessStepV0, ProcessWaterRateV0,
};
use super::{
    validate_linked_shared_input_bundle_v0, LinkedResolutionInputV0, LinkedSharedInputBundleV0,
};
use bincode::Options;
use serde::{Deserialize, Serialize};
use std::fmt;

pub const THIN_H_4KM_SCHEMA_VERSION_V0: &str = "orogen-owner-thin-h-4km-probe-v0";

const THIN_OWNER_PROFILE_V0: &str = "non-authoritative-4km-engineering-probe";
const THIN_H_CONTROL_DOMAIN_V0: &str = "orogen-owner-thin-v0/h-control-noncanonical";
const ELEVATION_ARRAY_DOMAIN_V0: &str = "orogen-organization-v0/elevation-array";
const EXPERIMENTAL_DISPLACEMENT_DOMAIN_V0: &str =
    "orogen-owner-thin-v0/experimental-cumulative-displacement";
const EXPERIMENTAL_BUNDLE_DOMAIN_V0: &str = "orogen-owner-thin-v0/experimental-input-bundle";
const EXPERIMENTAL_RESOLUTION_DOMAIN_V0: &str =
    "orogen-owner-thin-v0/experimental-input-resolution";
const TARGET_SPACING_KM: f64 = 4.0;
const PASS_COUNT: u32 = 200;
const CHECKPOINT_PASSES: [u32; 4] = [0, 50, 120, 200];
const SCHEDULE_HORIZON_MYR: f64 = 10.0;
const ACTIVITY_RAMP_MYR: f64 = 0.25;
const ACTIVITY_END_MYR: f64 = 6.0;
const ACTIVITY_INTEGRAL_MYR: f64 = 5.75;
const OPERATOR_EXPOSURE_PER_PASS_MYR: f64 = 0.05;
const REQUESTED_MAXIMUM_DT_MYR: f64 = 0.01;
const MINIMUM_DT_MYR: f64 = 1.0e-8;
const MAXIMUM_ADAPTIVE_ATTEMPTS: u32 = 16;
const MAXIMUM_ACCEPTED_STEP_COUNT: u64 = 100_000;
const K_KM_INVERSE: f64 = 1.0e-4;
const HILLSLOPE_DIFFUSIVITY_KM2_MYR: f64 = 0.1;
const MAXIMUM_DENUDATION_DEPTH_KM: f64 = 0.02;
const DENUDATION_SLOPE_COURANT: f64 = 0.25;
const HILLSLOPE_TIMESTEP_SAFETY: f64 = 0.4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinHOwnerErrorV0(pub String);

impl fmt::Display for ThinHOwnerErrorV0 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ThinHOwnerErrorV0 {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThinHProcessLimiterV0 {
    Requested,
    EffectiveDenudationAccuracy,
    EffectiveDenudationSlopeCourant,
    HillslopeStability,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ThinHLimiterHistogramV0 {
    pub requested: u64,
    pub effective_denudation_accuracy: u64,
    pub effective_denudation_slope_courant: u64,
    pub hillslope_stability: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinHPortalVolumeV0 {
    pub portal_id: u32,
    pub volume_km3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinHPortalRateV0 {
    pub portal_id: u32,
    pub rate_km3_myr: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinHIntegratedWaterLedgerV0 {
    pub supplied_volume_km3: f64,
    pub portal_outflow_volume_km3: Vec<ThinHPortalVolumeV0>,
    pub total_portal_outflow_volume_km3: f64,
    pub unresolved_sink_volume_km3: f64,
    pub balance_error_km3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinHProcessLedgerV0 {
    pub effective_denudation_export_km3: f64,
    pub hillslope_portal_transfers_km3: Vec<ThinHPortalVolumeV0>,
    pub total_hillslope_portal_transfer_km3: f64,
    pub water: ThinHIntegratedWaterLedgerV0,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinHElevationMomentLedgerV0 {
    pub initial_elevation_volume_moment_km3: f64,
    pub gross_hold_restoration_km3: f64,
    pub process: ThinHProcessLedgerV0,
    pub final_elevation_volume_moment_km3: f64,
    pub closure_error_km3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinHWaterRateV0 {
    pub total_supply_km3_myr: f64,
    pub portal_outflow_km3_myr: Vec<ThinHPortalRateV0>,
    pub total_portal_outflow_km3_myr: f64,
    pub unresolved_sink_rate_km3_myr: f64,
    pub balance_error_km3_myr: f64,
    pub unresolved_specific_discharge_cell_count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinHCheckpointV0 {
    pub pass: u32,
    pub schedule_coordinate_myr: f64,
    pub completed_operator_exposure_myr: f64,
    pub accepted_step_count: u64,
    pub physical_elevation_component_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinHOpportunityControlV0 {
    pub identity: OrganizationArtifactIdentityV0,
    pub config_hash: u64,
    /// A probe-local binding over the complete control observation. It is not a
    /// canonical organization-result hash and cannot become a predecessor in a
    /// promotion campaign.
    pub noncanonical_control_binding_hash: u64,
    pub final_elevation_component_hash: u64,
    pub gross_hold_restoration_km3: f64,
    pub initial_elevation_volume_moment_km3: f64,
    pub final_elevation_volume_moment_km3: f64,
    pub closure_error_km3: f64,
    pub checkpoints: Vec<ThinHCheckpointV0>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinHCompletionV0 {
    pub completed_pass_count: u32,
    pub completed_operator_exposure_myr: f64,
    pub accepted_step_count: u64,
    pub total_candidate_attempt_count: u64,
    pub maximum_attempts_for_one_step: u32,
    pub minimum_accepted_dt_myr: Option<f64>,
    pub maximum_accepted_dt_myr: Option<f64>,
    pub limiter_histogram: ThinHLimiterHistogramV0,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinH4KmObservationV0 {
    pub schema_version: String,
    pub profile: String,
    pub identity: OrganizationArtifactIdentityV0,
    pub control: ThinHOpportunityControlV0,
    pub config_hash: u64,
    pub final_elevation_component_hash: u64,
    pub cell_count: u64,
    pub final_min_elevation_km: f64,
    pub final_max_elevation_km: f64,
    pub completion: ThinHCompletionV0,
    pub ledger: ThinHElevationMomentLedgerV0,
    pub checkpoints: Vec<ThinHCheckpointV0>,
    pub maximum_effective_denudation_rate_km_myr: f64,
    pub maximum_linear_hillslope_abs_grade: f64,
    pub maximum_unresolved_specific_discharge_cell_count: u64,
    pub final_routing: ThinHWaterRateV0,
    pub final_elevation_km: Vec<f64>,
}

/// Explicit provenance for a noncanonical forcing-response experiment.
///
/// These hashes identify the derivative input. Public runners recompute this
/// binding from the supplied displacement and reject caller-selected values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThinHExperimentalForcingBindingV0 {
    pub synthetic_input_bundle_hash: u64,
    pub synthetic_input_resolution_hash: u64,
    pub cumulative_displacement_component_hash: u64,
}

/// Derive the only binding accepted for this bundle and replacement target.
pub fn derive_thin_h_experimental_forcing_binding_4km_v0(
    bundle: &LinkedSharedInputBundleV0,
    cumulative_rock_displacement_km: &[f64],
) -> Result<ThinHExperimentalForcingBindingV0, ThinHOwnerErrorV0> {
    validate_linked_shared_input_bundle_v0(bundle)
        .map_err(|error| fail(format!("linked input validation failed: {error}")))?;
    let input = accepted_4km_input_v0(bundle)?;
    validate_target_displacement_v0(input.mesh.cell_count(), cumulative_rock_displacement_km)?;
    derived_experimental_binding_v0(
        bundle,
        input,
        cumulative_rock_displacement_km,
    )
}

#[derive(Debug, Clone, Copy)]
struct HForcingBindingV0 {
    input_bundle_hash: u64,
    input_resolution_hash: u64,
    cumulative_displacement_component_hash: u64,
}

#[derive(Debug, Clone, PartialEq)]
struct HCommittedStateV0 {
    physical_elevation_km: Vec<f64>,
    completed_operator_exposure_myr: f64,
    gross_hold_restoration_km3: f64,
    process: ThinHProcessLedgerV0,
    accepted_step_count: u64,
    total_candidate_attempt_count: u64,
    maximum_attempts_for_one_step: u32,
    minimum_accepted_dt_myr: Option<f64>,
    maximum_accepted_dt_myr: Option<f64>,
    limiter_histogram: ThinHLimiterHistogramV0,
    maximum_effective_denudation_rate_km_myr: f64,
    maximum_linear_hillslope_abs_grade: f64,
    maximum_unresolved_specific_discharge_cell_count: u64,
}

#[derive(Serialize)]
struct ElevationArrayPreimageV0<'a> {
    domain: &'static str,
    identity: &'a OrganizationArtifactIdentityV0,
    elevation_km: &'a Vec<f64>,
}

#[derive(Serialize)]
struct ThinHControlPreimageV0<'a> {
    domain: &'static str,
    identity: &'a OrganizationArtifactIdentityV0,
    config_hash: u64,
    final_elevation_component_hash: u64,
    gross_hold_restoration_km3: f64,
    initial_elevation_volume_moment_km3: f64,
    final_elevation_volume_moment_km3: f64,
    closure_error_km3: f64,
    checkpoints: &'a Vec<ThinHCheckpointV0>,
    final_elevation_km: &'a Vec<f64>,
}

#[derive(Debug, Clone)]
struct OpportunityControlExecutionV0 {
    observation: ThinHOpportunityControlV0,
}

/// Execute the exact registered H target-only control and base at 4 km without
/// constructing a promotion-grade result or provenance tree.
pub fn run_thin_h_4km_v0(
    bundle: &LinkedSharedInputBundleV0,
) -> Result<ThinH4KmObservationV0, ThinHOwnerErrorV0> {
    validate_linked_shared_input_bundle_v0(bundle)
        .map_err(|error| fail(format!("linked input validation failed: {error}")))?;
    let input = accepted_4km_input_v0(bundle)?;
    run_validated_thin_h_4km_v0(
        input,
        &input.cumulative_rock_displacement_km,
        accepted_forcing_binding_v0(bundle, input),
    )
}

/// Run the exact H process owner against caller-supplied cumulative
/// displacement while retaining the validated accepted mesh, initial surface,
/// runoff, portals, and schedule.
///
/// This is deliberately noncanonical engineering evidence. The caller must
/// supply a distinct synthetic input identity and a component hash binding the
/// replacement displacement; the returned observation therefore cannot be
/// mistaken for an accepted linked-input H result.
pub fn run_thin_h_experimental_forcing_4km_v0(
    bundle: &LinkedSharedInputBundleV0,
    cumulative_rock_displacement_km: &[f64],
    binding: ThinHExperimentalForcingBindingV0,
) -> Result<ThinH4KmObservationV0, ThinHOwnerErrorV0> {
    validate_linked_shared_input_bundle_v0(bundle)
        .map_err(|error| fail(format!("linked input validation failed: {error}")))?;
    let input = accepted_4km_input_v0(bundle)?;
    validate_target_displacement_v0(input.mesh.cell_count(), cumulative_rock_displacement_km)?;
    let derived = derived_experimental_binding_v0(bundle, input, cumulative_rock_displacement_km)?;
    require(
        binding == derived,
        "experimental H forcing binding does not match supplied displacement",
    )?;
    validate_experimental_binding_v0(bundle, input, binding)?;
    let target_work: f64 = cumulative_rock_displacement_km
        .iter()
        .zip(&input.mesh.cell_area_km2)
        .map(|(depth, area)| depth * area)
        .sum();
    require_close_with_tolerance_v0(
        target_work,
        bundle.declaration.analytic_rock_volume_km3,
        1.0e-6,
        5.0e-7,
        "H experimental target work",
    )?;
    run_validated_thin_h_4km_v0(
        input,
        cumulative_rock_displacement_km,
        HForcingBindingV0 {
            input_bundle_hash: binding.synthetic_input_bundle_hash,
            input_resolution_hash: binding.synthetic_input_resolution_hash,
            cumulative_displacement_component_hash: binding.cumulative_displacement_component_hash,
        },
    )
}

/// Validate the accepted bundle once, execute the complete H probe twice and
/// reject any bit-level difference in the resulting observation.
pub fn run_repeated_thin_h_4km_v0(
    bundle: &LinkedSharedInputBundleV0,
) -> Result<ThinH4KmObservationV0, ThinHOwnerErrorV0> {
    validate_linked_shared_input_bundle_v0(bundle)
        .map_err(|error| fail(format!("linked input validation failed: {error}")))?;
    let input = accepted_4km_input_v0(bundle)?;
    let binding = accepted_forcing_binding_v0(bundle, input);
    let first =
        run_validated_thin_h_4km_v0(input, &input.cumulative_rock_displacement_km, binding)?;
    let second =
        run_validated_thin_h_4km_v0(input, &input.cumulative_rock_displacement_km, binding)?;
    require(
        fixed_bytes(&first)? == fixed_bytes(&second)?,
        "repeated thin H probe differs at bit-level comparison",
    )?;
    Ok(first)
}

fn run_validated_thin_h_4km_v0(
    input: &LinkedResolutionInputV0,
    cumulative_rock_displacement_km: &[f64],
    forcing_binding: HForcingBindingV0,
) -> Result<ThinH4KmObservationV0, ThinHOwnerErrorV0> {
    validate_h_input_v0(input)?;
    validate_target_displacement_v0(input.mesh.cell_count(), cumulative_rock_displacement_km)?;
    let control =
        run_opportunity_control_v0(input, cumulative_rock_displacement_km, forcing_binding)?
            .observation;
    let identity = identity_v0(forcing_binding, OrganizationRunPurposeV0::Base);
    let config_hash = registered_base_config_hash_v0(
        input,
        &identity,
        control.noncanonical_control_binding_hash,
        forcing_binding.cumulative_displacement_component_hash,
    )?;

    let initial_moment = shared_elevation_moment_v0(input, &input.initial_elevation_km)?;
    let zero_portals = zero_portal_volumes_v0(input);
    let mut state = HCommittedStateV0 {
        physical_elevation_km: input.initial_elevation_km.clone(),
        completed_operator_exposure_myr: 0.0,
        gross_hold_restoration_km3: 0.0,
        process: ThinHProcessLedgerV0 {
            effective_denudation_export_km3: 0.0,
            hillslope_portal_transfers_km3: zero_portals.clone(),
            total_hillslope_portal_transfer_km3: 0.0,
            water: ThinHIntegratedWaterLedgerV0 {
                supplied_volume_km3: 0.0,
                portal_outflow_volume_km3: zero_portals,
                total_portal_outflow_volume_km3: 0.0,
                unresolved_sink_volume_km3: 0.0,
                balance_error_km3: 0.0,
            },
        },
        accepted_step_count: 0,
        total_candidate_attempt_count: 0,
        maximum_attempts_for_one_step: 0,
        minimum_accepted_dt_myr: None,
        maximum_accepted_dt_myr: None,
        limiter_histogram: ThinHLimiterHistogramV0::default(),
        maximum_effective_denudation_rate_km_myr: 0.0,
        maximum_linear_hillslope_abs_grade: 0.0,
        maximum_unresolved_specific_discharge_cell_count: 0,
    };
    let mut checkpoints = Vec::with_capacity(CHECKPOINT_PASSES.len());
    checkpoints.push(checkpoint_v0(
        &identity,
        0,
        0.0,
        0.0,
        0,
        &state.physical_elevation_km,
    )?);

    for pass in 1..=PASS_COUNT {
        let (schedule_coordinate, progress) = target_progress_v0(pass, PASS_COUNT)?;
        let endpoint = pass_endpoint_v0(pass)?;
        execute_transactional_pass_v0(
            &mut state,
            &input.initial_elevation_km,
            cumulative_rock_displacement_km,
            &input.mesh.cell_area_km2,
            progress,
            |scratch| carve_to_endpoint_v0(input, endpoint, scratch),
        )?;
        require(
            state.completed_operator_exposure_myr.to_bits() == endpoint.to_bits(),
            "H pass did not land on its exact operator-exposure endpoint",
        )?;
        if CHECKPOINT_PASSES.contains(&pass) {
            checkpoints.push(checkpoint_v0(
                &identity,
                pass,
                schedule_coordinate,
                endpoint,
                state.accepted_step_count,
                &state.physical_elevation_km,
            )?);
        }
    }

    require(
        state.completed_operator_exposure_myr.to_bits() == SCHEDULE_HORIZON_MYR.to_bits(),
        "H did not complete exact 10 operator-Myr exposure",
    )?;
    require(
        checkpoints
            .iter()
            .map(|checkpoint| checkpoint.pass)
            .eq(CHECKPOINT_PASSES),
        "H checkpoint pass set is incomplete",
    )?;

    let final_moment = shared_elevation_moment_v0(input, &state.physical_elevation_km)?;
    let expected_final = initial_moment + state.gross_hold_restoration_km3
        - state.process.effective_denudation_export_km3
        - state.process.total_hillslope_portal_transfer_km3;
    let closure_error = final_moment - expected_final;
    require_close_v0(final_moment, expected_final, "H cumulative solid moment")?;
    let integrated_expected = state.process.water.total_portal_outflow_volume_km3
        + state.process.water.unresolved_sink_volume_km3;
    state.process.water.balance_error_km3 =
        state.process.water.supplied_volume_km3 - integrated_expected;
    require_close_v0(
        state.process.water.supplied_volume_km3,
        integrated_expected,
        "H integrated water",
    )?;

    let final_routing_process = fresh_routing_diagnostics_v0(
        &input.mesh,
        &state.physical_elevation_km,
        &input.local_runoff_supply_km3_myr,
    )
    .map_err(|error| fail(format!("H final routing failed: {error}")))?;
    validate_water_rate_v0(&final_routing_process)?;
    let final_routing = thin_water_rate_v0(&final_routing_process);
    let final_elevation_component_hash =
        elevation_hash_v0(&identity, &state.physical_elevation_km)?;
    let (final_min_elevation_km, final_max_elevation_km) =
        finite_min_max_v0(&state.physical_elevation_km)?;

    Ok(ThinH4KmObservationV0 {
        schema_version: THIN_H_4KM_SCHEMA_VERSION_V0.into(),
        profile: THIN_OWNER_PROFILE_V0.into(),
        identity,
        control,
        config_hash,
        final_elevation_component_hash,
        cell_count: u64::try_from(state.physical_elevation_km.len())
            .map_err(|_| fail("H cell count does not fit u64"))?,
        final_min_elevation_km,
        final_max_elevation_km,
        completion: ThinHCompletionV0 {
            completed_pass_count: PASS_COUNT,
            completed_operator_exposure_myr: state.completed_operator_exposure_myr,
            accepted_step_count: state.accepted_step_count,
            total_candidate_attempt_count: state.total_candidate_attempt_count,
            maximum_attempts_for_one_step: state.maximum_attempts_for_one_step,
            minimum_accepted_dt_myr: state.minimum_accepted_dt_myr,
            maximum_accepted_dt_myr: state.maximum_accepted_dt_myr,
            limiter_histogram: state.limiter_histogram,
        },
        ledger: ThinHElevationMomentLedgerV0 {
            initial_elevation_volume_moment_km3: initial_moment,
            gross_hold_restoration_km3: state.gross_hold_restoration_km3,
            process: state.process,
            final_elevation_volume_moment_km3: final_moment,
            closure_error_km3: closure_error,
        },
        checkpoints,
        maximum_effective_denudation_rate_km_myr: state.maximum_effective_denudation_rate_km_myr,
        maximum_linear_hillslope_abs_grade: state.maximum_linear_hillslope_abs_grade,
        maximum_unresolved_specific_discharge_cell_count: state
            .maximum_unresolved_specific_discharge_cell_count,
        final_routing,
        final_elevation_km: state.physical_elevation_km,
    })
}

/// Analytic integral of the linked episode's two smoothstep end ramps.
pub fn cumulative_linked_activity_myr_v0(t_myr: f64) -> Result<f64, ThinHOwnerErrorV0> {
    require_finite(t_myr, "H activity coordinate")?;
    let integral = if t_myr <= 0.0 {
        0.0
    } else if t_myr < ACTIVITY_RAMP_MYR {
        let s = t_myr / ACTIVITY_RAMP_MYR;
        let s2 = s * s;
        let s3 = s2 * s;
        let s4 = s3 * s;
        ACTIVITY_RAMP_MYR * (s3 - (0.5 * s4))
    } else if t_myr <= ACTIVITY_END_MYR - ACTIVITY_RAMP_MYR {
        t_myr - (0.5 * ACTIVITY_RAMP_MYR)
    } else if t_myr < ACTIVITY_END_MYR {
        let s = (ACTIVITY_END_MYR - t_myr) / ACTIVITY_RAMP_MYR;
        let s2 = s * s;
        let s3 = s2 * s;
        let s4 = s3 * s;
        ACTIVITY_INTEGRAL_MYR - (ACTIVITY_RAMP_MYR * (s3 - (0.5 * s4)))
    } else {
        ACTIVITY_INTEGRAL_MYR
    };
    require_finite(integral, "H cumulative activity")?;
    Ok(integral)
}

fn target_progress_v0(pass: u32, pass_count: u32) -> Result<(f64, f64), ThinHOwnerErrorV0> {
    require(
        pass <= pass_count && pass_count > 0,
        "invalid H pass coordinate",
    )?;
    let u = f64::from(pass) / f64::from(pass_count);
    let schedule_coordinate = SCHEDULE_HORIZON_MYR * u;
    let progress = if pass == 0 {
        0.0
    } else if pass == pass_count {
        1.0
    } else {
        cumulative_linked_activity_myr_v0(schedule_coordinate)? / ACTIVITY_INTEGRAL_MYR
    };
    require_finite(progress, "H target progress")?;
    Ok((schedule_coordinate, progress))
}

fn apply_hold_v0(
    elevation: &mut [f64],
    initial: &[f64],
    displacement: &[f64],
    area: &[f64],
    progress: f64,
) -> Result<f64, ThinHOwnerErrorV0> {
    let n = elevation.len();
    require(
        initial.len() == n && displacement.len() == n && area.len() == n,
        "H hold field length mismatch",
    )?;
    let mut pass_hold_volume = 0.0;
    for cell in 0..n {
        let target = initial[cell] + (progress * displacement[cell]);
        require_finite(target, "H target elevation")?;
        if target > elevation[cell] {
            let addition = target - elevation[cell];
            elevation[cell] = target;
            pass_hold_volume = pass_hold_volume + (addition * area[cell]);
        }
    }
    require_finite(pass_hold_volume, "H pass hold volume")?;
    Ok(pass_hold_volume)
}

fn execute_transactional_pass_v0<F>(
    committed: &mut HCommittedStateV0,
    initial: &[f64],
    displacement: &[f64],
    area: &[f64],
    progress: f64,
    carve: F,
) -> Result<(), ThinHOwnerErrorV0>
where
    F: FnOnce(&mut HCommittedStateV0) -> Result<(), ThinHOwnerErrorV0>,
{
    let mut scratch = committed.clone();
    let hold = apply_hold_v0(
        &mut scratch.physical_elevation_km,
        initial,
        displacement,
        area,
        progress,
    )?;
    scratch.gross_hold_restoration_km3 = scratch.gross_hold_restoration_km3 + hold;
    carve(&mut scratch)?;
    *committed = scratch;
    Ok(())
}

fn carve_to_endpoint_v0(
    input: &LinkedResolutionInputV0,
    endpoint: f64,
    state: &mut HCommittedStateV0,
) -> Result<(), ThinHOwnerErrorV0> {
    require_finite(endpoint, "H exposure endpoint")?;
    require(
        endpoint > state.completed_operator_exposure_myr,
        "H exposure endpoint is not ahead of committed coordinate",
    )?;
    while state.completed_operator_exposure_myr.to_bits() != endpoint.to_bits() {
        require(
            state.accepted_step_count < MAXIMUM_ACCEPTED_STEP_COUNT,
            "H reached maximum accepted-step count before its endpoint",
        )?;
        let start = state.completed_operator_exposure_myr;
        let remaining = endpoint - start;
        require_finite(remaining, "H remaining exposure")?;
        require(remaining > 0.0, "H remaining exposure is not positive")?;
        let ordinary = REQUESTED_MAXIMUM_DT_MYR.min(remaining);
        let tail = endpoint - (start + ordinary);
        let requested = if tail < MINIMUM_DT_MYR {
            remaining
        } else {
            ordinary
        };
        let mut candidate = requested;
        let mut final_limiter = ThinHProcessLimiterV0::Requested;
        let mut accepted: Option<(ProcessStepV0, f64, u32)> = None;

        for attempt in 1..=MAXIMUM_ADAPTIVE_ATTEMPTS {
            if candidate < MINIMUM_DT_MYR {
                return Err(fail(format!(
                    "H candidate dt {candidate:.17e} is below minimum {MINIMUM_DT_MYR:.17e}"
                )));
            }
            require_finite(candidate, "H candidate dt")?;
            require(candidate > 0.0, "H candidate dt is not positive")?;
            let midpoint = start + (0.5 * candidate);
            require_finite(midpoint, "H attempt midpoint")?;
            state.total_candidate_attempt_count = state
                .total_candidate_attempt_count
                .checked_add(1)
                .ok_or_else(|| fail("H candidate-attempt count overflow"))?;

            match attempt_active_process_v0(
                &input.mesh,
                &state.physical_elevation_km,
                &input.local_runoff_supply_km3_myr,
                candidate,
            )
            .map_err(|error| fail(format!("H active process failed: {error}")))?
            {
                ProcessAttemptV0::Accepted(step) => {
                    accepted = Some((step, candidate, attempt));
                    break;
                }
                ProcessAttemptV0::Retry { limiter, limit_myr } => {
                    require_finite(limit_myr, "H process limit")?;
                    require(limit_myr > 0.0, "H process limit is not positive")?;
                    final_limiter = thin_limiter_v0(limiter);
                    if attempt == MAXIMUM_ADAPTIVE_ATTEMPTS {
                        return Err(fail("H maximum adaptive attempts reached"));
                    }
                    candidate = candidate.min(limit_myr);
                }
            }
        }

        let (step, accepted_dt, attempts) =
            accepted.ok_or_else(|| fail("H adaptive driver produced no accepted step"))?;
        validate_process_step_v0(input, state, &step, accepted_dt)?;
        accumulate_process_step_v0(state, &step, accepted_dt, final_limiter, attempts)?;
        let initial_moment = shared_elevation_moment_v0(input, &input.initial_elevation_km)?;
        let candidate_expected_moment = initial_moment + state.gross_hold_restoration_km3
            - state.process.effective_denudation_export_km3
            - state.process.total_hillslope_portal_transfer_km3;
        require_close_v0(
            step.final_elevation_moment_km3,
            candidate_expected_moment,
            "H candidate cumulative solid moment",
        )?;
        let candidate_water_expected = state.process.water.total_portal_outflow_volume_km3
            + state.process.water.unresolved_sink_volume_km3;
        require_close_with_tolerance_v0(
            state.process.water.supplied_volume_km3,
            candidate_water_expected,
            1.0e-8,
            5.0e-12,
            "H candidate integrated water",
        )?;
        state.process.water.balance_error_km3 =
            state.process.water.supplied_volume_km3 - candidate_water_expected;
        state.physical_elevation_km = step.final_elevation_km;

        let end = if accepted_dt.to_bits() == remaining.to_bits() {
            endpoint
        } else {
            let ordinary = start + accepted_dt;
            require(
                ordinary > start && ordinary < endpoint,
                "H accepted step does not remain within endpoint",
            )?;
            ordinary
        };
        state.completed_operator_exposure_myr = end;
    }
    Ok(())
}

fn validate_process_step_v0(
    input: &LinkedResolutionInputV0,
    state: &HCommittedStateV0,
    step: &ProcessStepV0,
    accepted_dt: f64,
) -> Result<(), ThinHOwnerErrorV0> {
    require(
        step.final_elevation_km.len() == input.mesh.cell_count(),
        "H accepted process surface has wrong length",
    )?;
    let initial_moment = shared_elevation_moment_v0(input, &state.physical_elevation_km)?;
    let final_moment = shared_elevation_moment_v0(input, &step.final_elevation_km)?;
    require(
        step.initial_elevation_moment_km3.to_bits() == initial_moment.to_bits()
            && step.final_elevation_moment_km3.to_bits() == final_moment.to_bits(),
        "H process moment witnesses disagree with fresh reductions",
    )?;
    let actual_change = final_moment - initial_moment;
    let expected_change =
        0.0 - step.effective_denudation_export_km3 - step.hillslope_portal_transfer_km3;
    require(
        step.elevation_moment_change_km3.to_bits() == actual_change.to_bits()
            && step.expected_elevation_moment_change_km3.to_bits() == expected_change.to_bits()
            && step.process_solid_closure_error_km3.to_bits()
                == (actual_change - expected_change).to_bits(),
        "H process solid witnesses disagree with fresh reductions",
    )?;
    require_close_v0(actual_change, expected_change, "H per-step solid moment")?;
    validate_water_rate_v0(&step.water)?;
    require_finite(
        step.hillslope_internal_conservation_error_km3,
        "H hillslope internal conservation error",
    )?;
    require_finite(accepted_dt, "H accepted dt")?;
    Ok(())
}

fn accumulate_process_step_v0(
    state: &mut HCommittedStateV0,
    step: &ProcessStepV0,
    accepted_dt: f64,
    final_limiter: ThinHProcessLimiterV0,
    attempts: u32,
) -> Result<(), ThinHOwnerErrorV0> {
    state.process.effective_denudation_export_km3 =
        state.process.effective_denudation_export_km3 + step.effective_denudation_export_km3;
    accumulate_portal_volumes_v0(
        &mut state.process.hillslope_portal_transfers_km3,
        &step.hillslope_portal_transfers_km3,
    )?;
    state.process.total_hillslope_portal_transfer_km3 =
        state.process.total_hillslope_portal_transfer_km3 + step.hillslope_portal_transfer_km3;

    let supplied = step.water.total_supply_km3_myr * accepted_dt;
    let portal_total = step.water.total_portal_outflow_km3_myr * accepted_dt;
    let unresolved = step.water.unresolved_sink_rate_km3_myr * accepted_dt;
    state.process.water.supplied_volume_km3 = state.process.water.supplied_volume_km3 + supplied;
    accumulate_portal_rates_v0(
        &mut state.process.water.portal_outflow_volume_km3,
        &step.water,
        accepted_dt,
    )?;
    state.process.water.total_portal_outflow_volume_km3 =
        state.process.water.total_portal_outflow_volume_km3 + portal_total;
    state.process.water.unresolved_sink_volume_km3 =
        state.process.water.unresolved_sink_volume_km3 + unresolved;

    state.maximum_effective_denudation_rate_km_myr = state
        .maximum_effective_denudation_rate_km_myr
        .max(step.maximum_effective_denudation_rate_km_myr);
    state.maximum_linear_hillslope_abs_grade = state
        .maximum_linear_hillslope_abs_grade
        .max(step.maximum_linear_hillslope_abs_grade);
    state.maximum_unresolved_specific_discharge_cell_count = state
        .maximum_unresolved_specific_discharge_cell_count
        .max(step.water.unresolved_specific_discharge_cell_count);
    state.accepted_step_count = state
        .accepted_step_count
        .checked_add(1)
        .ok_or_else(|| fail("H accepted-step count overflow"))?;
    state.maximum_attempts_for_one_step = state.maximum_attempts_for_one_step.max(attempts);
    state.minimum_accepted_dt_myr = Some(
        state
            .minimum_accepted_dt_myr
            .map_or(accepted_dt, |value| value.min(accepted_dt)),
    );
    state.maximum_accepted_dt_myr = Some(
        state
            .maximum_accepted_dt_myr
            .map_or(accepted_dt, |value| value.max(accepted_dt)),
    );
    increment_limiter_v0(&mut state.limiter_histogram, final_limiter)?;
    Ok(())
}

fn accumulate_portal_volumes_v0(
    totals: &mut [ThinHPortalVolumeV0],
    step: &[super::organization_process::ProcessPortalVolumeV0],
) -> Result<(), ThinHOwnerErrorV0> {
    require(
        totals.len() == step.len(),
        "H hillslope portal vector length mismatch",
    )?;
    for (total, value) in totals.iter_mut().zip(step) {
        require(
            total.portal_id == value.portal_id,
            "H hillslope portal order mismatch",
        )?;
        total.volume_km3 = total.volume_km3 + value.volume_km3;
    }
    Ok(())
}

fn accumulate_portal_rates_v0(
    totals: &mut [ThinHPortalVolumeV0],
    water: &ProcessWaterRateV0,
    dt: f64,
) -> Result<(), ThinHOwnerErrorV0> {
    require(
        totals.len() == water.portal_outflow_km3_myr.len(),
        "H water portal vector length mismatch",
    )?;
    for (total, value) in totals.iter_mut().zip(&water.portal_outflow_km3_myr) {
        require(
            total.portal_id == value.portal_id,
            "H water portal order mismatch",
        )?;
        let volume = value.rate_km3_myr * dt;
        total.volume_km3 = total.volume_km3 + volume;
    }
    Ok(())
}

fn validate_water_rate_v0(water: &ProcessWaterRateV0) -> Result<(), ThinHOwnerErrorV0> {
    let expected = water.total_portal_outflow_km3_myr + water.unresolved_sink_rate_km3_myr;
    require(
        water.balance_error_km3_myr.to_bits() == (water.total_supply_km3_myr - expected).to_bits(),
        "H water balance witness disagrees with fresh reduction",
    )?;
    require_close_with_tolerance_v0(
        water.total_supply_km3_myr,
        expected,
        1.0e-6,
        5.0e-12,
        "H instantaneous water",
    )
}

fn thin_water_rate_v0(water: &ProcessWaterRateV0) -> ThinHWaterRateV0 {
    ThinHWaterRateV0 {
        total_supply_km3_myr: water.total_supply_km3_myr,
        portal_outflow_km3_myr: water
            .portal_outflow_km3_myr
            .iter()
            .map(|value| ThinHPortalRateV0 {
                portal_id: value.portal_id,
                rate_km3_myr: value.rate_km3_myr,
            })
            .collect(),
        total_portal_outflow_km3_myr: water.total_portal_outflow_km3_myr,
        unresolved_sink_rate_km3_myr: water.unresolved_sink_rate_km3_myr,
        balance_error_km3_myr: water.balance_error_km3_myr,
        unresolved_specific_discharge_cell_count: water.unresolved_specific_discharge_cell_count,
    }
}

fn thin_limiter_v0(limiter: ProcessLimiterV0) -> ThinHProcessLimiterV0 {
    match limiter {
        ProcessLimiterV0::EffectiveDenudationSlopeCourant => {
            ThinHProcessLimiterV0::EffectiveDenudationSlopeCourant
        }
        ProcessLimiterV0::EffectiveDenudationAccuracy => {
            ThinHProcessLimiterV0::EffectiveDenudationAccuracy
        }
        ProcessLimiterV0::HillslopeStability => ThinHProcessLimiterV0::HillslopeStability,
    }
}

fn increment_limiter_v0(
    histogram: &mut ThinHLimiterHistogramV0,
    limiter: ThinHProcessLimiterV0,
) -> Result<(), ThinHOwnerErrorV0> {
    let slot = match limiter {
        ThinHProcessLimiterV0::Requested => &mut histogram.requested,
        ThinHProcessLimiterV0::EffectiveDenudationAccuracy => {
            &mut histogram.effective_denudation_accuracy
        }
        ThinHProcessLimiterV0::EffectiveDenudationSlopeCourant => {
            &mut histogram.effective_denudation_slope_courant
        }
        ThinHProcessLimiterV0::HillslopeStability => &mut histogram.hillslope_stability,
    };
    *slot = slot
        .checked_add(1)
        .ok_or_else(|| fail("H limiter histogram overflow"))?;
    Ok(())
}

fn run_opportunity_control_v0(
    input: &LinkedResolutionInputV0,
    cumulative_rock_displacement_km: &[f64],
    forcing_binding: HForcingBindingV0,
) -> Result<OpportunityControlExecutionV0, ThinHOwnerErrorV0> {
    let identity = identity_v0(
        forcing_binding,
        OrganizationRunPurposeV0::OpportunityControl,
    );
    let config_hash = registered_control_config_hash_v0(
        input,
        &identity,
        forcing_binding.cumulative_displacement_component_hash,
    )?;
    let initial_moment = shared_elevation_moment_v0(input, &input.initial_elevation_km)?;
    let mut elevation = input.initial_elevation_km.clone();
    let mut gross_hold = 0.0;
    let mut checkpoints = Vec::with_capacity(CHECKPOINT_PASSES.len());
    checkpoints.push(checkpoint_v0(&identity, 0, 0.0, 0.0, 0, &elevation)?);

    for pass in 1..=PASS_COUNT {
        let (schedule_coordinate, progress) = target_progress_v0(pass, PASS_COUNT)?;
        let pass_hold = apply_hold_v0(
            &mut elevation,
            &input.initial_elevation_km,
            cumulative_rock_displacement_km,
            &input.mesh.cell_area_km2,
            progress,
        )?;
        gross_hold = gross_hold + pass_hold;
        if CHECKPOINT_PASSES.contains(&pass) {
            checkpoints.push(checkpoint_v0(
                &identity,
                pass,
                schedule_coordinate,
                0.0,
                0,
                &elevation,
            )?);
        }
    }

    for (cell, (&initial, &displacement)) in input
        .initial_elevation_km
        .iter()
        .zip(cumulative_rock_displacement_km)
        .enumerate()
    {
        require(
            elevation[cell].to_bits() == (initial + displacement).to_bits(),
            "H target-only final surface is not bit-identical to z0+D",
        )?;
    }
    let final_moment = shared_elevation_moment_v0(input, &elevation)?;
    let expected_final = initial_moment + gross_hold;
    let closure_error = final_moment - expected_final;
    require_close_v0(final_moment, expected_final, "H target-only solid moment")?;
    let final_elevation_component_hash = elevation_hash_v0(&identity, &elevation)?;
    let noncanonical_control_binding_hash = fnv1a64(&fixed_bytes(&ThinHControlPreimageV0 {
        domain: THIN_H_CONTROL_DOMAIN_V0,
        identity: &identity,
        config_hash,
        final_elevation_component_hash,
        gross_hold_restoration_km3: gross_hold,
        initial_elevation_volume_moment_km3: initial_moment,
        final_elevation_volume_moment_km3: final_moment,
        closure_error_km3: closure_error,
        checkpoints: &checkpoints,
        final_elevation_km: &elevation,
    })?);

    Ok(OpportunityControlExecutionV0 {
        observation: ThinHOpportunityControlV0 {
            identity,
            config_hash,
            noncanonical_control_binding_hash,
            final_elevation_component_hash,
            gross_hold_restoration_km3: gross_hold,
            initial_elevation_volume_moment_km3: initial_moment,
            final_elevation_volume_moment_km3: final_moment,
            closure_error_km3: closure_error,
            checkpoints,
        },
    })
}

fn registered_control_config_hash_v0(
    input: &LinkedResolutionInputV0,
    identity: &OrganizationArtifactIdentityV0,
    cumulative_displacement_component_hash: u64,
) -> Result<u64, ThinHOwnerErrorV0> {
    registered_h_config_hash_v0(
        input,
        identity,
        None,
        HProcessModeV0::TargetOnly,
        cumulative_displacement_component_hash,
    )
}

fn registered_base_config_hash_v0(
    input: &LinkedResolutionInputV0,
    identity: &OrganizationArtifactIdentityV0,
    noncanonical_control_binding_hash: u64,
    cumulative_displacement_component_hash: u64,
) -> Result<u64, ThinHOwnerErrorV0> {
    registered_h_config_hash_v0(
        input,
        identity,
        Some(noncanonical_control_binding_hash),
        HProcessModeV0::HoldAndCarve,
        cumulative_displacement_component_hash,
    )
}

fn registered_h_config_hash_v0(
    input: &LinkedResolutionInputV0,
    identity: &OrganizationArtifactIdentityV0,
    control_binding: Option<u64>,
    mode: HProcessModeV0,
    cumulative_displacement_component_hash: u64,
) -> Result<u64, ThinHOwnerErrorV0> {
    let active = mode == HProcessModeV0::HoldAndCarve;
    let active_process = active.then(|| ActiveProcessConfigWireV0 {
        routing: RoutingConfigWireV0 {
            depression_policy: RoutingDepressionPolicyV0::PriorityVirtualSurfaceNoBedrockWrite,
            flow_partition: FlowPartitionPolicyV0::MfdSlope,
            runoff_component_hash: input.component_hashes.local_runoff_hash,
        },
        denudation: EffectiveDenudationConfigWireV0 {
            k_km_inverse: K_KM_INVERSE,
            discharge_exponent_m: 1.0,
            slope_exponent_n: 1.0,
            support_policy: DischargeSupportPolicyV0::UnfilteredC0Physical,
        },
        hillslope: LinearHillslopeConfigWireV0 {
            diffusivity_km2_myr: HILLSLOPE_DIFFUSIVITY_KM2_MYR,
            timestep_safety: HILLSLOPE_TIMESTEP_SAFETY,
            boundary_policy: HillslopeBoundaryPolicyV0::LinearDirichletOnOpenFacesClosedElsewhere,
        },
        accuracy: ActiveProcessAccuracyConfigWireV0 {
            maximum_denudation_depth_km: MAXIMUM_DENUDATION_DEPTH_KM,
            denudation_slope_courant: DENUDATION_SLOPE_COURANT,
        },
    });
    let config = OrganizationArmConfigV0 {
        schema_version: ORGANIZATION_ARM_CONFIG_SCHEMA_VERSION_V0.into(),
        hash_version: ORGANIZATION_ARTIFACT_HASH_VERSION_V0.into(),
        identity: identity.clone(),
        predecessors: OrganizationPredecessorsV0 {
            opportunity_control_result_hash: control_binding,
            g_reference_4km: None,
        },
        payload: OrganizationArmConfigPayloadV0::H(HConfigWireV0 {
            config_schema: ORGANIZATION_H_CONFIG_SCHEMA_VERSION_V0.into(),
            process_mode: mode,
            pass_count: PASS_COUNT,
            checkpoint_passes: CHECKPOINT_PASSES.to_vec(),
            schedule_policy: HSchedulePolicyV0::LinkedEpisodeCumulativeActivityFraction,
            endpoint_policy: HEndpointPolicyV0::ExactZeroAndOneEndpoints,
            schedule_horizon_myr: SCHEDULE_HORIZON_MYR,
            activity_integral_myr: ACTIVITY_INTEGRAL_MYR,
            cumulative_displacement_component_hash,
            operator_exposure_per_pass_myr: if active {
                OPERATOR_EXPOSURE_PER_PASS_MYR
            } else {
                0.0
            },
            adaptive_integration: active.then_some(AdaptiveIntegrationConfigWireV0 {
                maximum_uplift_depth_km: None,
                minimum_dt_myr: MINIMUM_DT_MYR,
                maximum_adaptive_attempts: MAXIMUM_ADAPTIVE_ATTEMPTS,
                requested_maximum_dt_myr: REQUESTED_MAXIMUM_DT_MYR,
            }),
            split_order: active.then_some(SplitOrderV0::HoldThenRouteDenudeThenHillslope),
            active_process,
        }),
        derived_config_hash: 0,
    };
    organization_arm_config_hash_v0(&config)
        .map_err(|error| fail(format!("registered H configuration rejected: {error}")))
}

fn identity_v0(
    forcing_binding: HForcingBindingV0,
    purpose: OrganizationRunPurposeV0,
) -> OrganizationArtifactIdentityV0 {
    OrganizationArtifactIdentityV0 {
        input_bundle_hash: forcing_binding.input_bundle_hash,
        input_resolution_hash: forcing_binding.input_resolution_hash,
        nominal_spacing_km: TARGET_SPACING_KM,
        arm: OrganizationArmV0::H,
        purpose,
    }
}

fn accepted_4km_input_v0(
    bundle: &LinkedSharedInputBundleV0,
) -> Result<&LinkedResolutionInputV0, ThinHOwnerErrorV0> {
    bundle
        .resolutions
        .iter()
        .find(|input| input.nominal_spacing_km.to_bits() == TARGET_SPACING_KM.to_bits())
        .ok_or_else(|| fail("accepted bundle has no exact 4 km resolution"))
}

fn accepted_forcing_binding_v0(
    bundle: &LinkedSharedInputBundleV0,
    input: &LinkedResolutionInputV0,
) -> HForcingBindingV0 {
    HForcingBindingV0 {
        input_bundle_hash: bundle.derived_bundle_hash,
        input_resolution_hash: input.derived_resolution_hash,
        cumulative_displacement_component_hash: input
            .component_hashes
            .cumulative_rock_displacement_hash,
    }
}

fn validate_experimental_binding_v0(
    bundle: &LinkedSharedInputBundleV0,
    input: &LinkedResolutionInputV0,
    binding: ThinHExperimentalForcingBindingV0,
) -> Result<(), ThinHOwnerErrorV0> {
    require(
        binding.synthetic_input_bundle_hash != bundle.derived_bundle_hash
            || binding.synthetic_input_resolution_hash != input.derived_resolution_hash,
        "experimental H forcing reuses the accepted linked-input identity",
    )?;
    require(
        binding.cumulative_displacement_component_hash
            != input.component_hashes.cumulative_rock_displacement_hash,
        "experimental H forcing reuses the accepted displacement component hash",
    )
}

fn derived_experimental_binding_v0(
    bundle: &LinkedSharedInputBundleV0,
    input: &LinkedResolutionInputV0,
    cumulative_rock_displacement_km: &[f64],
) -> Result<ThinHExperimentalForcingBindingV0, ThinHOwnerErrorV0> {
    let cumulative_displacement_component_hash = fnv1a64(&fixed_bytes(&(
        EXPERIMENTAL_DISPLACEMENT_DOMAIN_V0,
        cumulative_rock_displacement_km,
    ))?);
    let identity_payload = (
        bundle.derived_bundle_hash,
        input.derived_resolution_hash,
        cumulative_displacement_component_hash,
    );
    Ok(ThinHExperimentalForcingBindingV0 {
        synthetic_input_bundle_hash: fnv1a64(&fixed_bytes(&(
            EXPERIMENTAL_BUNDLE_DOMAIN_V0,
            identity_payload,
        ))?),
        synthetic_input_resolution_hash: fnv1a64(&fixed_bytes(&(
            EXPERIMENTAL_RESOLUTION_DOMAIN_V0,
            identity_payload,
        ))?),
        cumulative_displacement_component_hash,
    })
}

fn validate_target_displacement_v0(
    cell_count: usize,
    cumulative_rock_displacement_km: &[f64],
) -> Result<(), ThinHOwnerErrorV0> {
    require(
        cumulative_rock_displacement_km.len() == cell_count,
        "H target displacement length mismatch",
    )?;
    for &value in cumulative_rock_displacement_km {
        require_nonnegative(value, "target cumulative displacement")?;
    }
    Ok(())
}

fn validate_h_input_v0(input: &LinkedResolutionInputV0) -> Result<(), ThinHOwnerErrorV0> {
    input
        .mesh
        .validate()
        .map_err(|error| fail(format!("invalid stored mesh: {error}")))?;
    require(
        input.nominal_spacing_km.to_bits() == TARGET_SPACING_KM.to_bits(),
        "H probe requires exact 4 km input",
    )?;
    let n = input.mesh.cell_count();
    require(
        input.initial_elevation_km.len() == n
            && input.cumulative_rock_displacement_km.len() == n
            && input.local_runoff_supply_km3_myr.len() == n,
        "H input field length mismatch",
    )?;
    for cell in 0..n {
        require_finite(input.initial_elevation_km[cell], "initial elevation")?;
        require_nonnegative(
            input.cumulative_rock_displacement_km[cell],
            "cumulative displacement",
        )?;
        require_nonnegative(
            input.local_runoff_supply_km3_myr[cell],
            "local runoff supply",
        )?;
    }
    Ok(())
}

fn checkpoint_v0(
    identity: &OrganizationArtifactIdentityV0,
    pass: u32,
    schedule_coordinate_myr: f64,
    completed_operator_exposure_myr: f64,
    accepted_step_count: u64,
    elevation: &Vec<f64>,
) -> Result<ThinHCheckpointV0, ThinHOwnerErrorV0> {
    let hash = elevation_hash_v0(identity, elevation)?;
    Ok(ThinHCheckpointV0 {
        pass,
        schedule_coordinate_myr,
        completed_operator_exposure_myr,
        accepted_step_count,
        physical_elevation_component_hash: hash,
    })
}

fn elevation_hash_v0(
    identity: &OrganizationArtifactIdentityV0,
    elevation: &Vec<f64>,
) -> Result<u64, ThinHOwnerErrorV0> {
    Ok(fnv1a64(&fixed_bytes(&ElevationArrayPreimageV0 {
        domain: ELEVATION_ARRAY_DOMAIN_V0,
        identity,
        elevation_km: elevation,
    })?))
}

fn shared_elevation_moment_v0(
    input: &LinkedResolutionInputV0,
    elevation: &[f64],
) -> Result<f64, ThinHOwnerErrorV0> {
    elevation_moment_v0(&input.mesh, elevation)
        .map_err(|error| fail(format!("H elevation-volume reduction failed: {error}")))
}

fn zero_portal_volumes_v0(input: &LinkedResolutionInputV0) -> Vec<ThinHPortalVolumeV0> {
    let mut portal_ids: Vec<u32> = input
        .mesh
        .outlet_portals
        .iter()
        .map(|portal| portal.id.0)
        .collect();
    portal_ids.sort_unstable();
    portal_ids
        .into_iter()
        .map(|portal_id| ThinHPortalVolumeV0 {
            portal_id,
            volume_km3: 0.0,
        })
        .collect()
}

fn finite_min_max_v0(values: &[f64]) -> Result<(f64, f64), ThinHOwnerErrorV0> {
    let (&first, rest) = values
        .split_first()
        .ok_or_else(|| fail("H final elevation is empty"))?;
    require_finite(first, "H final elevation")?;
    let mut minimum = first;
    let mut maximum = first;
    for &value in rest {
        require_finite(value, "H final elevation")?;
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    Ok((minimum, maximum))
}

fn pass_endpoint_v0(pass: u32) -> Result<f64, ThinHOwnerErrorV0> {
    require((1..=PASS_COUNT).contains(&pass), "invalid H pass endpoint")?;
    Ok(if pass == PASS_COUNT {
        SCHEDULE_HORIZON_MYR
    } else {
        f64::from(pass) * OPERATOR_EXPOSURE_PER_PASS_MYR
    })
}

fn fixed_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, ThinHOwnerErrorV0> {
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

fn require_close_v0(actual: f64, expected: f64, name: &str) -> Result<(), ThinHOwnerErrorV0> {
    require_close_with_tolerance_v0(actual, expected, 1.0e-8, 5.0e-12, name)
}

fn require_close_with_tolerance_v0(
    actual: f64,
    expected: f64,
    absolute: f64,
    relative: f64,
    name: &str,
) -> Result<(), ThinHOwnerErrorV0> {
    require_finite(actual, "H close actual")?;
    require_finite(expected, "H close expected")?;
    require_nonnegative(absolute, "H close absolute tolerance")?;
    require_nonnegative(relative, "H close relative tolerance")?;
    let difference = (actual - expected).abs();
    let limit = absolute + (relative * actual.abs().max(expected.abs()));
    require(
        difference <= limit,
        &format!("{name} does not close: actual={actual:.17e} expected={expected:.17e}"),
    )
}

fn require_finite(value: f64, name: &str) -> Result<(), ThinHOwnerErrorV0> {
    require(value.is_finite(), &format!("non-finite {name}"))
}

fn require_nonnegative(value: f64, name: &str) -> Result<(), ThinHOwnerErrorV0> {
    require_finite(value, name)?;
    require(value >= 0.0, &format!("negative {name}"))?;
    require(
        value != 0.0 || value.to_bits() == 0.0f64.to_bits(),
        &format!("noncanonical zero in {name}"),
    )
}

fn require(condition: bool, message: &str) -> Result<(), ThinHOwnerErrorV0> {
    if condition {
        Ok(())
    } else {
        Err(fail(message))
    }
}

fn fail(message: impl Into<String>) -> ThinHOwnerErrorV0 {
    ThinHOwnerErrorV0(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::landscape::build_linked_shared_input_bundle_v0;

    fn empty_process_ledger() -> ThinHProcessLedgerV0 {
        ThinHProcessLedgerV0 {
            effective_denudation_export_km3: 0.0,
            hillslope_portal_transfers_km3: Vec::new(),
            total_hillslope_portal_transfer_km3: 0.0,
            water: ThinHIntegratedWaterLedgerV0 {
                supplied_volume_km3: 0.0,
                portal_outflow_volume_km3: Vec::new(),
                total_portal_outflow_volume_km3: 0.0,
                unresolved_sink_volume_km3: 0.0,
                balance_error_km3: 0.0,
            },
        }
    }

    fn committed(elevation: Vec<f64>) -> HCommittedStateV0 {
        HCommittedStateV0 {
            physical_elevation_km: elevation,
            completed_operator_exposure_myr: 0.0,
            gross_hold_restoration_km3: 0.0,
            process: empty_process_ledger(),
            accepted_step_count: 0,
            total_candidate_attempt_count: 0,
            maximum_attempts_for_one_step: 0,
            minimum_accepted_dt_myr: None,
            maximum_accepted_dt_myr: None,
            limiter_histogram: ThinHLimiterHistogramV0::default(),
            maximum_effective_denudation_rate_km_myr: 0.0,
            maximum_linear_hillslope_abs_grade: 0.0,
            maximum_unresolved_specific_discharge_cell_count: 0,
        }
    }

    #[test]
    fn cumulative_activity_matches_registered_witnesses_and_symmetry() {
        let witnesses: [(f64, f64); 9] = [
            (0.0, 0.0),
            (0.125, 0.0234375),
            (0.25, 0.125),
            (3.0, 2.875),
            (5.75, 5.625),
            (5.875, 5.7265625),
            (6.0, 5.75),
            (8.0, 5.75),
            (10.0, 5.75),
        ];
        for (time, expected) in witnesses {
            assert_eq!(
                cumulative_linked_activity_myr_v0(time).unwrap().to_bits(),
                expected.to_bits()
            );
        }
        for time in [0.125, 0.25, 1.5, 3.0, 5.75, 5.875] {
            assert_eq!(
                cumulative_linked_activity_myr_v0(time).unwrap()
                    + cumulative_linked_activity_myr_v0(6.0 - time).unwrap(),
                ACTIVITY_INTEGRAL_MYR
            );
        }
    }

    #[test]
    fn base_progress_is_monotone_and_uses_exact_endpoints() {
        let mut previous = -1.0;
        for pass in 0..=PASS_COUNT {
            let (_, progress) = target_progress_v0(pass, PASS_COUNT).unwrap();
            assert!(progress >= previous);
            previous = progress;
        }
        assert_eq!(
            target_progress_v0(0, PASS_COUNT).unwrap().1.to_bits(),
            0.0f64.to_bits()
        );
        assert_eq!(target_progress_v0(PASS_COUNT, PASS_COUNT).unwrap().1, 1.0);
        assert_eq!(target_progress_v0(50, PASS_COUNT).unwrap().1, 19.0 / 46.0);
        assert_eq!(target_progress_v0(120, PASS_COUNT).unwrap().1, 1.0);
        assert_eq!(
            pass_endpoint_v0(PASS_COUNT).unwrap().to_bits(),
            10.0f64.to_bits()
        );
    }

    #[test]
    fn failed_carve_rolls_back_hold_surface_and_ledgers() {
        let mut state = committed(vec![0.0, 1.0]);
        let before = state.clone();
        let result = execute_transactional_pass_v0(
            &mut state,
            &[0.0, 1.0],
            &[2.0, 3.0],
            &[5.0, 7.0],
            0.5,
            |scratch| {
                scratch.physical_elevation_km[0] = -99.0;
                scratch.accepted_step_count = 12;
                scratch.process.effective_denudation_export_km3 = 123.0;
                Err(fail("injected carve failure"))
            },
        );
        assert!(result.is_err());
        assert_eq!(state, before);
    }

    #[test]
    #[ignore = "builds and validates the complete linked 8/4/2 bundle"]
    fn accepted_target_only_control_is_exact_and_repeatable() {
        let bundle = build_linked_shared_input_bundle_v0().unwrap();
        validate_linked_shared_input_bundle_v0(&bundle).unwrap();
        let input = bundle
            .resolutions
            .iter()
            .find(|input| input.nominal_spacing_km == 4.0)
            .unwrap();
        validate_h_input_v0(input).unwrap();
        let binding = accepted_forcing_binding_v0(&bundle, input);
        let first =
            run_opportunity_control_v0(input, &input.cumulative_rock_displacement_km, binding)
                .unwrap();
        let second =
            run_opportunity_control_v0(input, &input.cumulative_rock_displacement_km, binding)
                .unwrap();
        assert_eq!(
            fixed_bytes(&first.observation).unwrap(),
            fixed_bytes(&second.observation).unwrap()
        );
        assert_eq!(
            first.observation.identity.input_bundle_hash,
            bundle.derived_bundle_hash
        );
        assert_eq!(
            first.observation.identity.input_resolution_hash,
            input.derived_resolution_hash
        );
        assert_eq!(
            first.observation.config_hash,
            registered_control_config_hash_v0(
                input,
                &first.observation.identity,
                input.component_hashes.cumulative_rock_displacement_hash,
            )
            .unwrap()
        );
    }

    #[test]
    fn experimental_target_validation_rejects_wrong_shape_and_nonfinite_values() {
        assert!(validate_target_displacement_v0(2, &[0.0]).is_err());
        let mut invalid = vec![0.0; 2];
        invalid[0] = f64::NAN;
        assert!(validate_target_displacement_v0(2, &invalid).is_err());
    }

    #[test]
    #[ignore = "rebuilds the accepted bundle and executes two complete 4 km H runs"]
    fn accepted_linked_input_h_probe_is_bit_deterministic() {
        let bundle = crate::world::landscape::build_linked_shared_input_bundle_v0().unwrap();
        let result = run_repeated_thin_h_4km_v0(&bundle).unwrap();
        assert_eq!(result.completion.completed_pass_count, PASS_COUNT);
        assert_eq!(
            result.completion.completed_operator_exposure_myr.to_bits(),
            10.0f64.to_bits()
        );
        assert_eq!(result.checkpoints.len(), CHECKPOINT_PASSES.len());
        assert_eq!(
            result
                .checkpoints
                .last()
                .unwrap()
                .physical_elevation_component_hash,
            result.final_elevation_component_hash
        );
        let input = accepted_4km_input_v0(&bundle).unwrap();
        let binding = accepted_forcing_binding_v0(&bundle, input);
        let expected_identity = identity_v0(binding, OrganizationRunPurposeV0::Base);
        assert_eq!(result.identity, expected_identity);
        assert_eq!(
            result.config_hash,
            registered_base_config_hash_v0(
                input,
                &expected_identity,
                result.control.noncanonical_control_binding_hash,
                input.component_hashes.cumulative_rock_displacement_hash,
            )
            .unwrap()
        );
    }
}
