//! Exact 4 km C owner for the disposable organization engineering probe.
//!
//! This module deliberately does not implement campaign artifacts or publication.
//! It executes the registered uplift-only control and coevolving base algorithms,
//! retains the physical surface and essential numerical witnesses, and labels the
//! control binding as noncanonical so it cannot be mistaken for a promoted result.

use super::organization_artifact::{
    organization_arm_config_hash_v0, ActiveProcessAccuracyConfigWireV0, ActiveProcessConfigWireV0,
    AdaptiveIntegrationConfigWireV0, CConfigWireV0, CEndpointClippingPolicyV0,
    CForcingSamplingPolicyV0, CProcessModeV0, CVerticalRateAuthorityV0, DischargeSupportPolicyV0,
    EffectiveDenudationConfigWireV0, FlowPartitionPolicyV0, HillslopeBoundaryPolicyV0,
    LinearHillslopeConfigWireV0, OrganizationArmConfigPayloadV0, OrganizationArmConfigV0,
    OrganizationArmV0, OrganizationArtifactIdentityV0, OrganizationPredecessorsV0,
    OrganizationRunPurposeV0, RoutingConfigWireV0, RoutingDepressionPolicyV0, SplitOrderV0,
    ORGANIZATION_ARM_CONFIG_SCHEMA_VERSION_V0, ORGANIZATION_ARTIFACT_HASH_VERSION_V0,
    ORGANIZATION_C_CONFIG_SCHEMA_VERSION_V0,
};
use super::organization_owner::{ThinOwnerErrorV0, THIN_OWNER_PROFILE_V0};
use super::organization_process::{
    attempt_active_process_v0, elevation_moment_v0, fresh_routing_diagnostics_v0, ProcessAttemptV0,
    ProcessLimiterV0, ProcessStepV0, ProcessWaterRateV0,
};
use super::{
    validate_linked_shared_input_bundle_v0, DeformationEvaluator, DeformationFrame,
    LinkedResolutionInputV0, LinkedSharedInputBundleV0, OutletPortalId,
};
use bincode::Options;
use serde::{Deserialize, Serialize};

pub const THIN_C_4KM_SCHEMA_VERSION_V0: &str = "orogen-owner-thin-c-4km-probe-v0";

const TARGET_SPACING_KM: f64 = 4.0;
const ACTIVITY_INTEGRAL_MYR: f64 = 5.75;
const REQUESTED_MAXIMUM_DT_MYR: f64 = 0.01;
const MINIMUM_DT_MYR: f64 = 1.0e-8;
const MAXIMUM_ADAPTIVE_ATTEMPTS: u32 = 16;
const MAXIMUM_ACCEPTED_STEP_COUNT: u64 = 100_000;
const MAXIMUM_UPLIFT_DEPTH_KM: f64 = 0.02;
const MAXIMUM_DENUDATION_DEPTH_KM: f64 = 0.02;
const DENUDATION_SLOPE_COURANT: f64 = 0.25;
const DENUDATION_K_KM_INVERSE: f64 = 1.0e-4;
const HILLSLOPE_DIFFUSIVITY_KM2_MYR: f64 = 0.1;
const HILLSLOPE_TIMESTEP_SAFETY: f64 = 0.4;
const CHECKPOINT_TIMES_MYR: [f64; 4] = [0.0, 3.0, 6.0, 10.0];

const SOLID_ABSOLUTE_TOLERANCE_KM3: f64 = 1.0e-8;
const SOLID_RELATIVE_TOLERANCE: f64 = 5.0e-12;
const WATER_RATE_ABSOLUTE_TOLERANCE_KM3_MYR: f64 = 1.0e-6;
const WATER_VOLUME_ABSOLUTE_TOLERANCE_KM3: f64 = 1.0e-8;
const WATER_RELATIVE_TOLERANCE: f64 = 5.0e-12;
const CONTROL_VOLUME_ABSOLUTE_TOLERANCE_KM3: f64 = 1.0e-6;
const CONTROL_VOLUME_RELATIVE_TOLERANCE: f64 = 5.0e-7;
const CONTROL_CELL_TOLERANCE_KM: f64 = 2.0e-5;

const ELEVATION_ARRAY_DOMAIN_V0: &str = "orogen-organization-v0/elevation-array";
const THIN_CONTROL_BINDING_DOMAIN_V0: &str = "orogen-owner-thin-v0/c-control-binding";
const THIN_CHECKPOINT_DOMAIN_V0: &str = "orogen-owner-thin-v0/c-checkpoint";
const EXPERIMENTAL_STENCILS_DOMAIN_V0: &str = "orogen-owner-thin-v0/experimental-compiled-stencils";
const EXPERIMENTAL_EVALUATOR_DOMAIN_V0: &str =
    "orogen-owner-thin-v0/experimental-evaluator-chronology";
const EXPERIMENTAL_DISPLACEMENT_DOMAIN_V0: &str =
    "orogen-owner-thin-v0/experimental-cumulative-displacement";
const EXPERIMENTAL_BUNDLE_DOMAIN_V0: &str = "orogen-owner-thin-v0/experimental-input-bundle";
const EXPERIMENTAL_RESOLUTION_DOMAIN_V0: &str =
    "orogen-owner-thin-v0/experimental-input-resolution";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThinCStepLimiterV0 {
    Requested,
    UpliftAccuracy,
    DenudationAccuracy,
    DenudationSlopeCourant,
    HillslopeStability,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ThinCLimiterHistogramV0 {
    pub requested: u64,
    pub uplift_accuracy: u64,
    pub denudation_accuracy: u64,
    pub denudation_slope_courant: u64,
    pub hillslope_stability: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinCCompletionV0 {
    pub reached_time_myr: f64,
    pub final_revision: u64,
    pub accepted_step_count: u64,
    pub total_candidate_attempt_count: u64,
    pub maximum_attempts_for_one_step: u32,
    pub minimum_accepted_dt_myr: f64,
    pub maximum_accepted_dt_myr: f64,
    pub limiter_histogram: ThinCLimiterHistogramV0,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinCPortalVolumeV0 {
    pub portal_id: u32,
    pub volume_km3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinCIntegratedWaterLedgerV0 {
    pub supplied_volume_km3: f64,
    pub portal_outflow_volume_km3: Vec<ThinCPortalVolumeV0>,
    pub total_portal_outflow_volume_km3: f64,
    pub unresolved_sink_volume_km3: f64,
    pub balance_error_km3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinCSolidLedgerV0 {
    pub initial_elevation_volume_moment_km3: f64,
    pub forcing_interval_rock_uplift_moment_km3: f64,
    pub relaxation_interval_rock_uplift_moment_km3: f64,
    pub total_rock_uplift_moment_km3: f64,
    pub effective_denudation_export_km3: f64,
    pub hillslope_portal_transfers_km3: Vec<ThinCPortalVolumeV0>,
    pub total_hillslope_portal_transfer_km3: f64,
    pub final_elevation_volume_moment_km3: f64,
    pub closure_error_km3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinCProcessSummaryV0 {
    pub maximum_denudation_rate_km_myr: f64,
    pub maximum_linear_hillslope_abs_grade: f64,
    pub maximum_unresolved_specific_discharge_cells: u64,
    pub final_routing_hash: u64,
    pub final_routing_total_supply_km3_myr: f64,
    pub final_routing_total_portal_outflow_km3_myr: f64,
    pub final_routing_unresolved_sink_km3_myr: f64,
    pub final_routing_water_balance_error_km3_myr: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinCCheckpointV0 {
    pub time_myr: f64,
    pub revision: u64,
    pub elevation_component_hash: u64,
    pub diagnostic_checkpoint_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinCOpportunityAuditV0 {
    pub final_elevation_component_hash: u64,
    pub initial_elevation_volume_moment_km3: f64,
    pub integrated_rock_uplift_moment_km3: f64,
    pub forcing_interval_rock_uplift_moment_km3: f64,
    pub relaxation_interval_rock_uplift_moment_km3: f64,
    pub signed_displacement_volume_error_km3: f64,
    pub maximum_displacement_error_km: f64,
    pub area_weighted_l1_displacement_error_km: f64,
    pub area_weighted_rms_displacement_error_km: f64,
    pub final_elevation_volume_moment_km3: f64,
    pub solid_closure_error_km3: f64,
    pub completion: ThinCCompletionV0,
    pub checkpoints: Vec<ThinCCheckpointV0>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinC4KmObservationV0 {
    pub schema_version: String,
    pub profile: String,
    pub control_identity: OrganizationArtifactIdentityV0,
    pub control_config_hash: u64,
    /// A probe-local binding over the admitted control. It is not and cannot be
    /// relabelled as an organization campaign result hash.
    pub noncanonical_control_binding_hash: u64,
    pub opportunity: ThinCOpportunityAuditV0,
    pub base_identity: OrganizationArtifactIdentityV0,
    pub base_config_hash: u64,
    pub final_elevation_component_hash: u64,
    pub final_min_elevation_km: f64,
    pub final_max_elevation_km: f64,
    pub completion: ThinCCompletionV0,
    pub solid: ThinCSolidLedgerV0,
    pub water: ThinCIntegratedWaterLedgerV0,
    pub process: ThinCProcessSummaryV0,
    pub checkpoints: Vec<ThinCCheckpointV0>,
    pub final_elevation_km: Vec<f64>,
}

/// Explicit provenance for a noncanonical C forcing-response experiment.
///
/// Public runners recompute every field from the supplied evaluator and target
/// and reject caller-selected provenance values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThinCExperimentalForcingBindingV0 {
    pub synthetic_input_bundle_hash: u64,
    pub synthetic_input_resolution_hash: u64,
    pub compiled_stencils_component_hash: u64,
    pub frame_witnesses_component_hash: u64,
    pub cumulative_displacement_component_hash: u64,
}

/// Derive the only binding accepted for this bundle, evaluator and target.
pub fn derive_thin_c_experimental_forcing_binding_4km_v0(
    bundle: &LinkedSharedInputBundleV0,
    evaluator: &DeformationEvaluator,
    cumulative_rock_displacement_km: &[f64],
) -> Result<ThinCExperimentalForcingBindingV0, ThinOwnerErrorV0> {
    validate_linked_shared_input_bundle_v0(bundle)
        .map_err(|error| fail(format!("linked input validation failed: {error}")))?;
    let input = accepted_4km_input(bundle)?;
    validate_target_displacement(input.mesh.cell_count(), cumulative_rock_displacement_km)?;
    validate_evaluator(input, evaluator)?;
    derived_experimental_binding(bundle, input, evaluator, cumulative_rock_displacement_km)
}

#[derive(Debug, Clone, Copy)]
struct CForcingBindingV0 {
    input_bundle_hash: u64,
    input_resolution_hash: u64,
    compiled_stencils_component_hash: u64,
    frame_witnesses_component_hash: u64,
    cumulative_displacement_component_hash: u64,
}

#[derive(Debug, Clone)]
struct ControlRunV0 {
    identity: OrganizationArtifactIdentityV0,
    config_hash: u64,
    binding_hash: u64,
    audit: ThinCOpportunityAuditV0,
}

#[derive(Debug, Clone)]
struct DriverStateV0 {
    time_myr: f64,
    revision: u64,
    elevation_km: Vec<f64>,
    completion: CompletionAccumulatorV0,
    checkpoints: Vec<ThinCCheckpointV0>,
}

#[derive(Debug, Clone)]
struct CompletionAccumulatorV0 {
    accepted_steps: u64,
    attempts: u64,
    max_attempts: u32,
    min_dt: Option<f64>,
    max_dt: Option<f64>,
    histogram: ThinCLimiterHistogramV0,
}

impl Default for CompletionAccumulatorV0 {
    fn default() -> Self {
        Self {
            accepted_steps: 0,
            attempts: 0,
            max_attempts: 0,
            min_dt: None,
            max_dt: None,
            histogram: ThinCLimiterHistogramV0::default(),
        }
    }
}

#[derive(Debug, Clone)]
struct BaseAccumulatorV0 {
    forcing_uplift: f64,
    relaxation_uplift: f64,
    denudation_export: f64,
    hillslope_by_portal: Vec<ThinCPortalVolumeV0>,
    hillslope_total: f64,
    water: ThinCIntegratedWaterLedgerV0,
    max_denudation_rate: f64,
    max_hillslope_grade: f64,
    max_unresolved_cells: u64,
}

#[derive(Serialize)]
struct ElevationArrayPreimageV0<'a> {
    domain: &'static str,
    identity: &'a OrganizationArtifactIdentityV0,
    elevation_km: &'a Vec<f64>,
}

#[derive(Serialize)]
struct ThinControlBindingPreimageV0<'a> {
    domain: &'static str,
    identity: &'a OrganizationArtifactIdentityV0,
    config_hash: u64,
    final_elevation_component_hash: u64,
    integrated_rock_uplift_moment_km3: f64,
    initial_elevation_volume_moment_km3: f64,
    final_elevation_volume_moment_km3: f64,
    solid_closure_error_km3: f64,
    signed_displacement_volume_error_km3: f64,
    maximum_displacement_error_km: f64,
}

#[derive(Serialize)]
struct ThinCheckpointPreimageV0<'a> {
    domain: &'static str,
    identity: &'a OrganizationArtifactIdentityV0,
    time_myr: f64,
    revision: u64,
    elevation_km: &'a Vec<f64>,
}

pub fn run_thin_c_4km_v0(
    bundle: &LinkedSharedInputBundleV0,
) -> Result<ThinC4KmObservationV0, ThinOwnerErrorV0> {
    validate_linked_shared_input_bundle_v0(bundle)
        .map_err(|error| fail(format!("linked input validation failed: {error}")))?;
    let input = accepted_4km_input(bundle)?;
    let evaluator = accepted_evaluator(bundle, input)?;
    run_validated_thin_c_4km_v0(
        bundle,
        input,
        &evaluator,
        &input.cumulative_rock_displacement_km,
        accepted_forcing_binding(bundle, input),
    )
}

/// Run the exact C owner against an explicit replacement forcing evaluator and
/// cumulative target while retaining the validated accepted mesh, initial
/// surface, runoff, portals, time domain, and process configuration.
///
/// The target is used by the uplift-only opportunity audit; the evaluator is
/// the rate authority for both that control and the coevolving base. This is
/// noncanonical engineering evidence and requires a distinct synthetic input
/// identity plus explicit component bindings.
pub fn run_thin_c_experimental_forcing_4km_v0(
    bundle: &LinkedSharedInputBundleV0,
    evaluator: &DeformationEvaluator,
    cumulative_rock_displacement_km: &[f64],
    binding: ThinCExperimentalForcingBindingV0,
) -> Result<ThinC4KmObservationV0, ThinOwnerErrorV0> {
    validate_linked_shared_input_bundle_v0(bundle)
        .map_err(|error| fail(format!("linked input validation failed: {error}")))?;
    let input = accepted_4km_input(bundle)?;
    validate_target_displacement(input.mesh.cell_count(), cumulative_rock_displacement_km)?;
    validate_evaluator(input, evaluator)?;
    let derived =
        derived_experimental_binding(bundle, input, evaluator, cumulative_rock_displacement_km)?;
    require(
        binding == derived,
        "experimental C forcing binding does not match supplied evaluator and displacement",
    )?;
    validate_experimental_binding(bundle, input, binding)?;
    validate_experimental_forcing_target(
        input,
        evaluator,
        cumulative_rock_displacement_km,
        bundle.declaration.analytic_rock_volume_km3,
    )?;
    run_validated_thin_c_4km_v0(
        bundle,
        input,
        evaluator,
        cumulative_rock_displacement_km,
        CForcingBindingV0 {
            input_bundle_hash: binding.synthetic_input_bundle_hash,
            input_resolution_hash: binding.synthetic_input_resolution_hash,
            compiled_stencils_component_hash: binding.compiled_stencils_component_hash,
            frame_witnesses_component_hash: binding.frame_witnesses_component_hash,
            cumulative_displacement_component_hash: binding.cumulative_displacement_component_hash,
        },
    )
}

pub fn run_repeated_thin_c_4km_v0(
    bundle: &LinkedSharedInputBundleV0,
) -> Result<ThinC4KmObservationV0, ThinOwnerErrorV0> {
    validate_linked_shared_input_bundle_v0(bundle)
        .map_err(|error| fail(format!("linked input validation failed: {error}")))?;
    let input = accepted_4km_input(bundle)?;
    let evaluator = accepted_evaluator(bundle, input)?;
    let binding = accepted_forcing_binding(bundle, input);
    let first = run_validated_thin_c_4km_v0(
        bundle,
        input,
        &evaluator,
        &input.cumulative_rock_displacement_km,
        binding,
    )?;
    let second = run_validated_thin_c_4km_v0(
        bundle,
        input,
        &evaluator,
        &input.cumulative_rock_displacement_km,
        binding,
    )?;
    require(
        fixed_bytes(&first)? == fixed_bytes(&second)?,
        "repeated thin C probe differs at bit-level comparison",
    )?;
    Ok(first)
}

fn run_validated_thin_c_4km_v0(
    bundle: &LinkedSharedInputBundleV0,
    input: &LinkedResolutionInputV0,
    evaluator: &DeformationEvaluator,
    cumulative_rock_displacement_km: &[f64],
    forcing_binding: CForcingBindingV0,
) -> Result<ThinC4KmObservationV0, ThinOwnerErrorV0> {
    validate_c_input(input)?;
    validate_target_displacement(input.mesh.cell_count(), cumulative_rock_displacement_km)?;
    validate_evaluator(input, evaluator)?;
    let control = run_control(
        input,
        evaluator,
        cumulative_rock_displacement_km,
        bundle.declaration.analytic_rock_volume_km3,
        forcing_binding,
    )?;
    run_base(input, evaluator, forcing_binding, control)
}

fn validate_c_input(input: &LinkedResolutionInputV0) -> Result<(), ThinOwnerErrorV0> {
    let n = input.mesh.cell_count();
    require(
        input.initial_elevation_km.len() == n,
        "C initial elevation length mismatch",
    )?;
    require(
        input.local_runoff_supply_km3_myr.len() == n,
        "C stored runoff length mismatch",
    )?;
    require(
        input.cumulative_rock_displacement_km.len() == n,
        "C displacement length mismatch",
    )?;
    require(
        input
            .initial_elevation_km
            .iter()
            .all(|value| value.is_finite())
            && input
                .local_runoff_supply_km3_myr
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
            && input
                .cumulative_rock_displacement_km
                .iter()
                .all(|value| value.is_finite()),
        "C input contains invalid numerical values",
    )
}

fn accepted_4km_input(
    bundle: &LinkedSharedInputBundleV0,
) -> Result<&LinkedResolutionInputV0, ThinOwnerErrorV0> {
    bundle
        .resolutions
        .iter()
        .find(|value| value.nominal_spacing_km.to_bits() == TARGET_SPACING_KM.to_bits())
        .ok_or_else(|| fail("accepted bundle has no exact 4 km resolution"))
}

fn accepted_evaluator(
    bundle: &LinkedSharedInputBundleV0,
    input: &LinkedResolutionInputV0,
) -> Result<DeformationEvaluator, ThinOwnerErrorV0> {
    let evaluator = bundle
        .declaration
        .scenario
        .compile(&input.mesh)
        .map_err(|error| fail(format!("C forcing compilation failed: {error}")))?;
    require(
        evaluator.support_stencils() == input.compiled_stencils.as_slice(),
        "fresh C compiler stencils differ from accepted input",
    )?;
    Ok(evaluator)
}

fn accepted_forcing_binding(
    bundle: &LinkedSharedInputBundleV0,
    input: &LinkedResolutionInputV0,
) -> CForcingBindingV0 {
    CForcingBindingV0 {
        input_bundle_hash: bundle.derived_bundle_hash,
        input_resolution_hash: input.derived_resolution_hash,
        compiled_stencils_component_hash: input.component_hashes.compiled_stencils_hash,
        frame_witnesses_component_hash: input.component_hashes.frame_witnesses_hash,
        cumulative_displacement_component_hash: input
            .component_hashes
            .cumulative_rock_displacement_hash,
    }
}

fn validate_experimental_binding(
    bundle: &LinkedSharedInputBundleV0,
    input: &LinkedResolutionInputV0,
    binding: ThinCExperimentalForcingBindingV0,
) -> Result<(), ThinOwnerErrorV0> {
    require(
        binding.synthetic_input_bundle_hash != bundle.derived_bundle_hash
            || binding.synthetic_input_resolution_hash != input.derived_resolution_hash,
        "experimental C forcing reuses the accepted linked-input identity",
    )?;
    require(
        binding.compiled_stencils_component_hash != input.component_hashes.compiled_stencils_hash
            && binding.frame_witnesses_component_hash
                != input.component_hashes.frame_witnesses_hash
            && binding.cumulative_displacement_component_hash
                != input.component_hashes.cumulative_rock_displacement_hash,
        "experimental C forcing reuses an accepted forcing component hash",
    )
}

fn derived_experimental_binding(
    bundle: &LinkedSharedInputBundleV0,
    input: &LinkedResolutionInputV0,
    evaluator: &DeformationEvaluator,
    cumulative_rock_displacement_km: &[f64],
) -> Result<ThinCExperimentalForcingBindingV0, ThinOwnerErrorV0> {
    let compiled_stencils_component_hash = fnv1a64(&fixed_bytes(&(
        EXPERIMENTAL_STENCILS_DOMAIN_V0,
        evaluator.support_stencils(),
    ))?);
    // The accepted path binds its registered frame-witness component. The
    // experimental path binds the complete compiled evaluator so distinct
    // between-witness chronology cannot share a C configuration identity.
    let frame_witnesses_component_hash = fnv1a64(&fixed_bytes(&(
        EXPERIMENTAL_EVALUATOR_DOMAIN_V0,
        evaluator,
    ))?);
    let cumulative_displacement_component_hash = fnv1a64(&fixed_bytes(&(
        EXPERIMENTAL_DISPLACEMENT_DOMAIN_V0,
        cumulative_rock_displacement_km,
    ))?);
    let identity_payload = (
        bundle.derived_bundle_hash,
        input.derived_resolution_hash,
        cumulative_displacement_component_hash,
    );
    Ok(ThinCExperimentalForcingBindingV0 {
        synthetic_input_bundle_hash: fnv1a64(&fixed_bytes(&(
            EXPERIMENTAL_BUNDLE_DOMAIN_V0,
            identity_payload,
        ))?),
        synthetic_input_resolution_hash: fnv1a64(&fixed_bytes(&(
            EXPERIMENTAL_RESOLUTION_DOMAIN_V0,
            identity_payload,
        ))?),
        compiled_stencils_component_hash,
        frame_witnesses_component_hash,
        cumulative_displacement_component_hash,
    })
}

fn validate_target_displacement(
    cell_count: usize,
    cumulative_rock_displacement_km: &[f64],
) -> Result<(), ThinOwnerErrorV0> {
    require(
        cumulative_rock_displacement_km.len() == cell_count,
        "C target displacement length mismatch",
    )?;
    require(
        cumulative_rock_displacement_km.iter().all(|value| {
            value.is_finite()
                && *value >= 0.0
                && (*value != 0.0 || value.to_bits() == 0.0f64.to_bits())
        }),
        "C target displacement contains invalid values",
    )
}

fn validate_evaluator(
    input: &LinkedResolutionInputV0,
    evaluator: &DeformationEvaluator,
) -> Result<(), ThinOwnerErrorV0> {
    let n = input.mesh.cell_count();
    require(
        !evaluator.support_stencils().is_empty(),
        "C forcing evaluator has no support stencils",
    )?;
    for stencil in evaluator.support_stencils() {
        require(
            stencil.weight_per_km2.len() == n
                && stencil
                    .weight_per_km2
                    .iter()
                    .all(|weight| weight.is_finite() && *weight >= 0.0),
            "C forcing evaluator has an invalid support stencil",
        )?;
        let integral: f64 = stencil
            .weight_per_km2
            .iter()
            .zip(&input.mesh.cell_area_km2)
            .map(|(weight, area)| weight * area)
            .sum();
        require(
            integral.is_finite() && (integral - 1.0).abs() <= 1.0e-8,
            "C forcing evaluator stencil is not area normalized",
        )?;
    }
    for &time in &[0.0, 3.0, 6.0, 10.0] {
        let frame = evaluator.evaluate(time);
        validate_forcing_frame(input, &frame, time)?;
    }
    Ok(())
}

fn validate_experimental_forcing_target(
    input: &LinkedResolutionInputV0,
    evaluator: &DeformationEvaluator,
    cumulative_rock_displacement_km: &[f64],
    declared_work_km3: f64,
) -> Result<(), ThinOwnerErrorV0> {
    let target_work: f64 = cumulative_rock_displacement_km
        .iter()
        .zip(&input.mesh.cell_area_km2)
        .map(|(depth, area)| depth * area)
        .sum();
    require_close(
        target_work,
        declared_work_km3,
        CONTROL_VOLUME_ABSOLUTE_TOLERANCE_KM3,
        CONTROL_VOLUME_RELATIVE_TOLERANCE,
        "C experimental target work",
    )?;

    // At full activity the registered 0--6 Myr schedule integrates to 5.75
    // Myr. This cheap audit rejects an evaluator that does not actually encode
    // the supplied cumulative target before the expensive process run starts.
    let peak = evaluator.evaluate(3.0);
    let implied_displacement = peak
        .rock_vertical_rate_km_myr
        .iter()
        .map(|rate| f64::from(*rate) * ACTIVITY_INTEGRAL_MYR)
        .collect::<Vec<_>>();
    let zero = vec![0.0; input.mesh.cell_count()];
    let audit = displacement_audit_arrays(
        &zero,
        cumulative_rock_displacement_km,
        &input.mesh.cell_area_km2,
        &implied_displacement,
    )?;
    require(
        audit.signed_volume_error_km3.abs()
            <= CONTROL_VOLUME_ABSOLUTE_TOLERANCE_KM3
                + CONTROL_VOLUME_RELATIVE_TOLERANCE * declared_work_km3,
        "C evaluator full-activity rate does not close to target work",
    )?;
    require(
        audit.maximum_error_km <= CONTROL_CELL_TOLERANCE_KM,
        "C evaluator full-activity rate does not match target displacement",
    )
}

fn identity(
    forcing_binding: CForcingBindingV0,
    purpose: OrganizationRunPurposeV0,
) -> OrganizationArtifactIdentityV0 {
    OrganizationArtifactIdentityV0 {
        input_bundle_hash: forcing_binding.input_bundle_hash,
        input_resolution_hash: forcing_binding.input_resolution_hash,
        nominal_spacing_km: TARGET_SPACING_KM,
        arm: OrganizationArmV0::C,
        purpose,
    }
}

fn registered_c_config_hash(
    input: &LinkedResolutionInputV0,
    identity: &OrganizationArtifactIdentityV0,
    control_binding: Option<u64>,
    forcing_binding: CForcingBindingV0,
) -> Result<u64, ThinOwnerErrorV0> {
    let active = identity.purpose == OrganizationRunPurposeV0::Base;
    let active_process = active.then_some(ActiveProcessConfigWireV0 {
        routing: RoutingConfigWireV0 {
            depression_policy: RoutingDepressionPolicyV0::PriorityVirtualSurfaceNoBedrockWrite,
            flow_partition: FlowPartitionPolicyV0::MfdSlope,
            runoff_component_hash: input.component_hashes.local_runoff_hash,
        },
        denudation: EffectiveDenudationConfigWireV0 {
            k_km_inverse: DENUDATION_K_KM_INVERSE,
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
    let mut config = OrganizationArmConfigV0 {
        schema_version: ORGANIZATION_ARM_CONFIG_SCHEMA_VERSION_V0.into(),
        hash_version: ORGANIZATION_ARTIFACT_HASH_VERSION_V0.into(),
        identity: identity.clone(),
        predecessors: OrganizationPredecessorsV0 {
            opportunity_control_result_hash: control_binding,
            g_reference_4km: None,
        },
        payload: OrganizationArmConfigPayloadV0::C(CConfigWireV0 {
            config_schema: ORGANIZATION_C_CONFIG_SCHEMA_VERSION_V0.into(),
            process_mode: if active {
                CProcessModeV0::Coevolving
            } else {
                CProcessModeV0::UpliftOnly
            },
            start_time_myr: 0.0,
            forcing_end_time_myr: 6.0,
            end_time_myr: 10.0,
            checkpoint_times_myr: CHECKPOINT_TIMES_MYR.to_vec(),
            forcing_sampling: CForcingSamplingPolicyV0::CandidateMidpointResampledOnEveryRetry,
            endpoint_clipping: CEndpointClippingPolicyV0::ClipToCheckpointAndFinalEndpoint,
            vertical_rate_authority:
                CVerticalRateAuthorityV0::FreshCompilerEvaluatedF32AtCandidateMidpoint,
            forcing_compiler_id: "linked-cosine-support-area-normalized-v0".into(),
            compiled_stencils_component_hash: forcing_binding.compiled_stencils_component_hash,
            frame_witnesses_component_hash: forcing_binding.frame_witnesses_component_hash,
            cumulative_displacement_component_hash: forcing_binding
                .cumulative_displacement_component_hash,
            adaptive_integration: AdaptiveIntegrationConfigWireV0 {
                maximum_uplift_depth_km: Some(MAXIMUM_UPLIFT_DEPTH_KM),
                minimum_dt_myr: MINIMUM_DT_MYR,
                maximum_adaptive_attempts: MAXIMUM_ADAPTIVE_ATTEMPTS,
                requested_maximum_dt_myr: REQUESTED_MAXIMUM_DT_MYR,
            },
            split_order: active.then_some(SplitOrderV0::UpliftThenRouteDenudeThenHillslope),
            active_process,
        }),
        derived_config_hash: 0,
    };
    let hash = organization_arm_config_hash_v0(&config)
        .map_err(|error| fail(format!("registered C configuration rejected: {error}")))?;
    config.derived_config_hash = hash;
    Ok(hash)
}

fn run_control(
    input: &LinkedResolutionInputV0,
    evaluator: &DeformationEvaluator,
    cumulative_rock_displacement_km: &[f64],
    declared_work_km3: f64,
    forcing_binding: CForcingBindingV0,
) -> Result<ControlRunV0, ThinOwnerErrorV0> {
    let identity = identity(
        forcing_binding,
        OrganizationRunPurposeV0::OpportunityControl,
    );
    let config_hash = registered_c_config_hash(input, &identity, None, forcing_binding)?;
    let mut state = initial_driver_state(input, &identity)?;
    let initial_moment = elevation_moment_v0(&input.mesh, &state.elevation_km)
        .map_err(|error| fail(format!("C control initial moment failed: {error}")))?;
    let mut forcing_uplift = 0.0;
    let mut relaxation_uplift = 0.0;

    for &endpoint in &CHECKPOINT_TIMES_MYR[1..] {
        while state.time_myr.to_bits() != endpoint.to_bits() {
            require(
                state.completion.accepted_steps < MAXIMUM_ACCEPTED_STEP_COUNT,
                "C control reached the accepted-step ceiling",
            )?;
            let requested = requested_dt(state.time_myr, endpoint)?;
            let start = state.time_myr;
            let before_moment = elevation_moment_v0(&input.mesh, &state.elevation_km)
                .map_err(|error| fail(format!("C control moment failed: {error}")))?;
            let mut candidate_dt = requested;
            let mut final_limiter = ThinCStepLimiterV0::Requested;
            let mut accepted: Option<(Vec<f64>, f64, f64, u32)> = None;

            for attempt in 1..=MAXIMUM_ADAPTIVE_ATTEMPTS {
                require(
                    candidate_dt >= MINIMUM_DT_MYR,
                    "C control candidate dt fell below the registered minimum",
                )?;
                let midpoint = start + (0.5 * candidate_dt);
                let frame = evaluator.evaluate(midpoint);
                validate_forcing_frame(input, &frame, start)?;
                let uplift_limit = uplift_limit(&frame)?;
                if candidate_dt > uplift_limit {
                    state.completion.attempts =
                        checked_add(state.completion.attempts, 1, "C control attempt count")?;
                    final_limiter = ThinCStepLimiterV0::UpliftAccuracy;
                    if attempt == MAXIMUM_ADAPTIVE_ATTEMPTS {
                        return Err(fail("C control exhausted adaptive attempts"));
                    }
                    candidate_dt = propose_retry(candidate_dt, uplift_limit)?;
                    continue;
                }
                let (candidate, uplift_moment) =
                    apply_uplift(input, &state.elevation_km, &frame, candidate_dt)?;
                let after_moment = elevation_moment_v0(&input.mesh, &candidate)
                    .map_err(|error| fail(format!("C control moment failed: {error}")))?;
                require_close(
                    after_moment - before_moment,
                    uplift_moment,
                    SOLID_ABSOLUTE_TOLERANCE_KM3,
                    SOLID_RELATIVE_TOLERANCE,
                    "C control per-step solid moment",
                )?;
                state.completion.attempts =
                    checked_add(state.completion.attempts, 1, "C control attempt count")?;
                accepted = Some((candidate, uplift_moment, candidate_dt, attempt));
                break;
            }

            let (candidate, uplift_moment, accepted_dt, attempt_count) =
                accepted.ok_or_else(|| fail("C control produced no accepted candidate"))?;
            if start < 6.0 {
                forcing_uplift += uplift_moment;
            } else {
                require(
                    uplift_moment.to_bits() == 0.0f64.to_bits(),
                    "C control relaxation uplift is not exact positive zero",
                )?;
                relaxation_uplift += uplift_moment;
            }
            state.elevation_km = candidate;
            state.time_myr = advance_coordinate(start, endpoint, accepted_dt)?;
            state.revision = state
                .revision
                .checked_add(1)
                .ok_or_else(|| fail("C control revision overflow"))?;
            state
                .completion
                .accept(accepted_dt, attempt_count, final_limiter)?;
        }
        state.checkpoints.push(make_checkpoint(&identity, &state)?);
    }

    let completion = state.completion.finish(state.time_myr, state.revision)?;
    let integrated_uplift = forcing_uplift + relaxation_uplift;
    let final_moment = elevation_moment_v0(&input.mesh, &state.elevation_km)
        .map_err(|error| fail(format!("C control final moment failed: {error}")))?;
    let expected_final_moment = initial_moment + integrated_uplift;
    require_close(
        final_moment,
        expected_final_moment,
        SOLID_ABSOLUTE_TOLERANCE_KM3,
        SOLID_RELATIVE_TOLERANCE,
        "C control cumulative solid moment",
    )?;
    let solid_closure_error = canonical_zero(final_moment - expected_final_moment);
    let final_hash = elevation_component_hash(&identity, &state.elevation_km)?;
    let displacement =
        control_displacement_audit(input, cumulative_rock_displacement_km, &state.elevation_km)?;
    require(
        displacement.signed_volume_error_km3.abs()
            <= CONTROL_VOLUME_ABSOLUTE_TOLERANCE_KM3
                + (CONTROL_VOLUME_RELATIVE_TOLERANCE * declared_work_km3),
        "C opportunity control failed its signed displacement-volume gate",
    )?;
    require(
        displacement.maximum_error_km <= CONTROL_CELL_TOLERANCE_KM,
        "C opportunity control failed its maximum cell-displacement gate",
    )?;
    require(
        relaxation_uplift.to_bits() == 0.0f64.to_bits(),
        "C opportunity control accumulated nonzero relaxation uplift",
    )?;

    let binding_hash = fnv1a64(&fixed_bytes(&ThinControlBindingPreimageV0 {
        domain: THIN_CONTROL_BINDING_DOMAIN_V0,
        identity: &identity,
        config_hash,
        final_elevation_component_hash: final_hash,
        integrated_rock_uplift_moment_km3: integrated_uplift,
        initial_elevation_volume_moment_km3: initial_moment,
        final_elevation_volume_moment_km3: final_moment,
        solid_closure_error_km3: solid_closure_error,
        signed_displacement_volume_error_km3: displacement.signed_volume_error_km3,
        maximum_displacement_error_km: displacement.maximum_error_km,
    })?);
    Ok(ControlRunV0 {
        identity,
        config_hash,
        binding_hash,
        audit: ThinCOpportunityAuditV0 {
            final_elevation_component_hash: final_hash,
            initial_elevation_volume_moment_km3: initial_moment,
            integrated_rock_uplift_moment_km3: integrated_uplift,
            forcing_interval_rock_uplift_moment_km3: forcing_uplift,
            relaxation_interval_rock_uplift_moment_km3: relaxation_uplift,
            signed_displacement_volume_error_km3: displacement.signed_volume_error_km3,
            maximum_displacement_error_km: displacement.maximum_error_km,
            area_weighted_l1_displacement_error_km: displacement.l1_error_km,
            area_weighted_rms_displacement_error_km: displacement.rms_error_km,
            final_elevation_volume_moment_km3: final_moment,
            solid_closure_error_km3: solid_closure_error,
            completion,
            checkpoints: state.checkpoints,
        },
    })
}

fn run_base(
    input: &LinkedResolutionInputV0,
    evaluator: &DeformationEvaluator,
    forcing_binding: CForcingBindingV0,
    control: ControlRunV0,
) -> Result<ThinC4KmObservationV0, ThinOwnerErrorV0> {
    let base_identity = identity(forcing_binding, OrganizationRunPurposeV0::Base);
    let base_config_hash = registered_c_config_hash(
        input,
        &base_identity,
        Some(control.binding_hash),
        forcing_binding,
    )?;
    let mut state = initial_driver_state(input, &base_identity)?;
    let initial_moment = elevation_moment_v0(&input.mesh, &state.elevation_km)
        .map_err(|error| fail(format!("C base initial moment failed: {error}")))?;
    let mut totals = BaseAccumulatorV0::new(input)?;

    for &endpoint in &CHECKPOINT_TIMES_MYR[1..] {
        while state.time_myr.to_bits() != endpoint.to_bits() {
            require(
                state.completion.accepted_steps < MAXIMUM_ACCEPTED_STEP_COUNT,
                "C base reached the accepted-step ceiling",
            )?;
            let requested = requested_dt(state.time_myr, endpoint)?;
            let start = state.time_myr;
            let before_moment = elevation_moment_v0(&input.mesh, &state.elevation_km)
                .map_err(|error| fail(format!("C base moment failed: {error}")))?;
            let mut candidate_dt = requested;
            let mut final_limiter = ThinCStepLimiterV0::Requested;
            let mut accepted = None;

            for attempt in 1..=MAXIMUM_ADAPTIVE_ATTEMPTS {
                require(
                    candidate_dt >= MINIMUM_DT_MYR,
                    "C base candidate dt fell below the registered minimum",
                )?;
                let midpoint = start + (0.5 * candidate_dt);
                let frame = evaluator.evaluate(midpoint);
                validate_forcing_frame(input, &frame, start)?;
                let limit = uplift_limit(&frame)?;
                if candidate_dt > limit {
                    state.completion.note_attempt()?;
                    final_limiter = ThinCStepLimiterV0::UpliftAccuracy;
                    if attempt == MAXIMUM_ADAPTIVE_ATTEMPTS {
                        return Err(fail("C base exhausted adaptive attempts"));
                    }
                    candidate_dt = propose_retry(candidate_dt, limit)?;
                    continue;
                }
                let (after_uplift, uplift_moment) =
                    apply_uplift(input, &state.elevation_km, &frame, candidate_dt)?;
                match attempt_active_process_v0(
                    &input.mesh,
                    &after_uplift,
                    &input.local_runoff_supply_km3_myr,
                    candidate_dt,
                )
                .map_err(|error| fail(format!("C active process failed: {error}")))?
                {
                    ProcessAttemptV0::Retry { limiter, limit_myr } => {
                        require(
                            limit_myr.is_finite() && limit_myr > 0.0,
                            "C process returned an invalid retry limit",
                        )?;
                        state.completion.note_attempt()?;
                        final_limiter = map_process_limiter(limiter);
                        if attempt == MAXIMUM_ADAPTIVE_ATTEMPTS {
                            return Err(fail("C base exhausted adaptive attempts"));
                        }
                        candidate_dt = propose_retry(candidate_dt, limit_myr)?;
                    }
                    ProcessAttemptV0::Accepted(process) => {
                        state.completion.note_attempt()?;
                        let actual_change = process.final_elevation_moment_km3 - before_moment;
                        let expected_change = uplift_moment
                            - process.effective_denudation_export_km3
                            - process.hillslope_portal_transfer_km3;
                        require_close(
                            actual_change,
                            expected_change,
                            SOLID_ABSOLUTE_TOLERANCE_KM3,
                            SOLID_RELATIVE_TOLERANCE,
                            "C base per-step solid moment",
                        )?;
                        require_close(
                            process.water.total_supply_km3_myr,
                            process.water.total_portal_outflow_km3_myr
                                + process.water.unresolved_sink_rate_km3_myr,
                            WATER_RATE_ABSOLUTE_TOLERANCE_KM3_MYR,
                            WATER_RELATIVE_TOLERANCE,
                            "C base instantaneous water",
                        )?;
                        let expected_water_rate = process.water.total_portal_outflow_km3_myr
                            + process.water.unresolved_sink_rate_km3_myr;
                        require(
                            process.water.balance_error_km3_myr.to_bits()
                                == (process.water.total_supply_km3_myr - expected_water_rate)
                                    .to_bits(),
                            "C base water-balance witness disagrees with fresh reduction",
                        )?;
                        accepted = Some((process, uplift_moment, candidate_dt, attempt));
                        break;
                    }
                }
            }

            let (process, uplift_moment, accepted_dt, attempt_count) =
                accepted.ok_or_else(|| fail("C base produced no accepted candidate"))?;
            totals.accumulate(input, start, accepted_dt, uplift_moment, &process)?;
            let candidate_total_uplift = totals.forcing_uplift + totals.relaxation_uplift;
            let candidate_expected_moment = initial_moment + candidate_total_uplift
                - totals.denudation_export
                - totals.hillslope_total;
            require_close(
                process.final_elevation_moment_km3,
                candidate_expected_moment,
                SOLID_ABSOLUTE_TOLERANCE_KM3,
                SOLID_RELATIVE_TOLERANCE,
                "C base candidate cumulative solid moment",
            )?;
            let candidate_water_expected = totals.water.total_portal_outflow_volume_km3
                + totals.water.unresolved_sink_volume_km3;
            require_close(
                totals.water.supplied_volume_km3,
                candidate_water_expected,
                WATER_VOLUME_ABSOLUTE_TOLERANCE_KM3,
                WATER_RELATIVE_TOLERANCE,
                "C base candidate integrated water",
            )?;
            totals.water.balance_error_km3 =
                canonical_zero(totals.water.supplied_volume_km3 - candidate_water_expected);
            state.elevation_km = process.final_elevation_km;
            state.time_myr = advance_coordinate(start, endpoint, accepted_dt)?;
            state.revision = state
                .revision
                .checked_add(1)
                .ok_or_else(|| fail("C base revision overflow"))?;
            state
                .completion
                .accept(accepted_dt, attempt_count, final_limiter)?;
        }
        state
            .checkpoints
            .push(make_checkpoint(&base_identity, &state)?);
    }

    require(
        totals.relaxation_uplift.to_bits() == 0.0f64.to_bits(),
        "C base relaxation uplift is not exact positive zero",
    )?;
    let final_moment = elevation_moment_v0(&input.mesh, &state.elevation_km)
        .map_err(|error| fail(format!("C base final moment failed: {error}")))?;
    let total_uplift = totals.forcing_uplift + totals.relaxation_uplift;
    let expected_final =
        initial_moment + total_uplift - totals.denudation_export - totals.hillslope_total;
    require_close(
        final_moment,
        expected_final,
        SOLID_ABSOLUTE_TOLERANCE_KM3,
        SOLID_RELATIVE_TOLERANCE,
        "C base cumulative solid moment",
    )?;
    let water_expected =
        totals.water.total_portal_outflow_volume_km3 + totals.water.unresolved_sink_volume_km3;
    require_close(
        totals.water.supplied_volume_km3,
        water_expected,
        WATER_VOLUME_ABSOLUTE_TOLERANCE_KM3,
        WATER_RELATIVE_TOLERANCE,
        "C base integrated water",
    )?;
    totals.water.balance_error_km3 =
        canonical_zero(totals.water.supplied_volume_km3 - water_expected);

    let final_routing = fresh_routing_diagnostics_v0(
        &input.mesh,
        &state.elevation_km,
        &input.local_runoff_supply_km3_myr,
    )
    .map_err(|error| fail(format!("C final routing failed: {error}")))?;
    require_close(
        final_routing.total_supply_km3_myr,
        final_routing.total_portal_outflow_km3_myr + final_routing.unresolved_sink_rate_km3_myr,
        WATER_RATE_ABSOLUTE_TOLERANCE_KM3_MYR,
        WATER_RELATIVE_TOLERANCE,
        "C final routing water",
    )?;
    let final_routing_hash = diagnostic_routing_hash(&base_identity, &final_routing)?;
    let final_hash = elevation_component_hash(&base_identity, &state.elevation_km)?;
    let (final_min_elevation_km, final_max_elevation_km) = finite_min_max(&state.elevation_km)?;
    let completion = state.completion.finish(state.time_myr, state.revision)?;
    let closure_error = canonical_zero(final_moment - expected_final);

    Ok(ThinC4KmObservationV0 {
        schema_version: THIN_C_4KM_SCHEMA_VERSION_V0.into(),
        profile: THIN_OWNER_PROFILE_V0.into(),
        control_identity: control.identity,
        control_config_hash: control.config_hash,
        noncanonical_control_binding_hash: control.binding_hash,
        opportunity: control.audit,
        base_identity,
        base_config_hash,
        final_elevation_component_hash: final_hash,
        final_min_elevation_km,
        final_max_elevation_km,
        completion,
        solid: ThinCSolidLedgerV0 {
            initial_elevation_volume_moment_km3: initial_moment,
            forcing_interval_rock_uplift_moment_km3: totals.forcing_uplift,
            relaxation_interval_rock_uplift_moment_km3: totals.relaxation_uplift,
            total_rock_uplift_moment_km3: total_uplift,
            effective_denudation_export_km3: totals.denudation_export,
            hillslope_portal_transfers_km3: totals.hillslope_by_portal,
            total_hillslope_portal_transfer_km3: totals.hillslope_total,
            final_elevation_volume_moment_km3: final_moment,
            closure_error_km3: closure_error,
        },
        water: totals.water,
        process: ThinCProcessSummaryV0 {
            maximum_denudation_rate_km_myr: totals.max_denudation_rate,
            maximum_linear_hillslope_abs_grade: totals.max_hillslope_grade,
            maximum_unresolved_specific_discharge_cells: totals.max_unresolved_cells,
            final_routing_hash,
            final_routing_total_supply_km3_myr: final_routing.total_supply_km3_myr,
            final_routing_total_portal_outflow_km3_myr: final_routing.total_portal_outflow_km3_myr,
            final_routing_unresolved_sink_km3_myr: final_routing.unresolved_sink_rate_km3_myr,
            final_routing_water_balance_error_km3_myr: final_routing.balance_error_km3_myr,
        },
        checkpoints: state.checkpoints,
        final_elevation_km: state.elevation_km,
    })
}

impl CompletionAccumulatorV0 {
    fn note_attempt(&mut self) -> Result<(), ThinOwnerErrorV0> {
        self.attempts = checked_add(self.attempts, 1, "C candidate-attempt count")?;
        Ok(())
    }

    fn accept(
        &mut self,
        dt_myr: f64,
        attempts_for_step: u32,
        limiter: ThinCStepLimiterV0,
    ) -> Result<(), ThinOwnerErrorV0> {
        require(dt_myr.is_finite() && dt_myr > 0.0, "invalid accepted C dt")?;
        self.accepted_steps = checked_add(self.accepted_steps, 1, "C accepted-step count")?;
        self.max_attempts = self.max_attempts.max(attempts_for_step);
        self.min_dt = Some(self.min_dt.map_or(dt_myr, |value| value.min(dt_myr)));
        self.max_dt = Some(self.max_dt.map_or(dt_myr, |value| value.max(dt_myr)));
        let slot = match limiter {
            ThinCStepLimiterV0::Requested => &mut self.histogram.requested,
            ThinCStepLimiterV0::UpliftAccuracy => &mut self.histogram.uplift_accuracy,
            ThinCStepLimiterV0::DenudationAccuracy => &mut self.histogram.denudation_accuracy,
            ThinCStepLimiterV0::DenudationSlopeCourant => {
                &mut self.histogram.denudation_slope_courant
            }
            ThinCStepLimiterV0::HillslopeStability => &mut self.histogram.hillslope_stability,
        };
        *slot = checked_add(*slot, 1, "C limiter histogram")?;
        Ok(())
    }

    fn finish(&self, time_myr: f64, revision: u64) -> Result<ThinCCompletionV0, ThinOwnerErrorV0> {
        require(
            time_myr.to_bits() == 10.0f64.to_bits(),
            "C driver did not reach exact 10 Myr",
        )?;
        require(
            revision == self.accepted_steps,
            "C revision and accepted-step count disagree",
        )?;
        let histogram_total = self.histogram.requested
            + self.histogram.uplift_accuracy
            + self.histogram.denudation_accuracy
            + self.histogram.denudation_slope_courant
            + self.histogram.hillslope_stability;
        require(
            histogram_total == self.accepted_steps,
            "C limiter histogram does not reduce to the step count",
        )?;
        Ok(ThinCCompletionV0 {
            reached_time_myr: time_myr,
            final_revision: revision,
            accepted_step_count: self.accepted_steps,
            total_candidate_attempt_count: self.attempts,
            maximum_attempts_for_one_step: self.max_attempts,
            minimum_accepted_dt_myr: self
                .min_dt
                .ok_or_else(|| fail("C run contains no accepted dt"))?,
            maximum_accepted_dt_myr: self
                .max_dt
                .ok_or_else(|| fail("C run contains no accepted dt"))?,
            limiter_histogram: self.histogram.clone(),
        })
    }
}

impl BaseAccumulatorV0 {
    fn new(input: &LinkedResolutionInputV0) -> Result<Self, ThinOwnerErrorV0> {
        let portals = input
            .mesh
            .outlet_portals
            .iter()
            .map(|portal| ThinCPortalVolumeV0 {
                portal_id: portal.id.0,
                volume_km3: 0.0,
            })
            .collect::<Vec<_>>();
        require_portal_order(
            &input
                .mesh
                .outlet_portals
                .iter()
                .map(|p| p.id)
                .collect::<Vec<_>>(),
        )?;
        Ok(Self {
            forcing_uplift: 0.0,
            relaxation_uplift: 0.0,
            denudation_export: 0.0,
            hillslope_by_portal: portals.clone(),
            hillslope_total: 0.0,
            water: ThinCIntegratedWaterLedgerV0 {
                supplied_volume_km3: 0.0,
                portal_outflow_volume_km3: portals,
                total_portal_outflow_volume_km3: 0.0,
                unresolved_sink_volume_km3: 0.0,
                balance_error_km3: 0.0,
            },
            max_denudation_rate: 0.0,
            max_hillslope_grade: 0.0,
            max_unresolved_cells: 0,
        })
    }

    fn accumulate(
        &mut self,
        input: &LinkedResolutionInputV0,
        start_myr: f64,
        dt_myr: f64,
        uplift_moment_km3: f64,
        process: &ProcessStepV0,
    ) -> Result<(), ThinOwnerErrorV0> {
        if start_myr < 6.0 {
            self.forcing_uplift += uplift_moment_km3;
        } else {
            require(
                uplift_moment_km3.to_bits() == 0.0f64.to_bits(),
                "C base accepted nonzero uplift after 6 Myr",
            )?;
            self.relaxation_uplift += uplift_moment_km3;
        }
        self.denudation_export += process.effective_denudation_export_km3;
        require(
            process.hillslope_portal_transfers_km3.len() == self.hillslope_by_portal.len(),
            "C hillslope portal vector length mismatch",
        )?;
        for (stored, step) in self
            .hillslope_by_portal
            .iter_mut()
            .zip(&process.hillslope_portal_transfers_km3)
        {
            require(
                stored.portal_id == step.portal_id,
                "C hillslope portal order mismatch",
            )?;
            stored.volume_km3 += step.volume_km3;
        }
        self.hillslope_total += process.hillslope_portal_transfer_km3;

        require(
            process.water.portal_outflow_km3_myr.len()
                == self.water.portal_outflow_volume_km3.len(),
            "C water portal vector length mismatch",
        )?;
        self.water.supplied_volume_km3 += process.water.total_supply_km3_myr * dt_myr;
        for (stored, rate) in self
            .water
            .portal_outflow_volume_km3
            .iter_mut()
            .zip(&process.water.portal_outflow_km3_myr)
        {
            require(
                stored.portal_id == rate.portal_id,
                "C water portal order mismatch",
            )?;
            stored.volume_km3 += rate.rate_km3_myr * dt_myr;
        }
        self.water.total_portal_outflow_volume_km3 +=
            process.water.total_portal_outflow_km3_myr * dt_myr;
        self.water.unresolved_sink_volume_km3 +=
            process.water.unresolved_sink_rate_km3_myr * dt_myr;
        self.max_denudation_rate = self
            .max_denudation_rate
            .max(process.maximum_effective_denudation_rate_km_myr);
        self.max_hillslope_grade = self
            .max_hillslope_grade
            .max(process.maximum_linear_hillslope_abs_grade);
        self.max_unresolved_cells = self
            .max_unresolved_cells
            .max(process.water.unresolved_specific_discharge_cell_count);

        require(
            process.final_elevation_km.len() == input.mesh.cell_count(),
            "C process final surface length mismatch",
        )?;
        require(
            process
                .hillslope_internal_conservation_error_km3
                .is_finite()
                && process.process_solid_closure_error_km3.is_finite()
                && process.water.balance_error_km3_myr.is_finite(),
            "C process returned nonfinite diagnostics",
        )
    }
}

fn initial_driver_state(
    input: &LinkedResolutionInputV0,
    identity: &OrganizationArtifactIdentityV0,
) -> Result<DriverStateV0, ThinOwnerErrorV0> {
    let mut state = DriverStateV0 {
        time_myr: 0.0,
        revision: 0,
        elevation_km: input.initial_elevation_km.clone(),
        completion: CompletionAccumulatorV0::default(),
        checkpoints: Vec::with_capacity(CHECKPOINT_TIMES_MYR.len()),
    };
    state.checkpoints.push(make_checkpoint(identity, &state)?);
    Ok(state)
}

fn requested_dt(start: f64, endpoint: f64) -> Result<f64, ThinOwnerErrorV0> {
    let remaining = endpoint - start;
    require(
        remaining.is_finite() && remaining > 0.0,
        "C endpoint driver has invalid remaining time",
    )?;
    let ordinary = REQUESTED_MAXIMUM_DT_MYR.min(remaining);
    let tail = endpoint - (start + ordinary);
    let requested = if tail < MINIMUM_DT_MYR {
        remaining
    } else {
        ordinary
    };
    require(
        requested.is_finite() && requested > 0.0,
        "C endpoint driver produced invalid requested dt",
    )?;
    Ok(requested)
}

fn advance_coordinate(
    start: f64,
    endpoint: f64,
    accepted_dt: f64,
) -> Result<f64, ThinOwnerErrorV0> {
    let remaining = endpoint - start;
    let end = if accepted_dt.to_bits() == remaining.to_bits() {
        endpoint
    } else {
        let ordinary = start + accepted_dt;
        require(
            ordinary > start && ordinary < endpoint,
            "C accepted step crossed or failed to approach its endpoint",
        )?;
        ordinary
    };
    // Promotion-contract amendment required: in binary64, ordinary
    // `(start + dt) - start` does not generally recover the original `dt` bits
    // (for example at start=2 and dt=0.01). The thin probe keeps the physically
    // consumed dt unchanged and applies only the registered exact-endpoint rule.
    Ok(end)
}

fn propose_retry(previous: f64, limit: f64) -> Result<f64, ThinOwnerErrorV0> {
    require(
        previous.is_finite() && previous > 0.0 && limit.is_finite() && limit > 0.0,
        "C retry proposal received an invalid dt or limit",
    )?;
    Ok(previous.min(limit))
}

fn validate_forcing_frame(
    input: &LinkedResolutionInputV0,
    frame: &DeformationFrame,
    start_myr: f64,
) -> Result<(), ThinOwnerErrorV0> {
    let n = input.mesh.cell_count();
    require(
        frame.rock_vertical_rate_km_myr.len() == n
            && frame.horizontal_velocity_km_myr.len() == n
            && frame.dominant_episode.len() == n,
        "C compiler frame shape mismatch",
    )?;
    require(
        frame
            .rock_vertical_rate_km_myr
            .iter()
            .all(|rate| rate.is_finite()),
        "C compiler emitted a nonfinite vertical rate",
    )?;
    if start_myr >= 6.0 {
        require(
            frame
                .rock_vertical_rate_km_myr
                .iter()
                .all(|rate| rate.to_bits() == 0.0f32.to_bits()),
            "C compiler emitted non-positive-zero uplift after 6 Myr",
        )?;
    }
    Ok(())
}

fn uplift_limit(frame: &DeformationFrame) -> Result<f64, ThinOwnerErrorV0> {
    let mut maximum_rate: f64 = 0.0;
    for rate in &frame.rock_vertical_rate_km_myr {
        let value = f64::from(*rate).abs();
        require(value.is_finite(), "C uplift rate is nonfinite")?;
        maximum_rate = maximum_rate.max(value);
    }
    Ok(if maximum_rate == 0.0 {
        f64::INFINITY
    } else {
        MAXIMUM_UPLIFT_DEPTH_KM / maximum_rate
    })
}

fn apply_uplift(
    input: &LinkedResolutionInputV0,
    elevation_km: &[f64],
    frame: &DeformationFrame,
    dt_myr: f64,
) -> Result<(Vec<f64>, f64), ThinOwnerErrorV0> {
    let mut candidate = elevation_km.to_vec();
    let mut uplift_moment = 0.0;
    for (cell, elevation) in candidate.iter_mut().enumerate() {
        let depth = f64::from(frame.rock_vertical_rate_km_myr[cell]) * dt_myr;
        *elevation += depth;
        uplift_moment += depth * input.mesh.cell_area_km2[cell];
    }
    require(
        candidate.iter().all(|value| value.is_finite()) && uplift_moment.is_finite(),
        "C uplift produced a nonfinite candidate",
    )?;
    Ok((candidate, uplift_moment))
}

fn map_process_limiter(value: ProcessLimiterV0) -> ThinCStepLimiterV0 {
    match value {
        ProcessLimiterV0::EffectiveDenudationSlopeCourant => {
            ThinCStepLimiterV0::DenudationSlopeCourant
        }
        ProcessLimiterV0::EffectiveDenudationAccuracy => ThinCStepLimiterV0::DenudationAccuracy,
        ProcessLimiterV0::HillslopeStability => ThinCStepLimiterV0::HillslopeStability,
    }
}

struct ControlDisplacementV0 {
    signed_volume_error_km3: f64,
    maximum_error_km: f64,
    l1_error_km: f64,
    rms_error_km: f64,
}

fn control_displacement_audit(
    input: &LinkedResolutionInputV0,
    cumulative_rock_displacement_km: &[f64],
    final_elevation_km: &[f64],
) -> Result<ControlDisplacementV0, ThinOwnerErrorV0> {
    displacement_audit_arrays(
        &input.initial_elevation_km,
        cumulative_rock_displacement_km,
        &input.mesh.cell_area_km2,
        final_elevation_km,
    )
}

fn displacement_audit_arrays(
    initial: &[f64],
    displacement: &[f64],
    area: &[f64],
    final_elevation: &[f64],
) -> Result<ControlDisplacementV0, ThinOwnerErrorV0> {
    require(
        initial.len() == displacement.len()
            && initial.len() == area.len()
            && initial.len() == final_elevation.len(),
        "C control displacement arrays have unequal lengths",
    )?;
    let mut area_total = 0.0;
    let mut signed_volume = 0.0;
    let mut maximum_error: f64 = 0.0;
    let mut l1_sum = 0.0;
    let mut square_sum = 0.0;
    for cell in 0..initial.len() {
        let error = (final_elevation[cell] - initial[cell]) - displacement[cell];
        require(
            error.is_finite() && area[cell].is_finite() && area[cell] > 0.0,
            "invalid C displacement audit value",
        )?;
        area_total += area[cell];
        signed_volume += error * area[cell];
        maximum_error = maximum_error.max(error.abs());
        l1_sum += error.abs() * area[cell];
        square_sum += (error * error) * area[cell];
    }
    require(
        area_total.is_finite() && area_total > 0.0,
        "invalid C audit area",
    )?;
    Ok(ControlDisplacementV0 {
        signed_volume_error_km3: canonical_zero(signed_volume),
        maximum_error_km: maximum_error,
        l1_error_km: l1_sum / area_total,
        rms_error_km: (square_sum / area_total).sqrt(),
    })
}

#[derive(Serialize)]
struct ThinRoutingHashRateV0 {
    portal_id: u32,
    rate_km3_myr: f64,
}

#[derive(Serialize)]
struct ThinRoutingHashPreimageV0<'a> {
    domain: &'static str,
    identity: &'a OrganizationArtifactIdentityV0,
    total_supply_km3_myr: f64,
    portal_outflow_km3_myr: &'a Vec<ThinRoutingHashRateV0>,
    total_portal_outflow_km3_myr: f64,
    unresolved_sink_rate_km3_myr: f64,
    balance_error_km3_myr: f64,
    unresolved_specific_discharge_cell_count: u64,
}

fn diagnostic_routing_hash(
    identity: &OrganizationArtifactIdentityV0,
    routing: &ProcessWaterRateV0,
) -> Result<u64, ThinOwnerErrorV0> {
    let portal_outflow = routing
        .portal_outflow_km3_myr
        .iter()
        .map(|entry| ThinRoutingHashRateV0 {
            portal_id: entry.portal_id,
            rate_km3_myr: entry.rate_km3_myr,
        })
        .collect::<Vec<_>>();
    Ok(fnv1a64(&fixed_bytes(&ThinRoutingHashPreimageV0 {
        domain: "orogen-owner-thin-v0/c-final-routing",
        identity,
        total_supply_km3_myr: routing.total_supply_km3_myr,
        portal_outflow_km3_myr: &portal_outflow,
        total_portal_outflow_km3_myr: routing.total_portal_outflow_km3_myr,
        unresolved_sink_rate_km3_myr: routing.unresolved_sink_rate_km3_myr,
        balance_error_km3_myr: routing.balance_error_km3_myr,
        unresolved_specific_discharge_cell_count: routing.unresolved_specific_discharge_cell_count,
    })?))
}

fn make_checkpoint(
    identity: &OrganizationArtifactIdentityV0,
    state: &DriverStateV0,
) -> Result<ThinCCheckpointV0, ThinOwnerErrorV0> {
    let elevation_component_hash = elevation_component_hash(identity, &state.elevation_km)?;
    let diagnostic_checkpoint_hash = fnv1a64(&fixed_bytes(&ThinCheckpointPreimageV0 {
        domain: THIN_CHECKPOINT_DOMAIN_V0,
        identity,
        time_myr: state.time_myr,
        revision: state.revision,
        elevation_km: &state.elevation_km,
    })?);
    Ok(ThinCCheckpointV0 {
        time_myr: state.time_myr,
        revision: state.revision,
        elevation_component_hash,
        diagnostic_checkpoint_hash,
    })
}

fn elevation_component_hash(
    identity: &OrganizationArtifactIdentityV0,
    elevation_km: &Vec<f64>,
) -> Result<u64, ThinOwnerErrorV0> {
    Ok(fnv1a64(&fixed_bytes(&ElevationArrayPreimageV0 {
        domain: ELEVATION_ARRAY_DOMAIN_V0,
        identity,
        elevation_km,
    })?))
}

fn require_portal_order(ids: &[OutletPortalId]) -> Result<(), ThinOwnerErrorV0> {
    require(
        ids.windows(2).all(|pair| pair[0].0 < pair[1].0),
        "C input portal IDs are not strictly ascending",
    )
}

fn finite_min_max(values: &[f64]) -> Result<(f64, f64), ThinOwnerErrorV0> {
    let first = *values
        .first()
        .ok_or_else(|| fail("C final elevation is empty"))?;
    require(first.is_finite(), "C final elevation is nonfinite")?;
    let mut minimum = first;
    let mut maximum = first;
    for &value in &values[1..] {
        require(value.is_finite(), "C final elevation is nonfinite")?;
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
    role: &str,
) -> Result<(), ThinOwnerErrorV0> {
    require(
        actual.is_finite()
            && expected.is_finite()
            && absolute.is_finite()
            && relative.is_finite()
            && absolute >= 0.0
            && relative >= 0.0,
        format!("{role} received invalid closeness operands"),
    )?;
    let tolerance = absolute + (relative * actual.abs().max(expected.abs()));
    require(
        (actual - expected).abs() <= tolerance,
        format!(
            "{role} does not close: actual={actual}, expected={expected}, tolerance={tolerance}"
        ),
    )
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn checked_add(value: u64, increment: u64, role: &str) -> Result<u64, ThinOwnerErrorV0> {
    value
        .checked_add(increment)
        .ok_or_else(|| fail(format!("{role} overflow")))
}

fn fixed_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, ThinOwnerErrorV0> {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .serialize(value)
        .map_err(|error| fail(format!("thin C encoding failed: {error}")))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn require(condition: bool, message: impl Into<String>) -> Result<(), ThinOwnerErrorV0> {
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
    fn experimental_target_validation_rejects_wrong_shape_and_nonfinite_values() {
        assert!(validate_target_displacement(2, &[0.0]).is_err());
        let mut invalid = vec![0.0; 2];
        invalid[0] = f64::INFINITY;
        assert!(validate_target_displacement(2, &invalid).is_err());
    }

    #[test]
    fn endpoint_driver_lands_on_registered_bits_without_epsilon() {
        let start = 2.9975;
        let endpoint = 3.0;
        let remaining = endpoint - start;
        assert_eq!(
            requested_dt(start, endpoint).unwrap().to_bits(),
            remaining.to_bits()
        );
        assert_eq!(
            advance_coordinate(start, endpoint, remaining)
                .unwrap()
                .to_bits(),
            endpoint.to_bits()
        );
        let interior = advance_coordinate(2.0, 3.0, 0.01).unwrap();
        assert!(interior < 3.0);
        assert!(interior > 2.0);
    }

    #[test]
    fn retry_rule_retains_equal_limit_and_last_limiter_count() {
        let value = 0.004_f64;
        assert_eq!(
            propose_retry(value, value).unwrap().to_bits(),
            value.to_bits()
        );
        assert_eq!(
            propose_retry(0.01, value).unwrap().to_bits(),
            value.to_bits()
        );
        let mut completion = CompletionAccumulatorV0::default();
        completion.note_attempt().unwrap();
        completion.note_attempt().unwrap();
        completion
            .accept(0.004, 2, ThinCStepLimiterV0::DenudationSlopeCourant)
            .unwrap();
        assert_eq!(completion.attempts, 2);
        assert_eq!(completion.max_attempts, 2);
        assert_eq!(completion.histogram.denudation_slope_courant, 1);
        assert_eq!(completion.histogram.requested, 0);
    }

    #[test]
    fn manufactured_control_displacement_reductions_use_stored_cell_order() {
        let initial = [1.0, -2.0, 0.5];
        let displacement = [0.25, 0.5, 1.0];
        let area = [1.0, 3.0, 7.0];
        let errors = [1.0e-6, -2.0e-6, 0.5e-6];
        let final_elevation = [
            initial[0] + displacement[0] + errors[0],
            initial[1] + displacement[1] + errors[1],
            initial[2] + displacement[2] + errors[2],
        ];
        let audit =
            displacement_audit_arrays(&initial, &displacement, &area, &final_elevation).unwrap();
        let expected_signed = errors[0] * area[0] + errors[1] * area[1] + errors[2] * area[2];
        let expected_l1 =
            (errors[0].abs() * area[0] + errors[1].abs() * area[1] + errors[2].abs() * area[2])
                / area.iter().sum::<f64>();
        let expected_rms = ((errors[0] * errors[0] * area[0]
            + errors[1] * errors[1] * area[1]
            + errors[2] * errors[2] * area[2])
            / area.iter().sum::<f64>())
        .sqrt();
        assert!((audit.signed_volume_error_km3 - expected_signed).abs() < 2e-15);
        assert!((audit.maximum_error_km - 2.0e-6).abs() < 2e-15);
        assert!((audit.l1_error_km - expected_l1).abs() < 2e-15);
        assert!((audit.rms_error_km - expected_rms).abs() < 2e-15);
    }

    #[test]
    fn process_disabled_f32_midpoint_control_matches_frozen_oracle() {
        let full_rates: [f64; 4] = [0.0, 1.0 / 1024.0, 1.0 / 8.0, 1.0];
        let initial = [0.0_f64, 1.0, -1.0, 2.0];
        let mut elevation = initial;
        let mut time: f64 = 0.0;
        let mut steps = 0_u64;
        for endpoint in [3.0_f64, 6.0, 10.0] {
            while time.to_bits() != endpoint.to_bits() {
                let dt = requested_dt(time, endpoint).unwrap();
                let midpoint = time + (0.5 * dt);
                let activity = fixture_activity(midpoint);
                for (value, full_rate) in elevation.iter_mut().zip(full_rates) {
                    let emitted_rate = (activity * full_rate) as f32;
                    *value += f64::from(emitted_rate) * dt;
                }
                time = advance_coordinate(time, endpoint, dt).unwrap();
                steps += 1;
            }
        }
        let displacement = elevation
            .into_iter()
            .zip(initial)
            .map(|(final_value, initial_value)| final_value - initial_value)
            .collect::<Vec<_>>();
        assert_eq!(steps, 1_000);
        assert_eq!(
            displacement
                .iter()
                .copied()
                .map(f64::to_bits)
                .collect::<Vec<_>>(),
            [
                0.0,
                0.005615234374438227,
                0.7187499999286835,
                5.749999999429468,
            ]
            .map(f64::to_bits)
            .to_vec()
        );
        let area: [f64; 4] = [1.0, 3.0, 7.0, 11.0];
        let analytic = full_rates.map(|rate| rate * 5.75);
        let signed_error = displacement
            .iter()
            .zip(analytic)
            .zip(area)
            .fold(0.0, |sum, ((emitted, expected), cell_area)| {
                sum + ((*emitted - expected) * cell_area)
            });
        assert_eq!(
            signed_error.to_bits(),
            (-6.776753602721897e-9_f64).to_bits()
        );
    }

    fn fixture_activity(time: f64) -> f64 {
        if !(0.0..=6.0).contains(&time) {
            return 0.0;
        }
        let edge = (time / 0.25).min((6.0 - time) / 0.25).clamp(0.0, 1.0);
        edge * edge * (3.0 - (2.0 * edge))
    }

    #[test]
    #[ignore = "rebuilds the accepted bundle and executes two complete 4 km C runs"]
    fn accepted_linked_input_c_probe_is_bit_deterministic() {
        let bundle = super::super::build_linked_shared_input_bundle_v0().unwrap();
        let result = run_repeated_thin_c_4km_v0(&bundle).unwrap();
        assert_eq!(
            result.completion.reached_time_myr.to_bits(),
            10.0f64.to_bits()
        );
        assert_eq!(
            result.completion.final_revision,
            result.completion.accepted_step_count
        );
        assert_eq!(
            result.final_elevation_km.len(),
            bundle.resolutions[1].mesh.cell_count()
        );
        let input = accepted_4km_input(&bundle).unwrap();
        let binding = accepted_forcing_binding(&bundle, input);
        let expected_identity = identity(binding, OrganizationRunPurposeV0::Base);
        assert_eq!(result.base_identity, expected_identity);
        assert_eq!(
            result.base_config_hash,
            registered_c_config_hash(
                input,
                &expected_identity,
                Some(result.noncanonical_control_binding_hash),
                binding,
            )
            .unwrap()
        );
    }
}
