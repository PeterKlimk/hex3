//! Canonical wire authority for the preregistered orogen-organization campaign.
//!
//! This module is intentionally passive.  It defines semantic identities and
//! configuration bytes; it does not run H, C, or G and it does not make any
//! configuration selectable product behavior.

use bincode::Options;
use serde::{Deserialize, Serialize};
use std::fmt;

pub const ORGANIZATION_ARM_CONFIG_SCHEMA_VERSION_V0: &str = "orogen-organization-arm-config-v0";
pub const ORGANIZATION_ARTIFACT_HASH_VERSION_V0: &str = "fnv1a64-bincode-fixint-le-v0";
pub const ORGANIZATION_CONFIG_HASH_DOMAIN_V0: &str = "orogen-organization-v0/configuration";
pub const ORGANIZATION_H_CONFIG_SCHEMA_VERSION_V0: &str = "orogen-owner-h-config-v0";
pub const ORGANIZATION_C_CONFIG_SCHEMA_VERSION_V0: &str = "orogen-owner-c-config-v0";
pub const ORGANIZATION_G_CONFIG_SCHEMA_VERSION_V0: &str = "orogen-owner-g-config-v0";

const H_SCHEDULE_HORIZON_MYR: f64 = 10.0;
const H_ACTIVITY_INTEGRAL_MYR: f64 = 5.75;
const C_FORCING_COMPILER_ID: &str = "linked-cosine-support-area-normalized-v0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrganizationArtifactErrorV0(pub String);

impl fmt::Display for OrganizationArtifactErrorV0 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for OrganizationArtifactErrorV0 {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrganizationArmV0 {
    H,
    C,
    G,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrganizationRunPurposeV0 {
    OpportunityControl,
    Base,
    NumericalSensitivity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrganizationArtifactIdentityV0 {
    pub input_bundle_hash: u64,
    pub input_resolution_hash: u64,
    pub nominal_spacing_km: f64,
    pub arm: OrganizationArmV0,
    pub purpose: OrganizationRunPurposeV0,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GReferenceBindingV0 {
    pub result_hash_4km: u64,
    pub native_provenance_hash_4km: u64,
    pub amplitude_a_g_km_inverse: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrganizationPredecessorsV0 {
    pub opportunity_control_result_hash: Option<u64>,
    pub g_reference_4km: Option<GReferenceBindingV0>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingDepressionPolicyV0 {
    PriorityVirtualSurfaceNoBedrockWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowPartitionPolicyV0 {
    MfdSlope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DischargeSupportPolicyV0 {
    UnfilteredC0Physical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplitOrderV0 {
    HoldThenRouteDenudeThenHillslope,
    UpliftThenRouteDenudeThenHillslope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HillslopeBoundaryPolicyV0 {
    LinearDirichletOnOpenFacesClosedElsewhere,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingConfigWireV0 {
    pub depression_policy: RoutingDepressionPolicyV0,
    pub flow_partition: FlowPartitionPolicyV0,
    pub runoff_component_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectiveDenudationConfigWireV0 {
    pub k_km_inverse: f64,
    pub discharge_exponent_m: f64,
    pub slope_exponent_n: f64,
    pub support_policy: DischargeSupportPolicyV0,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearHillslopeConfigWireV0 {
    pub diffusivity_km2_myr: f64,
    pub timestep_safety: f64,
    pub boundary_policy: HillslopeBoundaryPolicyV0,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveIntegrationConfigWireV0 {
    pub maximum_uplift_depth_km: Option<f64>,
    pub minimum_dt_myr: f64,
    pub maximum_adaptive_attempts: u32,
    pub requested_maximum_dt_myr: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActiveProcessAccuracyConfigWireV0 {
    pub maximum_denudation_depth_km: f64,
    pub denudation_slope_courant: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActiveProcessConfigWireV0 {
    pub routing: RoutingConfigWireV0,
    pub denudation: EffectiveDenudationConfigWireV0,
    pub hillslope: LinearHillslopeConfigWireV0,
    pub accuracy: ActiveProcessAccuracyConfigWireV0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HProcessModeV0 {
    TargetOnly,
    HoldAndCarve,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HSchedulePolicyV0 {
    LinkedEpisodeCumulativeActivityFraction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HEndpointPolicyV0 {
    ExactZeroAndOneEndpoints,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HConfigWireV0 {
    pub config_schema: String,
    pub process_mode: HProcessModeV0,
    pub pass_count: u32,
    pub checkpoint_passes: Vec<u32>,
    pub schedule_policy: HSchedulePolicyV0,
    pub endpoint_policy: HEndpointPolicyV0,
    pub schedule_horizon_myr: f64,
    pub activity_integral_myr: f64,
    pub cumulative_displacement_component_hash: u64,
    pub operator_exposure_per_pass_myr: f64,
    pub adaptive_integration: Option<AdaptiveIntegrationConfigWireV0>,
    pub split_order: Option<SplitOrderV0>,
    pub active_process: Option<ActiveProcessConfigWireV0>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CProcessModeV0 {
    UpliftOnly,
    Coevolving,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CForcingSamplingPolicyV0 {
    CandidateMidpointResampledOnEveryRetry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CEndpointClippingPolicyV0 {
    ClipToCheckpointAndFinalEndpoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CVerticalRateAuthorityV0 {
    FreshCompilerEvaluatedF32AtCandidateMidpoint,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CConfigWireV0 {
    pub config_schema: String,
    pub process_mode: CProcessModeV0,
    pub start_time_myr: f64,
    pub forcing_end_time_myr: f64,
    pub end_time_myr: f64,
    pub checkpoint_times_myr: Vec<f64>,
    pub forcing_sampling: CForcingSamplingPolicyV0,
    pub endpoint_clipping: CEndpointClippingPolicyV0,
    pub vertical_rate_authority: CVerticalRateAuthorityV0,
    pub forcing_compiler_id: String,
    pub compiled_stencils_component_hash: u64,
    pub frame_witnesses_component_hash: u64,
    pub cumulative_displacement_component_hash: u64,
    pub adaptive_integration: AdaptiveIntegrationConfigWireV0,
    pub split_order: Option<SplitOrderV0>,
    pub active_process: Option<ActiveProcessConfigWireV0>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GPlanningPolicyV0 {
    InitialPlusCumulativeDisplacement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GForestPolicyV0 {
    MultiSourceMinimaxPortalForest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GQueueOrderV0 {
    PathMaximumTotalCmpPortalCellReceiverKindReceiver,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GAccumulationOrderV0 {
    ReverseFinalization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GReconstructionPolicyV0 {
    ReceiverRecursiveNextUpRunoffConditionedRise,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GCalibrationSolveConfigWireV0 {
    pub initial_upper_a_km_inverse: f64,
    pub bracket_growth_factor: f64,
    pub maximum_bracket_expansions: u32,
    pub maximum_iterations: u32,
    pub volume_absolute_tolerance_km3: f64,
    pub volume_relative_tolerance: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GAmplitudeAuthorityV0 {
    SolveAtThis4Km(GCalibrationSolveConfigWireV0),
    ReuseFrozen4Km(GReferenceBindingV0),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GConfigWireV0 {
    pub config_schema: String,
    pub planning_policy: GPlanningPolicyV0,
    pub forest_policy: GForestPolicyV0,
    pub queue_order: GQueueOrderV0,
    pub accumulation_order: GAccumulationOrderV0,
    pub reconstruction_policy: GReconstructionPolicyV0,
    pub cumulative_displacement_component_hash: u64,
    pub runoff_component_hash: u64,
    pub q_reference_km3_myr: f64,
    pub support_thresholds_km2: Vec<f64>,
    pub amplitude_authority: GAmplitudeAuthorityV0,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrganizationArmConfigPayloadV0 {
    H(HConfigWireV0),
    C(CConfigWireV0),
    G(GConfigWireV0),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrganizationArmConfigV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub identity: OrganizationArtifactIdentityV0,
    pub predecessors: OrganizationPredecessorsV0,
    pub payload: OrganizationArmConfigPayloadV0,
    pub derived_config_hash: u64,
}

#[derive(Serialize)]
struct ConfigurationPreimageV0<'a> {
    domain: &'static str,
    schema_version: &'a str,
    hash_version: &'a str,
    identity: &'a OrganizationArtifactIdentityV0,
    predecessors: &'a OrganizationPredecessorsV0,
    payload: &'a OrganizationArmConfigPayloadV0,
}

/// Computes the canonical configuration hash after checking every locally
/// decidable V0 registration rule. The stored `derived_config_hash` is excluded
/// from the preimage and is validated by roots which embed the configuration.
pub fn organization_arm_config_hash_v0(
    config: &OrganizationArmConfigV0,
) -> Result<u64, OrganizationArtifactErrorV0> {
    validate_organization_arm_config_shape_v0(config)?;
    let preimage = ConfigurationPreimageV0 {
        domain: ORGANIZATION_CONFIG_HASH_DOMAIN_V0,
        schema_version: &config.schema_version,
        hash_version: &config.hash_version,
        identity: &config.identity,
        predecessors: &config.predecessors,
        payload: &config.payload,
    };
    Ok(fnv1a64(&fixed_bytes(&preimage)?))
}

pub(crate) fn validate_organization_arm_config_shape_v0(
    config: &OrganizationArmConfigV0,
) -> Result<(), OrganizationArtifactErrorV0> {
    require(
        config.schema_version == ORGANIZATION_ARM_CONFIG_SCHEMA_VERSION_V0,
        "wrong organization config schema version",
    )?;
    require(
        config.hash_version == ORGANIZATION_ARTIFACT_HASH_VERSION_V0,
        "wrong organization config hash version",
    )?;
    require_registered_spacing(config.identity.nominal_spacing_km)?;
    if config.identity.purpose == OrganizationRunPurposeV0::NumericalSensitivity {
        require(
            config.identity.nominal_spacing_km.to_bits() == 4.0f64.to_bits(),
            "numerical sensitivity is registered only at 4 km",
        )?;
    }

    match (&config.identity.arm, &config.payload) {
        (OrganizationArmV0::H, OrganizationArmConfigPayloadV0::H(payload)) => {
            validate_h_config(config, payload)
        }
        (OrganizationArmV0::C, OrganizationArmConfigPayloadV0::C(payload)) => {
            validate_c_config(config, payload)
        }
        (OrganizationArmV0::G, OrganizationArmConfigPayloadV0::G(payload)) => {
            validate_g_config(config, payload)
        }
        _ => Err(error("arm and configuration payload disagree")),
    }
}

fn validate_h_config(
    config: &OrganizationArmConfigV0,
    payload: &HConfigWireV0,
) -> Result<(), OrganizationArtifactErrorV0> {
    require(
        payload.config_schema == ORGANIZATION_H_CONFIG_SCHEMA_VERSION_V0,
        "wrong H config schema",
    )?;
    require(
        payload.schedule_horizon_myr.to_bits() == H_SCHEDULE_HORIZON_MYR.to_bits()
            && payload.activity_integral_myr.to_bits() == H_ACTIVITY_INTEGRAL_MYR.to_bits(),
        "nonregistered H schedule",
    )?;
    require_canonical_nonnegative(payload.operator_exposure_per_pass_myr, "H exposure")?;
    require(
        config.predecessors.g_reference_4km.is_none(),
        "H has a G predecessor",
    )?;

    let (mode, passes, checkpoints, exposure, active): (HProcessModeV0, u32, &[u32], f64, bool) =
        match config.identity.purpose {
            OrganizationRunPurposeV0::OpportunityControl => (
                HProcessModeV0::TargetOnly,
                200,
                &[0, 50, 120, 200][..],
                0.0,
                false,
            ),
            OrganizationRunPurposeV0::Base => (
                HProcessModeV0::HoldAndCarve,
                200,
                &[0, 50, 120, 200][..],
                0.05,
                true,
            ),
            OrganizationRunPurposeV0::NumericalSensitivity => (
                HProcessModeV0::HoldAndCarve,
                400,
                &[0, 100, 240, 400][..],
                0.025,
                true,
            ),
        };
    require(payload.process_mode == mode, "wrong H process mode")?;
    require(payload.pass_count == passes, "wrong H pass count")?;
    require(
        payload.checkpoint_passes == checkpoints,
        "wrong H checkpoints",
    )?;
    require(
        payload.operator_exposure_per_pass_myr.to_bits() == exposure.to_bits(),
        "wrong H operator exposure",
    )?;
    validate_opportunity_predecessor(config, active)?;
    if active {
        let adaptive = payload
            .adaptive_integration
            .as_ref()
            .ok_or_else(|| error("active H lacks adaptive integration"))?;
        validate_adaptive(adaptive, None, 0.01)?;
        require(
            payload.split_order == Some(SplitOrderV0::HoldThenRouteDenudeThenHillslope),
            "wrong H split order",
        )?;
        validate_active_process(
            payload
                .active_process
                .as_ref()
                .ok_or_else(|| error("active H lacks process config"))?,
        )
    } else {
        require(
            payload.adaptive_integration.is_none()
                && payload.split_order.is_none()
                && payload.active_process.is_none(),
            "H control carries active-process configuration",
        )
    }
}

fn validate_c_config(
    config: &OrganizationArmConfigV0,
    payload: &CConfigWireV0,
) -> Result<(), OrganizationArtifactErrorV0> {
    require(
        payload.config_schema == ORGANIZATION_C_CONFIG_SCHEMA_VERSION_V0,
        "wrong C config schema",
    )?;
    require(
        payload.start_time_myr.to_bits() == 0.0f64.to_bits()
            && payload.forcing_end_time_myr.to_bits() == 6.0f64.to_bits()
            && payload.end_time_myr.to_bits() == 10.0f64.to_bits()
            && exact_float_bits(&payload.checkpoint_times_myr, &[0.0, 3.0, 6.0, 10.0]),
        "nonregistered C time domain",
    )?;
    require(
        payload.forcing_compiler_id == C_FORCING_COMPILER_ID,
        "wrong C forcing compiler",
    )?;
    require(
        config.predecessors.g_reference_4km.is_none(),
        "C has a G predecessor",
    )?;

    let active = config.identity.purpose != OrganizationRunPurposeV0::OpportunityControl;
    let requested_dt = if config.identity.purpose == OrganizationRunPurposeV0::NumericalSensitivity
    {
        0.005
    } else {
        0.01
    };
    validate_adaptive(&payload.adaptive_integration, Some(0.02), requested_dt)?;
    validate_opportunity_predecessor(config, active)?;
    if active {
        require(
            payload.process_mode == CProcessModeV0::Coevolving,
            "wrong C mode",
        )?;
        require(
            payload.split_order == Some(SplitOrderV0::UpliftThenRouteDenudeThenHillslope),
            "wrong C split order",
        )?;
        validate_active_process(
            payload
                .active_process
                .as_ref()
                .ok_or_else(|| error("active C lacks process config"))?,
        )
    } else {
        require(
            payload.process_mode == CProcessModeV0::UpliftOnly,
            "wrong C mode",
        )?;
        require(
            payload.split_order.is_none() && payload.active_process.is_none(),
            "C control carries active-process configuration",
        )
    }
}

fn validate_g_config(
    config: &OrganizationArmConfigV0,
    payload: &GConfigWireV0,
) -> Result<(), OrganizationArtifactErrorV0> {
    require(
        config.identity.purpose == OrganizationRunPurposeV0::Base,
        "G supports only the base purpose",
    )?;
    require(
        config
            .predecessors
            .opportunity_control_result_hash
            .is_none(),
        "G has an opportunity-control predecessor",
    )?;
    require(
        payload.config_schema == ORGANIZATION_G_CONFIG_SCHEMA_VERSION_V0,
        "wrong G config schema",
    )?;
    require(
        payload.q_reference_km3_myr.to_bits() == 500_000.0f64.to_bits()
            && payload.support_thresholds_km2 == [1_000.0, 2_000.0, 4_000.0],
        "nonregistered G support configuration",
    )?;
    match (
        config.identity.nominal_spacing_km,
        &config.predecessors.g_reference_4km,
        &payload.amplitude_authority,
    ) {
        (4.0, None, GAmplitudeAuthorityV0::SolveAtThis4Km(solve)) => validate_g_solve(solve),
        (8.0 | 2.0, Some(predecessor), GAmplitudeAuthorityV0::ReuseFrozen4Km(binding)) => {
            require_canonical_positive(binding.amplitude_a_g_km_inverse, "G amplitude")?;
            require(binding == predecessor, "G reference bindings disagree")
        }
        _ => Err(error("illegal G amplitude authority for spacing")),
    }
}

fn validate_opportunity_predecessor(
    config: &OrganizationArmConfigV0,
    required: bool,
) -> Result<(), OrganizationArtifactErrorV0> {
    require(
        config
            .predecessors
            .opportunity_control_result_hash
            .is_some()
            == required,
        "wrong opportunity-control predecessor presence",
    )
}

fn validate_adaptive(
    value: &AdaptiveIntegrationConfigWireV0,
    maximum_uplift_depth_km: Option<f64>,
    requested_maximum_dt_myr: f64,
) -> Result<(), OrganizationArtifactErrorV0> {
    require(
        value.maximum_uplift_depth_km.as_ref().map(|v| v.to_bits())
            == maximum_uplift_depth_km.as_ref().map(|v| v.to_bits())
            && value.minimum_dt_myr.to_bits() == 1e-8f64.to_bits()
            && value.maximum_adaptive_attempts == 16
            && value.requested_maximum_dt_myr.to_bits() == requested_maximum_dt_myr.to_bits(),
        "nonregistered adaptive integration",
    )
}

fn validate_active_process(
    value: &ActiveProcessConfigWireV0,
) -> Result<(), OrganizationArtifactErrorV0> {
    require(
        value.denudation.k_km_inverse.to_bits() == 1e-4f64.to_bits()
            && value.denudation.discharge_exponent_m.to_bits() == 1.0f64.to_bits()
            && value.denudation.slope_exponent_n.to_bits() == 1.0f64.to_bits()
            && value.hillslope.diffusivity_km2_myr.to_bits() == 0.1f64.to_bits()
            && value.hillslope.timestep_safety.to_bits() == 0.4f64.to_bits()
            && value.accuracy.maximum_denudation_depth_km.to_bits() == 0.02f64.to_bits()
            && value.accuracy.denudation_slope_courant.to_bits() == 0.25f64.to_bits(),
        "nonregistered active-process configuration",
    )
}

fn validate_g_solve(
    value: &GCalibrationSolveConfigWireV0,
) -> Result<(), OrganizationArtifactErrorV0> {
    require(
        value.initial_upper_a_km_inverse.to_bits() == 0.001f64.to_bits()
            && value.bracket_growth_factor.to_bits() == 2.0f64.to_bits()
            && value.maximum_bracket_expansions == 64
            && value.maximum_iterations == 128
            && value.volume_absolute_tolerance_km3.to_bits() == 1e-8f64.to_bits()
            && value.volume_relative_tolerance.to_bits() == 5e-12f64.to_bits(),
        "nonregistered G calibration solve",
    )
}

fn require_registered_spacing(value: f64) -> Result<(), OrganizationArtifactErrorV0> {
    require_canonical_positive(value, "nominal spacing")?;
    require(
        matches!(value, 8.0 | 4.0 | 2.0),
        "unregistered nominal spacing",
    )
}

fn exact_float_bits(values: &[f64], expected: &[f64]) -> bool {
    values.len() == expected.len()
        && values
            .iter()
            .zip(expected)
            .all(|(value, expected)| value.to_bits() == expected.to_bits())
}

fn require_canonical_nonnegative(
    value: f64,
    field: &str,
) -> Result<(), OrganizationArtifactErrorV0> {
    require(
        value.is_finite() && value >= 0.0,
        &format!("invalid {field}"),
    )?;
    require(
        value != 0.0 || value.to_bits() == 0.0f64.to_bits(),
        &format!("noncanonical zero in {field}"),
    )
}

fn require_canonical_positive(value: f64, field: &str) -> Result<(), OrganizationArtifactErrorV0> {
    require(
        value.is_finite() && value > 0.0,
        &format!("invalid {field}"),
    )
}

fn fixed_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, OrganizationArtifactErrorV0> {
    bincode_options()
        .serialize(value)
        .map_err(|error| OrganizationArtifactErrorV0(error.to_string()))
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

fn require(condition: bool, message: &str) -> Result<(), OrganizationArtifactErrorV0> {
    if condition {
        Ok(())
    } else {
        Err(error(message))
    }
}

fn error(message: &str) -> OrganizationArtifactErrorV0 {
    OrganizationArtifactErrorV0(message.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(
        arm: OrganizationArmV0,
        purpose: OrganizationRunPurposeV0,
    ) -> OrganizationArtifactIdentityV0 {
        OrganizationArtifactIdentityV0 {
            input_bundle_hash: 0x11,
            input_resolution_hash: 0x22,
            nominal_spacing_km: 4.0,
            arm,
            purpose,
        }
    }

    fn predecessors(required: bool) -> OrganizationPredecessorsV0 {
        OrganizationPredecessorsV0 {
            opportunity_control_result_hash: required.then_some(0x33),
            g_reference_4km: None,
        }
    }

    fn adaptive(uplift: Option<f64>, maximum_dt: f64) -> AdaptiveIntegrationConfigWireV0 {
        AdaptiveIntegrationConfigWireV0 {
            maximum_uplift_depth_km: uplift,
            minimum_dt_myr: 1e-8,
            maximum_adaptive_attempts: 16,
            requested_maximum_dt_myr: maximum_dt,
        }
    }

    fn active() -> ActiveProcessConfigWireV0 {
        ActiveProcessConfigWireV0 {
            routing: RoutingConfigWireV0 {
                depression_policy: RoutingDepressionPolicyV0::PriorityVirtualSurfaceNoBedrockWrite,
                flow_partition: FlowPartitionPolicyV0::MfdSlope,
                runoff_component_hash: 0x44,
            },
            denudation: EffectiveDenudationConfigWireV0 {
                k_km_inverse: 1e-4,
                discharge_exponent_m: 1.0,
                slope_exponent_n: 1.0,
                support_policy: DischargeSupportPolicyV0::UnfilteredC0Physical,
            },
            hillslope: LinearHillslopeConfigWireV0 {
                diffusivity_km2_myr: 0.1,
                timestep_safety: 0.4,
                boundary_policy:
                    HillslopeBoundaryPolicyV0::LinearDirichletOnOpenFacesClosedElsewhere,
            },
            accuracy: ActiveProcessAccuracyConfigWireV0 {
                maximum_denudation_depth_km: 0.02,
                denudation_slope_courant: 0.25,
            },
        }
    }

    fn h_config(purpose: OrganizationRunPurposeV0) -> OrganizationArmConfigV0 {
        let active_run = purpose != OrganizationRunPurposeV0::OpportunityControl;
        let sensitivity = purpose == OrganizationRunPurposeV0::NumericalSensitivity;
        let (pass_count, checkpoint_passes, exposure) = if sensitivity {
            (400, vec![0, 100, 240, 400], 0.025)
        } else {
            (
                200,
                vec![0, 50, 120, 200],
                if active_run { 0.05 } else { 0.0 },
            )
        };
        OrganizationArmConfigV0 {
            schema_version: ORGANIZATION_ARM_CONFIG_SCHEMA_VERSION_V0.into(),
            hash_version: ORGANIZATION_ARTIFACT_HASH_VERSION_V0.into(),
            identity: identity(OrganizationArmV0::H, purpose),
            predecessors: predecessors(active_run),
            payload: OrganizationArmConfigPayloadV0::H(HConfigWireV0 {
                config_schema: ORGANIZATION_H_CONFIG_SCHEMA_VERSION_V0.into(),
                process_mode: if active_run {
                    HProcessModeV0::HoldAndCarve
                } else {
                    HProcessModeV0::TargetOnly
                },
                pass_count,
                checkpoint_passes,
                schedule_policy: HSchedulePolicyV0::LinkedEpisodeCumulativeActivityFraction,
                endpoint_policy: HEndpointPolicyV0::ExactZeroAndOneEndpoints,
                schedule_horizon_myr: 10.0,
                activity_integral_myr: 5.75,
                cumulative_displacement_component_hash: 0x55,
                operator_exposure_per_pass_myr: exposure,
                adaptive_integration: active_run.then(|| adaptive(None, 0.01)),
                split_order: active_run.then_some(SplitOrderV0::HoldThenRouteDenudeThenHillslope),
                active_process: active_run.then(active),
            }),
            derived_config_hash: 0,
        }
    }

    fn c_config(purpose: OrganizationRunPurposeV0) -> OrganizationArmConfigV0 {
        let active_run = purpose != OrganizationRunPurposeV0::OpportunityControl;
        OrganizationArmConfigV0 {
            schema_version: ORGANIZATION_ARM_CONFIG_SCHEMA_VERSION_V0.into(),
            hash_version: ORGANIZATION_ARTIFACT_HASH_VERSION_V0.into(),
            identity: identity(OrganizationArmV0::C, purpose),
            predecessors: predecessors(active_run),
            payload: OrganizationArmConfigPayloadV0::C(CConfigWireV0 {
                config_schema: ORGANIZATION_C_CONFIG_SCHEMA_VERSION_V0.into(),
                process_mode: if active_run {
                    CProcessModeV0::Coevolving
                } else {
                    CProcessModeV0::UpliftOnly
                },
                start_time_myr: 0.0,
                forcing_end_time_myr: 6.0,
                end_time_myr: 10.0,
                checkpoint_times_myr: vec![0.0, 3.0, 6.0, 10.0],
                forcing_sampling: CForcingSamplingPolicyV0::CandidateMidpointResampledOnEveryRetry,
                endpoint_clipping: CEndpointClippingPolicyV0::ClipToCheckpointAndFinalEndpoint,
                vertical_rate_authority:
                    CVerticalRateAuthorityV0::FreshCompilerEvaluatedF32AtCandidateMidpoint,
                forcing_compiler_id: C_FORCING_COMPILER_ID.into(),
                compiled_stencils_component_hash: 0x66,
                frame_witnesses_component_hash: 0x77,
                cumulative_displacement_component_hash: 0x55,
                adaptive_integration: adaptive(
                    Some(0.02),
                    if purpose == OrganizationRunPurposeV0::NumericalSensitivity {
                        0.005
                    } else {
                        0.01
                    },
                ),
                split_order: active_run.then_some(SplitOrderV0::UpliftThenRouteDenudeThenHillslope),
                active_process: active_run.then(active),
            }),
            derived_config_hash: 0,
        }
    }

    fn g_config(spacing: f64) -> OrganizationArmConfigV0 {
        let reference = GReferenceBindingV0 {
            result_hash_4km: 0x88,
            native_provenance_hash_4km: 0x99,
            amplitude_a_g_km_inverse: 0.003,
        };
        let reuse = spacing != 4.0;
        let mut value = OrganizationArmConfigV0 {
            schema_version: ORGANIZATION_ARM_CONFIG_SCHEMA_VERSION_V0.into(),
            hash_version: ORGANIZATION_ARTIFACT_HASH_VERSION_V0.into(),
            identity: identity(OrganizationArmV0::G, OrganizationRunPurposeV0::Base),
            predecessors: OrganizationPredecessorsV0 {
                opportunity_control_result_hash: None,
                g_reference_4km: reuse.then(|| reference.clone()),
            },
            payload: OrganizationArmConfigPayloadV0::G(GConfigWireV0 {
                config_schema: ORGANIZATION_G_CONFIG_SCHEMA_VERSION_V0.into(),
                planning_policy: GPlanningPolicyV0::InitialPlusCumulativeDisplacement,
                forest_policy: GForestPolicyV0::MultiSourceMinimaxPortalForest,
                queue_order: GQueueOrderV0::PathMaximumTotalCmpPortalCellReceiverKindReceiver,
                accumulation_order: GAccumulationOrderV0::ReverseFinalization,
                reconstruction_policy:
                    GReconstructionPolicyV0::ReceiverRecursiveNextUpRunoffConditionedRise,
                cumulative_displacement_component_hash: 0x55,
                runoff_component_hash: 0x44,
                q_reference_km3_myr: 500_000.0,
                support_thresholds_km2: vec![1_000.0, 2_000.0, 4_000.0],
                amplitude_authority: if reuse {
                    GAmplitudeAuthorityV0::ReuseFrozen4Km(reference)
                } else {
                    GAmplitudeAuthorityV0::SolveAtThis4Km(GCalibrationSolveConfigWireV0 {
                        initial_upper_a_km_inverse: 0.001,
                        bracket_growth_factor: 2.0,
                        maximum_bracket_expansions: 64,
                        maximum_iterations: 128,
                        volume_absolute_tolerance_km3: 1e-8,
                        volume_relative_tolerance: 5e-12,
                    })
                },
            }),
            derived_config_hash: 0,
        };
        value.identity.nominal_spacing_km = spacing;
        value
    }

    #[test]
    fn every_registered_arm_purpose_shape_hashes() {
        for config in [
            h_config(OrganizationRunPurposeV0::OpportunityControl),
            h_config(OrganizationRunPurposeV0::Base),
            h_config(OrganizationRunPurposeV0::NumericalSensitivity),
            c_config(OrganizationRunPurposeV0::OpportunityControl),
            c_config(OrganizationRunPurposeV0::Base),
            c_config(OrganizationRunPurposeV0::NumericalSensitivity),
            g_config(8.0),
            g_config(4.0),
            g_config(2.0),
        ] {
            let first = organization_arm_config_hash_v0(&config).unwrap();
            assert_eq!(organization_arm_config_hash_v0(&config).unwrap(), first);
        }
    }

    #[test]
    fn base_h_configuration_has_frozen_known_hash() {
        assert_eq!(
            organization_arm_config_hash_v0(&h_config(OrganizationRunPurposeV0::Base)).unwrap(),
            0x680f_994b_0d85_34df
        );
    }

    #[test]
    fn c_and_g_configurations_have_frozen_known_hashes() {
        for (config, expected) in [
            (
                c_config(OrganizationRunPurposeV0::OpportunityControl),
                0xc54b_0bcc_526e_c8d0,
            ),
            (
                c_config(OrganizationRunPurposeV0::Base),
                0x4049_4edc_a992_5a94,
            ),
            (
                c_config(OrganizationRunPurposeV0::NumericalSensitivity),
                0x2d9e_fb24_849c_866d,
            ),
            (g_config(4.0), 0xea70_5195_e214_bb19),
            (g_config(8.0), 0xdd97_bb10_8c2c_1c0e),
            (g_config(2.0), 0x2a70_a762_dd50_f5ae),
        ] {
            assert_eq!(organization_arm_config_hash_v0(&config).unwrap(), expected);
        }
    }

    #[test]
    fn frozen_enum_discriminants_are_fixed_u32_little_endian() {
        for (value, ordinal) in [
            (OrganizationArmV0::H, 0u32),
            (OrganizationArmV0::C, 1),
            (OrganizationArmV0::G, 2),
        ] {
            assert_eq!(fixed_bytes(&value).unwrap(), ordinal.to_le_bytes());
        }
        for (value, ordinal) in [
            (OrganizationRunPurposeV0::OpportunityControl, 0u32),
            (OrganizationRunPurposeV0::Base, 1),
            (OrganizationRunPurposeV0::NumericalSensitivity, 2),
        ] {
            assert_eq!(fixed_bytes(&value).unwrap(), ordinal.to_le_bytes());
        }
        assert_eq!(
            fixed_bytes(&SplitOrderV0::HoldThenRouteDenudeThenHillslope).unwrap(),
            0u32.to_le_bytes()
        );
        assert_eq!(
            fixed_bytes(&SplitOrderV0::UpliftThenRouteDenudeThenHillslope).unwrap(),
            1u32.to_le_bytes()
        );

        for (config, payload_ordinal, amplitude_ordinal) in [
            (h_config(OrganizationRunPurposeV0::Base), 0u32, None),
            (c_config(OrganizationRunPurposeV0::Base), 1u32, None),
            (g_config(4.0), 2u32, Some(0u32)),
            (g_config(8.0), 2u32, Some(1u32)),
        ] {
            let payload_bytes = fixed_bytes(&config.payload).unwrap();
            assert_eq!(&payload_bytes[..4], &payload_ordinal.to_le_bytes());
            if let (Some(amplitude_ordinal), OrganizationArmConfigPayloadV0::G(payload)) =
                (amplitude_ordinal, config.payload)
            {
                let amplitude_bytes = fixed_bytes(&payload.amplitude_authority).unwrap();
                assert_eq!(&amplitude_bytes[..4], &amplitude_ordinal.to_le_bytes());
            }
        }
    }

    #[test]
    fn numerical_sensitivity_is_registered_only_at_4km() {
        for spacing in [8.0, 2.0] {
            let mut h = h_config(OrganizationRunPurposeV0::NumericalSensitivity);
            h.identity.nominal_spacing_km = spacing;
            assert!(organization_arm_config_hash_v0(&h).is_err());

            let mut c = c_config(OrganizationRunPurposeV0::NumericalSensitivity);
            c.identity.nominal_spacing_km = spacing;
            assert!(organization_arm_config_hash_v0(&c).is_err());
        }
    }

    #[test]
    fn config_hash_excludes_only_its_derived_hash() {
        let mut config = h_config(OrganizationRunPurposeV0::Base);
        let expected = organization_arm_config_hash_v0(&config).unwrap();
        config.derived_config_hash = u64::MAX;
        assert_eq!(organization_arm_config_hash_v0(&config).unwrap(), expected);
        if let OrganizationArmConfigPayloadV0::H(payload) = &mut config.payload {
            payload.cumulative_displacement_component_hash ^= 1;
        }
        assert_ne!(organization_arm_config_hash_v0(&config).unwrap(), expected);
    }

    #[test]
    fn illegal_purpose_predecessor_and_payload_combinations_reject() {
        let mut h = h_config(OrganizationRunPurposeV0::Base);
        h.predecessors.opportunity_control_result_hash = None;
        assert!(organization_arm_config_hash_v0(&h).is_err());

        let mut c = c_config(OrganizationRunPurposeV0::OpportunityControl);
        c.identity.arm = OrganizationArmV0::H;
        assert!(organization_arm_config_hash_v0(&c).is_err());

        let mut g = g_config(4.0);
        g.identity.purpose = OrganizationRunPurposeV0::NumericalSensitivity;
        assert!(organization_arm_config_hash_v0(&g).is_err());

        let mut reuse = g_config(8.0);
        if let OrganizationArmConfigPayloadV0::G(payload) = &mut reuse.payload {
            if let GAmplitudeAuthorityV0::ReuseFrozen4Km(binding) = &mut payload.amplitude_authority
            {
                binding.result_hash_4km ^= 1;
            }
        }
        assert!(organization_arm_config_hash_v0(&reuse).is_err());
    }

    #[test]
    fn illegal_active_options_and_g_authorities_reject() {
        let mut h = h_config(OrganizationRunPurposeV0::OpportunityControl);
        if let OrganizationArmConfigPayloadV0::H(payload) = &mut h.payload {
            payload.active_process = Some(active());
        }
        assert!(organization_arm_config_hash_v0(&h).is_err());

        let mut h = h_config(OrganizationRunPurposeV0::Base);
        h.predecessors.g_reference_4km = Some(GReferenceBindingV0 {
            result_hash_4km: 1,
            native_provenance_hash_4km: 2,
            amplitude_a_g_km_inverse: 0.003,
        });
        assert!(organization_arm_config_hash_v0(&h).is_err());

        let mut c = c_config(OrganizationRunPurposeV0::OpportunityControl);
        if let OrganizationArmConfigPayloadV0::C(payload) = &mut c.payload {
            payload.split_order = Some(SplitOrderV0::UpliftThenRouteDenudeThenHillslope);
        }
        assert!(organization_arm_config_hash_v0(&c).is_err());

        let mut g = g_config(4.0);
        let reference = GReferenceBindingV0 {
            result_hash_4km: 1,
            native_provenance_hash_4km: 2,
            amplitude_a_g_km_inverse: 0.003,
        };
        g.predecessors.g_reference_4km = Some(reference.clone());
        if let OrganizationArmConfigPayloadV0::G(payload) = &mut g.payload {
            payload.amplitude_authority = GAmplitudeAuthorityV0::ReuseFrozen4Km(reference);
        }
        assert!(organization_arm_config_hash_v0(&g).is_err());

        let mut g = g_config(8.0);
        g.predecessors.g_reference_4km = None;
        if let OrganizationArmConfigPayloadV0::G(payload) = &mut g.payload {
            payload.amplitude_authority =
                GAmplitudeAuthorityV0::SolveAtThis4Km(GCalibrationSolveConfigWireV0 {
                    initial_upper_a_km_inverse: 0.001,
                    bracket_growth_factor: 2.0,
                    maximum_bracket_expansions: 64,
                    maximum_iterations: 128,
                    volume_absolute_tolerance_km3: 1e-8,
                    volume_relative_tolerance: 5e-12,
                });
        }
        assert!(organization_arm_config_hash_v0(&g).is_err());

        let mut g = g_config(8.0);
        if let Some(reference) = &mut g.predecessors.g_reference_4km {
            reference.amplitude_a_g_km_inverse = 0.0;
        }
        if let OrganizationArmConfigPayloadV0::G(payload) = &mut g.payload {
            if let GAmplitudeAuthorityV0::ReuseFrozen4Km(reference) =
                &mut payload.amplitude_authority
            {
                reference.amplitude_a_g_km_inverse = 0.0;
            }
        }
        assert!(organization_arm_config_hash_v0(&g).is_err());

        let mut g = g_config(4.0);
        g.predecessors.opportunity_control_result_hash = Some(1);
        assert!(organization_arm_config_hash_v0(&g).is_err());
    }

    #[test]
    fn noncanonical_or_unregistered_floats_reject() {
        let mut h = h_config(OrganizationRunPurposeV0::OpportunityControl);
        if let OrganizationArmConfigPayloadV0::H(payload) = &mut h.payload {
            payload.operator_exposure_per_pass_myr = -0.0;
        }
        assert!(organization_arm_config_hash_v0(&h).is_err());

        let mut c = c_config(OrganizationRunPurposeV0::Base);
        c.identity.nominal_spacing_km = f64::NAN;
        assert!(organization_arm_config_hash_v0(&c).is_err());

        let mut c = c_config(OrganizationRunPurposeV0::Base);
        if let OrganizationArmConfigPayloadV0::C(payload) = &mut c.payload {
            payload.checkpoint_times_myr[0] = -0.0;
        }
        assert!(organization_arm_config_hash_v0(&c).is_err());

        let mut g = g_config(4.0);
        if let OrganizationArmConfigPayloadV0::G(payload) = &mut g.payload {
            payload.q_reference_km3_myr = f64::INFINITY;
        }
        assert!(organization_arm_config_hash_v0(&g).is_err());
    }
}
