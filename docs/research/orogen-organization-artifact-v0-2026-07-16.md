# Organization-owner artifact and provenance amendment V0

**Date:** 2026-07-16

**Status:** executable preregistration for semantic arm artifacts, native
provenance, traces, checkpoints, failures and atomic publication; not
implemented and not an H/C/G result

**Parent:** [organization-owner comparison design V0](orogen-organization-owner-v0-2026-07-16.md)

**Accepted predecessors:** [linked shared-input V0](orogen-linked-shared-input-v0-2026-07-15.md),
[common planar evidence-core V0](landform-common-core-v0-2026-07-15.md)

## Decision

Use an acyclic semantic artifact graph:

```text
accepted linked input
  -> registered arm configuration
  -> trace/checkpoint artifacts
  -> native provenance
  -> authoritative arm result
  -> later independent evidence
```

The arm result is deliberately thin. Native routing, graphs, ledgers, solver
coordinates and checkpoints never enter it. Native provenance binds the final
elevation **component hash**, not the arm-result hash; the arm result then binds
the native-provenance hash. This prevents a result/provenance hash cycle.

Three validation strengths remain distinct:

1. **standalone integrity** proves registered shape, canonical values, local
   algebra and hashes;
2. **predecessor consistency** proves exact binding to the supplied accepted
   input, opportunity/reference results, traces, checkpoints and mutually
   consistent result/provenance; and
3. **deterministic replay** runs the registered arm and requires identical
   semantic values and bytes.

Only replay establishes that the declared algorithm generated an internally
consistent final surface. FNV-1a is a deterministic identity checksum, not
authentication or proof of execution.

This amendment freezes schemas and equations. The subsequent numerical/admission
[amendment](orogen-organization-numerical-v0-2026-07-16.md) now owns closure
tolerances, the exact H activity primitive, the G amplitude solve, direct
sensitivity reductions and numerical admission. No arm implementation may
begin until the remaining evidence and presentation contracts are frozen.

## Corrections to the parent design

This executable amendment makes four parent-envelope details exact:

- H and C publish separate opportunity-control results before base or
  sensitivity results. G has no separable opportunity-control result: its
  reconstruction is its calibration and base result.
- G's 4 km calibration/base must be generated and frozen before G at 8 or
  2 km. The latter bind the exact 4 km result/native hashes and reuse the same
  `a_G` bits. The generic parent wording “8 km, then 4 km” does not apply to G.
- a successful run publishes result plus completed native provenance. A failed
  run publishes a typed failure root and **no** `arm-result.bin` or completed
  `native-provenance.bin`.
- scalar diagnostics for every accepted H/C substep live in a separately
  hashed trace artifact. Full declared H/C surfaces and G construction stages
  live in checkpoint artifacts rather than being hidden in JSON or the run
  envelope.

## Registered versions and limits

```text
semantic hash encoding       fnv1a64-bincode-fixint-le-v0
arm result schema            orogen-organization-arm-result-v0
arm config schema            orogen-organization-arm-config-v0
native provenance schema     orogen-organization-native-provenance-v0
step trace schema            orogen-organization-step-trace-v0
checkpoint schema            orogen-organization-checkpoint-v0
failure schema               orogen-organization-run-failure-v0
JSON projection family       orogen-organization-json-v0
run envelope schema          orogen-organization-run-envelope-v0

maximum arm-result bytes       16 MiB
maximum native bytes            8 MiB
maximum trace bytes            64 MiB
maximum one-checkpoint bytes   128 MiB
maximum failure bytes            8 MiB
maximum JSON file bytes         16 MiB
maximum checkpoint count         8
maximum cell count          250,000
maximum accepted-step count    100,000
maximum attempt count per step      16
maximum H pass count               400
maximum portal count                16
maximum support-threshold count      8
maximum claim count                  16
maximum semantic string bytes       128
```

Every semantic decoder checks the raw byte length before deserialization, uses
fixed-integer little-endian bincode with the matching `with_limit`, rejects
trailing bytes, and then performs standalone validation. Because `with_limit`
alone does not prevent a forged sequence length from driving an early reserve,
all wire `Vec`/`String` fields use bounded serde visitors (or an equivalent
preflight framing pass) that reject the announced element count before
allocation. Wire integers are fixed-width (`u32`/`u64`), never `usize`. No wire
struct contains a map, set, path from the host, human error string or platform-
dependent debug value.

All semantic floats are finite. Every zero is canonical positive zero,
including signed transfers whose nonzero values may be negative. Nonnegative
quantities additionally reject negative values. Runtime `+infinity` timestep
limits are encoded as `None`, never as a nonfinite float. Vectors use the exact
orders declared below; enum declaration order is wire order. A schema or order
change requires a new version.

## Common wire vocabulary

These Rust-like declarations freeze serialization field and enum order. All
strings shown as IDs are exact registered constants validated by value.

```rust
pub enum OrganizationArmV0 { H, C, G }

pub enum OrganizationRunPurposeV0 {
    OpportunityControl,
    Base,
    NumericalSensitivity,
}

pub struct OrganizationArtifactIdentityV0 {
    pub input_bundle_hash: u64,
    pub input_resolution_hash: u64,
    pub nominal_spacing_km: f64,
    pub arm: OrganizationArmV0,
    pub purpose: OrganizationRunPurposeV0,
}

pub struct GReferenceBindingV0 {
    pub result_hash_4km: u64,
    pub native_provenance_hash_4km: u64,
    pub amplitude_a_g_km_inverse: f64,
}

pub struct OrganizationPredecessorsV0 {
    pub opportunity_control_result_hash: Option<u64>,
    pub g_reference_4km: Option<GReferenceBindingV0>,
}

pub enum RoutingDepressionPolicyV0 { PriorityVirtualSurfaceNoBedrockWrite }
pub enum FlowPartitionPolicyV0 { MfdSlope }
pub enum DischargeSupportPolicyV0 { UnfilteredC0Physical }
pub enum SplitOrderV0 { HoldThenRouteDenudeThenHillslope, UpliftThenRouteDenudeThenHillslope }
pub enum HillslopeBoundaryPolicyV0 { LinearDirichletOnOpenFacesClosedElsewhere }

pub struct RoutingConfigWireV0 {
    pub depression_policy: RoutingDepressionPolicyV0,
    pub flow_partition: FlowPartitionPolicyV0,
    pub runoff_component_hash: u64,
}

pub struct EffectiveDenudationConfigWireV0 {
    pub k_km_inverse: f64,
    pub discharge_exponent_m: f64,
    pub slope_exponent_n: f64,
    pub support_policy: DischargeSupportPolicyV0,
}

pub struct LinearHillslopeConfigWireV0 {
    pub diffusivity_km2_myr: f64,
    pub timestep_safety: f64,
    pub boundary_policy: HillslopeBoundaryPolicyV0,
}

pub struct AdaptiveIntegrationConfigWireV0 {
    pub maximum_uplift_depth_km: Option<f64>,
    pub minimum_dt_myr: f64,
    pub maximum_adaptive_attempts: u32,
    pub requested_maximum_dt_myr: f64,
}

pub struct ActiveProcessAccuracyConfigWireV0 {
    pub maximum_denudation_depth_km: f64,
    pub denudation_slope_courant: f64,
}

pub struct ActiveProcessConfigWireV0 {
    pub routing: RoutingConfigWireV0,
    pub denudation: EffectiveDenudationConfigWireV0,
    pub hillslope: LinearHillslopeConfigWireV0,
    pub accuracy: ActiveProcessAccuracyConfigWireV0,
}
```

Registered base active-process values are the parent values: `K=1e-4 km^-1`,
`m=n=1`, linear `D_h=0.1 km2/Myr`, maximum uplift/denudation depth `0.02 km`,
denudation Courant `0.25`, minimum dt `1e-8 Myr`, 16 attempts, hillslope safety
`0.4`, requested maximum dt `0.01 Myr`, exact stored runoff and unfiltered
discharge. The C numerical sensitivity changes only requested maximum dt to
`0.005 Myr`. H's sensitivity changes pass/exposure partition as declared below.

## Exact arm configurations

```rust
pub enum HProcessModeV0 { TargetOnly, HoldAndCarve }
pub enum HSchedulePolicyV0 { LinkedEpisodeCumulativeActivityFraction }
pub enum HEndpointPolicyV0 { ExactZeroAndOneEndpoints }

pub struct HConfigWireV0 {
    pub config_schema: String, // "orogen-owner-h-config-v0"
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

pub enum CProcessModeV0 { UpliftOnly, Coevolving }
pub enum CForcingSamplingPolicyV0 { CandidateMidpointResampledOnEveryRetry }
pub enum CEndpointClippingPolicyV0 { ClipToCheckpointAndFinalEndpoint }
pub enum CVerticalRateAuthorityV0 { FreshCompilerEvaluatedF32AtCandidateMidpoint }

pub struct CConfigWireV0 {
    pub config_schema: String, // "orogen-owner-c-config-v0"
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

pub enum GPlanningPolicyV0 { InitialPlusCumulativeDisplacement }
pub enum GForestPolicyV0 { MultiSourceMinimaxPortalForest }
pub enum GQueueOrderV0 { PathMaximumTotalCmpPortalCellReceiverKindReceiver }
pub enum GAccumulationOrderV0 { ReverseFinalization }
pub enum GReconstructionPolicyV0 { ReceiverRecursiveNextUpRunoffConditionedRise }

pub struct GCalibrationSolveConfigWireV0 {
    pub initial_upper_a_km_inverse: f64,
    pub bracket_growth_factor: f64,
    pub maximum_bracket_expansions: u32,
    pub maximum_iterations: u32,
    pub volume_absolute_tolerance_km3: f64,
    pub volume_relative_tolerance: f64,
}

pub enum GAmplitudeAuthorityV0 {
    SolveAtThis4Km(GCalibrationSolveConfigWireV0),
    ReuseFrozen4Km(GReferenceBindingV0),
}

pub struct GConfigWireV0 {
    pub config_schema: String, // "orogen-owner-g-config-v0"
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

pub enum OrganizationArmConfigPayloadV0 {
    H(HConfigWireV0),
    C(CConfigWireV0),
    G(GConfigWireV0),
}

pub struct OrganizationArmConfigV0 {
    pub schema_version: String, // "orogen-organization-arm-config-v0"
    pub hash_version: String,
    pub identity: OrganizationArtifactIdentityV0,
    pub predecessors: OrganizationPredecessorsV0,
    pub payload: OrganizationArmConfigPayloadV0,
    pub derived_config_hash: u64,
}
```

Allowed configurations are exact:

- H opportunity control: 200 passes, `[0,50,120,200]`, target-only, zero
  exposure and no adaptive-integration, split-order or active-process payload;
- H base: 200 passes, the same checkpoints, `0.05 operator-Myr` per pass and
  the registered adaptive integration (`maximum_uplift_depth=None`) and active
  process with `HoldThenRouteDenudeThenHillslope`;
- H sensitivity: 400 passes, `[0,100,240,400]`, `0.025 operator-Myr` per pass
  and otherwise identical active process;
- C opportunity control: `0..10 Myr`, forcing ends at 6 Myr, checkpoints
  `[0,3,6,10]`, uplift-only and no split-order or active-process payload;
- C base/sensitivity: the same times with coevolution and their registered
  active-process values and `UpliftThenRouteDenudeThenHillslope`; and
- G: base only. `SolveAtThis4Km` is legal only at 4 km;
  `ReuseFrozen4Km` is mandatory at 8/2 km and must equal both the predecessor
  binding and the `a_G` in the supplied 4 km provenance. `SolveAtThis4Km` uses
  exact values `0.001` initial upper amplitude, `2.0` bracket growth, 64
  expansions, 128 midpoint iterations, `1e-8 km3` absolute volume tolerance
  and `5e-12` relative tolerance as frozen by the
  [numerical/admission amendment](orogen-organization-numerical-v0-2026-07-16.md).

H/C base and sensitivity configurations require the matching same-arm,
same-resolution opportunity-control result hash. Opportunity controls require
no predecessor. G requires no opportunity predecessor. No response-case input
is representable in V0.

Every C purpose retains adaptive integration with
`maximum_uplift_depth=Some(0.02 km)`, minimum dt `1e-8 Myr`, 16 attempts and
its registered requested maximum dt. `active_process=None` disables only
routing/denudation/hillslopes; it never erases the uplift integrator. Active H
uses the same minimum dt/attempt/requested-dt values with no uplift limit.

## Opportunity audit and authoritative result

`V` below is exactly the f64 bits copied from
`input.declaration.analytic_rock_volume_km3` in the predecessor bundle
(`100625.0` for accepted V0). It is not recomputed from a resolution's
cumulative-displacement and cell-area arrays. The exact input-side discrete
work is separately copied from the selected predecessor's
`LinkedResolutionInputV0.summary.cumulative_rock_volume_km3`. Result-side
`V_pos` is instead reduced from the emitted opportunity surface's
add-then-subtract differences; f64 arithmetic does not require it to equal the
input-side `sum(D[i]*A[i])` bit-for-bit. Both may differ from `V`; the result
audit retains `V_pos-V` and never silently selects resolution-specific
declared-work bits.

For any opportunity surface `z_op`, use the following exact f64 reductions.
Every accumulator starts at positive zero. Each assignment is an ordinary
nonfused operation with the shown parentheses, and the loop is stored cell
order:

```text
for i = 0..N:
  d = z_op[i] - z0[i]
  if d > 0:
    w = d * A[i]
    V_pos = V_pos + w
    support_area = support_area + A[i]
    support_count += 1
    min_d = min(previous_or_d, d)
    max_d = max(previous_or_d, d)
    sum_x = sum_x + (w * x[i])
    sum_y = sum_y + (w * y[i])
  else if d < 0:
    V_neg = V_neg + ((-d) * A[i])

signed_volume = V_pos - V_neg
cx = sum_x / V_pos
cy = sum_y / V_pos

for i = 0..N where (z_op[i] - z0[i]) > 0:
  d = z_op[i] - z0[i]
  w = d * A[i]
  dx = x[i] - cx
  dy = y[i] - cy
  sum_xx = sum_xx + ((w * dx) * dx)
  sum_xy = sum_xy + ((w * dx) * dy)
  sum_yy = sum_yy + ((w * dy) * dy)

covariance = [sum_xx / V_pos, sum_xy / V_pos, sum_yy / V_pos]
```

The centroid/covariance are `None` exactly when `V_pos==0`; support min/max are
`None` exactly when the positive-support count is zero. C displacement errors
use `e_i=(z_op[i]-z0[i])-D[i]` and these exact stored-order reductions:

```text
area_total = sum_i A[i]
max_error = max_i abs(e_i)
l1_error  = (sum_i (abs(e_i) * A[i])) / area_total
rms_error = sqrt((sum_i ((e_i * e_i) * A[i])) / area_total)
```

H target residual is `V_pos-V`. C uplift residual is its trace-summed rock
uplift moment minus `V`. G work residual is `V_pos-V`. G's
`next_up_only_positive_volume_km3` reconstructs over the identical frozen forest
with exact positive-zero `a=0`, then applies the same first-pass `V_pos`
reduction. The numerical amendment may register acceptance tolerances, but may
not change these hashed reductions after a result exists.

```rust
pub struct PositiveOpportunityMomentsV0 {
    pub positive_volume_km3: f64,
    pub negative_volume_km3: f64,
    pub signed_volume_km3: f64,
    pub positive_support_cell_count: u64,
    pub positive_support_area_km2: f64,
    pub minimum_positive_addition_km: Option<f64>,
    pub maximum_positive_addition_km: Option<f64>,
    pub positive_volume_centroid_km: Option<[f64; 2]>,
    pub positive_volume_covariance_km2: Option<[f64; 3]>,
}

pub enum OpportunityAuthorityV0 {
    SelfResult,
    SeparateControl { result_hash: u64 },
}

pub struct HOpportunityAuditV0 {
    pub target_elevation_component_hash: u64,
    pub target_positive_volume_km3: f64,
    pub target_minus_declared_work_km3: f64,
}

pub struct COpportunityAuditV0 {
    pub uplift_only_elevation_component_hash: u64,
    pub integrated_rock_uplift_moment_km3: f64,
    pub positive_volume_minus_declared_work_km3: f64,
    pub uplift_minus_declared_work_km3: f64,
    pub maximum_displacement_error_km: f64,
    pub area_weighted_l1_displacement_error_km: f64,
    pub area_weighted_rms_displacement_error_km: f64,
}

pub struct GCalibrationSolveAuditV0 {
    pub bracket_lower_a_km_inverse: f64,
    pub bracket_upper_a_km_inverse: f64,
    pub bracket_expansion_count: u32,
    pub iteration_count: u32,
    pub termination_residual_km3: f64,
}

pub struct GOpportunityAuditV0 {
    pub amplitude_a_g_km_inverse: f64,
    pub calibration_spacing_km: f64,
    pub reconstructed_elevation_component_hash: u64,
    pub positive_volume_minus_declared_work_km3: f64,
    pub solve: Option<GCalibrationSolveAuditV0>,
    pub next_up_only_positive_volume_km3: f64,
}

pub enum OpportunityAuditPayloadV0 {
    H(HOpportunityAuditV0),
    C(COpportunityAuditV0),
    G(GOpportunityAuditV0),
}

pub struct OpportunityAuditContentV0 {
    pub declared_work_volume_km3: f64,
    pub moments: PositiveOpportunityMomentsV0,
    pub payload: OpportunityAuditPayloadV0,
}

pub struct OpportunityAuditV0 {
    pub opportunity_authority: OpportunityAuthorityV0,
    pub content: OpportunityAuditContentV0,
}

pub struct OrganizationArmResultV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub identity: OrganizationArtifactIdentityV0,
    pub predecessors: OrganizationPredecessorsV0,
    pub initial_elevation_component_hash: u64,
    pub final_elevation_component_hash: u64,
    pub final_elevation_km: Vec<f64>,
    pub opportunity_audit: OpportunityAuditV0,
    pub native_provenance_hash: u64,
    pub derived_result_hash: u64,
}
```

For H/C opportunity controls, authority is `SelfResult`; the final surface is
the opportunity surface and the arm-specific opportunity component hash equals
the result final-elevation component hash. H/C base and sensitivity results use
`SeparateControl`, and their audit content must byte-equal the supplied
control's content.
For G, authority is always `SelfResult`: reconstruction is both opportunity
conversion and base surface. The 8/2 G audit may report work-volume drift but
cannot recalibrate it. G 4 km stores `solve=Some`; G 8/2 stores `solve=None`
rather than fake bracket/iteration zeroes.

For H/C base/sensitivity, `content` must byte-equal the supplied control's
`content`; authority differs deliberately. The `SeparateControl.result_hash`, result predecessor,
configuration predecessor and native-provenance predecessor must be the same
bits. For G 8/2, all duplicated 4 km bindings and amplitude bits must likewise
agree exactly.

The result contains no configuration, native receiver, portal assignment, flow
fraction, routing/fill surface, discharge, Strahler/support state, planning or
raw reconstruction array, pass/time/revision, operator diagnostic, ledger,
checkpoint, feature/evidence ID, score, camera or renderer state.

## Trace wire schema

The trace is mandatory for completed H/C base and sensitivity runs and for C
opportunity controls; it is absent for H target-only controls and G. It records
accepted C0 work and every candidate attempt needed to explain adaptive replay.
Values that a mode does not own are represented by enum variants, not fake
zeroes.

```rust
pub enum C0LimiterWireV0 {
    Requested,
    UpliftAccuracy,
    EffectiveDenudationAccuracy,
    EffectiveDenudationSlopeCourant,
    HillslopeStability,
}

pub enum CandidateAttemptOutcomeV0 {
    Accepted,
    Retry { limiter: C0LimiterWireV0, limit_myr: f64 },
}

pub struct CandidateAttemptTraceV0 {
    pub candidate_dt_myr: f64,
    pub midpoint_coordinate_myr: f64,
    pub outcome: CandidateAttemptOutcomeV0,
}

pub struct PortalRateV0 {
    pub portal_id: u32,
    pub rate_km3_myr: f64,
}

pub struct WaterRateTraceV0 {
    pub total_supply_km3_myr: f64,
    pub portal_outflow_km3_myr: Vec<PortalRateV0>,
    pub total_portal_outflow_km3_myr: f64,
    pub unresolved_sink_rate_km3_myr: f64,
    pub balance_error_km3_myr: f64,
    pub unresolved_specific_discharge_cell_count: u64,
}

pub enum UpliftStepContributionV0 {
    NotOwnedByH,
    C { rock_uplift_moment_km3: f64 },
}

pub struct ActiveProcessStepTraceV0 {
    pub effective_denudation_export_km3: f64,
    pub hillslope_portal_transfers_km3: Vec<PortalVolumeV0>,
    pub hillslope_portal_transfer_km3: f64,
    pub maximum_effective_denudation_rate_km_myr: f64,
    pub maximum_linear_hillslope_abs_grade: f64,
    pub water: WaterRateTraceV0,
}

pub enum StepProcessTraceV0 {
    Disabled,
    Enabled(ActiveProcessStepTraceV0),
}

pub struct AcceptedC0StepTraceV0 {
    pub step_index: u64,
    pub coordinate_start_myr: f64,
    pub coordinate_end_myr: f64,
    pub requested_dt_myr: f64,
    pub accepted_dt_myr: f64,
    pub attempts: Vec<CandidateAttemptTraceV0>,
    pub limiting_operator: C0LimiterWireV0,
    pub uplift_accuracy_limit_myr: Option<f64>,
    pub denudation_accuracy_limit_myr: Option<f64>,
    pub denudation_slope_courant_limit_myr: Option<f64>,
    pub hillslope_stability_limit_myr: Option<f64>,
    pub uplift: UpliftStepContributionV0,
    pub process: StepProcessTraceV0,
    pub elevation_volume_moment_change_km3: f64,
    pub closure_error_km3: f64,
}

pub struct HPassTraceV0 {
    pub pass: u32,
    pub schedule_coordinate_myr: f64,
    pub target_progress: f64,
    pub hold_restoration_km3: f64,
    pub post_hold_elevation_component_hash: u64,
    pub first_c0_step_index: u64,
    pub one_past_last_c0_step_index: u64,
}

pub struct HTracePayloadV0 {
    pub passes: Vec<HPassTraceV0>,
    pub c0_steps: Vec<AcceptedC0StepTraceV0>,
}

pub struct CTracePayloadV0 {
    pub c0_steps: Vec<AcceptedC0StepTraceV0>,
}

pub enum OrganizationStepTracePayloadV0 {
    H(HTracePayloadV0),
    C(CTracePayloadV0),
}

pub enum TraceExtentV0 {
    Complete,
    HCommittedPrefix { last_committed_pass: u32 },
    CCommittedPrefix { last_committed_time_myr: f64, revision: u64 },
}

pub struct OrganizationStepTraceV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub identity: OrganizationArtifactIdentityV0,
    pub config_hash: u64,
    pub extent: TraceExtentV0,
    pub payload: OrganizationStepTracePayloadV0,
    pub derived_trace_hash: u64,
}
```

Portal vectors are sorted by portal ID and contain every declared portal once.
Attempts are chronological, nonempty, and end in exactly one `Accepted` entry.
The accepted dt equals the last candidate dt; coordinate change equals accepted
dt. Complete H traces contain passes `1..=pass_count`; H prefix traces contain
`1..=last_committed_pass`. Both use contiguous substep ranges and H uplift
variants only. C traces contain C uplift variants only. H/C active runs
use `Enabled`; C uplift-only controls use `Disabled`, with every process limit
(`denudation_accuracy`, `denudation_slope_courant`, `hillslope_stability`)
`None`; its uplift limit remains present when finite. Step indices are
contiguous from zero and never exceed the registered maximum.

H step `coordinate_start/end_myr` is global cumulative operator exposure from
0 to 10 across all passes. It is neither within-pass exposure nor the linked
schedule coordinate; `HPassTrace.schedule_coordinate_myr` owns the latter.
Successful traces use `Complete`. A failure trace uses the matching committed
prefix variant and contains only fully committed steps/passes.

For every attempt, `midpoint=start+(0.5*candidate_dt)` with the shown
nonfused order. After a retry, the next candidate is
`min(previous_candidate, reported_limit)`. Attempt count is at most configured
maximum, the final attempt is the only `Accepted`, and the step's limiting
operator is `Requested` when no retry occurred or the last retry limiter
otherwise. H forbids uplift retry/limits; uplift-only C forbids all process
limiters. Enabled per-step portal vectors contain every portal in ID order and
their stored-order sum equals the step total.

Trace-to-ledger accumulation is step-index outer order and portal-ID inner
order, with each cumulative field updated by `total=total+term` and no fused
operations. Integrated water first forms `rate*accepted_dt` per step, then adds
it. The limiter histogram counts the **final limiting operator of each accepted
step**, not retry attempts; total candidate attempts is the sum of attempt-vector
lengths. H hold restoration accumulates pass order outer, stored cell order
inner. Residual fields are recomputed from the already accumulated totals using
the exact parenthesization in the conservation equations below; negative-zero
results are canonicalized to positive zero before serialization.

## Checkpoint wire schema

```rust
pub struct PortalVolumeV0 {
    pub portal_id: u32,
    pub volume_km3: f64,
}

pub struct IntegratedWaterLedgerV0 {
    pub supplied_volume_km3: f64,
    pub portal_outflow_volume_km3: Vec<PortalVolumeV0>,
    pub total_portal_outflow_volume_km3: f64,
    pub unresolved_sink_volume_km3: f64,
    pub balance_error_km3: f64,
}

pub struct SurfaceProcessLedgerV0 {
    pub effective_denudation_export_km3: f64,
    pub hillslope_portal_transfers_km3: Vec<PortalVolumeV0>,
    pub total_hillslope_portal_transfer_km3: f64,
    pub water: IntegratedWaterLedgerV0,
}

pub struct HElevationMomentLedgerV0 {
    pub initial_elevation_volume_moment_km3: f64,
    pub gross_hold_restoration_km3: f64,
    pub process: Option<SurfaceProcessLedgerV0>,
    pub final_elevation_volume_moment_km3: f64,
    pub closure_error_km3: f64,
}

pub struct CElevationMomentLedgerV0 {
    pub initial_elevation_volume_moment_km3: f64,
    pub forcing_interval_rock_uplift_moment_km3: f64,
    pub relaxation_interval_rock_uplift_moment_km3: f64,
    pub total_rock_uplift_moment_km3: f64,
    pub process: Option<SurfaceProcessLedgerV0>,
    pub final_elevation_volume_moment_km3: f64,
    pub closure_error_km3: f64,
}

pub struct GPortalAccumulationV0 {
    pub portal_id: u32,
    pub accumulated_runoff_km3_myr: f64,
    pub accumulated_area_km2: f64,
    pub owned_cell_count: u64,
}

pub struct GReconstructionLedgerV0 {
    pub initial_elevation_volume_moment_km3: f64,
    pub positive_added_volume_km3: f64,
    pub negative_added_volume_km3: f64,
    pub final_elevation_volume_moment_km3: f64,
    pub moment_identity_error_km3: f64,
    pub declared_work_volume_km3: f64,
    pub work_volume_residual_km3: f64,
    pub total_local_runoff_km3_myr: f64,
    pub total_portal_runoff_km3_myr: f64,
    pub runoff_balance_error_km3_myr: f64,
    pub total_cell_area_km2: f64,
    pub total_portal_area_km2: f64,
    pub area_balance_error_km2: f64,
    pub portals: Vec<GPortalAccumulationV0>,
}

pub enum OrganizationLedgerV0 {
    H(HElevationMomentLedgerV0),
    C(CElevationMomentLedgerV0),
    G(GReconstructionLedgerV0),
}

pub struct HCheckpointPayloadV0 {
    pub pass: u32,
    pub schedule_coordinate_myr: f64,
    pub completed_operator_exposure_myr: f64,
    pub one_past_last_c0_step_index: u64,
    pub physical_elevation_km: Vec<f64>,
    pub cumulative_ledger: HElevationMomentLedgerV0,
}

pub struct CCheckpointPayloadV0 {
    pub time_myr: f64,
    pub revision: u64,
    pub one_past_last_c0_step_index: u64,
    pub physical_elevation_km: Vec<f64>,
    pub cumulative_ledger: CElevationMomentLedgerV0,
}

pub enum GReceiverV0 {
    Portal { portal_id: u32 },
    Cell { receiver_cell: u32 },
}

pub struct GInputCheckpointPayloadV0 {
    pub planning_field_km: Vec<f64>,
    pub planning_field_component_hash: u64,
}

pub struct GForestCheckpointPayloadV0 {
    pub receivers: Vec<GReceiverV0>,
    pub finalization_order: Vec<u32>,
    pub path_maximum_km: Vec<f64>,
    pub accumulated_runoff_km3_myr: Vec<f64>,
    pub accumulated_area_km2: Vec<f64>,
    pub strahler_order: Vec<u32>,
    pub support_cell_counts: Vec<u64>,
    pub receiver_component_hash: u64,
    pub finalization_component_hash: u64,
    pub accumulated_runoff_component_hash: u64,
    pub accumulated_area_component_hash: u64,
    pub strahler_support_component_hash: u64,
}

pub struct GReconstructionCheckpointPayloadV0 {
    pub amplitude_a_g_km_inverse: f64,
    pub raw_edge_term_km2: Vec<f64>,
    pub raw_edge_term_component_hash: u64,
    pub physical_elevation_km: Vec<f64>,
    pub reconstruction_ledger: GReconstructionLedgerV0,
}

pub enum OrganizationCheckpointPayloadV0 {
    H(HCheckpointPayloadV0),
    C(CCheckpointPayloadV0),
    GInput(GInputCheckpointPayloadV0),
    GForest(GForestCheckpointPayloadV0),
    GReconstruction(GReconstructionCheckpointPayloadV0),
}

pub enum OrganizationCheckpointRoleV0 { Registered, FailureTerminal }

pub struct OrganizationCheckpointV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub identity: OrganizationArtifactIdentityV0,
    pub config_hash: u64,
    pub ordinal: u32,
    pub role: OrganizationCheckpointRoleV0,
    pub payload: OrganizationCheckpointPayloadV0,
    pub derived_checkpoint_hash: u64,
}

pub struct CheckpointBindingV0 {
    pub ordinal: u32,
    pub checkpoint_hash: u64,
    pub encoded_length_bytes: u64,
}

pub struct TraceBindingV0 {
    pub trace_hash: u64,
    pub encoded_length_bytes: u64,
}
```

H checkpoint ordinals follow declared checkpoint-pass order; pass zero is the
stored initial surface and pass 200/400 is after carving. C ordinals follow
`[0,3,6,10]` and land exactly on those times by endpoint clipping. A C
opportunity control uses the same four checkpoints and trace, even though its
active-process payload is absent. G has exactly three checkpoint ordinals in
`GInput`, `GForest`, `GReconstruction` order.

Successful runs use only `Registered`. A failed H/C run binds exactly one final
`FailureTerminal` checkpoint containing its full last-committed elevation and
cumulative ledger; it may be at an otherwise undeclared pass/time. A failed G
run binds a terminal checkpoint for its last completed stage when one exists.
The terminal binding is last and its coordinate equals the failure coordinate.
If the failure coordinate is `BeforeStart`/no completed G stage, no terminal
checkpoint is present.

If the last committed coordinate already has a `Registered` checkpoint, that
same payload/ordinal is relabeled `FailureTerminal`; no duplicate is appended.
Otherwise append one new terminal ordinal after all earlier registered
checkpoints. Thus every failed artifact has at most one checkpoint at any
coordinate and exactly one terminal role when committed state exists.

Every elevation/planning/forest array has exactly input `N` cells. G finalization
is a permutation of `0..N`; every receiver is legal, points to an earlier
finalized cell or a declared portal, and all paths terminate. Support counts are
in the exact configured threshold order. Component hashes are domain-separated
and recomputed; redundant arrays are retained because they are native authored
state, not common evidence.

G Strahler order is derived after the forest is frozen. A cell with no cell
children has order 1. Otherwise let `r` be the maximum child order; the cell is
`r+1` exactly when at least two cell children have order `r`, and `r` otherwise.
Portal roots are external and receive no order. Children are visited in
ascending cell index, though the count/max rule is order-independent. For each
configured threshold in vector order, `support_cell_counts[k]` is the count of
cells whose accumulated area is `>= support_thresholds_km2[k]`.

G queue counters use these event rules:

- `portal_seed_count` counts every open portal-face candidate evaluated,
  including candidates that do not improve the owner cell's current best key;
- `push_count` counts every heap insertion, including successful seed
  insertions;
- `pop_count` counts every heap removal;
- `stale_pop_count` counts a pop whose cell is already finalized or whose full
  key differs from that cell's current best key;
- `relaxation_count` counts every successful replacement of an unfinalized
  cell's best full key, including successful seeds;
- `tie_replacement_count` is the subset of successful replacements whose path-
  maximum compares equal by `f64::total_cmp` and whose improvement occurs in a
  later key field; and
- `maximum_queue_length` is updated immediately after each push.

Neighbor candidates are evaluated in ascending neighbor-cell index after
portal seeds are evaluated in portal-ID then boundary-face-record order. These
counters describe the frozen reference implementation, not algorithm quality.

## Conservation equations

Let `M(z)=sum_i z[i] A[i]`, accumulated left-to-right in stored cell order.
These are elevation-volume **moment** identities, not absolute crust volumes.

H closes:

```text
epsilon_H = M_f - (M_0 + H_gross - E_export - B_portal)
```

`H_gross` is the sum over passes and cells of
`max(target_k-z_before_hold,0)*A`; it may exceed declared opportunity `V`.
`E_export` is effective areal removal. `B_portal` is signed and equals the
stored-ID-order sum of per-portal hillslope transfers (positive export,
negative import). H target-only controls store `process=None` and interpret both
process terms as absent zero contributions.

C closes:

```text
U_total  = U_0_to_6 + U_6_to_10
epsilon_C = M_f - (M_0 + U_total - E_export - B_portal)
```

`U_6_to_10` must be exact positive zero. Opportunity residuals such as
`U_total-V` are calibration evidence, not part of this material identity. C
uplift-only controls store `process=None`; coevolving runs store `Some`.

For every accepted H/C substep with `process=Enabled`, and after exposure
integration:

```text
epsilon_water_rate = Q_supply - Q_portals - Q_unresolved_sink
W_x = sum_steps Q_x(step) * accepted_dt(step)
epsilon_water_integrated = W_supply - W_portals - W_unresolved_sink
```

The sink is instantaneous unresolved routing sink rate integrated over operator
exposure; it is not persistent physical water storage. In enabled process
payloads, per-portal water and hillslope vectors contain every input portal
exactly once in ID order.

G has no geological material/time ledger. It closes static reconstruction and
forest accumulation identities:

```text
P = sum_i max(z_f[i]-z0[i],0) * A[i]
N = sum_i max(z0[i]-z_f[i],0) * A[i]
epsilon_G_moment = M_f - (M_0 + P - N)
epsilon_G_runoff = sum_i local_runoff[i] - sum_portals accumulated_runoff
epsilon_G_area   = sum_i A[i] - sum_portals accumulated_area
work residual    = P - V
```

V0 reconstruction requires `N==+0.0`, strict authored receiver descent and
strict portal-base descent. Record counts and maximum magnitude of any
violations in native provenance; any nonzero count is a typed failure, not a
ledger tolerance.

G ledger portals and native `portal_owner_counts` each contain every input
portal exactly once in ID order. Their owner counts agree, sum to `N`, and equal
forest-derived terminal assignments. Portal runoff/area values are recomputed
from the forest arrays and their stored-order sums equal the corresponding
ledger totals.

For every G violation summary, maximum magnitude is `None` iff its count is
zero and `Some(nonnegative)` otherwise. Completed G provenance therefore has
zero counts/`None`; a reconstruction failure's `GInvariant` witness retains the
full counts/maxima plus the first offending cell in ascending cell order.

## Completion and native provenance

```rust
pub struct LimiterHistogramV0 {
    pub requested: u64,
    pub uplift_accuracy: u64,
    pub denudation_accuracy: u64,
    pub denudation_slope_courant: u64,
    pub hillslope_stability: u64,
}

pub struct HCompletionV0 {
    pub completed_pass_count: u32,
    pub completed_operator_exposure_myr: f64,
    pub accepted_c0_step_count: u64,
    pub total_candidate_attempt_count: u64,
    pub maximum_attempts_for_one_step: u32,
    pub minimum_accepted_dt_myr: Option<f64>,
    pub maximum_accepted_dt_myr: Option<f64>,
    pub limiter_histogram: LimiterHistogramV0,
}

pub struct CCompletionV0 {
    pub reached_time_myr: f64,
    pub final_revision: u64,
    pub accepted_c0_step_count: u64,
    pub total_candidate_attempt_count: u64,
    pub maximum_attempts_for_one_step: u32,
    pub minimum_accepted_dt_myr: f64,
    pub maximum_accepted_dt_myr: f64,
    pub limiter_histogram: LimiterHistogramV0,
}

pub struct GCompletionV0 {
    pub completed_stage: GCompletedStageV0,
    pub finalized_cell_count: u64,
}

pub enum GCompletedStageV0 { Reconstruction }
pub enum OrganizationCompletionV0 { H(HCompletionV0), C(CCompletionV0), G(GCompletionV0) }

pub struct ActiveProcessSummaryV0 {
    pub maximum_denudation_rate_km_myr: f64,
    pub maximum_linear_hillslope_abs_grade: f64,
    pub maximum_unresolved_specific_discharge_cells: u64,
}

pub struct FinalNativeRoutingWitnessV0 {
    pub routing_state_component_hash: u64,
    pub final_water_rate: WaterRateTraceV0,
}

pub struct HNativeSummaryV0 {
    pub active_process: Option<ActiveProcessSummaryV0>,
    pub final_routing: Option<FinalNativeRoutingWitnessV0>,
}

pub struct CNativeSummaryV0 {
    pub active_process: Option<ActiveProcessSummaryV0>,
    pub final_routing: Option<FinalNativeRoutingWitnessV0>,
}

pub struct GQueueCountersV0 {
    pub portal_seed_count: u64,
    pub push_count: u64,
    pub pop_count: u64,
    pub stale_pop_count: u64,
    pub relaxation_count: u64,
    pub tie_replacement_count: u64,
    pub maximum_queue_length: u64,
}

pub struct GNativeSummaryV0 {
    pub amplitude_a_g_km_inverse: f64,
    pub queue: GQueueCountersV0,
    pub portal_owner_counts: Vec<(u32, u64)>,
    pub authored_receiver_descent_violation_count: u64,
    pub maximum_receiver_descent_violation_km: Option<f64>,
    pub portal_base_violation_count: u64,
    pub maximum_portal_base_violation_km: Option<f64>,
}

pub enum OrganizationNativeSummaryV0 {
    H(HNativeSummaryV0),
    C(CNativeSummaryV0),
    G(GNativeSummaryV0),
}

pub enum OrganizationClaimV0 {
    LinkedOpportunityConversion,
    CalibratedOperatorExposure,
    PhysicalTime,
    EffectiveArealCellMeanDenudation,
    ConservativeLinearHillslopeTransport,
    AuthoredPortalReceiverForest,
    StaticGraphReconstruction,
    NoGeologicalChronology,
}

pub struct OrganizationNativeProvenanceV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub identity: OrganizationArtifactIdentityV0,
    pub predecessors: OrganizationPredecessorsV0,
    pub initial_elevation_component_hash: u64,
    pub final_elevation_component_hash: u64,
    pub config: OrganizationArmConfigV0,
    pub completion: OrganizationCompletionV0,
    pub ledger: OrganizationLedgerV0,
    pub trace: Option<TraceBindingV0>,
    pub checkpoints: Vec<CheckpointBindingV0>,
    pub summary: OrganizationNativeSummaryV0,
    pub declared_claims: Vec<OrganizationClaimV0>,
    pub derived_native_provenance_hash: u64,
}
```

Claim vectors are sorted by enum order with no duplicates. H base/sensitivity
declares linked opportunity conversion, calibrated exposure, effective areal
cell-mean denudation and conservative linear hillslopes; C base/sensitivity
declares linked opportunity conversion, physical time and the two process
claims; G declares linked opportunity conversion, authored forest, static
reconstruction and no chronology. Opportunity controls declare only the
ownership they execute: H declares linked opportunity conversion; C declares
linked opportunity conversion and physical time but no denudation/hillslope
claim. Claims are validation metadata, never quality scores.

Active H/C runs store both summary and final-routing witnesses as `Some`;
target/uplift-only controls store both as `None`. The routing hashes are native
replay witnesses only—the arrays are not common-result fields and cannot enter
independent scoring. C must consume the exact stored local-runoff array through
its adapter; reconstructing supply from the scalar depth-rate declaration is
not sufficient even when values happen to compare equal. At every candidate
midpoint, including every retry, C freshly evaluates the predecessor's stored
scenario with its registered compiler semantics on the accepted mesh and
consumes that evaluation's f32 vertical-rate output. The predecessor's stored
frame hashes are compiler-boundary witnesses, not arrays available to C and not
a substitute for midpoint evaluation. Horizontal velocity and dominant-episode
metadata remain input provenance and are not falsely listed as consumed state.
Active-process maxima are reductions over accepted pre-process routing/step
diagnostics only. The fresh reroute of the authoritative final post-hillslope
surface is owned exclusively by `final_routing.final_water_rate`; it is not
silently mixed into those maxima.

## Typed failure root

A failure artifact is semantic evidence that a registered invocation stopped at
a typed boundary. It is not a partial arm result. External termination that
prevents validated atomic publication (power loss, OOM kill, `SIGKILL`) leaves
no semantic failure claim; the external audit records it.

```rust
pub enum OrganizationFailurePhaseV0 {
    HTarget,
    HCarve,
    CForcing,
    CStep,
    GPlanning,
    GForest,
    GCalibrationBracket,
    GCalibrationSolve,
    GReconstruction,
    LedgerValidation,
    CheckpointValidation,
    ResourceCeiling,
}

pub enum OrganizationFailureCauseV0 {
    NonFiniteValue,
    NonCanonicalValue,
    LengthOrOrderMismatch,
    MinimumDtReached,
    MaximumAdaptiveAttemptsReached,
    InvalidOperatorLimit,
    RoutingOperatorFailure,
    DenudationOperatorFailure,
    HillslopeOperatorFailure,
    BoundaryOperatorFailure,
    RevisionOverflow,
    InternalInvariantFailure,
    EndpointNotReached,
    QueueInvariantFailure,
    ForestTerminationFailure,
    CalibrationBracketFailure,
    CalibrationIterationLimit,
    ReconstructionInvariantFailure,
    LedgerClosureFailure,
    WallTimeCeiling,
    MemoryCeiling,
    ArtifactSizeCeiling,
    MaximumAcceptedStepCountReached,
    OpportunityControlMismatch,
}

pub enum OrganizationFailureCoordinateV0 {
    BeforeStart,
    H { last_committed_pass: u32, completed_operator_exposure_myr: f64 },
    C { last_committed_time_myr: f64, revision: u64 },
    G { last_completed_stage: Option<GFailureStageV0> },
}

pub enum GFailureStageV0 { Input, Forest, Reconstruction }
pub enum OrganizationResourceKindV0 { WallTimeMilliseconds, PeakRssBytes, ArtifactBytes }
pub enum OrganizationFailureAuthorityV0 {
    ReplayableAlgorithmic,
    ObservationalResource,
}

pub enum OrganizationFailureWitnessV0 {
    None,
    Timestep {
        requested_dt_myr: f64,
        candidate_dt_myr: f64,
        midpoint_coordinate_myr: Option<f64>,
        attempt_number: u32,
        retry_limiter: Option<C0LimiterWireV0>,
        reported_limit_myr: Option<f64>,
    },
    GQueue {
        cell_index: Option<u32>,
        queue_length: u64,
    },
    GCalibration {
        lower_a_km_inverse: f64,
        upper_a_km_inverse: f64,
        iteration_count: u32,
        residual_km3: f64,
    },
    GInvariant {
        first_cell_index: u32,
        first_receiver_cell_index: Option<u32>,
        receiver_descent_violation_count: u64,
        maximum_receiver_descent_violation_km: Option<f64>,
        portal_base_violation_count: u64,
        maximum_portal_base_violation_km: Option<f64>,
    },
    Resource {
        kind: OrganizationResourceKindV0,
        observed: u64,
        ceiling: u64,
    },
    StepCount {
        accepted_step_count: u64,
        maximum_accepted_step_count: u64,
    },
}

pub struct OrganizationRunFailureV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub identity: OrganizationArtifactIdentityV0,
    pub predecessors: OrganizationPredecessorsV0,
    pub config: OrganizationArmConfigV0,
    pub authority: OrganizationFailureAuthorityV0,
    pub phase: OrganizationFailurePhaseV0,
    pub cause: OrganizationFailureCauseV0,
    pub coordinate: OrganizationFailureCoordinateV0,
    pub witness: OrganizationFailureWitnessV0,
    pub trace: Option<TraceBindingV0>,
    pub checkpoints: Vec<CheckpointBindingV0>,
    pub derived_failure_hash: u64,
}
```

No semantic field contains the underlying `Display`/`Debug` string from a
runtime error. Human-readable detail belongs in the nonsemantic envelope.
Failure traces/checkpoints, if present, must end at the declared last committed
coordinate. Transactional rejected attempts never masquerade as committed
state. Input, predecessor, configuration and publication failures occur outside
this semantic root and remain CLI/external-audit diagnostics: publication begins
only after exact input/predecessor/config validation.

An H pass is outer-transactional. Hold plus its complete declared carve exposure
commit together; failure during a pass discards that pass's surface and trace,
so the failure coordinate and prefix end at the preceding completed pass. C
retains its existing accepted-step transactionality. The final
`FailureTerminal` checkpoint makes the last committed H/C state and ledger
reconstructible even when it is not a registered success checkpoint.

`ReplayableAlgorithmic` is allowed only for cooperative algorithm/operator/
ledger failures and must reproduce exactly. Resource roots use observational
authority; replay returns the typed result `NotReplayable` rather than accepting
or invalidating them.
`ObservationalResource` requires `ResourceCeiling`, a matching wall/memory/size
cause and `Resource` witness. Every other legal phase/cause/witness combination
is registered explicitly in the numerical amendment under
`ReplayableAlgorithmic`; unregistered combinations reject. A deterministic-
repeat mismatch requires retaining both complete temporary semantic trees and
is recorded by the external audit rather than compressed into this failure
root.
A `MinimumDtReached` witness has no midpoint or retry limiter when rejection
occurs before forcing sampling. A maximum-attempt witness retains the last retry
limiter and reported limit. `MaximumAcceptedStepCountReached` requires the
`StepCount` witness and the registered equal current/maximum counts.
`OpportunityControlMismatch` is legal only at `LedgerValidation` with `None`;
the terminal control checkpoint supplies the recomputable surface and ledger.
Operator-family causes replace unstable runtime `Operator(String)` text; the
adapter must map every runtime error to one bounded variant before an active
run.

## Hash preimages

All component and root hashes are FNV-1a 64 over fixed-integer little-endian
bincode. Domain strings are exact. Each root preimage preserves the wire field
order above and excludes only its own `derived_*_hash` field.

```text
orogen-organization-v0/configuration
  domain, schema_version, hash_version, identity, predecessors, payload

orogen-organization-v0/elevation-array
  domain, input_bundle_hash, input_resolution_hash, nominal_spacing_km,
  arm, purpose, elevation_km

orogen-organization-v0/trace
  domain, schema_version, hash_version, identity, config_hash, extent, payload

orogen-organization-v0/checkpoint
  domain, schema_version, hash_version, identity, config_hash, ordinal, role,
  payload

orogen-organization-v0/native-provenance
  domain, schema_version, hash_version, identity, predecessors,
  initial_elevation_component_hash, final_elevation_component_hash, config,
  completion, ledger, trace, checkpoints, summary, declared_claims

orogen-organization-v0/arm-result
  domain, schema_version, hash_version, identity, predecessors,
  initial_elevation_component_hash, final_elevation_component_hash,
  final_elevation_km, opportunity_audit, native_provenance_hash

orogen-organization-v0/run-failure
  domain, schema_version, hash_version, identity, predecessors, config,
  authority, phase, cause, coordinate, witness, trace, checkpoints
```

The elevation-array component is used for result finals and H/C checkpoint
states; context determines the claim while
the component identifies exact array content. The accepted linked-input initial-
elevation component hash is reused by value and not redefined.

All remaining component payloads and orders are exact:

```rust
pub struct FinalNativeRoutingStateWireV0 {
    pub partition: FlowPartitionPolicyV0,
    pub directed_edge_fraction: Vec<f64>,
    pub directed_edge_flux_km3_myr: Vec<f64>,
    pub boundary_face_fraction: Vec<f64>,
    pub boundary_face_flux_km3_myr: Vec<f64>,
    pub portal_outflow_km3_myr: Vec<PortalRateV0>,
    pub routing_elevation_km: Vec<f64>,
    pub flat_potential: Vec<Option<u32>>,
    pub local_supply_km3_myr: Vec<f64>,
    pub available_supply_km3_myr: Vec<f64>,
    pub specific_discharge_vector_km2_myr: Vec<[f64; 3]>,
    pub specific_discharge_km2_myr: Vec<f64>,
    pub sink_rate_km3_myr: Vec<f64>,
    pub high_to_low_order: Vec<u32>,
    pub total_supply_km3_myr: f64,
    pub total_portal_outflow_km3_myr: f64,
    pub total_sink_rate_km3_myr: f64,
}

pub struct GStrahlerSupportComponentV0<'a> {
    pub strahler_order: &'a Vec<u32>,
    pub support_cell_counts: &'a Vec<u64>,
}

pub struct HPostHoldComponentV0<'a> {
    pub pass: u32,
    pub elevation_km: &'a Vec<f64>,
}

pub struct ElevationArrayPreimageV0<'a> {
    pub domain: &'static str, // "orogen-organization-v0/elevation-array"
    pub identity: &'a OrganizationArtifactIdentityV0,
    pub elevation_km: &'a Vec<f64>,
}

pub struct ConfiguredComponentPreimageV0<'a, T> {
    pub domain: &'static str,
    pub identity: &'a OrganizationArtifactIdentityV0,
    pub config_hash: u64,
    pub payload: &'a T,
}
```

`ConfiguredComponentPreimageV0` domains and payload types are below. Every full
domain string is the exact ASCII concatenation
`"orogen-organization-v0" + suffix`; the leading slash shown in each suffix is
the only separator, and the resulting full string is what bincode serializes.

| domain suffix | exact payload |
|---|---|
| `/h-post-hold` | `HPostHoldComponentV0` |
| `/final-routing-state` | `FinalNativeRoutingStateWireV0` in the exact field orders above |
| `/g-planning` | `Vec<f64>` in cell order |
| `/g-receiver` | `Vec<GReceiverV0>` in cell order |
| `/g-finalization` | `Vec<u32>` in finalization order |
| `/g-accumulated-runoff` | `Vec<f64>` in cell order |
| `/g-accumulated-area` | `Vec<f64>` in cell order |
| `/g-strahler-support` | `GStrahlerSupportComponentV0` |
| `/g-raw-edge-term` | `Vec<f64>` in cell order |

The H post-hold field uses its dedicated domain rather than the generic
elevation-array domain. Final-routing hashes are computed by freshly rerouting
the authoritative final post-hillslope surface. Directed edges, boundary faces
and portal rates use accepted mesh/portal order; high-to-low indices are narrowed
to checked `u32`. `sink_rate_km3_myr[i]` is the available supply retained at a
cell with no routed exit during that fresh solve; it is a rate diagnostic, not
persistent storage. These arrays are replay inputs to component hashes but are
not retained as common evidence. Standalone native validation checks their hash
syntax only; predecessor/replay validation recomputes their semantic meaning.

## Exact binding matrix

Predecessor validation requires all duplicated bindings to be exact:

- native `identity` and `predecessors` equal its embedded config values;
- result `identity` and `predecessors` equal native and config;
- result `initial_elevation_component_hash` and native initial hash equal the
  selected linked-input component hash;
- result final component equals a fresh elevation-array hash of its final
  vector and equals native final component;
- result `native_provenance_hash` equals native root hash;
- every trace/checkpoint identity and config hash equals native/config; binding
  ordinal, semantic hash and canonical encoded length equal the supplied bytes;
- checkpoint ordinals and filenames are contiguous from zero; the final H/C/G
  reconstruction checkpoint physical surface is bit-identical to the result
  final surface for successful runs;
- H/C `SeparateControl` hash equals both predecessor copies and the supplied
  same-arm/same-resolution opportunity-control result root; its audit content
  equals the control content;
- G `ReuseFrozen4Km` equals both predecessor copies and supplied G 4 km result/
  native hashes, and its `a_G` bits equal the 4 km native/audit/config authority;
  and
- current-run trace totals, checkpoint cumulative ledgers and native
  completion/ledger/summary agree under the exact reducers in this document.
  A `SelfResult` opportunity audit is reduced from that same result and, where
  applicable, its own trace. A `SeparateControl` audit is instead recomputed
  from the supplied, independently predecessor-validated control artifact (or
  accepted by byte-equality after that validation); it is never equated with
  the base/sensitivity run's trace or ledger.

Failure predecessor validation has the same exact DAG obligations:

- failure `identity` and `predecessors` equal its embedded config values, and
  the config hash recomputed from that config is the authority for all bound
  sidecars;
- each non-`None` failure predecessor equals the supplied artifact root and
  satisfies the same arm, purpose and resolution relationship registered for a
  successful run;
- a bound trace has the failure identity and config hash, and its semantic hash
  and canonical encoded length equal its binding;
- every bound checkpoint has the failure identity and config hash, and its
  ordinal, semantic hash and canonical encoded length equal its binding;
- H/C prefix extent, terminal-checkpoint coordinate, terminal surface and
  cumulative ledger all describe the failure's last committed coordinate. The
  trace reductions equal that terminal ledger. For G, the terminal payload is
  exactly the declared last completed stage; and
- `BeforeStart`, or G with `last_completed_stage=None`, has no trace and no
  checkpoint. Otherwise the last checkpoint binding is the unique
  `FailureTerminal` required above.

Standalone validation can check equality within one root but does not pretend
to validate absent predecessors or opaque routing witnesses.

## Validation API and authority

The implementation exposes these public library boundaries from a new
`world::landscape::organization_artifact` module:

```rust
organization_arm_config_hash_v0
organization_arm_result_bytes_v0
organization_native_provenance_bytes_v0
organization_step_trace_bytes_v0
organization_checkpoint_bytes_v0
organization_run_failure_bytes_v0

decode_organization_arm_result_v0
decode_organization_native_provenance_v0
decode_organization_step_trace_v0
decode_organization_checkpoint_v0
decode_organization_run_failure_v0

validate_organization_arm_result_standalone_v0
validate_organization_native_provenance_standalone_v0
validate_organization_step_trace_standalone_v0
validate_organization_checkpoint_standalone_v0
validate_organization_run_failure_standalone_v0

validate_organization_success_against_input_v0(
    input,
    result,
    native,
    trace,
    checkpoints,
    opportunity_control,
    g_reference_4km,
)

validate_organization_failure_against_input_v0(
    input,
    failure,
    trace,
    checkpoints,
    opportunity_control,
    g_reference_4km,
)

validate_organization_success_by_replay_v0(
    input,
    result,
    native,
    trace,
    checkpoints,
    opportunity_control,
    g_reference_4km,
)

validate_organization_failure_by_replay_v0(
    input,
    failure,
    trace,
    checkpoints,
    opportunity_control,
    g_reference_4km,
)
```

The optional arguments are typed `Option` values but their presence is fixed by
the legal arm/purpose matrix. Supplying or omitting an illegal predecessor is an
error, not “best effort.”

Standalone validation checks:

- byte cap, registered schema/hash strings, arm/purpose/payload agreement and
  exact legal configuration values available at that amendment level;
- finite canonical floats, nonnegative domains, fixed vector order, duplicate
  rejection and explicit global/per-vector count limits;
- component and root hashes, trace/checkpoint ordinal structure, local step
  continuity and internally stated ledger algebra; and
- absence/presence rules for configuration, trace, checkpoint and claim
  variants.

Predecessor validation first fully validates the accepted shared-input bundle,
selects the resolution by exact stored resolution hash, then checks exact bundle
hash, spacing bits, input component hashes, cell count, portal IDs, arm/purpose
predecessors and G 4 km amplitude bits. It decodes every bound trace/checkpoint,
checks its encoded length and semantic hash, recomputes elevation moments,
opportunity reductions, trace totals, ledgers, completion counters and
result/native/checkpoint final hashes, and requires exact H/C opportunity-audit
content equality with the supplied control.

Predecessor consistency still cannot prove the solver caused the final array. A
test must demonstrate that a mutually rewritten, well-formed alternate surface
can pass standalone and consistency checks when all dependent hashes and local
algebra are repaired. Replay must reject that witness by reconstructing the arm
from accepted input and registered configuration and comparing exact semantic
values and bytes. Until H/C/G exist, the replay function may be compiled only
behind the implementation rung, but its API and authority are frozen here.
For `ReplayableAlgorithmic`, failure replay likewise requires the registered
run to stop at the same typed phase, cause, coordinate and witness with
identical committed partial artifacts; a consistent failure record alone does
not prove the algorithm failed. Observational authorities return
`NotReplayable` as defined above.

The numerical amendment supplies tolerance predicates to predecessor/replay
validation. A semantic decoder never silently applies a morphology admission
threshold.

## JSON projections

Binary artifacts are semantic authority. JSON is regenerated from decoded
binary and is never accepted as an independent input.

The exact JSON projections are:

- `OrganizationArmResultJsonV0`: JSON schema, semantic schema/hash, identity,
  predecessor hashes as 16-digit lowercase hex, cell count, final min/max and
  final/opportunity/native/result hashes, and the complete opportunity audit;
- `OrganizationNativeProvenanceJsonV0`: JSON/semantic versions, identity,
  predecessor/config/final/native hashes, complete configuration, completion,
  ledger, trace/checkpoint bindings, summary and claims;
- `OrganizationCheckpointJsonV0`: identity, config hash, ordinal, payload kind,
  array counts and component/root hashes, plus checkpoint coordinate and ledger
  for H/C or invariant/counter summaries for G;
- `OrganizationStepTraceJsonV0`: identity, config/trace hashes, pass/step counts,
  coordinate bounds, attempt/limiter totals and ledger/water reductions, without
  per-step arrays; and
- `OrganizationRunFailureJsonV0`: identity, predecessor/config/failure hashes,
  phase, cause, coordinate, typed witness and bound partial-artifact summaries.

Each `from_semantic` constructor is the sole projection authority. `validate_against`
compares to a freshly constructed value. Pretty JSON ends in exactly one LF;
publishers reread, parse and compare it before publication. JSON hashes are
strings; semantic f64 quantities remain JSON numbers under the repository's
`float_roundtrip` serde_json feature.

## Atomic directory contract

The CLI receives a required new output directory. Canonical campaign paths are:

```text
artifacts/orogen-owner-v0/runs/opportunity-control/<h|c>/<8|4|2>-km/<attempt-id>/
artifacts/orogen-owner-v0/runs/base/<h|c|g>/<8|4|2>-km/<attempt-id>/
artifacts/orogen-owner-v0/runs/numerical-sensitivity/<h|c>/4-km/<attempt-id>/
```

`attempt-id` is nonsemantic campaign bookkeeping matching
`[a-z0-9][a-z0-9-]{0,63}` (for example `attempt-000`). Every invocation gets a
new directory. A typed failure therefore remains evidence without blocking a
later rerun, and no mutable “current” alias selects a result. Downstream
artifacts bind semantic result hashes, never attempt names.

Successful directories contain exactly:

```text
arm-result.bin
arm-result.json
native-provenance.bin
native-provenance.json
run-envelope.json
[step-trace.bin]
[step-trace.json]
checkpoints/
  000.bin
  000.json
  ...
```

Trace files are present exactly when the native trace binding is `Some`.
Checkpoint filenames are zero-padded ordinals and the set must exactly match
native bindings. A failure directory replaces the two result/native pairs with:

```text
run-failure.bin
run-failure.json
run-envelope.json
[step-trace.bin]
[step-trace.json]
checkpoints/
  ...
```

The public CLI is `orogen_owner`. Its semantic arguments are accepted shared-
input artifact path, `H|C|G`, `opportunity-control|base|numerical-sensitivity`,
exact `8|4|2` spacing, required output directory, and the predecessor directory
required by the legal matrix. It exposes no K, pass, timestep, graph, threshold,
amplitude, camera or morphology flags. G 8/2 requires the frozen G 4 km base
directory. H/C base and sensitivity require their same-resolution opportunity-
control directory. G 4 km and H/C controls require no predecessor directory.

The publisher reuses the accepted linked-input discipline:

1. capture source revision/dirty state before creating output paths;
2. require a present directory parent and absent target/temp sibling;
3. acquire a `create_new` sibling publication lock and recheck target absence;
4. build semantic values, perform predecessor validation, encode, decode and
   require value/byte round-trip identity;
5. create files with `create_new`, `sync_all`, reread and fully validate every
   binary and JSON projection from disk;
6. require the exact allowed filename set and construct `run-envelope.json`
   last, then `sync_all`, reread, parse and validate its artifact list against
   the on-disk files;
7. sync checkpoint directory, temporary root, atomically rename the sibling,
   and sync the parent; and
8. on ordinary error, remove only this invocation's temp and lock. Never
   overwrite a target or remove an unowned stale directory.

The nonsemantic run envelope records source revision/dirty state, manifest root,
invocation CWD, executable path/length/FNV, Rust toolchain, OS/kernel/WSL/CPU,
solver thread count (exactly one), available parallelism, command, elapsed time
to prepublication validation, `/proc/self/status` VmHWM witness, final external
measurement authority, success/failure kind and optional stable error summary.
Its artifact list contains every other file in lexical relative-path order with
length and file-byte FNV; it excludes itself. Timing, host and path values are
not semantic and are excluded from deterministic-repeat comparison.

Power loss can leave a temp/lock; the next invocation reports it and requires
explicit inspection rather than deleting evidence. Atomic rename guarantees
that a published target is complete, not that a killed process always publishes
a typed failure.

## Implementation and test gate

Implement the library in `src/world/landscape/organization_artifact.rs`, export
it from `src/world/landscape/mod.rs`, and add the thin `src/bin/orogen_owner.rs`
CLI only after the subsequent evidence/projection amendment and remaining
planar-review amendment are committed. The evidence amendment is now committed
in design; do not refactor the accepted linked-input publisher in this rung.

Required tests are:

1. exact H/C/G analytic fixtures produce repeated equal values and semantic
   bytes; every semantic type round-trips;
2. oversize, trailing, truncated, wrong-version, NaN, infinity, negative-zero,
   illegal option/variant, length, order, duplicate and cap witnesses reject;
3. repaired-hash mutations of every result/native/trace/checkpoint/failure
   family reject whenever a local invariant is violated;
4. wrong bundle, resolution, spacing, arm, purpose, initial hash, opportunity
   control, G reference, config, trace, checkpoint order/length and final hash
   reject predecessor validation;
5. H/C trace sums reproduce completion, integrated water and moment ledgers;
   G forest/reconstruction arrays reproduce its static ledgers and invariants;
6. a mutually repaired alternate final surface demonstrates the epistemic
   limit by passing integrity/consistency where mathematically self-consistent
   and failing deterministic replay;
7. JSON bytes are deterministic with one LF and every parsed projection equals
   a fresh projection; one-field edits reject;
8. envelope entries are exact, lexical and self-excluding, while the declared
   reproducible subset ignores timing/host/path values;
9. an ignored release publication test verifies the exact directory, on-disk
   full validation, preserved existing-target sentinel, and no temp/lock residue
   after ordinary failures; and
10. CLI parsing rejects missing predecessors, forbidden predecessors, missing
    arguments and any unregistered tuning flag.

Existing C0 transactional, midpoint-resampling, water-balance and hillslope-
conservation tests remain required predecessors; they do not substitute for
artifact mutation and publication tests.

## Stop boundary

This amendment completes item 1 of the parent design's executable stop
boundary. The subsequent
[numerical/admission amendment](orogen-organization-numerical-v0-2026-07-16.md)
completes item 2 and makes the small append-only failure-enum amendment above.
Do not implement artifact structs yet. The subsequent
[evidence/projection amendment](orogen-organization-evidence-v0-2026-07-16.md)
now completes item 3; the planar-review amendment still follows before any
active H/C/G run.
