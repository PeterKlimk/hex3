# Organization-owner evidence and projection amendment V0

**Date:** 2026-07-16

**Status:** executable preregistration for independent common extraction,
central projection, authored-G comparison, correspondence reduction and
claim-level materiality; not implemented and not an H/C/G result

**Parents:** [organization-owner comparison design V0](orogen-organization-owner-v0-2026-07-16.md),
[artifact/provenance amendment V0](orogen-organization-artifact-v0-2026-07-16.md),
[numerical/admission amendment V0](orogen-organization-numerical-v0-2026-07-16.md)

**Accepted instruments:** [common planar evidence core V0](landform-common-core-v0-2026-07-15.md),
[G0/S0](landform-object-packet-g0s0-2026-07-14.md),
[D0](landform-object-packet-d0-2026-07-15.md),
[O0a](landform-object-packet-o0-2026-07-15.md) and
[core-backed O0b](landform-object-packet-o0b-2026-07-15.md)

## Decision

Freeze a composition layer over accepted instruments rather than inventing a
new landform extractor. Every valid frozen final surface is independently
adapted to the same whole-domain S0/D0/reference-O0a core. A separately hashed
projection asks central-window and linked-forcing questions without clipping or
changing that core. Accepted core-backed O0b supplies mechanical many-to-many
correspondence. A G-only sidecar compares G's disclosed authored forest with
independent D0; authored state never enters common extraction.

This amendment does **not** choose an arm, name a natural landform kind, score
visual appeal, define a camera, add a response case or turn correspondence into
persistent identity. It does not call one-to-many, many-to-one or many-to-many
best components physical splits or merges. It retains those component kinds,
ties, nulls, context and topology exactly as O0b reports them.

The comparison layer separates three questions:

1. did the independent instrument build and validate;
2. is a quantity stable enough across resolution and numerical sensitivity to
   support a material-difference statement; and
3. what architecture evidence remains, including weak or unfavorable results.

Instrument failure blocks the affected comparison but does not retroactively
invalidate an arm. Missing objects are not zeroes. Incompatible objects are not
averaged until they look comparable. A nonmaterial difference is retained, not
declared equality. No count of favorable fields, weighted quality score or
computed Pareto frontier is introduced.

## Corrections and narrow choices

- The accepted organization population is whole-graph support. The stored
  central mask is consumed only by the projection. No partial-scored S0/D0 core
  is created.
- V0 uses reference O0a only. It makes no claim dependent on a favorable member
  of the ten-run O0a configuration-sensitivity suite, so that suite is neither
  generated nor accepted as a comparison predecessor.
- H/C 4 km numerical-sensitivity surfaces receive the same common-core,
  reference-O0a and central-projection treatment as their bases. This permits
  object-level stability to be measured rather than guessed from a scalar
  terrain RMS. G uses exact repeat and has no sensitivity evidence surface.
- G cell-forest Strahler and D0 retained-reach Strahler remain separately
  descriptive because they are orders on different graphs. V0 does not force a
  false same-cell equality between them.
- O0b already owns positive partners, exact maximum sets, ties, context,
  component cardinality and report-only topology. This amendment only joins
  those immutable records to central cohorts and quantity values.

## Registered identities and limits

```text
arm evidence schema              orogen-owner-arm-evidence-v0
central projection schema        orogen-owner-central-projection-v0
authored G/D0 schema             orogen-owner-g-authored-d0-v0
cross-arm surface schema         orogen-owner-cross-arm-surface-v0
numerical discrepancy schema     orogen-owner-numerical-discrepancy-v0
pairwise comparison schema       orogen-owner-pairwise-comparison-v0
comparison schema                orogen-organization-comparison-v0
cost observation schema          orogen-owner-cost-observation-v0
evidence failure schema          orogen-owner-evidence-failure-v0
JSON projection family           orogen-owner-evidence-json-v0
arm-evidence JSON schema         orogen-owner-arm-evidence-json-v0
central JSON schema              orogen-owner-central-projection-json-v0
authored-G/D0 JSON schema        orogen-owner-g-authored-d0-json-v0
numerical JSON schema            orogen-owner-numerical-discrepancy-json-v0
cross-arm surface JSON schema    orogen-owner-cross-arm-surface-json-v0
pairwise comparison JSON schema  orogen-owner-pairwise-comparison-json-v0
comparison JSON schema           orogen-organization-comparison-json-v0
failure JSON schema              orogen-owner-evidence-failure-json-v0
cost JSON schema                 orogen-owner-cost-observation-json-v0
semantic hash encoding           fnv1a64-bincode-fixint-le-v0

maximum common-core bytes        1 GiB
maximum reference-O0a bytes      512 MiB
maximum arm-evidence bytes         8 MiB
maximum central-projection bytes 256 MiB
maximum authored-G/D0 bytes      256 MiB
maximum cross-arm surface bytes    8 MiB
maximum numerical-discrepancy     64 MiB
maximum pairwise comparison      512 MiB
maximum one O0b bytes            1 GiB
maximum comparison bytes         512 MiB
maximum evidence-failure bytes     8 MiB
maximum JSON file bytes           32 MiB
maximum evidence directory         2 GiB
maximum pairwise directory         2 GiB
maximum comparison directory       2 GiB
maximum cells                    250,000
maximum projected object count  250,000 per family
maximum one member-cell vector   250,000
maximum total nested elements 32,000,000 per file
maximum semantic string bytes        128
maximum arm-evidence bindings          11
maximum numerical discrepancies         2
maximum cross-arm surface rows           3
maximum O0b artifacts                 22
```

Rust declaration order below is wire order. Enums use declaration order as
zero-based discriminants. Binary encoding is fixed-integer little-endian
bincode; decoders reject trailing bytes, oversized input, nonfinite values and
noncanonical negative zero. FNV is an identity checksum, not authentication.
Every `derived_*_hash` is excluded from its own preimage and every other field
is included in displayed order.

All new decoders combine the byte cap with bounded serde seeds/visitors. They
check each vector/string length before allocation, checked-add every nested
element count against the file aggregate and never trust bincode's declared
length. `with_limit` alone is insufficient. Fixed arrays are exact. The complete
root has 11 arm bindings, 3 pairwise-root hashes, 3 cross-arm surfaces, 2
numerical discrepancies, 22 O0b/reductions and 3 G bindings. Pairwise
cardinalities are frozen below.
Segment rows are 2, threshold rows 3,
portals 2 and cohort rows 2.
Accepted common-core/O0b child
decoders are wrapped by the same raw-byte cap and must gain equivalent bounded
visitors before they are accepted from an untrusted artifact directory.
Every extracted feature vector and every member/index vector is capped at
250,000. Expanded annotation/materiality record vectors are instead bounded by
the 32-million aggregate and, more tightly, by remaining bytes divided by their
fixed minimum wire size; this permits both-side object × quantity records
without pretending the feature-count cap is also a record-count cap. Before a vector allocation, also require
`declared_len<=remaining_raw_bytes/minimum_wire_bytes_per_element` using the
type's fixed minimum. Visitors never reserve the untrusted announced capacity:
they start empty, reserve at most 1,024 elements at once, and stop at the first
cap/remaining-byte breach.

## Exact evidence population and adapter

Evidence is built only from a valid, exactly repeated arm result and matching
native provenance. Required result purposes are all nine H/C/G bases at 8/4/2
km plus H/C numerical sensitivities at 4 km. Opportunity controls are numerical
predecessors, not terrain-quality evidence populations.

For each result:

1. fully validate the accepted input bundle/resolution, arm result, native
   provenance, trace/checkpoints and numerical validity;
2. require result final elevation and component hash to be the exact common
   physical-elevation source;
3. call `build_regular_hex_control_volumes_v0` on the stored `LandscapeMesh`,
   then `adapt_landscape_graph_v0`; do not reconstruct a nominal unported mesh
   as the live adapter input;
4. bind `PacketGeometryIdentityV0::LandscapeRegularPlanar` to nominal spacing
   and the freshly validated canonical graph hash;
5. set `scored_cell` to the stored `whole_graph_candidate` and require every
   value true;
6. set the common population to testbed Cartesian XY, declared `960 x 640 km`,
   `WholeGraphSupportV0`, `UniformPerAreaV0 { rate: 500.0 }` and the input's
   semantic portals in portal-ID order;
7. require generated `500.0*A[i]` runoff to bit-equal the stored
   `local_runoff_supply_km3_myr[i]` in cell order;
8. build registered S0 and D0, assemble and fully validate
   `CommonPlanarEvidenceCoreV0`, then build and fully validate the registered
   `ReferenceRelationshipEvidenceV0`; and
9. build the central projection from the validated core and the exact stored
   central-mask array.

The graph is validated both against the live stored mesh/control volumes and by
the accepted common-core deterministic regular-mesh rebuild. The latter remains
the core contract's independent identity check; it is not permission for the
organization adapter to ignore the stored mesh.

The central mask must bit-equal the accepted component and satisfy, for every
stored center in cell order:

```text
central[i] = abs(center.x) <= 320.0 && abs(center.y) <= 160.0
```

Its hash and true count must equal the accepted resolution manifest. Recompute
the manifest's area witness with its original ordinary stored-cell-order sum and
require bit equality. The projection's independently stored `physical_area_km2`
uses the Neumaier vocabulary below and closes to that witness under
`1e-8 km2 + 5e-12*max(abs(actual),abs(expected))`; the two reduction algorithms
are not required to share bits. No cell polygon is clipped for S0, D0, O0a or
O0b membership.

## Arithmetic and distribution vocabulary

New evidence reductions use binary64, one thread and the orders below. No FMA,
parallel reduction, quantization, result-dependent bin or arm-dependent weight
is legal. A sum uses this Neumaier recurrence in the declared input order:

```text
sum = +0.0
correction = +0.0
for x:
  next = sum + x
  if abs(sum) >= abs(x):
    correction = correction + ((sum - next) + x)
  else:
    correction = correction + ((x - next) + sum)
  sum = next
result = sum + correction
```

The sole arithmetic exception is `OrganizationNumericalDiscrepancyV0`: its
surface and ledger reductions are inherited bytes from the numerical amendment
and therefore use that amendment's ordinary stored-order binary64 arithmetic,
not this Neumaier recurrence. Cross-arm base-surface discrepancy introduced
below is new evidence and does use Neumaier.

Canonicalize a final exact zero to positive zero; do not canonicalize any
nonzero. Integer additions are checked. Direct arrays scan stored cell order.
Object arrays scan accepted common ID order. Portal arrays scan portal ID.
Threshold arrays scan `[1000.0,2000.0,4000.0]`.

```rust
pub struct WeightedDistributionV0 {
    pub population_count: u64,
    pub available_count: u64,
    pub unavailable_count: u64,
    pub total_weight: f64,
    pub minimum: Option<f64>,
    pub p05: Option<f64>,
    pub p25: Option<f64>,
    pub p50: Option<f64>,
    pub p75: Option<f64>,
    pub p95: Option<f64>,
    pub maximum: Option<f64>,
    pub mean: Option<f64>,
    pub rms: Option<f64>,
}
```

Available values and weights must be finite; weights are strictly positive.
Cell distributions use physical cell area. Object distributions explicitly use
unit weight, so their `total_weight==available_count as f64`. Sort quantile
items by `(value.total_cmp, stable_id)` and return the lowest value whose
Neumaier cumulative weight is `>= p*total_weight`, evaluating multiplication
before comparison. Minimum/maximum are first/last sorted values. Mean is
`sum(weight*value)/total_weight`; RMS is
`sqrt(sum(weight*(value*value))/total_weight)`. Empty populations have counts
and positive-zero weight with every optional statistic `None`. An unavailable
value increments `unavailable_count`; it never enters weight or becomes zero.

The projection stores summaries, not fitted histograms or raw sorted copies.
Any later distributional test requiring other quantiles or bins is a new schema,
not a post-result choice.

## Arm-evidence and projection schemas

```rust
pub struct OrganizationArmEvidenceV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub identity: OrganizationArtifactIdentityV0,
    pub input_bundle_hash: u64,
    pub input_resolution_hash: u64,
    pub arm_result_hash: u64,
    pub native_provenance_hash: u64,
    pub common_core_hash: u64,
    pub reference_o0a_hash: u64,
    pub central_projection_hash: u64,
    pub g_authored_d0_hash: Option<u64>,
    pub derived_arm_evidence_hash: u64,
}

pub struct OrganizationCentralProjectionV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub identity: OrganizationArtifactIdentityV0,
    pub input_bundle_hash: u64,
    pub input_resolution_hash: u64,
    pub arm_result_hash: u64,
    pub common_core_hash: u64,
    pub reference_o0a_hash: u64,
    pub central_mask_hash: u64,
    pub cell_summary: CentralCellSummaryV0,
    pub object_summary: CentralObjectSummaryV0,
    pub primary_highlands: Vec<CentralHighlandProjectionV0>,
    pub context_highlands: Vec<CentralHighlandProjectionV0>,
    pub saddles: Vec<CentralSaddleProjectionV0>,
    pub drainage_scales: Vec<CentralDrainageScaleV0>,
    pub relationship_summary: CentralRelationshipSummaryV0,
    pub forcing_summary: LinkedForcingProbeSummaryV0,
    pub transfer_probe: TransferProbeV0,
    pub derived_projection_hash: u64,
}
```

`g_authored_d0_hash` is `Some` exactly for G base evidence and `None` for H/C
and numerical sensitivities. The central projection binds reference O0a only.
The arm-evidence root is the semantic manifest joining accepted child hashes;
it does not duplicate their large arrays.

## Central cell evidence

```rust
pub struct RadiusDistributionV0 {
    pub radius_km: f64,
    pub values_km: WeightedDistributionV0,
    pub truncated_cell_count: u64,
}

pub struct CentralCellSummaryV0 {
    pub cell_count: u64,
    pub physical_area_km2: f64,
    pub local_runoff_km3_myr: f64,
    pub elevation_km: WeightedDistributionV0,
    pub physical_grade: WeightedDistributionV0,
    pub local_relief: Vec<RadiusDistributionV0>,
}
```

Use only true central-mask cells as distribution members, but compute their
grade and radius neighborhoods against the full whole-graph physical surface.
Physical grade is the accepted G0/S0 face-transmissibility-weighted least-
squares grade. Rank-deficient fits are unavailable. Local relief at exactly
25/50/100 km uses the accepted physical-center-distance neighborhood and full
whole-graph scored population. A central cell is `truncated` at a radius under
the accepted S0 physical/scored-domain test; truncation is reported but the
finite relief remains available. Radius records use configuration order.

Implementation exposes one versioned `build_common_planar_fields_v0` helper for
the grade and radius arrays. It shares the accepted S0 kernels and must reproduce
their raw per-cell value/availability bits. Its regression validator then uses
S0's existing ordinary `sum::<f64>()` total and `cumulative += area` weighted-
quantile reducer to reproduce every embedded highland summary bit-for-bit. New
projection summaries separately use the Neumaier vocabulary; they are not
required to equal legacy S0 summary bits. The projection may not duplicate
private field-formula variants.

## Central highlands, saddles and object summaries

```rust
pub enum CentralHighlandCohortV0 { PrimaryAnchor, ContextFootprint }

pub struct HighlandForcingProbeV0 {
    pub segment_id: u32,
    pub whole_footprint_stencil_integral: f64,
    pub central_footprint_stencil_integral: f64,
    pub centroid_to_segment_km: f64,
    pub acute_axis_difference_rad: Option<f64>,
}

pub struct CentralHighlandProjectionV0 {
    pub peak_id: u32,
    pub cohort: CentralHighlandCohortV0,
    pub peak_anchor_cell: u32,
    pub central_member_cells: Vec<u32>,
    pub central_footprint_area_km2: f64,
    pub central_local_runoff_km3_myr: f64,
    pub central_footprint_fraction: f64,
    pub central_mask_crossing: bool,
    pub physical_domain_contact: bool,
    pub forcing_probes: Vec<HighlandForcingProbeV0>,
}

pub struct CentralSaddleProjectionV0 {
    pub saddle_id: u32,
    pub central_backing_cells: Vec<u32>,
    pub elder_peak_id: u32,
    pub losing_peak_ids: Vec<u32>,
    pub equal_elder_ambiguous: bool,
}

pub struct KeyedDistributionV0 {
    pub key: f64,
    pub values: WeightedDistributionV0,
}

pub struct HighlandCohortSummaryV0 {
    pub cohort: CentralHighlandCohortV0,
    pub object_count: u64,
    pub central_footprint_area_km2: f64,
    pub persistence_km: WeightedDistributionV0,
    pub whole_footprint_area_km2: WeightedDistributionV0,
    pub equivalent_length_km: WeightedDistributionV0,
    pub equivalent_width_km: WeightedDistributionV0,
    pub anisotropy: WeightedDistributionV0,
    pub local_relief_p50_km: Vec<KeyedDistributionV0>,
    pub local_relief_p90_km: Vec<KeyedDistributionV0>,
    pub summit_cap_fraction: Vec<KeyedDistributionV0>,
    pub summit_cap_valid_grade_fraction: Vec<KeyedDistributionV0>,
    pub summit_cap_gentle_fraction: Vec<CapGradeDistributionV0>,
    pub summit_cap_merge_censored_count: Vec<KeyedCountV0>,
}

pub struct KeyedCountV0 { pub key: f64, pub count: u64 }

pub struct CapGradeDistributionV0 {
    pub cap_depth_km: f64,
    pub grade_threshold: f64,
    pub values: WeightedDistributionV0,
}

pub struct CentralObjectSummaryV0 {
    pub highland_cohorts: Vec<HighlandCohortSummaryV0>,
    pub saddle_elevation_km: WeightedDistributionV0,
    pub saddle_persistence_drop_km: WeightedDistributionV0,
    pub saddle_incident_peak_count: WeightedDistributionV0,
}
```

For a reference highland, `peak_anchor_cell` is its `PeakBranchV0.anchor_cell`.
It is primary exactly when that cell is central. Otherwise it is context exactly
when at least one `footprint_member` is central. Every reference highland is
therefore in exactly one of primary, context or omitted-outside. Primary and
context arrays are separately sorted by `peak_id`; summaries occur in enum
order and never combine them.

`central_member_cells` is the ascending intersection of the complete S0
footprint with the mask. Its area and runoff are Neumaier sums of stored common
values in that order. Fraction is central area divided by the complete S0
footprint area. `central_mask_crossing` means the complete footprint has at
least one central and at least one noncentral member; it is deliberately a
cell-centre-mask fact, not polygon containment. `physical_domain_contact` is
true when the complete branch owns at least one S0 physical boundary segment.
Existing S0 truncation/censorship flags remain authoritative and are not
rewritten as central censorship.

A saddle is retained when any `flat_saddle_cell` is central. Backing cells are
the ascending central intersection. Preserve the accepted elder and complete
sorted losing-peak vector; no pairwise reduction is substituted. Saddle
`persistence_drop` is `min(z_elder_peak-z_saddle,
z_losing_peak-z_saddle)` over the complete incident set in losing-ID order;
nonfinite or negative values fail. Incident count is one plus losing count.
Saddles have no O0b identity and support only cohort summaries.

All highland morphology distributions are unit-weighted whole-object fields
from the accepted S0 reference measurement, stratified by central cohort.
Radius keys use `[25,50,100]`, cap keys `[0.25,0.5,1.0]`, and cap/grade keys use
cap outer then `[0.005,0.010,0.020]` inner. S0 retains finite censored cap values,
but projection cap/gentle distributions mark each `cap_merge_censored` object
unavailable and exclude its value/weight; the semantic per-depth censored count
is retained. Object cap/gentle materiality likewise becomes `MissingQuantity`
when either matched object is censored. Thus censorship cannot produce a
supported difference. Ambiguous orientation is reflected by the
anisotropy value but has no orientation angle.

## Linked forcing probes

For each retained highland and each linked segment in ascending segment ID:

```text
whole stencil integral = sum_{i in footprint cell order}(stencil[i] * A[i])
central stencil integral = sum_{i in central_member_cells}(stencil[i] * A[i])
```

Both use the exact stored compiled f64 stencil and Neumaier recurrence. They are
fractions of each segment's normalized support, not rock volume or a score.

Distance uses the highland's accepted whole-footprint centroid and the finite
scenario segment `[start,end]`. With `v=end-start`, require `dot(v,v)>0`, set
`u=clamp(dot(centroid-start,v)/dot(v,v),0,1)`, and return the Euclidean distance
to `start+u*v`. Orientation is `None` when S0 marks the footprint ambiguous;
otherwise normalize the segment direction and S0 sign-canonical principal
axis, then compute `acos(clamp(abs(dot(axis,direction)),0,1))`. It is axial and
lies in `[0,pi/2]`; no segment is selected as a winner.

```rust
pub struct SegmentForcingSummaryV0 {
    pub segment_id: u32,
    pub primary_whole_stencil_integral: WeightedDistributionV0,
    pub primary_centroid_distance_km: WeightedDistributionV0,
    pub primary_axis_difference_rad: WeightedDistributionV0,
    pub context_whole_stencil_integral: WeightedDistributionV0,
}

pub struct LinkedForcingProbeSummaryV0 {
    pub segments: Vec<SegmentForcingSummaryV0>,
}
```

These are unit-weighted object distributions. The axis distribution records
ambiguous objects as unavailable. Segment rows are ascending and complete even
when an object cohort is empty.

## Central D0 projection

```rust
pub struct CentralReachProjectionV0 {
    pub reach_id: u32,
    pub outlet_portal_id: u32,
    pub whole_exclusive_area_km2: f64,
    pub whole_exclusive_runoff_km3_myr: f64,
    pub whole_nested_area_km2: f64,
    pub whole_nested_runoff_km3_myr: f64,
    pub central_exclusive_cell_count: u64,
    pub central_exclusive_area_km2: f64,
    pub central_exclusive_runoff_km3_myr: f64,
    pub central_nested_area_km2: f64,
    pub central_nested_runoff_km3_myr: f64,
}

pub struct CentralPortalContributionV0 {
    pub portal_id: u32,
    pub whole_exclusive_cell_count: u64,
    pub whole_exclusive_area_km2: f64,
    pub whole_exclusive_runoff_km3_myr: f64,
    pub central_exclusive_cell_count: u64,
    pub central_exclusive_area_km2: f64,
    pub central_exclusive_runoff_km3_myr: f64,
}

pub enum PortalTrunkRoleV0 { GreatestSupply, LongestTrunk, HighestOrder }

pub struct CentralTrunkProjectionV0 {
    pub portal_id: u32,
    pub role: PortalTrunkRoleV0,
    pub complete_reach_ids_source_to_outlet: Vec<u32>,
    pub whole_exclusive_area_km2: f64,
    pub whole_exclusive_runoff_km3_myr: f64,
    pub central_exclusive_area_km2: f64,
    pub central_exclusive_runoff_km3_myr: f64,
}

pub struct CentralDrainageScaleV0 {
    pub support_threshold_km2: f64,
    pub reaches: Vec<CentralReachProjectionV0>,
    pub portals: Vec<CentralPortalContributionV0>,
    pub trunks: Vec<CentralTrunkProjectionV0>,
    pub reach_length_km: WeightedDistributionV0,
    pub reach_tail_area_km2: WeightedDistributionV0,
    pub reach_tail_runoff_km3_myr: WeightedDistributionV0,
    pub reach_strahler_order: WeightedDistributionV0,
    pub central_exclusive_area_km2: WeightedDistributionV0,
    pub central_exclusive_runoff_km3_myr: WeightedDistributionV0,
}
```

Project all three registered D0 scales in threshold order. At one scale, scan
the complete `exclusive_owner` array in central cell order. Attribute each cell
to its exact `Reach(id)` or `Portal(id)` owner and add stored physical area and
local runoff. Retain a reach exactly when its central exclusive area or runoff
is positive; with accepted positive uniform runoff these conditions coincide.
Portal rows include every semantic portal even when zero. Reach rows sort by
reach ID and portal rows by portal ID.

Whole reach exclusive/nested fields copy the validated D0 catchment ledgers.
Whole portal-exclusive fields scan the complete D0 `exclusive_owner` array,
not terminal basin ownership; whole and central portal rows therefore have the
same population meaning and partition with reach-exclusive rows. D0 terminal-
basin portal ledgers remain available in the bound common core but are not
mislabelled portal-exclusive contribution here. For a retained reach, nested central contribution is
defined recursively as its own exclusive value followed by each
`child_reaches` nested value in the accepted stored child-vector order. Evaluate
roots in ascending reach ID with memoization, reject a cycle, and require a
repeated request for an already evaluated reach to reuse its exact stored bits
rather than add it twice. At every scale, retained reach
exclusive plus complete portal rows must close bit-for-bit to fresh Neumaier
central cell sums after the same owner-grouped reduction is itself recomputed;
the final physical close predicate is the D0 area/runoff tolerance. Nested
contributions are not added into that partition ledger.

Trunk role vectors are copied without ID changes from D0. Their central
contribution is the sum of disjoint central exclusive contributions owned by
the listed reaches, vector order outer and cell order inner. Rows sort portal ID
then role enum; whole contribution uses the corresponding whole exclusive
values in the same order. The object distributions are unit-weighted over retained
reaches. Whole reach length/tail area/tail runoff/Strahler come directly from
D0; central contributions come from this projection.

## Reference-O0a projection

```rust
pub struct CentralRelationshipSummaryV0 {
    pub reference_backed_face_indices: Vec<u32>,
    pub highland_boundary_peak_ids: Vec<u32>,
    pub saddle_association_keys: Vec<SaddleAssociationKeyV0>,
    pub reach_cross_section_ids: Vec<u32>,
    pub candidate_boundary_length_km: WeightedDistributionV0,
    pub bilateral_descent_length_ratio: WeightedDistributionV0,
    pub cross_section_relative_relief_span_km: WeightedDistributionV0,
    pub cross_section_boundary_relief_km: WeightedDistributionV0,
}

pub struct SaddleAssociationKeyV0 {
    pub saddle_id: u32,
    pub elder_peak_id: u32,
    pub losing_peak_id: u32,
}
```

The complete reference O0a backed-face index namespace is retained as
`0..face_count`; faces are not copied, filtered or renumbered in the projection.
This preserves every association and probe reference. The selected relationship
IDs then mean:

- highland boundary row when its peak is primary central-anchored;
- saddle association when its elder **or** losing peak is primary, or either
  association owner is a `Reach(id)` retained by the central reference-scale
  projection; saddle-backing centrality alone neither includes nor excludes the
  relationship; and
- reach cross-section probe when its reference-scale 2,000 km2 reach has a
  retained central projection.

Arrays use accepted source order after verifying it is ascending by its frozen
key. Boundary-length and bilateral-ratio distributions are unit-weighted over
selected highland rows, with `peak_id` as stable ID. Cross-section distributions
are unit-weighted over selected **reach** rows, not stations: for each reach,
scan stations by arclength; reduce available `relative_relief_span_km` values in
station order, and available boundary-relief values in station order with left
before right, using Neumaier sum divided by available count. That per-reach mean
is the distribution value and `reach_id` the stable ID. A reach with no available
value increments the relevant unavailable count once. Optional ratios/spans/
reliefs remain unavailable rather than zero. This projection does not concatenate
the ten O0a configuration sensitivities and does not count them as votes.

## Transfer probe

```rust
pub struct ProbeForcingIntegralV0 {
    pub segment_id: u32,
    pub stencil_integral: f64,
}

pub struct ProbeDrainageContributionV0 {
    pub support_threshold_km2: f64,
    pub reaches: Vec<ProbeReachContributionV0>,
    pub portals: Vec<ProbePortalContributionV0>,
}

pub struct ProbeReachContributionV0 {
    pub reach_id: u32,
    pub outlet_portal_id: u32,
    pub whole_exclusive_area_km2: f64,
    pub whole_exclusive_runoff_km3_myr: f64,
    pub whole_nested_area_km2: f64,
    pub whole_nested_runoff_km3_myr: f64,
    pub probe_exclusive_cell_count: u64,
    pub probe_exclusive_area_km2: f64,
    pub probe_exclusive_runoff_km3_myr: f64,
    pub probe_nested_area_km2: f64,
    pub probe_nested_runoff_km3_myr: f64,
    pub probe_mask_crossing: bool,
}

pub struct ProbePortalContributionV0 {
    pub portal_id: u32,
    pub whole_exclusive_cell_count: u64,
    pub whole_exclusive_area_km2: f64,
    pub whole_exclusive_runoff_km3_myr: f64,
    pub probe_exclusive_cell_count: u64,
    pub probe_exclusive_area_km2: f64,
    pub probe_exclusive_runoff_km3_myr: f64,
    pub probe_mask_crossing: bool,
}

pub struct ProbeCellSummaryV0 {
    pub cell_count: u64,
    pub physical_area_km2: f64,
    pub local_runoff_km3_myr: f64,
    pub elevation_km: WeightedDistributionV0,
    pub physical_grade: WeightedDistributionV0,
    pub local_relief: Vec<RadiusDistributionV0>,
}

pub struct TransferProbeV0 {
    pub bounds_km: [f64; 4],
    pub cell_summary: ProbeCellSummaryV0,
    pub saddle_ids: Vec<u32>,
    pub saddle_backing_cells: Vec<TransferSaddleBackingV0>,
    pub drainage: Vec<ProbeDrainageContributionV0>,
    pub forcing_integrals: Vec<ProbeForcingIntegralV0>,
}

pub struct TransferSaddleBackingV0 {
    pub saddle_id: u32,
    pub probe_backing_cells: Vec<u32>,
    pub probe_mask_crossing: bool,
}
```

Bounds are exactly `[min_x,max_x,min_y,max_y]=[-120,40,-72,72] km` and cell
membership is inclusive center membership. Cell grade/relief again use the
whole-domain common fields before filtering. A saddle is retained when any
backing cell is in the box; keep every such cell ascending and every saddle ID
ascending. Its mask-crossing flag is true when the complete backing set contains
both probe and nonprobe cells. Drainage uses the same exclusive-owner and nested-parent algorithms
as the central projection with the transfer mask substituted. Portal rows are
complete and every scale closes to the fresh box area/runoff sums. Forcing rows
are ascending segment ID and sum `stencil[i]*A[i]` over box cells.

Reach/portal `probe_mask_crossing` is true when that object's complete exclusive
owner support contains at least one probe and one nonprobe cell. This is the
registered transfer-box boundary-contact evidence. It is a cell-centre mask
fact, distinct from physical-domain contact and not polygon clipping.

The box creates no privileged saddle, reach, pass or through-going object. An
empty saddle/reach population is valid evidence. `bounds_km` are stored and
validated rather than accepted as runtime flags.

## Authored G versus independent D0

D0 is fully built and validated without reading native G. Only then may this
sidecar read the G forest checkpoint bound by native provenance. Cell indices
are comparable because both arrays use the same accepted mesh at the same
resolution; this is an explicit architecture probe, not persistent identity.

```rust
pub enum ReceiverAgreementV0 {
    ExactInternalReceiver,
    ExactPortalReceiver,
    SameClassDifferentTarget,
    DifferentClass,
}

pub struct DifferenceDistributionV0 {
    pub signed: WeightedDistributionV0,
    pub absolute: WeightedDistributionV0,
    pub symmetric_relative: WeightedDistributionV0,
}

pub struct SupportContingencyV0 {
    pub support_threshold_km2: f64,
    pub both_count: u64,
    pub g_only_count: u64,
    pub d0_only_count: u64,
    pub neither_count: u64,
    pub both_area_km2: f64,
    pub g_only_area_km2: f64,
    pub d0_only_area_km2: f64,
    pub neither_area_km2: f64,
    pub both_local_runoff_km3_myr: f64,
    pub g_only_local_runoff_km3_myr: f64,
    pub d0_only_local_runoff_km3_myr: f64,
    pub neither_local_runoff_km3_myr: f64,
}

pub struct GPortalD0ComparisonV0 {
    pub portal_id: u32,
    pub g_owned_cell_count: u64,
    pub d0_owned_cell_count: u64,
    pub g_accumulated_area_km2: f64,
    pub d0_accumulated_area_km2: f64,
    pub g_accumulated_runoff_km3_myr: f64,
    pub d0_accumulated_runoff_km3_myr: f64,
}

pub struct GAuthoredClaimDispositionV0 {
    pub terminal_portal_area_mismatch_fraction: f64,
    pub terminal_portal_runoff_mismatch_fraction: f64,
    pub support_area_mismatch_fractions: Vec<ThresholdFractionV0>,
    pub support_runoff_mismatch_fractions: Vec<ThresholdFractionV0>,
    pub architecture_correspondence_claim: ClaimDispositionV0,
}

pub struct ThresholdFractionV0 {
    pub support_threshold_km2: f64,
    pub fraction: f64,
}

pub struct OrganizationGAuthoredD0V0 {
    pub schema_version: String,
    pub hash_version: String,
    pub identity: OrganizationArtifactIdentityV0,
    pub arm_result_hash: u64,
    pub native_provenance_hash: u64,
    pub forest_checkpoint_hash: u64,
    pub common_core_hash: u64,
    pub d0_evidence_hash: u64,
    pub receiver_agreement_counts: [u64; 4],
    pub receiver_agreement_area_km2: [f64; 4],
    pub receiver_agreement_runoff_km3_myr: [f64; 4],
    pub terminal_portal_equal_count: u64,
    pub terminal_portal_equal_area_km2: f64,
    pub terminal_portal_equal_runoff_km3_myr: f64,
    pub accumulated_area_difference_km2: DifferenceDistributionV0,
    pub accumulated_runoff_difference_km3_myr: DifferenceDistributionV0,
    pub support: Vec<SupportContingencyV0>,
    pub portals: Vec<GPortalD0ComparisonV0>,
    pub d0_fill_supported_cell_count: u64,
    pub d0_flat_supported_cell_count: u64,
    pub d0_physically_non_descending_cell_count: u64,
    pub claim_disposition: GAuthoredClaimDispositionV0,
    pub derived_g_authored_d0_hash: u64,
}
```

For each cell in stored order, compare G `Cell(j)` with D0 `Cell { cell:j,.. }`
or G `Portal(p)` with D0 `Portal { portal_id:p,.. }`. D0 boundary-segment and G
directed-edge provenance do not have to match. Receiver categories are exclusive
and exhaustive in enum order. Derive G terminal portal by following its already-
validated forest; compare it with D0 `outlet_portal_id`.

Difference values are `g-d0`, `abs(g-d0)` and
`2*abs(g-d0)/(abs(g)+abs(d0))`. The relative value is unavailable exactly when
both operands are zero. Each distribution uses local physical cell area as its
weight and cell ID as stable ID. Nonnegative accumulated area/runoff is required.

At each registered threshold, G support is
`g_accumulated_area>=threshold`; D0 support is
`d0_structural_area>=threshold`. Accumulate the four contingency categories in
cell order, including count, local physical area and local runoff. Recompute G's
stored support count and D0's support membership before comparing. Portal rows
use portal ID order. Owned-cell counts scan terminal portal for all cells. G
area/runoff values come from the validated G portal ledger; D0 values come from
its validated `portal_ledgers`, which already sum accumulated root quantities at
immediate portal receivers. Do not sum nested quantities for every terminal-
owned cell. Both G and D0 portal totals must independently close to their own
validated domain totals.

Terminal mismatch fraction is
`(domain_total-equal_total)/domain_total`. Threshold-support mismatch is
`(g_only+d0_only)/domain_total`, with additions before division. The
architecture-correspondence claim is `Supported` at one resolution exactly
when terminal-portal mismatch is at most `0.10` of both domain physical area and
local runoff and, at every threshold, `g_only+d0_only` is at most `0.10` of both
domain area and runoff. Exact receiver agreement and accumulation-difference
distributions are retained but are not gates: the two graph constructions are
not expected to select identical local trees. The fixed 10% semantic-support
band is preregistered as a broad causal correspondence requirement, not fitted
to a result. Failure is `MaterialConflict`; nonfinite/zero domain denominators
are instrument failures. A scale-robust authored-G claim requires `Supported`
at 8, 4 and 2 km; one failed rung blocks that claim without invalidating G.

G and D0 Strahler vectors are reported separately through native/common
artifacts. They are not put in this same-cell structure or used to break a
support disagreement.

## Required core-backed O0b matrix

Build `CoreObjectCorrespondenceV1` in these 22 ordered directions after all
required cores freeze:

```text
H base 4->8, 8->4, 4->2, 2->4
C base 4->8, 8->4, 4->2, 2->4
G base 4->8, 8->4, 4->2, 2->4
H base 4->C base 4, C base 4->H base 4
H base 4->G base 4, G base 4->H base 4
C base 4->G base 4, G base 4->C base 4
H base 4->H sensitivity 4, H sensitivity 4->H base 4
C base 4->C sensitivity 4, C sensitivity 4->C base 4
```

No 8-to-2 artifact, sensitivity cross-arm artifact, opportunity-control
artifact or G-repeat artifact is implied. Same-mesh pairs still execute the
accepted polygon/line mechanics; no array resampling or cell-ID identity is
substituted for O0b.

Materiality has one frozen directional authority for each comparison. Cross-arm
records use only `H base 4->C base 4`, `H base 4->G base 4` and
`C base 4->G base 4`; an object anchored on the later arm reads that artifact's
`Target` assignment. Resolution stability uses only base `4->8` and `4->2` for
the operand arm. Numerical stability uses only base `4->sensitivity 4`.
Reverse-direction artifacts remain required descriptive/symmetry evidence but
never substitute as scalar authority. Their independently hashed assignments
need not be byte-interchangeable with the authoritative direction.

```rust
pub enum CentralCohortStatusV0 { PrimaryAnchor, ContextFootprint, Outside }

pub struct PartnerCohortV0 {
    pub partner_id: u32,
    pub cohort: CentralCohortStatusV0,
}

pub struct CentralAssignmentAnnotationV0 {
    pub side: PacketSideV0,
    pub family: ObjectFamilyV0,
    pub object_id: u32,
    pub channel: AssignmentChannelV0,
    pub object_cohort: CentralCohortStatusV0,
    pub positive_partners: Vec<PartnerCohortV0>,
    pub maximum_partners: Vec<PartnerCohortV0>,
}

pub struct CentralComponentAnnotationV0 {
    pub channel: AssignmentChannelV0,
    pub kind: ComponentKindV0,
    pub members: Vec<CentralBestMemberV0>,
    pub central_involved: bool,
}

pub struct CentralBestMemberV0 {
    pub side: PacketSideV0,
    pub family: ObjectFamilyV0,
    pub object_id: u32,
    pub cohort: CentralCohortStatusV0,
}
```

Annotations join complete immutable O0b assignments/components with source and
target projections. They preserve every positive/max partner and exact tie.
Classify a component before annotating/filtering it; `central_involved` means at
least one member is primary or context. Drainage objects use `PrimaryAnchor`
when their reference-scale reach projection exists and `Outside` otherwise;
there is no drainage context cohort. Context/primary transitions are records,
not object births, splits or merges.

## Object compatibility and null semantics

```rust
pub enum ObjectCompatibilityV0 {
    CompatibleOneToOneMajority,
    NoPositiveOverlap,
    NoExclusiveSupport,
    HierarchyAmbiguousSupport,
    ExactBestTie,
    NonReciprocalMaximum,
    NonOneToOneComponent,
    CoverageBelowHalf,
    MetricConflict,
    OutsideRequiredCohort,
}
```

For highland scalar comparison, use the exclusive-area channel. A source and
target are compatible exactly when both supports are eligible, each has one
maximum partner and those maxima are reciprocal, their best component is
`OneToOneBest`, the positive exclusive pair has both source and target coverage
`>=0.5`, neither side has an exact tie, and both are primary when a material
central-object claim is requested. Every central peak-anchored reference
highland is principal; there is no largest-N, observed-area or best-looking
subset.

Drainage catchment quantities use the exclusive-area channel with the same
reciprocal/one-to-one/majority rule. Reach geometry and cross-section quantities
use the line channel and both line coverages `>=0.5`. A quantity requiring both
catchment and line meaning additionally requires the same partner in both
channels and no `MetricConflict`. Positive but sub-majority pairs remain O0b
evidence but do not support scalar object materiality.

Test compatibility in the enum-order failure priority shown above after
`Compatible`: eligibility/null, tie, reciprocity, component, coverage, metric
conflict, cohort. Missing, tied or incompatible quantities store no fabricated
partner value or difference. One-to-many, many-to-one and many-to-many
components remain categorical vector evidence and are never averaged. Saddles
have no compatibility record.

## Quantity registry and materiality

```rust
pub enum DistributionStatisticV0 {
    Minimum, P05, P25, P50, P75, P95, Maximum, Mean, Rms,
}

pub enum EvidenceQuantityV0 {
    CellElevationKm(DistributionStatisticV0),
    CellGrade(DistributionStatisticV0),
    CellReliefKm { radius_km: f64, statistic: DistributionStatisticV0 },
    HighlandCount,
    HighlandPersistenceKm(DistributionStatisticV0),
    HighlandFootprintAreaKm2(DistributionStatisticV0),
    HighlandLengthKm(DistributionStatisticV0),
    HighlandWidthKm(DistributionStatisticV0),
    HighlandAnisotropy(DistributionStatisticV0),
    HighlandReliefP50Km { radius_km: f64, statistic: DistributionStatisticV0 },
    HighlandReliefP90Km { radius_km: f64, statistic: DistributionStatisticV0 },
    HighlandCapFraction { depth_km: f64, statistic: DistributionStatisticV0 },
    HighlandGentleFraction { depth_km: f64, grade: f64, statistic: DistributionStatisticV0 },
    SaddleCount,
    DrainageReachCount { support_km2: f64 },
    DrainageReachLengthKm { support_km2: f64, statistic: DistributionStatisticV0 },
    DrainageTailAreaKm2 { support_km2: f64, statistic: DistributionStatisticV0 },
    DrainageTailRunoffKm3Myr { support_km2: f64, statistic: DistributionStatisticV0 },
    DrainageStrahler { support_km2: f64, statistic: DistributionStatisticV0 },
    DrainageCentralExclusiveAreaKm2 { support_km2: f64, statistic: DistributionStatisticV0 },
    DrainageCentralExclusiveRunoffKm3Myr { support_km2: f64, statistic: DistributionStatisticV0 },
    ForcingStencilIntegral { segment_id: u32, statistic: DistributionStatisticV0 },
    ForcingCentroidDistanceKm { segment_id: u32, statistic: DistributionStatisticV0 },
    ForcingAxisDifferenceRad { segment_id: u32, statistic: DistributionStatisticV0 },
    O0aBoundaryLengthKm(DistributionStatisticV0),
    O0aBilateralDescentRatio(DistributionStatisticV0),
    O0aCrossSectionRelativeSpanKm(DistributionStatisticV0),
    O0aCrossSectionBoundaryReliefKm(DistributionStatisticV0),
    TransferElevationKm(DistributionStatisticV0),
    TransferGrade(DistributionStatisticV0),
    TransferReliefKm { radius_km: f64, statistic: DistributionStatisticV0 },
    TransferStencilIntegral { segment_id: u32 },
}

pub enum ObjectQuantityV0 {
    HighlandPersistenceKm,
    HighlandFootprintAreaKm2,
    HighlandLengthKm,
    HighlandWidthKm,
    HighlandAnisotropy,
    HighlandReliefP50Km { radius_km: f64 },
    HighlandReliefP90Km { radius_km: f64 },
    HighlandCapFraction { depth_km: f64 },
    HighlandGentleFraction { depth_km: f64, grade: f64 },
    HighlandO0aBoundaryLengthKm,
    HighlandO0aBilateralDescentRatio,
    DrainageReachLengthKm { support_km2: f64 },
    DrainageTailAreaKm2 { support_km2: f64 },
    DrainageTailRunoffKm3Myr { support_km2: f64 },
    DrainageStrahler { support_km2: f64 },
    DrainageCentralExclusiveAreaKm2 { support_km2: f64 },
    DrainageCentralExclusiveRunoffKm3Myr { support_km2: f64 },
    DrainageO0aMeanRelativeSpanKm,
    DrainageO0aMeanBoundaryReliefKm,
}

pub enum ClaimDispositionV0 {
    Supported,
    NotMaterial,
    MaterialConflict,
    IncompatiblePopulation,
    MissingQuantity,
    DescriptiveOnly,
}

pub struct AggregateMaterialityV0 {
    pub arm_a: OrganizationArmV0,
    pub arm_b: OrganizationArmV0,
    pub quantity: EvidenceQuantityV0,
    pub compatible_population: Option<CompatiblePopulationV0>,
    pub value_a_4km: Option<f64>,
    pub value_b_4km: Option<f64>,
    pub cross_arm_absolute_difference: Option<f64>,
    pub arm_a_resolution_allowance: Option<f64>,
    pub arm_b_resolution_allowance: Option<f64>,
    pub arm_a_numerical_allowance: Option<f64>,
    pub arm_b_numerical_allowance: Option<f64>,
    pub roundoff_guard: Option<f64>,
    pub materiality_threshold: Option<f64>,
    pub disposition: ClaimDispositionV0,
}

pub struct CompatiblePopulationV0 {
    pub family: ObjectFamilyV0,
    pub legs: Vec<PopulationCompatibilityLegV0>,
    pub arm_a_primary_count: u64,
    pub arm_b_primary_count: u64,
    pub arm_a_joint_count: u64,
    pub arm_b_joint_count: u64,
    pub arm_a_primary_measure: f64,
    pub arm_b_primary_measure: f64,
    pub arm_a_joint_measure: f64,
    pub arm_b_joint_measure: f64,
    pub arm_a_joint_count_fraction: Option<f64>,
    pub arm_b_joint_count_fraction: Option<f64>,
    pub arm_a_joint_measure_fraction: Option<f64>,
    pub arm_b_joint_measure_fraction: Option<f64>,
    pub majority_compatible: bool,
}

pub enum PopulationLegRelationV0 {
    CrossArm4Km,
    Resolution8Km,
    Resolution2Km,
    NumericalSensitivity4Km,
}

pub struct PopulationCompatibilityLegV0 {
    pub anchor_arm: OrganizationArmV0,
    pub partner_arm: OrganizationArmV0,
    pub relation: PopulationLegRelationV0,
    pub primary_count: u64,
    pub compatible_count: u64,
    pub primary_measure: f64,
    pub compatible_measure: f64,
    pub count_fraction: Option<f64>,
    pub measure_fraction: Option<f64>,
    pub majority_compatible: bool,
}
```

Arm pairs are `(H,C),(H,G),(C,G)`. Quantity order is enum then embedded key.
Validate every radius/depth/grade/threshold/segment against the registered
families. Counts are converted exactly to f64 only below `2^53`.

Aggregate quantity families are exact:

| quantity variants | compatibility family | status |
|---|---|---|
| cell elevation/grade/relief | none | materiality-eligible over the fixed physical population |
| highland morphology; forcing; highland O0a boundary/descent | `Highland` | materiality-eligible over the joint stable cohort |
| drainage length/area/runoff/Strahler/central supply at exactly 2,000 km2; O0a cross-section | `DrainageNode` | materiality-eligible over the joint stable cohort |
| drainage aggregates at 1,000 or 4,000 km2 | none | `DescriptiveOnly` because O0b owns no identity at those scales |
| highland/reach/saddle counts; every transfer quantity | none | `DescriptiveOnly` |

For highlands, primary measure is the sum of each primary object's central
footprint area (nested footprints deliberately remain object measures). For
drainage, it is reference-scale central exclusive catchment area. A leg counts
an anchor exactly when its authoritative O0b direction reports
`CompatibleOneToOneMajority` in the required primary cohort/channel. Leg order
is cross-arm A anchor, cross-arm B anchor; then for A and B respectively 8 km,
2 km and, for H/C only, numerical sensitivity. Within-arm legs have identical
anchor/partner arm values; relation distinguishes their runs. The leg vector is
therefore exactly 8 rows for H/C and 7 for H/G or C/G.

The joint population contains only reciprocal 4 km cross-arm pairs for which
both objects also pass every required resolution and numerical leg. At 4 km,
each arm's scalar is recomputed over its joint anchors. At 8/2 and H/C
sensitivity, it is recomputed only over those anchors' unique compatible
partners. Thus no complete-cohort average can hide object turnover. Cell-field
quantities alone use `compatible_population=None` because their physical
whole/central population is fixed independently of object identity; every
object-derived eligible quantity requires `Some`.

The identity joint is never silently narrowed per run by ordinary unavailable-
value exclusion. For each aggregate quantity, require that every identity-joint
anchor and each of its registered 4/8/2/sensitivity partners has that exact
quantity available. If any one is unavailable or censored, the record is
`MissingQuantity`, all scalar options are `None`, and the child summaries retain
where availability failed. Only a complete-case identity joint is reduced, so
every `q4/q8/q2/sensitivity` uses the same anchor set mapped through its unique
partners.

A fraction is `None` exactly when its denominator is zero; otherwise it is
numerator divided by denominator. Each leg and the joint gate require nonzero
count and measure plus count and measure fractions `>=0.5`. Top-level
`majority_compatible` requires every leg and both joint arm fractions to pass.
An empty valid cohort therefore records `None` fractions and
`IncompatiblePopulation`, not NaN or instrument failure. A materiality-eligible
aggregate with a false gate does not evaluate the scalar threshold.

For a finite aggregate quantity `q`:

```text
r_arm = max(abs(q4-q8), abs(q4-q2))
n_H/C = abs(q_base4-q_sensitivity4)
n_G = +0.0
guard = 64*f64::EPSILON*max(1,abs(all q values used))
threshold = r_a + r_b + n_a + n_b + guard
difference = abs(q_a4-q_b4)
```

Operations occur in shown order. `Supported` means `difference>threshold` and
both arms have all required finite values and, for object-derived quantities,
the compatible-population gate. Cell-field quantities require
`compatible_population=None` and no object gate.
Otherwise a finite complete record is `NotMaterial`. A missing or invalid
required predecessor blocks comparison construction before a root exists. An
unavailable statistic inside an otherwise valid predecessor is
`MissingQuantity`. This is a conservative
resolution-and-integration dominance test, not a confidence interval or claim
that smaller differences are physically equal.

Aggregate option/disposition legality is exact:

| disposition | compatible population | scalar option fields |
|---|---|---|
| `Supported` or `NotMaterial` | `Some` with true gate for object-derived quantities; `None` for cell fields | both values, difference, both resolution allowances, both numerical allowances, guard and threshold are all `Some` |
| `IncompatiblePopulation` | `Some` with false gate | every scalar option is `None`; unmatched cohort summaries are not copied here |
| `MissingQuantity` | required population is absent only for cell fields or present with a true gate | every scalar option is `None`; availability remains visible in child summaries |
| `DescriptiveOnly` | `None` | 4 km arm values are independently `Some` when available and difference is `Some` iff both are; every allowance, guard and threshold is `None` |

`MaterialConflict` is illegal for `AggregateMaterialityV0`. For
`ObjectMaterialityV0`, `Supported/NotMaterial` require a compatible cross-arm
partner, two `Some` 4 km values, complete fixed-arity compatible stability rows
and `Some` threshold. `IncompatiblePopulation` requires both scalar values and
threshold `None`; `MissingQuantity` likewise stores them `None` but retains the
compatible partner and stability identities. `DescriptiveOnly` and
`MaterialConflict` are illegal for object materiality. These rules are checked
before hashing and JSON projection.

Top-level `partner_object_4km` follows the same unique-candidate rule as a
stability row: `None` for no positive candidate or an exact best tie; `Some`
for a unique maximum even when reciprocity, component, coverage, channel or
cohort makes it incompatible. A compatible-but-missing quantity also retains
`Some`. The typed `cross_arm_compatibility` explains why scalar fields are null.

Run this registry for central cells, primary highlands, the 2,000 km2 reference
D0 reach cohort, primary forcing probes and reference-O0a summaries. The 1,000
and 4,000 km2 drainage summaries remain present but descriptive. Transfer fields,
all object counts, context highlands, saddles, O0b topology/components, portal
roles, G/D0 exact receiver categories, cost and human observations are
`DescriptiveOnly`; no scalar formula makes them votes.

```rust
pub struct StabilityPartnerV0 {
    pub relation: StabilityRelationV0,
    pub partner_id: Option<u32>,
    pub compatibility: ObjectCompatibilityV0,
    pub base_value: Option<f64>,
    pub partner_value: Option<f64>,
    pub absolute_difference: Option<f64>,
}

pub enum StabilityRelationV0 { Resolution8Km, Resolution2Km, NumericalSensitivity4Km }

pub struct ObjectMaterialityV0 {
    pub arm_a: OrganizationArmV0,
    pub arm_b: OrganizationArmV0,
    pub family: ObjectFamilyV0,
    pub anchor_side: PacketSideV0,
    pub anchor_object_4km: u32,
    pub partner_object_4km: Option<u32>,
    pub cross_arm_compatibility: ObjectCompatibilityV0,
    pub quantity: ObjectQuantityV0,
    pub value_a_4km: Option<f64>,
    pub value_b_4km: Option<f64>,
    pub arm_a_stability: Vec<StabilityPartnerV0>,
    pub arm_b_stability: Vec<StabilityPartnerV0>,
    pub materiality_threshold: Option<f64>,
    pub disposition: ClaimDispositionV0,
}
```

Object records cover every primary 4 km object on both sides: source-side IDs
ascending, then target-side IDs ascending, then quantity. Compatible pairs
therefore normally have two directional records; this deliberate redundancy
ensures an arm-B-only primary object receives a typed missing/incompatibility
record instead of disappearing. `value_a/value_b` always follow the declared
arm pair regardless of anchor side. A cross-arm pair is scalar-eligible only
under the compatibility predicate. For
each operand, 8/2 and H/C sensitivity partners must independently be compatible
for the quantity's channel and primary cohort. G omits numerical sensitivity
and contributes exact positive zero. The threshold is the sum of the maximum
8/2 object drift for each arm, its H/C sensitivity drift, and the same roundoff
guard. Missing/tied/cardinality/context/coverage failures produce
`IncompatiblePopulation` with typed compatibility and `None` difference; they
never become extreme zero-valued observations.

Stability vectors are fixed-arity, not sparse. For each H or C operand they
contain exactly `[Resolution8Km, Resolution2Km,
NumericalSensitivity4Km]`; for each G operand they contain exactly
`[Resolution8Km, Resolution2Km]`. A missing partner remains as a row with
`partner_id=None`, typed compatibility and all three value/difference options
`None`. If an object is anchored only on the opposite side of the cross-arm
comparison, the absent operand still owns its complete ordered stability vector
with all partner/value options `None`; it is never represented by an empty
vector.

Each stability row has one legal option state:

| row state | `partner_id` | compatibility | values/difference |
|---|---|---|---|
| no unique candidate (none or exact tie) | `None` | exact typed incompatibility | all `None` |
| unique candidate but incompatible by reciprocity/component/coverage/cohort/channel | `Some` candidate | exact typed incompatibility | all `None` |
| compatible candidate but either quantity unavailable/censored | `Some` candidate | `CompatibleOneToOneMajority` | all `None` |
| compatible and both values available | `Some` candidate | `CompatibleOneToOneMajority` | base, partner and absolute difference all `Some` |

No partial value pair is legal. Absolute difference is recomputed in shown
operand order whenever present.

Object highland values come from the identified S0 feature and its reference
O0a highland row. Object drainage values are legal only at the reference
`support_km2=2000`; the 1,000/4,000 km2 scales support aggregate evidence but
have no O0b object identity. For a reach's O0a mean span or boundary relief,
scan stations by arclength and left then right, take only available finite
values, and use the Neumaier sum divided by available count; no available value
makes that quantity missing. Thus every individual value has one frozen source
rather than borrowing a cohort statistic.

Object quantity legality and channel are exact:

| object quantity | family | required O0b channel |
|---|---|---|
| persistence, footprint, length, width, anisotropy, relief, cap, gentle, highland O0a boundary/descent | `Highland` | `HighlandExclusiveArea` |
| reach tail area/runoff, Strahler, central exclusive area/runoff | `DrainageNode` | `DrainageExclusiveArea` |
| reach length, drainage O0a mean span/relief | `DrainageNode` | `DrainageLine` |

No registered object quantity requires both drainage channels. A foreign
family/channel combination rejects during standalone shape validation;
`MetricConflict` remains categorical O0b evidence but is not invented as a
tie-break for the registered single-channel quantities.

## Direct cross-arm surface discrepancy

Before numerical-sensitivity sidecars, compare the three 4 km base surfaces
directly on their shared stored mesh. This detects spatial rearrangement that
equal marginal distributions cannot reveal.

```rust
pub struct OrganizationCrossArmSurfaceDiscrepancyV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub arm_a: OrganizationArmV0,
    pub arm_b: OrganizationArmV0,
    pub arm_a_result_hash: u64,
    pub arm_b_result_hash: u64,
    pub arm_a_evidence_hash: u64,
    pub arm_b_evidence_hash: u64,
    pub signed_moment_km3: f64,
    pub signed_mean_km: f64,
    pub l1_mean_km: f64,
    pub rms_km: f64,
    pub maximum_abs_km: f64,
    pub disposition: ClaimDispositionV0,
    pub derived_cross_arm_surface_hash: u64,
}
```

Rows are exactly `(H,C),(H,G),(C,G)`. Require bit-equal shared input-resolution,
canonical graph and physical-area vectors. In stored cell order set
`delta=z_b-z_a` and evaluate the same signed moment/mean, L1, RMS and maximum
formulas as the numerical amendment, but with this amendment's Neumaier sums.
`disposition` is always `DescriptiveOnly`: these magnitudes retain direct
physical difference and spatial sensitivity but are not a scalar terrain-
quality vote. Each row publishes independently as a semantic pairwise artifact.

## Direct H/C numerical discrepancy artifacts

The evidence layer retains, rather than replaces, the numerical amendment's
same-cell/checkpoint/ledger handoff.

```rust
pub enum NumericalCheckpointKeyV0 {
    H { base_pass: u32, sensitivity_pass: u32 },
    C { time_myr: f64 },
}

pub struct SurfaceDiscrepancyV0 {
    pub key: NumericalCheckpointKeyV0,
    pub signed_moment_km3: f64,
    pub signed_mean_km: f64,
    pub l1_mean_km: f64,
    pub rms_km: f64,
    pub maximum_abs_km: f64,
}

pub enum NumericalLedgerQuantityV0 {
    InitialElevationMoment,
    GrossHoldRestoration,
    ForcingIntervalRockUplift,
    RelaxationIntervalRockUplift,
    TotalRockUplift,
    EffectiveDenudationExport,
    HillslopePortalTransferTotal,
    HillslopePortalTransfer { portal_id: u32 },
    WaterSupply,
    WaterPortalOutflowTotal,
    WaterPortalOutflow { portal_id: u32 },
    WaterUnresolvedSink,
    WaterBalanceError,
    FinalElevationMoment,
    SolidClosureError,
}

pub struct NumericalLedgerDifferenceV0 {
    pub quantity: NumericalLedgerQuantityV0,
    pub base_value: f64,
    pub sensitivity_value: f64,
    pub signed_sensitivity_minus_base: f64,
    pub absolute_difference: f64,
    pub symmetric_relative_difference: Option<f64>,
}

pub struct LimiterCountV0 {
    pub limiter: C0LimiterWireV0,
    pub accepted_step_count: u64,
}

pub struct NumericalRunStepDiagnosticsV0 {
    pub accepted_step_count: u64,
    pub candidate_attempt_count: u64,
    pub limiter_counts: Vec<LimiterCountV0>,
    pub minimum_accepted_dt_myr: Option<f64>,
    pub maximum_accepted_dt_myr: Option<f64>,
}

pub struct OrganizationNumericalDiscrepancyV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub arm: OrganizationArmV0,
    pub base_result_hash: u64,
    pub sensitivity_result_hash: u64,
    pub base_trace_hash: u64,
    pub sensitivity_trace_hash: u64,
    pub checkpoint_discrepancies: Vec<SurfaceDiscrepancyV0>,
    pub ledger_differences: Vec<NumericalLedgerDifferenceV0>,
    pub base_diagnostics: NumericalRunStepDiagnosticsV0,
    pub sensitivity_diagnostics: NumericalRunStepDiagnosticsV0,
    pub derived_numerical_discrepancy_hash: u64,
}
```

Only H and C at 4 km are legal, ordered H then C. H checkpoint keys are
`(0,0),(50,100),(120,240),(200,400)`; C keys are exact `[0,3,6,10]` Myr.
Surface fields use the numerical amendment's exact stored-cell-order
`delta=z_sensitivity-z_base` reductions and physical area. Ledger rows include
every quantity owned by the arm in enum/portal order from the **final native
completion checkpoint's cumulative ledger only**; intermediate checkpoint
ledgers are validated but emit no ledger-difference rows. Variants not owned are
absent, not zero. Signed difference is sensitivity minus base; symmetric
relative is evaluated exactly as
`(2.0*abs(base-sensitivity))/(abs(base)+abs(sensitivity))`, multiplying before
adding the denominator terms and dividing last. It is unavailable exactly when
both denominator terms are zero. Limiter rows are
complete enum order, including zero counts. Min/max dt are unavailable only
when no accepted step exists. Full validation recomputes from bound traces,
checkpoints and ledgers; these records are not inferred from central summaries.

Each discrepancy artifact is a direct comparison predecessor and semantic
child of the final comparison. It is not an arm result and does not decide
materiality by itself.

## Comparison schema and evidence vector

```rust
pub struct ArmEvidenceBindingV0 {
    pub identity: OrganizationArtifactIdentityV0,
    pub arm_result_hash: u64,
    pub arm_evidence_hash: u64,
    pub common_core_hash: u64,
    pub reference_o0a_hash: u64,
    pub central_projection_hash: u64,
}

pub struct CorrespondenceReductionV0 {
    pub ordinal: u32,
    pub source_arm_evidence_hash: u64,
    pub target_arm_evidence_hash: u64,
    pub correspondence_hash: u64,
    pub assignment_annotations: Vec<CentralAssignmentAnnotationV0>,
    pub component_annotations: Vec<CentralComponentAnnotationV0>,
}

pub struct GAuthoredBindingV0 {
    pub nominal_spacing_km: f64,
    pub arm_evidence_hash: u64,
    pub g_authored_d0_hash: u64,
    pub disposition: ClaimDispositionV0,
}

pub struct ComparisonEvidenceVectorV0 {
    pub cross_arm_surface_discrepancies: Vec<CrossArmSurfaceBindingV0>,
    pub numerical_discrepancies: Vec<NumericalDiscrepancyBindingV0>,
    pub aggregate_materiality: Vec<AggregateMaterialityV0>,
    pub object_materiality: Vec<ObjectMaterialityV0>,
    pub correspondence_reductions: Vec<CorrespondenceReductionV0>,
    pub g_authored: Vec<GAuthoredBindingV0>,
}

pub struct NumericalDiscrepancyBindingV0 {
    pub arm: OrganizationArmV0,
    pub base_arm_evidence_hash: u64,
    pub sensitivity_arm_evidence_hash: u64,
    pub numerical_discrepancy_hash: u64,
}

pub struct CrossArmSurfaceBindingV0 {
    pub arm_a: OrganizationArmV0,
    pub arm_b: OrganizationArmV0,
    pub arm_a_evidence_hash: u64,
    pub arm_b_evidence_hash: u64,
    pub cross_arm_surface_hash: u64,
}

pub struct OrganizationPairwiseComparisonV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub input_bundle_hash: u64,
    pub arm_a: OrganizationArmV0,
    pub arm_b: OrganizationArmV0,
    pub arm_evidence: Vec<ArmEvidenceBindingV0>,
    pub cross_arm_surface: CrossArmSurfaceBindingV0,
    pub numerical_discrepancies: Vec<NumericalDiscrepancyBindingV0>,
    pub aggregate_materiality: Vec<AggregateMaterialityV0>,
    pub object_materiality: Vec<ObjectMaterialityV0>,
    pub correspondence_reductions: Vec<CorrespondenceReductionV0>,
    pub g_authored: Vec<GAuthoredBindingV0>,
    pub derived_pairwise_comparison_hash: u64,
}

pub struct OrganizationComparisonV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub input_bundle_hash: u64,
    pub arm_evidence: Vec<ArmEvidenceBindingV0>,
    pub pairwise_comparison_hashes: Vec<u64>,
    pub evidence_vector: ComparisonEvidenceVectorV0,
    pub orientation_robustness: ClaimDispositionV0,
    pub human_review: ClaimDispositionV0,
    pub derived_comparison_hash: u64,
}
```

Arm-evidence bindings are ordered base arm H/C/G outer and spacing 8/4/2 inner,
then H/C 4 km numerical sensitivities. Every one must exist and validate for a
complete comparison. If H is unavailable the campaign is blocked before a
comparison root. A failed C/G base remains a separately retained arm failure,
but no complete three-arm comparison root is fabricated around it.

`numerical_discrepancies` contains exactly two bindings, H then C. Each binds
the corresponding 4 km base and sensitivity arm-evidence roots plus its
independently validated discrepancy child. A missing or invalid child blocks
comparison publication; it cannot be represented as a claim disposition.
`cross_arm_surface_discrepancies` immediately precedes them and contains the
exact three registered arm pairs.

Pairwise comparison roots are ordered `(H,C),(H,G),(C,G)` and publish the
actual claim-level reductions without requiring the third arm. `arm_evidence`
contains both arms' base 8/4/2 bindings plus each present H/C arm's 4 km
sensitivity binding: exactly 8 for H/C and 7 for H/G or C/G. Each root also
contains its one surface binding, one numerical binding for each H/C arm
present, all materiality records for that pair, and
only these ascending global O0b ordinals:

```text
H/C: 00..07, 12,13, 18..21
H/G: 00..03, 08..11, 14,15, 18,19
C/G: 04..11, 16,17, 20,21
```

| pair root | arm evidence | surfaces | numerical | correspondence reductions | G bindings |
|---|---:|---:|---:|---:|---:|
| H/C | 8 | 1 | 2 | 14 | 0 |
| H/G | 7 | 1 | 1 | 12 | 3 |
| C/G | 7 | 1 | 1 | 12 | 3 |

`g_authored` is empty for H/C and the exact three spacing rows when G is
present. The complete comparison binds the three pairwise root hashes, copies
aggregate/object rows in pair order, emits each global correspondence ordinal
once, and deduplicates H/C numerical and G-authored bindings only after requiring
all duplicate semantic values and hashes to be equal. Repeated within-arm
`CorrespondenceReductionV0` ordinals likewise require exact hash, annotation
vector and complete value equality; the first occurrence in pair order is
emitted and later equal occurrences are discarded. Its three surface rows come
one from each pairwise root. It performs no new claim reduction.

Correspondence reductions use the exact 22-artifact ordinal matrix above.
Annotations are sorted by accepted O0b record order and freshly joined from the
two projections. Aggregate records sort arm pair then quantity; object records
sort arm pair, family, anchor side, anchor ID, quantity. G-authored bindings
sort 8/4/2 and
the complete scale-robust claim is supported only when all three stored
dispositions are supported.

`orientation_robustness` is `DescriptiveOnly` because the accepted bundle has
one lattice orientation. `human_review` is `MissingQuantity` until the separate
planar-review amendment executes; the comparison root is numerical/semantic
evidence, not a verdict. These fields prevent a later report from silently
implying either claim already passed.

The vector deliberately retains conflicting evidence. It has no favorable
count, weight, total score, rank or Pareto flag. Architecture ownership remains
a later explicit decision over:

- independent S0 organization and morphology;
- independent D0 basin/reach/trunk organization;
- O0a surface/drainage relationships;
- linked-forcing and transfer-box correspondence;
- authored-G versus independent-D0 correspondence;
- resolution and numerical stability;
- human visual communication; and
- multidimensional cost/reusability evidence.

## Observational cost and reusable-state report

Cost is required but not deterministic semantic terrain state. The comparison
directory contains a validated observational JSON sidecar:

```rust
pub enum DominantWorkV0 {
    HPassTimesAdaptiveC0,
    CAdaptiveC0,
    GPriorityForestAndReconstruction,
    CommonS0D0O0a,
    O0bPolygonAndLineIndex,
}

pub enum OperationKindV0 {
    HPass,
    CandidateAttempt,
    AcceptedStep,
    GQueuePush,
    GQueuePop,
    GFinalizedCell,
    EvidenceInputCell,
    SurfaceDiscrepancyCell,
    ReductionRecordVisited,
    O0bPolygonClip,
    O0bLineCandidateTest,
}

pub struct OperationCountV0 {
    pub kind: OperationKindV0,
    pub count: u64,
}

pub enum ReusableStateV0 {
    FinalPhysicalSurface,
    ProcessWaterAndSolidLedgers,
    FinalIndependentDrainage,
    AuthoredReceiverForest,
    AccumulatedAreaAndRunoff,
    CommonLandformCore,
}

pub struct ArchitectureCostObservationV0 {
    pub arm: OrganizationArmV0,
    pub nominal_spacing_km: f64,
    pub result_hash: u64,
    pub source_revision: String,
    pub source_dirty: bool,
    pub generation_wall_milliseconds: u64,
    pub generation_peak_rss_bytes: u64,
    pub retained_result_bytes: u64,
    pub evidence_wall_milliseconds: u64,
    pub evidence_peak_rss_bytes: u64,
    pub retained_evidence_bytes: u64,
    pub production_modules: Vec<String>,
    pub production_modules_added: u64,
    pub production_lines_added: u64,
    pub production_lines_deleted: u64,
    pub frozen_morphology_graph_constant_count: u64,
    pub seeded_object_count: u64,
    pub dynamic_bytes_per_cell: f64,
    pub retained_native_graph_bytes: u64,
    pub dominant_work: Vec<DominantWorkV0>,
    pub operation_counts: Vec<OperationCountV0>,
    pub reusable_state: Vec<ReusableStateV0>,
}

pub struct ComparisonCostObservationV0 {
    pub source_revision: String,
    pub source_dirty: bool,
    pub pairwise_invocation_wall_milliseconds_total: u64,
    pub pairwise_peak_rss_bytes_max: u64,
    pub retained_pairwise_bytes: u64,
    pub comparison_wall_milliseconds: u64,
    pub comparison_peak_rss_bytes: u64,
    pub retained_comparison_bytes: u64,
    pub surface_numerical_operation_counts: Vec<OperationCountV0>,
    pub correspondence_operation_counts: Vec<OperationCountV0>,
    pub reduction_operation_counts: Vec<OperationCountV0>,
    pub dominant_work: Vec<DominantWorkV0>,
}

pub struct OrganizationCostReportV0 {
    pub schema_version: String,
    pub implementation_base_revision: String,
    pub observations: Vec<ArchitectureCostObservationV0>,
    pub comparison: ComparisonCostObservationV0,
}
```

Observations sort arm then 8/4/2. Completed arm-result and arm-evidence
wall/RSS values come from their registered external supervisors and validated
envelopes. Before the first active result, a
committed implementation registry freezes the production module paths owned by
each arm, excluding tests, generated output and shared JSON projection code.
`production_modules` is that lexical list. Source counts run
`git diff --numstat <implementation_base_revision>..HEAD -- <registered paths>`;
`production_modules_added` counts registered paths absent at the base. Constant
and seeded-object counts come from the same committed registry, not a text
grep. Dynamic bytes per cell is peak live capacity bytes of arm-owned
vectors divided by N; allocator overhead is excluded and disclosed. Native
graph bytes are encoded checkpoint/provenance component bytes. Reusable-state
values, dominant-work values and operation-count rows are enum-sorted with no
duplicates. Operation rows are exhaustive for the work families legal to their
arm or comparison stage, including zero, and count actual loop events at the
named boundary; no single proxy counter stands for another algorithm. The
comparison observation covers all three cross-arm surface artifacts, both
numerical discrepancy artifacts, all 22 O0b builds, all three pairwise claim
reductions and final assembly.
Every `source_dirty` field must be false and
their revision must equal the committed implementation registry revision;
`git diff base..HEAD` is never used to conceal an uncommitted worktree.

Exact legal vectors, in displayed order, are:

| record | `dominant_work` | `operation_counts` |
|---|---|---|
| H architecture | `HPassTimesAdaptiveC0, CommonS0D0O0a` | `HPass, CandidateAttempt, AcceptedStep, EvidenceInputCell` |
| C architecture | `CAdaptiveC0, CommonS0D0O0a` | `CandidateAttempt, AcceptedStep, EvidenceInputCell` |
| G architecture | `GPriorityForestAndReconstruction, CommonS0D0O0a` | `GQueuePush, GQueuePop, GFinalizedCell, EvidenceInputCell` |
| comparison | `O0bPolygonAndLineIndex` | not stored in a single vector |
| comparison surface/numerical discrepancy vector | inherited comparison dominant work | `SurfaceDiscrepancyCell` |
| comparison correspondence vector | inherited comparison dominant work | `O0bPolygonClip, O0bLineCandidateTest` |
| comparison reduction vector | inherited comparison dominant work | `ReductionRecordVisited` |

`EvidenceInputCell` increments exactly once per stored scored cell in the
arm-evidence adapter's initial cell-order population loop; neighborhood,
receiver, child and later summary scans do not increment it.
`SurfaceDiscrepancyCell` increments exactly once for each stored cell consumed
by each direct cross-arm surface row or matched H/C checkpoint pair; ledger and
diagnostic rows do not increment it. `ReductionRecordVisited` increments once
for each input aggregate, object, annotation or authored-G record consumed by
each fresh pairwise reduction in H/C,H/G,C/G order, then once for each pairwise
materiality/reduction/G-binding record consumed by final concatenation and
deduplication. The
comparison dominant-work vector is exactly the table row; it does not pretend
the inexpensive discrepancy/reduction passes are asymptotically dominant.
The comparison worker counts its mandatory fresh rebuild of every copied
pairwise child, so these vectors neither trust nor sum observational counters
from the earlier independent publications.

Byte observations sum `metadata.len()` with checked u64 addition in lexical
relative-path order. `retained_result_bytes` includes every semantic success
`.bin` in the arm-result directory and excludes JSON, envelope, lock and temp
files. `retained_evidence_bytes` likewise includes every semantic `.bin` in its
arm-evidence directory. `retained_comparison_bytes` includes `comparison.bin`,
all three cross-arm surface `.bin` children, both numerical-discrepancy `.bin`
children, all 22 correspondence `.bin` children and all three pairwise-
comparison `.bin` children; it excludes JSON,
`cost-observation.json`, envelope, lock and temp files. Thus cost publication
is nonrecursive.

The pairwise cost population is exactly the 30 supplied immutable publication
directories: surfaces H/C,H/G,C/G; numerical H,C; correspondence 00..21;
pairwise comparison H/C,H/G,C/G. Each invocation's externally measured wall/RSS
already includes its mandatory internal fresh deterministic replay before
publication; replay is not a second published attempt. Wall total checked-adds
the 30 validated envelopes in that family/order and peak RSS is their maximum.
Retained pairwise bytes checked-add lexical paths within the same 30 directories
and include semantic binary and deterministic JSON files only; they exclude
every run envelope, lock and temp. The full comparison receives these exact
ordered primary directories, so no undeclared campaign registry is required.
These fields distinguish total preregistered pairwise campaign cost from the
minimal fresh rebuild cost measured by the final comparison worker.

Comparison publication uses a registered supervisor/worker boundary. The
measured worker exits after producing, rereading and fully validating every
semantic binary and deterministic JSON projection in the owned temporary
directory. The external supervisor then knows the worker's final wall time and
peak RSS, computes the frozen retained-byte set, writes and validates
`cost-observation.json`, writes the self-excluding run envelope, syncs and
atomically renames. Comparison wall/RSS therefore covers the complete semantic
worker but deliberately excludes observational sidecar/envelope serialization
and final rename. The supervisor is final authority for that declared boundary;
the cost file never depends on its own bytes or an unwritten envelope.

This report is regenerated and schema-validated but excluded from semantic
comparison hashing because timing, host and dirty state are observational. Its
file length/FNV is retained by the run envelope. The later decision must cite
it directly; exclusion from the semantic root is not permission to ignore cost
or select only a favorable resolution.

## Typed evidence failures

```rust
pub enum EvidenceInvocationV0 {
    ArmEvidence(OrganizationArtifactIdentityV0),
    CrossArmSurface { arm_a: OrganizationArmV0, arm_b: OrganizationArmV0 },
    NumericalDiscrepancy { arm: OrganizationArmV0 },
    Correspondence { ordinal: u32, source_core_hash: u64, target_core_hash: u64 },
    PairwiseComparison { arm_a: OrganizationArmV0, arm_b: OrganizationArmV0 },
    Comparison { input_bundle_hash: u64 },
}

pub enum EvidenceFailurePhaseV0 {
    PredecessorValidation,
    GraphAdaptation,
    SurfaceHierarchy,
    Drainage,
    ReferenceRelationships,
    CentralProjection,
    GAuthoredComparison,
    CrossArmSurface,
    NumericalDiscrepancy,
    Correspondence,
    MaterialityReduction,
    Serialization,
    ResourceCeiling,
}

pub enum EvidenceFailureCauseV0 {
    InvalidPredecessor,
    InstrumentError,
    NonFiniteValue,
    NonCanonicalValue,
    LengthOrOrderMismatch,
    LedgerClosureFailure,
    HashMismatch,
    InvariantFailure,
    WallTimeCeiling,
    MemoryCeiling,
    ArtifactSizeCeiling,
}

pub enum EvidenceFailureAuthorityV0 { ReplayableInstrument, ObservationalResource }

pub enum EvidenceFailureWitnessV0 {
    None,
    Index { family: EvidenceIndexFamilyV0, index: u64 },
    Hash { stored: u64, recomputed: u64 },
    Residual { actual: f64, expected: f64, residual: f64 },
    Resource { kind: OrganizationResourceKindV0, observed: u64, ceiling: u64 },
}

pub enum EvidenceIndexFamilyV0 {
    Cell, Highland, Saddle, Reach, Portal, Relationship, Checkpoint,
    NumericalLedger, Limiter, Correspondence, Quantity,
}

pub struct OrganizationEvidenceFailureV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub invocation: EvidenceInvocationV0,
    pub predecessor_hashes: Vec<u64>,
    pub authority: EvidenceFailureAuthorityV0,
    pub phase: EvidenceFailurePhaseV0,
    pub cause: EvidenceFailureCauseV0,
    pub witness: EvidenceFailureWitnessV0,
    pub derived_failure_hash: u64,
}
```

Predecessor hashes have the exact invocation-specific arity and registered
order. Repeated numeric hash values are not rejected merely because distinct
predecessors could theoretically collide.
Missing paths, raw-cap breaches, truncation, decode/framing failure, wrong
top-level schema or any other condition that prevents capturing every required
predecessor semantic root is a pre-invocation CLI diagnostic: it publishes no
semantic failure directory. `PredecessorValidation` begins only after every
root has decoded, passed standalone validation and been copied into the exact
hash vector. It then owns binding, order, cross-predecessor and replay failures,
for which a complete semantic failure is constructible.
For arm evidence they are input bundle, input resolution, arm result and native
provenance hashes. For one cross-arm surface they are arm-A result/evidence
then arm-B result/evidence hashes. For one numerical discrepancy they are base arm-result,
sensitivity arm-result, base trace and sensitivity trace hashes. For one
correspondence they are source arm-evidence, source
core, target arm-evidence and target core hashes. For one pairwise comparison
they are input bundle, its two arm-evidence groups in arm/spacing/purpose order,
one surface hash, applicable H/C numerical hashes, its displayed ascending O0b
hashes and any three G/D0 hashes. For comparison they are input
bundle, eleven arm-evidence hashes, the three cross-arm surface hashes, the H
then C numerical-discrepancy hashes,
22 correspondence hashes, three G/D0 hashes and the three pairwise comparison
hashes in their registered orders.
The separately supplied opportunity-control/G-4 directories are validation
witnesses already semantically bound by each arm result's predecessor record;
their hashes are deliberately not repeated in the evidence-failure vector.
Likewise, comparison's arm-run and aligned predecessor directories validate the
eleven arm-evidence children transitively and do not add another hash tier.

Legal phase/cause/witness families are:

| phase | cause | witness |
|---|---|---|
| `PredecessorValidation` | invalid predecessor, length/order | `None` or `Index` |
| `PredecessorValidation` | hash mismatch | `Hash` |
| graph/S0/D0/O0a/central/G/surface/numerical/O0b/materiality | instrument error | `None` |
| graph/S0/D0/O0a/central/G/surface/numerical/O0b/materiality | invariant failure | `None` or `Index` |
| graph/S0/D0/O0a/central/G/surface/numerical/O0b/materiality | nonfinite/noncanonical, length/order | `None` or `Index` |
| D0/central/G/numerical/materiality | ledger closure | `Residual` |
| any non-resource phase | hash mismatch | `Hash` |
| `Serialization` | noncanonical, length/order, hash mismatch, invariant | matching `None`, `Index` or `Hash` |
| `ResourceCeiling` | matching wall, memory or artifact-size ceiling | matching `Resource` |

Here the abbreviated algorithm phases expand to their exact enum variants in
declaration order. `InstrumentError` is legal only with `None`. `Residual` is
legal only for `LedgerClosureFailure`; `Hash` only for `HashMismatch`; and
`Index` only for length/order, nonfinite/noncanonical or invariant failure.
Every unlisted combination rejects.

For numerical discrepancy failures, `Checkpoint` indexes the registered
checkpoint-pair row, `NumericalLedger` the complete arm-owned ledger row, and
`Limiter` the limiter-enum row. Other index families are illegal for that
invocation.

Failure choice is total even for a multiply invalid input. Check raw byte cap
first; then bounded decode/framing failures as encountered by wire order
(invalid enum/option discriminants and announced string/vector/sequence lengths);
then trailing bytes after a successful decode; then schema/hash-version;
post-decode structural order/length checks; scalar finite/canonical/domain
checks; local algebra; stored root hash; predecessor-list arity/order; supplied
child validation and rebuild in the campaign's exact stage order; finally
publication. Decode/framing failures map to `Serialization` plus
`LengthOrOrderMismatch` and the earliest available `None` or `Index` witness.
Within a structural vector use stored element order; within a record use field
declaration order; within an enum-owned family use enum order. The first failing
check in that total order owns the failure. Existing typed
G0/S0, D0, O0a and O0b errors map to their phase plus `InstrumentError/None`;
their unstable `Display` text lives only in the envelope. Shape/finite/hash/
ledger checks performed by the wrapper use their dedicated cause/witness.

Every non-resource combination is `ReplayableInstrument` and must rebuild to
the same phase/cause/witness. `ResourceCeiling` with matching cause and resource
witness is the only `ObservationalResource` family and replay returns
`NotReplayable`. External kill/OOM before atomic publication creates no semantic
failure. A valid empty object cohort never maps to failure. Evidence failure
does not alter or replace its arm result.

## Validation boundaries

Standalone decoding validates raw caps/trailing decode, schema/hash version,
canonical scalar domains, enum/option legality, vector lengths/orders, local
count/shape algebra and root hash. It cannot recompute distribution reductions,
surface discrepancies or other summaries whose raw observations live only in
predecessors, and it does not claim that supplied child hashes exist.

Full predecessor validation additionally:

- fully validates the accepted input and arm artifacts;
- requires every result/native/final-elevation binding;
- rebuilds the common graph, S0, D0, common core and reference O0a;
- recomputes the stored central mask and every projection field;
- for G, validates the bound forest checkpoint before rebuilding the authored
  comparison;
- recomputes every available direct 4 km cross-arm surface artifact;
- recomputes both numerical-discrepancy children from their bound
  traces/checkpoints/ledgers;
- rebuilds every directional O0b from its two supplied cores;
- recomputes cohort annotations, compatibility, all materiality values and
  dispositions;
- rebuilds each available pairwise comparison root and the exact complete-root
  concatenation/deduplication rules; and
- requires exact child value, byte and hash equality.

Deterministic replay repeats the complete extraction/comparison in a fresh
same-process invocation and requires semantic value, binary byte, JSON byte and
hash equality. Closeness is never substituted for repeat. Observational cost
fields compare only their registered structural subset, not timing/RSS.

Full validation proves deterministic derivation from supplied accepted
predecessors. It does not prove those arm results were produced causally; arm
replay remains the artifact amendment's authority.

## JSON and atomic publication

Binary is semantic authority. `from_semantic` is the sole JSON constructor.
Hashes are 16-digit lowercase hex, semantic floats remain round-trip JSON
numbers, and pretty JSON ends in exactly one LF. `validate_against` compares a
parsed file to a fresh projection. JSON struct declaration order below is JSON
field order; no projection uses a map. Nested semantic types without hash or
cell-index-vector fields retain their binary declaration order exactly.
Serde's default externally tagged representation is mandatory: unit variants
are their exact Rust variant-name strings and data-bearing variants are
single-key objects whose value fields follow declaration order. Field and
variant names use exact Rust spelling; `rename`, `rename_all`, `tag`,
`content`, `flatten`, `skip` and `skip_serializing_if` attributes are forbidden.
Every declared `Option` field is present and `None` is JSON `null`.

```rust
pub struct OrganizationArtifactIdentityJsonV0 {
    pub input_bundle_hash: String,
    pub input_resolution_hash: String,
    pub nominal_spacing_km: f64,
    pub arm: OrganizationArmV0,
    pub purpose: OrganizationRunPurposeV0,
}

pub struct OrganizationArmEvidenceJsonV0 {
    pub json_schema_version: String,
    pub semantic_schema_version: String,
    pub semantic_hash_version: String,
    pub identity: OrganizationArtifactIdentityJsonV0,
    pub input_bundle_hash: String,
    pub input_resolution_hash: String,
    pub arm_result_hash: String,
    pub native_provenance_hash: String,
    pub common_core_hash: String,
    pub reference_o0a_hash: String,
    pub central_projection_hash: String,
    pub g_authored_d0_hash: Option<String>,
    pub derived_arm_evidence_hash: String,
}

pub struct CentralHighlandProjectionJsonV0 {
    pub peak_id: u32,
    pub cohort: CentralHighlandCohortV0,
    pub peak_anchor_cell: u32,
    pub central_member_count: u64,
    pub central_footprint_area_km2: f64,
    pub central_local_runoff_km3_myr: f64,
    pub central_footprint_fraction: f64,
    pub central_mask_crossing: bool,
    pub physical_domain_contact: bool,
    pub forcing_probes: Vec<HighlandForcingProbeV0>,
}

pub struct CentralSaddleProjectionJsonV0 {
    pub saddle_id: u32,
    pub central_backing_cell_count: u64,
    pub elder_peak_id: u32,
    pub losing_peak_ids: Vec<u32>,
    pub equal_elder_ambiguous: bool,
}

pub struct TransferSaddleBackingJsonV0 {
    pub saddle_id: u32,
    pub probe_backing_cell_count: u64,
    pub probe_mask_crossing: bool,
}

pub struct TransferProbeJsonV0 {
    pub bounds_km: [f64; 4],
    pub cell_summary: ProbeCellSummaryV0,
    pub saddle_ids: Vec<u32>,
    pub saddle_backing: Vec<TransferSaddleBackingJsonV0>,
    pub drainage: Vec<ProbeDrainageContributionV0>,
    pub forcing_integrals: Vec<ProbeForcingIntegralV0>,
}

pub struct OrganizationCentralProjectionJsonV0 {
    pub json_schema_version: String,
    pub semantic_schema_version: String,
    pub semantic_hash_version: String,
    pub identity: OrganizationArtifactIdentityJsonV0,
    pub input_bundle_hash: String,
    pub input_resolution_hash: String,
    pub arm_result_hash: String,
    pub common_core_hash: String,
    pub reference_o0a_hash: String,
    pub central_mask_hash: String,
    pub cell_summary: CentralCellSummaryV0,
    pub object_summary: CentralObjectSummaryV0,
    pub primary_highlands: Vec<CentralHighlandProjectionJsonV0>,
    pub context_highlands: Vec<CentralHighlandProjectionJsonV0>,
    pub saddles: Vec<CentralSaddleProjectionJsonV0>,
    pub drainage_scales: Vec<CentralDrainageScaleV0>,
    pub relationship_summary: CentralRelationshipSummaryV0,
    pub forcing_summary: LinkedForcingProbeSummaryV0,
    pub transfer_probe: TransferProbeJsonV0,
    pub derived_projection_hash: String,
}

pub struct OrganizationGAuthoredD0JsonV0 {
    pub json_schema_version: String,
    pub semantic_schema_version: String,
    pub semantic_hash_version: String,
    pub identity: OrganizationArtifactIdentityJsonV0,
    pub arm_result_hash: String,
    pub native_provenance_hash: String,
    pub forest_checkpoint_hash: String,
    pub common_core_hash: String,
    pub d0_evidence_hash: String,
    pub receiver_agreement_counts: [u64; 4],
    pub receiver_agreement_area_km2: [f64; 4],
    pub receiver_agreement_runoff_km3_myr: [f64; 4],
    pub terminal_portal_equal_count: u64,
    pub terminal_portal_equal_area_km2: f64,
    pub terminal_portal_equal_runoff_km3_myr: f64,
    pub accumulated_area_difference_km2: DifferenceDistributionV0,
    pub accumulated_runoff_difference_km3_myr: DifferenceDistributionV0,
    pub support: Vec<SupportContingencyV0>,
    pub portals: Vec<GPortalD0ComparisonV0>,
    pub d0_fill_supported_cell_count: u64,
    pub d0_flat_supported_cell_count: u64,
    pub d0_physically_non_descending_cell_count: u64,
    pub claim_disposition: GAuthoredClaimDispositionV0,
    pub derived_g_authored_d0_hash: String,
}

pub struct OrganizationNumericalDiscrepancyJsonV0 {
    pub json_schema_version: String,
    pub semantic_schema_version: String,
    pub semantic_hash_version: String,
    pub arm: OrganizationArmV0,
    pub base_result_hash: String,
    pub sensitivity_result_hash: String,
    pub base_trace_hash: String,
    pub sensitivity_trace_hash: String,
    pub checkpoint_discrepancies: Vec<SurfaceDiscrepancyV0>,
    pub ledger_differences: Vec<NumericalLedgerDifferenceV0>,
    pub base_diagnostics: NumericalRunStepDiagnosticsV0,
    pub sensitivity_diagnostics: NumericalRunStepDiagnosticsV0,
    pub derived_numerical_discrepancy_hash: String,
}

pub struct OrganizationCrossArmSurfaceDiscrepancyJsonV0 {
    pub json_schema_version: String,
    pub semantic_schema_version: String,
    pub semantic_hash_version: String,
    pub arm_a: OrganizationArmV0,
    pub arm_b: OrganizationArmV0,
    pub arm_a_result_hash: String,
    pub arm_b_result_hash: String,
    pub arm_a_evidence_hash: String,
    pub arm_b_evidence_hash: String,
    pub signed_moment_km3: f64,
    pub signed_mean_km: f64,
    pub l1_mean_km: f64,
    pub rms_km: f64,
    pub maximum_abs_km: f64,
    pub disposition: ClaimDispositionV0,
    pub derived_cross_arm_surface_hash: String,
}
```

Comparison JSON mirrors every disposition, compatibility reason and reduction
annotation, but does not duplicate the 22 complete O0b child tables:

```rust
pub struct ArmEvidenceBindingJsonV0 {
    pub identity: OrganizationArtifactIdentityJsonV0,
    pub arm_result_hash: String,
    pub arm_evidence_hash: String,
    pub common_core_hash: String,
    pub reference_o0a_hash: String,
    pub central_projection_hash: String,
}

pub struct NumericalDiscrepancyBindingJsonV0 {
    pub arm: OrganizationArmV0,
    pub base_arm_evidence_hash: String,
    pub sensitivity_arm_evidence_hash: String,
    pub numerical_discrepancy_hash: String,
}

pub struct CrossArmSurfaceBindingJsonV0 {
    pub arm_a: OrganizationArmV0,
    pub arm_b: OrganizationArmV0,
    pub arm_a_evidence_hash: String,
    pub arm_b_evidence_hash: String,
    pub cross_arm_surface_hash: String,
}

pub struct CorrespondenceReductionJsonV0 {
    pub ordinal: u32,
    pub source_arm_evidence_hash: String,
    pub target_arm_evidence_hash: String,
    pub correspondence_hash: String,
    pub assignment_annotations: Vec<CentralAssignmentAnnotationV0>,
    pub component_annotations: Vec<CentralComponentAnnotationV0>,
}

pub struct GAuthoredBindingJsonV0 {
    pub nominal_spacing_km: f64,
    pub arm_evidence_hash: String,
    pub g_authored_d0_hash: String,
    pub disposition: ClaimDispositionV0,
}

pub struct ComparisonEvidenceVectorJsonV0 {
    pub cross_arm_surface_discrepancies: Vec<CrossArmSurfaceBindingJsonV0>,
    pub numerical_discrepancies: Vec<NumericalDiscrepancyBindingJsonV0>,
    pub aggregate_materiality: Vec<AggregateMaterialityV0>,
    pub object_materiality: Vec<ObjectMaterialityV0>,
    pub correspondence_reductions: Vec<CorrespondenceReductionJsonV0>,
    pub g_authored: Vec<GAuthoredBindingJsonV0>,
}

pub struct OrganizationPairwiseComparisonJsonV0 {
    pub json_schema_version: String,
    pub semantic_schema_version: String,
    pub semantic_hash_version: String,
    pub input_bundle_hash: String,
    pub arm_a: OrganizationArmV0,
    pub arm_b: OrganizationArmV0,
    pub arm_evidence: Vec<ArmEvidenceBindingJsonV0>,
    pub cross_arm_surface: CrossArmSurfaceBindingJsonV0,
    pub numerical_discrepancies: Vec<NumericalDiscrepancyBindingJsonV0>,
    pub aggregate_materiality: Vec<AggregateMaterialityV0>,
    pub object_materiality: Vec<ObjectMaterialityV0>,
    pub correspondence_reductions: Vec<CorrespondenceReductionJsonV0>,
    pub g_authored: Vec<GAuthoredBindingJsonV0>,
    pub derived_pairwise_comparison_hash: String,
}

pub struct OrganizationComparisonJsonV0 {
    pub json_schema_version: String,
    pub semantic_schema_version: String,
    pub semantic_hash_version: String,
    pub input_bundle_hash: String,
    pub arm_evidence: Vec<ArmEvidenceBindingJsonV0>,
    pub pairwise_comparison_hashes: Vec<String>,
    pub evidence_vector: ComparisonEvidenceVectorJsonV0,
    pub orientation_robustness: ClaimDispositionV0,
    pub human_review: ClaimDispositionV0,
    pub derived_comparison_hash: String,
}
```

Failure JSON also has an exact recursive mirror:

```rust
pub enum EvidenceInvocationJsonV0 {
    ArmEvidence(OrganizationArtifactIdentityJsonV0),
    CrossArmSurface { arm_a: OrganizationArmV0, arm_b: OrganizationArmV0 },
    NumericalDiscrepancy { arm: OrganizationArmV0 },
    Correspondence { ordinal: u32, source_core_hash: String, target_core_hash: String },
    PairwiseComparison { arm_a: OrganizationArmV0, arm_b: OrganizationArmV0 },
    Comparison { input_bundle_hash: String },
}

pub enum EvidenceFailureWitnessJsonV0 {
    None,
    Index { family: EvidenceIndexFamilyV0, index: u64 },
    Hash { stored: String, recomputed: String },
    Residual { actual: f64, expected: f64, residual: f64 },
    Resource { kind: OrganizationResourceKindV0, observed: u64, ceiling: u64 },
}

pub struct OrganizationEvidenceFailureJsonV0 {
    pub json_schema_version: String,
    pub semantic_schema_version: String,
    pub semantic_hash_version: String,
    pub invocation: EvidenceInvocationJsonV0,
    pub predecessor_hashes: Vec<String>,
    pub authority: EvidenceFailureAuthorityV0,
    pub phase: EvidenceFailurePhaseV0,
    pub cause: EvidenceFailureCauseV0,
    pub witness: EvidenceFailureWitnessJsonV0,
    pub derived_failure_hash: String,
}
```

The observational cost JSON is exact as well:

```rust
pub struct ArchitectureCostObservationJsonV0 {
    pub arm: OrganizationArmV0,
    pub nominal_spacing_km: f64,
    pub result_hash: String,
    pub source_revision: String,
    pub source_dirty: bool,
    pub generation_wall_milliseconds: u64,
    pub generation_peak_rss_bytes: u64,
    pub retained_result_bytes: u64,
    pub evidence_wall_milliseconds: u64,
    pub evidence_peak_rss_bytes: u64,
    pub retained_evidence_bytes: u64,
    pub production_modules: Vec<String>,
    pub production_modules_added: u64,
    pub production_lines_added: u64,
    pub production_lines_deleted: u64,
    pub frozen_morphology_graph_constant_count: u64,
    pub seeded_object_count: u64,
    pub dynamic_bytes_per_cell: f64,
    pub retained_native_graph_bytes: u64,
    pub dominant_work: Vec<DominantWorkV0>,
    pub operation_counts: Vec<OperationCountV0>,
    pub reusable_state: Vec<ReusableStateV0>,
}

pub struct OrganizationCostReportJsonV0 {
    pub json_schema_version: String,
    pub semantic_schema_version: String,
    pub implementation_base_revision: String,
    pub observations: Vec<ArchitectureCostObservationJsonV0>,
    pub comparison: ComparisonCostObservationV0,
}
```

Exact `json_schema_version` values are respectively
`orogen-owner-arm-evidence-json-v0`,
`orogen-owner-central-projection-json-v0`,
`orogen-owner-g-authored-d0-json-v0`,
`orogen-owner-numerical-discrepancy-json-v0`,
`orogen-owner-cross-arm-surface-json-v0`,
`orogen-owner-pairwise-comparison-json-v0`,
`orogen-organization-comparison-json-v0`,
`orogen-owner-evidence-failure-json-v0`, and
`orogen-owner-cost-observation-json-v0`. Central JSON is the only lossy view:
it replaces highland member, saddle backing and transfer-saddle backing cell
vectors with exact counts. Every other semantic vector is complete. Rebuilding
the projection from binary must reproduce those counts and all JSON bytes.

Canonical directories are unique attempts:

```text
artifacts/orogen-owner-v0/evidence/<base|numerical-sensitivity>/<h|c|g>/<spacing>-km/<attempt-id>/
  arm-evidence.bin
  arm-evidence.json
  common-core.bin
  reference-o0a.bin
  central-projection.bin
  central-projection.json
  [g-authored-d0.bin]
  [g-authored-d0.json]
  run-envelope.json

artifacts/orogen-owner-v0/pairwise/numerical-discrepancy/<h|c>/<attempt-id>/
  numerical-discrepancy.bin
  numerical-discrepancy.json
  run-envelope.json

artifacts/orogen-owner-v0/pairwise/cross-arm-surface/<h-c|h-g|c-g>/<attempt-id>/
  cross-arm-surface.bin
  cross-arm-surface.json
  run-envelope.json

artifacts/orogen-owner-v0/pairwise/correspondence/<000..021>/<attempt-id>/
  correspondence.bin
  run-envelope.json

artifacts/orogen-owner-v0/pairwise/comparison/<h-c|h-g|c-g>/<attempt-id>/
  pairwise-comparison.bin
  pairwise-comparison.json
  run-envelope.json

artifacts/orogen-owner-v0/comparison/<attempt-id>/
  comparison.bin
  comparison.json
  cost-observation.json
  cross-arm-surface/
    000-h-c.bin
    000-h-c.json
    001-h-g.bin
    001-h-g.json
    002-c-g.bin
    002-c-g.json
  numerical-discrepancy/
    000-h.bin
    000-h.json
    001-c.bin
    001-c.json
  correspondence/
    000.bin
    ...
    021.bin
  pairwise-comparison/
    000-h-c.bin
    000-h-c.json
    001-h-g.bin
    001-h-g.json
    002-c-g.bin
    002-c-g.json
  run-envelope.json
```

G-authored files are present exactly for G base evidence. H/C sensitivity
evidence exists only at 4 km. A cooperative failure directory contains exactly
`evidence-failure.bin`, `evidence-failure.json` and `run-envelope.json` instead
of semantic successes. No partial common core or correspondence is published.

Publication reuses the artifact amendment's absent target, owned sibling lock,
`create_new`, encode/decode, reread/full-validation, `sync_all`, lexical
artifact manifest, atomic rename and explicit stale-lock rules. Attempt IDs and
envelopes are nonsemantic. Targets never overwrite. Individual caps are checked
in lexical relative-path order, then the 2 GiB complete-directory cap. Evidence
directories have 15-minute/2-GiB wall/RSS ceilings. Every pairwise attempt and
the complete comparison use the parent's 60-minute/2-GiB wall/RSS ceiling.
External measurement remains final authority.

Pairwise targets publish independently as soon as their own predecessors exist.
A later complete comparison copies their validated canonical bytes into the
displayed comparison child paths, rereads them there and requires exact
value/byte/hash identity. This deliberate self-contained duplication means a C
or G failure cannot erase valid H/C, H/G or within-arm evidence, while the final
three-arm directory remains one atomic immutable object.

The thin CLI is `orogen_owner_evidence`. It accepts only the registered input
bundle, invocation kind
`arm-evidence|cross-arm-surface|numerical-discrepancy|correspondence|comparison`,
and a required new output
directory. Arm evidence additionally receives one validated arm-run directory
and the run's predecessor directory exactly when required by the artifact
amendment's legal matrix: the same-resolution opportunity-control run for H/C
base or sensitivity, frozen G-4 base for G-8/2, and no predecessor for G-4.
It infers arm/purpose/spacing and rejects a missing, extra or mismatched slot.

Numerical-discrepancy receives the same-arm 4 km base and sensitivity evidence
directories plus their arm-run and required control directories.
Cross-arm-surface receives one registered ordered pair of 4 km base evidence
and arm-run/predecessor directories.
Correspondence receives its registered ordinal, source/target evidence
directories and each side's arm-run/optional run-predecessor directory. Each
invocation fully rebuilds its supplied evidence before publishing; neither
accepts a precomputed summary or runtime threshold.

Pairwise comparison is invoked as `comparison` with exactly one registered arm
pair and receives that pair's evidence/run/predecessor directories plus its
surface, applicable numerical and displayed O0b pairwise directories. The
three-arm comparison uses the same invocation with all three arms and additionally
receives the exact three pairwise-comparison directories. Pair arity is inferred
from the supplied registered arm set; no separate reduction mode or threshold
flag exists.

Three-arm comparison receives the exact ordered eleven evidence directories, their
eleven bound arm-run directories, and an aligned eleven-slot optional
run-predecessor vector. The predecessor slots contain H/C controls for all six
base runs and both 4 km sensitivity runs, frozen G-4 for G-8 and G-2, and
`None` only for G-4. Repeated paths are legal and expected. This permits full
predecessor validation rather than trust in child hashes. Comparison also
receives the exact ordered three cross-arm-surface, two discrepancy, 22
correspondence and three pairwise-comparison directories, copies their canonical
children, and fully rebuilds them from the bound arm evidence/runs before final
assembly.
It exposes no thresholds, masks, quantities, cohorts, O0a settings,
discrepancy settings or correspondence flags.

## Manufactured and regression gate

Before linked evidence execution:

1. **Adapter/core:** exact stored-mesh control-volume adaptation; whole mask;
   runoff/portal population; S0/D0/reference-O0a rebuild; split core bindings;
   exact repeat at manufactured 8/4/2.
2. **Common fields:** affine grade, radial relief and boundary truncation;
   whole-context neighborhoods for central cells adjacent to the mask; exact
   agreement with embedded S0 highland summaries.
3. **Central highlands:** primary/context/outside partition, flat maximum
   crossing the mask, central area/runoff, mask crossing versus physical-domain
   contact, empty cohort and ordering.
4. **Saddles/O0a:** a multi-losing-peak saddle with one central backing cell;
   complete incident identities; full reference face namespace with no index
   renumbering; highland/saddle/reach selection and optional-value reductions.
5. **Drainage:** reach/portal exclusive partitions close central and transfer
   area/runoff at all thresholds; nested contribution and three trunk roles;
   empty reach but nonempty portal context.
6. **Forcing/transfer:** exact normalized stencil integrals, endpoint-clamped
   segment distance, axial ambiguity/null, inclusive box edges and no selected
   saddle/reach narrative.
7. **G/D0:** exact/mismatched receiver class and target, terminal portal,
   accumulation difference, fill/flat disclosure, support contingency and
   pass/fail 10% boundary fixtures; portal closure and repeat.
8. **O0b annotation:** one-to-one, one-to-many without tie, exact tie,
   many-to-one, many-to-many, no overlap, both ineligible statuses, metric
   conflict and best partner outside central; ordered reversal preserves nulls/
   ties and swaps directional component meaning.
9. **Materiality:** exact threshold equality is not material; a constructed
    difference bit-equal to `next_up(materiality_threshold)` is material;
    resolution, sensitivity, missing, context, tie, coverage and
    cardinality blockers; exact leg arity/order, reciprocal joint intersection,
    count/measure fractions immediately below/equal/above 0.5, and joint-only
    4/8/2/sensitivity reductions; complete-cohort stability with total matched-object
    turnover or quantity-availability mismatch still rejects; empty cohorts
    and zero denominators store `None` fractions; `compatible_population`
    Some/None and every disposition option row are exact; 1,000/4,000
    km2 drainage remains descriptive; integer counts below `2^53`; zero relative
    denominator.
10. **Surface/numerical discrepancy:** identical and spatially permuted
    same-distribution 4 km cross-arm surfaces prove direct signed/L1/RMS/max
    reductions and Neumaier order; exact H/C checkpoint pairing and stored-order
    surface reductions; complete arm-owned ledger rows including portal order;
    zero/nonnull symmetric-relative cases; complete limiter diagnostics,
    accepted-step/attempt counts and empty-step dt nulls; exact repeat.
11. **Pairwise assembly:** exact H/C=8 and H/G=C/G=7 evidence bindings,
    registered ordinal subsets and numerical/G-authored presence; each valid
    two-arm root publishes with the third arm absent; wrong pair/ordinal rejects;
    final pair-order concatenation emits each global ordinal once and requires
    duplicate correspondence annotations/hashes plus numerical/G values/hashes
    equal before first-in-pair-order deduplication; copied
    pairwise roots are byte-identical.
12. **Mutation/caps:** repair outer hashes after mutating each predecessor,
    mask/cohort/member, reduction, G forest, O0b annotation and disposition;
    mutate every cross-arm and numerical-discrepancy reduction/ledger/diagnostic
    and child hash; every mutation rejects. Oversized files, including both
    discrepancy caps, and forged lengths reject before allocation.
13. **Publication:** exact independent pairwise and complete success/failure
    filename sets, survival with one arm absent, byte-identical final copies,
    supervisor worker failure, exact 30-directory replay-inclusive cost population/retained-byte
    set and cost-before-envelope handoff; JSON regeneration,
    absent-target sentinel, no temp residue after ordinary error and exact
    semantic/binary/JSON repeat for every arm evidence, discrepancy,
    correspondence, pairwise-comparison and final-comparison artifact.

Existing accepted G0/S0/D0/O0a/O0b fixture matrices remain required. These new
gates compose them; they do not amend their answers.

## Campaign advancement

After all base/sensitivity arm results are frozen and exactly repeated:

1. build 8/4/2 H base evidence and exact-repeat each; H failure blocks;
2. build 8/4/2 C and G base evidence and exact-repeat each, retaining any typed
   instrument failure;
3. build and exact-repeat H/C 4 km sensitivity evidence;
4. as soon as their own predecessors exist, independently publish and
   exact-repeat each available cross-arm surface row, H/C numerical-discrepancy
   artifact and O0b ordinal; a missing third arm does not suppress a valid pair;
5. publish and exact-repeat each pairwise comparison root as soon as its own
   surface, numerical and displayed O0b subset exists;
6. require all eleven evidence, three cross-arm surface, two numerical, 22 O0b
   and three pairwise-comparison roots only before constructing the complete
   three-arm comparison;
7. build and exact-repeat the complete comparison root without inspecting
   presentation;
8. publish the complete observational cost report; and
9. hand frozen evidence to the planar capture/human-review amendment.

An unappealing, object-poor or materially conflicting valid result remains in
the comparison. A failed extractor is repaired as an instrument under a new
contract amendment; it is not terrain tuning. No response case, rotated mesh,
global seed or product pipeline change is authorized by this rung.

## Implementation boundary

Only after the planar-review amendment is committed may implementation add:

```text
build_common_planar_fields_v0
build_organization_arm_evidence_v0
build_organization_central_projection_v0
build_g_authored_d0_comparison_v0
annotate_organization_correspondence_v0
reduce_organization_materiality_v0
validate_organization_evidence_*_v0
```

Reuse accepted common-core and core-backed-O0b builders. Do not fork S0/D0/O0a
formulas, expose private native terrain state to common extraction or refactor
the accepted arm publisher in this rung.

## Stop boundary

This amendment completes item 3 of the parent executable stop boundary. Do not
implement H/C/G or evidence yet. Next freeze planar capture, blinding, sheet
identity, human prompts, reveal order and review records. Only then may shared
composition/artifact/evidence infrastructure be implemented and the active
campaign begin.
