# Common planar evidence-core V0 executable contract

**Date:** 2026-07-15

**Status:** implemented and accepted as a bounded planar artifact boundary; see
the [dated audit](../audits/landform-common-core-2026-07-15.md)

**Parents:** [landform object packet v0](landform-object-packet-v0-2026-07-14.md),
[bounded O0b contract](landform-object-packet-o0b-2026-07-15.md),
[product-boundary decision](landform-product-boundary-decision-2026-07-15.md)

## Decision and correction

Implement the already-decided common planar artifact split before starting the
organization-owner comparison. Prove it by exact decomposition, reconstruction
and mechanical-equivalence tests over the accepted manufactured V0 packet
population.

Do **not** construct a pre-arm linked-deformation landform packet in this
checkpoint. The existing linked specification and implementations supply
candidate domain geometry, initial surfaces, time-dependent deformation
forcing, runoff and boundary conditions, not one newly accepted executable
bundle. More importantly, they deliberately do not supply an arm-neutral final
terrain. S0, D0 and O0a derive evidence from a final terrain, while O0b compares
two derived populations. Manufacturing a final terrain from the forcing would
insert an unregistered organization owner and make that owner a privileged
baseline.

The linked case therefore has two later and distinct artifacts:

1. a shared linked-input manifest, which binds the exact mesh and phase,
   declarative scenario and compiler identity, compiled/evaluated deformation
   field, episode and integrated-work ledgers, portals, initial state, runoff,
   homogeneous material, candidate evaluation geometry, cell counts and
   resource context, but no selected scoring population,
   final-surface objects, arm conversion or terrain-quality verdict; and
2. arm-result evidence cores produced only after H, C and G have generated
   authoritative final surfaces under the organization-owner preregistration.

This correction changes sequencing, not the accepted S0, D0, O0a or O0b
meaning.

## Scope

This checkpoint owns only:

- a slim common planar core retaining the inputs and evidence necessary to
  validate S0 and D0;
- separately hashed reference O0a and ten-run sensitivity artifacts;
- exact split and materialization of frozen `LandformObjectPacketCoreV0`;
- a new core-backed O0b artifact with unchanged mechanical evidence; and
- compatibility, mutation-binding, determinism and bounded-cost tests on
  existing accepted manufactured fixtures.

It does not own a new terrain, linked-case result, product adapter, spherical
D0/O0a/O0b, persistent object identity, natural-kind landform names, H/C/G
composition, renderer state or O0a storage factorization.

## Frozen schemas

Rust field order below is wire order. Every type derives deterministic
`Serialize`/`Deserialize`, `Debug`, `Clone` and `PartialEq` in the same manner
as the accepted V0 packet types.

```rust
pub struct CommonPlanarEvidenceCoreV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub population: CommonEvaluationPopulationV0,
    pub geometry_identity: PacketGeometryIdentityV0,
    pub graph: EvaluationSurfaceGraphV0,
    pub physical_elevation_km: Vec<f64>,
    pub scored_cell: Vec<bool>,
    pub local_runoff_supply: Vec<f64>,
    pub surface_config: SurfaceHierarchyConfigWireV0,
    pub drainage_config: DrainageConfigWireV0,
    pub surface_hierarchy: SurfaceHierarchyV0,
    pub drainage: EvaluationDrainageV0,
    pub derived_core_hash: u64,
}

pub struct ReferenceRelationshipEvidenceV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub core_hash: u64,
    pub payload: LandformRelationshipsWireV0,
    pub derived_reference_hash: u64,
}

pub struct RelationshipSensitivitySuiteV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub core_hash: u64,
    pub payloads: Vec<LandformRelationshipsWireV0>,
    pub derived_suite_hash: u64,
}
```

The common core omits the old packet's `relationship_configs`,
`relationship_payloads`, `surface_hierarchy_input_hash`,
`drainage_input_hash` and `predecessor_evidence_hashes`. The first two move to
sidecars. The latter three are duplicate ledgers reconstructible exactly from
the embedded S0, D0 and O0a evidence hashes.

Do not omit physical elevation, the exact local runoff-supply array, scored
mask, population, geometry, graph or full S0/D0 configurations. They bind the
evidence to its source physical arrays and extractor inputs even when O0b does
not read every field directly. Final physical elevation is an arm outcome,
whereas geometry, runoff and mask may be shared inputs.
`CommonEvaluationPopulationV0` remains the authoritative runoff declaration and
common-population identity; no second declaration is added.

The reference wrapper does not duplicate configuration or evidence hashes:
the unchanged `LandformRelationshipsWireV0` payload already contains its
configuration, namespace, geometry, predecessor hashes, work counts and
derived evidence hash.

### Registered identities

```text
CommonPlanarEvidenceCoreV0.schema_version
  = landform-common-planar-evidence-core-v0
ReferenceRelationshipEvidenceV0.schema_version
  = landform-reference-relationship-evidence-v0
RelationshipSensitivitySuiteV0.schema_version
  = landform-relationship-sensitivity-suite-v0

all three hash_version
  = fnv1a64-bincode-fixint-le-v0
```

The sensitivity suite contains exactly ten payloads in this order:

1. `StationSpacingLow`;
2. `StationSpacingHigh`;
3. `CrossSectionHalfLengthLow`;
4. `CrossSectionHalfLengthHigh`;
5. `CrossSectionSampleStepLow`;
6. `CrossSectionSampleStepHigh`;
7. `RelativeHeightFractionLow`;
8. `RelativeHeightFractionHigh`;
9. `MaximumDownstreamSupportLow`; and
10. `MaximumDownstreamSupportHigh`.

Each uses the already-registered one-factor configuration for its namespace.
`Reference` is forbidden in the suite; every sensitivity namespace is
forbidden in the reference wrapper. This checkpoint does not introduce a
factorized relationship payload.

## Encoding and hashes

All hashes use the accepted fixed-width, little-endian bincode encoding and
FNV-1a-64. Decoders reject trailing bytes. Derived outer hashes are excluded
from their own preimages.

The common-core preimage is, in exact order:

```text
schema_version
hash_version
population
geometry_identity
graph
physical_elevation_km
scored_cell
local_runoff_supply
surface_config
drainage_config
surface_hierarchy
drainage
```

The reference preimage is
`(schema_version, hash_version, core_hash, payload)`. The sensitivity-suite
preimage is `(schema_version, hash_version, core_hash, payloads)`.

Assembly canonicalizes declaration zeros before hashing. Decoding rejects
noncanonical negative zero in declarations, non-finite values, negative local
runoff, invalid lengths, invalid configuration or any mismatch between the
declared and recomputed hash.

“Declaration zeros” means only requested domain width/height; uniform runoff
rate or asymmetric-Y base/gradient; and portal span start/end/base level. These
inherit V0 assembly canonicalization. Negative zero in physical elevation or
the retained runoff array is rejected rather than canonicalized. No graph,
configuration or evidence float receives new zero rewriting in this contract.

## Semantic validation

`validate_common_planar_evidence_core_v0` must perform the accepted V0 packet
checks rather than merely trust the outer hash:

- require the registered regular-planar population, positive finite spacing,
  ordered unique portals, whole scored support and exact population hash;
- validate CSR, faces, polygons, physical measures and declared graph hash,
  then require exact equality with the deterministic regular-mesh rebuild;
- require all physical arrays to equal the graph length, physical elevation to
  be finite and local runoff to be finite and nonnegative;
- reconstruct formula-declared runoff arrays and require bitwise equality, or
  recompute and verify the canonical hash for `ExactSameMeshArrayV0`;
- validate the registered S0 and D0 wire configurations;
- recompute the embedded S0 and D0 evidence hashes; and
- rebuild S0 and D0 deterministically from retained inputs and require full
  value equality with the embedded evidence.

Standalone wrapper decoding validates its schema, ordering, namespace,
configuration and outer hash. Full sidecar validation takes a supplied,
fully-validated core and additionally requires exact `core_hash`, geometry
identity and S0/D0 predecessor hashes, then rebuilds O0a and requires complete
payload equality including work counts and embedded evidence hash. A wrapper
cannot become semantically valid merely by repairing its outer hash.

## Frozen V0 projection and materialization

Splitting a valid `LandformObjectPacketCoreV0` is mechanical:

1. copy the retained core fields without numerical conversion;
2. place the `Reference` O0a payload in the reference wrapper;
3. place the other ten payloads in registered order in the sensitivity suite;
4. validate all three artifacts against one another; and
5. compute the three new outer hashes.

The complete ten-run suite is mandatory for this exact V0 split and
materialization proof because frozen V0 contains all eleven O0a payloads. It is
optional only as retained evidence for later consumers that begin from a new
common core and do not request historical V0 materialization.

Materialization is the exact inverse. It restores the eleven registered
relationship configurations and payloads in canonical `Reference`-then-
sensitivity order; restores the top-level S0 and D0 input hashes from the
embedded evidence; rebuilds `predecessor_evidence_hashes` in canonical
namespace order; and recomputes the frozen V0 packet hash under the unchanged
V0 implementation.

For every valid V0 fixture in the registered matrix, materialization must equal
the original Rust value, serialized bytes and packet hash exactly. Splitting
that materialized value again must reproduce the same new artifact values,
bytes and hashes. Existing V0 types, version strings, encoders, decoders and
hash preimages must not change.

## Core-backed O0b

Introduce a new correspondence type; do not reinterpret
`ObjectCorrespondenceV0`:

```rust
pub struct CoreObjectCorrespondenceV1 {
    pub schema_version: String,
    pub hash_version: String,
    pub config: CorrespondenceConfigWireV0,
    pub source_core_hash: u64,
    pub target_core_hash: u64,
    pub highland_nested_pairs: Vec<AreaPairV0>,
    pub highland_exclusive_pairs: Vec<AreaPairV0>,
    pub drainage_nested_pairs: Vec<AreaPairV0>,
    pub drainage_exclusive_pairs: Vec<AreaPairV0>,
    pub drainage_line_pairs: Vec<LinePairV0>,
    pub context_records: Vec<ContextV0>,
    pub assignment_records: Vec<AssignmentV0>,
    pub best_components: Vec<BestComponentV0>,
    pub metric_conflicts: Vec<MetricConflictV0>,
    pub topology_records: Vec<TopologyV0>,
    pub work_counts: CorrespondenceWorkCountsV0,
    pub derived_correspondence_hash: u64,
}
```

```text
schema_version = landform-correspondence-o0b-core-v1
hash_version   = fnv1a64-bincode-fixint-le-core-v1
```

Its hash preimage substitutes `source_core_hash` and `target_core_hash` for the
old packet hashes and otherwise keeps the accepted O0b field order. Validation,
ordering, exact ties/nulls/conflicts, assignment rules, topology reporting and
work counts are unchanged. Reference and sensitivity hashes are not inputs.

The implementation should share one internal mechanical builder between old
packet-backed and new core-backed entry points. Frozen V0 O0b remains available
and byte-compatible. The new artifact makes a new identity claim: its schema,
preimage and bytes are not cross-decodable or interchangeable with old O0b.
Numeric FNV-1a-64 inequality is reported when observed, not an acceptance gate;
the identity distinction does not rely on collision absence.

## Executable matrix and gates

No linked-deformation result is in this matrix. Use only the existing frozen
fixture constructors and do not alter their surfaces to make a gate pass.

### Exact packet decomposition

At 8, 4 and 2 km:

- isolated-four-cone V0 splits, independently validates and materializes to
  exact original value, bytes and hash;
- asymmetric-Y V0 does the same; and
- core bytes/hash are constructed independently of sidecar presence and remain
  identical when paired with the uniquely valid reference and suite.

The failed linked-four-cone 2 km full packet remains a historical D0 ambiguity
halt. Do not manufacture a V0 value for it. The valid linked-four-cone 4 and
8 km packets may be used only for compatibility/equivalence coverage.

This V0 split deliberately inherits `WholeGraphSupportV0`. The intended linked
testbed instead describes a scored central window inside a buffer. Partial
scoring requires a new population/core schema identity, not reinterpretation of
this V0. Conversely, selecting whole-graph scoring would amend the testbed's
buffer-invariance contract. The organization comparison preregistration—not
this compatibility refactor or the shared-input manifest—must own that choice
and its explicit buffer semantics.

### Mechanical equivalence

For every direction in this matrix, old and new correspondence must have
bit-identical mechanical fields and work counts after excluding only version,
old-packet/new-core identity and derived outer-hash fields:

| fixture | directions |
|---|---|
| asymmetric Y | 4→8, 8→4, 4→2, 2→4 |
| isolated four cone | 4→8, 8→4, 4→2, 2→4 |
| linked four cone | 4→8, 8→4 |

The accepted equal-elder counterfactual and fixed five-cell remapping remain
shared-kernel regression gates; they operate on intentionally synthetic
hierarchies/populations that are not assembler-valid cores. Semantic artifact
rejection and whole-artifact reversal must additionally pass through the new
serialized/core-backed path without changing their expected mechanical
answers.

### Rejection and mutation binding

Focused tests reject:

- missing, duplicate, extra or incorrectly ordered sensitivity namespaces;
- reference payloads in the suite and sensitivity payloads in the reference;
- namespace/configuration mismatches;
- a valid sidecar bound to a foreign core;
- geometry or S0/D0 predecessor mismatch;
- a rehashed but semantically malformed O0a payload;
- any retained core-field mutation with only the outer hash repaired; and
- decoded trailing bytes, noncanonical declarations and malformed graphs or
  physical arrays.

A legitimate mutation may validate only after every causally affected
predecessor is rebuilt. It is a new core, not a mutation escape hatch.

The finite mutation-binding matrix uses asymmetric-Y 4 km and repairs only
`derived_core_hash` after each single mutation. It sets each version string to
`foreign`; adds 4 km to declared domain width without repairing its population
hash; flips the geometry identity's graph-hash low bit; adds `1e-6 km` to cell
0's graph-centre x coordinate; adds `1e-6 km` to physical elevation cell 0;
clears scored cell 0; adds `1e-6` to local runoff cell 0; adds `1e-6 km` to the
surface closure level; adds `1 km²` to the first D0 support threshold; and
flips the embedded S0 or D0 evidence-hash low bit in separate witnesses. Every
witness must fail semantic validation. A separate witness flips only the
stored outer hash and must fail hash validation. This covers every top-level
retained field class without an unbounded mutation search.

### Deterministic repeat

Independently rebuild asymmetric-Y 4→8 and isolated-four-cone 4→2 twice in one
process. For both source and target, require exact value, serialized-byte and
hash equality for the core, reference, suite and materialized V0; require the
same for the core-backed correspondence. Cross-process and cross-platform
byte stability remain consequences of the frozen encoding, not an additional
gate in this bounded checkpoint.

### Cost report and ceiling

The focused release evaluation reports, separately at isolated 8/4/2 km:

- serialized common-core, reference, sensitivity-suite and materialized-V0
  bytes;
- fixed-encoding bytes for every common-core preimage field separately:
  versions, population, geometry identity, graph, elevation, scored mask,
  runoff, both configurations, S0 and D0, plus the stored outer hash;
- assembly, split, validation, materialization and old/new correspondence wall
  times; and
- whole-process peak RSS from `/usr/bin/time -v`.

Compile time is excluded. Record CPU, available memory, OS, revision/dirty
state and exact command. The focused process has a preregistered ceiling of
2 GiB peak RSS and ten minutes wall time on the development WSL machine. A
ceiling breach stops this checkpoint for review; it does not license removal of
validation or evidence. Artifact size reduction is reported evidence, not a
semantic acceptance threshold.

## Acceptance and stop rules

Accept this checkpoint only if all frozen V0 preservation, exact inverse,
sidecar independence, mechanical-equivalence, invariance, mutation and
deterministic-repeat gates pass and the focused release run stays within its
resource ceiling.

Stop and amend before implementation continuation if exact V0 materialization
is impossible without hidden state, if new O0b changes a mechanical answer, if
a wrapper can validate against the wrong core, or if the split requires
changing an accepted predecessor meaning.

Passing this contract means only that the evaluation artifact boundary is
sound and cheaper evidence can be requested selectively. It does not validate
a linked terrain, choose an organization owner, or promote a product path.
