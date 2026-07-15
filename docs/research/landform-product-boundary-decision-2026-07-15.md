# Landform product boundary and packet-retention decision

**Date:** 2026-07-15

**Status:** architecture decision; common-core schemas and gates now separately
preregistered but not implemented

**Parents:** [landform object packet v0](landform-object-packet-v0-2026-07-14.md),
[bounded O0b contract](landform-object-packet-o0b-2026-07-15.md),
[landscape organization strategy](../landscape-strategy.md)

## Decision

Do not implement the umbrella's provisional combined “packet/product R0” as
one packet type or one correspondence problem. Split it into:

1. a **common planar evidence core** for arm-neutral final-surface comparison;
2. separately hashed **reference relationship evidence** and optional
   **sensitivity suites** over that core; and
3. a distinct **product-reference observation** over retained product state,
   with product-native hydrology and integration provenance in its own
   derivation namespace.

The unchanged product remains an external, noncompetitive reference. Inventory
after this decision found that the linked testbed defines shared inputs and
forcing, not an arm-neutral final terrain. The next checkpoint is therefore the
separately preregistered [common-core schema and equivalence
contract](landform-common-core-v0-2026-07-15.md), proved on accepted manufactured
packets. A later linked-input manifest may bind shared inputs
but may not emit final-surface landform evidence before an arm produces that
surface. Neither checkpoint waits for a product D0/O0a/O0b adapter or claims
cross-domain correspondence.

## Why the original boundary is wrong

The product and common-testbed drainage branches do not currently mean the
same thing.

- Product G0/S0 is implemented on the closed spherical Voronoi graph.
- Common D0 is a planar, portal-rooted, final-surface derivation. It virtually
  conditions the surface without changing physical elevation and requires
  every receiver chain to terminate at a declared boundary portal.
- Product hydrology uses ocean cells as sinks, retains an ocean-seeded receiver
  forest and physically lowers sparse outlet cells before the post-integration
  surface becomes authoritative.
- O0a uses planar point location, clipping, ray traversal and cross-section
  geometry. O0b accepts only registered regular planar packet populations.

A thin adapter cannot turn coastlines into planar boundary portals, product
cuts into D0 fill debt, or spherical relationship geometry into Cartesian
geometry without changing the question. Sharing an outer Rust type would hide
that semantic difference rather than solve it.

The complete causal picture may eventually contain both a common spherical
surface-derived drainage branch and a product-native observed branch. If both
are built, retain both with distinct hashes and provenance; neither may silently
substitute for the other.

## Product read boundary

After asserting that Stage 4 exists, the smallest physical-state boundary is:

```text
tessellation = &fine.base.tessellation
surface      = fine.eroded.as_ref().expect("Stage 4 product surface")
```

It supplies the exact product sphere, final post-integration elevation, exact
pre-integration elevation through sparse cut provenance, the precipitation
array actually used by hydrology, the native solid-angle array needed to
reconstruct the original `f32 precipitation * f32 area` local-supply operation,
receiver forest, accumulation, oceans, basins and water bodies. Local supply is
reconstructible but is not independently retained on `FineSurface`.

“Unchanged product” additionally requires envelope provenance: the complete
`RunManifest`, revision/dirty state, Stage 4 assertion, cache record and frozen
controls. Bind that identity metadata in the observation envelope while keeping
it outside the physical evidence hash. The physical derivation itself does not
require `World`, view-stage dispatch, tectonic/model fields, renderer state or
presentation settings.

Any later product-hydrology observation must preserve these distinctions:

- final physical elevation versus pre-integration elevation;
- exactly reconstructed native `f32` precipitation-times-solid-angle supply
  versus independently reconstructed `f64` G0 structural area;
- physical integration cuts versus virtual conditioning;
- native receiver/basin evidence versus visibility-selected river semantics;
  and
- product-local object IDs versus any later cross-packet correspondence.

The existing 250k product G0/S0 ancestry observation remains sufficient as the
current noncompetitive product context. A product drainage observer is deferred
until a concrete product question justifies it; it is not a prerequisite for
the common H/C/G testbed.

## Packet retention

The accepted `LandformObjectPacketCoreV0` remains frozen evidence. It requires
exactly eleven O0a namespaces, includes every payload and predecessor hash in
its packet hash, and must continue to decode and reproduce its accepted bytes.
Do not remove fields, reinterpret its hash or call a reference-only value V0.

That V0 shape is not the future runtime interchange contract. Current O0b
correspondence validates the complete packet hash but numerically consumes no
O0a relationship payload. All eleven O0a runs also duplicate invariant backed
faces, highland relationships and saddle associations. Among substantive
object-evidence arrays, only reach probes respond to the five one-factor
controls; configuration, work counts and derived hashes vary with them too.

The next executable packet-shape preregistration should therefore define:

```text
CommonPlanarEvidenceCore
  graph + physical elevation + scored mask
  local runoff supply + runoff declaration
  S0 and D0 configurations, evidence and hashes
  common population/geometry identity
  derived core hash

ReferenceRelationshipEvidence
  core hash + reference O0a config/evidence/hash

RelationshipSensitivitySuite       # optional audit artifact
  core hash + ten registered O0a sensitivity configs/evidence/hashes

ObjectCorrespondence
  source core hash + target core hash + mechanical O0b evidence/hash
```

Names, exact field order, hashes and compatibility gates are frozen by the
[common-core contract](landform-common-core-v0-2026-07-15.md). The dependency
direction is unchanged: sensitivities depend on a common core; correspondence
does not depend on a sensitivity suite. Reference O0a is requested explicitly
by consumers that judge boundary/descent/cross-section evidence.

The new core identity necessarily requires a new correspondence schema and
hash. It must reproduce accepted mechanical tables and assignments, not claim
the old V0 correspondence bytes or hashes.

For historical V0 evidence, a compatibility bundle may store exact sensitivity
payloads as sidecars and materialize the canonical V0 value when its old bytes
or hash are required. Recomputing old payloads is not a durable archive policy
unless the exact executable implementation is also retained.

Do not implement a factorized O0a wire format yet. It adds validation and
compatibility complexity and is justified only if repeated real evaluations
show that retaining all ten sensitivities has material decision value.

## Cost basis

The accepted isolated-four-cone measurements are:

| spacing | full V0 packet | reference O0a | ten sensitivities | packet minus sensitivity payload bytes |
|---|---:|---:|---:|---:|
| 8 km | 20.75 MB | 1.41 MB | 14.45 MB | about 6.30 MB |
| 4 km | 32.20 MB | 1.37 MB | 14.01 MB | about 18.19 MB |
| 2 km | 91.11 MB | 2.49 MB | 25.18 MB | about 65.94 MB |

Separating sensitivities is worthwhile, but it is not a complete scaling fix:
at 2 km the remaining common/base state dominates after sensitivities are
removed. A field-level breakdown has not established how that remainder divides
among geometry, physical arrays, S0 and D0. The common-core equivalence run must
still report full wall time, peak memory and retained artifact bytes under its
fixed resource ceiling.

## Revised checkpoint boundary

The exact slim-core/reference/sensitivity boundary is now preregistered
separately. It must preserve V0 compatibility as historical evidence and
reproduce the accepted O0b mechanical answers from the new core identity.

Do not call forcing an outcome. The linked deformation scenario contains no
arm-neutral final terrain from which a pre-arm S0/D0/O0a/O0b packet could be
derived. Its shared geometry, initial condition, forcing, runoff and portals
belong in a later input manifest. Final-surface evidence begins
only with separately preregistered H/C/G results.

It must not include:

- product-native hydrology adaptation;
- spherical D0, O0a or O0b;
- product-to-testbed object correspondence;
- persistent same/born/retired/split/merge language;
- H/C/G composition or a preferred terrain owner;
- renderer or cartographic state; or
- factorized O0a implementation without new cost/value evidence.

After the artifact split and shared-input manifest exist, preregister the
actual H/C/G comparison and extract objects independently from each arm's final
surface. Product drainage work remains a separate diagnostic decision, not a
hidden prerequisite.
