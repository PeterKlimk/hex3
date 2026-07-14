# Landform O0b correspondence candidate audit

**Date:** 2026-07-15
**Status:** implementation candidate; frozen evaluation halted; not accepted or promoted

## Outcome

The bounded planar O0b implementation candidate now assembles exact common
G0/S0/D0/O0a packet cores and computes mechanical cross-packet highland and
drainage-node correspondence. The ordinary manufactured kernels, sparse-index
oracles, asymmetric-Y 4-to-8/2 evidence and linked-four-cone 4-to-8 highland
evidence pass.

O0b does **not** receive a passing checkpoint verdict. The frozen linked-four-
cone 4-to-2 witness cannot be assembled into its required common packet: D0
returns the already-registered typed error
`DepressionHierarchyAmbiguity { depression: 1 }`. This occurs before O0a or
O0b correspondence. The contract's stop rule therefore applies. The exact
surface cannot be perturbed, D0 omitted or the expected answer weakened after
observing this result.

No product adapter, spherical correspondence, packet/product R0, persistent
identity/event language, H/C/G terrain arm or product decision was implemented
or evaluated.

## Candidate implementation

The candidate adds:

- deterministic common packet assembly over the exact common population,
  G0/S0, D0 and all eleven registered O0a namespaces;
- fixed-int little-endian serialization and FNV-1a hashes, with envelope
  metadata excluded from common bytes;
- convex cell intersection with compensated area/centroid accumulation;
- finite-capsule line proximity using pair-local covering radii and analytic
  segment intervals;
- nested and exclusive highland and reference-drainage support extraction;
- exact maximum sets, margins, ties, nulls, best-graph cardinality, metric
  conflict, two-way context and report-only topology;
- target cell and segment indexes with explicit work ledgers; and
- deterministic artifact serialization and strict semantic validation during
  build, hash, encode and decode, including rejection of rehashed malformed
  formulas, ordering, components, conflicts, topology, context and work counts.

The implementation deliberately emits mechanical evidence, not claims that an
object is the same, born, retired, split, merged or persistent.

## Evaluation ledger

| Gate | Result | Note |
|---|---|---|
| packet assembly, namespace canonicalization and hashes | pass | repeated/reordered inputs, envelope isolation, mutation, foreign/missing/duplicate namespace and round trip |
| exact area, line and local-radius arithmetic | pass | includes positive slivers, differing cell tilings, exact ties and no-contact controls |
| nested hierarchy | partial pass | parent/children, parent-only target, exact child tie and zero-exclusive parent pass; both explicit elder resolutions remain incomplete |
| assignment cardinality, nulls, metric conflict and topology | pass at kernel/candidate scope | no identity/event promotion |
| background, portal, ineligible and outside-domain context | pass | retained in both directions |
| enumeration, rigid-transform, same-mesh and sparse-index controls | pass at implemented kernel scope | 100-by-100 separated cell and line populations perform zero pair tests |
| ordinary library suite | pass | `308 passed; 14 ignored` after strict-wire validation |
| asymmetric-Y 4-to-8 and 4-to-2 | pass | trunk/west/east unique in exclusive-area and line channels; no metric conflict |
| linked-four-cone 4-to-8 | pass | all four 4 km reference labels have unique same-label exclusive-area maxima in both directions |
| linked-four-cone 4-to-2 | **halt** | D0 `DepressionHierarchyAmbiguity { depression: 1 }` while constructing the frozen 2 km predecessor packet |

The fixed full-reference remapping permutation, both explicit elder-resolution
packet oracles and whole-artifact source/target reversal matrix remain
incomplete. These are candidate gaps, not waived gates, and cannot turn the
halted checkpoint into a pass.

## Numerical correction exposed by the witness

The first linked 4-to-8 execution also exposed an O0a boundary-roundoff defect.
A longitudinal sample at the south portal midpoint differed from the polygon
boundary by floating roundoff, while point location required exact equality.
The spatial index now uses the already-declared G0/S0
`endpoint_match_abs_km = 1e-8 km` tolerance, searches every bucket touched by
that tolerance box, and resolves multiple containing cells by the existing
canonical centre ordering. A focused domain-boundary regression and the full
library suite pass.

This is an explicit predecessor numerical correction, not O0b semantic
authority. It does not relax graph validation or alter physical/presentation
ownership.

## Measured cost

Release-mode audit commands were measured after compilation. Peak process
memory and sensitivity-only duplication bytes were not instrumented and remain
unreported rather than inferred.

| Fixture | cells source/target | clips / full cell product | segments source/target | tests / full segment product | packet bytes source/target | correspondence bytes | audit wall time |
|---|---:|---:|---:|---:|---:|---:|---:|
| asymmetric-Y 4→8 | 896 / 224 | 2,538 / 200,704 (1.264%) | 25 / 13 | 92 / 325 (28.31%) | 1,204,999 / 520,434 | 6,759 | combined 4→8/2 test: 1.38 s |
| asymmetric-Y 4→2 | 896 / 3,520 | 10,808 / 3,153,920 (0.343%) | 25 / 48 | 189 / 1,200 (15.75%) | 1,204,999 / 3,737,936 | 6,060 | included above |
| linked four cone 4→8 | 67,200 / 16,800 | 200,282 / 1,128,960,000 (0.0177%) | 1,665 / 2,803 | 9,087 / 4,666,995 (0.195%) | 99,408,917 / 107,789,195 | 1,153,512 | 77.61 s |

The sparse geometry path satisfies the no-unreported-Cartesian-scan gate. The
dominant linked-world cost is nevertheless substantial: assembling and
retaining eleven O0a sensitivity payloads produces roughly 100 MB packet cores
before product-scale use. That is architectural evidence against casually
promoting this exact packet shape, even though the correspondence index itself
is sparse.

## Interpretation and next decision

The 2 km result invalidates the linked-four-cone surface as a *full common
packet* witness under the current D0 tree contract. It does not show that the
O0b overlap kernel chose the wrong highland match, because O0b never received
the target packet.

Two future paths are legitimate, but neither is authorized by this audit:

1. preregister a replacement full-packet highland witness whose exact drainage
   state is representable by D0, while retaining the original surface-only S0
   topology evidence; or
2. separately reconsider whether equal-spill multi-outlet structure justifies
   changing D0 from a unique-parent depression hierarchy to a richer
   representation.

The first is the Pareto default unless real/common evaluation shows the D0
ambiguity policy discarding valuable worlds. A D0 redesign should be purchased
for demonstrated downstream value, not solely to make one manufactured O0b
witness pass.

Until an amendment is frozen and the remaining matrix passes, O0b is an
available implementation candidate and diagnostic instrument, not accepted
architecture. R0 and H/C/G composition remain blocked behind that decision.
