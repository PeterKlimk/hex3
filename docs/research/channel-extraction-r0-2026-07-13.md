# Seeded channel-extraction R0

**Date:** 2026-07-13  
**Status:** invalidated during first implementation; no arm selected
**Predecessor:** [channel-ownership M0](../audits/channel-ownership-memory-m0-2026-07-13.md)
**Result:** [R0 audit](../audits/channel-extraction-r0-2026-07-13.md)

## Question

Given a conservative MFD face-flow field, a channel head and an outlet, what is
the cheapest deterministic way to derive one sparse centerline that corresponds
to the physical valley and remains stable under refinement and mesh-relative
orientation?

This is the missing extraction rung between continuum water and a persistent
reach object. It deliberately does **not** discover channel heads, apply the
initiation score, infer width, attach C1 state, preserve identity or alter water
or terrain. Those responsibilities cannot decide which path extractor wins.

## Why MFD water and a sparse centerline are separate

Multiple-flow routing is useful for sheet flow, divergent hillslopes and
conservative supply. A river centerline is nondispersive. Turning every MFD
face into a river would confuse distributed water support with channel
topology; replacing MFD water with SFD would discard the continuum result merely
to obtain a convenient graph.

Established GIS workflows commonly combine MFD/D-infinity accumulation or
channel-support evidence with a separate one-receiver reach tree. TauDEM and
GRASS both embody variants of this separation
([TauDEM](https://hydrology.usu.edu/taudem/taudem5/documentation.html),
[GRASS `r.stream.extract`](https://grass.osgeo.org/grass-stable/manuals/r.stream.extract.html)).
MFD and D-infinity reduce some grid restriction, while local single-direction
choices remain vulnerable to edge orientation and near ties
([Freeman 1991](https://doi.org/10.1016/0098-3004(91)90048-I),
[Tarboton 1997](https://doi.org/10.1029/96WR03137),
[Orlandini et al. 2003](https://doi.org/10.1029/2002WR001639)).

Global morphology/geodesic methods such as GeoNet integrate more context and
can bridge noisy local evidence
([Passalacqua et al. 2010](https://doi.org/10.1029/2009JF001254)). They also add
nonlinear filtering, curvature scale, endpoint detection and minimum-cost
parameters whose evidence is mainly metre-scale DEM structure. At Hex3's
kilometre-scale cells they are an offline upper-bound reference, not an eligible
first owner. Resolution is an identifiability limit: channel position, slope
and contributing area all degrade when cell spacing approaches hillslope or
valley scale
([Zhang and Montgomery 1994](https://doi.org/10.1029/93WR03553),
[McMaster 2002](https://doi.org/10.1029/2000WR000150)).

## Existing Hex3 substrate

- Production `Hydrology` constructs a priority-filled SFD graph using physical
  slope and accumulates precipitation-weighted area. `RiverNetwork` thresholds,
  computes hierarchy and collapses degree-one cell chains, but its cell paths
  and IDs are per-stage snapshots.
- Experimental `FaceFlowCache` already owns physical face fractions and fluxes,
  stable portal IDs, portal/sink water ledgers, a non-mutating depression route
  and a high-to-low DAG order.
- The parked A4 drainage pulse contains reusable reverse-topological Strahler
  propagation, but its SFD extraction is coupled to uplift and its older
  adaptive-mesh receiver chooses lowest neighbor rather than the corrected
  physical gradient. Reuse the graph pattern, not that receiver.
- Least-squares specific-discharge magnitude is not a valid sole gate at an
  unresolved confluence because opposing inflows can cancel. R0 traces face
  flux and scores integrated paths; it does not use pointwise `|q|` to decide a
  junction.

No MFD-to-sparse-thalweg implementation exists in current code or history.

## Registered arms

All arms are passive interpretations of one immutable mesh, physical surface,
supply field and `FaceFlowCache`.

### P0 — physical-gradient SFD control

At each cell, choose the strictly downhill internal or open-portal face with
maximum physical grade. Equal grades use a disclosed deterministic geometric
key. This is the planar testbed analogue of corrected production SFD, not a
claim that product and testbed hydrology are byte-identical.

### M0 — local dominant MFD flux

At each visited cell, choose the positive outgoing internal or portal face with
the largest continuum flux fraction. This is the cheapest direct thalweg
interpretation. Record the best/second-best margin; an index tie-break provides
repeatability only and is not evidence of physical uniqueness.

### M1 — cumulative dominant MFD path

Use reverse DAG dynamic programming to minimize the additive route cost

```text
C(cell -> receiver) = -ln(max(face_fraction, epsilon)) + C(receiver)
```

with zero terminal cost at open portals. This selects the maximum-product path
to an outlet and remains `O(N_face)` after the existing topological order. It
integrates several local decisions without adding curvature, filtering or a
global iterative solve. Ties use the same disclosed key and margin audit as M0.

### A0 — analytic skeleton reference

The known continuous V/Y centerline supplies geometry and topology error only.
It is not an eligible generator. A small manufactured flux diamond separately
checks that M0 and M1 differ when the locally largest first edge is not the
globally dominant path; that algebra check cannot select M1 by itself.

## Common fixtures

Use the existing uniform planar hex mesh at nominal 8/4/2 km. Run every physical
surface at two mesh-relative orientations and one fixed sub-cell translation.
All dimensions and supplies are held fixed in kilometres and km³/Myr.

### V — one resolved valley

A smooth sloping quadratic trough has one prescribed upstream head and one
fixed south portal. The analytic thalweg is a straight segment. Uniform physical
runoff supplies the surface. This tests outlet assignment, length, lateral
error, orientation and refinement without confluence ambiguity.

### Y — one confluence

Two smooth branch troughs join one trunk and fixed south portal. The two
prescribed heads should produce paths that merge once near the analytic junction
and share one suffix thereafter. The surface is generated from distance and
along-network coordinates of the analytic Y, not from burned cell paths.

### B — broad unresolved corridor control

A broad flat-bottom axial reach has no unique centreline at the represented
scale. Extractors may return a deterministic path, but it is not scored against
one privileged centre. Report seed/orientation spread and dominance margins.
Failure to identify a unique thalweg here is the correct result, not a reason to
tune a tie-break.

## Path and graph contract

- A path contains neighbor-connected cells followed by one open boundary face
  and stable `OutletPortalId`.
- A selected internal edge must carry positive MFD flux for M0/M1 and strictly
  descend the registered routing DAG. P0 must strictly descend physical grade.
- The union of head paths has out-degree at most one, is acyclic, and may merge
  but not split. Once two Y paths merge, their suffix is identical.
- Geometry uses physical cell-center distances plus final boundary-face center
  distance. Cell count is never length.
- Extraction records selected face fraction, local dominance margin, path cost,
  visited cells/faces and deterministic tie count.
- No selected path owns continuum discharge, channel width or active area.

## Registered metrics and gates

1. The shared continuum water ledger closes below `1e-10 km³/Myr`; extraction
   leaves every flow array and physical elevation bit-identical.
2. All paths terminate at the registered portal without cycles or guards,
   repeated runs are bit-identical, and Y paths merge once and never split.
3. For V, physical Hausdorff distance to A0 decreases in net from 8→2 km and is
   at most `1.5` nominal cells at 2 km. Finest-grid length error is below 5%.
4. For Y, each branch/trunk satisfies the same `1.5`-cell geometry envelope;
   the extracted confluence lies within two nominal cells of the analytic
   junction at 2 km. Report branch/trunk length separately.
5. Report orientation and translated-surface spreads. A method fails if its
   finest-grid maximum geometry error is more than twice its minimum solely
   because of mesh-relative orientation **and** the absolute spread exceeds one
   nominal cell. Tiny sub-cell ratios do not fail by arithmetic accident.
6. Report every selected-face dominance margin and all exact/near ties. A path
   that meets geometry gates only through an index tie is numerically
   repeatable but physically unresolved.
7. B has no geometry winner. Report path separation across seeds/orientations;
   no method may claim a physical centreline there.
8. P0 and M0 extraction are `O(path length)` after routing. M1 compilation is
   `O(N_cell + N_face)` and tracing is `O(path length)`. Report pass ownership
   and counts; no filtering, elliptic or iterative global solve is eligible.

Thresholds are numerical envelopes for this analytic fixture, not Earth
calibration or product acceptance criteria.

## Frozen interpretation

- If M0 passes V/Y topology, geometry and orientation gates, retain it: M1's
  extra full-DAG pass is not justified.
- Select M1 only if it materially reduces real V/Y orientation or near-tie
  error without worsening clear-valley geometry. Passing the manufactured
  diamond alone is insufficient.
- If P0 passes while both MFD interpretations fail, retain MFD as water truth
  but use physical-gradient SFD as the provisional centreline interpretation;
  disclose that separation.
- If all three fail similarly, the represented surface does not resolve a
  thalweg or the face-flow discretization is inadequate. Stop; do not add
  persistence, curvature tuning or GeoNet to conceal it.
- B disagreement does not fail an extractor. Treat it as evidence that a
  unique channel object is not identified at that scale.
- Passing R0 authorizes a seeded, state-free extractor only. Head discovery,
  initiation/resistance, physical support/width, births, lineage, C1 coupling,
  product integration and long landscape runs remain separate gates.
