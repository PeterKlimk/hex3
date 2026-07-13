# Irregular-Voronoi seeded channel extraction R1a

**Date:** 2026-07-13
**Status:** preregistered; G0 geometry substrate implemented and passed; routing/extraction pending
**Predecessor:** [invalidated planar R0](../audits/channel-extraction-r0-2026-07-13.md)
**Design basis:** [centreline geometry and confluence basis](channel-centerline-geometry-basis-2026-07-13.md)

## Decision

On an irregular Voronoi finite-volume mesh with unequal face widths, should a
seeded resolved-valley centreline use:

- **P0:** the strictly downhill face with greatest physical grade; or
- **M0:** the positive outgoing face carrying the greatest fraction of the
  cell's conservative MFD water?

R1a selects at most one local receiver owner and one geometry encoding. It does
not discover heads, create confluences, infer width, persist identity, attach
C1 state or change terrain.

M1 cumulative maximum-product routing is ineligible. R0 did not falsify it, but
no evidence earns a full-domain pass before the two path-local rules are validly
compared.

Implementation checkpoint: the guarded product-backend cap and its planar
finite-volume adapter pass the 8/4/2 km geometry, projection, determinism and
eight-versus-ten-guard gates. The registered A/V inputs, conservative route,
P0/M0 outgoing-rank conflict and path extraction remain unimplemented, so R1a
has not selected an arm. See the
[G0 audit](../audits/channel-extraction-r1a-g0-2026-07-13.md).

## Scope correction from R0

R1a deliberately has no analytic Y. Distinct integral curves of a smooth
steady gradient field cannot merge and then share a suffix. Confluence is a
network coarse-graining/morphology problem and will receive a separate topology
fixture without a fictitious exact junction point.

Cell-centre graph length is also ineligible for promotion. It remains a report-
only diagnostic. Eligible geometry connects the prescribed head through the
midpoints of selected shared Voronoi faces to the selected portal face.

## G0 — deterministic irregular S2 cap

Build a small Earth-radius cap through the product `s2-voronoi` backend rather
than generating an infeasible 2 km whole sphere.

### Frozen construction

- Tangent-domain extent: `256 × 224 km`, centred at the north-pole tangent
  origin; nominal generator spacings `8`, `4` and `2 km`.
- Start from the triangular generator lattice with basis vectors
  `(spacing, 0)` and `(spacing/2, √3 spacing/2)`, extending eight nominal
  spacings beyond the retained rectangle. For integer lattice coordinate
  `(q,r)`, seed ChaCha8 with `0x5231_A11C_E001 XOR
  q·0x9E3779B97F4A7C15 XOR r·0xD1B54A32D192ED03` using wrapping two's-complement
  `u64` arithmetic. Two draws select uniform-disk angle and square-root radius,
  capped at `0.18` nominal spacing. Expanding the guard therefore cannot change
  shared interior generators.
- Map tangent points to the unit sphere with the exponential map using
  `PLANET_RADIUS_KM = 6371`. Add an unjittered 128-point midpoint Fibonacci
  sphere using `z_k = 1 - 2(k+0.5)/128` and golden-angle longitude, discarding
  support points within `0.08 rad` of the north pole, then call
  `Tessellation::from_points_knn_clipping`.
- Retain cells whose generators lie inside the physical rectangle. A retained-
  to-retained shared edge becomes an internal face. A retained-to-guard edge
  becomes a boundary face. Accept only reciprocal pairs backed by exactly two
  shared Voronoi vertices; repaired nearest-neighbour links are ineligible.
- Project retained generators and Voronoi vertices back through the tangent log
  map. Use projected polygon areas, shared-edge lengths, centre distances and
  face midpoints consistently in the planar `LandscapeMesh` adapter. Report
  their difference from spherical area/arc measures.
- South-cut faces whose midpoint projects within `x = [-80, 80] km` share
  `OutletPortalId(401)` and base level `1 km`. A south-cut face has an excluded
  neighbour below `y = -112 km`; every other cut face is closed.

This is a product-backend irregular Voronoi mechanism fixture, not a claim that
its local jitter process reproduces the full product generator statistics.

### G0 gates

1. Rebuilding the same cap preserves cell count and retained adjacency, with
   unit-sphere coordinates equal within `1e-6`; repeated extraction on one
   built mesh is bit-identical.
2. Regenerating with a ten-spacing guard leaves every retained cell, shared
   edge and projected metric unchanged within `1e-3` nominal spacing.
3. Every retained polygon has positive finite area; every internal adjacency is
   reciprocal and has one two-vertex face; no orphan repair enters the cap.
4. Maximum tangent-projection edge-length distortion is below `0.1%` and total
   retained projected-versus-spherical area differs by less than `0.2%`.
5. Record internal face-width and area percentiles and require positive spread.
   Also record generator-to-polygon-centroid offsets: the state is a polygon
   mean while the current two-point operator uses generator distances, so A is
   explicitly the linear-consistency gate for that approximation.
   More importantly, if P0 and M0 have no outgoing-rank conflict at a cell
   visited by either arm anywhere in the registered A/V matrix, the
   discriminator is invalid rather than a tie.

Failure stops R1a before extractor implementation or score interpretation.

## Finite-volume inputs

For angle `θ`, define `u = (sin θ, cos θ)`, `v = (-cos θ, sin θ)` and analytic
outlet `o = (δ, -112 km)`, where `θ ∈ {0, 0.31 rad}` and
`δ ∈ {0, 0.7 km}`. Coordinates are `s = (p-o)·u` upstream and
`n = (p-o)·v` transverse. Physical dimensions and coefficients remain fixed
while this surface rotates/translates relative to the mesh.

For every polygon, integrate the registered affine/quadratic surface exactly
from projected polygon moments. Do not substitute the generator-point value.
Uniform runoff is the registered `500 km/Myr` (`0.5 m/yr`) times projected cell
area, producing local supply in `km³/Myr`. Route once with
`FaceFlowCache::route_with_portals` and
`FlowPartition::MfdSlope`; both extractors consume the same immutable result.

Executable clarification frozen before input implementation: A and V use the
closed-form first and second moments of the complete projected polygon. B is
piecewise quadratic, so polygons crossing `n = -12` or `+12 km` are clipped at
those lines and the two exterior quadratic pieces are integrated separately.
Whole-polygon moments alone are not an exact B input.

### A — affine direction control

```text
z(s,n) = 1 km + 0.01 s
```

One head lies at `s = 176 km, n = 0`; require that point to lie in exactly one
retained polygon and use that polygon as the seed. The analytic reference is
the straight down-gradient segment to the registered outlet. A has no claim to
being a river. It tests whether unequal cell faces corrupt a simple continuum
direction.

Head containment uses the convex projected polygon, accepts either winding and
tests every retained cell. An edge cross product within
`1e-10 × nominal_spacing²` of zero is boundary-adjacent: the case is invalid
rather than assigning an incidental build-index owner.

### V — resolved quadratic valley

```text
z(s,n) = 1 km + 0.01 s + 0.0008 km⁻¹ n²
```

Use the same head and outlet. The transverse minimum `n = 0` is the analytic
thalweg. This is the only R1a fixture allowed to support a seeded channel-
geometry claim.

### B — broad non-identifiability control

```text
z(s,n) = 1 km + 0.01 s
       + 0.0008 km⁻¹ max(|n| - 12 km, 0)²
```

Trace prescribed heads at `n = -8` and `+8 km`. Report path spread and margins;
do not score either path against a privileged centreline.

## Registered receiver arms

P0 and M0 compile no independent terrain or water state. A shared extraction
context may index boundary faces once; after that each arm inspects only faces
of visited cells.

- P0 scores internal and target-portal faces by strictly positive physical
  grade from the polygon-mean surface. On the MFD operator this is also the
  local flux-density rank with incidental face width removed.
- M0 scores positive internal and target-portal faces by the existing outgoing
  MFD fraction. Because available supply is common to a donor's faces, fraction
  and integrated face flux have the same rank.
- Both choose by exact numeric score. Bit-equal scores use projected face-
  midpoint coordinates, portal ID and only then build index. Near ties are
  diagnostics and never silently alter the winner.
- The API receives the required `OutletPortalId`; reaching another portal,
  closed boundary, sink, cycle or guard is a failure.

For every visited cell record both the best/second physical-grade margin and
the best/second MFD-fraction margin, regardless of the active arm. This keeps
local evidence distinct; no cumulative cost may be labelled a local margin.

Executable ordering clarification frozen before rank inspection: maximize the
numeric score; an exact tie chooses the lexicographically smallest
`(midpoint.x, midpoint.y, midpoint.z, portal_key, combined_face_index)`. Internal
faces use `portal_key = u32::MAX`; boundary build indices follow all CSR edge
indices in one combined index space. The reported normalized margin is
`(best-second)/max(abs(best), f64::MIN_POSITIVE)`. A sole eligible face uses a
zero second score. These conventions resolve numerical identity only and do
not make a near-tie physically decisive.

Before path tracers are implemented, scan every eligible donor in the complete
A/V spacing × angle × translation matrix and require at least one P0/M0 winner
conflict. This is a cheap necessary precheck, not G0 gate 5: promotion still
requires a conflict at a cell actually visited by either arm after tracing.
One routed case means one spacing, angle, translation and surface; B's two
heads share the same immutable route.

## Geometry encodings

### C0 — cell-centre diagnostic

Generator centre to generator centre, followed by the outlet-face midpoint.
Report cross-track error and length but never use C0 for promotion.

### F0 — selected-face midpoint baseline

Start at the prescribed physical head. For each selected internal transition,
append the midpoint of the actual shared Voronoi face; finish at the selected
portal-face midpoint. This convention adds no state and costs `O(path length)`.
It is not called a reconstructed velocity streamline.

If F0 fails the affine gates, stop. A conservative `H(div)`/`RT0` interior
reconstruction is a separately preregistered escalation, not an implementation-
time patch.

## Executable contracts and metrics

### Invariants

1. Every routed donor's positive internal/portal fractions sum to one; sink
   donors sum to zero. Reciprocal internal faces carry water in at most one
   direction. Source, portal and sink ledger error is at most `1e-12` of total
   supply; report the absolute `km³/Myr` residual as well.
2. Extraction leaves terrain, supply, fractions, fluxes, routing surface,
   potentials and ledgers bit-identical.
3. Every path is neighbour-connected, acyclic, strictly eligible under its arm
   and ends at portal `401` without a guard.
4. Repeating the complete extraction on the same built mesh is bit-identical.
5. Shared context construction is reported separately. P0/M0 tracing is
   `O(path faces)`; no hidden whole-domain sort, filter or iterative solve is
   eligible.

### Physical path metrics

For A/V and both C0/F0 encodings report:

- maximum absolute cross-track coordinate `|n|` over all polyline vertices;
- exact polyline arclength and relative error from `176 km`;
- outlet-face midpoint error from the analytic outlet;
- total positive along-track backtracking;
- path cell/face count and every local grade/fraction margin; and
- maximum/minimum/spread across orientations and translations.

Cross-track maxima and arclength are exact for the stored polyline. They replace
R0's sampled Hausdorff claim. The analytic endpoint error is reported
separately rather than hidden inside a sampled symmetric distance.

### Frozen F0 gates

For each eligible arm on both A and V:

1. finest-grid maximum cross-track error is at most `3 km` and decreases in net
   from 8 to 2 km;
2. finest-grid arclength error is below `5%`;
3. finest-grid total along-track backtracking is at most `2 km`;
4. finest-grid maximum cross-track error may not exceed twice its minimum
   solely through orientation/translation when the absolute spread also
   exceeds `2 km`; and
5. no gate is passed solely by a build-index tie. Near-tie paths remain
   numerically valid but their physical ambiguity is disclosed.

B has no geometry gate. Exact or large-margin routing on B cannot manufacture a
physically unique thalweg.

## Frozen interpretation

- If G0 fails or P0/M0 never differ, invalidate the fixture and stop.
- If P0 passes every A/V F0 gate, retain P0 unless M0 is required to pass a gate
  P0 fails. This prior keeps sparse geometry independent of incidental face
  width while MFD remains authoritative for water.
- If P0 fails and M0 alone passes all gates, select M0 provisionally and record
  that integrated face geometry improves the discrete representative path.
- If both fail, select neither. Do not add M1, smoothing, curvature or
  persistence to conceal the failure.
- C0 cannot win. F0 passing authorizes only a derived face-crossing geometry
  convention, not a physical channel width or subcell streamline.
- B disagreement is correct non-identifiability evidence.

Passing R1a authorizes a seeded local receiver and cheap path geometry on one
resolved valley. A conservative confluence topology fixture, geomorphic Y,
head discovery, promotion/retention, persistent reach geometry, C1 coupling,
sediment and product integration remain separate gates.
