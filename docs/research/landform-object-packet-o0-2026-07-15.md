# Landform object packet O0a relationship-probe executable contract

**Status:** frozen preregistration; implemented and evaluated as a bounded
common planar/testbed checkpoint; see the
[dated audit](../audits/landform-o0a-relationships-2026-07-15.md)

**Date:** 2026-07-15, before O0a implementation

**Parent:** [Landform object packet v0](landform-object-packet-v0-2026-07-14.md)

**Predecessors:** [G0/S0 executable contract](landform-object-packet-g0s0-2026-07-14.md)
and [D0 executable contract](landform-object-packet-d0-2026-07-15.md)

**Outcome:** bounded pass for deterministic relationship evidence only; no
O0b, product adapter, landform-class promotion or terrain arm was evaluated

## Decision question

Can the independent S0 surface hierarchy and D0 drainage tree support honest,
mesh-aware statements about drainage boundaries, bilateral physical descent,
saddle association and regional reach cross sections without inventing channel
beds, promoting arbitrary terrain classes or using an H/C/G arm's native answers?

O0a is relationship evidence, not another terrain or hydrology model. It may
show that a drainage boundary has unconditioned physical descent on both sides,
that an S0 saddle is near such a boundary, or that a reach-centred section has
two-sided boundary relief. It may also show absence, displacement, ambiguity or
censoring. Those are outcomes, not extractor failures.

## Physical interpretation and limits

A surface-water drainage divide is first a topological partition. On ordinary
resolved topography it often follows a local crest because physical descent on
both sides carries water away from the boundary. Flats, depressions, virtual
conditioning, coarse cells and unresolved flow can separate those ideas. O0a
therefore never equates a D0 partition seam with a geometric ridge by name.

The minimum causal chain is:

```text
physical cell-mean surface
  -> independent S0 highland/saddle structure
  -> independent D0 receiver/catchment structure
  -> face-level flow transition or lateral boundary candidate
  -> bilateral physical-descent probe
  -> highland-boundary and saddle-boundary relationships
  -> reach-centred regional cross-section probes
```

This is physically grounded for topographic surface drainage and an authentic
finite-volume probe at kilometre scale. It does not cover groundwater, karst,
ice divides, distributaries, floodplains or subcell thalwegs. A cross-section
probe is not a valley polygon, and bilateral descent is not persistent ridge
state or a natural-kind ridge classification.

## Scope

This checkpoint implements only the common planar/testbed relationship layer.
It consumes validated, mutually consistent G0, S0 and D0 outputs for one exact
physical surface and emits:

- a classification of every reference-scale raw D0 seam as an exact
  receiver-crossed flow transition or a lateral boundary candidate, with owner
  ancestry retained separately;
- face-local bilateral physical-descent probes and highland/saddle aggregates;
- reference-scale reach-centred longitudinal and cross-section probes; and
- deterministic O0a serialization and hashes.

O0a does not implement cross-packet correspondence or the combined packet;
those require a separately frozen O0b informed by these relationship objects.
It also does not adapt product `Hydrology`, implement product observation R0,
compose H/C/G, compare forcing fields, tune terrain, group boundaries into named
polylines or add channel, sediment, lake, ecology or renderer state.

## Frozen inputs and consistency

One O0a packet core receives:

- one validated planar `EvaluationSurfaceGraphV0`;
- the finite physical cell-mean elevation used by both S0 and D0;
- the exact scored-cell mask used by S0;
- the finite non-negative local runoff supply used by D0;
- the registered G0/S0 and D0 configurations;
- the exact `SurfaceHierarchyV0` produced with the registered G0/S0 config;
- the exact `EvaluationDrainageV0` produced with the registered D0 config;
- `PacketGeometryIdentityV0`, registered as
  `landscape-regular-planar-v0` plus finite positive nominal spacing for all
  competitive/testbed packets, or `projected-r1-voronoi-cap-v0` only for the
  manufactured irregular-geometry control; and
- caller metadata held outside the common evidence hash.

The S0 and D0 schema/hash strings must match their respective registered
versions; they are intentionally not the same schema. Their shared graph and
elevation inputs must be bit-identical. O0a requires positive-zero bits for
every zero in physical elevation and local runoff before either predecessor is
run; negative zero is a typed noncanonical-input error.
O0a reserializes the graph, elevation, scored mask, runoff, configs and result
payloads under the predecessor contracts and verifies both evidence hashes
before deriving relationships; an object record cannot be paired with a merely
equal-length foreign surface.
Only S0 reference highlands and D0's `2,000 km2` reference reach scale enter the
reference O0a object population. Sensitivity populations remain separate named
outputs and cannot add reference objects.

Native H/C/G graphs, tectonic fields, expected segment geometry, product river
importance, arm labels and presentation state are forbidden inputs to common
membership, geometry, relationships and hashes.

### Planar subdivision prerequisite

G0 local face backing is necessary but not sufficient for cross-section point
location. O0a additionally validates, using a deterministic bounding-box index:

- every cell center is strictly inside its own polygon;
- distinct cell interiors have no positive-area overlap greater than
  `endpoint_match_abs_km * (perimeter_a + perimeter_b)`;
- declared boundary segments form closed, non-self-intersecting loops; and
- summed cell area agrees with signed boundary-loop area to the registered G0
  planar relative-area tolerance.

Together with G0's exact internal-face/polygon backing, these are the v0 planar
subdivision invariant. Failure is typed before ray traversal. The geometry
identity also stores a canonical graph hash. The regular identity reruns the
exact regular-hex center-distance, face-width and vertex-radius formulas at its
declared spacing; the projected-cap identity must carry the graph hash produced
directly by that registered adapter. An arbitrary label cannot certify an
arbitrary graph.

## Exact configuration family

The reference cross-section policy is:

```text
along-reach station spacing       20 km
cross-section half-length        100 km
cross-section sample step          4 km
relative-height span fraction      0.25
maximum downstream reach support 400 km
tangent chord half-support          10 km (fixed)
longitudinal sample spacing          4 km (fixed)
```

The registered one-factor alternatives are respectively `10/40 km`,
`50/150 km`, `2/8 km`, `0.15/0.35` and `200/600 km`. A run is either the exact
reference policy or changes exactly one quantity to one registered alternative.
No Cartesian sweep, observed-object normalization or arbitrary runtime value is
valid under schema v0. Detailed object records use the reference policy; named
sensitivity runs report the same records in a separate namespace.

Saddle-boundary proximity uses physical mesh covering radii, not a fitted
distance knob. A cell covering radius is the
greatest Euclidean distance from its center to any polygon vertex. The local
proximity tolerance for a face is the greater covering radius of its two cells.
Every effective tolerance is retained in the evidence.

## Face backing, flow transitions and lateral boundaries

D0 raw boundary faces separate exclusive incremental owners. An upstream reach
hands flow to its downstream parent across one exact internal receiver face.
Other lateral faces may carry the same ancestry-related owner pair without
carrying flow; ancestry alone cannot decide that those faces are not catchment
boundaries.

At the reference reach scale define owner ancestry as follows:

- `Reach(a)` is an ancestor of `Reach(b)` exactly when repeatedly following
  `b.downstream_reach` reaches `a`; an owner is also its own ancestor;
- `Portal(p)` is the terminal ancestor of every reach whose ultimate outlet is
  `p`; and
- distinct portals are incomparable.

D0 raw records intentionally do not retain adjacent cell IDs. Before lineage
classification, O0a rederives every reciprocal internal face whose exclusive
owners differ, retaining its two cells and canonical directed-edge identity,
and compares the multiset of `(owners,endpoints,length)` records bit-for-bit
with D0. A missing or extra record is a typed inconsistency. If the same
canonical raw record is backed by more than one distinct cell pair, O0a returns
`AmbiguousBoundaryBacking`; it may not guess from enumeration order. All later
face heights, covering radii, footprint incidence and traces use this uniquely
backed record.

For each backed raw face, test whether either adjacent cell's frozen receiver is
the other adjacent cell through that exact directed edge. If so, emit
`FlowTransition` and retain receiver direction and owner ancestry; it is not a
lateral boundary. Otherwise emit `LateralBoundaryCandidate` and separately
label its owner pair `Incomparable` or `AncestryRelated`. This is a mechanical
face role, not yet a natural or topographic divide.

Every raw face appears exactly once in those two face roles. Role lengths close
to total raw boundary length under compensated summation. A face cannot be both
receiver-crossed and lateral, and ancestry never overrides the exact face role.

## Bilateral physical-descent probe

For a `LateralBoundaryCandidate` between cells `a` and `b`, independently trace
each cell downstream through the frozen D0 receiver forest until it first
reaches a supported cell whose `cell_reach` is that side's retained reach. A
portal-owned side traces through its exclusive cells to the terminal portal
midpoint. The trace may not cross to a different exclusive owner before its
declared target.

Each side remains a separate receiver trace. Cell samples use physical `z_i`
and a portal target uses its physical base elevation. The symmetric shared-face
value `(z_a+z_b)/2` is retained only as a reconstructed face-height proxy; it is
not an observed subcell surface height and cannot make a predicate pass. No
filled elevation or runoff enters a height calculation.

For side `k`, let `z_adj,k` be its adjacent-cell height, `z_target,k` its target
height and `delta z_e = z_donor-z_receiver` for every physical receiver segment
on the trace. Record:

```text
target drop              z_adj,k - z_target,k
minimum segment drop     min(delta z_e)
remote maximum excess    max(trace physical z)-z_adj,k
trace length/tortuosity  physical receiver length / endpoint distance
physically descending    target drop > 0, minimum segment drop > 0,
                         and remote maximum excess == 0
```

If the adjacent cell is already the target supported cell, emit
`TargetAtBoundaryCell`: target drop and remote maximum excess are zero, minimum
segment drop and tortuosity are unavailable, and `physically_descending` is
false. This is a valid zero-segment probe, not an error.

`bilateral_physical_descent` is true only when both sides are physically
descending. `unconditioned_bilateral_descent` additionally requires that no
segment on either trace is fill-supported, flat-supported or marked physically
non-descending by D0. Exact zero is not positive. Retain every signed margin so
epsilon-scale noise is visible; neither boolean is a quality gate or a
geomorphic-ridge label.

These traces are receiver-topology probes, not shortest cross-slope profiles.
Their target references and signed aggregate margins are retained so a long,
uphill or conditioned path cannot borrow a remote maximum and masquerade as
local crest evidence. The frozen D0 receiver forest plus physical elevation is
the authoritative reconstructible sequence; O0a does not duplicate a full path
per boundary face.

## Highland and saddle relationships

For each reference `HighlandFeatureV0`, collect lateral boundary candidates
incident to at least one cell in its exact S0 footprint. Report total candidate
length, unconditioned-bilateral-descent length, their ratio and the shares
supported by fill, flats and physical non-descent. Call this drainage-boundary
support within a highland footprint, not S0 ridge coincidence: S0 supplies no
independent ridge line.

When the highland's planar orientation is not ambiguous and at least one
unconditioned-bilateral-descent face exists, compute its length-weighted axial
orientation from face tangents using doubled angles. Report the acute angle in
`[0, pi/2]` between that boundary orientation and the S0 footprint principal
axis. If the doubled-angle resultant is zero, orientation is unavailable rather
than assigned by a tie-break.

For each S0 saddle, obtain the reference exclusive owners at its elder and each
losing peak anchor. Among lateral boundary candidates with that exact unordered
owner pair, select the nearest to the saddle's flat centroid by Euclidean
point-to-segment distance. Exact distance ties use coordinate-canonical face
endpoints, not face or cell IDs.

`SaddleBoundaryAssociationV0` records owner ancestry, separation, the effective
covering-radius proximity, saddle elevation minus reconstructed face-height
proxy, the face's physical-descent descriptors and equal-elder ambiguity.
`within_covering_radius` is proximity evidence only. O0a emits no
`transfer_low`, pass or along-boundary-low label because ungrouped faces cannot
establish those claims.

## Regional reach cross-section probes

Build each reference reach polyline from its ordered D0 cell centers and its
owned final receiver segment into the downstream reach or terminal portal. The
polyline remains centreline evidence only. Retain the downstream-most registered
maximum length; if truncated, the new first point is the exact arclength point
on the original polyline.

Parameterize the retained polyline source-to-downstream by physical arclength.
Cross-section stations are at `spacing/2 + k*spacing` and must remain strictly
inside the retained polyline. At station `s`, define the tangent from the chord
between arclength points `max(0, s-10 km)` and
`min(length, s+10 km)`. A zero or non-finite chord is a typed error. Orient the
tangent from source toward downstream. Its counter-clockwise planar rotation is
the left normal and its negative is the right normal; side labels therefore
rotate covariantly rather than changing under a global coordinate sign rule.

The regular sample lattice is exactly
`{k*sample_step | k is integer and abs(k*sample_step) <= half_length}`. It is
anchored at zero; a half-length endpoint is sampled only when divisible by the
step. A point uses the physical cell mean of its containing control volume.
Polygon containment includes the boundary; if several polygons contain an exact
boundary point, choose the coordinate-canonical cell center. Outside-domain
samples are retained as censored. No surface smoothing occurs at this stage.

Reconstruct the exact nested catchment of the station reach: a cell is inside
exactly when its exclusive owner is that reach or an upstream descendant of
that reach. If the station point is not inside, emit `AxisOutsideCatchment` and
no section. Otherwise traverse each normal ray no farther than the registered
half-length through control-volume face intersections in increasing positive
distance. Internal owner changes that remain inside the same nested catchment
are retained but do not select a flank. The first face leaving that nested
catchment is the only flank candidate; it must be a backed
`LateralBoundaryCandidate` incident to one inside and one outside cell. If the
domain boundary, a flow transition or ambiguous face geometry occurs first,
that side is censored with the exact reason. Thus an unrelated nearer seam or a
later sibling boundary cannot be selected. The axis height is the containing
cell mean at offset zero and boundary height is the symmetric mean of the two
cells adjacent to the selected face. Append the exact boundary intersection and
that reconstructed height proxy to the regularly spaced side samples; record it
as a proxy rather than an observed subcell elevation. If no nested-catchment
exit occurs within the half-length, emit
`NoCatchmentExitWithinSupport`.

Endpoint contact counts as an intersection. If the axis lies on a lateral
boundary, the section is `AxisOnBoundary` and no span is emitted. If a ray and
boundary are collinear over positive length, that side is `CollinearBoundary`
and censored; choosing either overlap endpoint would manufacture a flank.
Parallel disjoint segments do not intersect. These states precede ordinary
distance ties.

For each uncensored side record boundary relief, maximum sampled relief before
the boundary, boundary/maximum separation and the offset of the minimum sampled
physical elevation from the reach axis. `positive_boundary_relief` is a signed
mechanical predicate, not a valley label. For two positive sides, locate the
first outward crossing of

```text
axis_height + relative_height_fraction * side_boundary_relief
```

using linear interpolation between bracketing regular cell-mean samples, or
between the last regular cell-mean sample and the explicitly marked boundary
height proxy. Record which bracket type produced each crossing. The
`relative_relief_span_km` is the sum of the two crossing distances.
It is an operational cross-section statistic, not a valley boundary or width.
If either crossing is absent, relief is non-positive or the section is
censored, the span is unavailable with an explicit reason.

The longitudinal lattice is `s=0,4,8,... < length`, followed by the exact final
arclength when not already present. Elevation uses the containing-cell rule;
structural area and runoff use the donor cell of the receiver segment containing
the sample. An exact internal vertex belongs to the downstream segment; the
final portal or reach endpoint belongs to the preceding terminal segment.
Between consecutive samples, conditioning/flat/non-descent flags are the OR of
every receiver segment whose positive-length arclength interval intersects the
sample interval. Grade is the physical elevation difference divided by sample
arclength. It never substitutes filled elevation; zero-length intervals are
unavailable rather than repaired.

## Deferred O0b boundary

Cross-resolution/cross-surface correspondence and combined packet assembly are
not part of O0a. The initial all-in-one design was rejected before implementation
because it coupled unrelated relationship evidence to polygon union, line
buffering and uncalibrated identity-event language.

A later O0b preregistration may consume completed O0a packet cores. It must keep
the full geometric overlap table primary, use local rather than global
large-cell line uncertainty, account for nested hierarchy before assignment and
use mechanical labels such as `MutualBest`, `OneToManyBest` and
`NoPositiveOverlap`. It may not call microscopic overlap `Same`, `Born`,
`Retired`, `Split` or `Merge` without a separately justified acceptance rule.
Individual raw faces will not receive persistent cross-resolution identity.

## O0a records, ordering and hashes

`LandformRelationshipsV0` retains backed face roles, owner ancestry, bilateral
physical-descent probes, highland-boundary summaries, saddle-boundary
associations, reach cross-section probes, exact configuration, geometry
identity, predecessor input hashes and a distinct O0a evidence hash.

Object arrays order by their upstream stable IDs only after input hash
verification. New geometric records order by coordinate-canonical anchors,
owner keys and endpoints. Serialize using fixed-int little-endian bincode and
FNV-1a-64, excluding the result hash field itself. Reproduce each predecessor's
own serialization/canonicalization semantics when verifying its hash, then
preserve its payload bits exactly. Canonicalize negative zero in newly derived
O0a fields; NaN and infinity are typed errors.

Arm label, revision metadata and native provenance live in a caller envelope.
They are excluded from common hashes. A later O0b correspondence hash will be
separate and cannot alter this packet core.

## Manufactured gates

Before any H/C/G surface or product adapter is observed, pass:

1. **face-role fixture:** the exact receiver face handing an upstream reach to
   its parent is `FlowTransition`; a lateral face with the same ancestry-related
   owner pair and a sibling-owner face are `LateralBoundaryCandidate`; the two
   roles partition raw face length exactly;
2. **bilateral descent and flat control at 8/4/2 km:** a symmetric two-outlet
   surface has positive unconditioned physical descent on both sides of its
   central lateral boundary, while the identical graph at constant physical
   elevation has zero signed margins and no such descriptor;
3. **displaced/conditioned control:** a valid receiver partition whose physical
   maximum lies uphill on one side reports negative minimum segment drop,
   positive remote-maximum excess and conditioning flags; it cannot borrow that
   maximum to pass the bilateral descriptor. An adjacent cell already on its
   retained reach emits the frozen
   zero-segment/unavailable record;
4. **saddle fixture:** the linked highland surface retains nearest-boundary
   distance, height mismatch, face descriptors and equal-elder ambiguity, but
   emits no transfer-low or pass label; a nearby high/non-descending seam is an
   explicit negative semantic control;
5. **cross-section fixture:** an analytic reach-aligned quadratic trough has
   positive two-sided boundary relief, `80 km` relative-relief span and zero
   axis/minimum offset; monotone, shoulder, internal-ridge and truncated sides
   retain signed/censor evidence without a valley label;
6. **catchment-flank fixture:** a nearer unrelated seam and ancestry-related
   internal owner transition are ignored; each ray selects only its first exact
   exit from the station reach's nested catchment; the `50 km` half-length case
   is censored before the exits at `80 km`, and an off-grid `79 km` exit uses the
   disclosed cell-mean/proxy bracket;
7. **end-to-end geometry path:** the complete asymmetric-Y G0+D0 output passes
   raw-face multiset rederivation, unique backing, subdivision validation,
   indexed point location and nested-catchment ray traversal; indexed answers
   must equal an exhaustive polygon/face oracle on the same packet;
8. **rotation and reindexing:** tie-free relationship kernels preserve scalar
   evidence and rotate vectors. End-to-end predecessor packets are compared
   only when their coordinate-mapped S0/D0 topology is unchanged; otherwise
   covariance is censored because predecessor lexicographic ties are not
   rotation-covariant. Deterministic repeats of either indexing are identical;
9. **irregular geometry control:** the existing projected irregular Voronoi cap
   passes the O0a subdivision and unique raw-face-backing prerequisites using
   its stored face endpoints; no descent or saddle outcome is claimed; and
10. malformed foreign hashes, ambiguous boundary backing, noncanonical zero,
    overlapping/gapped control volumes, non-finite geometry/elevation, missing
    owner ancestry, inconsistent D0 targets, zero-length polylines and
    ledger/hash failures return typed errors without partial output.

### Frozen fixture annex

End-to-end planar fixtures use the existing
`LandscapeMesh::uniform_planar_hex_with_portals` lattice phase, exact regular
control volumes and spacings `8`, `4` and `2 km`. Local supply is
`0.1 * cell_area` unless a predecessor fixture already freezes another supply.

- The lineage case is the committed D0 asymmetric-Y fixture: a `128 x 96 km`
  domain, south portal 23 over `[-1,1] km`, outlet `(0,-48)`, junction `(0,0)`,
  heads `(-48,48)` and `(48,48)`, and the existing segment-network cost surface.
  Its face-role kernel additionally freezes three unit-length faces: receiver-
  crossed `f0` between owners `Reach(P)` and its upstream `Reach(U)`, lateral
  `f1` with the same owner pair, and lateral `f2` between incomparable
  `Reach(U)` and `Reach(V)`. Only `f0` is a flow transition.
- The symmetric-shed/flat pair uses a `256 x 192 km` domain with full-height
  west portal 1 and east portal 2, both at base `0`. The symmetric surface is
  `z(x,y)=0.25 + 0.005*(128-abs(x)) km`; its analytic crest is `x=0`. The flat
  control is exactly `z=1 km` with both portal bases translated to `1 km`.
- The displaced-crest kernel uses unit spacing and the ordered support
  `portal_L <- L <- a | b -> R -> portal_R` at x coordinates
  `[-3,-2,-0.5,0.5,2,3]`, with the candidate face at `x=0`, physical heights
  `[0,2,1,1,0.5,0]`, owners terminating at distinct portals, and receivers
  `a -> L -> portal_L` and `b -> R -> portal_R`. On the left, target drop is
  positive but minimum segment drop is `-1 km` and remote maximum excess is
  `1 km`; `a -> L` is declared fill-supported. This is an explicit
  relationship-kernel table, not a claim that the table is a natural surface.
- The saddle case reuses the exact linked four-cone G0/S0 fixture. Its explicit
  D0 owner table assigns A/B to incomparable sibling owners, C to B's downstream
  ancestor and D to an equal-elder ambiguity. The table freezes nearest-face
  distance and height difference only; no case has a transfer-low oracle. The
  unit kernel places a `1 km` saddle at `(0,0)`, an A/B boundary face from
  `(-1,0)` to `(1,0)` with reconstructed height `1 km`, and covering radius
  `1 km`. Its negative repeats the geometry with reconstructed height `2 km`
  and non-descending face traces. Equal-elder true/false is crossed independently.
- The cross-section kernel uses axis `P(s)=(s,0,0)`, `0 <= s <= 400 km`, nested
  catchment exits crossing every station at `y=-80` and `y=80 km`, and
  `z(s,y)=2-0.002*s+(y/80)^2 km`. Boundary relief is exactly `1 km`; at reference
  fraction `0.25` the crossings are `y=+-40 km` and relative-relief span is
  `80 km`. An internal same-catchment transition at `y=30 km` and unrelated seam
  at `y=20 km` must not replace the `y=80 km` exit. The monotone control replaces
  the lateral term with `0.005*y`; the shoulder moves the axis to `y=30 km`, the
  internal-ridge control negates the quadratic term, and the censored control
  removes the positive-y exit. Additional cases move both exits to `+-79 km`
  and run half-length `50 km`. These are explicit probe-kernel tables. The full
  asymmetric-Y packet additionally compares indexed containment/traversal with
  exhaustive polygon/face enumeration, including exact raw-boundary multiset
  backing.
- The irregular control is the existing
  `build_r1_voronoi_cap(VoronoiCapConfig::r1(8.0))` projected through
  `adapt_projected_voronoi_cap_graph_v0`; its exact stored face endpoints, not
  a regular-hex reconstruction, must pass subdivision and unique face backing.
  No O0a descent, saddle or cross-section semantic record is expected from this
  geometry-only control.

Rotate only the tie-free explicit cross-section and descent kernels by exactly
30 degrees for the rotation gate. Reindex fixtures with the fixed permutation
`new=(17*old+3) mod N` when `gcd(17,N)=1`; otherwise use the smallest odd
multiplier greater than 17 coprime to `N`. Compare reindexed outputs only after
coordinate-based remapping; do not assert byte or hash equality across distinct
serialized graphs. Expected answers above come from the analytic fields or
explicit tables, never from extractor output. Changing any dimension, phase,
portal, equation, owner table or expected answer is a contract amendment made
before O0a implementation continues.

## Cost and complexity gate

O0a may use deterministic spatial indices, but they cannot change answers.
Cross-section point location and face-intersection candidates must not scan the
full graph per sample. Precompute the target, signed extrema, minimum segment
drop, conditioning flags and physical length for every cell once in reverse D0
receiver order; do not retrace a full downstream path independently per face.
Index physical face and polygon bounding boxes once, then enumerate only boxes
intersecting each ray/sample. Public accumulation remains compensated and order
stable.

Report isolated O0a wall time and peak memory at 8/4/2 km, plus counts of faces,
receiver-trace segments, stations, samples and candidate face tests. There is no
hardware-specific wall-clock promotion threshold. Stop and review if work
growth is consistent with unindexed per-sample full-graph behavior or if O0a
evidence costs more than the terrain alternatives it is intended to judge.

## Stop and amendment rule

Stop after the O0a implementation, manufactured matrix and dated audit. Then
preregister O0b correspondence/packet assembly from observed O0a objects and
costs. Do not adapt product hydrology, run product R0, compose H/C/G, tune
terrain or add named ranges/passes/valleys, channel geometry, persistent
lineage, sediment or presentation selection in this checkpoint.

Amend before implementation continues if:

- a raw owner seam must be called a divide from ancestry or ownership alone;
- bilateral descent needs a remote maximum, filled elevation or a conditioned
  segment to pass;
- a cross-section statistic is needed as a named valley boundary or width;
- O0b correspondence or persistent event language leaks back into O0a;
- packet bytes or common hashes change with arm/provenance metadata; or
- a manufactured expected answer, tolerance or fixture must change after H/C/G
  or product output is inspected.

Passing O0a establishes only that the common evidence packet can state these
relationships reproducibly. It does not establish natural-kind terrain
semantics, topographic ridge identity, mountain passes, valley polygons,
persistent geographic identity, Earth calibration or a preferred landscape
architecture.
