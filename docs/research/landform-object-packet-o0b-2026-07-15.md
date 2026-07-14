# Landform object packet O0b correspondence and assembly executable contract

**Status:** preregistered; implementation and outcomes unknown

**Date:** 2026-07-15, after the bounded O0a audit and before O0b implementation

**Parent:** [Landform object packet v0](landform-object-packet-v0-2026-07-14.md)

**Predecessors:** [G0/S0 executable contract](landform-object-packet-g0s0-2026-07-14.md),
[D0 executable contract](landform-object-packet-d0-2026-07-15.md) and
[O0a executable contract](landform-object-packet-o0-2026-07-15.md)

**Observed basis:** [O0a relationship-probe audit](../audits/landform-o0a-relationships-2026-07-15.md)

## Decision question

Can completed common planar G0/S0, D0 and O0a evidence be assembled without
changing its meaning, then compared across a shared physical frame with a
deterministic geometric best-map that exposes one-to-one, one-to-many,
many-to-one, many-to-many, tie, null and metric-conflict outcomes without
claiming persistent geographic identity?

O0b is an evaluation-instrument rung. It is not a terrain model, landform
classifier, temporal tracker or product adapter. Passing it establishes only
that packet-local reference objects can be assembled and related mechanically.

## Why this rung is separate

O0a showed why raw object counts and nearest relationships cannot be treated as
identity. On the unchanged asymmetric-Y fixture, reference-highland count is
`2 / 1 / 1` at 8/4/2 km, saddle count is `14 / 27 / 52`, only two saddles at
each resolution lie within the local covering radius, and sampled two-sided
spans are non-monotone. Raw boundary-face count also grows strongly under
refinement. O0b must preserve those changes rather than normalize them into a
clean story.

The provisional umbrella used a single `confidence` field and a flat list of
best-map labels. This contract replaces that provisional scalar with raw
directional coverage, exact maximum sets, normalized best/second margins,
anchor displacement, explicit ambiguity and separate topology evidence. There
is no calibrated probability or weighted composite in O0b.

## Bounded scope

This checkpoint implements only:

1. a self-contained common planar/testbed packet core assembled from one exact
   G0 graph/input population and its completed S0, D0 and O0a outputs; and
2. a separate ordered-pair correspondence artifact between two compatible
   packet cores in the same registered Cartesian kilometre frame.

Only these reference object families receive correspondence:

- S0 reference highlands, through nested and exclusive physical area support;
- D0 reference-scale drainage nodes, identified by `reach_id`, through nested
  and exclusive catchment area; and
- the same drainage nodes through full D0 centre-receiver polylines and local
  cell-scale buffered proximity.

No persistent correspondence is assigned to raw S0 peaks below the reference
population, saddle nodes, depressions, cells, receiver segments, backed O0a
faces, highland-boundary aggregates, saddle associations, cross-section
stations/sides/samples/spans or sensitivity-population objects. Those records
remain inside their exact packet and may later be compared descriptively only
through each retained owning candidate/maximum set; a tie never becomes a
selected winner.

Individual raw boundary faces never receive cross-resolution identity. O0b
does not group them into ridges or divides.

Product/spherical packets, the projected-R1 geometry-only control, native G
graphs, forcing/provenance relationships and H/C/G output are outside this
checkpoint. The current O0a implementation is planar; O0b does not conceal
that missing product adapter.

## Forbidden inputs and leakage barrier

Common assembly and correspondence may consume only the completed common
physical inputs and predecessor evidence named below. They must not consume:

- `World`, `OrogenModel`, arm identity or native object IDs;
- G's authored graph membership or H/C native topology;
- tectonic masks, forcing fields, expected segment answers or target labels;
- product river importance, renderer state or presentation transforms;
- an observed per-arm maximum, fitted acceptance threshold or arm-specific
  normalization; or
- shared cell indices as geographic truth.

Caller revision, dirty state, arm label and native-provenance hash live only in
an envelope. Arbitrarily changing them must leave common packet bytes, packet
hashes, correspondence records and correspondence hashes unchanged.

## Common packet assembly

### Required live inputs

One `LandformObjectPacketCoreV0` assembly receives and retains:

- the exact validated planar `EvaluationSurfaceGraphV0`;
- physical cell-mean elevation, scored-cell mask and local runoff arrays;
- registered `SurfaceHierarchyConfigV0`, `DrainageConfigV0` and every registered
  `LandformRelationshipConfigV0`;
- the exact `SurfaceHierarchyV0` and `EvaluationDrainageV0`;
- exactly eleven `LandformRelationshipsV0` payloads: reference plus the ten
  registered one-factor namespaces, each exactly once;
- the registered `PacketGeometryIdentityV0`; and
- `CommonEvaluationPopulationV0`, the frozen coordinate frame, declared
  comparison domain/portal registry, scored policy and runoff policy.

O0b accepts no arbitrary population string. The exact v0 enums are:

```text
coordinate_frame = LandscapeTestbedCartesianXyKmV0
declared_domain   = RequestedRegularPatchV0 { width_km, height_km }
scored_policy     = WholeGraphSupportV0
runoff_policy     = ExactSameMeshArrayV0 { canonical_array_hash }
                 | UniformPerAreaV0 { rate }
                 | AsymmetricYAffinePerAreaV0 {
                       base_rate: 0.3,
                       x_gradient_per_km: 0.002
                   }
semantic_portals  = coordinate-ordered Vec<DeclaredPortalV0> {
                       id, side, span_start_km, span_end_km, base_level_km
                   }
```

`CommonEvaluationPopulationV0` stores the canonical hash of all these fields.
Assembly validates `UniformPerAreaV0` as `rate * cell_area_km2` and the
asymmetric policy in the exact operation order
`0.3 * (1.0 + 0.002 * center.x) * cell_area_km2`. Same-mesh comparison under
`ExactSameMeshArrayV0` additionally requires bit-identical runoff arrays.
Cross-resolution comparison permits only the two reconstructible formula
variants and validates every cell. A later H/C/G contract must preregister any
new policy enum before it can enter common correspondence. An arm label is not
a population policy.

`canonical_array_hash` is FNV-1a-64 over fixed-int little-endian bincode of the
canonicalized `Vec<f64>` runoff array, including its serialized vector length.
`population_definition_hash` uses the same encoding/hash over
`coordinate_frame`, `declared_domain`, `scored_policy`, `runoff_policy` and
`semantic_portals` in that field order, excluding the result field itself.

For v0, assembly rebuilds the regular `LandscapeMesh`, control volumes and G0
adapter from `RequestedRegularPatchV0`, nominal spacing and the declared portal
records, and requires the canonical graph hash to equal the supplied graph.
The declaration therefore cannot spoof a different domain or portal layout.

### Validation before assembly

Assembly must reproduce each predecessor's own canonical serialization and
hash, without changing predecessor schemas or bytes. It verifies:

- graph, array lengths, finite values and positive-zero input conventions;
- exact G0/S0 and D0 evidence hashes;
- every O0a predecessor-input hash and O0a evidence hash;
- one reference and all ten distinct sensitivity namespaces, with no duplicate
  or missing namespace;
- identical graph/elevation/scored/runoff/config inputs across those O0a
  payloads;
- the exact registered regular-planar geometry identity and graph hash; and
- all predecessor object IDs, references, ledgers and canonical ordering.

The assembly cannot repair, reorder or recompute a predecessor payload into a
different valid-looking payload. Failure returns a typed error and no partial
packet.

The persistent artifact must round-trip from owned bytes. O0a's current
borrowed version strings and missing public deserializer are implementation
constraints, not permission to change its canonical hash. Use a validated wire
representation or custom decoder that reconstructs the already frozen values
bit-for-bit.

### Core and envelope

The core records:

```text
schema/hash version
common evaluation population
geometry identity and canonical graph hash
exact graph, physical elevation, scored mask and runoff
registered configs
exact S0, D0 and eleven O0a payloads
separate predecessor evidence hashes
derived common-packet hash
```

`LandformPacketEnvelopeV0` may additionally record run ID, revision, dirty
state, arm label and native-provenance hash. The envelope is outside the common
packet hash. The core is immutable after successful construction.

The packet hash uses fixed-int little-endian bincode followed by FNV-1a-64,
excluding only its own result field. Negative zero in newly introduced fields
is canonicalized to positive zero; non-finite derived evidence is invalid.

## Compatible correspondence pairs

`ObjectCorrespondenceV0` is a separate artifact over an ordered
`(source_packet_hash, target_packet_hash)` pair. It never changes either core.

The exact new namespaces are:

```text
packet schema          = landform-object-packet-o0b-v0
correspondence schema  = landform-correspondence-o0b-v0
hash                   = fnv1a64-bincode-fixint-le-v0
```

`CorrespondenceConfigV0` has no numerical knob. Its only accepted values name
the frozen policies in this contract: `ExclusiveIntersectionAreaV0`,
`LocalCoveringRadiusSumV0`, `ExactMaximumSetV0` and
`ReportOnlyTopologyV0`, plus the schema/hash strings above. Any other value is
an unregistered configuration. Same-mesh versus cross-mesh comparison is
derived from the two graph hashes, not selected by the caller.

Both cores must have:

- the O0b schema and accepted predecessor schema/hash versions;
- `EvaluationDomainV0::Planar` and regular-planar geometry identities;
- exact-equal validated common-evaluation population values/hashes; and
- reference S0, 2,000 km2 D0 and reference O0a populations available.

For the bounded cross-resolution path, nominal spacing must be one of 8, 4 or
2 km and every cell must be scored. Same-mesh cross-surface comparison requires
an identical canonical graph hash and scored mask. Cross-mesh comparison uses
physical geometry only.

Cross-resolution regular patches intentionally have different sawtooth unions
of full control volumes. Raw G0 outer boundary chains and total boundary length
therefore are not required to match. The declared requested patch and semantic
portal registry must match. Each graph must independently validate that its
open boundary segments have the declared portal ID/base level and physical-face
backing. Sort and union their stored `projected_span_km` intervals; that union,
not the sawtooth physical segments, must realize the declared side/span within
`endpoint_match_abs_km`.

Area comparison occurs only on the actual intersection of the two G0 physical
supports. In the same artifact retain, in both directions, support area outside
the opposite graph as `OutsideTargetDomain` or `OutsideSourceDomain` context.
Neither is `NoPositiveOverlap` unless no positive object candidate also exists.
Different sawtooth support is representation evidence, not a correspondence
failure or an event.

Portal IDs are compatibility/context keys supplied by the common testbed, not
objects discovered by correspondence.

## Area-support construction

### Highland supports

For reference highland `h`, nested support `F(h)` is the exact union of the G0
cell polygons named by its S0 `footprint_members`.

Reference highland footprints are nested or disjoint. Mechanical assignment
uses exclusive reference support `E(h)`: assign each member cell to the deepest
retained reference highland whose nested footprint contains it. Equivalently,
subtract every proper retained-descendant footprint from `F(h)`. Validate that
these exclusive supports do not overlap and that their union equals the union
of reference nested support. A reference highland with zero exclusive area is
retained in the nested table with `NoExclusiveSupport`; it does not enter the
exclusive best graph.

This prevents an ancestor's deliberately nested footprint from manufacturing
a false one-to-many pattern.

Equal-elder resolution can change packet-local ancestry and therefore support.
Mark highland `h` as `HierarchyAmbiguousSupport` when `h`, its nearest retained
ancestor path, or any retained-descendant path subtracted from `F(h)` traverses
a `PeakBranchV0` or key `SaddleNodeV0` with `equal_elder_ambiguous`. Retain the
frozen S0 nested geometry and its nested overlaps, but emit no exclusive pair
row involving a marked highland and exclude it from exclusive assignment,
margins, components and topology credit. Eligible support overlapping a marked
opposite highland is retained as
`IneligibleHighlandSupport { peak_id, status, area_km2 }` context. Do not assert
that an alternate valid elder choice has the same support.

This context intersects eligible support with each marked highland's frozen
packet-local exclusive `E(h)`, not its nested footprint. Those frozen exclusive
supports remain disjoint inside one resolved packet even though another valid
elder resolution may change them. Attribute every contribution by marked
`peak_id`; a marked object with zero `E(h)` emits no positive context entry.

### Drainage-node catchment supports

Use only the D0 scale whose support threshold is exactly 2,000 km2. A reach's
exclusive catchment support is the union of cells whose
`exclusive_owner == Reach(reach_id)`. Its nested support additionally includes
every exclusive reach owner whose downstream-parent chain reaches that reach.
Portal-owned cells remain disjoint context, grouped by semantic portal ID.

O0b support area is canonical polygon/self-clip area, while predecessor ledgers
sum stored `cell_area_km2`. Validate the difference with a composed bound equal
to the G0 per-cell polygon-area allowance accumulated in coordinate order plus
the D0 compensated-summation balance allowance. Do not apply the stricter D0
relative tolerance alone and do not substitute stored cell area for polygon
area. Nested and exclusive tables are both retained; only exclusive support
controls the drainage-node area channel.

### Exact planar overlap

For source support `A` and target support `B`, compute each positive cell-pair
intersection once using their stored physical control-volume polygons. Use a
deterministic bounding-box index; do not scan every opposite cell for every
object.

Cell polygons are convex and counter-clockwise. Weld each polygon, rotate its
vertex sequence so the numerical-lexicographic minimum `(x,y,z)` is first and
compare complete sequences by length then `f64::total_cmp` coordinates. Clip
the lesser sequence first by the other with Sutherland-Hodgman half-planes. A signed cross
product whose magnitude is at most
`endpoint_match_abs_km * clip_edge_length` is on the boundary. Weld consecutive
result vertices within `endpoint_match_abs_km`, remove a duplicate closing
vertex and evaluate shoelace area and centroid in canonical vertex order with
compensated sums. The polygon-pair coordinate order, not source/target role,
controls clipping so source/target reversal uses identical intersection bits.

Sort every positive cell-pair contribution by those two canonical polygon
sequences before accumulating object-pair area and first moments. Use the same
compensated order under source/target reversal. Define
`area_tolerance = endpoint_match_abs_km^2 +
planar_area_match_relative * max(As,At)`. If accumulated `I` exceeds
`min(As,At)` by more than this tolerance, fail; if the excess is within the
tolerance, canonicalize `I` to `min(As,At)`. Coverage greater than one after
this rule is invalid.

Zero-area contact is not a positive pair. Every finite area greater than zero
after the frozen clipping/welding procedure is retained; O0b has no minimum
overlap threshold.

For each positive nested or exclusive pair retain:

```text
intersection_area_km2 = I
source_area_km2       = As
target_area_km2       = At
union_area_km2        = As + At - I
source_coverage       = I / As
target_coverage       = I / At
jaccard                = I / (As + At - I)
dice                   = 2 I / (As + At)
source/target support-centroid displacement
```

Nested rows use centroids of the full nested supports; exclusive rows use
centroids of exclusive supports. Compute both from the same compensated polygon
area/first moments, not stored cell centres. `NoExclusiveSupport` and
`HierarchyAmbiguousSupport` have no exclusive centroid, margin, null or best
component.

All areas are positive for an emitted pair. The logical nested table includes
every source-target pair. The logical exclusive table includes every
eligible-source/eligible-target pair. Serialization stores their positive rows
in source-object, target-object order, while an omitted eligible row means exact
zero. A pair involving `NoExclusiveSupport` or
`HierarchyAmbiguousSupport` is ineligible, not a zero row; positive overlap with
a marked highland is retained only in `IneligibleHighlandSupport` context as
specified above. Per-object `NoPositiveOverlap` and context coverage make the
sparse eligible representation complete.

For every exclusive source object retain area falling into target highland
background, target portal context and outside-target-domain context. Retain the
exact target-to-source counterparts in the same artifact. Context never enters
the best graph and is never called absence, birth or retirement.

Highland context partitions eligible-highland, ineligible-highland,
highland-background and outside-domain area. Drainage-node context separately
partitions eligible drainage-node, portal-owned and outside-domain area. These
two family partitions are never added together as if they were one support.

## Reach-line construction and proximity

Line correspondence uses the full D0 reference reach, not O0a's optionally
source-truncated 400 km probe.

For reach cells in source-to-downstream order, use their G0 centres and append
the exact receiver target of the tail cell: the receiver-cell centre or the
terminal portal-segment midpoint. Each resulting segment is the D0 receiver
segment of its donor cell. Geometry uses those endpoints, but its authoritative
measure length is `drainage.routing.segment_length_km[donor_cell]`, not
recomputed Euclidean centre distance. Accumulate those measures in donor order
with the D0 compensated rule and require bit-identical reproduction of the
stored reach length.

For a cell, define covering radius as the maximum Euclidean distance from its
stored centre to any stored polygon vertex. An internal receiver segment's
local radius is the maximum covering radius of its donor and receiver cells; a
terminal portal segment uses its donor-cell radius. For source segment `i` and
target segment `j`, the allowed proximity radius is exactly

```text
R_ij = local_radius_source_i + local_radius_target_j
```

There is no global maximum-cell buffer and no fitted physical width. This is
mesh-local geometric proximity support, not a calibrated uncertainty bound,
channel, bank or valley width.

For source segment `p(t)=a+t u`, `t in [0,1]`, and target segment
`q(s)=c+s v`, the covered source parameters satisfy

```text
min_(s in [0,1]) |p(t)-q(s)|^2 <= R_ij^2.
```

Compute the interval analytically. Split `[0,1]` where the unclamped projection
`((p(t)-c) dot v)/(v dot v)` equals 0 or 1. On each remaining interval the
closest target parameter is 0, affine or 1, so squared distance is one
quadratic `A t^2 + B t + C`. Intersect its `<= R_ij^2` solution with that
parameter interval. Reject degenerate/non-finite segments. A tangent root has
zero covered length and is not positive evidence.

Canonicalize each segment key as its numerical-lexicographic endpoint pair,
then measure length and local radius. Order each cross-packet segment pair by
the two complete keys, independent of source/target role, and solve both
directional interval sets once in that call. For quadratic
`A t^2 + B t + (C-R_ij^2) <= 0`, use these exact branches:

- `A == 0, B == 0`: the whole subinterval passes iff `C <= R_ij^2`;
- `A == 0, B != 0`: use the single root `-(C-R_ij^2)/B`;
- `A > 0, D < 0`: empty;
- `A > 0, D == 0`: the zero-length tangent only; and
- `A > 0, D > 0`: compute
  `q=-0.5*(B+copysign(sqrt(D),B))`, roots `q/A` and
  `(C-R_ij^2)/q`, falling back to `-B/(2*A)` only when `q==0`, then sort.

Any `A < 0` or non-finite coefficient is a typed numerical failure. No
discriminant epsilon is introduced.

For each source segment, union its coordinate-sorted covered intervals over all
indexed target-segment candidates. Interval `[t0,t1]` contributes
`(t1-t0) * measure_length_km`; repeat in the target direction. A positive row
requires both directional covered lengths to be strictly positive. A one-sided
result is a typed numerical inconsistency. Retain:

```text
source_covered_length_km
target_covered_length_km
source_coverage_fraction
target_coverage_fraction
source/target total physical length
half-arclength anchor displacement_km
minimum_positive_candidate_separation_km
```

Total length and each half-arclength anchor use authoritative D0 segment
measure, interpolating geometrically within the segment at the corresponding
measure fraction. The minimum-separation field ranges only over segment pairs
actually tested for this positive row; it is not a global Hausdorff or nearest-
line distance.

Sort interval contributions by the role-independent complete segment-pair key
before the compensated reach-pair sums. Reversing packets reuses the same
pairwise interval bits and accumulation order, then swaps the two directional
fields.

Segment candidates use bounding boxes expanded by their own local radii. Count
every exact segment-pair test. Parallel lines inside the local radius are only
buffered-proximity evidence.

## Mechanical best-map assignment

Area assignment uses exclusive intersection area. For a source drainage node's
line channel it uses source covered length; for the target direction it uses
target covered length. Nested overlap, Jaccard, Dice, anchor displacement,
topology and O0a measurements never change the maximum set.

For each object and channel:

1. retain every partner with positive primary score;
2. retain the full set whose score is exactly the maximum under `f64::total_cmp`;
3. never choose one member of an exact maximum set by coordinates or IDs;
4. record `ExactBestTie` when that set has more than one member; and
5. record normalized margin `(best - second_distinct) / object_measure`, using
   zero for an exact tie and `second_distinct = 0` when only one positive score
   exists.

`NoPositiveOverlap`, `NoExclusiveSupport` and
`HierarchyAmbiguousSupport` have no best score or margin; unavailable is never
serialized as numerical zero.

Near-ties remain visible as small margins. O0b does not invent an epsilon or
probability. A same-graph optimization may cache the canonical polygon
self-clip result; it may not substitute stored `cell_area_km2`. Shared cell IDs
cannot add candidates or break ties, and the cached path must be bit-identical
to general clipping.

Form a channel-qualified undirected best graph containing an edge when either
endpoint includes the other in its directed maximum set. Every member is
tagged `(Source|Target, Highland|DrainageNode, object_id)`. Connected
components receive only their mechanical cardinality:

- `OneToOneBest`;
- `OneToManyBest`;
- `ManyToOneBest`; or
- `ManyToManyBest`.

Objects with no positive partner receive `NoPositiveOverlap` on their own side.
`ExactBestTie` is an orthogonal flag, not a selected winner. A microscopic
positive pair can participate in a best component when nothing else overlaps;
that still does not mean `Same`.

Highlands have one exclusive-area best graph. Drainage nodes have two
independent, explicitly channel-qualified graphs: exclusive-catchment area and
buffered-line proximity. For each source and target drainage node, project both
maximum sets to partner `reach_id`; unequal sets record a side-specific
`MetricConflict`. Reversal swaps the side and preserves the conflict. Do not
expose an unqualified drainage-node component, and do not weight, vote or use
topology to select one.

The labels `Same`, `Born`, `Retired`, `Split`, `Merge`, `Captured` and
`Persistent` are forbidden. O0b cardinality describes a best graph at one
ordered comparison, not an event or natural identity.

## Topology evidence after assignment

Topology is report-only. It cannot veto, repair or rerank geometry.

For reference highlands, induce a retained hierarchy by walking each retained
peak's S0 parent chain to the nearest retained ancestor or root. A traversed
edge is unavailable when any peak or key saddle on that path is
`equal_elder_ambiguous`; affected supports are already excluded above.

For drainage nodes, use the exact D0 downstream-reach or terminal-portal edge.
Evaluate catchment-area and line best graphs separately.

`TopologyTargetV0` is `Highland(id)`, `DrainageNode(id)`, `Portal(id)` or
`HighlandRoot`.
For an object-to-object source edge, take the Cartesian product of the two
endpoint maximum sets. For an edge to a terminal portal or highland root, take
only the source object's maximum set and compare each mapped target object's
declared topology target with the exact same semantic portal ID or root.
Portals/roots have no object maximum set.

Record `TopologyAvailabilityV0::HierarchyAmbiguous` for an affected highland
edge and `NoMappedEndpoint` when any required object endpoint lacks a maximum
set; in either case `mapped_adjacency` is `None`. Otherwise availability is
`Available` and record:

- `MappedAdjacency::All`, `Some` or `None` according to whether all, a strict
  nonempty subset or none of those mapped endpoint pairs have the target edge
  or terminal portal.

For an available object-to-object edge, retain the separate
`endpoints_in_same_best_component: Some(bool)`; it is not a contraction event.
Portal/root and unavailable records use `None`. Retain raw edge records and
unweighted category counts.
No topology outcome changes a correspondence component. A source/target
reversal reports the target topology in the same way rather than
pretending the two directed graphs are identical.

The terminal oracle's compatible population declares only portal 41. It maps
one source drainage node and one target drainage node uniquely, with both
terminating there: availability is `Available` and adjacency is `All` in both
directions. Changing a D0 terminal to undeclared portal 42 fails packet
validation before correspondence. In a separate compatible oracle whose common
population declares both portals 41 and 42, map source and target nodes
uniquely but terminate them at 41 and 42 respectively; availability is
`Available` and adjacency is `None`. A portal ID is never geometrically
reassigned.

## Correspondence records, ordering and hash

The artifact retains:

- exact ordered source/target packet hashes and frozen O0b config;
- positive nested and exclusive highland area pairs plus two-way background/
  domain context;
- positive nested and exclusive drainage-node catchment pairs plus two-way
  portal/domain context;
- positive drainage-node line-proximity pairs;
- directed maximum sets, margins, nulls and best-graph components for every
  assignment channel;
- side-specific drainage-node `MetricConflict`, exact-tie and hierarchy-
  ambiguity flags;
- report-only topology edge evidence;
- work counts for polygon candidates/clips, membership contributions, segment
  candidates/tests, positive rows and best-graph edges; and
- a separate derived correspondence hash.

Object references are `(family, packet-local id)`. Sort families as highland,
drainage node; within a table sort source ID then target ID. Sort maximum
sets by partner ID without selecting a winner. Sort components by the lowest
`(anchor.x, anchor.y, anchor.z, side, id)` member, then their complete
side-tagged member lists. Highland and drainage-area anchors are their eligible
exclusive-support polygon centroids; drainage-line anchors are D0-measure
half-arclength points. Newly derived coordinates use numerical `total_cmp`; raw
bits never define physical order.

Highland object ID is its reference `peak_id`. Drainage-node object ID is the
reference-scale `reach_id`; its catchment and line channels are a within-packet
bijection, not separate object families. No ID from the 1,000 or 4,000 km2 D0
scales is admissible.

### Frozen wire order

All enum discriminants are their zero-based order as listed here:

```text
PacketSideV0       = Source | Target
ObjectFamilyV0     = Highland | DrainageNode
AreaSupportV0      = Nested | Exclusive
CoordinateFrameV0  = LandscapeTestbedCartesianXyKmV0
DeclaredDomainV0   = RequestedRegularPatchV0 {
                        width_km: f64, height_km: f64
                      }
ScoredPolicyV0     = WholeGraphSupportV0
RunoffPolicyV0     = ExactSameMeshArrayV0 { canonical_array_hash: u64 }
                    | UniformPerAreaV0 { rate: f64 }
                    | AsymmetricYAffinePerAreaV0 {
                        base_rate: f64, x_gradient_per_km: f64
                      }
DeclaredPortalSideV0 = North | East | South | West
AreaAssignmentPolicyV0 = ExclusiveIntersectionAreaV0
LineSupportPolicyV0 = LocalCoveringRadiusSumV0
MaximumPolicyV0    = ExactMaximumSetV0
TopologyPolicyV0   = ReportOnlyTopologyV0
AssignmentChannelV0 = HighlandExclusiveArea
                    | DrainageExclusiveArea
                    | DrainageLine
ComponentKindV0    = OneToOneBest | OneToManyBest
                    | ManyToOneBest | ManyToManyBest
SupportStatusV0    = Eligible | NoExclusiveSupport
                    | HierarchyAmbiguousSupport | NoPositiveOverlap
MappedAdjacencyV0  = All | Some | None
TopologyAvailabilityV0 = Available | HierarchyAmbiguous | NoMappedEndpoint
TopologyTargetV0   = Highland(u32) | DrainageNode(u32) | Portal(u32)
                    | HighlandRoot
DomainContextV0    = HighlandBackground | IneligibleHighlandSupport
                    | Portal | OutsideSourceDomain | OutsideTargetDomain
```

Wire structs serialize fields in this exact order:

```text
LandformObjectPacketCoreWireV0 {
  schema_version, hash_version, population, geometry_identity,
  graph, physical_elevation_km, scored_cell, local_runoff_supply,
  surface_config, drainage_config, relationship_configs,
  surface_hierarchy, drainage, relationship_payloads,
  surface_hierarchy_input_hash, drainage_input_hash,
  predecessor_evidence_hashes, derived_common_packet_hash
}

CommonEvaluationPopulationWireV0 {
  coordinate_frame, declared_domain, scored_policy, runoff_policy,
  semantic_portals, population_definition_hash
}
RequestedRegularPatchWireV0 { width_km, height_km }
DeclaredPortalWireV0 {
  id, side, span_start_km, span_end_km, base_level_km
}
CorrespondenceConfigWireV0 {
  area_policy, line_policy, maximum_policy, topology_policy,
  schema_version, hash_version
}
PredecessorEvidenceHashesWireV0 {
  surface_hierarchy_hash, drainage_hash, relationship_hashes
}
RelationshipEvidenceHashWireV0 { run_namespace, evidence_hash }

ObjectCorrespondenceWireV0 {
  schema_version, hash_version, config,
  source_packet_hash, target_packet_hash,
  highland_nested_pairs, highland_exclusive_pairs,
  drainage_nested_pairs, drainage_exclusive_pairs,
  drainage_line_pairs, context_records,
  assignment_records, best_components, metric_conflicts,
  topology_records, work_counts, derived_correspondence_hash
}

AreaPairWireV0 {
  source_id, target_id, support_kind,
  intersection_area_km2, source_area_km2, target_area_km2,
  union_area_km2, source_coverage, target_coverage, jaccard, dice,
  source_centroid_km, target_centroid_km, centroid_displacement_km
}

LinePairWireV0 {
  source_id, target_id, source_covered_length_km,
  target_covered_length_km, source_coverage, target_coverage,
  source_length_km, target_length_km, source_anchor_km,
  target_anchor_km, anchor_displacement_km,
  minimum_positive_candidate_separation_km
}

AssignmentWireV0 {
  side, family, object_id, channel, support_status,
  positive_partner_ids, maximum_partner_ids,
  best_score, second_distinct_score, normalized_margin, exact_best_tie
}

BestComponentWireV0 { channel, kind, members }
BestMemberWireV0    { side, family, object_id }
MetricConflictWireV0 { side, drainage_node_id,
                       area_maximum_ids, line_maximum_ids }
ContextWireV0       { side, family, object_id,
                      background_area_km2, ineligible_highland_areas,
                      portal_areas_km2,
                      outside_domain_area_km2 }
IneligibleHighlandAreaWireV0 { peak_id, support_status, area_km2 }
PortalAreaWireV0    { portal_id, area_km2 }
TopologyWireV0      { side, family, channel, from_id, target,
                      availability,
                      mapped_adjacency: Option<MappedAdjacencyV0>,
                      endpoints_in_same_best_component: Option<bool> }
TopologyTargetWireV0 = Highland { id: u32 }
                     | DrainageNode { id: u32 }
                     | Portal { id: u32 }
                     | HighlandRoot {}
WorkCountsWireV0   {
  source_cells, target_cells, cell_box_candidates, polygon_clips,
  positive_cell_intersections, nested_membership_contributions,
  source_segments, target_segments, segment_box_candidates,
  segment_pair_tests, positive_highland_nested_rows,
  positive_highland_exclusive_rows, positive_drainage_nested_rows,
  positive_drainage_exclusive_rows, positive_line_rows,
  best_graph_edges
}
```

Unavailable optional numerical fields serialize as `None`, never a sentinel.
Every `Option` uses `None | Some` discriminants 0/1. Portal vectors sort by
`(id, side, span_start, span_end, base_level)`; relationship hashes sort by
`RelationshipRunNamespaceV0`; portal/ineligible context vectors sort by ID.
Vectors otherwise use the canonical orders already frozen above. The O0a owned wire
mirror uses exactly the current `LandformRelationshipsV0` field order and the
source-declaration order of every existing enum. Its `schema_version` and
`hash_version` decode through owned strings, validate against the frozen
constants, and reserialize to the identical existing O0a bytes. O0b may expose
existing hash helpers but may not change predecessor field/discriminant order
or canonical serialization.

Serialize with fixed-int little-endian bincode and FNV-1a-64, excluding the
result hash itself. Repeated ordered input must be byte-identical. Reversing the
ordered packet pair normally changes bytes/hash, but must swap directional
coverages and `OneToManyBest`/`ManyToOneBest` while preserving positive-pair
geometry, tie, conflict and null facts.

## Frozen manufactured gates

All formulas, shapes and answers below are frozen before O0b code and before
any product or H/C/G output is inspected.

### 1. Assembly and hash

Assemble the existing 4 km asymmetric-Y G0/S0/D0/O0a reference plus all ten
O0a sensitivities.

- deterministic repeats are byte-identical;
- all eleven namespaces occur exactly once in enum order;
- reordered caller enumeration canonicalizes to the same core;
- one-bit input mutation, foreign predecessor, missing/duplicate namespace and
  tampered hash return typed errors;
- serialization round-trips to the identical core bytes/hash; and
- arbitrary H/C/G labels, revision strings and native-provenance payloads in
  the envelope do not change common bytes/hash.

### 2. Exact area arithmetic

Use analytic rectangles in square kilometres:

```text
A = [0,100] x [0,50]
B = [25,125] x [0,50]
```

`A` against itself has `I=5000`, both coverages, Jaccard and Dice equal to 1.
`A` against `B` has `I=3750`, union `6250`, both coverages and Dice `0.75`, and
Jaccard `0.60`. Edge-only contact and disjoint rectangles emit no positive row.
`A` against `[99.999,199.999] x [0,50]` retains the positive `0.05 km2` sliver
and source coverage `0.00001`; it acquires no identity/event label.

Exercise the production cell-pair path rather than clipping object rectangles
directly. Represent source `A` by vertical cells `[0,50]` and `[50,100]` and an
identical target by horizontal cells `[0,100] x [0,25]` and
`[0,100] x [25,50]`; the four cell intersections are exactly `1250 km2` and
sum to `5000`. Represent `B` by vertical cells `[25,75]` and `[75,125]`; the
three positive source/target cell contributions are exactly `1250 km2` and sum
to `3750`. Enumeration and tiling differences cannot double-count support.

### 3. Best-graph cardinality and ties

Let source `A=[0,100] x [0,50]`.

- targets `[0,60] x [0,50]` and `[60,100] x [0,50]` have intersections
  `3000` and `2000`; output is `OneToManyBest` without a source tie;
- reversing the ordered pair produces `ManyToOneBest`;
- equal targets `[0,50] x [0,50]` and `[50,100] x [0,50]` produce an exact
  two-member maximum set and `ExactBestTie`, never a selected winner; and
- a disjoint source/target pair produces source and target
  `NoPositiveOverlap`.

In common domain `[0,40] x [0,10]`, source object `[0,10] x [0,10]` and target
object `[20,30] x [0,10]` have no positive pair. Retain exactly `100 km2` of
source-to-target-background and `100 km2` of target-to-source-background
context; reversal swaps those records. Repeat with the opposite support owned
by declared portal 7 for drainage-node portal context.

The many-to-many fixture uses source vertical partitions at `x=60` and target
horizontal partitions at `y=60` over `[0,100] x [0,100]`. Its intersection
table is `[[3600,2400],[2400,1600]]`. The union of directed maximum sets is one
two-by-two `ManyToManyBest` component without an exact tie.

### 4. Nested hierarchy

Use one parent footprint `[0,100] x [0,100]`, a left child
`[0,40] x [0,100]` and right child `[60,100] x [0,100]`. With identical source
and target hierarchies, the nested table contains the expected parent/child
cross-level positives, while exclusive supports produce exactly three
`OneToOneBest` components with areas `2000`, `4000` and `4000`.

When the target retains only the parent, the three source exclusive objects
form `ManyToOneBest`; the target parent has an exact two-member maximum set over
the equal-area source children and records `ExactBestTie`. No object is called
merged or retired. Replacing the two children by `[0,50]` and `[50,100]` gives
the parent zero exclusive support and must emit `NoExclusiveSupport`. Marking
that object does not also create `NoPositiveOverlap`, a numerical margin or a
component.

For equal-elder ambiguity, construct both valid elder resolutions of an exact-
height two-branch hierarchy so their packet-local parent/descendant supports
differ. Every reference highland whose retained-ancestor or retained-descendant
subtraction path traverses that ambiguity must record
`HierarchyAmbiguousSupport` and remain outside exclusive assignment in both
resolutions. Frozen nested geometry may differ and remains reportable; neither
elder choice emits an exclusive pair row or receives topology credit. Overlap
from an eligible opposite support appears in `IneligibleHighlandSupport`
context in both directions.

Run the same support arithmetic once as highlands and once as drainage-node
catchments, with both directions of portal/background context retained.

### 5. Line proximity

Use non-degenerate line segments with local radius `0.5 km` on both sides, so
the pair radius is exactly `1 km`.

- `[0,100] x {0}` against the identical line, and against the same line split
  at 40 km, has both directional coverages equal to 1;
- against `[25,75] x {0}`, endpoint caps cover source interval `[24,76]`, so
  source coverage is `0.52` and target coverage is 1;
- targets `[0,60] x {0}` and `[60,100] x {0}` cover respectively 61 and 41 km
  of the source and all of each target; they produce `OneToManyBest`, with the
  reverse producing `ManyToOneBest`;
- a parallel line at `y=1.001 km` is outside support and has no positive pair;
- a parallel line at `y=0.75 km` has full buffered proximity but no identity
  label; and
- equal candidates at `y=+0.5` and `y=-0.5 km` retain an exact maximum set.

The local-radius control uses source line `[0,100] x {1.5}` at radius `0.5`,
ordinary target `T0=[0,100] x {0}` at radius `0.5`, and far target
`T1=[1000,1010] x {0}` at radius `100`. Neither pair is positive: `T0` is
outside its local sum radius 1 and `T1` is farther than its pair radius 100.5.
A forbidden global-maximum buffer would falsely match `T0`; the registered
pair-local rule cannot.

### 6. Metric and topology conflict

Use source reach `S0` with line `[0,100] x {0}` and exclusive catchment
`[0,100] x [0,20]`. Target `T0` has the identical line and exclusive catchment
`[0,40] x [0,20]`; target `T1` has line `[0,100] x {10}` and exclusive
catchment `[40,100] x [0,20]`. Every segment has local radius `0.5 km`.

The source line's unique maximum is `T0`, while exclusive catchment
intersections are `800` and `1200 km2`, making `T1` the unique area maximum.
Both tables and maximum sets remain unchanged. Source `S0` records
`MetricConflict`; target `T1` also records conflict because its area maximum is
`S0` while its line maximum is unavailable. Reversing source and target swaps
and preserves both side-specific conflicts.

For a separate two-edge topology kernel, source reaches have edge `S0 -> S1`,
the unique geometry map is `S0 -> T0`, `S1 -> T1`, and the target edge is
`T1 -> T0`. The source edge records `MappedAdjacency::None` without changing
either assignment. In a tied control, `S0` maps to `{T0,T1}`, `S1` maps to
`{T2}`, and only target edge `T0 -> T2` exists; the source edge records
`MappedAdjacency::Some`, never all-preserved credit.

### 7. Invariance

- isolated area/line/assignment kernels are invariant to polygon, cell, segment
  and object input enumeration after their frozen canonicalization;
- in those isolated kernels, the fixed permutation `new=(17*old+3) mod N`, or
  the smallest odd multiplier above 17 coprime to `N`, rebuilds every
  CSR/member/owner reference and preserves coordinate-remapped evidence;
- also in the isolated kernels, translating by `(17,-23) km` and then rotating
  both tie-free geometries by exactly 30 degrees preserves structural records
  and assignments; every best/second separation must exceed the applicable
  numerical error bound before this gate runs, and after inverse transformation
  coordinate/length differences are at most `endpoint_match_abs_km` while area
  relative differences are at most `planar_area_match_relative`;
- same-mesh shared-cell acceleration is bit-identical to general clipping;
- deterministic repeats are byte-identical; and
- source/target swap obeys the reversal rules for directional coverage,
  margins, contexts, side-specific nulls/conflicts and cardinality.

Only deterministic repeats of identical ordered packet inputs require byte
identity. Rigid-transform and reindexing controls do not pass through packet
assembly, whose registered graph rebuild intentionally rejects transformed or
noncanonical graph bytes; they test correspondence geometry/assignment only.
Exact-tie fixtures are tested only under identical or exactly representable
symmetry bytes, never through floating 30-degree covariance.

### 8. Registered 8/4/2 resolution evidence

Complete the already frozen analytic linked-four-cone surface into a packet
fixture without changing its cells or surface formula. At 8/4/2 km use the
existing `1120 km x 480*sqrt(3) km` regular patch, add declared south portal 41
with span `[-16,16] km` and base level 0, score every cell, canonicalize sampled
zero elevation to positive zero, and supply runoff `0.1 * cell_area_km2`.
Build G0/S0, D0 and all eleven O0a namespaces. Portal segmentation may change
boundary records but the frozen central four-cone S0 analytic topology and
retention answers must still hold.

Use the existing cone labels only inside manufactured assertions. Each 4 km
reference highland must have the unique exclusive-area maximum with the same
analytic label at 8 and 2 km, and the reverse directional maxima must agree.
Report every raw nested/exclusive coverage, displacement, margin and topology
delta; do not require monotone metric convergence.

For the D0 asymmetric-Y fixture, label the terminal reach as trunk and the two
upstream reaches by the sign of their frozen analytic head position. Each 4 km
trunk/west/east reach must have the unique same-label maximum at 8 and 2 km in
both exclusive-catchment and line channels, with no `MetricConflict`. Report
raw coverage, displacement, margins and topology deltas. O0a station censors
remain packet-local and are not corresponded.

The asymmetric-Y highland population itself remains an observation-only stress
case because O0a already measured `2 / 1 / 1` reference highlands. Report its
full mechanical graph without adding an expected identity/event answer or
changing the extractor.

These gates validate correspondence arithmetic on named analytic fixtures
only. They do not establish that every real or future packet object has a
positive, unique or stable cross-resolution match.

Analytic cone and trunk/west/east labels exist only in test assertions. They
are forbidden builder inputs and never serialize into packet or correspondence
bytes/hashes.

### 9. Malformed inputs

Typed failures cover incompatible frame/population/domain/portal/config,
unsupported geometry identity, bad predecessor hash, duplicate/cyclic
hierarchy, invalid object reference, overlapping exclusive supports,
non-finite or zero nested-object area, degenerate/non-finite reach geometry,
area/length ledger failure, non-canonical derived zero, overflow and
serialization failure. A valid zero exclusive support remains the explicit
`NoExclusiveSupport` outcome. No error may return a partial core or
correspondence artifact.

## Cost and complexity gate

Build one spatial index over target cell boxes and one over target reach-segment
boxes. A cell-polygon pair is clipped once and its positive measure distributed
to the relevant exclusive/nested membership pairs. A segment-pair interval is
solved once. Do not run a full opposite-cell scan per object or a full
opposite-segment scan per source segment.

Nested hierarchy can create a legitimately large positive output. Report
separately:

- cell-box candidates and actual polygon clips;
- positive cell intersections and nested membership contributions;
- segment-box candidates and analytic segment-pair tests;
- positive object-pair rows and best-graph edges;
- polygon-clip count divided by the full source-cell/target-cell Cartesian
  product, and segment-test count divided by the full segment product;
- canonical packet bytes and sensitivity-payload duplication bytes;
- assembly and correspondence wall time; and
- process peak memory.

The index oracle places 100 unit-square source cells in `[0,10] x [0,10]` and
100 target cells in `[100,110] x [100,110]`; it must return zero polygon
candidates rather than 10,000 clips. The analogous 100 source and 100 target
unit line segments, with pair radii 1 km and the same separation, must return
zero rather than 10,000 segment tests.

Measure 4-to-8 and 4-to-2 correspondence for the frozen linked-highland and
asymmetric-Y populations. There is no hardware-specific promotion time or
comparison against terrain arms that do not yet exist. The real-fixture ratios,
bytes, wall time and memory are report-and-review evidence. The checkpoint
fails only if the frozen sparse oracle or code path reveals an explicit
unindexed Cartesian scan.

## Stop and amendment rule

Stop after O0b implementation, the manufactured matrix, measured cost and a
dated audit. Do not implement packet/product R0, a product O0a adapter, H/C/G
composition or identity/event language in this checkpoint.

Amend before implementation continues if:

- arm/native state, forcing answers or shared cell IDs are needed to obtain an
  expected match;
- an observed H/C/G/product result is needed to choose a threshold, weight,
  buffer, score, winner or fixture answer;
- topology, anchor distance, Jaccard or Dice must override the frozen exclusive
  or line maximum sets;
- reference and sensitivity populations must mix;
- line/catchment `MetricConflict`, exact ties, nulls or background/portal
  coverage would be hidden;
- raw faces, saddles or stations need persistent IDs;
- a best component must be called same/born/retired/split/merge/persistent;
- caller metadata changes common bytes/hash or correspondence mutates a source
  core;
- product/spherical support enters this bounded rung;
- indexed work degenerates into unreported all-pairs scans; or
- a frozen manufactured answer changes after product or competitive output is
  inspected.

If the named analytic 4-to-8/2 gates do not have the frozen unique dominant
matches, stop and determine whether the witness or the provisional umbrella
expectation was invalid. Absence is a valid general outcome; the gate cannot
be weakened into an unfalsifiable universal-positive claim.

Passing O0b would license only a separately committed preregistration for the
next packet comparison/adapter boundary. It would not license O0b identity
semantics, product R0, a terrain arm or a preferred architecture.
