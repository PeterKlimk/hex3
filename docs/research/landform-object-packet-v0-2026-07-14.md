# Landform object packet v0

**Date:** 2026-07-14
**Status:** umbrella evidence contract; G0/S0 completed and D0 executable rung
preregistered; O0/R0 not yet preregistered
**Parent decision:** [Landscape organization strategy](../landscape-strategy.md)

## Decision

Before defining executable H/C/G terrain arms, build one presentation-independent
object packet which can judge every final authoritative surface through the same
derived evidence.

The packet is an experiment instrument, not promoted world ontology. It owns no
terrain, water or authored feature state. It derives operational highland,
peak/saddle, basin, divide, reach and valley-probe objects for comparison and
retains the physical backing needed to audit them.

Do not promote the existing 1.5 km mountain components, 50 km crest bins or
100 m crest lows as the shared definition. Those diagnostics helped expose the
tableland defect but encode one global elevation threshold and a one-dimensional
pass proxy. They remain secondary hypotheses.

## Question

Can one deterministic, mesh-aware extractor expose the topology and morphology
needed to distinguish hold-and-carve, reduced coevolution and graph-first
reconstruction without using an arm's native graph, tectonic answer field or
presentation state?

Passing v0 means only that the arms can later be compared through a common
semantic lens. It does not promote an arm or the object definitions themselves.

## Ownership and exclusions

The extractor may consume:

- physical cell centers, areas, adjacency, edge lengths and boundary portals;
- undisplaced authoritative elevation in kilometres;
- a scored-domain mask;
- a common evaluation-routing result derived from that surface; and
- common runoff supplied independently of an arm.

It may associate objects after extraction with shared deformation, runoff,
repair or arm-native provenance fields.

It must not consume as membership inputs:

- `World`, `OrogenModel`, experiment-arm identity or native H/C/G object IDs;
- tectonic feature masks, expected segment geometry or target range answers;
- renderer relief, river width, color, lighting, camera or visibility policy;
- product `Major` river selection; or
- cell-count thresholds or per-arm normalization by observed maxima.

An arm-native graph lives in a separate provenance namespace. Its agreement
with the independently extracted packet is an outcome.

For competitive testbed packets, the scored-domain mask is identical across
all arms and conditions and is derived only from the shared physical domain
geometry before any arm output exists. It cannot depend on elevation, water,
runoff, forcing response or arm identity. A product observation may use a
separately named land/domain policy; it is not the same comparison population.

## Packet header

`LandformObjectPacketV0` records:

- schema and extractor version;
- run/revision/dirty identity supplied by the caller;
- arm label as metadata only;
- authoritative surface and evaluation-hydrology hashes;
- mesh/domain/backend identity and physical resolution;
- scored-domain geometry;
- complete extractor configuration and rationale;
- deterministic derived-evidence hash; and
- optional native-provenance hash.

No camera or presentation parameter is serializable into this header.

The derived-evidence hash excludes arm label and all native provenance.
Identical common graph, surface, runoff, scored mask and extractor configuration
must produce byte-identical common objects, relationships and aggregates under
H, C and G labels and under arbitrary native-provenance payloads. Native graph
correspondence lives in a separate section/hash and cannot alter common object
IDs or summaries.

Object IDs are opaque and stable only inside one exact packet. Cross-arm,
cross-resolution and changed-condition identity uses explicit correspondence;
cell indices are never persistent geographic identity.

## G0: common physical surface graph

Introduce one derived `EvaluationSurfaceGraphV0` used only by semantics and
validation. It contains:

```text
domain: planar | spherical(radius_km)
cell_center_km: Vec<DVec3>
cell_area_km2: Vec<f64>
edge_offsets: Vec<u32>
edge_neighbor: Vec<u32>
edge_distance_km: Vec<f64>
edge_shared_width_km: Vec<Option<f64>>
edge_reciprocal: Vec<Option<u32>>
edge_face_endpoints_km: Vec<Option<[DVec3; 2]>>
cell_polygon_offsets: Vec<u32>
cell_polygon_vertices_km: Vec<DVec3>
scored_cell: Vec<bool>
outlet/closed boundary segment geometry and metadata
```

Adapters are required for both `LandscapeMesh` and product `Tessellation`.
The spherical adapter uses great-circle distance and physical spherical area;
the planar adapter preserves the stored finite-volume geometry. Adapter tests
must prove reciprocal adjacency, positive measures, deterministic CSR order and
area preservation.

Physical footprint, buffered-line and cross-resolution overlap must use this
control-volume/face geometry. A rung contract must freeze the planar/spherical
polygon clipping and line-buffer procedure before correspondence is implemented;
cell centers plus nominal area are not a permitted silent approximation.

The product adapter exists to observe the unchanged product reference. H/C/G
competition uses the same testbed graph and exact adapter, so product and
testbed hydrology need not pretend to be one implementation.

## S0: surface peak–saddle hierarchy

The first surface object is a deterministic superlevel-component **split tree**,
not a thresholded "mountain range."

Algorithm:

1. Sort scored cells by decreasing physical elevation.
2. Activate exact-equal-elevation cells as one batch; union their mutual edges
   before relating the batch to higher components. This prevents cell index
   from fragmenting a flat summit or selecting a false saddle.
3. A batch with no higher neighbor births a flat-maximum component. Exact
   equality does not make it a geomorphic plateau.
4. A batch joining two or more higher components is a saddle node. The highest
   peak survives by the elder rule; exact ties use the lowest anchor cell only
   as a deterministic implementation tie-break. Mark an exact-height elder tie
   as semantic ambiguity; its arbitrary parent direction cannot contribute to
   topology-quality comparisons.
5. Record each losing peak's key saddle, vertical persistence and superlevel
   footprint immediately above the merge.
6. The surviving root closes at the lowest scored elevation or declared base
   level, identified explicitly in the packet.

Run this algorithm as a deterministic forest, one rooted split tree per
connected scored-domain component. D0 may still route on the full physical
graph. `SurfaceHierarchyV0` retains flat-maximum nodes, saddle nodes, parent/child
links, persistence, physical footprints and boundary contact. A retained node
is called a `HighlandFeatureV0`; it is not yet a promoted range or massif.

Required highland measurements are physical area, peak and key-saddle
elevation, persistence, sampled maximum extent, mean width, orientation,
boundary contact and fixed-radius local relief.

Historical absolute-elevation range components may be attached as a named
secondary overlay. They cannot determine S0 membership or pass/fail.

## D0: common evaluation drainage

H/C/G evaluation uses one independent drainage derivation over each final
physical surface. Native C flux and G's authored graph are report-only for this
packet.

For the bounded testbed, D0 uses the existing portal-seeded priority-fill/flat
potential and a deterministic single-receiver evaluation tree. It preserves
physical elevation, uses genuine boundary portals and records structural
catchment area separately from runoff-conditioned discharge. A receiver graph
is permitted here as a topology coarse graining; it makes no subcell thalweg or
channel-bed geometry claim.

The exact receiver ordering, flat rule, portal ownership and tie-break must be
committed before any H/C/G surface is generated. Cycles, disconnected cells,
unexplained sinks or water-balance failure invalidate the packet.

Conditioning is never invisible. D0 records each depression, spill elevation,
maximum and integrated virtual fill, affected physical area/volume, and every
receiver/reach segment whose descent is supported only by the fill/flat
potential rather than the physical surface. Longitudinal profiles always use
physical elevation and flag conditioned segments. This permits a valid
evaluation tree without rewarding a pit-ridden surface as clean drainage.

The unchanged product observation may adapt retained product `Hydrology`, but
its derivation ID must differ and it is not numerically compared with the
testbed packet as though routing were identical.

## Required v0 objects

### `HighlandFeatureV0`

Derived only from S0:

- flat-maximum anchor and complete peak–saddle subtree;
- key saddle and persistence;
- physical footprint, boundary, area and extent/orientation;
- 25/50/100 km local-relief summaries;
- summit-cap area at registered physical depth below the peak;
- gentle-area fraction under the registered physical-grade probe; and
- boundary/contact and scale-sensitivity flags.

Several retained highland nodes may later compose one range candidate. V0 does
not freeze that higher grouping before ridge/divide evidence exists.
Plateau evidence is continuous cap area, local relief and gentle-area fraction;
an exact-elevation flat maximum is not itself classified as a plateau.

### `DrainageBasinGraphV0`

Derived only from D0:

- cell-to-outlet and retained sub-basin membership;
- physical area, structural contributing area and supplied runoff;
- stable semantic portal/outlet identity and open/terminal status;
- parent/child containment at retained confluences; and
- raw shared divide edges between adjacent retained basins.

Nested basin records may overlap, but the divide partition may not. Each cell is
assigned to one **exclusive incremental catchment** owned by the first retained
downstream reach or portal it enters. Divide edges are emitted only between
different exclusive IDs; nested parent/child relationships are stored
separately. The D0 executable contract freezes confluence retention and this
assignment/tie rule before implementation.

Raw divide edges are not prematurely grouped into named polylines. A geometric
ridge claim additionally requires positive bilateral relief toward the streams
reached on both sides.

### `RiverReachGraphV0`

Use a frozen physical catchment threshold to derive source/confluence/outlet-
bounded reaches from D0:

- directed reach topology and ordered cell support;
- physical length and structural contributing area;
- common-runoff supply/discharge;
- Strahler order;
- outlet and retained-basin relationships; and
- separate greatest-supply, longest-trunk and highest-order roles.

There is no single `major` truth in the evaluation packet. Product visibility
policy may be attached as a secondary overlay.

### `RidgeDivideRelationV0`

Relate S0 highland/saddle structure to D0 basin boundaries after both are
independently extracted:

- ridge/divide coincidence and separation length;
- saddle connection between adjacent retained basins;
- bilateral divide-to-reach relief and asymmetry;
- highland-axis versus divide orientation; and
- transfer-low presence as an observed relationship, never a supplied target.

### `ValleyCorridorProbeV0`

V0 does not claim a universal valley polygon. For retained trunks/reaches it
records sampled evidence:

- longitudinal physical-surface profile;
- fixed-spacing cross sections;
- drainage-relative elevation;
- flanking divide relief/asymmetry; and
- corridor width under the frozen relative-height rule.

This remains a probe until its boundary and cross-resolution behavior earn
semantic promotion. It represents regional valley support, never a channel bed,
bank or floodplain.

## Frozen v0 scale family

Reference values are diagnostic policies, not quality targets:

| Quantity | Reference | Sensitivity bracket |
|---|---:|---:|
| Mesh spacing | 4 km | 8 / 2 km convergence |
| Peak–saddle persistence | 0.10 km | 0.05 / 0.20 km |
| Minimum retained highland footprint | 2,500 km² | 1,250 / 5,000 km² |
| Local-relief radius | 50 km | 25 / 100 km |
| Summit-cap depth below peak | 0.50 km | 0.25 / 1.00 km |
| Gentle physical grade | 1% | 0.5 / 2% |
| Reach catchment support | 2,000 km² | 1,000 / 4,000 km² |
| Along-reach cross-section spacing | 20 km | 10 / 40 km |
| Cross-section half-length | 100 km | 50 / 150 km |
| Cross-section sample step | 4 km | 2 / 8 km |
| Relative-height corridor fraction | 0.25 | 0.15 / 0.35 |
| Maximum probed trunk length | 400 km | 200 / 600 km |

All arms use the reference population. Sensitivity brackets vary one factor at
a time around that population; they do not form an opportunistic Cartesian
search and cannot replace the reference result. No arm may select its most
flattering threshold or scale. Absence at a reference setting is an outcome,
not permission to lower its threshold.

The S0 executable contract must freeze the exact primary retention predicate,
root treatment, extent/width/orientation formulas and all ties. The D0 contract
must freeze sub-basin/confluence retention and reach endpoint/length rules. O0
must freeze ridge–divide tolerance, valley sampling/interpolation and
correspondence buffering/assignment. The scale table alone does not make those
algorithms executable.

No spatial smoothing is part of v0. Vertical persistence and 8/4/2 km object
correspondence are the registered mesh-noise controls. Adding a fixed-support
surface filter requires a new checkpoint and must leave the raw result visible.

## Correspondence

`ObjectCorrespondenceV0` is separate from object identity and records:

```text
source and target packet/object IDs
physical footprint or buffered-line overlap
anchor displacement
topology agreement
dominant mutual match
same | split | merge | born | retired | ambiguous
confidence and tie margin
```

Cross-resolution and cross-arm matching uses physical geometry. Shared cell
indices may break a tie only within the same mesh and cannot establish identity.
G's native graph uses the same correspondence structure but a distinct
provenance namespace.

## Cause and provenance relationships

Only after extraction may the packet compute:

- overlap/orientation with the common deformation field;
- range-end distance from a declared segment termination;
- longitudinal/transverse drainage relative to forcing;
- reach crossing through highland/divide structure;
- initial/final and changed-forcing correspondence;
- native-graph to extracted-graph agreement; and
- hydrologic repair overlap where a product observation supplies it.

No object qualifies because it appears where an arm was expected to put it.

## Aggregate summaries

Aggregates never replace object records. The minimum summary distinguishes
object-, area- and length-weighted measurements and includes:

- highland count/area, persistence, extent, relief and summit-cap distributions;
- peak/saddle node and retained-link counts;
- basin-area hierarchy and divide length;
- reach length, contributing area, supply and order distributions;
- physical drainage density;
- ridge–divide coincidence;
- object survival/split/merge/displacement across controls; and
- runtime, peak memory, deterministic repeat and 8/4/2 km response.

Cell counts are resolution context only. Topology counts are descriptive, not
quality scores by themselves.

## Ordered implementation rungs

1. **G0/S0:** common graph adapters, batch-safe split tree, highland records and
   manufactured peak/plateau/saddle tests.
2. **D0:** common testbed evaluation routing, basin/reach/divide objects and
   fork/confluence/flat/portal tests.
3. **O0:** ridge–divide relationships, valley probes, correspondence and packet
   serialization.
4. **R0:** one unchanged product-reference packet and one pre-arm linked-testbed
   manufactured packet at 8/4/2 km.

Each rung is a separate checkpoint. Do not implement H, C or G composition
while the common evidence packet is incomplete.

This document freezes the shared ownership, object vocabulary, scale family,
neutrality and rung order. It is not permission to implement a rung. Each rung
requires a separately committed executable preregistration containing every
formula, ordering, retention predicate, approximation and tie-break before its
code is written.

## Frozen tests and gates

### Geometry and determinism

- Planar and spherical adapters preserve area and reciprocal adjacency.
- Identical common input produces byte-identical derived evidence and hashes
  across arm labels and arbitrary native-provenance payloads.
- Equal-height flat maxima do not split or choose false saddles by cell order.
- Invalid lengths, non-finite fields, invalid CSR and drainage cycles
  return typed errors rather than partial packets.

### Manufactured surface hierarchy

- One smooth hill produces one root highland and no interior saddle.
- Two unequal hills with a prescribed connecting saddle produce the correct
  elder/child relation and persistence within discretization error.
- A broad flat maximum is one node and has more cap/gentle-area evidence than
  an equal-height narrow hill; neither is classified as a plateau by equality.
- A linked two-segment surface retains two massif nodes and one transfer saddle
  at 8/4/2 km under the registered persistence family.

### Drainage and relationships

- Fork/confluence fixtures conserve structural area and common runoff.
- Portal, flat and nested depression fixtures terminate deterministically
  without changing physical elevation.
- Depression conditioning reports virtual rise/volume and flags physically
  non-descending receiver/reach segments.
- Each divide edge is emitted once with physical length and two distinct basin
  IDs.
- Greatest-supply, longest-trunk and highest-order roles remain distinct when
  the fixture makes them distinct.
- A divide with no bilateral relief is not promoted to a geometric ridge.

### Resolution

- Objects retained by the 4 km reference predicates have dominant physical
  matches at 8 and 2 km.
- Report raw deltas; the later H/C/G preregistration freezes promotion
  tolerances after these manufactured baselines, before arm outcomes.

## Stop and amendment rules

Stop and amend before H/C/G if:

- an object definition depends materially on cell index, arm identity or
  presentation;
- G could obtain credit from its native graph without reconstructed-surface
  support;
- the registered scale family reverses the manufactured object topology;
- the same surface produces incompatible physical area/length accounting across
  adapters; or
- a proposed "pass," "ridge," "range" or "valley" cannot be stated as an
  operational surface/drainage relationship.

An amendment is a new preregistered checkpoint. Do not tune this extractor on
H/C/G output.

## Promotion limits

Passing v0 does not establish persistent identity, natural-kind terrain
classes, channel beds or widths, product river importance, Earth calibration,
sediment, ecological meaning or cartographic generalization. It establishes
only a common reproducible evidence layer for the landscape architecture
comparison.
