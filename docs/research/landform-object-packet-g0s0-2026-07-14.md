# Landform object packet G0/S0 executable contract

**Date:** 2026-07-14  
**Status:** preregistered; planar structural slice implemented/evaluated; full
rung incomplete
**Parent:** [Landform object packet v0](landform-object-packet-v0-2026-07-14.md)

## Preregistered amendment A: regular-hex width quantization

**Date:** 2026-07-14, before any H/C/G surface or packet was generated.

The first manufactured regular-hex closure check exposed a representation
detail in existing `LandscapeMesh`: centers and cell areas are `f64`, while
internal operator face widths are stored as `f32`. Reconstructing physical
vertices from the quantized width misses the exact center/area regular hex by
nanometres and cannot satisfy the already registered `1e-10` area gate.

For the verified uniform regular-hex companion only, derive physical face width
as `center_distance / sqrt(3)` in `f64` and build its vertices from that width.
The builder must separately verify that the stored operator width is the nearest
`f32` representation within two `f32` ulps. G0 stores the derived physical face
width; the source operator value remains solver provenance, not a second
control-volume geometry. General planar adapters still copy their explicit
companion geometry and may not apply this repair. No tolerance, terrain
threshold or arm result changes.

## Decision

Implement only the common physical surface graph (**G0**) and the independent
surface peak–saddle hierarchy (**S0**) before constructing H, C or G.

The first planar structural checkpoint now passes. Its evidence and remaining
scope are recorded in the [structural-slice audit](../audits/landform-g0s0-structural-slice-2026-07-14.md).

This rung asks whether a final authoritative surface contains stable regional
highland organization. It does not reward deeper physics, infer terrain from
tectonic answers or promote a universal mountain ontology. The same extractor
must report an attractive authentic hack and a reduced physical model without
knowing which produced the surface.

No code may change the formulas, populations, fixture family or gates below
under this schema version. A failed manufactured gate requires a new committed
amendment before implementation continues. H/C/G output may not be used to tune
the instrument.

## Scope

G0/S0 produces:

- one validated planar or spherical finite-volume graph;
- a complete, unpruned superlevel-component split forest;
- compact cell ownership and exact branch footprints;
- reference and one-factor-sensitivity highland populations;
- physical area, persistence, geometry, local-relief, summit-cap and gentle-
  area evidence; and
- a deterministic arm-neutral evidence hash.

It does not produce drainage, basins, ridges, divides, passes, valleys,
cross-resolution identity or a product-visible terrain classification. D0 and
O0 own those later relationships.

## Frozen input contract

The extractor accepts only:

```text
EvaluationSurfaceGraphV0
elevation_km: Vec<f64>
scored_cell: Vec<bool>
SurfaceHierarchyConfigV0
```

It cannot accept an arm label, deformation/uplift field, runoff, routing,
hydrology, native graph, expected feature geometry, renderer state or product
importance policy. Run and arm metadata are attached only after the common
evidence has been serialized and hashed.

All competitive arms receive the same graph, scored mask and configuration.
The testbed mask is fixed from shared domain geometry before any arm output.
The reference closure level is the shared physical base level:

```text
closure_level_km = 0.0
active S0 cell = scored_cell && elevation_km > closure_level_km
```

The comparison preregistration may further restrict the geometry-only scored
mask, but may not change this extractor or derive membership from an arm's
elevation, water, forcing response or observed objects. A separately named
noncompetitive product observation may supply its own frozen scored-domain
policy and records that policy in its header.

`SurfaceHierarchyConfigV0` is exactly:

```text
closure_level_km: 0.0
reference_persistence_km: 0.10
reference_min_footprint_km2: 2500.0
persistence_sensitivity_km: [0.05, 0.20]
footprint_sensitivity_km2: [1250.0, 5000.0]
local_relief_radii_km: [25.0, 50.0, 100.0]
summit_cap_depths_km: [0.25, 0.50, 1.00]
gentle_grade_thresholds: [0.005, 0.010, 0.020]
endpoint_match_abs_km: 1e-8
planar_area_match_relative: 1e-10
sphere_area_closure_relative: 1e-6
linear_rank_relative: 1e-12
orientation_ambiguity_anisotropy: 0.10
spherical_nonlocal_radius_rad: 0.50
schema_version: "landform-g0s0-v0"
hash_version: "fnv1a64-bincode-fixint-le-v0"
```

All threshold comparisons are inclusive except the strict closure test above.
Finite `-0.0` inputs are canonicalized to `+0.0` before comparison or hashing.
There is no elevation epsilon and no per-arm normalization.

## G0: normalized physical graph

### Common representation

`EvaluationSurfaceGraphV0` stores:

```text
domain: Planar | Spherical { radius_km }
cell_center_km: Vec<DVec3>
cell_area_km2: Vec<f64>
cell_polygon_offsets: Vec<u32>
cell_polygon_vertices_km: Vec<DVec3>
edge_offsets: Vec<u32>
edge_neighbor: Vec<u32>
edge_reciprocal: Vec<u32>
edge_distance_km: Vec<f64>
edge_shared_width_km: Vec<f64>
edge_face_endpoints_km: Vec<[DVec3; 2]>
boundary_segments: Vec<EvaluationBoundarySegmentV0>
```

Each boundary segment records its canonical ID, owner cell, directed endpoints,
physical length, closed/open condition and, when open, semantic portal ID and
base level. Planar projected-span metadata remains separate from physical
length. A scored/unscored internal face is mask contact, not a physical outlet.

Every internal directed edge has one reciprocal directed edge and one physical
shared face. Neighbor lists are sorted by neighbor cell ID. Directed-face IDs
are their canonical CSR positions. A repaired nearest-neighbor link without a
shared control-volume face is not physical adjacency and returns
`NonPhysicalAdjacency`; the existing product fallback width must not enter G0.

Cell polygons have at least three vertices and no repeated closing vertex.
Planar polygons are counter-clockwise in `(x,y)`. Spherical polygons are
counter-clockwise as viewed from outside the sphere. Their first vertex is the
lowest source vertex ID where one exists, otherwise the lexicographically
smallest canonical `(x_bits,y_bits,z_bits)` vertex. Face endpoints are stored
in the directed polygon order. Reciprocal endpoints therefore reverse.

### Planar `LandscapeMesh` adapter

The general adapter requires a `LandscapeControlVolumesV0` companion containing
CCW cell polygons plus CSR-aligned internal and boundary face endpoints. A bare
`LandscapeMesh` has operator centers, areas, distances, widths and normals but
does not uniquely specify arbitrary control-volume polygons. Missing companion
geometry returns `MissingControlVolumeGeometry`; the adapter may not invent a
polygon from centers.

The current uniform regular-hex constructor has a separately tested companion
builder. Only after proving the regular lattice, midpoint faces and constant
hex geometry may that builder use, for an internal directed edge `i -> j`:

```text
normal = normalize(center[j] - center[i])
tangent = (-normal.y, normal.x, 0)
face_center = 0.5 * (center[i] + center[j])
endpoints = face_center +/- 0.5 * shared_width * tangent
```

Under amendment A, `shared_width = center_distance / sqrt(3)`; the stored `f32`
operator width must pass the separate quantization check.

Endpoint order is chosen to follow the owning cell counter-clockwise. Boundary
segment endpoints are reconstructed from `LandscapeBoundaryFace.center_km`,
`outward_normal` and `width_km`; portal-split collinear segments remain
explicit. Internal and boundary endpoints are gathered and angle-sorted around
the cell center to reconstruct the full regular-hex control volume, retaining
portal cut points as collinear vertices.

The general adapter copies native cell centers, areas, distances, face widths,
outlet IDs and base levels in kilometres and validates them against the
companion geometry. It rejects non-planar centers, a directed tangent
inconsistent with the center displacement, reciprocal widths/endpoints outside
the registered tolerance, a polygon that does not close through its faces, or
polygon shoelace area differing from stored finite-volume area beyond
`planar_area_match_relative`. The tangent-projected irregular Voronoi cap must
likewise supply its already constructed projected polygons and add exact
directed endpoints; it cannot pass through the regular-hex builder.

### Spherical product `Tessellation` adapter

Generators and Voronoi vertices are normalized in `f64` and multiplied by the
canonical planet radius. Cell vertex order comes from the source Voronoi cell.
For adjacent cells, the shared face is the exactly two consecutive shared
Voronoi vertex IDs. Anything else returns `NonPhysicalAdjacency`.
Canonical adjacency is rebuilt from polygon-edge ownership and must equal the
stored tessellation adjacency exactly; a nearest-generator orphan repair is
therefore exposed as non-geometric instead of silently entering S0.

Distances and face widths use the robust great-circle formula:

```text
arc(a,b) = radius_km * atan2(length(cross(unit(a), unit(b))),
                             dot(unit(a), unit(b)))
```

Cell area is recomputed in `f64` by summing oriented spherical triangles with
the Van Oosterom–Strackee formula and multiplying steradians by `radius_km^2`.
The closed-sphere sum must match `4*pi*radius_km^2` within the registered
`sphere_area_closure_relative`. The full sphere has no physical boundary segments.
The product elevation adapter converts the authoritative normalized surface to
kilometres once through `units::elevation_to_km` before S0.

### G0 validation

Validation completes before S0 allocates partial output. Typed errors cover:

- empty graph, empty scored mask or mismatched array lengths;
- non-finite geometry, elevation or configuration;
- non-positive area, edge distance or face width;
- malformed/overflowing CSR or polygon offsets;
- self edges, duplicate neighbors or missing/non-unique reciprocals;
- reciprocal distance, width or endpoints outside registered tolerance;
- invalid polygon winding/area or internal face not backed by both polygons;
- invalid, duplicate or overlapping semantic outlet assignments; and
- spherical radius/normalization/closed-area failure.

Disconnected scored regions are valid. A nonempty scored mask with no cell
strictly above closure produces a valid empty S0 forest.

## S0: split-forest construction

### Exact-level batching

Process active cells in descending canonical `f64` elevation. Cells occupy the
same level only when their canonical values have identical bits.

For every distinct level `h`:

1. Find connected components of all not-yet-active cells at exactly `h`.
2. Union each complete equal-level component before examining higher active
   neighbors.
3. Collect the distinct higher active branches touching that level component.
4. With zero higher branches, birth one flat-maximum peak branch.
5. With one higher branch, extend it without creating a critical node.
6. With two or more higher branches, create one flat-saddle node and merge all
   branches at `h`.

This is an operational split tree on the represented cell graph, not the exact
Morse complex of an unknown subcell surface. Exact-elevation peak and saddle
batches are called flat maxima and flat saddles, not geomorphic plateaus.

At a merge the elder branch is selected by:

1. greater peak elevation;
2. lexicographically smaller area-weighted flat-maximum centroid in the
   canonical planar frame or spherical `(x,y,z)` frame; then
3. lower anchor cell ID as a final same-packet implementation tie-break.

All losing branches pair with the same multiway saddle. An equal peak-height
elder choice records `equal_elder_ambiguous`; its arbitrary parent direction is
excluded from later topology-quality scoring. Multi-cell maxima and saddles
also retain explicit flat-batch flags and sorted member IDs.

Cells exactly at or below closure are background and cannot merge components.
Each surviving component becomes one root paired with the external closure
level:

```text
root persistence = peak_elevation - closure_level
root footprint = active cells in that root component
```

Root closure and contact with the scored-mask boundary are reported. They are
not errors.

### Complete structure and compact footprint ownership

The complete raw forest is serialized before thresholding. Pruning never
changes topology, persistence or another branch's footprint.

Every active cell is assigned once to the peak branch that owns it when its
level batch is activated. At a merge the saddle batch is assigned to the elder
branch. A peak branch's footprint is the union of exclusive cells owned by that
branch and all losing descendant branches beneath it. This compact
`cell_peak_owner` plus the peak/saddle tree represents nested footprints in
`O(cells + critical nodes)` storage.

Equivalently, an interior losing branch's footprint is its connected component
in `{cell | elevation > saddle_elevation}` immediately before the saddle batch
activates. The saddle batch is excluded from that losing footprint. For a root,
substitute the strict closure level.

Materialized member IDs, union-boundary directed-face IDs and physical boundary-
segment IDs are ascending.
Footprint area is the sum of member control-volume areas. Nested footprint
areas overlap by design; aggregate occupied area uses a physical union or
exclusive ownership and never sums ancestor and child areas.

### Reference and sensitivity populations

A branch is a reference `HighlandFeatureV0` exactly when:

```text
persistence >= 0.10 km
AND footprint_area >= 2500 km2
```

Roots use the same predicate and remain separately flagged. Sensitivities vary
one factor at a time:

```text
persistence 0.05 / 0.20 km with area fixed at 2500 km2
area 1250 / 5000 km2 with persistence fixed at 0.10 km
```

They are report-only populations. There is no Cartesian search, `OR`
retention, per-arm threshold, observed-maximum threshold or substitution of a
sensitivity population for the reference result. Absence is evidence.

## Frozen highland measurements

All measurements use the physical footprint union. Unavailable geometry is an
explicit optional result and flag, never NaN or a fabricated zero.

### Area, centroid, moments and orientation

Planar footprints use exact polygon shoelace area, centroid and second moments
summed over their non-overlapping member cells.

For a spherical footprint, first compute the normalized physical-area-weighted
sum of member cell centroid directions. Construct a deterministic tangent basis
by selecting the Cartesian axis least aligned with that centroid, crossing it
with the centroid, normalizing and fixing the first nonzero component positive.
Map polygon vertices into that plane with the azimuthal-equidistant spherical
log map. Each projected cell's polygon moments are scaled by
`physical_cell_area / projected_polygon_area` before summation.

If the centroid resultant fails the registered rank tolerance or any vertex is
antipodal within that tolerance, area and persistence remain valid but local
geometry is `NonLocalGeometry`. Maximum footprint angular radius above `0.50`
radians sets `spherical_nonlocal_warning` without invalidating the metrics.

Let the area covariance eigenvalues be `lambda1 >= lambda2`. Report:

```text
equivalent_ellipse_length_km = 4 * sqrt(lambda1)
equivalent_ellipse_width_km  = 4 * sqrt(lambda2)
anisotropy = (lambda1 - lambda2) / (lambda1 + lambda2)
```

The principal eigenvector is a signless orientation; serialize its sign with
the first nonzero component positive. When anisotropy is below `0.10`, retain
the continuous anisotropy but mark orientation ambiguous so an almost circular
footprint cannot earn a meaningful alignment score.

Also report a deterministic two-sweep center extent: start at the lowest member
cell ID, select the physically farthest center (lowest ID on an exact tie), then
select the farthest center from it. `two_sweep_extent_km` is the final distance
and `mean_width_km = footprint_area / two_sweep_extent_km`. It is explicitly an
approximation, not a maximum polygon diameter.

### Fixed-radius local relief

For scored cell `i` and registered radius `R`:

```text
N_R(i) = scored cells with physical center distance(i,j) <= R
relief_R(i) = max(elevation over N_R(i)) - min(elevation over N_R(i))
```

Distance is Euclidean on the plane and great-circle on the sphere, never hop
count. The neighborhood is not clipped to a highland footprint. A spatial
index or adjacency candidate walk is permitted only if the final inclusion
test uses the exact registered distance. Cells whose radius intersects the
physical or scored-domain boundary are flagged as truncated.

For every footprint and radius report area-weighted p50 and p90. A weighted
quantile is the lowest ordered value whose cumulative physical area reaches the
requested fraction; exact value ties use cell ID only for serialization.

### Physical grade, summit caps and gentle area

Estimate one physical tangent-plane gradient per scored cell from all scored
reciprocal neighbors. For neighbor displacement `q_ij`, elevation difference
`dz_ij`, shared face length `l_ij` and center distance `d_ij`:

```text
w_ij = l_ij / d_ij
M_i = sum(w_ij * q_ij * transpose(q_ij))
b_i = sum(w_ij * q_ij * dz_ij)
g_i = inverse(M_i) * b_i
physical_grade_i = length(g_i)
```

Planar displacement uses native `(x,y)`. Spherical displacement uses the log
map at the cell center. If the smaller eigenvalue of `M_i` is no greater than
`linear_rank_relative * larger_eigenvalue`, the fit is unavailable and the
cell is flagged. This face-transmissibility weighting is identical on both
backends. An affine planar field must reconstruct to numerical tolerance.

For peak elevation `z_peak`, branch footprint `F` and cap depth `d`:

```text
cap(d) = { i in F | elevation_i >= z_peak - d }
cap_area(d) = sum(area_i for i in cap(d))
cap_fraction(d) = cap_area(d) / footprint_area
gentle_fraction(d,g) =
    sum(area_i for i in cap(d) with valid grade_i <= g) / cap_area(d)
valid_grade_fraction(d) =
    sum(area_i for i in cap(d) with valid grade_i) / cap_area(d)
```

Invalid fits do not count as gentle, so a broken backend cannot improve the
fraction. Report the full registered depth/grade family; `d=0.50 km` and
`g=0.010` are the reference descriptor. If `d >= persistence`, set
`cap_merge_censored`; do not choose a shallower feature-specific cap.

Large cap fraction, low local relief and high gentle fraction are continuous
plateau-like evidence. None changes feature retention or declares the exact
flat-maximum batch a plateau.

## Canonical IDs, serialization and hashes

Peak nodes are ordered by descending peak elevation, flat-maximum centroid and
anchor cell. Saddle nodes are ordered by descending saddle elevation, centroid
and anchor cell. IDs are assigned only after that ordering. Root, child and
saddle references and materialized cell/face lists are then sorted by canonical
ID. No hash-bearing structure contains an unordered map.

Normalize finite zero before serialization. Optional measurements use tagged
options rather than NaN. The canonical common byte stream uses bincode v1 fixed
integer encoding, little-endian primitives and the declared field order. Its
64-bit FNV-1a digest is `derived_evidence_hash` and covers:

- graph identity and physical geometry;
- scored mask, physical elevation and complete configuration;
- the complete raw forest and compact cell ownership;
- all retained populations, materialized reference objects and aggregates.

FNV-1a is a reproducibility checksum, not a security claim. A separate packet-
envelope hash may include run, arm and native provenance. Changing only arm
label or native provenance must leave the common bytes, common object IDs and
derived hash byte-identical.

Determinism is required for the same build and target. Cross-platform floating-
point bit identity is not claimed in v0.

## Typed errors and nonfatal evidence

Fatal typed errors include invalid G0 geometry, non-finite input or derived
values, impossible negative persistence, multiple exclusive owners for one
cell, cyclic parentage, footprint/member area mismatch, overflow and failure to
reproduce the canonical byte stream on an immediate second serialization.

The following are valid reportable outcomes: empty forest, multiple roots,
boundary contact, flat maximum/saddle, equal elder ambiguity, orientation
ambiguity, nonlocal spherical geometry, rank-deficient local grade, root
closure, threshold-sensitive presence and unfavorable morphology. Do not turn
a poor arm result into an extractor error.

## Frozen manufactured fixtures

Every fixture runs with the reference config except the explicitly paired
translation-invariance check, which translates both physical elevation and the
declared closure by the same frozen offset. Graph-order tests additionally
permute cell IDs and CSR neighbor order before canonical adaptation.

### Exact graph topology fixtures

Use small valid planar polygon graphs with explicitly assigned elevations to
assert:

1. one hill above closure: one root peak and no interior saddle;
2. unequal peaks joined above closure: correct elder, child, saddle and
   persistence;
3. equal peaks: physical-centroid elder tie-break plus ambiguity flag,
   invariant under cell reindexing at the geometry/topology level;
4. one multi-cell flat maximum: one peak under two cell/CSR permutations;
5. one multi-cell saddle joining three peaks: one saddle and two losing pairs;
6. two hills connected only at closure: two roots and no saddle;
7. the same low raised above closure: one root and the expected saddle;
8. an unscored hole: correct forest plus boundary flags, with no invented
   critical node;
9. no scored elevation above closure: a valid empty forest; and
10. a prescribed piecewise-constant field: exact exclusive ownership,
    footprint members, union boundary and physical area.

### Analytic morphology fixtures

Use regular planar hex meshes at nominal 8/4/2 km over a geometry large enough
that every reference feature is more than 100 km from the scored boundary.

The two-cone saddle fixture is:

```text
p1 = (-80, 0), h1 = 2.0 km
p2 = ( 80, 0), h2 = 1.8 km
slope = 0.010
z(x,y) = max(h1 - slope * distance((x,y),p1),
             h2 - slope * distance((x,y),p2))
```

Its continuous merge is at `(10,0)` and `1.10 km`. It must produce two peaks,
one root, one interior saddle and the correct elder at all resolutions.

The cap pair uses the same isolated center and flank slope:

```text
narrow(r) = 2.0 - 0.015 * r
broad(r)  = 2.0 - 0.015 * max(0, r - 40 km)
```

The broad surface must have one flat maximum, greater cap area at every
registered depth, greater gentle fraction at grade `0.005` and `0.010`, and no
lower gentle fraction at `0.020` than the narrow surface.

The linked-segment fixture is the maximum of four cones:

```text
(-180,-40,2.4), (-60,0,2.0), (60,0,2.2), (180,40,1.9)
common flank slope = 0.010 km/km
```

The adjacent analytic contact heights, descending, are approximately
`1.56754`, `1.50000` and `1.41754 km`. Non-adjacent contacts are lower. The
expected forest is therefore one A-rooted merge sequence with three interior
events in that order; B, C and D lose respectively to the branch containing A.
These answers come from the frozen cone geometry, not extractor output.

Additional morphology fixtures cover an axis-aligned rectangle, the same
rectangle rotated 30 degrees, a symmetric disk, a planar affine field of grade
0.010, and translation plus equal elevation/closure translation. They assert
area/moment orientation, rotation invariance, disk orientation ambiguity,
affine grade reconstruction and invariance of relative topology/measurements.

### Error and neutrality fixtures

Malformed CSR, a nonreciprocal edge, a synthetic neighbor without a shared
face, negative area, invalid polygon and non-finite elevation each return the
registered typed error. Deliberately corrupted ownership/tree invariants fail
before serialization. Changing only arm/provenance metadata preserves every
common byte and the derived hash.

## Frozen 8/4/2 gates

For the analytic one-hill, two-cone, cap-pair and linked-segment fixtures:

- expected root/peak/saddle topology is identical at 8, 4 and 2 km;
- reference retention is identical and deliberately away from thresholds;
- no unexpected birth, retirement, split or merge occurs;
- broad-versus-narrow cap/gentle ordering and orientation ambiguity are stable;
- graph and footprint area close to the adapter tolerance; and
- peak/saddle elevation absolute error is no more than
  `L * cell_circumradius`, and anchor displacement no more than one cell
  circumradius, using each analytic fixture's declared Lipschitz bound `L`.

For the rectangle and linked objects at 2 km, area and equivalent-ellipse
length relative errors must be at most 5%, and equivalent-ellipse width error
at most 7.5%. Their absolute errors must not increase from 8 to 4 to 2 km.

G0/S0 does not preregister general cross-resolution correspondence. Fixture
matching uses the known analytic peak labels only. Dominant physical-overlap
matching remains O0 ownership; it cannot be improvised here and then silently
become the H/C/G comparison rule.

## Stop and amendment rules

Stop before code continues if the implementation needs to choose or change:

- the scored mask, closure or exact-equality convention;
- batch ordering, elder rule, root handling or footprint definition;
- `AND` retention, inclusive comparisons or sensitivity construction;
- physical polygon/face geometry or an adapter fallback;
- morphology, relief, cap or grade formulas;
- object ordering, serialization or hash scope;
- a fixture input, expected topology or convergence tolerance; or
- smoothing, simplification or any arm-specific exception.

A new preregistered checkpoint may correct a mathematically impossible gate or
an invalid physical approximation using manufactured evidence. It may not use
H/C/G output. Passing G0/S0 establishes only a neutral landscape instrument;
it neither validates current terrain nor selects a future organization owner.
