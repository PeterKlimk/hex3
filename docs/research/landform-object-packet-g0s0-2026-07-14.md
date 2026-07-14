# Landform object packet G0/S0 executable contract

**Date:** 2026-07-14  
**Status:** manufactured G0/S0 pass; planar analytic 8/4/2, product-spherical
and projected-cap G0 adapters, and synthetic-sphere S0/morphology pass; product
landforms remain unobserved
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

## Preregistered amendment B: analytic fixture completion

**Date:** 2026-07-14, before any analytic 8/4/2 outcome test was written or
run.

Implementation review found that the analytic matrix named a one-hill and a
rectangle without defining them, left the regular-hex lattice phase free, and
did not say where the continuous field was sampled. It also found two gates
that are not consequences of centre-sampled graph topology: a graph saddle can
activate nearly one centre spacing from a continuous contact, and full-cell
quadrature errors on non-nested hex lattices need not decrease monotonically at
every refinement. Freeze the missing inputs and replace those two claims as
follows.

Every analytic resolution uses:

```text
spacing = 8, 4 or 2 km
width   = 1120 km
height  = 480 * sqrt(3) km
mesh    = uniform_planar_hex_with_portals(width, height, spacing, &[])
mask    = every cell scored
z_i     = continuous z evaluated at cell_center_km[i]
closure = 0 km
```

This produces respectively `140×120`, `280×240` and `560×480` cells. Assert
those dimensions and the resulting centred phase before extracting anything.
The physical boundary, not the requested width/height, determines the buffer;
every reference-footprint member centre must be strictly farther than 100 km
from every boundary segment and every 100 km relief truncation list must be
empty.

The analytic one-hill is the isolated cone

```text
center = (0,0), peak = 2.0 km, slope = 0.010
z = 2.0 - 0.010 * r
```

Its continuous closure footprint is a radius-200-km disk. The cap-pair center
is also `(0,0)`. The rectangle fixtures are centre-classified plateaux:

```text
axis aligned: z = 1 km when |x| <= 160 km and |y| <= 60 km, else 0
rotated:      apply the same test after rotating the point by -30 degrees
```

Their continuous area is `38400 km²`; equivalent-ellipse length and width are
`369.504172281 km` and `138.564064606 km`. The previously named symmetric-disk,
affine and translation cases remain local formula fixtures rather than extra
large resolution matrices.

Match discrete peaks to analytic labels independently at each resolution; IDs
are never cross-resolution correspondence. For a point apex, peak elevation
error remains bounded by `L * cell_circumradius` and the anchor centre by one
circumradius. For the broad cap, require its flat-maximum support to represent
the analytic radius-40-km maximum set; do not compare its serialization anchor
to the center. For each cone contact, replace the saddle bounds with:

- elevation error at most `L * spacing`; and
- distance from the analytic contact to the union of the flat-saddle support
  cell polygons at most one spacing.

The continuous linked root oracle is the union of closure disks with radii
`240, 200, 220, 190 km`. Its registered references are:

```text
area                 307636.56239 km²
centroid              (-36.18947048, -11.29139307) km
covariance xx/xy/yy   42009.22364 / 5791.31912 / 15430.84532 km²
ellipse length/width  831.541290 / 477.053652 km
```

The B, C and D losing footprints are tangent-at-merge disks with respective
radii `43.2455532`, `70.0` and `48.2455532 km`, areas `5875.337063`,
`15393.804003` and `7312.476002 km²`, and ellipse diameters `86.4911064`,
`140.0` and `96.4911064 km`. The 2 km area/ellipse gates apply to the root and
all three losing linked objects, plus both rectangles. Report errors at 8, 4
and 2 km, but remove the unjustified requirement that each absolute error be
non-increasing. The frozen 2 km limits of 5% for area/length and 7.5% for width
remain unchanged.

These are fixture-definition and validity corrections, not terrain or
extractor tuning. Expected topology, retention, cap/gentle ordering, reference
thresholds and the H/C/G exclusion do not change.

## Evaluated amendment C: physical merge-support complex

**Date:** 2026-07-14, after the first analytic run stopped at its first failed
gate and before the remaining matrix continued.

The 8 km two-cone case placed the analytic contact `8.6698845 km` from the
polygon of the lower activation cell, failing amendment B's one-spacing
support-cell gate. That polygon alone is not the physical support of a graph
merge: the event is created by its faces to cells in the strictly higher,
already-active components. Judging only the lower cell mistakes a serialization
support for the causal adjacency.

Define the **merge-support complex** as the union of every flat-saddle cell and
every face-neighbor polygon whose sampled elevation is strictly above the
saddle level. The analytic contact must be within one center spacing of this
complex. Keep the `L * spacing` saddle-elevation bound unchanged. Tests must use
the frozen sampled field and graph adjacency to construct this complex; they
may not choose neighbors by proximity to the analytic answer.

This amendment corrects the physical meaning of an already registered
location check. It does not change the graph, surface, expected event, spacing,
threshold, or error limit. No later outcome from the matrix was available when
it was committed.

## Evaluated amendment D: merge-support sampling radius

**Date:** 2026-07-14, after the resumed matrix reached the 8 km linked C–D
event and before any finer resolution ran.

The physical merge-support complex from amendment C passed the two-cone event
but lay `8.7459305 km` from the linked C–D analytic contact, just beyond one
8 km center spacing. One spacing still omits the lattice covering radius: an
analytic contact lies in a regular Voronoi cell whose center is at most one
cell circumradius away, and the lower endpoint that activates a graph bridge
may be one face-neighbor spacing beyond that center.

Replace both cone-contact sampling bounds with the geometry-derived radius

```text
sampling_radius = spacing + cell_circumradius
```

The merge-support complex must lie within `sampling_radius` of the analytic
contact, and saddle elevation error must be no more than
`L * sampling_radius`. For a regular hex this is `(1 + 1/sqrt(3)) * spacing` at
every resolution; it is not fitted to the observed 8 km miss. No topology,
surface, phase, threshold or final morphology-accuracy gate changes.

## Preregistered amendment E: product-geometry adapter authority

**Date:** 2026-07-14, before the spherical product or projected irregular-cap
G0 adapters were implemented.

Adapter inventory found three authority conflicts that must be resolved before
code can produce canonical evidence.

First, spherical source vertex IDs are available only while adapting a product
`Tessellation`, while `EvaluationSurfaceGraphV0` stores physical coordinates
and cannot later validate ID-based polygon rotation. Canonical polygon starts
therefore use the lexicographically smallest canonical physical
`(x_bits,y_bits,z_bits)` vertex in every domain. Source vertex IDs remain the
authoritative way to discover shared edges, but do not enter evidence identity.
This also makes the hash invariant to a harmless renumbering of one shared
source vertex table.

Second, product adjacency equality means **validated neighbor-set equality**.
The adapter rejects out-of-range, self or duplicate stored neighbors, sorts a
copy, and requires exact equality with adjacency rebuilt from two-owner
consecutive polygon edges. Raw source neighbor order is not semantic and does
not have to equal canonical G0 CSR order. A nearest-generator orphan repair has
no polygon edge and therefore still fails `NonPhysicalAdjacency`.

Third, spherical point-radius checks use `endpoint_match_abs_km`. Stored cell
area is exactly the adapter's recomputed signed-`f64` solid-angle area; cached
`f32` product area is not a second authority. Per-cell winding/positive-area
validation precedes the registered closed-sphere sum against
`4*pi*radius_km^2`. Full-sphere G0 has no boundary segments. This checkpoint
implements and validates only G0; `build_surface_hierarchy_v0` continues to
reject spherical input until the separately underspecified morphology,
gradient and relief rules receive an executable amendment.

For the planar companion seam, native finite-volume measures remain evidence:

- retain `LandscapeMesh.cell_area_km2` after requiring it to match the explicit
  polygon's signed shoelace area within `planar_area_match_relative`;
- retain each native `f32` internal face width and distance only after requiring
  agreement with the explicit chord and center distance within two `f32`
  relative epsilons;
- require each boundary face center and width to match its explicit directed
  endpoints within `endpoint_match_abs_km`; and
- keep the native operator direction check against center displacement, but do
  not require a general projected polygon chord to be perpendicular to that
  displacement. Orthogonality remains a separately proven property of the
  regular-hex companion, not a property of arbitrary projected Voronoi cells.

The tangent-projected irregular cap must retain its exact directed endpoints at
their existing source-edge construction sites. Reconstructing them from a
midpoint, width and assumed perpendicular is forbidden.

## Preregistered amendment F: executable spherical S0 geometry

**Date:** 2026-07-14, after spherical G0 passed and before any spherical S0
outcome or product terrain was inspected.

The original spherical morphology text left several numerical choices
underspecified. Freeze the following operational definitions. They complete
the existing formulas; they do not add a terrain model, product threshold or
new evidence category.

For each member cell with directed unit polygon edges `a -> b`, first define its
physical centroid direction from the spherical polygon rather than substituting
the Voronoi generator:

```text
m_i = sum(angle(a,b) * normalize(a cross b))
u_i = normalize(m_i)
```

A zero or non-finite cell resultant is fatal after valid positive-area G0.
For member areas `A_i` and sphere radius `R`, define

```text
A = sum(A_i)
s = sum(A_i * u_i)
rho = length(s) / A
```

If `rho <= linear_rank_relative`, footprint geometry is the valid reportable
outcome `NonLocalGeometry`. Otherwise the footprint reference direction is
`c = s / length(s)` and its serialized physical centroid is `R*c`. This does
not change the structural elder tie-break: flat-batch centroids retain their
existing unnormalized area-weighted Cartesian definition.

Construct the deterministic right-handed tangent basis at any unit direction
`c` as follows:

1. choose the Cartesian axis with smallest absolute dot product with `c`, with
   ties ordered X, Y, Z;
2. `e1 = normalize(axis cross c)`;
3. flip `e1` when its first nonzero Cartesian component is negative; and
4. `e2 = c cross e1` after that sign choice.

For unit target direction `u`, define the azimuthal-equidistant log map:

```text
theta = atan2(length(c cross u), c dot u)
t = normalize(u - (c dot u) * c)
q = R * theta * t
(x,y) = (q dot e1, q dot e2)
```

Coincident directions map to `(0,0)`. A target is operationally antipodal when
`pi - theta <= linear_rank_relative` radians; its log map is unavailable.

Project every member cell polygon with that map. Each projected polygon must
retain positive signed area, have no repeated consecutive vertex and have no
intersection between non-adjacent closed edges. Antipodal or invalid projected
geometry yields `NonLocalGeometry`, not a fabricated metric. Otherwise multiply
all six raw polygon-moment terms by
`physical_cell_area / projected_signed_area` before summing. The rescaled area
must match the authoritative member-area sum within
`sphere_area_closure_relative`. Non-finite arithmetic or a covariance that is
negative beyond the registered rank tolerance is a fatal typed spherical-
moment error rather than a nonlocal outcome.

Spherical measurements serialize as:

```text
SphericalHighlandMeasurementsV0 {
    footprint_geometry:
        Local(SphericalLocalFootprintGeometryV0) | NonLocalGeometry,
    two_sweep_extent_km,
    mean_width_km,
    local_relief,
    rank_deficient_grade_cells,
    summit_caps,
}

SphericalLocalFootprintGeometryV0 {
    area_km2,
    centroid_km = R*c,
    projected_centroid_km = [x,y],
    tangent_covariance_km2 = [xx,xy,yx,yy],
    equivalent_ellipse_length_km,
    equivalent_ellipse_width_km,
    anisotropy,
    principal_axis,
    orientation_ambiguous,
    maximum_angular_radius_rad,
    spherical_nonlocal_warning,
}
```

The principal eigenvector is mapped back to the global tangent vector
`vx*e1 + vy*e2` and its first nonzero Cartesian component is made positive.
The maximum angular radius is the greatest `theta` over every member polygon
vertex; it sets `spherical_nonlocal_warning` only when strictly greater than
`spherical_nonlocal_radius_rad`. Two-sweep extent uses robust great-circle
center distance and the existing exact-tie/lower-cell-ID rule. It and
`mean_width = authoritative_area / extent` remain available even when tangent-
plane footprint geometry is `NonLocalGeometry`; zero extent yields two `None`
values.

Spherical grade uses the same registered weighted least-squares system as the
plane. At owner center `u_i`, build its deterministic tangent basis and use the
neighbor log-map coordinates as `q_ij`; retain
`w_ij = shared_face_length / center_distance`. An antipodal neighbor or
non-finite solve is fatal. Only the existing matrix rank rule makes grade
unavailable.

Spherical fixed-radius relief includes a scored candidate only when its robust
great-circle center distance is at most the registered physical radius. A
spatial index may conservatively overselect candidates, but the final test is
the exact arc formula. Full-sphere physical boundaries remain absent. Internal
scored/unscored faces are minor great-circle boundary arcs.

For unit query `p` and minor-arc endpoints `a,b`, let

```text
delta = angle(a,b)
n = normalize(a cross b)
```

Project `p` onto the great-circle plane and test both normalized projection
signs `q`. Define the oriented parameter

```text
t = atan2(n dot (a cross q), a dot q)
if t < 0: t += 2*pi
```

The projection is on the closed minor arc exactly when `t <= delta`. The
point-to-arc distance is the smallest query-to-valid-projection arc, or the
smaller endpoint arc if neither projection lies on the segment. A relief
neighborhood is truncated exactly when this distance is less than or equal to
its radius. Degenerate or antipodal boundary endpoints are invalid G0, not
repaired here.

Manufactured implementation must not inspect product elevation. It uses an
unjittered 2,048-site Fibonacci product Voronoi sphere, adapted through G0 and
uniformly rescaled with all length and area measures to a 100 km validation
radius. The end-to-end local fixture is

```text
center = the lowest-ID generator nearest +Z
d_i = great-circle distance(center, cell_i)
z_i = 0.40 km - 0.010 * d_i
closure = 0 km
all cells scored
```

It must produce one retained root with local spherical geometry, finite relief
and cap families, repeat-identical hierarchy bytes and no truncated relief on
the fully scored sphere. Private manufactured checks additionally require:

- a central log-affine field with grade `0.010` reconstructs that grade and is
  invariant under a rigid 3-D rotation;
- a tangent-plane elongated selection recovers its signless global orientation
  under rigid rotation, while a symmetric local cap is orientation ambiguous;
- an exactly balanced antipodal resultant reports `NonLocalGeometry` without
  suppressing relief, grade-validity or cap evidence;
- centers exactly inside, on and outside each registered relief radius obey
  inclusive great-circle membership; and
- boundary-arc interior and endpoint cases reproduce analytic angular
  distances and are rigid-rotation invariant.

No D0 arm, product landform or renderer output may be inspected until these
manufactured spherical checks pass.

## Decision

Implement only the common physical surface graph (**G0**) and the independent
surface peak–saddle hierarchy (**S0**) before constructing H, C or G.

The planar structural, unit-morphology and analytic 8/4/2 checkpoints now pass.
Their evidence and remaining scope are recorded in the
[structural-slice audit](../audits/landform-g0s0-structural-slice-2026-07-14.md),
the [morphology-slice audit](../audits/landform-g0s0-planar-morphology-2026-07-14.md),
and the [analytic audit](../audits/landform-g0s0-planar-analytic-2026-07-14.md).
The independent product-spherical and projected irregular-cap G0 adapters also
pass their physical-geometry and adversarial checks, recorded in the
[geometry-adapter audit](../audits/landform-g0-geometry-adapters-2026-07-14.md).
The preregistered spherical S0 seam now also passes its bounded synthetic-sphere
fixtures, recorded in the
[spherical morphology audit](../audits/landform-g0s0-spherical-morphology-2026-07-14.md).
This completes the manufactured G0/S0 checkpoint. Product elevation and
landforms remain uninspected, and no H/C/G surface has been observed.

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
counter-clockwise as viewed from outside the sphere. Under amendment E, every
polygon's first vertex is the lexicographically smallest canonical
`(x_bits,y_bits,z_bits)` vertex. Face endpoints are stored in the directed
polygon order. Reciprocal endpoints therefore reverse.

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
Canonical adjacency is rebuilt from polygon-edge ownership and its neighbor
sets must equal the validated stored tessellation neighbor sets exactly; a
nearest-generator orphan repair is therefore exposed as non-geometric instead
of silently entering S0.

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
- point-apex peak elevation error is no more than
  `L * cell_circumradius`, and its anchor displacement no more than one cell
  circumradius; cone-contact saddle elevation error is no more than `L` times
  the amendment-D sampling radius, with the analytic contact no farther than
  that radius from its merge-support complex. Broad-cap maxima use the support
  rule in amendment B.

For the rectangle and linked objects at 2 km, area and equivalent-ellipse
length relative errors must be at most 5%, and equivalent-ellipse width error
at most 7.5%. Record, but do not gate on, monotonic error across the non-nested
8/4/2 lattices.

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
