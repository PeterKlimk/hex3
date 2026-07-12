# Bounded orogen organization testbed — 2026-07-13

Status: Slice 1 implemented and numerically screened. Core invariants pass, but
erosion-budget and post-relaxation relief convergence fail; Slice 2 and product
integration are blocked. The representation question is now resolved for the
next rung, but its analytic contracts are not implemented. See the
[Slice 1 audit](../audits/orogen-testbed-slice1-2026-07-13.md) and
[channel/surface scaling decision](channel-surface-scaling-2026-07-13.md).

Research basis: [mountain-system design basis](mountain-system-design-basis-2026-07-13.md).

## Decision

Build a CPU-only bounded landscape testbed around a new dimensioned surface-graph
contract. Do not construct `World` or `FineWorld`, and do not expose the current
`ErosionState` as the new architecture.

The cheapest possible harness—feeding prescribed height envelopes through the
existing erosion function—would repeat O1–O3A. It cannot test time-varying rock
uplift, divide migration, capture, material inheritance or relaxation after
forcing stops. Prescribed-height erosion remains useful only as the locked
hold-and-carve control.

The testbed must support both bounded planar and spherical graph adapters. The
first experiments use a uniform planar hexagonal finite-volume patch because it
provides high spatial resolution, explicit boundaries and cheap convergence
tests. Product integration later uses the same graph operators through a
spherical `Tessellation` adapter.

## Ownership contract

```text
ScenarioCompiler
  owns mesh, boundary conditions, episode geometry, geology and forcing schedule

DeformationEvaluator
  owns time-dependent rock velocity fields; cannot write elevation

LandscapeSolver
  is the sole mutator of bedrock and optional mobile cover

DrainageSolver
  owns a revision-keyed derived routing/discharge cache; cannot repair terrain

LandformAnalyzer
  derives channels, basins, divides, ridges, passes and captures independently

Renderer/exporter
  consumes snapshots; cannot affect state or scoring
```

These are not direct height inputs in the coupled arms:

- collision work, crustal thickening or deformation-segment shape;
- uplift history or inherited weakness;
- drainage, ridge or valley skeletons;
- erosion potential, sediment load or isostatic response;
- relief exaggeration.

Only the neutral initial datum and declared boundary/base levels begin as
height. Thereafter authoritative surface height is evolved bedrock plus optional
mobile cover.

## Units

The testbed uses dimensioned values rather than relabeling the current
dimensionless erosion epoch:

| Quantity | Unit |
|---|---|
| Horizontal position/distance | km |
| Cell/face measure | km² / km |
| Bedrock, cover and base level | km |
| Time | Myr |
| Rock vertical/horizontal velocity | km/Myr, normally reported as mm/yr |
| Runoff depth rate | km/Myr, normally configured/reported as m/yr |
| Discharge | km³/Myr |
| Hillslope diffusivity | km²/Myr |
| Physical slope | grade = vertical km / horizontal km |
| Integrated material terms | km³ |

For every erosion law, the manifest records its discharge/support convention,
`m`, `n` and the derived units of `K`. C0 uses specific discharge `q` and an
effective areal law such as `E = K q^m S^n`; the P pathway control may use
accumulated discharge `Q` but cannot claim cell-volume export. No testbed value
may silently reuse a native elevation-per-radian product parameter.

## Minimum data contract

The following Rust shapes express the contract, not a requirement to preserve
these exact names:

```rust
struct LandscapeMesh {
    cell_center_km: Vec<DVec3>,
    cell_area_km2: Vec<f64>,
    edge_offsets: Vec<u32>,
    edge_neighbor: Vec<u32>,
    edge_distance_km: Vec<f32>,
    edge_face_width_km: Vec<f32>,
    edge_outward_tangent: Vec<Vec3>,
    boundary: Vec<BoundaryCondition>,
}

enum BoundaryCondition {
    Interior,
    Closed,
    OpenBaseLevel { elevation_km: f32 },
}

struct LandscapeState {
    time_myr: f64,
    revision: u64,
    mean_bedrock_elevation_km: Vec<f32>,
    mobile_cover_km: Option<Vec<f32>>,
    drainage: DrainageCache,
    ledger: LandscapeLedger,
}
```

`LandscapeMesh` is the only geometry consumed by routing, incision and
hillslopes. It can be compiled from a planar hex mesh or `Tessellation` without
forking the laws.

### Deformation history

```rust
struct DeformationEpisode {
    id: EpisodeId,
    active_myr: Range<f64>,
    ramp_myr: f32,
    rock_volume_rate_km3_myr: TimeSeries,
    segment_shares: Vec<SegmentShare>,
}

struct DeformationSegment {
    id: SegmentId,
    geometry: SegmentGeometry,
    width_km: f32,
    along_strike_taper: Taper,
    vergence: Vec3,
    links: Vec<SegmentLink>,
}

struct DeformationFrame {
    rock_vertical_rate_km_myr: Vec<f32>,
    horizontal_velocity_km_myr: Vec<Vec3>,
    dominant_episode: Vec<Option<EpisodeId>>,
}
```

Each segment compiles to an area-normalized support stencil. Episode shares sum
to one; overlap cannot create extra forcing. Geological affinity may redistribute
the stencil, after which it is renormalized to the episode's declared integrated
rock-volume rate. The output is velocity, never accumulated terrain height.

Horizontal velocity is present in the contract but zero in the first rung. When
enabled, it must move bedrock, cover and material identity together with an
explicit boundary ledger.

### Shared geology

```rust
struct GeologicProvince {
    id: ProvinceId,
    deformation_affinity: f32,
    channel_erodibility: f32,
    hillslope_diffusivity_km2_myr: f32,
    critical_slope_grade: f32,
    fabric_axis: Vec3,
    fabric_strength: f32,
}
```

One `province_by_cell` field supplies persistent identity. The same province
must condition deformation localization, incision and hillslope behavior. A
separate height/noise mask does not implement the hypothesis.

### Drainage

The derived cache records current receivers, discharge, basin/outlet identity
and routing age. Routing must:

- consume the current evolved surface;
- use runoff × cell area as local water supply;
- terminate at declared outlets or explicit closed sinks;
- produce an acyclic graph;
- recompute often enough to permit divide migration and capture;
- use filling only to select routes, never mutate authoritative elevation;
- expose any stale-routing policy and error bound.

For the C0 coupled arm, routing is a finite-volume face-flux calculation and
fluvial forcing uses a validated specific-discharge field, not raw accumulated
cell discharge. Multiple-flow direction is the research default; SFD remains a
control. Fixed convergence cases terminate at physical outlet portals whose
identity, geometry and base level do not change with cell count. Dirichlet base
level is applied at the boundary face/ghost state, not by pinning an interior
row.

Current drainage-integration cuts are intentionally absent. A later terrain
repair arm may add them explicitly with separate before/after state.

### Surface operators

One dimensioned step is:

```text
evaluate deformation at t + dt/2
choose and report a stable dt
apply rock vertical velocity
route current surface and accumulate discharge
apply C0 effective areal fluvial denudation
apply conservative nonlinear hillslope transport
optionally transport/deposit mobile cover
close ledgers and advance time/revision
```

The first rung uses physical grade and a regularized nonlinear transport law
approaching a material critical slope. Linear diffusion may remain a control.
No local Airy rebound is included. A later broad load-response operator must be
separately conservative and cannot be inferred cell by cell from erosion.

The first rung also omits mobile sediment: detached bedrock is exported. Add one
nonnegative cover quantity only after the uplift–drainage–hillslope system is
understood. Its ledger must then close bedrock detachment = cover storage +
deposition + outlet export.

The authoritative C0 elevation is a finite-volume cell mean. Stream power is
interpreted as effective areal fluvial denudation, so its volume term is
`E * cell_area * dt`. This representation does not claim resolved channel-bed
height, width, gorge geometry or valley cross-section. A path-elevation solver
may be retained as a non-volumetric control. A dual mean/bed/width/reach state
is deferred until missing bed–interfluve disequilibrium is demonstrated.

The first coupled smoke uses the frozen normalized direct-q regime in the
[channel/surface scaling decision](channel-surface-scaling-2026-07-13.md):
`R0=500 km/Myr`, `q0=50,000 km²/Myr`, `S0=0.02`, `E0=0.1 km/Myr`,
`m=n=1`, algebraic `K=1e-4 km^-1`, `D=0.1 km²/Myr`, `Sc=0.7`. This is an
explicit transient response-time prior, not the default or a visual fit. The
legacy Q-form `K=0.03` has different dimensions and cannot be transferred.

## Fixed domain

- Planar full-hex finite-volume patch selected around a `960 × 640 km` target
  extent. The union of complete cells is the conservative domain; every run
  records its actual area, exposed boundary and offset from the target.
- Scored interior: central `640 × 320 km`; all metrics exclude the buffer.
- Reference cell spacing: `4 km`; convergence meshes at `8 km` and `2 km`.
- North/south: fixed portal IDs and projected coordinate spans on genuine
  exposed hex faces for convergence cases; physical sawtooth arc length may
  refine. A continuously open coast is a separate later experiment.
- East/west: closed/no-flux.
- Initial terrain: neutral low-relief outward-draining surface plus one fixed
  deterministic perturbation below 10 m; no arm-specific noise.
- Common initial material except in the province case.
- Duration: 10 Myr; forcing active from 0–6 Myr, relaxation from 6–10 Myr.
- Reference cumulative central rock uplift: approximately 3 km, corresponding
  to 0.5 mm/yr for 6 Myr. Exact forcing is frozen before arm comparison.
- Full scalar snapshots every 0.25 Myr; object checkpoints at
  `0, 1, 3−, 3+, 4.5, 6, 8, 10 Myr`.

The forcing axes should not align exactly with the hex lattice. A rotated-mesh
control must show that the claimed range/channel organization is not a grid
direction artifact.

## Causal cases

| ID | Question | Construction |
|---|---|---|
| U | What does an unstructured physical null produce? | One smooth finite-width uplift block with a flat along-strike interior |
| L | Does linked forcing create range ends, massifs and a transfer saddle? | Two 350-km tapered, slightly en-echelon segments with an 80-km overlap/transfer zone; same integrated work as U |
| P | Does inherited geology create mutually consistent tectonic and drainage organization? | L plus one 60-km oblique weak province crossing the transfer zone |
| R | Does forcing history leave memory, migration and capture? | Initial segment active 0–3 Myr; replacement shifted 80 km cross-strike active 3–6 Myr; no terrain reset |
| R0 | Is history more than final cumulative forcing? | R's time-integrated spatial uplift distributed continuously over 0–6 Myr |
| W/D | Does climate condition process rather than paint style? | L at 2× and 0.5× reference runoff; identical tectonic work/material |

U is a scientific null, not a candidate expected to look good. `R − R0` is the
central history discriminator.

## Representation arms

| ID | Representation | Role |
|---|---|---|
| H | Current-style height envelope plus repeated rebuild | Locked hold-and-carve baseline |
| S | Synthetic authoritative drainage graph and compatible terrain | Cheap topology control |
| C | Time-varying rock uplift + evolving drainage + stream-power incision + nonlinear hillslopes | Preferred minimum physical hypothesis |
| D | C plus shared geological province state | Likely target extension |
| E | Explicit joint ridge/pass/valley/drainage skeleton conditioned on episodes | Structural upper bound/fallback hack |

D must be bit-identical to C when no province is declared. An arm's native graph
is reported but never used to score itself; one independent analyzer extracts
objects from every evolved surface.

The complete reference matrix is H/S/C/E for U, L, R, R0, W and D, plus
H/S/C/D/E for P: 29 runs before convergence repeats. Implementation proceeds in
slices and does not run this matrix until the numerical core passes U and L.

## Matching contract

Do not equalize final peak height, plateau coverage or visual relief.

Every arm receives the same:

- mesh, initial terrain, outlets and scored window;
- episode schedule and exact integrated uplift-work ledger;
- runoff and geological inputs;
- duration and snapshot times;
- maximum segment, province and seeded-outlet count;
- deterministic perturbation, with no arm-specific random fields.

Synthetic arms have a preregistered graph-complexity cap and may use only the
declared segments, province and outlet seeds. Relief-normalized renders are
allowed as secondary morphology views, never physical scoring inputs.

Each arm receives one literature- or dimensionally justified regime. Stability
fixes are allowed; changing morphology parameters after viewing results creates
a new registered experiment.

## Independent outputs

Per run:

```text
artifacts/orogen-testbed/<run-id>/
  manifest.json
  metrics.ndjson
  checkpoints/<time>.bin.gz
  objects/<time>.json
  views/<field>-<time>.png
  profiles.json
  summary.json
```

The manifest records revision/dirty state, mesh/scenario/solver hashes, units,
arm, case, seed, timestep policy, parameters, thread count and wall time.

Required time-resolved fields are rock-uplift rate/cumulative uplift, bedrock and
surface change, runoff/discharge/routing, incision, hillslope flux, slope/critical
slope ratio, province/material identity, and all ledger terms. Add cover,
deposition and sediment export only when the sediment rung exists.

The common analyzer derives:

- range and massif components;
- watershed divides and ridge graph;
- saddles and passes;
- basins and persistent outlets;
- channels and Strahler order;
- valleys and longitudinal profiles;
- capture events and divide trajectories.

Views use locked plan cameras and ranges. Primary sheets show physical
elevation/hillshade, forcing, channels/divides/passes, slope-threshold ratio and
erosion/transport across arms and checkpoint times. Cartographic relief may be
shown separately but cannot establish success.

## Metrics and causal contrasts

Keep a vector of evidence; do not collapse it into one quality score.

### Forcing correspondence

- range-axis overlap/orientation with active segments;
- range-end distance from segment termination;
- massif relief and transfer-zone saddle depth;
- pass occurrence/connectivity through the transfer zone;
- uplift and denudation attribution per episode.

### Drainage and topology

- basin-area hierarchy and fragmentation;
- Strahler/Horton structure and trunk persistence;
- longitudinal/transverse channel orientation;
- ridge–channel dual consistency;
- divide displacement through time;
- capture timing and transferred basin area;
- crossings of active ranges and inherited structures.

### Morphology

- 25/50/100-km relief;
- summit-cap and broad-flat measures;
- slope/curvature distributions and critical-slope occupancy;
- pass-to-summit relief ratio;
- range elongation, branching and internal massif count.

Interpret the registered contrasts:

- `L − U`: organization due to linked forcing;
- `P:D − P:C`: value of shared geological inheritance;
- `R − R0`: value of temporal history;
- `W − D`: climate sensitivity at fixed tectonics;
- each physical arm relative to H, bracketed by S and E.

An effect counts only when it exceeds resolution and timestep uncertainty.

## Invariants and numerical gates

- Arrays match mesh size and remain finite; cover, when present, is nonnegative.
- Segment stencils integrate to one; episode shares sum to one; integrated
  uplift equals the schedule independently of resolution.
- Internal face flux is antisymmetric; hillslopes change volume only at open
  boundaries.
- Routing face transfers connect neighbors, the directed flux graph is acyclic,
  and portal outflow plus declared closed storage equals runoff supply.
- Filling/routing never changes physical elevation.
- Surface-volume change closes against uplift, detachment, storage and export
  to `1e-6` of cumulative uplift volume, or the run is invalid.
- Identical seed/config/thread count repeats bit-for-bit; cross-thread results
  preserve major objects within floating-point tolerance.
- Halving timestep changes major scalar summaries by less than 5%, object
  measures by less than 10%, and event time by at most one output interval.
- Error decreases monotonically from 8→4→2 km. Between 4 and 2 km, major-object
  measures change by less than 15%, major-trunk overlap exceeds 0.7, and
  principal range/pass/basin identities persist.
- No scored result may depend on a buffer boundary or mesh orientation.
- The timestep audit reports uplift, incision, hillslope and future-advection
  limits separately. A run stabilized mainly by clamps fails.

These are initial numerical gates, not geological calibration targets. Relaxing
one requires diagnosing which claimed truth exceeds the testbed's resolution.

## Falsification and stop conditions

Reject an organization owner when any applies:

- relief changes without range/drainage/divide object changes;
- linked forcing does not create a reproducible termination and transfer saddle;
- reorganization produces no persistent adjustment, divide migration or basin
  transfer beyond numerical uncertainty;
- wet/dry forcing merely rescales a style rather than changing rates and
  adjustment time;
- a shared province lacks coherent deformation and drainage consequences;
- sharpness requires widespread slope-limit violation or global rounding;
- success disappears under timestep, resolution or mesh rotation;
- case-specific fitting is required;
- the arm's native semantic graph materially disagrees with the independent
  terrain-derived graph;
- S or E dominates a physical arm functionally and visually while costing over
  an order of magnitude less.

Do not run global seeds until one physical arm:

1. passes L, R/R0 and W/D;
2. improves at least three independent object families over H beyond numerical
   uncertainty;
3. demonstrates a province benefit in D or evidence that province state is
   unnecessary;
4. provides reusable channel/divide/range objects;
5. has a defensible compute/benefit position relative to S and E.

If C fails while S/E succeeds, revisit missing physical state before tuning. If
all arms fail, the testbed or analyzer is inadequate. If D alone succeeds,
shared geological inheritance is minimum state rather than optional detail.

## Implementation sequence

### Slice 1 — dimensioned coupled null

- Add experimental `world::landscape` module with `LandscapeMesh`, boundaries,
  dimensioned state, forcing frames, routing, incision, nonlinear hillslopes and
  ledgers.
- Add a uniform planar hex-mesh constructor and later a `Tessellation` adapter.
- Add two normalized linked-segment stencils with vertical uplift only.
- Add `src/bin/orogen_testbed.rs` with U and L under C.
- Export manifests, ledgers and scalar checkpoints.
- Pass determinism, conservation, timestep and 8/4-km smoke convergence.

This slice proves infrastructure and causal response; it cannot choose the
product architecture by itself.

### Slice 1R — representation and analytic repair

- Make C0 state explicitly mean finite-volume bedrock elevation.
- Replace per-cell outlet semantics with fixed physical portals and
  boundary-face/ghost conditions.
- Use the actual union of complete hex control volumes. Do not treat a nominal
  rectangle drawn through full boundary cells as finite-volume geometry; exact
  rectangular analytic flux uses a small 1-D/Cartesian fixture.
- Implement conservative face-flux routing and validate a continuum specific
  discharge field on plane, radial, convergent and rotated meshes.
- Separate the existing implicit incision routine as a P/pathway reference;
  do not assign its path change a cell-volume meaning.
- Implement C0 effective areal denudation and pass prescribed-field volume and
  relief convergence at 8/4/2 km.
- Pass depression/flat routing without mutating the physical surface.
- Rerun U/L only after the analytic ladder passes.

This slice follows the
[channel/surface scaling decision](channel-surface-scaling-2026-07-13.md).
It is a prerequisite for Slice 2, not parameter tuning within Slice 1.

### Slice 2 — independent objects and controls

- Add common basin/channel/divide/ridge/pass extraction and time identity.
- Add H and S controls, locked before comparative viewing.
- Run U/L and reject C if linked forcing changes only relief.

### Slice 3 — history and inheritance

- Add R/R0 and capture/divide tracking.
- Add the shared weak province and D.
- Add wet/dry forcing.
- Run the 29-case reference matrix only after numerical gates pass.

### Slice 4 — upper bound and optional missing state

- Add E only as a structural upper bound.
- Add one mobile cover/sediment state only if transient failures justify it.
- Add horizontal advection only by moving bedrock, cover and geology together.
- Add spherical adapter and planet transfer only after an architecture passes.

Do not add current `tectonic_thickening`, `arc`, `collision`, `coarse_target`,
`emergent_uplift_shape`, drainage integration, local isostatic rebound, glacial
erosion or presentation controls to Slice 1.
