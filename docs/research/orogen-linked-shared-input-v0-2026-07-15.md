# Linked orogen shared-input manifest V0

**Date:** 2026-07-15

**Status:** preregistered executable contract; not implemented or evaluated

**Parents:** [landscape organization strategy](../landscape-strategy.md),
[bounded orogen organization testbed](orogen-testbed-spec-2026-07-13.md),
[common planar evidence-core V0](landform-common-core-v0-2026-07-15.md)

## Decision

Materialize one immutable, arm-neutral linked-case input bundle at 8, 4 and
2 km before preregistering or implementing H, C or G. The bundle binds what the
three terrain organizations are allowed to receive. It does not turn those
inputs into terrain and it does not decide which organization wins.

This checkpoint corrects three ambiguities in the older testbed material:

1. use the coordinate-defined parabolic-taper initial surface with no pinned
   portal cells, not the legacy linearly tapered routing-bearing state;
2. declare one homogeneous present base substrate, without importing any arm's
   erodibility, diffusivity or critical-slope parameters; and
3. freeze both whole-graph and central-window candidate masks, while deferring
   the accepted evaluation population to the organization-comparison contract.

The manifest is an input identity, not a supposedly neutral pre-arm landform.
S0, D0, O0a and O0b require an authoritative final surface and therefore begin
only after an arm has produced one.

## Scope and semantic nonclaims

The semantic bundle owns:

- the exact centered planar mesh, lattice phase and physical boundary faces;
- north/south outlet portal declarations and their compiled face ownership;
- the declarative linked deformation scenario;
- the compiler identity, normalized support stencils and evaluated forcing
  witnesses;
- the analytic activity and positive rock-volume-input ledgers;
- one raw initial mean-elevation field;
- one uniform local runoff-supply field;
- one homogeneous base-substrate membership field;
- whole-graph and central-window candidate masks; and
- units, schema versions and deterministic hashes.

It makes no claim about:

- final elevation, range or river morphology, drainage, routing or discharge;
- a natural-kind range end, transfer low, pass, ridge, valley or catchment;
- arm-specific application or retention of the common vertical-rate and
  displacement fields as authoritative height, uplift, graph priority or any
  other terrain opportunity;
- H calibrated steps, C integration policy or G construction passes;
- terrain snapshot times, including the older undefined `3-`/`3+` pair;
- the accepted scoring population or landform-quality thresholds;
- opportunity calibration, arm admission or the comparison resource ceiling;
- product/spherical correspondence, persistent identity or renderer state; or
- promotion of the existing forcing compiler as a complete tectonic model.

The segment links and vergence are retained declarative causal metadata. In the
current compiler they do not affect support or evaluated fields, and horizontal
velocity is identically zero. The manifest must disclose that fact rather than
imply an implemented transfer or kinematic response. The visible low between
segments, if present in the forcing, follows only from their offset, finite
support and along-strike taper.

## Frozen domain and mesh

For each spacing in the exact order `[8.0, 4.0, 2.0] km`, construct a fresh:

```rust
LandscapeMesh::uniform_planar_hex(960.0, 640.0, spacing_km)
```

This is the current centered pointy-row full-hex patch:

```text
row_step = spacing * sqrt(3) / 2
columns  = round(960 / spacing)
rows     = round(640 / row_step)
y0       = -(rows - 1) * row_step / 2
even-row x phase = -spacing / 4
odd-row  x phase = +spacing / 4
cell area          = sqrt(3) * spacing^2 / 2
internal face width = spacing / sqrt(3)
```

Cells are row-major. The constructor's fixed six-neighbour discovery order is
semantic because it fixes CSR bytes and ordered reductions. The full serialized
`LandscapeMesh`, not a prose reconstruction or nominal rectangle, is the
authoritative geometry.

The expected integer topology is:

| spacing | rows x columns | cells | directed internal edges | raw exposed faces | split boundary records |
|---:|---:|---:|---:|---:|---:|
| 8 km | 92 x 120 | 11,040 | 65,394 | 846 | 848 |
| 4 km | 185 x 240 | 44,400 | 264,702 | 1,698 | 1,700 |
| 2 km | 370 x 480 | 177,600 | 1,062,202 | 3,398 | 3,400 |

The requested rectangle is not an exact area or perimeter claim. The generated
bundle reports the ordered floating sums and bit patterns for actual cell area,
physical boundary arc, projected portal coverage and physical open arc. Decimal
inventory values may be printed for inspection but are not hard-coded as exact
floating fixtures.

### Portals

Freeze the constructor's existing full-width declarations:

| ID | side | declared span | base level |
|---:|---|---:|---:|
| 0 | south | `[-480, 480] km` | `0 km` |
| 1 | north | `[-480, 480] km` | `0 km` |

East and west are closed. The bundle retains all physical boundary-face
segments, including projected width, physical width, owner cell and portal ID.
Projected portal coverage and physical sawtooth boundary arc are distinct
measures and must be reported separately.

Expected open-face records total 480/960/1,920. Unique open-face owner counts
are 120/240/480 per portal and 240/480/960 in total at 8/4/2 km. Compatibility
cell boundary flags are
retained as mesh evidence; they do not authorize pinning the initial or evolved
surface to base level.

## Frozen linked forcing

Use `linked_scenario()` with scenario ID `L`:

| segment | start km | end km | width | along taper | vergence | link |
|---:|---:|---:|---:|---|---|---|
| 0 | `(-360,-65)` | `(-10,-22)` | 100 km | cosine ends, fraction 0.22 | +Y | transfer to 1 |
| 1 | `(-70,22)` | `(280,65)` | 100 km | cosine ends, fraction 0.22 | +Y | transfer to 0 |

Episode 0 declares the interval `0.0..6.0 Myr`; the evaluator has closed-end
support but its smoothstep makes the field exactly zero at both endpoints and
positive only on `(0,6)`. It has `0.25 Myr` end ramps, total declared
rock-volume rate `17,500 km3/Myr`, and assigns a share of `0.5` to each
segment. The case observation horizon is
`10 Myr`: forcing occupies 0--6 and the shared forcing field is zero from
6--10. This horizon does not impose physical chronology on H or G.

### Compiler identity

Register compiler ID `linked-cosine-support-area-normalized-v0` with the current
ordered f64 algorithm. For cell centre `x`, segment start `a`, unit axis `e`,
length `L`, width `W` and cosine-end fraction `f`:

```text
along = dot(x - a, e) / L
cross = length((x - a) - e * dot(x - a, e))

cross_weight = 0,                                  cross >= W/2
             = 0.5 * (1 + cos(pi * cross/(W/2))), otherwise

along_weight = 0,                                  along outside [0,1]
             = 0.5 * (1 - cos(pi * along/f)),     along < f
             = 0.5 * (1 - cos(pi * (1-along)/f)), along > 1-f
             = 1,                                  otherwise

raw = cross_weight * along_weight
support[i] = raw[i] / sum_j(raw[j] * cell_area[j])
```

The existing clamp `f = clamp(end_fraction, 1e-6, 0.5)` is part of the compiler
identity even though this scenario supplies 0.22. Each f64 stencil must have
units `km^-2`, be nonnegative and area-integrate to one within the registered
ordered-sum tolerance. Overlap adds the two assigned contributions; it creates
no extra declared work.

The implementation requires a narrow read-only export of the compiled
`SupportStencil` values. Do not duplicate the compiler in the manifest binary.

### Activity and work ledger

Register activity policy `smoothstep-episode-ends-v0`:

```text
activity(t) = 0 outside [start,end]
r = min(max(ramp, 0), (end-start)/2)
activity(t) = 1 inside [start,end] when r == 0
edge = clamp(min((t-start)/r, (end-t)/r), 0, 1), otherwise
activity(t) = edge^2 * (3 - 2*edge), otherwise
```

For this episode, the exact analytic activity-time integral is `5.75 Myr`.
The declared positive rock-volume ledger is therefore:

```text
total:     17,500 * 5.75       = 100,625 km3
segment 0: 17,500 * 5.75 * 0.5 = 50,312.5 km3
segment 1: 17,500 * 5.75 * 0.5 = 50,312.5 km3
```

For each resolution, materialize a cumulative rock-displacement input array:

```text
displacement_km[i]
  = sum_segments(50,312.5 km3 * support_segment[i] km^-2)
```

Its area-weighted sum must close to `100,625 km3`. This is integrated forcing,
not final terrain height, solid uplift retained by an arm, or an erosion
opportunity calibration.

### Evaluated forcing witnesses

Evaluate the existing `DeformationEvaluator` at these exact forcing-oracle
times, in order:

```text
[0, 0.125, 0.25, 3, 5.75, 5.875, 6, 8, 10] Myr
```

Expected activities are `[0, 0.5, 1, 1, 1, 0.5, 0, 0, 0]`. These are compiler
witnesses, not terrain snapshot times. For every witness, retain the time,
activity, exact hash of the f32 vertical-rate array, exact hash of the f32
horizontal-velocity array, exact hash of dominant-episode values and the
ordered f64 area-integrated rate. Horizontal velocity must be bitwise zero.
At activity one, the integrated f32 rate must differ from `17,500 km3/Myr` by
less than `2e-7` relative at all three registered resolutions.

Keep three ledgers distinct:

1. analytic declared activity and volume;
2. compiled f64 stencil normalization and cumulative displacement; and
3. evaluated f32 frame integration residuals.

## Frozen initial surface

Register generator ID `linked-low-relief-parabolic-taper-v0`, seed `12345` and
the following coordinate-defined f64 field. Coordinates and elevation are km:

```text
yhat  = clamp(2*y / 640, -1, 1)
taper = max(1 - yhat^2, 0)

z = taper * (
      0.020
    + 0.0020 * sin( 0.071*x + 0.043*y + p1)
    + 0.0015 * sin(-0.038*x + 0.063*y + p2)
    + 0.0010 * sin( 0.027*x - 0.052*y + p3)
)
```

For stream `s` in `[1,2,3]`:

```text
v0 = seed XOR wrapping_mul(s, 0x9e3779b97f4a7c15)
v1 = wrapping_add(v0, 0x9e3779b97f4a7c15)
v2 = wrapping_mul(v1 XOR (v1 >> 30), 0xbf58476d1ce4e5b9)
v3 = wrapping_mul(v2 XOR (v2 >> 27), 0x94d049bb133111eb)
r  = v3 XOR (v3 >> 31)
u  = f64(r >> 11) / f64(1_u64 << 53)
p  = u * TAU
```

The current seed produces phases approximately
`[4.4037979935143605, 1.4440520298135509, 2.055503350315996]` radians; the
algorithm and materialized array, not the printed decimals, are authoritative.

Store only the raw initial mean-elevation array. Do not construct or hash a
`LandscapeState`, receiver forest, fill surface, discharge cache or portal-cell
pin. The field is a refinement-continuous low-relief symmetry breaker, not an
arm output or accepted terrain grammar.

## Frozen runoff and material

Register runoff ID `uniform-depth-local-supply-v0`:

```text
runoff_depth_rate = 500 km/Myr = 0.5 m/year
local_runoff_supply[i] = 500 km/Myr * cell_area_km2[i]
```

The f64 array is materialized and hashed. Its ordered sum must equal
`500 * actual_domain_area_km2` under the registered reduction tolerance.
Receivers, routed discharge, specific-discharge support, lake storage and
outlet assignment are arm/evidence outputs and are forbidden here.

Register material ID `uniform-present-base-material-v0`. Materialize one
`base_material_present` boolean per cell, all `true`. This is only substrate/
domain membership: it does not mean that a cell participates in, or is eligible
for, an arm's erosion, uplift or graph operation. No province IDs, weak zones,
layers, erodibility, diffusivity, critical slope or response coefficient are
implied. Geological inheritance is a possible later comparison rung, not a
hidden shared prior.

## Candidate evaluation geometry

Materialize and hash both terrain-independent masks:

```text
whole_graph[i]   = true
central_window[i] = abs(center.x) <= 320 && abs(center.y) <= 160
```

The central predicate is a new frozen choice replacing the older prose-only
"central 640 x 320 km" statement. Expected central cell counts are
3,680/14,880/58,880 at 8/4/2 km. Nominal cell-area products are approximately
203,966.3031/206,183.3281/203,966.3031 km2; the generated ordered sums and bits
are authoritative.

This is cell-centre inclusion, not full-cell containment; boundary cells may
cross the nominal window edge.

Neither mask is selected here as the accepted comparison population. Current
common evidence V0 accepts only `WholeGraph`; selecting the central window
would require a separately preregistered partial-population schema and boundary
semantics. That decision belongs to the organization comparison.

## Artifact and encoding

Rust field order is wire order. Names below are normative; the implementation
may add private helpers but not semantic fields without amending this contract.

```rust
pub struct LinkedSharedInputBundleV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub declaration: LinkedInputDeclarationV0,
    pub resolutions: Vec<LinkedResolutionInputV0>,
    pub derived_bundle_hash: u64,
}

pub struct LinkedInputDeclarationV0 {
    pub requested_width_km: f64,
    pub requested_height_km: f64,
    pub spacings_km: Vec<f64>,
    pub case_horizon_myr: f64,
    pub mesh_constructor_id: String,
    pub scenario: LandscapeScenario,
    pub forcing_compiler_id: String,
    pub forcing_compiler_semantics: ForcingCompilerSemanticsV0,
    pub activity_policy_id: String,
    pub forcing_oracle_times_myr: Vec<f64>,
    pub analytic_activity_integral_myr: f64,
    pub analytic_rock_volume_km3: f64,
    pub work_ledgers: Vec<DeclaredWorkLedgerV0>,
    pub initial_surface: InitialSurfaceDeclarationV0,
    pub runoff: RunoffDeclarationV0,
    pub material_id: String,
    pub candidate_geometry_id: String,
    pub units: LinkedInputUnitsV0,
}

pub struct LinkedResolutionInputV0 {
    pub nominal_spacing_km: f64,
    pub mesh: LandscapeMesh,
    pub initial_elevation_km: Vec<f64>,
    pub local_runoff_supply_km3_myr: Vec<f64>,
    pub base_material_present: Vec<bool>,
    pub whole_graph_candidate: Vec<bool>,
    pub central_window_candidate: Vec<bool>,
    pub compiled_stencils: Vec<SupportStencil>,
    pub cumulative_rock_displacement_km: Vec<f64>,
    pub frame_witnesses: Vec<ForcingFrameWitnessV0>,
    pub summary: LinkedResolutionSummaryV0,
    pub component_hashes: LinkedInputComponentHashesV0,
    pub derived_resolution_hash: u64,
}

pub struct InitialSurfaceDeclarationV0 {
    pub generator_id: String,
    pub seed: u64,
    pub phase_streams: [u64; 3],
    pub base_elevation_km: f64,
    pub amplitudes_km: [f64; 3],
    pub wave_vectors_per_km: [[f64; 2]; 3],
    pub taper_full_height_km: f64,
}

pub struct RunoffDeclarationV0 {
    pub generator_id: String,
    pub depth_rate_km_myr: f64,
}

pub struct ForcingCompilerSemanticsV0 {
    pub field_consumed: Vec<String>,
    pub identity_or_output_only: Vec<String>,
    pub retained_but_ignored: Vec<String>,
    pub horizontal_velocity_policy: String,
}

pub struct DeclaredWorkLedgerV0 {
    pub episode_id: EpisodeId,
    pub segment_id: Option<SegmentId>,
    pub activity_integral_myr: f64,
    pub share: f64,
    pub positive_rock_volume_km3: f64,
}

pub struct LinkedInputUnitsV0 {
    pub coordinate: String,
    pub elevation: String,
    pub area: String,
    pub time: String,
    pub vertical_rate: String,
    pub runoff_depth_rate: String,
    pub runoff_supply: String,
    pub volume: String,
    pub support: String,
}

pub struct ForcingFrameWitnessV0 {
    pub time_myr: f64,
    pub expected_activity: f64,
    pub vertical_rate_hash: u64,
    pub horizontal_velocity_hash: u64,
    pub dominant_episode_hash: u64,
    pub integrated_rate_km3_myr: f64,
}

pub struct LinkedResolutionSummaryV0 {
    pub cell_count: u64,
    pub directed_edge_count: u64,
    pub raw_exposed_face_count: u64,
    pub split_boundary_record_count: u64,
    pub actual_domain_area_km2: f64,
    pub physical_boundary_arc_km: f64,
    pub center_bounds_km: [f64; 4],
    pub portal_summaries: Vec<LinkedPortalSummaryV0>,
    pub initial_min_km: f64,
    pub initial_max_km: f64,
    pub local_runoff_total_km3_myr: f64,
    pub base_material_present_count: u64,
    pub whole_graph_count: u64,
    pub central_window_count: u64,
    pub central_window_area_km2: f64,
    pub cumulative_rock_volume_km3: f64,
    pub stencil_summaries: Vec<LinkedStencilSummaryV0>,
}

pub struct LinkedPortalSummaryV0 {
    pub portal_id: OutletPortalId,
    pub face_record_count: u64,
    pub owner_cell_count: u64,
    pub projected_coverage_km: f64,
    pub physical_open_arc_km: f64,
}

pub struct LinkedStencilSummaryV0 {
    pub segment_id: SegmentId,
    pub support_cell_count: u64,
    pub minimum_weight_per_km2: f64,
    pub maximum_weight_per_km2: f64,
    pub area_integral: f64,
}

pub struct LinkedInputComponentHashesV0 {
    pub mesh_hash: u64,
    pub initial_elevation_hash: u64,
    pub local_runoff_hash: u64,
    pub base_material_present_hash: u64,
    pub whole_graph_candidate_hash: u64,
    pub central_window_candidate_hash: u64,
    pub compiled_stencils_hash: u64,
    pub cumulative_rock_displacement_hash: u64,
    pub frame_witnesses_hash: u64,
}
```

`center_bounds_km` is ordered `[min_x, max_x, min_y, max_y]`. Unit strings are
exactly `km`, `km`, `km2`, `Myr`, `km/Myr`, `km/Myr`, `km3/Myr`, `km3` and
`km^-2` in the field order above. `phase_streams` is exactly `[1,2,3]`; the remaining
initial-surface fields reproduce the frozen equation and are validated against
it rather than serving as an alternate configurable generator.
`taper_full_height_km` is exactly `640` and is consumed as
`yhat = 2*y/taper_full_height_km`.

Other exact declaration values not already tabulated are:

```text
requested_width_km = 960
requested_height_km = 640
spacings_km = [8,4,2]
case_horizon_myr = 10
mesh_constructor_id = "centered-pointy-row-full-hex-v0"
initial generator/base/amplitudes =
  "linked-low-relief-parabolic-taper-v0", 0.020, [0.0020,0.0015,0.0010]
initial wave_vectors_per_km = [[0.071,0.043],[-0.038,0.063],[0.027,-0.052]]
runoff generator/depth = "uniform-depth-local-supply-v0", 500
material_id = "uniform-present-base-material-v0"
candidate_geometry_id = "whole-and-central-cell-centre-candidates-v0"
```

The compiler-semantics vectors are exact and ordered:

```text
field_consumed = [
  "segment.id", "segment.geometry", "segment.width_km",
  "segment.along_strike_taper",
  "episode.active_myr", "episode.ramp_myr",
  "episode.rock_volume_rate_km3_myr",
  "episode.segment_shares.segment_id", "episode.segment_shares.share"
]
identity_or_output_only = ["episode.id"]
retained_but_ignored = ["scenario.id", "segment.vergence", "segment.links"]
horizontal_velocity_policy = "identically-zero-v0"
```

The three work ledgers are ordered total, segment 0, segment 1. The total uses
`segment_id=None` and share `1`; segment rows use `Some(0/1)` and share `0.5`.
All use episode 0 and activity integral 5.75; their volumes are 100,625,
50,312.5 and 50,312.5 km3. `forcing_oracle_times_myr` is exactly the nine-time
list registered above.

All semantic hashes use schema ID `orogen-linked-shared-input-v0`, hash ID
`fnv1a64-bincode-fixint-le-v0`, fixed-width little-endian bincode and FNV-1a-64.
Derived hashes are excluded from their own preimages. Resolution order and
frame-witness order are exact. Decoders reject trailing bytes, wrong schemas,
noncanonical declaration zeros, non-finite values, invalid lengths and any
recomputed component, resolution or bundle hash mismatch.

The exact domain strings and preimages are:

```text
component hash = FNV(fixed((
  domain, schema_version, hash_version, nominal_spacing_km, payload
)))

frame-array hash = FNV(fixed((
  domain, schema_version, hash_version, nominal_spacing_km, time_myr, payload
)))

derived_resolution_hash = FNV(fixed((
  "orogen-linked-input-v0/resolution", schema_version, hash_version,
  nominal_spacing_km, mesh, initial_elevation_km, local_runoff_supply_km3_myr,
  base_material_present, whole_graph_candidate, central_window_candidate,
  compiled_stencils, cumulative_rock_displacement_km, frame_witnesses,
  summary, component_hashes
)))

derived_bundle_hash = FNV(fixed((
  "orogen-linked-input-v0/bundle", schema_version, hash_version,
  declaration, resolutions
)))
```

The nine component domains, in field order, are:

```text
orogen-linked-input-v0/mesh
orogen-linked-input-v0/initial-elevation
orogen-linked-input-v0/local-runoff
orogen-linked-input-v0/base-material-present
orogen-linked-input-v0/whole-graph-candidate
orogen-linked-input-v0/central-window-candidate
orogen-linked-input-v0/compiled-stencils
orogen-linked-input-v0/cumulative-rock-displacement
orogen-linked-input-v0/frame-witnesses
```

The three frame-array domains are
`orogen-linked-input-v0/frame-vertical-rate`,
`orogen-linked-input-v0/frame-horizontal-velocity` and
`orogen-linked-input-v0/frame-dominant-episode`. Array hashes are compact
replay witnesses: semantic validation freshly evaluates the scenario on the
stored mesh and compares full array hashes and integrated rates. The frame
arrays themselves are deliberately not duplicated in the artifact.

Assembly canonicalizes every floating zero in declarations to positive zero
before hashing. Negative zero is rejected, rather than rewritten, in initial
elevation, local runoff, support, cumulative displacement, evaluated vertical
rate, evaluated zero velocity and nonnegative computed summaries. Signed mesh
and forcing geometry must equal the fresh authoritative build and receives no
additional zero-rewriting rule.

FNV-1a is an identity checksum, not a security claim. Domain separation prevents
equal bytes in different roles from being substituted. The bundle hash covers
the full declaration and resolution values, not only their nested hashes.
Exact transcendental/f32 field hashes certify replay under the recorded
executable, Rust toolchain and WSL platform. The serialized bundle remains the
cross-platform input authority; another platform may claim numerical replay
only if its full hashes match, otherwise it reports the mismatch and may use
the stored inputs without rewriting the bundle.

Materialization writes one atomic directory:

```text
artifacts/orogen-linked-input-v0/
  shared-input.bin
  manifest.json
  run-envelope.json
```

`shared-input.bin` is the semantic authority. `manifest.json` is a deterministic
human-readable projection containing schema, bundle/component hashes, exact
integer counts, units, ledgers and round-trippable floating values; it must
validate against the binary. Its exact schema is:

```rust
pub struct LinkedSharedInputManifestJsonV0 {
    pub schema_version: String,
    pub semantic_schema_version: String,
    pub hash_version: String,
    pub derived_bundle_hash_hex: String,
    pub declaration: LinkedInputDeclarationV0,
    pub resolutions: Vec<LinkedResolutionManifestJsonV0>,
}

pub struct LinkedResolutionManifestJsonV0 {
    pub nominal_spacing_km: f64,
    pub summary: LinkedResolutionSummaryV0,
    pub component_hashes_hex: Vec<String>,
    pub derived_resolution_hash_hex: String,
}
```

The JSON schema is `orogen-linked-shared-input-manifest-json-v0`; semantic and
hash versions repeat the binary values. Hash strings are lower-case, zero-padded
16-digit hexadecimal without `0x`. Component strings follow the nine-field
order above. Resolution order is 8/4/2. Serialize with
`serde_json::to_vec_pretty` and append one LF. JSON decimals need only round-trip
to the binary f64 values; semantic hashes bind their exact bits.

`run-envelope.json` is nonsemantic provenance: source revision and dirty state,
executable identity, Rust toolchain, OS/CPU, thread count, command, elapsed time
to pre-publication validation, `/proc/self/status` `VmHWM` at that point, and
the lengths and FNV hashes of `shared-input.bin` and `manifest.json`. It does not
hash or measure itself. It explicitly names external `/usr/bin/time -v` as the
authority for final process wall time and whole-process peak RSS; those values
belong in the dated audit because a process cannot record its own post-exit
measurements. Changing the envelope must not change the semantic bundle hash.

The binary is exactly `orogen_linked_input`. Its only argument is the required
`--output-dir <PATH>`; V0 has no default, overwrite flag, arm or terrain
parameter. The target must not exist. A collision fails before modification;
successful publication writes and validates a temporary sibling directory and
atomically renames it to the target. The registered invocation is:

```bash
cargo run --release --bin orogen_linked_input -- \
  --output-dir artifacts/orogen-linked-input-v0
```

## Frozen numerical comparisons

For a registered mixed comparison, define:

```text
close(actual, expected; abs_tol, rel_tol)
  := abs(actual - expected)
     <= abs_tol + rel_tol * max(abs(actual), abs(expected))
```

Use these exact gates:

| quantity | absolute tolerance | relative tolerance |
|---|---:|---:|
| each compiled support area integral versus 1 | `1e-12` | `1e-12` |
| cumulative rock-volume integral versus 100,625 km3 | `1e-8 km3` | `5e-12` |
| runoff sum versus 500 times ordered area | `1e-6 km3/Myr` | `5e-12` |
| activity-one evaluated rate versus 17,500 km3/Myr | `0` | `2e-7` |

Mesh closure uses the current independent geometric checks, not the mixed table:

- exact fresh-constructor equality for every serialized mesh field;
- exact reciprocal neighbour indices, with reciprocal f32 distances, widths and
  opposite tangents checked by stored values;
- per-cell vector Gauss residual `< 2e-6 km`;
- per-cell area-moment residual `< 2e-6 km2`;
- global boundary-area-moment and `cell_count * analytic_cell_area` residuals
  `< 1e-5 km2`;
- boundary-face centre-distance residual `< 1e-6 km`, unit-normal residual
  `< 1e-12` and positive face measures;
- total physical arc before versus after portal splitting residual `< 1e-8 km`;
  and
- each portal's projected coverage versus the ordered sum of intersections
  between the unsplit side-face intervals and the declared portal interval,
  residual `< 1e-10 km`.

The last comparison is deliberately neither the nominal 960 km span nor the
full generated side coverage. The staggered full-cell patch projects slightly
beyond one declared endpoint on each north/south side. Every positive-measure
face/portal intersection must be open with the right ID and base level; every
out-of-span remainder and every east/west face must be closed. This produces
projected portal coverage 958/959/959.5 km at 8/4/2 and one closed outboard
split sliver on each of north and south. Portal IDs must be unique and every
open record must resolve exactly once. `owner_cell_count` in a portal summary
means the count of unique open-face owner cells for that portal.

Stored arrays, declarations, hashes and summaries that rebuild through the same
frozen ordered algorithm must be bit-identical. The tolerances above apply only
to independently reduced or analytically compared quantities; they are not a
license to accept a different mesh, input array or stencil.

## Executable gates

Implementation is accepted only if all of the following pass:

1. **Exact rebuild:** two same-process builds are identical by Rust value,
   binary bytes, JSON semantic projection and every hash.
2. **Decode boundary:** trailing bytes, field mutation with repaired outer
   hash, wrong order, wrong schema and non-finite values reject. Negative values
   reject only for fields registered as nonnegative: measures, widths,
   distances, rates, work, initial elevation, runoff, support and cumulative
   displacement. Signed coordinates, directions and wave vectors remain valid.
3. **Mesh:** all three fresh meshes pass native validation plus the exact
   reciprocal, closure, boundary, portal and integer-topology gates above.
4. **Forcing declaration:** IDs are unique, links resolve, vergence and numeric
   fields are finite, episode times/rates/shares are valid, shares sum to one,
   compiler-consumed versus retained-metadata fields are reported, and the
   canonicalized value exactly equals a fresh `linked_scenario()` plus the
   registered compiler-semantics and work declarations.
5. **Support:** both stencils are nonnegative, have nonempty support and
   area-integrate to one at every resolution. Stored stencils exactly equal a
   fresh compile.
6. **Work:** analytic activity is 5.75 Myr; total and segment ledgers are
   100,625 and 50,312.5/50,312.5 km3; each cumulative array closes to the total
   and exactly rebuilds from its stencils.
7. **Frames:** all registered activity oracles, f32 field hashes, horizontal
   zeros, dominant IDs and integrated-rate tolerances pass at 8/4/2 km.
8. **Initial state:** the seed/formula and arrays rebuild exactly, all values
   are finite and nonnegative, refinement samples use coordinates rather than
   cell IDs, and portal owners receive no special rewrite.
9. **Runoff/material/masks:** exact lengths and formulas rebuild; runoff closes
   to area, material and whole-graph arrays are all true, and central-window
   counts/predicate pass.
10. **Semantic exclusions:** the schema contains no final surface, receiver,
    discharge, arm conversion/configuration, common evidence, score, terrain
    verdict or presentation field.

Exact value/byte gates apply where arrays are reconstructed by the same frozen
algorithm; the frozen numerical comparisons apply to independent analytic
closures.

The manifest validator owns the stronger declaration checks in gate 4. The
current forcing compiler does not by itself reject every duplicate ID, invalid
link, non-finite vergence or non-finite/negative schedule, rate, ramp and share;
successful `compile()` is necessary but not sufficient manifest validation.

### Repaired-hash mutation matrix

Mutation rejection must not rely on one stale nested checksum. Starting from a
valid 4 km bundle, perform each witness below separately, then recompute every
affected component hash, resolution hash and bundle hash through the top of the
chain. Decode plus semantic validation must still reject:

| owner | registered mutation |
|---|---|
| declaration | increment segment 0 endpoint x by one finite f64 ULP |
| mesh | increment one interior cell-centre x by one finite f64 ULP |
| initial elevation | increment one interior value by one finite f64 ULP |
| runoff | increment one local-supply value by one finite f64 ULP |
| base material | change one `true` membership to `false` |
| whole mask | change one `true` value to `false` |
| central mask | flip one cell immediately outside the coordinate predicate |
| stencil | increment one positive segment-0 weight by one finite f64 ULP |
| cumulative work | increment one positive displacement by one finite f64 ULP |
| frame witness | increment the activity-one integrated rate by one finite f64 ULP |
| summary | increment the central-window count by one |

Also reverse the 8/4/2 resolution order and swap two forcing witnesses, repair
the affected outer hashes, and require rejection on canonical order. For the
frame-field replay boundary, separately replace one stored vertical-rate hash
with the correctly encoded hash of a one-cell-mutated f32 array, repair the
frame/resolution/bundle chain and require rejection against fresh evaluation.

## Bounded implementation cost

After release compilation, bundle construction, validation and atomic write on
the development WSL machine must complete within 2 minutes wall time and
1 GiB whole-process peak RSS. Record the authoritative final process time/RSS
with external `/usr/bin/time -v`, alongside the envelope's pre-publication
witnesses and artifact bytes. This is only the input-materialization budget; it
is not the later common H/C/G resource ceiling.

## Implementation order and stop condition

1. Commit this preregistration before production code.
2. Move the private parabolic initial generator into `world::landscape` and
   make `c0_orogen_smoke` consume that single authority; do not copy it.
3. Add the read-only compiled-stencil export and manifest types/builder.
4. Implement the frozen `orogen_linked_input` binary, semantic validation,
   hashing, atomic binary/JSON output and mutation/determinism tests.
5. Run the registered 8/4/2 materialization once, record a dated audit and stop.

Do not proceed from a passing manifest directly into a favored arm. The next
document is the organization-owner comparison preregistration. It must decide
evaluation population, opportunity calibration, H/C/G conversions and
chronologies, snapshot/response protocols, admission gates, shared arm resource
ceiling, independent evidence extraction and matched human presentation.
