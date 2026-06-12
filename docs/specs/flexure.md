# Spec: Elastic Plate Flexure for Subduction Zones

Implementation spec for roadmap item #2 (`docs/physically-inspired-roadmap.md`).
This document is the contract for the implementation; where it is silent,
match the existing style of the file being edited and do NOT invent new
mechanisms, smoothing passes, or post-processing. Design questions that come
up during implementation should be left as `// SPEC:` comments, not resolved
ad hoc.

## Goal

Replace the symmetric exponential trench profile with the analytic flexure
profile of a thin elastic plate, and add the coupled forearc depression on
the overriding plate. After this change a subduction zone reads, going
seaward-to-landward:

```
outer rise (subtle bulge) -> deep narrow trench -> forearc depression
recovering landward -> volcanic arc (existing Gaussian, unchanged)
```

All of this is *dynamic topography*: it stays in the `features.trench`
channel consumed by `elevation.rs` as a subtraction from the isostatic
surface. Do not touch the crust-thickness / isostasy path.

## Physics

A broken elastic plate end-loaded at the trench axis deflects as

```
w_sub(d) = w0 * exp(-d/alpha) * (cos(d/alpha) + sin(d/alpha))      d >= 0
```

where `d` is distance from the trench axis along the subducting plate and
`alpha` is the flexural parameter. Properties (these become unit tests):

- `w_sub(0) = w0` (maximum deflection at the trench axis)
- zero crossing at `d = 3*pi/4 * alpha ~= 2.356 * alpha`
- minimum (the outer rise, since deflection sign flips) at `d = pi * alpha`
  with value `w0 * -exp(-pi) ~= -0.0432 * w0`
- `|w_sub| < 0.005 * w0` for `d > 2*pi*alpha`

The overriding plate is not broken; its edge is dragged down by coupling to
the slab and recovers monotonically-ish landward (continuous-plate member of
the same family):

```
w_over(d) = w0_over * exp(-d/alpha_f) * cos(d/alpha_f)             d >= 0
```

- `w_over(0) = w0_over`
- first zero crossing at `d = pi/2 * alpha_f`
- small overshoot (minimum) at `d = 3*pi/4 * alpha_f`, value
  `~= -0.067 * w0_over` — keep it, it is physical (back-bulge)

Sign convention everywhere: **positive = downward deflection** (consistent
with the existing `trench` field, which `elevation.rs` subtracts).

## Changes

### 1. `src/world/features.rs` — profile functions

Add two pure functions next to `exp_decay` / `gaussian_band` (same `pub fn`
style, doc comments stating the physics):

```rust
pub fn flexure_broken(dist: f32, alpha: f32) -> f32   // (cos+sin) form, returns w/w0
pub fn flexure_coupled(dist: f32, alpha: f32) -> f32  // cos form,     returns w/w0
```

Both return the *normalized* profile (caller multiplies by w0). Guard
`alpha <= 0` and non-finite `dist` by returning 0.

### 2. `src/world/features.rs` — subducting side

In the per-cell loop (currently `trench[i] = depth * exp_decay(d, TRENCH_DECAY)`
for oceanic cells), replace with:

```rust
let alpha = TRENCH_FLEX_ALPHA
    * (TRENCH_FLEX_ALPHA_YOUNG_MULT
        + (TRENCH_FLEX_ALPHA_OLD_MULT - TRENCH_FLEX_ALPHA_YOUNG_MULT) * age);
trench[i] = depth * flexure_broken(d, alpha);
```

`age` is the existing `oceanic_age_factor_from_ridge_distance(ridge_dist[i])`
already computed in that block for the depth multiplier — reuse it, do not
recompute. `depth` (the `sqrt_response` of smoothed trench forcing with the
age multiplier) is unchanged.

The trench forcing smoothing support distance (currently
`4.0 * TRENCH_DECAY`) must cover the outer rise: change to
`(PI + 1.0) * TRENCH_FLEX_ALPHA * TRENCH_FLEX_ALPHA_OLD_MULT`.

The `trench` field hereby becomes **signed** (negative on the outer rise).
Audit every consumer — see section 5.

### 3. `src/world/features.rs` — overriding side (forearc)

New seed arrays `forearc_seed_strength` / `forearc_seed_dist0`, built in the
same boundary loop where trench seeds are built. When polarity says A
subducts and A is oceanic (i.e. exactly the condition that seeds a trench),
seed the *overriding* cell `cell_b` with the **subducting side's** force
(`subd_force_a * area_scale(b.cell_b)`) and `dist0_b`; mirror for
B-subducts. Rationale: the forearc is pulled down by the slab, so its
amplitude is the trench's amplitude, not the arc's.

Then:
- `forearc_dist` via `distance_field_from_edge_seed_cells(..., true)`
  (plate-restricted, like the others)
- `forearc_forcing` via `compute_smoothed_boundary_forcing` with support
  `(0.75 * PI + 1.0) * FOREARC_ALPHA`
- per-cell, for **any crust type** (continental forearcs exist):

```rust
if forearc_dist[i].is_finite() {
    let w0 = FOREARC_COUPLING
        * sqrt_response(forearc_forcing[i], TRENCH_SENSITIVITY, TRENCH_MAX_DEPTH);
    trench[i] += w0 * flexure_coupled(forearc_dist[i], FOREARC_ALPHA);
}
```

Reusing `TRENCH_SENSITIVITY` / `TRENCH_MAX_DEPTH` keeps the two walls of the
trench meeting at comparable depth; `FOREARC_COUPLING` is the knob for how
much of the trench depth the overriding edge inherits. No age multiplier on
the forearc (the overriding plate's age is not the slab's age; out of scope).

Note `+=`: a cell could in principle carry both subducting-side and
forearc deflection (e.g. small plates); summing is correct superposition.

### 4. `src/world/constants.rs`

Remove `TRENCH_DECAY`. Add, in its place, with doc comments in the existing
style (state Earth-equivalent length for radian values):

```rust
/// Flexural parameter alpha for the subducting plate (radians).
/// Sets the whole trench-to-outer-rise geometry: trench wall zero-crossing
/// at 2.36*alpha, outer rise crest at pi*alpha (4.3% of trench depth, up).
/// 0.018 rad ~= 115 km on Earth -> outer rise ~360 km from the axis.
pub const TRENCH_FLEX_ALPHA: f32 = 0.018;
/// Alpha multiplier for young (near-ridge) lithosphere: hot, thin, floppy.
pub const TRENCH_FLEX_ALPHA_YOUNG_MULT: f32 = 0.6;
/// Alpha multiplier for old lithosphere: cold, thick, stiff -> wide flexure.
pub const TRENCH_FLEX_ALPHA_OLD_MULT: f32 = 1.4;
/// Flexural parameter for the overriding plate's forearc (radians).
/// Recovery is ~complete by pi/2*alpha_f ~= 0.024 rad, landward of which
/// the arc Gaussian (peak 0.04-0.05 rad) takes over.
pub const FOREARC_ALPHA: f32 = 0.015;
/// Fraction of the trench-axis depth inherited by the overriding plate edge.
pub const FOREARC_COUPLING: f32 = 0.8;
```

These initial values are starting points; do not tune them to make any
number look better — tuning happens after review, by us.

### 5. Signed-field consumers (audit + update)

- `src/world/elevation.rs` (~line 350): `- features.trench[i]` is already
  correct for a signed field (outer rise negative -> net uplift). Update the
  module doc comment ("dynamic(trench)") to mention flexure/outer rise.
- `src/app/coloring.rs` `FeatureLayer::Trench` (~line 427): currently a blue
  scale assuming `value >= 0`. Make it diverging: blues for positive
  (down), warm/red for negative (outer rise). Keep the existing
  normalization approach for the positive side.
- `src/app/export.rs`: exports `trench` verbatim — no change needed, but
  confirm nothing clamps it.
- `src/bin/diagnose.rs` arc-trench gap: masks on `t > 0.3 * trench_peak`
  where `trench_peak` is a `fold(0.0, max)` — still correct for signed
  values (peak is the deepest point). Verify, don't restructure.
- Grep for any other `features.trench` / `"trench"` consumer (tests,
  scripts) and check sign assumptions. `scripts/` changes are out of scope —
  flag with a `// SPEC:` note if a script assumes non-negative.

### 6. Diagnostics

Add to `src/bin/diagnose.rs`, alongside the arc-trench gap block, a
"Flexure profile" section computed from the `features.trench` field alone
(diagnose builds the world in-process, so the field is available; do not
expose new internals from features.rs for this):

- deepest deflection (`max`), strongest outer rise (`-min`), and their
  ratio. Expected ratio ~0.04-0.07 (bulge is 4.3% per seed, forcing overlap
  varies it). `[Earth: outer rise ~200-500 m vs trenches 2-8 km -> ~0.05]`
- count of cells with negative trench (outer-rise cells) — must be > 0 on
  seed 12345.

### 7. Unit tests

In `features.rs` tests module (create one if absent, matching repo test
style):

- `flexure_broken(0.0, a) == 1.0` (within 1e-6)
- zero crossing: sign change between `d = 2.35*a` and `2.37*a`
- outer rise: minimum near `pi*a`, value within 5% of `-0.0432`
- `flexure_broken` for `d > 2*pi*a` has magnitude `< 0.005`
- `flexure_coupled(0.0, a) == 1.0`; zero crossing near `pi/2*a`; overshoot
  near `3*pi/4*a` within 5% of `-0.0670`
- both return 0.0 for `alpha = 0.0` and `dist = f32::INFINITY`

## Validation (run all; paste output into the PR/branch description)

```bash
cargo fmt && cargo clippy -- -D warnings && cargo test
cargo run --release -- --headless --seed 12345 --export /tmp/w12345.json.gz
cargo run --release --bin diagnose -- --seed 12345
```

Acceptance on seed 12345:
- all existing tests pass, including `tests/field_smoothness.rs` (Moran's I
  guards — flexure must not introduce speckle)
- diagnose: outer-rise cells > 0; deflection ratio in [0.02, 0.12]
- land fraction unchanged at the `LAND_FRACTION` target, 26.0% (sea-level
  solve should absorb the small volume change)
- arc-trench gap diagnostic still in its previous range (forearc must not
  shift the arc crest)

## Non-goals / forbidden

- No changes to crust thickness, isostasy, rifts, ridges, collision, or the
  arc Gaussian (beyond the forearc term defined here).
- No new smoothing/blur passes, no clamps or caps not specified here, no
  retuning of existing constants.
- No new dependencies.
- No changes under `scripts/`.
- Spreading-rate-based ocean age is a separate roadmap item; keep using
  `oceanic_age_factor_from_ridge_distance`.
