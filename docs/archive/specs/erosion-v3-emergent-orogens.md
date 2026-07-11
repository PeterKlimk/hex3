> **Archived:** Historical design or experiment record. It does not define current architecture, defaults, or priorities. Start at [`docs/README.md`](../../README.md).

# Erosion-v3: emergent orogens (uplift-rate forcing, not painted relief)

**Provenance.** 2026-06-20, after the P1a/b/c fine-synthesis work (erosion-v2 Phase 1)
hit a ceiling: P1a+erosion gives "dissected noise", P1b gave "tectonic corduroy"
(sand-dune ridges), and the user's read + two codex physical reviews converged on a
deeper diagnosis. This spec is the rearchitecture those reviews pointed to. It
SUPERSEDES the P1 painted-substrate approach for orogen morphology (P1a survives only
as a faint seed; P1b/P1c are retired from the default path — see §7).

## Root cause (verified in code)

The fine erosion stage is calibrated as a **postprocessor**, not a mountain generator:

- The fine base is the **finished, saturated coarse elevation** interpolated onto fine
  cells (`fine.rs` `interpolate_coarse_elevation`). The coarse forcing fields are
  distance-decay functions that SATURATE in orogen interiors → the base is a flat-
  topped plateau, at full orogen height, before erosion runs.
- Erosion HAS active uplift during the fluvial loop (`erosion.rs`: `u_thick` built from
  `uplift_scale*((arc+collision)/slope + rift)`, added each step), but it is
  deliberately TINY. `constants.rs` (EROSION_UPLIFT_SCALE doc): *"uplift … roughly
  BALANCES erosion … It must not re-inject the full orogen — the coarse elevation
  already encodes the static height — so SCALE … must stay small."*

So height is baked into the inherited base, and uplift is forbidden (by calibration)
from rebuilding it, to avoid double-counting. Erosion can only DISSECT the flat plateau
it's handed. The flat top is structurally guaranteed; erosion is calibrated not to fix
it; painting (P1) was the workaround, and its character (noise / corduroy) is the
ceiling because erosion only sands the painted substrate.

**Review status.** Codex design review (2026-06-20, against code): **SOUND-WITH-FIXES**
— direction right, but the spec over-claimed three things now corrected below: the
decomposition is exact only PRE-sea-level (not "preserves the mask by construction");
height is NOT preserved by construction (no target attractor in an n=1 LEM); and self-
organization is plausible but unproven in this solver. All load-bearing numbers verified
(`isostatic_elevation` strictly linear `slope·thk+offset`; `uplift_scale·steps·dt =
0.003·60 = 0.18` → today rebuilds only ~18% of the orogen). Fixes folded into the
sections below + the First Prototype section.

## The fix: hand erosion an uplift-RATE field + a low envelope, not finished elevation

Real ridge-and-valley relief is **emergent** from erosion acting on an actively
uplifting surface (the fluvial-incision vs hillslope-diffusion competition self-
organizes a valley-spacing wavelength; Perron/Dietrich/Kirchner). So let erosion BUILD
the ranges:

```
coarse model  →  (a) broad ENVELOPE elevation (no saturated orogen peak)
                 (b) uplift-RATE field U(x)  (where rock is actively added)
fine base     =  interpolate(envelope) + faint seed
fine erosion  =  active LEM: U(x) builds relief up while incision carves valleys
                 → emergent dendritic ranges; final height ≈ coarse target (preserved)
```

### The decomposition is exact and double-count-free (key enabler)

The coarse elevation already separates the orogen peak as a distinct additive term.
From `elevation.rs`: `thickness = base_thickness(continentality) + thickening + rift`,
`thickening = (arc+collision)/slope`, and `structural_elevation =
isostatic_elevation(thickness) + thermal + ridge − trench`. Because
`isostatic_elevation` is ~`slope·thickness`, the **elevation contribution of the orogen
thickening is exactly `arc + collision`** (elevation units). The SAME `arc+collision`
already drives `u_thick`. So:

- **Envelope** = coarse elevation − `λ·(arc + collision)`   (λ∈[0,1]: fraction of the
  orogen peak removed from the static base; λ=1 = build entirely from uplift, λ=0 =
  current behaviour). Continentality dome, rifts, oceans, trenches, ridges, thermal —
  all UNCHANGED (only the orogen peak is demoted).
- **Uplift rate** `U(x)` ∝ `arc + collision`, scaled so that the uplift integrated over
  the erosion epoch (minus what erosion removes) rebuilds ≈ `λ·(arc+collision)` of
  relief — i.e. the emergent orogen returns to ~the coarse target height, but as
  dissected ranges instead of a flat plateau.

This removes the static orogen height from the base and re-supplies it as uplift, so
there is no double-count. The intent is that the macro height is **approximately
preserved** — but NOT "by construction" (see the corrections below): an n=1 stream-power
LEM has no target-height attractor, so the emergent height must be *calibrated* toward
the coarse target, not assumed.

### Corrections from review (the claims to NOT make)

- **Decomposition is exact only on the PRE-datum structural field.** `arc+collision`
  cancels cleanly *before* the coarse sea-level solve, but the finished coarse elevation
  has already had a uniform sea-level shift applied (`assemble_heightmap` area-weighted
  solve). Subtracting `λ·(arc+collision)` does NOT commute with that solve. So: **inherit
  the coarse sea-level datum (do not re-solve at fine), and AUDIT/clamp land-mask drift**
  — don't claim the envelope preserves the land/ocean mask by construction.
- **Uplift gate bug to fix.** `erosion.rs` gates uplift `if base[i] < 0.0 { return 0.0 }`.
  If demotion pushes an intended-land orogen cell below 0, it will never uplift back
  (dead orogen). Fix: **gate the builder uplift on the FULL coarse-target land mask**
  (`coarse_target[i] >= 0`), or keep the envelope land-positive, not on the demoted base.
- **Exclude `rift_delta` from the builder uplift.** Current `u_thick =
  uplift_scale·((arc+collision)·inv_slope + rift_delta)`. The builder should re-supply
  ONLY the demoted orogen term `(arc+collision)`; rifts are not what we demoted.
- **Bookkeeping.** `derive_elev = base + slope·(thick − thick_init)`, step 0 = base. With
  `base = coarse − λF` and uplift `= uplift_scale·(F/slope)`, the no-erosion rebuild after
  N steps is `coarse − λF + uplift_scale·N·dt·F`, so **rebuild ⇔ `uplift_scale·N·dt ≈ λ`**
  (with erosion, higher). Today 0.18; λ=0.5 ⇒ `uplift_scale ≈ 0.008`, λ=1.0 ⇒ ≈ 0.017.
  Keep `thick_init` as a pure delta-reference (numerically fine) or de-thicken it by
  `λF/slope` for cleaner diagnostics.

## Why height preservation matters (atmosphere validity)

The coarse atmosphere (temperature, circulation, orographic precip, rain shadows) was
solved on the FINISHED coarse elevation. If emergent orogens drifted materially in
height, the coarse atmosphere would be inconsistent (the same concern P1a guarded with
zero-mean + land-drift, but bigger). Constraint (softened per review): **calibrate the emergent orogen toward the coarse
target so low-order hypsometry / orogen-scale mean stay within tolerance** — exact
height-matching fights the LEM and isn't the goal; "reshaping, roughly same mean" is.
Fine orographic precip re-derives *local* modulation on fine relief but keeps the coarse
wind/moisture field — so this is a first-order approximation: good while broad mean height
and barrier footprint stay close, NOT a guarantee of atmospheric consistency if the form
change is drastic. Sea-level datum and the
land/ocean mask are inherited from the envelope (oceans/continents untouched).

## New fine pipeline (what changes)

1. `FineBase`: build `envelope = interpolate(coarse_elevation) − λ·interpolate(arc+collision)`
   as the erosion base, AND keep `coarse_base_elevation` (full interpolant) as the
   height TARGET / atmosphere-lapse baseline. Store the interpolated `arc+collision`
   uplift field at fine.
2. Seed: a faint, broadband, zero-mean symmetry-breaking perturbation on the orogen
   envelope (P1a demoted: amplitude ~0.005, just enough to break drainage symmetry —
   NOT a structural substrate). Keep the gating + zero-mean + land-drift guard.
3. Erosion (`FineSurface::generate`): run as an ACTIVE LEM —
   - `uplift_scale` raised to BUILD (uplift integral ≈ λ·orogen height over the epoch),
     no longer "hold & carve". `EROSION_UPLIFT_SMOOTH_KM` ON (broad tectonic uplift,
     not cell-speckled).
   - nonlinear hillslope diffusion (Roering, `hillslope_critical_slope`) ON — needed for
     a real wavelength-setting hillslope law (linear D=2e-8 only polishes).
   - MFD ON (`mfd_exponent`) — SFD inherits mesh/seed quirks; MFD gives coherent dispersal.
   - `steps` likely raised (building from a low envelope takes longer than dissecting a
     high base) — perf cost to measure.
4. Calibrate to land near the coarse target height (probe: orogen mean elevation vs the
   `coarse_base_elevation` target; aim drift small).

## Calibration strategy (the real work — "active LEM", not "preserve + carve")

This re-tunes the erosion stage as a landscape-evolution problem. Levers, in order:
- `λ` (orogen-demotion fraction): start ~0.5 (leave a gentle swell, build the rest), push
  toward 1.0 if emergent ranges look good and heights can be matched.
- `uplift_scale` × `steps`: the uplift integral sets how much relief is built; tune so the
  emergent orogen mean ≈ coarse target. Watch perf (steps).
- `K` (incision) and `D`/`hillslope_crit` (hillslope): set the valley SPACING and the
  relief/dissection texture (the incision-vs-diffusion competition wavelength).
- All are existing CLI/sweep knobs (`--erosion-uplift-scale`, `--erosion-steps`,
  `--erosion-k`, `--erosion-hillslope-crit`, `--erosion-uplift-smooth`, `--erosion-mfd-exponent`),
  so the calibration can be SWEPT visually (the sweep harness is the tuning instrument).

## What happens to P1a/P1b/P1c

- **P1a** → demoted to the faint seed (§3.2). The synthesis/gating/zero-mean code is
  reused at low amplitude; it is no longer the mountain generator.
- **P1b** (strike bands) → RETIRED from the default (`front_strike_weight = 0`). Structural
  along-strike grain, if wanted later, belongs in the **erodibility field K(x)** (already
  exists, `lithology_erodibility`) as a SECOND-ORDER overprint on emergent dendritic
  drainage — not as painted height. (codex: "move structure into K(x), not height".)
- **P1c** (margin contrast) → RETIRED from the default; active/passive margin morphology
  is already handled at coarse scale (continentality shelf width) and can re-enter via
  K(x) / uplift shape if needed.
- The knobs stay in the code (cache-hashed) so the old painted path is A/B-able against
  the emergent path, but default to off.

## Risks & open questions (for codex review)

1. **Steady-state height matching.** Stream-power steady-state relief ~`(U/K)^(1/n)`
   integrated up-channel — the emergent height is a function of U, K, climate, time, NOT
   directly the coarse target. Can we reliably land near the coarse height by tuning, or
   does height-matching fight texture quality? Is λ<1 (keep part of the orogen static)
   the pragmatic hedge?
2. **Convergence in a fixed step budget.** Building from a low envelope may need many more
   steps than dissecting. Does the existing `steps`/`dt` reach a mature landform, or is
   this a perf blowup? Is there a non-dimensional time (U·t/K·…) target?
3. **Isostatic feedback.** Erosion evolves `thick`; the base envelope was de-thickened.
   Does removing `λ·(arc+collision)` from the base but re-supplying via `u_thick` interact
   correctly with the per-step `thick`/`elev` re-derivation (`erosion.rs` derive_elev)?
   Must verify the isostatic bookkeeping doesn't drift.
4. **Cratons / non-orogen relief.** Only orogens get demoted+rebuilt; cratonic dome,
   passive margins, ocean stay in the envelope. Correct, or do interiors also need an
   active component?
5. **Atmosphere coupling.** Is "preserve macro height" sufficient, or does the changed
   orogen FORM (ridges vs plateau) shift orographic precip enough to matter (and is the
   fine re-derivation enough)?
6. **Does the solver actually self-organize at these settings?** The capability is
   present (stream-power + diffusion + active uplift) but unproven at "builder" scale.
   This spec assumes it does; the first prototype must demonstrate emergent dendritic
   ranges before the full calibration.

## OUTCOME (2026-06-20): premise FALSIFIED with the current erosion solver

The prototype was built (self-calibrating builder: rebuild `target−base` over the epoch,
×`EMERGENT_REBUILD_GAIN`, so height tracks target and `steps` is a pure build-vs-carve
dial). Mechanically sound: height rebuilds to ~90% target, land flips ~2% at λ=0.5.

**But the emergent orogen does NOT dissect into ranges — it stays a smooth swell.**
Numerical pre-screen (seed 12345, λ=0.5, MFD+nonlinear hillslope on):

| steps | summit eroded slope p50 | p90 | max elev |
|---|---|---|---|
| 120 | 2.41e-4 | 1.32e-3 | 0.447 |
| 240 | 2.32e-4 | 1.14e-3 | 0.442 |
| 400 | 2.28e-4 | 1.05e-3 | 0.437 |

More carving time → SMOOTHER (slope falls), not sharper. Aggressive channelization
(channel-support 3, diffusivity 5e-9, SFD) also fails: drainage density ~0.001 km/km²,
valley spacing ~1000 km (Earth: 0.5–5 km/km²). Emergent summits sit at ~2.3e-4 — BELOW
global land (3.8e-4), i.e. smooth swells; painted P1a was ~2.8e-3 (~10× more dissected).

**Cause:** stream-power incision is n=1 (linear in slope); a broad smooth uplift dome
has gentle slopes → weak stream power → no channel cutting; hillslope diffusion then
smooths it (more steps = smoother). There is no channelization instability / threshold
incision to manufacture relief from low perturbations. THIS solver dissects relief it is
GIVEN (why painted P1a works — high rough substrate with steep local slopes), but cannot
BUILD ranges from a smoothly-rebuilt envelope.

**Conclusion:** the emergent architecture is mechanically correct but the "erosion builds
the mountains" premise needs an erosion model with threshold/n>1 incision + proper
hillslope–fluvial competition — a landscape-evolution-solver project, not this interface
swap. The envelope/uplift/self-calibrating-builder scaffolding (behind `emergent_lambda`,
default 0) is retained and reusable IF that solver work is done. For now: painted P1 is
the pragmatic path (this erosion dissects given relief well). See the decision in-session.

## First prototype (do THIS before the full retune — codex)

The smallest diagnostic path that answers "does the active-LEM premise work at all,"
before any texture tuning. Gated behind a knob (`--emergent-lambda`, default 0 = current
behaviour) so the default path is untouched.

1. **Envelope:** `envelope = interpolate(coarse_elev) − λ·interpolate(arc+collision)` as
   the erosion base; keep `coarse_base_elevation` (full interpolant) as the height TARGET
   and the lapse baseline. λ via the knob.
2. **Builder uplift:** `u_thick = uplift_scale·(arc+collision)/slope` (rift EXCLUDED),
   gated on the coarse-target land mask (not `base<0`). `uplift_scale` set to the rebuild
   target (≈0.008–0.02 for λ=0.5, swept).
3. **Terminal lakes OFF for the test** — pre-hydrology on the LOW envelope would freeze
   low lakes as base levels that pin the emergent range (`terminal_lake_base_levels`).
   Compare on/off; if material, defer lake base levels to a post-warm-up recompute.
4. **Seed:** faint P1a (`interior_relief ≈ 0.005`), P1b/P1c off.
5. **Measure (diagnose, numerical first — no GPU):**
   - area-weighted orogen mean + p90 elevation DRIFT vs the coarse target (height landed?),
   - **land-mask flips:** count cells where `coarse_target ≥ 0` but `envelope < 0` (the
     uplift-gate breaker — risk item #2),
   - ridge-and-valley morphology / summit-slope distribution (the mountain-top probe):
     real dissected ranges vs plateau vs noise.
6. **Then** a Windows visual sweep over `λ × uplift_scale` (and `K`, `steps`,
   `hillslope_crit`) — the texture-quality gate (user).

Decision gate: if the active build produces coherent dissected ranges at a tunable height
with tolerable land drift → commit to the full retune. If it produces uplifted mush or
can't hold height → the solver can't be the generator here, fall back to naturalistic
painted substrate.

## Validation

- **Objective:** orogen mean-elevation drift vs `coarse_base_elevation` target (height
  preserved); ridge-and-valley spacing in the emergent ranges (a real length scale, not
  flat and not noise); the mountain-top probe (summit slope distribution should look like
  dissected ranges, not plateau and not fBm); land-fraction/datum invariants.
- **Visual (user):** the sweep harness — `--erosion-uplift-scale`/`steps`/`k`/`hillslope-crit`
  grids — judged for "real ranges" vs "noise"/"dunes"/"mush". This is the gate.
- **A/B:** emergent path vs the P1a-painted path at matched height, same seed/camera.

## References
- Root-cause finding + this rearchitecture: this session's codex reviews (physical +
  architecture), verified against `erosion.rs` (u_thick:435/740, derive_elev:570),
  `constants.rs` (uplift "hold & carve":846), `elevation.rs` (decomposition:311/426).
- Supersedes the orogen-morphology role of [`erosion-fine-synthesis.md`](erosion-fine-synthesis.md)
  (P1a/b/c); P1a's seed + invariants carry forward.
- Erosion model: [`erosion.md`](erosion.md). LEM physics: Perron/Dietrich/Kirchner valley spacing.
