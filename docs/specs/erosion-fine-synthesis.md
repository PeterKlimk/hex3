# Fine-Terrain Synthesis (erosion-v2 Phase 1): structure in the mid-band

**Provenance.** 2026-06-20. After localizing the "flat range tops covered in
fractal cottage-cheese / dendritic spirals" artifact to its true source (see
[[hex3-summit-cottage-cheese]]): it is **pre-erosion**, in the fine *base*, which
is nothing but the coarse elevation interpolated onto fine cells. This is
erosion-v2 ROOT CAUSE #1 ("erosion carves an interpolant, not synthesized fine
terrain") — the band between the coarse Nyquist (~70 km) and the channel scale
(~few km) is **empty of structure**, so summits are smooth interpolated plateaus
with no gradient, and erosion (which can't drain a flat) spirals into cottage
cheese instead of dissecting real ranges. Glacial (now default-off) and the whole
routing/MFD/escalation line were downstream of this and never touched it.

**Philosophy (erosion-v2 noise rule).** Mid-band content must be an *input a
process organizes*, not painted output: physical heterogeneity (structural relief,
erodibility) that erosion then carves into landforms — NOT fake painted ridges.

**Review status.** Codex round 2 (2026-06-20, against code): **SOUND-WITH-FIXES** —
ordering / cache-version / zero-mean+area-weighted / soft-isotropic traps confirmed;
three new must-fixes folded in (scarp-knob cache boundary → decision A; temperature-
lapse baseline split; P1a needs a real interior height term, erodibility alone won't
test the thesis). All five pre-P1a fixes live in the Traps + rung table below.

**P1a status: LANDED** (2026-06-20, commits on `sweep-harness`). `fine.rs`
`add_interior_structural_relief` (zero-mean per-coarse-cell isotropic fBm height
field, gated to high orogen interiors) + scarps now built in `FineBase` before
pre-hydrology; `fault_scarp_height`+`interior_relief` migrated to
`FineStructureParams` (hashed, cache v4); temperature lapse split onto
`coarse_base_elevation`; area-weighted land-drift guard added. Codex round-3 review
(against the diff): SOUND-WITH-FIXES, all 3 folded in (temperature accessor exposed
the corrected field; land-drift check in code; scarp-asymmetry documented). Probe
(seed 12345): summit pre-slope p50 1.7e-4→7.9e-4 (flat plateau broken, now above
global land); erosion REVERSES from adding summit pit% (3.0→3.4) to organizing it
(8.1→5.4); land fraction + elevation percentiles unchanged. **Visual summit-zoom
sign-off (user) still pending** — needs the Windows GPU sweep.

**P1b status: LANDED** (2026-06-20). Realized as the spec's "evaluator over coarse
boundary-source primitives" path (NOT a full FeatureFields recompute). `OrogenFronts`
= the convergent plate boundaries as great-circle ARCs (each front's shared Voronoi
edge: `seg_a`/`seg_b`) + per-front overriding side, built on the coarse mesh. The fine
grain is a banded `cos` of the (unsigned) great-circle distance to the nearest
*compatible* front ARC — `point_to_arc_distance`, gathered within the influence radius
via a KD-tree of edge midpoints. Distance-to-arc (not to a point anchor) makes the
iso-contours TRUE offsets from the boundary polyline → coherent front-parallel ridge-
and-valley, no scalloped bullseyes, and no per-cell strike vector (sidesteps the
orientation-tensor seam problem the design weighed). Blended with P1a isotropic by
`front_strike_weight` (clamped [0,1]; cache v5 hashes the arc geometry + the P1a/P1b
shape constants). TWO codex reviews folded in — design (real edge anchors, side
awareness, magnitude-stays-gated, cache) and diff (the arc-vs-point-distance fix #1,
within-radius gather, constant hashing, weight clamp). Probe: `front_strike_weight=0`
reproduces P1a exactly (7.92e-4); default 0.7 → 2.81e-3 summit pre-slope, erosion still
organizes (pit% 11.0→8.6), land drift +0.00e0, datum preserved. **Strike-alignment is
a visual property — user's call via the GPU sweep.**

**P1c status: LANDED** (2026-06-20). Active/passive margin contrast for the structural
relief. The coarse model already does the active/passive SHELF (continentality →
bathymetry); P1c adds the sub-coarse STRUCTURAL contrast at the margin: the raw
`Crust.signed_margin_distance` is interpolated to fine (reusing
`interpolate_coarse_elevation` — no struct change) and drives a coastal-band amplitude
scale `1 + margin_contrast·coastal·(lerp(passive 0.5, active 1.3, activity) − 1)`,
where `activity` = convergent forcing and `coastal` fades over `FINE_MARGIN_WIDTH`.
Sharpens an active (convergent) coast, damps a passive one, neutral inland.
Amplitude-only within the elevation gate → land mask untouched (drift +0.00e0). Knob
`margin_contrast` (cache v6; knob + shape constants hashed). Probe: `margin_contrast=0`
reproduces P1b exactly (2.810e-3); default 2.852e-3 (active-coast boost on the coastal
subset), datum preserved. Margin morphology is the user's visual call.

**All three rungs (P1a/P1b/P1c) landed.** Each was implemented, codex-reviewed
(SOUND/SOUND-WITH-FIXES, fixes folded in), and validated against the mountain-top
probe with invariants (land fraction, datum) preserved throughout. Remaining: the
VISUAL sign-off (summit-zoom + margin sweeps) — the user's call, needs the Windows GPU
(`--sweep interior_relief|front_strike_weight|margin_contrast`).

## The problem precisely

- Fine base elevation = `interpolate_coarse_elevation(coarse)` (fine.rs:372). The
  forcing fields ARE resolution-independent distance-decay functions (features.rs:4),
  but they *saturate* in orogen interiors → broad smooth flat-topped highs;
  sampling them finer keeps them flat.
- Flat top → no drainage gradient → degenerate erosion → dendritic-spiral /
  cottage-cheese (the artifact). **The flat top and the cottage cheese are one
  problem**: fix the substrate and erosion organizes.
- The fine mesh already TRANSFERS coarse fields to fine (`transfer_fields`,
  fine.rs:1285 → `FineFields`, fine.rs:96: elevation_fields, temperature, precip,
  uplift, wind). Elevation is interpolated (not regenerated) deliberately, to
  inherit the coarse **sea-level datum**, which is a global planet datum solved
  once on the coarse mesh and never re-solved (elevation.rs:163,378).

## Core principle (resolves the datum + field-validity concerns by construction)

```text
fine_base = interpolate_coarse_elevation(coarse)   // KEEP: datum + coarse-scale agreement
          + fine_detail                            // NEW: sub-coarse structure, ~zero-mean at coarse scale
```

- **Datum preserved (with care — codex):** the interpolated term carries the
  coarse sea-level shift exactly, and `fine_detail` must be **coarse-cell-local
  zero-mean** (mean removed per coarse cell, not globally). NOTE this only kills
  *vertical* bias — land **fraction** can still drift because land/ocean is
  thresholded at elev 0.0, so zero-mean bumps flip near-sea-level cells
  asymmetrically *by area* (coastal lowlands, shelves; sea-level clamps like
  `apply_fault_scarps`' basin-drop floor add positive bias). So gate
  `fine_detail` to **well above sea level** (orogen interiors, where the artifact
  is) and validate with an **area-weighted** land-fraction drift check (see
  Validation), not a mean check. No fine sea-level re-solve.
- **Coarse-field validity preserved (the concern):** because the synthesized
  surface AGREES with the coarse elevation at every scale the coarse fields can
  see, and only adds detail BELOW the coarse Nyquist where the coarse atmosphere/
  etc. had no information anyway, the coarse-derived fields stay valid at the
  scales they operate. You are filling in *under* them, not invalidating them.
- **Relief-sensitive couplings already re-derived at fine (codex-confirmed):**
  temperature-lapse delta (fine.rs:496), hydrology (:510), orographic precip
  (fine.rs:688). **NOT re-derived** (computed on coarse elevation,
  atmosphere.rs:65): wind, pressure/circulation, wind-projection uplift, full
  moisture transport. Deliberately punted (needs fine atmosphere sim); second-
  order *provided the land/ocean mask stays ~stable* — which is the area-weighted
  drift check above. If fine detail moved the mask materially, coarse precip/
  evaporation assumptions would degrade, so keeping detail above sea level matters
  for climate validity too, not just the datum.

This is the "trick" for the unavoidable coupling: keep the added structure
coarse-consistent (zero-mean at coarse scale) and re-derive only the relief-
sensitive modulations — no clever fix needed, just that discipline.

## What stays coarse vs what's synthesized at fine

- **Coarse (unchanged):** plate assignment, Euler poles/dynamics, boundary
  classification, atmosphere simulation. Inherently plate-/planet-scale; running
  on millions of cells is waste with no physical gain. These feed in as SOURCE.
- **Fine (the new synthesis):** the `fine_detail` structure terms below.

## `fine_detail` content (the mid-band structure)

1. **Structural relief — the dominant term.** Range-front + intra-orogen faulting
   (fault blocks / horst-graben → ridge-and-valley) and fold-belt grain, gated by
   the orogen/convergence forcing so it follows the range geometry. Machinery
   exists: `apply_fault_scarps` (fine.rs:423, `FAULT_SCARP_HEIGHT`),
   `litho_grain` (fold grain). Currently weak/default-low; this makes it the
   substrate erosion dissects. Breaks the flat top → gives drainage a gradient.
2. **Erodibility heterogeneity.** `litho_sigma` / `litho_grain` (fine.rs:587) →
   differential erosion turns uniform uplift into grain. Already evaluated at fine;
   strengthen + ensure it's structurally (not just fBm) keyed.
3. **Sub-coarse forcing residual (optional, P1b).** Re-evaluate features at fine
   and take the part above the interpolated coarse — sharpens fronts near
   boundaries. Small away from boundaries (smooth fields interpolate ~exactly), so
   not the main content; do it only if structure placement is too soft.

## Rungs (climb in order; measure each)

| Rung | Change | Effort | Measure |
|---|---|---|---|
| **P1a** | Add `fine_detail` = structural relief (faults/folds) + erodibility, gated by the **already-transferred** fine feature fields, onto the base **inside `FineBase` before `generate_pre`** (see ordering trap). **Must include a zero-mean structural HEIGHT term across orogen interiors** (not just erodibility + front scarps — see "erodibility alone" trap). Scarp/detail params become hashed `FineBase` inputs (decision A); temperature lapse baseline split off (see traps). NO FeatureFields refactor. Codex confirms feasible: transferred `ElevationFields` carry trench/ridge/convergent/divergent/arc/collision/rift_delta/continentality gates (existing hooks `lithology_erodibility` and `apply_fault_scarps` already use them). LIMIT: transferred fields lack `activity`/`transform`/raw arc&collision distances/boundary strike → P1a structure is **soft/isotropic** (no crisp boundary-normal/strike-aware fronts). | med | mountain-top probe: summit gains sub-coarse relief (slope), erosion ORGANIZES (summit pit% falls, dendritic spiral gone); area-weighted land-fraction drift ~0; summit-zoom render (user judges) |
| **P1b** | Crisp/strike-aware fronts: **real refactor** (not a small source/target split — codex). `FeatureFields::compute` (features.rs:97) couples to the target mesh throughout: distance fields need `plates.cell_plate` on the *target* mesh (features.rs:1003), screened diffusion needs same-mesh plate membership (:1185), eval reads `crust.crust_type(i)` (:628), boundary extraction uses coarse adjacency (`collect_plate_boundaries`, boundary.rs:310). So P1b needs **explicit fine-cell plate + crust classification** (assign each fine cell a plate/crust from coarse geometry), then re-evaluate the distance/diffusion/feature pipeline on fine — or a new evaluator over coarse boundary-source primitives. | high | crisper fronts; same probes |
| **P1c** | Crust margin at fine — interpolate `Crust.signed_margin_distance` (continuous) for active/passive-margin structural contrast (passive = wide shelf, active = sharp front). | low–med | margin morphology |

**Stop-and-evaluate after P1a:** if strengthened structural relief breaks the
flat-top cottage cheese and erosion dissects real ranges, P1b/P1c are polish. If
it does NOT, the substrate wasn't the lever → reassess the erosion regime
(uplift/incision maturity) before the bigger refactor.

## Cost

- P1a: arithmetic over fine cells gated by existing transferred fields → seconds,
  dwarfed by erosion. The plate/atmosphere sim cost is unchanged.
- P1b: feature recompute at fine — the smoothing iterations scale with resolution
  (features.rs:116: more hops for the same physical distance) → seconds-to-low-
  tens, one-time, pre-erosion. This is the only real "earlier stages cost more".

## Traps

- **ORDERING (codex — important):** the structural detail is part of the
  *pre-erosion* base, so add it to `FineBase.base_elevation` **before**
  `FineWorld::generate_pre` builds pre-hydrology (fine.rs:133). Today
  `apply_fault_scarps` runs *later*, inside `FineSurface::generate` (fine.rs:423),
  so terminal-lake base levels (`terminal_lake_base_levels`, fine.rs:671) are
  computed on the *unfaulted* base — an inconsistency. Move structural relief into
  base construction (or before pre-hydrology), not into the erosion stage.
- **Fine-cache version bump (codex):** the base detail is *code-generated*, not an
  input field, so the input-hash cache key (fine_cache.rs:63) won't change and
  will serve stale bases. Bump `FINE_BASE_CACHE_VERSION` when base generation
  changes, AND hash the new fine-detail params into the key (see scarp-knob trap).
- **Scarp/detail params cross the cache boundary — DECIDED: (A) hash as fine-base
  inputs (codex round 2).** `fault_scarp_height` is today an `ErosionParams` rerun
  knob (erosion.rs:129), consumed at the *erosion* stage (fine.rs:427); `rerun_eroded`
  (fine.rs:166) re-runs erosion WITHOUT regenerating the cached `FineBase`. Moving
  structural relief into `FineBase` therefore makes that knob (and the sweep-harness
  `fault_scarp` / diagnose `--fault-scarp` overrides, app/world.rs:142,204,
  sweep.rs:105) **inert** unless the base regenerates. **Choice (A):** promote
  `fault_scarp_height` + the new `fine_detail` knobs into explicit `FineBase`
  generation inputs and **hash them into the cache key** — keeps them sweepable (the
  sweep harness is *for* tuning exactly these), at the cost of a base regen per
  sweep value. (Rejected (B) = freeze them as constants + retire the rerun knob:
  cleaner cache, but kills visual sweep-tuning of the substrate.) Net: the scarp
  knob migrates from `ErosionParams` to the fine-base input set.
- **Temperature-lapse baseline breaks silently (codex round 2 — important).**
  `FineSurface::from_eroded` corrects temperature by `eroded − base.base_elevation`
  (fine.rs:505), a deliberate no-op for the pre-erosion surface (eroded ==
  base_elevation). If `base_elevation` now carries `fine_detail`, the transferred
  coarse `fields.temperature` is still baked against the *coarse* relief, and NO
  lapse correction is applied for the new fine relief → pre-erosion summits get the
  wrong temperature (hence wrong evaporation/precip). **Fix:** store the interpolated
  coarse base separately as the lapse baseline (correct `eroded` against the *coarse*
  datum, not `base_elevation`), OR adjust `fields.temperature` by the fine-detail
  lapse when adding it. Pick the baseline-separation fix; it also keeps the eroded-
  surface correction honest.
- **Do NOT re-solve sea level at fine** (elevation.rs:163). Inherit the coarse
  shift via the interpolated term; keep `fine_detail` coarse-cell-local zero-mean.
- **`fine_detail` must be coarse-consistent** (zero-mean over a coarse cell) or it
  shifts land fraction and silently invalidates the coarse atmosphere — the very
  thing we're protecting. Verify land-fraction drift is ~0.
- **Structure must be a SUBSTRATE erosion carves, not painted final relief**
  (erosion-v2 noise philosophy). Fault/fold relief is the input; ridge-and-valley
  topography is the erosion OUTPUT. Don't paint the output.
- **Gate structure by orogen geometry**, not free fBm everywhere, or you get
  uniform noise (back to cottage cheese, just intentional). Faults/folds live in
  convergence belts; cratons stay quiet.
- **Erosion must be able to organize on it:** the point of structural relief is to
  give drainage a gradient. Confirm summit pit% / dendritic spiral DROP after
  erosion (not just that relief was added).
- **Erodibility alone does NOT test the thesis (codex round 2 — important).**
  Differential erodibility (`litho_sigma`/`litho_grain`) changes incision *rate* but
  gives a flat summit no initial gradient; `apply_fault_scarps` only sharpens *range
  fronts*, not orogen *interiors* — the exact place the cottage cheese lives. So P1a
  MUST include an **actual zero-mean structural HEIGHT term across high orogen
  interiors** (fault blocks / fold grain as real relief), not just stronger lithology
  + front scarps. Without it a null result is uninterpretable (can't tell "substrate
  didn't help" from "we never added interior relief").

## Validation

- **Objective:** the mountain-top plateau probe (diagnose) — summit pre vs eroded
  roughness + slope; expect summits to gain coherent sub-coarse relief and erosion
  to REDUCE pit%/spiral (organized drainage) rather than add it. **Area-weighted**
  land-fraction drift ~0 and sea-level shift unchanged (datum check — a plain mean
  check is insufficient, codex). Global curv-rms a sanity bound.
- **Visual (USER judges, [[hex3-sweep-image-reading]]):** summit-zoom sweep grids,
  before/after P1a, glacial off. The "real ranges vs cottage cheese" call is the
  user's.
- **Morphometry:** ridge-and-valley spacing, range-front facets, summit form
  (peaked/dissected vs flat) — distinguishes real structure from noise.

## Open decisions

1. **Start at P1a** (structural relief on interpolated base, existing transferred
   fields) and gate on the summit visual — vs commit to P1b's FeatureFields
   refactor up front? Lean: P1a first (cheap, tests the thesis).
2. **How much structural relief** is wanted (subtle vs dramatic fault/fold grain)?
   Sets the `FAULT_SCARP_HEIGHT` / `litho_grain` / interior-height targets — a user
   aesthetic call. RESOLVED how (not how-much): these knobs stay sweepable as hashed
   `FineBase` inputs (decision A), so the sweep harness tunes them visually.
3. **Couple to Phase 3 later?** This synthesis is the substrate the eventual
   coupled erosion↔uplift loop (erosion-v2 Phase 3) needs; keep that path open.

## References

- erosion-v2 Phase 1 + root cause #1 + noise philosophy:
  [`erosion-v2.md`](erosion-v2.md)
- [[hex3-summit-cottage-cheese]] (the localized diagnosis this fixes)
- Code: fine.rs:372 (interpolate, the term to augment), :1325/:96 (transferred
  fine fields incl. trench/ridge/convergent/divergent/arc/collision/rift_delta/
  continentality), :133 (generate_pre — add detail before this), :418/:423/:427
  (terminal-lake-before-scarp ordering), :505 (from_eroded temperature lapse
  baseline), :166 (rerun_eroded — re-runs erosion without base regen),
  features.rs:97/:1003/:1185 (compute, mesh-native distance + plate-screened
  diffusion — the P1b refactor surface), fine_cache.rs:63 (`FINE_BASE_CACHE_VERSION`
  + input hash), erosion.rs:129 (`fault_scarp_height` — migrates to fine-base input),
  elevation.rs:163,378 (sea-level datum — inherit, don't re-solve), crust
  `signed_margin_distance` (margin transfer).
