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

- **Datum preserved:** the interpolated term carries the coarse sea-level shift
  exactly; `fine_detail` is ~zero-mean at coarse scale, so land fraction / sea
  level barely move (verify, see Validation). No fine sea-level re-solve.
- **Coarse-field validity preserved (the concern):** because the synthesized
  surface AGREES with the coarse elevation at every scale the coarse fields can
  see, and only adds detail BELOW the coarse Nyquist where the coarse atmosphere/
  etc. had no information anyway, the coarse-derived fields stay valid at the
  scales they operate. You are filling in *under* them, not invalidating them.
- **Relief-sensitive couplings are already re-derived at fine:** orographic precip
  (rain-shadow) and the temperature-lapse re-application. The genuine residual
  (planetary wind deflected by the new fine ranges) needs fine *atmosphere
  simulation* — deliberately punted; second-order (circulation doesn't resolve
  individual ranges; the orographic proxy carries the first-order effect).

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
| **P1a** | Add `fine_detail` = structural relief (faults/folds) + erodibility, gated by the **already-transferred** fine feature fields, onto the interpolated base. NO FeatureFields refactor. | med | mountain-top probe: summit gains sub-coarse relief (slope), erosion ORGANIZES (summit pit% falls, dendritic spiral gone); summit-zoom render (user judges) |
| **P1b** | If P1a placement is too blurry: refactor `FeatureFields::compute` (features.rs:97) to **decouple source (coarse boundary midpoints) from target (fine cells)** and re-evaluate distance fields at fine → crisp range-fronts. | high | crisper fronts; same probes |
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

- **Do NOT re-solve sea level at fine** (elevation.rs:163). Inherit the coarse
  shift via the interpolated term; keep `fine_detail` zero-mean at coarse scale.
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

## Validation

- **Objective:** the mountain-top plateau probe (diagnose) — summit pre vs eroded
  roughness + slope; expect summits to gain coherent sub-coarse relief and erosion
  to REDUCE pit%/spiral (organized drainage) rather than add it. Land-fraction /
  sea-level drift ~0 (datum check). Global curv-rms a sanity bound.
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
   Sets the `FAULT_SCARP_HEIGHT` / `litho_grain` targets — a user aesthetic call.
3. **Couple to Phase 3 later?** This synthesis is the substrate the eventual
   coupled erosion↔uplift loop (erosion-v2 Phase 3) needs; keep that path open.

## References

- erosion-v2 Phase 1 + root cause #1 + noise philosophy:
  [`erosion-v2.md`](erosion-v2.md)
- [[hex3-summit-cottage-cheese]] (the localized diagnosis this fixes)
- Code: fine.rs:372 (interpolate, the term to augment), :1285/:96 (transferred
  fine fields), features.rs:97 (compute, source/target decouple for P1b),
  elevation.rs:163,378 (sea-level datum — inherit, don't re-solve), crust
  `signed_margin_distance` (margin transfer).
