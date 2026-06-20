---
name: hex3-fine-synthesis-p1
description: June 2026 — erosion-v2 Phase 1 (P1a/P1b/P1c fine structural-relief synthesis) LANDED on sweep-harness; fixes the summit cottage-cheese; visual sign-off pending
metadata:
  type: project
---

erosion-v2 Phase 1 (fine-terrain synthesis, the fix for [[hex3-summit-cottage-cheese]] / root cause #1) — all three rungs implemented, codex-reviewed, numerically validated, pushed on branch `sweep-harness` (2026-06-20). Spec: `docs/specs/erosion-fine-synthesis.md`. Core code: `src/world/fine.rs` `add_interior_structural_relief` + `OrogenFronts`.

The fine base was the coarse elevation INTERPOLATED onto fine cells → flat orogen summits → erosion spiraled into cottage-cheese. P1 synthesizes a zero-mean mid-band structural HEIGHT field into `FineBase.base_elevation` BEFORE pre-hydrology (the SUBSTRATE erosion carves), gated to high orogen interiors, coarse-cell-local zero-mean (area-weighted) so it never moves the datum/land mask.

- **P1a** (knob `interior_relief`, default 0.04): isotropic fBm height grain. Also migrated `fault_scarp_height` ErosionParams→`FineStructureParams` (decision A: structural knobs are hashed fine-base inputs), split temperature-lapse baseline onto `coarse_base_elevation`, added area-weighted land-drift guard.
- **P1b** (knob `front_strike_weight`, default 0.7): strike-aware — banded `cos` of the distance to the nearest compatible convergent front ARC (`OrogenFronts` = coarse boundaries as great-circle segments, side-aware via overriding plate). Distance-to-arc (not point) → coherent front-parallel ridges, no bullseyes. The spec's "evaluator over coarse boundary primitives" path, NOT a full FeatureFields recompute.
- **P1c** (knob `margin_contrast`, default 1.0, clamped [0,1]): active/passive margin contrast — coastal-band amplitude scale from interpolated `signed_margin_distance` × convergent forcing (active coast sharpened, passive damped). Amplitude-only → land-safe.

Validated via diagnose mountain-top probe (seed 12345): summit pre-slope 1.7e-4→2.85e-3 (flat plateau broken), erosion REVERSES from adding summit pit% to organizing it, land-fraction drift +0.00e0, datum preserved. Each knob at 0 cleanly reduces to the prior rung. Cache `FINE_BASE_CACHE_VERSION` now 6; shape constants hashed.

**Still pending: the VISUAL sign-off (the user's call, per [[hex3-erosion-methodology]] / [[hex3-sweep-image-reading]]).** Numbers are necessary-not-sufficient; the strike-alignment + margin morphology are visual properties to judge via the Windows GPU sweep harness ([[hex3-sweep-harness]]): `--sweep interior_relief|front_strike_weight|margin_contrast`. The default constant values (amplitude, frequency, band width, factors) are first guesses to be tuned on the maps.
