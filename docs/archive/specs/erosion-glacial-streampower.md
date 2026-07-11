> **Archived:** Historical design or experiment record. It does not define current architecture, defaults, or priorities. Start at [`docs/README.md`](../../README.md).

# Erosion: Glacial Stream-Power (fix the glacial "prickle")

**Provenance.** 2026-06-20. After the objective sweep + phase-split diagnosis
(codex-confirmed) showed the cell-scale roughness ("swiss cheese / prickle",
proxy `curv-rms`) is produced by the **glacial** pass, not fluvial erosion — see
[[hex3-prickle-is-glacial]] and below. Supersedes the fluvial root-cause
attribution in [`erosion-routing-ladder.md`](erosion-routing-ladder.md) and
[`erosion-escalations.md`](erosion-escalations.md) for the prickle specifically:
the fluvial engine is clean (curv-rms ≈ base); MFD Rungs 1–3 and escalations #1/#2
were fluvial/hillslope levers and so were no-ops on the (glacial) artifact.

**Philosophy (unchanged).** Unreasonably physically inspired; mechanisms over
hacks; parameters as playground knobs. This spec replaces a degenerate glacial
model with the literature's *large-scale-appropriate* one — not a cosmetic smooth.

## State of play (the evidence)

Headless `diagnose`, seed 12345, full fine res (~1.3M land):
- Phase-split roughness: base curv-rms 2.69e-3 → post-fluvial 2.62e-3 (flat) →
  post-glacial **2.05e-2** (~8× jump, entirely in the glacial pass).
- `--glacial-k 0` → final curv-rms back to 2.62e-3. Disabling glaciation removes
  the whole artifact.
- Decomposition (`--glacial-overdeepen-max`): 0.012→0.006→0 gives curv-rms
  2.05e-2→1.55e-2→1.07e-2 and pit% 0.57→0.39. So over-deepening = ~half the
  roughness + the closed pits; the other half (1.07e-2, still ~4× base) is the SFD
  abrasion itself.
- **Mechanism probe** (`glacial_erode`, info-gated): the SFD ice-discharge field
  is white-noise spiky — **checker 48.6%** (elev base ~41.6%), **pit 9.86%** of
  glaciated cells are local flux minima. Single-flow dumps all ice into one
  downstream cell and starves neighbours → spiky flux → spiky abrasion. **Ice is
  being point-routed like water.**

## The current model (what's wrong)

`glacial_erode` (src/world/erosion.rs:211–306), per step:
- builds **SFD** routing on the bed (`Routing::build(.., 0.0)`, line 256),
- accumulates ice load `(elev − snowline)` to the **single** receiver (line 271),
- abrades `elev -= glacial_k · flux`, floored at `pre[receiver] − overdeepen_max`
  (lines 286–287).

Versus the literature (web search 2026-06-20): the standard large-scale glacial
LEM **OpenLEM** (Liebl et al., GMD 16:1315, 2023) uses a glacial **stream-power
law** `E = K_g · A_i^m · S^n` (A_i = ice flux as equivalent catchment area, S =
**ice-surface slope**) and — crucially — keeps the flow from concentrating in one
cell by an explicit **glacier-width** parameterization `w ∝ A_i^α`. hex3's pass is
a **degenerate sub-case**: `E ∝ flux` only — **no slope term, no width spread, no
ice-surface model**. It is below the published baseline, which is exactly why it
alone produces the 1-cell artifact that even SFD literature models avoid.

(Gold standard is a distributed-ice model — SIA + higher-order, iSOSIA — giving
smooth diffusive over-deepenings and true U-valleys. The literature explicitly
positions the stream-power approach as a valid *large-scale complement*, not a
poor substitute, so SIA is a deferred frontier, not the target here.)

## Target law (hex3 mapping)

Per glaciated cell, per step:

```text
E_i = dt · K_g · A_i^m · S_i^n          (then clamp / over-deepen as today)
```

- **A_i — distributed ice flux.** Replace SFD accumulation with **MFD**: build the
  ice routing with a glacial MFD exponent (`Routing::build(.., glacial_mfd_exponent)`)
  and accumulate the ice supply down the MFD DAG. This is the Voronoi-native
  analogue of OpenLEM's glacier-width spread — flux fans across all downslope
  neighbours instead of one, which is the direct fix for the 48.6%-checker
  spikiness. Reuse the existing MFD infra: `build_mfd_flow` (erosion.rs:1200),
  `MfdFlow`, and an ablating variant of `accumulate_wet_area_mfd` (erosion.rs:1685)
  — the only delta is per-cell snowline ablation during the downstream pass
  (`flux[i] = max(0, flux[i] − ablation·below_i)` before distributing).
- **S_i — ice-surface slope.** hex3 has no ice-thickness field, so use the **bed
  slope toward the steepest-descent receiver** as the first-cut proxy:
  `S_i = max(0, (elev[i] − elev[receiver]) / dist)` (chord distance, already in
  `Routing.dist` / `NeighborGeometry`). Note this is the bed surface, not the ice
  surface — a known approximation (ice fills hollows, so true ice-surface slope is
  gentler over over-deepenings). Good enough to introduce the slope dependence;
  the ice-thickness upgrade is a later rung.
- **K_g, m, n — new knobs.** `GLACIAL_K` already exists (becomes K_g). Add
  `GLACIAL_M` (area exponent, ~1.0 start — OpenLEM range) and `GLACIAL_N` (slope
  exponent; OpenLEM ties n to the sliding-law exponent l, ~1–2). Keep
  `glacial_overdeepen_max`, `glacial_ablation`, snowlines, `glacial_steps`.

**Over-deepening coherence.** Keep over-deepening (it's a real feature: cirques,
tarns, fjord sills), but it should emerge at **valley scale**, not per cell. With
MFD flux it largely self-corrects (convergent ice over-deepens where flux is high
and smooth). Keep the `pre[receiver] − overdeepen_max` reverse-gradient cap as a
guard; once flux is distributed the cap fires on coherent valley floors instead of
isolated cells. (If pits persist, a small valley-scale smoothing of the floor —
not the elevation — is the next lever.)

## The rungs (climb in order; measure each)

| Rung | Change | Effort | Measure (vs current 2.05e-2) |
|---|---|---|---|
| **G0** | **Relabel diagnose** "eroded" → "fluvial" + add a "glacial" row (it currently mislabels the post-glacial surface as "eroded", confounding every metric). Prereq cleanup. | trivial | n/a (correctness) |
| **G1** | **MFD ice flux** — distribute A_i (reuse MFD infra + ablating accumulator); abrade ∝ A_i (current form otherwise). | med | flux checker% should fall toward ~42%; curv-rms toward the over-deepen-only ~1e-2 floor |
| **G2** | **Stream-power law** — `E = K_g·A_i^m·S_i^n` with the bed-slope proxy + `GLACIAL_M`/`GLACIAL_N`. | low (after G1) | curv-rms + visual U-valley/over-deepening shape; tune m,n |
| **G3** | **Over-deepening coherence** — verify pits are valley-scale; if not, valley-scale floor smoothing. | low–med | pit% back toward fluvial ~0.43%; tarns survive as coherent basins |
| **G4** (frontier) | **Ice-thickness (SIA-lite)** for a real ice-surface slope + a quarrying term. Deferred; only if G1–G3 leave the morphology wanting. | large | true ice-surface slope; U-valley cross-section |

A natural **stop-and-evaluate** after G2: if curv-rms is back near base and the
sweep-harness render looks right, G4 is unnecessary at planet scale (per the lit).

## Traps (heed before building)

- **MFD accumulator must use the ICE supply + ablation**, not precip×area. Don't
  blindly call `accumulate_wet_area_mfd` — fork it for the per-cell melt term.
- **Keep the SFD/flat fallback.** MFD on flats degenerates; the existing
  `flat_resolution` path in `Routing`/`build_mfd_flow` must still apply to ice.
- **Slope proxy sign/units.** `S_i` uses chord distance (radians) like fluvial
  incision; clamp `S ≥ 0` (a cell that has gone non-downhill between re-routes
  contributes no erosion, never negative). Watch the `S^n` blow-up on cliffs —
  cap S as the fluvial code caps slopes.
- **Mass / over-erosion.** The abrasion is explicit lowering; the over-deepen
  floor is the only stability guard. Re-tune `GLACIAL_K` after adding `A_i^m·S^n`
  (the magnitude changes) and watch the logged abraded volume + glaciated %.
- **Don't let MFD "fix" by just over-smoothing.** The goal is distributed flux,
  not blur — verify over-deepenings/U-valleys SURVIVE (morphometry), not just that
  curv-rms dropped (Goodhart: curv-rms can't tell a real tarn from a grid pit).
- **Bed-slope ≠ ice-surface slope.** Acceptable for G2, but document it; it will
  under-erode broad over-deepenings (where ice surface is steeper than the bed).

## Validation

- **Objective (diagnose, noise-free):** phase-split curv-rms (fluvial vs glacial),
  glacial ice-flux checker%/pit% (the mechanism probe), pit%/peak% of the final
  surface, carved-dissection, local relief. Target: glacial curv-rms back toward
  the fluvial ~2.6e-3 baseline (or a deliberate, defensible level), flux checker%
  toward ~42%.
- **Visual (USER judges — see [[hex3-sweep-image-reading]]):** sweep-harness grids,
  old vs new glacial law (and glacial-off), zoomed close-ups. The aesthetic call
  (prickly→clean, U-valleys/tarns present and plausible) is the user's, not the
  metric's.
- **Morphometry (the physical check):** U-valley cross-section, over-deepening
  long-profile (smooth sill vs spiky pits), cirque/valley spacing — distinguishes
  "real glacial landform" from "grid artifact" in a way curv-rms cannot.

## Does this fit the rendering? (coherence check)

Ice is **not a rendered entity** — there is no ice extent/thickness field. The
white "snow/ice" caps are a *decoupled cosmetic*: `apply_snow_cap` (coloring.rs:122)
whitens cells above a global `SNOW_LINE = 0.28` (+ latitude), plus a
`MATERIAL_ICE_SNOW` glint in unified.wgsl:196. So glaciation reaches the screen
**only as terrain shape**, and two things matter:

- **It is still legitimate as terrain.** Glacial landforms (U-valleys, cirques,
  tarns, fjords, arêtes) persist long after the ice — most of Earth's glaciated
  terrain has no ice on it now. So a glacial *erosion* pass that leaves recognizable
  morphology is justified with zero ice rendered. The morphology IS the payoff —
  which is exactly the bar the current (noise-producing) model fails and this rework
  must clear.
- **There is a coherence gap.** The erosion "where is ice" snowline
  (`GLACIAL_SNOWLINE_*`, latitude-varying ~0.02–0.30) ≠ the rendering snow-cap line
  (`SNOW_LINE = 0.28`, global). So "where it looks icy" and "where it's glacially
  carved" don't correspond. Cheap patch: relate the two snowlines (derive the
  cosmetic cap from the glacial snowline, or share a constant) so the caps roughly
  mark the carved zone.

Crucially, this re-weights the rungs: **G1–G3 improve glacial terrain but produce
NO renderable ice field** — the cap-vs-carving mismatch persists; they're justified
purely on the relict-terrain argument. **G4 (ice-thickness) is the rung that makes
glaciation a *coherent feature*** — one ice field drives both the erosion AND a real
ice render, retiring the cosmetic hack. "Most physical" (G4) is therefore also "most
coherent with rendering." Treat G4 as a deliberate *make-glaciation-a-feature*
project, not a bolt-on.

## Open decisions (user is the deciding vote)

1. **What is glaciation FOR?** This sets the whole scope:
   - *(a) Terrain character only* → do G1–G2, unify the snowlines as a cheap
     coherence patch, accept ice-as-cosmetic. Lowest effort; fixes the prickle.
   - *(b) A real feature* → climb to G4 (ice field drives erosion + a real ice
     render). Bigger, coherent end-to-end.
   - *(c) Not worth it* → default glacial **off** (it's been a noise source for a
     non-surfaced feature), ship fluvial + cosmetic caps, revisit later.
   Lean: (a) — do G1–G2, **gate hard on morphometry/visual**; if it doesn't read as
   glacial, fall to (c); pursue (b) only as a separate feature project.
2. **Default-on?** Glaciation is currently default-on (`GLACIAL_K>0`) and was
   producing the prickle. Until the new law's morphology is signed off, default it
   **off** (the safe state — option (c) is the fallback at every rung).
3. **Over-deepening intent:** how much closed-basin (tarn/fjord) character is
   *wanted* aesthetically vs read as "noise"? Sets the `overdeepen_max` target.

## References

- OpenLEM glacial stream-power, benchmarked vs iSOSIA — Liebl et al., GMD 16:1315
  (2023): https://gmd.copernicus.org/articles/16/1315/2023/
- Power-based abrasion law for LEMs — https://par.nsf.gov/servlets/purl/10550156
- Subglacial quarrying LEM — Ugelvig et al., JGR 2016:
  https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2016JF003960
- Glacial valley cross-section evolution — https://arxiv.org/pdf/0901.1177
- See also: [`erosion-routing-ladder.md`](erosion-routing-ladder.md) (MFD infra
  this reuses), [`erosion-escalations.md`](erosion-escalations.md),
  [`erosion-v2.md`](erosion-v2.md).
