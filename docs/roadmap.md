# hex3 roadmap / idea inventory

Living record of candidate work + status, so ideas don't get lost between sessions.
Goal of the project: **good-looking mountains & mountain ranges** ("unreasonably physically
inspired", not Earth-accurate).

## Process rule (the rain-shadow lesson, 2026-06-22)
Every "better mountains" ADD must pass an **elevation-first A/B** (same mesh/seed) BEFORE it
counts: mountain-mask `p50/p90/p99 |Δelevation|`, slope `p90/p99`, blind render. **Derived
fields (flow, precip, climate) do NOT count unless elevation also moves.** Bar: p90 mountain
`|Δ| ≥ 0.01`. (Rain-shadow's p90 was 0.003 — it moved rivers, not mountains.) → Build this
probe into `diagnose` before the next mountain ADD.

## Architecture first-principles review (2026-06-21)

### LOSE — DONE (merged to main)
- ✅ coarse-asymmetry envelope (default-off, double-applied with O0)
- ✅ dead `transform` feature field (zero terrain consumers; classifier kept)
- ✅ write-only atmosphere struct fields (continentality/upper_temperature/pressure/moisture/phi)
- ✅ orphaned `normalize_positive_field`
- ⏸️ glacial pass — KEEP-DISABLED (scaffold for the MFD-on-ice rework), not deleted
- ⏸️ coarse-hydrology PREVIEW — trim-only (flow-prior fast path), LOW priority

### ADD — status & priority (re-gated by the elevation-first rule)
1. ✅ **Rain shadow** (downwind lee drying) — DONE, default-off. RECLASSIFIED: a river/climate
   feature, NOT a mountain feature (m/n=0.25 → precip is a weak relief lever). Useful 0.2-0.4.
2. ⬜ **O0 along-strike passes / water-gaps** — as an UPLIFT/base-shape change (O0 segmentation
   floors at 0.15 so crests never break). Codex rates FIRST-ORDER on relief: summit-gap-summit
   rhythm + river outlets. **Top mountain candidate.** Needs fine-cache bump.
3. ⬜ **Marine deposition** (deltas / shelf aprons) — extend `deposit()` below sea level.
   Coastlines, not mountains. Small-medium effort.
4. ⬜ **Stratigraphic / depth-dependent erodibility** (K = f(depth below a folded datum)) —
   benches / cuestas / cap-rock. Erosion TEXTURE; speculative; hot-path cost. Lower priority.

## Mountain-quality open issues
- ⬜ **Wide-mountain residual spikiness** — root-caused to the FINE-BASE synthesis (pre-erosion,
  cell-scale), not erosion. Fix = flat/coherent broad-massif macro shape + better fine-base noise.
- ⬜ **Coarse-asymmetry single-owner redesign** (C/A hybrid) — for atmosphere rain-shadow
  consistency; deferred (one shared OrogenFronts signed-distance product; coarse owns envelope,
  O0 drops its front-profile; collision stays symmetric).

## Other candidate directions
- 🆕 **River re-work / analysis** — rivers look bad graphically in 3D (and possibly drainage
  quality). UNDER CONSIDERATION (raised 2026-06-22). Needs a grounded analysis: rendering
  (how river ribbons are drawn/displaced on the 3D relief) + hydrology (network plausibility).
- ⬜ **Perf** — default gen ~3× slower (n=2 Newton + 200 erosion steps); `EROSION_STEPS` is the
  quality↔speed dial. Revisit if iteration speed bites.

## Validated current state (the baseline to beat)
O0 structured-emergent uplift (asymmetric front + along-strike segmentation) + n=2 incision +
channel_support≈4 + hillshade render fix. Shipped as the default. See
`docs/specs/orogen-structure.md`.
