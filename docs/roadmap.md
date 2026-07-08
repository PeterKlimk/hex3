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
- **River RENDERING re-work — ✅ DONE (merged 2026-06-22).** Floating quad ribbons → rivers
  baked into an equirect **SDF texture** the terrain shader samples by lat/lon and
  reconstructs as **thin, crisp, screen-space-AA'd** rivers draped on the surface; water-
  shaded (fresnel/glint, distinct from ocean). The existing **Off/Major/All** river mode is
  now the real density control (is-major in the SDF B channel; was non-functional). Non-relief
  modes keep line rivers; dead quad path removed.
- ✅ **River HYDROLOGY pass — DONE (merged 2026-06-22), see `docs/specs/drainage-integration.md`.**
  Root cause (measured + Codex-confirmed): drainage did depression FILLING but no INTEGRATION
  (~30% endorheic coarse, ~50% fine). Shipped a basin-integration pre-pass that carves outlet
  channels (micro-pit geometry criterion + carve along the priority-flood spill path), adopted
  into the rendered terrain as real water gaps. **Fine endorheic 42% → 17.2%** (≈Earth), rivers
  now reach the sea. The "pluvial overflow" criterion was tested and rejected (step function,
  over-integrates); climate left at 0.15 (a user dial). Possible follow-ups: lakes too sparse
  (climate), and a targeted high-discharge criterion if a major river still ends inland.
- ⬜ **Lakes** — the drainage integration traded lakes for sea-reaching rivers (lakes ~0.8% →
  ~0%; breached basins can't pond). **DIAGNOSED (2026-06-22 probe + Codex):** lake-capable
  basins collapse **61 → 4** through integration via TWO mechanisms — (1) direct over-selection
  (`MICRO_BASIN_DEPTH 0.012 > MIN_LAKE_DEPTH 0.01` + `OR` area gate breach lake-capable basins),
  (2) collateral carving (a micro-pit's outlet `carve_outlet` walks the global flood-parent tree
  and cuts THROUGH preserved deep basins — 2,449 carved cells, 100% in lake-capable basins).
  Climate inertness is by-design (endorheic = topology metric), NOT a `calculate_water_levels`
  bug. **Fix (both needed):** (a) fix the predicate so lake-capable basins aren't directly
  breached (lower `MICRO_BASIN_DEPTH` below `MIN_LAKE_DEPTH` / guard `!lake_capable`), AND
  (b) make `carve_outlet` basin-aware — stop at / route into a preserved basin rather than
  slicing through it. Then fill-to-overflow → lakes WITH outlets. See
  `docs/specs/drainage-integration.md`.
- ⬜ **Perf** — default gen ~3× slower (n=2 Newton + 200 erosion steps); `EROSION_STEPS` is the
  quality↔speed dial. Revisit if iteration speed bites.

## Validated current state (the baseline to beat)
O0 structured-emergent uplift (asymmetric front + along-strike segmentation) + n=2 incision +
channel_support≈4 + hillshade render fix. Shipped as the default. See
`docs/specs/orogen-structure.md`.
