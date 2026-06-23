# Drainage integration: making rivers reach the sea

Status: **IMPLEMENTED & merged 2026-06-22** (the core integration pass). Codex-reviewed.

## OUTCOME (what shipped)

A drainage-integration pre-pass (`integrate_basins` in hydrology.rs) carves outlet channels
so rivers reach the sea, run before the present-climate hydrology:
- **Selection (v1): micro-pit geometry** — breach basins that are small (`total_area <
  MICRO_BASIN_AREA`) or shallow (`spill−bottom < MICRO_BASIN_DEPTH`). Physically: low-relief
  basins integrate over geologic time; deep high-relief basins can stay endorheic (Caspian).
- **Carve:** `carve_outlet` walks the priority-flood `flood_parent` tree (oceanward) from each
  selected basin's lowest cell, lowering only the spill saddle / barrier into a monotonic
  descent (a thin notch, `CARVE_SLOPE`), then re-runs the hydrology on the carved surface.
- **Terrain consistency:** the carved elevation is adopted as the rendered fine surface
  (`from_eroded`), so the outlet channels are real water gaps. `integrate_basins` is
  idempotent on an already-carved surface (no closed basins → nothing to breach).
- Gated by `HEX3_NO_DRAINAGE_INTEGRATION` (A/B).

**Measured (`--drainage-audit`, seed 12345, area-weighted):** fine endorheic land
**42% → 17.2%** @40k (≈ Earth's 18%, in the 15–25% target), coarse 23% → 4.8%, fine−coarse
inflation **19 pp → ~5 pp** (Codex's <5–8 pp target). Basins 630 → ~180. 49 tests pass.

**Tested & rejected — the "pluvial overflow" criterion (Codex step 3).** Evaluating basins
under a wetter paleo-climate and breaching the ones that overflow is a *step function*, not a
tuning dial (off=17%, on=10.6% at any ratio 0.18–0.6) — it over-integrates below target with
no useful control. The micro-pit geometry proxy hits the target cleanly and is physically
defensible, so it's the shipped criterion. (Code path left available for future revisit.)

**Climate (Codex step 4): tried `0.3`, REVERTED to `0.15` — the climate-for-lakes retune does
NOT work.** Measured: climate `0.15` vs `1.5` give IDENTICAL lake_fraction (~0%) AND endorheic
land (17.2%) — climate is inert here. Root cause: **the integration breached the lake-holding
basins** (lakes ~0.8% pre-integration → ~0% after). A drained basin can't pond, and climate
can't refill a basin that's been carved open. A `"climate"` sweep knob was added as tooling
(`set_active_climate_ratio` post-gen).

**PROBE (2026-06-22, Codex-recommended, measured via `--drainage-audit` + integration log).**
Confirms breaching is the lake killer; the earlier "possible `calculate_water_levels` bug"
flag was a FALSE ALARM. Seed 12345, 40k, fine eroded surface:
- Lake-capable basins (depth ≥ `MIN_LAKE_DEPTH`) collapse **61 → 4** through integration.
  The 4 survivors DO pond (3/4 `is_lake` even at arid 0.15) — fill + extraction work; the
  geometry is simply gone. This is Codex's candidate (d).
- **Two mechanisms, both significant.** (1) Direct over-selection: `MICRO_BASIN_DEPTH=0.012 >
  MIN_LAKE_DEPTH=0.01` + the `OR` area gate breach lake-capable basins on their own clause.
  (2) Collateral carving: **2,449 carved cells landed in PRESERVED basins, 100% of them
  lake-capable** — a micro-pit's `carve_outlet` walks the global `flood_parent` tree to the
  sea and slices open deep basins en route. → Fixing the predicate alone is NOT enough; the
  carve must be made basin-aware (stop at / route into a preserved basin).
- **Climate is by-design inert, not buggy.** Endorheic-land is a `basin_id` topology metric
  (fixed at gen time); `set_climate_ratio` only moves water levels/bodies. And at 0.15 the few
  surviving lake-capable basins already overflow, so a wetter climate has no unfilled reserve
  to act on.
- Instrumentation kept: `carve_outlet` records lowered cells; `integrate_basins` logs
  lake-capable count + collateral carving; `--drainage-audit` prints a lake-capability
  breakdown (lake-capable basins, has-water/overflowing, `is_lake` bodies).

**FIX (2026-06-22): lake-aware breaching — lakes restored, climate dial live again.** Two
changes to `integrate_basins`/`carve_outlet`:
1. **Predicate (keep lake-capable basins):** breach a basin only if it's too shallow to hold a
   lake (`depth < MIN_LAKE_DEPTH`) OR a genuinely negligible deep spike (`area <
   TINY_SPIKE_AREA = 0.0002`). The old `depth < 0.012` + `OR`-area gate (which breached the
   `[0.01,0.012)` lake band and small deep basins) is gone. `MICRO_BASIN_AREA/DEPTH` removed.
2. **Carve-aware routing:** `carve_outlet` takes `basin_id` + `keep[]` and STOPS when its
   oceanward path reaches a preserved basin — the breached pit drains INTO that lake (which
   handles its own overflow) instead of slicing through it.

**Measured (seed 12345, 40k, fine eroded):** collateral cells through preserved basins
**4048 → 0**; lake-capable basins surviving **4 → 26**; `is_lake` bodies **3 → 17**;
lake_fraction **0.01% → 0.41%** (~40×); endorheic land **16.7% → 13.7%** (breached pits now
route into kept basins that overflow seaward). Climate is no longer inert: of 26 lake-capable
basins, 17 overflow at the arid 0.15 and 9 pond below spill, so a wetter climate has basins to
fill. (NOTE: this CHANGES the rendered fine terrain — carved water gaps differ — so it needs a
visual sweep before merge. Climate-sweep confirmation of the live dial is the next check.)
49 tests pass.

**CODEX PHYSICAL REVIEW (2026-06-22, 2 rounds) + hardening.** Codex endorsed the basin-aware
carve but rated the depth keep/breach as a *pragmatic heuristic*, not a geomorphic model.
Round-2 convergence + what we did about it:
- **Water-budget criterion is degenerate in THIS solver, root cause identified.** We'd already
  tested the "pluvial overflow" criterion (breach basins that overflow under a wetter paleo-
  climate) and found it a STEP FUNCTION (off 17% / on 10.6% at any ratio 0.18–0.6). Mechanism:
  a basin overflows when `catchment·(r/evap) ≥ capacity`, so `r_crit = capacity·evap/catchment`;
  `catchment` uses **first-basin attribution**, making network position near-BINARY (trunk →
  `r_crit≪1` always overflows; isolated pit → `r_crit≳1` never overflows). `r_crit` is bimodal
  with a gap across the usable climate range → a single multiplicative knob can't land in it.
  Codex agreed and RETRACTED "use paleo-overflow instead of depth"; the real fix is **storage-
  aware surplus transfer** (fill local basins, evaporate from lake area, pass only excess
  downstream) to de-bimodalize `r_crit` — deferred as the long-term water-budget model.
- **Depth accepted as an incision-resistance proxy** (sill height an outlet must cut), not just
  a ponding/render threshold — and it's *continuous*, which is why it gives a usable dial.
- **DONE — split thresholds.** `MIN_INTEGRATION_SILL_RELIEF` (integration keep) is now a
  separate constant from `MIN_LAKE_DEPTH` (lake render), even though both = 0.01 today.
- **DONE — over-connection audit** (`HEX3_OVERCONNECT_AUDIT`): re-solves water levels with
  breached-pit precip masked out (≈ pre-integration, water trapped) on the same carved geometry
  and counts kept basins that flip overflowing→not. **Result: 0/17 (0.0%)** on the fine eroded
  surface (0/1 pre-erosion, 0/2 coarse) — the cascade is NOT manufacturing connectivity; the
  16.7%→13.7% improvement is real. `integrate_basins` now returns the breached-cell mask.
- **DONE — `TINY_SPIKE_AREA` → `MIN_DEEP_CELLS` (deep-cell spike filter).** Replaced the
  scale-coupled absolute-area cutoff with a resolution-aware one-cell-spike test: a basin resists
  integration iff its DEEP body (cells submerged > `MIN_LAKE_DEPTH` at spill-full) spans
  ≥ `MIN_DEEP_CELLS` (=2) cells — i.e. its sill relief isn't carried by a single anomalous cell.
  Histogram-driven (`HEX3_SPIKE_HISTO` dumps sill-resisting basins bucketed by total/deep cells):
  on the fine eroded surface the deep-cell distribution is `deep=1: 18 spikes` then a coherent
  tail `deep≥2: 43`, so 2 is the natural knee. Resolution-robust (a real depression gains deep
  cells as the mesh refines; a true spike stays at 1). **Measured vs the area cutoff (fine eroded,
  seed 12345 40k):** kept basins 24→**43**, `is_lake` 17→**33** (lake_fraction 0.41%→**0.44%** —
  the recovered basins are small), endorheic land 13.7%→**16.7%** (back INTO the 15–25% target,
  ≈Earth 18% — the area cutoff had been over-breaching real small deep basins), collateral 0,
  over-connection **0/32 (0%)**. `MIN_DEEP_CELLS` is the tunable (raise to 3–4 for a stricter
  "minimum coherent lake" if speckle persists visually). 49 tests pass. CHANGES rendered terrain
  → still needs the visual sweep before merge.
- **Still open (roadmap):** resolution-scale `CARVE_SLOPE` (per-cell gradient). Explicit basin
  spill-hierarchy receiver (medium). Storage-aware surplus-transfer water budget (the principled
  long-term model that would revive the water-budget keep criterion).

**LAKES are now a separate follow-up — not a climate dial.** The integration traded lakes for
sea-reaching drainage. To get lakes back without re-breaking drainage, the lever is the
breach/keep BALANCE: breach only tiny noise pits, KEEP real basins, and let them fill to spill
and OVERFLOW so they become lakes *with outlets* (lake + drains to sea). That's a deliberate
rebalance of `MICRO_BASIN_*` + climate, not a one-knob bump — and Codex's warning ("overflow by
lake-filling masks the geometry") means it needs care.

---

Original plan (below). Codex-reviewed.

## Problem

Rivers look sparse and many "end inland" (terminate at interior lakes) instead of reaching
the coast.

## Evidence (measured — `diagnose --drainage-audit`, area-weighted, seed 12345)

"Endorheic land" = land draining to a non-overflowing interior basin instead of the sea.

| | endorheic land | basins (endorheic %) |
|---|---|---|
| **Coarse macro shape** | ~23–31% | ~30–50 (~91–96%) |
| **Fine (eroded)** | ~42–50% | ~600–1000 (~95%) |

Three stacked contributors:
1. **Coarse macro shape** already ~30% endorheic — the floor, baked into the coarse
   continents/orogens (intermontane basins, foreland lows, rain-shadow interiors).
2. **Fine refinement** adds ~+18 pp — structural relief + erosion carve 600–1000 small
   interior pits (basin count explodes ~20–30×) that don't overflow.
3. **Arid default climate** (`DEFAULT_CLIMATE_RATIO = 0.15`, hydrology.rs:165) — most basins
   never fill to their spill point, so ~95% read as non-overflowing.

## Root cause (verified)

The generator does depression **FILLING** (`priority_flood_with_basins`) but **NO drainage
INTEGRATION** — there is no breach/carve/incision code in `hydrology.rs`. A basin is endorheic
whenever `water_level < spill_elevation` *today* (`Basin::is_overflowing`). So every basin that
doesn't fill *now* is treated as a permanent present-day sink.

But real drainage networks are **historical scars**: basins get breached during wetter
(pluvial) epochs or captured by headward erosion over geologic time. The model has no memory
of that → it is **too geomorphically young** → 30–50% endorheic. *Fill tells you where water
would pond; integration decides where time has cut an outlet.*

## Architecture context
- Hydrology runs **once, on the fine mesh** (coarse `world.hydrology` is None). Fine elevation
  = interpolated coarse + structural relief + erosion, so macro basins are coarse-inherited.
- `Basin` (hydrology.rs:38): `spill_elevation`, `water_level`, `spill_target_cell` (cell just
  OUTSIDE the basin where overflow exits), `overflow_target: Option<usize>` (None = spills to
  ocean). It does NOT currently store the inside spill **saddle** cell — needed for breaching.

## The fix — a basin-integration / outlet-breaching pass

NOT "force terrain to drain coastward" (the closed basins are physically correct — keep them)
and NOT "just crank the climate wetter" (masks the geometry — overflow by lake-filling →
too many lakes, climate-swingy drainage). The physical, high-leverage fix:

1. Run the current priority-flood + basin detection.
2. Build basin groups + spill links (which basin spills into which).
3. Evaluate each basin/group under a **wetter "geologic integration" climate**
   (`INTEGRATION_CLIMATE_RATIO` > present; "pluvial highstands + long-term capture", not
   today's lake area).
4. **Mark for integration** basins that are shallow/small artifacts, would overflow under the
   pluvial climate, or have enough discharge/relief to plausibly incise their outlet.
5. **Breach** those outlets: lower a narrow channel through the spill saddle and along the
   downstream path.
6. **Rerun hydrology on the modified fine surface at the ACTUAL present climate.**

→ Drainage reflects geologic HISTORY, not just current lake levels. Arid worlds still keep
genuine endorheic interiors (a deliberate climate/tectonic outcome, not the baseline).

Code touchpoint: extend `Basin` to store the inside spill saddle cell/edge (scan basin
boundary edges after `basin_id` is known) — the current per-basin spill data can't recover
true external outlets through chains (the cycle limitation around hydrology.rs:1023).

## Ordered plan
1. **Extend the audit** (`--drainage-audit`): endorheic **discharge** (precip-weighted flow
   failing to reach ocean), endorheic area by basin size/depth bucket, coarse-vs-fine delta,
   and a climate sweep (0.15 / 0.25 / 0.35 / 0.5 / 1.0). Measure before changing behaviour.
2. **Micro-basin breaching** (fine mesh): aggressively breach shallow/small pits → cut the
   fine-only inflation from +18 pp toward **<5–8 pp**, without touching real macro basins.
3. **Geologic basin integration** (the big one): steps 1–6 above.
4. **Retune `DEFAULT_CLIMATE_RATIO`** to ~0.25–0.35 (river-rich but not wetland-dominated);
   keep 0.15 as an arid preset.

## Success criteria (measure with the audit, across seeds)
- Default fine endorheic land median **15–25%** (Earth ≈ 18%).
- Fine − coarse endorheic inflation **<5–8 pp**.
- Micro/small terminal pits **<5%** of land area.
- Most high-discharge rivers reach the ocean (or a major terminal lake *intentionally*
  classified as endorheic).
- Arid preset still preserves large inland basins.

## What this is NOT
- Not a mountain-relief change (per the elevation-first gate, this is rivers/hydrology).
- Not the deferred-but-separate atmosphere/coarse-asymmetry work.
