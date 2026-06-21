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

**Climate (Codex step 4): left at `DEFAULT_CLIMATE_RATIO = 0.15`.** Integration handles the
sea-reaching goal at 0.15 with in-target endorheic; raising climate would only trade endorheic
(below target) for more lakes. Climate stays a user dial (Up/Down). Revisit if lakes look too
sparse (`lake_fraction ≈ 0.07%`).

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
