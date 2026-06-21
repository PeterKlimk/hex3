# Fine-mesh downwind rain shadow (side-aware orographic precipitation)

Status: IMPLEMENTED, default-off. **RECLASSIFIED to a river/climate-realism toggle — it is
NOT a mountain-shape feature** (see OUTCOME). Codex-reviewed twice (design + outcome).

## OUTCOME (2026-06-22): a DRAINAGE feature, not a relief feature

Implemented and measured (seed 12345, stage 4, downwind 0.0 vs 0.6, same base mesh, 648k
land cells):
- **Flow accumulation (rivers/discharge): median +30.6%, p90 +90%, 86% of cells shift >10%**
  — substantial drainage reshaping (windward-dense / lee-sparse).
- **Elevation (the landform): median |Δ| = 0.00000, p90 = 0.00315** — essentially unchanged.

So the rain shadow moves rivers a lot and mountains almost not at all. This is arithmetic,
not a bug: with `EROSION_M=0.5, EROSION_N=2.0`, equilibrium slope `S ∝ A^(−m/n) = A^(−0.25)`,
so halving lee discharge gives only ~18% steeper slope (the "spikier lee" seen in renders);
incision sees `A^m`, so +30% flow = only ~14% incision-power change. Compounded by transient
(non-equilibrium) erosion that re-erodes from `structured_base` each outer pass, the
mean-invariant renorm removing net denudation, and uplift being precip-independent.

**The original "priority-1, HIGH on visible mountains" estimate was WRONG** — it conflated
precip-asymmetry with relief-asymmetry. Lesson adopted: every ADD is now gated on an
**elevation-first A/B** (mountain-mask p90 |Δelev| ≥ 0.01; derived fields like flow/precip
do not count). Rain-shadow's p90 (0.003) fails that bar by 3×.

**KEEP as:** asymmetric river networks / hydrology / climate maps. Default-off. Useful range
~0.2–0.4 (0.6 = aggressive diagnostic; audit median ~0.52, floor-hit ~27%). **Do NOT** put it
on the mountain-shape roadmap or build a non-mean-invariant "dry the planet" variant for
relief — that's the wrong lever (Codex-confirmed).

---

Original design notes (Codex-reviewed → revised; global-axis ordering and fetch units were
fixed — see "Codex revisions" below). Default-off; lands dark and is swept before any merge.

## The gap

`fine_precipitation` (src/world/fine.rs:1458) is a **purely local, per-cell** operator:

```
oro[i]   = wind[i] · ∇elev[i] · height_factor          // signed: + windward, − lee
precip[i]= coarse_precip[i] · clamp(1 + s·oro_norm, OROGRAPHIC_PRECIP_MIN, _MAX)
```

A flat lee basin behind a crest has `wind·∇elev ≈ 0`, so it **snaps back to coarse precip
and never progressively dries downwind**. The coarse atmosphere (moisture.rs) does genuine
advective transport, but only at ~100 km resolution; its rain shadow is transferred as the
base `coarse_precip`. The missing increment is **downwind PROPAGATION of the lee dry-anomaly
at fine scale**.

Why it matters & why it's sound for THIS solver: precip → discharge (`precip × steradian`)
sets BOTH the channel-initiation threshold `a_crit` (erosion.rs:556, from mean land precip)
AND the `A^m` incision rate (erosion.rs:1241). So asymmetric precip → **visibly asymmetric
wet-windward / dry-leeward dissection**. Precip is also the field hydrology/rivers/lakes
consume later (fine.rs:837, hydrology.rs:239), so it is the correct shared integration point
— modulating discharge or `a_crit` directly would desync the final rivers. It re-weights the
already-advected coarse precip — paints no relief, smooths nothing → trips no validated
failure mode.

## The known trap (designed around)

A prior FULL moisture-transport attempt on the adaptive fine mesh **over-dried** it: tiny
fine cells → CFL `dt` collapse → rainout outran advection → interiors desiccated. The project
reverted to local-only. This design is therefore **NOT advection** (no timestep) — it is a
single bounded, ordered relaxation over a receiver DAG.

## Design — two phases inside `fine_precipitation`

**Phase 1 (unchanged):** the existing local modulation → `precip_initial[i]`. Byte-identical
when the new knob is 0.

**Phase 2 (new, gated on `DOWNWIND_SHADOW_STRENGTH`, default 0):**

1. **Lee dry-anomaly seed:** `D[i] = max(0, coarse_precip[i] − precip_initial[i])` on land
   (`elev>0`), else 0. (Honest framing: this is the *local lee negative anomaly*, a bounded
   fine-scale dry-anomaly to propagate — **not** true air-column rainout. Good enough to
   reweight discharge; see open decision 2.)
2. **Per-cell downwind receiver (LOCAL wind, not a global axis):** for each land cell `i`
   with `|wind[i]|² > ε`, receiver = the land neighbor `j` maximizing
   `cos(wind[i], c_j−c_i)`, accepted only if `cos > DOWNWIND_CONE_COS`. Cells with near-zero
   wind, or no qualifying neighbor, get **no receiver** (their anomaly rains out locally).
   Out-degree ≤ 1 → a **functional graph**.
3. **Break cycles → DAG:** a functional graph's cycles are disjoint simple loops (winds that
   curve back). Detect via path-coloring in O(N); in each cycle remove the **weakest edge**
   (lowest `cos`). Result is a forest-of-DAGs.
4. **Topo sweep (sources→sinks), one O(N) pass:** process each cell before its receiver
   (Kahn on in-degree). `accum[i]` starts at `D[i]`; carry
   `carried = accum[i]·exp(−chord_dist(i,r)/fetch_chord)` into `incoming[r]` (and `accum[r]`);
   the un-carried remainder is dropped (rained out — cannot amplify). Because of topo order,
   every cell is final before it's read. **Deterministic** (fixed graph + fixed order).
5. **Apply the dry-anomaly under a floor:** `precip_shadow[i] = max(precip_initial[i] −
   DOWNWIND_SHADOW_STRENGTH·incoming[i], DOWNWIND_SHADOW_FLOOR·coarse_precip[i])`.
6. **Mean-invariant renorm (reuse fine.rs:1508-1526):** rescale to the area-weighted land
   mean `PRECIP_GLOBAL_SCALE`. The lee's lost moisture returns globally as a uniform
   multiplier → the **global budget hydrology calibrates against is unchanged; only the
   spatial distribution shifts** windward-wet / lee-dry.
7. **Post-renorm AUDIT (log, every run with strength>0):** because renorm runs *after* the
   floor, the floor is **soft after rescale** (a sub-1 multiplier can dip cells below it, a
   >1 multiplier can push windward past `OROGRAPHIC_PRECIP_MAX`). Log the area-weighted
   median / p10 / p25 of `precip_shadow/coarse_precip`, the floor-hit fraction, the
   over-`MAX` fraction, and the **largest connected dry land region** — the honest collapse
   tripwires (median alone misses a big *regional* failure under half the land).

### Over-dry guardrails (four, stacked)
1. **Floor** (`≤ OROGRAPHIC_PRECIP_MIN=0.3`) — applied pre-renorm; audited post-renorm (7).
2. **Fetch-bounded decay in RADIANS/chord** (`exp(−chord/fetch_chord)` per hop) — the plume
   dies after a bounded **arc length**, independent of cell count → resolution-robust (the
   explicit fix for the trap: cell-count dependence → arc-length dependence).
3. **Single ordered pass over a DAG, no CFL** — the resolution-dependent `dt` that caused the
   trap does not exist.
4. **Mean-invariant renorm** — the *mean* land precip is pinned by construction; only the
   *spread* changes (so the median/regional audit, not the mean, is the over-dry signal).

## Insertion & threading

- **Insertion:** new helper `downwind_lee_shadow(tess, elevation, wind, coarse_precip,
  precip_initial, strength, fetch_chord) -> Vec<f32>` in fine.rs, called inside
  `fine_precipitation` between Phase 1 and the existing renorm block. Reuses `chord_gradient`,
  `tess.neighbors / cell_center / cell_areas`, and the renorm block.
- **Combined early-return:** today's `if strength <= 0.0 { return coarse_precip }` becomes
  `if local_strength <= 0 && downwind_strength <= 0 { return coarse_precip }`. (With local=0
  the seed `D` is 0, so downwind is a no-op by construction — it propagates the LOCAL lee
  anomaly.)
- **Threading (mirror `orographic_precip_strength`):** new `ErosionParams.downwind_shadow_strength`
  (default `DOWNWIND_SHADOW_STRENGTH`); read at the call site (fine.rs:820); override in
  app/world.rs; sweep name `"downwind_shadow"` in app/sweep.rs (registry + apply_knob).
- **Ordering guarantee:** precip is finalized inside `fine_precipitation` and returned to the
  `precip` loop var BEFORE the next `erode()` (fine.rs:801) consumes it as discharge.

## Knobs (all new, in constants.rs)

| Constant | Meaning | Default |
|---|---|---|
| `DOWNWIND_SHADOW_STRENGTH` | master enable; fraction of incoming dry-anomaly applied (0 = OFF, bit-identical) | `0.0` |
| `DOWNWIND_SHADOW_FETCH_KM` | fetch decay length in **km** (converted to chord via `PLANET_RADIUS_KM`); bounds reach | `300.0` (≈0.047 rad) |
| `DOWNWIND_SHADOW_FLOOR` | floor as fraction of coarse precip (≤ 0.3; audited post-renorm) | `0.25` |
| `DOWNWIND_CONE_COS` | min cos(wind, i→receiver) for a downwind neighbor (0.5 ≈ ±60°) | `0.5` |

## Open design decisions (resolved post-Codex)

1. **Sweep order axis** — ~~global mean wind~~ → **LOCAL per-cell wind receivers + cycle-break
   + topo sweep.** A single global axis is broken here: the circulation is latitude-banded
   with zonal reversal (circulation.rs), so the global mean wind is ~zero/meaningless.
2. **Seed** — **lee-anomaly only** for v1 (budget-safe, minimal), documented honestly as a
   fine-scale dry-anomaly, not air-column rainout. (v2 could deplete the windward column.)
3. **Apply every outer erode↔precip pass** (matches `orographic_precip_strength`;
   `EROSION_PRECIP_OUTER_ITERS=2` bounds it).
4. **Floor independent constant, ≤0.3**, with the post-renorm audit (7) as the real check.

## Possible v1.1 (only if artifacts appear)
- **Multi-receiver split** (2–3 downwind neighbors weighted by `cos·edge_length`) if the
  single-receiver plume shows mesh-aligned filaments. Still one-pass if the graph stays a DAG.

## Validation plan

1. **Determinism / default-unchanged (WSL2, first):** with `DOWNWIND_SHADOW_STRENGTH=0`,
   stage-2/4 export byte-identical to a pre-change build. `cargo test`: a fine-path analogue
   of `precipitation_normalized_and_finite` (area-weighted land mean == `PRECIP_GLOBAL_SCALE`
   ± 1e-4, all finite, strength>0). Unit-assert the receiver graph is acyclic after cycle-break.
2. **Over-dry audit (the tripwire):** the Phase-2 step-7 logging. Trap = median sliding below
   ~0.85, floor-hits past ~25-30%, or a large connected dry region, as strength rises.
3. **Windows diagnose sweep:** `--sweep downwind_shadow --sweep-values 0,0.3,0.6,1.0,1.5`.
   Pair with an OBJECTIVE probe — windward-vs-lee channel-density ratio (flow-accum-above-
   threshold per area, split by sign of `wind·∇elev` across major ranges); expect >1 and
   rising (necessary-not-sufficient).
4. **Blind-rank (user judges aesthetics; I do NOT narrate fine images — confabulation risk):**
   unlabeled strength tiles → user picks best windward-dense/lee-sparse dissection, flags
   over-dried.
5. **Resolution check:** re-sweep at `--cells/--fine-scale` variants — asymmetry should persist
   (fetch is arc-length) and median land precip must not drift with N.

## Effort

~80-120 lines (up from the first estimate — local-wind receivers + cycle-break + topo sweep
adds ~25 lines over the global-axis sort): one helper + a ~6-line insertion + one
`ErosionParams` field + 4 constants + ~3 lines each in world.rs/sweep.rs + the audit logging.
No erosion-loop restructuring, no touching moisture.rs/erosion.rs internals. Risk is in
TUNING (fetch + floor on Windows), not code volume. Default-off → lands dark, swept before
merge.

## Codex revisions (second opinion, verified in code)
- **Global-axis ordering → local receivers + toposort** (verdict's biggest fix; the global
  mean wind is meaningless under zonal reversal — verified circulation.rs).
- **Fetch unit bug fixed:** `_SR`/steradians + `0.0008` (≈5 km) → `_KM` (300 km ≈ 0.047 rad),
  converted via `PLANET_RADIUS_KM` (verified the math).
- **Near-zero wind skip** in receiver selection (`|wind|² > ε`).
- **Floor is soft after renorm** → added the post-renorm audit (median/p10/floor-hit/over-MAX/
  largest-dry-region) instead of claiming a hard floor.
- **Deficit-only reframed** honestly as a fine-scale dry-anomaly, not air-column rainout.
- **Confirmed precip is the right integration point** (hydrology consumes it downstream).
