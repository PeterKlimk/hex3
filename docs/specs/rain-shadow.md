# Fine-mesh downwind rain shadow (side-aware orographic precipitation)

Status: DESIGNED (not implemented). Priority-1 ADD from the first-principles architecture
review. Default-off; lands dark and is swept before any merge.

## The gap

`fine_precipitation` (src/world/fine.rs:1458) is a **purely local, per-cell** operator:

```
oro[i]   = wind[i] · ∇elev[i] · height_factor          // signed: + windward, − lee
precip[i]= coarse_precip[i] · clamp(1 + s·oro_norm, OROGRAPHIC_PRECIP_MIN, _MAX)
```

A flat lee basin behind a crest has `wind·∇elev ≈ 0`, so it **snaps back to coarse precip
and never progressively dries downwind**. The coarse atmosphere (moisture.rs) does genuine
advective transport, but only at ~100 km resolution; its rain shadow is transferred as the
base `coarse_precip`. The missing increment is **downwind PROPAGATION of the lee deficit at
fine scale**.

Why it matters & why it's sound for THIS solver: precip → discharge (`precip × steradian`)
sets BOTH the channel-initiation threshold `a_crit` (erosion.rs:347, derived from mean land
precip) AND the `A^m` incision rate. So asymmetric precip → **visibly asymmetric
wet-windward / dry-leeward ridge-and-valley dissection** across crests. It re-weights the
already-advected coarse precip — paints no relief, smooths nothing → trips no validated
failure mode.

## The known trap (designed around)

A prior FULL moisture-transport attempt on the adaptive fine mesh **over-dried** it: tiny
fine cells → CFL `dt` collapse → rainout reactions outran advection → interiors desiccated.
The project reverted to local-only. This design is therefore **NOT advection** (no timestep)
— it is a single bounded, ordered relaxation.

## Design — two phases inside `fine_precipitation`

**Phase 1 (unchanged):** the existing local modulation → `precip_initial[i]`. Byte-identical
when the new knob is 0.

**Phase 2 (new, gated on `DOWNWIND_SHADOW_STRENGTH`, default 0):**

1. **Lee-deficit seed:** `D[i] = max(0, coarse_precip[i] − precip_initial[i])` on land
   (`elev>0`), else 0. This is the moisture the local term just suppressed at the crest/lee
   — the budget we now carry downwind instead of letting it snap back.
2. **Downwind order (built once):** sort land cells by **decreasing** wind-projection key
   `s[i] = ŵ_global · cell_center[i]` (ŵ_global = area-weighted mean unit wind). A single
   global axis → a deterministic total order, no iteration.
3. **Receiver (precomputed once):** each cell's single downwind receiver = the land neighbor
   `j` maximizing `cos(wind[i], c_j−c_i)`, accepted only if `cos > DOWNWIND_CONE_COS`
   **and** `s[j] < s[i]` (strictly downwind in the global order). The strict-`s` gate makes
   the receiver graph a **DAG along `s`**, so one ordered pass is exact (no cycles, no
   convergence loop). No qualifying neighbor → plume terminates (rains out locally).
4. **Single O(N) sweep (upwind→downwind):** `accum[i]` starts at `D[i]`. In sorted order,
   each cell carries `carried = accum[i] · exp(−chord_dist(i,r)/FETCH_SCALE_SR)` into its
   receiver's `incoming[r]` and `accum[r]`; the un-carried remainder is dropped (rained out,
   cannot amplify). Because we process in strictly-decreasing `s` and push only to smaller
   `s`, every cell is final before it's read.
5. **Apply under a HARD FLOOR:** `precip_shadow[i] = max(precip_initial[i] − strength·incoming[i],
   DOWNWIND_SHADOW_FLOOR · coarse_precip[i])`.
6. **Mean-invariant renorm (reuse fine.rs:1508-1526 verbatim):** rescale to the area-weighted
   land mean `PRECIP_GLOBAL_SCALE`. Whatever the lee loses is returned globally as a uniform
   multiplier → the **global precip budget hydrology calibrates against is unchanged; only
   the spatial distribution shifts** windward-wet / lee-dry.

### Over-dry guardrails (four, stacked)
1. **Hard floor** (`≤ OROGRAPHIC_PRECIP_MIN=0.3`) — no cell driven to drought; can only equal,
   never undercut, today's lee minimum.
2. **Fetch-bounded decay in STERADIANS** (`exp(−chord/FETCH_SCALE_SR)` per hop) — the plume
   dies after a bounded **arc length**, independent of cell count → resolution-robust (this
   is the explicit fix for the trap: cell-count dependence → arc-length dependence).
3. **Single ordered pass, no CFL** — the resolution-dependent `dt` that caused the trap does
   not exist.
4. **Mean-invariant renorm** — median/mean land precip cannot collapse; the mean is pinned by
   construction, only the spread changes.

## Insertion & threading

- **Insertion:** new private helper `downwind_lee_shadow(tess, elevation, wind, coarse_precip,
  precip_initial, strength) -> Vec<f32>` in fine.rs, called inside `fine_precipitation`
  between Phase 1 and the existing renorm block. Reuses `chord_gradient`, `tess.neighbors /
  cell_center / cell_areas`, and the renorm block verbatim.
- **Combined early-return:** today's `if strength <= 0.0 { return coarse_precip }` becomes
  `if local_strength <= 0 && downwind_strength <= 0 { return coarse_precip }` so the two
  phases are independent. (Note: with local=0 the deficit seed is 0, so downwind is a no-op
  by construction — the shadow spreads the LOCAL lee deficit.)
- **Threading (mirror `orographic_precip_strength`):** new `ErosionParams.downwind_shadow_strength`
  (default from `DOWNWIND_SHADOW_STRENGTH`); read at the call site (fine.rs:820); override in
  app/world.rs; sweep name `"downwind_shadow"` in app/sweep.rs (registry + apply_knob).
- **Ordering guarantee:** precip is finalized inside `fine_precipitation` and returned to the
  `precip` loop var BEFORE the next `erode()` (fine.rs:801) consumes it as discharge.

## Knobs (all new, in constants.rs)

| Constant | Meaning | Default |
|---|---|---|
| `DOWNWIND_SHADOW_STRENGTH` | master enable; fraction of accumulated deficit applied (0 = OFF, bit-identical) | `0.0` |
| `DOWNWIND_SHADOW_FETCH_SCALE_SR` | fetch decay length in steradians per hop (bounds reach; ~few hundred km) | `0.0008` |
| `DOWNWIND_SHADOW_FLOOR` | hard floor as fraction of coarse precip (≤ 0.3) | `0.25` |
| `DOWNWIND_CONE_COS` | min cos(wind, i→receiver) for a downwind neighbor (0.5 ≈ ±60°) | `0.5` |

## Open design decisions (v1 recommendations)

1. **Sweep order axis** — GLOBAL mean wind vs per-cell. **REC: global v1** (deterministic,
   matches the "one bounded sweep" brief; receiver DIRECTION still uses per-cell `wind[i]`,
   only the ORDER is global). Revisit if maps show mis-oriented shadows under strong banding.
2. **Deficit seed** — deficit-only vs also carry windward excess. **REC: deficit-only v1**
   (budget-safe, minimal, avoids re-doing coarse advection).
3. **Apply every outer erode↔precip pass vs after-first.** **REC: every pass** (matches how
   `orographic_precip_strength` already behaves; `EROSION_PRECIP_OUTER_ITERS=2` bounds it).
4. **Floor independent vs tied to `OROGRAPHIC_PRECIP_MIN`.** **REC: independent, defaulted
   ≤0.3** (composable; shadow can only deepen toward, never raise, the local lee floor).

## Validation plan

1. **Determinism / default-unchanged (WSL2, do first):** with `DOWNWIND_SHADOW_STRENGTH=0`,
   stage-2/4 export byte-identical to a pre-change build. `cargo test`: a fine-path analogue
   of `precipitation_normalized_and_finite` (area-weighted land mean == `PRECIP_GLOBAL_SCALE`
   ± 1e-4, all finite, with strength>0). Unit-assert the DAG invariant (`s[receiver] < s[i]`).
2. **Median-collapse watch (the honest over-dry tripwire):** log land MEDIAN & p10 of
   `precip_shadow/coarse_precip` and the floor-hit fraction. Trap = median sliding below
   ~0.85 or floor-hits past ~25-30% as strength rises (the mean is pinned by renorm, so the
   MEDIAN is the honest signal).
3. **Windows diagnose sweep (compute shaders need Windows):** `--sweep downwind_shadow
   --sweep-values 0,0.3,0.6,1.0,1.5`. Pair with an OBJECTIVE probe — windward-vs-lee
   channel-density ratio (flow-accum-above-threshold per area, split by sign of `wind·∇elev`
   across major ranges); expect >1 and rising with strength (necessary-not-sufficient).
4. **Blind-rank (user judges aesthetics; I do NOT narrate fine images — confabulation risk):**
   unlabeled strength tiles → user picks best windward-dense/lee-sparse dissection and flags
   over-dried (lee bone-dead).
5. **Resolution check:** re-sweep at `--cells/--fine-scale` variants — asymmetry should
   persist (fetch is arc-length) and median land precip must not drift with N.

## Effort

~60-100 lines, additive: one helper + a ~6-line insertion + one `ErosionParams` field + 4
constants + ~3 lines each in world.rs/sweep.rs. No erosion-loop restructuring, no touching
moisture.rs/erosion.rs internals. Risk is in TUNING (fetch + floor on Windows), not code
volume. Default-off → lands dark, swept before merge.
