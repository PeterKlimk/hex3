> **Archived:** Historical design or experiment record. It does not define current architecture, defaults, or priorities. Start at [`docs/README.md`](../../README.md).

# Improvement Plan — 2026-06-16 (post-`f15eecb` audit + Codex review)

Synthesis of a fresh full-pipeline audit on `main @ f15eecb` (six parallel
subsystem deep-reads) plus an independent `codex exec` review of the findings.
This **succeeds** [`improvement-plan-2026-06.md`](improvement-plan-2026-06.md):
that plan's erosion phases 1–3 have landed, and the older
[`algorithm-audit-2026-06-15.md`](../../audits/algorithm-audit-2026-06-15.md) was on the
pre-merge branch (`58d8ce1`) — its headline **C1** (erosion uplift applied to
ocean/sink cells) is **fixed** in `88c242f`. This pass is what remains on the
current tip.

**Philosophy constraint (unchanged).** Add the *mechanism* with named constants
as knobs; do **not** tune constants to chase probe numbers. Probes are
references for judgement, not optimisation targets. The user judges maps by eye;
we review magnitudes. Where this plan is silent, match the module's style and
leave a `// SPEC:` comment rather than resolving ad hoc.

**Confidence tags.** ✓ = re-verified at the cited lines by both the sweep and
Codex. ⚠ = real but the *intent*/magnitude is a judgement call — a resolution
path (physical argument or test) is given. Each item says how sure we are and,
where not certain, exactly what to test.

---

## Headline

No critical or crash-level bug on the current tip. The core engines are sound
(stream-power incision unit-tested, priority-flood basins, area-weighted
discharge, three-cell circulation, fixed-seed determinism). The real findings
cluster in one theme — **the climate/sediment layers don't fully track the
*actual* fine relief and resolution** — plus a few cleanups. Codex confirmed the
four substantive items, trimmed three over-confident interpretations, and added
one worthwhile miss (stale fine temperature).

| # | Item | Type | Confidence | Priority |
|---|------|------|-----------|----------|
| 1 | Moisture reactions not `dt`-scaled (advection is) | Correctness / resolution | ✓ | **P1** |
| 2 | En-route deposition ~inert (repose slope ≈ 0.1% grade) | Tuning / missing realism | ✓ (mag reasoned) | **P1** |
| 8 | Fine temperature stale vs eroded/glaciated relief | Missing coupling | ✓ | **P2** |
| 3 | Diffusion clamps to sea level, ignores lake base-level | Correctness (small) | ✓ | **P2** |
| 4 | Merged-cycle lakes forced endorheic (budget leak) | Correctness (rare) | ✓ | **P2** |
| 6 | `.unwrap()` on `partial_cmp` can panic on NaN | Hardening | ✓ | **P3** |
| 7 | Craton seed spacing uses flat-area formula | Cosmetic | ⚠ | **P3** |
| 5 | `trench_support_dist` folds in slab-age stiffening | Clarify intent | ⚠ | **P3** |
| 9 | Lake-evap halo isotropic, not wind-aware | Fidelity frontier | ✓ | **P4** |

Two **physical judgement calls** (not bugs, by the "physically-inspired"
philosophy) are recorded at the end: ocean-ocean subduction polarity by approach
speed, and no oceanic ITCZ rain.

---

## Implementation status (2026-06-16, same day)

Acted on the plan immediately. All changes verified with `cargo test --lib` (45
pass) and an end-to-end stage-4 `diagnose` run (seed 12345); no panic/NaN.

**Landed (kept):**
- **#6 NaN-safe sort** — `hydrology.rs:728` now `unwrap_or(Equal)`. Pure hardening.
- **#3 diffusion clamp to base level** — `erosion.rs:1002` clamps to
  `routing.base_floor[i]` (sea level *or* lake surface) instead of `0.0`.
- **#8 fine temperature lapse correction** — `fine.rs` `from_eroded` recomputes
  temperature against the eroded relief before hydrology; verified active
  (`basin evaporation: mean factor 0.97`). Display path (`active_temperature`)
  still shows the coarse-interp field — acceptable; the evaporation consumer is
  what mattered.
- **#2 deposition repose slope 1.0 → 6.0** — `constants.rs`, with the grade
  derivation (`grade ≈ SLOPE × 0.1%`) documented; unit test rebuilt to a
  production-scale `dist` (was masking with `dist = 1.0`). **NEEDS VISUAL
  REVIEW on Windows** — it changes erosion character (more valley-floor/fan
  aggradation). Shifts fine zonal precip slightly via orographic modulation on
  the new relief (e.g. +15..+30 band 3.51 → 3.68 at seed 12345/40k); continents
  unchanged. Calibrate with `diagnose --erosion-deposition-slope` + maps.

**#1 moisture `dt` — landed as the PROPER (reference-normalized) fix.**
The first attempt (naive `reactions × dt`) was reverted: it discards the
eye-calibration (the rainout constants were tuned as per-iteration fractions, an
implicit `dt=1`), so balanced advection dumped all rain at the equator and killed
subtropical+midlatitude rain (seed 12345/40k: equatorial band 2.53 → 12.94;
+15..+30 3.51 → 0.23) at ~20× cost. The correct fix scales reactions
(evaporation, rainout, eddy diffusion) by `react_dt = dt / MOISTURE_DT_REF`,
where `MOISTURE_DT_REF = 0.01` is the design-resolution advective `dt`. This pins
the reaction:advection ratio to `1/DT_REF` independent of mesh resolution and
wind speed, **while preserving the calibrated climate at the design point** (at
100k cells `dt ≈ DT_REF` so `react_dt ≈ 1`). The loop now iterates to a tolerance
(reactions slow with resolution) with a warm start; at 100k it converges in ~tens
of iters and breaks. Verified: new vs baseline @ 100k — precip Moran's I 0.968 vs
0.970, aridity 72% vs 74%, zonal pattern matched (big-land bands within ~2%, no
equator dump); stage-2 time 2.3s (no regression). `simulate_moisture` currently
runs only at the fixed 100k coarse mesh, so this is a no-op for *today's* world —
but it removes the latent resolution-dependence (and the spurious peak-wind
dependence), so running moisture on a finer/adaptive mesh later won't over-dry.
New constants: `MOISTURE_DT_REF`, `MOISTURE_MAX_ITERATIONS`, `MOISTURE_CONV_TOL`.

**#7 craton spacing — fixed (angular formula), signed off.**
`crust.rs` now uses the spherical-cap `acos(1 - 2/N)` (matching `plates.rs`)
instead of the flat-area `sqrt(4π/N)`. This *does* change the accepted craton
seed set → different continent layout for every seed (it was initially deferred
to preserve the seed-12345 reference world, then explicitly signed off as a
worthwhile correctness fix). Verified the world still generates sanely at seed
12345: 3 continents (79M/53M/38M km², close to the prior 81M/53M/39M), land
95.4%, aridity 68%; crust coverage/survival tests pass.

**Deferred per plan:** #4 (merged-cycle saddle — until a ring basin misbehaves),
#5 (trench `OLD_MULT` — needs a Windows render A/B), #9 (wind-aware lake halo —
frontier), and the two judgement calls.

---

## Group A — Climate/terrain coupling (the real theme)

### 1. Moisture reactions are not `dt`-scaled while advection is — **P1** ✓

**Problem.** The moisture relaxation is operator-split advection–reaction, but
only advection carries the timestep:
- `dt = MOISTURE_CFL / max_outflow` (`moisture.rs:121-126`)
- evaporation: `m += EVAPORATION_RATE * (cap − m)` — no `dt` (`moisture.rs:142`)
- rainout: `rain = m * (rain_rate + convective) + …` — no `dt` (`moisture.rs:158`)
- transport: `transported = dt * amount` (`moisture.rs:182`)

Reactions run at an implicit `dt = 1`; advection at `dt = CFL/max_outflow < 1`.
So moisture rains/evaporates over a **shorter advected distance than intended**,
and because `dt` is set by `max_outflow` (the windiest/smallest cell) it shifts
with **wind speed and mesh resolution**.

**Physical reading.** The inland moisture-penetration length is
`L ≈ u · τ_rain`. With reactions at `dt=1` but advection throttled by `dt`, the
effective `L` is compressed by `dt`, and `dt` is resolution-dependent. Precip is
renormalised to a land mean, so the *absolute* level is pinned but the
**coastal→interior drying gradient is too steep and gets steeper at higher
resolution.** This is the most likely root cause of the "full moisture transport
over-dries the adaptive mesh" symptom that pushed us to local-only orographic
modulation on the fine mesh ([[hex3-climate-feedback]]).

**Fix.** Multiply the reaction terms by the same `dt` (evaporation, baseline +
orographic + convective rainout, over-capacity rainout). At steady state `dt`
then cancels from the balance, making the pattern dt/resolution-independent. The
absolute rainout-per-iteration drops, so confirm 80 iterations still reach
steady state (it may need more, or a convergence check).

**Confidence / how to resolve.** Mechanism ✓ certain. Impact = **test before
fixing constants:** (a) instrument the loop to log max per-cell Δmoisture per
iteration — confirm whether 80 iters is converged or transient; (b) run a
coarse-mesh resolution sweep (e.g. 30k/100k/300k coarse cells) logging the
land-mean coast→interior precip falloff. If the falloff moves with resolution,
the bug bites. Apply the `dt`-consistent fix and re-check that the falloff
becomes resolution-stable. Do **not** re-tune `RAINOUT_*` to a number — judge
the resulting precip map by eye once the gradient is resolution-stable.

### 8. Fine-mesh temperature is stale vs the eroded/glaciated relief — **P2** ✓ (Codex)

**Problem.** Fine temperature is interpolated **once** from the coarse
atmosphere field (`fine.rs:1406`, used at `:503`) and never recomputed.
But the coarse temperature baked in lapse against the **coarse** elevation
(`atmosphere.rs:201-205`, `LAPSE_RATE=1.5`), while the fine surface has since
been carved (valleys down), held/sharpened (peaks ~+15%), and glacially
over-deepened (`fine.rs:445-478`). So the temperature the fine basins evaporate
with ignores the very relief they drain.

**Physical reading.** The error is the lapse of the *erosional relief change*:
a held peak at +0.1 elevation unit is ~0.15 temp-units too warm; a carved valley
too cold. Basin evaporation (`hydrology.rs:1276` `exp((mean−global)·1.0)`) sees
catchment-mean temperature, so the effect on lake level is ~`exp(±0.1)` ≈ ±10%
per affected catchment — modest but real, and it's free to fix. (Glacial snowline
uses its own latitude+elevation rule, `constants.rs:909-919`, so glaciers are
**not** affected — only the evaporation/display path.)

**Fix.** After the eroded surface exists, correct fine temperature by the
fine-vs-coarse relief lapse delta:
`T_fine[i] = T_coarse_interp[i] − LAPSE_RATE · (elev_fine[i] − elev_coarse_interp[i])`
(only the positive-elevation part, matching `atmosphere.rs:203`). Cheap, local,
no re-advection. Integration: recompute in the eroded-surface assembly where
`fine.rs` builds `FineSurface`, before hydrology reads temperature.

**Confidence / how to resolve.** Mechanism ✓. Magnitude is modest — log lake-area
/ total-evaporation change on seed 12345 before/after to confirm it's worth the
code. If negligible everywhere, downgrade to a `// SPEC:` note and skip.

### 9. Lake-evaporation halo is isotropic, not wind-aware — **P4** ✓ (Codex)

`boost_precip_near_lakes` spreads lake humidity by symmetric neighbour averaging
(`fine.rs:792-831`). Real lake-effect precip is **downwind**. Fine wind is
available. Low-priority fidelity: bias the halo diffusion along the wind
(upwind-weighted steps) instead of isotropic. Frontier, judged on maps.

---

## Group B — Sediment & water budget on the fine mesh

### 2. En-route deposition is ~inert at production resolution — **P1** ✓ (magnitude reasoned)

**Problem.** `target = elev[r] + EROSION_DEPOSITION_SLOPE · routing.dist[c]`
(`erosion.rs:1054`), default `SLOPE = 1.0` (`constants.rs:781`). Deposition only
fires where `elev[c] < target`, i.e. where the local channel grade is gentler
than `SLOPE`. The unit test uses `dist = 1.0` (a **full radian** ≈ 6371 km reach),
which hides the scale (`erosion.rs:1259-1279`).

**Physical reading (this resolves Codex's "PARTIAL").** `routing.dist` is chord
distance (`erosion.rs:1112-1121`); chord ≈ arc to 1e-8 at km scale, so that
nuance doesn't change the result. Convert the constant to a physical grade,
which is cell-size-independent:
`grade ≈ SLOPE · M / R`, with `M ≈ 6300 m/elev-unit` (`CONTINENTAL_BASE 0.08 ≈
500 m`) and `R = 6.371e6 m` → **`grade ≈ SLOPE × 0.1%`.** So `SLOPE = 1.0` is a
**0.1% repose grade** — only deltas/lake floors are that flat. Alluvial fans,
floodplains, and foreland fills sit at ~0.5–2% → **`SLOPE ≈ 5–20`.** The default
is ~10× too low to build the landforms its docstring describes; deposition is
effectively confined to sink-fill. (My earlier "~1–6 m" was an unjustified
absolute; the grade framing above is the right invariant — Codex was correct to
flag the meters claim.)

**Fix.** Raise `EROSION_DEPOSITION_SLOPE` toward the 5–20 band (a ~0.5–2% repose
grade) and judge fans/floodplains on maps; the constant stays the knob. Fix the
unit test to use a realistic `dist` (~1e-3 rad) so the masking goes away.

**Confidence / how to resolve.** ✓ inert at 1.0; the 5–20 target is a physical
estimate, **calibrate by eye** with `diagnose --erosion-deposition-slope` and the
mass ledger (lost-to-ocean should drop as `SLOPE` rises). Not a number to hit.

### 3. Diffusion clamps to sea level, ignoring terminal-lake base levels — **P2** ✓

Incision grades cells to `routing.base_floor` (sea level **or** a terminal-lake
surface `L>0`, propagated `erosion.rs:899-910`), but the hillslope-diffusion
Jacobi sweep clamps only `.max(0.0)` (`erosion.rs:1002`). A cell graded to a lake
surface `L>0` can then be diffused **below** `L`, undoing the lake-grade
invariant incision just enforced. Magnitude small (`D=2e-8`) but unguarded.

**Fix.** Clamp diffusion to the local datum: `.max(base_floor[i])` instead of
`.max(0.0)`. Thread `base_floor` into the diffusion sweep (or apply a post-sweep
per-cell `max(base_floor)`). Same `.max(0.0)` is also a tiny coastal mass source
— acknowledged, leave unless it shows up in the ledger.

**Confidence.** ✓ certain; low magnitude. Clean correctness fix, do it with #4.

### 4. Merged-cycle lakes are forced endorheic — downstream budget leak — **P2** ✓ (author-acknowledged)

When overflow targets form a cycle (A→B→C→A), the members are merged and capped
at the **highest member rim** with `overflow_group = None` even if the union has
a real external saddle; the overflow volume is then dropped, not cascaded
(`hydrology.rs:1023-1038`, the comment admits it). Mass not conserved for
downstream recipients. Rare.

**Fix (correct algorithm, when we get to it).** On merging a cycle into one
water body, recompute the **union's minimum external saddle** — the lowest rim
cell whose lowest neighbour lies outside the union — and route the merged
overflow there (re-entering the topological cascade). Defer until a map shows a
mis-behaving ring basin; document the algorithm now so it isn't re-derived.

---

## Group C — Cleanups & intent clarifications

### 6. `.unwrap()` on `partial_cmp` can panic on a NaN elevation — **P3** ✓

`hydrology.rs:728` `sort_by(|a,b| a.0.partial_cmp(&b.0).unwrap())` hard-crashes
worldgen if any cell elevation is NaN; the merged-pair sort at `:1062` already
uses `.unwrap_or(Equal)`. No current NaN source found, but make them consistent —
one-line hardening.

### 7. Craton seed spacing uses a flat-area formula on the sphere — **P3** ⚠

`crust.rs:124` `ideal_spacing = sqrt(4π / num_cratons)` is flat-tiling linear
spacing, then compared against angular distances (`:143-147`); `plates.rs:83`
correctly uses the spherical-cap `acos(1 − 2/N)`. **Cosmetic** — only
`NUM_CRATONS = 5` and a relaxation loop compensates. Fix for consistency by
switching to the angular formula, or leave with a `// SPEC:` note. Codex's
caveat: the "plates does it right" comparison is heuristic, not proof of a
visible defect — so this is a tidy-up, not a bug.

### 5. `trench_support_dist` folds in the slab-age stiffening constant — **P3** ⚠ (clarify intent)

`features.rs:530`: `(PI+1.0)·TRENCH_FLEX_ALPHA·TRENCH_FLEX_ALPHA_OLD_MULT`. The
sibling support distances (`:531-535`) use only peak/width geometry with no age
multiplier; `OLD_MULT` (1.4) is the per-cell *age* stiffening applied separately
at `:636-650`. The 40% widening of the trench-forcing smoothing footprint is
real; whether it's a leftover or an intentional "use the widest old-lithosphere
flexure as the support radius" is **unproven** (Codex's correction — I called it
a leftover too confidently).

**How to resolve (test, low priority).** Render trenches/forearcs on seed 12345
with the line as-is vs with `OLD_MULT→1.0` in that expression only. If
indistinguishable → drop the factor for consistency. If the 1.4× visibly
de-blocks the trench forcing → keep it and add a one-line comment documenting it
as a deliberate widest-flexure support radius. Either way the ambiguity goes away.

---

## Physical judgement calls (record, not fix — by the "physically-inspired" philosophy)

- **Ocean-ocean subduction polarity by approach speed, not age/density**
  (`boundary.rs:225-247`). There is no crustal-age field, so the faster-approaching
  plate is elected to subduct — the opposite of the physical driver (older/colder/
  denser sinks). Defensible as a proxy; the *real* fix is a slab-age field, which
  is the long-horizon "tectonics as history" item from the prior plan. Leave.
- **No oceanic ITCZ rain** (`moisture.rs:151-156`, `:205-222`). Convective rain is
  land-only and precip is normalised to a land mean, so the wettest zone on a real
  planet (the tropical ocean band) is suppressed by construction. Baseline rainout
  still wets ocean, so the precise statement is "no oceanic *convective* rain."
  Intentional (maritime rain recycles to ocean anyway); revisit only if ocean
  precip is ever consumed downstream.

---

## Verified correct (clean bill — for confidence)

Re-checked by both the sweep and Codex and found sound, so we don't re-litigate:
Voronoi duality + circumcenter hemisphere disambiguation; spherical-triangle area
(Van Oosterom–Strackee in f64); adjacency determinism (sort+dedup); uniform
sphere sampling; Coriolis hemisphere flip (`atmosphere.rs:391-399`); lapse-rate
sign; zonal trade/westerly directions; signed orographic windward/lee
(`atmosphere.rs:911-920`); priority-flood basins, area-weighted discharge,
topological overflow cascade; chord-vs-acos discipline in the erosion/fine path;
fixed-seed reproducibility (Lloyd/adjacency nondeterminism fixes present).

The earlier ruled-out "evaporation `exp()` overflow" is **not** a hazard, but the
reasoning is corrected: temperature is **not** strictly [0,1] — it can go negative
at altitude via lapse (`atmosphere.rs:31-33`); it stays bounded because
catchment-mean deviations from the global mean are small, not because the field
is clamped.

---

## Suggested order

1. **P1 — #1 (moisture `dt`)** and **#2 (deposition slope)**: the two highest
   realism wins. #1 needs the convergence/resolution test first; #2 is a constant
   raise + test fix, calibrated by eye.
2. **P2 — #8, #3, #4**: the fine-relief coupling and water-budget correctness
   cluster. #8 and #3 are cheap; #4 only when a ring basin misbehaves.
3. **P3 — #6, #7, #5**: one-line hardening (#6), a formula tidy (#7), and a
   one-render intent test (#5).
4. **P4 — #9** and the two judgement calls: frontier / record-only.

Open the work with the **#1 moisture convergence+resolution probe** — it both
confirms the highest-value bug and tells us whether the iteration count needs to
move when reactions become `dt`-scaled.
