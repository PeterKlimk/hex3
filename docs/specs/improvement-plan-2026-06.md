# Improvement Plan — June 2026 (post-erosion audit)

Synthesis of two independent audits of the stage-3/4 (hydrosphere + erosion)
pipeline. Both converged on the same short list; this plan orders it by a
dependency story — **fix the one correctness bug first, then close the
depositional/base-level realism gap, then the frontier work** — and pins each
item to concrete integration points.

Philosophy constraint (unchanged, from `physically-inspired-roadmap.md`): add
the *mechanism* with named constants as knobs; do **not** tune constants to
chase probe numbers. The roughness / blue-noise / mass-ledger probes are
references for judgement, not optimization targets. The user judges maps by eye;
we review magnitudes, not targets. Where this plan is silent, match the module's
existing style and leave a `// SPEC:` comment rather than resolving ad hoc.

## Status of the audit

Most of `erosion-v2.md` has already landed (channel-initiation threshold,
lithologic K-field, uplift-on "Hold & carve", coupled erode↔precip loop,
lakes-as-evaporation, diffusion in the mass ledger). What remains splits into
one bug and a set of physical/aesthetic gaps:

| # | Item | Type | Phase |
|---|------|------|-------|
| 1 | Hydrology is cell-count based on the adaptive mesh | **Correctness bug** | 1 |
| 2 | Deposition is point-like (no floodplains/fans/deltas) | Missing physical half | 2 |
| 3 | Erosion base-level is only `elev < 0` (lakes ignored mid-loop) | Missing coupling | 2 |
| 4 | Uplift/K/steps mass balance unverified at production resolution | Tuning fragility | 3 |
| 5 | Erodibility is free fBm, not tied to geology | Aesthetic frontier | 3 |
| 6 | Global water budget pinned by normalization | Design ceiling | 4 |
| 7 | Tectonics is a snapshot, not a history | Long horizon | 4 |
| 8 | Dead `FineFields.uplift` transfer | Trivial cleanup | 3 |

### Implementation status (June 2026)

- **Phase 1 — DONE** (item 1). Hydrology area-weighted end to end; `flow_accumulation`
  is now a physical discharge with `mean_cell_discharge` for legacy count thresholds.
- **Phase 2 — DONE** (items 2, 3). Terminal-lake base levels + transport-aware
  (repose-slope) deposition; lost-to-ocean 43%→15% at the default, mass closes.
- **Phase 3 — DONE** (items 4, 5, 8). Balance re-verified resolution-robust at
  300k–2.5M (no retune; comment-truth pass instead). Erodibility tied to geology
  (cratons hard / arcs soft). Item 8 was a non-issue — `FineFields.uplift` IS read
  via `World::active_uplift` for the fine Climate→Uplift view + export.
- **Phase 4 — IN PROGRESS**. Item 6 (global wetness, `PRECIP_GLOBAL_SCALE`) done.
  Item 7 (geologic epochs / time evolution) is the open long-horizon project,
  deferred — it's about tectonic *history*, not per-range visual quality.

### Erosion-visual track ("make erosion look good")

A separate track from the phases above: how mountains *read*, independent of
tectonic history (the `ideas.md` lithosphere-structure + glacial cluster). The
`ideas.md` review split "mountains look generic" three ways, each a spatial fix:

- Textural sameness (one K) → geology-tied erodibility — **DONE (Phase 3 item 5)**.
- Rounded/blobby crests → **glacial erosion — v1 DONE** (`glacial_erode`: snowline
  ice over-deepening → U-troughs, tarns, sharpened peaks). **v2 DEFERRED**: explicit
  U-valley *widening* (cross-valley planation) — the genuinely hard part on an
  irregular Voronoi mesh; revisit only if v1's troughs read as too V-shaped.
- Isotropic dendritic everywhere → **structural grain** (anisotropic/striped
  erodibility along fold axes → ridge-and-valley / trellis drainage) — **next**.
  Experimental: `ideas.md` flags trellis as "an outcome to test, not a promise."
- (Range-scale blobbiness → discrete faulting / range-front facets — not started.)

---

## Phase 1 — Area-weight hydrology end-to-end (the bug)

### Why

Hydrology was written for the ~equal-area coarse mesh, where cell *count* ∝
*area*, so seeding flow with `precipitation[i]` instead of `precipitation[i] *
area[i]` only changed a global constant — harmless. It now runs on the
**adaptive fine mesh** (`fine.rs`), where land cell areas span ~50–60:1 within a
single basin (1.5 km mountains vs 12 km plains) and ~1000:1 globally. Every
hydrology quantity is therefore distorted in a way that *correlates with terrain*
(systematic artifacts at the mountain→plain transition, not noise), and it
**disagrees with erosion**, which already uses precip × area ("wet area",
`erosion.rs:734`). Confirmed count-based sites:

- `compute_flow_accumulations` (`hydrology.rs:933-934`): seeds
  `precipitation_flow = precipitation.to_vec()` and `cell_flow = 1.0` per cell.
- `compute_basin_catchments` (`hydrology.rs:697-702`): `catchment = Σ precip[c]`,
  `catchment_cells = basin.cells.len()`; the field stored as `catchment_area`
  (`:727`) is a precip-sum, not an area.
- `calculate_water_levels` (`hydrology.rs:819-847`): `target_surface` is "in
  cells", `capacity = basin.cells.len()`, and partial fill submerges
  `sorted_elevations[target_surface-1]` — floods N cells *by count*, area-blind,
  so lake shorelines ignore hypsometry.

Blast radius is bounded and the fix is tractable: erosion's inputs are clean (it
accumulates its own wet area, and the mesh-density prior uses the *coarse*
equal-area preview hydrology), so only the **displayed** rivers/lakes/basins and
the minor lake-evap feedback are wrong. The areas are already on the
tessellation — this is threading `cell_areas()`, not a redesign.

### Mechanism

Compute `let areas = tessellation.cell_areas();` once at the top of
`generate_from_continentality` (`hydrology.rs:180`) and thread it through. Make
every water quantity an **area integral**:

1. `compute_flow_accumulations(drainage_dir, precipitation, temperature, areas)`:
   seed `precipitation_flow[i] = precipitation[i] * areas[i]`,
   `temperature_flow[i] = temperature[i] * areas[i]`, and replace the count
   `cell_flow = 1.0` with `cell_flow[i] = areas[i]` (it becomes *drained area*).
   `mean_temperature` averaging then divides the area-weighted temperature sum by
   the area sum — a true area-weighted mean.
2. `compute_basin_catchments`: seed `catchment = Σ precip[c]*areas[c]`,
   `catchment_area_sum = Σ areas[c]` (rename `catchment_cells`),
   `temperature_sum = Σ temp[c]*areas[c]`. `basin.catchment_area` becomes a real
   precip×steradian discharge; mean temp = temperature_sum / catchment_area_sum.
3. `calculate_water_levels`: `target_surface` becomes a target **area**
   (steradians), `capacity = Σ areas of basin cells`, and partial fill walks the
   hypsometric curve accumulating cell **area** to the target. This needs the
   basin to carry areas aligned with `sorted_elevations` — extend
   `priority_flood_with_basins` to store a parallel `sorted_areas` (or sort
   `(elev, area)` tuples). The `floor()` "suppress tiny on/off lakes" trick
   (`:826`) moves to an area threshold (reuse `MIN_LAKE_DEPTH`-style gating).

### Coupled normalization fix (absorbs the mechanical half of audit #4)

For the discharge units to mean "water volume", precip must be normalized by
**area**, not count. Switch the fine-mesh land-mean-1.0 renormalizations from
count to area-weighted:

- `fine_precipitation` (`fine.rs:535-544`): `mean = Σ precip[i]*areas[i] /
  Σ areas[i]` over land.
- `boost_precip_near_lakes` (`fine.rs:611-618`): same.

(The coarse `moisture.rs:210` renorm can stay — equal-area mesh — but matching it
for consistency is fine; note with a comment if left as-is.) With precip at
area-mean 1.0, `catchment_area` ≈ physical catchment area where precip ≈ 1, and
`lake_area = catchment_area × climate_ratio` is dimensionally consistent.

### Downstream consumers to update

- River-width / visibility thresholds that read `flow_accumulation` (app render
  + `visualization.rs`): the field changes from count-like to precip×steradian.
  Rescale thresholds to physical drainage area; the V-key Off/Major/All tiers
  should key off a steradian/km² cutoff, not a magic count.

### Knobs

None new. Existing `climate_ratio` keeps its meaning (now over true areas).

### Risk

Low-to-moderate. The strongest regression guard: **on the coarse equal-area mesh
the change must be near-neutral** (count ∝ area there), so any large coarse-mesh
delta is a sign of a units mistake. Lake counts/extents on the *fine* mesh will
shift — expected and correct.

### Acceptance

- `cargo test` green; add a unit test that area-weighting reproduces the
  count-based result on a synthetic equal-area mesh (up to a global scale).
- New diagnostic in `diagnose`: correlation of `hydrology.flow_accumulation`
  against erosion wet-area at matching fine cells — should be ~1 where both are
  defined (the two now agree). Report before/after.
- Maps (stage 3 + 4, seed 12345): rivers track the eroded valley network; lake
  shorelines follow hypsometry, not cell-count contours.

---

## Phase 2 — Depositional & base-level realism

Items 2 and 3 are the same gap (the landscape has no working sediment sink other
than the open ocean) and should land together.

### 2a — Lakes/basins as erosion base level during the loop (audit #3)

**Why.** `Routing::build` treats only `elev < 0` as sinks (`erosion.rs:626`), so
terminal lakes and endorheic basins are priority-flooded and drained over their
spill *while terrain carves*. Internal drainage, playas, and lake outlets are
therefore weaker than the machinery already supports.

**Mechanism.** The stage-3 `pre` surface already computes hydrology basins
*before* erosion runs (`FineWorld.pre`, `fine.rs:103`). Pass its terminal /
overflowing basins into the erosion loop as **fixed local base levels**: in
`compute_eroded` (`fine.rs:124`), hand `pre.hydrology` (water levels + basin
membership) to `FineSurface::generate` → `ErosionState::new`, and in
`Routing::build` mark a cell as a sink when it sits at or below its basin's
pre-erosion water level (in addition to `elev < 0`). The sink elevation is the
lake surface, so channels grade to the lake, not through it. This is one-way
(pre-erosion lakes seed base level; post-erosion hydrology still re-derives the
final lakes), so no chicken-and-egg loop. Leave a `// SPEC:` note on whether to
refresh base levels at each re-route (`EROSION_REROUTE_INTERVAL`) or freeze them
for the run — start frozen (cheaper, and pre-erosion basins are a good prior).

**Knob.** Optionally `EROSION_LAKE_BASE_LEVEL` on/off; default on.

### 2b — Spread deposition (audit #2)

**Why.** `deposit()` (`erosion.rs:826`) routes every catchment's sediment to the
single coastal sink it drains into and caps fill there; the rest is "lost to
ocean". Correct as a mass ledger, but it cannot build floodplains, alluvial
fans, deltas, foreland basins, or sediment wedges — roughly half of real
continental surface is depositional. Codex's framing, which I endorse: a simple
"spread sediment across low downstream cells" pays off more than retuning K.

**Mechanism (design fork — pick the lighter one first).**
- *Light (recommended first):* after the incision sweep, run a downstream
  **fill pass** — carry each cell's sediment to its receiver as today, but when a
  cell's receiver is *not* downhill enough (low gradient, basin floor, lake/sea
  margin) deposit the excess locally up to a gradient/fill cap, raising the bed
  (thickness) instead of dumping it all at the terminal sink. This reuses the
  existing routing + `eroded_vol` ledger; the new state is a per-cell deposition
  cap from local slope.
- *Full:* transport-limited stream power — give each cell a transport capacity
  `Qs_cap ∝ A^m S^n`; deposit where supply > capacity. Larger change (turns the
  detachment-limited solver into a hybrid), higher payoff, more retuning. Defer
  unless the light version's fans/floodplains read as inadequate.

Couple with 2a: lakes/closed basins become deposition sites, so internal basins
aggrade instead of vanishing.

**Knobs.** Deposition cap / transport coefficient; reuse
`EROSION_DEPOSIT_FILL_FRACTION` semantics where possible.

**Risk.** Moderate. Deposition can fight incision into oscillation if the cap is
loose; keep it gradient-limited and watch the mass ledger
(`eroded ≈ deposited + lost`, with `lost` now much smaller — that's the success
signal). Re-check the uplift/erosion balance after (Phase 3 #4) since deposition
changes the denudation side.

**Acceptance.** Mass ledger shows `lost-to-ocean` dropping sharply; maps show
piedmont fans below ranges, valley-fill, and river-mouth deltas; endorheic
basins read as flats, not carved bowls.

---

## Phase 3 — Tuning re-verification & cheap wins (independent, do anytime)

### 4 — Re-verify the "Hold & carve" mass balance at production resolution

`EROSION_UPLIFT_SCALE = 0.003` / `EROSION_STEPS = 60` were calibrated at
`--fine-max 300000` (per the constant comments and the `*_fine300k.bin`
artifact), but production runs at `FINE_MAX_CELLS = 8_000_000` (design target
~2.5M) — ~8–25× finer. The knobs are *designed* resolution-robust (K on
steradian wet-area, support in km², diffusivity in steradian/step), so this is a
**verification, not a retune**: run the `--erosion-uplift-scale` / `--erosion-
steps` sweep at 2.5M and confirm `uplift-in ≈ eroded` still crosses ~1 near the
defaults. If it doesn't, the units have a hidden resolution dependence worth
finding before anyone "fixes" terrain that looks off at full res. Do this
*before* Phase 2 tuning so the baseline is honest, and again *after* (deposition
moves the balance).

### 5 — Tie erodibility to geology (cheap version of the aesthetic frontier)

`lithology_erodibility` (`fine.rs:459`) is free isotropic fBm — "rock varies
randomly", not "rock varies by geology". The fine mesh already carries
`continentality`, `arc`, `collision`, `ridge_age_distance` (transferred). Drive
the K-multiplier from those (old cratonic interiors hard, arcs/young crust soft)
*in addition to* the fBm seed, still normalized to unit land mean so it only
redistributes incision. Near-zero cost, more coherent differential relief.
Anisotropic structural grain / fold fabric (`ideas.md` A1) is the bigger, later
win and stays out of this plan.

### 8 — Drop the dead `FineFields.uplift`

`uplift` is interpolated, stored, and **serialized into the disk cache**
(`fine.rs:71,1202`) but never read on the fine mesh (erosion uplift comes from
tectonic features; the orographic feedback recomputes `wind·∇elev` directly).
Remove it from `FineFields`, `transfer_cell`, and the cache key — or wire it into
a fine climate-uplift view if that was the intent. Trivial; do it alongside any
`FineBase` cache-version bump (the key changes, so caches rebuild anyway).

---

## Phase 4 — Horizon

### 6 — Explicit global water-balance knob

Renormalizing precip to land-mean-1.0 pins the planet's total water, so you
cannot generate a globally wetter/drier world; `climate_ratio` (a lake-level
lever) is the only global knob and it isn't a precipitation lever. Once Phase 1
makes precip area-true, expose a global precip-scale (or redefine `climate_ratio`
as the explicit P/E water-balance knob) so "wet planet vs desert planet" is
generable, not just spatially redistributed. Small, but only meaningful after
Phase 1.

### 7 — A few geologic epochs instead of a snapshot

Euler poles are random (`dynamics.rs:53`) and elevation/erosion infer "history"
from current boundaries only. A handful of coarse time-steps (crust ages, ridges
migrate, collisions narrow) would do more for believable mountains, drainage
capture, antecedent rivers, foreland basins, and terrane structure than perfect
single-step physics — and it's the umbrella that makes Phase 2's deposition and
base-level work produce *history* rather than static landforms. Large, open-
ended; the substrate (steppable `ErosionState`, `FineBase`/`FineSurface` split)
already exists. Out of scope here beyond naming the first step: run erosion
*during* a 3–5 epoch uplift sequence on the reference world and look.

---

## Validation (all phases)

```bash
cargo fmt && cargo test
cargo run --release --bin hex3 -- --headless --seed 12345 --export /tmp/w.json.gz
cargo run --release --bin diagnose -- --seed 12345            # add --fine-max 2500000 for prod-res checks
```

- All existing tests pass, including field smoothness (Moran's I) and the
  erosion steady-state unit tests — no phase may introduce cell-scale speckle.
- New unit tests where checkable in isolation (Phase 1 area-weighting ≡ count on
  equal-area mesh; Phase 2a base-level grading; Phase 2b mass-ledger closure).
- Report magnitudes on seed 12345 in the commit message (land %, lake %/extent,
  denudation %, `eroded/deposited/lost` ledger, uplift-in vs eroded) — changes
  are expected; we review magnitudes, not targets.

## Non-goals

- No retuning of existing constants except where a phase explicitly re-verifies
  a balance (Phase 3 #4) or a mechanism demands it (Phase 2b deposition cap).
- No MFD routing, glacial/aeolian/coastal erosion, stratigraphy, or submarine
  fluvial erosion (sea level stays a fixed datum).
- No wind/pressure/circulation changes; Phase 1 touches only precip
  *normalization*, not the moisture solve.
- No new dependencies; no changes under `scripts/`.

## Recommended sequence

**1 → 2a+2b → (3 re-verify) → 4/horizon.** Phase 1 is a bug and unblocks honest
judgement of everything downstream (rivers/lakes are currently wrong on the
adaptive mesh). Phase 2 is the biggest *physical* payoff and the two halves
interlock. Phase 3 items are independent and cheap — slot #8 and #5 in whenever;
run #4 as the standing balance check before and after Phase 2. Phase 4 is the
long game.
