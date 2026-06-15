# World-Generation Algorithm Audit — 2026-06-15

A code-level audit of the world-generation pipeline on branch `fault-facets` (tip across all
branches at the time: `58d8ce1`). Unlike [algorithm-review-2026-06.md](algorithm-review-2026-06.md)
(which critiqued algorithm *choice* and missing couplings), this audit reads the **actual
implementation** for correctness bugs, numerical issues, determinism hazards, and places where
comments/docs over-claim the physics. Docs and comments were treated as untrusted and verified
against code.

**Method.** Six parallel deep-reads of the subsystems (geometry/tessellation; plates/crust/
dynamics/boundary; features/elevation; atmosphere/circulation/moisture; hydrology; fine-mesh/
erosion/cache), plus an independent `codex exec` source audit. Findings below merge both. Each
item notes whether it was directly re-verified at the cited lines (✓), and its source
(**sweep** = the subsystem reads, **codex** = the Codex pass, **both** = independently found by
each).

**Headline pattern:** comments and module docs systematically over-claim physics that the code
does not implement. This recurs in every subsystem and is called out per-finding below. Treat
docstrings as aspirational, not descriptive.

**Overall:** the core engines are sound (stream-power erosion is unit-tested; priority-flood basin
extraction, area-weighted discharge, and the three-cell circulation are well-built; determinism
for a fixed `--seed` is intact — the previously-documented Lloyd/adjacency nondeterminism fixes
are present and verified). The real bugs cluster in (a) the **newest code** — erosion's add-on
passes and the fine-mesh cache — and (b) the **mesh-plumbing / water-budget** layer.

---

## Critical

### C1 — Erosion uplift is applied to every cell, including ocean and sink cells ✓ (sweep)
`erosion.rs:550-552`
```rust
for i in 0..n { self.thick[i] += self.u_thick[i] * self.params.dt; }
```
No `base[i] >= 0` / `!is_sink` guard. Incision and diffusion skip sink cells, so uplift on
submerged margins (where arc/collision/rift forcing is nonzero) is a one-way thickness→elevation
injection with nothing removing it. Over the step count this **lifts trench-margin ocean cells
across sea level and manufactures new land**, breaking the fixed sea-level datum and undercutting
the "resolution-robust height distribution" calibration. The mass ledger comment even admits
uplift is "NOT in that balance."
**Fix:** gate uplift to land / non-sink cells and re-clamp to `base_floor`.

### C2 — Tectonic forcing scales ~√N with resolution ✓ (sweep)
`features.rs:112,210-213`
```rust
let area_scale = |cell_idx: usize| -> f32 { mean_area / cell_areas[cell_idx].max(1e-10) };
// ...
arc_seed_strength_cont[b.cell_b] += uplift_force_b * area_scale(b.cell_b); // force already × edge_length
```
Boundary forcing is multiplied by **both** `edge_length` and `area_scale = mean_area/cell_area`.
Since `edge_length ~ √area` and `area ~ 1/N`, per-cell seed strength scales as `1/√area ~ √N`, so
feature amplitude after `sqrt_response` drifts as ~N¼. The `max` caps mask it on coarse worlds.
This contaminates the **sea-level solve and every field transferred to the fine mesh**, since both
are computed on the coarse mesh. The comment at `features.rs:107-109` claims the opposite
("total integrated forcing is constant regardless of resolution") — false.
**Check:** compare the debug sums at `features.rs:640-680` across two `NUM_CELLS` values.
**Fix:** drop one of the two factors (keep `edge_length`, drop `area_scale`, or vice-versa) so the
per-cell integrated forcing is resolution-invariant; re-tune sensitivities.

### C3 — Fine-mesh cache key omits a transferred input (`atmosphere.wind`) ✓ (both)
`fine_cache.rs:49,86` / `fine.rs:1367,420`
```rust
mix_f32s(&mut h, &atmosphere.temperature);
mix_f32s(&mut h, &atmosphere.precipitation);
mix_f32s(&mut h, &atmosphere.uplift);     // wind is NOT mixed in
// fine.rs:1367 — wind IS stored and used:
wind: support.interpolate_vec3(&atmosphere.wind),
```
`atmosphere.wind` is transferred into the fine base and used for orographic precipitation / rain
shadows, but is not part of the cache key. A change that alters wind without moving temp/precip/
uplift yields a **false cache hit with stale wind**. Independently flagged by both passes.
Separately, the sampling/transfer *logic* is guarded only by a hand-bumped
`FINE_BASE_CACHE_VERSION` (still `2`) — editing `sample_fine_points`/`transfer_cell` without
bumping it silently serves stale geometry.
**Fix:** add `atmosphere.wind` to `fine_base_key`; bump `FINE_BASE_CACHE_VERSION`.

### C4 — Circumcenter hemisphere-sign test is wrong for obtuse facets ✓ (sweep)
`voronoi.rs:299` (default convex-hull backend)
```rust
let centroid = (a + b + c).normalize();
if center.dot(centroid) < 0.0 { -center } else { center }
```
The robust test is `center.dot(a) < 0.0`. For an obtuse spherical Delaunay facet the circumcenter
lies outside the triangle and can have `center·centroid < 0` while `center·a > 0`, flipping an
already-correct Voronoi vertex into the antipodal hemisphere. Latent root cause of the stray
vertices / topology defects `validation.rs` exists to detect.
**Fix:** two-line change to test against a vertex.

---

## High

### H1 — Orphan (no-neighbor) fine cells are accepted with only a warning ✓ (codex)
`tessellation.rs:227-235,267-271`
```rust
.filter(|&&n| n != s2_voronoi::adjacency::NO_NEIGHBOR)
// ...
let orphan_count = adjacency.iter().filter(|a| a.is_empty()).count();
if orphan_count > 0 { log::warn!("s2-voronoi: {} cells have no neighbors (orphans)", orphan_count); }
```
`NO_NEIGHBOR` entries are filtered out; orphans are only counted and warned, never repaired or
errored. Downstream hydrology (`priority_flood_with_basins` seeds ocean cells, never checks all
cells were reached) and erosion routing assume a connected graph, so orphan fine cells silently
become no-drainage islands / lose flow / are skipped by erosion.
**Fix:** make orphans a hard error or repair adjacency before worldgen continues.

### H2 — WITHDRAWN on verification: flat-fill does NOT create drainage cycles
`hydrology.rs:690,1010`
The sweep originally claimed the ε=0 flat fill lets `flood_parent` route uphill and form drainage
cycles that silently drop discharge. **Closer analysis disproves this.** `flood_parent` is the BFS
tree rooted at the ocean seeds, and every place it is assigned (normal `>= current_elev`
processing, far-shore cells, and basin-interior fill) sets the parent to an **equal-or-lower**
filled elevation. Steepest descent only ever targets a **strictly-lower** filled cell. So along any
drainage path filled elevation is monotonically non-increasing (strictly decreasing on
steepest-descent edges); a cycle would require an all-flat loop of `flood_parent` edges, which is
impossible because `flood_parent` is a tree. Therefore `drainage_dir` is acyclic, the Kahn
accumulation completes for every cell, and no discharge is dropped. The `lake_outflow_paths`
10000-step guard protects the *basin overflow* graph (basins can mutually overflow), not
`drainage_dir`. **No Barnes ε pass needed.** The real residual issues here are H5 (drainage ignored
edge length) and H4/M8 (overflow-cascade accounting).

### H3 — Unbounded loop in overflow routing = latent hang (sweep)
`hydrology.rs:842`
`compute_overflow_targets` follows `drainage_dir` with no iteration cap — unlike
`lake_outflow_paths`, which caps at 10000 *because* these cycles are known to exist. A cycle that
touches neither ocean nor a basin tag hangs generation.
**Fix:** add an iteration cap / cycle guard.

### H4 — Basin water budget: cross-basin inflow + overflow cascade can double-count ✓ (codex; partial sweep)
`hydrology.rs:800-814` (catchment from `flow_accumulation` on the *filled* graph) and `:930`
(overflow cascade). `flow_accumulation` routes water through basin interiors as if they already
spill — *before* lake levels are solved — so a downstream basin's catchment includes upstream-basin
water that should be captured by the upstream lake; then `calculate_water_levels` adds overflow
*again* for basins that do spill. (Note: the catchment *scan itself* correctly adds only
boundary-crossing edges once — the double-count is between the two accounting stages, not within
the scan.) Same root issue as the overflow-cascade ordering bug (M-series below). Skews lake/river
sizing; does not break generation.
**Fix:** solve lake levels and route inflow/overflow in topological order of the overflow graph.

### H5 — Steepest descent ignores edge length ✓ (sweep)
`hydrology.rs:1000`
Drainage picks the lowest-*elevation* neighbor, not lowest elev/distance, on a mesh with ~50:1
spacing variation — biasing channels toward distant large cells. Docstring claims it "follows the
gradient"; it does not.
**Fix:** normalize the drop by inter-cell chord distance.

### H6 — Glacial pass ignores lake datums and is non-conservative (sweep)
`erosion.rs:218,248`
Routes with `no_lakes = NEG_INFINITY`, abrasion floored only to sea level, eroded rock vanishes
(logged, not deposited). Can gouge below carved terminal-lake floors the fluvial stage respected.

### H7 — Fault-scarp datum mismatch + coastal submergence (sweep)
`fine.rs:390-416,596-606`
`faulted_base` becomes the erosion base, but hydrology's lake set was derived from the *un-faulted*
elevation. The basin-drop only skips cells already `< 0`, so a `+0.001` coastal cell can be driven
to `−0.02` — **silently submerging land before erosion starts**.

### H8 — Ocean-ocean subduction polarity defaults to a motion-independent constant (sweep)
`boundary.rs:282-285`
```rust
let base_polarity = ocean_ocean_polarities.get(&key).copied()
    .unwrap_or(SubductionPolarity::ASubducts);
```
Fires on near-threshold pairs (the per-edge vote threshold and the aggregate-mean kind classifier
disagree), giving a fixed bias unrelated to motion.
**Fix:** fall back to the aggregate mean-normal direction; reconcile the two thresholds.

### H9 — Climate uplift cannot suppress rain (sweep)
`atmosphere.rs:893,911`
Convergence and orographic uplift are clamped `≥0`; only the lone signed circulation term
(weight 0.7) opposes rain. Hadley-descent subtropical deserts are under-modeled, and the `uplift`
value fed to `static_rain_rate` is a renormalized dimensionless blend, not a rate. Comment claims
subsidence "can suppress" rain (`moisture.rs:62`) — effectively unreachable.

---

## Medium

### M1 — Dead "asymmetric continental response" model + dead constants (sweep)
`elevation.rs` docs + `constants.rs:147-167`. The documented compression→mountains / tension→rifts
asymmetric model and its 7 tuning constants (`CONT_COMPRESSION_SENS`, `CONT_MAX_MOUNTAIN`,
`OCEAN_SENSITIVITY`, …) are **dead** — the live path is a pure isostatic-thickness model.
`DIV_OCEAN_CONT` (passive-margin uplift) is also never read. **Fix:** delete or wire the constants.

### M2 — Unnormalized "gradients" (valence/resolution biased) (sweep)
`atmosphere.rs:339` (`compute_pressure_gradient`), `elevation.rs:212` (`gradient`). Both sum
unit-direction × slope over neighbors with no division by count/area → systematically larger for
high-valence cells, non-convergent under refinement. Not true ∇.

### M3 — Unit confusion: `mean_cell_area().sqrt()` used as a length (sweep)
`atmosphere.rs:886`, `moisture.rs:53`. It is √steradian on the unit sphere; works only because
everything is renormalized, but `MOISTURE_DIFFUSIVITY` is labeled "rad²" while divided by a
steradian area. Latent km/radian trap if physical constants are ever wired in.

### M4 — Wind-projection edge fluxes discarded; moisture re-derives flux from reconstructed cell winds (codex)
`atmosphere.rs:746` / `moisture.rs:105`. The projection's corrected edge-normal fluxes are thrown
away; moisture recomputes flux from least-squares cell-centered winds, which cannot preserve all
corrected edge normals on valence-5/6 cells — so the "post-projection divergence-free" claim does
not actually hold for the transported field.

### M5 — Moisture diffusion not conservative on unequal-area meshes (both)
`moisture.rs:192`. Unweighted neighbor average while advection treats moisture as area-normalized
concentration → changes total integrated moisture, biases small/large cells.

### M6 — `is_continental` re-thresholded on the fine mesh (sweep)
`fine.rs:1300` (`continentality >= 0.5`) instead of carrying the coarse boolean, so cells flip type
at coastlines while still carrying coarse-crust feature magnitudes.

### M7 — `macro_dt` excluded from transferred thickness → fine base ≠ displayed coarse terrain (sweep)
`elevation.rs:324` stubs macro noise to 0 in `coarse_elevation_fields` but adds it in the rendered
coarse elevation, so the fine erosion base diverges from the coarse map.

### M8 — Overflow cascade ordered by spill elevation, not overflow topology (sweep)
`hydrology.rs:888-894,931`. Correct only if every basin's overflow target has a lower spill
elevation; when it doesn't, the downstream increment is silently lost (undersized lake). Same
family as H4.

### M9 — Deposition reads stale `elev`, writes `thick` (sweep)
`erosion.rs:1033`. Chained low-gradient reaches under-fill; mass-conserved but order-dependent, not
the repose surface claimed.

### M10 — `base_floor` clamp uses stale routing-time datum across evolving steps (sweep)
`erosion.rs:514,881`. Reroute interval is 6 steps; a cell whose terminal sink changed can be
clamped to a stale lake surface. Bounded by the slope guard, but not re-derived.

### M11 — Craton seeding can return < `NUM_CRATONS` (sweep)
`crust.rs:116`. No final "accept-anything" fallback (plate seeding has one at `plates.rs:150`).

---

## Low

### L1 — `_lloyd_iterations` parameter ignored ✓ (codex)
`tessellation.rs:144,157`. The public arg is discarded; code always uses internal
`LLOYD_ITERATIONS`. App's `LLOYD_ITERATIONS = 5` knob is dead (docstring is honest about it).

### L2 — `partial_cmp().unwrap()` panics on NaN (sweep)
`hydrology.rs:731,889`.

### L3 — `acos(dot)` in fine interpolation weight (both)
`fine.rs:1446`. Safe today (coarse separations don't collapse) but the lone non-chord distance in
the fine path; use chord for consistency / future-proofing.

### L4 — Hydrology `drainage_dir == None` docstring is materially false (codex)
`hydrology.rs:118,988`. Comment says `None` means "ocean/lake or no outlet," but only ocean cells
are treated as sinks at drainage time (lakes aren't known yet), so lake cells can still carry a
downstream direction.

### L5 — Determinism fragilities (latent, not live) (sweep)
Default `HashMap` in `boundary.rs` (happens to feed only order-independent reductions);
ocean-ocean polarity decided by `min_votes >= max_votes` on summed f32 votes (`boundary.rs:250`) —
a 1-ULP flip changes which side gets the trench. Prefer `BTreeMap` / explicit tie-breaks.

### L6 — Cross-cutting numerical/edge-case notes (sweep)
f32 `circumcenter_on_sphere` while qhull/area/validation use f64 (`voronoi.rs:285`); f32
`/slope`·`slope` round-trip drift over erosion steps (`erosion.rs:547`); percentile-normalization
scale can floor to `1e-6` on near-flat worlds (`atmosphere.rs:836`); sea-level percentile biases
land fraction slightly high (`elevation.rs:409`).

---

## Cross-check: sweep vs Codex

- **Independently corroborated by both:** C3 (cache wind), M5 (moisture diffusion), L3 (fine `acos`).
- **Codex-only (verified, new):** H1 (orphan cells), M4 (projection→moisture flux mismatch), L1
  (dead Lloyd param), L4 (drainage `None` docstring).
- **Sweep-only (Codex missed):** C1 (ocean uplift), C2 (√N forcing), C4 (circumcenter sign), M1
  (dead asymmetric model), H2/H3 (flat-fill cycles + unbounded loop). Codex's pass was shallower on
  erosion/features physics, deeper on mesh plumbing.
- **Apparent disagreement, resolved:** H4 (basin double-count) — both partly right; the scan itself
  is fine, the cross-stage accounting is not.

---

## Suggested fix order

1. **C3** cache key + wind, bump version — silent stale data; corroborated twice. (cheap)
2. **C1** erosion ocean-uplift guard — corrupts the headline terrain output.
3. **H3** overflow-loop cap (latent hang) + **H5** distance-normalized drainage (quality).
4. **C4** circumcenter sign → `center.dot(a)` — two lines, removes a topology-defect source.
5. **H1** orphan-cell repair — robustness of the fine mesh.
6. Then decide on **C2** (resolution-scaling of feature forcing) and prune/wire the dead
   elevation constants (**M1**); revisit the basin water budget (**H4/M8**).

---

## Fix status (branch `audit-fixes`, 2026-06-15)

Applied and verified (`cargo build`/`test`/`clippy` clean, 52 tests pass):

- **C3** — `fine_cache.rs`: `atmosphere.wind` added to the key; `FINE_BASE_CACHE_VERSION` 2→3
  (the bump also covers H5's effect on the fine density prior).
- **C1** — `erosion.rs`: tectonic uplift gated to land (`base >= 0`) at its source; submerged cells
  no longer accrue one-way thickness. Terminal-lake sinks intentionally left uplifting.
- **C4** — `voronoi.rs`: circumcenter hemisphere test now uses `center.dot(a)`.
- **H3** — `hydrology.rs`: `compute_overflow_targets` walk bounded by cell count (defensive; the
  walk is acyclic per the H2 analysis, but this guards against future routing changes).
- **H5** — `hydrology.rs`: `compute_steepest_descent` now picks the neighbor of greatest
  distance-normalized slope (chord), not lowest elevation. **Behavior-changing** — alters river
  networks and (via the coarse-hydrology preview) the fine-mesh sampling; needs Windows visual
  sign-off.
- **H1** — `tessellation.rs`: orphan cells in the s2/fine mesh are repaired by linking to the
  nearest generator (symmetric), instead of only a warning. kd-tree built only when orphans exist.

- **C2** — resolution invariance (follow-up commit). Corrected analysis: only the **7
  magnitude seeds** (trench/forearc/arc/ridge/collision/rift) were resolution-dependent — they feed
  `compute_smoothed_boundary_forcing` → `sqrt_response` with a fixed sensitivity, and per-cell
  forcing scaled ~1/√N (so amplitudes ~N^−¼). The 4 regime seeds were already invariant (their
  `compute_influence_field` reference scales the same way). Fix: those 7 seeds are now an
  edge-length-weighted **mean** rate × a **fixed** `FEATURE_FORCE_REF_SPACING` (= measured 0.012558
  at the 100k design resolution) — intensive, so amplitudes are cell-count-independent. Distance
  fields gate on `strength > 0` only, so geometry is unchanged; `*_SENSITIVITY`/`*_MAX` need no
  re-tuning. Verified by `force_seed_normalization_is_intensive` (a full-world 2-resolution test was
  tried and discarded — it conflates forcing-scaling with kernel-sampling and plate-layout noise).
  Feature values do shift slightly at 100k (multi-edge boundary cells now average, not sum); the
  fine-cache key hashes feature fields directly, so it auto-invalidates (no version bump).

- **H4 + M8** — basin water-budget rewrite (follow-up commit). Two coupled fixes:
  - **Local catchment (H4):** `compute_basin_catchments` no longer reads `flow_accumulation`
    (which routes through filled basins as if all spill). Each cell is now attributed to the
    **first basin its drainage reaches** (`compute_capturing_basin`), so an upstream basin's water
    is counted once, in that basin — not also downstream. `compute_flow_accumulations` is kept (it
    still feeds rivers / the density prior) but simplified to return just the discharge field.
  - **Topological cascade + merge (M8):** `calculate_water_levels` now merges mutually-overflowing
    basins (cycles in the functional overflow graph, via `group_overflow_cycles`) into one lake
    group with combined catchment + hypsometry, bounded by the group's lowest external saddle, then
    solves groups in **topological order** (Kahn) so received overflow is known before a group's
    level is computed. The old single pass ordered by spill elevation and silently dropped overflow
    into any downstream basin with a higher spill.
  - Verified by `basin_catchments_use_local_first_basin_attribution` (corrected the prior test,
    which had encoded the double-count: downstream 15→9), `overflow_cascade_follows_topology_not_
    spill_elevation`, and `mutually_overflowing_basins_merge_to_one_level`. Full-res stage-3 headless
    runs clean at 2.16M cells. **Behavior note:** removing the double-counted inflow makes some lakes
    smaller — expected and more correct; `climate_ratio` (Up/Down keys) compensates if lakes read too
    sparse. Needs Windows visual review.

Withdrawn: **H2** (no real cycle bug — see above).

Not yet addressed: **H6/H7/H8/H9**, all **M*/L*** items.
