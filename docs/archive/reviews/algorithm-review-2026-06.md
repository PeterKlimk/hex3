> **Archived:** Historical design or experiment record. It does not define current architecture, defaults, or priorities. Start at [`docs/README.md`](../../README.md).

# Algorithm Review — June 2026

A review of hex3's simulation algorithms (not code quality): tectonics, terrain/elevation,
atmosphere, hydrology, and the geometry pipeline. Based on four parallel domain critiques plus
an empirical analysis of a generated world (seed 12345, 100k cells, stage 3 — see
[analysis-seed-12345.md](../../generated/analysis-seed-12345.md) and `analysis-seed-12345.png`).

**Overall verdict:** the individual algorithms are well-chosen and mostly correctly implemented.
The quality ceiling is set by three missing *couplings* between subsystems, not by the algorithms
themselves:

1. **Monolithic plate typing** — whole plates are continental or oceanic, so continent edges and
   plate boundaries are forced to coincide. No passive margins; continent count is a dice roll.
2. **No moisture model** — atmosphere computes wind/uplift but nothing consumes them; hydrology
   runs on uniform rainfall. Climate cannot affect rivers.
3. **No erosion** — rivers are drawn atop terrain they never carved; noise amplitude is uniform,
   so lowlands are as rough as mountains.

Context note: moisture and erosion were consciously deferred ("we'll get to it") pending Voronoi
performance work, on the expectation that erosion needs re-tessellation or very high cell density.
The density math supports this: at 100k cells, mean spacing ≈ 0.0126 rad ≈ **80 km** at Earth
scale — one cell is wider than most mountain ranges. Resolving km-scale relief needs ~5×10⁸ cells
uniform, or ~10⁷ with aggressive variable density on orogens; that is the workload the s2-voronoi
kNN-clipping backend exists for. Two readings of the roadmap follow: (a) items 1–2 (crust typing,
moisture) are *resolution-independent* — they work at 100k and carry over unchanged to any future
density; (b) a cheap erosion *proxy* (elevation/flow-scaled noise amplitude + one smoothing pass
along drainage lines) runs on the existing 100k mesh with no re-tessellation and buys most of the
visual benefit at render scale, while full iterative erosion waits for the high-density mesh.

---

## Empirical baseline (seed 12345)

| Metric | Value | Reference / note |
|---|---|---|
| Plates | 14 (**1 continental**, 13 oceanic) | Greedy type assignment let the largest plate absorb the whole 30% continental budget |
| Collision feature max | **0.000** | One continent → no continental collisions possible |
| Land coverage | 16.8% | Earth ≈ 29% |
| Continental crust | 29.3% | ⇒ ~43% of continental crust submerged (Earth ≈ 25%) |
| Land elevation p90 | 0.017 (max 0.313) | Nearly all land is flat lowland; relief confined to tiny areas |
| Elevation histogram | 14.7% of surface in 0–0.04 band | Thin land tail: only 1.9% above 0.04 |
| Ridge feature max | 0.017 (trench 0.161, arc 0.400) | Ridge field nearly dead — thermal subsidence already provides the swell |
| Hypsometry | Bimodal (abyssal ≈ −0.35, shelf/land ≈ 0) | The hardest distributional property — present and correct |
| Islands (oceanic land) | 0.92% | |
| Lake coverage | 0.21% | |

Reproduce with:
```bash
cargo run --release -- --headless --seed 12345 --export world.json.gz
scripts/.venv/bin/python scripts/analyze_terrain.py world.json.gz
```

---

## A. Tectonics (plates, dynamics, boundaries)

### Sound
- **Weighted flood fill** (`plates.rs`): global priority queue with per-plate state (seed position,
  target size, noise offset); distance + noise + perimeter-preference produces organic,
  non-fractal boundaries. Multi-level seed-spacing relaxation handles tight packings gracefully.
- **Log-normal target sizes** approximate a varied size distribution.
- **Euler pole math** is correct: v = ω × r, with proper normal/tangential decomposition of
  relative velocity at boundaries (`boundary.rs`).
- **Two-pass boundary classification**: per-edge kinematics, then plate-pair aggregation —
  treats a long boundary as a coherent regime instead of sign-flipping micro-faults.
- **Subduction polarity voting** weighted by convergence magnitude and edge length, with
  age proxy (ridge distance) for ocean–ocean pairs.

### Weak
- **Plate typing is monolithic** (`dynamics.rs` ~70–114): plates sorted by size, greedily marked
  continental until 30% global coverage. Consequences:
  - Continent edges ≡ plate boundaries → **passive margins cannot exist** (most of Earth's
    coastline is passive margin: continental crust with oceanic seafloor offshore on the *same*
    plate).
  - Continent count is unstable across seeds (seed 12345: one continent).
  - Continental collisions are rare-to-impossible (need two adjacent continental *plates*).
  - The margin-distance machinery in `elevation.rs` has almost nothing to anchor to.
- **Velocities are pure RNG** (`dynamics.rs` ~117–134): angular velocity ~ Uniform(−1,1), axis
  uniform. No slab pull / ridge push / drag. A "convergent" boundary converges only by accidental
  geometry; kinematics are divorced from morphology.
- **Size distribution** is log-normal with a 4× ratio cap; Earth's is closer to power-law with a
  ~100× spread. Small plates (which drive much boundary complexity) are underrepresented.
- **No segmentation of long boundaries**: a 100° arc with locally varying kinematics is averaged
  into one regime by the plate-pair pass.

### Missing (ranked by impact)
1. **Time evolution** — snapshot simulation: no crust aging from actual spreading, no trench
   migration, no collision narrowing. Even 3–5 timesteps would create emergent geometry.
2. **Back-arc basins** — arcs are pure uplift; real overriding plates often extend behind the arc
   (Japan Sea, Mariana Trough). A subsidence lobe opposite the trench is a one-day change.
3. **Hotspots** — no intraplate volcanism; ocean interiors are monotonous. Fixed mantle-frame
   points + plate motion gives seamount chains nearly free once time evolution exists.
4. **Terrane accretion / plate fragmentation** — plate count is static.
5. **Rift shoulder uplift** — continental rifts lack the shoulder-uplift + axial-graben morphology.

---

## B. Terrain & elevation (features, elevation)

### Sound
- Edge-anchored Dijkstra distance fields (seeds at Voronoi edge midpoints, not cell centers) avoid
  quantizing features to boundary-cell centers at coarse resolution.
- Area normalization (`mean_area / cell_area`) keeps integrated forcing resolution-independent.
- Thermal subsidence uses √(distance-from-ridge), the right functional form (Parsons–Sclater),
  plate-constrained so crust doesn't "age" across plate boundaries.
- Isostatic base elevation (continental base / margin depth / oceanic thermal) produces genuine
  hypsometric bimodality — confirmed empirically.
- Four-layer noise (macro/hills/ridge/micro) with regime modulation is a reasonable hierarchy.

### Weak
- **Trenches are symmetric** (`exp_decay` both sides of boundary). Real trenches: steep seaward
  wall, gradual forearc rise on the overriding side. Subduction polarity is already computed —
  asymmetric decay is cheap.
- **Arc–trench gap uncoupled**: trench (exponential from boundary) and arc (Gaussian at
  `ARC_CONT_PEAK_DIST` = 0.05 rad ≈ 287 km) are independent fields. Real arc-trench gaps are
  ~100–200 km and consistent; consider 0.03–0.04 rad and a coupling constraint.
- **Collision band too narrow**: `COLLISION_WIDTH` = 0.02 rad ≈ 127 km produces ridgelines, not
  the 300–500 km plateau-bearing ranges of real continental collisions. Widen (0.035–0.05 rad)
  and add a broad low-amplitude secondary component for plateaus (Tibet pattern).
- **Ridge feature field is nearly redundant** (empirical max 0.017): thermal subsidence already
  produces the ridge swell. Fold it in or delete it.
- **Spreading rate ignored**: distance-from-ridge proxies age, implicitly assuming constant
  spreading rate. Divergence magnitude at the ridge is available and could modulate the
  subsidence width per ridge segment.
- **No-ridge fallback is crude**: plates with no ridge get uniform `ABYSSAL_DEPTH`; a mid-range
  depth (~−0.35) would read as "old but varied" rather than uniformly abyssal.
- **Hills suppressed in orogens** (`hills_orogen_suppress = 1 − 0.8·comp_driver`): arguably
  backwards — real orogens are the *rough* places (erosion roughens them). Tied to the erosion gap.
- **Ridge noise layer biased positive** (remapped to [0,1]) and gated only on convergence — in
  extensional zones this makes rifts read as uplands instead of grabens.
- **Passive margin profile is a linear blend** (margin depth → base over shelf width). Real
  margins: shelf → shelf break (~200–400 m) → steep slope → gentle rise → abyssal plain. Also
  uniform `CONTINENTAL_SHELF_WIDTH` = 0.04 rad everywhere; real shelves are wide on passive
  margins, narrow on active ones.
- **No isostatic root feedback**: very thick crust (collision + noise) doesn't depress its base.

### Constants flagged for re-tuning
| Constant | Current | Question |
|---|---|---|
| `TRENCH_DECAY` | 0.020 rad (127 km) | Too narrow if symmetric; asymmetric split preferred |
| `ARC_CONT_PEAK_DIST` | 0.05 rad (287 km) | ~50% farther inland than real arc-trench gaps |
| `COLLISION_WIDTH` | 0.02 rad (127 km) | Real collision zones 300–500 km wide |
| `THERMAL_SUBSIDENCE_WIDTH` | 1.5 rad (9550 km) | Consider 1.0–1.2 rad for a steeper aging profile |
| `CONTINENTAL_SHELF_WIDTH` | 0.04 rad uniform | Should vary by boundary regime (narrow active, wide passive) |
| Noise amplitudes | macro 0.12 vs hills+ridge+micro ≈ 0.29 | Noise outweighs tectonic signal; revisit balance |

---

## C. Atmosphere

### Sound
- Geostrophic deflection is implemented correctly (`−pos.cross(pressure_grad)` rotates the
  gradient 90° in the tangent plane; blend strength scales with |latitude| so it vanishes at the
  equator where Coriolis breaks down).
- Terrain permeability formulation `k = 1/(1+g²)^(p/2)` for routing wind around mountains is
  sensible; uphill blocking + katabatic acceleration are reasonable single-layer effects.
- The pressure-projection (Chorin-style finite-volume Poisson solve with SOR on the Voronoi mesh)
  is mathematically correct.

### Weak / structural
- **Circulation is prescribed, not emergent**: three hard-coded zonal bands (trades 0–30°,
  westerlies 30–60°, polar easterlies 60–90°). *Editorial note: this is a defensible procgen
  choice — emergent circulation is a research project. The bands are fine; what's missing is what
  they should transport (moisture).*
- **No moisture, anywhere** (confirmed by search): no humidity/precipitation field exists in any
  struct. Without it: no rain shadows, no deserts vs rainforests, no monsoons, no wet equator.
  Uplift is computed but drives nothing.
- **The divergence-free projection works against the goal**: enforcing zero divergence on a
  single-layer steady wind field removes exactly the convergence signal (ITCZ-style) that drives
  precipitation in reality — which is why uplift must use *pre-projection* winds as a proxy. For
  a slab model, allowing divergence and interpreting it as vertical motion (→ precipitation) is
  more useful than conserving mass.
- **No land–ocean thermal contrast**: temperature depends only on latitude + elevation lapse.
  Continentality (land heats/cools faster) is the driver of monsoons and interior deserts; the
  elevation struct already knows crust type, so a thermal bias is a small change.
- **Geostrophic signal is heavily damped**: effective pressure-gradient contribution to final wind
  is ~0.12× (PRESSURE_WEIGHT 0.4 × PRESSURE_WIND_SCALE 0.3); zonal bands dominate.
- `SURFACE_CORIOLIS_ANGLE` (45° Ekman-style turning) is defined in constants but never used.
- Uplift is percentile-normalized, destroying absolute magnitude — fine for visualization, lossy
  as a future precipitation driver.

### Recommended pivot
Implement a simple moisture balance *before* anything else atmospheric: moisture picked up over
ocean, advected with the (existing) wind field, precipitated proportional to uplift × saturation,
with the residual continuing downwind. This single feature unlocks ~80% of climate realism and
gives hydrology a real input.

---

## D. Hydrology

### Sound
- **Priority-flood depression filling** is correctly implemented (ocean-seeded, floods inward by
  elevation, fills basins on detection), with flood-parent fallback giving deterministic drainage
  across filled flats.
- **Flow accumulation** via topological sort is correct.
- **Endorheic basins emerge naturally**: water level equilibrium `lake_area = catchment ×
  climate_ratio` with overflow-target chains means terminal (Caspian-style) lakes happen when a
  basin equilibrates below its spill elevation. This is a genuinely nice property.
- The climate-ratio mechanism is a mathematically consistent precipitation/evaporation *balance*.

### Weak
- **Uniform rainfall**: every cell contributes exactly 1.0 to flow accumulation. Desert and
  rainforest river networks are topologically identical. The fix is one line
  (`flow[cell] = precip[cell]`) once a precipitation field exists — the bottleneck is atmosphere
  (section C), not hydrology.
- **Climate ratio is a global scalar** — effectively a sea/lake-level slider, not spatial
  hydrological rebalancing.
- **No erosion feedback**: hydrology is read-only from elevation. Rivers don't carve valleys,
  no deltas, no incision. River width rendering is cosmetic.
- Flat-basin drainage on perfectly level filled plateaus depends on flood exploration order —
  pathological-case only, low priority.

---

## E. Geometry pipeline

### Verdict: clean bill of health
- **Fibonacci lattice + jitter (0.25× spacing) + k-means Lloyd (2 iterations × 20 samples/site)**
  is a justified speed/quality tradeoff: ~CV 0.10–0.12 cell-area variation at 100k cells in ~1 s,
  vs full Lloyd's marginal improvement at ~60× the cost.
- **Spiral/banding artifacts**: the golden-ratio spiral survives as very low-frequency modulation
  but is below perceptual threshold after jitter + Lloyd, and masked by noise/tectonics. If ever
  observed, bump `LLOYD_ITERATIONS` 2→3 (+~0.5 s).
- **Area normalization** in features compensates properly for residual cell-size variance.
  Two non-normalized consumers exist but don't matter at 100k cells: flow accumulation
  (1.0/cell regardless of area) and noise sampling — both only visible at ≥1M cells.

---

## Consolidated roadmap

| # | Change | Unlocks | Effort | Notes |
|---|--------|---------|--------|-------|
| 1 | **Per-cell crust type** (continents grown independently of plates) | Reliable multi-continent worlds, passive vs active margins, actual collisions, controllable land fraction | ~2–3 days | Contained to `dynamics.rs` typing + `elevation.rs` margins; `boundary.rs` already works per cell pair |
| 2 | **Moisture → precipitation → flow weights** | Deserts, rain shadows, climate-coupled rivers, biome groundwork | ~2–3 days | Advect over existing wind field; hydrology side is one line |
| 3 | **Cheap erosion proxy** | Carved valleys, flat floodplains, rough orogens | ~1 day | Elevation/flow-scaled noise amplitude + smoothing pass along drainage; no re-tessellation needed |
| 4 | **Feature geometry fixes** | Recognizable tectonic landforms | ~1–2 days | Trench asymmetry, collision width+plateau, arc-trench gap, back-arc subsidence lobe, rift grabens |
| 5 | **Plate motion plausibility / time stepping** | Self-consistent kinematics, real crust age, hotspot trails, emergent boundary geometry | open-ended | The fun rabbit hole; full iterative erosion at high density also lives here, where the s2-voronoi backend pays off |

Quick wins independent of the above: widen `COLLISION_WIDTH`; delete or fold in the ridge feature
field; add land–ocean thermal bias to upper temperature; use `SURFACE_CORIOLIS_ANGLE` or remove it.
