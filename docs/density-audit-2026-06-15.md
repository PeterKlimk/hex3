# Fine-Mesh Density Allocation Audit — 2026-06-15

Empirical audit of the adaptive fine-mesh cell budget: **is the cell count spent where the terrain
needs it, and can it be lowered by coarsening ocean/plains without starving mountains?** Seed 12345,
the default coarse world (100k cells); fine mesh emergent ≈ **2.17M cells** at the current knobs.

The density prior is `compute_areal_density` (`fine.rs`): ocean cells get a fixed coarse size; land
interpolates from `FINE_PLAINS_CELL_KM` to `FINE_MOUNTAIN_CELL_KM` by a normalized **demand** =
weighted blend of slope / log-flow / activity, each raised to `FINE_DENSITY_FEATURE_EXPONENT`. Total
cell count *emerges* from integrating density over the sphere; `FINE_MAX_CELLS` is a guardrail that
uniformly coarsens if exceeded.

## The measure

**Relief-per-cell** `Δh_i = max_neighbour |elev_i − elev_j|` — the resolved elevation step across a
cell — as an **equidistribution monitor**. Local interpolation error scales like
`|∇h|·spacing ≈ Δh`, so an optimally adaptive mesh flattens `Δh` over the terrain it cares about.
Where `Δh/cell` is tiny (plains) cells are smaller than the terrain needs (wasted budget); where it
is large (mountains) cells carry the representation error. Reported per terrain class (ocean /
lowland / upland / mountain, by eroded elevation) with cell budget, **area-weighted** `Δh`
percentiles, and a **slope (Δh/km)** column that strips out cell size.

Implemented as the "Density allocation audit" section of `src/bin/diagnose.rs`.

**Caveat (important).** This is a *warning light for mis-allocation, not an absolute "dense enough"
proof.* It is measured on the already-eroded fine mesh, so detail that finer cells *would* create is
invisible — a low mountain `Δh` could mean "well resolved" or "never formed." Per-cell `Δh` also
shrinks trivially as cells shrink, so it cannot be the *convergence* signal (see Open items). It
answers "is the budget balanced," not "is class X absolutely adequate." (Independent review by Codex
concurred; the slope column and area-weighting were added on its critique.)

## Tooling

Runtime density knobs (`FineDensityParams`, defaulting to the `FINE_*` consts; part of the fine-base
cache key) thread through `World` → `FineWorld::generate_pre` → `compute_areal_density`, mirroring
`ErosionParams`. `diagnose` exposes them as `--fine-plains-km / --fine-ocean-km / --fine-mountain-km
/ --fine-density-exponent / --fine-{slope,flow,activity}-weight` (any override forces fine-base
regeneration, bypassing the cache). Sweep without recompiling.

Compare the **emergent** count (the density integral, logged before the cap), not the achieved count
— the latter drifts run-to-run by up to ~20% from s2-voronoi weld nondeterminism. A capped run still
logs the true emergent count, so emergent sweeps are fast.

## Baseline allocation (2.17M cells)

| class | % cells | % area | median cell | Δh/cell p90 | slope p90 (Δh/km) |
|---|---|---|---|---|---|
| ocean | 9.3% | 74.1% | 50 km | 0.054 | 0.0015 |
| lowland | 32.7% | 9.1% | 6 km | 0.0065 | 0.0012 |
| upland | **39.4%** | 15.6% | 11 km | 0.013 | 0.0020 |
| mountain | 18.6% | 1.5% | **3.9 km** | 0.104 | 0.025 |

Equidistribution (area-weighted Δh p90): mountain/lowland **12.7×**, mountain/upland **9×**,
mountain/ocean ~1.6×. Slope ratio (cell size removed): mountain/lowland **~20×**.

Reading: **lowland + upland = 72% of the budget** carrying 8–12× less relief and ~20× less slope per
cell than mountains — over-resolved. Ocean's high Δh is *cell size, not steepness* (slope ratio ~1),
so its relief doesn't indicate a need for resolution; it's underwater (erosion skips it). Mountains
hold nearly all the relief on 18.6% of cells, and achieve a **3.9 km median vs the 1.5 km knob** —
the `exponent=3` demand curve only reaches full mountain density at peak slope+flow+activity.

## Q1 — Lowering the cell count

Emergent vs baseline (2,165,375), single-knob:

| change | emergent | Δ |
|---|---|---|
| ocean 60→100 km | 2.10M | **−3.1%** |
| ocean 60→150 km | 2.08M | −4.1% |
| plains 12→20 km | 1.59M | **−26.7%** |
| plains 12→30 km | 1.41M | **−35.0%** |
| exponent 3→4 | 1.88M | −13.4% |
| exponent 3→2 | 2.82M | +30.2% |
| mountain 1.5→1.0 km | 3.61M | +66.8% |

- **Plains is the strong lever** (−27% at 20 km). The flow term keeps river valleys fine regardless,
  but the bulk of plains/upland sits near the size floor, so raising it cuts deep.
- **Ocean is a weak lever and saturates fast** (60→100 = −3.1%; 100→150 only −1% more). Ocean is
  only ~9% of cells, so there isn't much there — coarsen it (it's underwater and visually unimportant)
  but don't expect more than ~4–5% from it. *Decision (2026-06-15): coarsen ocean further anyway
  (~100–120 km) — it looks over-large and the detail is worthless — accepting the small gain.*
- **Exponent concentrates the budget**: *higher* exponent (sharper demand) → fewer total cells, more
  of them in mountains; *lower* spreads to gentle terrain (more cells). (Note: this is the opposite
  of a "lower the exponent" intuition — sharpening is what concentrates on mountains.)

## Q2 — Are mountains dense enough?

**Relatively under-resourced** (firmly established): they carry 8–12× the relief and ~20× the slope
per cell of plains on 18.6% of cells, and the `exponent=3` dilution makes the achieved median 3.9 km
vs the 1.5 km knob. Real mountain ridge/valley spacing is ~2–10 km, so 3.9 km cells under-resolve
fine dissection.

**Absolutely adequate?** Not yet proven. A first (noisy) read showed denser mountains (1.0 km) raised
drainage density ~2×, consistent with under-resolution, but that needs the convergence study below.

## Recommendation: rebalance, don't just cut

Coarsening plains/ocean and sharpening the demand curve **cuts ~40–50% of cells while raising the
mountain share** — the gentle-terrain over-resolution funds finer mountains:

| config | emergent | Δ | mountain share |
|---|---|---|---|
| baseline | 2.17M | — | 18.6% |
| plains 20 + ocean 100 + exponent 4 | 1.23M | **−43%** | **25.8%** |
| plains 25 + ocean 100 + exponent 4 | 1.11M | **−49%** | ~26% |

The rebalance also improves the equidistribution ratio (mtn/lowland 12.7× → 6.2×). The production
`FINE_*` constants are **unchanged** pending a visual sign-off — applying the rebalance is a next
step, not done here.

## Open items / next steps

1. **Rigorous mountain convergence study** (the absolute "dense enough" answer). Needs a
   *scale-controlled* metric — fixed-radius local relief (max−min elevation within e.g. 10/25/50 km)
   and/or drainage density — measured **uncapped** across a `FINE_MOUNTAIN_CELL_KM` 1.5→1.0→0.75
   sweep. Stop when halving cell size moves mountain local-relief p90/p99 and drainage density by
   only a few percent. Per-cell `Δh` can't serve here (shrinks with cell size).
2. **Coastal/shelf band.** A fixed coarse ocean prior is fine offshore, but coarser ocean risks
   blocky coastlines/shelves/deltas/lake-outlets. If that shows visually, add a near-shore
   refinement term rather than a single flat ocean size.
3. **Apply the rebalance** to the production constants after visual review, with the cache version
   bumped (or rely on the content-hash of the changed knobs).
4. **Protect lowland rivers**: when raising `FINE_PLAINS_CELL_KM`, keep or raise the flow weight so
   major valleys stay resolved as interfluves coarsen.
