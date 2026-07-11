> **Dated audit evidence:** Results are revision/configuration specific and do not define current defaults.

# Resolution-independence test (2026-06-16)

How invariant is world-gen output to mesh resolution? Two **independent**
resolution axes, tested separately:

- **Coarse axis** — `NUM_CELLS` (the Voronoi cell count). Drives tessellation →
  plates → crust → features → elevation → atmosphere → coarse hydrology
  (stages 1–2). Now CLI-settable: `--cells N`.
- **Fine axis** — the adaptive erosion mesh, sized by absolute km targets
  (`FINE_*_CELL_KM`), independent of `NUM_CELLS`. Drives the erosion mesh, fluvial
  erosion, and fine hydrology (stages 3–4). Now sweepable with one knob:
  `--fine-scale F` (uniform multiplier on the three km targets; >1 coarsens,
  <1 refines).

## Tooling added

- `src/main.rs`: `--cells N`, `--fine-scale F` on the headless/export path.
- `src/bin/diagnose.rs`: `--fine-scale F` convenience multiplier.
- `scripts/resolution_compare.py`: cross-resolution comparison harness.
  - `--mode coarse`: **statistical-invariance** test across seeds (see below).
  - `--mode fine`: rasterized **spatial** cell-for-cell convergence vs the
    highest-res reference (valid only when the coarse world is held fixed).

### Why the coarse axis needs a *statistical* test, not a spatial one

The coarse mesh is a **Fibonacci lattice + jitter** of `N` points
(`fibonacci_sphere_points_with_rng`), and plate/craton seeds are placed
*relative to those cells*. So changing `N` relocates the entire geography — the
"same seed" does **not** give the same continents at a different cell count.
A cell-for-cell spatial comparison across coarse resolutions therefore measures
*world difference*, not resolution dependence (empirically: elevation grid
correlation between 100k and 200k is ~0.2, while latitude-driven temperature is
~0.96). The meaningful question is statistical: **does resolution move a metric
more than reseeding does?** The harness builds a `[seed × resolution]` matrix per
metric and reports `ratio = (drift across resolution) / (spread across seeds)`,
gated by a 2%-relative practical-significance floor.

## Coarse-axis results (seeds 12345, 777 × cells 50k/100k/200k/400k, stage 2)

`python scripts/resolution_compare.py /tmp/restest/coarse/*.json.gz --mode coarse`

| System / metric | Verdict | Notes |
|---|---|---|
| Land fraction, continental fraction | **invariant (by construction)** | sea level solved to pin area-weighted land at `LAND_FRACTION=0.26` on any mesh (`elevation.rs:404`); both seeds give 0.2600 |
| Precipitation mean (land) | **invariant** | resΔ 0.5% |
| Land hypsometry (p50/p95), thermal structure | **invariant / seed-dominated** | resolution within seed noise (temperature is latitude-driven) |
| Ocean depth — `elev_p5 / p50 / mean` | **drifts ~8–12%** | monotonic deepening with resolution; ratio 8–13 |
| Atmosphere uplift — `uplift_p99` | **drifts ~18%** | monotonic ↓ (1.41→1.17); the least resolution-robust field |
| Arid land fraction | **drifts** | 0.69→0.81 with resolution |
| Feature p99 tails (activity, arc, ridge) | drifts, but extreme-value stats | sharpen on finer meshes inherently |

**Interpretation.** Area-weighted *intensive* invariants hold. The genuine soft
spots are (1) the elevation **distribution tails** — with sea level pinned to
hold land area, fixed-frequency noise resolves deeper ocean troughs as cells
shrink, widening the elevation distribution (a property of sampling
band-unlimited noise, not a normalization bug); and (2) the **atmosphere uplift
field**, consistent with the self-documented fixed-50-iteration projection
solver. Extreme-value (p99) feature stats drift because extremes always sharpen
with resolution — read feature **means/area-weighted** quantities for the
invariant signal.

### Fix applied — adaptive projection solver

The wind-projection Poisson solve (`project_wind_field`) ran a **fixed 50**
Gauss-Seidel/SOR sweeps. It is an *unscreened* Poisson solve (unlike the
*screened* diffusion solver, which is already resolution-adaptive), so its
low-frequency error decays ~O(1/h²) per sweep — a fixed count under-develops the
large-scale `phi` (and the convergence-derived uplift) as cells shrink. Fix:
scale the sweep count by `(PROJECTION_REFERENCE_SPACING / mean_h)²`, clamped to
`[50, 800]`, and **precompute the edge conductances once** (lifts an `acos` out
of the hot loop so the extra sweeps are cheap). Result:

| metric | before (ratio / resΔ) | after (ratio / resΔ) |
|---|---|---|
| `uplift_p99` | 13.8 DEPENDENT / 0.242 | **2.2 watch / 0.053** |
| `precip_arid_frac_land` | 7.2 DEPENDENT / 0.120 | 3.7 / 0.064 |

`uplift_p99` is now flat at ~1.41 across 50k–400k (was 1.41→1.17); drift cut
~4.6× to the seed-noise floor. Cost: ~103 sweeps at the 100k default
(sub-second); 411 sweeps / ~14 s at 400k (coarse-mesh-only, one-time). **Note:**
the 100k baseline `uplift_p99` shifts ~1.32→1.41 (more converged) — eyeball the
climate map on Windows to confirm the look is still good.

## Fine-axis results (seed 12345, `--fine-scale` 2.0→0.65, stage 4 erosion)

Instrument: `diagnose`'s built-in **fixed-radius local relief** probe (max−min
eroded elevation within R km — resolution-*controlled*, so a rising value means
"still under-resolved" and a flat value means "converged").

| F | fine cells | R10 p50/p90/p99 | R25 p50/p90/p99 | top orogen peak | RSS |
|---|---|---|---|---|---|
| 2.0 | 363k | .019 / .127 / .273 | .085 / .242 / .417 | +0.54 | 0.37 GB |
| 1.5 | 638k | .039 / .141 / .273 | .096 / .241 / .428 | +0.54 | 0.60 GB |
| 1.2 | 992k | .063 / .149 / .282 | .099 / .235 / .415 | +0.54 | 0.84 GB |
| 1.0 | 1.43M | .070 / .148 / .273 | .101 / .231 / .414 | +0.54 | 1.18 GB |
| 0.8 | 2.23M | .074 / .146 / .266 | .101 / .221 / .406 | +0.54 | 1.66 GB |
| 0.65 | 3.37M | .075 / .143 / .263 | .099 / .216 / .396 | +0.54 | 2.38 GB |

(The F=0.65 row required fixing a `diagnose` KD-tree probe panic — see below. Its
flat p90/p99 confirm the relief tail is converged through 3.37M cells.)

**Converged.** The relief-bearing **tail (p90/p99) is flat across the 6× cell
range**, and the top orogen peak is +0.54 at every scale — major relief does not
grow under refinement. Only the R=10km **median** (p50) is still rising at coarse
fine-meshes (.019→.074) but decelerating (Δ +.020,+.024,+.007,+.004): typical
mountain small-scale dissection converges right around the **F=1.0 baseline**.
Refining to 2.23M cells barely moves the relief tail. Confirms the design intent
(relief outcome stable ±few% across 300k–2.5M cells).

One metric still trends: **drainage-density wet/arid ratio rises 1.96→4.57** with
resolution — finer meshes resolve more low-order channels in wet uplands. Worth a
look if drainage texture matters.

### Fine-axis spatial convergence (`--mode fine`)

Because the coarse world is *fixed* across `--fine-scale`, cell-for-cell spatial
comparison IS valid here. Stage-4 exports at F=1.5/1.0/0.8 (645k/1443k/2251k fine
cells), rasterized to a 720×360 grid and compared to the finest as reference
(`--mode fine --plot`):

| field | 645k (nRMSE/corr) | 1443k (nRMSE/corr) | trend |
|---|---|---|---|
| elevation | 0.140 / 0.990 | 0.102 / 0.995 | **converging** |
| temperature | 0.026 / 1.000 | 0.017 / 1.000 | converged |
| precipitation | 0.175 / 0.995 | 0.081 / 0.998 | converging |
| uplift | 0.084 / 0.996 | 0.057 / 0.998 | converging |
| feat_* (trench/arc/…) | 0.22–0.47 / 0.89–0.98 | 0.16–0.34 / 0.94–0.99 | converging |
| **log_flow (rivers)** | 1.36 / 0.43 | 1.16 / **0.41** | **NOT converging** |

Eroded **terrain, climate, and features all converge** (corr → 1, nRMSE → 0 as
resolution rises). The elevation difference map shows residual only as thin lines
on coastlines and ridge crests (sub-grid-sharp edges), white interiors — the
signature of a converged field.

The **river flow field does not converge spatially** (corr ~0.41): the drainage
network reorganizes between fine meshes. The flow difference map concentrates in
the wet climate bands — a mix of (a) diffuse low-order accumulation being
discretization-dependent (finer mesh spreads upstream area over more cells) and
(b) trunk channels taking different paths. The aggregate drainage *density* is
roughly stable, but individual channel *placement* is resolution-sensitive — the
fine-axis counterpart to the coarse "geography moves with N" finding, and largely
inherent to drainage networks on adaptive meshes. (`max_flow_equiv` appears to
drift 75%, but that is a metric artifact: the count-equivalent divides physical
discharge by `mean_cell_discharge`, which shrinks as the adaptive mesh refines —
the physical discharge is the invariant.)

### Known issue surfaced — FIXED

`diagnose`'s final "Fine-scale local relief" probe panicked (exit 101) at ~3.37M
cells (F=0.65): kiddo's `ImmutableKdTree::new_from_slice` hits `mid > len` on the
near-coincident points that appear at multi-million-cell fine meshes. Fixed by
switching that probe to the mutable `KdTree` (the pattern `fine.rs` already uses
at the same scale). Verified clean at 3.37M cells.

## Reproduce

```bash
# Coarse axis
for s in 12345 777; do for c in 50000 100000 200000 400000; do
  ./target/release/hex3 --headless --seed $s --cells $c --stage 2 \
    --export /tmp/restest/coarse/s${s}_c${c}.json.gz
done; done
python scripts/resolution_compare.py /tmp/restest/coarse/*.json.gz --mode coarse

# Fine axis
for F in 2.0 1.5 1.2 1.0 0.8; do
  ./target/release/diagnose --seed 12345 --fine-scale $F
done
```
