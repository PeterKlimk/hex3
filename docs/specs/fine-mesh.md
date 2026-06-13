# Spec: Fine-Mesh Refinement Stage (erosion infrastructure)

Part (a) of the erosion project; `docs/specs/erosion.md` builds on this.
Loose spec: mechanism, contracts, and invariants — implementation layout is
the implementer's call. Leave `// SPEC:` comments for genuine ambiguities
instead of resolving them ad hoc. No smoothing hacks, no constant retunes.

## Goal

Stage 3 gains a refinement step: the coarse world (~100k cells, stages 1-2
unchanged) is resampled onto an adaptive-density fine mesh (default ~2.5M
cells: tiny cells on mountains and river corridors, huge cells on the
abyssal plain). Elevation is rebuilt on the fine mesh from transferred
structure; hydrology then runs on the fine mesh instead of the coarse one.
After this spec, the fine world should look like the coarse world rendered
denser — erosion (separate spec) is what will make it look *better*.

## Mechanism

1. **Density prior from the coarse world.** Run coarse hydrology (existing
   code, unchanged) as a preview at the end of stage 2. Per coarse cell:

       density = OCEAN below sea level;
                 on land: BASE + slope term + flow term

   with the slope term from the coarse elevation gradient, the flow term
   from log-scaled coarse flow accumulation (river corridors attract
   resolution), and an optional tectonic-activity term (uplift forcing) —
   pre-erosion slopes understate where erosion will carve detail, e.g. on
   active plateaus. Knobs: total budget `FINE_NUM_CELLS`, ocean-vs-land
   base ratio, slope/flow/activity weights, and a max:min density ratio
   (default ~50x). The coarse hydrology preview is discarded afterward.

2. **Weighted sampling, no Lloyd.** Generate fine points with density
   proportional to the prior: jittered-Fibonacci candidates thinned by
   rejection against the local density (deterministic from the world seed).
   Must hit the target count within a few percent and stay blue-noise-ish
   (no clumping artifacts at density steps). Tessellate via the existing
   knn-clipping backend (native adjacency landed in `04b30a5`).

3. **Fine-to-coarse mapping.** Store, per fine cell, its nearest coarse
   generator. This mapping is a keeper — render modes that color by
   plate/feature at stage 3 use it too.

4. **Field transfer.** Two classes, no exceptions:
   - *Smooth fields*: interpolate (inverse-distance over a small coarse
     neighborhood — k-nearest generators or nearest-cell-plus-neighbors,
     implementer's choice). Transfer what elevation reassembly and the
     later erosion/hydrology need: crust thickness, continentality fraction,
     ridge_age_distance, signed trench/flexure, ridge feature, the noise
     modulation fields (stress/regime weights), temperature, precipitation,
     uplift. These are all low-frequency by construction (Moran's I guards),
     so interpolation cannot imprint the coarse cells.
   - *Noise*: NEVER interpolated. Re-evaluate the same seeded fBm functions
     at the fine cell centers — they are continuous on the sphere, so this
     is exact and resolution-free.

5. **Elevation reassembly** on the fine mesh uses the existing formula
   (isostatic(thickness) + thermal(age)*(1-cont) + ridge - trench + noise)
   with the SAME code paths where practical (refactor elevation assembly to
   take fields as inputs rather than duplicating it). Noise layers on the
   fine mesh: macro and micro only — ridges and hills are retired there
   (erosion will supersede them; until the erosion spec lands the fine
   world is legitimately smoother than the coarse one at medium scales).
   Sea level: re-solve once on the fine pre-erosion elevation (same
   LAND_FRACTION machinery), then treat as fixed.

6. **Hydrology moves to the fine mesh.** Hydrology runs on the fine
   tessellation with transferred precipitation/temperature. Do NOT
   synthesize a fake fine `Crust` to satisfy the current signature:
   refactor Hydrology to take the specific inputs it actually uses (and
   derive what's needed, e.g. continental/oceanic from the transferred
   continentality fraction). Rivers, basins, lakes, and the V-key river
   rendering all come from the fine world at stage 3+. Coarse hydrology no
   longer ships in the final World.

7. **Integration.** World carries the fine tessellation + fine fields from
   stage 3 on; rendering and export switch to the fine mesh at stage 3
   (relief view, terrain coloring, rivers). The fine mesh renders as a
   SHARED-VERTEX, vertex-colored mesh (colors averaged from adjacent
   cells): per-cell flat coloring duplicates ~6 vertices per cell, which at
   2.5M cells is over a GB of vertex data for cell edges that are subpixel
   anyway. Shared vertices are ~5x smaller and relief displacement is
   already per-vertex. The coarse mesh path (stages 1-2, plate/feature
   views) keeps its current representation. Climate views may render
   interpolated fine values via the transfer — they're smooth either way.
   Wind particles keep using the coarse wind field.

## Performance contract

Benchmarked budget on the user's machine (see `bench_mesh`): tessellation
+ native adjacency ~2.5s at 2.5M. The whole refinement stage (density,
sampling, tessellation, transfer, reassembly, fine hydrology) should land
in roughly 10s at the default budget; log at info level: per-phase
timings, fine cell count, achieved max:min density ratio, and GPU buffer
sizes (vertex/index bytes) for the fine mesh. No phase may be accidentally
quadratic in cell count.

## Validation

- `cargo test` green, including field smoothness on fine fields (Moran's I
  on the fine mesh; transferred fields must stay smooth).
- Headless export of the fine world; render elevation/precipitation maps at
  coarse and fine and compare by eye: same world, denser — no coarse-cell
  staircase imprint, no density-boundary artifacts in elevation.
- Land fraction on the fine mesh within 0.5pp of LAND_FRACTION.
- diagnose runs against the fine world where its measures apply (continents,
  hypsometry, rivers, lakes) and reports fine cell count + density ratio
  achieved.
- A render of cell areas (or density) as a map layer, to eyeball that
  mountains/rivers actually attracted resolution.

## Non-goals

Erosion itself (next spec). Submarine detail. Changes to stages 1-2
physics. GPU-side LOD tricks. Backward compatibility of exports that only
made sense on the coarse mesh.
