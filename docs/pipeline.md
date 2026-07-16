# Stage and state pipeline

This is the canonical description of Hex3 generation stages and retained state.
It distinguishes computation from presentation and coarse state from fine state.
It describes the implemented frontier only. Stage 4 is not a declaration that
the world model is complete, and the present stage boundaries may be redesigned.

## Stage summary

| Stage | Runtime name | Spatial state | Principal products |
|---:|---|---|---|
| 0 | Geometry/bootstrap | Coarse | Spherical tessellation and adjacency |
| 1 | Lithosphere | Coarse | Plates, crust, motion, boundaries/history/features, elevation |
| 2 | Atmosphere | Coarse | Temperature, pressure, surface/upper wind, uplift, precipitation |
| 3 | Hydrosphere (pre-erosion) | Fine | Adaptive fine base, transferred fields, pre-erosion hydrology |
| 4 | Erosion | Fine | Eroded surface, local climate adjustments, final hydrology |

Stage names describe runtime milestones, not closed scientific domains. For
example, hydrology exists at Stage 3 and is recalculated during Stage 4; erosion
depends on tectonic and atmospheric inputs; presentation is available throughout.

## Stage 0: geometry/bootstrap

`World::new_with_options` creates the coarse `Tessellation`. The default backend
uses convex-hull duality; the optional kNN clipping backend is faster and
approximate. Lloyd-style relaxation improves generator distribution. The result
provides spherical cells, vertices, compact adjacency and cell areas.

This stage is reported as stage 0 until elevation exists. The interactive app
normally proceeds directly to Stage 1 rather than presenting geometry alone.

## Stage 1: lithosphere

Generation order is:

1. `generate_plates` partitions cells into motion units.
2. `generate_crust` grows independent crust/craton regions.
3. `generate_dynamics` assigns Euler-pole rotations to plates.
4. `generate_features` computes tectonic history and boundary-driven fields
   using the selected `OrogenModel`.
5. `generate_elevation` produces coarse elevation and bathymetry.

The stage is complete when `World.elevation` exists. It retains all intermediate
state because atmosphere, fine sampling, diagnostics, overlays and exports
consume more than elevation alone.

`World::generate_all` is a legacy name for precisely this Stage-1 sequence; it
does not generate atmosphere, hydrology or erosion.

## Stage 2: atmosphere

`generate_atmosphere` consumes coarse elevation, crust setting and tessellation.
It derives connected-ocean identity before hydrology exists, then computes
temperature, circulation, pressure/wind/uplift and moisture/precipitation on the
coarse graph. The atmosphere remains retained after refinement because wind
particles, coarse views and fine-field transfer consume it.

Stage 2 changes modeled state but not spatial resolution.

## Stage 3: adaptive fine base and pre-erosion hydrology

`generate_fine_pre_with_cap` requires Stage-1 and Stage-2 products. It:

1. derives convergent-front primitives on the coarse graph;
2. computes or loads a cache-keyed `FineBase`;
3. creates an adaptive fine tessellation and coarse-to-fine mapping;
4. transfers coarse crust, feature, terrain and climate fields;
5. synthesizes configured fine substrate structure;
6. derives a `FineSurface` named `pre`;
7. computes hydrology on that un-eroded surface.

Before fine erosion or hydrology consumes it, transferred or locally adjusted
precipitation is area-normalized over land defined by the same connected-ocean
classifier. This is required even with climate modifiers disabled because
adaptive transfer does not preserve the land water budget automatically.

Creating fine state clears the legacy top-level hydrology fallback. From this
stage onward, active terrain normally comes from `FineWorld`.

The fine cell cap is a quality/cost control and diagnostics may use smaller caps.
The default can reach millions of cells, making cache identity, compact
adjacency, parallel construction and area-aware algorithms architectural rather
than incidental implementation details.

## Stage 4: erosion and final hydrology

`generate_fine_eroded` reuses the existing `FineBase` and `pre` surface. The
fine erosion pipeline evolves elevation through configured fluvial, hillslope,
uplift and optional experimental processes. It applies bounded fine climate
feedback and derives hydrology again on the eroded terrain. The resulting
surface is retained as `FineWorld.eroded`; Stage 3 is not destroyed.

Hydrology may lower outlet-path cells during its explicit drainage-integration
hack. Each `Hydrology` now retains sparse exact records of those cuts and a
bitset of source-basin cells selected for breaching. The final surface remains
the post-integration authoritative terrain, but diagnostics can reconstruct the
terrain supplied to hydrology and distinguish erosion from integration cuts
without retaining a second full elevation array.

`rerun_fine_eroded` replaces only the eroded surface using current erosion
parameters. It deliberately avoids regenerating the expensive fine base, making
controlled erosion comparisons practical.

For headless/batch use, `generate_hydrology_with_fine_cap` is the convenience
path that computes both Stage 3 and Stage 4. Despite its name, its terminal
result is a full eroded fine world.

## Computed stage versus viewed stage

`World::current_stage()` reports the furthest computed state:

- elevation present: 1;
- atmosphere present: 2;
- fine pre-surface present: 3;
- fine eroded surface present: 4.

The app separately stores `viewed_stage` and applies it through
`World::set_view_stage`. The `active_*` accessors then expose the appropriate
tessellation, elevation, climate and hydrology:

```text
view 1-2 -> coarse tessellation and coarse elevation
view 3   -> fine tessellation and FineWorld.pre
view 4   -> fine tessellation and FineWorld.eroded, if computed
```

Backspace changes the viewed stage without removing later state. Space first
steps forward through already-computed views; only at the latest view does it
compute the next stage. GPU buffers for inactive visited stages are cached for
fast comparison.

This is stage navigation, not time reversal or simulation rollback.

## Field ownership across stages

| Field | Coarse authority | Fine behavior |
|---|---|---|
| Plate/crust/dynamics/history/features | Stage 1 | Referenced or transferred; not recomputed as a full fine tectonic model |
| Elevation | Coarse Stage 1 envelope | Fine base synthesis, then separate pre/eroded elevations |
| Temperature/precipitation | Coarse Stage 2 global climate | Transferred and locally adjusted on fine surfaces |
| Wind | Coarse Stage 2 | Consumed for local feedback/presentation; no full fine circulation solve |
| Hydrology | No normal pre-fine authority | Derived independently for pre- and post-erosion fine surfaces |
| Semantic/render data | Derived from the active view | Rebuilt or selected when stage/view changes |

## Invalidation and reuse

- Changing Stage-1 inputs requires rebuilding all later stages.
- Changing atmosphere inputs requires rebuilding Stage 2 and later stages.
- Changing fine density or substrate inputs changes the fine-base cache identity
  and requires Stage 3/4 regeneration.
- Changing erosion-only parameters can reuse `FineBase` and rerun Stage 4.
- Changing the lake climate ratio mutates hydrologic water levels on the active
  fine surface and rebuilds presentation buffers; it is not a display-only knob.
- Changing relief scale, river stroke width, lighting, camera or render mode
  changes presentation only.

Cache correctness depends on the fine-base key and manual cache-format/version
discipline. A cache hit is part of the reproducible state and should be recorded
in future evidence metadata.

## Current limits and naming debt

- Stage 3 is called Hydrosphere but also owns adaptive refinement and substrate
  synthesis; the name is a UI milestone, not full ownership documentation.
- Stage 4 is called Erosion but includes climate adjustment and final hydrology.
- `generate_all` and `generate_hydrology` are broader/narrower than their actual
  effects suggest.
- `World.hydrology` is legacy-compatible state beside fine-surface hydrology.
- Optional stage fields and `expect` calls enforce order at runtime rather than
  in types.

These names may be improved in code later. Canonical documentation describes
their present behavior rather than silently inventing a cleaner API.

## Beyond Stage 4

No Stage 5 or later number is currently canonical. Water Geography V0 is a
completed enabling slice over current climate/hydrology/semantic state; Living
Surface V0 is the leading but not-yet-authorized first expansion. Other
candidates include cryosphere, persistent sediment and soils, resources, and eventually
culture/civilization. Their dependency shape is likely a graph rather than a
simple numbered sequence:

```text
terrain + climate + water
  -> sediment/soil ---------> vegetation/ecology
  -> cryosphere ------------> water/erosion feedback
  -> biome constraints -----> vegetation/resources
  -> semantic regions ------> settlement/culture/civilization
  -> all of the above ------> cartographic/game presentation
```

This diagram identifies plausible dependencies, not committed implementations.
The [cross-system disposition](system-disposition.md) supplies the current
ordering; stage numbers wait until a candidate demonstrates whether it needs
persistent simulated state, can be derived semantically, feeds back into
existing systems and produces enough emergence or visible value to justify its
cost.
