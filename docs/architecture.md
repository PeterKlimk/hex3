# Current architecture

This document describes the accepted Hex3 architecture as implemented. It
focuses on ownership and data flow; model fidelity and alternatives will live
in the [system assessments](system-assessments.md) and
[experiment registry](experiment-registry.md). See the [project thesis](thesis.md),
[documentation policy](documentation-policy.md), and detailed
[pipeline](pipeline.md).

This is a current-state description, not a preservation mandate. Systems and
even the coarse/fine, staged, or model/semantic/presentation boundaries may be
reworked when criticism and evidence support a better architecture.

## Architectural shape

Hex3 has three conceptual layers and two main spatial scales.

```text
WORLD MODEL
  coarse planet graph
    geometry -> tectonics -> global terrain -> atmosphere
                                      |
                                      v
  adaptive fine graph
    transferred fields -> substrate -> hydrology -> erosion -> hydrology
                                      |
                                      v
SEMANTIC MODEL
  materials, water state, river hierarchy, feature fields, overlays
                                      |
                                      v
PRESENTATION MODEL
  globe/map projection, relief, strokes, color, lighting, particles
```

The world layer is causal state. The semantic layer derives meaning and
selection from that state. The presentation layer makes it legible and
attractive. Some semantic derivations currently live inside rendering code;
making their ownership explicit is an architectural direction, not a claim that
they are already centralized.

## World root and state ownership

`World` in `src/world/mod.rs` is the aggregate root. It owns the reproducible
seed, coarse tessellation, optional stage products, model selections and
fine-stage parameters. Stage products are optional because the interactive app
constructs the world progressively.

`World::manifest` derives a serializable effective-run record from that retained
state. It includes source revision/dirty state, backend, generation parameters,
computed/viewed stage and fine-cache identity/outcome. World export embeds the
same record and diagnostic tools print its compact summary, preventing each tool
from inventing a different provenance header.

The [elevation and unit contract](units.md) defines the conversion boundaries
used by that manifest and distinguishes model elevation, reference crust-column
thickness, native simulation slopes and physical grades.

The coarse world owns:

- `Tessellation`: spherical Voronoi geometry, compact adjacency and cell area;
- `Plates`: motion-unit assignment;
- `Crust`: independent continental/oceanic crust and craton structure;
- `Dynamics`: Euler-pole motion;
- `TectonicHistory` and `FeatureFields`: boundary history and tectonic response;
- `Elevation`: coarse terrain/bathymetry;
- `Atmosphere`: temperature, pressure, wind, uplift and precipitation.
- `EcologySemantics`: derived seasonless ecological constraints and broad biome
  labels; interpretive and replaceable, not physical pipeline state.

`FineWorld` owns the adaptive tessellation, transferred fields and two retained
surfaces:

- `pre`: the synthesized fine base with pre-erosion hydrology;
- `eroded`: the optional eroded surface with re-derived climate adjustments and
  hydrology.

The legacy top-level `World.hydrology` remains as a fallback, but normal current
generation stores fine hydrology on these surfaces.

## Geometry and scale ownership

Global systems run on a coarse spherical Voronoi graph. This keeps plate,
boundary, feature and atmospheric work tractable and gives every system shared
areas and adjacency.

Stage 3 constructs a separate adaptive fine graph. Density is allocated using
coarse terrain, climate and tectonic signals so expensive resolution is spent
where drainage, erosion or appearance benefits. Smooth coarse fields are
transferred; fine substrate synthesis supplies unresolved structure; hydrology
and erosion then operate on fine geometry.

This is not transparent supersampling. Algorithms on the fine graph must use
physical area and distance rather than assume uniform cells, and architecture
must say whether a field is coarse-authoritative, transferred, synthesized, or
re-derived. The fine-base disk cache is part of this boundary: it makes erosion
iteration practical, but cached construction is authoritative and requires
explicit version/provenance management.

## Lithosphere and terrain

Plates and crust are generated independently, allowing motion units to contain
mixed crust and passive margins. Euler rotations supply per-cell velocity.
Shared plate edges are classified from relative motion and crust interaction.
History, boundary rates and response profiles produce tectonic feature fields;
elevation combines crustal/bathymetric structure with the selected orogen model.

The default product path remains the legacy orogen model. Numerous conserved,
history-aware, thin-sheet and moving-carrier alternatives are implemented for
evaluation. They are architecture experiments, not simultaneous features of a
normal world and not accepted product behavior merely because their code exists.

At fine scale, the coarse envelope and fields guide adaptive sampling and
substrate construction. Fluvial incision, hillslope diffusion, uplift and
limited deposition evolve the surface. Hydrology is recomputed on the result.

## Atmosphere and water

The atmosphere is a steady, simplified global model. Temperature derives from
latitude, continentality and elevation. An analytic circulation supplies
surface/upper winds and vertical motion. A finite-volume moisture transport
model generates precipitation from winds, capacity, evaporation and rainout.

Fine erosion uses transferred climate plus bounded local feedbacks such as
lapse correction and optional orographic, lee-side and lake effects. It does
not rerun a global dynamic climate after every surface change.

Hydrology identifies ocean, resolves drainage and depressions, accumulates
flow, constructs basin/overflow relationships, and derives equilibrium lakes
and rivers. Pre-erosion hydrology supplies routing and base levels to erosion;
post-erosion hydrology describes the final retained surface.

Drainage integration is a declared terrain-repair operator inside hydrology,
not erosion. Its sparse cut/source provenance is retained so water gaps and
river paths can disclose when this authentic hack changed their terrain.

## Semantic derivation

There is not yet one semantic-model module. Its responsibilities are distributed
across world, app and diagnostic code:

- water states and basins identify ocean, lake and land;
- catchment thresholds and outlet tracing select visible river networks;
- material derivation identifies land/water/snow-like rendering classes;
- feature fields and boundary aggregation identify tectonic regimes;
- coloring derives thematic interpretations of elevation, climate and flow;
- ecological potentials derive inspectable living-world constraints without
  yet owning vegetation placement or presentation;
- diagnostics derive connected objects, ranges and structural measurements;
- visualization derives arrows, pole markers and boundary colors.

The first extractions are now implemented in `world::semantics` and
`world::ecology`: shared [semantic objects](semantics.md) define per-stage water
identity, river selection/hierarchy and provisional ecological potentials.
Rendering consumes lightweight river selection; audits consume the same water,
river and ecology definitions. Other semantic responsibilities remain
distributed.

These operations should be documented as interpretation rather than physical
state or raw drawing. A future explicit semantic layer can support consistent
scale-dependent cartography, legends, object inspection and game-facing regions.

## Application and presentation

`src/app` owns progressive generation, view state, CPU-side color/material
derivation, GPU buffer construction, overlays, export and sweep orchestration.
`src/render` owns wgpu context, pipelines, camera, buffers, elevation cubemap,
renderer and wind-particle implementation. WGSL programs live under
`src/shaders`.

The intended ownership and scale policy for derived features and visual
communication is specified in the
[semantic and presentation architecture](semantic-presentation.md).

Two surface paths currently coexist:

- a unified material/elevation mesh for Relief and wind views, supporting
  radial displacement, water materials and a draped river texture;
- a CPU-colored Voronoi mesh for most thematic modes.

Additional line/marker buffers render cell edges, plate diagnostics and rivers
outside the unified path. This split is functional but causes capability drift
between modes and Globe/Map views. It should be treated as presentation debt,
not as multiple world truths.

Relief scale and river width are renderer-only. They do not change elevation,
hydrology, validation or later generation. The renderer may exaggerate them
cartographically under the [presentation contract](presentation.md).

## Validation and tooling

Validation spans several forms:

- geometry topology, containment and area checks;
- deterministic and resolution-aware unit tests;
- physical-time conversion and subdivision checks;
- conservation/material ledgers for experimental tectonics;
- hydrology, roughness and structural diagnostics;
- command-line scorecards and exported world analysis;
- controlled offscreen render sweeps and human visual judgment.

These provide strong internal evidence in places but do not make the full planet
a predictive scientific simulation. Reproducibility metadata and evidence types
must remain explicit.

## Known architectural boundaries

- Stage dependencies are enforced through optional fields and runtime
  precondition checks, not a typed state machine.
- `generate_all` currently stops after Stage 1 despite its broad name.
- Computed and viewed stage are separate; rendering uses `active_*` accessors.
- Fine graph construction and caching have their own determinism contract.
- Physical units are uneven: motion uses explicit km/Myr while several terrain,
  temperature and precipitation quantities remain normalized.
- Persistent sediment, dynamic tectonic/surface feedback, global water inventory,
  ecology and biomes are absent or partial rather than hidden stages.
- Experimental code must not be presented as the product architecture.

## Implemented frontier and future domains

The implemented world pipeline currently ends with an eroded surface and final
hydrology. That is a development frontier, not the intended conceptual endpoint
of Hex3.

Plausible later domains include:

- ocean circulation, currents, sea ice and coastal processes;
- glaciers, snowpack, sediment, soils and persistent surface materials;
- biomes and ecological constraints;
- vegetation structure, including forests and visible trees at appropriate
  scales;
- resources, disturbance and long-timescale landscape/ecology feedback;
- culture, settlement, infrastructure and civilization-scale systems;
- semantic regions and cartographic generalization for board/game views.

These do not yet have an approved order or fidelity. Some may be semantic
derivations, some simulations, some authentic hacks, and some presentation-only
systems. Several may share inputs and run in parallel rather than form one
linear Stage 5–N chain. The roadmap should compare them by prerequisite
readiness, downstream reach, emergence, visible payoff and compute cost.
