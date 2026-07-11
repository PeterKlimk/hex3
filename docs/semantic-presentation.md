# Semantic and presentation architecture

Hex3 should show a coherent world without requiring physically tiny features to
be literally visible at globe scale. This document defines the architecture
between generated state and pixels: interpretation, generalization, styling and
rendering.

See the [project thesis](thesis.md), [current architecture](architecture.md), and
[presentation contract](presentation.md).

## Layer contract

```text
WORLD STATE
  elevation, crust, motion, climate, water, flow, material, history
      |
      | deterministic derivation; no visual settings
      v
SEMANTIC STATE
  named/typed features, hierarchy, importance, regions, relationships
      |
      | view/profile/scale-dependent selection and generalization
      v
CARTOGRAPHIC SCENE
  visible objects, displaced surfaces, strokes, symbols, labels, materials
      |
      | GPU rendering and compositing
      v
IMAGE
```

The world model answers **what exists and why**. The semantic model answers
**what it means and what is important**. The cartographic scene answers **what
should be visible here**. Rendering answers **how those choices become pixels**.

Only an explicit modeled feedback may move information upward into world state.
Relief scale, line width, camera distance, color or symbol visibility must never
change terrain, hydrology or acceptance metrics.

## Semantic state

Semantic state is derived from world fields but is not merely a color lookup.
It should eventually contain stable, inspectable objects and relationships such
as:

- oceans, seas, lakes, wetlands and drainage basins;
- river networks with sources, trunks, tributaries, mouths and hierarchy;
- mountain ranges, plateaus, passes, valleys and drainage divides;
- active/passive margins, trenches, arcs, ridges, rifts and collision belts;
- climate regions, snow/ice zones, biomes and vegetation regions;
- coasts, islands, shelves and sedimentary basins;
- later, resources, routes, settlements, cultural regions and political areas.

A semantic object should, where applicable, record:

- stable identity for the generated world/stage;
- type and membership/geometry;
- physical measurements and units;
- parent/child or network relationships;
- modeled provenance—the fields or events that caused it;
- importance at multiple map scales;
- confidence or ambiguity when classification is heuristic;
- invalidation dependencies.

Semantic state must remain presentation-independent. A river can have a
modeled discharge/catchment, semantic rank and rendered width; these are three
different values with different owners.

## Existing implicit semantic layer

Current code already performs semantic work in several places:

- hydrology classifies land, ocean, lake, basin and river structure;
- river preparation selects catchment-qualified networks and major outlets;
- coloring derives terrain/water/material-like classes;
- boundary aggregation and visualization identify tectonic interaction types;
- diagnostics identify ranges, components and river/lake objects;
- application state selects active stage and visible layer.

This is useful but distributed across `src/world`, `src/app/coloring.rs`,
`src/app/world.rs`, `src/app/visualization.rs` and diagnostic binaries. The
future semantic layer should centralize reusable meaning, not centralize every
rendering calculation. Shader-specific antialiasing, lighting and pixel widths
remain presentation concerns.

## Scale hierarchy

One semantic object may need several representations:

| Scale | Primary questions | Typical representation |
|---|---|---|
| Planet | What organizes the whole world? | plates, continents, major ranges, climate belts, major rivers |
| Regional | What defines this region? | basins, tributary systems, passes, coasts, biomes, settlements |
| Local | What occupies and shapes this place? | valleys, channels, vegetation cover, individual terrain structures |
| Detail | What creates surface richness? | trees, rocks, ripples, small channels, material texture |

More detail is not always more truthful. Planet views need selection,
aggregation and exaggeration; rendering every fine cell or tree would be both
illegible and expensive. Conversely, a local view should not stretch one
planet-scale symbol into literal geometry.

Generalization may:

- suppress low-importance objects;
- merge related objects into a parent region/network;
- simplify geometry while preserving topology and character;
- widen, separate or symbolically displace features for legibility;
- replace geometry with a marker, texture or statistical coverage;
- change labels and annotation density.

Generalization must preserve declared invariants. River simplification should
not reverse flow or disconnect a mouth; coast simplification should not invent
major islands; range aggregation should preserve principal orientation and
extent. These are semantic/cartographic validation questions, not terrain-model
questions.

## Presentation profiles

A presentation profile bundles choices that should otherwise not be inferred
from scattered toggles.

### Physical inspection

- true-scale or explicitly stated 1× geometry;
- physical units, datum, scale and vertical-exaggeration readout;
- minimal illustrative styling;
- optional slope-angle, contour and uncertainty overlays;
- intended for diagnosing world state, not product spectacle.

### Diagnostic

- thematic fields with legends, units/normalization and range disclosure;
- selectable stage and model provenance;
- topology, object boundaries, vectors and numeric inspection;
- fixed cameras/settings for reproducible comparisons.

### Cartographic

- legibility-oriented relief and screen-space river treatment;
- scale-dependent feature selection and generalized geometry;
- illustrative materials, lighting and labels;
- the intended default globe/board interpretation.

The current code/UI preset named `Authentic` supplies the cartographic relief
scale, but a complete profile would also own river policy, lighting, overlays,
labels and generalization.

### Dramatic/showcase

- consciously stronger atmosphere, lighting, animation or exaggeration;
- designed for discovery, screenshots and “wow” value;
- still records distortions and never changes modeled acceptance.

Profiles may share a renderer. They differ in communication contract, not in
which world is loaded.

## Surface and feature ownership

| Concern | World owner | Semantic owner | Presentation owner |
|---|---|---|---|
| Mountains | elevation, crust, erosion | range/plateau/pass objects and importance | relief displacement, hillshade, snow/rock styling |
| Rivers | drainage and flow | network hierarchy, major status, named reach | stroke width, opacity, color, antialiasing |
| Lakes/ocean | basin/water level and ocean state | water-body identity/type/importance | flattened water surface, color, glint, shoreline treatment |
| Climate | temperature, wind, precipitation | climate/biome region classification | thematic palette, particles, symbols |
| Plates | assignments and Euler motion | boundary type, plate identity, important interactions | arrows, colors, pole markers, decluttering |
| Vegetation | future ecological/coverage state | forest/biome objects and density | canopy texture, instanced trees, scale transition |
| Civilization | future population/network state | settlement, route, cultural/political objects | symbols, labels, borders, board/game styling |

This table is a responsibility guide, not a requirement that every semantic
owner become a heavyweight stored database. Cheap deterministic derivations may
remain computed on demand if their definition is shared and testable.

## Current rendering architecture

Two primary surface paths coexist:

1. the unified material/elevation mesh used by Relief and wind views, with
   shader displacement, water materials and the draped river texture;
2. the CPU-colored Voronoi mesh used by most thematic views.

Line and marker paths add edges, plate overlays and non-unified rivers. Globe
and Map projections support different subsets. This creates presentation-path
behavior rather than profile-driven behavior: lighting, river availability and
surface semantics can change with the selected pipeline.

This is accepted current debt, not desired long-term ownership. A rework should
seek shared semantic inputs and declared profile behavior; it need not force all
modes through one shader if multiple paths remain cheaper and clearer.

## Current visual derivations

The following are presentation or semantic heuristics, not modeled materials:

- fixed elevation snow styling;
- slope-derived exposed rock;
- flow-based green-valley moisture styling;
- nonlinear optical-looking water palettes;
- river Fresnel/glint and screen-space width;
- fixed-size velocity arrows and Euler-pole markers;
- illustrative hemisphere lighting.

They are legitimate while labeled. If a future climate, vegetation or material
system replaces one, the old heuristic should be removed or retained only as an
explicit style—not silently stacked with its physical replacement.

## Presentation data and reproducibility

A reproducible visual should identify:

- seed and model/configuration;
- computed and viewed stages;
- view mode and thematic layer;
- presentation profile and individual overrides;
- relief scale/preset and river selection/width;
- lighting mode;
- camera/projection, viewport and output resolution;
- revision and cache provenance.

World export currently records substantial model state but not enough camera or
presentation metadata to reconstruct an image. Capture metadata should become a
first-class sidecar or export section.

## Pareto priorities

1. Extract shared semantic definitions for rivers, water bodies, ranges and
   materials before building a universal renderer.
2. Add quantitative legends, scale and exaggeration disclosure to physical and
   diagnostic profiles.
3. Make river and overlay availability consistent across Globe/Map and surface
   paths through declared generalization policy.
4. Record complete capture metadata.
5. Use zoom-dependent selection before increasing fine rendering density.
6. Treat biomes, vegetation and later civilization as semantic systems first;
   decide simulation and rendering depth independently by scale and payoff.
7. Prefer illustrative lighting improvements over expensive physically based
   rendering until materials and viewing distance make that depth worthwhile.

