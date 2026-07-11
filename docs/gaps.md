# Gap and Pareto analysis

This document compares missing capabilities, weak couplings and possible
fundamental reworks. It asks what would most improve coherent emergence, visual
appeal, semantic richness and future extensibility per unit of compute and
complexity.

It is not a promise to implement every plausible planet subsystem. Current
architecture remains open to replacement, and missing systems compete with
reworking systems that already exist.

## Evaluation dimensions

Candidates are compared qualitatively on:

- **Visible payoff** — how directly the result changes what people see;
- **Emergence** — whether interactions produce varied outcomes difficult to
  author directly;
- **Downstream reach** — how many later systems consume the output;
- **Authenticity gain** — whether it replaces arbitrary state with coherent
  causes or consequences;
- **Readiness** — whether current inputs, units and scale are adequate;
- **Runtime cost** — compute and memory at intended resolution;
- **Architecture cost** — state, invalidation, parameters and ownership burden;
- **Evidence confidence** — how well the mechanism/payoff is understood.

“High value” does not mean “build a scientifically complete version.” The
preferred scope is the cheapest model that preserves the important causal and
visible consequences.

## Comparative matrix

| Candidate | Visible payoff | Emergence / reach | Cost / readiness | Provisional Pareto view |
|---|---|---|---|---|
| Explicit semantic objects and regions | High through inspection, cartography and later gameplay | Very high; reused by rendering, validation, ecology and civilization | Low–medium; much logic already exists implicitly | **Do early** |
| Presentation profiles, legends and capture metadata | High clarity; moderate spectacle | High diagnostic and decision value | Low; ready now | **Do early** |
| Consistent scale-dependent rivers/overlays | High at globe/map scales | Medium; improves all thematic views | Medium; semantic generalization needed first | **Early** |
| Elevation/unit/scale contract | Indirect visually, critical diagnostically | Very high across terrain, erosion and rendering | Low–medium; requires careful normalization audit | **Do early** |
| Unified experiment/config provenance | Indirect but prevents invalid decisions | Very high across all development | Low–medium | **Do early** |
| Biome constraints and ecological regions | Very high recognizability and world identity | High; consumes climate, water, elevation and future soils/resources | Medium; current inputs are usable but incomplete | **Early prototype** |
| Vegetation coverage and scale-aware forest rendering | Very high spectacle and board/globe richness | Medium–high; feeds ecology, resources and settlement | Medium; split state from tree rendering | **Early after biome semantics** |
| Persistent sediment budget v0 | High potential at rivers, basins, coasts and range surroundings | Very high coupling across erosion, lithology, flexure and hydrology | Medium–high; time/material ownership needed | **Research then bounded prototype** |
| Soil/moisture substrate | Moderate directly, high through vegetation/agriculture | High for ecology and civilization | Medium; depends partly on sediment/weathering choices | **After biome prototype clarifies need** |
| Same-clock tectonics and erosion | Potentially high terrain authenticity | Very high if stable; connects mountain age, uplift and denudation | High; tectonic carrier operators currently fail convergence | **Research gate, not product rewrite yet** |
| Force-derived plate motion | Low direct; potentially broad tectonic coherence | Medium–high | High uncertainty; current kinematics already useful | **Hold pending causal payoff study** |
| Ocean currents and heat transport | Moderate/high regional climate/coast payoff | High for climate, ecology and ice | High; global dynamic solve not ready or necessary | **Research authentic shortcut** |
| Coastal processes and deltas | High local/regional visual payoff | High with sediment, rivers and sea level | Medium after sediment state | **Sequence behind sediment v0** |
| Glaciers/cryosphere | High in suitable worlds | Medium–high with water, climate, erosion and sea level | Medium–high; current glacial pass is incomplete | **Later coherent subsystem** |
| Groundwater/wetlands | Moderate visual/ecological payoff | High locally for rivers, lakes and biomes | Medium; hydrology semantics ready in part | **Targeted later addition** |
| Dynamic seasons/weather | High animation/variety | Medium geographic reach unless coupled over time | Very high for full solve; cheaper presentation/authentic hacks possible | **Hold; research only if product view needs it** |
| Resources and suitability fields | Moderate directly; high board/game meaning | Very high bridge to settlement/civilization | Low–medium once geology/ecology semantics exist | **Prepare interfaces, add later** |
| Settlement, routes and cultural diffusion | Very high narrative and board appeal | Very high emergent meaning from geography | Medium if aggregate; enormous if agent-heavy | **Later bounded world-history layer** |
| Full civilization/economic simulation | Potentially very high but changes project center of gravity | Very high | Extremely high and weakly bounded | **Distant option, not present roadmap** |
| Physically based material renderer | Moderate visual improvement | Low model reach | High compared with illustrative lighting at globe scale | **Hold** |
| Dynamic 3-D atmosphere/ocean | Scientifically rich, uncertain product payoff | High in principle | Extremely high compute/validation cost | **Reject as default direction absent a new need** |

## Highest-leverage missing layer: semantics

Hex3 already generates enough state to derive reusable objects: water bodies,
river networks, mountain systems, tectonic belts and climate regions. Today
these meanings are distributed across diagnostics, thresholds, coloring and
buffer generation.

An explicit semantic layer is a force multiplier because it can provide:

- scale-dependent cartography without changing physical state;
- object inspection and causal explanation;
- reusable object-level validation;
- stable inputs for biomes, resources and settlement;
- named/typed features for a board or game view;
- a clean boundary between generated fields and visual heuristics.

The first semantic work should reuse existing derivations rather than begin with
a generic entity framework. Rivers/water bodies and ranges are the best initial
objects because their topology and diagnostics already exist.

## Highest-leverage new visible domain: living surface

Biomes and vegetation can produce a large visual and semantic jump using current
temperature, precipitation, elevation, continentality, water and terrain state.
The problem should be split into layers:

1. ecological constraints and limiting factors;
2. semantic biome/vegetation regions;
3. coverage, density and disturbance state;
4. scale-dependent presentation—from globe color/texture to regional canopy and
   optional local tree instances.

The first version should not simulate individual trees or full succession. It
should make forests, grasslands, deserts, wetlands and alpine/polar zones arise
from coherent inputs, expose why a region received its state, and preserve room
for later soils, fire, disturbance or human land use.

Risks include producing a climate lookup-table painted over terrain, allowing
noisy cell classifications, or treating rendered trees as ecological state.

## Highest-leverage large coupling: sediment

Persistent sediment could connect many systems that currently stop at ledgers:

```text
erosion source
  -> hillslope/channel transport
  -> floodplain/terminal basin/coastal deposition
  -> sediment load and basin fill
  -> soil/lithology/coast/flexure consequences
```

A Pareto version should not begin with stratigraphic layers and detailed grain
classes. A bounded v0 might track one conserved mobile sediment quantity on the
fine graph, move it through the existing drainage network, deposit by transport
capacity/base-level opportunity, and retain accumulated thickness for later
consumers. It must demonstrate visible basin/coastal consequences and stable
mass behavior before gaining more geology.

This candidate depends on a clearer time/units contract and may require careful
memory design at millions of cells.

## Fundamental rework candidates

### Geological time and tectonic evolution

The desirable outcome is not “more physical tectonics” in isolation. It is
terrain whose width, elevation, drainage and decay reflect forcing history on a
shared clock. Current carrier/lifecycle work proves useful causal ideas but
fails spatial convergence and product morphology gates.

Do not rebuild the product around it until boundary forcing and material
projection converge. Research should isolate operators and test whether a
coarser, more aggregate history representation can deliver the important
consequences more cheaply.

### Stage/state architecture

Optional fields and runtime `expect` calls work but make invalidation, future
branches and feedback harder to reason about. New ecology/sediment/civilization
domains will stress a single linear stage integer.

Potential rework: an explicit dependency graph of immutable products with
versioned inputs and retained snapshots. This could improve caching and
experimentation, but it should follow concrete dependency pressure rather than
precede it as framework work.

### Rendering architecture

Two surface paths and view-specific overlays create capability drift. The
desired rework is shared semantic/generalization inputs plus presentation
profiles, not necessarily one universal shader. Migrate incrementally around
rivers, legends and capture metadata before considering a broad renderer reset.

### Physical units

Motion has explicit physical rates while elevation, temperature, precipitation
and erosion chronology mix normalized and physical interpretations. A unit
audit may expose fundamental model changes. That is acceptable: coherent units
and conversion boundaries matter more than preserving current parameter names.

## Important missing couplings

- tectonic loading/unloading ↔ erosion, sediment and flexure;
- sediment ↔ basins, coastlines, lithology and soils;
- surface water ↔ climate beyond local lake humidity;
- climate/soil/water ↔ vegetation and fire/disturbance;
- ice/snow ↔ water, erosion, loading and sea level;
- ocean heat/current structure ↔ regional climate and ice;
- ecology/resources/terrain ↔ settlement and routes;
- semantic objects ↔ validation, cartography and interaction.

Coupling is not automatically good. Feedback should be added only when it
produces stable, interpretable behavior and meaningful consequences.

## Research questions

Before major implementation, targeted external research should compare:

1. simplified biome and dynamic-vegetation models used in Earth-system science,
   strategy games and procedural graphics;
2. multiscale vegetation rendering and generalization from globe coverage to
   regional/individual trees;
3. reduced sediment-routing and landscape-evolution models that preserve mass
   and generate floodplains/deltas without full stratigraphy;
4. authentic ocean heat/current shortcuts suitable for spherical procedural
   worlds;
5. aggregate settlement, route and cultural-diffusion models grounded in
   terrain, water, ecology and resources;
6. feature extraction/generalization techniques for river networks, ranges,
   regions and labels;
7. dependency-graph and provenance approaches for expensive procedural
   pipelines with retained stage variants.

Research should return mechanism, payoff, compute, failure modes and a proposed
Hex3-scale approximation—not a survey of maximum-fidelity simulations.

