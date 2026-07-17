# Gap and Pareto analysis

This catalogue supplies candidates to the [model strategy](model-strategy.md)
and [roadmap](roadmap.md). It is not a queue. Missing systems compete with
repairing, simplifying or removing implemented systems. The current selection
and ordering live in the [cross-system disposition](system-disposition.md).

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

The completed disposition pass selected **Water Geography V0** as the immediate
enabling slice; that slice now passes. **Living Surface V0** also passes its
bounded proof and is retained as an on-demand semantic/presentation layer.
The current portfolio selects product/laboratory separation as immediate
enabling work and Consequential Geography V0 as the next bounded expansion.
Persistent sediment remains the leading large physical candidate behind its own
design gate.

| Candidate | Visible payoff | Emergence / reach | Cost / readiness | Provisional Pareto view |
|---|---|---|---|---|
| Product/research boundary and presentation allocation | Indirect visual payoff; large iteration and comprehensibility gain | Very high across every future change | Medium; research code is about 54k LOC and stage-local river textures alone are ~128 MiB each | **Immediate enabling work** |
| Landform semantics: ranges, plateaus, ridges, valleys, divides and passes | High through inspection and feature-aware rendering | Very high across causal validation, cartography, ecology and later culture | Medium; the completed packet is reusable evidence but product definitions remain incomplete | **Retain evidence; continue only for a consumer** |
| Regional mountain organization owner | Very high; targets tableland grammar and weak drainage hierarchy | Very high across terrain, water, climate barriers, semantics and later worlds | High; bounded H/C/G/F/I discriminators found no winning owner | **Highest-priority rework; bound one product-native structural slice** |
| Water-geography truth contract and object integration | High for lakes, rivers, coasts, inspection and cartography | Very high across climate validation, hydrology, ecology and later settlement | Low–medium; retained inputs, compact graph and exact diagnostic geometry now exist | **Completed enabling slice** |
| Controlled correspondence batteries for current systems | Indirect visually; prevents expensive false confidence | Very high across tectonics, climate, hydrology, erosion and remapping | Low–medium; idealized cases and ablations | **Use inside bounded decisions** |
| Presentation transform ledger plus matched physical/cartographic views | High clarity and prevents false geometric inference | High diagnostic and rendering leverage | Low; physical/cartographic split already exists | **Immediate** |
| Explicit semantic objects and regions | High through inspection, cartography and later gameplay | Very high; reused by rendering, validation, ecology and civilization | Low–medium; much logic already exists implicitly | **Derive when a route/site/presentation consumer needs them** |
| Presentation profiles, legends and capture metadata | High clarity; moderate spectacle | High diagnostic and decision value | Low; ready now | **Do early** |
| Consistent scale-dependent rivers/overlays | High at globe/map scales | Medium; improves all thematic views | Medium; semantic generalization needed first | **Early** |
| Elevation/unit/scale contract | Indirect visually, critical diagnostically | Very high across terrain, erosion and rendering | Low–medium; requires careful normalization audit | **Do early** |
| Unified experiment/config provenance | Indirect but prevents invalid decisions | Very high across all development | Low–medium | **Do early** |
| Biome constraints and ecological regions | Very high recognizability and world identity | High; consumes climate, water, elevation and future soils/resources | Medium; diagnostic prototype exposes unvalidated inputs, not an ecology stage | **Still unselected; require a consumer and stronger causal grammar beyond the accepted fractional layer** |
| Vegetation coverage and scale-aware forest rendering | Very high spectacle and board/globe richness | Medium–high; feeds ecology, resources and settlement | Medium; split state from tree rendering | **Fractional Living Surface cover accepted; forest rendering remains unselected and gated** |
| Persistent sediment budget v0 | High potential at rivers, basins, coasts and range surroundings | Very high coupling across erosion, lithology, flexure and hydrology | Medium–high; time/material ownership needed | **Research then bounded prototype** |
| Soil/moisture substrate | Moderate directly, high through vegetation/agriculture | High for ecology and civilization | Medium; depends partly on sediment/weathering choices | **After a concrete ecology, sediment or human-geography consumer clarifies need** |
| Same-clock tectonics and erosion | Potentially high terrain authenticity | Very high if stable; connects mountain age, uplift and denudation | High; tectonic carrier operators currently fail convergence | **Research gate, not product rewrite yet** |
| Force-derived plate motion | Low direct; potentially broad tectonic coherence | Medium–high | High uncertainty; current kinematics already useful | **Hold pending causal payoff study** |
| Ocean currents and heat transport | Moderate/high regional climate/coast payoff | High for climate, ecology and ice | High; global dynamic solve not ready or necessary | **Research authentic shortcut** |
| Coastal processes and deltas | High local/regional visual payoff | High with sediment, rivers and sea level | Medium after sediment state | **Sequence behind sediment v0** |
| Glaciers/cryosphere | High in suitable worlds | Medium–high with water, climate, erosion and sea level | Medium–high; current glacial pass is incomplete | **Later coherent subsystem** |
| Groundwater/wetlands | Moderate visual/ecological payoff | High locally for rivers, lakes and biomes | Medium; hydrology semantics ready in part | **Targeted later addition** |
| Dynamic seasons/weather | High animation/variety | Medium geographic reach unless coupled over time | Very high for full solve; cheaper presentation/authentic hacks possible | **Hold; research only if product view needs it** |
| Named relative opportunity components | Moderate directly; high board/game meaning | Very high bridge from planet state to aggregate sites | Low–medium with current water, terrain and Living Surface inputs | **Selected inside Consequential Geography V0; no soil/resource claims** |
| Aggregate sites and routes | Very high narrative and board appeal | Very high emergent meaning and a functional test of existing geography | Medium if aggregate; enormous if agent-heavy | **Selected bounded expansion; no population/agents/economy/culture** |
| Full civilization/economic simulation | Potentially very high but changes project center of gravity | Very high | Extremely high and weakly bounded | **Distant option, not present roadmap** |
| Physically based material renderer | Moderate visual improvement | Low model reach | High compared with illustrative lighting at globe scale | **Hold** |
| Dynamic 3-D atmosphere/ocean | Scientifically rich, uncertain product payoff | High in principle | Extremely high compute/validation cost | **Reject as default direction absent a new need** |

## Consumer-driven semantic layer

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

Semantic work should reuse existing derivations rather than begin with a generic
entity framework. Rivers and water bodies are already useful objects. New
regions, passes, crossings and barriers should be pulled into existence by a
route, settlement, label or inspection consumer.

These objects supplied evidence for the completed landscape-owner discriminator.
Semantics alone can improve explanation and presentation, but it cannot repair
an unorganized physical surface. Water objects now provide the most useful
semantic integration boundary because they also test climate and hydrology.

## Completed bounded visible domain: living surface

Biomes and vegetation can produce a large visual and semantic jump using current
temperature, precipitation, elevation, continentality, water and terrain state.
The bounded design review selects equilibrium physiognomy and splits the
problem into layers:

1. ecological constraints and limiting factors;
2. fractional bare/herbaceous/woody/wet coverage;
3. optional semantic regions and later disturbance state;
4. scale-dependent presentation—from globe color/texture to regional canopy and
   optional local tree instances.

The accepted version does not simulate individual trees or succession. Its
on-demand physiognomic fractions arise from coherent inputs and can support
later affordances without becoming a retained ecology stage. Named biomes,
regions and richer vegetation remain separate decisions. See the
[V0 decision](living-surface.md).

The first corpus shows provisional biome transition coverage ranging from about
14% to 40% across seeds. This does not diagnose climate quality, but it confirms
that calibration should not advance before climate controls and landform
semantics are evaluated.

Risks include producing a climate lookup-table painted over terrain, circular
wetland halos around map-selected rivers, allowing noisy cell classifications,
or treating rendered trees as ecological state.

## Selected missing consequence layer

The [Consequential Geography V0 decision](consequential-geography.md) uses
current terrain, water/coast identity and Living Surface opportunity to derive
traversability, named relative opportunity components, aggregate sites and
least-cost routes. It is selected because it
directly advances the globe/board identity while testing several retained
systems together.

It must remain an authentic aggregate hack. Independent ablations of grade,
freshwater, coast and living opportunity should move sites and routes in
intelligible ways. If authored weights dominate, routes exploit mesh artifacts,
or every site merely follows the largest river, stop rather than add economy or
cultural simulation.

The first same-site route panel now shows a cheap, material terrain consequence
in corridor geometry but no endpoint-topology change. This closes the missing
traversability-to-route seam without closing the broader consequence layer:
the aggregate site prior remains provisional, and gaps, crossings and
chokepoints do not yet have honest route-local explanations.

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

Targeted external research should now compare:

1. aggregate settlement-site and route-network models used in procedural games,
   spatial history, transport geography and strategy maps;
2. least-cost and network-formation methods over irregular terrain, including
   harbor opportunities, crossings, route-local gaps and chokepoints;
3. structural range generators and reduced tectonic/landscape models that avoid
   universal smooth tablelands without requiring full geodynamics;
4. reduced sediment-routing models that preserve mass and generate one useful
   floodplain, terminal-fill or delta target without full stratigraphy;
5. cheap seasonal hydroecology and authentic ocean heat/current shortcuts; and
6. multiscale vegetation and feature-generalization methods only when a selected
   consumer needs them.

Research should return mechanism, payoff, compute, failure modes and a proposed
Hex3-scale approximation—not a survey of maximum-fidelity simulations.
