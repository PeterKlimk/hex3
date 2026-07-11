# Hex3 roadmap

This roadmap turns the [project thesis](thesis.md),
[system assessments](system-assessments.md), and [gap analysis](gaps.md) into an
ordered decision process. It is intentionally revisable: evidence may reorder
work or justify fundamental rework of any current system.

The roadmap optimizes for coherent emergence, visual appeal, explanatory depth,
iteration speed and “wow” value—not for completing a conventional list of
planet subsystems.

## Roadmap rules

1. Preserve one known product baseline while experiments are evaluated.
2. Fix ownership, units or convergence failures before tuning downstream systems
   around them.
3. Prefer shared state and couplings that benefit several consumers.
4. Separate world, semantic and presentation changes in implementation and
   evidence.
5. Use bounded vertical slices before committing to deep simulations.
6. Promotion requires the [validation policy](validation.md); implementation
   alone does not advance roadmap status.
7. Retire or park overlapping mechanisms when a new owner is promoted.

## Horizon 0: documentation and decision substrate

Status: **substantially complete; cleanup validation in progress**.

Purpose: establish a trustworthy account of the current system before choosing
large reworks or new stages.

Completed in this sprint:

- project thesis and fidelity vocabulary;
- code/document/render inventories;
- current architecture and stage pipeline;
- documentation authority/status policy;
- system fidelity/Pareto assessments;
- experiment registry;
- semantic/presentation architecture;
- validation and reproducibility policy;
- gap analysis and this roadmap.

Remaining:

- establish one current experiment/configuration manifest.

Completed after the initial roadmap draft:

- human-facing root `README.md`;
- minimal assistant policy linking canonical docs;
- archived/reclassified superseded specs, roadmaps, reviews, research and
  generated outputs;
- corrected the stale top-level world-stage module description and historical
  source-document paths.
- added a serializable effective-run manifest to world exports and diagnostic
  headers, including build revision/dirty state and fine-cache identity/outcome.

Exit gate: a new contributor can find the product path, current experiments,
validation rules and active roadmap without reading chronological specs.

## Horizon 1: observability and semantic foundation

Purpose: make the existing planet understandable, comparable and reusable
before increasing model depth.

### 1A. Units and evidence envelope

- **Completed:** document and test the end-to-end elevation
  datum/unit/slope/render conversion, with unit-contract metadata in exports;
- audit normalized versus physical temperature, precipitation, time and erosion
  quantities;
- extend the shared world manifest with presentation/camera capture metadata;
- add presentation/camera metadata to controlled captures;
- define fixed seed and resolution panels for product promotion.

Exit gate: numeric and visual results can be reproduced and cannot silently mix
physical and cartographic scales.

### 1B. First semantic objects

- **Completed:** extract shared water-body and river-network semantics from
  hydrology/render preparation/diagnostics;
- extract range/plateau/pass semantics from existing mountain diagnostics;
- define stable per-world-stage identities, measurements, provenance and
  importance;
- expose objects to diagnostics and presentation without changing world state.

Exit gate: the renderer and audits consume the same definition of a major river
or range, and objects can explain their modeled causes.

### 1C. Presentation profiles

- implement declared Physical, Diagnostic, Cartographic and Dramatic profiles;
- add legends, units, scale and vertical-exaggeration disclosure;
- make river/overlay policy consistent across Globe/Map and relevant modes;
- use scale-dependent generalization/decluttering;
- decide whether to integrate displaced-facet hillshade interactively;
- verify/remove stale rendering paths only as replacements become clear.

Exit gate: presentation choices are reproducible profile state, and no visual
mode is mistaken for physical evidence.

## Horizon 2: living-world vertical slice

Purpose: deliver the largest near-term expansion in world identity and visual
richness using existing terrain, climate and water.

### 2A. Biome/ecological constraint prototype

- derive limiting factors from temperature, precipitation, seasonless moisture
  availability, elevation, water proximity and terrain;
- produce continuous ecological potentials before categorical biome labels;
- classify stable semantic regions with uncertainty/transition zones;
- validate geographic coherence and control response across seeds;
- make deliberate approximations explicit where seasons/soils are absent.

### 2B. Vegetation coverage and rendering

- derive vegetation form/coverage/density separately from biome name;
- render planet-scale coverage and regional forest structure;
- add individual/tree-cluster instances only where camera scale justifies them;
- keep placement deterministic and bounded by semantic coverage;
- evaluate memory, draw cost, silhouette and transition behavior.

### 2C. Minimal disturbance, only if needed

Evaluate whether static equilibrium looks too painted. If so, prototype one
bounded disturbance/history mechanism—such as moisture variability, fire age or
successional patchiness—rather than a full ecosystem simulation.

Exit gate: living regions are caused by current world state, visually transform
the globe, remain explainable, and do not require individual-organism simulation.

## Horizon 3: surface-material coupling decision

Purpose: determine whether persistent sediment is the next major emergence
engine or whether present erosion/hydrology should remain simpler.

### 3A. Research and design gate

- compare reduced sediment-routing/landscape-evolution approaches;
- define state, units, time relation and mass ledger;
- identify a minimal visible target: floodplains, terminal basin fill, deltas or
  foreland/coastal deposition;
- budget fine-graph memory and update cost;
- define interaction with current deposition and hydrology ownership.

### 3B. Bounded sediment v0, conditional

If the design gate is favorable:

- track one persistent mobile/deposited sediment quantity;
- route through existing drainage with explicit capacity/opportunity rules;
- conserve source, stored material and ocean/export sink;
- produce at least one visible and one downstream semantic consequence;
- avoid stratigraphic layers, detailed grain classes and global flexural
  feedback until v0 proves value.

Exit gate: sediment creates coherent basin/coastal geography worth its cost and
does not merely add a ledger or smooth terrain.

## Horizon 4: choose a deep physical coupling

This horizon is a decision point, not a commitment to perform all branches.

Candidates:

- shared-clock tectonic uplift, erosion and denudation;
- sediment/load-driven flexure and basin formation;
- coherent glaciers/cryosphere with water and erosion feedback;
- reduced ocean currents/heat transport for regional climate;
- soil and wetland hydrology for ecology and later agriculture.

Selection criteria:

- prerequisites are stable;
- at least three systems or major visible outcomes benefit;
- a bounded authentic model exists;
- physical ownership replaces rather than stacks over a heuristic;
- resolution and runtime risks have a credible validation plan.

The current carrier tectonic models cannot enter the product branch until their
boundary/deformation operators pass resolution gates. A different, simpler
history representation is allowed and may be preferable.

## Horizon 5: geography becomes human meaning

Purpose: prepare and then optionally build the Civilization-board dimension of
the project without allowing it to eclipse planet generation prematurely.

### Foundations

- semantic regions and traversability;
- freshwater/coast access and hazard/opportunity fields;
- ecological productivity and resource affordances;
- routes, chokepoints and settlement suitability;
- scale-aware symbols, labels and borders.

### Bounded world-history candidate

Prototype aggregate settlement growth, route formation and cultural diffusion
before individual agents or detailed economies. Geography should constrain and
differentiate outcomes; generated history should in turn create visible map
structure and stories.

Exit gate: the human layer reveals consequences of the generated planet and
produces emergent narrative. If it behaves like independent noise over a map,
deepen the coupling or stop.

Full civilization/economic simulation remains a distant option requiring a
separate scope decision.

## Continuous workstreams

These run when justified rather than waiting for one horizon to finish:

### Correctness and performance

- resolve discovered topology, unit, convergence and cache defects;
- profile before optimizing;
- protect iteration speed and Windows GPU viability;
- add tests around previously costly failures.

### Product visual quality

- improve lighting, materials and animation when they have strong visible
  return and preserve layer separation;
- maintain controlled captures and presentation regression cases;
- favor semantic/generalization improvements over brute-force density.

### Current-system criticism

- periodically reassess whether plate/crust initialization, stage architecture,
  adaptive refinement, climate, erosion and rendering still earn their shape;
- permit deletion or fundamental replacement when a simpler or more generative
  architecture is demonstrated;
- do not turn this roadmap into protection for current code.

### Research

Use targeted subagent/external research for the questions listed in
[gaps.md](gaps.md). Each study should return a Hex3-sized mechanism and compute/
benefit recommendation, not only a survey.

## Near-term sequence

Unless new evidence changes priorities:

1. finish the documentation/archive and contributor-entry work;
2. implement provenance and the elevation/unit contract;
3. extract river/water-body semantics, then range semantics;
4. establish presentation profiles, legends and consistent map/globe
   generalization;
5. research and prototype biome constraints;
6. build scale-aware vegetation presentation;
7. run the sediment v0 design/research gate;
8. choose the next deep coupling from measured prerequisites and payoff.

This sequence deliberately alternates infrastructure, visible payoff and deeper
physical work. It should produce useful product improvements without closing off
larger reworks.
