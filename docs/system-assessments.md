# System fidelity and Pareto assessment

This document assesses the accepted product architecture by mechanism: what it
buys, what kind of fidelity it claims, what it costs, and whether deeper work is
currently justified. It is not a roadmap. Assessments are provisional decisions
to review as evidence and project priorities change.

Prioritization based on these assessments lives in the
[gap analysis](gaps.md) and [roadmap](roadmap.md).
Current implementation and product-role dispositions live in the
[cross-system disposition](system-disposition.md). The
[subtractive architecture audit](subtractive-audit.md) is earlier evidence.

No “Foundation” or “Retain” label protects a system from fundamental rework. It
means the capability is valuable under current evidence, not that its present
representation or algorithm must survive.

See the [thesis](thesis.md) for fidelity vocabulary, the
[documentation policy](documentation-policy.md) for evidence/status terms, and
the [experiment registry](experiment-registry.md) for non-default alternatives.

## Assessment scale

The **Pareto posture** is one of:

- **Foundation** — essential shared substrate; maintain and validate.
- **Retain** — good benefit/cost at current depth.
- **Clarify** — implementation is useful, but units, ownership or claims need
  tighter definition before deeper work.
- **Deepen selectively** — a specific coupling or extension may unlock several
  valuable outcomes; do not broaden indiscriminately.
- **Hold** — more sophistication is not presently justified.
- **Missing candidate** — absent capability with plausible multi-system value;
  requires comparison before roadmap commitment.

No complete Hex3 subsystem is classified as simulation-grade. That does not
discount numerical methods inside it; it limits the quantitative claim made for
the integrated planet.

## Geometry and computation

### Spherical tessellation

- **Purpose/payoff:** common spherical cells, topology, adjacency, distance and
  area for every model and renderer.
- **Fidelity:** computational geometry, physically neutral. Convex-hull duality
  is the exact construction; kNN clipping is an approximate backend.
- **Cost:** moderate on the coarse graph and dominant during million-cell fine
  construction. Compact adjacency, parallel work and caching have high value.
- **Evidence:** topology/containment/area validation, backend comparison and
  resolution audits exist.
- **Limits:** fine construction and cache reproducibility have a special
  determinism contract; adaptive cells stress uniform-grid assumptions.
- **Pareto posture:** **Foundation.** Prefer stronger convergence, provenance and
  performance contracts over a more elaborate geometric representation.

### Coarse/fine allocation

- **Purpose/payoff:** keep planetary processes tractable while spending density
  on drainage, erosion and visible terrain.
- **Fidelity:** authentic compute-allocation strategy, not a physical process.
- **Cost:** complexity in field transfer, thresholds, cache identity and
  resolution-sensitive numerics; large GPU and CPU memory footprints.
- **Evidence:** density/resolution audits and allocation logging exist, but no
  universal cross-system convergence contract does.
- **Pareto posture:** **Foundation + Clarify.** This is the strongest current
  compute bargain. Formalize scale ownership and convergence before adding more
  fine-only mechanisms.

## Lithosphere

### Plate partition and crust initialization

- **Purpose/payoff:** establishes motion units, mixed-crust plates, continents,
  active/passive margins and interaction variety cheaply.
- **Fidelity:** authentic procedural hack. Plate flood-fill and independently
  grown cratons create useful causal state without mantle convection or crustal
  evolution.
- **Cost:** low relative to downstream value; conceptual risk arises when an
  initializer is mistaken for geological history.
- **Evidence:** structural and deterministic tests; visual and downstream value
  are clear, empirical distribution validation is limited.
- **Pareto posture:** **Retain.** Even with future time evolution, preserve an
  efficient world initializer. Let history modify it rather than requiring a
  full genesis simulation unless that produces demonstrable emergence.

### Euler motion and boundary kinematics

- **Purpose/payoff:** provides coherent rigid plate velocity, convergence,
  divergence, shear, rates and subduction polarity.
- **Fidelity:** physically based kinematic primitive with synthetic forcing and
  heuristic interaction classification.
- **Cost:** low/moderate and reused by features, history, diagnostics and future
  evolution.
- **Evidence:** unit conversion, subdivision and resolution-aware boundary
  normalization tests.
- **Limits:** axes and speeds are random; no torque, slab pull, ridge push,
  mantle flow or plate-force balance.
- **Pareto posture:** **Retain + Clarify.** Force-derived motion is attractive
  only if it improves several visible/systemic outcomes, not merely provenance.

### Tectonic history, deformation and feature fields

- **Purpose/payoff:** turns boundary motion and crust interaction into duration,
  crustal work, trenches, arcs, ridges, collision structure and activity.
- **Fidelity:** mixed. Physical rates, histories, volume ledgers and flexural
  forms coexist with calibrated response kernels and a legacy uplift baseline.
- **Cost:** high conceptual surface and potentially high runtime in carrier and
  thin-sheet variants. The accepted product uses only a subset.
- **Evidence:** extensive invariants, conservation audits and scorecards;
  experimental carrier deformation currently fails resolution/promotion gates.
- **Pareto posture:** **Clarify + Deepen selectively.** The feature bridge is
  high leverage and should remain. Geological time and material continuity are
  promising, but only through convergent operators and same-clock surface
  coupling—not by accumulating more selectable terrain responses.

### Elevation, crust thickness and sea level

- **Purpose/payoff:** supplies the global terrain/bathymetry envelope consumed
  by climate, fine allocation, erosion, hydrology and presentation.
- **Fidelity:** physically based Airy/thermal/flexural skeleton mixed with
  authentic legacy orogen parameterization and macro crustal variation.
- **Cost:** efficient at coarse scale; enormous downstream leverage and risk.
- **Evidence:** area/conservation tests, numeric scorecards and visual audits.
- **Limits:** model elevation and physical-kilometre/render conversion need one
  explicit contract. Sea level is solved to a target land fraction rather than
  a conserved global water inventory.
- **Pareto posture:** **Foundation + Clarify.** Stabilize units and ownership.
  Deeper sea-level/isostatic coupling earns priority only with sediment, ice or
  evolving loading that can use it.

## Atmosphere

### Temperature and circulation

- **Purpose/payoff:** latitude structure, continental contrast, winds, vertical
  motion and terrain interaction for precipitation and visualization.
- **Fidelity:** authentic physically informed temperature plus a prescribed
  physically inspired circulation.
- **Cost:** cheap compared with dynamic climate and highly reusable.
- **Evidence:** circulation continuity/sign/balance tests and qualitative climate
  behavior; limited observational calibration.
- **Limits:** no energy balance, seasons, greenhouse physics, storms, ocean heat
  transport or dynamically evolving weather.
- **Pareto posture:** **Retain.** Dynamic 3-D climate is not justified. Consider
  longitude-asymmetric heat/circulation only if it materially changes regional
  geography, not solely to make the atmosphere more scientific.

### Moisture and precipitation

- **Purpose/payoff:** couples ocean, wind, terrain and climate to rainfall,
  drainage and erosion using coherent transport rather than painted wet bands.
- **Fidelity:** physically based finite-volume transport skeleton with tuned,
  normalized source/sink physics.
- **Cost:** iterative but affordable on the coarse graph; convergence caps and
  normalization limit predictive claims.
- **Evidence:** numerical tests and climate/moisture analyses; stronger
  cross-resolution and geographic distribution validation would help.
- **Pareto posture:** **Retain + Clarify.** This is an excellent authentic
  simulation bargain. Improve evidence and units before adding more processes.

### Fine local climate feedback

- **Purpose/payoff:** lapse, orographic, lee and lake modulation lets changed
  fine terrain affect erosion without rerunning global climate.
- **Fidelity:** authentic physical shortcut.
- **Cost:** approximately proportional to outer feedback passes and often small;
  risk is circular terrain tuning or overstated climate meaning.
- **Evidence:** isolated rain-shadow work showed hydrologic value but weak
  mountain-shape payoff; most optional terms are parked.
- **Pareto posture:** **Deepen selectively.** Keep the bounded architecture;
  promote only feedbacks with independent climate/hydrology value.

## Surface processes

### Hydrology

- **Purpose/payoff:** drainage topology, basins, flow, lakes and river hierarchy;
  supplies erosion routing and highly visible geography.
- **Fidelity:** physically based topological skeleton with authentic depression
  integration, breach and equilibrium-lake hacks.
- **Cost:** scalable graph work on millions of cells; sensitive cleanup and lake
  policies can strongly alter geography.
- **Evidence:** river/basin/lake audits, area-aware routing and known integration
  studies. Transient water balance and groundwater evidence are absent.
- **Pareto posture:** **Foundation + Deepen selectively.** Hydrology is a central
  emergence engine. Favor persistent water/sediment and meaningful basin
  behavior over small additions such as water chemistry.

### Fluvial and hillslope erosion

- **Purpose/payoff:** converts broad tectonic terrain into organized valleys,
  divides, relief and drainage-responsive landforms.
- **Fidelity:** physically based process model using stream power, hillslope
  transport, thickness/isostatic state and limited deposition; calibration and
  elapsed time remain worldbuilding-scale.
- **Cost:** one of the dominant iterative costs. Rerouting, implicit diffusion
  and climate passes multiply work.
- **Evidence:** operator tests, roughness/mass ledgers, erosion validations,
  cross-seed terrain audits and controlled visual sweeps.
- **Limits:** the 200-step epoch is numerical maturity, not geological time;
  sediment is not a persistent material system.
- **Pareto posture:** **Foundation + Clarify.** Same-clock tectonics/erosion and
  sediment continuity are more valuable than adding further isolated shaping
  operators.

### Glacial process

- **Purpose/payoff:** potential alpine specificity—over-deepening, cirques,
  sharper divides and high-latitude terrain.
- **Fidelity:** early authentic process hack; default off.
- **Cost:** additional routing/abrasion passes; current model omits ice dynamics,
  mass balance, valley widening and loading response.
- **Evidence:** implementation exists, but product benefit and convergence have
  not earned promotion.
- **Pareto posture:** **Hold.** Revisit as one coherent glacier/ice feature only
  after climate/time/terrain ownership is clear; do not tune the current pass as
  generic mountain texture.

### Persistent sediment and surface materials

- **Purpose/payoff:** would connect erosion to floodplains, foreland basins,
  deltas, shelves, lithology, flexural loading and tectonic recycling.
- **Fidelity:** currently absent as persistent state; local deposition and loss
  ledgers are partial hooks.
- **Cost:** potentially high state, routing, time and validation complexity.
- **Pareto posture:** **Missing candidate.** It has unusually broad coupling and
  visual potential. Start with the smallest conserved sediment budget that can
  create basin/coastal consequences; avoid a full stratigraphic simulator.

### Biomes, vegetation and ecology

- **Purpose/payoff:** turns climate, terrain, water and soils into recognizable
  regions and living structure; provides one of the clearest routes from planet
  model to visual richness and board/game semantics.
- **Fidelity:** a diagnostic semantic potential/constraint prototype exists;
  ecological regions, vegetation state and a living-surface product stage are
  absent. A tree renderer would be presentation driven by that state, not the
  ecology model itself.
- **Cost:** semantic biomes can be cheap; persistent vegetation, competition,
  disturbance and fine tree placement can become large state/render problems.
- **Pareto posture:** **Accepted bounded layer.** Equilibrium fractional
  physiognomy is implemented on demand and its linear presentation is
  selectable. Regions, biome calibration and richer vegetation remain separate
  gated decisions. See [Living Surface V0](living-surface.md).

### Culture, settlement and civilization

- **Purpose/payoff:** would make generated geography consequential to agents,
  routes, resources, political regions and history, strongly serving the
  Civilization-board side of the project vision.
- **Fidelity:** one authentic aggregate site-and-terrestrial-route slice now
  exists on demand. Population, culture and history remain absent; plausible
  deeper approaches range from procedural cultural regions to dynamic
  population/economic simulation.
- **Cost:** extremely elastic; a deep agent simulation could eclipse the planet
  generator, while grounded settlement/route/cultural diffusion models may
  yield much of the visible narrative value.
- **Pareto posture:** **Retain the completed Consequential Geography V0
  operators; stop vertical expansion.** Terrain-sensitive routes and a
  conservative lower-corridor explanation pass cheaply. The site prior remains
  provisional, and broad route comparisons do not justify pass, gap or
  chokepoint semantics. It claims no population, history or civilization
  simulation. Deeper human systems require a future portfolio choice. See the
  [V0 decision](consequential-geography.md).

## Semantic and presentation systems

### Semantic feature derivation

- **Purpose/payoff:** converts fields into major rivers, water/material states,
  regions, ranges, overlays and scale-dependent objects meaningful to people or
  future gameplay.
- **Fidelity:** authentic interpretation; currently distributed across world,
  app and diagnostics.
- **Cost:** relatively low compute; architectural cost comes from duplicated
  thresholds and ambiguous ownership.
- **Evidence:** existing diagnostics and visual selectors demonstrate value.
- **Pareto posture:** **Missing candidate / Deepen selectively.** Formalizing
  this layer is likely one of the cheapest ways to improve legibility,
  inspection, cartography and a Civilization-board feel simultaneously.

### Relief, rivers, color and lighting

- **Purpose/payoff:** makes physically tiny or abstract features readable and
  produces the product's visual character.
- **Fidelity:** explicit cartographic and visual hacks grounded in world fields.
- **Cost:** generally excellent visual return, with exceptions such as the fixed
  high-resolution river texture and duplicated surface paths.
- **Evidence:** relief and river sweeps, human review, limited preset unit tests.
- **Limits:** metadata, legends, zoom generalization, unit display and parity
  across Globe/Map and unified/colored paths are incomplete.
- **Pareto posture:** **Foundation + Deepen selectively.** Improve declared
  presentation profiles, consistency and capture evidence before pursuing
  heavier physically based rendering.

### Wind particles and diagnostics

- **Purpose/payoff:** communicates vector fields and internal model state with
  unusually high comprehension and spectacle.
- **Fidelity:** model-grounded visualization with visual particle dynamics.
- **Cost:** efficient GPU work; diagnostic views are mostly cheap recoloring.
- **Evidence:** strong observational utility, limited automated visual and GPU
  correctness testing.
- **Pareto posture:** **Retain.** Add legends, metadata and correctness tests;
  avoid turning diagnostic particles into a second atmospheric simulation.

## Portfolio-level conclusions

1. The bounded mountain comparison located a real forcing/organization defect,
   but H remains a control and C, G and manufactured I do not earn product
   ownership. The inheritance follow-up does not divide the reviewed belt
   honestly. Drop continuity alone as a rejection criterion for the reviewed
   parent, but keep the legacy generic tableland owner as unresolved debt and
   reopen it only under the ordinary-world trigger in the
   [landscape organization strategy](landscape-strategy.md).
2. Water Geography V0 now joins connected ocean identity, moisture source,
   lakes/basins/outlets, river roles, repair provenance and exact raw coast
   geometry into one coherent derived account. Scale-aware cartographic
   selection remains consumer-owned.
3. Coarse climate remains near a useful Pareto point. Transported moisture has
   earned retention over the tested cheaper conditional climatology through
   stable regional and hydrologic consequences plus manufactured causal signs.
4. Semantic derivation and reproducible presentation profiles are comparatively
   cheap, high-leverage architecture, not substitutes for physical state.
5. Water/climate inputs now have declared meaning, bounded equilibrium
   physiognomy is accepted on demand, and Consequential Geography has completed
   its bounded decision. World Readability V0 is the next expansion because it
   composes several retained systems at product scale and directly advances the
   board/globe identity; persistent sediment remains a strong physical candidate
   behind a deliberate material/time/river/lowland gate.
6. Geological time matters only if processes share it. A physical clock attached
   to otherwise uncoupled or non-convergent operators is not progress by itself.
7. The implemented Stage 4 frontier leaves major creative space: aggregate
   inhabited geography can build on current state without requiring every
   intermediate domain—or human history—to become a full simulation.
