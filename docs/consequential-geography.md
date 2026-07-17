# Consequential Geography V0 decision

Status: **selected roadmap priority; traversal/access foundation implemented and
operator-tested; sites, routes, product evaluation, promotion, default,
population state and Stage 5 remain absent**, 2026-07-17.

## Product question

Given one final generated world and one disclosed aggregate mobility and
site-selection prior, does terrain, water and living opportunity materially
constrain one plausible configuration of sites and routes?

The intended payoff is a world that reads as both planet and board: a viewer can
see why a site is near this lake, why a route uses that gap, and why a coastal or
river relationship matters. The result is a functional test of retained world
systems as well as a new visible consequence.

V0 does not ask where real people would certainly settle or reconstruct a
history. Many configurations can be plausible on the same geography.

## Fidelity and causal reference

Consequential Geography V0 is an **authentic aggregate hack**. Higher-fidelity
settlement and transport models preserve several relationships that a useful
compression should not discard. Settlement-location work frames sites through
the cost of accessing critical resources rather than the value of one cell,
while settlement-system models add spatial interaction and feedback
([Wood](https://doi.org/10.2307/279249),
[Sikk and Caruso](https://doi.org/10.1016/j.ecolmodel.2024.110652)):

- movement pays physical distance and terrain-dependent effort;
- water access and relative biological opportunity constrain viable sites and
  their accessible catchments;
- coast and freshwater access are different affordances;
- settlements compete for nearby opportunity and are not independent top-cell
  samples;
- routes connect destinations through low-cost corridors, balance construction
  against travel efficiency and tend to reuse useful connections; and
- gaps, crossings and coastal junctions matter because of the surrounding
  network, not because they received decorative labels.

Reality also includes technology, path dependence, institutions, conflict,
trade, migration and historical accident. V0 omits those causes and therefore
claims causal sensitivity and intelligible topology, not historical prediction.

The closest compact precedent is a neutral least-cost landscape model that
combines irregular sites, least-cost catchments, a sparse neighbor graph and
reduced cost along established routes
([Etherington et al.](https://doi.org/10.1007/s10980-024-01836-w)). V0 adapts
that causal shape to Hex3's actual terrain, water and living inputs rather than
copying its neutral landscape.

## Ownership and data flow

```text
final Stage-4 world state
  active Voronoi graph + post-repair terrain + hydrology
      |
accepted on-demand semantics
  water/coast identity + aggregate river policy + Living Surface fractions
      |
Consequential Geography derivative
  named factor fields -> aggregate sites -> sparse route objects/relations
      |
cartographic scene
  markers + strokes + labels + scale-dependent emphasis
```

V0 is deterministic, presentation-independent and derived on demand. It does
not add state to `World`, create a numbered stage or feed back into terrain,
hydrology or Living Surface. A future consumer may justify retained state or
cross-stage identity; V0 does not prebuild that architecture.

The clean implementation seam is one product module beside existing water and
living semantics, with explicit inputs rather than a generic entity framework:

```text
ConsequentialGeography::build(
    tessellation,
    hydrology,
    water_semantics,
    aggregate_river_policy,
    living_surface,
    config,
)
```

`WaterGeographyReport`, the superseded ecology classifier, renderer styling and
the quarantined landscape laboratory are not dependencies.

## Input truth contract

### Terrain and movement

- Use the active Stage-4 Voronoi adjacency.
- Use `Hydrology::elevation` as effective terrain because it includes the
  authoritative drainage integration applied after erosion.
- Record route overlap with drainage-integration cuts so repaired corridors do
  not become invisible evidence.
- Derive directed edge ascent and descent from elevation difference and stable
  physical centre-to-centre distance. Do not use relief scale, screen distance,
  raw elevation-per-radian slope or a cell gradient whose opposing faces can
  cancel.
- Treat traversal as generalized cost, not travel time, until a transport
  regime and empirical speed law earn that claim.

### Water and coast

- Submerged, ocean and proper-lake identity come from retained hydrology and
  `WaterBodySemantics`.
- Freshwater access and coast access remain separate components.
- An aggregate river policy may identify major freshwater corridors, but its
  threshold must be disclosed as a board/world scale—not literal potable-water
  distance, discharge reliability, width or navigability.
- Ponds do not count as reliable freshwater in V0. Groundwater, seasonality,
  drought reliability, event floods and channel width are unknown.

### Living opportunity and constraints

- Consume accepted Living Surface continuous values and fractions, not biome
  labels or the superseded ecology classifier.
- Call the result relative living opportunity, cover or wetness. It is not
  productivity, carrying capacity, crop yield, soil fertility or biomass.
- Physical grade is a terrain constraint. Drainage saturation and water
  limitation may be disclosed exposure proxies, but V0 has no flood, drought,
  landslide, seismic or volcanic hazard model.
- Do not add noise-based resources, ore bodies or soil claims. A later consumer
  may justify specifically named material affordances.

## Required semantic outputs

Keep the factors inspectable rather than immediately collapsing the result into
one opaque suitability map.

### Factor fields

- directed edge traversal cost, physical length and ascent/descent;
- freshwater access;
- coast access;
- relative living opportunity; and
- any narrowly named terrain or wetness constraint actually used.

Every combined site or route score retains its component contributions and
configuration. Authored combination rules are legitimate, but must be visible
and sensitivity-tested.

### Aggregate sites

Generate 12–30 deterministic, spatially separated site anchors. This is an
authored legibility and worldbuilding budget, not an inferred population count.
Each site records:

- stage-local deterministic identity and anchor cell;
- physical position and landmass/water relationships where available;
- component opportunity/constraint values and combined rank;
- the exclusion/spacing decision that admitted it; and
- input and configuration provenance.

Candidate selection first applies hard or limiting viability, then evaluates
accessible opportunity over a bounded travel-cost catchment. Site selection
should preserve multiple constraints and competition for nearby opportunity.
It must not simply choose the highest cells of one additive weighted field;
abundant woody cover cannot compensate without limit for inaccessible
freshwater or unbuildable terrain.

### Sparse route network

Connect a bounded neighbor graph among selected sites using least-generalized-
cost paths. Directional slope plus a friction surface is a well-established
minimum movement model (for example, the official
[GRASS `r.walk` model](https://grass.osgeo.org/grass-stable/manuals/r.walk.html)),
but V0 calls the result generalized cost rather than calibrated walking time.
Least-cost paths are corridors under one disclosed behavior model, not uniquely
correct historical roads ([White](https://doi.org/10.7183/2326-3768.3.4.407)).

Begin with a connected, MST-like backbone. Add only a few links whose travel
savings justify their construction, preserving the spatial-network trade
between material cost and route efficiency
([Gastner and Newman](https://doi.org/10.1103/PhysRevE.74.016117)). A bounded
discount along accepted routes may create deterministic route reuse; it is an
authentic path-dependence hack and must be ablated. The contract requires these
consequences, not a particular graph algorithm.

Each route records:

- endpoint site identities and ordered cell path;
- physical length, cumulative generalized cost and ascent/descent;
- factor contribution summary;
- shared-network use if the selected rule rewards reuse; and
- overlap with authoritative drainage repair.

V0 must not run every site pair or build a general routing service. Bound the
candidate connections and measure cost on default and high-end product meshes.

### Route-derived relationships

Create a relationship only where route geometry supports it:

- a **route-local gap** is a locally cheaper crossing of a surrounding terrain
  barrier, not a complete mountain-pass ontology;
- a **river crossing** is an approximate route/network intersection, not a
  modeled bridge, ford or channel width;
- **harbor opportunity** means a site and land route meet semantic ocean access;
  it does not imply nautical accessibility, a navigable harbor or maritime
  route; and
- a **chokepoint** has high route use or removal/detour consequence in this
  generated network; it is not a geopolitical prediction.

For the small site graph, chokepoint evidence can be exact: combine weighted
route use with the increase in disconnection or travel cost after removing the
candidate. Betweenness alone is not sufficient, although it can be computed
cheaply with the sparse-graph method of
[Brandes](https://doi.org/10.1080/0022250X.2001.9990249).

If the current cell-centre representation cannot support one of these
relationships without ambiguity, report the ambiguity or omit the label.

## Authored priors

V0 must freeze and disclose only the priors needed to make one configuration
well-defined:

- one aggregate terrestrial transport regime;
- factor combination and hard-exclusion rules;
- site count and physical spacing budget;
- route candidate/connection and optional reuse policy; and
- deterministic tie-breaking.

Modest parameter changes should alter marginal choices without completely
rewriting the world. If that is not true, the authored prior rather than the
planet is probably owning the result.

## Deliberately absent

- population, calibrated carrying capacity or settlement growth;
- agents, economy, trade volume, politics, culture, war or borders;
- migration, temporal evolution or historical lock-in;
- soils, crops, ore bodies or calibrated resources;
- maritime routing, currents, ship technology or harbor geometry;
- event hazards or defensibility claims;
- feedback into physical or living systems;
- persistent identity across stages, remeshing or regenerated worlds;
- a generic region, entity or dependency framework; and
- richer ecology or a mountain-model replacement.

## Discriminator battery

One compact battery is sufficient. Do not create another rung campaign.

### Controlled causal fixtures

1. A neutral field produces only the disclosed spacing/network baseline.
2. A steep barrier with one low gap sends the route through the gap; closing it
   creates an intelligible detour.
3. Freshwater, coast and living-opportunity changes move the corresponding site
   preference in the declared direction without changing upstream world state.
4. Rotation and cell-ID permutation preserve the semantic result apart from
   deterministic identity remapping.
5. Useful mesh-resolution changes do not make sites or paths follow adaptive
   density, seams or zero-length numerical edges.
6. Removing route reuse increases parallel corridors; removing optional
   efficiency links produces the disclosed cheaper but more fragile backbone.

### Product counterfactuals

On a small fixed world panel compare:

- the full disclosed configuration;
- uniform opportunity and uniform traversal nulls; and
- independent grade, freshwater, coast and living-opportunity ablations.

Report which objects moved, which routes changed and why. A factor need not be
important everywhere, but its claimed relationship must appear where the world
provides a relevant choice. Also report runtime, peak/retained memory and route
search count.

### Product proof

One matched packet should contain:

- the ordinary authentic/cartographic world;
- a diagnostic factor, site and route-provenance view;
- a cartographic site-and-route view; and
- one regional view where a gap, crossing or harbor opportunity is legible.

The sidecar records seed, stage, configuration, profile/camera, object summary,
counterfactual changes and cost. Site and route semantics remain identical
across Physical, Authentic and Dramatic presentation; marker size, stroke width,
labels and decluttering belong to presentation.

Human review answers whether the result adds legible board/globe meaning and
whether its explanations feel grounded rather than decorative.
This follows the useful procedural-settlement precedent of judging adaptation,
functionality, narrative and aesthetics separately on unseen maps rather than
optimizing one utility score
([GDMC challenge](https://arxiv.org/abs/1803.09853)).

## Exit and kill conditions

V0 passes only if several retained world systems create visible, explainable
and counterfactually demonstrated consequences at acceptable cost.

Stop or reduce the slice if:

- the neutral baseline is visually or structurally equivalent;
- removing a factor does not affect its claimed relationship;
- most sites merely follow the largest river or one authored score;
- routes follow mesh density, cell seams, projection or repair artifacts;
- modest prior changes completely rewrite the world;
- the result reads as decorative game noise; or
- compute, memory or architecture cost exceeds its explanatory and visual
  payoff.

On partial failure, retain only useful traversal/access semantics. Do not add
population, economics or more authored layers to rescue a weak geographic
signal.

## Current implementation checkpoint

`world::ConsequentialGeographyComponents` now builds the first bounded substrate
on demand. It retains raw freshwater access, coast access, accepted Living
Surface values, drainage-repair provenance, exact source masks, river-policy
provenance and traversal configuration. It adds no state to `World`.

For adjacent cells, directed generalized cost is physical great-circle distance
plus separately disclosed uphill and downhill penalties times ascent and
descent. The land-only access fields use half the two directed costs on each
edge: a direction-neutral there-and-back burden per leg, without a fixed edge
toll or display-scale input. Freshwater sources are selected river land cells
and land beside semantic lakes; ocean coast is separate and ponds are excluded.

Focused fixtures establish flat-distance behavior, reverse-direction component
symmetry, water as a traversal barrier, distinct ocean/lake/pond source
semantics and reduced access cost through a lower gap. These are operator tests,
not evidence that the authored penalty values, future site configuration or
visible product are fit for purpose.

## Implementation boundary

The smallest honest implementation sequence is:

1. Build and test physical edge cost plus named opportunity/access components.
2. Add deterministic spaced site selection with explanation records.
3. Add one bounded sparse route network and route-local relationships.
4. Run the counterfactual packet and make one pass/kill/reduce decision.

Code, focused tests and one result record are enough. V0 does not require a
family of contract amendments, a new experiment registry ladder or a general
semantic framework.
