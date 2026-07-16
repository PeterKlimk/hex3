# Living Surface V0 decision

Status: **current architecture decision; semantic kernel implemented and
diagnostically evaluated, not promoted**, 2026-07-17.

This decision selects the next bounded world expansion after Water Geography
V0. It replaces the long-term role proposed for the current ecology classifier;
it does not promote that classifier or authorize a dynamic ecosystem.

## Decision

Build **equilibrium physiognomy**: a cheap, cause-first description of how much
land is bare, herbaceous, woody or wet. Continuous cover is primary. Named
biomes, ecological regions and rendering recipes are derived consumers.

```text
final temperature + relative precipitation/demand
  + drainage-relative wetness + scale-declared terrain exposure
  -> thermal and water limitations
  -> growth opportunity and total vegetation cover
  -> woody/herbaceous/wet partition
  -> optional regions and presentation recipes
```

This is an **authentic hack** grounded in the dependency order of equilibrium
vegetation models. It does not claim carbon balance, NPP, LAI, species,
succession or predictive ecology. The objective is coherent, explainable world
structure with high visual and downstream payoff.

## Why this wins the current portfolio choice

Living Surface V0:

- consumes the now-coherent cheap climate, terrain, drainage and water objects;
- changes the identity of the whole visible planet rather than one rare site;
- provides reusable inputs for resources, settlement and later human geography;
- is an `O(n)` derived product with a narrow state contract; and
- can improve later when seasons, soils, sediment or disturbance earn their cost.

The alternative source-to-sink sediment system has greater eventual physical
coupling, but its smallest honest version needs separate bedrock and alluvial
cover, mobile load, one authoritative transport route, rerouting and material
ledgers. Its most attractive outputs—floodplains and deltas—also lie where the
current product mesh is coarsest. It remains the next large physical candidate,
behind explicit surface-material ownership and localized lowland/coastal
resolution. Re-enabling the old deposition scalar would repeat a parked
experiment, not establish Sediment V0.

No expansion loses because the current inputs are already adequate for a
bounded proof and the expected visual/downstream payoff is unusually broad.
The stop gate below still permits rejection after implementation evidence.

## Existing prototype disposition

`EcologySemantics` is an on-demand diagnostic transform, not state owned by
`World`, a product stage or a renderer input. Preserve only its useful shape:

- pure deterministic derivation;
- continuous limitations before labels;
- authoritative water masking and physical unit conversions; and
- area-weighted audits and transition diagnostics.

Do not inherit its formulas or categorical contract as architecture:

- `precipitation / demand` is a relative moisture-supply ratio, not a physical
  aridity index;
- dividing that ratio by each world's land mean cancels global wetness changes;
- freshwater distance is measured from map-selected rivers/proper lakes and can
  cross irrelevant catchments or ocean, so it is not soil or groundwater state;
- absolute elevation stress partly duplicates the lapse correction already in
  final temperature;
- maximum one-edge grade is resolution/faceting sensitive; and
- forest, steppe and wetland labels overclaim normalized seasonless inputs.

The representative seed-12345 audit at 100k coarse / 255,866 actual fine cells
assigned 67.8% of land to Desert, left several labels effectively unused, and
placed 18.4% of land below a 0.2 score-dominance margin. Area-weighted mean
heat/moisture/vegetation/tree/wetland scores were
`0.74/0.18/0.17/0.14/0.11`. These numbers do not validate or falsify the
climate; they show that the classifier thresholds are not a product basis.

The original `living-surface-preview` packet mapped those limiting factors and
consequences without presenting a biome palette. The same selectable packet now
maps the replacement kernel's causes and exclusive fractions. Its sidecar
records known semantic limits. It remains diagnostic evidence, not a product
view.

## Implemented semantic checkpoint

`LivingSurfaceSemantics` now exists as an on-demand derived product beside the
quarantined classifier. It is not stored by `World` and has no product renderer
or export consumer. The kernel currently implements:

- final precipitation used without land-mean normalization; the current build
  fixes its multiplier at 1.0 until one upstream control can rebuild both
  precipitation and hydrology coherently;
- a dimensionless temperature-dependent demand and saturating relative-water
  response;
- geometric contributing area routed conservatively over authoritative
  `Hydrology::drainage_dir`, independent of precipitation and `RiverSelection`;
- HAND against drainage-integration-aware physical elevation, proper-lake water
  surfaces and ocean datum, with terminal dry basins left unresolved;
- a nominal 2,000 km2 drainage reference with a four-local-cell resolution
  floor, 30 m vertical decay and a disclosed 0.35 subcell saturation cap on
  non-reference land; reference-network membership alone claims no channel or
  floodplain width; and
- exact bare/herbaceous/woody/wet partition closure on land and zero terrestrial
  fractions under water.

Fourteen manufactured tests cover water-control response, absolute-dry-world
behavior, precipitation and
drainage causal signs, closure, submerged state, no second elevation cutoff,
conservative unequal-area routing, bluff/floodplain HAND, channel/dry-basin
reference rules, explicit refusal to infer wetland occupancy from reference
membership, adaptive large-cell anchor rejection, cell-ID permutation,
continuous HAND response and cycle rejection. The real fine-mesh smoke test
also checks finite bounded output and fraction closure. Direct lake-fringe and
cross-resolution landform correspondence remain promotion gates, not claims.

The first untuned 2,000 km2 / 150 m implementation failed immediately: on seed
12345 at 255,866 cells it gave mean drainage saturation `0.833` and wetland
fraction `0.762`, because the reference catchment was below one evidence cell
and most cells self-referenced. That result is retained as a scale-contract
failure, not a tuning arm.

With the initial global resolution floor and still-conflated channel-reference
semantics, seed 12345
gives mean thermal/water/drainage/cover of `0.906/0.224/0.180/0.298` and mean
bare/herbaceous/woody/wet fractions of `0.702/0.109/0.113/0.075`. Seed 8675309
gives `0.878/0.226/0.183/0.306` and
`0.694/0.098/0.131/0.077`. Maximum observed closure error is `1.2e-7`; about
3.7-5.3% of land terminates without a drainage reference. Those values are
retained as a failed intermediate witness, not the accepted result.

Review found two distinct meanings hidden in that result. First, one locally
large adaptive cell could self-anchor with no upstream catchment. The kernel now
requires both 2,000 km2 and four times each candidate cell's local area. Second,
drainage-reference membership identifies a datum but supplies no evidence for
the channel/floodplain width inside that cell. Reference cells therefore claim
zero wetland occupancy; low-HAND non-reference land and lake fringe can still
carry saturation. A later width/valley-floor owner may replace that conservative
unknown, but catchment membership alone may not.

The regenerated accepted checkpoint for seed 12345 at 255,866 cells gives mean
thermal/water/drainage/cover of `0.906/0.224/0.067/0.234` and mean
bare/herbaceous/woody/wet fractions of `0.766/0.081/0.120/0.033`. Maximum
closure error is `1.2e-7`; 5.0% of land terminates without a reference. The
release derivation takes about `18 ms`. The result record is 48 bytes/cell (11.7 MiB
there; 366 MiB at 8M), but output plus the current graph/area/HAND working
buffers approach roughly 0.5 GiB at 8M before the already-retained world is
counted. Peak RSS has not been isolated. Keep it on demand and retain only a
compact fraction contract if a consumer earns ownership.

The corrected maps expose latitudinal cold, transported-moisture/coastal
gradients and low-HAND texture without the old circular freshwater-distance
halo or an automatically wet reference network. They do **not** yet pass the
product stop gate: the scalar views remain dominated by broad climate bands,
interiors are mostly bare, no blended physical view has been judged, and a
second seed has not been regenerated after the review fix. This may be an
upstream precipitation-distribution finding or a missing cover cause; do not
answer it by tuning biome or cover thresholds.

## V0 semantic contract

### Inputs

- final lapse-corrected temperature;
- final precipitation and a declared temperature-dependent demand proxy;
- final precipitation with no normalization inside this stage; a future
  planetary wetness control must alter precipitation and rebuild hydrology from
  the same upstream setting rather than recolor vegetation after water bodies
  have already been solved;
- authoritative submerged/ocean/lake identity;
- physical drainage topology and contributing area, independent of the
  cartographic river-selection threshold; and
- a scale-declared robust slope/relief exposure measure if it materially
  improves exposed-rock geography.

Precipitation remains dimensionless relative supply. Therefore V0 calls its
water result **relative water limitation**, not PET, soil moisture or physical
aridity.

### Continuous derived fields

1. `thermal_opportunity`
2. `relative_water_limitation`
3. `drainage_saturation`
4. `growth_opportunity`
5. `vegetation_cover`
6. `woody_share`

Drainage saturation should use hydrologic position such as height above the
downstream drainage reference (HAND), optionally combined with contributing
area and slope. It must distinguish a floodplain from a bluff beside the same
river. Reference-network membership is not a floodplain-width estimate.
Euclidean distance to rendered water is rejected as the owner.

The final exclusive terrestrial fractions are:

```text
wetland    = cover * saturation
woody      = (cover - wetland) * woody_share
herbaceous = (cover - wetland) * (1 - woody_share)
bare       = 1 - cover
```

The implementation may use algebraically equivalent bounded functions. Every
fraction must be finite, lie in `[0, 1]`, and sum to one on land. Submerged cells
receive no terrestrial fractions. These are equilibrium cover opportunities,
not persistent biomass or ecological history.

### Ownership and scale

V0 is a derived post-surface semantic product. The implemented proof computes
on the active tessellation and retains nothing in `World`. If a renderer, export or
later stage adopts it, retain only the compact fractional contract and declare
invalidation from temperature, precipitation, hydrology and terrain. Do not
copy the current 48-byte diagnostic record onto every maximum-density
cell by default.

Connected physiognomy regions may be derived after continuous state is sound.
Semantic fractions remain free of presentation noise. Rendering can blend
substrate by fractions and use deterministic density-driven symbols or tree
instances; noise may vary placement, hue and assets but not semantic totals.

## Deliberately absent from V0

- biome lookup tables as primary state;
- calibrated physical productivity, carbon or soil-water balance;
- seasons, deciduousness, fire, disturbance and succession;
- species or individual-tree ecology;
- vegetation feedback into runoff or erosion;
- procedural noise that changes semantic cover; and
- automatic sediment, soil, cryosphere or civilization scope.

Seasons are the leading causal prerequisite if actual biome identity, fire,
snow ecology or woody/grass ambiguity later proves visibly important.

## Manufactured correspondence gates

Before a product palette is judged:

1. Constant flat land and climate produce constant fields with no semantic
   noise.
2. More precipitation at fixed demand monotonically reduces water limitation
   and increases cover up to saturation.
3. More demand at fixed precipitation increases water limitation.
4. A planetary wetness change affects global cover; no internal normalization
   cancels it.
5. Reversing a manufactured windward/lee precipitation field reverses the
   cover relationship.
6. Lapse-corrected temperature changes cover with altitude without a second
   absolute-elevation cutoff.
7. Low-HAND floodplain cells become wetter than nearby high bluffs;
   reference-network membership alone does not assert wetland occupancy, and a
   future width owner must distinguish a steep narrow channel from a broad flat
   floodplain.
8. Fractions are bounded, close exactly, are absent under water and respond
   continuously to small input changes.
9. Cell-ID permutation and globe rotation preserve corresponding output.
10. Reference identity, saturation area, area-weighted cover and region areas
    converge on the same manufactured landform across uniform and adaptive
    useful mesh resolutions. This gate is still open.
11. Presentation seed/noise changes placement detail but not semantic coverage.

## Product proof and stop gate

The complete product proof should produce one matched global packet containing:

- causal inputs and limiting factors (**implemented**);
- the four cover fractions (**implemented**) and their blended semantic result;
- ordinary Physical, Authentic and Dramatic presentation at globe and regional
  scale; and
- one sidecar with cover closure, area totals and runtime (**implemented**), plus
  coupled response controls and measured peak memory.

Use one representative world and one known climate outlier first. Expand only
if the mechanism survives. V0 earns promotion only if it creates recognizable
rain-shadow, latitudinal, riparian/floodplain and exposed-terrain structure,
reads as a living planet, and supplies more useful causal state than a direct
climate color lookup. If it merely paints smooth climate bands or drainage
halos, revise the causal owner or stop; do not tune category thresholds.

## Research basis

The dependency order follows equilibrium vegetation models such as
[BIOME3](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/96GB02344):
physiological viability and resource limitation precede vegetation structure
and biome summaries. [CASA](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/93GB02725)
supports the reduced “opportunity times limiting factors” shape, without
granting Hex3 its NPP interpretation. HAND provides the drainage-relative
wetness prior ([Nobre et al. 2011](https://doi.org/10.1016/j.jhydrol.2011.03.051)).
Dynamic global vegetation models such as
[LPJ](https://infoscience.epfl.ch/entities/publication/371b2746-66c3-4a26-864e-1d5d4a4d824e)
define the fuller causal graph but are deliberately not V0 templates.

For presentation, production systems support the separation
`semantic mask -> authored procedural recipe`: see
[Minecraft's layered world-generation overview](https://learn.microsoft.com/en-us/minecraft/creator/documents/world-generation?view=minecraft-bedrock-stable)
and [Ubisoft's Far Cry 5 procedural-world account](https://news.ubisoft.com/en-us/article/16TjVZmAtD85EWcvHtxHXL/far-cry-5-creating-curiosity-in-a-familiar-world).
