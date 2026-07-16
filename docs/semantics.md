# Semantic objects

Semantic layers interpret retained modeled state without changing it. They are
replaceable, derived products: changing their thresholds cannot alter tectonics,
terrain, climate or hydrologic flow.

## Hydrology

This document describes the first implemented semantic-object slice: water
bodies and river networks. These objects interpret modeled hydrology without
changing it and contain no camera, relief, color or stroke settings.

Hydrology semantics live in `src/world/semantics.rs`; their compact whole-world
report lives in `src/world/water_geography.rs`; ecological semantics live in
`src/world/ecology.rs`.

## Water bodies

`WaterBodySemantics` derives connected objects from a tessellation and one
retained hydrology surface.

Each `SemanticWaterBody` records:

- a per-world-stage identity;
- ocean, lake or sub-threshold pond classification;
- member cells;
- physical area, surface elevation and maximum depth;
- terminal, ocean or downstream-basin outlet relationship.

Identity is `(basin_id, anchor_cell)`. The anchor is the deterministic deepest
cell, with cell index as a tie-break. Oceans have no basin ID and are represented
as connected components. This identity is stable for a retained stage and more
robust to climate-level changes than the transient `Hydrology.water_bodies`
vector index, but it is not promised to survive remeshing or terrain changes.

Ponds remain semantic objects even though they do not qualify as hydrologic lake
sinks. This preserves the distinction between “water exists,” “proper lake,” and
“should be rendered at this scale.”

## River policy and selection

`RiverThresholdPolicy` owns the definition of a semantically visible network:

- `CatchmentKm2` is the product policy. It uses a physical catchment-area
  threshold at mean wetness and resolution-aware coarse-mesh floor.
- `Legacy` preserves historical count-equivalent fractions for controlled A/Bs.

The default minimum catchment is 2000 km². Major rivers are selected from large
outlets and traced upstream through sufficiently large branches. Major selection
is intersected with the All network, making it a true subset.

`RiverSelection` computes only:

- All and Major cell masks;
- maximum physical/count-equivalent flow values;
- overflowing-lake outlet paths.

Rendering consumes this lightweight object. Screen-space width, opacity, color,
texture baking and glint remain presentation responsibilities.

## Full river network

`RiverNetwork` extends the same selection with:

- deterministic upstream adjacency;
- per-cell Strahler order;
- semantic mouths;
- source/confluence/mouth-bounded reaches;
- approximate physical reach length;
- major-river membership.

Reach cells are ordered upstream to downstream. Confluence cells may belong to
both incoming and outgoing reaches so topology remains explicit. Reach identity
is deterministic within the built network but is not yet a persistent identity
across terrain, climate or policy changes.

The river audit consumes these shared masks, adjacency, Strahler values and
mouths rather than reconstructing a renderer-like network independently. Lake
audits consume semantic area, depth, identity and outlet classification.

## Whole-world water-geography report

`WaterGeographyReport` is a deterministic derived summary used by the dossier
packet. It does not add physical state or persistent identity. From one retained
hydrology surface plus `WaterBodySemantics` and `RiverNetwork`, it records:

- connected ocean and geographic-land component counts and descending areas;
- ocean coastline and classified-lake shoreline length;
- basin, lake, pond, terminal and overflow counts;
- river mouth/reach counts and independent highest-discharge, longest-trunk and
  highest-Strahler roles;
- drainage-integration cut/source footprint and intersection with selected
  channel masks; and
- semantic ownership, component and mouth consistency failures.

An overflowing basin with `overflow_target == None` is reported only as having
no basin target. Current hydrology uses that representation for normal ocean
exit as well as rare unresolved/cyclic walks, so the report does not manufacture
a distinction that retained physical provenance cannot support.

## Ownership

| Concern | Owner |
|---|---|
| Drainage direction, flow and water level | `Hydrology` world state |
| Water-body identity/type and river hierarchy | Semantic objects |
| Whole-world water/coast/repair summary | Derived `WaterGeographyReport` |
| All/Major importance policy | `RiverThresholdPolicy` |
| Stroke width, color, opacity and antialiasing | Presentation/renderer |
| Earth-reference comparisons and gates | Diagnostics/validation |

Changing river width must not rebuild semantic topology. Changing a semantic
threshold may change which reaches are visible but cannot change hydrologic
flow. Changing climate or terrain invalidates semantic objects because their
modeled input changed.

## Current limitations

- River reaches use cell-center paths and chord-based approximate lengths.
- Catchment area and trunk-profile summaries remain diagnostic calculations,
  not stored reach properties.
- Names, cross-stage correspondence and cross-resolution identity do not exist.
- Water-body semantics currently build on demand rather than being cached as a
  retained stage product.
- Ponds/wetlands need a more deliberate ecological and presentation policy.
- River generalization still selects cells; geometry simplification and
  zoom-dependent reach aggregation remain future cartographic work.
- Land/ocean components currently retain aggregate ranked areas rather than
  reusable coast geometry, strait topology or cross-resolution ancestry.
- Drainage integration retains exact sparse cuts but not per-breach event
  identity, reason or endpoint kind.

## Ecology and biome proxies

`EcologySemantics` is the first living-world semantic prototype. It consumes
one retained tessellation, elevation, temperature, precipitation and optional
hydrology surface. It derives continuous per-cell potentials for heat,
moisture, vegetation, trees and wetlands, plus cold, water, alpine and terrain
stress and freshwater access.

Moisture uses `precipitation / temperature-dependent demand`, normalized to an
area-weighted land mean of one. Terrain stress uses physical grade, alpine
stress uses elevation in kilometres, and freshwater access is graph distance
from semantic rivers and non-ocean water in kilometres.

Each cell also receives a broad `BiomeKind` and classification confidence.
Confidence measures dominance over the runner-up label; low confidence is an
explicit transition zone. Oceans and retained lake water come directly from
hydrology.

These labels are calibrated, seasonless ecological proxies. They are **not**
Köppen classes or claims about real vegetation. The climate has no seasonality,
precipitation has no calibrated physical unit, and the model lacks soil,
substrate, disturbance, ecological history and interspecies dynamics.
Continuous potentials should drive future coverage and rendering; labels are
primarily for inspection, summaries and region identity.

The `--biome-audit` diagnostic reports area-weighted potential means, biome
coverage, transition coverage and region coherence. No biome palette or
vegetation renderer is authoritative yet. Initial audit output may legitimately
show broad transition coverage or unused labels; cross-seed calibration and
control-response validation remain promotion gates, not reasons for hidden
single-seed tuning.
