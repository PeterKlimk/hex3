# World evaluation synthesis

Status: active first-pass synthesis after the clean `reference-v1` corpus,
2026-07-12. This is an evidence-directed diagnosis, not a final model-strategy
decision. Matched visual inspection and feature correspondence are still open.

## Evidence combined

- clean revision `cb506c07f971ec5792661b1aad9c852c3274d943`;
- ten canonical seeds, product legacy model, stage 4;
- 100,000 coarse cells and explicit 1,000,000 fine cap;
- metric registry v3, cache disabled;
- numerical corpus under `artifacts/evaluation/reference-v1`;
- current code/document inventories;
- [scientific, game and rendering priors](research/correspondence-priors-2026-07-12.md).

The corpus artifacts are intentionally not committed. Manifests, summaries and
run IDs make them reproducible.

## Population overview

| Measurement | min | median | max | Interpretation limit |
|---|---:|---:|---:|---|
| actual fine cells | 984,164 | 995,524 | 998,536 | resolution context only |
| total runtime | 64.0 s | 73.8 s | 87.9 s | WSL release timing, no memory/GPU |
| dry-land area | 25.90% | 26.22% | 26.76% | strongly influenced by solved datum |
| peak elevation | 5.64 km | 8.39 km | 10.32 km | extreme-cell descriptor, not morphology |
| land elevation p90 | 0.99 km | 1.23 km | 1.88 km | area-weighted upper envelope |
| land elevation p99 | 2.89 km | 3.98 km | 4.63 km | area-weighted tail |
| physical neighbor-grade p90 | 0.0062 | 0.0091 | 0.0125 | edge-scale roughness, not fixed-scale slope |
| 10-km local relief p90 | 385 m | 827 m | 1,324 m | deterministic mountain-cell sample |
| 25-km local relief p50 | 278 m | 456 m | 872 m | not area-weighted |
| 25-km local relief p90 | 1,027 m | 1,980 m | 2,878 m | max-minus-min, sensitive to extrema |
| semantic river cells | 59,148 | 62,811 | 69,107 | resolution-dependent cell population |
| lake / terrestrial area | 0.08% | 1.64% | 3.30% | needs object/water-balance interpretation |
| biome transition area | 14.5% | 29.3% | 39.5% | provisional classifier behavior only |

Runtime varies 37% despite less than 1.5% variation in active cells. Erosion
dominates with a median about 62 seconds and a 54–74 second range. A future cost
audit should explain morphology-dependent work rather than treating cell count
as the full cost model.

## Outlier hypotheses, not verdicts

### Seed 8675309: broad elevated, low-relief candidate

This seed has the highest land p90 elevation (1.885 km), but the second-lowest
10-km relief p90 (418 m), lowest 25-km median relief (278 m), and near-lowest
25-km p90 relief (1,126 m). This is strong evidence for broad elevated smooth
terrain. It may be a plateau, high plain, overly broad mountain envelope or a
sampling/classification effect. Range/plateau objects and matched physical-scale
views must decide which.

The seed was a lifecycle plateau outlier historically, but this corpus uses the
legacy product model. The shared seed identity does not establish the same cause.

### Seed 9001: low-relief, low-peak world

Peak is 5.64 km, 10-km relief p90 is 385 m and 25-km relief p90 is 1,027 m, all
population minima. This may be legitimate diversity or insufficient tectonic/
erosional structure. Causal range counts and visual character are needed before
calling it bland or defective.

### Seed 12345: high short-scale relief

It has the highest 10-km p90 relief (1,324 m) and 25-km p90 relief (2,878 m), but
not the highest peak. This is evidence of stronger dissection/short-wavelength
structure rather than simply taller terrain. The former pillar case should be a
standing physical-versus-cartographic regression view.

### Climate/ecology ambiguity

Biome transition coverage spans 14.5–39.5%. This primarily diagnoses the
provisional classifier and normalized aridity basis. It cannot distinguish a
bad climate from appropriately broad ecotones. Climate correspondence controls
must precede biome calibration.

## System correspondence assessment

| System | Current correspondence claim | What it demonstrably preserves | What remains unestablished |
|---|---|---|---|
| Voronoi/adaptive mesh | numerical substrate | spherical topology, adjacency, area and targeted density | whether remeshing preserves every physical/semantic quantity appropriately |
| Plates and Euler dynamics | physically grounded kinematic abstraction | rigid rotational velocity and boundary-relative setting | force balance, realistic motion reorganization, diffuse deformation and plate-lifetime priors |
| Product tectonic features/elevation | authentic tectonic terrain hack with physical setting | convergence/divergence polarity, crust type, isostatic/thickness-inspired response and broad location | geological time, material budget as surface cause, belt-width scaling, inherited-versus-current forcing |
| Experimental lifecycle | stronger material/topology correspondence, not product terrain | explicit ownership, boundary work and conservative ledgers | resolution robustness and visible/semantic value of history; constant poles weaken its history claim |
| Atmosphere/moisture | mechanism-informed climatology proxy | latitude/elevation temperature, analytic circulation, wind transport, rainout and orographic terms | energy/water closure, wind-reversal response, seasonality, ocean heat transport and absolute climate calibration |
| Hydrology | physically grounded topology plus equilibrium hacks | drainage direction, basins, spill topology, lakes and catchment accumulation | dynamic storage, discharge semantics, remesh-stable basin/reach identity, wetlands and groundwater |
| Erosion | reduced landscape-evolution morphology | stream-power incision, hillslope transport, rerouting and some uplift coupling | shared geological time, complete mass/source-to-sink ledger, sediment cover, causal steady-state/control response |
| Ecology semantics | provisional interpretation | explicit continuous constraints and uncertainty | ecological calibration, seasons/soil/history and stability across climate controls |
| Rendering | cartographic presentation over physical state | separated relief and river exaggeration | complete transform provenance, physical/cartographic matched views and scale-aware landform/coast generalization |

## What metrics do not answer

- Earth-like peaks do not prove tectonic correspondence.
- Stable land fraction is partly datum construction, not an empirical success.
- River cell count is not physical river length, discharge or hierarchy quality.
- Lake fraction cannot distinguish many coherent lakes from one implausible sea.
- Local relief cannot identify a plateau or range without object geometry and
  causal placement.
- A biome transition percentage cannot validate climate or ecology.
- No current corpus metric measures “wow,” geographic identity, causal
  explainability or matched physical/cartographic legibility.

## Immediate evaluation priorities

1. Extract provisional range/plateau/peak/pass/ridge/divide objects and apply
   them to seeds 8675309, 9001 and 12345.
2. Produce matched physical, current-cartographic and hillshade-led views with a
   complete transform ledger for those seeds.
3. Run controlled correspondence batteries for climate, uplift–erosion,
   hydrology and remapping before changing their default mechanisms.
4. Add structured object and relationship metrics: tectonic attribution,
   catchment/reach stability, range geometry, stage survival and cost per object.
5. Use the results to decide whether existing uplift–erosion coupling needs
   validation, simplification or replacement; do not assume it is absent.
6. Design a sediment/source-to-sink v0 only after the existing erosion ledger
   and visible payoff are understood.

Feature expansion into vegetation, culture or sophisticated PBR remains paused.
Missing systems are ranked by causal and downstream leverage, not by scientific
completeness.

