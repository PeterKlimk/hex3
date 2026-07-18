# Numerical instrumentation inventory

Status: code/document synthesis updated 2026-07-19. This inventory describes what
exists; it does not endorse historical thresholds.

## Diagnostic surface

`diagnose` is both a default world report and an experimental harness. It
supports dedicated audits for tectonic history, rebuild fidelity, drainage,
lakes, detail survival, mountains, rivers, biome proxies and cross-resolution
pilot predictiveness. Most audits require full fine generation; tectonic history
exits at the coarse stage.

| Audit | Principal populations and measurements | Main cautions |
|---|---|---|
| Tectonic history | plate-pair episodes, boundary length, velocity, duration, displacement, lifecycle ledgers and event footprints | mixes episode evidence with lifecycle experimental state |
| Rebuild fidelity | exact active-builder components, target/final peaks and volume, subsidy/tax and overshoot | only audit with an executable warn/fail threshold; applies to an experimental builder |
| Drainage | coarse/fine endorheic land, lakes, basins and lake capability | reconstructs overflow chains outside shared water semantics |
| Lakes | semantic lakes/ponds, depth, area, shape, outlet, catchment and climate-dial response | component-to-semantic-object correspondence is assumed; some object summaries are count-weighted |
| Detail survival | fixed tectonic footprint across coarse, fine base and erosion; support, volume and relief | footprint is relative to each world's maximum forcing |
| Mountains | elevation-mask components, range geometry, peaks, passes, profiles, relief spectra and drainage grain | “mountain” is a threshold mask, not shared semantics; many nested thresholds |
| Rivers | shared All/Major masks, Strahler/Horton structure, mouths and trunk profiles | length is cell-area approximation; default report still uses incompatible river populations |
| Biomes | shared seasonless potentials, broad labels, uncertainty and component coherence | provisional classifier, not calibrated ecology |
| Resolution pilot | short solved-surface response versus native-reference slope, neighbour relief and selected-channel heads/junctions; global and within-density-band area-budget capture | predictor evidence only, not a remeshing result; native reference inherits the current density prior and cannot reveal regions it never sampled |

The default report additionally covers mesh resolution, erosion incision,
hypsometry, Moran's I, continent/island masks, tectonic placement, climate and
aridity, river grading, adaptive-density allocation, fixed-radius relief and a
roughness/plateau probe.

## Definition collisions

### Mountains, ranges and plateaus

- The common mountain/range mask is elevation at least `0.15`, currently 1.5 km.
- Significant mountain components commonly require 20,000 km².
- Lifecycle reporting also uses a 2 km mountain threshold.
- Terrain-density bins, summit probes and default range reports repeat related
  constants rather than consuming one semantic definition.
- “Plateau” is not an object. The main current proxy selects the top elevation
  decile inside an absolute mountain mask and examines steepest downhill slope.
  Historical cap analysis instead measures areas within 0.5/1.0 km of each
  component summit and calls cells below 1% physical grade flat.

These definitions can conflate elevated continental interiors, broad orogenic
envelopes, unresolved relief and genuine plateaus. They are high-priority audit
targets because presentation exaggeration previously made a plausible physical
slope look pillar-like.

### Rivers

The dedicated audit uses shared `RiverNetwork` semantics and a 2,000 km² product
catchment threshold. The default report also uses:

- cells above 1% of the world's maximum flow;
- count-equivalent flow thresholds of 50 or 100;
- up to 50 long-profile rivers above 200 km;
- an explicitly unreliable population slope-area fit.

“River” therefore still names incompatible populations inside one executable.
The aggregate long-profile bow/profile metric remains useful historical
evidence; the population slope-area metric is explicitly superseded.

### Climate and ecology

The current aridity index is precipitation divided by a normalized
temperature-dependent demand, then normalized again to each world's land mean.
It describes within-world spatial water stress but cannot preserve absolute
dryness differences across worlds. Earth aridity thresholds applied afterward
are therefore analogies, not direct empirical comparisons.

Biome labels consume the same style of normalized aridity plus terrain and
freshwater potentials. Classification confidence exposes ambiguity, but the
labels have not passed cross-seed calibration or control-response gates.

## Units and weighting risks

- Shared component geometry generally uses physical km and km².
- Elevation now uses exactly 10 km per model unit.
- Several reported “slope” quantities are elevation units per km rather than
  physical grade. Historical reports may predate the current unit contract.
- Many volumes, drainage fractions and ecological summaries are area-weighted.
- Several global quantiles, temperature means, aridity fractions, causal
  fractions and summit populations remain cell-weighted. On the adaptive fine
  mesh these do not estimate planetary area.
- Channel length often sums square roots of selected cell areas. This is a
  useful approximation but is mesh-shape and resolution sensitive.
- Relative-to-world-maximum masks for arcs, trenches and process footprints can
  change extent when only one extreme value changes.
- Fine “margin distance” in the default report is reconstructed from
  continentality and is not a retained geometric distance field.

Every registry entry must state units and weighting explicitly. Existing output
headers that imply all fractions are area-weighted should be verified per field.

## Historical and current gates

| Rule | Present interpretation |
|---|---|
| tectonic peak >12 km warn, >14 km fail | historical user-calibrated anomaly guard; not an Earth limit |
| carrier peak drift >2 km or land drift >2 percentage points | standing experimental robustness/convergence rule; intended claim needs clarification |
| rebuild final-target overshoot >2 km warn, >4 km fail | executable experimental-builder gate |
| material residual near roundoff | invariant evidence, but tolerance should be registered per ledger |
| plateau cap area/flat-cap fractions | descriptive historical metrics; no accepted thresholds |
| river long-profile bow/concavity | descriptive correctness evidence; no maintained pass threshold |
| population slope-area concavity | superseded/unreliable |
| land fraction 0.26 invariant | historical by-construction behavior, not a maintained empirical gate |

Archived PASS/WARN/FAIL labels from one- or two-seed sweeps remain local
experiment decisions unless they meet the current validation policy.

## Corpus-relevant seeds and axes

- Standing ten seeds: `12345, 777, 4242, 9001, 314159, 271828, 8675309,
  20260711, 42, 1001`.
- Lifecycle carrier axis: 4096/8192/16384 cells on a fixed 100,000-cell terrain.
- Historical coarse statistical axis: 50k/100k/200k/400k for seeds 12345/777;
  changing coarse resolution regenerates geography, so cellwise same-seed
  convergence is invalid.
- Historical fine axis: scale 2.0 through 0.65 on fixed coarse geography,
  roughly 363k to 3.37M cells.
- Important outliers: lifecycle plateaus `8675309` and `1001`; pillar and
  resolution continuity case `12345`; historical carrier divergence `4242`.

## Highest-priority registry work

1. Separate area-weighted planetary distributions from cell/sample summaries.
2. Register all slope and relief quantities against the unit contract.
3. Split mountain-mask, range-component, summit-cap and plateau-proxy meanings.
4. Retire or quarantine default-report river definitions that bypass shared
   semantics.
5. Clarify whether carrier-resolution rules claim continuum convergence or
   robustness across accepted procedural resolutions.
6. Attach tolerance and control response to every invariant/gate.
7. Add missing relationship and product-character metrics rather than expanding
   isolated field summaries indefinitely.
