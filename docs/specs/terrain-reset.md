# Terrain reset: physical ownership baseline

Status: product baseline reset implemented 2026-07-11; geological-time redesign pending.

## Why reset

The fine-terrain stack was optimized while relief rendering exaggerated elevation by
roughly 127x. Several systems were accepted because they repaired apparent pillars,
plateaus, softness, or missing structure under that presentation. At the authentic
cartographic scale, those observations no longer justify a cumulative mechanism stack.

This reset treats the old stack as experimental evidence rather than architecture.
A mechanism belongs in the product default only when it has a clear physical owner and
survives an isolated numerical A/B without borrowing credit from rendering.

## Ownership rule

- Coarse crust and tectonic forcing own the orogen envelope: width, mean elevation,
  broad taper, and whether a wide collision produces a plateau.
- Direct tectonic uplift supplies material during the erosion epoch.
- Stream-power incision and hillslope transport own valleys, drainage divides, and
  local relief.
- Lithology and climate feedback may redistribute erosion only after their fields and
  rates have an independently justified scale.
- Noise may break exact numerical symmetry, but must converge toward irrelevance; it
  may not own kilometres of final relief.
- Rendering owns legibility only and never supplies a terrain acceptance target.

## Reset product path

The product default now preserves the solved coarse tectonic surface and uses the
existing direct hold-and-carve source:

`uplift = EROSION_UPLIFT_SCALE * (tectonic_thickening + rift_delta)`

The following remain active because they are core process/correctness mechanisms:

- legacy coarse orogen response;
- Barnes flat resolution and SFD drainage;
- stream-power incision with `m=0.5`, `n=2`;
- linear hillslope diffusion;
- physical channel-initiation support;
- 200-step provisional mature epoch;
- hydrology integrated from the carved surface.

The following are parked neutral behind their existing flags:

- emergent demotion and self-calibrating target rebuild;
- O0 structured/global-normalized uplift redistribution;
- P1 interior height relief, strike bands, margin contrast, and fault scarps;
- Roering critical-slope modifier (physically motivated, numerically inert in the
  isolated reset A/B at the current mesh/parameter);
- synthetic lithology and structural-grain erodibility;
- fine erosion↔orographic-precipitation feedback and lake evaporation boost;
- en-route deposition calibration;
- meso fields, drainage pulse, rain shadow, MFD, confinement, and glacial carving.

Parking is not deletion. A parked mechanism can return only with a named physical role,
dimensioned or derived scale, isolated A/B, cross-seed stability, and resolution test.

## Numerical evidence

Seed 12345, 100k coarse cells, full fine mesh, all local additions neutral:

| Path | mountain land | max peak | cap area p50/p90 | flat-cap p50/p90 | relief 25/50/100 km |
|---|---:|---:|---:|---:|---:|
| Erosion only, no uplift | 5.7% | 5.9 km | 18,221/199,582 km² | 39/94% | 151/326/658 m |
| Direct uplift, n=1 | 10.2% | 9.1 km | 16,305/31,918 km² | 53/71% | 203/447/905 m |
| Direct uplift, n=2, linear hillslope | 7.8% | 9.1 km | 8,646/22,841 km² | 32/62% | 296/608/1,129 m |
| Direct uplift, n=2, Roering S_c=200 | 7.8% | 9.0 km | 7,680/21,816 km² | 29/61% | 298/607/1,109 m |

The genuine dune-to-structure improvement is primarily nonlinear incision plus direct
uplift. The Roering modifier is nearly inert. No procedural base relief or structured
target reconstruction is required for the measured dissection.

Cross-seed stripped direct-uplift results:

| Seed | mountain land | max peak | spire gate | relief 25/50/100 km |
|---:|---:|---:|:---:|---:|
| 12345 | 7.8% | 9.1 km | pass | 296/608/1,129 m |
| 777 | 9.7% | 9.3 km | pass | 183/414/832 m |
| 4242 | 8.3% | 8.9 km | pass | 186/410/850 m |

Large low-relief caps remain on some wide orogens. They are observations, not universal
failures: plateau plausibility must be conditioned on orogen width, drainage state, and
age rather than optimized away globally.

## Required upstream redesign: geological time

`EROSION_UPLIFT_SCALE`, `EROSION_K`, diffusivity, and `steps * dt` currently form a
dimensionless regime. The next architectural task is to give them a shared clock:

1. assign plate boundaries/orogens an age or active duration;
2. express convergence-derived crustal supply as a rate;
3. express incision and hillslope transport against the same time unit;
4. distinguish growing, flux-balanced, and decaying orogens;
5. make plateaus conditional outcomes of forcing width, drainage/base level, and age;
6. verify resolution convergence at fixed physical duration.

Until then, 200 steps is a provisional numerical maturity budget, not a geological age.
