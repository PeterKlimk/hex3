# Moving-carrier tectonic promotion scorecard — 2026-07-11

The `tectonic_scorecard` binary audits the coarse geological state before fine
erosion or rendering. It compares the reset legacy baseline with
`history-carrier-evolved`, sweeps carrier resolution independently of the fixed
100,000-cell terrain mesh, and reports absolute plausibility separately from
resolution convergence.

Run the standing ten-seed, three-resolution scorecard with:

```bash
cargo run --release --bin tectonic_scorecard
```

The default sweep is intentionally headless: ten seeds, 4096/8192/16384 carrier
cells, a 2 Myr snapshot interval, and no atmosphere/fine generation.

## What is measured

- absolute peak and area-weighted land hypsometry;
- mountain-land coverage above 1.5 km;
- significant-range count, median width/elongation, and summit-cap geometry;
- positive and negative crust-work volume;
- explicit arc addition as a fraction of positive thickening;
- inherited positive thickness outside the cells carrying 90% of present uplift;
- area-weighted inherited/current-field cosine similarity;
- moving-forcing, gap/overlap, mass-residual, and runtime ledgers;
- per-seed drift across carrier resolution.

The absolute peak gate retains the user-calibrated terrain thresholds: >12 km is
a warning and >14 km is a failure. The convergence gate fails when carrier
refinement moves a peak by >2 km or land coverage by >2 percentage points.

## Resolution result: failed

All runs below use a fixed 100,000-cell terrain mesh. Land coverage is invariant,
but peak height, mountain coverage, inherited relief, and range fragmentation are
not. The instability therefore belongs to carrier deformation/projection rather
than sea-level solving.

| seed | carrier peaks 4k/8k/16k | peak span | mountain-land 4k/8k/16k | inherited outside active 4k/8k/16k | verdict |
|---:|:---|---:|:---|:---|:---|
| 12345 | 13.48 / 13.92 / 19.08 km | 5.60 km | 42.9 / 47.1 / 48.6% | 66.3 / 74.0 / 77.4% | FAIL |
| 777 | 8.98 / 11.99 / 13.09 km | 4.11 km | 47.7 / 49.3 / 51.4% | 71.7 / 77.5 / 81.0% | FAIL |
| 4242 | 10.90 / 16.83 / 30.28 km | 19.38 km | 40.0 / 40.6 / 44.1% | 67.7 / 74.9 / 76.3% | FAIL |

The reset legacy baseline for these seeds peaks at 5.85–6.17 km with 10.5–12.9%
mountain land. The moving model produces 33–51% mountain land and roughly
80–194 significant coarse ranges. This is not merely an isolated summit-tail
problem: the evolved world is systematically too thickened and fragmented.

Mass residuals remain `9.5e-7`–`1.4e-5` relative to explicit arc addition, well
inside the conservation gate. Positive/negative crust-work volume increases with
carrier resolution, while fixed land fraction hides that redistribution in the
sea-level offset. Conservation is working; spatial concentration is not converged.

## Ten-seed 8192-cell result: failed

| seed | peak | mountain-land | ranges | arc / positive work | inherited outside active | verdict |
|---:|---:|---:|---:|---:|---:|:---|
| 12345 | 13.92 km | 47.1% | 125 | 55.9% | 74.0% | WARN |
| 777 | 11.99 km | 49.3% | 110 | 49.1% | 77.5% | PASS |
| 4242 | 16.83 km | 40.6% | 128 | 53.9% | 74.9% | FAIL |
| 9001 | 18.28 km | 37.1% | 159 | 51.9% | 75.3% | FAIL |
| 314159 | 15.13 km | 46.8% | 103 | 25.7% | 81.3% | FAIL |
| 271828 | 9.74 km | 40.5% | 145 | 39.8% | 72.6% | PASS |
| 8675309 | 18.65 km | 44.1% | 128 | 46.9% | 71.7% | FAIL |
| 20260711 | 20.92 km | 41.0% | 146 | 51.9% | 72.6% | FAIL |
| 42 | 9.89 km | 35.9% | 129 | 44.6% | 66.9% | PASS |
| 1001 | 26.81 km | 32.9% | 121 | 49.0% | 70.5% | FAIL |

Six of ten seeds fail the absolute peak gate, one warns, and three pass. Median
range widths cluster around 227–254 km and median elongation around 1.8–2.1,
indicating numerous compact blobs rather than a small number of long belts.

## Decision

`history-carrier-evolved` is not ready for erosion work or promotion. Adding
denudation now would compensate for an upstream discretization failure and repeat
the render-scale tuning mistake.

The next mechanism should make the carrier deformation operator resolution-
independent before changing any physical coefficient:

1. audit boundary traction integrated per physical kilometre at each carrier
   resolution;
2. audit parcel volume distribution/gather as ownership support changes;
3. audit carrier-to-100k projection for extrema and area-weighted volume;
4. normalize extensive boundary work by dual face length/receiving area exactly
   once;
5. rerun this scorecard unchanged.

Do not tune arc retention, stress-transmission length, mobility, or erosion against
the current peaks until those operator audits close.

## Operator-isolation follow-up

The scorecard now runs the requested five-rung ladder only when its runtime carrier
configuration enables `operator_audit`; normal experimental worlds do not pay this
cost. Seed 12345 at a fixed 100k terrain mesh gives:

| carrier | mean boundary length | convergent swept area / Myr | boundary support | one-step max | frozen-100 max | moving-native max | projected max | projection net residual |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 416,671 km | 3.98M km² | 36.0% | 0.127 | 2.407 | 2.018 | 2.018 | −1.66e-3 |
| 8192 | 513,035 km | 4.48M km² | 30.4% | 0.182 | 2.498 | 2.253 | 2.253 | +1.59e-3 |
| 16384 | 646,608 km | 4.74M km² | 26.2% | 0.249 | 3.001 | 2.834 | 2.834 | +8.04e-4 |

The first divergent rung is the single 2 Myr step from uniform crust: its maximum
thickness response spans 0.122 before parcel history, while total positive/negative
one-step volumes remain relatively close. The boundary becomes 55% longer and its
support occupies a shrinking fraction of the mesh as resolution increases. Long-time
integration amplifies that localized response, but does not create it.

Carrier-native and projected maxima are identical at every resolution. Projection
changes net volume by only ~1e-3 thickness·steradian and is therefore not the primary
cause. Parcel gather/distribute also cannot be the first cause because it is absent from
the already-divergent one-step rung.

The next fix belongs in boundary forcing: define traction on edges as a finite-volume
boundary flux or regularize it over a fixed physical fault-zone width before solving
sheet velocity. It must make integrated swept area and maximum one-step response
converge without changing plate speed, arc retention, mobility, or erosion.

### Exact boundary-velocity A/B: falsified and reverted

A constrained screened-Laplace experiment fixed every receiver-boundary cell to its
closure-derived velocity and solved only the free interior. This removed the shrinking
volumetric-source interpretation without adding a traction coefficient. It worsened
every gate:

| carrier | one-step max | frozen-100 max | final coarse peak |
|---:|---:|---:|---:|
| 4096 | 0.176 | 3.548 | 14.23 km |
| 8192 | 0.232 | 5.275 | 17.72 km |
| 16384 | 0.505 | 6.623 | 20.34 km |

The one-step span increased from 0.122 to 0.330 and the frozen-100 span from
0.594 to 3.075. Exact constraints preserve every increasingly jagged boundary-cell
corner and therefore concentrate, rather than regularize, deformation. The experiment
was reverted; `history-carrier-evolved` retains its prior volumetric-source result.

This A/B rules out raw per-cell Dirichlet forcing. The next candidate must first define
a resolution-independent physical boundary geometry—e.g. fixed-scale chain
regularization or a finite-width forcing measure—before either Dirichlet velocity or
Neumann traction can converge.

### Fixed-width boundary-band A/B: falsified and reverted

A 440 km compact band (the existing sheet stress-transmission length) propagated
closure velocity inward on each plate/crust domain. The first nearest-seed version
created internal Voronoi seams and produced a 0.301 one-step span. Blending nearby
boundary directions by edge length removed those seams but still produced:

| carrier | one-step max | transport-only max | magma-only max | final coarse peak |
|---:|---:|---:|---:|---:|
| 4096 | 0.197 | 0.197 | 0.024 | 14.12 km |
| 8192 | 0.250 | 0.250 | 0.031 | 13.89 km |
| 16384 | 0.336 | 0.336 | 0.059 | 20.85 km |

The blended one-step span is 0.139, still worse than the original 0.122, and its
terrain peak span is 6.96 km. Source attribution shows the one-step maximum is
transport-dominated, not arc-magma dominated. The band machinery was removed; only
the transport/magma attribution remains in the standing ladder.

At this point 4096/8192/16384 are better understood as different physical procedural
resolutions, not a convergent discretization family: a 440 km deformation gradient is
barely one carrier edge at 4096 cells. Hex3 does not require a physics-grade continuum
limit. The honest near-term choice is to retain the canonical 8192-cell (~250 km)
carrier, keep its resolution failure visible, and assess whether same-clock denudation
regulates the otherwise authentic moving history. Do not claim numerical convergence.
