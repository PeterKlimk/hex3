# Tectonic source-viability audit

Status: **completed bounded source gate**, 2026-07-21. This audit is evidence,
not product architecture or a terrain promotion.

## Question

Can the retained default tectonic history honestly drive the time-aware forcing
assumed by Terrain Slice A, or would that slice begin by fabricating history?

## Method

Revision `362c670` adds a research-only extension to the existing
`diagnose --tectonic-history-audit`. It collects exact great-circle arcs from
the current convergent-front compiler, associates them with retained boundary
episodes and emits deterministic JSON. For the fixed seeds `12345`, `8675309`
and `9001`, each world used 100,000 requested and actual coarse cells:

```bash
cargo build --release --features research-landscape --bin diagnose
target/release/diagnose --seed 12345 --cells 100000 \
  --tectonic-history-audit \
  --tectonic-history-audit-out artifacts/tectonic-source-viability-v1/seed-12345.json
```

The same command was repeated for the other two seeds. All artifact provenance
records revision `362c670375e87d3cf34e364b8a2961cfbf7e7108`, the convex-hull
backend and the Legacy model.

The audit derives onset intervals from unique convergent-component durations.
An interval contains the exact present support of every episode old enough to
be active. Opportunity is exact arc length times positive local convergence
rate times duration. Adjacent normalized L1 compares opportunity composition;
rank-one residual tests whether a frame is merely a global rescaling of the
present field. The `0.10` L1 classification threshold is a screening heuristic,
not a physical constant.

## Result

| Seed | Present convergent components | Derived frozen-support intervals | Integrated shortening-area opportunity (km²) | Largest component | Max adjacent L1 | Max rank-one residual |
|---:|---:|---:|---:|---:|---:|---:|
| 12345 | 17 | 12 | 1.604e8 | 27.7% | 1.074 | 0.963 |
| 8675309 | 14 | 10 | 1.615e8 | 31.1% | 0.528 | 0.891 |
| 9001 | 12 | 8 | 1.370e8 | 33.5% | 0.989 | 0.971 |

All three worlds pass the narrow static-onset screen: differently aged present
components produce substantial planet-wide changes in the active opportunity
composition. This is more informative than one global uplift-amplitude scalar.

All three fail the moving/reversing-support prerequisite. The default history
retains zero historical spatial opportunity frames. Within each connected
component, forcing through time is exactly rank-one: its present geometry,
local rates, crust setting, regime, polarity and receiver remain fixed. Large
global composition-centroid shifts arise when separate stationary components
activate and must not be called front migration.

## Architectural consequence

Terrain Slice A may proceed only as a finite-age, frozen-support coupled
landscape test. It can test unequal belt maturity, incomplete adjustment,
drainage competition, divide movement and hillslope response. It cannot test
range-internal forcing migration, abandonment, polarity reversal or material
transition.

If that bounded slice creates coherent and varied dissection, richer tectonic
history is not a prerequisite for the first replacement. If each component
still becomes an internally uniform roof despite responsive drainage, the next
causal intervention is an upstream deformation-history owner or the explicit
topology upper bound—not finer cells or a response-parameter campaign.
