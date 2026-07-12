# Reference corpus fine-budget audit — 2026-07-12

Status: accepted evaluation-budget decision, not a product-quality or model
promotion gate.

## Question

What fine-mesh cap is cheap enough for the standing ten-seed reference corpus
while preserving the terrain statistics and fixed-physical-scale relief needed
for evaluation?

## Reproducibility

- revision: `82fac5c96a6459bfcfa83453aeb053c91163dfd5` (clean);
- command: `cargo run --release --bin corpus -- --spec
  docs/corpora/reference-budget-v1.json`;
- seed: 12345;
- product coarse mesh: 100,000 cells, one Lloyd iteration;
- backend/model: convex-hull, legacy orogen;
- computed/viewed stage: 4/4;
- fine cache: disabled;
- requested fine caps: 250k, 1M, 4M;
- metric registry: v3;
- platform: WSL2 CPU run; no GPU/render evidence;
- artifacts: 88 KiB total without full world exports.

The 4M cap is a guardrail, not a requested density. The adaptive sampler
naturally stopped at 1,367,641 cells, so this rung represents the current
uncapped/default-density result for this seed.

## Results

| requested cap | actual cells | total s | fine-pre s | erosion s | land % | peak km | land p90/p99 km |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 250k | 256,847 | 26.8 | 1.3 | 20.6 | 26.350 | 9.033 | 1.171 / 3.891 |
| 1M | 997,650 | 84.0 | 6.8 | 72.6 | 25.968 | 9.050 | 1.220 / 3.984 |
| 4M cap | 1,367,641 | 113.0 | 11.8 | 95.9 | 25.901 | 9.057 | 1.233 / 4.011 |

### Resolution-sensitive terrain measurements

| requested cap | grade p50 | grade p90 | grade p99 | relief R10 p90 m | relief R25 p50 m | relief R25 p90 m |
|---:|---:|---:|---:|---:|---:|---:|
| 250k | 0.00135 | 0.00771 | 0.0838 | 1,064 | 652 | 2,920 |
| 1M | 0.00119 | 0.01019 | 0.0795 | 1,324 | 785 | 2,878 |
| 4M cap | 0.00115 | 0.01103 | 0.0777 | 1,323 | 782 | 2,828 |

Physical grade is a one-edge diagnostic and is expected to change as the edge
scale changes. It is not the convergence gate. Fixed-radius local relief is the
scale-controlled measurement.

The 250k mesh is inadequate for the reference corpus: 250k→1M increases 10-km
p90 relief by about 24.5% and 25-km median relief by about 20.4%. It is useful
for smoke and coarse control runs only.

The 1M→1.37M marginal changes are small:

- 10-km relief p90: -0.1%;
- 25-km relief p50: -0.3%;
- 25-km relief p90: -1.7%;
- land elevation p90/p99: +1.1% / +0.7%;
- land coverage: -0.067 percentage points;
- peak: +7 m;
- ecological transition coverage: -0.18 percentage points.

Runtime rises about 34.6% for 37.1% more cells. Erosion owns roughly 86% of the
1M runtime and remains the dominant corpus cost.

River cell count rises 18% between 1M and 1.37M. This does not overturn the
terrain decision: it is a cell-count metric whose resolution dependence is
already registered, not a physical river-length or topology convergence gate.
It reinforces the need to promote reach/object measurements before using river
counts for cross-resolution decisions.

## Instrumentation correction discovered during the sweep

The initial metric implementation used `acos(dot)` for neighbor angular
distance. Near-coincident fine cells caused f32 precision collapse and a false
physical-grade p99 of 108 at the highest rung. Metric registry v3 and ecological
terrain stress now use stable chord-to-arc conversion. The corrected p99 is
0.0777. The rejected value was an instrumentation artifact, not a terrain or
renderer result.

## Decision

Use an explicit 1,000,000 fine-cell cap for the ten-seed stage-4 reference
corpus. Keep 100,000 coarse cells and product defaults. Disable the fine cache
for first-pass cost comparability and record actual cells.

Expected sequential runtime is roughly fourteen minutes for ten seeds on the
measured machine, before exports or views. The cap is an evaluation budget, not
a product default and not proof of convergence for every subsystem. Systems
with finer characteristic scales may require targeted convergence corpora.

