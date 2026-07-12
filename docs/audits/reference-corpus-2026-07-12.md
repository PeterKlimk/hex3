# Product-baseline reference corpus — 2026-07-12

Status: immutable numerical evidence. Interpretation is maintained in
[`../evaluation-synthesis.md`](../evaluation-synthesis.md).

## Reproducibility

- revision: `cb506c07f971ec5792661b1aad9c852c3274d943` (clean);
- command: `cargo run --release --bin corpus -- --spec
  docs/corpora/reference-v1.json`;
- model/backend: product `legacy`, convex-hull;
- stage: 4/4;
- resolution: 100,000 coarse cells, 1,000,000 fine cap, actual cells recorded;
- cache: disabled;
- metric registry: v3;
- platform: WSL2 CPU release build;
- generated artifact root: `artifacts/evaluation/reference-v1` (264 KiB,
  ignored by Git; reproducible from the specification).

All ten runs completed. No images or full world exports belong to this audit.

## Per-seed results

| seed | cells | runtime s | land % | peak km | land p90/p99 km | R10 p90 m | R25 p50/p90 m | lake % terrestrial | biome transition % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12345 | 997,650 | 87.9 | 25.968 | 9.05 | 1.22 / 3.98 | 1,324 | 785 / 2,878 | 3.30 | 20.6 |
| 777 | 995,430 | 68.9 | 26.384 | 9.26 | 1.47 / 4.44 | 1,041 | 558 / 2,333 | 1.64 | 21.3 |
| 4242 | 995,967 | 72.2 | 26.207 | 8.87 | 1.22 / 4.52 | 771 | 516 / 1,882 | 1.25 | 38.6 |
| 9001 | 984,164 | 73.7 | 26.014 | 5.64 | 0.99 / 3.74 | 385 | 410 / 1,027 | 1.65 | 39.5 |
| 314159 | 995,524 | 81.4 | 26.626 | 7.44 | 1.55 / 3.69 | 827 | 380 / 1,866 | 2.37 | 14.5 |
| 271828 | 994,735 | 77.4 | 26.217 | 8.39 | 1.22 / 3.25 | 1,201 | 872 / 2,437 | 2.08 | 34.3 |
| 8675309 | 996,032 | 65.1 | 26.216 | 6.70 | 1.88 / 3.93 | 418 | 278 / 1,126 | 1.02 | 22.3 |
| 20260711 | 994,539 | 78.6 | 26.307 | 10.32 | 1.23 / 4.10 | 999 | 456 / 2,335 | 0.44 | 21.7 |
| 42 | 998,536 | 73.8 | 25.901 | 6.83 | 1.05 / 2.89 | 723 | 405 / 1,566 | 1.35 | 29.3 |
| 1001 | 994,432 | 64.0 | 26.756 | 7.97 | 1.51 / 4.63 | 794 | 380 / 1,980 | 0.08 | 35.2 |

R10/R25 are max-minus-min local relief at 10/25 km radius over a deterministic
sample of cells above 1.5 km. Biome transition is area below provisional
classification confidence 0.20; it is classifier evidence, not ecological
validation.

## Population summaries

| measurement | min | median | mean | max |
|---|---:|---:|---:|---:|
| runtime s | 64.0 | 73.8 | 74.3 | 87.9 |
| land % | 25.90 | 26.22 | 26.26 | 26.76 |
| peak km | 5.64 | 8.39 | 8.05 | 10.32 |
| land elevation p90 km | 0.99 | 1.23 | 1.33 | 1.88 |
| land elevation p99 km | 2.89 | 3.98 | 3.92 | 4.63 |
| R10 relief p90 m | 385 | 827 | 848 | 1,324 |
| R25 relief p50 m | 278 | 456 | 504 | 872 |
| R25 relief p90 m | 1,027 | 1,980 | 1,943 | 2,878 |
| lake / terrestrial area % | 0.08 | 1.64 | 1.52 | 3.30 |
| biome transition area % | 14.5 | 29.3 | 27.7 | 39.5 |

Median stage times were approximately 3.1 s lithosphere, 1.6 s atmosphere,
7.5 s fine-pre and 62.2 s erosion. Erosion is the dominant cost.

No table entry is a maintained promotion threshold. Land fraction is partly
datum-controlled; peak is an extreme; river cells are resolution-dependent;
lake coverage needs object interpretation; and ecological labels are
provisional.

