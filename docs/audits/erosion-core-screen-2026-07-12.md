# Erosion core screening ablation — 2026-07-12

Status: immutable two-seed screening evidence. It rejects neutral simplification
claims; it does not promote an erosion configuration.

## Question

Can the active erosion core be made substantially cheaper by reducing 200 steps
to 50/100 or replacing the `n=2` stream-power slope exponent with the closed-form
`n=1` path without materially changing terrain structure?

## Reproducibility

- revision: `644fded82f49e933f968d6c36a0c44e6b6b82e28` (clean);
- command: `cargo run --release --bin corpus -- --spec
  docs/corpora/erosion-core-screen-v1.json --force`;
- seeds: 12345 and broad/low-relief candidate 8675309;
- model/backend: product legacy, convex-hull;
- resolution: 100,000 coarse cells, 250,000 fine cap;
- stage/cache: 4, cache disabled;
- metric registry: v3;
- purpose: cheap screen; 250k is known to under-resolve 10–25 km relief.

## Results

### Seed 12345

| configuration | erosion s | peak km | land p90/p99 km | grade p90 | R10 p90 m | R25 p50/p90 m |
|---|---:|---:|---:|---:|---:|---:|
| 200 steps, n=2 | 19.9 | 9.03 | 1.17 / 3.89 | 0.0077 | 1,064 | 652 / 2,920 |
| 50 steps, n=2 | 5.5 | 6.77 | 1.23 / 3.21 | 0.0068 | 680 | 394 / 2,020 |
| 100 steps, n=2 | 10.4 | 7.53 | 1.20 / 3.42 | 0.0068 | 883 | 484 / 2,430 |
| 200 steps, n=1 | 10.2 | 9.06 | 1.33 / 4.39 | 0.0109 | 617 | 524 / 2,017 |

### Seed 8675309

| configuration | erosion s | peak km | land p90/p99 km | grade p90 | R10 p90 m | R25 p50/p90 m |
|---|---:|---:|---:|---:|---:|---:|
| 200 steps, n=2 | 15.1 | 6.65 | 1.74 / 3.92 | 0.0082 | 91 | 230 / 1,101 |
| 50 steps, n=2 | 4.3 | 5.05 | 1.67 / 3.14 | 0.0068 | 68 | 157 / 567 |
| 100 steps, n=2 | 10.1 | 5.60 | 1.66 / 3.40 | 0.0071 | 76 | 180 / 801 |
| 200 steps, n=1 | 22.6 | 6.72 | 2.09 / 3.94 | 0.0098 | 83 | 208 / 716 |

## Interpretation

### Step-count simplification is not neutral

At 100 rather than 200 steps:

- seed 12345 loses about 17% of R10 p90 relief, 26% of R25 median relief and
  17% of R25 p90 relief; peak falls 1.50 km;
- seed 8675309 loses about 17% of R10 relief, 21% of R25 median relief and 27%
  of R25 p90 relief; peak falls 1.05 km.

The direction is consistent across two very different worlds. Fifty steps
diverges further. The default 200-step state is not a converged physical-time
solution, but 50/100 are not arithmetic-preserving cost reductions.

Do not reduce the product default on timing evidence alone. A 150-step rung may
still offer a useful calibrated trade, but would be a changed landscape regime
requiring object and visual review.

### `n=1` is a different morphology

At 200 steps, `n=1` preserves similar peaks but raises neighbor-grade p90 and
upper elevation tails while reducing fixed-radius relief substantially:

- seed 12345 R10/R25-p90 fall about 42%/31%;
- seed 8675309 R10/R25-p90 fall about 10%/35%, while land p90 rises from 1.74 to
  2.09 km.

This combination is consistent with steeper local edges but less organized
relief at fixed scales. It may worsen broad elevated terrain, but range/plateau
objects and matched views are required.

`n=1` timing is not yet established: it was faster for seed 12345 and slower for
8675309 in the clean run. Single wall-clock observations mix morphology,
routing, allocation and machine noise. Performance promotion requires repeated
per-phase measurements.

## Decision

- Retain 200 steps and `n=2` as the product baseline for now.
- Reject 50/100 steps and `n=1` as drop-in simplifications.
- Do not spend a 1M confirmation on equivalence: the 250k differences are
  already too large.
- Revisit 150 steps only as an explicit quality/cost trade after landform
  semantics and matched views exist.
- Prioritize arithmetic-neutral allocation/buffer improvements and diffusion/
  deposition ablations next; these have a better chance of reducing cost
  without silently selecting a different landscape.

