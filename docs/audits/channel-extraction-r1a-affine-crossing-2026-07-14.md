# R1a affine continuous-crossing audit

**Date:** 2026-07-14
**Status:** completed incomplete causal discriminator; X0 passes, X1 not fully
evaluable
**Specification:** [R1a affine continuous-crossing discriminator](../research/channel-extraction-r1a-affine-crossing-2026-07-14.md)
**Parent evidence:** [R1a affine generator-point control audit](channel-extraction-r1a-generator-control-2026-07-14.md)

## Verdict

The analytic continuous-crossing arm X0 passes all 12 registered affine cases.
The checked Voronoi segment substrate, semantic portal and entry-point-aware
ray traversal are therefore valid for registered A. The earlier affine failure
is not inherent to crossing this cap continuously.

The paired causal claim remains incomplete. The frozen literal all-cell
centroid reconstruction exceeds its `1e-10` linear-exactness prerequisite in
one case, `h=2 km, theta=0.31, delta=0`. Its maximum relative vector error is
`1.101602e-10` and maximum sine-angle error is `1.031036e-10`. X1 is correctly
not constructed or judged for that case. It reaches portal 401 and passes every
geometry gate in the other 11 cases, always with the same crossed-face sequence
as X0.

This is not an X1 traversal failure and does not reject continuous crossing.
It rejects the frozen reconstruction as a complete 12-case manufactured
solution at the registered tolerance. Do not relax the tolerance or run the
censored trace post hoc. Because reconstruction is an ordered prerequisite,
the result does not yet establish that the maximum-face plus F0 graph
representation is the load-bearing bundle, even though the 11 evaluated pairs
strongly support that hypothesis.

## Frozen result

| spacing | X0 gate | reconstruction gate | judged X1 gate | maximum judged X1 cross-track | judged X1 relative-length range |
|---:|---:|---:|---:|---:|---:|
| 8 km | 4 / 4 | 4 / 4 | 4 / 4 | `7.745e-13 km` | `1.770–2.301%` |
| 4 km | 4 / 4 | 4 / 4 | 4 / 4 | `1.435e-12 km` | `0.288–0.512%` |
| 2 km | 4 / 4 | 3 / 4 | 3 / 3 | `4.999e-12 km` | `0.105–0.328%` |

All 12 X0 paths reach the required portal, have zero backtracking, remain at
the numerical cross-track floor and have relative arclength error below 5%.
All 11 judged X1 paths do the same. Their face sequences equal X0 exactly;
maximum paired crossing-coordinate differences range from about `4.73e-13` to
`2.11e-11 km`. Endpoint differences are at most `5.23e-12 km`. These tiny
differences are descriptive rather than a selected post-hoc agreement gate.

Portal intersections deliberately differ from the nominal analytic outlet.
At 8 km the endpoint offset is `3.12–4.05 km`; at 4 km it is `0.51–0.90 km`;
at 2 km it is `0.17–0.58 km`. The arclength trend and signed terminal positions
show ordinary boundary-segment discretization, not lateral path drift.

The internal-score equivalence gate passes all 12 cases with zero eligibility
conflicts and zero winner conflicts. The smallest observed absolute generator
score is `4.338e-8`; the maximum symmetric normalized score error is
`2.118e-6`. This establishes equivalence of the two internal ranking formulas
on the registered matrix, but it cannot override the failed reconstruction
prerequisite or extend to portal scoring.

## What was validated

The test-only implementation:

- exactly maps every cyclic polygon segment to one CSR or boundary face and
  consumes every stored face once;
- checks reciprocal owners and bit-equal segment endpoints;
- reconstructs polygon centroids and areas independently;
- uses all CSR neighbors in fixed order for the frozen literal normal-equation
  solve;
- retains actual intersection points without midpoint substitution or nudging;
- checks the receiving-cell inward predicate and reciprocal containment;
- preserves typed boundary, degeneracy, cycle and guard failures; and
- repeats contexts, reconstructions, score audits and traces bit-identically
  without mutating the registered cap or case.

An independent implementation review found no load-bearing geometry,
reconstruction-formula, ranking, traversal, metric or censoring defect. It did
identify and correct a reporting ambiguity: `reconstruction not judged` and
`traversal attempted but failed` are now separate counts. The observed matrix
contains one of the former and zero of the latter.

## Consequence and next rung

The evidence separates three facts that should not be collapsed:

1. continuous crossing with the analytic affine direction is sound on this
   cap;
2. the chosen centroid-gradient formula is accurate enough to produce 11
   essentially identical paths, but misses its deliberately strict all-domain
   numerical contract once; and
3. internal maximum-face choices are equivalent under the generator and
   reconstructed formulations throughout the matrix.

The smallest principled follow-up is numerical linear consistency, not a new
river mechanism and not terrain tuning. Preregister one stable affine
reconstruction control that preserves the same polygon means, centroids,
neighbors and crossing code while changing only the numerical solve/input
evaluation convention. It should distinguish loss in polygon-moment/elevation
differences from loss in the 2x2 normal-equation solve and must retain an
all-cell gate. Candidate stable methods should be justified by standard
least-squares practice rather than selected to make this one case pass.

Do not implement V, RT0, discharge integration, confluence, persistence or a
product river extractor from this incomplete result. If a preregistered stable
reconstruction passes and the same X1 geometry passes 12/12, the original
maximum-face/F0 bundle localization becomes warranted. If it does not, the
input/reconstruction semantics remain load-bearing and need resolution first.

The next rung was
[preregistered](../research/channel-extraction-r1a-stable-reconstruction-2026-07-14.md)
as a row-input × solve factorial. It preserves this audit's geometry and gates
and does not weaken the censored result.

Later evidence: the
[stable reconstruction audit](channel-extraction-r1a-stable-reconstruction-2026-07-14.md)
finds that QR does not rescue registered rows, while both direct affine-row
oracle solves pass 12/12. Registered mean/difference numerics remain the formal
blocker; no crossing or score failure is observed.

## Executed checks

```bash
cargo test --lib channel_extraction_r1a_affine_crossing -- --nocapture
cargo test --lib world::landscape::channel_extraction_r1a_affine_crossing::registered_affine_crossing_reproduces_frozen_incomplete_matrix -- --ignored --exact --nocapture
```

Routine tests pass. Independent full-matrix runs reproduce the same
classification; the first completed in `49.78 s` and the independent review in
`58.57 s`. A passing harness means it deterministically reproduces this
incomplete causal result, not that X1 passed the 12-case promotion gate.
