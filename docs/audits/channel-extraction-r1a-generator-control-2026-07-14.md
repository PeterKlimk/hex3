# R1a affine generator-point control audit

**Date:** 2026-07-14
**Status:** completed negative causal control; both receiver arms still reject
**Specification:** [R1a affine generator-point causal control](../research/channel-extraction-r1a-generator-control-2026-07-14.md)
**Parent result:** [R1a path audit](channel-extraction-r1a-path-2026-07-14.md)

## Verdict

Changing only the affine state from exact polygon means to exact values at the
Voronoi generators is insufficient. It repairs none of the registered
termination failures and neither P0 nor M0 passes the affine measuring gates.
The generator-sampled control therefore does not rescue either receiver or
change R1a's “select neither” decision.

The earlier polygon-mean/generator-geometry mismatch is numerically material
but is not established as a load-bearing cause of the frozen rejection. P0's
six common successful traces all improve in cross-track and arclength, but P0
still fails every rotated registration and still misses the finest-grid cross-
track and length gates. M0 becomes worse overall: it loses the two rotated 2 km
terminations that polygon means happened to reach.

The leading remaining explanation is the reduction of a distributed local
field to one maximum-face graph walk. On affine generator samples, internal
Voronoi two-point P0 grades already recover the analytic face-normal derivative
up to stored-geometry precision, yet the selected walk can accumulate large
lateral displacement and reach a closed cut. The control does not by itself
fully separate face argmax, boundary interaction and path encoding, so that
claim remains a mechanism diagnosis rather than a promoted replacement.

## Frozen result

Independent per-representation summaries retain R1a's successful-trace
censoring and report termination separately.

| representation | arm | portal failures | 8 km successful cross-track | 2 km successful cross-track | 2 km length-error range | aggregate A result |
|---|---:|---:|---:|---:|---:|---:|
| polygon mean | P0 | 6 / 12 | 18.219–18.919 km | 8.664–9.364 km | 6.824–6.915% | fail |
| polygon mean | M0 | 4 / 12 | 8.795–30.774 km | 4.786–7.338 km | 3.345–5.162% | fail |
| generator point | P0 | 6 / 12 | 12.953–13.653 km | 4.840–5.540 km | 4.899–5.063% | fail |
| generator point | M0 | 6 / 12 | 10.546–22.641 km | 8.487–9.187 km | 5.389–5.838% | fail |

Both generator arms pass only the backtracking, robustness and build-index-tie
subgates. Successful paths have zero backtracking; successful paths and
retained failure prefixes have zero exact-score/build-index ties. Partial-
prefix backtracking was not measured. Water ledgers close to at most `1.11e-15`
relative error.

All six generator failures for each arm are the rotated `theta = 0.31` cases:
two translations at every spacing. There are zero polygon-failure/generator-
success termination repairs. M0 instead changes from four polygon failures to
six generator failures.

Failure prefixes make the categorical result physically legible. They enter a
closed-cut sink after overshooting the analytic outlet along-track:

- terminal `s` is about `-9.9` to `-37.5 km`;
- terminal `|n|` is about `36.6` to `120.3 km`; and
- the failures contain 35–156 visited cells with positive local margins.

These are not head-adjacent numerical stalls. The strictly downhill one-face
walk has left the intended corridor before conservative sink storage receives
its water. A closed sphere would remove the artificial cut termination, but it
would not turn a path displaced tens to hundreds of kilometres into the
prescribed centreline.

## Paired interpretation

Only two of four registrations are common portal successes at each spacing for
either arm. Per the preregistration, no paired geometry subgate is therefore
causally evaluable. Raw common-success changes remain descriptive:

- P0 improves cross-track and arclength on all six common successes. At 2 km,
  polygon cross-track `8.664–9.364 km` becomes `4.840–5.540 km`; the state-
  placement change removed a substantial error contribution but not enough to
  meet the contract.
- M0 improves only the unrotated `8 km, delta = 0` pair. Its other five common
  successes worsen in both cross-track and arclength. Unequal integrated face
  width remains appropriate to distributed MFD water, but this result gives no
  support for using its largest share as a sparse centreline owner.

Do not describe the first bullet as a repaired geometry gate: the rotated
baseline failures censor half of every spacing's paired set. Do not describe
the control as exact for the complete boundary operator: generator sampling
makes internal affine Voronoi grades linearly consistent, while portal grading,
closed cuts and discrete extraction remain unchanged.

## Implementation and invariants

The control is deliberately test-only and provenance-wrapped. The registered
polygon-mean builder still constructs its original elevation vector, then both
representations enter one private route/rank/audit assembly. The diagnostic:

- rejects V and B configurations;
- uses projected `mesh.cell_center_km` generators and the frozen affine formula;
- independently recomputes route, fractions, ranks and ledgers;
- reuses the unchanged tracer and metric code;
- repeats every case build and every trace exactly; and
- verifies immutable caps, cases and trace contexts.

The paired polygon baseline reproduces the previous R1a audit exactly. An
independent implementation review and full-matrix rerun found no load-bearing
sampling, routing, aggregation or censoring defect.

## Consequence for the causal ladder

A polygon-centroid, linear-exact gradient followed by the same P0 maximum-face
rule is redundant on affine A. Such a reconstruction recovers the constant
analytic gradient, and projecting it onto actual Voronoi face normals produces
the same internal affine P0 ordering already exercised by the generator
control. It could change curved V behavior, but A remains the required linear-
consistency stop gate.

The smallest nonredundant next discriminator should change path geometry, not
retune face scores:

1. reconstruct the constant affine vector from polygon means at polygon
   centroids and verify it against the analytic A vector;
2. start at the physical head and cross each polygon at the actual intersection
   of that vector ray with the polygon boundary;
3. continue from the entry point rather than choosing a maximum face or its
   midpoint; and
4. require the semantic portal and the same affine path metrics.

This affine-only continuous-crossing rung can distinguish the graph-walk
representation from the cap and reconstruction before paying for a general
conservative `H(div)`/RT0 field on V. It must be separately preregistered. Do
not implement it, a centroid-distance heuristic or a general flux replacement
as part of this completed control.

Later evidence: the
[affine continuous-crossing audit](channel-extraction-r1a-affine-crossing-2026-07-14.md)
validates X0 on all 12 cases and records 11/11 judged X1 geometry successes, but
one frozen all-cell reconstruction prerequisite fails narrowly. The suggested
graph-bundle localization therefore remains plausible but unestablished; the
next rung is stable affine numerical reconstruction, not a richer river model.

## Executed checks

```bash
cargo test --lib channel_extraction_r1a -- --nocapture
cargo test --lib world::landscape::channel_extraction_r1a_generator_control::registered_affine_generator_control_reports_paired_causal_matrix -- --ignored --exact --nocapture
```

The routine R1a tests pass. The paired audit matrix passes in `48.13 s`; an
independent rerun passes and reproduces the same classification. “Passes” means
the harness deterministically reproduces this negative causal result.
