# Validation and reproducibility policy

Hex3 validation asks whether a change is correct, coherent, worthwhile and
visually successful without confusing those questions. No single metric,
render, seed or scientific citation is sufficient.

Status and evidence terminology follows the
[documentation policy](documentation-policy.md). Model/presentation separation
follows the [semantic and presentation architecture](semantic-presentation.md).

## Start with the claim

Every consequential change should state:

1. the problem or opportunity;
2. the system and layer that own it;
3. the expected causal mechanism;
4. the observable outcomes that should change;
5. the important quantities that should not change;
6. the cost expected to buy that result;
7. the evidence that would falsify or park the proposal.

Validation should test this claim, not collect every available number after the
fact. A climate mechanism does not become a mountain improvement because it
changes precipitation; a presentation improvement does not repair terrain; a
conservative operator is not useful if it puts conserved material in the wrong
places.

## Evidence dimensions

Assess these independently:

| Dimension | Question | Example evidence |
|---|---|---|
| Correctness | Does implementation satisfy its stated algorithm/invariants? | unit/property tests, topology checks, identity gates |
| Dimensional consistency | Do units, extensive/intensive quantities and time scales compose? | conversion tests, unit ledger, scale analysis |
| Conservation | Is tracked matter/flux created or lost only where declared? | area/volume/material residuals |
| Numerical behavior | Is the solution stable and converged enough? | timestep/iteration sweeps, residuals, operator isolation |
| Resolution behavior | Does physical behavior survive discretization changes? | fixed-world coarse/fine/backend/resolution sweeps |
| Structural behavior | Are objects and relationships plausible and coherent? | range, river, basin and topology scorecards |
| Empirical grounding | Are scales/distributions compatible with useful references? | observed ranges and scientific literature |
| Visual acceptance | Does the intended presentation communicate and delight? | controlled A/B, fixed camera, human review |
| Downstream value | Do consumers gain meaningful behavior or emergence? | before/after effects in dependent systems |
| Cost | Is compute, memory, complexity and iteration burden justified? | timings, allocations, profile data, maintenance impact |

Scientific grounding establishes mechanisms and useful ranges; it does not turn
Earth values into universal optimization targets. Hex3 may generate non-Earth
worlds and deliberately stylized presentations. References should constrain
claims and catch implausible failures without Goodharting the generator into one
planet.

## Validation layers

### Operator level

Test local algorithms in controlled cases:

- exact/identity behavior at zero controls;
- conservation and sign conventions;
- known analytic or monotonic cases;
- rotational/permutation invariance where applicable;
- stability across timestep, iteration or subdivision;
- boundaries, poles, seams, sinks and degenerate cells.

Operator correctness is necessary but cannot promote a system.

### System level

Test a subsystem on representative worlds:

- field distributions and physical ranges;
- topology and object statistics;
- causal response to its own controls;
- interaction with immediate inputs/consumers;
- resolution and seed sensitivity;
- time/memory scaling.

Use object-level metrics where aggregate means hide the feature being judged.
Lake area alone cannot distinguish a few implausible inland seas from many good
lakes; maximum elevation cannot characterize range width or structure.

### Pipeline level

Test information survival and ownership across stages:

- coarse cause to fine representation;
- pre-erosion to post-erosion change;
- climate to hydrology/erosion response;
- topology before and after generalization;
- model state to semantic object to presentation;
- invalidation and cache reuse boundaries.

A downstream system must not take credit for an input change it cannot preserve,
and a new physical owner should normally retire the heuristic it replaces.

### Product/presentation level

Judge the complete experience under declared profiles:

- legibility at intended globe, regional and local scales;
- silhouette, hierarchy, clutter and visual rhythm;
- consistency among modes and stage views;
- appeal and “wow” value;
- absence of misleading presentation artifacts;
- acceptable generation, memory and frame cost.

Human visual review is required when appearance is part of the claim. It is not
an embarrassing fallback metric; it is the correct evidence for communication
and aesthetic goals, provided the comparison is controlled.

## Controlled comparisons

A useful A/B changes one conceptual variable at a time. It should use:

- identical seed and upstream cached/base state where valid;
- identical model parameters apart from the named change;
- identical computed/viewed stage;
- fixed camera, viewport and presentation profile;
- numeric output before interpreting screenshots;
- clear labeling and randomized/blind ordering when practical.

Composed-regime experiments are sometimes necessary. When effects interact,
include component A/Bs and state that the result depends on the composition;
do not attribute synergy to one mechanism.

### Seed sets

- One seed is for debugging and rapid iteration.
- A small fixed panel is for controlled design comparison.
- A broader fixed set is required for promotion when stochastic morphology is
  central to the claim.
- New/random seeds are useful as an overfitting check after thresholds are set.

Do not change the seed panel after seeing an unfavorable result without keeping
the original result in the record.

### Resolution sets

Vary the resolution that owns the questioned operator while holding unrelated
representations fixed. For carrier tectonics, carrier resolution and terrain
resolution must be distinguishable. For rendering, viewport resolution and
world mesh density are different axes. Compare extensive quantities by physical
area/volume and intensive quantities without accidental cell-count weighting.

## Physical and presentation separation

World-model gates use modeled or derived physical quantities: elevation units,
slope, distance, area, topology, flow, conservation and stage-to-stage state.
They must not depend on relief scale, line width, lighting or camera.

Elevation, crust-column, distance, slope and relief claims must use the
[elevation and unit contract](units.md). In particular, native
elevation-per-radian slope is not physical grade.

Presentation gates use declared visual conditions: apparent silhouette,
screen-space width, contrast, clutter, labels, seam behavior and readability.
Every visual result must identify its profile and camera. If a screenshot looks
wrong, inspect world, semantic and presentation layers before choosing which
system to change.

True-scale or diagnostic renders help interpret numeric state; they do not
replace numeric validation. Dramatic renders help judge spectacle; they do not
serve as physical evidence.

## Performance and complexity evidence

Measure cost at the scale where the feature is intended to run:

- wall time split by major stage/operator;
- peak and retained memory;
- cache hit/miss behavior and cache size;
- GPU frame/compute cost for animated presentation;
- output/buffer/texture size;
- effect of resolution and pass/step counts;
- conceptual cost: new state, parameters, invalidation and interacting modes.

A mechanism can be computationally cheap but architecturally expensive if it
adds another overlapping shape owner or tuning dimension. Conversely, an
expensive shared substrate may be justified when several visible systems consume
it.

Performance claims should identify build mode, platform/backend and measurement
method. WSL `cargo build` verifies compilation; GPU behavior and representative
runtime performance must be checked on Windows as required by repository policy.

## Reproducibility record

Every durable audit should start with:

```text
title / hypothesis
date and revision
dirty-worktree status
seed(s)
command and overrides
model and experiment status
backend and coarse/fine/carrier resolutions
computed and viewed stages
cache version/key/hit status
units and normalization
presentation profile, camera and viewport (if images)
machine/backend and build mode (if performance)
```

Record raw outputs or machine-readable summaries when practical. A derived table
should retain the command/configuration needed to reproduce it. Generated world
exports include a `manifest` with build revision/dirty state, effective world
model parameters, backend, stages and fine-cache provenance. Presentation and
camera state are not yet part of that world manifest, so visual reports must
supply them separately.

## Current tools

- `cargo test` covers unit, identity, conservation and numerical invariants.
- `cargo build`/`cargo clippy`/`cargo fmt` provide compilation and static hygiene.
- `diagnose` supplies general statistics and dedicated drainage, lake, mountain,
  rebuild-fidelity, detail-survival, tectonic-history and river audits.
- `dossier` supplies selected spatial objects, aggregate water geography and a
  frozen-terrain diagnostic conditional-climatology comparison. That null is
  fitted from each product world and therefore tests residual value, not
  out-of-sample replacement-climate performance.
- `tectonic_scorecard` performs cross-seed/carrier-resolution promotion checks
  without fine erosion or rendering.
- comparison/sample binaries support targeted orogen experiments.
- the offscreen sweep harness produces fixed-camera tiles/contact sheets for
  parameter and presentation comparisons; its bounded `water-geography` packet
  pairs ordinary relief with exact categorical ownership, raw shorelines and
  spill/integration-cut provenance at two derived cameras.
- interactive stage navigation enables retained pre/post comparisons.
- JSON/gzip export supports external object/field analysis.

Tool availability does not imply that every audit has a maintained threshold or
complete metadata. The validation roadmap should consolidate common configuration
and report headers rather than add another bespoke output for each experiment.

## Promotion gate

Promotion requires evidence proportional to risk and reach. For a substantive
model default, require:

1. a named owner and claim;
2. operator correctness and identity behavior;
3. physical/dimensional rationale appropriate to its fidelity claim;
4. isolated A/B against the current baseline;
5. structural/object evidence on a fixed seed panel;
6. relevant resolution convergence;
7. compute/memory assessment;
8. downstream regression checks;
9. controlled visual acceptance when appearance matters;
10. explicit disposition of mechanisms it replaces or overlaps.

Presentation-only changes need not pass physical-model gates, but must prove
they cannot alter modeled state and must be tested across relevant views,
scales, seams and profiles.

## Failure, parking and uncertainty

A failed hypothesis is useful evidence. Record whether failure belongs to the
mechanism, discretization, calibration range, coupling, cost or intended payoff.
Revert failed implementation branches when appropriate while preserving the
audit/decision.

Park work when the mechanism remains plausible but evidence, prerequisites,
visual acceptance or cost justification is missing. Mark uncertain values and
status explicitly rather than promoting through optimistic prose.

Do not tune downstream systems to compensate for a known upstream convergence or
presentation error. Fix or isolate the faulty layer first.
