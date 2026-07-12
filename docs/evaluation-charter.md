# World evaluation charter

Status: strategy decision reached; geographic correspondence pass active,
2026-07-12.

This pass pauses feature expansion to determine what Hex3 currently generates,
which systems earn their complexity, where authenticity breaks, and which model
class should guide later work. It evaluates numerical state, semantic
interpretation and presentation separately, then studies their relationships.

The charter governs evaluation and research. It does not promote current
thresholds, classifiers or historical scorecards merely by cataloguing them.

## Decisions this pass must inform

1. Which implemented systems should be retained, simplified, replaced, parked
   or removed?
2. Which observed problems belong to the physical model, discretization,
   semantic interpretation or presentation?
3. Where is additional compute buying visible or downstream value?
4. Which missing systems are Pareto-favorable given authenticity, emergence,
   spectacle and architectural leverage?
5. Should each domain behave as an Earth-scale reduced physical model, a
   physically grounded mini-planet, or an authentic game/world-generation hack?
6. What is the next implementation or research decision after this evaluation?

## Outputs

The pass should leave five durable products:

- a registry of implemented metrics, feature definitions and gates;
- a fixed, provenance-complete corpus of numerical outputs and matched views;
- a failure-mode catalogue linking observations to layers and candidate causes;
- comparable external research briefs;
- a model-strategy decision that revises system assessments and the roadmap.

The [model-strategy decision](model-strategy.md) is now complete. It selects a
hybrid authentic systemic world generator and makes coherent geographic objects
and matched physical/cartographic inspection the next evidence target. This
charter remains active for that correspondence pass; it is not a mandate for
further parameter optimization.

The first application is the preliminary
[seed-12345 planet dossier](dossiers/seed-12345.md). It records causal lineage
and object probes. A CPU packet now selects diverse named mountain, lake and
river targets with sparse drainage-repair provenance; matched Windows views and
diagnostic overlays remain required.

The first corpus/research synthesis is maintained in
[evaluation-synthesis.md](evaluation-synthesis.md); it remains provisional until
feature correspondence and matched visual inspection are added.

Raw reports and images are evidence. The synthesis documents own current
interpretation; dated audit records remain immutable historical evidence.

## Evaluation questions

### Numerical and field behavior

- What are the area-weighted distributions, tails, spatial scales and
  correlations of terrain, climate, hydrology, erosion and tectonic fields?
- Which quantities are invariant, converged, conservative or resolution-sensitive?
- Are metrics expressed in physical units, model coordinates or presentation
  space, and is that distinction enforced?

### Features and classifications

- What exactly constitutes a peak, range, plateau, valley, basin, lake, river,
  coast, climate region or biome?
- Is it a stable object, a threshold mask, a diagnostic sample or a visual mark?
- Which definitions conflate broad envelopes, unresolved terrain, physical
  slope, renderer exaggeration or arbitrary thresholds?
- How uncertain are classifications, and how do objects correspond across
  stage, resolution and configuration changes?

### Causal structure

- Do features occur where their claimed causes predict?
- When a failure is absent in reality or a higher-fidelity simulation, what
  mechanism prevents it there? Which observable and downstream consequences of
  that mechanism must survive in a reduced model?
- Does a control change its intended response while preserving declared
  invariants?
- Does information survive coarse-to-fine transfer and later processing?
- Can unusual worlds be explained from retained state rather than excused after
  viewing them?

### Product character and cost

- How diverse are features within a world and across seeds?
- What creates geographic identity, legibility and memorable structure at
  globe, board and regional scales?
- What compute, memory, parameter and maintenance cost produces each benefit?
- Would a simpler authentic hack preserve the useful output?

## Canonical corpus

Corpus membership is fixed before calibration. Results are never silently
removed because they are unfavorable.

### Seed panel

The standing ten-seed panel inherited from the tectonic scorecard is:

`12345, 777, 4242, 9001, 314159, 271828, 8675309, 20260711, 42, 1001`.

Within it:

- `12345` is the continuity/reference seed used by many historical audits;
- `8675309` and `1001` are retained known lifecycle plateau outliers;
- `42` is a convenient rapid-iteration seed, not a promotion surrogate.

New random holdout seeds may test overfitting only after thresholds are frozen.

### Run tiers

| Tier | Purpose | Initial membership |
|---|---|---|
| Smoke | Harness and schema debugging | seeds 42 and 12345 at deliberately small coarse/fine caps |
| Reference | Cross-seed product characterization | all ten seeds, product defaults, one declared practical reference coarse/fine resolution |
| Convergence | Separate discretization effects from seed variation | seeds 12345, 8675309 and 1001 across at least three owner-resolution values |
| Controls | Establish metric response and confounders | fixed seeds with one conceptual input or mechanism changed at a time |
| Holdout | Detect panel/threshold overfitting | newly sampled seeds after registry and gates freeze |

The reference corpus uses 100,000 coarse cells and an explicit 1,000,000
fine-cell cap, selected by the
[reference-budget audit](audits/reference-budget-2026-07-12.md). This is an
evaluation budget, not a product default or universal convergence claim. The
product guardrail of eight million cells is not automatically the evaluation
budget. Every result records actual emergent cell counts.

### Required pairings

Where relevant, retain matched outputs for:

- coarse and active fine state;
- pre-erosion and eroded fine surfaces;
- physical numerical state and declared presentation profile;
- baseline and isolated control;
- repeated identical runs for determinism checks.

## Metric registry rules

Every durable metric uses the schema in [metric-registry.md](metric-registry.md).
Metrics are classified as:

- **invariant:** correctness, topology, conservation or determinism;
- **field descriptor:** distribution or spatial statistic;
- **feature measurement:** geometry or property of a defined object;
- **relationship metric:** causal placement, coupling or stage survival;
- **product indicator:** diversity, legibility, spectacle or identity proxy;
- **promotion gate:** a justified decision boundary with known response.

A reference comparison is not automatically a gate. A historical gate is not
automatically current policy. A threshold mask is not automatically a semantic
object. No metric becomes a promotion gate until its expected control response,
confounders and failure interpretation are recorded.

Metrics likely to be optimized directly must include a Goodhart warning. Peak
height, land fraction and plateau coverage can all improve while morphology or
causal attribution worsens.

## Feature-analysis vocabulary

Reports must distinguish:

- **field:** a value at every cell;
- **mask:** cells passing a rule;
- **component:** connected cells in a mask;
- **sampled feature:** a component or path measured for one report;
- **semantic object:** reusable identity/topology derived from modeled state;
- **presentation primitive:** geometry, color, line or displacement shown;
- **reference class:** an external empirical or design comparison;
- **gate:** an explicit pass/warn/fail decision rule.

Terms such as mountain, plateau and river must link to their operational
definition. Reports should say “cells above 1.5 km” or “connected cap under the
declared local-slope rule” when no stable semantic object exists.

## Evidence and provenance

All runs follow [validation.md](validation.md) and record revision/dirty state,
seed, command, model, backend, stages, resolutions, cache provenance, units and
normalization. Visual evidence additionally records profile, relief scale,
camera, viewport and platform/backend.

Machine-readable rows should carry metric IDs from the registry. Human reports
may aggregate them, but must preserve the command and manifest needed to recover
the source run.

## Research brief contract

External research is divided by domain but uses one output template:

1. problem and Hex3 decision it informs;
2. how reality avoids or resolves the failure, followed by the mechanism used
   in scientific models, simulations, games or rendering;
3. fidelity class and deliberate approximations;
4. state, inputs, outputs, coupling and characteristic scales;
5. compute and implementation cost;
6. visible/emergent benefit;
7. known failure modes and validation methods;
8. applicability and mismatch with Hex3;
9. smallest discriminating experiment;
10. recommendation: adopt, adapt, research, park or reject.

Research should prefer primary scientific/technical sources and direct
developer documentation or talks for game/rendering techniques. A survey without
a Hex3-sized mechanism and cost/benefit judgment is incomplete.

The full causal account is useful even when it is computationally infeasible.
It acts as the reference model against which a simpler latent model or authentic
hack is judged: simplification may discard internal process, but should do so
deliberately and retain the causal signatures needed by visible geography and
future consumers.

## Sequence and stop conditions

1. inventory existing diagnostics, gates, tools and historical definitions;
2. populate the metric registry and mark uncertainty/staleness;
3. establish corpus harness/output formats and measure practical cost;
4. run smoke, reference, convergence and control tiers;
5. inspect numerical outliers with matched views;
6. synthesize focused external research;
7. **Complete:** write the model-strategy decision and revise the roadmap;
8. extract geographic objects, inspect matched views and make final system
   dispositions before choosing another stage.

Feature expansion remains paused until geographic objects, causal
correspondence and matched views can identify its opportunity cost. Small
instrumentation or semantic extraction changes are allowed when they expose
existing state without tuning the generated world. Local ablations require a
specific object-level decision and stop condition; producing another optimizable
metric is not sufficient reason.
