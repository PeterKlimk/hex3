# Documentation policy

This document defines how Hex3 documentation states authority, status, and
evidence. It exists to prevent design proposals, implemented experiments, and
the product architecture from drifting into one indistinguishable record.

## Authority

For questions about current behavior, authority descends in this order:

1. executable code and tests;
2. canonical current-state documentation;
3. current experiment and roadmap records;
4. dated audits and decision records;
5. superseded specifications, research notes, and generated reports.

The [project thesis](thesis.md) is authoritative for goals and decision
principles, not for implementation facts. Code being authoritative does not
make every code path part of the product: defaults and runtime selection decide
which implemented path is active.

Canonical documents should link to evidence rather than absorb chronological
implementation diaries. When code and a canonical document disagree, record
the discrepancy and correct the document or code deliberately; do not treat the
older prose as an implicit requirement.

“Canonical” means authoritative about the present, not protected from change.
Canonical architecture is a versioned snapshot of accepted behavior. It may be
criticized or replaced wholesale when a better design is justified; the
documentation must then change with it.

## Document classes

| Class | Purpose | Expected maintenance |
|---|---|---|
| Thesis/policy | Goals, vocabulary, decision and evidence rules | Changes rarely and deliberately |
| Current architecture | What the accepted product does now and how state flows | Updated with architectural changes |
| System assessment | Physical basis, approximations, cost, payoff, validation and gaps | Updated when a model or its evidence changes materially |
| Gap analysis | Comparative options, missing couplings and research questions | Updated when evidence changes the option set |
| Roadmap | Ordered horizons, gates and active priorities | Kept actively curated; not a preservation mandate |
| Evaluation charter/metric registry | Active corpus, measurement meanings and decision boundaries | Stable during an evaluation pass; revised explicitly between passes |
| Experiment registry | Status of selectable or proposed alternatives | Updated at every state transition |
| Decision record | Why a consequential choice was made | Immutable apart from corrections/addenda |
| Audit/evidence | Results at a named revision and configuration | Immutable, reproducible where possible |
| Research note | External findings or exploratory reasoning | Non-authoritative until verified and synthesized |
| Archive | Superseded material retained for provenance | Not maintained as current truth |

The directory name `specs` does not establish a document's status. Existing
files there include all of the classes above and will be reclassified during
the replacement sprint.

## Implementation status

Use these terms precisely:

- **Proposed** — described but not implemented.
- **Implemented** — code exists; this says nothing about selection or quality.
- **Selectable** — reachable through a supported runtime or diagnostic option.
- **Numerically evaluated** — relevant invariants or quantitative gates have
  been run and recorded.
- **Visually evaluated** — controlled renders have received human review.
- **Promoted** — accepted for the product path.
- **Default** — selected without an override in the stated runtime context.
- **Parked** — retained but neutral, disabled, or outside active work.
- **Falsified** — the tested hypothesis failed its stated purpose or gates.
- **Removed** — no longer present in current code.

These states are not a linear ladder. A useful diagnostic can remain selectable
without being a product candidate; a numerically valid model can fail visual or
cost gates; a promoted mechanism may later be superseded.

Every architecture description should distinguish the product/default path
from implemented alternatives.

## Fidelity and evidence

Use the fidelity vocabulary from the [thesis](thesis.md): simulation-grade,
physically based, authentic hack, visual hack, and unjustified hack. Assign it
to a specific mechanism and claim, not to Hex3 as a whole. A subsystem may mix
a physically based transport solver, empirical source terms, and visual-only
presentation.

Keep these evidence dimensions separate:

- physical or scientific basis;
- numerical correctness and invariants;
- conservation and dimensional consistency;
- empirical or comparative grounding;
- structural behavior and resolution convergence;
- compute and memory cost;
- visual acceptance;
- downstream and emergent value.

Passing one dimension does not imply the others.

## Terminology

- **World model** means generated state and processes that may affect later
  systems.
- **Semantic model** means derived interpretation such as major rivers, ranges,
  basins, materials, or regions. It does not alter physical state unless an
  explicit feedback is modeled.
- **Presentation model** means geometry displacement, strokes, color, lighting,
  visibility and other visual communication.
- **Computed stage** is the furthest stage whose state exists.
- **Viewed stage** is the retained stage currently exposed through active
  accessors and rendering. Moving backward does not undo computation.
- **Coarse** and **fine** identify distinct tessellations and ownership scales,
  not merely quality levels.
- **Hydrology** covers drainage topology, basins, water bodies and river data.
  It is computed both before and after erosion.
- **Erosion** changes the fine surface and then requires hydrology to be derived
  again.
- **Current stage** means an implemented runtime milestone. It does not imply
  that the project considers the planet complete at that stage.
- **Future stage** means a candidate dependency milestone, not a commitment to
  a strictly linear pipeline. Later domains may introduce feedback into earlier
  state or run on a separate time/scale hierarchy.
- **Default**, **product**, and **implemented** are never synonyms.

Use **cartographic** for legibility-oriented presentation. The existing relief
preset named `Authentic` remains a code/UI name for now, but prose should call it
the authentic/cartographic presentation preset so it is not confused with true
physical scale.

## Reproducibility envelope

The operational evidence rules live in the
[validation policy](validation.md). At minimum, new quantitative reports should
record, when applicable:

- revision and dirty-worktree status;
- seed or seed set;
- command and runtime overrides;
- relevant constants/configuration or a stable config hash;
- Voronoi backend and coarse/fine cell counts;
- computed and viewed stages;
- cache key/version and whether cached state was used;
- units and normalization of reported fields;
- presentation profile, camera and viewport when images are evidence;
- machine/backend and timing method for performance claims.

A screenshot without seed, stage, camera, and presentation settings is useful
inspiration but is not sufficient evidence for changing the world model.

## Lifecycle rule for this sprint

Old documents remain in place until their current decisions, open questions,
and useful evidence are represented in the replacement set. Moving a document
to an archive does not endorse its claims; deleting one requires confidence
that it has neither durable evidence nor unresolved reasoning worth preserving.
