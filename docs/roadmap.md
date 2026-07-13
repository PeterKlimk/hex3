# Hex3 roadmap

This roadmap turns the [project thesis](thesis.md), chosen
[model strategy](model-strategy.md), [system assessments](system-assessments.md),
and [gap analysis](gaps.md) into an ordered decision process. It is intentionally
revisable: evidence may reorder work or justify fundamental rework of any
current system.

The roadmap optimizes for coherent emergence, visual appeal, explanatory depth,
iteration speed and “wow” value—not for completing a conventional list of
planet subsystems.

## Roadmap rules

1. Preserve one known product baseline while experiments are evaluated.
2. Fix ownership, units or convergence failures before tuning downstream systems
   around them.
3. Prefer shared state and couplings that benefit several consumers.
4. Separate world, semantic and presentation changes in implementation and
   evidence.
5. Use bounded vertical slices before committing to deep simulations.
6. Promotion requires the [validation policy](validation.md); implementation
   alone does not advance roadmap status.
7. Retire or park overlapping mechanisms when a new owner is promoted.
8. Require an object-level or relationship-level product question before local
   tuning or deeper simulation becomes roadmap work.

## Horizon 0: documentation and decision substrate

Status: **complete enough to govern the next pass**.

Purpose: establish a trustworthy account of the current system before choosing
large reworks or new stages.

Completed in this sprint:

- project thesis and fidelity vocabulary;
- code/document/render inventories;
- current architecture and stage pipeline;
- documentation authority/status policy;
- system fidelity/Pareto assessments;
- experiment registry;
- semantic/presentation architecture;
- validation and reproducibility policy;
- gap analysis and this roadmap.
- explicit hybrid causal world-model strategy and domain truth contracts.

Remaining:

- establish one current experiment/configuration manifest.

Completed after the initial roadmap draft:

- human-facing root `README.md`;
- minimal assistant policy linking canonical docs;
- archived/reclassified superseded specs, roadmaps, reviews, research and
  generated outputs;
- corrected the stale top-level world-stage module description and historical
  source-document paths.
- added a serializable effective-run manifest to world exports and diagnostic
  headers, including build revision/dirty state and fine-cache identity/outcome.

Exit gate: a new contributor can find the product path, current experiments,
validation rules and active roadmap without reading chronological specs.

## Horizon 1: observability and semantic foundation

Purpose: make the existing planet understandable, comparable and reusable
before increasing model depth.

### 1A. Units and evidence envelope

- **Completed:** document and test the end-to-end elevation
  datum/unit/slope/render conversion, with unit-contract metadata in exports;
- audit normalized versus physical temperature, precipitation, time and erosion
  quantities;
- extend the shared world manifest with presentation/camera capture metadata;
- add presentation/camera metadata to controlled captures;
- define fixed seed and resolution panels for product promotion.

Exit gate: numeric and visual results can be reproduced and cannot silently mix
physical and cartographic scales.

### 1B. First semantic objects

- **Completed:** extract shared water-body and river-network semantics from
  hydrology/render preparation/diagnostics;
- extract range/plateau/pass semantics from existing mountain diagnostics;
- define stable per-world-stage identities, measurements, provenance and
  importance;
- expose objects to diagnostics and presentation without changing world state.

Exit gate: the renderer and audits consume the same definition of a major river
or range, and objects can explain their modeled causes.

### 1C. Presentation profiles

- implement declared Physical, Diagnostic, Cartographic and Dramatic profiles;
- add legends, units, scale and vertical-exaggeration disclosure;
- make river/overlay policy consistent across Globe/Map and relevant modes;
- use scale-dependent generalization/decluttering;
- decide whether to integrate displaced-facet hillshade interactively;
- verify/remove stale rendering paths only as replacements become clear.

Exit gate: presentation choices are reproducible profile state, and no visual
mode is mistaken for physical evidence.

## Horizon 2: geographic coherence decision

Status: **active**.

Purpose: determine whether the current pipeline creates coherent, memorable
geography before adding another domain or optimizing an isolated mechanism.

### 2A. Geographic objects

- extract scale-aware ranges, plateaus, passes, valleys, divides and coast
  hierarchy, complementing existing river and water-body identity;
- distinguish durable semantic objects from masks, diagnostic samples and
  presentation primitives;
- record object measurements, uncertainty, provenance and importance;
- characterize diversity within worlds and across the reference panel.

### 2B. Causal correspondence

- test whether geographic objects occur where their claimed tectonic,
  erosional, climatic and hydrologic causes predict;
- use controls and counterexamples to assess sign, ordering, topology, scale and
  downstream consequence—not Earth-like marginal distributions alone;
- identify missing relationships as well as wrong, overbuilt or redundant
  mechanisms;
- make the legacy product path and major experimental paths answer the same
  object-level questions where comparison is useful.

### 2C. Matched presentation

- implement reproducible Physical, Diagnostic, Cartographic and Dramatic
  profiles sufficient for controlled inspection;
- record relief, river, color, lighting, camera and generalization state;
- inspect the same objects across numerical state and presented form;
- judge legibility, identity and spectacle as product outcomes without using
  them as physical validation.

### 2D. System disposition

Update every material system to **retain**, **simplify**, **replace**,
**quarantine**, **remove** or **research**. Each decision must name its truth
contract, visible/downstream payoff, evidence, cost and cheaper alternative.

Exit gate: representative worlds can be discussed in terms of geographic
objects and causes; physical and cartographic judgments cannot be confused; and
the next build/rework choice follows from a cross-system comparison rather than
the availability of a tunable subsystem.

## Horizon 3: choose the next world expansion

Purpose: choose among missing systems and fundamental reworks using the Horizon
2 evidence. No candidate is the default merely because an early prototype or
research note exists.

Candidate vertical slices include:

- living surface: ecological constraints, vegetation coverage and bounded
  disturbance;
- source-to-sink geography: sediment, floodplains, basin fill and deltas;
- climate/water repair: seasonality, ocean heat shortcut, storage or wetlands;
- tectonic/landform repair: a cheaper history representation or shared forcing
  where object correspondence demonstrates the need;
- coast, ice or soil systems where they unlock several visible consequences;
- semantic geography for resources, traversability, settlement and routes.

Selection requires a bounded authentic model, at least one striking visible
outcome, meaningful downstream leverage, explicit cost and a discriminating
vertical-slice test.

### Sediment candidate: research and design gate

- compare reduced sediment-routing/landscape-evolution approaches;
- define state, units, time relation and mass ledger;
- identify a minimal visible target: floodplains, terminal basin fill, deltas or
  foreland/coastal deposition;
- budget fine-graph memory and update cost;
- define interaction with current deposition and hydrology ownership.

### Sediment candidate: bounded v0, conditional

If the design gate is favorable:

- track one persistent mobile/deposited sediment quantity;
- route through existing drainage with explicit capacity/opportunity rules;
- conserve source, stored material and ocean/export sink;
- produce at least one visible and one downstream semantic consequence;
- avoid stratigraphic layers, detailed grain classes and global flexural
  feedback until v0 proves value.

Exit gate: sediment creates coherent basin/coastal geography worth its cost and
does not merely add a ledger or smooth terrain.

## Horizon 4: choose a deep physical coupling

This horizon is a decision point, not a commitment to perform all branches.

Candidates:

- shared-clock tectonic uplift, erosion and denudation;
- sediment/load-driven flexure and basin formation;
- coherent glaciers/cryosphere with water and erosion feedback;
- reduced ocean currents/heat transport for regional climate;
- soil and wetland hydrology for ecology and later agriculture.

Selection criteria:

- prerequisites are stable;
- at least three systems or major visible outcomes benefit;
- a bounded authentic model exists;
- physical ownership replaces rather than stacks over a heuristic;
- resolution and runtime risks have a credible validation plan.

The current carrier tectonic models cannot enter the product branch until their
boundary/deformation operators pass resolution gates. A different, simpler
history representation is allowed and may be preferable.

## Horizon 5: geography becomes human meaning

Purpose: prepare and then optionally build the Civilization-board dimension of
the project without allowing it to eclipse planet generation prematurely.

### Foundations

- semantic regions and traversability;
- freshwater/coast access and hazard/opportunity fields;
- ecological productivity and resource affordances;
- routes, chokepoints and settlement suitability;
- scale-aware symbols, labels and borders.

### Bounded world-history candidate

Prototype aggregate settlement growth, route formation and cultural diffusion
before individual agents or detailed economies. Geography should constrain and
differentiate outcomes; generated history should in turn create visible map
structure and stories.

Exit gate: the human layer reveals consequences of the generated planet and
produces emergent narrative. If it behaves like independent noise over a map,
deepen the coupling or stop.

Full civilization/economic simulation remains a distant option requiring a
separate scope decision.

## Continuous workstreams

These run when justified rather than waiting for one horizon to finish:

### Correctness and performance

- resolve discovered topology, unit, convergence and cache defects;
- profile before optimizing;
- protect iteration speed and Windows GPU viability;
- add tests around previously costly failures.

### Product visual quality

- improve lighting, materials and animation when they have strong visible
  return and preserve layer separation;
- maintain controlled captures and presentation regression cases;
- favor semantic/generalization improvements over brute-force density.

### Current-system criticism

- periodically reassess whether plate/crust initialization, stage architecture,
  adaptive refinement, climate, erosion and rendering still earn their shape;
- permit deletion or fundamental replacement when a simpler or more generative
  architecture is demonstrated;
- do not turn this roadmap into protection for current code.

### Research

Use targeted subagent/external research for the questions listed in
[gaps.md](gaps.md). Each study should return a Hex3-sized mechanism and compute/
benefit recommendation, not only a survey.

## Near-term sequence

Feature expansion is paused for Horizon 2. The corpus, research synthesis and
[model-strategy decision](model-strategy.md) have already established the
evaluation substrate; the current sequence is:

1. extract provisional range/plateau/ridge/divide/peak/pass semantics;
2. produce matched Physical, Diagnostic, Cartographic and Dramatic inspections
   for a small representative/outlier panel;
3. assess causal correspondence for those objects, including absent or
   contradictory relationships;
4. give current systems and high-leverage missing systems comparable
   dispositions;
5. choose one bounded rework or expansion—or deliberately choose subtraction—
   from the resulting cross-system case.

The [subtractive architecture audit](subtractive-audit.md) remains evidence,
not a standing optimization queue. Cleanup, ablation or performance work enters
this sequence only when it resolves an identified geographic or iteration-cost
decision.

Seed 12345 now has a first spatial dossier packet with diverse mountain, lake
and river targets and exact drainage-integration provenance. This is evidence
toward steps 1–2, not completion: range/plateau/divide semantics and matched
diagnostic layers outside the completed range-ancestry packet remain open.

The first range-ancestry packet and binary repeated-uplift control now locate
the flat-plateau defect upstream of erosion: the coarse distance-band envelope
is already table-like, fine structural synthesis is a no-op at product defaults,
and repeated uplift amplifies rather than originates the grammar. The next
decision is no longer another fixed-height mountain comparison. The
[mountain-system design basis](research/mountain-system-design-basis-2026-07-13.md)
finds that the current path collapses tectonic forcing into terrain height before
drainage can coevolve with it. The next bounded implementation is therefore the
[orogen organization testbed](research/orogen-testbed-spec-2026-07-13.md): a
dimensioned, time-resolved uplift–drainage–hillslope system compared against
locked hold-and-carve, synthetic-topology and explicit-skeleton controls. It
must pass linked-segment, forcing-reorganization and wet/dry causal cases before
any global-seed tuning or product-pipeline replacement.

Slice 1 of that testbed is now implemented. Its deterministic forcing, routing,
water/solid ledgers, timestep smoke and full-run stability pass, but its erosion
budget and post-relaxation relief do not converge with mesh spacing. The
[channel/surface scaling decision](research/channel-surface-scaling-2026-07-13.md)
now selects an explicitly cell-mean finite-volume continuum, effective areal
fluvial denudation driven by validated specific discharge, and fixed physical
outlet portals. The next build is its analytic boundary/routing/denudation gate,
not a channel-width patch, Slice 2 semantics, global tuning or product
integration. Current numerical evidence:
[Slice 1 audit](audits/orogen-testbed-slice1-2026-07-13.md).

The first Slice 1R analytic rung now passes the isolated pathway law, genuine
full-hex boundary geometry, interior MFD plane/ridge convergence and
manufactured areal denudation. Portal balance, a separately derived
depression/flat potential and the exact linear Dirichlet fixture also pass their
first analytic controls. Physical mean-surface gradient reconstruction now also
passes affine and smooth-radial cases, and the genuine-boundary hillslope
operator now closes internal/portal volume without pinned cells. Radial and
one-sided convergent fields converge, and flow-aligned physical grade passes,
but `|q_vector|` retains only 16.1% of two-sided convergence-line strength.
That exact sink has no unique local continuum vector; RMS/L1/cell-width
fallbacks are not invariant substitutes. The resolved downstream-reach and
integrated-cut follow-up passes. A separate transactional C0 arm is now
implemented without a local fallback, with effective denudation zero pending a
dimensioned regime; U/L remains blocked until manufactured spatial and temporal
convergence pass. Manufactured tests pass, but the first unseen 0.1 Myr U/L
screen fails at 2 km through concentrated denudation and timestep collapse.
Post-run diagnosis found a portal physical-grade defect and absent fluvial
slope CFL; correction and the unchanged rerun remove the false 2 km runaway.
The subsequent fixed-16-km supported-intensity comparison stabilizes local q
but is not promoted: it materially changes export, largely replaces explicit
step cost with global filter cost, and has no drainage-divide ownership. The
next bounded task is therefore the manufactured minimum C1
`{z_bar,z_c,f_c}` volume-mixing fixture, not long U/L, filter/K tuning, sediment
or product integration. Current evidence:
[C0 support discriminator](audits/c0-support-discriminator-2026-07-13.md).
That fixture now passes fixed-area, channel-history, export, mean-volume,
internal-transfer and zero-width gates at 8/4/2 km. The next bounded rung is a
prescribed conservative receiver network with confluence and reorganization
controls, retaining the same physical reach area and local C1 mixing. Network
and width ownership—not sediment, ecology, rendering or a long terrain run—are
the remaining questions. Current evidence:
[C1 manufactured fixture](audits/c1-manufactured-fixture-2026-07-13.md).
The prescribed routed follow-up now also passes stable reach identity,
conservative confluence/capture routing, C1 response and overlap-remap gates at
8/4/2 km. The subsequent ownership review rejects all three candidate mechanisms
as exclusive owners. Conservative continuum flow owns instantaneous supply; a
dimensioned discharge-slope/resistance rule proposes channel support; a sparse
active/dormant graph owns C1 state and lineage; semantic and presentation
networks remain derived. It preregisters a snapshot-versus-persistent-memory
discriminator under receiver jitter and one genuine capture—not product
integration or initiation/width tuning. That decision is recorded in:
[Routed C1 fixture](audits/c1-routed-fixture-2026-07-13.md) and
[drainage-network ownership](research/drainage-network-ownership-2026-07-13.md).
The narrower M0 mechanism rung now passes on prescribed physical observations,
including hysteretic retention across a manufactured threshold dip, exactly
one capture and transactional composition with unchanged C1 intervals. It does
not execute the preregistered discriminator: S0 loses the marginal reach by
construction, and the MFD-to-thalweg extraction rule remains absent. The
next decision is therefore that extraction rung and its comparison with current
production SFD semantics—not birth/retirement, sediment, long U/L or product
integration. Current evidence:
[M0 ownership audit](audits/channel-ownership-memory-m0-2026-07-13.md).
That extraction decision is now split cleanly from initiation and persistence.
The first planar R0 implementation invalidated the discriminator rather than
selecting an arm. Its centre-to-centre V length gate is unattainable on the
chosen lattice orientation, equal face widths alias physical-gradient P0 with
local dominant-MFD M0, and the draft Y is not smooth at its junction. No draft
extractor is retained and M1 is not rejected; its added pass simply remains
unearned. The next bounded rung is now preregistered: a guarded local
Earth-radius S2 Voronoi cap compares path-local P0/M0 on affine and smooth-V
flow, while cell topology and selected-face geometry receive separate gates.
Research also removes analytic merging Y from this rung: confluence is a later
network/morphology gate, because distinct smooth gradient trajectories cannot
merge and share a suffix. The cap geometry substrate now passes its 8/4/2 km
determinism, reciprocal-face, projection and guard-independence gates. The
exact-input rung also passes: polygon-mean A/V/B inputs feed one immutable
conservative route per case, domain ranks differ materially, and six A/V head
cells provide a necessary visited-cell P0/M0 conflict witness. The completed
path rung rejects both arms. Rotated affine traces can reach real sinks, and
even successful A paths drift kilometres because exact polygon means are
interpreted through generator-based two-point geometry. V is closer but cannot
rescue a rule that fails A. Before any richer tracer, preregister a causal
operator-consistency ladder: generator-point A as a non-promotable control,
then a polygon-mean linear-exact local gradient, and only if justified a shared
conservative non-orthogonal face flux. This is not initiation, persistence,
sediment or ecology. Current evidence:
[seeded channel extraction R0](research/channel-extraction-r0-2026-07-13.md)
and [R0 audit](audits/channel-extraction-r0-2026-07-13.md),
[centreline geometry basis](research/channel-centerline-geometry-basis-2026-07-13.md),
[R1a specification](research/channel-extraction-r1a-2026-07-13.md),
[R1a G0 audit](audits/channel-extraction-r1a-g0-2026-07-13.md) and
[R1a input/rank audit](audits/channel-extraction-r1a-input-rank-precheck-2026-07-13.md),
and [R1a path audit](audits/channel-extraction-r1a-path-2026-07-14.md).
The first ladder rung is now
[preregistered](research/channel-extraction-r1a-generator-control-2026-07-14.md):
change only affine state sampling, reuse the complete route/tracer stack, score
paired subgates, and stop before implementing a new operator.
The rejected nominal rectangular boundary remains an important scope decision:
exact cut-cell hex geometry is not currently justified for a testbed intended
to transfer to a closed sphere. Earlier analytic evidence remains in the
[Slice 1R audit](audits/orogen-testbed-slice1r-2026-07-13.md).
