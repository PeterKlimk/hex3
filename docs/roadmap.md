# Hex3 roadmap

This roadmap turns the [project thesis](thesis.md), chosen
[model strategy](model-strategy.md), [system assessments](system-assessments.md),
[gap analysis](gaps.md) and current
[cross-system disposition](system-disposition.md) into an ordered decision
process. It is intentionally revisable: evidence may reorder work or justify
fundamental rework of any current system.

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

Status: **complete enough to select the next slice**.

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

**Completed:** update every material system to **retain**, **simplify**, **replace**,
**quarantine**, **remove** or **research**. Each decision must name its truth
contract, visible/downstream payoff, evidence, cost and cheaper alternative.
See the [cross-system disposition](system-disposition.md).

Exit gate: representative worlds can be discussed in terms of geographic
objects and causes; physical and cartographic judgments cannot be confused; and
the next build/rework choice follows from a cross-system comparison rather than
the availability of a tunable subsystem.

## Horizon 3: choose the next world expansion

Status: **active; Water Geography V0 selected as the enabling slice**.

Purpose: choose among missing systems and fundamental reworks using the Horizon
2 evidence. No candidate is the default merely because an early prototype or
research note exists.

The current frontier is ordered rather than tied:

1. water geography: climate/runoff value, connected ocean identity, lakes,
   basins, outlets, river roles, repair provenance, coasts and presentation;
2. living surface: ecological constraints, semantic regions, vegetation
   coverage and multiscale presentation, conditional on coherent inputs;
3. source-to-sink geography: persistent sediment, basin fill, floodplains and
   deltas, conditional on a design gate;
4. aggregate resources, traversability, routes and settlement; and
5. reduced tectonic forcing replacement when it can consume real fronts,
   history and material state.

Selection requires a bounded authentic model, at least one striking visible
outcome, meaningful downstream leverage, explicit cost and a discriminating
vertical-slice test.

### Water Geography V0: selected enabling slice

- compare transported climate/runoff with a cheap latitude–elevation–coast
  baseline on frozen terrain;
- use connected-ocean identity consistently for moisture source attribution;
- derive a shared report for oceans, lakes, basins, outlets, spills, river
  roles, coasts/islands and repair-cut contribution;
- distinguish discharge/catchment importance, network hierarchy, longest trunk
  and cartographic importance; and
- inspect the same objects in Physical, Diagnostic and Cartographic views.

Exit gate: the retained climate and hydrology have declared meanings, stable
causal consequences and coherent objects suitable for presentation and a
living-surface consumer. Do not add seasons, ocean circulation, groundwater,
sediment, vegetation or erosion tuning inside this slice.

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

The current sequence is deliberately short:

1. implement **Water Geography V0** as the bounded climate/water truth-contract
   and semantic integration slice described above;
2. decide whether transported moisture earns retention over the cheap baseline
   and whether current lake/repair hacks remain acceptable with object-level
   provenance;
3. if the inputs are coherent, design and build **Living Surface V0** as the
   first actual expansion; and
4. keep sediment, human geography and renewed mountain forcing in their stated
   order and behind their gates.

Do not turn this into another broad scoring campaign. Use a small set of known
worlds and causal controls, inspect named objects at suitable framing, and stop
when the retained truth contract and next consumer decision are clear.

### Completed mountain discriminator (history)

The following sequence records the completed organization work. It is retained
for provenance and is not the active queue:

1. extract the minimum shared object packet needed to compare ranges, ridge/
   divide graphs, passes, valleys, basins and river hierarchy, reusing existing
   diagnostic and semantic code. Its arm-neutral vocabulary and rung order are
   now frozen by the [landform object packet v0](research/landform-object-packet-v0-2026-07-14.md);
   the executable [G0/S0 surface-graph and split-tree contract](research/landform-object-packet-g0s0-2026-07-14.md)
   now passes manufactured planar/spherical gates and the first unchanged 250k
   product observation. The preregistered
   [D0 common drainage rung](research/landform-object-packet-d0-2026-07-15.md)
   now passes its bounded planar manufactured matrix. The
   [O0a relationship probe](research/landform-object-packet-o0-2026-07-15.md)
   is now implemented and evaluated as a bounded common checkpoint, recorded in
   its [dated audit](audits/landform-o0a-relationships-2026-07-15.md). The
   bounded planar [O0b correspondence/assembly contract](research/landform-object-packet-o0b-2026-07-15.md)
   is now accepted as the bounded common planar evaluation-instrument
   checkpoint. Amendment A passes the isolated-four-cone 4→8/2,
   equal-elder, frozen-remapping and whole-artifact reversal gates. The flat
   apron remains compatibility-only, while the failed linked-four-cone 2 km
   witness remains historical S0/D0 representation evidence. This does not
   accept product/spherical O0b, persistent identity/events, a product O0a
   adapter, packet/product R0 or H/C/G composition. Those product and temporal
   boundaries remain separately unregistered and unimplemented. The
   [product-boundary decision](research/landform-product-boundary-decision-2026-07-15.md)
   retires the combined packet/product R0 formulation: product-native evidence
   and common planar evidence remain separate;
2. **completed:** implement and evaluate the preregistered
   [slim common-core boundary](research/landform-common-core-v0-2026-07-15.md)
   on accepted manufactured V0 packets. Preserve exact historical V0 bytes and
   mechanical O0b answers while hashing reference O0a and the optional ten-run
   sensitivity suite separately; report actual bytes, wall time and peak
   memory. The [dated audit](audits/landform-common-core-2026-07-15.md) accepts
   the boundary and finds retained graph geometry dominates the 2 km core;
3. **completed:** materialize the
   [linked shared-input manifest](research/orogen-linked-shared-input-v0-2026-07-15.md)
   at 8/4/2 km. It binds exact mesh identity and phase, declarative and
   compiled/evaluated forcing, schedule and integrated work, initial state,
   runoff, portals, homogeneous material, candidate evaluation geometry and a
   separate resource envelope. It must not manufacture an arm-neutral final
   terrain, contain an arm conversion, select the scoring population or emit a
   landform-quality verdict. The
   [dated audit](audits/orogen-linked-shared-input-2026-07-15.md) accepts the
   exact artifact identity and measured cost without promoting a terrain arm;
4. **design envelope completed:** the
   [bounded organization-owner comparison](research/orogen-organization-owner-v0-2026-07-16.md)
   freezes shared admissible inputs, work-matched opportunity, whole-domain
   independent extraction plus a central report, resource philosophy and the
   three owners: hold-and-carve H, reduced coevolution C and graph-first G. It
   deliberately stops short of executable wire/verdict status;
5. commit its four executable amendments: **completed:** exact
   [artifact/provenance](research/orogen-organization-artifact-v0-2026-07-16.md);
   **completed:** exact
   [numerical/admission](research/orogen-organization-numerical-v0-2026-07-16.md);
   **completed:** exact
   [evidence/projection](research/orogen-organization-evidence-v0-2026-07-16.md);
   **completed:** exact
   [planar capture/human review](research/orogen-organization-planar-review-v0-2026-07-16.md);
6. **completed negative discriminator:** build the smallest end-to-end 4 km
   discriminator before more promotion infrastructure.
   Exact H/C/G execution over the accepted linked input, deterministic repeats,
   final physical surfaces, essential ledgers/hashes and direct in-memory reuse
   of S0/D0/reference-O0a now exist. Numerical output and sharp matched profiles
   show the shared static ribbon already owns the long roof: H preserves it, C
   only locally dissects it and G makes it discontinuous. No arm advances;
7. **completed discriminator:** compare accepted ribbon B, work-matched
   full-cosine finite parents F, and independent inheritance-conditioned child
   graph I before terrain execution. F removes the exact roof coherently; I
   creates separated hot spots and does not earn its complexity. All work
   ledgers close; no direct elevation/drainage input enters the compiler;
8. **completed negative discriminator:** feed F once through H as the
   direct-forcing control and C once as the evolving response. Finite support
   removes the exact roof, but both outputs remain smooth elongated massifs. C
   adds real denudation and drainage differences without coherent visible
   valley/divide/pass organization sufficient to earn roughly `2.22x` H's
   runtime. This is not a fair owner tournament because H restores its target
   while C receives gradual uplift; and
9. **completed:** return to the cross-system disposition table. It selects Water
   Geography V0 and preserves the mountain conclusion: when this slice resumes,
   improve causal forcing from real front topology/history/material state before
   adding deeper terrain physics.

Do not resume the R1 arithmetic ladder, run a global seed sweep, tune legacy
shape amplitudes or optimize an undecided composition. The organization slice
is explicitly allowed to select the cheaper graph-first/hybrid architecture
over the reduced physical arm.

The [subtractive architecture audit](subtractive-audit.md) remains evidence,
not a standing optimization queue. Cleanup, ablation or performance work enters
this sequence only when it resolves an identified geographic or iteration-cost
decision.

Seed 12345 has a first spatial dossier packet with diverse mountain, lake and
river targets and exact drainage-integration provenance. Its lake, river and
repair targets are starting evidence for Water Geography V0, not its verdict.

### Research evidence constraining this sequence

The following chain records why the current slice has this shape. It is
evidence and history, not a queue of automatic next rungs.

The first range-ancestry packet and binary repeated-uplift control now locate
the flat-plateau defect upstream of erosion: the coarse distance-band envelope
is already table-like, fine structural synthesis is a no-op at product defaults,
and repeated uplift amplifies rather than originates the grammar. The next
decision is no longer another fixed-height mountain comparison. The
[mountain-system design basis](research/mountain-system-design-basis-2026-07-13.md)
finds that the current path collapses tectonic forcing into terrain height before
drainage can coevolve with it. At that historical rung, it selected the
[orogen organization testbed](research/orogen-testbed-spec-2026-07-13.md): a
dimensioned, time-resolved uplift–drainage–hillslope system compared against
locked hold-and-carve, synthetic-topology and explicit-skeleton controls. It
did not authorize immediate completion of all such response cases; the current
owner comparison and its four executable amendments now supersede that old
"next implementation" wording.

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
[evaluated](audits/channel-extraction-r1a-generator-control-2026-07-14.md).
Generator sampling repairs zero terminations and both arms still reject. It
also makes a linear-exact gradient feeding the same affine maximum-face P0
redundant. Before a general flux or V mechanism, preregister an affine-only
entry-point-aware continuous face-crossing discriminator.
That discriminator is now
[evaluated](audits/channel-extraction-r1a-affine-crossing-2026-07-14.md).
Analytic crossing validates the cap and continuous traversal in all 12 affine
cases. The reconstructed arm passes all 11 judged cases with identical face
sequences, but one 2 km all-cell reconstruction prerequisite narrowly fails;
therefore the maximum-face/F0 bundle has not yet been causally localized. The
next bounded rung is a separately preregistered stable affine linear-
consistency control which distinguishes polygon-mean input error from the
literal normal-equation solve. Do not advance to V, RT0, discharge integration
or product extraction before that prerequisite is resolved.
That control is now
[evaluated](audits/channel-extraction-r1a-stable-reconstruction-2026-07-14.md).
RQ does not rescue the one registered-input failure; ON and OQ pass 12/12 at
machine precision, and the failing stencil is well-conditioned. Registered
affine mean/difference numerics, not normal-equation stability, remain the
formal blocker. Do not promote QR. Either preregister one local-coordinate
polygon-mean evaluation control to close that numerical identity, or take the
explicit Pareto stop—the elevation discrepancy is about `1e-12 km`—and return
to the wider architecture question without claiming the graph bundle proven.
The rejected nominal rectangular boundary remains an important scope decision:
exact cut-cell hex geometry is not currently justified for a testbed intended
to transfer to a closed sphere. Earlier analytic evidence remains in the
[Slice 1R audit](audits/orogen-testbed-slice1r-2026-07-13.md).
