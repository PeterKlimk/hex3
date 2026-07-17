# Hex3 roadmap

Status: **current portfolio decision**, 2026-07-18.

This roadmap turns the [project thesis](thesis.md), [model strategy](model-strategy.md)
and [cross-system disposition](system-disposition.md) into a short, revisable
sequence. It is not a subsystem checklist or a history of experiments.

Hex3 optimizes for coherent emergence, visual appeal, explanatory depth,
iteration speed and “wow” value. Missing systems compete directly with fixing,
simplifying or deleting current systems.

## Rules

1. Preserve one usable product baseline while a replacement is judged.
2. Begin with the viewer or downstream outcome, not a favored mechanism.
3. Keep world state, semantic interpretation and presentation ownership distinct.
4. Prefer the cheapest model that preserves the important causal consequences.
5. Use one bounded vertical slice before deeper simulation or framework work.
6. A new owner replaces or retires overlapping machinery; it does not merely
   stack another control over it.
7. Implementation, a plausible screenshot or an improved scalar metric does not
   by itself earn promotion.

## Completed foundation

The documentation/decision substrate, first cross-system evaluation and hybrid
model-strategy decision are complete enough to govern work. The current product
has explicit unit, stage, validation and physical-versus-cartographic contracts.

Two bounded integration slices are also complete:

- [Water Geography V0](system-disposition.md#completed-bounded-slice-water-geography-v0)
  retains the cheap climate/hydrology spine and shared water, river, coast and
  repair-provenance account.
- [Living Surface V0](living-surface.md) retains on-demand fractional
  physiognomy and a selectable linear relief palette without creating a retained
  ecology stage.

The large mountain/landscape experiment campaign is historical evidence, not an
active roadmap. Its current conclusion is maintained in
[landscape strategy](landscape-strategy.md); exact contracts and outcomes remain
indexed under [research](research/README.md) and [audits](audits/README.md).

## Current portfolio choice

The previous “sediment, then human geography, then renewed tectonics” order was
inherited from candidate availability. It is retired. Completed slices are not
frontier items, and a scientifically attractive missing coupling is not
automatically the best product move.

| Product promise | Candidate intervention | Current decision |
|---|---|---|
| The product is lean enough to reason about and iterate | Separate product code/API/CLI from the experimental laboratory; remove unconditional rendering allocations | **Immediate enabling work** |
| Geography has visible consequences | Traversability, water/coast access, named relative opportunities, aggregate sites and least-cost routes | **Bounded discriminator complete; retain useful operators without vertical expansion** |
| Terrain is morphologically convincing | Replace scalar final-height ownership with one structural range/forcing owner or disclosed authentic range hack | **Selected next model rework** |
| The planet is readable and explainable | Consumer-driven regions, passes, crossings, chokepoints, labels and scale generalization | **Build only as pulled by the selected consequence layer** |
| Landscapes connect source to sink | Persistent mobile/deposited material, floodplains, terminal fill and deltas | **Research/design gate; not next by default** |
| The world feels more visibly alive | Forest structure, vegetation assets, seasons or disturbance | **Gated; accepted Living Surface does not authorize vertical expansion** |

This is a qualitative portfolio decision, not a scorecard. Each future card must
state the user outcome, owner changed or connected, causal reference, cheapest
honest slice, visible/downstream payoff, compute/architecture burden, kill
condition and cheaper alternative.

## Immediate sequence

### 1. Contract the product surface

Separate the ordinary application from the research laboratory without deleting
historical evidence or reusable operators.

- hide quarantined orogen models and default-off erosion/fine knobs from the
  normal product CLI;
- keep the landform/landscape testbeds and their binaries behind the explicit
  non-default `research-landscape` build/module boundary;
- keep the legacy terrain owner as the disclosed usable control, not as an
  endorsed final architecture;
- retain the now-lazy alternative colored fill meshes and relief edges plus the
  one-texel pre-hydrology river fallback; share or retire further inactive-stage
  resources only where measurement justifies it; and
- measure actual product build/runtime/memory changes, without turning cleanup
  into a new model campaign.

Exit condition: a contributor can identify the product path without navigating
rejected terrain rungs, and ordinary rendering no longer allocates hundreds of
MiB for unused stage-local resources.

### 2. [Consequential Geography V0](consequential-geography.md) — bounded decision complete

Use existing final terrain, physical grade, water geography and Living Surface
fractions to test whether the generated planet constrains one plausible
aggregate site-and-route map.

Minimum honest slice:

```text
terrain grade + water/coast access + relative living opportunity
  -> traversability and named opportunity/constraint components
  -> 12–30 aggregate sites
  -> least-cost terrestrial routes
  -> conservative lower-terrain-corridor evidence and cartographic diagnosis
```

This is an authentic aggregate hack, not population, economy, politics or full
civilization simulation. Regions and landform semantics are derived only where
the site/route consumer needs containment, barriers or crossings. V0 uses named
relative opportunity components, not ore-body, soil, crop-yield or generic
resource simulations.

Discriminator: independently ablate grade, freshwater, coast and living
opportunity. Sites and routes must change intelligibly; they must not be
invariant, collapse to “everything follows the largest river,” or primarily
track mesh density. The slice must create visible board/globe meaning and expose
which retained physical systems have real downstream consequences.

The first representative site-only packet rejects its candidate proposal. All
60 baseline anchors across three worlds collapse onto the exact intersection of
selected rivers, ocean coast, saturated local living opportunity and maximum
coast bonus before catchments can choose among a diverse support. The cheap
access/catchment substrate remains useful. The next bounded action is
preference-neutral, physically diverse candidate support plus the same tiered
counterfactual, not routes or parameter tuning. See the
[site-probe audit](audits/consequential-geography-site-probe-2026-07-17.md).

That correction now passes the collapse discriminator: joint river/coast
anchors fall to 11/60, the support contains all requested relationship classes,
and 512-candidate selection costs less than 0.3 s at about 255k cells. It remains
a provisional route-discriminator input, not a product prior. Freshwater and
catchment-scale living opportunity matter strongly; coast matters modestly;
site-local grade is nearly inert.

The bounded same-site route discriminator now earns that missing terrain
consequence. Across the three representative worlds, physical and zero-grade
arms select the same endpoint graph but only 47--56% of selected paths are
exact; selected edge overlap is 0.40--0.67, physical ascent is 31--48% lower,
and each network costs about 0.2 seconds at 255k cells. Retain the operator, but
do not promote the site prior or claim network-topology consequence. This led
to one final route-local lower-corridor discriminator using existing path
evidence—not population, site-weight calibration or another route parameter
campaign. See the
[route-probe audit](audits/consequential-geography-route-probe-2026-07-17.md).

That final relationship decision is also complete. Five to seven selected
routes per world contain an elementary branch that is longer in distance but
cheaper and lower under the physical traversal. The mechanism is reproducible,
but split/rejoin spans are continental (3,163--5,891 km) and only one of three
automatic images communicates it clearly. Retain the conservative
`lower-terrain-corridor` evidence and typed omissions; reject automatic gap,
pass, ridge-crossing and chokepoint semantics. Stop this vertical slice rather
than building a relationship ladder, calibrating sites or adding population.
See the [lower-corridor audit](audits/consequential-geography-lower-corridor-2026-07-18.md).

Kill condition: if authored weights dominate geography or the result is a
decorative overlay, stop. Retain only useful traversability/access semantics.

### 3. [Lithosphere Inheritance V0](lithosphere-inheritance.md) — explicit semantics pass, generated history next

The tableland/“long Uluru” range grammar remains the largest established visible
model defect. The product-native structural compiler has now shown that the
current source does not contain enough causal state to repair it honestly.

The selected task is therefore upstream state, not mountain response. Add
coherent basement provinces plus a sparse directional graph of inherited
sutures/rifts. This state is generated before terrain, has explicit topology
and orientation, and must serve both collision organization and rift
localization. A competence-noise raster cannot substitute for it.

The inert front/episode record, sparse finite-parent compiler and closed
shortening-opportunity ledger are implemented. Fixed attribution rejected seed
`12345`'s reviewed height component as one belt: its core merges one long
collision front and two disconnected subduction fronts.

The revised contract now selects entirely in the generated source domain. Its
primary continental-capable component is a coherent 3,254 km plate-pair system
with one long collision parent and two exact subduction transitions; the old
reviewed parent is independently rank three. Exact legacy response ownership
then binds it to one dominant 1.436 million km² visible component with no mixed-
seed ambiguity and negligible hydrologic-repair ownership.

The [Structural Mountain V0](structural-mountain.md) organization gate fails
cleanly. The 3,200 km collision parent has
substantial broad curvature and modest continuous kinematic variation, but one
uniform episode state. Its finite-end-tapered opportunity contains one
persistent maximum and no internal minimum. Bends can concentrate rather than
terminate shortening, so converting five curvature maxima into independent
tapers would manufacture the needed lows. Structural Mountain V0 stops with
`insufficient-causal-segmentation`; no terrain response is authorized.

The crust follow-up finds craton `3` on both sides of every parent edge and only
one broad ocean-margin-distance envelope. Existing craton labels, experimental
strain and lifecycle damage do not provide pre-collision structural memory.
Build and audit the new source state before revisiting terrain. If it cannot
produce coherent multi-consumer relationships cheaply, relax the universal
internal-hierarchy target and retain continuous finite massifs. Do not add
independent ridge noise, lower a bend threshold, revive the manufactured
organization ladder, tune amplitudes, or stack relief over the current scalar
height owner.

The first on-demand seam is now implemented without changing product terrain:
connected basement provinces, exact candidate province contacts and a generic
boundary/structure query. At 100k cells it costs about 0.05 s and would retain
about 0.83 MB of vector payload. The selected 3,200 km collision parent crosses
two short contact-aligned runs, but neither reaches a multi-trace incidence and
the shared Voronoi support can inflate exact overlap. These contacts have not
been assigned geological history; this is useful source structure, not yet a
license for segmented relief.

The explicit relationship gate now passes independently of generated history.
Endpoint hyperedges distinguish continuation, junction, finite offset transfer
and crossing-unlinked. Identical four-arm geometry produces two components when
declared as an unlinked crossing and one when declared as a junction; an offset
transfer is invalid without a finite connector. Collision and continental-rift
assessments share the same geological contact while retaining separate
applications. Generated basement state intentionally contains zero such
relationships, so the next gate is a plate-blind, terrain-blind prior selecting
a sparse finite suture/rift/transfer history. Only coherent fixed-corpus evidence
from that prior may reopen terrain response.

## Later decision set

- **Seasonal hydroecology:** a cheap two-season or monthly analytic cycle,
  snow/root-zone reservoirs and fixed-topology seasonal runoff may offer more
  visible coupling than deeper steady climate. Research after Consequential
  Geography exposes the need.
- **Sediment/source to sink:** require bedrock/alluvial ownership, one mobile
  load, time and mass contracts, one authoritative route, local lowland/coastal
  representation and one visible target before implementation.
- **Reduced ocean structure:** research a steady surface-current/SST anomaly
  authentic hack only if named coastal climates and living geography need it.
- **Cryosphere:** replace the old shaping pass with coherent ice geometry and
  meltwater ownership; terrain carving and loading come later if earned.
- **Richer vegetation:** forests and scale-aware assets remain presentation and
  semantic consumers of accepted cover, not permission for individual-tree
  ecology.

Full civilization/economics, dynamic atmosphere/ocean, full stratigraphy,
individual-tree ecology, generic dependency frameworks and physically based
material rendering are dominated at the current frontier.

## Continuous obligations

- Fix correctness, topology, unit, cache and convergence defects when found.
- Profile demonstrated iteration bottlenecks before optimizing them.
- Preserve Windows GPU validation and representative human visual review.
- Maintain Physical/Diagnostic/Authentic/Dramatic separation.
- Ask how reality or higher-fidelity simulations avoid an important defect,
  then preserve only the consequences Hex3 actually needs.
- Reopen this portfolio whenever evidence changes the product promise with the
  highest expected payoff.
