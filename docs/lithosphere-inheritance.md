# Lithosphere Inheritance V0 decision

Status: **source-only chronological history and initial consumer audit complete;
assembly/suture prior retained experimentally, paleorift arm not justified, and
nothing promoted into the product**, 2026-07-18.

This decision follows the source-only failure in
[Structural Mountain V0](structural-mountain.md). The contract below records the
smallest missing causal owner that was built and evaluated before any mountain
terrain response; it is no longer an active implementation instruction.

## Decision

The evaluated design added **tectonic memory** as two related but distinct forms
of on-demand pre-terrain state:

1. coherent basement/material provinces; and
2. a sparse directional graph of inherited sutures, rifts and their linkage or
   transfer relationships.

Do not add a generic competence-noise raster and do not revive the full moving
carrier/lifecycle model. The selected representation is an authentic systemic
hack: it preserves material identity, structural geometry, orientation and
connectivity without claiming a depth-resolved lithosphere simulation or a
reconstructed geological history.

The first implementation is source-only. It may organize plate-boundary and
rift response, but it cannot read elevation, drainage, landforms or rendered
relief and cannot write terrain directly.

## Why the existing state is insufficient

The accepted seed `12345` source audit now checks the existing crust seam, not
only boundary kinematics. All 70 edges of the 3,199.8 km collision parent place
craton `3` on both plates. There is no craton transition. Distance to the
continental margin rises from zero near one end to one broad interior maximum
and returns to zero near the other; after three-width smoothing it has one
maximum and no internal material boundary. It can support a finite continental
envelope, not a segmented belt.

That label also has a weaker causal meaning than its name suggests. Current
cratons are noisy flood-fill regions generated independently of plates and
overlaid on them. Their margin distance records only continent-ocean
transitions. It does not record terrane boundaries, sutures, rifts, material
age or a plate-carried assembly history. Calling craton `3` an inherited suture
would therefore turn a layout label into fictional geology.

Other implemented fields do not fill the gap:

- legacy product `thin_sheet_strain` and compression axes are zero;
- experimental thin-sheet strain is a response to the present forcing, not an
  antecedent cause;
- lifecycle weakness/fabric begins empty and is written by collision, so it
  cannot explain where that collision first localizes; and
- the current craton macro field is generated during elevation assembly as
  thickness variation. Reading it back into tectonic organization would invert
  ownership, and its scalar position noise has no linkage or direction.

The evidence is retained in
[`generated/structural-mountain-seed-12345-organization-audit-v1.json`](generated/structural-mountain-seed-12345-organization-audit-v1.json).

## Reality and simulation prior

Reality avoids a universal smooth collision roof through several interacting
causes: inherited rifts and sutures localize later strain; their offset,
orientation and connectivity partition deformation; thermal/mechanical age
changes broad strength and detachment style; active structures grow, link and
abandon; and surface processes express the supplied differences.

The important reduction is structural, not equation count:

- 3-D rift-inversion models produce different range continuity, polarity and
  transfer-zone topography from different inherited rift linkage and offset
  ([Wolf et al. 2026](https://doi.org/10.1038/s41467-025-66695-8));
- coupled extension-inversion models show inherited weak zones controlling
  localization and orogen width, with erosion and sediment modifying rather
  than inventing that organization
  ([Erdős et al. 2014](https://doi.org/10.1002/2014JB011408));
- lithospheric age changes the broad style and amount of shortening
  ([Mouthereau, Watts and Burov 2013](https://doi.org/10.1038/ngeo1902)); and
- practical geodynamic models retain accumulated damage as material history
  and directional fabric when scalar viscosity is insufficient
  ([Glerum et al. 2018](https://doi.org/10.5194/se-9-267-2018),
  [Duretz et al. 2025](https://doi.org/10.1029/2025GC012409)).

A scalar weakness field can order localization intensity. It cannot say that
two structures connect, that one ends, that offset segments form a transfer
zone, or that their directions favor common versus opposing polarity. The
earlier manufactured inheritance lattice demonstrated the product consequence:
it created separated hot spots and a bead grammar, not a credible range
network.

## Reduced state contract

The conceptual state is:

```text
continental/crust initializer
  -> coherent basement provinces
  -> inherited suture/rift segments and graph relationships
  -> present plate-boundary and rift interaction
  -> finite deformation/uplift opportunity
  -> evolving drainage and hillslopes
  -> physical terrain
  -> semantic ranges/divides/passes
  -> declared cartographic presentation
```

The minimum material record is a coarse province identity plus an optional
effective thermo-mechanical competence rank. Competence is a disclosed
depth-integrated proxy, not viscosity or crustal age in physical units.

The minimum graph record retains:

- exact spherical support and host/adjacent province identity;
- finite segments with tangent/director, width class and relative maturity;
- structure class, initially suture or inherited rift;
- tips, continuations, offsets, linkage/transfer nodes and junctions; and
- provenance sufficient to distinguish initialized inheritance from damage
  created by the current tectonic episode.

Erodibility or lithology is not the same quantity as tectonic competence. A
later province material record may correlate them, but each keeps separate
ownership and counterfactual behavior. Observed erodibility contrasts can
reorganize drainage and capture, which gives province identity a valuable later
consumer without justifying that conflation
([Forte 2018](https://doi.org/10.1016/j.epsl.2018.04.029)).

## Pareto implementation rung

V0 should preserve the current continental envelope while subdividing each
large generated craton into a small number of coherent basement terranes on the
same coarse Voronoi graph. Constrained growth supplies province identity and
exact internal contacts. A sparse subset of those contacts plus bounded finite
rift traces supplies the inherited graph. Randomness may choose unresolved
province history or irregularity, but it may not choose mountain highs, lows or
passes after seeing terrain.

This rung is intentionally smaller than geological-time replay:

- no mantle convection, force-derived plate motion or depth-resolved rheology;
- no new moving raster/carrier;
- no tensor field when a graph tangent plus strength is sufficient;
- no direct elevation, ridge skeleton or pass placement;
- no guarantee that every convergent belt is segmented; and
- no erosion or renderer changes.

If a generated collision has no relevant inherited relationship, the honest
source outcome is one continuous finite massif. The project-level model should
produce several belt grammars across worlds; every individual belt need not be
forced to contain a transfer low.

## First bounded task and gates

Implement and inspect the state seam before changing the product surface:

1. add deterministic province identity and an exact sparse structure graph to
   the coarse initializer;
2. compile graph/front relationships for manufactured continuous, aligned,
   crossing, offset-linked and unrelated cases;
3. audit the seed `12345` source and a small fixed corpus entirely before
   terrain, reporting intersections, aligned runs, tips and transfer nodes;
4. expose the same state to a rift-localization query so the owner has a second
   real consumer; and
5. only then decide whether the structural mountain compiler has enough causal
   evidence to emit more than a finite continuous parent.

The rung passes only if:

- province and graph state are independent of elevation, drainage and target
  selection;
- graph topology and physical-scale statistics are stable enough under coarse
  resolution changes to support the claim;
- generated contacts are sparse, coherent and non-periodic rather than a
  disguised noise lattice;
- manufactured orientation/linkage counterfactuals change the classified
  structural relationship in the expected direction;
- at least collision organization and rift localization can consume the same
  state without sharing a painted result; and
- time and memory remain negligible beside existing Stage-1 feature generation.

Stop if graph generation merely guarantees attractive subdivisions, if a
scalar score again becomes independent hot-spot selection, or if the graph has
no useful consumer beyond mountain styling. Do not tune terrain to rescue the
state seam.

## Basement skeleton checkpoint

The first implementation generated deterministic basement provinces inside
the existing continental/craton envelope and compiles every exact internal
province contact into a typed spherical graph. It also exposes one generic
plate-boundary relationship query: a boundary may be unrelated, coincide with
an inherited edge or touch one at an exact vertex, with shared graph segments,
geometric endpoint incidence and the minimum unoriented tangent angle reported.
The canonical cell-edge identity now belongs to tessellation rather than the
mountain module.

This was deliberately built first as an on-demand diagnostic seam. It is not
retained in `World`, selected by the ordinary product, cached, or read by
elevation, drainage, semantics or rendering. At this checkpoint the generator
emitted candidate basement contacts only. Divergent boundaries gained a
separate inert application assessment, but not a localization response or
terrain consequence.

The fixed seed `12345` report at 100,000 coarse cells records:

- 68 connected provinces with median area 2.021 million km²;
- 4,157 exact contact edges compiled into 134 open segments, totaling
  179,107 km;
- about 0.05 s to generate the state in a release build and approximately
  0.83 MB of retained vector payload if it were kept; and
- 20 contacts along 2,157 divergent boundary edges, including eight exact
  overlaps, showing that the same relationship query is not collision-specific.

The selected 3,199.8 km collision parent has 60 unrelated edges, six coincident
edges and four vertex contacts. Its coincidences form two short runs on separate
basement-contact segments, around 1,423--1,506 km and 1,709--1,807 km along the
parent; neither run reaches a multi-trace endpoint incidence, and neither has an
explicit geological relationship. This is new source information, but it does
**not** yet identify either contact as a suture or justify a transfer low,
segmented uplift or any topographic sign. Because both graphs use the same
coarse Voronoi edges, exact coincidence also overstates geometric precision and
must not be treated as independent validation.

Spot checks at 25k, 50k, 100k and 200k cells retain roughly the intended
province scale and yield 125, 101, 134 and 152 graph segments respectively.
That is adequate for a prototype seam, not a convergence claim: current world
initialization itself changes geometry with resolution, and the 50k graph is a
moderate outlier.

The checkpoint therefore passes independence, determinism, connected-province,
exact-geometry and cost tests, but does not pass the complete V0 gate. Compiling
all terrane contacts is only a basement skeleton, not yet the selected sparse
history model.

## Explicit relationship checkpoint

Geometric coincidence is no longer allowed to imply geological connectivity.
The graph now has a separate endpoint-relationship layer with four declarations:

- continuation between two compatible trace ends;
- a three-or-more-arm junction at one exact vertex;
- offset transfer through a finite `TransferLink` segment whose endpoints close
  exactly against the two primary traces; and
- crossing-unlinked, which pairs four coincident ends into two branches while
  explicitly forbidding cross-branch connectivity.

Unreferenced endpoints remain tips. Candidate `BasementContact` segments cannot
participate in these relationships; they must first receive an explicit suture
or inherited-rift interpretation. Every endpoint can belong to at most one
relationship, and canonical endpoint ordering makes serialization deterministic.
The basement skeleton emits an empty relationship set. Generated history also
leaves it empty unless an event supplies an exact relationship; it does not turn
incidence into geology.

The decisive manufactured counterfactual now passes. Four arms with identical
vertex geometry and an explicit crossing-unlinked declaration compile into two
connected components. Replacing only that declaration with a four-arm junction
compiles the same geometry into one component. A continuation joins two aligned
traces. Two offset traces connect only when a finite transfer segment spans their
endpoints; changing that connector endpoint makes validation fail. Thus topology
comes from declared source history, not proximity, angle or mesh incidence.

A shared inert assessment also distinguishes current consumers without painting
a result. The same named inherited-rift contact remains the same geology when
read by continental collision or continental rifting, while the application is
typed separately. Ocean-ocean divergence remains spreading, and a pair-level
divergent edge that is locally closing remains ineligible. The assessment assigns
no localization strength, deformation sign or topographic consequence.

The corrected fixed audit calls the automatically compiled endpoints geometric
incidences: 64 tips and 68 multi-trace incidences. It records zero explicit
geological relationships globally and on the selected collision parent. The 20
divergent contacts remain candidate basement coincidences, not inherited-rift
localization. The retained report is
[`generated/lithosphere-inheritance-seed-12345-v1.json`](generated/lithosphere-inheritance-seed-12345-v1.json).

This passes the explicit-topology and manufactured multi-consumer semantics gate.

## Chronological source-history checkpoint

The selected prior is chronological rather than a random structural mask:

```text
coherent basement provinces
  -> per-craton terrane assembly forest
  -> finite preserved portions of assembly sutures
later independent extension axes
  -> finite intra-province paleorift traces
present collision/rifting
  -> typed application only
```

Each assembly event joins two previously separate terrane components. Only a
bounded physical-length portion of its contact remains as named inherited
structure; the rest stays an ordinary candidate basement contact. This keeps the
history causally sufficient to assemble each generated craton without claiming
that every contact or the whole original suture is preserved equally.

After assembly, at most four cratons receive one later paleorift event. Its host
province and latent great-circle axis are selected without present plates or
terrain. The trace follows a simple exact Voronoi-edge path inside that province
to a target measured in kilometres, bounded by host scale. This is a coarse
rift-system envelope, not a claim that natural faults follow cell edges.

Every named edge and compiled segment carries its chronological source event.
Candidate contacts carry none, and source-event identity is part of segment
chaining, so coincident histories cannot merge accidentally. A validator checks
that assembly events own only sutures between different provinces, paleorift
events own only intra-province rifts, all events own support, and explicit
relationship topology remains valid.

Basement and history seeds are independently controllable. Resampling only
history must leave cell/province ownership and the complete candidate-contact
geometry fixed while changing which finite traces are named. This is the key
causal counterfactual; present plates, dynamics, features, mountains, elevation,
hydrology and rendering are absent from the generator API.

V0 intentionally generates no transfer, junction or crossing relationships yet.
Those semantics remain tested and available, but absence is more honest than a
proximity repair or a mandatory showcase. Exact later reuse of an assembly
suture is also deferred because the current one-event-per-edge provenance would
have to become an overlay rather than overwrite the older event. The audit must
show that this additional expressiveness pays before adding it.

The next gate is therefore empirical, still before terrain: inspect physical
trace scale and sparsity, verify the fixed-history counterfactual, and ask whether
real continental-collision and continental-rift boundaries encounter useful
named state. Do not connect the history to localization or topography before
that evidence exists.

## Initial source audit and verdict

The compact source audit passes the assembly/suture representation gate but not
the paleorift gate, and it does **not** reopen the reviewed mountain response.

At seed `12345` and 100k coarse cells, 63 assembly events and four paleorift
events name 42,956 km of structure against 179,107 km of original candidate
contact. The named/contact length ratio is 24.0%; preserved sutures alone own
22.3% of candidate-contact length. Of named length, 96.9% belongs to a
segment at least 300 km long with at least four exact source edges, while 0.15%
belongs to one-edge fragments. Median segment length is 570 km for sutures and
725 km for paleorifts. On-demand generation takes about 0.07 seconds in the
release audit.

The assembly reduction is provisionally stable rather than edge-count stable.
At 50k / 100k / 200k cells for seed `12345`, named/contact ratios are 21.1 /
24.0 / 22.3%, coherent named shares are 93.7 / 96.9 / 96.7% and suture medians
are 578 / 570 / 539 km. Paleorift medians are 483 / 725 / 405 km; four traces
are too few and too unstable to support the same claim. The generated worlds
themselves are not feature-identical across resolution; these are distributional
checks only.

At 100k cells, spot-check seeds `12345`, `8675309` and `1001` have named/contact
ratios of 24.0, 18.7 and 22.5% and coherent named shares of 96.9, 92.1 and
90.4%. Every world has named suture contacts under both real continental
collision and real continental-rifting applications. Seed `12345` also has six
collision edges contacting a paleorift. No tested continental-rifting edge
contacts a named paleorift, and generated explicit relationship count remains
zero. Suture reactivation therefore supplies the demonstrated second consumer;
the new paleorift traces do not yet demonstrate a present rift-nucleation
consumer.

The fixed-basement counterfactual behaves correctly. Resampling only history at
seed `12345` preserves candidate geometry exactly, changes named-edge Jaccard to
0.153, keeps named length close (42,956 versus 43,429 km) and changes 51 current
plate-boundary contact locations.

Most importantly, all ten contacts on the selected 3,200 km collision parent
remain unnamed candidate basement contacts. They carry no history event. The
prior therefore supplies no new causal reason to divide that parent, and
Structural Mountain V0 remains stopped at one finite continuous opportunity.
Choosing another history seed until this target receives a convenient suture
would violate the decision.

Retain the assembly/suture generator as an inexpensive experimental source seam.
The current paleorift arm is an implemented negative result: its physical-scale
evidence is unstable and it has no demonstrated rift-nucleation consumer. Do not
store either arm in `World`, tune them toward the reviewed mountain, or connect
them to terrain. Before retaining or expanding paleorifts, transfer generation
or multi-event edge overlays, require one genuine consumer that can use interior
inherited rifts to alter where extension nucleates; the present boundary-contact
query is not that mechanism. The retained seed report is
[`generated/lithosphere-inheritance-seed-12345-v2.json`](generated/lithosphere-inheritance-seed-12345-v2.json).

## Consequences for current work

Structural Mountain V0 is paused at its successful source compiler and rejected
organization gate. Its compiler, target attribution and legacy observation
binding remain useful evidence. Its same-belt terrain replacement is no longer
the immediate task because the accepted source lacks the state needed to make
that replacement honest.

The expensive lifecycle implementation remains quarantined. Useful concepts—
material identity, persistent damage, rotated fabric and conservative ledgers—
may be extracted later, but the named model is not a prerequisite for V0.

The portfolio has now taken that alternative: drop continuity alone as a
rejection criterion for the reviewed parent, relax the universal
internal-hierarchy target and spend the next project budget on
[World Readability V0](roadmap.md#2-world-readability-v0). This does not endorse
the legacy generic tableland owner; the ordinary-world terrain escalation gate
remains explicit. The source test earned a useful experimental assembly/suture
seam, not guaranteed promotion or further topology work.
