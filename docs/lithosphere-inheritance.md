# Lithosphere Inheritance V0 decision

Status: **selected upstream architecture; source-state prototype not yet
implemented or promoted**, 2026-07-18.

This decision follows the source-only failure in
[Structural Mountain V0](structural-mountain.md). It chooses the smallest
missing causal owner worth building before any mountain terrain response.

## Decision

Add **tectonic memory** to the coarse world model as two related but distinct
forms of pre-terrain state:

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

## Consequences for current work

Structural Mountain V0 is paused at its successful source compiler and rejected
organization gate. Its compiler, target attribution and legacy observation
binding remain useful evidence. Its same-belt terrain replacement is no longer
the immediate task because the accepted source lacks the state needed to make
that replacement honest.

The expensive lifecycle implementation remains quarantined. Useful concepts—
material identity, persistent damage, rotated fabric and conservative ledgers—
may be extracted later, but the named model is not a prerequisite for V0.

The alternative remains explicit: if sparse inherited state does not create
coherent multi-consumer relationships cheaply, retain finite continuous
massifs, relax the universal internal-hierarchy target, and spend the project
budget on a higher-value missing system. The decision is to test the causal
owner, not to guarantee its promotion.
