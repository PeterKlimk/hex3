# Terrain architecture design space

Status: **current design decision; coupled scalar-support Slice A, RDS0 terrain
ownership and the bounded B0 channel/hillslope dual are insufficient; no
terrain replacement or next owner is selected**, 2026-07-23.

This document defines the terrain ownership boundary and the smallest credible
alternatives to Legacy. It does not change product behavior. Legacy remains the
usable control until a replacement earns promotion.

See the [landscape strategy](landscape-strategy.md),
[terrain causal-attribution audit](audits/terrain-causal-attribution-2026-07-21.md),
[tectonic source-viability audit](audits/tectonic-source-viability-2026-07-21.md)
and historical
[mountain-system design basis](research/mountain-system-design-basis-2026-07-13.md).

## Decision

Do not select a monolithic “terrain owner.” Define one coherent ownership
boundary containing explicit cooperating systems for:

1. deformation and material opportunity;
2. rock-uplift forcing through a small number of meaningful relative epochs;
3. authoritative surface state and material accounting;
4. evolving drainage, incision and nonlinear hillslope response; and
5. derived drainage/divide/range evidence for later consumers.

Three decisions must remain separate:

- **forcing:** where, when and at what rate rock is advected through the surface
  control volume, distinct from any creation or accretion of solid material;
- **response:** how drainage and hillslopes turn that opportunity into relief;
- **representation:** which state is explicit on the mesh and which compatible
  detail is reconstructed below or between process scales.

Prior work repeatedly entangled these decisions. A better interpolator cannot
repair a causally uniform forcing program; a more expensive erosion loop cannot
guarantee missing source organization; and a detailed mesh cannot create a
range hierarchy absent from both.

The two architecture families were initially ordered as:

- **A — finite-age coupled landscape evolution**, initially using static
  present-front supports, the reduced physical candidate;
- **B — causal dual drainage/divide construction**, a cheap authentic-hack
  upper bound and fallback.

Slice A and the subsequent RDS0 moving-support composition have now supplied
that discriminator. They materially change relief and receivers, but still let
a scalar positive opportunity field own regional topography. RDS0's apparent
linked lows resolve into dense source-linked closed depressions rather than a
coherent drainage/divide system. The condition for B was therefore met. The
bounded B0 implementation has now tested and rejected that upper bound as
sufficient ownership. Do not tune B0 or select another terrain owner by
momentum; the next task is to synthesize what the coupled and constructed
failures jointly imply.

Three plausible responses to the RDS0 result remain distinct:

| Response | What it would test | Decision now |
|---|---|---|
| Persistent thickness/material state plus coupled evolution | Whether advection, retention and relaxation remove lobe/depression grammar | Physically attractive but does not isolate the newly demonstrated topology defect; existing lifecycle machinery is too broad and unresolved |
| Source-conditioned erodibility or drainage feedback inside the current loop | Whether fabric or drainage can organize an otherwise valid uplift state | Possible later ingredient; alone it leaves the rejected positive-lobe owner intact and risks repeating A4's circular uplift feedback |
| Channel/hillslope dual with compatible reconstruction | Whether explicit drainage/base-level relations are the missing cheap causal consequence | **Implemented bounded upper bound; rejected as sufficient ownership** |

## Product outcome and causal reference

The desired outcome is not “fewer plateaus.” Hex3 should produce several
causally distinct mountain families whose ranges, massifs, ends, saddles,
divides, valleys, basins and trunk rivers agree with one another. Broad
low-relief highlands remain legitimate when supported by long-wavelength block
uplift, incomplete dissection, drainage starvation, internal basins or a
regime-specific process such as glaciation. A wet active collision belt should
not receive the same smooth roof by default.

The design reference is:

```text
plate/front history + crust/material setting
  -> finite deformation and material opportunity
  -> time-dependent rock-uplift program
  -> authoritative evolving surface + control-volume solid ledger
       <-> drainage growth, competition, divide migration and capture
       <-> drainage-area incision and real base levels
       <-> low-slope creep and threshold hillslope response
  -> final physical terrain + drainage/catchment/divide evidence
  -> water, climate, living surface, routes and cartographic presentation
```

Finite deformation support supplies range ends. Segmented or heterogeneous
support, overlap geometry and localized migration may supply large structural
lows. Drainage competition supplies much of the branching valley/divide skeleton.
Finite age prevents every belt from reaching the same steady state. Nonlinear
hillslopes turn stronger forcing primarily into faster wasting rather than
arbitrarily steep roofs. Structured erodibility can pin divides, organize
longitudinal valleys and retain resistant ridges, but must come from shared
material state rather than independent texture.

These consequences are supported by reduced landscape and divide-migration
models: finite fault-driven topographic growth
([Curry and Barnes 2015](https://doi.org/10.1130/GES01156.1)),
time-dependent stream-power response
([Whipple and Tucker 1999](https://doi.org/10.1029/1999JB900120)),
ridge/valley spacing from drainage competition
([Perron et al. 2009](https://doi.org/10.1038/nature08174)), divide migration
and capture ([Willett et al. 2014](https://doi.org/10.1126/science.1248765)),
uplift/advection control of divide position
([He et al. 2021](https://doi.org/10.1038/s41467-020-20748-2)), and
threshold hillslope behavior
([Larsen and Montgomery 2012](https://doi.org/10.1038/ngeo1479)).
The eSCAPE model demonstrates that spatial/temporal uplift, routing,
stream-power incision, creep and source-to-sink accounting can form a compact
landscape model without continuum geodynamics
([Salles 2019](https://doi.org/10.5194/gmd-12-4165-2019)).

## Minimum consequences to preserve

| Consequence | Minimum representation | Not required initially |
|---|---|---|
| Finite and heterogeneous mountain support | Two-dimensional rock-uplift opportunity with honest ends, local variation and overlap | Explicit faults, folds or crustal rheology |
| Transience | Relative component age, activation and incomplete adjustment; deactivation only when retained by a future source | Calibrated geological chronology |
| Branching relief | Recomputed drainage area and base level while terrain evolves | Event hydrology or hydraulic flow depth |
| Mobile divides | Basin ownership may change; capture is possible rather than scripted | A complete persistent landform ontology |
| Bounded steep terrain | Low-slope transport plus material-aware nonlinear transport/relaxation near a threshold, never a hard slope clip | Explicit landslide bodies |
| Shared heterogeneity | Optional coherent erodibility/material classes with a homogeneous control | Independent height or lithology noise |
| Explainable mass change | Separate uplift-through-datum, erosion, bulk deposition/storage and export ledgers with a declared control volume | Grain classes, stratigraphy or full flexure |
| Downstream identity | One final terrain and compatible receivers, water bodies and river hierarchy | Final cartographic generalization inside the solver |

Full viscoplastic tectonics, earthquake cycles, mantle convection, detailed
sediment classes, flexure, glaciers and two-way climate feedback are omitted.
They may later own named regimes or consequences; none is the generic cure for
the current roof.

## Current code boundary

The product already has useful ingredients:

- classified spherical plate-boundary adjacencies with convergence, shear,
  regime and polarity, plus an on-demand compiler for exact convergent arcs;
- retained boundary episodes and material/work diagnostics;
- a source-only finite-front compiler with an opportunity ledger;
- a fixed adaptive fine mesh, pre- and post-erosion hydrology, normalized
  precipitation supply, default SFD stream-power incision and linear hillslope
  diffusion, with nonlinear hillslopes implemented but default-off;
- raw hydrology plus lightweight river selection used by rendering and Living
  Surface, and richer water/river semantics used by dossiers and routes; and
- a quarantined landscape solver with dimensioned forcing frames, terrain
  revision, adaptive stepping and conservative ledgers.

But these pieces do not form the product ownership boundary:

- Legacy converts normalized/diffused arc and collision response directly into
  coarse height, then repeatedly injects a scaled thickness source derived from
  the same arc/collision load plus rift state during erosion;
- history and work arrays are computed but Legacy terrain does not consume
  them; model-specific material/lifecycle arrays remain empty or zero unless
  their experimental model is selected;
- the product erosion loop and final hydrology contain separate routing logic;
- material and erosion ledgers are transient or experimental rather than
  retained product state;
- the mesh is fixed before drainage reorganizes and extensive coarse-to-fine
  transfer is not conservative; and
- there are competing terrain authorities: ordinary rendering and general
  active-elevation access read pre-integration `FineSurface::elevation`, while
  routes, Consequential Geography and hydrologic semantics read hydrology's
  post-repair elevation.

A replacement therefore needs a typed, unit-bearing seam rather than another
height field. Rock uplift advects material and changes surface elevation; it is
not itself solid creation. Any ledger must define its control volume and keep
crustal/material accretion separate:

```text
DeformationProgram
  -> RockUpliftFrame[]
  -> LandscapeState { authoritative elevation, revision, solid ledger/storage }
  <-> DrainageState { receivers, accumulation, catchment ownership }
  -> FinalTerrain { elevation, hydrology, repair provenance }
```

Names are illustrative, not a request to build a framework first. Adapt existing
operators directly into the first vertical slice; generalize an interface only
after the slice demonstrates value.

## Source-viability result

Legacy history is adequate for a deliberately narrow first slice, not for the
full causal reference above. `BoundaryEpisode` describes a connected component
of the **present** boundary and attaches one inferred contact age to it. It does
not retain past boundary geometry, rate changes, deactivation, polarity or
receiving-side changes, or material transitions. Replaying it honestly therefore
means that fixed present-day components switch on at different ages; it does not
mean that a front migrates or reorganizes.

The fixed 100k-cell worlds nevertheless contain enough between-component age
variation to test unequal belt maturity:

| Seed | Convergent components | Derived onset intervals | Largest opportunity share | Max adjacent normalized composition L1 |
|---:|---:|---:|---:|---:|
| 12345 | 17 | 12 | 27.7% | 1.074 |
| 8675309 | 14 | 10 | 31.1% | 0.528 |
| 9001 | 12 | 8 | 33.5% | 0.989 |

These are planet-wide composition changes as different fixed belts activate.
Within every connected component, temporal spatial forcing is exactly rank-one:
the geometry, local rate and regime are constant and only an onset scalar
changes. The large composition-centroid shifts are consequently **not** front
migration. The `0.10` L1 screen is a useful discrimination heuristic, not a
physical threshold.

Proceed with A as a static-support, finite-age test of landscape response. It
can answer whether evolving drainage and hillslopes turn differently aged belts
into coherent, variably dissected terrain. It cannot answer whether migrating,
abandoned or reversing deformation would solve the roof grammar. If A produces
only different amplitudes or the same internal roof on each belt, treat missing
deformation history as an upstream causal limitation; do not compensate with an
erosion tuning ladder.

## Architecture options

| Family | What it preserves | Cost and principal risk | Disposition |
|---|---|---|---|
| Legacy height/rebuild variants | Broad tectonic location and cheap usable relief | Generic distance-band grammar; tuning and unsmoothing already discriminated | **Control only** |
| Finite deformation organizer alone | Honest ends, source provenance and bounded opportunity | Current source may be genuinely continuous; prior finite parents remained smooth massifs | **Input component, not terrain owner** |
| Finite-age coupled landscape evolution | Uplift–drainage–divide–hillslope feedback and unequal belt maturity | Present supports are frozen; repeated routing can be expensive | **Slice A, narrow source gate passed** |
| Channel/hillslope drainage dual | Sparse hydrologically compatible channels, divides and regional relief at low cost | Can paint the answer, repeat a grammar or reconstruct amoebas/steps | **Selected bounded Slice B upper bound after RDS0 rejection** |
| Multiscale erosion, adaptive cells or sub-grid synthesis | Buys tributaries and local texture after regional structure exists | Patch/LOD drainage seams and self-similar grooves; cannot repair macro grammar | **Later representation strategy** |
| Thin-sheet/lifecycle/geodynamic depth | Richer deformation and material history | Existing experiments cost heavily and have not earned visible organization | **Reference or source research, not next** |

Graphics precedents reinforce this separation. Uplift plus a stream graph can
create shared rivers and watersheds cheaply
([Cordonnier et al. 2016](https://doi.org/10.1111/cgf.12820)). Drainage-first
generation makes channel hierarchy and watersheds authoritative
([Génevaux et al. 2013](https://doi.org/10.1145/2461912.2461996)). Orometric
divide trees expose a sparse peak–saddle range hierarchy
([Argudo et al. 2019](https://people.cs.uct.ac.za/~jgain/wp-content/papercite-data/pdf/argudo2019.pdf)).
Multiscale erosion can add detail while retaining a coarse organization, but is
limited by global drainage continuity and whole-surface cost
([Schott et al. 2024](https://doi.org/10.1145/3658200)). These are architecture
precedents and authentic-hack controls, not evidence that authored graph rules
are geology.

## Slice A — finite-age coupled landscape evolution

Hypothesis: an honestly finite-age rock-uplift program on fixed present-front
supports, combined with mobile drainage and nonlinear hillslopes, can produce
coherent regional organization without an authored ridge skeleton.

The minimum slice:

1. derives a few relative onset frames from exact present convergent fronts and
   their supported episode ages;
2. keeps each component's geometry, local rate, regime and material setting
   fixed, and reports unsupported migration, deactivation or transitions rather
   than inventing them;
3. starts from the same non-orogenic base, climate/runoff and sea-level policy
   as Legacy;
4. applies uplift incrementally while recomputing routing, drainage area,
   stream-power incision and nonlinear hillslope response;
5. retains an elevation revision and separate uplift-through-datum,
   removal, bulk deposition/storage and export ledger; and
6. runs final water semantics from that same authoritative surface.

The quarantined landscape solver is an operator source, not a package to
promote wholesale. Its dimensioned forcing, adaptive stepping and ledger are
valuable; its planar mesh, fixture APIs and experiment artifact machinery are
not product requirements.

Kill A if its change is merely fine texture, if selected ranges retain the same
roof grammar in at least two of the three fixed worlds, or if drainage basins,
trunks and divides do not respond coherently to forcing counterfactuals. A
visually promising prototype may be unoptimized. Before promotion it must
retire overlapping buffers/operators, fit the ordinary Stage-4 memory envelope
and state an accepted runtime budget; any remaining cost increase must buy a
material visible or downstream consequence.

This kill result distinguishes two failures. If drainage and divides barely
move under valid age/onset counterfactuals, the coupled response has not earned
itself. If they respond but each fixed component still becomes one internally
uniform roof, the next missing owner is time-varying deformation support (or the
explicit topology upper bound B), not more response calibration.

### Current checkpoint

The first quarantined vertical slice is implemented at `6010d3c`. It demotes the
entire Legacy convergent height contribution, gives each fine source cell one
nearest exact present-front owner, and derives its uplift rate and finite active
suffix from positive local convergence and retained episode age. It then uses
the ordinary Stage-4 rerouting, incision, nonlinear hillslope, deposition and
final hydrology path. Present supports remain frozen; this is deliberately not
a migration model. A separate diagnostic can match the counterfactual all-old
integral, but the candidate does not: finite age changes integrated work.

The three-world pilot shows recurring structural change rather than mere
texture: summits descend more steeply, crests are generally closer and trunk
flow is more often transverse. It also roughly halves mountain-land coverage,
but human review finds the matched visuals much better and clears the
architectural direction.

At the ordinary roughly 256k-cell surface, maximum height remains within 5% of
Legacy in every world while 25 km relief p90 rises 34--126% and median summit
descent rises 153--665%. Global pit and checkerboard rates do not worsen. The
candidate therefore concentrates relief into narrower organized ranges instead
of merely deleting amplitude or adding global noise. Wall time and peak memory
are at practical parity in one cache-disabled pair per world; exact-front
ownership still adds aggregate CPU work and should be reused before promotion.

The component correspondence pass localizes the remaining uncertainty. It
cross-checks exact edge/episode identity, reconstructs the unsmoothed target-land
builder and finite-age schedule, and measures final response only on the supplied
target-land footprint. The three ordinary worlds give:

| Seed | Static builder budget (million km³) | Present-support share | Finite-age share of supported | Scheduled→positive-response share L1 / Spearman |
|---|---:|---:|---:|---:|
| `12345` | 45.9 | 91.2% | 55.0% | 0.212 / 0.904 |
| `8675309` | 37.2 | 96.1% | 67.6% | 0.230 / 0.917 |
| `9001` | 25.4 | 91.4% | 52.3% | 0.180 / 0.881 |

Seed `9001` is therefore not an anomalous downstream under-response. Its source
model supplies about half the scheduled volume of the other worlds, its largest
exact-opportunity component has no target-land source cells, and the coupled
response expresses the work it is actually given about as consistently as the
other worlds. This validates source-to-response correspondence, not the physical
adequacy of frozen present supports or the land gate.

The repeated-tooth discriminator now localizes the defect upstream. At the
ordinary capped surface, seed `8675309` episode `9` has 73 owner-convergence
peaks, 80 scheduled-uplift peaks and 75 final-crest peaks along one actual chain;
their mean contiguous spacings are 162, 155 and 173 km. Owner convergence
predicts final crest strongly (`rho=0.871`), including adjacent-station changes
(`rho=0.688`), whereas the demoted substrate has only 15 detected peaks and a
weaker `rho=0.404`. The thresholds are diagnostic rather than calibrated
landform semantics, but the source-to-final spatial correspondence is enough:
the coupled landscape is not creating this cadence from smooth forcing.

Slice A therefore remains useful but exposes a source-flux coupling error. The
exact episode carries `411,625 km²/Myr` signed normal flux and `502,096 km²/Myr`
after per-edge positive clipping. Exact-midpoint reevaluation is bit-near
equivalent to the stored rate, so location is not the cause. Continuous normals
at ±127/±382 km materially change orientation and reduce local maxima, but also
increase signed flux by 26.5/33.2%; direct normal replacement is not
conservative and is rejected.

Signed-rate aggregation before clipping is the accepted source operation. The
implemented finite-volume arm redistributes signed rate along one uninterrupted
causal segment with edge length as conserved measure and no-flux ends. On seed
`8675309`, episode `9`, both tested scales close the `411,625 km²/Myr` ledger to
roundoff. The 127 km arm removes 66.0% of rectification excess while retaining
25 positive maxima and `0.692` raw/output correlation. Broadening to 382 km buys
only another five percentage points of rectification reduction but collapses
the source to six maxima and `0.601` correlation. Select 127 km as the
production-shaped source candidate and retain 382 km only as the rejecting
sensitivity.

The coupled implementation is now an explicit research-only finite-age source
model. It applies the fixed 127 km operation to every emitted causal segment,
maps the signed result back by exact edge ID, then performs positive
classification in the unchanged frozen-support source. Raw remains the default;
closed-loop and other omission edges remain explicit and unchanged. On the
complete seed-`8675309` source the global signed ledger closes to `4.7e-9`
km²/Myr over 2.48 million km²/Myr, while positive-clipped flux falls 10.7%.

The first coupled comparison is a fixed-budget spatial-grammar test because the
existing terrain builder normalizes both sources to the demoted Legacy volume.
For episode `9`, final-crest maxima fall from 72 to 29 and mean spacing expands
from 176 to 520 km. The matched images remove the repeated comb, but human
review confirms that the candidate replaces it with a broad smooth mound rather
than a convincing organized range. The corrected operation order is retained
as necessary source representation; the static nearest-front adapter is
rejected as sufficient terrain architecture. The `105` scheduled-source peak
count is secondary fine-support variation, not visual evidence against that
reading: only 29 peaks survive into a terrain that reads smooth.

The selected
[regional deformation program](regional-deformation-support.md) now has a
source-only RDS0a implementation and one RDS0b terrain discriminator. The first
categorical center-owner transfer was rejected as non-surjective on the
adaptive process mesh. A sparse exact spherical control-volume overlap now
closes every active donor and provenance ledger without changing the mesh or
inventing a smoothing width. Under one continuous, fixed demoted-Legacy budget,
the moving four-frame arm materially changes local relief and 56.6% of support
drainage receivers. Matched views replace the control roof with finite linked
lobes, lows and differentiated ends, but the fixed relationship readout shows
that the apparent intermontane organization is a dense closed-depression
grammar: 100 support depressions versus 3, 86 entering highland, and no
represented major-river cell inside any highland component. RDS0 is therefore
rejected as sufficient terrain ownership. Its source representation and exact
transfer remain reusable; no other worlds, generated prior, width, source-rate
or erosion tuning are licensed for this composition.

## Slice B — product-native channel/hillslope dual

Hypothesis: a sparse topology conditioned by tectonic opportunity, base level,
runoff and supported material state can preserve the important mountain
consequences much more cheaply than repeated global landscape evolution.

The minimum slice jointly constructs a drainage/catchment scaffold and its
divide dual, then reconstructs a smooth terrain constrained by base levels,
channel long profiles, broad deformation opportunity and a bounded relief or
solid budget. A short product-native erosion/reconciliation pass makes the
reconstructed surface and routed hydrology agree.

The critical representation split is:

```text
all-cell receiver forest        owns possible drainage and catchment membership
sparse promoted channel graph   owns descending long profiles and runoff hierarchy
non-channel surface             owns continuous hillslopes, interfluves and massifs
catchment adjacency             derives divides and candidate saddles
```

The old fast G arm collapsed the first three rows: every receiver edge added a
strict elevation rise, so graph depth became steps and spikes. B may reuse that
failure as a negative oracle, but not its reconstruction. Likewise A4 may
remain a drainage-feedback control, but its burn-in topology and
trunk/interfluve uplift redistribution are not the new owner.

The physical reference remains the reduced landscape equation already used by
the better scientific precedents: rock uplift competes with drainage-area and
slope-dependent incision while hillslope transport supplies the non-channel
surface. Divide migration is then a consequence of unequal erosion and uplift,
not a separately drawn crest. B makes the drainage scaffold explicit only to
obtain that relationship cheaply. Analytic/procedural stream-power work shows
that channel long profiles can be reconstructed without a long whole-surface
simulation, while drainage-first graphics shows the visual upper bound and the
risk of obtaining the answer by construction
([physically based analytical erosion](https://doi.org/10.1111/cgf.15033),
[Génevaux et al. 2013](https://doi.org/10.1145/2461912.2461996)).

### Selected B0 vertical slice

Use the already fixed seed `8675309`, episode `9` and ordinary process mesh.
Do not introduce another world, history prior or tuning ladder. The candidate:

1. consumes the retained RDS source frames, axial fabric and additive
   provenance through the existing exact coarse-to-process overlap; the fixed
   dyadic history remains a counterfactual, not a generated geological prior;
2. uses the demoted non-orogenic substrate, real ocean/lake base-level policy,
   runoff and eligible material to construct one provisional receiver forest;
   a whole-cell forest is routing scaffolding, not a channel mask;
3. accumulates runoff and promotes only a sparse, physically scaled channel
   subset, keeping the channel-versus-hillslope distinction explicit;
4. integrates monotonically descending channel profiles from their base levels
   using a reduced slope-area/stream-power relation whose amplitude is bound by
   one declared regional relief or work budget;
5. reconstructs non-channel cell means with a smooth conservative hillslope
   response toward those channel constraints and the broad deformation
   envelope; it does not recursively add one rise per graph edge;
6. derives catchments, adjacency divides and candidate saddles from the same
   scaffold, then performs one short reroute/incision/hillslope reconciliation;
   and
7. finalizes once through the ordinary `FineSurface`/product-hydrology path so
   downstream water, rivers, Living Surface and routes read the same terrain.

The implementation should be one research-only runner over the existing
`FineBase`, not a new solver framework. Reuse `RegionalDeformationProgramV0`,
its sparse exact overlap transfer, the product erosion routing/accumulation
scratch and `FineSurface::from_eroded`. Do not import the planar landscape
mesh, its artifact machinery, the exact spherical hierarchy into the iterative
loop, or persistent C1 reach identity. Those remain operator/evidence sources.

Opportunity is not direct height in B0. Its integrated support and fabric may
condition routing resistance, relief capacity or channel orientation only
through a named rule with a neutral control. Per-frame scalar positive lobes
may not be injected and then cosmetically carved. This retains the useful RDS
state without repeating the failed ownership relation.

The first output is one matched Physical/Diagnostic relationship packet, not a
campaign. It must show the provisional channel graph, final product receivers,
catchment/divide ownership, physical depressions and deformation support on the
same views. The existing lightweight RDS relationship readout is sufficient;
do not materialize a second full f64 geometry merely to obtain exact saddles.

Kill B if constructed channel edges are not monotonically descending to a
compatible base level, if rerouting changes catchment/divide ownership
materially, or if the graph invents cross-basin ridge links. Also kill it if the
surface contains tents, swollen amoebas, steps, corduroy or repeated graph
grammar, or if tectonic/material counterfactuals do not change the topology in
the expected direction. A visible valley network is insufficient if it is only
the provisional scaffold burned into an otherwise unrelated surface. B is an
explicit authentic hack even if successful.

### B0 outcome

B0 is implemented behind `research-landscape` and evaluated in the original
fixed seed `8675309`, episode `9`, 100k-coarse/~255k-process world. It uses
product priority-flood routing, promotes 7,938 sparse channel cells from
high-order source-intersecting or fabric-aligned paths, grades every promoted
edge monotonically to a compatible base level, and reconstructs the remaining
surface with a finite-volume steady hillslope solve. Integrated opportunity is
the solve's source term; the removed Legacy relief supplies only a global solid
budget and maximum-relief cap. The local reproducibility packet is
`artifacts/b0-terrain-2026-07-23/`.

The upper bound fails its declared admission test. Final product hydrology
agrees with the provisional scaffold on only **41.59%** of
deformation-support cells and only **58.01%** of promoted-channel cells, with
91,151 receiver mismatches overall. The solved slope-area amplitude hits the
Legacy-derived 4.00 km relief cap after retaining only 2.60% of the available
positive relief budget, while the base-level reconstruction removes 7.08
million km³ from the demoted substrate. It nearly eliminates the RDS0
depression grammar, but source-in-highland falls to 9.68%; the largest >1.5 km
highland is still 0.89 million km² and 579 km mean width, and no highland
component contains a represented major-river cell. Matched Physical and
Authentic views are largely flat with a few narrow isolated remnants rather
than coherent ranges. This is a reroute/relief-collapse failure anticipated by
the kill rule, not a request for parameter tuning.

Retain the product-routing reuse, compact source schedule, finite-volume
reconstruction seam and scaffold-versus-final diagnostic as operator evidence.
Do not promote B0, extend it into a tuning ladder, or infer that a graph-first
terrain owner is now required. Together, RDS0 and B0 show that neither moving
positive support nor drainage-first compatible reconstruction supplies the
missing regional organization from the present source state.

## Shared comparison and decision rule

Use the existing fixed seeds `12345`, `8675309` and `9001`, with 100,000 coarse
cells and the ordinary roughly 250,000-cell fine surface. Do not begin at one
million cells. Higher resolution, adaptive remeshing and conditioned sub-grid
detail are follow-ups only after a candidate establishes regional organization
at the ordinary scale.

Both slices must:

- remove Legacy convergent direct height and repeated convergent uplift inside
  the candidate path rather than stack over them;
- retain the same non-orogenic base, climate/precipitation supply and sea-level
  policy; candidate arms share one declared source budget, while their mapping
  to Legacy's unbudgeted calibrated height response remains explicit rather
  than claiming false budget parity;
- preserve or deliberately replace downstream water, river, living-surface and
  route contracts;
- expose Physical and Diagnostic evidence before Authentic presentation;
- show range-scale organization and a drainage/divide consequence in at least
  two of three worlds without per-seed tuning; and
- allow an honestly continuous source to remain a broad massif or plateau when
  its response and edge/drainage relationships support that outcome.

Prefer A if it meets those outcomes within a plausible cost. Prefer B only if
it preserves the same useful consequences materially more cheaply or reveals a
topological requirement A cannot recover. If both succeed, do not stack them by
default: choose the simpler ownership or assign non-overlapping responsibilities
and require an ablation to justify each. If both fail, revisit the adequacy of
the generated deformation/material state; do not answer automatically with
deeper physics, more cells or independent noise.

## Machinery retired by a successful replacement

A promoted boundary would absorb or remove:

- Legacy arc/collision direct-height ownership;
- repeated hold-and-carve convergent uplift;
- emergent demote/rebuild, structured-uplift and drainage-pulse variants;
- overlapping default-off orogen thickness solvers not retained as named
  controls;
- fine structural generators whose responsibility the new boundary actually
  replaces;
- duplicate routing once evolution and final hydrology share a contract; and
- post-hoc hydrology terrain writing as a separate authority, either by
  absorbing repair into evolution or making the repaired terrain canonical.

Retain spherical geometry, plate/crust setting, boundary kinematics, the current
isostatic conversion contract, climate, water semantics, river hierarchy,
declared repair provenance and explicit cartographic presentation. Persistent
sediment, richer inheritance, glaciers, dynamic isostasy/flexure and ecology
remain later systems with their own payoff gates.
