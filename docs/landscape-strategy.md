# Landscape organization strategy

Status: **current architecture decision**, 2026-07-14.

This document decides how terrain, drainage and landform meaning should fit
together after the geographic-coherence pass. It does not promote a new terrain
model. It records the completed comparison, the rejected ownership shortcuts
and which parts of the current system have earned a role.

See the [project thesis](thesis.md), [model strategy](model-strategy.md),
[current architecture](architecture.md), [mountain-system design basis](research/mountain-system-design-basis-2026-07-13.md)
and [experiment registry](experiment-registry.md).

## Decision

Hex3's weakest architectural seam is **regional geographic organization**:
the transformation from broad tectonic, climatic and hydrologic causes into
coherent ranges, ridges, divides, passes, valleys, basins and river networks.

Retain this target causal shape:

```text
plate/crust setting + linked deformation forcing
  -> broad elevation/base-level envelope and rock-uplift forcing
  <-> drainage growth, competition and capture
  <-> channel incision and slope-limited hillslope response
  -> authoritative land surface + retained drainage/landform structure
  -> scale-aware geographic objects with provenance
  -> explicit cartographic presentation
```

This is a statement about ownership and preserved consequences, not a demand
for a full geodynamic or hydraulic simulation. The completed architecture
comparison allowed a reduced coupled model and cheaper graph-first controls to
compete on common geographic outcomes. None earned product ownership: current
G and manufactured I impose the wrong structure, while C's real process effects
do not justify their cost under the compact F forcing. The next mountain owner
must therefore begin with better causal forcing organization from product
fronts, history and material state; it need not be a deeper simulation.

The unchanged current legacy path remains the product reference. A separately
defined idealized hold-and-carve adapter is the bounded testbed control; it must
not be mistaken for a physical-time reproduction of the product's 200-step
calibrated chronology. Neither is the presumed target architecture. Further
tuning of scalar mountain profiles, additive fine texture, relief gain or local
path scores is not the next project-level task.

## Why this is the missing seam

The project philosophy is working at both ends:

- spherical Voronoi geometry, units and area-aware topology provide a credible
  numerical substrate;
- plate ownership, Euler kinematics and boundary relationships provide useful
  broad causal setting;
- climate and hydrology are comparatively cheap, reusable inputs;
- physical, diagnostic and exaggerated presentation are now separated; and
- Authentic and Dramatic relief communicate the same world successfully.

The middle is weaker. The default convergent-terrain response is a smooth,
capped distance-band height field. Fine structural synthesis is neutral at
product defaults. Erosion creates substantial relief and dissection but mostly
carves the inherited tableland. Hydrology supplies drainage and basin topology,
yet current river selection does not provide stable, agreed importance or a
general landform hierarchy.

The resulting failure is not simply "too many plateaus." Broad high surfaces
are physically legitimate. The failure is that one plateau-like grammar is the
generic response to convergence and often lacks an internal hierarchy of range
ends, massifs, saddles, passes, branching divides and valleys.

Scientific landscape models retain the incision–hillslope feedback: forcing
creates potential relief, routed water concentrates incision, catchments
compete and hillslope transport limits and redistributes slopes. Production
terrain systems often preserve the resulting drainage/ridge hierarchy by
construction rather than simulating that loop. Full hydraulics, stratigraphy
and continuum geodynamics are not prerequisites for either reduction.

## What has earned a place

| System or capability | Disposition | Earned role and limit |
|---|---|---|
| Spherical Voronoi/adaptive geometry | **Retain** | Shared topology, area and distance foundation. Adaptive allocation remains subject to a simpler-mesh value comparison. |
| Plate/crust initialization | **Retain** | Cheap authentic initializer with mixed crust and passive-margin variety; not geological genesis. |
| Euler motion and boundary kinematics | **Retain** | Physically meaningful broad setting and polarity; synthetic forcing, not predictive plate history. |
| Product boundary feature bridge | **Retain/reframe** | Preserve sign, rate, polarity, present crust setting and connected fronts; stop treating its scalar response as finished mountain terrain. |
| Legacy tectonic height and repeated uplift | **Retain as control** | Efficiently locates broad high terrain and keeps the product usable. Its direct-height plus hold-and-carve ownership is the architecture under test. |
| Active routing, incision and hillslope operators | **Retain as operators** | Demonstrably create material relief and dissection. Their calibrated chronology and present composition are not yet a promoted landscape-evolution model. |
| Hydrologic topology, basins and area-aware accumulation | **Retain** | High-leverage physical topology. Equilibrium lake storage and drainage integration remain disclosed authentic hacks. |
| Cheap climate and transported moisture | **Retain/characterize** | Valuable conditioner for runoff, erosion, future ecology and spectacle. Do not deepen it before isolating its conditional geographic benefit. |
| Water and river semantic foundations | **Retain/deepen** | Shared water identity, catchment policy, reaches and Strahler order are useful. Importance, persistence and scale generalization remain incomplete. |
| Cartographic relief, river width and diagnostic views | **Retain** | Successful explicit presentation. Visual exaggeration does not own or validate terrain morphology. |
| C0/C1/continuous-crossing research operators | **Retain as research library** | Establish representation and ownership constraints. They are not a product drainage or landscape stack. |

## What should be reframed, simplified or quarantined

### Scalar mountain construction

Replace the assumption that tectonic work becomes a final height footprint.
The broad envelope may survive, but linked deformation should preferably expose
rock-uplift/loading forcing which another owner turns into surface form. A later
rung may add inherited material structure if the base comparison identifies the
consequence it needs to own.

O1 cross-section shaping, O2 scalar along-strike modulation and O3A prescribed
fine structure did not create a convincing functional hierarchy. This is
evidence against another amplitude/profile sweep, not against every authentic
range generator.

### One elevation field owning incompatible meanings

A finite-volume mean land surface is not simultaneously a resolved channel bed,
a tectonic work ledger, a presentation mesh and a semantic ridge/valley graph.
Keep those meanings related but explicit. Additional state is justified only
where a consumer needs the distinction.

### Experimental superstructure

The tectonic-model ladder, inactive fine-shape branches and erosion alternatives
contain useful tested operators but exceed the accepted product architecture.
Move toward a small product configuration plus research-only compositions.
Default-zero state and CLI parameters still impose maintenance and reasoning
cost.

### River-role conflation

Separate these owners:

```text
continuum topology and water supply
  -> physical channel-support/initiation proposal
  -> optional persistent reach state and lineage
  -> semantic hierarchy and importance
  -> scale-dependent cartographic selection
```

The R1 evidence rejects the registered P0/M0 local maximum-face constructions
for the stated seeded-centreline role. It validates continuous polygon traversal
on its manufactured domain while leaving the broader graph bundle formally
unproven, and reaches a physically irrelevant registered arithmetic floor. Take
the recorded Pareto stop; do not promote QR, weaken the gate or continue that
ladder without a concrete architecture dependency.

### Lake, ocean and repair ownership

Core basin topology stays. The climate source mask, connected ocean identity,
lake equilibrium, outlet/spill provenance and drainage-integration cuts need
one shared semantic account. A climate-ratio control that changes lakes without
recomputing rivers, rainfall or erosion must remain an explicit storage hack,
not be described as whole-world wetness.

## Pareto-important missing capabilities

### 1. Landform objects and generalization

Ranges, plateaus, ridge/divide graphs, passes, valleys, catchments, coast/island
hierarchy and object ancestry are the cheapest high-leverage missing layer.
They are required to judge any terrain replacement and later support
cartography, ecology, resources, settlement and explanation.

This is not a request for a universal ontology. Build only the objects needed
by current comparisons, retain provenance, and allow definitions to be
scale- and purpose-dependent.

### 2. One regional-organization owner

The product needs either:

- a reduced evolving uplift–drainage–hillslope system;
- a tectonically conditioned drainage/ridge skeleton with compatible terrain
  reconstruction; or
- a hybrid in which the reduced physical loop establishes topology and an
  authentic reconstruction makes it legible at finite resolution.

These are competing architecture families. Do not stack all three.

### 3. Shared geological inheritance

A small number of persistent provinces, weak zones or linked deformation
episodes could condition both forcing and erosion. This is more causally useful
than independent noise fields and can later support soils and resources. It is
an optional second rung, not required to prove basic drainage organization.

### 4. Persistent sediment, conditionally

One conserved mobile sediment/cover quantity could connect mountains, basins,
plains, deltas, coasts and soils. It is not the first repair for flat mountain
tops. Add it only after the organization comparison identifies a need for
transient cover/deposition and a stable channel/support owner.

### 5. Living and human layers, after the seam

Existing climate, water and provisional ecological potentials make biomes and
vegetation a strong near-future spectacle slice. Coast hierarchy, resources,
settlement and routes also have high board-world value. They should not be
blocked on maximal landscape physics, but they should consume coherent
geographic objects rather than compensate for their absence.

## Next architectural experiment: organization-owner slice

Prepare one bounded **organization-owner slice**, not another global seed sweep
and not another isolated numerical rung. The
[comparison design envelope](research/orogen-organization-owner-v0-2026-07-16.md)
now selects the three families and common evaluation philosophy. It is not yet
an executable wire/verdict contract: the exact
[artifact/provenance amendment](research/orogen-organization-artifact-v0-2026-07-16.md)
and exact
[numerical/admission amendment](research/orogen-organization-numerical-v0-2026-07-16.md)
and exact
[evidence/projection amendment](research/orogen-organization-evidence-v0-2026-07-16.md)
and exact
[planar capture/human-review amendment](research/orogen-organization-planar-review-v0-2026-07-16.md)
are now preregistered. A deliberately thin 4 km H/C/G source implementation and
common numerical/visual discriminator now exist. They are not promotion-grade
arm packets and no arm is promoted.

### Question

Which minimum owner produces causally placed and reusable range/drainage
organization at a justified cost: reduced coevolution or graph-first
reconstruction?

### First discriminator result

The first result is negative for the shared input representation before it is a
choice between owners. Each linked segment contains about `196 km` of exactly
constant along-strike forcing; both segments share one synchronous episode and
equal work. Their declared links and vergence have no compiler mechanics,
horizontal velocity is zero, and material and runoff are homogeneous.

Sharp common-crop views and matched profiles show long physical roofs in both H
and C. H maintains them through its pointwise hold. C adds local drainage
texture but has no regional inherited or moving state with which to break the
roof. G turns the same opportunity into discontinuous steps and spikes. The
current G reconstruction therefore stops; H remains the null/control; and the
present C instance does not justify its roughly two-times-H runtime. This does
not reject the reduced coupled family under a better causal forcing owner.

The compiler-only `B/F/I` comparison shows that the first useful upstream change
is smaller than a persistent product organization graph. A full-cosine
finite-parent control removes the exact longitudinal roof while conserving the
accepted work and width. The manufactured inheritance graph also conserves
work, but breaks the belt into disconnected hot spots and does not earn its
complexity.

### Second discriminator result

The one-shot F response confirms that finite support fixes only the exact roof.
H and C both receive the same compact F opportunity; F closes `100,625 km³`
near roundoff and matches the compiler probe to `3.11e-15 km`. C coevolution
produces real process consequences--`25,345 km³` of denudation and longer portal
trunks--but the sharp physical view remains two smooth elongated massifs. Its
larger whole-domain critical-point count is not massif-local evidence because
the entire domain is positive land. C does not supply coherent range-scale valleys,
divides or passes that justify `746.10 s` versus H's `336.43 s`.

H's repeated target restoration (`130,385 km³` gross) makes this a mechanism
discriminator rather than a fair work-matched owner tournament. The result does
not reject coupled landscape response in general. It does reject the idea that
more terrain coupling should precede better causal forcing organization here.
Current G and manufactured I stop; H remains a control; C remains research
machinery rather than selected product behavior. On return to mountains, derive
the next forcing owner from real front chains, tectonic history and inherited
material state rather than tuning these toy fields. See the updated
[mountain-system design basis](research/mountain-system-design-basis-2026-07-13.md).

### Shared setting

Use the existing bounded landscape-testbed geometry and one linked, tapered
deformation pair with a termination/transfer low. Every competitive arm receives
the same admissible geometry, deformation field, outlet/base-level geometry,
runoff field, material mask, characteristic spatial scales and presentation
settings.

The non-presentation subset is now frozen by the accepted
[linked shared-input manifest V0](research/orogen-linked-shared-input-v0-2026-07-15.md):
exact 8/4/2 km meshes and portals, declarative and compiled forcing, analytic
and evaluated work ledgers, a coordinate-defined raw initial surface, uniform
runoff, homogeneous base-substrate membership and two candidate evaluation
masks; see the [dated audit](audits/orogen-linked-shared-input-2026-07-15.md).
The design envelope now selects work-matched opportunity, arm-specific
conversion/chronology, whole-domain common extraction with a separately bound
central projection and a shared safety ceiling. Exact schemas, reductions,
ledger gates and presentation details are divided among four executable
amendments. Artifact/provenance, numerical/admission, evidence/projection and
planar human-review contracts are complete. One non-authoritative 4 km
engineering probe may now de-risk exact H/C/G composition without selecting an
arm or producing campaign artifacts. Promotion-grade implementation and the
frozen base campaign still require their registered artifact, replay, renderer
and review gates.

Here segment terminations and transfer-zone forcing opportunity are prescribed
testbed geometry controls. They do not guarantee that any arm produces a
terrain transfer low or saddle. The current common packet does not recognize or
emit a named range end or transfer-low object.

The design matches positive input-work volume at the reference resolution but
does not pretend that direct height, rock-uplift rate and graph conditioning are
one forcing quantity or that their spatial opportunity envelopes are equal.
Each arm keeps its own units and reports its conversion and spatial moments.

C owns physical duration. H owns a declared calibrated step horizon. G owns a
construction/reconstruction pass and makes no chronology claim. All arms share
a maximum runtime and memory ceiling and report actual cost; cheap arms are not
required to spend the budget.

The linked base case comes first. The design bounds honest response protocols:
H repeats its registered hold/carve operator, C evolves through changed input,
and G performs a disclosed frozen-amplitude reconstruction. Exact wet/dry and
forcing-reorganization input identities still must be committed before those
cases. This tests each architecture's response rather than granting dynamics
only to C.
Every competitive arm meeting the shared base-case gate advances under the same
rule; H remains the standing control in every admitted case. Record a dropped
arm's base failure explicitly rather than silently excluding it from comparison.

### Arms

An unchanged product-reference observation is retained outside the competitive
testbed. The three comparable testbed families are:

1. **H — idealized hold-and-carve control:** a locked smooth height/uplift
   envelope plus a frozen adapter of the current reduced routing/incision/
   hillslope behavior. Its chronology is calibrated rather than physical.
2. **C — reduced coupled owner:** time-varying rock uplift, conservatively routed
   runoff recomputed as elevation evolves, stream-power/effective denudation and
   linear hillslope transport as the minimal first control. Admit nonlinear or
   threshold-limited hillslopes only if steep-slope behavior requires them. No
   sediment, flexure, glaciers or explicit fault mesh.
3. **G — graph-first authentic control:** generate a hierarchical drainage
   network conditioned by the shared deformation field and outlets, derive
   compatible valleys/divides/ridges, and reconstruct a surface. Its authored
   topology is disclosed rather than treated as emergent physics.

An explicit ridge/valley skeleton is not a fourth product candidate. If useful,
score it separately as a noncompetitive quality upper bound; do not hide it
inside G.

### Required outputs

Run the same independent semantic extractor over every final authoritative
surface. G cannot score its authored construction graph as though it were an
independently recovered result; retain that graph only as provenance and test
its correspondence with the extracted objects. Freeze G without access to the
target object answers and charge its authored priors and parameters as
architecture cost.

The shared vocabulary, neutrality rules and implementation order are now
preregistered in the [landform object packet v0](research/landform-object-packet-v0-2026-07-14.md).
That document is an umbrella evidence contract, not one monolithic extractor.
G0/S0, D0 and O0a are now implemented and evaluated within their bounded
scopes. They establish common physical geometry, surface hierarchy, planar
drainage and mechanical boundary/descent/cross-section evidence—not a terrain
arm or named natural-kind landforms. The bounded planar
[O0b correspondence and assembly contract](research/landform-object-packet-o0b-2026-07-15.md)
is now accepted as the bounded common planar evaluation-instrument checkpoint
for exact packet assembly and mechanical highland/drainage-node best maps; see
the [dated audit](audits/landform-o0b-correspondence-2026-07-15.md). Amendment A
passes the isolated-four-cone 4→8/2, equal-elder, frozen-remapping and
whole-artifact reversal gates. Its flat-routing apron is compatibility-only,
not realistic drainage morphology. The failed linked-four-cone 2 km witness is
retained as historical S0/D0 representation evidence rather than rewritten or
treated as a passing packet. This acceptance does not extend to
product/spherical O0b, persistent identity/events, a product O0a adapter,
packet/product R0 or any H/C/G composition or promotion.

The [product-boundary decision](research/landform-product-boundary-decision-2026-07-15.md)
now retires that provisional combined packet/product R0 formulation. Product
G0/S0 remains external noncompetitive context; product-native hydrology is a
different derivation from common planar D0 and will not be cast into the same
packet. The next organization prerequisite is the executable
[common-core split](research/landform-common-core-v0-2026-07-15.md), proved
against accepted manufactured packets. Inventory corrected an important
category error: the linked scenario defines forcing and shared inputs, not a
pre-arm final terrain. Its shared-input manifest must remain separate from
the final-surface evidence produced by H/C/G. The thin source implementation
now exercises that separation, but does not establish product-to-testbed
correspondence or authorize promotion infrastructure.

The current common packet supplies operational reference highlands,
peak/saddle topology, drainage nodes/catchments/reaches, raw boundary evidence
and reach probes. The eventual organization experiment must additionally answer
the following questions, but these are desired comparison semantics rather
than fields already emitted by O0b:

- range extent, massifs, range ends and transfer lows;
- ridge/divide graphs, passes and saddles;
- drainage basins, trunks, hierarchy, confluences and outlet relationships;
- retained network/landform provenance where an arm constructs persistent
  structure rather than deriving it from the final surface;
- correspondence between forcing, divides, longitudinal/transverse drainage
  and retained relief;
- state changes under forcing reorganization and wet/dry conditioning;
- water and solid ledgers where claimed;
- runtime, memory and resolution/timestep response; and
- matched Physical and Diagnostic views, followed by one fixed Cartographic
  view for human judgment.

Named ranges, ridges, divides, passes, transfer lows and valleys require a
separate future semantic contract. They cannot be inferred from O0a raw faces
or O0b best-component cardinality.

Earth distributions are priors, not the target function. The decisive evidence
is object topology, causal response, readable morphology and compute/complexity
cost—not peak height or one relief statistic.

### Decision rule

- Choose from the non-dominated arms by object quality, causal response,
  downstream leverage, runtime, memory and architecture cost; physical depth
  is not a tie-breaker by itself.
- Prefer C over a cheaper G only when its dynamics and counterfactuals provide
  demonstrated object-level or downstream value commensurate with their
  incremental cost.
- Prefer G or a C/G hybrid when it preserves the required causes and downstream
  semantics more cheaply or reliably; it is not merely C's fallback.
- Retain H only if neither alternative produces enough object-level benefit to
  justify replacement; that result would redirect effort to semantics and
  presentation rather than another physical solver.
- Add geological provinces or sediment only when a failed case identifies the
  missing consequence they own.

No arm is promoted from a single attractive image. No global product rewrite
begins until an arm passes the linked, reorganization and wet/dry cases under
its preregistered response protocol and is visually preferred under declared
presentation.

## What is deliberately not next

- completing the `1e-12 km` affine arithmetic identity;
- tuning legacy uplift, erosion gain, relief scale or river visibility;
- promoting RT0, C1 lineage or a stable solve without a product consumer;
- adding full sediment, dynamic vegetation, glaciers or civilization;
- force-derived plate dynamics, mantle convection or a 3-D atmosphere;
- a physically based material-renderer rewrite; or
- performance optimization of an undecided landscape composition.

These remain available when a chosen architecture exposes a specific need.
The near-term goal is to decide who owns organized geography.
