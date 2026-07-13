# Drainage-network ownership decision

**Date:** 2026-07-13  
**Status:** testbed architecture selected; bounded discriminator preregistered  
**Predecessor:** [routed C1 audit](../audits/c1-routed-fixture-2026-07-13.md)

## Decision

Do not choose continuum extraction, a channel-initiation threshold or an
explicit skeleton as one universal river owner. They answer different
questions.

```text
conservative surface flow
  -> physically dimensioned initiation evidence
  -> sparse active/dormant reach graph with lineage
  -> C1 bed/channel state and downstream process carriers
  -> semantic importance and cartographic selection
```

- **Continuum face flow owns instantaneous water supply and drainage
  possibility.** It retains sheet flow, split flow and exact water ledgers. A
  skeleton may preferentially route an unresolved channel fraction, but it may
  not invent water or silently burn its bed into authoritative cell-mean
  elevation.
- **A reduced physical initiation rule proposes channel support.** Discharge,
  slope and a declared resistance prior are stronger evidence than cell count
  or a visibility threshold. With no storm, infiltration, grain-size,
  roughness or vegetation state, this remains an authentic physical closure,
  not a universal channel-head law.
- **A sparse persistent graph owns reach lineage and subgrid channel state.**
  It carries physical intervals, width/support, bed state, C1 compartments and
  later sediment/ecology attachments between flow rebuilds and remeshes.
- **Identity is provenance, not physical immortality.** Dominant physical
  overlap can preserve a `ReachId`; initiation, abandonment, split, merge,
  capture, cutoff and avulsion are explicit events. Receiver changes need not
  destroy an otherwise continuous tributary, while newly formed or destroyed
  geometry must not be hidden behind an old ID.
- **Semantic and presentation networks remain derived.** Catchment importance,
  hierarchy, named rivers, displayed trunks and screen width are not the same
  as active geomorphic channel support.

In short: **the continuum supplies, the physical gate promotes, and the sparse
graph remembers**. This is a testbed direction, not promotion into the current
product path.

## Why none of the three candidates is sufficient alone

| Candidate alone | What it gets right | Decisive deficiency |
|---|---|---|
| Re-extract from continuum flow | Cheap, terrain-responsive topology; divide migration and capture can emerge | Receiver ties and remeshing churn identity; no safe owner for C1 bed, width, sediment or history |
| Explicit persistent skeleton | Stable process carrier, exact topology and strong graphics/semantic interface | Prescribing the graph obtains the desired answer by construction and can contradict terrain/runoff |
| Physical initiation threshold | Gives channel presence a causal reduced prior in physical units | A snapshot mask has no memory, lineage or conservative state correspondence; one threshold also collapses several absent resistance processes |

The selected layering also prevents four existing quantities from being
conflated: finite-volume face width, C1 active-channel width, represented
swath width and renderer stroke width.

## Existing Hex3 seam

Hex3 already has most pieces on either side of the missing bridge:

- production `Hydrology` owns a per-stage single-receiver cell graph and
  area/precipitation-weighted accumulation;
- `RiverNetwork` thresholds that graph, builds upstream adjacency and Strahler
  order, and collapses cells into reaches, but assigns IDs from rebuild order
  and does not identify a receiver reach at confluences;
- experimental `FaceFlowCache` owns conservative physical multi-face flux and
  water closure, but only as a mesh-indexed derived cache;
- the routed C1 fixture owns stable reach IDs, physical intervals, a validated
  DAG, conservative overlap remapping and atomic receiver changes, but its
  network, width and grade are prescribed;
- persistent outlet portals and forcing episodes already demonstrate the
  useful pattern: a mesh-independent physical object is compiled into
  mesh-specific support.

The missing system is therefore narrower than a new hydrology solver. It is a
channel **promotion, correspondence and lineage layer** between recomputed
physical flow and persistent C1 state.

## Physical basis and limits

DEM/landscape-evolution models commonly rebuild receiver graphs from current
topography. D8 and related methods are efficient, but extracting a drainage
density is partly a convention rather than a channel-creation law
([O'Callaghan and Mark 1984](https://doi.org/10.1016/S0734-189X(84)80011-0),
[Tarboton et al. 1991](https://doi.org/10.1002/hyp.3360050107),
[Braun and Willett 2013](https://doi.org/10.1016/j.geomorph.2012.10.008)).
Morphometric extraction is valuable as an initializer or audit, but channel-head
classification and slope-area parameters can be grid-sensitive
([Passalacqua et al. 2010](https://doi.org/10.1029/2009JF001254),
[Clubb et al. 2014](https://doi.org/10.1002/2013WR015167)).

Field and reduced-model evidence supports discharge/area-slope initiation, but
also shows that landsliding, seepage, saturation overland flow, substrate and
storm statistics produce different channel heads
([Montgomery and Dietrich 1988](https://doi.org/10.1038/336232a0),
[Dietrich et al. 1993](https://doi.org/10.1086/648220),
[Istanbulluoglu et al. 2002](https://doi.org/10.1029/2001WR000782)). A future
Hex3 initiation score should therefore be dimensioned and spatially conditioned,
with uncertainty disclosed. Catchment area alone remains a cheap baseline.

Willgoose et al. explicitly distinguish hillslope and channelized states and
give channelization memory, although their switching law is phenomenological
and permanent channelization risks fossil channels
([1991](https://doi.org/10.1029/91WR00935)). This supports active/dormant state
with separate initiation and abandonment behavior, not an irreversible binary
flag.

Explicit river-network models demonstrate why a graph is valuable after a
network exists: it can carry bed profiles, width, discharge, sediment and
provenance through confluences
([GRLP](https://doi.org/10.5194/esurf-7-17-2019),
[D-CASCADE](https://doi.org/10.1029/2021WR030784)). Subgrid channels coupled to
a 2D floodplain are also an established reduced representation
([Neal et al. 2012](https://doi.org/10.1029/2012WR012514)). These precedents do
not solve global channel genesis or persistent identity for Hex3.

Real and modeled drainage reorganizes. Divide migration and capture can cause
discrete area transfers, while purely vertical-incision models can make
networks too static
([Willett et al. 2014](https://doi.org/10.1126/science.1248765),
[Whipple et al. 2017](https://doi.org/10.1002/2016JF003973),
[Kwang et al. 2021](https://doi.org/10.1073/pnas.2015770118)). This is why the
contract preserves lineage where justified and records topology events where
it is not.

Graphics provides useful controls, not physical authority. Drainage-first
terrain synthesis shows the visual and semantic upper bound of a graph-owned
world ([Génevaux et al. 2013](https://doi.org/10.1145/2461912.2461996)).
Interactive centerline models can evolve bends and perform procedural cutoffs
or avulsions, but whole-network junction evolution remains difficult
([Paris et al. 2023](https://doi.org/10.1145/3618350)). They support a later
lowland planform layer, not global mountain-channel genesis.

## Preregistered memory discriminator

### Question

Does persistent, hysteretic reach ownership materially outperform snapshot
channel extraction when the physical flow evidence is unchanged except for
numerical jitter and one genuine capture?

This test decides whether the memory/lineage layer earns its cost. It does not
select a universal initiation law, infer channel width, evolve sediment, tune a
landscape or enter the product path.

### Common manufactured world

Use the rectangular landscape fixture at nominal 8/4/2 km spacing with fixed
physical extent, source supply and outlets. Prescribe a smooth, slowly moving
divide/valley surface and route the same conservative continuum flow in three
phases:

1. **stable:** two tributary systems, one confluence and two outlets;
2. **jitter:** bounded sub-cell surface perturbations make near-tied dominant
   receiver faces alternate without crossing the registered initiation or
   abandonment margins;
3. **capture:** a larger monotone divide displacement creates one sustained
   transfer from the first outlet system to the second.

Use one frozen dimensionless initiation evidence function

```text
I = (q / q_ref)^m (S / S_ref)^n / R
```

where `q`, `q_ref`, physical distance and slope retain declared units; `R` is a
declared dimensionless resistance field. Freeze all values before the first
comparison. This is a controlled discharge-slope proxy, not a claim that the
chosen exponents or resistance are a product law. Prescribe one physical width
per manufactured reach so width evolution is not smuggled into this test.

### Arms

- **S0 — snapshot:** promote wherever `I >= I_on`, select a deterministic
  dominant-flux thalweg and rebuild reaches/IDs from scratch each phase.
- **S1 — persistent:** use the same candidates, `I_on` for initiation and a
  lower frozen `I_off` for dormancy/abandonment. Match by physical overlap and
  connectivity, preserve lineage only where correspondence is dominant, and
  emit explicit topology events.
- **S2 — prescribed skeleton control:** use the known analytic channel graph
  and event as an upper bound on topology/identity. It is not eligible for
  promotion as the physical owner.

All arms consume the same continuum water result. The graph is initially a
passive observer/process carrier: it may not alter flow or cell-mean terrain.

### Registered gates

1. Continuum source, face, sink and outlet water ledgers are bit-identical
   between S0 and S1 and close at every spacing and phase.
2. Physical active-channel length and prescribed-width support converge across
   8/4/2 km; report error against S2 rather than requiring accidental exact
   cell-path equality.
3. In the jitter phase, S1 emits no topology event, retains stable IDs and
   preserves attached C1 channel/interfluve moments. Measure S0 ID/path churn;
   do not require it to fail by construction.
4. In the capture phase, S1 emits exactly one capture transaction. Unchanged
   dominant-overlap reach geometry retains lineage; born, retired or materially
   replaced intervals receive explicit parent/child provenance.
5. Receiver replacement remains acyclic and transactional. A failed match or
   proposed cycle leaves graph and attached state unchanged.
6. Overlap remapping plus explicit birth/retirement ledgers close channel,
   interfluve and total elevation-volume moments. State may not leak across a
   divide merely because two paths are spatially close.
7. Repeated extraction, matching and event generation are bit deterministic.
8. Work is `O(N_cell + N_candidate + N_reach)` apart from deterministic sorting;
   report counts and owned passes. No global elliptic/filter solve is allowed.

### Interpretation frozen in advance

- If S0 is already stable and conservatively attachable under jitter,
  persistent lineage is not justified yet; reuse snapshot extraction and defer
  the object layer.
- If S1 suppresses numerical churn and handles the real capture without hiding
  it, promote the lineage layer inside the isolated C1 testbed.
- If S1 suppresses both jitter and the real capture, reject the hysteresis/event
  policy as over-persistent.
- If both S0 and S1 change materially with resolution, the failure belongs to
  routing/initiation support, not identity matching.
- Passing does not promote the initiation equation, prescribed width, C1
  landscape response or product integration.

## Deferred questions

- the product initiation/resistance closure and its climate, material, soil or
  vegetation conditioning;
- width/depth geometry and bankfull discharge;
- channel feedback into continuum routing without double-counting terrain;
- basin, water-body and reach identity correspondence across the global
  coarse/fine product pipeline;
- lateral migration, floodplains, avulsion, deltas and sediment provenance;
- semantic hierarchy, naming and scale-dependent cartographic selection.

