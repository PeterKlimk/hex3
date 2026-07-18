# Landscape organization strategy

Status: **current architecture decision**, 2026-07-19.

This document decides how terrain, drainage and landform meaning fit together.
It does not promote a terrain replacement. [Structural Mountain V0](structural-mountain.md)
and [Lithosphere Inheritance V0](lithosphere-inheritance.md) have completed their
bounded source decisions; prior H/C/G and landform-packet documents remain
research evidence rather than active policy.

See the [project thesis](thesis.md), [model strategy](model-strategy.md),
[current architecture](architecture.md) and
[mountain-system design basis](research/mountain-system-design-basis-2026-07-13.md).

## Decision

Hex3's weakest established seam is **regional geographic organization**: the
transformation from broad tectonic, climatic and hydrologic causes into coherent
ranges, ridges, divides, lows, valleys, basins and river networks.

Retain this target causal shape:

```text
plate/crust setting + linked deformation forcing
  -> broad elevation/base-level envelope and rock-uplift opportunity
  <-> drainage growth, competition and capture
  <-> channel incision and slope-limited hillslope response
  -> authoritative land surface + retained drainage/landform evidence
  -> consumer-specific geographic objects with provenance
  -> explicit cartographic presentation
```

This is an ownership rule, not a demand for full geodynamics or hydraulics. The
current legacy path remains the usable product control. It is not the presumed
target architecture.

If terrain ownership is reopened, the candidate must replace rather than stack
over the legacy convergent height and repeated uplift owners. It should treat
finite uplift forcing, drainage/divide organization and nonlinear hillslope
response as one coupled ownership problem. A new source-segmentation ladder is
not the current task.

## Precise defect

The default convergent response is a smooth, capped distance-band height field.
Fine structural synthesis is neutral at product defaults. Erosion creates
substantial relief and dissection but mostly carves the inherited tableland.
Hydrology supplies valuable drainage and basin topology, but no stable range,
divide, pass or valley hierarchy is currently shared with consumers.

Broad high surfaces are physically legitimate. The failure is that one
plateau-like grammar is the generic response to convergence and often lacks
finite massifs, range ends, branching divides, saddles and organized valleys.
Optimizing global plateau coverage would therefore target the wrong problem.

The earlier pillar incident remains a separate warning: a physically supported
slope was made visually absurd by roughly hundredfold relief exaggeration. It
does not justify flattening physical terrain or judging morphology from a
cartographic view.

## Current ownership and retained systems

The implemented default path is:

```text
plate/crust regions + Euler motion
  -> classified present boundary edges
  -> scalar arc/collision response
  -> coarse tectonic thickening and elevation
  -> adaptive fine interpolation
  -> repeated legacy uplift + routed incision + hillslope response
  -> drainage-integration repair
  -> final physical surface
```

The following capabilities have earned a role:

| Capability | Disposition | Earned role and limit |
|---|---|---|
| Spherical Voronoi geometry | **Retain** | Shared topology, area and distance foundation; adaptive work must remain area-aware. |
| Plate/crust initialization | **Retain** | Cheap mixed-crust and margin setting; not geological genesis. |
| Euler motion and boundary kinematics | **Retain** | Useful sign, rate, shear and polarity; synthetic rather than predictive plate history. |
| Boundary fronts and history episodes | **Retain/reframe** | Supply real product chains, finite causal grouping and relative work; stop converting them immediately into finished height. |
| Legacy tectonic height and repeated uplift | **Retain as control** | Keeps the product usable and locates broad high terrain; direct-height plus hold-and-carve ownership is under replacement. |
| Routing, incision and hillslope operators | **Retain as operators** | Create real drainage-related relief; their calibrated schedule is not geological time. |
| Hydrologic topology and repair provenance | **Retain** | Central emergence engine and cheap input to later stages; repair remains an explicit terrain writer. |
| Climate, runoff and transported moisture | **Retain/condition** | Cheap reusable conditioning; no deeper atmosphere is required for the mountain slice. |
| Water, river and catchment semantics | **Retain/deepen on demand** | Useful shared identity and hierarchy; importance and scale generalization remain consumer-specific. |
| Physical/Diagnostic/Authentic/Dramatic views | **Retain** | Separate truth, inspection and spectacle; presentation never validates morphology. |
| Research landscape and landform libraries | **Retain quarantined** | Reusable operators and negative evidence; not product state or an active promotion ladder. |

## What prior work established

The mountain campaign produced several durable conclusions:

- ancestry on seed `12345` located the smooth roof in coarse legacy elevation;
  fine interpolation preserves it, erosion dissects it, repeated uplift
  reinforces it and presentation only amplifies it;
- cross-section reshaping, scalar along-strike modulation and equal-RMS
  MassifCorridor structure did not establish a functional hierarchy;
- isotropic or strike-aligned fine grain supplies local texture, with regular
  strike bands risking dune/corduroy morphology;
- conserved redistribution and thin-sheet/lifecycle solvers improved some
  numerical and causal properties but did not earn useful range organization;
- on the planar H/C/G comparison, a uniform forcing ribbon made H preserve a
  roof, C add expensive local texture and G create steps/spikes;
- a finite tapered-parent control removed the exact roof cheaply, but finite
  tips alone still produced smooth elongated massifs; and
- drainage and erosion express supplied organization but do not reliably invent
  first-order range structure from a symmetric ribbon.

These are constraints on any future owner, not reasons to tune old rungs. The old
experiment amendments, artifact schemas and planar promotion campaign are no
longer prerequisites. Their results remain indexed under
[research](research/README.md) and [audits](audits/README.md).

## Current disposition and reopening gate

Structural Mountain V0 compiled one coherent 3,200 km collision parent, then
stopped because its source contained one continuous opportunity maximum and no
defensible internal low. The source-only inheritance follow-up generated cheap,
coherent terrane-assembly sutures, but none of the reviewed parent's contacts
carry that history. Its paleorift arm also lacks physical-scale stability and a
demonstrated rift-nucleation consumer.

The principled result is not to manufacture missing segmentation. Continuity
alone is no longer a rejection criterion for this causally continuous parent.
This relaxes the universal requirement that every mountain system expose
internal hierarchy; it does not promote a new terrain response, endorse the
legacy generic capped ribbon or establish that the tableland defect is rare.

Terrain work therefore pauses. Reopen the broader terrain owner only if a small
fixed ordinary-world Physical/Diagnostic corpus shows that smooth roofs are
recurrent modeled morphology. The [World Readability V0](roadmap.md#1-world-readability-v0)
A/B may establish that the morphology materially harms the cartographic product;
it may not establish physical validity. A reopened comparison must replace
scalar final-height plus repeated-uplift ownership and demonstrate a
drainage/divide consequence. It may not select a convenient seed, tune relief,
add independent crest texture or expand geological history merely to force a
break in one belt.

## Pareto-important capabilities while terrain is paused

- **Scale-aware world readability:** make current product-control relief and
  retained water, river and Living Surface relationships visible before
  inferring that another physical mechanism is missing.
- **Consumer-driven landform objects:** ranges, divides, saddles, valleys and
  crossings remain valuable, but should be built only when terrain evaluation,
  maps or consequence systems require them.
- **Shared geological inheritance:** a few coherent provinces or weak
  structures may condition both deformation and erodibility if existing
  crust/history transitions prove insufficient. Independent noise is not
  inheritance.
- **Persistent sediment:** one mobile/deposited material could connect source,
  basin, floodplain, delta and soil. It is not the first cure for mountain
  roofs.
- **Seasonal climate, cryosphere and richer ecology:** valuable future
  conditioners/consumers, not generic mountain repairs.

## What is deliberately not next

- tuning legacy uplift, erosion gain, relief scale or river width;
- reviving O1/O2/O3, A4 or the H/C/G promotion campaign;
- adding a crest texture, independent geological noise or decorative passes;
- completing the experimental carrier/lifecycle model because it exists;
- adding sediment, explicit landslides, flexure or glaciers to cure the generic
  belt grammar;
- building a universal landform ontology or artifact bureaucracy; or
- optimizing an undecided landscape composition.

The near-term goal is product-facing and bounded: make named existing geographic
relationships more legible without falsifying state, while retaining an
independent Physical/Diagnostic trigger for a genuine terrain-owner rework if
the tableland grammar is endemic rather than exceptional.
