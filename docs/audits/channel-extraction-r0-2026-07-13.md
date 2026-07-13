# Seeded channel-extraction R0 audit

**Date:** 2026-07-13
**Decision checkpoint:** `451a3ef`
**Status:** invalidated during first implementation; no arm selected

## Outcome

The first implementation attempt exposed defects in the
[preregistered R0 discriminator](../research/channel-extraction-r0-2026-07-13.md)
before a valid comparison was completed. No extractor is selected or rejected.
The draft implementation is not retained: its most prominent failures measure
the fixture and geometry encoding rather than the candidate algorithms.

This is a useful experimental stop, not a reason to tune the valley
coefficients or move downstream to initiation, lineage, C1 coupling, sediment
or ecology.

## Why R0 is invalid

### The V length gate is impossible for the registered encoding

R0 requires path length to be the sum of cell-centre neighbour distances plus
the terminal face distance, then demands less than 5% error. The registered
vertical V thalweg lies halfway between two available directions on the regular
hex graph. Any centre-to-centre path must alternate directions 30 degrees from
the target, so its asymptotic length ratio is

```text
1 / cos(30°) = 1.1547...
```

The first run produced about 15.34% finest-grid error, matching this geometric
lower bound. Refinement can reduce lateral Hausdorff distance while never
passing the length gate. The observed rejection therefore diagnoses the
centre-to-centre encoding, not P0, M0 or M1.

### The regular fixture aliases P0 and M0

`MfdSlope` ranks an outgoing face by `face_width × physical_grade`. Every
internal face of the uniform planar fixture has the same width, so local
dominant-MFD M0 and physical-gradient P0 make the same internal choice. Their
V/Y paths were identical by construction of the substrate. R0 cannot answer
its main P0-versus-M0 question on that mesh.

This does not generalize to Hex3's product geometry. The application uses an
irregular spherical S2 Voronoi tessellation with unequal shared-edge widths and
neighbour directions, where physical grade and finite-volume flux are genuinely
different hypotheses.

### The Y surface is not the registered smooth analytic Y

The draft surface took the minimum of three independently clamped segment
troughs. Near the junction, the clamped trunk endpoint undercuts the branch
surface for roughly the first 12.5 km, producing an isotropic junction bowl and
derivative seams. The resulting junction displacement cannot rank centreline
extractors against a claimed smooth Y reference.

### The draft harness did not implement the full contract

The review also found that:

- the manufactured diamond used non-normalized, non-conservative face
  fractions and was not a realizable `FaceFlowCache`;
- M1 reported a downstream total-cost margin under a field named like a local
  face-dominance margin, hiding the distinction R0 intended to audit;
- P0/M0 built a whole-domain boundary lookup despite claiming path-local work,
  while the single-head M1 API recompiled its full-domain pass per head;
- several orientation, refinement, outlet, immutability and ledger gates were
  printed or implied rather than independently asserted; and
- the approximate Hausdorff sampler did not justify three-decimal claims near
  a 3 km gate.

These are implementation defects, but repairing them would not rescue the two
load-bearing fixture defects above.

## Indicative observations only

The discarded run did show that the three algorithms can be implemented as
deterministic, merge-only paths over the existing routing DAG. V lateral error
generally decreased under refinement, and the broad flat corridor produced
many aligned exact ties, correctly warning that a deterministic path need not
be a physically identified thalweg. M1 performed a whole-domain pass and did
not visibly cure the malformed Y case.

None of those observations is promotion or rejection evidence. In particular,
M1 remains an unearned extra mechanism, not a falsified one.

## Disposition

- **No arm selected or rejected.** The comparison is inconclusive because the
  fixture aliases P0/M0 and its length gate is unattainable.
- **No draft extractor retained.** A public experimental API should not be
  checkpointed with misleading margin semantics, pass ownership or invalid
  evidence tests.
- **R0 remains provenance.** Its failure explains why the next experiment must
  separate routing ownership, path geometry and mesh substrate.
- **Downstream channel mechanics remain blocked.** Prescribed C1 and lineage
  fixtures are still mechanism evidence, not terrain-derived product systems.

## Required R1 correction

Preregister the next discriminator before implementing it:

1. Use a deterministic irregular S2 Voronoi cap derived from the product
   tessellation path, locally projected only where the planar finite-volume API
   requires it. Preserve reciprocal adjacency, actual shared-edge widths,
   physical areas and a short stable outlet cut.
2. Construct analytic V/Y surfaces that are smooth at their junctions, and
   validate the surface itself before scoring any route.
3. Compare P0 and M0 on real unequal face geometry. Keep any cumulative M1 arm
   behind an explicit Pareto burden; do not let a manufactured path alone earn
   its full-domain pass.
4. Score routing/topology separately from geometry encoding. Compare the raw
   cell graph with a face-crossing or within-cell polyline; do not require an
   impossible graph length to choose the water-to-centreline owner.
5. Use only normalized, conservative routed caches. Make target portal,
   compiler ownership, local versus cumulative margins and multi-head reuse
   explicit in the API and cost audit.
6. Use exact or conservatively bounded geometry metrics and executable frozen
   scorecards. Tests should fail on invariant violations and report promotion
   status without making a known negative outcome the behavior to preserve.

This R1 is the next bounded task. It is still a seeded, state-free extraction
test, not product integration or a channel-initiation model.
