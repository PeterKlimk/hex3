# Channel-ownership memory M0 audit

**Date:** 2026-07-13  
**Decision checkpoint:** `193d954`  
**Status:** preliminary mechanism gate passed; preregistered discriminator remains unrun

## Scope correction

The [ownership decision](../research/drainage-network-ownership-2026-07-13.md)
preregistered a snapshot-versus-persistent comparison on continuum flow over a
moving surface. Implementing that literally would first require choosing an
unvalidated rule that turns a multi-face flux DAG into thalwegs, confluences and
physical reach intervals. Any result would then confound two questions:

1. does the extractor correspond to the physical surface and converge; and
2. does persistent lineage suppress numerical identity churn without hiding
   real reorganization?

M0 does not yet answer either comparison empirically. It consumes prescribed,
physically dimensioned reach observations containing discharge, grade,
resistance, support, width and receiver candidates and checks that the proposed
mechanism is internally coherent. Its API accepts no continuum-flow or terrain
object, so non-coupling is true by construction rather than a tested coupling
result. M0 does not claim terrain-derived channel genesis, physical-width
inference, resolution convergence of extracted paths or that memory earns its
cost.

Unlike the frozen full discriminator, M0 deliberately lowers B's initiation
evidence from `1.2` to `0.9`, crossing `I_on=1.0` while remaining above
`I_off=0.75`. S0 therefore loses B by construction. This validates the
hysteresis mechanism; it is not evidence that snapshot extraction is unstable
under physically equivalent numerical jitter.

## Implemented mechanism

The fixture adds:

- ephemeral per-snapshot `ChannelCandidateId` values which are never promoted
  to semantic identity;
- a frozen dimensionless discharge-slope/resistance evidence score;
- separate initiation (`1.0`) and retention (`0.75`) thresholds;
- deterministic physical-anchor ordering and a bounded two-pointer
  correspondence pass;
- rejection of overlapping anchor neighborhoods that M0 cannot match uniquely;
- stable `ReachId` assignment on matched observations;
- explicit initiation, abandonment and capture event types;
- complete candidate DAG validation before mutation;
- a composition smoke test with the existing C1 remapper over unchanged
  reach-local intervals and segmentation;
- an explicit stop when reach birth or retirement would require a state ledger
  that M0 does not yet implement.

Correspondence is `O(N log N)` for deterministic physical sorting plus linear
matching, network construction and existing C1 remapping. The registered
four-reach case compares at most four anchor pairs per update.

## Manufactured observations

All resolutions use the routed-C1 physical network: four reaches, `288 km`
total length and `48 km²` prescribed active-channel area.

| Phase | Observation change | Snapshot S0 | Persistent S1 |
|---|---|---|---|
| Stable | Four scores above `I_on` | Four build-order IDs | Four physical-order lineage IDs |
| Threshold dip | Candidate IDs/order replaced; anchors move ±0.2 km; B crosses `I_on` to `0.9` | B is removed by construction; three IDs rebuilt | All four retained by hysteresis; no event |
| Capture | Anchors return; B receiver changes C→D | Rebuilt snapshot | One capture event; B lineage retained |

The anchor displacement is below the frozen `0.5 km` correspondence margin.
The capture uses the same physical B reach, so changing its receiver while
retaining B's lineage is intentional. M0 does not test newly cut or abandoned
physical intervals.

## Results

At nominal 8/4/2 km segmentation:

- S0 contains four reaches before the threshold dip and three during it, as
  required by the manufactured scores;
- S1 retains all four IDs within the unique-anchor fixture and emits no event;
- channel, interfluve and total elevation-volume moments close across the
  identical-interval remap within `1e-11`, `1e-9` and `1e-9 km³` respectively;
- the capture produces exactly one `B: C→D` event at each spacing;
- attached C1 state remains bit-identical through the capture comparison;
- an introduced A↔C cycle fails during candidate-network validation and leaves
  the old network and state unchanged;
- forcing B below the retention threshold reports one unsupported abandonment
  and stops before discarding state;
- overlapping anchor neighborhoods are rejected rather than silently swapping
  reach IDs;
- repeated promotion, matching and capture produce bit-identical audits and
  retained fixture state.

Six focused tests pass.

## Disposition

- **Pass M0 mechanism:** hysteretic promotion, uniquely matchable physical
  anchors and explicit capture compose transactionally with routed C1 state.
- **Architecture by construction, not test evidence:** ownership is kept outside
  continuum water and cell-mean terrain APIs at this rung.
- **Useful stop:** birth/retirement cannot proceed until explicit state
  initialization, dormancy and retirement ledgers are specified.
- **Not tested:** whether continuum-derived candidates flicker, whether their
  support converges, whether the chosen anchor correspondence is sufficient for
  branched geometry, whether moved/resized channel footprints remap
  conservatively, or whether persistence earns its product/runtime cost.
- **Not promoted:** the physical initiation equation, resistance field,
  prescribed widths, lineage layer, C1 landscape response or any product path.

M0 is therefore only implementation evidence that a narrow lineage mechanism
is coherent. It does not execute or pass the preregistered memory discriminator
and is not empirical evidence that the layered architecture is necessary. The
next principled decision is the missing extraction rung: select a bounded way
to obtain sparse channel candidates and physical overlap from MFD flow without
confusing face width, cell paths, active-channel width or renderer strokes.
Compare it against production SFD extraction and the prescribed skeleton
control before adding birth/retirement, sediment or long landscape runs.
