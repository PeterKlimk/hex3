# Routed C1 ownership and remapping audit

**Date:** 2026-07-13
**Predecessor checkpoint:** `abead00`
**Status:** manufactured routed gate passed; network generation remains absent

## Scope

This audit evaluates the frozen network in
[`../research/c1-routed-fixture-2026-07-13.md`](../research/c1-routed-fixture-2026-07-13.md).
It is an isolated prescribed-network fixture. Terrain does not choose receivers;
width and reach geometry do not evolve; no sediment, valley, ecology or product
state is present.

The implementation adds:

- stable `ReachId` ownership independent of vector position;
- deterministic DAG validation and topological routing;
- exact physical interval segmentation with per-reach ownership ranges;
- conservative headwater/lateral supply, reach and outlet audits;
- atomic receiver replacement with unchanged geometry/state ownership;
- fixed unit-stream-power response composed through the existing C1 excavation
  ledger;
- conservative overlap remapping by `(ReachId,[s0,s1])`;
- explicit per-length internal compartment transfer through the existing C1
  transfer ledger.

Routing uses precomputed contiguous per-reach segment ranges. Remapping uses a
per-reach two-pointer interval sweep. Both perform segment-linear work with
small deterministic ordered-reach overhead and no global solve.

## Registered network result

| Spacing | Segments | Total reach length | Active-channel area | Pre-capture outlets C/D | Post-capture outlets C/D |
|---:|---:|---:|---:|---:|---:|
| 8 km | 36 | 288 km | 48 km² | 18.54 / 3.95 | 13.70 / 8.79 km³/Myr |
| 4 km | 72 | 288 km | 48 km² | 18.54 / 3.95 | 13.70 / 8.79 km³/Myr |
| 2 km | 144 | 288 km | 48 km² | 18.54 / 3.95 | 13.70 / 8.79 km³/Myr |

Total source and outlet flow are `22.49 km³/Myr` before and after capture.
Segment closure is exact zero; reach/network closure is within `5e-14`.

With fixed `K=1e-4 km^-1`, `dt=0.1 Myr`, prescribed grade and midpoint
discharge, integrated C1 excavation export is `0.0002018352 km³` at every
spacing. The midpoint rule is exact for the registered linear lateral-supply
field. Mean-volume closure is within `2e-10 km³`.

## Identity and reorganization result

The registered event changes only B's receiver from C to D.

- B and A segment flows and incision response remain bit-identical.
- C loses B's terminal flow; D gains exactly that flow.
- C/D response changes after the event as required.
- All reach IDs, segment physical intervals, C1 mean/channel state bits and
  geometry remain unchanged by the topology event itself.
- An invalid event that introduces a cycle fails before network or state
  mutation.

Thus topology is mutable while semantic reach ownership is stable. This is the
same separation required elsewhere in Hex3 between physical state and changing
relationships.

## Conservative remapping result

The 8→4→2 remap length-averages channel and reconstructed interfluve state only
over physical overlap within the same ReachId, then reconstructs the destination
mean.

- global channel/interfluve/total elevation-volume moments are preserved within
  `3e-10 km³`;
- per-reach channel moments are preserved within `2e-12 km³`;
- per-reach interfluve moments are preserved within `2e-10 km³`;
- changing B's receiver during remeshing does not move state between reaches;
- missing reach coverage, incompatible physical geometry and incomplete
  intervals fail explicitly.

Refinement does not invent subsegment variation. The claim is conservative
ownership, not recovery of absent detail.

An explicit per-length internal transfer on A and D moves `0.368 km³` between
compartments, leaves every authoritative mean bit unchanged and closes with
exact net zero.

## Verification

- 8 routed-C1 focused tests pass.
- 88 landscape tests pass.
- Full suite passes with 188 library tests plus every integration group; one
  existing ignored test remains ignored.
- `cargo build --bin hex3`, formatting and diff checks pass.
- Clippy reports the existing 35 library warnings and none from the new module.

## Disposition

- **Pass:** stable semantic reach and physical interval ownership.
- **Pass:** conservative confluence routing and capture reallocation.
- **Pass:** resolution-invariant physical support and manufactured response.
- **Pass:** conservative compartment-aware remapping without cross-reach state
  transfer.
- **Pass:** local/near-linear work ownership; no global filter.
- **Promote for the isolated testbed:** prescribed semantic receiver graphs are
  a viable owner for C1 state and topology events.
- **Still absent:** a justified system that creates, prunes and evolves the
  receiver graph and physical widths from world state.
- **Do not:** identify vector indices as reaches, generate networks by incidental
  grid paths, infer width from cell size, attach sediment/ecology, or run long
  U/L/product integration.

## Next architectural decision

The next task should zoom out one level and select the network owner before more
C1 mechanics. Compare, on paper and then with one bounded discriminator:

1. a semantic graph extracted from conservative continuum face flow;
2. an explicit persistent drainage skeleton evolved as its own object system;
3. a physical channel-initiation/support threshold that promotes only stable
   portions of the continuum graph.

Judge identity stability through reorganization, physical correspondence,
coupling to C1 area/width, cost and downstream value for rivers, valleys,
sediment and ecology. Do not presume that the most simulation-like owner wins.
