# Routed C1 ownership and remapping fixture

**Status:** preregistered manufactured network gate
**Date:** 2026-07-13
**Predecessor:** minimum C1 fixture at `abead00`

## Question

The minimum C1 algebra now preserves physical channel area, channel evolution
and authoritative cell-mean volume. It still assumes that reach length, width
and ownership are already known. This rung asks whether stable semantic reaches
can own those quantities through refinement, a confluence and one controlled
capture event without losing water, material or state identity.

It does not generate drainage, infer width, use terrain to choose receivers,
carry sediment or run coupled tectonics.

## Prescribed semantic network

Four stable reaches form two outlets:

```text
before:  A ─┐
             ├→ C → outlet C
         B ─┘

         D ───→ outlet D

after:   A ───→ C → outlet C
         B ───→ D → outlet D
```

Only B's downstream receiver changes. Reach IDs, physical geometry and all C1
state remain unchanged by the topology event itself.

| Reach | Length | Channel width | Represented swath | Grade | Headwater Q | Lateral supply per km |
|---|---:|---:|---:|---:|---:|---:|
| A | 64 km | 0.12 km | 12 km | 0.012 | 2.00 km³/Myr | 0.10 km³/Myr/km |
| B | 48 km | 0.10 km | 10 km | 0.015 | 1.00 km³/Myr | 0.08 km³/Myr/km |
| C | 96 km | 0.22 km | 16 km | 0.008 | 0.50 km³/Myr | 0.05 km³/Myr/km |
| D | 80 km | 0.18 km | 14 km | 0.009 | 0.75 km³/Myr | 0.04 km³/Myr/km |

Total physical reach length is `288 km`; total active-channel area is exactly
`48 km²`. Before capture, outlet flows are `18.54` and `3.95 km³/Myr`; after
capture they are `13.70` and `8.79 km³/Myr`. Total outflow remains
`22.49 km³/Myr`.

## Discretization and routing

Each semantic reach is partitioned independently at nominal 8/4/2 km. Segment
identity is `(ReachId, physical interval [s0,s1])`; vector index is not semantic
identity. Segment length partitions the declared reach exactly.

For local lateral supply density `r`, route in upstream-to-downstream order:

```text
Q_out = Q_in + r Δs
Q_mean = Q_in + 0.5 r Δs.
```

The downstream reach receives the sum of terminal upstream flows plus its own
declared headwater input. The reach graph must be a deterministic acyclic graph
with unique IDs. Total sources equal total outlet flow near roundoff.

## Manufactured C1 response

Use prescribed unit-stream-power channel lowering

```text
E_c = K (Q_mean / w) S
```

with fixed `K=1e-4 km^-1`, the declared reach grade and a fixed `dt=0.1 Myr`.
Apply `dz_c=-E_c dt` through the already validated C1 excavation operator.
Because `Q(s)` is linear and the midpoint rule is exact, integrated export must
be invariant at 8/4/2 km without resolution-dependent coefficients.

This validates ownership and composition only. Grade is prescribed rather than
derived from evolving channel state; no equilibrium/profile claim is made.

## Conservative state remapping

Remap state between segmentations by physical interval overlap within the same
stable ReachId. Independently length-average `z_c` and reconstructed `z_i`, then
reconstruct `z_bar` from the destination fraction. The remap must preserve:

```text
sum(w Δs z_c)
sum((A - w Δs) z_i)
sum(A z_bar)
```

near roundoff. It may not move state between reaches or use nearest vector
index. Refinement cannot invent subsegment variation; preservation, not hidden
detail reconstruction, is the claim.

## Registered gates

1. At 8/4/2 km, every ReachId retains exact length and channel area; network
   totals remain `288 km` and `48 km²`.
2. Water closes at each segment, confluence, reach and outlet. The registered
   before/after outlet flows are recovered at every spacing.
3. The midpoint unit-stream-power response produces invariant integrated
   export and closes the C1 mean-volume ledger.
4. Tributary A/B water and incision are unchanged by B's receiver change;
   only trunks C/D and their outlet allocation respond afterward.
5. The topology event leaves all C1 state bits and stable ReachIds unchanged.
6. 8→4→2 overlap remapping preserves channel, interfluve and total moments;
   no cross-reach state transfer occurs.
7. A cycle, unknown receiver, duplicate ID, invalid physical geometry or
   incomplete overlap fails transactionally/deterministically.
8. Internal hillslope→channel transfer remains explicit and cancelling when
   composed per physical reach length.
9. Repeated construction, routing, remapping and response are bit deterministic.

No long U/L run, parameter sweep or visual score follows automatically.

## Cost and disposition

Construction, routing, response and remap should be `O(N_segment + N_reach)`;
no global elliptic solve is permitted. Report segment count and work ownership,
but do not optimize before correctness.

Passing selects stable semantic reach ownership as the next C1 testbed
architecture. It still leaves network generation and width evolution as
separate missing systems. Failure stops before coupling to landscape routing.
