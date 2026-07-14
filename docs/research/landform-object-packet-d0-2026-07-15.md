# Landform object packet D0 executable contract

**Status:** implemented/evaluated; common planar checkpoint passes

**Date:** 2026-07-15, before D0 implementation

**Parent:** [Landform object packet v0](landform-object-packet-v0-2026-07-14.md)

**Predecessor:** [G0/S0 executable contract](landform-object-packet-g0s0-2026-07-14.md)

**Outcome:** [D0 common drainage audit](../audits/landform-d0-common-drainage-2026-07-15.md)

## Decision question

Can one arm-neutral, physically measured drainage tree expose basins, reaches,
catchment boundaries and conditioning debt on every bounded-testbed surface
without claiming subcell channel geometry or rewarding an arm's native drainage
representation?

D0 is evaluation infrastructure. It does not promote a terrain, river or
hydrology model. It derives one deterministic topology from the final physical
surface supplied by each later H/C/G arm. Native C fluxes, G's authored graph,
product river visibility and presentation state are excluded from its decisions.

## Scope

This checkpoint implements only the common planar/testbed derivation over an
already validated `EvaluationSurfaceGraphV0`. It emits drainage conditioning,
an exhaustive cell-to-portal receiver forest, retained reach topology, nested
catchments, exclusive incremental catchments and raw catchment-boundary faces.

The separately identified product-`Hydrology` adapter permitted by the parent
contract is deferred to a later checkpoint. Its routing and physical surface
history differ and must never be numerically presented as the common derivation.
O0 owns ridge/divide relationships, valley probes, cross-surface correspondence
and the combined evidence packet. D0 does not consume S0.

## Frozen input and configuration

Inputs are:

- one validated planar `EvaluationSurfaceGraphV0` with at least one genuine
  `OpenBaseLevel` boundary segment;
- finite physical cell-mean elevation `z_i` in kilometres;
- finite non-negative local runoff supply `q_i` in a caller-declared common
  volume-per-time unit; and
- the complete physical domain. D0 has no scored or terrain-arm mask.

The reference retained-reach support is `2,000 km2`. Independent sensitivity
outputs use `1,000 km2` and `4,000 km2`; they cannot select or tune an arm.
No elevation, runoff or graph smoothing is permitted. Configuration and schema
are exact v0 values, not runtime knobs.

## Virtual conditioning

Let `f_i` be the minimum, over all graph paths from cell `i` to an open portal
face, of the maximum of every physical cell elevation on the path and that
portal's base elevation. Compute it with a deterministic minimax priority flood
seeded by every open portal segment. Physical `z_i` is never changed.

For every exact-equal-`f` component, seed potential zero at every cell having a
strictly lower-`f` neighbour or an open portal face with base elevation at most
`f_i`. Breadth-first graph distance within that exact-equal component is the
flat potential `p_i`. A cell disconnected from all portals or left without a
potential is a typed error.

A depression record is one maximal connected component of cells having
`f_i > z_i` and the same exact `f_i`. Its stable ordering is by spill elevation,
then coordinate-canonical anchor. Record the spill elevation, anchor, affected
cells, physical area, maximum fill depth and integrated virtual fill volume
`sum(area_i * (f_i - z_i))`.

After receivers are frozen, a depression's parent is the first different
positive-fill component encountered downstream from any of its exiting
receiver segments. All exits must agree on that first component or absence;
otherwise the depression hierarchy is a typed ambiguity error. This retains
nested spill levels without pretending the positive-fill mask is one flat
basin.

## Frozen single-receiver rule

An internal face `i -> j` is eligible exactly when:

- `f_i > f_j`; or
- `f_i == f_j` and `p_j < p_i`.

An open portal segment owned by `i` is eligible exactly when:

- `f_i` is greater than its base elevation; or
- the values are equal and `p_i == 0`.

Each eligible candidate receives the same unnormalised finite-volume weight as
the existing portal-aware slope partition:

```text
internal strict drop: shared_width * (f_i - f_j) / center_distance
internal equal flat:  shared_width / center_distance
portal strict drop:   segment_width * (f_i - base) / center_distance
portal equal flat:    segment_width / center_distance
```

The portal center distance is the Euclidean distance from the owner-cell center
to the segment midpoint. Non-finite or non-positive geometry or weight is a
typed error.

Choose the greatest weight using `f64::total_cmp`. Exact ties use, in order:
lower downstream routing elevation, lower downstream flat potential (zero for a
portal), lexicographically smaller directed-face midpoint `(x,y,z)`, internal
before portal, lexicographically smaller destination center for internal faces,
then smaller semantic portal ID for portal faces. Cell index and CSR visit order
are never semantic tie-breaks.

Every cell has exactly one receiver, either another cell or a semantic portal.
The receiver key strictly decreases in `(f, p)`, so cycles are invalid rather
than repaired. Every chain must terminate at a portal. A terminal cell,
disconnected component, unknown portal or cycle is a typed error.

## Structural and runoff accumulation

Accumulate in the reverse receiver order:

```text
structural_area_i = physical_cell_area_i + sum(upstream structural_area)
supplied_runoff_i = q_i + sum(upstream supplied_runoff)
```

Structural area and runoff supply remain separate. Per-portal totals must close
against total physical area and total local supply. Use compensated `f64`
summation for public ledgers. Each absolute residual must be no greater than
`max(1e-9, 1e-12 * abs(total))` in the reported units. Failure is typed; no
residual is assigned to a sink.

Mark a receiver segment as fill-supported when either endpoint has positive
virtual fill, and flat-supported when it uses the exact-equal flat rule. Mark it
physically non-descending when the receiver's physical elevation is not lower
than the donor's, or the portal base is not lower. Reach records retain the OR
of these flags and the count and physical length of affected segments.

## Retained reaches

Materialize a complete scale record independently at `1,000`, `2,000` and
`4,000 km2`; reach IDs and catchments are local to one scale. At a scale, a
channel-support cell is one whose structural contributing area is greater than
or equal to its threshold. Let its upstream degree count only supported cells.

A reach begins at every supported cell whose supported upstream degree is not
one. It owns that cell and consecutive downstream supported cells while the
next cell has supported upstream degree exactly one. A confluence cell belongs
to the downstream reach; tributary reaches stop at their preceding cells. Every
supported cell belongs to exactly one reach.

The reach owns every receiver segment originating in its cells, including its
last segment into the next reach or portal. Physical length is the sum of
center-to-center distances for internal segments and center-to-segment-midpoint
distance for a terminal portal segment. Record ordered cell support, upstream
and downstream reach IDs, terminal portal, head/tail structural area and runoff,
conditioning flags and segment counts.

Strahler order is computed on the retained reach DAG: a source is order one; a
downstream reach takes the maximum upstream order, incremented only when that
maximum occurs at least twice. Exact topology, not runoff, breaks no Strahler
tie.

For each portal-rooted retained network, record three separate main-stem paths:

- **greatest supply:** at each upstream choice select greatest tail runoff;
- **longest trunk:** select greatest cumulative source-to-choice physical
  length; and
- **highest order:** select greatest Strahler order.

Remaining exact ties use greater structural area, greater physical length, then
the coordinate-canonical reach head. The three paths are reported independently
even when they coincide. There is no D0 `major` boolean.

## Catchments and raw boundary faces

Each retained reach defines a nested contributing catchment containing all cells
upstream of its owned receiver segments. Do not store quadratic member lists;
record physical area, local runoff, total supplied runoff, parent reach and child
reaches so nesting is reconstructed from the reach tree. Nested catchments may
overlap only by declared reach ancestry.

Exclusive incremental ownership is a partition. Starting at a cell, follow its
receiver chain; the first supported cell encountered owns the cell through that
cell's unique retained reach. If no supported cell occurs, the terminal portal
owns it. Because supported-cell ownership is unique, no further tie exists.

For each reciprocal internal face whose two cells have different exclusive
owners, emit exactly one raw catchment-boundary record using the lower
coordinate-canonical directed-face representation. Store both owner IDs,
endpoints and physical shared length. These partition seams are not called
geomorphic divides or grouped into polylines until O0 demonstrates bilateral
relief and ridge correspondence.

## Evidence and hashing

D0 has its own schema and evidence hash, separate from frozen G0/S0. The hash
includes domain geometry identity, physical elevation, local supply, exact
configuration and all derived D0 evidence. It excludes arm labels, native C/G
graphs, product semantics and presentation state. Changing those excluded
labels must leave output and hash byte-identical.

## Manufactured gates

Before any competitive surface is observed, pass:

1. one portal-draining plane: all cells terminate, ledgers close and no
   conditioning is reported;
2. exact flat with two portals: BFS routing terminates deterministically,
   physical elevation is bit-identical and portal identity is stable;
3. nested depressions: spill hierarchy, fill area/volume and every
   non-descending segment are reported without physical mutation;
4. asymmetric fork/confluence: structural area and runoff close locally and
   globally, reach segmentation and Strahler order are exact;
5. role-separation fixture: greatest-supply, longest-trunk and highest-order
   paths are deliberately distinct;
6. two-basin fixture: exclusive ownership is complete, nested areas are honest
   and every raw boundary face appears once; and
7. malformed CSR, non-finite/negative input, absent/unknown portal,
   disconnection, receiver cycle and ledger failure return typed errors without
   partial output.

Run applicable geometry/topology cases at 8/4/2 km and report raw object and
ledger deltas. No cross-resolution promotion tolerance is invented in D0.

## Stop rule

Stop after the common D0 implementation, manufactured matrix and dated audit.
Do not adapt product hydrology, implement O0, compose H/C/G, tune terrain or add
channel beds, widths, persistent lineage, sediment, valley polygons or renderer
selection in this checkpoint. Choose the next rung from the observed D0
validity, cost and representation limits.
