# Bounded orogen testbed Slice 1 — 2026-07-13

Status: implemented and numerically screened. Core invariants pass; the
resolution gate does not yet pass, so Slice 2 and product integration remain
blocked.

Design contract:
[`research/orogen-testbed-spec-2026-07-13.md`](../research/orogen-testbed-spec-2026-07-13.md).

## Implemented boundary

Slice 1 is isolated from `World` and `FineWorld` in the experimental
`world::landscape` module and `orogen_testbed` CPU binary. It includes:

- a dimensioned uniform planar hex finite-volume mesh;
- north/south open base level and east/west closed boundaries;
- area-normalized uniform U and linked-segment L deformation histories;
- time-ramped vertical rock-uplift rates, never prescribed terrain height;
- evolving priority-flood routing without terrain mutation;
- runoff/discharge accumulation;
- implicit stream-power incision;
- conservative nonlinear threshold hillslope transport;
- operator-specific adaptive timestep audits;
- solid-volume and water ledgers;
- atomic manifests, interval checkpoints, NDJSON metrics and summaries.

The neutral initial perturbation is a band-limited function of physical
coordinates and seed. Earlier cell-index white noise was removed because it
gave different initial landscapes to each resolution.

Not implemented: inherited provinces, sediment/cover, horizontal advection,
isostasy, drainage-integration cuts, climate feedback, glaciers, landform
objects, synthetic-topology controls or rendering.

## Numerical/unit evidence

Tests establish:

- reciprocal mesh faces and valid boundaries;
- U/L area-integrated forcing equality at 8 and 4 km;
- forcing cessation after the declared episode;
- routing acyclicity and water closure without surface mutation;
- internal hillslope conservation;
- exact linear implicit-incision reference behavior;
- deterministic full steps and state snapshots;
- solid-volume ledger closure;
- distinct U/L response under matched forcing;
- less than 5% change in the first response when timestep is halved.

The provisional discharge-form incision coefficient is `0.03`. This corrects
the dimensional conversion to `Q` in km³/Myr; the initial `2.5e-8` value made
incision inert by roughly five orders of magnitude. It is not visually fitted
and is not a product default.

## Full 8/4-km runs

Commands use the release `orogen_testbed` binary with 10 Myr duration, 0.01 Myr
requested steps and 0.25 Myr checkpoints.

| Case/resolution | Cells | Runtime | Uplift km³ | Incision export km³ | Peak relief km | Relative ledger residual |
|---|---:|---:|---:|---:|---:|---:|
| U / 8 km | 11,040 | 8.5 s | 100,625.000 | 83,593 | 2.703 | `4.4e-14` |
| U / 4 km | 44,400 | 43.3 s | 100,625.000 | 73,284 | 3.045 | `1.1e-13` |
| L / 8 km | 11,040 | 8.5 s | 100,625.000 | 83,934 | 3.038 | `1.2e-14` |
| L / 4 km | 44,400 | 43.2 s | 100,625.000 | 73,847 | 3.427 | `3.9e-14` |

The solver remains stable through 6 Myr forcing plus 4 Myr relaxation. U and L
remain distinct under identical integrated forcing. These facts establish a
working causal harness, not a successful terrain architecture.

## Resolution result

At 1 Myr, relief is already close and changes monotonically toward the 2-km
result:

| Case | 8 km relief | 4 km relief | 2 km relief |
|---|---:|---:|---:|
| U | 0.510 km | 0.515 km | 0.516 km |
| L | 0.569 km | 0.576 km | 0.577 km |

But cumulative incision export does not converge over the same sequence:

| Case | 8 km incision | 4 km incision | 2 km incision |
|---|---:|---:|---:|
| U | 1,567 km³ | 1,147 km³ | 816 km³ |
| L | 1,576 km³ | 1,155 km³ | 827 km³ |

By 10 Myr, the budget difference has accumulated into a large final-relief
split: U is 1.589 km at 8 km versus 2.379 km at 4 km; L is 1.778 versus
2.674 km. The current representation therefore fails the declared convergence
gate despite excellent algebraic ledger closure.

This is not fixed by the common continuous initial field. The leading diagnosis
is more precise than “the channel needs a width”: the solver asks one elevation
to be both a one-dimensional channel/path sample and a finite-volume cell-mean
surface. It then applies the path incision rate across the whole cell. Coarse
cells therefore imply wider established channels, while fine headwaters also
change because nearly every SFD flow path is incision-eligible. Outlet count,
catchment partitioning and the width/location of the pinned base-level row all
change with refinement. These are representation and boundary questions, not
candidates for compensating `K` by resolution.

The subsequent [channel/surface scaling decision](../research/channel-surface-scaling-2026-07-13.md)
selects a cell-mean finite-volume continuum with specific-discharge-driven
effective denudation for the next rung. A channel-path solver remains a valid
control, but cannot claim cell-volume export; a dual subgrid channel is deferred
until its additional state answers a demonstrated need.

## Disposition

- **Pass:** experimental isolation, dimensional state, forcing normalization,
  determinism, water/solid bookkeeping, timestep smoke and full-run stability.
- **Fail/block:** erosion-budget and post-relaxation relief convergence.
- **Do not:** tune `K`, uplift or relief separately per resolution; begin Slice
  2 object scoring; compare visual mountain quality; enter the product pipeline.
- **Resolved design question:** use an explicitly cell-mean continuum rather
  than a width patch, with fixed physical outlet portals and boundary-face
  conditions. Implement the analytic boundary, routing, depression and
  manufactured-denudation gate before rerunning U/L.

Evidence roots:

- `artifacts/orogen-testbed/slice1-continuous-8km/`
- `artifacts/orogen-testbed/slice1-continuous-4km/`
- `artifacts/orogen-testbed/slice1-continuous-2km-1myr/`
