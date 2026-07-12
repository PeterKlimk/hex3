# C0 fixed-support discriminator and minimum C1 contract

**Status:** preregistered representation experiment; not product architecture
**Date:** 2026-07-13
**Control:** corrected unfiltered C0 from the Slice1R audit

## Question

The corrected C0 solver converges tightly in short-run integrated relief, but
its peak cellwise denudation grows by `4.4–5.1×` from 8→2 km and its explicit
work grows rapidly. Is that remaining behavior adequately represented by a
declared fixed physical support for regional surface response, or does Hex3
need simultaneous channel/interfluve state?

This is a representation test, not a calibration sweep. Forcing, runoff,
`K`, exponents, initial field, portals and hillslope parameters remain frozen.

## Why reality does not have a one-cell channel

Real fluvial work occupies finite channel cross-sections, migrates laterally,
exchanges material with banks and hillslopes, and may be buffered by sediment.
A grid model that sends conservative hillslope supply into a one-cell river
shrinks that river's physical support with grid spacing. Pointwise lowering can
therefore change even when the receiver profile or integrated relief looks
converged.

This is the structural grid-dependence identified by Hergarten (2020,
<https://doi.org/10.5194/esurf-8-367-2020>) and revisited by Hergarten (2025,
<https://doi.org/10.5194/esurf-13-277-2025>). Coatléven and Chauveau (2024,
<https://doi.org/10.5194/esurf-12-995-2024>) instead declare a minimum physical
continuum scale. Those approaches answer different questions from an explicit
subgrid channel.

## Registered arms

### C0-V — unfiltered control

Use the corrected finite-volume face routing, raw reconstructed scalar
specific-discharge magnitude, physical outgoing-face grade and effective
areal law exactly as checkpoint `4a613a3`.

### C0-Q16 — fixed-support discharge intensity

After conservative water routing and physical grade construction, solve

```text
(M + alpha² L) q_alpha = M q_raw
```

with `alpha=16 km` and homogeneous-Neumann physical boundaries. Sixteen
kilometres is preregistered because it is the smallest fixed scale already
shown by the analytic filter transfer test to be numerically represented at
all 8/4/2 km spacings. It is an experimental support scale, not a fitted river
width. Helmholtz `alpha` is a decay length and not a compact cutoff radius.

`q_alpha` is named **supported discharge intensity**. It is supplied to both
the slope-response CFL and `E=K q_alpha^m S_physical^n`. It must not be reported
as water flux or used to replace raw hydrologic diagnostics.

The arm deliberately does **not** filter:

- authoritative mean elevation;
- routing/fill elevation;
- directed water flux, portal flow or sink storage;
- flow direction or physical grade;
- tectonic forcing or hillslope transport.

Thus a flat/uphill filled connection still has exactly zero erosive grade, and
the first-step water topology is identical to C0-V. Later topology may differ
only because the two arms evolve different physical surfaces.

The filter must converge at its frozen tolerance, return finite nonnegative
intensity, and remain transactional. No negative clamp, local RMS/L1 fallback,
resolution-dependent `alpha` or resolution-dependent `K` is allowed. A failed
filter fails the trial.

This is a deliberately minimal authentic closure. It is not the coupled
topography/water large-scale formulation of Coatléven and Chauveau, and it does
not claim to resolve channels.

### C1 — paper contract, not implemented in this experiment

The minimum defensible dual subgrid state is:

```text
z_bar  authoritative cell-mean surface elevation
z_c    channel-bed elevation
f_c    physical channel occupied fraction
```

For cell area `A`, routed reach length `L` and physical width `w`,

```text
A_c = min(w L, A)
f_c = A_c / A
z_bar = f_c z_c + (1 - f_c) z_i
```

so the interfluve mean `z_i` is reconstructable. Channel lowering changes the
cell elevation-volume moment by `A_c dz_c`, not `A dz_c`; hillslope/channel
exchange must close with equal and opposite ledgers. Direction, reach length
and water discharge may remain network diagnostics rather than duplicated
state.

This core is only erosion-geometry-ready. Sediment requires separate alluvial
cover/storage following models such as SPACE (Shobe et al. 2017,
<https://doi.org/10.5194/gmd-10-4577-2017>). Valley and ecology consumers need
a distinct valley/channel-belt fraction or width; channel width is not a proxy
for the multi-flood disturbed corridor. Disturbance succession eventually
requires age/time-since-reworking. These meanings must not be collapsed into
one channelization scalar.

## Analytic gates before the coupled screen

1. Omitted support is byte-identical to the checkpointed C0-V control.
2. `alpha=0` reproduces raw intensity and state exactly, apart from declared
   filter metadata.
3. C0-V and C0-Q16 have identical one-step water flux/topology diagnostics from
   the same initial state.
4. Constant supported intensity is unchanged; the fixed-scale sinusoidal
   transfer test retains monotone 8→4→2 convergence.
5. The filter audit is exposed; non-convergence or invalid output leaves state
   byte-unchanged.
6. The supported intensity is used consistently by both denudation and its CFL.
7. Depression/uphill and below-base portal controls retain zero invented
   denudation.

The later C1 implementation, if justified, must additionally prove invariant
`sum(w L)`, channel-bed evolution and export at 8/4/2 km; exact `z_bar` volume
mixing; exact hillslope/channel transfer cancellation; and a zero-width limit.

## Frozen coupled comparison

Run C0-V and C0-Q16 for U and L at 8/4/2 km for `0.1 Myr`, requested
`dt=0.01 Myr`, using the existing seed and smoke regime. Preserve every arm in
separate JSON artifacts. Do not run 1 Myr before disposition.

Report independently:

- relief and denudation export at each spacing;
- raw and supported intensity maxima and area-weighted integrals;
- accepted steps, attempts, governing limiter, filter iterations and runtime;
- solid/water closure, sinks and unresolved raw-q cells;
- first-step water identity and filter determinism;
- whether refinement error decreases monotonically for each metric.

No composite score may hide a failure.

## Disposition rules

- **Retain C0-V** if local divergence is a declared truth limit and C0-Q16 does
  not materially improve it without semantic leakage or disproportionate cost.
- **Retain C0-Q16 as a regional arm** only if fixed support stabilizes local
  intensity and refinement behavior while water semantics, physical-grade
  controls and ledgers remain intact. It still does not become a channel model.
- **Advance C1** if simultaneous channel/interfluve relief or physical occupied
  fraction is required, if fixed support merely smears work across divides, or
  if future sediment/valley/ecology consumers cannot use C0 honestly.
- **Stop** on a failed analytic gate. Do not tune `alpha`, `K`, runoff, MFD
  weights or forcing to rescue the selected arm.

Whatever wins this discriminator remains an isolated testbed representation.
Product integration and visual morphology evaluation require separate gates.
