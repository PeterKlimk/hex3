# C0 fixed-support discriminator audit

**Date:** 2026-07-13
**Control checkpoint:** `4a613a3`
**Status:** C0-Q16 evaluated; useful numerical control, not promoted

## Registered comparison

The preregistered contract is
[`../research/c0-support-discriminator-2026-07-13.md`](../research/c0-support-discriminator-2026-07-13.md).
Forcing, runoff, `K`, exponents, portals, initial field and hillslopes remain
frozen. C0-V is the corrected unfiltered control. C0-Q16 filters only the raw
scalar specific-discharge magnitude with a homogeneous-Neumann Helmholtz
operator at fixed `alpha=16 km`, then supplies that supported intensity to both
the slope CFL and denudation. It never filters elevation, water flux, routing,
physical grade or water diagnostics.

This arm is absent by default and the library's denudation `K` remains zero by
default. It is an isolated representation experiment, not selectable product
behavior.

## Analytic implementation gates

The implementation passes:

- omitted support retains the unfiltered control;
- `alpha=0` produces identical state evolution;
- first-step raw water diagnostics are identical between arms;
- supported intensity is consumed consistently by the CFL and denudation;
- filter non-convergence/invalid output is transactional;
- depression/uphill and below-base portal controls still invent no erosive
  grade;
- raw and supported maxima/integrals plus filter audit are serialized;
- the existing fixed-scale transfer test converges monotonically at 8/4/2 km.

At 2 km one U trial exposed recursive PCG residual drift: the recursive
residual crossed tolerance while the independently recomputed true residual
missed by `0.25%`. The trial correctly failed. The filter now recomputes
`rhs-Ax` at every apparent convergence, accepts only the true residual, and
otherwise performs a deterministic residual-replacement restart within the
unchanged iteration budget and tolerance. A regression exercises that path.

## Frozen 0.1 Myr U/L result

All values below are from separate preserved JSON artifacts. No result was used
to change `alpha`, `K` or forcing.

| Arm | Case | h | Relief | Export | Maximum E | Steps | Runtime |
|---|---|---:|---:|---:|---:|---:|---:|
| C0-V | U | 8 km | 31.126 m | 38.105 km³ | 0.0339 km/Myr | 13 | 2.51 s |
| C0-V | U | 4 km | 31.252 m | 38.422 km³ | 0.0679 km/Myr | 53 | 20.48 s |
| C0-V | U | 2 km | 31.284 m | 39.212 km³ | 0.1478 km/Myr | 253 | 162.30 s |
| C0-Q16 | U | 8 km | 30.928 m | 46.730 km³ | 0.00571 km/Myr | 10 | 0.26 s |
| C0-Q16 | U | 4 km | 31.214 m | 47.095 km³ | 0.00672 km/Myr | 10 | 1.48 s |
| C0-Q16 | U | 2 km | 31.273 m | 47.500 km³ | 0.00733 km/Myr | 20 | 121.36 s |
| C0-V | L | 8 km | 31.910 m | 38.317 km³ | 0.0284 km/Myr | 14 | 1.62 s |
| C0-V | L | 4 km | 31.976 m | 38.584 km³ | 0.0601 km/Myr | 58 | 16.89 s |
| C0-V | L | 2 km | 32.013 m | 39.333 km³ | 0.1456 km/Myr | 250 | 113.51 s |
| C0-Q16 | L | 8 km | 31.730 m | 47.146 km³ | 0.00739 km/Myr | 10 | 0.25 s |
| C0-Q16 | L | 4 km | 31.944 m | 47.499 km³ | 0.00794 km/Myr | 11 | 3.73 s |
| C0-Q16 | L | 2 km | 32.002 m | 47.885 km³ | 0.00906 km/Myr | 24 | 100.84 s |

Every ledger closes within `1.9e-10 km³`; portal water closes; sink storage and
unresolved raw-q cell counts are zero.

## What Q16 resolves

Raw peak intensity in the Q16-evolved surfaces still grows strongly with
refinement. Supported peak intensity does not:

| Case | Supported q max, 8/4/2 km | Per-refinement change |
|---|---|---|
| U | `0.589 / 0.623 / 0.654 million km²/Myr` | `+5.7% / +4.9%` |
| L | `0.969 / 1.029 / 1.058 million km²/Myr` | `+6.2% / +2.8%` |

The homogeneous-Neumann area-weighted q integral closes near solver tolerance.
Peak denudation grows much less than C0-V, relief error decreases strongly on
the second refinement, and export drift is held below `0.9%` per refinement.
Thus a fixed physical support does isolate the one-cell intensity defect; that
part of the hypothesis succeeds.

Accepted steps also fall by roughly an order of magnitude at 2 km. This does
not yield an order-of-magnitude runtime win: repeated global PCG solves leave
Q16 at `121/101 s` versus C0-V's `162/114 s` for U/L. The closure is faster at
8/4 km but only `1.34×/1.13×` faster at 2 km. It changes the cost owner rather
than resolving refinement cost.

## Why Q16 is not promoted

Q16 changes more than the divergent peak. At fixed `K`, export is about
`20–24%` above C0-V because smoothing changes the spatial covariance between
intensity and physical grade. That is a legitimate outcome of a different
closure, but it means Q16 is not a harmless numerical regularizer.

More importantly, the isotropic Helmholtz operator knows mesh adjacency but no
drainage-basin or channel-corridor topology. Its positive connected-domain
response necessarily transfers supported intensity across any divide not
encoded as a physical boundary. Raw water remains conservative and unchanged,
but the erosional closure can attribute one basin's concentrated intensity to
a neighboring positive-grade surface that receives none of that water. Calling
the result hydrologic specific discharge would therefore be false; calling it
regional supported intensity is honest but insufficient for channel, valley,
sediment or ecology consumers.

The coupled topography/water large-scale formulation in the literature avoids
this contradiction by declaring sub-alpha topography absent. Q16 deliberately
keeps that topography and therefore cannot borrow the same interpretation.
Making the filter basin-aware, anisotropic or wider would be another unregistered
model family, not a correction to this result.

## Disposition

- **Pass:** fixed physical support stabilizes the local intensity diagnostic
  and preserves the raw water and solid accounting contracts.
- **Pass:** true-residual PCG auditing and deterministic replacement.
- **Mixed:** relief and export refinement improve, but export is not strictly
  asymptotic and the closure changes its magnitude materially.
- **Fail for promotion:** isotropic supported intensity has no channel/divide
  ownership and global filter work largely replaces the saved explicit steps.
- **Retain:** C0-V as the conservative regional-surface control with its local
  truth limit; C0-Q16 as a falsified/diagnostic support arm, not product state.
- **Advance:** only the minimal C1 analytic core `{z_bar, z_c, f_c}` with exact
  volume mixing and fixed physical channel area. Do not add sediment, valley
  belts or ecology state until that core passes manufactured gates.
- **Do not:** tune `alpha` or `K`, implement a basin-aware Q16 rescue, run 1/10
  Myr, or integrate either arm into the product pipeline.

The next bounded task is a C1 manufactured fixture, not a coupled global model:
prove resolution-invariant `sum(wL)`, channel-bed evolution/export, exact
`z_bar` mixing and hillslope/channel transfer cancellation at 8/4/2 km.
