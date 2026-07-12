# Bounded orogen testbed Slice 1R — 2026-07-13

Status: **analytic infrastructure partially passes; coupled C0 remains
blocked**. This rung implements the representation decision without changing
the legacy Slice 1 U/L solver.

Design basis:
[channel support, surface meaning and mesh convergence](../research/channel-surface-scaling-2026-07-13.md).

## Implemented boundary

The experimental landscape module now contains three deliberately separate
operators:

- the existing SFD/implicit stream-power path solver, retained as the P control;
- an unfilled, strictly-downhill finite-volume MFD face-flow operator which
  reconstructs a continuum specific-discharge vector;
- a transactional C0 effective areal-denudation operator on cell-mean bedrock.

The production U/L path still uses Slice 1 SFD, cell-wide implicit incision and
pinned boundary rows. No result below promotes or silently changes it.

## A. Pathway-law control — pass

A fixed one-dimensional receiver chain prescribes discharge and the analytic
steady slope

```text
S = (U / (K Q^m))^(1/n).
```

After uniform donor uplift, the existing backward-Euler incision step restores
the nonlinear equilibrium profile to floating-point tolerance. Cell areas are
explicitly arbitrary and the routine's area-weighted return is ignored. This
establishes that the incumbent implicit mathematics is valid as a sampled
path/channel-bed law; it does not establish physical excavation volume.

## B. Boundary geometry — conservative foundation pass, exact rectangle deferred

An initial implementation partitioned the requested `960 × 640 km` rectangle
into nominal boundary strips. Review found those strips were not faces of the
stored full-hex control volumes: cell areas and internal faces described a
jagged union of complete cells, while the nominal perimeter described a second
incompatible domain. That implementation was replaced before flux consumed it.

The retained mesh now creates one genuine exposed face for every missing
lattice neighbor:

- physical width `h / sqrt(3)` and center distance `h / 2`;
- f64 midpoint and outward normal;
- stable portal IDs and fixed projected coordinate spans;
- explicit distinction between projected portal coverage and physical
  sawtooth boundary arc length;
- actual-domain area and boundary-arc reporting.

Tests close the discrete Gauss vector and area moment per cell, close the global
boundary area against the sum of cell areas, prove missing-face uniqueness and
prove portal split/identity/coverage behavior at 8/4/2 km.

The conservative domain is therefore the union of full hex cells approximating
the target rectangle. A genuinely exact rectangular hex domain would require a
complete cut-cell/constrained-Voronoi mesh, including clipped areas, shortened
internal faces and sliver handling. That work is deferred unless boundary
sensitivity becomes a discriminator. Exact Dirichlet boundary-flux formulas
still require the registered 1-D/Cartesian manufactured fixture.

## C. Interior finite-volume routing — pass for the implemented analytic cases

`FaceFlowCache` is isolated from `DrainageCache`. It:

- sorts cells into a strict high-to-low DAG;
- partitions each cell's available water with
  `weight = face_width × physical_slope` (`p = 1`);
- records CSR-aligned directed fractions and water fluxes;
- closes local and global supply/storage balances;
- reconstructs a specific-discharge vector from signed `Q / face_width` samples
  using an f64 weighted two-dimensional least-squares solve;
- optionally routes through genuine open boundary subfaces using the same
  `face_width × slope` normalization and aggregates outflow by stable portal ID;
- treats flats and local minima as explicit instantaneous storage rather than
  inventing an index-dependent route.

Fixed-physical-mask errors decrease monotonically at four lattice-relative
angles:

| Case | 8 km relative vector error | 4 km | 2 km |
|---|---:|---:|---:|
| Plane, angle range | 1.9–3.4% | 1.1–1.4% | 0.57–0.70% |
| Ridge, angle range | 8.3–10.0% | 4.2–5.1% | 2.1–2.5% |

This passes the initial interior plane/ridge test at the preregistered `p = 1`;
the exponent was not fitted.

Portal tests additionally prove local/global water closure, zero flow across
closed faces and stable one/two-portal IDs at 8/4/2 km. In a symmetric
two-portal case, the left-portal outflow share refines
`0.52525 → 0.49773 → 0.49958`; the comparison uses actual-domain supply rather
than pretending the jagged domains have identical area.

Subsequent radial and convergent gates refine the result:

- compact radial-divergent vector/magnitude error decreases
  `9.43% → 4.49% → 2.29%` at 8/4/2 km;
- convergent V/strip vector error at one-sided probes decreases to
  `2.08–2.52%` at 2 km over four orientations;
- **the convergence-line scalar fails:** analytic two-sided sink strength
  converges to `9.6 km²/Myr`, while `|q_vector|` reaches only
  `1.38 → 1.49 → 1.54 km²/Myr`, or 14.4–16.1% of that strength.

Opposing face inflows cancel in a single vector. This does not invalidate the
vector as a net direction away from convergence, but it blocks using its
magnitude as the sole scalar water support for C0 denudation. The failure is
preserved; `p`, `K` and test thresholds were not changed.

A resolved broad axial reach supplies the missing discriminator. Integrated
cut flux closes to `1.2e-8` relative error or better and the support-corrected
downstream vector/magnitude errors at 2 km are below 0.5% for aligned and
rotated cases. Thus C0 retains the consistent LS vector and magnitude on
resolved continuum flow. The exact line sink remains a declared singular truth
limit, scored by cut flux rather than an invented pointwise scalar.

## D. Depressions and flats — initial analytic pass

The direct unfilled control intentionally stores water at flats/minima. The C0
depression route instead performs two separate derivations:

1. a portal-seeded minimax priority fill determines the lowest spill elevation
   connecting each cell to an outlet;
2. an independent multi-source BFS starts at genuine lower-filled neighbors and
   open portal cells, assigning graph distance across each exact equal-filled
   component.

Routing orders higher fill levels first, then higher flat potential, and sends
equal-level flow only toward decreasing potential. It uses no epsilon slope,
heap visitation rank or cell-index potential, and physical elevation remains
byte-unchanged.

Tests pass for an exact flat at 8/4/2 km with two stable portals and for
constructed nested bowls whose inner and outer pits fill to their declared 4 km
and 1 km sills. Routes are deterministic, acyclic and conservative, with no
unexplained sink storage in these cases. Wider mesh-rotation and natural-flat
convergence remain future coupled evidence.

## E. Manufactured C0 denudation — pass in isolation

The new operator evaluates

```text
E = K q^m S^n
```

as effective areal lowering of an explicitly cell-mean bedrock surface. It
validates all inputs transactionally, returns per-cell `E`, and records exactly
`sum(E × cell_area × dt)` as solid export. It makes no channel-width or
channel-bed claim.

For a smooth separable manufactured field with an analytic area integral, the
relative export errors at 8/4/2 km are approximately
`8.03e-4 / 1.47e-4 / 1.98e-6`. Field values and the per-mesh ledger also close
to floating-point tolerance.

## F. Physical mean-surface gradient and flow alignment — pass in isolation

A separate weighted two-dimensional least-squares operator reconstructs the
gradient of physical cell-mean bedrock from real neighboring cells. It never
consumes the filled routing surface or invents portal ghost elevations. Affine
planes at four angles reproduce their vector and grade to near roundoff in the
fixed physical interior at 8/4/2 km. A smooth radial surface decreases error
monotonically over the same refinement sequence, and repeats are bit
deterministic.

The derived grade is explicitly `max(-q_hat · grad(z_mean), 0)`, with exact
zero for exact zero discharge. Routing/fill elevation is not an input, so a
filled connection across physically flat or uphill terrain cannot invent an
erosional slope. Net direction remains useful even though scalar intensity at
convergence needs a separate representation. The coupled solver does not use
this LS grade for denudation: it uses the stricter outgoing-face construction
described below so an open portal's Dirichlet datum is represented exactly.

## G. Conservative C0 hillslopes — pass in isolation

A new operator visits each reciprocal internal face once and applies the
nonlinear threshold transport law with an explicit Jacobian timestep limit.
Internal volume transfer is exactly antisymmetric. Closed physical faces carry
zero flux; genuine open portal faces use the separately validated **linear**
Dirichlet control and report signed transfer by stable portal ID. Cells are
never pinned.

Closed affine/perturbed fields conserve volume, open-face transfer matches the
Dirichlet fixture, smooth manufactured response converges at 8/4/2 km, and
unstable or non-finite trials are transactional. Nonlinear threshold behavior
at an external base-level face is intentionally not claimed.

### Analytic fixed-support filter foundation — pass in isolation, not selected

A deterministic f64 finite-volume Helmholtz solve is available for the later
representation comparison. It solves `(M + alpha² L)y = Mx` with genuine
Neumann or portal-Dirichlet boundaries and reports true residual and iteration
count. Identity, constant preservation, Neumann integral conservation,
positivity tolerance, portal localization and deterministic repeat tests pass.
For a fixed `alpha=16 km` sinusoidal control, transfer-function error at
8/4/2 km is `0.001628 / 0.000410 / 0.000105`.

This is infrastructure, not permission to blur authoritative elevation or a
post-hoc cure for the invalid first smoke. The comparison must preregister the
filtered quantity, physical support and downstream semantics; unfiltered C0
remains the control arm.

## H. Separate coupled C0 solver — implemented, promotion evidence open

`C0LandscapeSolver` is a separate type family rather than a mode inside the
legacy path solver. Its transactional Lie-split trial performs:

1. midpoint vertical rock motion on scratch state;
2. depression-aware face-flow routing and water balance;
3. outgoing-flux-weighted physical face grade, including portal Dirichlet
   faces;
4. effective areal denudation using C0-V intensity;
5. conservative genuine-boundary hillslope transport;
6. elevation-volume-moment ledger closure and one committed revision.

Rejected trials leave state byte-unchanged and forcing callbacks are resampled
at the new midpoint. Portal cells are never pinned. Unit tests pass for zero and
uplift-only steps, direct isolated-operator equivalence, all-operator ledger
closure, adaptive denudation-depth and slope-Courant retries, forcing
resampling, below-base portal behavior and transactional failure.

The coupled coefficient defaults to zero: no dimensioned C0 response regime has
yet been accepted as a default, and the legacy Q-form `K=0.03` has incompatible
units. A first unseen smoke regime is preregistered separately:
`R=500`, `q0=50,000`, `S0=0.02`, `E0=0.1`, `m=n=1`, `K=1e-4`,
`D=0.1`, `Sc=0.7` in the documented km/Myr unit system. The solver also
enforces a routed-distance fluvial slope-change Courant limit in addition to
uplift/denudation-depth accuracy and hillslope stability limits.
It counts accumulated-flow cells whose reconstructed vector cancels instead of
substituting raw discharge or a floor.

Manufactured coupled tests use a separate test-only `K=1e-5` with `m=n=1`.
They pass bit determinism, water/solid closure, decreasing `dt/dt2/dt4` error,
and independent buffered mean, RMS, uplift and denudation improvement from
8→4→2 km against a 1 km reference. A depression-fill control carries water but
produces exactly zero denudation wherever the physical flow-aligned grade is
flat/uphill. This promotes the numerical composition gate, not the smoke regime.

### First unseen 0.1 Myr U/L smoke — fail/block

The frozen `m=n=1`, `K=1e-4` smoke was run without viewing/tuning at 8/4/2 km.

| Case | Spacing | Relief | Denudation export | Maximum E | Accepted dt minimum |
|---|---:|---:|---:|---:|---:|
| U | 8 km | 31.8 m | 44.28 km³ | 0.035 km/Myr | 0.010 Myr |
| U | 4 km | 34.1 m | 45.64 km³ | 0.085 km/Myr | 0.010 Myr |
| U | 2 km | 87.5 m | 56.38 km³ | 7.79 km/Myr | 0.00101 Myr |
| L | 8 km | 32.7 m | 44.53 km³ | 0.035 km/Myr | 0.010 Myr |
| L | 4 km | 34.7 m | 45.93 km³ | 0.112 km/Myr | 0.010 Myr |
| L | 2 km | 179.7 m | 59.93 km³ | 19.30 km/Myr | 0.000244 Myr |

All solid ledgers close near `1e-10 km³`, portal water balance closes, no sink
storage occurs and no reconstructed-q cancellation cell is reported. The 8 and
4 km responses are close; 2 km is qualitatively different and repeatedly hits
the denudation-depth limiter. This is not a bookkeeping failure and must not be
repaired by reducing `K` at 2 km.

Post-run cell diagnostics show this screen mixes a real support warning with
two earlier correctness gaps:

1. portal-adjacent cells can fall below base level while filled routing still
   sends water outward; the LS physical gradient omits the portal Dirichlet
   face and then interprets the deep cell against higher interior neighbors as
   outward-downhill, causing runaway incision;
2. the coupled solver exposes but does not enforce the fluvial slope Courant
   limit. At 2 km, `Kq` makes the requested `0.01 Myr` step several cells per
   step even before the worst feedback.

The L-2 km maximum lies 0.44 km inside the north portal and reaches
`E=19.3 km/Myr`. With a smaller requested step, both U and L eventually run
away in portal cells to roughly `-1.4 km`; smaller dt therefore does not repair
the boundary law. Independently, initial/interior maximum q and E approximately
double from 4→2 km at nearly unchanged available catchment supply, confirming a
real one-cell support-narrowing question remains after correctness is fixed.

Fix the face-consistent physical grade at open portals and implement the
fluvial slope CFL first, then rerun the same frozen 0.1 Myr screen. A fixed-
physical coarse-graining arm versus C1 remains the likely next representation
comparison only if that corrected rerun still fails. The 1 Myr smoke is stopped.

### Corrected 0.1 Myr U/L smoke — correctness pass, support question open

The exact frozen screen was rerun after replacing the LS-derived erosive grade
with outgoing-flux-weighted physical face grade (including portal faces) and
enforcing the slope Courant limit. New artifacts preserve the invalid run and
use the `c0-smoke-corrected-*` prefix.

| Case | Spacing | Relief | Denudation export | Maximum E | Accepted steps |
|---|---:|---:|---:|---:|---:|
| U | 8 km | 31.126 m | 38.105 km³ | 0.0339 km/Myr | 13 |
| U | 4 km | 31.252 m | 38.422 km³ | 0.0679 km/Myr | 53 |
| U | 2 km | 31.284 m | 39.212 km³ | 0.1478 km/Myr | 253 |
| L | 8 km | 31.910 m | 38.317 km³ | 0.0284 km/Myr | 14 |
| L | 4 km | 31.976 m | 38.584 km³ | 0.0601 km/Myr | 58 |
| L | 2 km | 32.013 m | 39.333 km³ | 0.1456 km/Myr | 250 |

The catastrophic 2 km boundary feedback is gone. Relief changes by at most
`0.40%` per refinement and all ledgers close within `1.9e-10 km³`; portal water
is stable, with zero sink storage and zero unresolved-q cells. This promotes
the boundary-grade and timestep correctness repair.

It does **not** promote unregularized C0 as the eventual landscape
representation. Peak cellwise denudation grows about `4.4–5.1×` from 8 to
2 km, the 4→2 export drift (`1.9–2.1%`) is larger than the 8→4 drift
(`0.7–0.8%`), and the governing timestep scales approximately with `h²`.
Those are the clean signature of a one-cell drainage support narrowing under
refinement. Integrated relief is currently robust; local intensity, derivative
consumers and compute cost are not. A longer run would amplify runtime without
answering the representation question, so 1 Myr remains stopped.

## Disposition

- **Pass:** P/pathway analytic control; genuine full-hex boundary geometry;
  stable semantic portals and conservative portal outflow; interior unfilled
  MFD conservation and plane/ridge specific-discharge convergence; isolated C0
  denudation and solid ledger.
- **Pass:** the exact linear Dirichlet fixture recovers affine boundary flux at
  8/4/2 km and closes finite-volume storage plus export.
- **Truth limit:** `|q_vector|` is not pointwise throughput at an unresolved
  confluence or line sink. Keep it as the consistent local continuum intensity
  on resolved smooth fields; use physical cut flux for singular topology.
- **Pass:** resolved downstream-reach local intensity and integrated cut flux;
  no RMS/L1/cell-width fallback is added.
- **Pass:** coupled transactional composition and per-step ledgers in unit
  controls.
- **Pass:** manufactured spatial/temporal convergence under a test-only direct-q
  regime and the depression/uphill-grade control.
- **Invalid evidence retained:** the first unfiltered 0.1 Myr U/L screen mixed
  an open-base physical-grade defect with an absent slope CFL.
- **Pass:** face-consistent portal grade, slope-Courant control and the exact
  corrected short rerun; the prior catastrophic refinement feedback is gone.
- **Open:** unregularized C0 has convergent short-run relief but narrowing local
  support, non-asymptotic export and `h²` timestep cost. Compare one fixed-
  physical-support C0 arm against explicit C1 requirements before longer runs.
- **Blocked:** 1/10 Myr U/L, Slice 2 objects/controls, product integration and
  morphology tuning until that representation comparison is resolved.
- **Do not:** feed the new `q` field into the legacy path incision coefficient;
  interpret closed sink storage as modeled lakes; attach nominal rectangular
  faces to full cells; tune MFD exponent or denudation parameters from visual
  terrain before the remaining analytic contracts pass.

The next gate is a representation comparison, not parameter tuning: retain the
corrected unfiltered C0 result as the baseline, evaluate one preregistered fixed-
physical-support C0 arm, and state the minimum C1 channel state needed by future
sediment, valley and ecology consumers. Do not rescale `K` with grid spacing,
score pointwise intensity on an exact sink or add a local RMS/L1 fallback.
