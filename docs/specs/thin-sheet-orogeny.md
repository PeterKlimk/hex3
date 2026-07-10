# Thin-sheet orogeny

Status: T0 research prototype implemented, not a product replacement, 2026-07-11.

## Problem

The legacy coarse model projects independently capped arc and collision response
fields into elevation. It preserves visually useful belt structure but has no
shared shortening-volume or rheological constraint. Commit `37f4cb5` replaced
that with a conserved boundary-volume source followed by isotropic scalar
diffusion. Seed-12345 A/B diagnostics show that this removes too much organized
terrain while only modestly improving the pillar profile.

The first causal ladder isolates two failures:

1. The chosen kinematic episode supplies far less integrated crust than the
   legacy response implied. Before relaxation, the area with >=0.2/0.5/1.0 km
   isostatic tectonic support falls from 100/88/67% to 21/15/11% of the fixed
   arc+collision process footprint.
2. Scalar isotropic redistribution has no representation of plate velocity,
   stress, strain direction, connected deformation fronts, or yield. It can
   only trade a sharp blob for a broad blob.

The product default remains `legacy` until the replacement passes physical,
structural, and visual gates.

The runtime model ladder is `legacy`, `legacy-yield`, `conserved-local`,
`conserved-feature-footprint`, `conserved-isotropic`, and `thin-sheet`. The
conserved-* bridges are deliberately causal experiments, not candidate terrain
styles; `legacy-yield` (below) is the one CANDIDATE rung.

### Legacy byte-identity (2026-07-11)

The walk-back initially left the legacy path off by float round-trips
(`(arc+col)·inv_slope` for the historical `/slope`; thickness assembled as
`(base+rift).max(0.05)+tectonic` for the historical
`(base+thickening+rift).max(0.05)`; demotion as `λ·slope·thickening` for the
historical `λ·(arc+col)`). Coarse deltas were ~1e-7 but the fine/erosion chaos
cascade amplified them to ±2.6 km at orogen peaks — silently invalidating every
measured A4/meso baseline. All three expressions now reproduce the historical
float ops exactly in the `Legacy` arm; verified BYTE-IDENTICAL (all shared
export fields, seed 4242, cells 20k, fine-scale 4, no cache) against the
pre-ladder pipeline (64ae119). Lesson: "semantically legacy" is not legacy —
identity claims are verified by export diff, never by reading the code.

### legacy-yield — the candidate middle rung (2026-07-11)

Root cause of the seed-12345 mesa/pillar (measured): a MICROPLATE with
convergence on all sides carries arc≈0.27 AND collision≈0.26 simultaneously
(the legacy response stacks them, no strength limit) → compact cliff-edged 6-km
coarse block → the fine emergent shape stack concentrates the rebuilt volume
into its peaks (8.9 km at baseline, 11.8 composed). Spire-probe signature:
ring-p90 drop flat out to ~100 km then 3-6 km cliff at 100-250 km (Earth tapers
monotonically; Everest ≈ 2→4 km across the same rings).

`legacy-yield` = the EXACT legacy source, then `deformation::yield_relax` on
the tectonic thickening: material above `OROGEN_YIELD_ELEV` (0.35 ≈ 3.5 km of
orogenic contribution; strength/gravity playground dial) spreads conservatively
over `OROGEN_YIELD_SPREAD_KM` (250) with Picard re-yielding. Sub-yield cells
keep their bit-exact legacy thickening — this rung CANNOT lose belt detail (the
measured failure of the conserved/isotropic experiments); it can only cap
over-strength compact loads and build their foothill aprons from the shed
volume. The taper is emergent, not painted. Physically this is the
gravitational half of T1 applied to the legacy source instead of the T0 flux.

Gate results (2026-07-11, s50 g1, seeds 12345/777; 25-km p95-p05 p50 @ max
peak, spire = worst 100-250 ring-p90 drop):

| config              | 12345                   | 777                    |
|---------------------|-------------------------|------------------------|
| legacy baseline     | 193 @ 8.9, spire 3.6    | 158 @ 8.3              |
| yield baseline      | 182 @ 6.7, spire 1.3 ok | 154 @ 6.6 ok           |
| legacy + A4 composed| 330 @ 11.8, spire 6.3   | 249 @ 10.5             |
| yield + A4 composed | **315 @ 8.7, spire ok** | **247 @ 8.7, spire ok**|

The pillar's flat-then-cliff profile is GONE (1.3 km ring-250 drop, Earth-
normal taper); structure metrics are unchanged from legacy (elongation p50
5.0/4.6, trunk grammar 24/32/44 vs 23/33/44, mountain land 8.5→10.5%, spacing
CV in band) — the sub-yield-untouched construction did what it promised. The
composed A4 candidate keeps ~95-99% of its relief while peaks drop 11.8/10.5 →
8.7/8.7 (right at Earth max): the yield mechanism self-regulates the peak
budget the shape stack kept violating, and REOPENS dial headroom (deeper
meso/pulse now plausible-by-construction). Costs: baseline peaks land 6.6-6.7
(a touch under Earth's 8.8 — raise OROGEN_YIELD_ELEV if summits read low), and
baseline 25-km relief −3..−6% (the mesa's own samples leaving the pool).
Status: numbers green both seeds; USER VISUAL pending (sweep knob
`orogen_model` 0=legacy 1=legacy-yield for the A/B montage).

## Principle

Treat continental crust as a deformable thin sheet riding on plate motion.
Solve horizontal deformational velocity first; derive thickening from mass
continuity. Topography is then a consequence of thickness and isostasy.

For sheet thickness `H` and horizontal velocity `u`:

```text
div(sigma(u, H)) - grad(P_g(H)) + basal_coupling(u_plate - u) = 0
dH/dt + div(H u) = S_magma
```

`sigma` is viscoplastic. `P_g` is gravitational potential pressure. The second
equation is conservative: collision redistributes existing continental crust;
only retained arc magma is a positive material source.

## State

Coarse per-cell state:

- crust thickness;
- horizontal deformational velocity (tangent vector);
- accumulated plastic strain invariant;
- principal compression direction (tangent axis);
- shortening/thinning rate;
- magmatic addition rate;
- isostatic uplift rate.

Arc/collision labels remain process diagnostics. They do not prescribe height
or deformation width.

## Boundary conditions

- Plate-interior target velocity comes from its Euler pole.
- Connected continental crust is one deformable domain across collisional
  sutures; plate identity is a velocity boundary condition, not a no-flow wall.
- At a subduction margin, overriding-side coupling transmits a fraction of
  closing motion into the overriding sheet. Subducting oceanic crust does not
  become continental shortening volume.
- Retained magma enters `S_magma` on the overriding side.
- Crust-type coastlines are no-flux boundaries for continental material unless
  an explicit accretion/extension process crosses them.

## Discretization ladder

### T0: linear velocity sheet

Minimize viscous dissipation plus basal departure from rigid plate velocity on
the Voronoi graph. Solve the two tangent velocity components with a symmetric
positive-definite screened operator. Compute conservative face fluxes of `H u`
and update thickness with CFL-derived substeps.

This is the first implementation target. Unlike scalar height diffusion, it
preserves convergence direction and obtains thickening from velocity divergence.

The current T0 implementation is intentionally narrower than the final boundary
conditions above. It solves deformational velocity anomalies independently
inside each plate/crust block, represents suture traction with boundary target
velocities, and prohibits cross-suture material flux. That gives a conservative
test of the velocity/continuity handoff without claiming that colliding
continental sheets are already one coupled domain.

### T1: viscoplastic yield

Picard-iterate effective viscosity from the previous strain rate. Below yield,
the sheet follows rigid plate motion; above yield, viscosity drops and strain
localizes. Gravitational pressure increases with excess thickness, causing tall
loads to spread preferentially.

The prototype currently implements only the gravitational half of this rung:
excess thickness above a strength/gravity yield threshold is relaxed with
conservative Picard steps. Strain-dependent effective viscosity and localization
in the horizontal velocity solve remain future work.

### T2: elastic/flexural response

Apply an optional lithospheric-flexure solve to the final crustal load. This
creates foreland response but never determines shortening volume.

## Worldbuilding axes

- tectonic episode duration / accumulated displacement;
- basal coupling versus sheet viscosity (stress-transmission length);
- yield strength relative to gravitational pressure;
- strain-rate exponent;
- subduction coupling;
- retained magmatic fraction;
- optional effective elastic thickness.

Earth-like defaults are one vector. Extreme values deliberately allow stagnant
rigid lids, diffuse plateau worlds, hyper-mobile crust, or magmatically dominated
arc planets. No parameter directly specifies mountain height or Gaussian width.

## Fine-mesh coupling

Transfer strain invariant, principal compression direction, and uplift rate.

- Structural grain eligibility follows plastic strain, not absolute elevation.
- Grain strike follows principal compression/extension axes.
- Emergent uplift volume follows the coarse thickness-change budget.
- Submarine and low-elevation active structures remain eligible.
- Erosion organizes supplied structure but does not decide whether tectonic
  structure exists.

## Evaluation

All models must run through a common fine tessellation or be resampled onto a
fixed evaluation mesh. Export coarse envelope, fine base, pre-erosion surface,
eroded surface, tectonic thickness, structural delta, and erosion delta.

Required gates:

- area-weighted crust volume conservation;
- resolution-independent integrated shortening;
- support/detail survival across the complete process footprint;
- connected relief-network length, branching, elongation, and cross-range ridge
  count;
- multiscale relief energy and screen-space normal variation;
- seed-12345 pillar gate without regression in mountain coverage or belt
  organization;
- fixed-camera Windows visual A/B approval across several seeds.

## Current result

The diagnostic and state plumbing is usable, but the prototype does not pass the
terrain gate. On seed 12345, increasing the episode duration enough to restore
mountain area creates a broad, low sheet (about 28% mountain land, median belt
elongation 2.6, and roughly 2,100 km crest spacing in the tested run). The shorter
episode preserves more belt elongation but remains underpowered. Accordingly,
`legacy` is still the exact default and `thin-sheet` is an explicit experiment.
