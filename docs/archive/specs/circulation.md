> **Archived:** Historical design or experiment record. It does not define current architecture, defaults, or priorities. Start at [`docs/README.md`](../../README.md).

# Spec: Meridional Overturning Circulation

Roadmap item #3 (`docs/physically-inspired-roadmap.md`). Replaces three
independently prescribed latitude functions — the zonal wind bands
(`zonal_wind()` in `atmosphere.rs`), the desert belt, and the ITCZ band
(`latitude_rain_factor()` in `moisture.rs`) — with derivations from a single
prescribed meridional overturning streamfunction. This spec is intentionally
looser than `flexure.md`: it states the mechanism, formulas, and invariants;
implementation layout is the implementer's call. Where a genuine design
question arises, leave a `// SPEC:` comment rather than resolving it ad hoc.
Do not add smoothing passes or shape-fixing hacks; do not tune unrelated
constants.

## Mechanism

Prescribe Psi(phi): the northward mass transport of the *surface* branch of
the overturning circulation, as a function of latitude phi. Everything else
derives:

1. **Surface meridional wind**: v(phi) ∝ Psi(phi) / cos(phi)
   (normalization knob `CIRC_MERIDIONAL_SCALE`).
2. **Surface zonal wind** from Coriolis turning under linear surface
   friction: u(phi) = (f / eps) * v(phi), with f = 2 * OMEGA * sin(phi) and
   friction knob eps (`CIRC_FRICTION`). The trade/westerly/polar-easterly
   band structure must EMERGE from Psi's sign pattern — `zonal_wind()` and
   its hand-shaped strength curves are deleted.
3. **Vertical motion**: w(phi) ∝ -dPsi/dphi / cos(phi) (analytic
   derivative of the Psi parameterization, not a numerical mesh
   derivative). Ascent (w > 0) where the surface flow converges, subsidence
   (w < 0) where it diverges.
4. **Upper wind** (visualization-quality only): meridional component is the
   return flow -v; zonal component from angular-momentum conservation for a
   parcel ascending at each cell's ascent latitude phi_a:
   u_up(phi) = OMEGA_SURFACE_SPEED * (cos^2(phi_a) - cos^2(phi)) / cos(phi),
   magnitude-clamped by a knob. This yields westerly jets at cell edges for
   the particle view. Replaces however the upper wind is currently produced.

## Psi parameterization

Piecewise cells. Each cell spans [phi_k, phi_k+1] with amplitude A_k and
sign s_k in {-1, +1}; within a cell

    Psi(phi) = s_k * A_k * sin(pi * (phi - phi_k) / (phi_k+1 - phi_k))

so Psi vanishes at every cell edge (v continuous through zero, w peaks at
edges and interior ascent/descent comes out of the derivative). Sign
convention check, northern hemisphere: Hadley surface flow is equatorward,
so s_Hadley makes Psi < 0 on (0, phi_H); this must give ascent (w > 0) at
the equator and subsidence at phi_H. Southern hemisphere mirrors.

Cells are configured per hemisphere (count, edge latitudes, amplitudes) so
asymmetric worlds are expressible. The default constructor builds an
Earth-like 3-cell hemisphere from a single rotation knob
`PLANET_ROTATION_RATE` (1.0 = Earth):

- Hadley edge phi_H = 30 deg / sqrt(PLANET_ROTATION_RATE), clamped to
  [8, 70] deg; Ferrel edge at min(2 * phi_H, 80 deg); polar cell to the
  pole. If phi_H clamps high enough that cells collapse, degrade gracefully
  to fewer cells (a slow rotator legitimately has one giant cell per
  hemisphere).
- Amplitudes: Hadley strongest, Ferrel ~half, polar ~quarter (knobs).

OMEGA in the Coriolis formula is proportional to PLANET_ROTATION_RATE, so
one knob coherently moves cell widths, turning strength, and jet positions.

## Integration (decisions already made — do not revisit)

- **w feeds the uplift field, not a rain multiplier.** In
  `compute_uplift()`, the large-scale w joins phi-convergence and
  orographic uplift as a third component with its own weight
  (`UPLIFT_CIRCULATION_WEIGHT`). The combined uplift field becomes SIGNED
  (subsidence negative): keep normalizing the two positive components as
  today, then add the (independently normalized, signed) circulation term.
  `latitude_rain_factor()` and the `DESERT_BELT_*` / `ITCZ_*` constants are
  deleted; rainout in `moisture.rs` consumes signed uplift, clamping each
  rain rate at >= 0 so subsidence suppresses (including orographic rain)
  but never produces negative rain. Audit uplift consumers for sign
  assumptions (climate Uplift view should become a diverging colormap;
  export passes through).
- **v and u replace the zonal background flow** in
  `generate_initial_wind()`; the pressure-gradient/geostrophic component,
  terrain effects, and the divergence-free projection are untouched — they
  remain the regional layer on top of the mean circulation. (The projection
  will remove the mean flow's divergence from the *wind field*, which is
  fine: rain takes its large-scale signal from the analytic w, not from the
  projected wind.)
- Temperature generation is untouched.

## Validation

`cargo test` (field-smoothness guards must pass), then headless export on
seeds 12345 and at least one other; render wind and precipitation maps
(`scripts/render_map.py`) and check with eyes:

- three alternating zonal wind bands per hemisphere at default knobs, with
  band edges at the configured cell edges (not at hard-coded 30/60);
- an equatorial rain band and subtropical dry belts that line up with the
  derived w, not with any latitude constant;
- diagnose climate line: land arid fraction within +/- 10 percentage points
  of its pre-change value on seed 12345 (we retune later; the mechanism
  just must not collapse the rain budget);
- a one-cell world (PLANET_ROTATION_RATE small enough to collapse cells)
  generates without panics and shows a single wet band / dry pole
  structure.

Unit-test the analytic pieces: Psi continuity at cell edges, the sign
convention (equatorial ascent for an Earth-like default), u band signs, and
w integrating to ~zero over the sphere.

## Non-goals

Ocean currents / gyres ("Gulf Stream"), Walker-type zonally-asymmetric
cells, seasons, heat advection by the circulation, any change to the
projection or pressure machinery. Final tuning of rain/aridity balance is
ours, post-review.
