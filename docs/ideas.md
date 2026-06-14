# Idea Backlog (post-roadmap)

Living list of mechanism ideas after the `physically-inspired-roadmap.md` items
landed (crust-thickness isostasy, trench flexure, overturning circulation,
fluvial erosion + fine mesh). Same philosophy: *unreasonably physically
inspired*, not Earth simulation — mechanisms are physical because that makes
worlds coherent; parameters stay playground knobs. Same pattern to exploit:
**the physical quantity is often already computed, just not used.**

These are sketches to not-forget, not specs. Promote to `docs/specs/` when one
is chosen.

**Noise — input vs output.** Noise is allowed; the distinction is *how*. Noise as
the *output* (noise being the terrain/field, e.g. the retired hills/ridges fBm)
is what a mechanism replaces. Noise as an *input/modulator to a physical
skeleton* is legitimate and often the right call — it supplies irregularity a
deterministic model can't and stands in for detail below the scale the mechanism
sets. Rule of thumb: skeleton physical (orientation, location, the simulated
process), texture noise-fed. Keep things less fake where it makes sense, but
don't invent mechanics you don't need just to avoid noise.

## Central insight (June 2026)

Erosion is validated (river concavity θ≈0.5, climate-responsive drainage
density) but mountains still look "off" in a way *tuning can't fix*. Diagnosis:
erosion carves a crust with **no internal structure** — smooth interpolated
uplift, uniform erodibility (single `K`), a scalar thickness field — so it can
only produce generic isotropic dendritic terrain (dendritic valleys + diffusion-
smoothed ridges). Real mountains get their character from *geologic structure
that erosion exploits*. So the next frontier is **giving the lithosphere
internal structure for erosion to express**, plus glacial for high elevations.

## A. Lithosphere internal structure (the "ranges look generic" cluster)

### A1. Structural grain / fold fabric  — lead candidate
- **Symptom:** ranges are isotropic dendritic blobs; real fold-thrust belts are
  strongly *lineated* — strike-parallel ridge-and-valley (Valley-and-Ridge
  Appalachians, Zagros).
- **Mechanism:** compression folds/thrusts crust, with fold axes ⟂ to the
  shortening direction (= parallel to the belt trend). Erosion then dissects the
  folded competence layers into strike-parallel ridges.
- **Approach (current lead):** structure the *input*, not the erosion — make
  `EROSION_K` a striped erodibility field, leave the validated isotropic stream-
  power untouched. Stripe phase = distance-from-convergent-boundary /
  fold_wavelength, so stripes are the distance field's iso-contours → belt-
  parallel grain that curves for free, no orientation field to construct.
  Erosion carves weak bands → strike ridges, and **trellis drainage emerges**
  (the other fold-belt signature). A1 and A2 are the same machinery — A1 is A2
  with an oriented periodic pattern.
- **Skeleton physical, texture noise-fed:** orientation/location/erosion are
  deterministic; noise feeds wavelength jitter + phase breaks/terminations/
  en-echelon so it's a quasi-periodic belt, not a corduroy comb. (See the noise
  principle above.) Optional later: sawtooth stripe for thrust vergence/facets.
- **Already computed:** distance-from-boundary (`collision_distance`/
  `arc_distance` in `features.rs`); `convergent`/`collision` influence to gate
  amplitude. Would transfer a distance field to the fine mesh + make `K` per-cell.
- **Resolution fit:** fold wavelength ~5–20 km on Earth = fine-mesh scale
  (cells 1.5–12 km), so this belongs in the fine/erosion stage, not the coarse.
- **Open Qs:** striped-K vs corrugated-thickness vs both; tie wavelength to
  crustal thickness (physical: ∝ competent-layer thickness) or knob+noise;
  extensional grain too (same machinery, gated by `divergent` → basin-and-range).
- **Effort:** medium.

### A2. Lithological contrast / erodibility field
- **Symptom:** uniform `K` → uniform smoothness; no escarpments, hogbacks,
  resistant ridges, differential relief.
- **Mechanism:** spatial rock-resistance field; differential erosion. Make `K`
  (and maybe diffusivity) a field, not a constant.
- **Already computed (partial):** craton age & structure (`crust.rs` cratons),
  continentality, arc (volcanic), ridge-age. Map these → an erodibility field.
- **Effort:** low–medium (mostly a field + `K(x)` in the erosion loop).

### A3. Discrete faulting / range-front facets
- **Symptom:** ranges have smooth fronts; real ranges are fault-bounded
  (triangular facets, steep fronts, abrupt basins).
- **Mechanism:** discrete normal/thrust fault traces offsetting elevation.
- **Already computed:** least of the three — needs a fault-trace generator
  (seed along boundary stress / convergent field).
- **Effort:** medium–high.

## B. Glacial erosion  — biggest pure-visual mountain win
- **Symptom:** fluvial + hillslope diffusion → rounded crests; no alpine
  character (cirques, arêtes, U-valleys, hanging valleys, sharp peaks). Likely a
  big part of "doesn't look perfect."
- **Mechanism:** above the equilibrium-line altitude (ELA), glacial erosion
  dominates — headward cirque erosion, valley widening/over-deepening (U-shape).
  A sibling process to fluvial in (or after) the erosion loop.
- **Already computed:** ELA from temperature + elevation (+ precip for
  accumulation); both fields live on the fine mesh.
- **Note:** explicit non-goal in `erosion.md`; would be the first extension.
- **Effort:** medium–high.

## C. Broader threads (not mountain-specific)
- **Crust / terrane model:** crust is just continental/oceanic + craton growth;
  real crust is accreted terranes with provenance/age structure — feeds A2.
- **Plate kinematics from forces:** Euler poles are RNG; least-squares fit to
  slab-pull + ridge-push (roadmap #4, deferred to the time-evolution project).
- **Time evolution:** thickness is the state variable; collision *and* erosion
  both modify it. Run tectonics+erosion as a coupled time loop (erosion runs
  once today). Big project; the thickness-based design was chosen to enable it.
- **Zonally-asymmetric circulation:** ocean gyres / Walker cells (roadmap
  "future craziness") — a longitude-dependent layer on top of the zonal-mean Ψ.

## Deferred / known
- **Rivers:** rendering rework (huge blocks don't suit high-density cells) +
  possibly a deeper rework. Out of scope for erosion-as-written.
- **s2-voronoi perf:** compute (r=2 cold path on adaptive density) is being
  fixed upstream; fine-mesh relaxation can then reuse a bare cube-grid kNN
  (see the note by `relax_fine_points`). Re-profile when it lands.
