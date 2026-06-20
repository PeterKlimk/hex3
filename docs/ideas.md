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

**Diagnose the failure first (which "generic"?).** A skeptical review (codex,
gpt-5.5, June 2026) pointed out the symptom splits three ways with different
fixes, so confirm which before committing:
- textural sameness / no lineation, uniform drainage → **A2** (+ A1 for belts)
- rounded, un-alpine crests / smooth peaks → **B (glacial)**; K/lithology can't
  fix this, only glacial gives cirques/arêtes/U-valleys
- blobby at the range scale (smooth fronts, no abrupt basins/facets) → **A3 /
  macro uplift**, untouched by erodibility fields

**Ordering update (from that review): A2 before A1.** A general erodibility
field improves many terrain types and avoids fake repetition; A1 is a *targeted*
specialization (A1 = A2 with an oriented periodic pattern), not the generic lead.

### A1. Structural grain / fold fabric  — targeted mode (was "lead")
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
  Erosion carves weak bands → strike ridges. (Earlier claim "trellis drainage
  emerges" is DOWNGRADED per review: K changes incision *rate*, not geometry;
  detachment-limited vertical incision has no planation/capture/migration, so
  trellis is an outcome to *test*, not a promise — needs K-contrast to beat the
  slope term + relief organized across stripes.) A1 = A2 with an oriented pattern.
- **Pair K with diffusivity (and maybe a little structural relief).** `K(x)`
  alone may barely show — equal-elevation bands routing the same differentiate
  only slowly. Vary hillslope `D(x)` too (resistant rock resists rounding, else
  ridgelines smooth back to sameness), and/or seed small structural relief so
  bands start differentiated. Review's phrasing: "layered competence fabric,"
  not "fold fabric."
- **Distance-from-boundary caveats:** medial-axis artifacts where the nearest
  boundary segment flips; ignores oblique/arcuate/transpressional strike. Use A1
  only on belts where belt-parallel grain is defensible, not every orogen.
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

## A'. Missing couplings (the "systems aren't talking yet" cluster)

Reframing from the review (June 2026): some "mountains/surroundings look
generic" symptoms aren't a missing *field* but a missing *interaction* between
systems we already have. "Not Earth-accurate" is the wrong dismissal — the
absence flags an emergent coupling worth having (see design-philosophy memory).

### A'1. Flexure ⇄ erosion ⇄ deposition → foreland basins
- **Symptom:** orogens have no adjacent lowlands/sediment wedges; ranges sit on
  smooth ground instead of beside basins fed by their own erosion.
- **Coupling:** orogen load → flexural subsidence of the neighbouring plate →
  basin → traps sediment eroded off the orogen.
- **Ingredients (all present, uncoupled):** flexure (currently trench-only —
  `flexure_broken/coupled`), erosion sediment volume, deposition (currently
  coastal/per-mouth only). Extend flexure to orogen loads; let deposition fill
  the flexural low. Effort: medium. (Promote? ties to `docs/specs/flexure.md`.)

### A'2. Tectonics ⇄ erosion over time → water gaps, antecedence, capture
- **Symptom:** no transverse rivers cutting rising ridges, no water/wind gaps,
  no drainage capture — rivers have no *history* relative to the uplift.
- **Coupling:** drainage established, THEN uplift rises through it (antecedence);
  networks compete and capture as divides migrate. Needs erosion running *during*
  uplift, not once after — the time-evolution thread below. This is its marquee
  payoff, not abstract "time stepping." Effort: large.

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

## Tooling
- **Staged snapshots + steppable erosion** → promoted to `docs/specs/staging.md`.
  Object-level structural sharing (Arc copy-on-write) so stages are navigable
  without recompute; stage 3 split into 3a `FineBase` / 3b erosion / 3c
  hydrology so A/B + re-run share the expensive mesh; resumable `ErosionState`
  to watch carving (arrow keys, rough); disk-cached `FineBase` to survive
  rebuilds. The substrate the time-evolution thread (A′2) will reuse. Needed
  because before/after on stage-3 work is otherwise too slow to see.

## Performance / algorithmic optimization notes (June 2026)

These are category-2 candidates: likely bigger than allocation cleanup, but not
same-output by default unless the cache/invalidation boundaries are exact. Keep
same-output work separate from parameter tuning and intentional output changes.

### Reuse hydrology/erosion graph scaffolding
- **Observation:** hydrology and erosion repeatedly derive graph views from the
  same primitives: `drainage_dir`, upstream contributors, topological order,
  flow accumulation, basin/outflow traversals.
- **Candidate:** materialize reusable structures for the active drainage graph:
  compact `drains_into`, a topological processing order, and maybe common
  per-cell flags/masks used by rivers, erosion, and basin analysis.
- **Why:** each O(N) rebuild plus allocation is tolerable on coarse meshes but
  expensive on the adaptive fine mesh; erosion can repeat this work whenever
  routing is recomputed.
- **Same-output path:** cache only pure derived data keyed by the exact
  `drainage_dir`/active stage. Invalidate whenever elevation/routing changes.
- **Risk:** stale graph caches are subtle and wrong; make invalidation explicit.

### Share render topology and index construction
- **Observation:** colored terrain, unified relief, elevation-map, and edge
  buffers all walk the same Voronoi topology and fan-triangulate cells in similar
  ways. Fine meshes already use shared vertices for some paths, while coarse
  modes duplicate per-cell vertices for sharp coloring/wrap behaviour.
- **Candidate:** introduce a reusable render-topology layer per tessellation:
  shared triangle indices, per-cell triangle ranges, unique edge index list, and
  maybe antimeridian/wrap metadata for map mode. Attribute builders would fill
  colors/elevations/materials without rediscovering topology.
- **Why:** large fine-mesh index buffers and topology walks are deterministic
  work currently repeated across buffer rebuilds and mode-specific meshes.
- **Same-output path:** start with shared-vertex fine surfaces where topology is
  already identical. Keep a separate duplicated-vertex topology for coarse/map
  paths so sharp per-cell colours and wrap handling remain unchanged.
- **Risk:** forcing all modes through one mesh abstraction would blur legitimate
  differences between coarse cell-flat rendering and fine shared rendering.

### Fine mesh sampling/relaxation and erosion reroute cadence
- **Observation:** stage 3/4 cost is dominated by how many fine cells are made,
  how they are sampled/relaxed, Voronoi construction, and how often erosion
  recomputes routing over the fine graph.
- **Candidate:** investigate adaptive relaxation only where density gradients or
  quality metrics need it; reuse nearest-neighbour/kd structures across relax
  passes where valid; recompute erosion routing less often or only where
  elevation changed enough; use coarse-to-fine hydrology hints.
- **Why:** this attacks the high-order cost drivers rather than just removing
  duplicate allocations.
- **Same-output path:** limited. Reusing an identical nearest-neighbour
  structure inside an exact algorithm may qualify, but changing relaxation,
  route cadence, or adaptive thresholds changes the generated world and belongs
  in tuning/algorithm experiments.
- **Risk:** these may improve speed while changing terrain character; evaluate
  with visual A/B and diagnostics, not just wall-clock time.

## Deferred / known
- **Rivers:** rendering rework (huge blocks don't suit high-density cells) +
  possibly a deeper rework. Out of scope for erosion-as-written.
- **s2-voronoi perf:** compute (r=2 cold path on adaptive density) is being
  fixed upstream; fine-mesh relaxation can then reuse a bare cube-grid kNN
  (see the note by `relax_fine_points`). Re-profile when it lands.
