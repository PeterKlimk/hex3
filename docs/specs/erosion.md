# Spec: Fluvial Erosion on the Fine Mesh

Part (b) of the erosion project; requires `docs/specs/fine-mesh.md`.
Loose spec: the formulas and invariants below are the contract; layout is
the implementer's call. Leave `// SPEC:` comments for genuine ambiguities.
No shape-fixing hacks; constants are knobs with stated defaults, not values
to tune against targets.

## Goal

Replace painted mountain texture with the real thing: rivers that carved
their valleys. The erosion loop runs on the fine mesh between the fine
elevation step (`Elevation::refine_from_base`) and final hydrology,
modifying CRUST THICKNESS (not elevation directly) so isostasy responds and
the state composes with future time evolution.

NOTE (changed since this spec was drafted): the fine mesh no longer
recomputes elevation from crust thickness — `refine_from_base` interpolates
the already-solved coarse elevation onto the fine cells (the "base"). So
couple thickness to elevation as an isostatic DELTA on top of that base:

    elev_i = base_i + ( isostatic_elevation(thickness_i)
                        - isostatic_elevation(thickness_initial_i) )

The working thickness is initialized from the transferred coarse crust
thickness (already on each fine cell in `fields.elevation_fields.crust_thickness`,
currently only consumed by export). At t=0 the delta is zero (elev = base);
incision (thinning) lowers the surface with the rebound the isostatic
relation provides; uplift (thickening) raises it. This preserves the base's
fixed sea-level datum and the thickness-based composability the spec wants.

## Model

Detachment-limited stream power with hillslope diffusion, plus simple
sink/coastal deposition. Per timestep, on land cells:

1. **Route**: steepest-descent receiver per cell (existing hydrology
   pattern). Resolve pits first (see Lakes below).
2. **Accumulate**: precipitation-weighted drainage area A per cell
   (existing topological-order pattern), area-weighted so A has units of
   "wet area", making wet ranges dissect more finely than desert ranges.
3. **Incise** (Braun & Willett 2013 implicit scheme, n = 1 so each cell is
   a linear solve given its receiver): processing cells in downstream-first
   order,

       h_i = (h_i + dt * K * A_i^m * h_rcv / d_i) / (1 + dt * K * A_i^m / d_i)

   with d_i the distance to the receiver and m = EROSION_M (default 0.5).
   This is unconditionally stable: ~EROSION_STEPS (default 40) large
   timesteps suffice. K = EROSION_K.
4. **Diffuse**: linear hillslope diffusion (soil creep) with diffusivity
   EROSION_DIFFUSIVITY; explicit with a CFL-safe substep or implicit,
   implementer's choice.
5. **Uplift source**: add U_i * dt to thickness, where U_i is the tectonic
   uplift rate derived from the transferred TECTONIC forcing fields in
   `fields.elevation_fields` (convergent/collision + arc positive, divergent/
   rift negative; `trench`/`ridge` available too) — NOT `fields.uplift`, which
   is atmospheric uplift (name collision). Scaled by EROSION_UPLIFT_SCALE.
   Active orogens approach
   a U/K equilibrium (concave river profiles, ridge-valley relief);
   inactive terrain just decays. This is the mechanism that makes mountain
   character emerge — do not substitute noise.
6. **Deposit**: track eroded volume routed downstream; where flow enters a
   lake, closed pit, or the ocean, deposit it there (raise thickness,
   spreading over the sink's low cells) up to a fill limit. Deltas at
   mouths and filled basin floors are the desired emergent result. Full
   transport-limited sediment routing is out of scope.

All of 1-6 modify the working thickness field; elevation for slope
computation is re-derived each step via the isostatic-delta coupling above
(or an equivalent incremental update — state the choice in a comment). Sea
level stays fixed (inherited via `base_i` from the coarse solve; not
re-solved). No erosion below sea level; submarine change comes only from
deposition (6).

## Lakes / pits during the loop

Pits and flat areas must not stall routing: use priority-flood to assign
receivers across them each step (carve-or-fill, implementer's choice —
state it). Real terminal lakes are fine; the final lake set is computed by
the existing hydrology after the loop ends.

## Performance

Route + accumulate + incise measured at ~0.3s/step single-threaded at 2.5M
(bench_mesh). Route, incise, diffuse, and uplift are per-cell/per-edge and
trivially parallel; the budget for the whole loop is ~10-15s at defaults.

## Validation

- Maps: render fine elevation before/after; valleys must follow the river
  network (flow accumulation overlay), drainage density visibly higher in
  wet regions than arid ones at similar uplift.
- River long-profiles (sample a few major rivers mouth-to-source from the
  drainage graph): concave-up in active orogens.
- Hypsometry stays bimodal; land fraction drift from deposition reported
  (small drift is physical, sea level is not re-solved).
- Mass sanity: total eroded volume ~= total deposited volume + (lost to
  ocean sinks), logged per run.
- Moran's I on the final fine elevation stays high (erosion produces
  coherent valleys, not speckle).
- `cargo test` green; unit-test the implicit incision step against a
  1-D analytic steady state (h' = (U/(K A^m))^(1/n) slope at equilibrium).

## Non-goals

Glacial/coastal/aeolian processes, full sediment transport, stratigraphy,
submarine erosion, time-evolution coupling (erosion runs once after
tectonics for now), GPU compute. Retuning unrelated constants.
