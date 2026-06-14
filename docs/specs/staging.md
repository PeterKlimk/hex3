# Spec: Stage Navigation + Steppable Erosion (before/after tooling)

Loose spec: the model and invariants below are the contract; layout is the
implementer's call. Leave `// SPEC:` comments for genuine ambiguities. The point
is to make before/after on the expensive stage-3 work *fast to see* — so the
mountain-realism mechanisms (lithology, glacial, fold grain, couplings) can be
evaluated without ~90s regen per change.

## Scope decision (June 2026) — read this first

An audit (this branch) reshaped the original Arc-everything design. Two findings:

1. **Coarse stages are already retained additively.** `World::generate_*`
   accumulate onto one `&mut World` (`mod.rs:180–288`); advancing a stage adds
   fields, it doesn't discard prior ones (the lone exception: coarse `hydrology`
   is nulled when `fine` is built, `mod.rs:287`). So *navigating* between already-
   computed stages needs no recompute and no snapshot history — it's a render
   concern. (It is NOT currently *supported*: `current_stage()` derives from which
   fields are populated and rendering always targets the latest. So back-nav is
   new work — just cheap new work: a viewed-stage flag + render selection.)
2. **The fine mesh is the only thing worth persisting.** Coarse stages are cheap
   to recompute, so wrapping them in `Arc` buys nothing. The expensive, worth-
   storing object is the fine mesh base (`fine.rs` steps 1–7).

So this spec is **de-scoped from the original**:
- **No Arc / no snapshot-history model.** `FineBase` is a plain owned struct held
  in `World` alongside the re-runnable `FineSurface`. Stage navigation is a
  "viewed stage" render flag, not a list of immutable snapshots.
- **No in-session A/B.** Erosion variants are **re-run-to-replace** (cheap,
  reuses `FineBase`), seen sequentially (before → after), not held in pairs.
  Holding two surfaces at once was the *only* thing that forced `Arc<FineBase>`;
  it's out, so Arc is out. The only multi-view is stage navigation (stages
  stacked: coarse ↔ fine ↔ sub-stages).
- **Climate-ratio mutation stays in-place** (it was only a problem under the
  immutability model, which is gone).

What survives unchanged: the `FineBase`/`FineSurface` split, steppable
`ErosionState`, runtime erosion knobs, disk-cached `FineBase`, and the
viewed-stage navigator.

## Goal

Two capabilities, on two different axes:

1. **Structural axis — stage navigation.** Move back and forth between
   already-computed stages without recompute (render an earlier stage).
2. **Temporal axis — watch/step erosion.** Advance erosion incrementally and see
   valleys carve; jump back to the pre-erosion mesh.

These are orthogonal: stage navigation concerns stage *boundaries* and says
nothing about within-stage computation, so within-stage iteration (stepping
erosion, re-running with new knobs) composes freely with it.

## Model

### Stage 3 sub-stages — the `FineBase` / `FineSurface` split

Split the monolithic `FineWorld::generate_with_target` (`fine.rs:59–208`) at the
erosion seam (the `super::erosion::erode` call, `fine.rs:169`) so the expensive,
stable part is reused and the cheap, experimental part varies:

- **3a Fine mesh → `FineBase`** (owned): steps 1–7 — coarse-hydrology preview,
  density field, sampling, blue-noise relaxation, kNN tessellation, field
  transfer, base-elevation interpolation. Expensive (~most of stage 3); the
  reused object. Holds: `tessellation`, `coarse_cell`, `fields` (`FineFields`),
  `base_elevation`, `density`, `achieved_density_ratio`.
- **3b Erosion → eroded surface**: steps 8–9 — runs `ErosionState` over
  `FineBase`, then `Elevation::refine_from_base` micro-noise. Output = the eroded
  fine `Elevation`.
- **3c Hydrology → final**: step 10 — `Hydrology::generate_from_continentality`
  over the eroded surface.

`FineWorld` therefore splits into `FineBase` (owned, expensive) and
`FineSurface` (`elevation` + `hydrology`, cheap, per-variant). Re-running erosion
= a new `FineSurface` reading `&FineBase` and replacing the old one. The split
seam is load-bearing: if `FineWorld` stays monolithic, re-running erosion
recomputes the mesh — defeating the point.

`World` holds both as sibling fields (e.g. `fine_base: Option<FineBase>`,
`fine_surface: Option<FineSurface>`). Rendering reads both; nothing is
self-referential, so no `Arc`/lifetime gymnastics. (Keep a `FineWorld`-shaped
accessor or thin wrapper if it eases the `app/world.rs` call sites.)

### Erosion as a distinct stage (pre/post snapshots) — DONE

**Design pivot (June 2026).** The original plan was a *temporal* within-stage
stepper (Left/Right by % of `EROSION_STEPS`). Built, then dropped: stepping is
slow (full per-step rebuild) and, without keyframe storage (which the user does
not want), useless for back-and-forth. What's actually wanted for evaluating
mountain mechanisms is **pre-erosion vs eroded, snapped instantly** — the
*structural* axis. So erosion became its own stage:

- **Stage 3 (Hydrosphere)** — fine mesh + hydrology on the **un-eroded** base.
- **Stage 4 (Erosion)** — full fluvial erosion + re-derived hydrology.

`FineWorld` now holds `{ base: FineBase, pre: FineSurface, eroded:
Option<FineSurface> }`. Both surfaces share the one (disk-cached) `FineBase`;
the ~11s erosion cost is paid once on entering stage 4. `World::active_*` select
`pre` vs `eroded` by the view stage. `World::rerun_fine_eroded()` re-runs just
the eroded surface with tweaked `ErosionParams` (reuses the base) — the
iteration loop, available for a future runtime-knob UI.

The resumable `ErosionState` (`new/step/elevation`) survives as the internal
engine of `erode()` (and the substrate for a future time-evolution loop), but
the per-step UI is gone. `FineSurface::from_eroded` builds the pre surface (from
`base.base_elevation`) and is shared with full generation.

### GPU — per-stage buffer cache (instant snap)

Stages 3 and 4 share fine topology but differ in elevation/colors/rivers, so
snapping rebuilds buffers — too slow to do every snap at full res. Instead the
app caches a full `WorldBuffers` per non-active stage (`AppState.inactive_buffers`,
keyed by stage). Snapping swaps the active `WorldBuffers` for the cached one
(instant) and stashes the leaving stage's; on a cache miss it builds fresh. The
colored mesh (mode-specific) is re-derived via `regenerate_colors` only in
non-Relief modes (Relief uses the mode-independent unified mesh, so erosion
snapping in Relief is fully instant). The elevation map (wind-particle terrain)
is refreshed per snap. Memory: a `WorldBuffers` per visited stage (2 large for
fine stages 3/4, 2 small coarse) — acceptable; invalidated on regenerate (R) and
should be invalidated on a future erosion re-run.

### Disk cache (survive rebuilds)

The fine mesh is expensive to *recompute* after a code rebuild — which is most
before/afters (new mechanism = new erosion-or-later code). Serialize `FineBase`
to disk, keyed by seed + the params that affect the mesh (coarse gen + density
params). On startup, load if the key matches; otherwise regenerate. Then a
recompile of erosion/downstream code reloads the mesh and jumps straight to the
new code. Invalidate when a mesh-affecting param changes. (Optionally cache the
coarse `World` too, to land on a rendered stage 3 immediately on launch; the
coarse recompute is cheap, so this is a nicety.)

### Runtime erosion knobs

To re-run erosion with a tweaked value *without* a recompile, the experimental
erosion knobs (`EROSION_K`, diffusivity, uplift scale, steps) move from
compile-time `const` to runtime params carried in app/world state and passed into
`ErosionState::new`. Other constants can stay `const`.

## Navigation / UX

- **Stage forward / back + erosion snap** — DONE. `AppState.viewed_stage` tracks
  the rendered stage separate from the max computed. Space moves the view forward
  (computing the next stage only when already at the latest: 1→2→3→4), Backspace
  moves it back — both via `snap_to_stage`, which swaps cached per-stage buffers
  for instant toggling. Snapping stage 3 (pre-erosion) ↔ stage 4 (eroded) is the
  before/after-on-erosion comparison. A transient `World::view_stage` cap makes
  the `active_*` accessors + `mode_uses_fine_mesh` expose data only up to the
  viewed stage (stage 1/2 render the coarse mesh; stage 3 the pre-erosion fine
  surface; stage 4 the eroded one); default `u32::MAX` keeps headless/batch
  identical. Rivers auto-hide below stage 3 (no hydrology); particles gate on the
  viewed stage. Up/Down stay climate-ratio (applies to the viewed surface).

## Open decisions (call out in the PR)

- Exact `World` shape for the fine split (two `Option` fields vs a small
  `Fine { base, surface }` substruct) — implementer's call; keep `app/world.rs`
  call sites tidy.
- Disk-cache key: exactly which params invalidate `FineBase` (seed + coarse
  `NUM_CELLS`/`LLOYD_ITERATIONS`/`NUM_PLATES` + craton params + density params).
- Disk-cache format/location (e.g. `bincode` under a cache dir keyed by a hash).
- Keyframe scrubber: deferred (rough forward stepping + re-run-from-base back is
  enough per the user).

## Invariants

- Re-running erosion reuses `FineBase` by reference — no mesh recompute.
- `ErosionState` stepped to completion == batch `erode()` (same final state). ✓
- Stage navigation never recomputes an already-computed stage.
- Disk-loaded `FineBase` round-trips losslessly, AND regeneration reproduces it.
  World gen is now deterministic per seed (the staging work fixed two
  order-dependent sources — see [[hex3-fine-mesh-nondeterminism]]: the Lloyd
  parallel float-sum reduce in `sphere.rs` and the `HashMap`-ordered
  `build_adjacency` in `tessellation.rs`). So the content-hash cache key
  (version + seed + max_cells + fine consts + coarse generators/elevation/
  atmosphere) is STABLE across runs and the cache hits as intended. ✓ The
  version field still guards generation-code changes the content hash can't see.

## Non-goals

In-session A/B (hold two erosion surfaces at once); a full Arc snapshot-history
model; full free-scrub of every field; per-erosion-step first-class stages;
backward within-stage scrubbing beyond rough re-run-from-base; compressing the
mesh itself (it's stored/cached whole). Time-evolution (A′2) is out of scope
here, but the steppable `ErosionState` + the `FineBase`/`FineSurface` split are
deliberately the substrate it will reuse.
