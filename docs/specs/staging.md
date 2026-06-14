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

### Steppable erosion (within-stage 3b) — DONE

Status: implemented on this branch. `erode()` is now a thin wrapper over a
resumable `ErosionState` (`erosion.rs`) holding the loop-carried state
(`thick`, cached `routing`, `area`, step counter) plus once-built
`geom`/`areas`/`u_thick` and owned input copies (`base`, `thick_init`,
`precipitation`) — so stepping needs **no `Tessellation` borrow**, only
construction does (this is what lets the UI hold it across frames). API:
`new(tess, fields, base, precipitation)`, `step(n)`, `elevation()`,
`step_count()`, `is_halted()`. `Routing::build` now takes the prebuilt
`NeighborGeometry` instead of `&Tessellation`.

Invariant met: `new(..).step(EROSION_STEPS)` reproduces the batch run (same ops,
order, reroute schedule; `geom.tess_neighbors` is in `tess.neighbors` order). The
two unit tests still pass; the no-sinks early-`break` became a `halted` flag.

- **Forward** (the common case, "watch it happen"): continue the live
  `ErosionState` by some % of `EROSION_STEPS` per keypress — cheap, incremental.
- **Backward** (occasional): re-run from `FineBase` to the target step.
  Stateless, no keyframe storage; rough is fine (user's call). Avoid
  re-running-from-base on *every* forward press (quadratic) — forward continues
  the live state, only backward re-runs.

Keyframe storage / a fine scrubber is explicitly OPTIONAL and deferred; the
re-run-from-base approach covers rough scrubbing without it.

### GPU

Erosion variants/steps share topology, so the index buffer is built once and
reused; only the per-vertex buffer (elevation + color) is re-uploaded when
stepping erosion (topology unchanged) — fast enough to watch. No `Arc` on the
buffer is needed (one variant rendered at a time); just don't rebuild the index
buffer on an erosion step. Today the index/vertex buffers are already separate in
both render paths (`world.rs:393, 570`), and climate-ratio already does a full
rebuild with unchanged topology — that's the pattern to reuse, trimmed to
vertex-only on an erosion step.

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

- **Stage forward / back**: Space advances (unchanged); add a back key. Track a
  `viewed_stage` separate from the max-computed stage; back/forward selects which
  already-computed stage renders. Stage 3 exposes 3a (pre-erosion) / 3b (eroded)
  / 3c (hydrology).
- **Erosion stepping**: Left/Right advance/retreat erosion by some % (rough).
  Confirmed free in the keymap (`app/mod.rs`); Up/Down stay climate-ratio.
  Forward continues the live `ErosionState`; back re-runs from `FineBase`.

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
