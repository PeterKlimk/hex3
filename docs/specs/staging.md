# Spec: Staged Snapshots + Steppable Erosion (before/after tooling)

Loose spec: the model and invariants below are the contract; layout is the
implementer's call. Leave `// SPEC:` comments for genuine ambiguities. The point
is to make before/after on the expensive stage-3 work *fast to see* — so the
mountain-realism mechanisms (lithology, glacial, fold grain, couplings) can be
evaluated without ~90s regen per change.

## Goal

Two capabilities, on two different axes:

1. **Structural axis — navigable, shared stage snapshots.** Move back and forth
   between locked stages without recompute, and A/B-compare variants that share
   an expensive prefix (chiefly the fine mesh).
2. **Temporal axis — watch/step erosion.** Advance erosion incrementally and see
   valleys carve; jump back to the pre-erosion mesh; compare two erosion runs.

These are orthogonal: the snapshot model defines stage *boundaries* and says
nothing about within-stage computation, so within-stage iteration (stepping
erosion, re-running with new knobs) composes freely with it.

## Model

### Structural sharing (the snapshot spine)

Stages are immutable snapshots with object-level structural sharing
(copy-on-write via `Arc`), NOT field-level diffs. A stage transition is
`input snapshot -> output snapshot`, where the output references every object
that didn't change and holds a fresh object for what did. Because stages are
additive (each sets new fields), a snapshot is mostly shared `Arc`s plus the one
new object.

- Stage objects (`plates`, `crust`, …, `atmosphere`, fine-mesh artifacts) become
  `Arc<T>`; the world snapshot is a bundle of `Arc`s; history is a list of
  snapshots. Back/forth = pick a snapshot (free). A/B = two snapshots sharing
  the prefix by reference (expensive mesh shared, not duplicated, not recomputed).
- "Locked" = committed to history and treated as immutable. Mutation-in-place
  becomes produce-a-new-snapshot (see climate, below).

### Stage 3 sub-stages

Split the monolithic `FineWorld::generate` into three locked stages so the
expensive, stable part is shared and the cheap, experimental part varies:

- **3a Fine mesh → `FineBase`** (Arc): sampling + relaxation + tessellation +
  field transfer + pre-erosion base elevation. Expensive; the shared object.
- **3b Erosion → eroded surface**: shares `FineBase`; its internal computation is
  the steppable `ErosionState` (below). Locked output = the chosen eroded
  elevation/thickness.
- **3c Hydrology → final**: shares the above.

`FineWorld` therefore splits into `FineBase` (Arc, expensive, locked) and
`FineSurface` (erosion result + hydrology, cheap, per-variant). Re-running
erosion or A/B = a new `FineSurface` pointing at the same `FineBase`. The split
seam is load-bearing: if `FineWorld` stays monolithic, nothing shares and the
mesh recomputes — defeating the point.

### Steppable erosion (within-stage 3b)

Refactor `erode()` from run-to-completion into a resumable `ErosionState`
holding (`thick`, `thick_init`, `base`, `geom`, `areas`, `u_thick`, cached
routing, step counter) with `step(n)` advancing `n` steps. Invariant: stepping
to `EROSION_STEPS` must produce the SAME result as today's batch `erode()` for
the same inputs — this is a refactor, not a new algorithm (the existing unit
tests still hold).

- **Forward** (the common case, "watch it happen"): continue the live
  `ErosionState` by some % of `EROSION_STEPS` per keypress — cheap, incremental.
- **Backward** (occasional): re-run from the cached `FineBase` to the target
  step. Stateless, no keyframe storage; rough is fine (user's call). Avoid
  re-running-from-base on *every* forward press (that's quadratic) — forward
  continues the live state, only backward re-runs.

Keyframe storage / a fine scrubber is explicitly OPTIONAL and deferred; the
re-run-from-base approach covers rough scrubbing without it.

### GPU

Erosion variants share topology, so the index buffer is shared (Arc on the GPU
buffer); only the per-vertex buffer (elevation + color) is per-variant. Stepping
erosion re-derives elevation and re-uploads the vertex buffer (topology
unchanged) — fast enough to watch.

### Disk cache (survive rebuilds)

`Arc` makes in-session history cheap, but the fine mesh is still expensive to
*recompute* after a code rebuild — which is most before/afters (new mechanism =
new erosion-or-later code). Serialize `FineBase` (and optionally the coarse
world) to disk, keyed by seed + the params that affect the mesh (coarse gen +
density params). On startup, load if the key matches; otherwise regenerate. Then
a recompile of erosion/downstream code reloads the mesh and jumps straight to the
new code. Invalidate when a mesh-affecting param changes.

### Runtime erosion knobs

To re-run erosion with a tweaked value *without* a recompile, the experimental
erosion knobs (`EROSION_K`, diffusivity, uplift scale, steps) move from
compile-time `const` to runtime params carried in app/world state. Other
constants can stay `const`.

## Navigation / UX

- **Stage forward / back**: Space advances; add a back key (data persists, so
  back = render an earlier snapshot). Stage 3 now has 3a/3b/3c.
- **Erosion stepping**: arrow keys advance erosion by some % (rough). NOTE key
  conflict — Up/Down currently adjust the climate ratio at stage 3; pick a free
  binding (e.g. Left/Right for erosion step) or scope by sub-stage. `// SPEC:`
- **A/B**: toggle between two locked erosion results sharing one `FineBase`.

## Open decisions (call out in the PR)

- Climate-ratio mutation (`Hydrology::set_climate_ratio`, currently in-place)
  becomes produce-a-new-snapshot under immutability. Arguably better (each
  climate setting is a lockable state) but it's a real touch.
- Disk-cache key: exactly which params invalidate `FineBase`.
- Keyframe scrubber: deferred (rough forward stepping + re-run-from-base back is
  enough per the user).

## Invariants

- Locked snapshots are immutable; sharing must never deep-copy the fine mesh.
- Re-running erosion reuses `FineBase` by reference — no mesh recompute.
- `ErosionState` stepped to completion == batch `erode()` (same final state).
- Disk-loaded `FineBase` is bit-identical to a freshly generated one for the
  same key (deterministic generation).

## Non-goals

Full free-scrub of every field; per-erosion-step first-class stages; backward
within-stage scrubbing beyond rough re-run-from-base; compressing the mesh
itself (it's stored/cached whole). Time-evolution (A′2) is out of scope here,
but the steppable `ErosionState` + snapshot spine are deliberately the substrate
it will reuse.
