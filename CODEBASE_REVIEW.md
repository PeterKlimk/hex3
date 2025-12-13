# Hex3 Codebase Review Notes

This document captures observations and improvement suggestions from a quick codebase review.

## Likely Bugs / Correctness Issues

### Map rendering depth interactions

- `PipelineBuilder` enables depth by default (`TextureFormat::Depth32Float`, `CompareFunction::Less`, depth writes on).
- In map mode, triangles are effectively coplanar (`z = 0.0` in `src/geometry/mesh.rs`), and wrapping can create overlaps.
- Result: some triangles/edges can fail the depth test and disappear in map view, depending on draw order and overlaps.

Suggested fix:
- Disable depth entirely for map pipelines (preferred), or at least disable depth writes and use `LessEqual`.
- Files: `src/main.rs` (map pipeline setup), `src/render/pipeline.rs` (defaults), `src/geometry/mesh.rs` (map projection).

### Convergence dot product mixes non-tangent normal with tangent velocity

- Convergence is computed as `relative_vel.dot(boundary_normal)` where:
  - `relative_vel` is tangent to the sphere at `boundary_point` (`velocity_at` returns `axis × point`).
  - `boundary_normal = (neighbor_pos - cell_pos).normalize()` is a chord direction, not tangent to the sphere at `boundary_point`.
- This mixes spaces and can bias convergence magnitude/sign, especially for larger inter-cell angles.

Suggested fix:
- Project the boundary direction into the tangent plane at `boundary_point` before dotting:
  - `tangent_normal = boundary_normal - boundary_point * boundary_normal.dot(boundary_point);`
  - Normalize if length > eps.
- Files: `src/geometry/tectonics.rs` (`calculate_boundary_stress`), `src/geometry/plates.rs` (edge color convergence).

### Seed selection can yield fewer than requested plates

- `select_seeds_spaced` may return fewer than `num_plates` if rejection sampling fails to find enough candidates.
- Downstream assumes `num_plates` in multiple places (e.g. target size generation, plate type assignment, Euler poles).

Suggested fix:
- Add a fallback phase to fill remaining seeds:
  - Relax the minimum distance progressively, or
  - Fill remaining with random unpicked cells.
- Files: `src/geometry/plates.rs` (`select_seeds_spaced`, `min_seed_distance`).

## Panic / NaN Footguns

- `partial_cmp(...).unwrap()` will panic if any compared value is NaN.
  - `src/geometry/voronoi.rs`: `order_vertices_ccw` sorts angles.
  - `src/geometry/lloyd.rs`: nearest-neighbor distance `min_by` uses `partial_cmp`.

Suggested fix:
- Use `f32::total_cmp` when sorting numeric keys, or explicitly handle NaNs (e.g. treat NaN as `+∞` or filter).

## Performance Hotspots (Notable At ~20k Cells)

### Stress propagation is O(N × boundary_per_plate)

- `propagate_stress` sums contributions from *all* boundary cells in the same plate for *every* cell.
- Complexity grows quickly as cell count increases.
- File: `src/geometry/tectonics.rs` (`propagate_stress`).

Suggested alternatives:
- Multi-source graph propagation on the adjacency graph with cutoff (ignore contributions once `exp(-d/L)` is negligible).
- Iterative diffusion / relaxation within each plate.

### Repeated HashSet allocations + intersections for shared edge detection

- `calculate_boundary_stress` and boundary edge coloring build `HashSet`s per cell and intersect repeatedly.
- This is allocation-heavy and repeated work, given adjacency is already derived from shared edges.
- Files: `src/geometry/tectonics.rs`, `src/geometry/plates.rs`.

Suggested fix:
- Extend `build_adjacency` to also return shared edge vertex pairs and optionally precomputed edge arc length.
- Reuse that data for stress and edge coloring.

## Modeling / Tuning Notes

### Continental coverage can undershoot

- Plate types are assigned by a greedy “do not overshoot” rule, which can stop below the target coverage.
- File: `src/geometry/tectonics.rs` (`assign_plate_types_by_coverage_with_rng`).

Suggested fix:
- Allow the last selected plate to overshoot, or pick the closest-to-target plate among remaining candidates.

### Voronoi “circumcenter” naming

- `circumcenter_on_sphere` computes (and hemisphere-corrects) the facet normal direction, which is the correct dual for spherical Voronoi via hull duality.
- The implementation is fine; the name may be slightly misleading relative to Euclidean circumcenters.
- File: `src/geometry/voronoi.rs`.

## Structure / Maintainability Suggestions

### Split `src/main.rs`

`src/main.rs` currently mixes:
- input/event handling,
- GPU setup and pipeline creation,
- world generation and mesh/buffer creation,
- rendering and UI state (view/render modes, edges, FPS title updates).

Suggested refactor:
- `src/app.rs`: `App`, `AppState`, event handling, render loop hooks.
- `src/world.rs`: `generate_world_buffers` and related mesh/buffer prep.
- `src/config.rs`: constants (`NUM_CELLS`, `NUM_PLATES`, `LLOYD_ITERATIONS`) and perhaps defaults.
- Keep `src/main.rs` as a thin entry point.

### Add reproducibility/config entrypoints

- Seed is printed, which is great; make it injectable.
- Consider a small CLI or config for `seed`, `NUM_CELLS`, `NUM_PLATES`, and `LLOYD_ITERATIONS`.

## Housekeeping

- `cargo test` passes.
- `cargo clippy` passes.
- `cargo fmt --check` fails with formatting-only diffs in:
  - `src/main.rs`
  - `src/geometry/mesh.rs`
  - `src/geometry/plates.rs`
  - `src/geometry/tectonics.rs`

Suggested fix:
- Run `cargo fmt`.

