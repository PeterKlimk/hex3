# GPU-style spherical Voronoi: soundness issue + proposals

## Current soundness issue

The current incremental clipping implementation in
`src/geometry/gpu_voronoi/cell_builder.rs` only bootstraps a polygon when
exactly three neighbor planes exist, and it assumes those *first three* planes
form a valid spherical triangle inside all half-spaces. When those three planes
are not the active supporting set (common at high density), their pairwise
intersections lie outside other half-spaces and the initialization fails. The
cell then remains empty and never recovers, even though a valid Voronoi cell
exists.

In short: **"first 3 kNN planes must form a valid seed triangle" is not a sound
assumption**, especially at large cell counts.

## Robust bootstrap inside the current incremental clipper

Keep the existing O(k*m) clipping pipeline, but replace the fragile
"planes.len() == 3" bootstrap with a small, bounded search for a valid seed
triangle among an early prefix of planes (e.g., the first 6-12 in kNN order).
Once any valid triplet is found, initialize the polygon and then clip all
remaining planes (including the earlier ones not used in the seed).

Why this is sound and fast:
- It does not add extra half-spaces; it only chooses a correct initialization.
- Work is bounded and only occurs for the failing cases.
- The hot path remains the current incremental clipper.

Implementation sketch:
- Accumulate planes until a seed exists.
- Test triplets that include the newest plane and prior planes (cheap O(m^2)).
- A triplet is valid if its great-circle intersections yield three vertices that
  satisfy all three half-space constraints, and the winding is consistent with
  the generator.
- After initializing, continue clipping with all planes in original order
  (skipping those already used for the seed).