# Geometry / Normalization / Precision Notes

This document summarizes findings and follow-ups from investigating geometry shortcuts,
normalization strategy, and precision/performance trade-offs.

## Geometry Shortcuts / Invariants

- The system assumes unit-sphere inputs across geometry (e.g., points/vertices are
  on the unit sphere in `src/geometry/sphere.rs`, `src/geometry/lloyd.rs`,
  `src/geometry/voronoi.rs`, and `src/geometry/gpu_voronoi/cell_builder.rs`).
- For many checks, angle comparisons do not need `acos`. Use dot-product comparisons
  against `cos(threshold)` or chord-length proxies `2 * (1 - dot)` where only
  ordering or thresholds matter.
- Small-angle approximations can avoid `acos` for falloff curves, e.g.
  `angle ≈ sqrt(2 * (1 - dot))` for small angles.
- Core geometry already uses correct spherical formulas where needed:
  - Spherical excess via `atan2` in `src/geometry/lloyd.rs`.
  - Great-circle clipping for GPU Voronoi in `src/geometry/gpu_voronoi/cell_builder.rs`.

## Normalization (Where + Why It Broke)

- Inputs are normalized at generation/relaxation time; geometry code assumes unit
  length inside hot loops.
- The certification regression happened because the f64 fallback normalized points
  again, producing f64 plane normals that differed from the f32 plane normals used
  for clipping. That inflated the per-plane error and caused widespread
  "ill-conditioned" failures.
- Fix: treat f32 positions as canonical for f64 certification (avoid re-normalizing
  in the fallback path) so cert uses the same basis as the f32 clipping planes.
- Recommendation: normalize only at the boundaries (generation/import) and keep
  inner loops consistent with the build path (no extra normalization unless required
  for correctness).

## Precision / Performance

- f32 should remain the core geometry type; f64 is best limited to certification and
  validation to avoid consistency mismatches and performance cost.
- Hot operations include `acos`, `normalize`, and `cross` inside loops in:
  - `src/world/atmosphere.rs`
  - `src/world/elevation.rs`
  - `src/world/features.rs`
  - `src/world/boundary.rs`
- Performance win: replace distance checks that only compare thresholds with
  dot-product or chord comparisons, and precompute `cos(threshold)`.
- For visualization-only paths, consider an approximate `acos` option to reduce
  overhead further.

## Suggested Follow-Ups

1. Convert the hottest `acos` call sites to dot/chord comparisons or small-angle
   approximations.
2. Add a debug-only unit-length assertion at module boundaries to prevent
   inconsistencies from creeping into hot loops.
3. (Optional) Add a toggle for approximate `acos` in visualization-only paths.
