# Channel extraction R1a G0 Voronoi-cap audit

**Date:** 2026-07-13
**Status:** G0 geometry substrate passes; routing/rank-conflict gate pending
**Specification:** [Irregular-Voronoi seeded extraction R1a](../research/channel-extraction-r1a-2026-07-13.md)

**Later checkpoint:** Exact inputs, immutable routing and the visited-cell rank-
conflict subgate subsequently passed; see the
[input/rank audit](channel-extraction-r1a-input-rank-precheck-2026-07-13.md).
This audit retains the narrower status of the geometry checkpoint it records.

## Verdict

The guarded Earth-radius S2 Voronoi cap is a valid substrate for the R1a
discriminator. The product `s2-voronoi` path produces deterministic, reciprocal
two-vertex finite-volume geometry with unequal cell areas and face widths at
8/4/2 km. Eight- and ten-spacing guards agree within the frozen tolerance, and
the tangent adapter remains well inside its projection-error gates.

This does **not** pass R1a or select an extractor. The registered affine/V
polygon means, conservative MFD route, visited-cell P0/M0 rank conflict and F0
path gates are not implemented. In particular, the second half of G0 gate 5
remains pending: if P0 and M0 never disagree in the registered matrix, the
discriminator is still invalid.

## Implemented substrate

`src/world/landscape/voronoi_cap_fixture.rs` now owns:

- the frozen coordinate-keyed triangular lattice, ChaCha8 jitter, exponential
  sphere map and Fibonacci far-field support;
- the `Tessellation::from_points_knn_clipping` product-backend construction;
- retention by projected generator position and actual polygon-edge ownership,
  without consuming repaired nearest-neighbour adjacency;
- planar finite-volume cell areas, shared-face widths/midpoints, generator
  distances, cut boundary faces and portal `401`;
- retained unit-sphere geometry for rebuild/guard comparison; and
- area, face-width, centroid-offset and projection audits.

Only edges with exactly two polygon owners enter the adapter. Internal edges
are emitted reciprocally, and a retained cell pair cannot own more than one
face; retained-to-guard edges become explicit physical boundary faces.
`LandscapeMesh::validate` checks positive geometry, CSR shape and reciprocal
internal adjacency.

## Numerical results

The table reports the eight-spacing construction. The ten-spacing construction
has the same retained counts and passes the cell-, polygon-, edge- and boundary-
metric comparison at `1e-3` nominal spacing.

| spacing | retained cells | internal faces | boundary / portal faces | area p10 / p50 / p90 (km²) | face width p10 / p50 / p90 (km) | generator-centroid p50 / p95 (km) | max edge distortion | total area error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 km | 1,057 | 3,043 | 256 / 40 | 50.420 / 55.269 / 60.867 | 3.417 / 4.718 / 5.817 | 0.458 / 0.839 | 0.01125% | 0.00402% |
| 4 km | 4,160 | 12,224 | 512 / 80 | 12.553 / 13.841 / 15.240 | 1.712 / 2.356 / 2.920 | 0.235 / 0.429 | 0.01128% | 0.00397% |
| 2 km | 16,514 | 49,029 | 1,026 / 160 | 3.128 / 3.461 / 3.813 | 0.855 / 1.177 / 1.458 | 0.118 / 0.215 | 0.01161% | 0.00395% |

Area and face-width p90 exceed p10 at every resolution. Generator-to-polygon-
centroid offsets are material relative to the cells and scale down with nominal
spacing; R1a must therefore retain its polygon-mean terrain input and treat the
generator-distance two-point operator as an approximation tested by A, not as
an exact polygon-centroid discretization.

Maximum projected-versus-spherical centre-distance error is
`0.01162–0.01169%`; maximum projected-chord-midpoint versus projected spherical-
midpoint displacement is `0.000014–0.000021 km`; and maximum individual-cell
area error is `0.01175–0.01178%`. These are report-only construction diagnostics,
not newly invented promotion gates.

At 8 and 4 km the printed eight- and ten-guard audit summaries are bit-equal.
At 2 km their total projected areas differ by about `1.0e-5 km²`; all matched
unit generators/polygon vertices and projected per-cell, face and boundary
metrics remain inside the preregistered tolerances.

The area comparison interprets the registered length tolerance dimensionally
as `1e-3 × spacing²`; independent vertex-by-vertex length checks also constrain
every polygon.

## Numerical issue found during implementation

The first adapter used `acos(z)` for the spherical logarithmic map. Near the
north-pole tangent point, the backend's `f32` unit coordinates make that form
ill-conditioned; it falsely reported up to 17.8% edge distortion even though
the geometry was sound. The equivalent
`atan2(horizontal_length, z)` formulation is well-conditioned there. Spherical
arc comparison also normalizes the stored directions in `f64` before measuring
the chord. The frozen projection gates were not relaxed.

## Executed checks

Default smoke and deterministic rebuild:

```bash
cargo test --lib r1_cap -- --nocapture
```

Result: three passed; the full matrix is intentionally ignored in routine test
runs because it takes roughly two minutes in an unoptimized build.

Full preregistered 8/4/2 km, eight-versus-ten-guard matrix:

```bash
cargo test --lib r1_cap_passes_g0_geometry_and_guard_gates_at_all_spacings -- --ignored --nocapture
```

Result: passed in 128.89 seconds. The test compares retained unit and projected
polygons, cell areas, adjacency, center distances, face widths/tangents/
midpoints, boundary ownership/geometry and portal assignments. Rebuilding the
complete fixture is bit-identical independently at 8, 4 and 2 km.

## Next bounded step

Implement only the registered polygon-moment A/V/B surfaces and one immutable
`FaceFlowCache::route_with_portals` result per case. Before writing either path
tracer, inspect the outgoing physical-grade and MFD-fraction ranks over the
registered A/V matrix. If no visited candidate can distinguish P0 from M0,
invalidate R1a without adding another extractor or tuning the mesh.
