# Landform G0 geometry-adapter audit

**Date:** 2026-07-14
**Verdict:** pass; G0 geometry-adapter checkpoint only
**Contract:** [G0/S0 executable contract](../research/landform-object-packet-g0s0-2026-07-14.md)

## Result

The remaining product and irregular-testbed geometry adapters now pass their
focused manufactured and adversarial checks. This completes the registered G0
geometry seam; it does not complete spherical S0 or evaluate a terrain owner.

The product `Tessellation` adapter now:

- discovers each spherical face from the exactly two source Voronoi cells that
  own one consecutive source-vertex-ID edge;
- rejects an edge without exactly two opposite owners and requires set-equal
  agreement between polygon-derived and stored product adjacency;
- normalizes generators and Voronoi vertices to the canonical radius in `f64`;
- derives center distances and face widths with the robust great-circle
  `atan2(cross, dot)` formula;
- recomputes positive, signed per-cell solid-angle area in `f64` and validates
  point radius, polygon winding, global `4*pi*R^2` closure, connectedness and
  the absence of physical boundary segments; and
- retains coordinate-canonical evidence independent of harmless source vertex
  renumbering while still using source IDs to establish physical ownership.

The projected 8 km product-Voronoi cap adapter now carries the exact directed
face endpoints created by the source projection. It retains native
finite-volume cell areas, face widths and center distances only after checking
them against the explicit polygons and faces. It does not reconstruct an
irregular face from a midpoint or assume that every projected face is
perpendicular to the generator displacement.

## Adversarial evidence

Focused tests prove that:

- an off-radius spherical point is rejected;
- reversing a spherical polygon invalidates its physical geometry;
- adding stored product adjacency without a two-owner polygon edge is rejected
  as `NonPhysicalAdjacency`;
- hidden prefix data in either compact product adjacency or Voronoi-cell index
  storage is rejected as malformed rather than silently ignored;
- corrupting projected-cap cell area or native face width is rejected;
- moving an exact projected-cap endpoint is rejected; and
- `build_surface_hierarchy_v0` continues to reject spherical input after the
  independent G0 graph has passed validation.

These checks defend geometry authority. They do not validate spherical
highland moments, tangent-plane grade, fixed-radius boundary distance or any
other S0 morphology.

## Verification

- `cargo test --lib world::landforms --no-fail-fast`: **19 passed, 1 ignored**.
- `cargo test --lib --no-fail-fast`: **230 passed, 7 ignored**.
- `cargo build --bin hex3`, formatting and diff checks pass.
- The ignored test is the already evaluated long-running planar analytic
  8/4/2 audit matrix, not a G0 geometry failure.

## Scope and interpretation

This checkpoint establishes two physical adapters for the common evidence
instrument:

1. closed spherical product Voronoi geometry; and
2. the explicit projected irregular-cap geometry used by the bounded testbed.

It makes no statement about product landform quality, drainage organization or
whether hold-and-carve, reduced coevolution or graph-first reconstruction
should own terrain. No H/C/G surface or native semantic graph was consumed.

Spherical S0 and spherical morphology remain deliberately unavailable. They
require their own executable contract amendment and manufactured evidence;
passing spherical G0 must not be treated as permission to reuse planar
measurements implicitly.
