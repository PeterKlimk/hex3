# Landform G0/S0 planar morphology-slice audit

**Date:** 2026-07-14

**Verdict:** partial pass; planar reference-highland morphology only

**Contract:** [G0/S0 executable contract](../research/landform-object-packet-g0s0-2026-07-14.md)

## Result

The arm-neutral extractor now measures the registered planar morphology of the
reference highland population. Manufactured unit fixtures pass. This is still
not a completed G0/S0 packet: the registered 8/4/2 analytic surfaces and the
spherical/product geometry path have not passed, and no product terrain or
H/C/G output has been observed.

Implemented in `src/world/landforms.rs`:

- translated-frame polygon-union area, centroid and covariance with closure
  against authoritative cell area;
- equivalent-ellipse dimensions, anisotropy, sign-canonical orientation,
  ambiguity, two-sweep extent and mean width;
- physical-radius local relief with area-weighted quantiles and exact
  point-to-segment scored/physical-boundary truncation;
- face-width/distance weighted least-squares grade with the registered
  eigenvalue rank rule;
- the full summit-depth/gentle-grade matrix, valid-grade fractions and merge
  censoring; and
- hash-covered planar measurements for the frozen reference population.

Sensitivity populations remain structural report-only sets. They do not
materialize duplicate morphology records under schema v0.

## Contract corrections caught before checkpoint

Independent review found three boundary failures. First, the general planar
adapter trusted native operator distance and direction fields even though grade
weights depend on them. The adapter and public planar graph validator now check
centre distance and directed face-normal consistency with explicit source-f32
quantization tolerances.

Second, non-finite least-squares arithmetic and unexpected nonpositive
determinants were being reported as ordinary rank deficiency. Only the frozen
eigenvalue predicate now yields an unavailable grade; other numerical failures
are typed fatal errors.

Third, the public hierarchy builder could accept a spherical graph before the
spherical G0 winding, radius and area-closure adapter exists. It now returns an
explicit unsupported-domain error. A spherical measurement status remains in
the schema for the future adapter, but cannot be emitted as apparently valid
partial evidence today.

## Verification

- `cargo test --lib world::landforms::tests`: **16 passed**.
- `cargo test --lib`: **227 passed, 6 ignored**.
- `cargo build --bin hex3`: **passed**.
- `cargo fmt --check`: **passed**.
- `git diff --check`: **passed**.

The ignored tests are existing long-running registered audit matrices, not new
G0/S0 failures.

## Remaining contract

This checkpoint does not yet establish:

- the frozen one-hill, two-cone, cap-pair and linked-segment 8/4/2 gates;
- cross-resolution topology, retention, ordering or error bounds;
- the spherical product `Tessellation` adapter and spherical morphology;
- explicit control volumes for the irregular planar Voronoi cap;
- the remaining multiway, permutation and error fixture family;
- serialized packet aggregates or product-reference observation; or
- D0, O0, R0 or any H/C/G terrain comparison.

## Next bounded step

Implement the already frozen planar 8/4/2 analytic suite against this code.
Treat failures as evidence about the instrument or registered gates, not an
invitation to tune thresholds. Checkpoint that result before adding spherical
or product adapters.
