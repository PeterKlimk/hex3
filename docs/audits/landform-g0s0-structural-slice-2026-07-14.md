# Landform G0/S0 planar structural-slice audit

**Date:** 2026-07-14  
**Verdict:** partial pass; structural planar checkpoint only  
**Contract:** [G0/S0 executable contract](../research/landform-object-packet-g0s0-2026-07-14.md)

## Result

The first arm-neutral landform-instrument slice passes its manufactured planar
geometry and surface-topology tests. It is not a completed G0/S0 packet and has
not observed product terrain or any H/C/G output.

Implemented in `src/world/landforms.rs`:

- a canonical planar `EvaluationSurfaceGraphV0` with explicit polygons,
  reciprocal faces, physical boundary segments and semantic portals;
- an explicit `LandscapeControlVolumesV0` seam, so a bare operator mesh cannot
  fabricate physical polygons;
- a verified regular-hex companion builder and general planar adapter;
- complete edge ownership validation: every polygon segment has exactly one
  internal-face or boundary-segment owner;
- exact-bit, simultaneous-level superlevel split forests;
- flat maxima/saddles, closure roots, elder ambiguity and deterministic IDs;
- exclusive cell ownership, nested physical footprints and union boundaries;
- inclusive persistence-plus-area reference and one-factor populations; and
- canonical bincode/FNV reproducibility checks independent of arm metadata.

## Defects caught before checkpoint

Adversarial review found that processing disconnected equal-height saddle
supports sequentially could make merge support and losing-branch ownership
depend on cell order. The implementation now snapshots all incidences with the
strictly higher surface and resolves each connected level event simultaneously.
A planar K2,2 fixture proves one event with both supports under cell reindexing.

Review also found that accepted public graphs could vary polygon starts and
boundary ordering while hashing differently, and that declared faces did not
prove complete polygon-edge ownership. Validation now requires canonical zero,
polygon starts, CSR, boundary ordering/IDs and exact edge backing.

The first regular-hex closure run exposed mixed precision in existing
`LandscapeMesh`: f64 centers/areas but f32 operator widths. Amendment A derives
the exact physical regular-hex face from center spacing, verifies the stored
width is its quantized operator copy, and retains the original tight area gate.
This was corrected before any competitive surface existed.

## Verification

- `cargo test --lib world::landforms --no-fail-fast`: **10 passed**.
- `cargo test --lib`: **221 passed, 6 ignored**.
- `cargo build --bin hex3`: **passed**.
- `cargo fmt --check`: **passed**.
- `git diff --check`: **passed**.

The ignored tests are existing long-running registered audit matrices, not new
G0/S0 failures. Repository-wide clippy with warnings denied is not a clean
baseline because of pre-existing lints outside this slice.

## Remaining contract

This checkpoint does not implement or validate:

- the spherical product `Tessellation` adapter;
- explicit endpoint geometry for the irregular planar Voronoi cap;
- equivalent-ellipse orientation/extent/width;
- fixed-radius local relief;
- least-squares physical grade, summit caps or gentle-area evidence;
- the full multiway/permutation/error fixture family;
- the registered analytic cone/cap/linked-segment 8/4/2 gates;
- serialized packet aggregates or product-reference observation; or
- D0 drainage, O0 relationships/correspondence, or any H/C/G composition.

## Next bounded step

Implement the planar highland morphology fields and frozen analytic 8/4/2
fixtures against this structural core. Add the spherical and irregular-cap
geometry adapters before declaring G0/S0 complete. Do not begin competitive
terrain arms or tune object thresholds on generated output.
