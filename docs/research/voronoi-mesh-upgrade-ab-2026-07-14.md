# Modern Voronoi backend controlled upgrade

**Status:** evaluated positive after upstream correction; promoted physical
geometry substrate at exact revision `e8804a639ea3c989e1ce9ea44b4c66c5f2d7e060`

**Date:** 2026-07-14

**Hex3 baseline revision:** `3208f9a`

## Question

Does replacing the currently pinned `s2-voronoi` revision with the modern
upstream `voronoi-mesh` revision produce a valid fine physical control-volume
graph, at acceptable cost, without a Hex3-side topology repair?

This is a substrate experiment, not a terrain mechanism or a relaxation of the
landform G0 contract. It was prompted by the frozen seed-12345 product
observation, where fine cell 199413 contains distinct vertex IDs whose unit
directions form a zero-angle physical edge. The old backend pin predates a
large upstream robustness rework. The modern backend is therefore a plausible
correction, but its strict validation is principally topological and does not
contractually guarantee that every pair of distinct returned `f32` vertices
has positive angular separation. The unchanged Hex3 G0 predicate remains the
decisive witness.

## Frozen arms

### A: retained baseline

- dependency repository: `https://github.com/PeterKlimk/s2-voronoi`;
- revision: `8ee131ca0b4415aa1638d02fc8542623a7c20eca`;
- current Hex3 wrapper drops native `NO_NEIGHBOR` entries and connects orphan
  cells to their nearest generator;
- fine-base cache schema: version 13;
- frozen product result: coarse G0 passes, fine G0 rejects cell 199413 edge 6.

### B: modern candidate

- dependency package: `voronoi-mesh`, retained under the local Cargo alias
  `s2-voronoi` so the dependency rename is not confused with semantic changes;
- dependency repository: `https://github.com/PeterKlimk/s2-voronoi`;
- exact revision: `da99a8a6384f2fb641c0d3389a6787dec78fc5b4`;
- entry point: `compute_with_report(points, VoronoiConfig::default())`;
- physical mesh: the returned preferred/effective diagram, not the public
  diagram containing original-input aliases after welding;
- fine-base cache schema: version 14;
- no change to sampling, world settings, G0 geometry, S0 morphology, terrain,
  erosion, hydrology or presentation.

The candidate consumes the effective solved partition because welded public
cells are aliases for API cardinality, not distinct physical control volumes.
Fine cell-aligned product fields are created after tessellation, so their
cardinality follows the returned effective mesh.

## Candidate acceptance contract

Hex3 will retain a serializable, Hex3-owned summary of the backend report on a
clipped-Voronoi `Tessellation`. At minimum it records original and effective
site counts, merged sites, the preferred validation verdict, pre-repair edge
mismatches, whether local repair was attempted and accepted, post-repair
unpaired edges and whether degeneracy perturbation was applied.

Construction must reject rather than conceal any candidate for which:

- the preferred diagram is not strictly valid;
- post-repair residuals remain;
- great-circle degeneracy perturbation was applied;
- native adjacency contains `NO_NEIGHBOR`;
- adjacency is asymmetric, disconnected from its physical face ownership, or
  otherwise fails the existing product geometry adapter;
- any physical polygon edge has zero angular length or any cell fails the
  unchanged G0 area and closure gates.

An upstream local 3D repair which is reported as attempted and accepted is
allowed and must remain observable. Hex3's old behavior of dropping defective
edge entries and inventing nearest-generator links is removed in B; it cannot
be used to make the candidate pass.

## Checks before the product rerun

The implementation checkpoint must include focused tests for:

1. strict report acceptance and absence of native missing neighbors;
2. exact reciprocal adjacency and agreement between adjacency and shared
   polygon edges;
3. deterministic repeated construction on a fixed point set;
4. effective-diagram consumption for a manufactured near-coincident input
   which triggers welding;
5. existing product spherical G0 and irregular-cap fixtures;
6. cache invalidation from version 13 to 14;
7. all binaries and tests under the renamed package/API.

Commands, run from WSL2:

```bash
cargo fmt --check
cargo build --bins
cargo build --release --bin hex3
cargo build --release --bin landform_baseline
cargo test --all-targets --no-fail-fast
cargo clippy --all-targets
```

The candidate's direct release runtime is measured after compilation so build
time is not attributed to tessellation. Record input, effective cell, merge,
vertex and directed-edge counts together with wall time and peak process RSS.
Exact fine hashes may change because the mesh backend is the experimental
variable. Coarse convex-hull Stage 1/2 behavior is outside the changed owner and
must not change.

## Frozen decisive rerun

After the focused checkpoint, repeat the previously registered product command
without changing seed, cap, model, stages, cache mode, G0 tolerances or output
path:

```bash
cargo build --release --bin landform_baseline
/usr/bin/time -v timeout 20m target/release/landform_baseline \
  artifacts/landforms/seed-12345-product-g0s0-250k-v0.json
```

The first decisive question is whether fine G0 completes. If it does not, no
S0 or product morphology inference is permitted. If it does, the unchanged
harness may continue through the five registered ancestry surfaces; those
results remain descriptive and do not promote an H/C/G terrain arm.

## Decision table

| Outcome | Disposition |
|---|---|
| Candidate backend errors, reports residual topology, perturbs inputs, or cannot supply an unaliased effective partition | Retain A; audit the incompatibility |
| Candidate passes its report gates but fails unchanged fine G0 | Do not claim a fix; retain or separately justify B only if its independent value clearly pays |
| Candidate passes G0 but causes material non-owner regressions or unjustified cost | Retain A or park B pending a scoped correction |
| Candidate passes focused/full regressions and unchanged fine G0 at acceptable cost | Promote B as geometry substrate, then resume the already-frozen product observation |

No outcome permits weakening physical G0, adding a renderer workaround, tuning
terrain, or interpreting an uncompleted ancestry packet.

## Outcome

Candidate B passed its report, focused, full-regression and integration gates
but failed unchanged fine G0 at cell 57852 edge 7. Two distinct vertex IDs are
stored at bit-identical f32 directions despite a topology-clean upstream
report. No product surface reached S0. The candidate remains an implemented
integration scaffold, not a promoted physical substrate, pending an upstream
valid-or-error correction and the same frozen rerun.

See the [A/B audit and upstream handoff](../audits/voronoi-mesh-zero-edge-handoff-2026-07-14.md).

### Resolution

Upstream subsequently added exact stored-zero discovery, safe transactional
contraction and output-resolution reporting. Hex3 pins revision
`e8804a639ea3c989e1ce9ea44b4c66c5f2d7e060`, requires zero remaining edges in
both the output report and independent validator, and bumps the fine cache to
version 15. The focused/full regressions and unchanged product G0/S0 run pass.
This promotes the geometry substrate only; the resulting descriptive terrain
evidence is recorded in the
[product completion audit](../audits/landform-product-g0s0-completion-2026-07-15.md).
