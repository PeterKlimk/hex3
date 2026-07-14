# Product G0/S0 ancestry observation audit

**Date:** 2026-07-14

**Verdict:** stopped at fine-G0 prerequisite; no product morphology observed

**Continuation:** the prerequisite was subsequently corrected upstream and the
unchanged observation completed. See the
[2026-07-15 completion audit](landform-product-g0s0-completion-2026-07-15.md).

**Contract:** [Product G0/S0 ancestry observation](../research/landform-product-g0s0-observation-2026-07-14.md)

## Result

The frozen seed-12345 coarse/250k product observation did not reach S0. The
convex-hull coarse tessellation passed product G0 adaptation. The adaptive fine
tessellation then failed physical spherical cell validation at:

```text
cell: 199413
polygon edge: 6
reason: DegenerateEdge
```

The source polygon uses distinct vertex IDs, but those endpoints collapse to a
zero-angle edge after the registered normalization to the physical sphere. G0
correctly rejected it. No tolerance was relaxed, polygon reversed, area made
absolute, face dropped or adjacency repaired.

Because the harness builds both graphs before extracting any surface, no coarse
or fine elevation entered S0 and no highland topology or morphology result was
seen. There is therefore no evidence here about tablelands, caps, relief, stage
ownership or the quality of the legacy terrain.

## Reproducibility

- preregistration revision: `b218f41`;
- harness revision: `d19c100`, with diagnostic context in `471ab4a`;
- typed geometry diagnostic revision: `90df961`;
- worktree: clean before the final run;
- seed: 12345;
- backend: convex hull coarse, product adaptive clipped-Voronoi fine;
- requested resolution: 100,000 coarse cells and 250,000 fine cap;
- model/stages/cache: legacy product defaults, Stage 4, fine cache disabled;
- platform/build: WSL2 CPU, release;
- command:

```bash
/usr/bin/time -v timeout 20m cargo run --release --bin landform_baseline -- \
  artifacts/landforms/seed-12345-product-g0s0-250k-v0.json
```

The final diagnostic invocation exited in 49.19 seconds, including a 19.53
second release rebuild. `/usr/bin/time` reported 1,002,200 KiB maximum RSS for
the whole Cargo/compiler/binary invocation. This is not a retained-memory or
extractor-only measurement. No JSON artifact was published.

The earlier untyped run stopped with `InvalidSphericalGeometry`; a diagnostic-
only harness change localized it to fine G0, and a formula-neutral typed error
change then identified the exact cell, edge and reason. These retries used the
same frozen world and extractor configuration and exposed no morphology.

## What this establishes

- The product coarse convex-hull geometry reaches the current exact G0 seam.
- The 2,048-cell manufactured clipped-Voronoi sphere did not cover the
  adaptive fine mesh's degenerate-edge tail.
- The current fine `Tessellation` is sufficient for its existing approximate
  product consumers but is not yet a fully face-backed physical control-volume
  graph under the frozen G0 contract.
- Generic `InvalidSphericalGeometry` was too weak for product-scale diagnosis;
  the new typed cell-area reason adds provenance without changing acceptance.

The fine path is materially different from the coarse path. It uses the
clipped-Voronoi backend, copies `f32` source generators/vertices and cell cycles,
and does not preserve the backend weld map as product geometry authority. Fine
sites are also generated adaptively rather than as the regular manufactured
sphere. The observed degenerate edge is consistent with that representation
seam, but this audit does not yet assign one unique upstream cause.

## Required next checkpoint

Do not bypass G0 or tune landform thresholds. First extract a minimal geometry
witness for cell 199413 and its shared face, including source vertex IDs and
coordinates, ownership count, weld provenance and adjacent cell cycles. Then
choose an upstream correction according to the witness:

- canonicalize/deduplicate fine sites and all aligned fields if site identity is
  duplicated;
- preserve or correctly consume backend weld/effective-cell identity if the
  copied topology is stale;
- repair fine tessellation construction if it emits a genuine zero-length
  physical face; or
- preregister a more robust spherical-area predicate only if topology and
  positive face length are valid and the failure is solely a numerical fan
  test.

Any change which drops/reorders faces, repairs topology, changes tolerance or
changes the area predicate requires an explicit contract decision before the
product observation is resumed. A valid upstream implementation of the already
frozen physical graph requirements must still receive its own manufactured
regression and audit checkpoint.

The product G0/S0 observation remains incomplete. D0, O0, R0 and all H/C/G
terrain comparisons remain out of scope.
