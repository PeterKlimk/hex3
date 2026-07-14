# Product G0/S0 ancestry observation completion

**Date:** 2026-07-15

**Verdict:** product G0/S0 completes at the registered 250k diagnostic scale;
upstream geometry prerequisite resolved; descriptive ancestry evidence now
supports the existing terrain-ownership diagnosis

**Contract:** [Product G0/S0 ancestry observation](../research/landform-product-g0s0-observation-2026-07-14.md)

**Prior stopped audit:** [Product G0/S0 ancestry observation audit](landform-product-g0s0-observation-2026-07-14.md)

## Substrate resolution

Hex3 now pins `voronoi-mesh` revision
`e8804a639ea3c989e1ce9ea44b4c66c5f2d7e060` at Hex3 revision `2f35b80`.
The upstream revision detects and safely contracts exact stored-zero boundary
edges, including the minimized four-generator core extracted from the previous
Hex3 failure. Hex3 independently requires:

- no exact zero edges remaining in the backend output-resolution report;
- no zero-length edges in independent backend validation;
- strict topology, no unresolved repair signals or perturbation;
- complete, reciprocal, edge-backed native adjacency; and
- unchanged product spherical G0 area and positive-edge acceptance.

The focused adapter suite passes 10/10, including exact deterministic output,
effective post-weld identity and the natural-core contraction regression. The
full repository suite passes: 235 library tests with seven registered audit
tests ignored, all binary/integration targets, and both required release
binaries.

## Reproducibility

- Hex3 revision: `2f35b807065ef299b2dc3b3748d5e942571253e5`;
- worktree embedded in artifact: clean;
- seed: 12345;
- coarse/fine: 100,000 convex-hull cells / 256,847 adaptive fine cells from a
  250,000 cap;
- model/stages/cache: legacy product defaults, Stage 4, cache disabled,
  fine-cache schema 15;
- platform: WSL2 Linux x86-64, release CPU build;
- command:

```bash
cargo build --release --bin landform_baseline
/usr/bin/time -v timeout 20m target/release/landform_baseline \
  artifacts/landforms/seed-12345-product-g0s0-250k-v0.json
```

- artifact SHA-256:
  `d303be2dd0ad8426a5a7bd980bcdb09445dbc6eae0fceb18fe4bc89b4d92a799`;
- elapsed wall time: 114.59 seconds;
- maximum RSS: 1,242,016 KiB;
- product generation: 26.88 seconds;
- coarse/fine G0 adaptation: 0.84 / 1.79 seconds;
- five S0/morphology passes: 0.69 / 19.76 / 19.74 / 18.81 / 18.14 seconds.

Both graphs close to the same physical sphere area within roundoff:

| Graph | Cells | Directed edges | Area (km²) |
|---|---:|---:|---:|
| coarse G0 | 100,000 | 599,988 | 510,064,471.909791 |
| fine G0 | 256,847 | 1,541,064 | 510,064,471.909786 |

## Stage-localized result

The complete artifact retains all raw reference objects, hashes and evidence.
The compact ancestry summary is:

| Surface | Reference objects | Raw peaks | Persistence p50 (km) | Footprint p50 (km²) | Extent p50 (km) | Width p50 (km) | Relief p50 at 25/50/100 km (km) |
|---|---:|---:|---:|---:|---:|---:|---:|
| coarse Stage 1 | 78 | 317 | 0.248 | 56,401 | 460 | 169 | 0.000 / 0.000 / 0.609 |
| fine base raw | 108 | 2,653 | 0.232 | 22,170 | 266 | 89 | 0.132 / 0.349 / 0.787 |
| fine Stage 3 final | 108 | 2,707 | 0.232 | 22,170 | 266 | 89 | 0.135 / 0.349 / 0.787 |
| fine Stage 4 raw | 183 | 3,301 | 0.529 | 13,256 | 162 | 81 | 0.375 / 0.941 / 1.979 |
| fine Stage 4 final | 183 | 3,318 | 0.529 | 13,256 | 162 | 81 | 0.375 / 0.941 / 1.979 |

`FineBase.coarse_base_elevation` and `FineBase.base_elevation` are byte-
identical. At product defaults the fine structural layer therefore contributes
no terrain organization. Stage-3 hydrologic integration changes the elevation
hash and 54 raw peaks, but leaves the registered reference population and
nearly every object summary unchanged.

Erosion is the material morphology-changing stage in this packet. From Stage 3
to raw Stage 4 it:

- increases reference objects from 108 to 183 and raw peaks from 2,707 to
  3,301;
- more than doubles median persistence and 25/50/100 km local relief;
- reduces median footprint, extent and width; and
- changes median 0.25 km summit-cap fraction from 1.0 to 0.437.

This is consistent with erosion carving and subdividing the inherited broad
terrain rather than the fine substrate supplying an independent range
hierarchy. It does not establish that every new peak is desirable or that
erosion owns the original range envelope.

Final Stage-4 hydrologic integration is secondary at this object vocabulary.
It adds 17 raw peaks and reduces active area fraction from 27.11% to 26.83%, but
the registered reference count, persistence, footprint, extent, width and
25/50/100 km relief summaries are unchanged. Median anisotropy moves only from
0.566 to 0.575.

## Broad-cap evidence and cautions

The result does not define a binary plateau class. At the final surface:

- median 0.25 km cap fraction is 0.437;
- median 0.50 and 1.00 km cap fractions remain 1.0;
- median gentle fraction below 1% grade is 0.491 for the 0.50 km cap;
- median 25/50/100 km relief is 0.375 / 0.941 / 1.979 km.

Together these show that erosion adds substantial local relief and trims the
shallowest summit caps while broad, gentle highland support remains common at
deeper cap levels. This is compatible with the human-visible “long Uluru”
diagnosis, but one seed and an operational split-tree population do not prove a
global plateau rate.

Cap interpretation is censored for many objects: 90 of 108 fine-base reference
objects and 115 of 183 Stage-4 reference objects encounter at least one merge
limit. Each surface also records six spherical nonlocal warnings; all reference
objects nevertheless have local geometry, with no rank-deficient grade cells
or relief truncation. Read cap summaries with this declared censoring rather
than as an uncensored geomorphic census.

## What changes now

The product geometry prerequisite is resolved for this registered seed and
scale. The modern Voronoi substrate is promoted for Hex3's current physical
control-volume use, while Hex3 retains explicit rejection of reported
unresolved zero geometry. This promotes the substrate, not the terrain model.

The observation now supplies the missing descriptive product evidence for the
landscape comparison:

1. broad organization is already inherited from the coarse/base terrain;
2. product fine structure is absent;
3. pre-erosion hydrologic integration is not the organization owner;
4. erosion materially increases relief and segmentation but does not erase
   broad cap support; and
5. final integration is secondary to the measured range morphology.

D0 drainage relationships, O0 cross-object relationships, multiple seeds,
resolution convergence, H/C/G comparison and visual acceptance remain
separate checkpoints. No terrain tuning or architecture arm is promoted by
this one descriptive observation.

