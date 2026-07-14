# Product G0/S0 ancestry observation

**Date:** 2026-07-14, before any product surface entered G0/S0

**Status:** preregistered, not yet implemented or evaluated

**Parents:** [Landform object packet v0](landform-object-packet-v0-2026-07-14.md),
[G0/S0 executable contract](landform-object-packet-g0s0-2026-07-14.md)

## Decision boundary

Run one bounded, noncompetitive observation of the unchanged legacy product
terrain through the passing G0/S0 instrument. This is not R0, an H/C/G arm, a
promotion gate or permission to tune terrain or the extractor. It observes only
surface highland topology and morphology; D0 drainage and O0 relationships do
not yet exist.

The observation asks two descriptive questions:

1. Can the instrument execute at a useful product diagnostic scale and at what
   CPU/memory cost?
2. At which already retained terrain stage do the measured highland topology,
   broad-cap evidence and internal relief materially change?

Unfavourable, threshold-sensitive, nonlocal, ambiguous, empty or expensive
results remain evidence. No threshold, mask, stage or population may be changed
after output is seen without a separately committed amendment.

## Frozen product run

Use exactly:

```text
seed: 12345
coarse requested cells: 100000
coarse backend: convex-hull
Lloyd metadata: 1 (currently ignored by the convex-hull generator)
plates: NUM_PLATES_DEFAULT
orogen model: legacy
fine cap: 250000
fine scale and all terrain/climate/hydrology/erosion controls: product defaults
fine cache: disabled
computed stage: 4
build: release, WSL2 CPU
```

This seed preserves continuity with the existing causal and visual diagnosis.
The 250k cap is a diagnostic-cost slice, not the accepted 1M corpus resolution
and not a claim of resolution convergence. A 1M observation, other seeds and
H/C/G surfaces require later checkpoints.

The harness must record the complete `RunManifest`, actual coarse/fine cell
counts, cache record, command, output schema, platform description and build
revision/dirty state. Product generation timing is split into lithosphere,
atmosphere, fine-pre and erosion. G0 adaptation, each extraction and artifact
serialization have separate wall times. Whole-process peak RSS is captured by
`/usr/bin/time -v`; retained memory is explicitly unavailable.

The run is attempted with a 20-minute wall guard. Timeout, allocation failure,
typed geometry failure or invalid output is a valid scale/correctness result,
not grounds to reduce the resolution or alter the terrain. Do not attempt 1M in
this checkpoint.

## Frozen scored-domain and elevation policy

Every product cell is scored. S0 activity remains the already frozen strict
test:

```text
active = elevation_km > 0.0
```

Thus local relief may include below-datum coastal or ocean cells and the closed
sphere has no artificial scored boundary. This is an explicitly named product
observation policy, not the future competitive testbed mask.

For every observed surface, convert the authoritative native `f32` value once:

```text
elevation_km = f64(native_elevation) * f64(ELEVATION_UNIT_KM)
```

No hydrologic water level, renderer relief, camera, river importance, tectonic
field, native mountain component or per-surface normalization may enter G0/S0.
Use `SurfaceHierarchyConfigV0::default()` unchanged.

## Frozen ancestry surfaces

Build G0 once for the coarse tessellation and once for the retained fine
tessellation. Observe these five surfaces in order:

| ID | Geometry | Exact elevation source | Interpretation limit |
|---|---|---|---|
| `coarse-stage1` | coarse G0 | `World.elevation.values` | authoritative broad Stage-1 product envelope |
| `fine-base-raw` | fine G0 | `FineBase.base_elevation` | raw fine substrate supplied to Stage-3 hydrology |
| `fine-stage3-final` | fine G0 | `FineWorld.pre.elevation.values` | authoritative post-integration Stage-3 surface |
| `fine-stage4-raw` | fine G0 | `FineWorld.eroded.hydrology.pre_integration_elevation(i)` | eroded terrain supplied to final hydrology |
| `fine-stage4-final` | fine G0 | `FineWorld.eroded.elevation.values` | authoritative post-integration product surface |

Also report whether `FineBase.coarse_base_elevation` and `base_elevation` are
byte-identical at product defaults. Do not run a duplicate hierarchy merely
because they are equal. The four fine observations share exactly one graph, so
their changes are cell-aligned. The coarse and fine graphs are not compared by
cell ID.

## Frozen artifact

Write one atomic JSON artifact containing:

- schema version, observation identity, scored policy, frozen hierarchy config,
  manifest and timings;
- for each graph: domain/radius, cells, directed edges, physical area and G0
  adaptation time;
- for each surface: elevation hash, active cell/area fraction, raw peak, saddle
  and root counts, all five frozen population counts, derived-evidence hash and
  S0/morphology time;
- the complete reference-highland records plus their associated peak and key
  saddle fields;
- counts of flat maxima/saddles, equal-elder ambiguity, root/child features,
  local/nonlocal geometry, spherical nonlocal warnings, orientation ambiguity,
  rank-deficient grade cells, relief truncation and cap-merge censorship;
- union area of reference footprints, never the sum of nested footprint areas;
- object-weighted min/p10/p50/p90/max summaries for persistence, footprint area,
  two-sweep extent, mean width, equivalent ellipse length/width, anisotropy,
  each 25/50/100 km relief p50/p90, each 0.25/0.50/1.00 km cap area/fraction and
  each 0.5/1/2% gentle fraction; and
- the five largest reference objects by footprint area and the five most
  persistent, ties by peak ID.

Quantiles use the sorted finite object values and nearest-rank index
`ceil(q*n)-1`, clamped into the nonempty vector. Missing optional measurements
are counted and omitted from numeric summaries, never replaced with zero.
Counts and object-weighted summaries are descriptive; the artifact must contain
no aggregate quality score or pass/fail label.

The hierarchy builder's canonical double serialization remains the registered
determinism check. The observation stores its derived hash but does not perform
a second expensive extraction. Pretty-printing the complete raw hierarchy is
forbidden because nested footprints make it a poor product-scale artifact.

## Interpretation guardrails

- `HighlandFeatureV0` is an operational retained split-tree branch, not a
  promoted range or massif. A flat maximum is not a geomorphic plateau.
- Broad-cap evidence is continuous: summit-cap fraction, local relief and
  gentle fraction must be read together. No binary plateau classifier is added.
- Stage changes localize ownership only. They do not prove that a mechanism is
  realistic, useful or the unique cause.
- With no D0/O0, make no claim about basins, rivers, ridge/divide agreement,
  passes, valleys, range composition, correspondence or native graph quality.
- Product reference is not the idealized H control. G0 is the graph adapter and
  is not the graph-first G architecture arm.
- One seed and one capped resolution cannot validate the product, rank H/C/G,
  establish Earth calibration or justify a terrain rewrite.
- Renderer profiles and human visual evidence remain separate. Any later image
  comparison must declare physical versus presentation state explicitly.

After the artifact exists, write a descriptive audit which preserves the raw
result, cost and limitations. The only allowed next decision is whether the
instrument and observed distinctions justify a separate scale/seed checkpoint
or reveal a prerequisite defect.
