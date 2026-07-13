# Channel extraction R1a input and rank-precheck audit

**Date:** 2026-07-13
**Status:** exact inputs, immutable routing and visited-cell rank-conflict subgate pass; extraction pending
**Specification:** [Irregular-Voronoi seeded extraction R1a](../research/channel-extraction-r1a-2026-07-13.md)
**Geometry substrate:** [R1a G0 audit](channel-extraction-r1a-g0-2026-07-13.md)

**Later checkpoint:** Path extraction subsequently rejected both P0 and M0;
see the [path audit](channel-extraction-r1a-path-2026-07-14.md). This audit
retains the narrower passing status of the input/rank rung it records.

## Verdict

The registered affine A, resolved-valley V and broad-control B surfaces now
compile to exact projected-polygon means on the passing irregular Voronoi cap.
Each spacing, orientation, translation and surface is routed exactly once with
the conservative MFD face operator, and P0 physical-grade ranks and M0 outgoing-
fraction ranks are derived from that same immutable case.

P0 and M0 disagree throughout the domain and, more importantly, disagree at a
prescribed A/V head in six cases across the complete matrix. Every eventual
path must visit its prescribed head, so this passes the existential visited-
cell anti-alias subgate that R0 could not pass. It is stronger evidence than a
domain-only scan, but it does not select an arm.

This is **not** a pass of R1a. P0/M0 path tracing, neighbour/acyclic/portal
termination, F0 geometry, physical path gates and complete-extraction
repeatability remain unimplemented. B remains report-only.

## Implemented contract

`src/world/landscape/channel_extraction_r1a_fixture.rs` now owns:

- closed-form first and second polygon moments for A and V;
- exact B integration after clipping the polygon at both broad-valley
  shoulders, rather than evaluating a generator or centroid;
- unique convex-polygon ownership of the prescribed head, with boundary-
  adjacent heads rejected at the frozen tolerance;
- uniform `500 km/Myr` runoff multiplied by projected polygon area;
- one `FaceFlowCache::route_with_portals(..., MfdSlope)` result per registered
  case, retained with its source terrain and supply;
- P0 and M0 best/second ranks using the preregistered face ordering and margin;
  and
- donor, head-conflict and conservative-ledger audits.

The routed-case constructor accepts only the registered 8-spacing-guard cap.
The ten-spacing cap remains a geometry comparison and cannot accidentally
become a second physical input.

## Numerical result

All 36 spacing × orientation × translation × surface cases pass. The table
summarizes the four registrations at each spacing for each surface.

| spacing | surface | donor cells per case | domain P0/M0 conflicts per case | cases with a head conflict |
|---:|:---:|---:|---:|---:|
| 8 km | A | 1,053–1,056 | 198–225 | 1 / 4 |
| 8 km | V | 1,057 | 161–221 | 0 / 4 |
| 8 km | B | 1,056–1,057 | 164–219 | report only |
| 4 km | A | 4,152–4,159 | 802–891 | 0 / 4 |
| 4 km | V | 4,160 | 596–875 | 0 / 4 |
| 4 km | B | 4,159–4,160 | 623–855 | report only |
| 2 km | A | 16,498–16,513 | 3,210–3,725 | 3 / 4 |
| 2 km | V | 16,514 | 2,426–3,520 | 2 / 4 |
| 2 km | B | 16,513–16,514 | 2,605–3,547 | report only |

The broad fractions of conflicting donors—roughly 14–23% depending on surface
and registration—show that unequal face geometry creates a real discriminator,
not an isolated build-index accident. The six A/V head conflicts establish the
required visited-cell witness without assuming what either downstream path will
do.

Maximum global water-ledger relative error is about `1.18e-15`; maximum donor
fraction-sum error is `3.33e-16`. Maximum absolute cell closure residual is
`7.45e-9 km³/Myr` and passes the registered local-throughput-scaled tolerance.
Total supply is approximately `2.927e7`, `2.883e7` and `2.860e7 km³/Myr` at
8, 4 and 2 km respectively. V sends the complete supply to portal 401. A can
terminate at closed-domain sinks because its unconfined plane also drains
toward non-portal cap edges; rotated B cases can do likewise. Those sinks are
conservatively accounted for and are not yet evidence that either path arm
reaches the required portal.

Some local normalized margins are small (down to order `1e-6`). Exact frozen
ordering keeps the result deterministic, but those margins remain numerical-
dominance diagnostics rather than physical confidence.

## Invariant and determinism evidence

The audit rejects negative or non-finite fractions, fluxes, supplies, sinks or
aggregates. It checks donor fractions, exact stored face flux construction,
local incoming/source/available closure, donor-versus-sink exclusivity,
reciprocal-face direction, closed boundaries, non-target portals, per-portal
totals and global source/portal/sink closure.

Rebuilding every registered case on its already-built cap is bit-identical,
including input vectors, flow cache, heads, local ranks and audit. The cap is
unchanged by all case builds. This establishes routed-case repeatability; the
separate complete-extraction repeatability invariant remains pending because
no extractor exists yet.

Independent review found no remaining load-bearing correctness issue after
finite-value, portal-ledger, convexity, guard and asymmetric/oblique exact-
integration tests were added.

## Executed checks

```bash
cargo test --lib channel_extraction_r1a_fixture::tests -- --nocapture
cargo test --lib registered_full_input_matrix_passes_and_reports_rank_precheck -- --ignored --nocapture
```

The routine tests pass; the ignored full 8/4/2 km matrix passes and repeats
every case exactly.

## Next bounded step

Implement only the local P0 and M0 tracers over the existing immutable case and
construct C0/F0 polylines. Then enforce connectivity, acyclicity, eligibility
and portal-401 termination before interpreting the frozen physical path
metrics. Do not add M1, head discovery, persistence, confluence, width or
terrain feedback to this rung.
