# Channel extraction R1a path audit

**Date:** 2026-07-14
**Status:** completed; P0 and M0 both fail; select neither
**Specification:** [Irregular-Voronoi seeded extraction R1a](../research/channel-extraction-r1a-2026-07-13.md)
**Input checkpoint:** [Exact-input and rank-precheck audit](channel-extraction-r1a-input-rank-precheck-2026-07-13.md)

## Verdict

R1a is a valid negative experiment. Both registered local receiver arms are
deterministic, path-local and distinguishable, but neither passes the required
affine A plus resolved-valley V contract. P0 and M0 are therefore both rejected
for this role; R1a authorizes no seeded centreline mechanism or product change.

The failure is not a near-tie, relief-presentation or F0-midpoint artifact. The
affine control exposes a load-bearing mismatch between exact polygon-mean state
and the generator geometry used by the current two-point face operator. Some
rotated affine paths reach genuine closed-domain sinks. Terminating affine paths
still wander kilometres laterally. V is much closer, but it cannot rescue a
receiver that fails the simpler linear-consistency control.

Do not tune the valley, relax the gates, add M1 or apply RT0 to the current face
fluxes. The next decision must isolate and repair state/operator consistency
before paying for richer path geometry.

## Implemented tracer

`src/world/landscape/channel_extraction_r1a_path.rs` implements:

- one shared boundary-face index per cap;
- P0 and M0 traces that recompute both ranks from only each visited cell's
  faces and compare them exactly with the immutable case audit;
- strict CSR-edge ownership, neighbour, positive-grade/fraction and semantic-
  portal validation;
- path-sized cycle state, with no hidden domain pass in either tracer;
- typed sink/cycle/guard failures retaining every validated step, both rank
  margins, tie decisions and partial C0/F0 vertices;
- C0 generator-centre and F0 selected-face-midpoint polylines; and
- the preregistered signed transverse, arclength, outlet and backtracking
  metrics.

The complete outcome—success or failure prefix—is bit-identical on repeat.
Tracing does not mutate the cap, surface, supply, route, ranks or ledgers.

The rank helper is now a one-pass best/runner-up selection. It records whether
score, a physical midpoint component, portal identity or the final build index
decided each receiver. No successful A/V path or failed A/V prefix contains an
exact-score or build-index tie.

## Frozen F0 result

Ranges below are over successful orientation/translation registrations. The
failure column is over all 12 spacing × orientation × translation cases for
that surface/arm. A failure already disqualifies the arm; conditional metric
ranges do not conceal it.

| surface | arm | portal failures | 8 km worst cross-track | 2 km cross-track range | 2 km worst length error | verdict |
|:---:|:---:|---:|---:|---:|---:|:---|
| A | P0 | 6 / 12 | 18.919 km | 8.664–9.364 km | 6.915% | fail termination, cross-track, length |
| A | M0 | 4 / 12 | 30.774 km | 4.786–7.338 km | 5.162% | fail termination, cross-track, length |
| V | P0 | 0 / 12 | 5.156 km | 1.605–3.205 km | 4.816% | fail cross-track |
| V | M0 | 0 / 12 | 5.156 km | 1.789–3.220 km | 5.563% | fail cross-track and length |

P0's V result is close but not passing: the worst finest-grid cross-track error
is `3.205 km`, above the frozen inclusive `3 km` limit. M0 misses that limit by
slightly more and also exceeds the strict `5%` length limit. Both V arms improve
in worst cross-track error from 8 to 2 km. Every successful A/V path has zero
along-track backtracking; both arms pass the backtracking, orientation-
robustness and build-index-tie gates.

The affine failures are systematic:

- P0 reaches sink cell 0 for both rotated registrations at all three spacings;
- M0 does so for both rotated registrations at 8 and 4 km, but reaches portal
  401 at 2 km; and
- failed prefixes contain 34–138 selected faces, positive local margins and no
  exact-score ties. They are not an early guard or incidental index choice.

The broad B control remains non-gating. Nineteen of its 48 head/arm traces end
at sinks. Successful pairs often occupy broad, asymmetric transverse envelopes
or end on different sides of the analytic centre. This is compatible with the
control's intended non-identifiability and supplies no privileged thalweg
evidence.

## Why the affine control fails

For affine `z(p) = a + g·p`, the exact mean of polygon `i` is

```text
z_mean_i = a + g·c_i
```

at its area centroid `c_i`. The current two-point operator instead attaches
that value difference to Voronoi generators `x_i` and `x_j`:

```text
grade_ij = g·(c_i - c_j) / |x_i - x_j|
```

Writing `c_i = x_i + e_i` introduces the spurious directional term
`g·(e_i-e_j)`. G0 measured p95 generator-centroid offsets of `0.839`, `0.429`
and `0.215 km` at 8, 4 and 2 km: approximately a constant fraction of spacing.
The absolute perturbation shrinks, but it need not shrink relative to the
adjacent-cell affine drop, so local receiver ranks need not become linearly
faithful under refinement.

Ordinary Voronoi two-point flux is attractive when values are point samples at
the generators, because the generator connector is normal to the face. Exact
polygon means are located at polygon centroids instead; centroid connectors are
not generally face-normal. Merely replacing the denominator with centroid
distance would therefore be another diagnostic, not a principled conservative
fix.

A secondary limitation remains even after consistency is repaired: selecting
one maximum face from distributed MFD flow creates an irregular lattice walk,
not a continuum streamline. The similar C0 and F0 lateral drift shows that the
current failure already exists in the visited-cell sequence rather than being
created by face-midpoint encoding.

## Executed checks

```bash
cargo test --lib channel_extraction_r1a_path::tests -- --nocapture
cargo test --lib registered_full_path_matrix_reports_frozen_f0_gates -- --ignored --nocapture
```

The routine tests pass. The complete 8/4/2 km A/V/B matrix passes its audit
harness in `72.47 s`; an independent rerun passes in `66.83 s`. “Passes” here
means the implementation reproduces and correctly classifies the frozen
negative result, not that either receiver passes promotion.

Independent review found no remaining load-bearing tracer, indexing,
complexity, failure-reporting, metric or aggregation defect.

## Next principled discriminator

Preregister a compact causal ladder before more implementation:

1. As a report-only falsification control, evaluate affine generator-point
   samples with the unchanged two-point operator. If A becomes well directed,
   it isolates state-location mismatch; it does not authorize reinterpreting
   conservative polygon means as point values.
2. Preserve polygon means and test a centroid-based, linear-exact local gradient
   receiver projected onto actual face normals. This is the cheapest physically
   coherent candidate for repairing P0.
3. Only if M0 remains justified, test one shared conservative, linear-exact
   non-orthogonal face flux from polygon means—a reconstructed-gradient,
   mimetic or multipoint finite-volume rung.
4. Only after face fluxes are trustworthy, use a conservative `H(div)`/RT0
   field and continuous face crossings to test whether maximum-face lattice
   walking is the remaining limitation.

This sequence distinguishes state placement, flux consistency and sparse path
extraction without weakening the polygon-mean physical state or paying for a
richer mechanism before its causal need is established.
