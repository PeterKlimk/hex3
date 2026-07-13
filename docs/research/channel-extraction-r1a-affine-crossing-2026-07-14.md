# R1a affine continuous-crossing discriminator

**Date:** 2026-07-14
**Status:** preregistered; not yet evaluated
**Parent evidence:** [R1a generator-point control audit](../audits/channel-extraction-r1a-generator-control-2026-07-14.md)

## Decision

Test whether entry-point-aware continuous polygon crossing repairs affine A
after exact internal face-normal grades failed as a maximum-face graph walk.
Run two test-only arms:

- **X0 analytic oracle:** use the registered analytic downhill unit vector;
- **X1 reconstructed crossing:** reconstruct a local affine vector from exact
  polygon means at polygon-centroid locations, then cross actual polygon
  segments from the physical entry point.

This is a geometry and causal-localization experiment, deliberately close to a
manufactured solution. It is not a river model, a third P0/M0 receiver, a
general vector-field solver or a product candidate.

## Frozen scope

Use A only over the complete registered matrix:

```text
spacing h:    8, 4, 2 km
orientation:  0, 0.31 rad
translation:  0, 0.7 km
head:         one prescribed physical head at s = 176 km, n = 0
portal:       semantic OutletPortalId(401)
arms:         X0 analytic, X1 polygon-centroid reconstruction
```

Do not construct V or B, consume MFD fractions or discharge, add runoff, smooth
directions, use generator-valued terrain, add portal ghost elevations, or
implement RT0. The registered polygon-mean case supplies immutable A values and
the existing uniquely owned head; its water route is not a direction input.

## Checked polygon-face context

Build one test-only context per cap. For every cyclic segment `(a,b)` in
`cell_polygons_km[cell]`, compute the exact stored-arithmetic midpoint
`0.5*(a+b)` and match it to exactly one of:

- an internal CSR edge in that cell's edge range whose
  `edge_face_midpoint_km` is bit-equal; or
- a `boundary_face` owned by that cell whose `center_km` is bit-equal.

Do not use nearest-midpoint matching. Require polygon segment count to equal
CSR degree plus owned boundary-face count, consume every face exactly once and
verify every internal segment has a reciprocal neighbor segment with the same
bit-equal endpoints up to reversal. Segment index is never treated as a CSR
index.

Boundary meaning comes only from the stored `BoundaryFaceCondition`. Do not
reclassify the actual intersection by nominal x/y coordinates or split a
midpoint-classified portal segment analytically.

For every polygon, recompute signed area and centroid in `f64` from the complete
projected vertex loop. Require finite nonzero area and agreement with
`mesh.cell_area_km2` under the already-used `1e-10 h^2` area tolerance.

## Frozen X1 reconstruction

For each retained cell `i`, use all real internal face-neighbors in fixed CSR
order. With polygon centroid `c_i`, exact registered mean `z_i`, spacing `h`,

```text
r_ij = (c_j - c_i) / h
q_ij = (z_j - z_i) / h
M_i  = sum_j r_ij r_ij^T
b_i  = sum_j r_ij q_ij
g_i  = inverse(M_i) b_i
```

Use uniform weights and the literal `f64` 2×2 normal-equation solve. Do not use
generator coordinates, face widths, regularization, fallback stencils or a
different summation order. Reject the cell as singular when values are
non-finite or

```text
det(M_i) <= 1e-12 * trace(M_i)^2.
```

Scan every retained cell before tracing so linear-exactness is not conditioned
on the realized path. For analytic `g* = 0.01 u`, require positive alignment
and report the maximum

```text
relative vector error = |g_i - g*| / |g*|
sine-angle error      = |cross(g_i, g*)| / (|g_i| |g*|).
```

Both must be at most `1e-10`; no retained cell may be singular. X1 uses local
unit direction `d_i = -g_i/|g_i|`. X0 uses `d* = -u`.

## Frozen internal-score equivalence

Before calling another maximum-face P0 redundant, execute the comparison. On
every directed internal edge `i -> j`, use generator values `zgen` from the
prior control, projected generator connector `x_j-x_i` and the stored `f32`
distance promoted to `f64`:

```text
Sgen_ij = (zgen_i - zgen_j) / f64(edge_distance_ij)
Sx_ij   = -dot(g_i, x_j - x_i) / f64(edge_distance_ij)
```

Eligibility is the exact predicate `S > 0`. For each cell, rank only eligible
internal faces by greatest score, then the existing lexicographically smallest
physical midpoint `(x,y,z)` and finally directed CSR edge index. Compare best
face `Option<edge>` including the `None` case when no internal face is positive.
Report eligibility conflicts, winner conflicts, both normalized margins,
minimum `|Sgen|` distance to zero and maximum symmetric normalized score error

```text
|Sx-Sgen| / max(|Sx|, |Sgen|, f64::MIN_POSITIVE).
```

The equivalence gate requires zero eligibility and winner conflicts. Boundary
portal scoring is excluded and no redundancy claim may be extended to it. If
this gate fails, continuous crossing can still be evaluated, but it cannot
establish that a centroid-gradient maximum-face P0 would reproduce the prior
internal choice.

## Frozen traversal

Start at the exact registered physical head in its existing owner polygon.
Within cell `i`, intersect ray

```text
p + t d_i,  t > 0
```

with every actual polygon segment `a + u(b-a)`. For `e=b-a`, solve in `f64`

```text
den = cross(d_i, e)
t   = cross(a-p, e) / den
u   = cross(a-p, d_i) / den.
```

Append the smallest valid positive-`t` intersection. For an internal segment,
transition through its checked CSR identity and carry the reciprocal segment
as the next cell's excluded entry face. Keep the exact intersection point; do
not nudge it. Let `w` be the sign of the new polygon's signed area and `e` its
oriented reciprocal entry segment. Require

```text
w * cross(e, d_i) > 1e-10 |e|.
```

An absolute value within that bound is `TangentEntryAmbiguity`; the outward
side is `NonAdvancing`.

At a boundary segment, succeed only for stored
`OpenBaseLevel { portal_id: 401, ... }`. Closed boundary, wrong portal, missing
exit, non-advancing direction, repeated cell and cell-count guard are distinct
typed failures. Retain every valid prefix.

Use these registered numerical predicates:

```text
length tolerance tau_x = 1e-10 h
parameter tolerance     = 1e-10
```

- exclude the known reciprocal entry face topologically;
- require other exits to have `t > tau_x` and segment parameter within
  `[-1e-10, 1+1e-10]`;
- treat `|cross(d,e)| <= 1e-10 |e|` as parallel; if the ray is also within
  `tau_x` of the segment line, using
  `|cross(a-p,e)|/|e| <= tau_x`, return `CollinearAmbiguity`, otherwise ignore it;
- return `VertexAmbiguity` if an exit is within `tau_x` of either endpoint or
  two eligible exit parameters differ by at most `tau_x`; and
- never resolve an ambiguity by face, CSR or build index.

Report minimum vertex clearance and first/second eligible-exit parameter gap.
Iteration order must not decide a valid crossing.

The stored polyline is only

```text
physical head -> actual internal intersections -> actual portal intersection.
```

It contains no generators, face midpoints or analytic outlet point.

## Invariants and metrics

Repeat every context, centroid, reconstruction and trace exactly. Require
bit-identical outputs and no mutation of cap, polygon means, head, context or
registered case. Every successful path is neighbor-connected, face-continuous,
acyclic, has every local exit at `t > tau_x` and passes the frozen inward-entry
test. Global progress is measured only by the analytic-s backtracking metric;
X1 has no single global ray parameter. Each intersection must lie on its
declared segment within `tau_x` and on the reciprocal segment for an internal
crossing.

For every trace report:

- typed termination, visited cells and crossed face identities;
- actual crossing coordinates and maximum segment residual;
- maximum analytic cross-track `|n|`;
- Euclidean arclength and relative error from `176 km`;
- along-track backtracking;
- endpoint error from the analytic outlet and terminal signed `s,n`;
- minimum vertex clearance and exit-parameter gap; and
- for X1, visited and all-domain gradient errors.

Also report X0/X1 face-sequence equality, endpoint difference and metric
differences. Per-crossing coordinate differences are evaluated only when the
declared face sequences are identical; otherwise they are explicitly non-
evaluable. A different cell sequence is not automatically a physical failure
when both polylines meet the continuous geometry gates, but it must remain
visible.

Failures retain prefixes but have no full geometry score. Aggregate each
spacing over all four registrations with minimum, maximum, spread and censor
count. Paired causal geometry requires all four X0/X1 successes.

## Frozen gates and interpretation

Let a successful trace cross `N` faces. X0 must reach portal `401` in all 12
cases with maximum `|n|` and backtracking at most `tau_x`. X1 must reach it in
all 12 with maximum `|n|` and backtracking at most `(N+1) tau_x`. Both must have
relative arclength error below `5%` and no typed ambiguity.

Do not reuse R1a's strict `worst cross_2 < worst cross_8` clause. An exact ray
can sit at the numerical floor at both spacings. Compatibility with R1a's
absolute cross-track bound is reported, but line adherence and X0/X1 agreement
are the operator gates. Endpoint error and orientation/translation spread
remain report-only.

Interpret the ordered prerequisites:

1. If X0 fails, the polygon traversal, cap or portal contract is invalid; X1
   and graph-localization claims are uninterpretable.
2. If X0 passes but the all-cell reconstruction gate fails, this local centroid
   reconstruction is inadequate; continuous X1 geometry is not judged.
3. If reconstruction passes but X1 fails, local continuation, intersection
   numerics or a preregistered geometric degeneracy remains load-bearing.
4. X0/X1 agreement means only that both independently pass their frozen portal,
   line-adherence, length, backtracking and ambiguity gates; no post-hoc
   polyline distance is selected. If both pass while the internal-score
   equivalence gate passes, entry-point-aware crossing is sufficient to repair
   registered affine A. The maximum-face plus F0 graph representation **as a
   bundle** is then load-bearing for A.
5. If crossing passes but internal-score equivalence fails, the continuous
   convention is viable on A but the claimed redundancy of another local P0 is
   not established.

Even a complete pass cannot promote P0/M0, validate V or natural terrain,
establish a conservative flux/velocity field, identify a thalweg, justify a
general `H(div)`/RT0 reconstruction, or support confluence, persistence or C1
claims.

## Stop rule

Stop after the 12-case paired matrix and write a dated audit. Do not implement
V, RT0, a general field integrator or product integration in the same
checkpoint. The next rung must be chosen from the causal outcome, not assumed
from the expected near-manufactured pass.
