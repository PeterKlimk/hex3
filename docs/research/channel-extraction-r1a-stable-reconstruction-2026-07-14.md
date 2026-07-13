# R1a stable affine-reconstruction causal control

**Date:** 2026-07-14
**Status:** preregistered; not implemented or evaluated
**Parent evidence:** [R1a affine continuous-crossing audit](../audits/channel-extraction-r1a-affine-crossing-2026-07-14.md)

## Decision

Determine whether the single censored X1 case in the affine continuous-
crossing experiment comes from the registered polygon-mean right-hand side or
from solving the local least-squares problem through normal equations. Run a
test-only two-by-two causal control over the same registered affine matrix:

| arm | neighbor-row input | least-squares solve |
|---|---|---|
| `RN` | registered polygon-mean differences | frozen normal equations |
| `RQ` | registered polygon-mean differences | streaming Givens QR |
| `ON` | direct affine centroid-difference oracle | frozen normal equations |
| `OQ` | direct affine centroid-difference oracle | streaming Givens QR |

`RN` is the bit-reproduced baseline, not a new candidate. `RQ` is the only arm
that could rescue the existing physical input by changing the solve alone.
`ON` and `OQ` are manufactured numerical oracles used only for attribution.
They are not product terrain states and cannot be promoted.

This is a numerical linear-consistency checkpoint. It does not introduce a
new river model, direction stencil, flux field, receiver, mesh, portal rule or
crossing convention.

## Frozen scope and shared state

Use affine A over the unchanged 12-case matrix:

```text
spacing h:    8, 4, 2 km
orientation:  0, 0.31 rad
translation:  0, 0.7 km
head:         one prescribed physical head at s = 176 km, n = 0
portal:       semantic OutletPortalId(401)
```

Reuse the checked polygon/face context, independently recomputed polygon
centroids, CSR order, analytic frame, exact registered case and continuous-
crossing implementation from the parent experiment. Do not alter polygon
moments, stored case elevations, routing, face geometry, head ownership,
boundary classification, traversal tolerances, ambiguity handling, metrics or
gates. Do not construct V or B, consume MFD fractions or discharge, add
regularization, change the stencil, use face weights, smooth directions, add
ghost elevations, implement RT0 or integrate a product path.

The registered case and cap remain immutable. The oracle is a derived neighbor-
row control and must never be assembled into an `R1RegisteredCase` or routed.

## Frozen row geometry and input factor

For every retained cell `i`, visit all real internal neighbors in fixed CSR
order. Reuse the parent scaled centroid connector and analytic gradient:

```text
r_ij = (c_j - c_i) / h
g*   = 0.01 u
```

The input factor changes only the scalar row right-hand side:

```text
registered R: qR_ij = (z_j - z_i) / h
oracle O:     qO_ij = dot(g*, r_ij)
```

`z` is the unchanged registered polygon-mean elevation. The oracle evaluates
the affine centroid difference directly; do not form two absolute oracle
elevations and subtract them. This deliberately removes polygon-moment and
absolute-elevation differencing loss together. It does not distinguish those
two subcauses from each other.

Before either solve, report over all directed internal rows:

```text
row defect              = qR_ij - qO_ij
grade-normalized defect = |row defect| / |g*|
row-normalized defect   = |row defect| / (|g*| |r_ij|)
```

Report the maximum of all three defect measures and the cell/edge identity
attaining each. Also form the report-only absolute centroid oracle

```text
zO_i = 1.0 + dot(g*, c_i - outlet)
```

and report maximum `|z_i-zO_i|` and maximum
`|z_i-zO_i|/(|g*|h)`, with cell identities. This does not enter either solve.
These quantities are diagnostics, not alternate gates.

## Frozen normal-equation solve

For `RN` and `ON`, use the parent literal `f64` solve without changing
summation order or arithmetic:

```text
M = sum r r^T
b = sum r q
g = inverse(M) b
```

Retain the existing singularity condition

```text
det(M) <= 1e-12 trace(M)^2
```

and all existing finite/alignment checks. `RN` must reproduce the committed
baseline reconstruction bit-for-bit, including the one failed all-cell gate.
If it does not, the control is invalid.

## Frozen stable solve

For `RQ` and `OQ`, solve the same unweighted row system `A g ~= q` through an
incremental Givens QR update in fixed CSR row order. Do not form `A^T A`, pivot,
reorder rows, rescale columns, regularize or call a platform-dependent external
linear algebra library.

Maintain the upper-triangular augmented state

```text
[ r00 r01 | y0 ]
[  0  r11 | y1 ]
```

initialized to zero. For each incoming row `(a0,a1 | beta)`, first eliminate
`a0` against `r00`, then eliminate the transformed `a1` against `r11`. For a
pair `(x,a)`, compute the frozen scaled hypotenuse

```text
scale = max(|x|, |a|)
rho   = 0                                          if scale == 0
rho   = scale * sqrt((x/scale)^2 + (a/scale)^2)   otherwise
```

and use `(c,s)=(1,0)` only when `rho == 0`; otherwise use
`c=x/rho`, `s=a/rho`. Apply the rotation

```text
[ c  s]
[-s  c]

top'    =  c top + s bottom
bottom' = -s top + c bottom
```

to the corresponding triangular/new-row entries and augmented right-hand
sides using ordinary `f64` multiplication and addition. Preserve the old pair
values until both rotated values are computed. This convention makes each
new diagonal nonnegative and fixes signs and operation order.

After all rows, apply the same precomputed normal-matrix singularity gate used
by the N arm, require finite positive `r00` and `r11`, then back-substitute in
this order:

```text
g_y = y1 / r11
g_x = (y0 - r01 g_y) / r00
```

Do not use explicit fused multiply-add. After solving every arm, recompute its
row residuals `q-r dot g` in CSR order and report the maximum absolute and RMS
residual. For QR also record `r00`, `r01` and `r11`. Compute the two eigenvalues
of the shared 2x2 `M` and report the cellwise design condition estimate
`sqrt(lambda_max/lambda_min)`, determinant ratio
`det(M)/trace(M)^2` and stencil degree at each arm's maximum-error cell. These
are report-only and must not select an arm.

## Reconstruction and pair diagnostics

For every arm, scan every retained cell before any trace. Retain the parent
definitions:

```text
relative vector error = |g_i - g*| / |g*|
sine-angle error      = |cross(g_i, g*)| / (|g_i| |g*|)
```

Require positive alignment, no singular cell and maxima at most `1e-10` for an
arm to be trace-eligible. Do not relax this tolerance. Report the maximum and
attaining cell for both errors, plus maximum all-cell gradient differences:

```text
|g_RQ - g_RN| / |g*|    solve effect on registered rows
|g_OQ - g_ON| / |g*|    solve effect on oracle rows
|g_ON - g_RN| / |g*|    input effect under normal equations
|g_OQ - g_RQ| / |g*|    input effect under QR
```

These paired effects are descriptive. They do not replace the independently
frozen all-cell gates.

Run the parent internal-score equivalence audit for every arm using its own
gradient. Preserve exact eligibility and winner predicates, internal-only
scope, generator comparator, stored-distance convention and tie order. Report
all conflicts and margins. A graph-bundle localization requires zero
eligibility and winner conflicts for the relevant physical-input arm; oracle
equivalence alone is insufficient.

## Frozen crossing evaluation

Run X0 once per registered case as an exact baseline reproduction. For each
reconstruction arm whose all-cell gate passes, run the unchanged continuous
crossing tracer with its local unit directions. If an arm fails reconstruction,
record `not judged` and do not trace it. Keep `not judged`, attempted traversal
failure and successful traversal as distinct states.

Preserve every parent traversal and geometry gate:

- required semantic portal 401;
- X0 maximum cross-track and backtracking at most `1e-10 h`;
- reconstructed maximum cross-track and backtracking at most
  `(N+1) 1e-10 h` for `N` crossed faces;
- relative arclength error below 5%; and
- no typed ambiguity or other failure.

Report face sequences, actual crossings, X0/arm sequence equality and all
parent metrics. Coordinate differences are evaluated only for equal declared
face sequences and remain report-only. An independently passing path need not
share X0's sequence to meet the geometry gate.

Repeat the complete factor construction, reconstruction, diagnostics and
traces exactly on the same target/build; require bit-identical results and no
input mutation. Do not claim cross-platform bit identity. For every reported
argmax, scan cells and CSR rows in their existing order and replace the stored
identity only on strict `>` so diagnostic ties have a fixed first-attaining
meaning.

Keep the committed parent matrix test unchanged. Add a separate ignored four-
arm matrix which preserves the existing `ReconstructionAudit` and
`CrossingArm` meanings, labels RN/RQ/ON/OQ only in an enclosing control result,
traces X0 once per case and censors each reconstruction independently. Do not
call the paired parent observer four times or aggregate unlike arms into one
spacing summary.

## Frozen interpretation

Interpret the complete matrix without selecting a tolerance or method after
seeing results:

1. If `RN` does not reproduce the parent baseline exactly, invalidate this
   control.
2. If X0 no longer passes 12/12, invalidate the reused crossing substrate and
   all reconstruction attribution.
3. If `RQ` passes reconstruction and crossing 12/12 while `RN` retains its
   single reconstruction failure, the normal-equation solve is load-bearing
   for the censor. If `RQ` also has zero internal-score conflicts, the earlier
   maximum-face/F0 graph-bundle localization becomes warranted for registered
   affine A.
4. If `RQ` fails while `ON` and `OQ` pass, the registered polygon-mean/
   elevation-difference right-hand side is sufficient to explain the censor;
   QR alone does not rescue the physical input, and graph-bundle localization
   remains unestablished.
5. If only `OQ` passes among the three nonbaseline arms, the result is an input-
   by-solve interaction: neither changing the registered input path nor
   changing its solve is sufficient alone.
6. If `RN` and `ON` fail while `RQ` and `OQ` pass, normal equations are strongly
   localized as the common numerical cause; `RQ` remains the required physical-
   input repair.
7. If both oracle arms fail, the manufactured row geometry, solve
   implementation or `1e-10` numerical contract is inconsistent; do not
   interpret the registered-input comparison. If neither QR arm passes, the
   factorial has not localized a sufficient repair.
8. `ON` versus `OQ` reports normal-equation loss on an exactly compatible row
   system. `RN` versus `ON` and `RQ` versus `OQ` report the combined registered-
   input/differencing contribution. Do not claim this control separates
   polygon integration, global-coordinate centroid error, affine evaluation or
   absolute-elevation subtraction. The row and elevation defects only identify
   where to inspect next.
9. A 12/12 `RQ` pass supports only the affine numerical and graph-
   representation diagnosis. It cannot promote a general QR reconstruction,
   validate V, establish conservative flux physics, identify a thalweg or
   justify product rivers.

## Stop rule

Stop after the complete four-arm affine matrix and write a dated audit. Do not
change registered polygon-moment construction, add a third input convention,
implement V/RT0/discharge/confluence/persistence or integrate any arm into the
product in this checkpoint. Choose the next rung from the causal outcome.
