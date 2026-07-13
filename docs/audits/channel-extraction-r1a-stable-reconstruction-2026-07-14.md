# R1a stable affine-reconstruction control audit

**Date:** 2026-07-14
**Status:** completed negative solve control; registered row input remains
load-bearing
**Specification:** [R1a stable affine-reconstruction causal control](../research/channel-extraction-r1a-stable-reconstruction-2026-07-14.md)
**Parent evidence:** [R1a affine continuous-crossing audit](channel-extraction-r1a-affine-crossing-2026-07-14.md)

## Verdict

Stable QR does not rescue the registered polygon-mean reconstruction. RN and
RQ both pass 11/12 and fail the same `h=2 km, theta=0.31, delta=0` all-cell
gate. Both direct affine centroid-difference oracle arms pass 12/12 at machine
precision. The frozen interpretation is therefore that registered polygon-
mean/elevation differencing is sufficient to explain the remaining censor;
the normal-equation solve is not.

Every trace which is eligible to run reaches portal 401, passes the unchanged
geometry gates and has the same crossed-face sequence as X0. Every internal-
score equivalence audit passes. This strengthens the continuous-crossing
hypothesis but does not complete the physical-input causal claim: RQ, the only
candidate which preserves registered input while changing only the solve,
still fails its ordered reconstruction prerequisite.

Do not promote streaming QR from this result. On this well-conditioned affine
stencil it changes the registered result only at the floating floor and is not
a sufficient repair. Do not relax the `1e-10` threshold after observing the
miss, and do not substitute an oracle arm for registered terrain.

## Frozen matrix

| arm | row input | solve | reconstruction | judged crossing | score equivalence | X0-equal judged faces |
|---|---|---|---:|---:|---:|---:|
| X0 | analytic direction | none | n/a | 12 / 12 | n/a | reference |
| RN | registered differences | normal equations | 11 / 12 | 11 / 11 | 12 / 12 | 11 / 11 |
| RQ | registered differences | Givens QR | 11 / 12 | 11 / 11 | 12 / 12 | 11 / 11 |
| ON | direct affine row oracle | normal equations | 12 / 12 | 12 / 12 | 12 / 12 | 12 / 12 |
| OQ | direct affine row oracle | Givens QR | 12 / 12 | 12 / 12 | 12 / 12 | 12 / 12 |

The sole RN and RQ non-pass is not a crossing failure. Those two traces are not
constructed or judged. There are zero attempted traversal failures and zero
score eligibility or winner conflicts in the matrix.

## Numerical attribution

At the common failed cell 15875:

| arm | maximum relative vector error | maximum sine-angle error | maximum row residual | design condition |
|---|---:|---:|---:|---:|
| RN | `1.101601900e-10` | `1.031035947e-10` | `2.766390e-13` | `1.205859` |
| RQ | `1.101602506e-10` | `1.031035608e-10` | `2.766416e-13` | `1.205859` |

The stencil has degree three and determinant ratio
`det(M)/trace(M)^2 = 0.241440`, close to the isotropic two-dimensional maximum
of `0.25`; it is not near the frozen rank boundary. The largest registered
normal-versus-QR gradient change anywhere in the matrix is only
`1.044158e-15` relative to the analytic grade.

By contrast, the largest registered-versus-oracle gradient effect is
`1.101601e-10` under normal equations and `1.101604e-10` under QR, both at the
failed registration and cell. The direct oracle arms reduce worst all-cell
relative vector error to `6.940e-16` for ON and `6.952e-16` for OQ; worst
sine-angle error is at most `3.388e-16`.

Across the matrix, the largest absolute registered-row defect is
`1.139399e-12`, or `1.139399e-10` normalized by the analytic grade. The largest
row-length-normalized defect is `1.211849e-10`. The largest discrepancy between
a registered cell elevation and the report-only affine value at the checked
centroid is `1.374456e-12 km`, or `6.872281e-11` after normalization by
`|g*|h`. These magnitudes are physically negligible but numerically decisive
under the deliberately strict manufactured-solution gate.

This control does not distinguish polygon-moment accumulation, global-
coordinate centroid error, affine-value evaluation and cancellation in
`z_j-z_i`. It establishes only their combined registered input/differencing
side of the factorial.

## Implementation and review

The implementation remains entirely test-only. It preserves the committed RN
reconstruction bit-for-bit and leaves `trace_crossing`, `CrossingArm`, the
registered fixture and product code unchanged. It adds:

- direct registered and oracle row factors in fixed CSR order;
- a deterministic scaled streaming Givens solve without `A^T A`, pivoting,
  regularization or an external linear algebra library;
- the shared frozen normal-matrix singularity predicate for all arms;
- strict-first row, cell, residual, condition and pair-effect diagnostics;
- one X0 trace per case and independently censored RN/RQ/ON/OQ observations;
  and
- bit-repeat, immutability, known-Givens and RN-reproduction tests.

Independent review found no load-bearing factor, rotation, rank, diagnostic,
censoring, score or crossing defect. It found one non-load-bearing inherited
gate mismatch: reconstructed cross-track had been checked against the stricter
X0 `tau_x` rather than `(N+1) tau_x`. The implementation now matches both
preregistrations. All observed reconstructed traces already passed the stricter
accidental bound, so no count or interpretation changed.

## Consequence and next decision

The causal ladder now says:

1. the checked Voronoi cap, semantic portal and analytic continuous traversal
   pass registered A;
2. changing the local solve from normal equations to stable QR is insufficient;
3. making the affine neighbor rows algebraically consistent is sufficient for
   both solves and all crossings; and
4. the remaining formal blocker lies in numerical construction/differencing of
   registered affine means, not in path traversal or stencil conditioning.

If formal maximum-face/F0 graph-bundle localization is still worth earning,
the next smallest control is one preregistered local-coordinate affine-mean
evaluation. Standard numerical practice avoids cancellation in polygon moments
by translating each polygon to a nearby origin before accumulating moments.
That control should preserve polygon-mean semantics and ordinary absolute
registered elevations, change neither solve nor crossing, and compare its row
defects with RN/RQ. It must be a new checkpoint, not a post-hoc fifth arm here.

There is also a legitimate Pareto stop: the discrepancy is around
`1e-12 km` in elevation and has no physical or visual importance. If the formal
localization is not needed to choose the next architecture experiment, record
the manufactured numerical floor and return to the wider hydrology/terrain
decision rather than accumulating more solver machinery. What is not justified
is silently weakening this experiment's threshold or calling RQ a pass.

This audit does not promote QR, a continuous field, V, RT0, discharge
integration, a thalweg, confluence, persistence or product river extraction.

## Executed checks

```bash
cargo test --lib channel_extraction_r1a_affine_crossing -- --nocapture
cargo test --lib world::landscape::channel_extraction_r1a_affine_crossing::registered_affine_stable_reconstruction_reproduces_frozen_input_result -- --ignored --exact --nocapture
```

Routine tests pass. Full-matrix runs by the implementation worker, primary
agent and independent reviewer reproduce the same classification in roughly
`51–58 s`. A passing harness means it reproduces this negative solve control;
it does not mean RQ passed the 12-case reconstruction gate.
