# Organization-owner numerical and admission amendment V0

**Date:** 2026-07-16

**Status:** executable preregistration for H/C/G arithmetic, operator execution,
closure, completion, failure classification and campaign advancement; not
implemented and not an H/C/G result

**Parents:** [organization-owner comparison design V0](orogen-organization-owner-v0-2026-07-16.md),
[artifact/provenance amendment V0](orogen-organization-artifact-v0-2026-07-16.md)

**Accepted input:** [linked shared-input V0](orogen-linked-shared-input-v0-2026-07-15.md)

## Decision

Freeze four boundaries that must not be conflated:

1. **exact consistency** recomputes stored fields, reductions, hashes and
   sidecar bindings bit-for-bit;
2. **numerical validity** applies preregistered tolerances only to independently
   recomputed physical residuals and requires exact completion/invariants;
3. **campaign advancement** decides which valid or failed invocations may be
   scheduled next; and
4. **architecture evidence** reports resolution drift, numerical sensitivity,
   cost and morphology without retroactively changing whether an honest run
   completed its declared algorithm.

An ugly, flat, weakly organized, resolution-sensitive or expensive completed
surface is not a numerical failure. It remains evidence about the owner. A
forged residual cannot pass because its stored bits first have to equal a fresh
reduction; only then is the tolerance predicate applied. Deterministic replay,
not closeness, establishes execution by the declared algorithm.

This amendment introduces no new semantic artifact root. Its constants populate
the configuration fields already frozen by the artifact amendment, and its pure
predicates are called by predecessor/replay validation. The later comparison
artifact records their dispositions. Binary arm results and native provenance
remain the authorities for run state.

## Corrections to the parent boundary

Three parent requirements are narrowed before any result exists:

- G is work-matched exactly at its 4 km calibration resolution. Reusing the
  same amplitude at 8/2 km is a scale-transfer experiment. Its finite work drift
  is reported and may be material architecture evidence, but no arbitrary
  cross-resolution percentage turns an otherwise valid reconstruction into a
  solver failure or licenses recalibration.
- This amendment owns direct same-cell H/C base-versus-sensitivity reductions.
  O0b partners, missing objects, ties, splits, merges and quantity-specific
  object compatibility belong to the evidence/projection amendment because
  those evidence bytes do not yet exist.
- Common extraction and central projection occur only after arm results freeze.
  Their failure blocks the evidence instrument, not numerical validity of the
  already-published arm result.

The parent phrases “cross-resolution opportunity-volume admission band” and
“quantity-by-quantity correspondence reduction” are superseded accordingly.
This is not leniency: G still has exact forest/reconstruction invariants and a
tight 4 km solve, while unresolved scale drift remains visible rather than
being hidden by a pass/fail label.

## Arithmetic and tolerance vocabulary

The reference implementation uses one solver thread. Mesh, cell, edge,
boundary-face, portal, pass, step and finalization orders are the accepted or
explicitly registered orders. No parallel reduction, fused multiply-add,
extended-precision accumulator, compensated sum, global quantization or hidden
endpoint epsilon is permitted.

Except for the forcing compiler's declared f32 vertical-rate output and stored
f32 mesh geometry, arithmetic is IEEE-754 binary64 with round-to-nearest,
ties-to-even. Every expression below is evaluated in the shown parenthesized
order. Runtime positive infinity is permitted only as a local “no finite
limit” value and is serialized as `None`. Semantic zeroes are canonicalized to
positive zero only after the registered reduction; no nonzero value is rounded
to zero.

The one closeness predicate is:

```text
close(actual, expected; absolute, relative) =
  abs(actual - expected)
    <= absolute + (relative * max(abs(actual), abs(expected)))
```

The subtraction, absolute values, maximum, multiplication and addition occur in
that order. Both operands and both tolerances must be finite; tolerances are
nonnegative. The registered residual predicates are:

| residual | absolute tolerance | relative tolerance |
|---|---:|---:|
| H/C per-step solid moment | `1e-8 km3` | `5e-12` |
| H/C cumulative solid moment | `1e-8 km3` | `5e-12` |
| H/C instantaneous water rate | `1e-6 km3/Myr` | `5e-12` |
| H/C integrated water volume | `1e-8 km3` | `5e-12` |
| G reconstruction moment | `1e-8 km3` | `5e-12` |
| G 4 km work solve | `1e-8 km3` | `5e-12` |
| G runoff rate | `1e-6 km3/Myr` | `5e-12` |
| G accumulated area | `1e-8 km2` | `5e-12` |

For an identity stored as `error = actual-expected`, validation freshly
recomputes `actual`, `expected` and `error` with the artifact amendment's exact
orders, requires stored-error bit equality, then calls `close(actual,expected)`.
It never merely checks `abs(stored_error)`.

Exact invariants do not use this table: array lengths/orders, H target equality,
checkpoint coordinates, G forest termination, receiver legality and descent,
portal descent, owner counts, `N==+0.0`, nonnegative domains and deterministic
semantic repeat either match exactly or fail.

For one accepted H/C active-process step, freshly reduce `M_before` and
`M_after` in stored cell order and set:

```text
actual_step_change = M_after - M_before
H expected_step_change = +0.0 - denudation_export - hillslope_portal_transfer
C expected_step_change = uplift_moment - denudation_export - hillslope_portal_transfer
```

The per-step solid predicate compares those exact actual/expected values. The
cumulative solid predicate uses the H/C equations frozen by the artifact
amendment and freshly reduced final moment. Instantaneous water compares
`supply` with `portal_outflow+unresolved_sink` for every accepted active step
and for the active run's fresh final reroute. Integrated water compares the
artifact ledger's time-integrated values. Disabled controls own none of these
water or active-surface-process predicates. H target-only owns no step
predicate, but its cumulative solid predicate compares final-minus-initial
moment with gross hold restoration. C uplift-only owns its per-step and
cumulative solid identity, with expected change equal to uplift moment and
`process=None`.

## Exact H schedule and execution

### Cumulative activity primitive

For the accepted episode, let `r=0.25`, `T=6.0` and `I_total=5.75`, all in Myr.
The cumulative primitive is:

```text
t <= 0:
  I = +0.0

0 < t < r:
  s  = t / r
  s2 = s * s
  s3 = s2 * s
  s4 = s3 * s
  I  = r * (s3 - (0.5 * s4))

r <= t <= T-r:
  I = t - (0.5 * r)

T-r < t < T:
  s  = (T - t) / r
  s2 = s * s
  s3 = s2 * s
  s4 = s3 * s
  I  = I_total - (r * (s3 - (0.5 * s4)))

t >= T:
  I = I_total
```

This is the analytic integral of the accepted `3s^2-2s^3` end ramps. Registered
witnesses are:

| `t` Myr | `I(t)` Myr |
|---:|---:|
| 0 | `0` |
| 0.125 | `0.0234375` |
| 0.25 | `0.125` |
| 3 | `2.875` |
| 5.75 | `5.625` |
| 5.875 | `5.7265625` |
| 6, 8, 10 | `5.75` |

The implementation also requires `I(t)+I(T-t)==I_total` at the symmetric
interior witnesses under the shown evaluation and monotonic nondecrease over
every registered pass coordinate.

For pass count `P` and `k=0..P`:

```text
u_k = (k as f64) / (P as f64)
t_k = 10.0 * u_k

k == 0: p_k = +0.0
k == P: p_k = 1.0
otherwise: p_k = I(t_k) / I_total

target_k[i] = z0[i] + (p_k * D[i])
```

The endpoint branches are authoritative rather than relying on division to
recreate their bits. Base uses `P=200`; sensitivity uses `P=400`. Base pass 50
has `t=2.5`, `I=2.375` and mathematical progress `19/46`; base pass 120 and
sensitivity pass 240 have exact `p=1` through the `t>=6` primitive branch.

### Target-only control

H opportunity control executes every configured pass in order. Pass zero is
the unchanged initial checkpoint. At each pass, for cells in stored order:

```text
if target_k[i] > z[i]:
  addition = target_k[i] - z[i]
  z[i] = target_k[i]
  pass_hold_volume = pass_hold_volume + (addition * A[i])
```

`pass_hold_volume` starts at positive zero for each pass and becomes that
pass's ledger contribution. After the pass succeeds it is added once to
cumulative `gross_hold_restoration_km3`; target-only control uses the same
pass-local then cumulative ordering but publishes no pass/step trace.

There is no route, C0 step, process trace or operator exposure. The final array
must be bit-identical to the stored-order computation `z0[i]+D[i]`. Running all
passes, rather than replacing them with a direct final copy, is part of replay
and checkpoint identity.

### Hold-and-carve base and sensitivity

An H pass is one outer transaction:

1. clone the preceding committed surface and cumulative ledger;
2. apply the hold rule and hash the post-hold surface;
3. evolve zero-uplift active processes to the pass exposure endpoint; and
4. commit surface, ledger, trace range and checkpoint, if any, together.

Base exposure per pass is `0.05 operator-Myr`; sensitivity exposure is `0.025`.
The global operator coordinate is distinct from `t_k`:

```text
pass k < P: exposure_endpoint = (k as f64) * exposure_per_pass
pass k == P: exposure_endpoint = 10.0
```

Every process request is the lesser of `0.01` and the remaining exposure. H
uses an explicit no-uplift mode: it evaluates no forcing and owns no uplift
accuracy limiter. It must not emulate absence with an arbitrary large limit.
The final result is the post-carve surface of the final pass and is never
repinned.

Within each H scratch pass, every attempt computes the global-exposure midpoint
and then follows C's active attempt order starting at routing: minimum-dt check,
route, slope-Courant retry, scratch denudation/depth retry, linear-hillslope
stability retry, closure validation and scratch-step commit. Retry equality,
attempt 16 and trace metadata use the same rules as C. A successful manufactured
H retry must therefore produce an artifact-valid attempt vector even though it
never samples uplift.

The 100,000 accepted-step ceiling is invocation-global. It counts committed
steps plus accepted scratch steps in the current uncommitted pass. The cap is
checked before requesting another scratch step. If it fires inside a pass, the
pass is rolled back and its steps are absent from the committed trace, while
the `StepCount` witness retains the exact global work count of 100,000. Replayed
failure must reproduce both the shorter committed prefix and that witness.

Failure anywhere after the hold but before the pass endpoint discards the
entire scratch pass, including its hold, steps and ledger. The failure prefix
ends at the preceding pass as required by the artifact amendment.

## Exact event clipping

H exposure and C physical time share one endpoint driver. The ordered endpoint
sets are H's pass endpoints and C's `[3.0,6.0,10.0]`. Given exact committed
coordinate `x` and next endpoint `e`:

```text
remaining = e - x
requested = min(configured_maximum_dt, remaining)
```

`remaining` and `requested` must be finite and positive. Adaptive trials may
accept less than `remaining`. When an accepted dt bit-equals `remaining`, the
committed coordinate is assigned the registered endpoint bits `e`; the trace
still records `coordinate_end-coordinate_start == accepted_dt` under the same
subtraction. Otherwise the committed coordinate is the ordinary
`x+accepted_dt` and must be strictly between `x` and `e`. Crossing, epsilon
completion, silent snapping of a non-remaining step, or failure to land on an
endpoint is `EndpointNotReached`.

This explicit event assignment is coordinate bookkeeping, not a modification
of the physical update: every operator consumes the unmodified accepted dt.

## Exact C integration

C opportunity, base and sensitivity start at exact `time=+0.0`, `revision=0`
and the stored initial surface. Base/control requested maximum dt is `0.01 Myr`;
sensitivity is `0.005 Myr`. All clip first to 3, then 6, then 10 Myr.

Every candidate attempt follows this order:

1. if `candidate_dt < 1e-8`, fail before forcing evaluation;
2. compute `midpoint=start+(0.5*candidate_dt)` in that nonfused order;
3. freshly evaluate the stored scenario/compiler on the accepted mesh and
   consume its f32 vertical-rate array;
4. compute the uplift limit and retry when `candidate_dt > limit`;
5. apply uplift to scratch state in cell order;
6. for active runs, route exact stored local runoff, compute physical grade and
   slope-Courant limit, and retry when `candidate_dt > limit`;
7. apply denudation to scratch state, compute its realized depth limit, and
   retry when `candidate_dt > limit`;
8. evaluate/apply the linear hillslope operator, retrying when
   `candidate_dt > its stability limit`; and
9. validate the candidate, advance coordinate/revision once and commit.

Equality with any finite limit is accepted. A retry proposes
`min(previous_candidate,reported_limit)` and resamples forcing even when the
reported value bit-equals the previous candidate. Attempt 16 is executed; if it
returns a retry, that retry and proposed limit are recorded but the proposed
17th candidate is never tested. This is `MaximumAdaptiveAttemptsReached`.
All rejected candidate physical values are scratch and cannot affect state,
physical ledgers or final-routing evidence. Their attempt dt, midpoint, limiter
and reported limit remain mandatory chronological trace entries and therefore
do affect attempt counters and limiter reductions.

Opportunity control executes only steps 1--5 and 9. It performs no routing,
denudation or hillslope work and carries no process limiters. Base/sensitivity
use `UpliftThenRouteDenudeThenHillslope`. From the exact 6 Myr endpoint onward,
the compiler output and every stored uplift contribution must be positive zero.

For every C trial, scan f32 rates in stored cell order. The maximum absolute
rate owns the uplift limit `0.02/maximum_rate`, or no finite limit when the
maximum is zero. After that limit passes, apply and ledger in the same order:

```text
depth = f64(rate[i]) * candidate_dt
z[i] = z[i] + depth
uplift_moment = uplift_moment + (depth * A[i])
```

No f64 stencil or analytic activity value replaces the compiler's f32 output in
the actual C update.

The driver stops before accepting step 100,001. Reaching 100,000 accepted steps
without the exact endpoint publishes the replayable cause
`MaximumAcceptedStepCountReached`; it is not folded into `EndpointNotReached`.

## Frozen C0 active operators

### Stored runoff and routing

H/C route the exact predecessor `local_runoff_supply_km3_myr` array. The C0
adapter therefore accepts a validated supply slice and its component hash; it
does not regenerate `depth_rate*area`, even when the current accepted values
happen to compare equal.

The existing portal-seeded depression surface, equal-level potential, MFD-slope
partition, high-to-low accumulation, physical-face specific-discharge
reconstruction and unresolved instantaneous sink semantics are retained. The
derived routing surface never writes physical elevation. Final-routing evidence
is one fresh reroute of the final post-hillslope surface and is excluded from
accepted-step maxima.

### Effective areal denudation

Because the registered exponents are exactly `m=n=1`, the owner implementation
uses the specialized non-`powf` law:

```text
rate[i]  = (K * q[i]) * grade[i]
depth[i] = rate[i] * dt
z1[i]    = z0[i] - depth[i]
export   = export + ((rate[i] * A[i]) * dt)
```

All right-hand values are computed from the same pre-denudation surface and the
state update is simultaneous. Negative elevation is legal. There is no
base-level clamp or deposition.

The physical grade is the outgoing-flux-weighted mean of nonnegative physical
face grades; fill-only flat/uphill routes contribute zero. Directional length
is the outgoing-flux-weighted harmonic mean over those faces. For `n=1`, the
slope response used by the Courant limit is `(K*q[i])`, including at grade zero;
only positive response with finite directional length contributes a candidate
`0.25*length/response`. Zero response means no finite limit.

Both limiter reductions start at positive zero and scan cells in stored order.
The slope-Courant limit is the minimum finite positive candidate above, or no
finite limit if there is none. Any nonfinite candidate is an operator failure;
a finite nonpositive candidate is an invariant failure. For the depth limit,
compute every nonnegative finite `rate[i]`, reduce
`maximum_rate=max(maximum_rate,rate[i])`, and return `0.02/maximum_rate`, or no
finite limit when the maximum is positive zero. The fully computed scratch
denudation result and this limit exist before the candidate is accepted or
rejected. A reported infinity is encoded as `None`; no large finite sentinel is
legal. This retains the transactional ordering and trace identity.

### Conservative linear hillslopes

Add a separate linear operator; do not route through the current nonlinear
critical-slope denominator with a large threshold. Read the pre-step physical
surface only. For cells in ascending order and stored directed edges, visit an
internal face exactly when `neighbor > cell`:

```text
grade = (z[cell] - z[neighbor]) / f64(edge_distance_km)
rate  = (D_h * grade) * f64(edge_face_width_km)

volume_rate[cell]     = volume_rate[cell] - rate
volume_rate[neighbor] = volume_rate[neighbor] + rate

conductance = (D_h * f64(edge_face_width_km)) / f64(edge_distance_km)
conductance_sum[cell] = conductance_sum[cell] + conductance
conductance_sum[neighbor] = conductance_sum[neighbor] + conductance
```

Then visit boundary faces in stored order. Closed faces contribute nothing.
For an open face:

```text
grade = (z[cell] - f64(base_level_km)) / center_distance_km
rate  = (D_h * grade) * width_km
volume_rate[cell] = volume_rate[cell] - rate
portal_rate[portal] = portal_rate[portal] + rate
conductance[cell] = conductance[cell] + ((D_h * width_km) / center_distance_km)
```

Positive portal rate is export; negative is import. Portal accumulators are in
input portal-ID order. The maximum absolute grade covers every visited internal
face and every open portal face before the update.

After all faces, sum portal rates in portal-ID order. The internal conservation
residual is `(sum_cell_order(volume_rate)+total_portal_rate)*dt`. Convert each
portal rate to its trace volume with one `rate*dt` multiplication, and compute
total boundary transfer as `total_portal_rate*dt`; do not instead sum rounded
per-face volumes. Step solid closure then uses the exact actual/expected rule
above.

For each cell with positive conductance:

```text
cell_limit = (0.4 * A[cell]) / conductance[cell]
```

The stability limit is the stored-cell-order minimum, or no finite limit when
all conductances are zero. A candidate at the limit is legal. The simultaneous
candidate is `z[i] + ((volume_rate[i]*dt)/A[i])`; it is fully checked before
caller state changes. Internal transfer is antisymmetric by construction, and
only the ordered open-face rates enter the solid boundary ledger.

## C opportunity-control admission

The input-side f64 oracle is the selected predecessor cumulative-displacement
array `D`, not analytic `V`. For each cell in stored order:

```text
e[i] = (z_control[i] - z0[i]) - D[i]
signed_volume_error = signed_volume_error + (e[i] * A[i])
```

The opportunity control is usable as a base/sensitivity predecessor only when:

```text
abs(signed_volume_error) <= 1e-6 + (5e-7 * V)
max_i(abs(e[i]))         <= 2e-5 km
```

where `V` is the exact analytic declared-work bits. Its stored maximum, L1 and
RMS displacement errors are still recomputed exactly. The volume gate is about
reproducing the resolution-specific f64 input displacement through fresh f32
midpoint forcing; it is not a requirement that either quantity bit-equal
analytic `V`. The relaxation uplift ledger must be exact positive zero.

Before any linked 10 Myr run, a fixed process-disabled oracle uses four cells
with areas `[1,3,7,11] km2`, initial elevations `[0,1,-1,2] km`, and full-
activity f64 rates `[0,1/1024,1/8,1] km/Myr`. At every midpoint it evaluates
the analytic smoothstep activity above, multiplies by the full rate, casts that
product to f32, converts back to f64 and applies the exact C endpoint driver
with maximum dt `0.01`, uplift depth limit `0.02`, and endpoints `[3,6,10]`.
It has no retries and exactly 1,003 accepted steps under the registered endpoint
arithmetic.

The f64 analytic displacements are the full rates times `5.75`; their weighted
volume is exactly `68.298095703125 km3`. The emitted displacements must bit-
equal, in cell order:

```text
[0.0,
 0.005615234374438227,
 0.7187499999286835,
 5.749999999429468]
```

The signed emitted-minus-analytic volume error must bit-equal
`-6.776753602721897e-9 km3`; maximum absolute cell error must bit-equal
`5.705320660354118e-10 km`. These are surface-derived displacements
`z_final-z_initial`, not a separately accumulated uplift array. It must also
pass the linked control tolerances with fixture `V=68.298095703125`. Together
with the accepted input's independent
compiler frame/stencil gates, this exercises temporal integration and f32
conversion without using H/C/G morphology. Failure requires amending the
integration representation rather than fitting a looser linked-result
tolerance.

## Exact G construction

G is a fresh implementation under its own schema. Product erosion routing,
common D0 and independent extracted objects are not executable dependencies.

### Planning and queue order

Planning is the stored-cell-order operation `p[i]=z0[i]+D[i]`. It must remain
finite; no smoothing or fill mutates it. The queue key is:

```text
(path_maximum total_cmp, portal_id, candidate_cell,
 receiver_kind, receiver_index)
```

`receiver_kind` uses the artifact enum order: `Portal=0`, `Cell=1`.
`receiver_index` is portal ID for a seed and receiver-cell index otherwise.
Portal inheritance is part of every internal candidate key.

Evaluate all open-face seeds in ascending portal ID, then accepted boundary-face
record order within a portal. Multiple faces of one portal may seed the same
cell; every candidate is counted, but only a full-key improvement is pushed.
For a seed owned by cell `i`:

```text
path_maximum = max(f64(base_level_km), p[i])
receiver = Portal(portal_id)
```

After a cell `j` is finalized, evaluate its neighbors in ascending neighbor-cell
index, independent of CSR record order. For an unfinalized neighbor `i`:

```text
path_maximum = max(final_path_maximum[j], p[i])
portal_id = inherited_portal[j]
receiver = Cell(j)
```

Compare the complete key. A better key replaces the current best and is pushed.
A pop is stale when the cell is finalized or the popped key is not bit-identical
to its current best. A nonstale pop freezes the cell, receiver, inherited portal
and path maximum. No finalized state is revised.

If the heap empties before all `N` cells finalize, fail. After the `N`th
finalization, drain the heap without neighbor evaluation so all remaining pops
are counted stale and successful completion has `pop_count==push_count` and an
empty queue. Let `M` be the accepted directed-edge-record count and `S` the
open-face seed-candidate count. The structural bounds are:

```text
portal_seed_count == S
push_count <= S + M
pop_count <= push_count
relaxation_count == push_count
tie_replacement_count <= relaxation_count
maximum_queue_length <= push_count
```

Breaching a structural bound is `QueueInvariantFailure`; no arbitrary queue-
operation budget is added. At completion every cell rank is unique, each cell
receiver has lower rank, every chain terminates at the stored inherited portal,
the path-maximum recurrence is exact, and after the mandatory drain:

```text
pop_count == push_count
stale_pop_count == pop_count - N
```

An equal duplicate seed is counted but causes no replacement/push. Receiver
legality uses the donor-owned CSR record; selected portal roots own a matching
accepted open-face record; terminal chain portal equals inherited portal.

### Reverse accumulation and edge geometry

Initialize accumulated runoff and area from the exact stored local-supply and
cell-area arrays and `accumulated_cell_count[i]=1`. Traverse the reverse of
finalization order so every donor adds its accumulated runoff, area and count to
its receiver cell before that receiver is visited. This pass does not add
portal totals. Then, for each portal in ascending ID order, scan cells in
ascending stored index and add accumulated runoff/area/count exactly when that
cell's receiver is that portal. Integer count addition is checked for overflow.
Stored cell and portal totals must close under the registered runoff/area
predicates; portal owned-cell counts must sum exactly to `N` and equal the
native forest's terminal-portal assignment counts. Every accumulated area is
finite and strictly positive. Every cell's accumulated runoff/area bit-equals
its local value plus donor contributions in reverse-finalization traversal
order; its accumulated count equals one plus those donor counts in that order.

For an internal receiver, `ell[i]` is the f32 distance converted to f64 from the
accepted directed CSR record owned by donor `i` whose neighbor is the receiver.
Exactly one such record must exist. For a portal receiver, visit accepted
boundary-face records owned by `i` with the selected portal ID and take the
stored-order minimum `center_distance_km`; at least one must exist. Reciprocal
edge geometry is not substituted.

With `Q_ref=500000.0`:

```text
ratio = accumulated_runoff[i] / Q_ref
root  = sqrt(ratio)
numerator = D[i] * ell[i]
w[i] = numerator / root
```

Every accumulated runoff, ratio and root is finite and strictly positive;
`D`, numerator and `w` are finite and nonnegative; `ell` is finite and strictly
positive. No denominator floor is permitted.

### Exact `next_up` and reconstruction

`next_up(x)` accepts only finite canonical f64. For positive zero or positive
`x`, increment the IEEE-754 bit pattern by one; for negative `x`, decrement it
by one. If that decrement produces negative-zero bits (the
`-MIN_SUBNORMAL` case), return canonical positive zero. A result that is
otherwise nonfinite fails. Negative zero input is already rejected by semantic
validation.

For proposed finite `a>=+0.0`, traverse finalization order (portal/downstream
receivers before donors/upstream cells):

```text
Portal receiver:
  receiver_floor = next_up(f64(portal_base_level_km))

Cell receiver j:
  receiver_floor = next_up(z_a[j])

floor = max(z0[i], receiver_floor)
rise  = a * w[i]
z_a[i] = floor + rise
```

Multiplication and addition are separate. Fresh validation in ascending cell
order requires finite elevation, `z_a[i]>=z0[i]`, strict descent to an internal
receiver and strict descent to portal base. Negative added volume is exact
positive zero; violation counts are exact failures, not tolerated residuals.

### Frozen 4 km amplitude solve

The configuration values are:

```text
initial_upper_a_km_inverse = 0.001
bracket_growth_factor = 2.0
maximum_bracket_expansions = 64
maximum_iterations = 128
volume_absolute_tolerance_km3 = 1e-8
volume_relative_tolerance = 5e-12
```

Freeze the 4 km forest, accumulated runoff/area, `ell` and `w` before solving.
Every evaluation freshly reconstructs all elevations and reduces

```text
F(a) = sum_i(max(z_a[i]-z0[i],+0.0) * A[i])
R(a) = F(a) - V
```

in stored cell order. The exact solve is:

```text
lo = +0.0
F_lo = F(lo)
require finite F_lo and F_lo < V

hi = 0.001
expansions = 0
F_hi = F(hi)
require finite F_hi

while F_hi < V:
  if expansions == 64: CalibrationBracketFailure
  lo = hi
  F_lo = F_hi
  hi = hi * 2.0
  expansions += 1
  require finite positive hi
  F_hi = F(hi)
  require finite F_hi

require F_lo < V and F_hi >= V

for iteration = 1..=128:
  mid = lo + (0.5 * (hi - lo))
  if mid == lo or mid == hi: CalibrationIterationLimit
  F_mid = F(mid)
  require finite F_mid
  require F_lo <= F_mid and F_mid <= F_hi
  residual = F_mid - V
  if close(F_mid,V;1e-8,5e-12): return mid
  if residual < 0: lo=mid; F_lo=F_mid
  else: hi=mid; F_hi=F_mid
  require F_lo < V and V <= F_hi and lo < hi

CalibrationIterationLimit
```

`bracket_expansion_count` counts multiplications of `hi`, not evaluations.
`iteration_count` counts midpoint evaluations. The successful solve audit
stores the enclosing `lo`/`hi` that existed immediately before the accepted
midpoint and stores that midpoint's signed residual. There is no closest-
endpoint fallback, interpolation, Newton step or result-dependent initial
bracket. A zero-amplitude solution, all-zero `w`, nonfinite evaluation or
unbracketed root is a typed failure.

`F(+0.0)` must bit-equal the opportunity audit's
`next_up_only_positive_volume_km3`. Fixed nonnegative `w` makes `F` monotone;
every bracket/midpoint monotonicity assertion above is therefore an exact
implementation invariant, not an empirical morphology gate.

At 8/2 km, the exact 4 km amplitude bits are reused with a freshly constructed
same-resolution forest and reconstruction. The result reports `F(a_G)-V` and
all spatial moments. It is never recalibrated, normalized or failed merely for
finite scale drift.

### Pre-contract exploratory disclosure

During review, before this amendment was committed, an unregistered out-of-tree
probe applied the draft G algorithm to the accepted input. It suggested a 4 km
root near `0.0098470873 km^-1` and frozen-amplitude work drift of roughly `-28%`
at 8 km and `+40%` at 2 km. It produced no repository artifact, hash, audit or
admissible comparison result and must be rerun under the final contract.

This known diagnostic is disclosed so “preregistered” is not misread as blind
to likely scale behavior. It did not select a tolerance. Instead it exposed why
8/2 drift is architecture evidence about the graph/reconstruction
discretization rather than solver residual or permission to fit a broad gate.
The later evidence amendment must allow such drift to block any dependent
scale-robustness claim.

## Run-level validity

A successful run satisfies its exact consistency rules and every applicable
predicate below:

### H

- configured pass count and checkpoint set are exact;
- target-only final surface equals `z0+D` bit-for-bit;
- active H reaches exact `10.0` operator-Myr and every pass endpoint;
- every pass is wholly committed, trace ranges are contiguous and completion
  counters reduce exactly;
- active base/sensitivity per-step and cumulative solid/water identities pass
  their registered tolerance; and
- active base/sensitivity final routing is a fresh reroute of the final physical
  surface, while target-only stores no process/final-routing witness.

### C

- exact physical endpoint `10.0`, checkpoints `[0,3,6,10]` and no epsilon stop;
- final revision equals accepted-step count and trace/counters reduce exactly;
- every purpose uses fresh midpoint f32 forcing; only active base/sensitivity
  consumes exact stored runoff;
- opportunity controls satisfy their displacement gate or publish typed
  `OpportunityControlMismatch` instead of a successful result;
- forcing-interval and relaxation uplift ledgers split at exact 6 Myr, with the
  latter exact positive zero; and
- every owned per-step/cumulative solid and active-process water identity passes
  its registered tolerance.

### G

- planning, forest and reconstruction stages complete; finalization count is
  exactly `N` and the queue is drained;
- forest, portal inheritance, accumulation, Strahler/support, strict descent,
  nonnegative reconstruction and ownership invariants pass exactly;
- moment, runoff and area identities pass their registered predicates;
- 4 km returns the first accepted bisection midpoint; 8/2 bind and reuse those
  exact amplitude bits.

For all arms, individual semantic file caps and directory filename rules remain
artifact-validity requirements. No hidden clamp, forbidden operator, native-
evidence feedback, result-dependent normalization or post-run repair is legal.
A successful invocation is not campaign-admissible until a separate same-
process replay reproduces every semantic value, byte, hash and success/failure
sidecar. This applies to H, C and G; controls, bases and sensitivities; and every
resolution. Nonsemantic envelopes are compared only on their registered
reproducible subset. A mismatch retains both trees in the external audit and
blocks use/advancement without inventing an ordinary run-failure root.

## Numerical sensitivity is evidence, not validity

H base versus 400-pass sensitivity and C base versus half-maximum-dt sensitivity
are compared at 4 km after both independently pass run validity. For any paired
surface in accepted stored cell order:

```text
area_total = sum_i A[i]
delta[i] = z_sensitivity[i] - z_base[i]

signed_moment = sum_i(delta[i] * A[i])
signed_mean   = signed_moment / area_total
l1            = sum_i(abs(delta[i]) * A[i]) / area_total
rms           = sqrt(sum_i((delta[i] * delta[i]) * A[i]) / area_total)
maximum       = max_i(abs(delta[i]))
```

Each accumulator starts at positive zero; loops are stored cell order and the
shown multiplication/parentheses are exact. Report the same reductions for
every matched checkpoint: H pairs `[0,50,120,200]` with
`[0,100,240,400]` by normalized progress; C pairs exact physical times.

Also report base, sensitivity, signed difference, absolute difference and
symmetric relative difference

```text
2*abs(a-b) / (abs(a)+abs(b))
```

for every corresponding solid/water ledger scalar. The relative value is
`None` exactly when both denominator terms are zero. Accepted-step/attempt
counts, limiter histograms and minimum/maximum dt are paired diagnostic values,
not quantities to subtract into a physical “error.”

No universal percentage makes either valid result invalid. A later claim may
rely on an object or scalar only through the quantity-specific compatibility
predicate preregistered by the evidence amendment. A large sensitivity blocks
that claim and remains evidence about the owner; it does not permit selecting
the more attractive discretization as a new arm. G has no alternative pass/dt
run and instead requires exact repeat.

## Failure classification

The artifact failure schema is amended by appending cause variants
`MaximumAcceptedStepCountReached` and `OpportunityControlMismatch` and witness
variant `StepCount`; appending within each enum preserves every earlier
discriminant. Legal replayable phase/cause/witness families are:

| phase | legal cause families | witness |
|---|---|---|
| `HTarget` | nonfinite/noncanonical, length/order, internal invariant | `None` |
| `HCarve` | minimum dt, maximum attempts | `Timestep` |
| `HCarve` | maximum accepted steps | `StepCount` |
| `HCarve` | nonfinite/noncanonical, length/order | `None` |
| `HCarve` | routing, denudation, hillslope, boundary, revision, endpoint, internal invariant | `None` |
| `CForcing` | nonfinite/noncanonical, length/order, internal invariant | `None` |
| `CStep` | minimum dt, maximum attempts | `Timestep` |
| `CStep` | maximum accepted steps | `StepCount` |
| `CStep` | nonfinite/noncanonical, length/order | `None` |
| `CStep` | routing, denudation, hillslope, boundary, revision, endpoint, internal invariant | `None` |
| `GPlanning` | nonfinite/noncanonical, length/order | `None` |
| `GForest` | queue invariant | `GQueue` |
| `GForest` | forest termination, nonfinite, length/order, internal invariant | `None` |
| `GCalibrationBracket` | bracket failure | `GCalibration` |
| `GCalibrationBracket` | nonfinite or invalid invariant | `None` |
| `GCalibrationSolve` | iteration limit | `GCalibration` |
| `GCalibrationSolve` | nonfinite or invalid invariant | `None` |
| `GReconstruction` | reconstruction invariant | `GInvariant` |
| `GReconstruction` | nonfinite, length/order, internal invariant | `None` |
| `LedgerValidation` | ledger closure | `None` |
| `LedgerValidation` | opportunity-control mismatch | `None` |
| `CheckpointValidation` | noncanonical, length/order, internal invariant | `None` |

Every listed row has `ReplayableAlgorithmic`. The only legal observational
combinations are `ResourceCeiling` with matching wall-time, memory or artifact-
size cause and matching `Resource` witness; those have
`ObservationalResource`. Every other combination rejects. Timestep witnesses
use the last candidate/attempt information fixed by the artifact amendment.
Maximum accepted-step failure requires
`StepCount { accepted_step_count: 100000,
maximum_accepted_step_count: 100000 }` and is detected before another request.

`InvalidOperatorLimit` has no legal V0 combination. A computed/reported
nonfinite limit maps to `NonFiniteValue` with `None`; a finite nonpositive limit
maps to `InternalInvariantFailure` with `None`. A finite positive limit that
does not reduce a candidate is retained in the attempt trace and may eventually
cause maximum-attempt exhaustion.

Failure selection follows execution order and stops at the first failing check.
For H/C scratch candidates, check output shape/finite state, per-step solid,
instantaneous water, candidate cumulative solid and candidate integrated water,
in that order, before commit. C retains the preceding committed step on
failure; H rolls back the entire current pass. After a completed control,
opportunity displacement is checked before a success root. Checkpoints are
checked when constructed. Active final rerouting checks routing first and its
instantaneous water balance second. Thus simultaneous bad residuals have one
deterministic cause/coordinate.

Owner-path operators expose typed errors; no adapter parses `Display` text.
Failure at the forcing callsite maps to `CForcing`; routing construction maps to
`RoutingOperatorFailure`; specialized denudation maps to
`DenudationOperatorFailure`; an internal linear-hillslope failure maps to
`HillslopeOperatorFailure`; and an open-face geometry/flux failure maps to
`BoundaryOperatorFailure`. Length/nonfinite checks that precede those calls use
their dedicated causes. Invalid parameters and unavailable adapters are
configuration errors before a semantic run.

G witness construction is exact:

- a queue failure tied to a candidate uses `cell_index=Some(candidate)` and heap
  length immediately before the failing action. Post-drain/global counter checks
  use `None` and current heap length. Check counters in this order: seed count,
  push bound, pop/push equality, stale algebra, relaxation equality, tie bound,
  maximum-queue bound, empty heap;
- `F(0)>=V` uses calibration witness `(lower=0,upper=0,iteration=0,
  residual=F(0)-V)`. Expansion exhaustion uses current finite `lo`/`hi`,
  iteration zero and `F(hi)-V`; these are the current search points, not a
  falsely successful enclosing bracket. Midpoint stagnation uses current
  `lo`/`hi`, the about-to-run iteration number and `F(lo)-V`. Iteration
  exhaustion uses final `lo`/`hi`, count 128 and the last evaluated midpoint
  residual. Nonfinite paths instead use `NonFiniteValue/None`; and
- reconstruction scans cells ascending. Internal violation is `z_cell<=z_recv`
  with magnitude `max(z_recv-z_cell,+0.0)` and receiver `Some`; portal violation
  is `z_cell<=base` with magnitude `max(base-z_cell,+0.0)` and receiver `None`.
  The first ascending offender owns the witness. Counts cover all offenders and
  maxima reduce ascending; equality therefore legally yields `Some(+0.0)`.

Operator adapters map errors before execution begins. Free-form runtime text is
external detail. A candidate producing a bad numerical value fails
transactionally; it is never converted into a zero, clamp or successful partial
step.

## Resource stops

Campaign ceilings are unchanged:

| spacing | wall milliseconds | peak RSS bytes | retained directory bytes |
|---:|---:|---:|---:|
| 8 km | `1,800,000` | `2,147,483,648` | `2,147,483,648` |
| 4 km | `5,400,000` | `2,147,483,648` | `2,147,483,648` |
| 2 km | `21,600,000` | `2,147,483,648` | `2,147,483,648` |

A cooperative internal check may publish `ObservationalResource` with exact
observed/ceiling values. Replay returns `NotReplayable`. `/usr/bin/time -v` (or
the registered external equivalent) remains final wall/RSS authority; internal
elapsed time and VmHWM are witnesses. A kill, OOM termination or power loss
that prevents validated publication creates no semantic failure claim.

Per-file decoder caps are deterministic semantic safety limits. A decoder that
receives an oversized supplied file rejects it externally and does **not**
manufacture a semantic failure root. A cooperative generator that discovers
its would-be file or complete directory exceeds a registered generation budget
may instead publish observational `ArtifactSizeCeiling`. After constructing the
complete would-be success tree in memory, check every file with a registered
individual cap in lexical relative-path order, then sum every retained would-be
file in lexical order and check the 2 GiB directory cap. The first
failing check owns `Resource { kind=ArtifactBytes, observed, ceiling }`.
The semantic witness deliberately carries no filename; the nonsemantic stable
error summary may name it. These contexts are never interchanged.

If the external wrapper reports a wall/RSS breach only after a numerically valid
semantic success exists, retain that success unchanged and record external
campaign disposition `ResourceExceededExternal`. It is not rewritten as a
numerical failure or semantic resource root, but it cannot be used as a
predecessor or advance to the next rung. `WithinCeiling` from the final external
authority is required alongside exact repeat. Completing below a ceiling says
nothing about Pareto value, and a breach never licenses a larger budget or
coarser hidden settings.

## Manufactured and regression gate

All gates below pass before a linked 10 Myr run:

1. **H activity:** exact primitive witnesses, symmetry, monotonic pass progress,
   endpoint overrides, 200/400 common-progress correspondence and exact total
   operator exposure.
2. **H transaction:** target-only bit equality and checkpoints; one injected
   carve failure proves whole-pass rollback including hold, trace and ledger.
   A separate manufactured slope-Courant retry succeeds on its second attempt
   and proves the retained rejected-attempt metadata, unmodified pre-pass state,
   accepted scratch step, exposure coordinate and cumulative ledger all replay.
3. **Routing:** exact-flat two-portal drainage, known-sill nested bowls,
   byte-unchanged physical surface, stable portal IDs, deterministic repeat,
   water closure and zero physical denudation on fill-only uphill routes.
4. **Denudation:** specialized `K*q*S` equality, simultaneous update/export
   identity and the existing smooth 8/4/2 analytic convergence with 2 km
   relative error below `5e-4`.
5. **Linear hillslopes:** uniform closed identity, two-cell antisymmetric
   exchange, exact affine Dirichlet portal flux, closed quadratic response
   `4*D_h*c` with monotone 8/4/2 error and 2 km error below `2e-7 km/Myr`,
   accepted-at-limit/rejected-next-up-limit and byte-unchanged failure.
6. **C integration:** no-retry and every-limiter retry paths, fresh midpoint
   samples, attempt-16 exhaustion, exact endpoint clipping, stored-runoff
   consumption, exact post-6 zero uplift, control-versus-f64 opportunity oracle,
   transactionality and trace/counter reductions.
7. **Coupled short runs:** use the uniform planar hex fixture of width `64 km`
   and height `48 km`. For cell center `(x,y)`, freeze
   `s=1-(2*y/48)^2`, initial elevation
   `0.35+0.22*s+0.012*cos(pi*x/64)*max(s,0)`, and constant f32 forcing
   `f32(0.012*exp(-(x/18)^2-(y/12)^2))`. Freeze `K=1e-5`, `m=n=1`, runoff
   depth rate `100 km/Myr`, zero hillslope diffusivity and uplift/denudation
   depth limits `1 km`. The buffer weight is
   `w[i]=exp(-(x[i]/22)^8-(y[i]/15)^8)*A[i]`, reduced in cell order.

   At 4 km and `0.016 Myr`, requested dt `[0.004,0.002,0.001]` must satisfy
   `RMS(0.002,0.001) < 0.8*RMS(0.004,0.002)`, where paired RMS is
   `sqrt(sum_i(w[i]*(z_a[i]-z_b[i])^2)/sum_i(w[i]))` in cell order. At
   `0.012 Myr`, dt `0.001`, define buffered mean as `sum(w*z)/sum(w)`, buffered
   RMS as `sqrt(sum(w*z*z)/sum(w))`, uplift moment as
   `sum(w*f64(forcing)*time)`, and denudation export as
   `sum(w*(z_initial+f64(forcing)*time-z))`, all in cell order. Absolute errors
   versus the separately executed 1 km fixture must each decrease strictly at
   8→4→2 km. Then a 0.1 Myr accepted-linked-L prefix at 8/4/2 need only
   complete, repeat exactly and pass the frozen solid/water predicates; old
   nonlinear output magnitudes are not golden bytes for the new linear owner.
8. **G forest:** single chain, minimax diamond, symmetric two-portal tie under
   CSR permutation, equal-key duplicate seed, duplicate portal fragments,
   portal-ID-outer accumulation, fork accumulation/Strahler and every
   queue/path/portal invariant. Include a stale-key fixture that independently
   attains nonzero stale pops and replacement count.
9. **G reconstruction/solve:** `D=0` next-up chain; `next_up(+0)`,
   `next_up(-MIN_SUBNORMAL)` and finite-maximum edge cases; one-cell affine-
   volume root; tolerance root; monotonic bracket updates; no bracket;
   endpoint-stagnation; iteration exhaustion; strict descent; exact negative-
   volume zero; exact repeat identity; and one accepted-input 4 km bundle smoke
   that validates its forest, solve audit, reconstruction and ledgers without
   asserting a morphology magnitude.
10. **Validation mutation:** independently corrupt every residual, endpoint,
    limiter, counter, amplitude, forest rank and resource authority; repaired
    hashes do not bypass exact recomputation or numerical predicates.

The manufactured 8/4/2 G family reports frozen-amplitude work drift but has no
linked-result-derived pass band. Any later proposal for scale normalization is
a new architecture family and requires its own causal justification.

## Campaign advancement

After all manufactured gates pass:

1. publish valid H/C opportunity controls at 8/4/2 and exact-repeat each before
   it can be a predecessor;
2. publish and exact-repeat G 4 km calibration/base, then freeze its amplitude
   before any G 8/2;
3. run and exact-repeat H/C base at 8 then 4 km and G base at 8 km;
4. if H is invalid or resource-stopped at a required rung, block the cross-arm
   campaign; retain the failure;
5. advance H and each valid, non-resource-stopped C/G arm to 2 km, exact-repeat
   every result, and block the complete cross-arm campaign if required H 2 km
   fails; finite G 8 km work drift is reported, not an advancement veto;
6. freeze all arm results, then run and exact-repeat 4 km H/C numerical
   sensitivities; and
7. hand only frozen valid results and direct discrepancy records to the later
   evidence/projection amendment.

An algorithmic failure is not a weak result. A resource stop is not a numerical
failure. A valid but unappealing result is not dropped. Sensitivity can block a
later materiality claim but cannot retroactively erase a base artifact.

## Implementation boundary and APIs

Only after the remaining evidence and presentation amendments are committed,
implementation adds narrowly owned helpers:

```text
cumulative_linked_activity_myr_v0
organization_close_v0
validate_organization_numerics_v0
measure_organization_sensitivity_v0
solve_g_amplitude_v0
classify_organization_failure_v0
```

The linear hillslope mode, specialized owner `K*q*S` denudation/Courant path,
stored-runoff C0 adapter, explicit process-disabled mode, exact endpoint driver,
H outer transaction and G builder are implementation work. Existing generic
`powf` denudation, nonlinear hillslopes and scalar-runoff C0 entry points remain
separate research behavior; they are not silently changed into the owner path.

## Stop boundary

This amendment completes item 2 of the parent executable stop boundary. Do not
implement H/C/G yet. Next commit the evidence/projection amendment: exact common
core/O0a/central inputs, authored-G comparison, object and cohort reductions,
missing/tie/split/merge semantics, materiality predicates and comparison bytes.
The final planar capture/human-review amendment still follows before any active
arm run.
