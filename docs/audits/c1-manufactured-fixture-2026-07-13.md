# Minimum C1 manufactured fixture audit

**Date:** 2026-07-13
**Predecessor checkpoint:** `db9d14a`
**Status:** analytic representation gate passed; routed prototype not built

## Scope

This audit evaluates the contract in
[`../research/c1-manufactured-fixture-2026-07-13.md`](../research/c1-manufactured-fixture-2026-07-13.md).
The implementation is an isolated f64 compartment/geometry module. It does not
route water, select channels, infer width, carry sediment, form valleys or
modify the product world.

For each cell it prescribes cell area `A`, reach length `L` and active-channel
width `w`, then stores authoritative cell-mean elevation `z_bar` and channel-
surface elevation `z_c`. Active fraction is derived as `f_c=wL/A`; interfluve
mean is reconstructed from exact volume mixing.

## Implemented operators

- **Channel-only excavation:** applies prescribed `dz_c<=0`, changes `z_bar`
  by `f_c dz_c`, exports `-wL dz_c`, and leaves reconstructed interfluve mean
  unchanged.
- **Internal interfluve→channel transfer fixture:** raises `z_c` by `V/(wL)`,
  leaves authoritative `z_bar` bit-identical and therefore lowers reconstructed
  interfluve mean by `V/(A-wL)`. Compartment ledgers cancel exactly.

The transfer fixture proves algebra only. It has no sediment availability,
cover, transport-capacity or channel/interfluve ordering law and makes no such
claim.

All geometry, state and per-cell inputs are validated before mutation.
`wL>=A`, negative/non-finite geometry or change, and positive transfer into
zero channel area fail transactionally. With `w=0`, both channel operators are
inert and the authoritative mean reduces exactly to C0.

## Registered 8/4/2 result

The manufactured reach is 128 km long and 0.2 km wide inside a 16 km physical
swath. Four prescribed channel-lowering increments total `0.1 km`.

| Nominal spacing | Cells | Sum reach length | Sum channel area | Final z_c | Export |
|---:|---:|---:|---:|---:|---:|
| 8 km | 16 | 128 km | 25.6 km² | 0.5 km | 2.56 km³ |
| 4 km | 32 | 128 km | 25.6 km² | 0.5 km | 2.56 km³ |
| 2 km | 64 | 128 km | 25.6 km² | 0.5 km | 2.56 km³ |

Per-step elevation-volume-moment closure is within `1e-11 km³`; reconstructed
interfluve invariance is within `5e-16 km`; the internal-transfer net ledger is
exact positive zero; authoritative mean bits are unchanged by internal
transfer. Repeated runs are bit deterministic.

Seven focused tests cover the registered geometry/excavation gates, transfer
cancellation, zero-width reduction, invalid transactional behavior,
non-finite rejection and determinism. The complete landscape module has 80
passing tests at this point in the audit.

## Representation and cost verdict

The minimum C1 representation passes the failure that C0-Q16 could not address:
physical support belongs to a routed reach inside a cell rather than an
isotropic field crossing unspecified drainage divides. Neither channel
incision nor export contains grid spacing or a resolution-dependent
coefficient. The authoritative regional surface remains a conservative mean.

The irreducible dynamic cost over C0 is one f64 channel-surface elevation per
participating cell. Fraction is derived from prescribed physical geometry;
local excavation/mixing work is `O(N_channel)` and needs no global Helmholtz
solve. Network geometry and width ownership remain real missing systems, not
hidden inside the fixture.

This is a representation promotion only:

- **Pass:** fixed physical occupied area, channel evolution/export and mean-
  surface volume are invariant at 8/4/2 km.
- **Pass:** exact zero-width C0 reduction and transactional validation.
- **Pass:** internal compartment transfer bookkeeping.
- **Promote for the testbed:** `{z_bar,z_c,f_c}` is the minimum routed C1
  candidate and is preferable to isotropic Q16 for channel-local consumers.
- **Not implemented:** network/reach ownership, width law, lateral migration,
  alluvium/sediment, valley corridor, disturbance age and coupling to C0
  hillslopes/water.
- **Do not:** call `z_c` bedrock beneath sediment, infer width from cell size,
  treat internal transfer as deposition, add product/rendering state, or run
  long U/L yet.

## Next bounded gate

Before another coupled terrain run, attach the C1 core to one prescribed
conservative receiver network and prove:

1. reach length/channel area remain physical under 8/4/2 remeshing;
2. routed water `Q` and channel incision evolve `z_c` while `z_bar` closes by
   `A_c/A` mixing;
3. hillslope delivery is an explicit internal transfer rather than whole-cell
   erosion or untracked loss;
4. channel identity survives a simple confluence and forcing reorganization;
5. cost remains local enough to beat the rejected global support filter.

That is still a manufactured routed prototype, not Slice 2 semantics or
product integration.
