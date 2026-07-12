# Minimum C1 manufactured fixture

**Status:** implementation contract; isolated representation test
**Date:** 2026-07-13
**Predecessor:** C0 support discriminator at `db9d14a`

## Decision being tested

C0-Q16 proves that a fixed physical support can stabilize local erosion
intensity, but its isotropic support has no drainage-divide ownership. The next
question is narrower than “build channels”: can a dual channel/interfluve cell
representation preserve physical occupied area, local channel evolution and
the authoritative cell-mean volume without grid-dependent coefficients?

This fixture tests only that bookkeeping and geometry. It does not route water,
choose a network, predict channel width, carry sediment, form valleys or
integrate with the product world.

## Minimum state and prescribed geometry

For each participating cell:

```text
A       physical cell area
L       physical routed reach length inside the cell
w       prescribed physical active-channel width
A_c     w L
f_c     A_c / A
z_bar   authoritative cell-mean surface elevation
z_c     channel-surface elevation
z_i     reconstructed interfluve mean elevation
```

with

```text
z_bar = f_c z_c + (1 - f_c) z_i
z_i   = (z_bar - f_c z_c) / (1 - f_c).
```

`A_c < A` is mandatory. The fixture represents an unresolved subgrid channel;
`f_c→1` is outside its regime. At `w=0`, channel operators are inert and the
state reduces to the authoritative C0 mean surface.

Width and reach length are prescribed physical geometry in this rung. They are
not inferred from mesh spacing or discharge. A later system may diagnose or
evolve them, but doing so requires a separate causal contract.

## Operators

### Channel-only excavation

For prescribed channel-surface lowering `dz_c < 0`, exported solid volume is

```text
V_export = -A_c dz_c.
```

The authoritative mean changes by

```text
dz_bar = f_c dz_c,
```

so reconstructed `z_i` remains unchanged. The operator records the exact
elevation-volume-moment change `A dz_bar` and export ledger. No whole-cell
incision or resolution-scaled `K` is allowed.

### Internal interfluve→channel transfer fixture

For a prescribed positive internal transfer volume `V_t`, lower the
interfluve compartment by `V_t / (A-A_c)` and raise the channel surface by
`V_t/A_c`. `z_bar` and total cell volume moment remain unchanged. Equal and
opposite compartment ledgers must cancel exactly.

This is a bookkeeping/manufactured operator, not a sediment model. Without
alluvial storage, grain flux and cover state, it must not be described as
deposition or transport capacity.

## Required 8/4/2 gates

Use a straight `128 km` prescribed reach, fixed physical width and exact
segment partition at nominal spacings 8/4/2 km.

1. `sum(L)=128 km` and `sum(wL)` are invariant near roundoff.
2. The same prescribed channel-lowering history produces identical `z_c(t)`
   and total export at every spacing.
3. `sum(A z_bar)` closes against channel export near roundoff.
4. Reconstructed `z_i` remains invariant under channel-only excavation.
5. Internal transfer changes channel/interfluve compartments equally and
   oppositely while leaving `z_bar` byte-unchanged.
6. Zero width leaves the mean state byte-unchanged and exports zero.
7. `A_c>=A`, non-finite geometry/rates, negative transfer and positive transfer
   into zero channel area fail before mutation. This bookkeeping rung does not
   invent a sediment-availability or channel/interfluve ordering law.
8. Repeated runs are deterministic.

No composite score may hide a failed invariant.

## Cost and state claims

The irreducible dynamic addition over C0 is one channel-surface elevation per
participating cell. Physical fraction may be stored or derived from prescribed
`wL/A`; reach direction and length can remain network geometry. The local
mixing/excavation work is `O(N_channel)` and requires no global scalar filter.

This is only erosion-geometry-ready. Honest extensions remain separate:

- sediment: bedrock channel elevation plus alluvial cover/storage;
- valley/floodplain: a distinct corridor fraction or width greater than or
  equal to the active-channel fraction;
- ecology: corridor disturbance history/age if succession matters;
- lateral mobility: an evolution law for corridor occupancy, not reuse of
  active channel width.

Passing this fixture permits a bounded routed C1 prototype. It does not promote
C1 into the product or justify those extensions automatically.
