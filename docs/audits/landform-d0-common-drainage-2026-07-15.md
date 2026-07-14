# Landform D0 common drainage audit

**Date:** 2026-07-15

**Verdict:** pass for the bounded common planar/testbed checkpoint

**Implementation:** `e142bd2c36638b4711abb50be32c72820e1feb74`

**Contract:** [D0 executable contract](../research/landform-object-packet-d0-2026-07-15.md)

## Result

The common D0 derivation now produces deterministic, arm-neutral drainage
evidence from a validated planar `EvaluationSurfaceGraphV0`, physical
cell-mean elevation and independently supplied runoff. The implementation does
not consume S0, arm identity, native C flux, G's authored graph, product river
selection or renderer state.

The checkpoint retains:

- a portal-seeded minimax virtual fill and exact-flat routing potential while
  leaving the supplied physical surface unchanged;
- one exhaustive deterministic cell-to-cell-or-portal receiver forest;
- structural area and runoff accumulation as separate compensated ledgers;
- outlet provenance and explicit conditioning depth, area, volume and parent
  hierarchy;
- independent 1,000, 2,000 and 4,000 km2 retained reach graphs, confluences and
  Strahler order;
- separate greatest-supply, longest and highest-order portal trunks;
- nested catchments, exclusive incremental ownership and one-copy raw
  catchment-boundary faces; and
- a D0-specific schema and deterministic evidence hash.

These raw boundary faces are partition seams, not yet geomorphic divides or
grouped ridge polylines. O0 owns that interpretation.

## Manufactured evidence

All eleven D0 tests pass. The matrix covers the required success and rejection
cases without product terrain or competitive H/C/G output.

| fixture | result |
|---|---|
| deterministic plane at 8/4/2 km | two portals retained; ledgers close; no conditioning; physical surface unchanged |
| exact flat at 8/4/2 km | deterministic potential resolves routing and retains both portals |
| nested fill at 8/4/2 km | depression hierarchy and conditioning debt retained without mutating the surface |
| resolved asymmetric Y at 8/4/2 km | 3 reaches, 1 confluence and maximum Strahler order 2 at every resolution |
| narrow portal/confluence | exact confluence ownership and incremental catchment closure |
| two basins | complete exclusive ownership and every raw partition face emitted once |
| distinct trunk roles | greatest supply `[2, 1, 3]`, longest `[4, 3]`, highest order `[0, 1, 3]` |
| malformed and numerical inputs | typed rejection of bad graph/input/configuration, missing or unknown portal, disconnection, cycle, ambiguous depression hierarchy and non-finite accumulation/conditioning |

The asymmetric Y emits 44, 90 and 246 raw boundary faces at 8, 4 and 2 km.
That growth is expected for ungrouped physical partition faces and is not a
claim of increasing geomorphic divide count.

An initial plane/tie fork fixture lost its apparent confluence at 2 km. It was
rejected rather than relaxed: the coarse-grid coincidence had not encoded a
resolved valley. The replacement prescribes an explicit Y-shaped surface and
retains the same reach topology across 8/4/2 km. This is a useful example of the
manufactured gate correcting the witness rather than tuning the extractor to a
grid accident.

## Corrections found during review

Pre-audit review caught and corrected evidence-invalidating edge cases before
this result was recorded:

- accumulation and portal ledgers now use compensated sums and reject finite
  input whose derived accumulation becomes non-finite;
- a depression with both parent-depression and portal exits is rejected as
  ambiguous, and same-component spill re-entry walks to the first distinct
  downstream component;
- non-finite conditioning area, depth or volume is a typed failure;
- raw boundary canonicalization preserves the endpoints and length from the
  same physical edge;
- coordinate ordering uses numerical floating-point order rather than raw bit
  order; and
- terminal portal ownership and exclusive catchment assignment are retained
  once, with memoized downstream resolution.

## Cost and verification

The warmed focused debug run completed the eleven D0 tests in `3.34 s`; the
whole command took `4.14 s` and reached `76,628 kB` maximum resident set size.
This measures the manufactured matrix, not a product-scale adapter or production
runtime.

- `cargo test`: **passed** — 246 library tests passed, 7 ignored, with every
  binary and integration target also passing.
- `cargo clippy --all-targets`: **passed** with the existing repository warning
  backlog; D0 introduces no remaining ordinary Clippy warning.
- `cargo build --release --bin hex3`: **passed**.
- `cargo fmt --all -- --check`: **passed**.
- `git diff --check`: **passed**.

## Interpretation and next boundary

D0 is fit for the common planar evidence role. It establishes a cheap shared
drainage topology and honest conditioning disclosures; it does not validate a
terrain arm or select a product hydrology model.

The representation is deliberately coarse: single-receiver, cell-centred and
without channel beds, widths, within-cell geometry, sediment or persistent
lineage. It has not been run through a separately specified product
`Hydrology` adapter. It also cannot by itself judge ridge/divide meaning,
longitudinal versus transverse valleys or cross-surface object correspondence.

The next principled rung is therefore an executable O0 preregistration using
the now-observed D0 representation, cost and limits. Do not compose H/C/G or
adapt product hydrology before that contract decides the minimum relationship
evidence actually needed.
