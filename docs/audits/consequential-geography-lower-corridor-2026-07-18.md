# Consequential Geography lower-corridor explanation

Status: **mechanical explanation accepted; geomorphic and product labels
rejected; visual communication mixed**, 2026-07-18.

## Question and boundary

Can the accepted terrain-sensitive route operator explain one changed path in
terms of existing physical terrain, without inventing a pass, gap, ridge or
chokepoint model?

Commit `7f28dd8` adds the bounded explanation. It compares the physical and
zero-grade paths for the same candidate endpoints, divides them only at common
split/rejoin cells in the same order, and rescores both branches with the
physical traversal. A branch is called a `lower-terrain-corridor` only when it
is strictly longer in physical distance, cheaper under physical traversal and
lower at its maximum effective elevation. Otherwise it emits a typed omission.

This is relative counterfactual evidence, not independent barrier topology. It
cannot establish a geomorphic pass or gap, route construction, travel time,
road reuse, a crossing or a chokepoint.

## Reproduction

```bash
cargo build --release --bin hex3
target/release/hex3 \
  --sweep-stack consequential-geography --stage 4 --seed 12345 \
  --cells 100000 --fine-max 250000 --no-fine-cache \
  --sweep-width 768 --sweep-height 768 --sweep-rivers major \
  --out-dir sweep_out/consequential-geography/seed-12345-100k-250k-probe-v4-lower-corridor
```

Replace the seed and output directory for `8675309` and `1001`. The schema-v3
sidecar retains the prior site and route packet and adds
`route_probe.lower_corridor_explanation`. When evidence exists, the directory
also contains `route-local-lower-terrain-corridor.png`: orange is the physical
branch, magenta the distance null, white rings are split/rejoin cells and small
colored crosses mark each branch's maximum-elevation cell. No explanation image
is emitted when the contract omits every selected candidate.

The packets were generated from the exact dirty worktree immediately before
checkpoint `7f28dd8`; their compile-time manifests therefore name parent
`8b3fa83` with `git_dirty: true`. `7f28dd8` is the clean commit containing the
tested source.

## Numerical result

| Seed | Physical-selected routes | Explained | Typed omissions | Strongest split/rejoin span | Extra distance | Physical cost saved | Ascent saved | Maximum elevation saved | Sampled separation |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 12345 | 19 | 7 | 9 exact, 3 not lower | 3,432 km | 22 km | 124 km (3.28%) | 9.77 km | 1.32 km | 326 km |
| 8675309 | 19 | 6 | 10 exact, 3 not lower | 3,163 km | 16 km | 116 km (3.33%) | 8.81 km | 1.64 km | 250 km |
| 1001 | 16 | 5 | 9 exact, 2 not lower | 5,891 km | 100 km | 350 km (5.17%) | 30.01 km | 2.98 km | 376 km |

All three fixed worlds contain several selected physical routes satisfying the
strict relationship. The strongest branch is not the whole endpoint-to-endpoint
route in any seed. Seed `12345` has 2.67% repaired-edge support on its strongest
physical branch; both branches of the other two strongest comparisons have
none. Repair therefore does not own the result. The support length remains a
conservative proxy: an edge's full length is counted when either endpoint was
lowered by drainage integration.

The split/rejoin spans are 3,163--5,891 km. These are broad continental routing
choices, not localized openings through a resolved barrier. The sampled
separation is measured only at retained Voronoi cell centres and is not a
continuous corridor width or polyline Hausdorff distance.

## Controlled evidence and validation

Nine focused fixtures cover the existing sparse route network plus exact-path
omission, a manufactured low opening, non-monotonic common-cell omission,
non-simple-path rejection and repair-provenance closure. The full test suite
passes: 148 library tests, 16 application tests, all binary tests and all
integration tests. `cargo check --bin hex3` also passes. The only reported
warning is the pre-existing unused test constructor in `tessellation.rs`.

Assessment validates matched networks, canonical land anchors, simple
Voronoi-adjacent paths, finite metrics and an exact zero-grade null. Elementary
divergences require common cells to occur in the same order; it does not use an
LIS or greedy repair to manufacture a comparison. Candidate, forest and extra-
link ties disclose their site-ID ordering; Dijkstra path ties remain dependent
on cell identity and adjacency order.

## Visual reading

Seed `12345` communicates the causal comparison well: the magenta distance-null
branch crosses the bright mountain belt while the orange physical branch takes
a longer lower route. Seed `8675309` shows broad parallel alternatives around
an elevated region, but the cause is subtle. Seed `1001` is visually ambiguous
at the automatically selected maximum-separation camera; local relief color can
make the orange route appear higher even though the complete branch's measured
maximum is lower.

This is not a reason for another camera or styling campaign. The scalar and
path evidence establish the route mechanism; two of three images show that the
relationship is often too broad or context-dependent to work as an automatic
cartographic annotation. Human visual-fidelity judgment remains authoritative.

## Disposition

Retain the core lower-terrain-corridor evidence and the optional diagnostic
image. It gives the route operator a conservative causal explanation and an
honest omission path at low marginal cost.

Do not promote `gap`, `pass`, `ridge crossing` or `chokepoint` labels, and do
not extend Consequential Geography into a relationship ladder, population or
culture on the strength of this result. Independent barrier/landform topology
and a more local consumer would be required for those semantics.

This completes the bounded route relationship decision. The higher-value next
portfolio move is the structural mountain-range problem: it is already the
largest established visual defect and better range organization could later
improve both terrain itself and any genuinely local crossings a consumer needs.
