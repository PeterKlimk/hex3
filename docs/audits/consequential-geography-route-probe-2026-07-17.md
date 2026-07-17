# Consequential Geography route probe

Status: **bounded terrestrial route operator accepted; corridor consequence
passes; endpoint-topology consequence not demonstrated**, 2026-07-17.

## Question and boundary

On the exact same provisional aggregate sites, does physical Stage-4 terrain
materially change a sparse terrestrial route network relative to a zero-grade
distance null, at a cost justified by the visible consequence?

This is not a road, travel-time, settlement-history or maritime model. It does
not evaluate population, trade, route reuse, bridges, passes, chokepoints or a
product default. Implementation commit `7627869` supplies the bounded operator,
fixtures and schema-v2 packet used here.

## Implemented causal slice

For every occupied landmass, the probe builds a bounded candidate graph from a
physical-distance spanning tree plus four symmetric nearest neighbors per site.
It runs one land-only multi-target Dijkstra search per canonical source site,
using physical great-circle edge length plus the direction-neutral half-round-
trip cost formed from the access substrate's disclosed asymmetric uphill and
downhill burden. Candidate paths retain ordered cells, physical length,
generalized cost, ascent/descent, maximum grade and exact drainage-repair
overlap.

A generalized-cost Kruskal forest supplies the backbone. At most three extra
links are admitted when the existing-network detour is at least `1.35` times
the direct candidate cost. Candidate pairs are capped at 96 and settled cells
at 10 million. Disconnected landmasses remain a forest and single-site
landmasses remain explicitly isolated. No water edge is traversable.

The matched null preserves the world, sites, landmass ownership, candidate
endpoint pairs and graph policy. It changes only uphill and downhill penalties
to zero. Consequently, changed paths isolate terrain burden; changed selected
endpoint pairs would isolate a network-topology consequence.

## Reproduction

The panel uses the same seeds and world controls as the corrected 512-support
site packet:

```bash
cargo build --release --bin hex3
target/release/hex3 \
  --sweep-stack consequential-geography --stage 4 --seed 12345 \
  --cells 100000 --fine-max 250000 --no-fine-cache \
  --sweep-width 768 --sweep-height 768 --sweep-rivers major \
  --out-dir sweep_out/consequential-geography/seed-12345-100k-250k-probe-v3-routes
```

Replace the seed and output directory for `8675309` and `1001`. Each directory
contains 30 matched PNGs, a ten-by-three montage and
`consequential-geography.json`. The first eight rows preserve the site panel;
the final two show physical and zero-grade routes over the exact baseline sites.
The regional camera targets the largest selected-path divergence. The expected
WSL Mesa/Vulkan adapter warning precedes a working offscreen fallback.

The packets were generated from the exact tested worktree immediately before
checkpoint `7627869`. Their compile-time manifests therefore name parent
`e028fe8` with `git_dirty: true`; `7627869` is the clean commit containing that
source. This audit is documentation-only relative to the executable result.

## Numerical result

| Seed | Active cells | Candidate paths | Selected routes | Exact selected paths | Selected edge Jaccard | Max selected Hausdorff | Physical / null ascent | Physical / null build |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12345 | 255,866 | 40 | 19 | 9/19 | 0.667 | 326 km | 32.2 / 48.4 km | 202 / 185 ms |
| 8675309 | 255,376 | 41 | 19 | 10/19 | 0.624 | 361 km | 34.8 / 50.6 km | 234 / 237 ms |
| 1001 | 255,238 | 34 | 16 | 9/16 | 0.404 | 376 km | 38.5 / 74.5 km | 221 / 223 ms |

All selected endpoint-pair sets are identical between physical and zero-grade
arms (`Jaccard = 1.0`). Terrain therefore does not earn a graph-topology claim
on this panel. It does materially control the corridors connecting that graph:
only 47%, 53% and 56% of selected candidate paths are exact, and the physical
networks use 31--48% less ascent while adding only 58--125 km across roughly
54,000--67,000 total route-km. Candidate-path median Hausdorff displacement is
20, 20 and 74 km; selected maxima are 326--376 km.

The two route builds settle 0.76--0.92 million cells each, far below the
10-million guardrail, and cost about 0.19--0.24 seconds at roughly 255,000
cells. This is cheap relative to the existing 22--27 second world generation
and comparable to one access or 512-candidate site pass.

Drainage integration is not hidden. Repaired edges contribute 1.9%, 2.4% and
1.9% of total selected physical route length. The largest single-route repaired
fraction is 11.3%, 7.8% and 14.1%. Repair participates in some corridors but
does not dominate the network-level terrain signal.

## Controlled evidence and visual reading

Focused fixtures pass for deterministic sparse connectivity, disconnected
landmass forests, explicit budgets, forward/reverse metric closure, repair
provenance and a manufactured barrier. Opening one low gap redirects the
physical path; closing it raises physical cost, while the zero-grade path is
unchanged.

The representative regional images communicate the same mechanism. Seed
`12345` is the clearest: the zero-grade line cuts across a bright range while
the physical line shifts to a visibly lower corridor. The other two worlds
show distinct matched corridors as well, although their causal terrain reading
is less immediate. This is preliminary inspection; final visual-fidelity
judgment remains human.

## Disposition

Accept the bounded land-only least-cost operator and matched packet as useful
on-demand Consequential Geography machinery. Terrain grade has earned a real,
cheap and visible downstream consequence in route geometry, even though it was
nearly inert in site-anchor selection.

Do not promote all of Consequential Geography V0 or freeze the authored site
prior. The panel does not show terrain changing network endpoint topology, and
the operator does not yet explain route-local gaps, crossings, junctions or
reuse. It also retains deterministic cell-ID tie debt and does not isolate peak
memory beyond establishing that no state is retained in `World`.

The next bounded decision should be functional rather than another parameter
sweep: derive one route-local explanation from geometry already computed—first
a lower-corridor/gap explanation with comparison to nearby alternatives—and
ask whether it adds legible board/globe meaning. If that relationship cannot be
made honest from current terrain and route evidence, retain the routes without
inventing labels and stop the slice there.
