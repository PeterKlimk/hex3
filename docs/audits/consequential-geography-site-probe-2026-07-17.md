# Consequential Geography site probe

Status: **first representative site prior rejected; access substrate retained**,
2026-07-17.

## Question and boundary

Does the first disclosed aggregate-site prior allow Stage-4 terrain, freshwater,
coast and Living Surface opportunity to constrain a varied, explainable set of
site anchors, or does the authored selection machinery determine the result?

This is a site-only discriminator. It does not evaluate routes, settlement,
population, history or a product default. The packet implementation is commit
`cccb716`; all recorded representative sidecars report that clean revision.

## Reproduction

The fixed panel uses seeds `12345`, `8675309` and `1001`, 100,000 coarse cells,
a 250,000-cell fine guardrail, Stage 4, the convex-hull Voronoi backend, disabled
fine cache and the legacy product terrain control. Each world is generated once
and reused for baseline, tight/loose and grade/freshwater/coast/living
counterfactual selections.

```bash
cargo build --release --bin hex3
target/release/hex3 \
  --sweep-stack consequential-geography --stage 4 --seed 12345 \
  --cells 100000 --fine-max 250000 --no-fine-cache \
  --sweep-width 768 --sweep-height 768 --sweep-rivers major \
  --out-dir sweep_out/consequential-geography/seed-12345-100k-250k-probe-v0
```

Replace the seed and output directory for the other two worlds. Each directory
contains 21 matched PNGs, a seven-by-three montage and
`consequential-geography.json`. The expected WSL Mesa/Vulkan adapter warning
appeared before the working offscreen fallback; all packets completed.

## Cost and completion

| Seed | Active cells | World generation | Access components | Baseline selection | Sites |
|---:|---:|---:|---:|---:|---:|
| 1001 | 255,238 | 21.67 s | 285.96 ms | 34.88 ms | 20/20 |
| 12345 | 255,866 | 27.02 s | 228.98 ms | 46.01 ms | 20/20 |
| 8675309 | 255,376 | 22.93 s | 219.48 ms | 33.46 ms | 20/20 |

All variants produced the requested 20 sites without catchment-visit or search
shortfall. The on-demand component and site operators are cheap relative to the
existing world generation. Compute is not the failure.

## Counterfactual signal

Values below are exact baseline-anchor retention and the median physical
distance from each variant anchor to its nearest baseline anchor. Nearest-site
distance is directional and does not solve bipartite matching.

| Seed | Tight | Loose | No grade | Uniform freshwater | No coast bonus | Uniform living |
|---:|---:|---:|---:|---:|---:|---:|
| 1001 | 80%, 0 km | 55%, 0 km | 10%, 311 km | 70%, 0 km | 10%, 198 km | 30%, 1,264 km |
| 12345 | 50%, 27 km | 45%, 29 km | 5%, 656 km | 40%, 31 km | 20%, 275 km | 50%, 337 km |
| 8675309 | 45%, 322 km | 35%, 260 km | 15%, 462 km | 45%, 16 km | 5%, 384 km | 55%, 0 km |

The factors are not numerically inert. Grade, coast and living ablations move
many anchors, while the freshwater null changes a smaller or more marginal
subset. This initially looks discriminating, but the baseline composition shows
that it is the wrong signal.

## Rejection evidence

Every one of the 60 baseline anchors is simultaneously:

- a selected-river freshwater source with zero freshwater access burden;
- an ocean-coast source with zero coast access burden and the maximum `1.2`
  coast multiplier;
- at effectively saturated local vegetation cover (mean `0.999999` to `1.0`);
  and
- far inside the grade gate (mean trimmed grade about `1.3e-5` to `1.6e-5`
  against a `0.15` limit).

The preliminary proposal score is the minimum living, freshwater and terrain
margin times the coast multiplier. A flat coastal selected-river cell therefore
receives almost the theoretical maximum:

```text
min(1, 1, approximately 1) * 1.2 = approximately 1.2
```

Each world has roughly 2,500--2,700 proposal local maxima, but only the first
160 spatially filtered members of that preference-ranked list receive a
catchment search. Catchment opportunity varies and does choose among survivors,
but it can choose only within a candidate support already dominated by joint
river/coast sources. Freshwater and coast then influence final admission again.

The large coast-ablation movement is consequently a candidate-support
bifurcation, not evidence that a modest coast benefit gently resolves otherwise
plausible alternatives. The nearby prior arms also retain only 35--80% of exact
anchors; some are geographically close substitutions, while a few move
thousands of kilometres. Greedy competition amplifies these support changes.

This fails the V0 condition that sites not merely collapse onto the largest
river/coast conjunction and that computational preselection not become a
second hidden site model. The marker packet is legible enough to expose the
problem, but it is not a product proof.

## Disposition and next discriminator

Retain the physical traversal/access components, bounded catchments,
competition records and packet. Reject this authored candidate proposal as the
product prior. Do not add routes, population or further content on top of it.

The next bounded correction is architectural rather than parametric:

1. Keep hard land, freshwater-reachability, living and grade viability.
2. Make computational candidate support preference-neutral and physically
   diverse instead of ranking it by the same freshwater/coast/site factors.
3. Apply factor preferences and catchment opportunity only after alternatives
   have reached evaluation.
4. Rerun the same three worlds with a tier census at eligible, proposed,
   catchment-passed and selected populations: joint freshwater/coast source,
   freshwater only, coast only, and neither-but-freshwater-viable.
5. Compare the corrected bounded pool with a 512-candidate diagnostic oracle,
   and keep candidate support fixed where an ablation is intended to isolate a
   final preference rather than eligibility.

If diverse support still selects only exact joint sources, then preselection
starvation is disproved and the next owner is semantic saturation: comfortable
near-water access or catchment relationships must replace exact source-cell
singular optima. Reducing `coast_bonus`, changing spacing or merely enlarging
the same preference-ranked pool does not test that causal distinction.
