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

## Follow-up: factor-neutral support

Commit `5f93526` removes the preference-ranked local-maxima proposal. Hard
viability now defines an eligible population, and deterministic maximin squared
chord distance builds a physically dispersed support without freshwater, coast,
terrain-margin or catchment preference. Exact cell-ID ties remain disclosed.
The selection records every support anchor and relationship composition at the
eligible, support, catchment-passed and selected tiers.

The first corrected run used 160 candidates and eliminated the joint-source
collapse, but a 512-candidate diagnostic found 17--20% more total authored site
score and retained only 5--20% of the exact 160-pool anchors. A 160-point cover
is also spatially coarser than the 450 generalized-km catchment over planetary
land area. Since the 512 selection cost remained below 0.31 s, commit `1e32852`
makes 512 the probe baseline and retains 160 as the deliberately under-resolved
comparison. This is a compute/representation decision, not a changed factor
weight or product default.

The clean 512-support packets use the same worlds and commands above with
output directories ending `probe-v2-512-support`. Their baseline relationship
composition is:

| Seed | Joint river/coast | Freshwater only | Coast only | Neither exact source, freshwater viable |
|---:|---:|---:|---:|---:|
| 1001 | 7 | 11 | 0 | 2 |
| 12345 | 3 | 8 | 1 | 8 |
| 8675309 | 1 | 14 | 0 | 5 |

The original 60/60 joint-source failure falls to 11/60. Catchment evaluation
now receives a genuinely mixed support and selects some merely near-water
anchors. The 160 comparison remains inadequate: it retains 5--20% of exact
512-baseline anchors, has median nearest-baseline displacement of 361--667 km
and produces 13--15% less total authored score.

At 255,238--255,866 active cells, access components take 224--260 ms and the
512-candidate baseline selection 230--288 ms. All variants still return 20
sites within the visit budget.

## Corrected counterfactual interpretation

With adequate neutral support, removing local grade/traversal retains 90--100%
of exact anchors and removing the coast bonus retains 80--85%. Their dramatic
first-packet effects were primarily candidate-support artifacts. Uniform
freshwater retains 0--5%, while uniform living opportunity retains 5--10%; the
existing hydrology and catchment-scale Living Surface therefore have material
downstream consequences.

The result is not a frozen product prior:

- 55--90% of baseline anchors remain exact selected-river sources, and every
  nearest freshwater relationship in this panel is a selected river rather
  than a proper lake;
- site-local physical grade is effectively nonbinding on these lowland choices;
  terrain may matter more honestly in route geometry than in the anchor gate;
- the tight and loose arms change many hard gates and scales at once and retain
  only 0--25% of exact baseline anchors, so they are not evidence of modest
  one-parameter robustness; and
- 512 remains a bounded cover rather than exhaustive continuous optimization.

The correction passes the specific candidate-starvation blocker and is cheap
enough to serve as the provisional input to one bounded route discriminator.
It does not earn promotion, default status, population semantics or a claim
that the authored site prior is calibrated. Route work must retain these site
limitations and must demonstrate terrain-sensitive gaps/corridors rather than
using more content to conceal weak grade evidence.
