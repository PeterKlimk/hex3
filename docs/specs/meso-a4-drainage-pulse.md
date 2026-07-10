# A4 — two-stage drainage-aware uplift pulse (implementation hand-off)

Status: DESIGNATED NEXT ARCHITECTURE STEP (2026-07-10). Not started.
Parent: relief-spectrum-redesign.md (§12 consult, §13 addenda measured facts).
Consult origin: meso-design-consult-gpt56.md §4.3 (GPT 5.6; its preferred construction).

## 1. Why A4 is necessary (measured, 2026-07-10)

The uplift-shape channel hit a MEASURED ceiling: the emergent builder volume-
normalizes the uplift shape, so any volume the meso field carves (corridors) is
repaid as global uplift — peaks rise ~+2 km at meso 0.9 even at gain 1, and
massif-side caps cannot prevent it (only total meso depth moves peaks; verified
across MASSIF_CAP 0.35→0.2 with identical 12.4 km peaks). At the user's peak
budget (plausibility self-gate: ≤~12 km ok, 14+ absurd — encoded in
--mountain-audit) the channel tops out at ~1.6× baseline 25-km relief
(313 m @ 11.8 km, the m0.7/s50/g1 candidate). The Earth-alpine band (≥600 m at
25 km) is SERVO-UNREACHABLE in that channel. rebuild_gain is retired (three
independent confirmations of candidate-B self-similar scaling: relief bought by
gain arrives as peak height).

A4 escapes the coupling because erosion removes volume AFTER the height budget
is set: valleys deepen by incision along a pre-organized network, not by uplift
deficit that the normalizer refunds.

## 2. Design (consult §4.3 + our constraints)

1. **Burn-in epoch**: run the existing emergent build + erosion with the meso
   field OFF (macro envelope + broadband noise only), enough steps for drainage
   to self-organize (topology, not maturity — likely ~60-100 steps; the current
   default pipeline already produces this state, so v1 can literally reuse the
   standard stage-4 run or a shortened one).
2. **Extract** the stable order-≥3 drainage network + divides on the fine mesh
   (Strahler machinery exists in hydrology + river-audit; divides = watershed
   boundaries between major basins).
3. **Smooth** the network signal to 10-40 km (graph-distance falloff, ~like the
   corridor Gaussian widths; nothing below ~8 km — mesh floor ~3.9 km cells).
4. **Zero-mean uplift modifier**: lower uplift along major trunks (~0.5-0.7×),
   higher on broad interfluves/massif cores (~1.1-1.3×), mean-normalized per
   orogen so the servo is NEUTRAL (zero net volume change — this is the point).
5. **Final short epoch**: re-run erosion (steps ~50) with the modifier FROZEN.
   ONE feedback pass only — consult warning: continuous feedback locks into
   exaggerated spokes.

Relationship to the massif-corridor field (committed, style 1): A4's drainage-
derived scaffold REPLACES the lattice corridors (A3); the massif/saddle
component (A2) may survive as the interfluve-boost shape. Consult's alpine
recipe: ~10-15% structural grain, 30-35% massif/saddle, 50-60% drainage scaffold.

## 3. Constraints & gotchas (hard-won, do not rediscover)

- **Cache**: the burn-in→modifier→rerun happens EROSION-side (stage 4), but if
  any new param shapes the fine BASE it must enter fine_base_key. The modifier
  itself is erosion-state; a fresh FineSurface for the second pass keeps stage-3
  (pre-erosion) semantics intact. Mind the per-stage GPU buffer cache + staging.
- **Perf**: erosion is the dominant cost; two passes ≈ 2×. Burn-in at reduced
  steps + the ~4× speedup of steps 50 (vs 200) keeps total near today's default.
- **Determinism**: extraction must be order-stable (sort by cell index, not
  HashMap iteration — the June 2026 nondeterminism lesson).
- **Plausibility self-gate stays**: max range peak ≤12 km before ANY visual
  hand-off. Also gate: trunk-flow-orientation (transverse-leaning), crest-train
  spacing (no lattice spike — drainage spacing should be emergent), elongation,
  rivers (0 inland dead-ends), roughness proportionate to relief.
- **Identity**: modifier off ⇒ bit-identical to today. Sweepable depth dial
  (like meso_relief) + diagnose/main/overrides/sweep plumbing, per the
  meso_style template (commits 4396fd6, b8a580f, 14d669f).
- The all-cell flow-orientation split is hillslope-dominated and class-blind —
  use the TRUNK (top-decile accumulation) split.

## 4. Current state (for the fresh session)

Branch feature-scorecard (all pushed). Candidate default from the uplift-shape
channel: meso_style 1, meso_relief 0.7, steps 50, gain 1 — all gates pass both
seeds; user visual A/B pending (`--sweep-stack meso`). A4 proceeds regardless of
that verdict: the candidate is the floor, A4 is the path to real valley depth.
Instruments live in `diagnose --mountain-audit`: relief spectrum, crest-train
spacing, trunk flow orientation, plausibility self-gate, roughness+summit probes.
