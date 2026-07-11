> **Archived:** Historical design or experiment record. It does not define current architecture, defaults, or priorities. Start at [`docs/README.md`](../../README.md).

# A4 — two-stage drainage-aware uplift pulse (implementation hand-off)

Status: IMPLEMENTED + gate-validated 2 seeds (2026-07-10, same day). See §5.
User visual pending.
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

## 5. Implementation + measured results (2026-07-10)

### What was built (erosion-side only; fine base + cache untouched)

- `erosion::drainage_pulse_modifier`: SFD routing of the burn-in surface →
  wet-area accumulation → Strahler orders on channel cells (channel-initiation
  gate = `channel_support_km2` floored at 1 km²) → trunk mask (order ≥
  `EROSION_PULSE_TRUNK_ORDER` = 3) → multi-source Dijkstra graph distance →
  Gaussian falloff t (σ = `pulse_smooth_km`, floored at 8 km) → modifier
  `1 + depth·(0.25·(1−t) − 0.4·t)` clamped ≥ 0, then mean-normalized PER OROGEN
  (connected component of shape > 0; Σ area·shape·mod = Σ area·shape) so the
  volume-normalizing builder's between-orogen distribution is untouched.
  Deterministic (index-ordered traversals). No trunks ⇒ None ⇒ unmodified shape.
- `FineSurface::generate`: when `drainage_pulse > 0` (+ emergent + shape):
  burn-in `erode()` at `pulse_burnin_steps` (80) with the unmodified shape,
  extract modifier, run the normal erode↔precip loop with shape×modifier from a
  fresh `structured_base`. Modifier FROZEN (one feedback pass, per consult).
- Knobs: `drainage_pulse` (0 = off), `pulse_burnin_steps`, `pulse_smooth_km` in
  ErosionParams/constants + full main/diagnose/overrides/sweep plumbing.
- Tests: tree-graph Strahler trunk suppression + per-orogen volume conservation;
  chain-graph → None fallback. Identity: pulse-0 export BYTE-IDENTICAL to HEAD
  (seed 4242, cells 20k, no-fine-cache). Full suite + clippy clean.

### Measured (s50 g1, full res; 25-km p95-p05 p50 @ max range peak)

Seed 12345 depth ladder at σ15, meso 0: 193 @ 8.9 (base) → 210 @ 9.4 (p0.5) →
226 @ 9.9 (p1.0) → 242 @ 10.4 (p1.5). LINEAR: +32 m and +1.0 km peak per unit.
Seed 777 replicates (+18% @ +0.9 km at p1.0).

KEY FINDING — volume neutrality ≠ peak neutrality: the servo refund is gone
(this channel is volume-neutral by construction), but the interfluve BOOST lands
on massif cores, so peaks still rise. The coupling is via WHERE the boost sits
(a design DOF), not the normalizer.

σ bracket @ p1.5 (12345): σ8 = 234 @ 9.8 (46 m relief per km-peak, BEST); σ15 =
242 @ 10.4 (33); σ25 = 257 @ 11.1 (29 — wide Gaussian ⇒ renorm inflates boost
to 2.4× ⇒ crest inflation). NARROW suppression carves trunk valleys without
paying peaks ⇒ σ = 8 km (the mesh floor).

Depth ladder @ σ8 (12345): p1.5 234 @ 9.8 → p2.5 252 @ 10.2 → p3.5 272 @ 10.4.
Suppress side saturates (trunk cores reach zero uplift, modifier 0.00..3.95)
but peaks nearly STOP rising — the last depth unit costs +0.2 km for +20 m.
Seed 777 @ p3.5σ8: 158 → 207 @ 9.5.

Composition with the meso candidate (m0.7): at σ15/p1.0 SUB-ADDITIVE (295 @
11.4, below meso-alone 313 @ 11.8 — trunks overlap corridors, renorm washes
out). At σ8 it STACKS (see post-review finals below).

### Codex review (same day) — 3 confirmed fixes, finals re-measured

1. Per-orogen normalization summed over ALL shape>0 cells, but the builder only
   uplifts target-land — ocean-side shape (demotion at subduction margins)
   skewed component scales AND bridged components through the sea (42 → 85
   components once masked to `target >= 0`).
2. All-arid worlds: mean_precip 0 ⇒ a_crit 0 ⇒ `acc >= 0` painted every land
   cell a channel — now returns None (no fabricated trunks).
3. A small component whose modifier fully clamps to 0 (entirely inside trunk
   suppression at high depth) kept scale 1.0 and silently leaked its volume to
   other orogens via the global normalizer — now falls back to identity.
All three unit-tested (target-mask + fallback + all-arid). Exploration-ladder
numbers above are PRE-fix (trends unchanged); finalists re-measured post-fix:

- A4-alone σ8/p3.5: 262 @ 10.9 (12345), 210 @ 9.3 (777) — gates pass.
- Composed m0.7 + σ8/p2.5: 339 @ 12.2 (12345) — the corrected land-only
  normalization concentrates the boost; peak now over budget. 777: 256 @ 10.6.
- **Composed m0.7 + σ8/p2.0: 330 @ 11.8 (12345), 249 @ 10.5 (777)** — recovers
  the relief at meso-alone's exact peak (777: relief wash vs meso-alone 250 but
  peak −0.5 km). The composed CANDIDATE dial.

### Gates (every rung, both seeds)

Plausibility ≤ 12 km [ok] everywhere (max 11.6). Trunk grammar stable
transverse-leaning (23-29 / 32-34 / 37-44). Crest spacing CV 1.6-1.9, no
lattice spike. Elongation 4.6-5.7 ≈ baseline. Mountain land 8.4% stable.
Rivers: fine endorheic-land 16.3% (12345 p3.5σ8) / 20.5-22.3% (777; slightly
above the ~17% norm — watch on visual). Perf: burn-in 80 + final 50×2 ≈
today's default 200×2 budget (burn-in extraction ~8 s at 200k cells).

### Candidates for user visual (numbers final, eyes decide)

- A4-alone: `--drainage-pulse 3.5 --pulse-smooth-km 8` (+ s50 g1) — 262/210 m,
  peaks 10.9/9.3, biggest headroom, drainage-consistent grammar by construction.
- Composed: `--meso-relief 0.7 --drainage-pulse 2.0 --pulse-smooth-km 8` —
  330/249 m @ 11.8/10.5. Best measured 25-km relief at the peak budget to date
  (12345: +17 m over meso-alone at the same peak; 777: wash at −0.5 km peak).
- Sweep: `--sweep drainage_pulse --sweep-values 0,1.5,2.5,3.5` (σ8 via
  `--pulse-smooth-km 8`), or a 3-row stack vs baseline/meso.
