> **Dated audit evidence:** Results are revision/configuration specific and do not define current defaults.

# Erosion Validation — River Grading — 2026-06-15

Investigation into whether the fine-mesh fluvial erosion is *functional* — specifically whether
rivers **grade** (develop concave-up long profiles, the signature of stream-power equilibrium).
Triggered by a density-convergence study whose plateau was ambiguous: finer mountain cells revealed
no new relief, which could mean "mountains are dense enough" **or** "erosion isn't carving the relief
a finer mesh would capture." So: is erosion working?

## TL;DR

**River grading is functional. There was no bug.** The alarm — a population slope-area concavity
`θ ≈ 0` ("under-graded") — was a **measurement artifact**. The robust metric (median long-profile
concavity over the ~40 largest rivers) shows baseline rivers are **concave-up / graded** (median
normalized bow −0.12, 77% concave, source third steeper than mouth). Raising `EROSION_K` barely
changes grading; it modestly increases relief/dissection (a taste knob, not a fix). **No erosion
constant needs changing for correctness.**

The lasting deliverables are diagnostic: a trustworthy **aggregate long-profile grading probe**, and
the **retirement of the population-θ probe**, which is unreliable on this mesh and caused the false
alarm.

## What happened (so we don't repeat it)

1. **Symptom.** `diagnose`'s population slope-area probe reported `θ ≈ 0` (target ~0.5 for graded
   stream-power channels) in every world → looked like rivers weren't grading.
2. **First over-read.** A deposition-off + K4× run gave `θ = +0.82` ("graded"), so I concluded
   *en-route deposition was flattening profiles*. Plausible mechanism (deposition aggrades reaches to
   a repose slope ≈ the median channel slope), but…
3. **Calibration broke it.** Sweeping deposition × K, `θ` swung +0.82 / −0.18 / +0.01 / −0.06 across
   small parameter changes and produced non-physical values. The population θ is **noise** here:
   `flow_accumulation` routes through filled/non-overflowing lakes (artificial drainage area),
   single-edge `drop/dist` is noisy on irregular Voronoi cells, small bins, mixed regimes. (Codex
   flagged exactly this — "probe artifact first" — on the second opinion.)
4. **Second over-read.** A single-river long profile looked convex at baseline and concave at K0.08
   (cap 800k) → "under-incision → convex transient." But at full resolution the verdict *flipped*
   (baseline concave, K0.08 convex) — because each config's "highest channel head" is a **different
   river** (lengths 848 vs 2381 vs 11763 km). The single-river profile is selection-noise.
5. **Robust metric → verdict.** Aggregating long-profile concavity over the ~40 largest rivers
   (deduplicated, ≥200 km, normalized bow) is stable across resolution and config, and says
   **graded** everywhere:

   | config | median norm-bow | % concave | source/mouth slope | local relief R25 p90 |
   |---|---|---|---|---|
   | baseline (K=0.04) | −0.122 (concave) | 77% | 1.22 | 0.220 |
   | K=0.08 | −0.091 (concave) | 73% | 1.48 | 0.246 |
   | K=0.12 | −0.107 (concave) | 78% | 1.21 | 0.266 |

   (Population θ for the same runs: +0.09 / +0.04 / −0.00 — flat, contradicting the long profiles.
   The population θ is wrong, not the rivers.)

## Conclusions

- **Grading works.** Baseline fluvial erosion produces concave-up graded main-stem rivers. The
  density-convergence plateau is therefore *not* explained by broken erosion (re-examine it on its
  own terms when the density track resumes — likely the relief simply is captured by ~1.5 km cells,
  with strength as a secondary lever).
- **Strength (`EROSION_K`) is a taste knob, not a bug.** Higher K monotonically raises local relief
  (R25 p90 +21% at K=0.12) and dissection, with grading essentially unchanged. Whether baseline is
  "under-powered" is a visual call for the artist, not a correctness issue. Left at the default.
- **Deposition is not the problem.** The deposition-off result that implicated it was a θ-noise draw;
  the aggregate metric shows baseline (deposition on) rivers grade fine.

## Tooling changes (in `src/bin/diagnose.rs`)

- **Added** "River grading (aggregate over largest mountain rivers)" — trace each of the N largest
  rivers source→sea, measure normalized midpoint bow + source/mouth slope ratio, report the median +
  % concave. The trustworthy grading metric.
- **Added** "Fine-scale local relief" (fixed-radius max−min elevation) — scale-controlled relief, for
  density/strength convergence (per-cell Δh shrinks trivially with cell size and can't serve).
- **Retired/relabeled** the population slope-area θ probe: kept as a documented negative result, but
  its verdict is marked UNRELIABLE and points here. Do **not** judge grading by it.
- **Lake-cleaning** of the θ channel population (exclude lake-water + non-overflowing-basin cells,
  and receivers in lakes) — improves it but does not make it trustworthy; the aggregate probe is the
  one to use.

## Lesson

Don't trust a single noisy probe — especially a population statistic on an irregular mesh with lakes.
Two confident-but-wrong conclusions (deposition; under-incision) came from over-reading θ and a
single river. The aggregate-over-many-features metric is what settled it. When a probe disagrees with
itself across resolution/config/selection, fix the probe before theorizing about the physics.
