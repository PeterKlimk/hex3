# Relief-spectrum redesign — mountains that read as mountains

Status: DRAFT for review (2026-07-09). Instrument-first redesign; no code yet.
Instrument: `diagnose --mountain-audit` relief spectrum (branch feature-scorecard).

## 1. Problem & evidence

The fine eroded terrain has Earth-plausible **macro** mountain structure but reads as
hills up close: giant smooth ramps to high summits. Measured (seed 12345, mountain-mask
samples, p50 [p90] meters; stable vs seed 777):

| window | pre-erosion base | eroded | Earth alpine ballpark |
|-------:|-----------------:|-------:|----------------------:|
|  10 km |          14 [60] | 62 [317] | 1000–1800 |
|  25 km |         55 [202] | 225 [903] | 1500–3000 |
|  50 km |        133 [404] | 486 [1784] | 2000–3500 |
| 100 km |        273 [812] | 981 [3028] | ~saturates 3000+ |

(5-km row omitted: below mesh resolution — mountain cell target is 1.5 km
(`FINE_MOUNTAIN_CELL_KM`), achieved spacing coarser; a 5-km window spans ~3 cells.)

Reading:
- **The deficit is broadband below ~100 km and grows toward short wavelengths**:
  ~3.5× at 100 km → ~8× at 25 km → ~20× at 10 km. Earth's alpine spectrum is nearly
  FLAT across 10–100 km (steep valley walls: a 10-km window catches full valley
  depth). Ours is steeply red: relief accumulates only along long ramps.
- **Erosion is currently the main producer of sub-100-km relief** (~4× the pre base at
  every band) — but 5–20× too weak, and its dials are exhausted (§4).
- **Only the ≥100-km band (O0 envelope) approaches Earth** — consistent with the v4
  blind-ranking win.
- Relief/peak ratio ~2–3% vs Earth ~30%: summits are fine, the terrain between them
  is not.

## 2. Target

**Relief spectrum on the mountain mask: p50 roughly flat at 1000–3000 m across the
10–100 km windows, saturating (not growing) beyond.** Earth values are references for
plausibility, not optimization targets (Goodhart guard); the pass bar for any candidate:

- ≥5× improvement at the 10- and 25-km windows without losing the macro wins
  (mountain land ~8–12%, significant-range count/elongation, crest asymmetry).
- Artifact gates hold: pit%/checkerboard%/curv-rms not materially worse than baseline
  (`diagnose` roughness counters), no summit cottage-cheese regression (summit-local
  probe), drainage still integrates (0 inland dead-end mouths, `--river-audit`).
- Resolution-independent (spectrum stable under `--fine-scale` sweep) and
  deterministic per seed.
- Final judgment: user's eyes on sweep-harness renders. Numbers triage only.

## 3. Incumbent inventory (measured, not assumed)

| Mechanism | Band it actually delivers | Verdict going in |
|---|---|---|
| Coarse tectonics + O0 structured-emergent uplift (fronts, segmentation, asymmetry) | ≥100 km, near-Earth | **Keeper** (only confirmed one) |
| P1 fine-base structural grain (`interior_relief`/strike/margin) | none measurable (×3 amplitude → no change at any window) | **On notice — delete if superseded**; its original job (summit texture) must be re-covered |
| n=2 stream-power erosion + diffusion | multiplies all bands ~4× over base; can't reach target (§4) | Role re-scoped: organizer/dissector; NOT the relief budget owner |
| `uplift_scale` knob | nothing (self-calibrating builder ignores it — erosion.rs:497) | **Dead — delete** |
| Coarse fBm noise layers | unaudited | Audit during design |

No band may end up with two owners (the coarse-asymmetry lesson: double-applied
asymmetry was dead weight).

## 4. Why knobs cannot fix it (falsifications, all measured)

10-point single-knob sweep at the 25-km window (baseline 224/914 m):
- K×4 → 251/1259 — best mover, but fragments mountains (land 8.5→7.2%, components
  55→120) long before approaching target.
- steps×2 → 204/773 — **worse** (more erosion = smoother; also the v3 lesson).
- diffusivity×0.5 → 241/1105 with fragmentation risk (components →100).
- channel_support 4→1 → 206/793 — worse.
- uplift ×2/×4 → byte-identical no-op (dead knob).
- interior_relief ×3 → no change (wrong wavelength: 1.5-km grain, ~50–150 m).

Physical account: with n=2 stream power the steady channel slope is
S = (U/(K·A^m))^(1/n); at our U and K the graded slopes are gentle everywhere, so
valleys are wide and shallow. Earth alpine relief is **threshold-hillslope-limited**
(walls stand at ~30–40° because rock strength, not stream power, sets them); nothing
in the current system produces or preserves near-threshold slopes at 1.5-km cells.

## 5. Candidate mechanisms (replace-vs-add is OPEN)

**A. Broadband structural generator (replace P1 + extend O0 downward).**
One mechanism producing correlated structure from ~10 to ~500 km: fault-block /
fold-train relief keyed to the existing `OrogenFronts` (strike-aligned ridge/valley
trains, spur rhythm, block rotations), amplitude prescribed per band to hit the
spectrum. Erosion keeps its measured role (organize + dissect + 4×). Physical story:
real orogens ARE fold/fault trains; erosion reveals them.
- Deletes: P1 grain (superseded), possibly parts of `compute_emergent_uplift_shape`.
- Risk: "painted dunes" failure mode from erosion-v2 — mitigate by feeding it through
  the UPLIFT path (structure grows under erosion, v4's validated pattern), not by
  stamping the base.

**B. Threshold-hillslope dissection regime (change the solver's relief ceiling).**
Make relief slope-limited instead of flux-limited: reactivate/redesign the nonlinear
hillslope law (Roering S_c exists, default-off, measured "mild" — possibly because
incision never steepened slopes toward S_c), raise incision depth until walls hit a
critical slope, possibly detachment-limited n=2 with much larger U·epoch and K
retuned together. Erosion becomes the relief owner it currently half-is.
- Deletes: nothing structural; retires the "erosion can't do it" assumption with one
  more (bounded) attempt — the sweep varied knobs one at a time; the U+K+S_c JOINT
  regime was never tested.
- Risk: prior escalation history says this family disappoints (mild/no-op/artifacts);
  cost: hot-path; may fight the fold-back/stability constraints. Time-box it.

**C. Seeded valley skeleton (v4 pattern extended down-spectrum).**
Generate a meso drainage/valley skeleton (10–50 km spacing, strike-aware water gaps
and cross-valleys) and seed it as NEGATIVE structure (incision guide / erodibility
corridors / demoted-envelope valleys) so erosion deepens pre-organized valleys instead
of discovering weak ones. Complements A (A seeds ridges, C seeds valleys).
- Deletes: possibly the litho-grain experiment (superseded corridor mechanism).
- Risk: valley spacing becomes authored rather than emergent; dendritic realism
  depends on the skeleton generator quality.

**D. Declare the mesh the binding constraint (do less, cheaper).**
If 1.5-km cells cannot carry threshold slopes without artifacts, no mechanism above
reaches the 10-km band honestly; then the right move may be: hit 25–100 km only
(A/C at reduced scope), and take the 10-km band at RENDER time (normal/detail
mapping), accepting stylization. Cheapest; least "unreasonably physical".

Recommendation to review: **A as primary, C as its valley-side complement, B time-boxed
as a calibration experiment first** (it's the only one that's pure parameters — if a
joint U·K·S_c regime moves the 25-km p50 past ~600 m without fragmenting, the whole
design gets cheaper), D as fallback. But the review should challenge this.

## 6. Constraints

- Perf: erosion already ~3× gen cost; A/C are per-cell synthesis (cheap); B touches
  the hot path. No new global solver passes without a measured budget.
- Determinism per seed; resolution independence under `--fine-scale` (spectrum is the
  probe); fine-base cache key must include any new structure params.
- The elevation-first gate applies (these ARE elevation changes; trivially passed,
  but the artifact gates in §2 are the real bar).
- Mega-seas are out of scope (separate coarse-elevation problem; roadmap).

## 7. Evaluation protocol

1. Implement candidate behind a default-off param (sweepable via diagnose flags).
2. `diagnose --seed {12345,777} --mountain-audit --river-audit` → spectrum + gates.
3. `--fine-scale {0.7,1.0,1.5}` spot-check for resolution independence.
4. Bracket 3–4 parameterizations through the sweep harness → PNG grid → **user judges**.
5. Winner flips default; losers and superseded incumbents are DELETED, not disabled.

## 8. Open questions for review

- Is the flat-spectrum target right, or should the 10-km band be explicitly ceded to
  rendering (option D) given the 1.5-km cell floor?
- Can A avoid the painted-dunes failure while still hitting amplitude, i.e. how much
  of the structure must arrive via uplift-during-erosion vs base?
- Is B's joint-regime hypothesis worth its time-box, given three prior falsifications
  in this family?
- What owns summit texture if P1 dies — A's short-wavelength tail, or nothing (were
  the P1 aesthetics ever validated)?
- Spectrum instrument: is max-min per window robust enough, or do we need banded
  std-dev to avoid single-outlier inflation?
