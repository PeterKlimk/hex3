# Relief-spectrum redesign — mountains that read as mountains

Status: DRAFT v2 (2026-07-09) — Codex review folded in (all findings verified against
the repo). Instrument-first redesign; no code yet.
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

(5-km row omitted: below effective mesh resolution. The mountain cell TARGET is 1.5 km
(`FINE_MOUNTAIN_CELL_KM`) but the ACHIEVED spacing is ~3.9 km per the density audit
(constants.rs ~652) — a 10-km window is only ~3 effective cells in many places. This
caveat shapes the target, §2.)

Reading:
- **The deficit is broadband below ~100 km and grows toward short wavelengths**:
  ~3.5× at 100 km → ~8× at 25 km → ~20× at 10 km. Earth alpine relief is much flatter
  across 10–100 km (steep valley walls: a 10-km window catches most of a valley's
  depth). Ours is steeply red: relief accumulates only along long ramps.
- **Erosion is currently the main producer of sub-100-km relief** (~4× the pre base at
  every band) — but 5–20× short of Earth, and the accessible dials are exhausted (§4).
- **Only the ≥100-km band (O0 envelope) approaches Earth** — consistent with the v4
  blind-ranking win.
- Relief/peak ratio ~2–3% vs Earth ~30%: summits are fine, the terrain between them
  is not.

## 2. Target (revised per review)

Two-tier target, split at the mesh floor:

- **Primary (mesh-resolved): 25–100 km windows — p50 relief ≥ ~1000 m at 25 km,
  1500–3000 m at 50–100 km, saturating beyond.** This is the physical target every
  candidate is scored on.
- **10-km band: an explicit DECISION, not a default goal.** At ~3.9 km achieved cells,
  1000+ m in a 10-km window means near-cell-scale cliffs — high artifact risk.
  Options: (a) accept partial (300–600 m) physical relief there, (b) cede the band to
  render-side detail (normal/detail mapping — option D), (c) refine the mountain mesh
  first. Decide AFTER the 25–100 km winner is known.

Earth values are plausibility references, not optimization targets (Goodhart guard).
Pass bar for any candidate:
- ≥4× improvement in 25-km p50 (≥~900 m) without losing the macro wins (mountain land
  ~8–12%, significant-range count/elongation, crest asymmetry).
- Artifact gates hold: pit%/checkerboard%/curv-rms not materially worse (roughness
  counters), no summit cottage-cheese regression (summit-local probe), drainage still
  integrates (0 inland dead-end mouths, `--river-audit`).
- Metric suite (§7) stable under `--fine-scale` {0.7, 1.0, 1.5}.
- Final judgment: user's eyes on sweep-harness renders. Numbers triage only.

## 3. Incumbent inventory (measured, not assumed; review-corrected)

| Mechanism | Band it actually delivers | Verdict going in |
|---|---|---|
| Coarse tectonics + O0 structured-emergent uplift (`compute_emergent_uplift_shape`, fronts/segmentation/asymmetry; default fully structured) | ≥100 km, near-Earth | **Keeper** (only confirmed one) |
| P1 fine-base grain — NOTE: P1b strike (0.0) and P1c margin (0.0) are already OFF by default ("painted dunes" lesson); only faint P1a `interior_relief=0.005` is live | none measurable at ≥10 km | **On notice.** Its remaining job (breaking drainage symmetry / seeding incision variance at cell scale) must get an explicit owner before deletion |
| n=2 stream-power erosion + diffusion (NOTE: module/constants comments still say "n=1" — doc drift, fix in passing) | multiplies all bands ~4× over base; can't reach target alone (§4) | Role re-scoped: organizer/dissector + amplifier |
| `uplift_scale` knob | **dead in the DEFAULT emergent path only** (self-calibrating builder, erosion.rs:497); still live in the painted/non-emergent path (erosion.rs:516) | Deprecate-or-rescope, don't blind-delete |
| Lithologic K (`litho_sigma`, default-ON) + fold-belt structural-grain erodibility (default-ON, fine.rs:962) | drainage-aligned differential relief (unquantified per band) | **Audit during design** — candidate C overlaps these; no double-owners |
| Deposition, channel-support, confinement (active defaults); glacial (disabled) | context the winner must coexist with | Document interactions in implementation spec |
| Coarse fBm noise layers | unaudited | Audit during design |

No band may end up with two owners (the coarse-asymmetry lesson).

## 4. Why the accessible knobs cannot fix it (falsifications, measured)

10-point single-knob sweep at the 25-km window (baseline 224/914 m):
- K×4 → 251/1259 — best mover, but fragments mountains (land 8.5→7.2%, components
  55→120) long before approaching target.
- steps×2 → 204/773 — **worse**. In emergent mode steps is a build-vs-carve dial
  (same total uplift, more carving+diffusion time) — consistent with smoother output.
- diffusivity×0.5 → 241/1105 with fragmentation risk; channel_support 4→1 → worse.
- uplift_scale ×2/×4 → no-op — **expected**: wrong knob in emergent mode (see below).
- interior_relief ×3 → no change (faint cell-scale grain; wrong wavelength).

**Review correction: the joint high-relief REGIME was not tested.** In emergent mode
the actual uplift controls are `emergent_lambda`, `EMERGENT_REBUILD_GAIN`, the coarse
target amplitude, and `steps` — not `uplift_scale`. A joint (U↑, K↑, S_c on) regime
remains untested and is exactly candidate B.

Physical account: with n=2 stream power the steady channel slope is
S = (U/(K·A^m))^(1/n); at current effective U and K, graded slopes are gentle
everywhere, so valleys are wide and shallow. Earth alpine relief is
**threshold-hillslope-limited** (~30–40° walls set by rock strength). Nothing in the
current default regime produces or preserves near-threshold slopes.

## 5. Candidate mechanisms (replace-vs-add is OPEN)

**B. Joint high-relief regime test — RUN 2026-07-09, VERDICT: amplitude-only, spectral
shape invariant. FALSIFIED as the relief owner; gain retained as a secondary dial.**
8-run sweep (seed 12345, p95-p05 25-km p50, baseline 191 m): gain 2 → 301; gain 3 →
454; gain 3 + K×4 + S_c → 542 (bar was ≥600). BUT relief scaled with PEAK HEIGHT
(10 → 21.7 km peaks at gain 3): relief/peak stayed ~3-4% at every setting — the solver
scales terrain self-similarly, it cannot flatten the spectrum (10-km band stayed ~100 m
even with 21-km peaks). Reaching 1000 m at 25 km by pure gain needs ~40-km peaks.
Also: S_c 200 = the DEFAULT already (constants.rs:788 — the "Roering default-off"
memory is stale); sweeping it produced no useful differentiation. K×4 fragments
(228 components). CONCLUSION: spectral shape must come from structure → candidate A;
gain ~1.5-2 available as an amplitude trim afterward. Original design below. Vary together: coarse target amplitude and/or `EMERGENT_REBUILD_GAIN`
(more mountain volume to carve), K (keep pace), Roering `S_c` ON (~150–300) to hold
walls near threshold, `steps` as build-vs-carve balance. The solver is numerically
bounded throughout (Newton clamps to receiver, Roering caps S/S_c at 0.95 with an
implicit solve, fold-back is exact) — the risk is artifacts and calibration, NOT
stability, and prior "escalation" failures never ran this joint regime. Needs small
diagnose flags for gain/target-amp. **Time-box; success = 25-km p50 ≥ ~600 m without
tripping gates.** Even partial success rescales what A/C must supply.

**A. Mid-band structured uplift (EXTEND O0, replace P1's structural role).**
Not a new system: extend `compute_emergent_uplift_shape` with a 10–50 km component —
strike-aligned fold/fault-block trains (ridge/valley rhythm, spur trains, along-strike
passes/water-gaps), amplitude prescribed per band. Structure arrives through the
UPLIFT path so erosion organizes it as it grows (v4's validated pattern; avoids the
painted-dunes failure of P1b, which stamped the base). Explicit delta vs today: O0's
shape field gains mid-band spatial structure; P1a's symmetry-breaking job moves here
(then P1 dies); coarse-asymmetry stays dead.

**C. Seeded valley skeleton (define as delta vs EXISTING erodibility fields).**
Today's litho-K + fold-belt grain already modulate WHERE erosion cuts (default-on).
C is the stronger version: a coherent meso drainage skeleton (10–50 km spacing,
strike-aware) expressed as erodibility corridors or demoted-envelope valleys, so
erosion deepens pre-organized valleys. Must either subsume the existing grain
mechanism or be rejected as duplicate — no coexistence without a measured split of
ownership.

**D. Cede the shortest band to rendering.** If B+A/C hit 25–100 km but the 10-km band
stays artifact-bound at 3.9-km cells, take 10 km at render time (normal/detail
mapping) or via targeted mountain-mesh refinement. Cheapest; least "unreasonably
physical"; decide last (§2).

Recommended order (revised): **B first** (days, parameters only, directly informs how
much amplitude A/C must add), then **A** sized by B's residual gap, with **C** only if
A's ridge-side structure leaves valley organization visibly poor. D decided last.

## 6. Constraints (review-hardened)

- **Mesh adaptivity ordering**: fine density is sampled from coarse elevation +
  preview hydrology BEFORE structural relief exists (fine.rs:541). Any A/C mid-band
  relief must either (a) feed the density prior so steep new structure gets cells, or
  (b) prove spectrum convergence under `--fine-scale` {0.7, 1.0, 1.5}. Otherwise the
  10–25 km metrics measure the mesh prior, not the algorithm.
- **Cache & determinism**: fine generation is CACHE-deterministic, not
  regen-deterministic (s2-voronoi weld drift; fine_cache.rs:7). Every new
  param/constant that shapes the base MUST enter `fine_base_key` (or bump
  `FINE_BASE_CACHE_VERSION`); erosion-side params (B) don't touch the base cache.
- Perf: erosion already ~3× gen cost; A/C are per-cell synthesis (cheap); B is
  parameter-only. No new global solver passes without a measured budget.
- The elevation-first gate applies; the artifact gates in §2 are the real bar.
- Mega-seas are out of scope (separate coarse-elevation problem; roadmap).
- Doc-drift cleanup rides along: stale "n=1" comments (erosion.rs:3,
  constants.rs:680).

## 7. Metric suite & evaluation protocol (review-upgraded)

Metrics (all per window, mountain mask, area-weighted and range-stratified sampling):
- p50/p90 of **p95–p05 relief** (robust range; max-min kept for continuity but not a
  gate — one spike/pit/lake edge can inflate it),
- detrended std-dev (plane-fit residual) per window,
- slope percentiles (p50/p90 max-downhill),
- existing artifact counters + summit probe + `--river-audit` gates.

Protocol:
1. Implement candidate behind default-off params, sweepable via diagnose flags
   (B needs: rebuild-gain, target-amplitude; A/C: their shape params, cache-keyed).
2. `diagnose --seed {12345,777} --mountain-audit --river-audit` → spectrum + gates.
3. `--fine-scale {0.7,1.0,1.5}` convergence check (mandatory for A/C, spot for B).
4. Bracket 3–4 parameterizations through the sweep harness → PNG grid → **user judges**.
5. Winner flips default; losers and superseded incumbents are DELETED, not disabled
   (P1 deletion requires its symmetry-breaking role explicitly re-owned; uplift_scale
   resolution depends on whether the painted path survives this redesign).

## 8. Open questions

- Should the painted (non-emergent) path survive at all, or is this redesign the
  moment it dies with its knobs (`uplift_scale`, P1b/P1c A/B path)?
- If B alone reaches ~600–900 m at 25 km, is A still worth its complexity, or does
  "B + P1a retuned" suffice aesthetically? (User's eyes decide.)
- Does the density prior need slope-of-STRUCTURE input (A/C feed-forward), or is
  `--fine-scale` convergence proof enough?
- 10-km band: physical, rendered, or refined-mesh (§2 decision, deferred)?
- Semivariogram vs windowed detrended std-dev — is the simpler one enough? (Start
  simple; upgrade only if gates disagree with eyes.)

---

## 9. Candidate A — implementation spec (hand-off)

**Goal.** Extend the O0 structured uplift shape with a MESO band (10–50 km
ridge/valley structure) so erosion organizes pre-seeded relief. Success bar (§2):
25-km p95-p05 p50 ≥ ~600 m on seed 12345 at a default-candidate amplitude, with peaks
within ~+20% of baseline, no component explosion (≤ ~2× baseline count), summit/
roughness gates clean.

**The key mechanism — redistribution, not addition.** `Erosion*::new` volume-
normalizes the uplift shape (`shape_c`: excess volume / shape volume, erosion.rs
~:468-488). Therefore meso-structure added INSIDE the shape field redistributes uplift
from proto-valleys to proto-ridges at constant total volume — relief WITHOUT peak
inflation (the failure that killed candidate B). Implement A entirely as a modulation
of the shape returned by `compute_emergent_uplift_shape` (fine.rs:997).

**Where.** In `sample()` (fine.rs ~:1020-1080), after `profile`/`seg` are computed,
the code already has the orogen-intrinsic coordinate frame: signed cross-strike
distance `v` (radians), along-strike arc coordinate `u = fronts.arc_u[best_front]`,
and `chain_id` for phase decorrelation. Build the meso field in (u, v) FRONT
coordinates (structure aligns with the orogen like real fold trains), not raw 3D
position — but blend a minority isotropic 3D-noise component for naturalness where
`best_front` is far/degenerate (fallback path returns `demoted` today; keep that).

**Meso field sketch (Codex may improve within constraints):**
- Fold-train component: quasi-periodic ridges across strike, wavelength
  `meso_wavelength_km` (default 25), phase-modulated along strike by 1-D fbm of
  (u, chain) so ridges are wavy/broken, not corduroy — the P1b "sand dunes" failure
  was PERIODIC UNMODULATED banding stamped on the base; avoid it via (a) strong phase/
  amplitude modulation, (b) delivery through uplift (erosion reshapes as it grows).
- Along-strike spur/gap rhythm: 1-D fbm of u at ~2× the segmentation frequency,
  multiplying the fold-train amplitude (creates passes/water-gap opportunities).
- Combine to `meso ∈ [-1, 1]`, then `shaped *= (1 + meso_relief * meso)` clamped ≥ 0,
  applied BEFORE the `(1-blend)/blend` mix. Valleys must be able to reach near-zero
  uplift locally (that's where erosion cuts through) but the field must stay
  non-negative.

**Params (add to `FineStructureParams`, fine.rs:67; Default; and BOTH override plumbs
— diagnose flags `--meso-relief`, `--meso-wavelength-km`, and `ErosionOverrides` +
main.rs flags like `interior_relief`):**
- `meso_relief: f32` — modulation depth 0..1. DEFAULT 0.0 (off) until visual sign-off.
- `meso_wavelength_km: f32` — default 25.0.

**Cache (MANDATORY).** Both new params + every new shape constant must be mixed into
the fine-base cache key next to the existing structure knobs (fine_cache.rs ~:106-140,
see the "shape constants" block). The uplift shape is stored in `FineBase`
(`emergent_uplift_shape`), so a param change MUST be a cache miss. Bump
`FINE_BASE_CACHE_VERSION` if anything the hash can't see changes.

**Constraints.** Deterministic (seeded noise only — reuse the `Fbm`/`Perlin` pattern
with a new seed offset); per-cell cost only (the sample loop is already parallel); no
new solver passes; do NOT touch the painted path or P1a (their fate is a later
decision); `--fine-scale` convergence is the mesh-adaptivity proof for v1 (density
prior feed-forward is out of scope).

**Validation (after build + `cargo test`):**
`diagnose --seed 12345 --mountain-audit --rebuild-gain -1 --meso-relief {0.3,0.6,0.9}`
→ p95-p05 spectrum vs baseline; check components/mountain-land/peaks in the same
output; then seed 777 for the best; then `--fine-scale {0.7,1.5}` on the best.
