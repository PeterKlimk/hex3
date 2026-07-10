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

---

## 10. A′ — meso relief in the BASE elevation (hand-off #2)

**Why (measured, 2026-07-09).** The uplift-redistribution variant of A was implemented
correctly (identity-verified, monotonic dial) but the solver attenuates 25-km uplift-
RATE structure ~100×: meso_relief 0.9 (valleys 0.1×, ridges 1.9× uplift) moved 25-km
p95-p05 p50 only 191→218 m. Cranking the EXISTING erodibility fields (litho grain ×4,
sigma ×2) is also flat (≤+3%) — they are incoherent at meso wavelength. Conclusion:
at 10-50 km the solver flattens rate/volume anomalies; the only channel it demonstrably
respects is relief ALREADY IN the elevation it dissects (the v4 finding). So: paint the
meso band into `base_elevation`, at a wavelength (25 km) far above P1b's corduroy
failure regime (1-5 km), with the same anti-corduroy defenses.

**Implementation.**
- Factor the meso field out of `compute_emergent_uplift_shape` into ONE shared helper
  (fold-train in front (u,v) coordinates + phase warp + wavelength jitter + spur
  gating + isotropic blend — keep constants/behavior identical so the uplift variant
  stays available for A/B). No duplicate implementations.
- Apply it in the base-synthesis path next to `add_interior_structural_relief`
  (fine.rs ~:676): `base_elevation[i] += meso_base_relief * envelope(i) * meso(i)`,
  where `envelope` reuses the SAME orogen gating the interior grain uses
  (arc+collision-scaled, so plains stay clean) and the result is approximately
  zero-mean like the interior grain (the existing area-weighted datum drift check
  must stay quiet).
- New param `meso_base_relief: f32` in `FineStructureParams` — ELEVATION units
  (0.01 ≈ 100 m), default 0.0 (off). Shares `meso_wavelength_km`. Plumb exactly like
  `meso_relief`: diagnose `--meso-base-relief`, main.rs flag, `ErosionOverrides`,
  sweep knob, AND the fine-base cache hash.
- Validation ladder: identity at 0.0 (bit-identical spectrum), then
  `--meso-base-relief {0.02, 0.05, 0.10}` (≈200/500/1000 m seed amplitude) on seed
  12345 — expect the PRE-erosion 25-km band to rise to ≈ the seeded amplitude and the
  ERODED band to retain a large fraction; gates: components/mountain-land stable,
  summit probe clean.

---

## 11. Composed-regime gate battery (2026-07-10, measured)

Instrument: patched `diagnose` — `--mountain-audit` now also emits the roughness
counters + summit plateau probe (they were behind the audit early-return).
Configs are seed 12345 unless noted; "composed" = meso_relief 0.9 + rebuild_gain 2
+ steps 50. All runs: rivers clean (0 inland dead-end mouths, Dd unchanged).

### Attribution ladder (25-km p95-p05 p50 / max range peak / components / elong p50)

| config | relief | peak | comps | elong |
|---|---:|---:|---:|---:|
| baseline (s200 g1 m0) | 191 m | 10.0 km | 55 | 5.0 |
| meso only (s200) | 218 m | ~10 km | 60 | 4.8 |
| steps 50 only | 220 m | 10.2 km | 132 | 4.9 |
| meso + s50 (g1) | 386 m | 11.0 km | 193 | 5.1 |
| meso + s50 + g1.5 | 479 m | 13.1 km | 193 | 4.5 |
| meso + s50 + g2 (composed) | 648 m | 16.5 km | 194 | 3.2 |
| meso + s100 + g2 | 463 m | 9.9 km | 138 | 4.0 |
| meso 0.6 + s50 + g2 | 536 m | — | 186 | 3.3 |

Findings:
- **Synergy is the mechanism**: meso alone +27 m, short epoch alone +29 m, together
  +195 m (at g1). The seeded structure needs a short epoch to survive transport;
  the short epoch needs structure to expose. Consistent with §10's attenuation story.
- **Every gate cost decomposes onto the regime dials, not meso**: peak inflation
  (+65% at g2) and elongation loss (5.0→3.2, incl. a merged 7,740-km supergiant)
  are rebuild_gain's (candidate-B signature); the component rise (55→132) is the
  short epoch's (small high patches survive above the 0.15 mask). Meso itself is
  gate-clean at every setting.
- **Three dials, three axes**: meso_relief = mid-band relief depth (monotone
  368/536/648 at m 0/0.6/0.9, g2 s50); rebuild_gain = relief↔peak-inflation trade;
  steps = relief↔cleanliness trade (s100 at g2 returns peaks to baseline and halves
  the summit-pit rise while keeping 463 m).

### Gates on the composed endpoint

- **Cross-seed**: seed 777 replicates (167→583 m, 3.5×; land 11.6%; elongation
  HELD at 5.0 — the blob/supergiant effect is seed-12345's geography, not systematic).
- **Fine-scale convergence (mandatory, §6): PASS** — composed/baseline ratio 3.33 /
  3.39 / 3.28 at fine-scale 0.7 / 1.0 / 1.5. The 10-km band is mesh-bound as
  expected (165→263 at fs 0.7) — still the §2 deferred decision.
- **Roughness**: checker% improves (45.5→42.1); pit% 0.40→0.70 (well under the
  swiss-cheese regime); curv-rms 2.9e-3→8.3e-3 (~2.9×) tracks the 3.4× relief —
  amplitude-proportional, not the 8× glacial-artifact signature; most of it is the
  regime's (regime-only 6.7e-3), meso adds +24%.
- **Summit probe**: plateaus are gone (summit slope p50 1.4e-3 → 8.8e-3 elev/km) —
  the intended effect. WATCH-ITEM: summit pit% 2.9→6.6 (s100 variant: 4.8) —
  numbers can't distinguish artifact ponding from legitimate hanging valleys;
  visual call.

### Verdict & handoff to visual

Numbers-side: the mechanism is validated; no disqualifying gate. The §2 bar
(≥600 m) is met only at g2 s50, which buys it with +65% peaks and blobbier belts on
some seeds; g1 s50 (386 m, peaks +10%, elong intact) and g2 s100 (463 m, peaks ~base)
are the clean middle candidates. This is now an aesthetic trade — four tiles are in
the sweep harness: `--sweep-stack meso` (baseline / g1-s50 / g2-s100 / g1.5-s50 /
g2-s50; `rebuild_gain` also added as a plain sweep knob). Component counts 132-194
vs baseline 55 are mostly sub-significant crumbs (significant ranges stable 8-13);
if they read as scatter in renders, raising the audit mask or a min-area filter is
cosmetic, not structural.

### §11 addendum — visual verdict → irregularity dial (2026-07-10)

User verdict on `--sweep-stack meso`: mechanism reads (dunes < smooth ramps), gain
ladder visible as measured, but **"the ridges are too consistent"** — the 1-D fold
train is phase-locked across strike (every ridge a copy of its neighbor; pure sine
= the "dune-y" cross-section). Fix: `meso_irregularity` (0..1, default 0.7) scaling
three metronome-breakers: cross-strike decorrelation of phase/spur (per-ridge
wobble/termination), a second incommensurate fold octave (0.618λ beats → variable
prominence/spacing), crest sharpening (|fold|^0.65). Measured (seed 12345, composed
g2 s50): irregularity 0 is IDENTITY with the 1-D field (spectrum line-identical
through a full cache-miss regen); 0.7 and 1.0 retain 89% of the 25-km gain
(648→576 m) and slightly improve components (194→171) / pit% / curv-rms; rivers
clean. 0.7→1.0 is numerically saturated — the difference is texture, eyes only.

## 12. Crest-train grammar audit + design consultation (2026-07-10)

User verdict #2 (irregularity sweep): "generally similar … not sure what they're
supposed to look like." Two responses, both landed:

**New instrument** (`--mountain-audit` crest-train block): per significant range,
cross-strike transect crest detection (150 m prominence) → ridge spacing
distribution + ridges/transect, and steepest-descent orientation vs strike
(longitudinal/oblique/transverse). Measured (seed 12345):

| config | spacing p25/p50/p75 km | ridges/transect | flow L/O/T % |
|---|---|---|---|
| baseline | 32/68/172 | 2 | 30/33/38 |
| meso irr 0 | 20/28/40 | 12 | 30/33/37 |
| meso irr 0.7 | 20/28/48 | 10 | 30/33/36 |

Readings: (1) the train IS in the terrain (median spacing ≈ λ, 12 ridges/transect);
(2) **drainage ignores it completely** — flow orientation identical to baseline, no
trellis grammar (real fold belts drain ALONG strike valleys with transverse water
gaps); (3) the irregularity dial does not change the terrain class (as the user saw).
Our meso terrain is "stamped ribs + indifferent drainage" — neither fold-belt nor
alpine grammar. The ridges and valleys must AGREE for terrain to read as real.

**External design consultation** (GPT 5.6 via codex exec, memo verbatim in
`meso-design-consult-gpt56.md`, refs spot-checked): longitudinal fold train is the
wrong DEFAULT prior for alpine collision belts (right for a Zagros/V&R foreland
apron component); a lay eye keys on NEGATIVE space (branching trunk valleys, spurs,
unequal massifs/saddles/peak groups), not ridge statistics; more irregularity
polishes the wrong prior. Also: the 0.618λ octave (~15.5 km ≈ 4 cells) is below
mesh representability; secondary structure belongs at 35-60 km. Amplitude honesty:
~600 m at 25 km reads Appalachian, not alpine (S. Alps transverse ridges 1000-1500 m).

Recommended successor constructions (all uplift-rate channel, §10-compatible):
- **A2 massif-and-saddle**: irregular anisotropic Gaussians in (u,v) (u-spacing
  25-60 km, heavy-tailed amplitudes, flank-offset centers, saddle gaps) — object
  vocabulary becomes "massifs separated by corridors".
- **A3 corridor trees**: branching LOW-uplift corridors from front roots (outlet
  spacing ≈ half divide-front distance, Hovius R≈2; trunks transverse ±20-40°;
  10-20% cross-belt; widths ≥8-15 km; −10-25% uplift) — seed the negative space,
  inter-corridor ground becomes spurs/ridges for free.
- **A4 two-stage drainage-aware pulse** (preferred long-term): low-amplitude
  burn-in → extract order≥3 drainage/divides → smooth 10-40 km → zero-mean uplift
  modifier (low on trunks, high on interfluves) → short high-relief epoch. One
  feedback pass only. Resolves the short-epoch vs drainage-maturity tension.
- Fold train survives as a FOREland/fold-thrust preset, not the default.

## 13. A2+A3 — massif-corridor meso field (implementation spec)

**Goal.** Replace the fold train as the DEFAULT meso construction with one field that
changes the object vocabulary from "parallel ridges" to "massifs separated by
transverse valley corridors" (§12 consult; v1 corridors are single wobbled oblique
trunks — BRANCHING hierarchy is A4's job). Delivery is UNCHANGED: same uplift-shape
modulation point, same `meso_relief` depth dial, same composed-regime dials
(gain/steps), same cache discipline. The fold train survives behind a style switch
as the foreland/fold-thrust preset.

**New param** `meso_style` (usize: 0 = fold train, 1 = massif-corridor; DEFAULT 1 —
safe because `meso_relief` defaults 0 = off). Plumb exactly like `meso_relief`
(FineStructureParams, cache key, diagnose/main flags, ErosionOverrides, sweep knob).
Identity requirements: (a) `meso_relief 0` → bit-identical to today at any style;
(b) `meso_style 0` → bit-identical to the committed fold train at any irregularity.

**Construction (in the existing (u,v) front frame, one sampler struct):**
All scales derive from `meso_wavelength_km` (λ, default 25) so one dial scales the
whole construction; W_h = FINE_OROGEN_HINTERLAND_WIDTH, the range's half-width.

- **Massifs (A2):** jittered 1-D lattice along u per chain, period 1.6λ (~40 km);
  per-site hash (seed, chain, k) drives: u-jitter ±40% of period, center offset
  v_i ∈ [-0.15·W_f, +0.6·W_h] (flank-offset, not all on the crest), L_u ∈ [0.4, 1.2]λ,
  L_v ∈ [0.3, 0.8]λ, heavy-tailed amplitude a = 0.35 + 0.65·h³ (h ~ U[0,1] — a few
  dominant massifs). Sum anisotropic Gaussians from the 2 nearest lattice sites each
  side. Consult mesh floor: no σ below ~2 cells (clamp L_u, L_v ≥ 8 km).
- **Corridors (A3):** per hinterland flank, jittered u-lattice period 1.8λ (~45 km,
  ≈ Hovius outlet spacing at these W_h); each corridor k: root at outer hinterland
  (v = W_h), path u(v) = u_k + tan(θ_k)·(W_h − v) + fbm wobble (θ_k ∈ ±[20°, 40°],
  wobble amplitude ~0.3λ, frequency ~1/(2λ)); Gaussian cross-section width
  w ∈ [0.35, 0.6]λ (≥ 8 km); depth d = full meso amplitude (the corridor is the
  valley seed — −10..25% uplift arrives via the meso_relief dial). Head gate: 85%
  of corridors fade out at v ∈ [0.1, 0.3]·W_h (heads interdigitate below the crest);
  15% (by hash) cross fully through v=0 into the foreland (water gaps / antecedent
  trunks). No corridors rooted on the narrow foreland flank in v1.
- **Combine:** M = clamp(massifs − corridors, −1, 1), then the SAME 20% isotropic
  blend and delivery `shaped *= (1 + meso_relief·M).max(0)` as the fold train.
  No octave2/sharpening machinery (those are fold-train-specific).

**Determinism:** all randomness from a small splitmix-style hash of
(seed, chain_id, lattice index, salt) + the existing seeded Fbm for wobble. No
Date/thread-order dependence (the sample loop stays per-cell parallel).

**Success gates (instruments already exist):** on the composed regime
(meso 0.9 · gain 2 · steps 50, seed 12345): flow-orientation transverse share
RISES clearly above the 37% baseline plateau; crest-train spacing loses the λ
spike (p50 well off 28 km or dispersion up); 25-km p95-p05 p50 stays ≥ ~450 m;
components/elongation/summit/river gates as §11; `meso_style 0` and
`meso_relief 0` identities hold. Then a `meso` stack / style A/B for the user.

### §13 addendum — implementation + gate results (2026-07-10)

Implemented (massif-corridor sampler, style switch, full plumbing). Codex diff review
found 2 fatal geometry bugs, both fixed: corridor site search was windowed on the
cell's u while oblique paths drift up to ~7 lattice periods (fix: per-sign
inverted-root windows); u was midpoint-quantized at ~70-km coarse segments AND
mirror-folded at the BFS seed (fix: new `u_lin`/`u_dir` endpoint-ordered oriented
chain coordinate + within-segment projection — `arc_u` untouched for fold-train
identity). Plus: obliquity sign salt, massif window ±3, new constants hashed,
λ floor, iso-frequency scales with λ. Declined: conditional cache mixing (identity
proven empirically through regen; collision risk), A′ style-dispatch (shared-field
design per §10). v1 corridors are single oblique trunks (branching = A4).

Gates (seed 12345 composed g2 s50, vs fold train same regime): identities PASS
(meso off + style 0 both file-identical through cache-miss regen); 25-km p95-p05
514 m (fold 648, bar 450); spacing λ-spike GONE (p50 40 km, IQR 20-88, ridges/
transect 7 vs 12); roughness BETTER (pit 0.54 vs 0.70, summit slope softer);
rivers clean; seed 777: 440 m, land 11.8%, comps 113. NEW trunk-flow-orientation
metric (top-decile SFD accumulation; all-cell split is hillslope-dominated and
class-blind): baseline 24/32/44 L/O/T, fold train 36/33/31 (longitudinal shift =
half-formed trellis — strike trunks without water gaps: the measured "corduroy"),
massif-corridor 29/33/38 (transverse-leaning restored). Corridors imprint only
partially at 50 steps (drainage self-organizes before corridor uplift-deficit
accumulates) — if visual wants stronger valley grammar, next levers: deeper
corridor depth via field weighting, s100, or A4. USER VISUAL: style A/B sweep.

### §13 second addendum — peak budget closes the gain path; corridor-heavy candidate (2026-07-10)

User verdict on the regime ladder: peaks "reach the heavens" past ~12 km — gain is
RETIRED from the meso path (candidate-B self-similarity, third confirmation). New
plausibility SELF-GATE in --mountain-audit (max range peak: <=12 ok / 14+ FLAG-ABSURD,
vetoed before visual — the 18-km rows should never have reached the user's eyes).

Measured servo coupling: the volume-normalizing builder repays any meso-carved
volume as global uplift (~peaks +2 km at meso 0.9 g1) — massif caps can't lower
peaks, only total meso depth can. Corridor-heavy field (MASSIF_CAP 0.2,
CORRIDOR_GAIN 1.6): relief from valleys down. Depth ladder at g1 s50:
m0.7 = 313 m @ 11.8 km [ok]; m0.8 = 339 @ 12.1; m0.9 = 362 @ 12.4 [borderline].

**Candidate default: meso_style 1, meso_relief 0.7, steps 50, gain 1.** Full gates
pass BOTH seeds (777: 250 m @ 11.0 km): elongation fully recovered (5.0/5.3 ≈
baseline), rivers clean, roughness proportionate, trunk grammar transverse-leaning
(40/37%). This is the honest ceiling of the uplift-shape channel at the user's peak
budget (~1.6× baseline relief; the ≥600 m bar is UNREACHABLE in this channel
without absurd peaks — servo-coupled). Deeper relief at fixed peaks = A4
(drainage-organized dissection removes volume AFTER the height budget) — the
designated next architecture step if the candidate still reads too smooth.
Visual A/B: `--sweep-stack meso` (baseline | candidate | m0.9-borderline).
