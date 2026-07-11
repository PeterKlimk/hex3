> **Archived:** Historical design or experiment record. It does not define current architecture, defaults, or priorities. Start at [`docs/README.md`](../../README.md).

# OrogenStructure: tectonic global structure, erosion dissects (erosion-v4)

**Provenance.** 2026-06-20, after the erosion-v2 (painted substrate) and erosion-v3
(emergent uplift) investigations + two independent first-principles reviews (cross-model
codex + clean-context claude subagent). The decisive empirical findings:
- Erosion DISSECTS structure; it does not CREATE it from smoothness. A smooth uplift
  dome stays a smooth swell regardless of step budget, channelization, or slope exponent.
- n>1 incision sharpens dissection (landed, `EROSION_N`) but does not supply structure.
- **Painted global structure fails.** P1a (local heterogeneity) works because erosion
  organizes it; P1b (a `cos`-distance strike band trying to paint the range's global
  grain) produces uniform "tectonic corduroy / sand dunes" — confirmed by both reviews
  and visually. Global structure (where ranges segment, branch, plunge, which way they
  face, fold spacing) is **tectonic** and cannot be a fine-mesh noise grain.

Both reviews converged on the same architecture. This spec is that architecture.

## Principle (the three-way split)

- **Tectonics owns GLOBAL structure** — the orogen's trend, width, asymmetry, active
  front, along-strike segmentation, adjacent basins, fold/thrust grain. Output: a set of
  STRUCTURAL FIELDS, not a finished height.
- **Erosion owns DISSECTION** — valleys, drainage, ridges-as-negative-space (n>1,
  emergent active uplift, the v3 scaffolding).
- **Noise owns LOCAL heterogeneity** — sub-band erodibility/relief that erosion
  organizes (P1a-style, keyed to a mechanism). NOT global shape.

Our mistake was using painted noise (P1b/P1c) to do tectonics' job. OrogenStructure moves
that job to where it belongs.

## What OrogenStructure produces

A layer between coarse tectonics and fine erosion that derives, per convergent belt, the
fields a real range needs. Consumed by the fine stage as an **uplift-RATE field** (driving
the emergent build) + an **erodibility-K field** (driving differential dissection) +
optional **basin/subsidence** — NOT as painted height. Per (coarse or fine) cell:

1. **`uplift_rate(x)` — the orogen as an actively-rising, ASYMMETRIC, SEGMENTED welt.**
   - Asymmetric cross-section: steep on the pro-wedge (subducting/overriding-front) side,
     long gentle retro/back-slope — keyed on the KNOWN `SubductionPolarity` (overriding
     plate) rather than a symmetric Gaussian. (Replaces the symmetric `gaussian_band`
     collision/arc cross-section that makes domes.)
   - Along-strike SEGMENTATION: low-frequency variation ALONG the front so the range
     rises and plunges, offsets, branches, and terminates — instead of a uniform belt.
   - Finite DEFORMATION WIDTH: a strain zone of realistic width, not a thin band.
2. **`erodibility_K(x)` — fold/thrust hardness bands** parallel to the front, VARIABLE
   wavelength (not a single global `cos`), tightest near the suture — so erosion expresses
   trellis/ridge-and-valley grain via DIFFERENTIAL EROSION (a rate, the gold-standard noise
   use), not painted height. Subsumes P1b's intent done right.
3. **`basin(x)` (optional, later) — foreland/backarc flexural lows** adjacent to the
   range: the sediment-filled lowlands that make big ranges read as real.

Erosion (n>1, emergent) then BUILDS the asymmetric segmented welt under active uplift and
DISSECTS it, with the K-bands steering the valley grain. Ranges emerge tectonic, not
painted.

## Build from what we have

- **`OrogenFronts`** (fine.rs) already extracts the convergent boundaries as great-circle
  ARCS (`seg_a`/`seg_b`) with per-front overriding plate (`accept_plate`). It is the seed
  of the primitive set. Extend it to carry, per front: strike (tangent), normal
  (cross-boundary, oriented toward the overriding plate), convergence/shear magnitude, and
  an ALONG-STRIKE coordinate.
- **The emergent builder** (erosion.rs, `coarse_target − base` self-calibrating uplift)
  already rebuilds demoted relief over the epoch. OrogenStructure REPLACES the uniform
  demoted-envelope rebuild with the STRUCTURED `uplift_rate(x)` (asymmetric/segmented), so
  the built relief has tectonic shape.
- **`lithology_erodibility`** (fine.rs) already builds a per-fine-cell K field gated by
  transferred fields. OrogenStructure feeds it the fold/thrust band field.
- **n>1** (`EROSION_N`) is the dissection sharpener.

## The new primitives (the geometry to solve)

Per fine cell, relative to the nearest compatible convergent front:
- **signed front-normal distance** `v` (toward overriding plate +, foreland −): drives the
  asymmetric cross-section and the K-band phase. (`point_to_arc_distance` gives unsigned;
  add the sign from `dot(cell − foot, normal)`.)
- **along-strike coordinate** `u`: drives segmentation. THE hard primitive — needs the
  boundary edges chained into connected polylines with an arc-length parameter, OR a
  cheaper proxy (low-frequency 3D noise sampled at the nearest-front point, which varies
  along the front for free). **Open decision (see below).**
- **overriding-side membership** + convergence magnitude (have these).

## Review status (codex, 2026-06-20): SOUND-WITH-FIXES

Core bet sound (asymmetric/segmented uplift-rate is the right lever). Fixes folded in:
- **Do O0 first** (below) — test the core bet as a contained hack BEFORE building the
  machinery. If structured uplift can't beat smooth-emergent in an A/B, the full project
  isn't justified.
- **Volume-normalize** the structured uplift (don't let height go fully emergent yet):
  `raw_rate(x) = convergence·asym_profile(v)·segmentation(u)·mask`, scaled so
  `∫ raw_rate dA · epoch · slope ≈ gain · ∫ demoted_orogen_height dA` (preserve total
  orogen work, let the DISTRIBUTION be tectonic). Track mean/p95/max/land-flip/volume.
- **n must be ≥1.5** in any decisive run (default `EROSION_N=1` — set it explicitly).
- **Proxy segmentation is smoke-test-only.** Low-freq noise at the nearest-front foot is
  NOT an along-front coordinate (blobs/seams/incoherent at kinks); if O0 with the proxy
  looks bad the result is AMBIGUOUS. If O0 is promising, do real arc-length CHAINING
  before O2 (moderate engineering: carry edge endpoint/vertex IDs, group convergent edges
  by plate-pair+polarity, endpoint graph, walk degree-2 chains, split at triple junctions,
  arc-length = accumulated angular distance; gaps = separate polylines).
- **Coarse = macro envelope.** Once O0 is accepted, move the asymmetric front + broad
  segmentation + deformation width into the COARSE `arc`/`collision` cross-section (drives
  crust thickness, sea-level, atmosphere, transfer, emergent target) — NOT patched only at
  fine handoff (leaves coarse atmosphere baked against the old symmetric dome). What breaks
  is mostly intended (sea level re-solves, coastlines/rain-shadows move, climate sees the
  new mountains). Keep FINE-scale segmentation/K-bands at fine (avoid coarse aliasing).
- **80/20:** asymmetric coarse envelope + belt-normalized segmented uplift + n>1 + existing
  lithology fBm, P1b height bands off. Defer basins, explicit branching, FeatureFields
  recompute, any global param not derived from boundary kinematics.

## O0 — the decisive hack (do this first)

Contained test of the core bet, no new machinery. In the EMERGENT uplift source only:
1. P1b/P1c height painting OFF; faint P1a seed only; `n ≈ 2`; `emergent_lambda ≈ 0.5`.
2. Replace the uniform `(target−base)` per-cell rebuild with a structured shape:
   `shape(x) = asym_profile(v) · segmentation(u) · orogen_mask`, where `v` = signed
   front-normal distance (sign from overriding-vs-foreland plate membership — reuse the
   `OrogenFronts.accept_plate` side test; no normal vector needed), `asym_profile` = steep
   narrow foreland flank → crest at the front → gentle wide hinterland, `segmentation` =
   low-freq proxy noise (smoke-test only), `orogen_mask` = where `target−base > 0`.
3. **Volume-normalize** `shape` so total uplift = `gain · Σ(target−base)·area` (global for
   O0; per-belt later). `u_thick[i] = C·shape[i]`, `C = gain·Σ(demoted·area)/(epoch·slope·
   Σ(shape·area))`.
4. A/B same seed/camera vs (current smooth emergent) and (painted P1). Numerical:
   summit-slope / drainage-density vs both; visual: does it read as an asymmetric,
   plunging RANGE vs a dissected dome?

Implementation: compute `shape` in `generate_with_target` (OrogenFronts + signed
`point_to_arc_distance` are there), store on `FineBase`, pass to `erode`; the builder
normalizes + uses it as `u_thick` when present. Behind a knob (default off → v3 behaviour).

DECISION GATE: O0 beats smooth-emergent → proceed (coarse envelope → chaining → K-bands).
O0 ≈ smooth-emergent → the tectonic-uplift idea is weak; reconsider (maybe painted is the
ceiling after all). O0 ambiguous (proxy-limited) → do chaining and retest before judging.

## O-coarse OUTCOME (2026-06-21): implemented, but DEFERRED — asymmetry needs a single owner

Built the asymmetric coarse cross-section (`features.rs` `asym_band` + `compute_overriding_
side`, behind `coarse_asymmetry`, default 0). It WORKS in isolation (painted path → a
broad asymmetric massif) but **double-applies with O0**: the coarse band shapes the front
with its crest offset `peak` INTO the overriding plate, while O0's structured uplift
(`compute_emergent_uplift_shape`) applies ANOTHER front profile with its crest at the
boundary. Stacked → two offset crests → an "inner massif + surrounding BARRIER" artifact
(user-caught; codex-confirmed). Plus a secondary bug: the coarse band's SIGN
(`compute_overriding_side`, nearest-midpoint KD-tree) and MAGNITUDE (plate-screened
Dijkstra distance) come from different sources that disagree near curves/triple junctions
→ wrong-sign lobes. (O0 avoids this — it derives side + distance from the same front.)

**Decision (codex): B for now.** O0 OWNS orogen asymmetry; `coarse_asymmetry` stays 0
(default). The atmosphere being symmetric (no side-aware rain shadows) is a second-order
loss vs. breaking the validated emergent terrain. A guard in `generate_fine_pre_with_cap`
warns if both are enabled. The O-coarse code is retained (default-off) as scaffolding.

**Deferred redesign (C/A hybrid) — the single-owner architecture, when revisited:**
1. ONE shared signed-distance product from real front arcs (reuse/extend `OrogenFronts`:
   segment endpoints + overriding plate + chain + arc-length), NOT KD-midpoint-side +
   Dijkstra-distance from different sources.
2. Coarse owns only the broad asymmetric ENVELOPE the atmosphere reads.
3. O0 STOPS applying its front-normal profile — its source becomes `demoted ×
   segmentation(u)` (the demoted target is ALREADY the coarse asymmetric envelope; the
   existing volume-normalizing builder keeps height sane). O0 keeps segmentation,
   dissection, active uplift, erosion timing.
4. Collision stays SYMMETRIC (no subduction-style overriding/foreland band) until a real
   bilateral-collision model exists.

**Validated locked state:** O0 (structured emergent uplift) + n≈2 incision +
`channel_support≈4` + the hillshade render fix. Coarse asymmetry OFF.

## Incremental plan (visually gate each — do NOT build it all at once)

1. **O1 — asymmetric, segmented uplift-rate.** Replace the uniform emergent rebuild with a
   structured `uplift_rate(x)`: front-anchored, asymmetric across `v` (steep pro / gentle
   retro), modulated along strike (proxy segmentation first). Drives emergent + n>1. This
   ALONE tests the core thesis: does structured uplift + dissection give tectonic ranges
   vs the v3 smooth dome? Cheapest decisive increment.
2. **O2 — fold/thrust K-bands.** Variable-wavelength erodibility bands parallel to the
   front into `lithology_erodibility`; retire P1b. Trellis grain via differential erosion.
3. **O3 — along-strike coordinate done properly** (chain boundaries → arc-length) if the
   O1 proxy segmentation is too crude.
4. **O4 — foreland/backarc basins** (flexural lows) for the range-plus-basin gestalt.
5. Cleanup: retire P1b/P1c from the default; keep painted path behind knobs for A/B.

## Keep the A/B (user's point)

The painted path (P1a/b/c, `emergent_lambda`, etc.) stays behind knobs. OrogenStructure is
a new default path; every increment is A/B-able against painted noise on the same seed —
if principled doesn't beat noise, we'll see it.

## Open decisions (for review)

1. **Coarse vs fine computation.** Build the structural fields on the COARSE mesh (cheap,
   then transfer/interpolate like other fields) or directly on FINE (sharper, more cost)?
   The uplift-rate is broad → coarse+transfer likely fine; the K-bands are sub-coarse →
   may need fine. Possibly split.
2. **Along-strike coordinate:** proxy (nearest-front-point noise) for O1, real arc-length
   chaining for O3 — or commit to chaining up front?
3. **Does this REPLACE the coarse `collision`/`arc` elevation cross-section** (make it
   asymmetric there too, so the macro envelope is right and the atmosphere sees it), or
   only add structure at the fine handoff? (Codex leaned toward fixing the coarse cross-
   section; that touches the atmosphere-validity story.)
4. **Height preservation:** the emergent build must still land near a sensible orogen
   height (the v3 self-calibrating gain). Structured uplift makes the per-cell target
   non-uniform — recalibrate.

## Risks

- The along-strike segmentation is the make-or-break for "not a uniform belt"; a weak
  proxy may still read as too-regular. O1 must demonstrate real along-strike variation.
- More tectonic realism = more parameters; resist re-creating the P1 knob sprawl. Lean on
  derived (kinematic) quantities, not free knobs.
- Atmosphere validity if the coarse envelope changes (decision 3).
- This is a real project; each increment gates on the visual before the next.

## References
- Erosion-v3 finding (emergent + n>1, why pure-emergent fails): `erosion-v3-emergent-orogens.md`,
  [[hex3-emergent-orogens-finding.md]].
- Painted substrate (P1a/b/c, what to keep/retire): `erosion-fine-synthesis.md`.
- Both first-principles reviews (this session): tectonics owns global structure; noise =
  rate not height; cut dormant systems (glacial/MFD/confinement/uplift-smooth).
- Code: `fine.rs` OrogenFronts + emergent builder; `erosion.rs` n>1 incision + u_thick;
  `boundary.rs` SubductionPolarity/convergence/shear; `features.rs`/`elevation.rs` coarse
  collision/arc cross-section.
