# hex3 roadmap / idea inventory

Living record of candidate work + status, so ideas don't get lost between sessions.
Goal of the project: **good-looking mountains & mountain ranges** ("unreasonably physically
inspired", not Earth-accurate).

## Process rule (the rain-shadow lesson, 2026-06-22)
Every "better mountains" ADD must pass an **elevation-first A/B** (same mesh/seed) BEFORE it
counts: mountain-mask `p50/p90/p99 |Δelevation|`, slope `p90/p99`, blind render. **Derived
fields (flow, precip, climate) do NOT count unless elevation also moves.** Bar: p90 mountain
`|Δ| ≥ 0.01`. (Rain-shadow's p90 was 0.003 — it moved rivers, not mountains.) → Build this
probe into `diagnose` before the next mountain ADD.

## Architecture first-principles review (2026-06-21)

### LOSE — DONE (merged to main)
- ✅ coarse-asymmetry envelope (default-off, double-applied with O0)
- ✅ dead `transform` feature field (zero terrain consumers; classifier kept)
- ✅ write-only atmosphere struct fields (continentality/upper_temperature/pressure/moisture/phi)
- ✅ orphaned `normalize_positive_field`
- ⏸️ glacial pass — KEEP-DISABLED (scaffold for the MFD-on-ice rework), not deleted
- ⏸️ coarse-hydrology PREVIEW — trim-only (flow-prior fast path), LOW priority

### ADD — status & priority (re-gated by the elevation-first rule)
1. ✅ **Rain shadow** (downwind lee drying) — DONE, default-off. RECLASSIFIED: a river/climate
   feature, NOT a mountain feature (m/n=0.25 → precip is a weak relief lever). Useful 0.2-0.4.
2. ⬜ **O0 along-strike passes / water-gaps** — as an UPLIFT/base-shape change (O0 segmentation
   floors at 0.15 so crests never break). Codex rates FIRST-ORDER on relief: summit-gap-summit
   rhythm + river outlets. **Top mountain candidate.** Needs fine-cache bump.
3. ⬜ **Marine deposition** (deltas / shelf aprons) — extend `deposit()` below sea level.
   Coastlines, not mountains. Small-medium effort.
4. ⬜ **Stratigraphic / depth-dependent erodibility** (K = f(depth below a folded datum)) —
   benches / cuestas / cap-rock. Erosion TEXTURE; speculative; hot-path cost. Lower priority.

## Mountain-quality open issues
- 🟨 **Mesa/pillar + legacy-yield orogen rung — IMPLEMENTED, gates pass 2 seeds, user
  visual pending** (2026-07-11): the user's "Everest but a cliff on all sides" pillar =
  a broad double-forcing dome (arc+collision stack, no shared strength limit) →
  6-km coarse load → fine shape stack amplifies to 9-12 km. The earlier "microplate"
  diagnosis was an export-analysis radians bug. Detected by the
  new spire probe in `--mountain-audit` (ring-p90 drop; mesa = flat-then-cliff, Earth
  tapers). Fix = `--orogen-model legacy-yield`: exact legacy source + gravitational
  yield relaxation of only over-strength thickening (`OROGEN_YIELD_ELEV` dial); sub-yield
  belts bit-untouched. Measured: cliff 3.6→1.3 km [ok], structure unchanged, composed A4
  candidate keeps 315/247 m relief at 8.7/8.7 km peaks (was 330/249 @ 11.8/10.5).
  Also: legacy default restored BYTE-IDENTICAL after the model-ladder walk-back (float
  round-trips had invalidated all baselines ±2.6 km); thin-sheet T0 stays an explicit
  experiment. `--rebuild-fidelity-audit` is now the standing coarse→fine gate. It
  independently falsified per-orogen normalization as the pillar fix: the global
  normalizer TAXES this dome 11.7% (local c 2.991 > global 2.619), while the
  structured-vs-uniform A/B changes its peak 7.62→4.85 km. Fix the within-orogen
  shaped allocation, not cross-orogen volume transfer. Full spec:
  `docs/specs/thin-sheet-orogeny.md`.
  **Renderer follow-up (2026-07-11):** a Windows renderer-only sweep at identical
  terrain/camera (`relief_scale` 0.20/0.08/0.04/0.02) proved the photographed tower
  was dominated by 0.20's ~127× vertical exaggeration. Default is now **0.04**
  (~25×): other ranges remain plainly visible while the monolith disappears.
  `--mountain-audit` now also reports p50/p10 surroundings and an apparent-wall
  render gate. Relief rivers' SDF width was corrected from fixed texture texels
  (~7–21 km full width) to true screen-pixel width. Interactive runs accept
  `--relief-scale`; use `--relief-scale 0.00157` for approximately physical 1×.
- 🟨 **A4 two-stage drainage-aware uplift pulse — IMPLEMENTED, gates pass 2 seeds, user
  visual pending** (2026-07-10): burn-in erosion → Strahler-≥3 trunk extraction →
  per-orogen zero-mean uplift modifier → frozen final epoch. Measured: σ8/p3.5 gives
  262/210 m 25-km relief @ 10.9/9.3 km peaks (vs 193/158 baseline); composed with the
  meso candidate (m0.7 + σ8/p2.0) = **330 m @ 11.8 (12345) — best at the peak budget**,
  +17 m over meso-alone at the same peak (post Codex-review fixes: target-land
  normalization mask, all-arid guard, zero-component fallback). Key finding: volume
  neutrality ≠ peak neutrality (the interfluve boost sits on massif cores — the residual coupling is boost
  ALLOCATION, a design DOF). Knobs `--drainage-pulse/--pulse-smooth-km/--pulse-burnin-steps`;
  pulse-0 byte-identical. Full results: `docs/specs/meso-a4-drainage-pulse.md` §5.
- ⬜ **25-km local-relief gap — KNOBS FALSIFIED (2026-07-09 sweep)**: baseline p50/p90 =
  224/914 m vs Earth alpine 1500-3000. Best knob (K×4) reaches only 251/1259 while
  fragmenting mountain land 8.5→7.2%; steps×2 makes it WORSE (smoother); uplift_scale is
  a NO-OP by design in the O0/emergent path (self-calibrating builder — hygiene: dead
  knob); interior_relief×3 inert (cell-scale grain, wrong wavelength). The deficit is a
  missing 10-50 km MESO wavelength (ridge/valley rhythm) between the O0 envelope (100s km)
  and the P1 grain (1-5 km). Fix = seed meso-structure in the O0 uplift shape (along-strike
  passes/water-gaps + cross-strike spur/valley rhythm) and let erosion organize it —
  same principle as v4 (structure seeds, erosion dissects). Instrument: --mountain-audit.
  **2026-07-10 UPDATE — gates PASS, awaiting visual.** Meso field built (default-off);
  composed regime (meso 0.9 + rebuild_gain 2 + steps 50) = 648 m at 25 km (3.4×,
  cross-seed replicated, fine-scale-convergent, rivers clean). Mechanism is SYNERGY
  (meso alone +27 m, steps 50 alone +29 m, together +195 m at gain 1); all gate costs
  (peaks +65%, elongation loss, component rise) decompose onto the REGIME dials, not
  meso. Clean middle candidates: g1+s50 (386 m, peaks +10%) and g2+s100 (463 m, peaks
  ~base). Visual A/B: `--sweep-stack meso`. Full table: relief-spectrum spec §11.
- ⬜ **Wide-mountain residual spikiness** — root-caused to the FINE-BASE synthesis (pre-erosion,
  cell-scale), not erosion. Fix = flat/coherent broad-massif macro shape + better fine-base noise.
- ⬜ **Coarse-asymmetry single-owner redesign** (C/A hybrid) — for atmosphere rain-shadow
  consistency; deferred (one shared OrogenFronts signed-distance product; coarse owns envelope,
  O0 drops its front-profile; collision stays symmetric).

## Other candidate directions
- **River RENDERING re-work — ✅ DONE (merged 2026-06-22).** Floating quad ribbons → rivers
  baked into an equirect **SDF texture** the terrain shader samples by lat/lon and
  reconstructs as **thin, crisp, screen-space-AA'd** rivers draped on the surface; water-
  shaded (fresnel/glint, distinct from ocean). The existing **Off/Major/All** river mode is
  now the real density control (is-major in the SDF B channel; was non-functional). Non-relief
  modes keep line rivers; dead quad path removed.
- ✅ **River HYDROLOGY pass — DONE (merged 2026-06-22), see `docs/specs/drainage-integration.md`.**
  Root cause (measured + Codex-confirmed): drainage did depression FILLING but no INTEGRATION
  (~30% endorheic coarse, ~50% fine). Shipped a basin-integration pre-pass that carves outlet
  channels (micro-pit geometry criterion + carve along the priority-flood spill path), adopted
  into the rendered terrain as real water gaps. **Fine endorheic 42% → 17.2%** (≈Earth), rivers
  now reach the sea. The "pluvial overflow" criterion was tested and rejected (step function,
  over-integrates); climate left at 0.15 (a user dial). Possible follow-ups: lakes too sparse
  (climate), and a targeted high-discharge criterion if a major river still ends inland.
- ✅ **Lakes** — DONE (merged 2026-07-08): lake-aware breaching (split MIN_INTEGRATION_SILL_RELIEF
  from MIN_LAKE_DEPTH + basin-aware carve) restores lakes with outlets; climate dial revived
  (smooth + monotonic, verified by `--lake-audit`). Original diagnosis kept below for the record.
  **NEW follow-up → see "Mega-sea problem".**
  ORIGINAL: the drainage integration traded lakes for sea-reaching rivers (lakes ~0.8% →
  ~0%; breached basins can't pond). **DIAGNOSED (2026-06-22 probe + Codex):** lake-capable
  basins collapse **61 → 4** through integration via TWO mechanisms — (1) direct over-selection
  (`MICRO_BASIN_DEPTH 0.012 > MIN_LAKE_DEPTH 0.01` + `OR` area gate breach lake-capable basins),
  (2) collateral carving (a micro-pit's outlet `carve_outlet` walks the global flood-parent tree
  and cuts THROUGH preserved deep basins — 2,449 carved cells, 100% in lake-capable basins).
  Climate inertness is by-design (endorheic = topology metric), NOT a `calculate_water_levels`
  bug. **Fix (both needed):** (a) fix the predicate so lake-capable basins aren't directly
  breached (lower `MICRO_BASIN_DEPTH` below `MIN_LAKE_DEPTH` / guard `!lake_capable`), AND
  (b) make `carve_outlet` basin-aware — stop at / route into a preserved basin rather than
  slicing through it. Then fill-to-overflow → lakes WITH outlets. See
  `docs/specs/drainage-integration.md`.
- ⬜ **Mega-sea problem** (found by `--lake-audit`, 2026-07-08): lake area is dominated by ~6
  ROUND Caspian-to-4×Caspian inland seas (largest 1.45M km²; total lake cover 3.5% of land vs
  Earth ≈1.8% at default dial 0.15; Earth-like at ≈0.05). Catchment/lake ratios 3–6× (Earth
  10–100×) = the depression is its own watershed. Present in COARSE hydrology → inherited from
  coarse ELEVATION (deep closed tectonic depressions), NOT a hydrology bug. Levers: coarse
  closed-basin depth/size, terminal-sea evaporative equilibrium (sit well below spill), dial
  default; aesthetic: elongated rift lakes (elong 6–7 tail lakes look right) over round blobs.
- ⬜ **Feature scorecard** (2026-07-08, user mandate): object-level feature audits with Earth
  refs + distributions + dial-response curves as the primary quality instrument for ALL systems
  (Claude judges numbers, user judges images). `--lake-audit` is the template; NEXT: mountain
  panel (range objects: elongation, crest-gap spectrum, front asymmetry, valley spacing) and
  river panel (Horton bifurcation ≈3–5, trunk concavity θ≈0.45, drainage density, sinuosity),
  then a `--scorecard` umbrella. Earth values are references, NOT targets (Goodhart guard).
- ⬜ **Perf** — default gen ~3× slower (n=2 Newton + 200 erosion steps); `EROSION_STEPS` is the
  quality↔speed dial. Revisit if iteration speed bites.

## Validated current state (the baseline to beat)
O0 structured-emergent uplift (asymmetric front + along-strike segmentation) + n=2 incision +
channel_support≈4 + hillshade render fix. Shipped as the default. See
`docs/specs/orogen-structure.md`.
