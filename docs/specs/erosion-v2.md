# Roadmap: Erosion v2 (fine-terrain synthesis + coupled evolution)

Synthesis of two independent reviews of the erosion stage and its upstream
feeders (June 2026): one code-internals read and one geomorphology-checklist
read. They converged on the same top-tier findings; this doc merges them into
one prioritized plan. Items are tagged `[me]` / `[codex]` / `[both]` by which
review surfaced them.

Philosophy reminder (unchanged): the goal is "unreasonably physically
inspired," not Earth simulation. Add *mechanisms* with named constants as
playground knobs; do not tune constants to chase numbers — the numerical
probes (roughness, Moran's I, denudation %, drainage density) are a REFERENCE
for judgment, not optimization targets. The user judges maps. See
`docs/specs/erosion.md`, `docs/specs/fine-mesh.md`, and
`docs/physically-inspired-roadmap.md`.

## Root cause

One sentence ties every finding together: **erosion is the only source of
sub-coarse-cell land character, but it currently runs in a degenerate
regime** — relaxation-only, carving an interpolated coarse surface, with the
isostasy machinery inert. Three compounding facts:

1. **Erosion carves an interpolant, not synthesized fine terrain.** The fine
   mesh interpolates the already-solved coarse elevation onto fine cells
   (`fine.rs:304`) and erosion applies only an isostatic delta on top of it
   (`erosion.rs:210`). There is no new structural or noise detail in the band
   between the coarse Nyquist (~70 km) and the erosion channel scale (~few km)
   — only what incision carves. `[both]`

2. **The loop is relaxation-only.** The uplift path exists (`erosion.rs:165`)
   but `EROSION_UPLIFT_SCALE = 0` (`constants.rs:745`). This correctly avoids
   double-counting the coarse orogen, but it means there is no U/K
   equilibrium, no sustained relief, no graded-profile mechanism — stage 4
   decays a static initial condition. "More steps" = more denudation, not
   "closer to graded." `[both]`

3. **The isostasy/thickness layer is inert for terrain shape.** `derive_elev`
   (`erosion.rs:210`) and the thickness fold-back (`erosion.rs:300`) are exact
   inverses (`thick - thick_init ≡ (elev - base)/slope`); the operators work
   in elevation space, so thickness carries no information beyond elevation
   except at deposition sinks. "Isostasy responds" describes nothing dynamic
   in this regime — no delayed or spatial rebound. It only becomes meaningful
   once uplift is on (Phase 3). `[me]`

The plan moves erosion from "cosmetic carve on a smooth interpolant" to "real
landscape evolution with something to grade against."

---

## Noise philosophy

Noise was built when the coarse mesh *was* the finished product, so it painted
the *appearance* of terrain. Erosion + the fine mesh change the job: noise
should now be an *input* a physical process organizes, not a painted output.
Sort by role, not scale.

A noise term is justified only if it is one of:

1. **Heterogeneity of a physical state variable or parameter** — it perturbs an
   input that a mechanism then organizes; the landform is emergent, not painted.
   It lives in that state variable and at the scale of the process that consumes
   it. (thickness → isostasy; erodibility → incision.)
2. **Render-only dressing** — sub-mesh visual texture that *never enters the
   simulation*, purely so large flat-shaded cells don't look faceted.

It is NOT justified when it paints the appearance of a landform a process now
produces — that competes with and contaminates the simulated result (painted
"ridges" don't align with drainage; erosion then superimposes its own grain on
the fake grain). Those layers were honest stopgaps; the mesh + erosion work
that supersedes them has landed (`physically-inspired-roadmap.md` "Defended
as-is" anticipated exactly this).

Verdict by layer:

- **Macro** (`elevation.rs:108`) — **keep.** Role 1 done right: a crust-thickness
  perturbation, isostatically compensated, consumed by Airy. The model of "the
  crust isn't uniform." (Next candidate for a *physical* replacement —
  craton-structure-derived thickness — but defensible as-is; not in scope here.)
- **Hills / Ridge** (`elevation.rs:133,152`) — **retire** as elevation paint.
  Ridge is convergence-gated fake mountain grain; hills is redundant regional
  undulation. Their function (mountain grain, interior relief) returns
  physically via erodibility heterogeneity + erosion of structural relief.
- **Micro** (`elevation.rs:175`) — **demote** to render-only; drop from the
  world data model (the shader already carries `MICRO_AMPLITUDE`). Justified
  only where cells are large (coarse stages, ocean).

Two corollaries that answer "why on the coarse mesh at all":

- **None of these layers were ever fine** — their highest octaves bottom out
  ~200–500 km. True 1–50 km detail has only ever come from the cosmetic micro
  cap; that band is exactly what erosion now fills.
- **Only role-1 noise feeding a coarse process belongs on the coarse mesh** —
  i.e. macro alone (the coarse sim consumes thickness for the sea-level solve,
  land mask, and isostasy). Erodibility feeds a *fine* process, so evaluate it
  at fine cell centers, never coarse-then-interpolated — interpolation
  band-limits to ~140 km and *then* smooths, so it cannot serve as fine detail.
  Stripped of paint, the coarse elevation becomes honest *structural* relief
  (isostatic thickness incl. macro + thermal + ridge-feature − trench),
  appropriate for the "Lithosphere" view.

---

## Phase 0 — Make the current pass correct & legible

Small effort, low risk, no regime change. Prerequisites that make all later
tuning trustworthy. Do these first regardless of strategic direction.

- **Channel-initiation threshold** `[codex]`. Stream power currently incises
  wherever drainage area is nonzero, including hillslopes. Gate `incise_step`
  (`erosion.rs:485`) by a drainage-area (or area·slope) cutoff; below it,
  diffusion only. Physically the channel head — the hillslope→channel
  transition. New const `EROSION_AREA_CRIT`. This is the cheapest single item
  that changes terrain character. Effort: low.

- **Diffusion in the mass ledger** `[both]`. `eroded_vol` is incision-only
  (`erosion.rs:269`); diffusion (`:286`) and the `.max(0.0)` clamp (`:698`)
  move and leak mass invisibly. Track diffusion's net volume change (or at
  least log its residual) so the eroded/deposited/lost balance actually
  closes. Effort: low.

- **Diffusion convergence on the finest cells** `[me]`. On ~1.5 km mountain
  cells the implicit coupling is `c ≈ 2`, so the `EROSION_DIFFUSION_ITERS = 6`
  Jacobi sweeps under-converge → mild speckle exactly where the result should
  be cleanest. Verify on a real mesh; raise the sweep count if the finest
  cells show cell-scale spikes (the constant's own doc already says "8 is
  ample", `constants.rs:716`). Effort: low.

- **Comment-truth pass** `[me]`. The code claims things that are no longer
  true: `EROSION_STEPS` "closer to graded profiles" wording (`constants.rs:674`),
  false while uplift is off; and the `erosion.rs` header's "isostasy responds"
  framing — add a note that the thickness↔elevation round-trip is currently an
  identity (root cause #3). (The stale ridge "RidgedMulti" comment is moot —
  Phase 1 deletes that code.) Effort: low.

## Phase 1 — Synthesize real fine terrain

Medium effort, medium risk, **highest visual payoff per effort**. Attacks root
cause #1 and is a hard prerequisite for Phase 3 (the loop needs fine structure
to evolve). Applies the noise philosophy above: retire the appearance-paint,
replace its *function* with role-1 seeds that erosion organizes.

(Supersedes an earlier draft item, "make ridge noise genuinely ridged + ungate
it" — that was backwards: it enriches the paint instead of retiring it.)

- **Retire hills + ridge from elevation assembly** `[me/codex, resolved]`.
  Remove the two additive elevation-paint layers (`elevation.rs:133,152`) and
  their constants. The coarse elevation becomes honest structural relief; the
  fine base interpolated from it is smooth structural relief for erosion to
  carve. Effort: small (deletion).

- **Demote micro to render-only** `[me]`. Drop micro from the world data model;
  keep it as a shader detail where cells are large (`unified.wgsl` already has
  `MICRO_AMPLITUDE`). Effort: small.

- **Erodibility heterogeneity `K(x)` on the fine mesh** `[both]`. The
  replacement for ridge's function: instead of painting mountain grain, give
  incision heterogeneous rock and let the grain emerge *aligned with drainage*,
  with knickpoints at the contrasts. A role-1 seed — drive `K` from a cheap
  field (crust age / continentality / fBm) evaluated **at fine cell centers**
  (never coarse-then-interpolated; fine-mesh spec §4). Per-cell `k` in
  `incise_step`; carry the field on `FineFields`. Effort: medium.

- **(Conditional) high-frequency thickness band for quiet interiors** `[me]`.
  Only if, on real maps, macro + erosion leave low-slope cratonic interiors
  glassy-flat. The honest fix is a *thickness* perturbation (isostatic,
  composes with erosion), not a return to additive hills paint — effectively a
  higher-frequency macro octave. Ship without it first; add only if needed.
  Effort: small.

## Phase 2 — Fluvial realism (the depositional half)

Medium effort, independent of the coupled loop. Pull sediment-spreading
forward into Phase 0.5 if pointy deltas are visible on current renders.

- **Sediment spreading / transport-limited fill** `[both]`. Deposition
  currently dumps each catchment's sediment into the single coastal sink it
  drains to (`erosion.rs:715`), capped by local depth → pointy mouths and
  underdeveloped lowlands. Spread it across basin-floor low cells, floodplains,
  and fans. Effort: medium-large.

- **Dynamic lakes as base level during erosion** `[me/codex]`. Routing fills
  pits to spill (`erosion.rs:515`); real terminal lakes should act as a local
  base level during the loop, not just be recovered by hydrology afterward.
  Effort: medium.

- **(Optional) Multiple-flow-direction routing** `[codex]`. Single steepest-
  descent receivers give parallel, somewhat artificial drainage. MFD helps
  divergent flow on hillslopes/fans; single-flow is fine for channels, so this
  is lower priority. Effort: medium.

## Phase 3 — Coupled uplift↔erosion (strategic centerpiece)

Large effort, high risk/reward. Resolves the most findings at once and is the
north star: it makes the loop a real landscape-evolution model rather than a
one-shot relaxation. This is the "A'2" time-evolution thread in
`docs/ideas.md`.

- **Turn uplift on — but fix the double-count properly first.** Today's choice
  is binary (`EROSION_UPLIFT_SCALE = 0`) because the static IC already bakes
  the full orogen height (`092803b`). Provide uplift as a *rate* and remove the
  equilibrium height from the initial condition — either start the fine crust
  thinner and grow the orogen over the loop, or subtract the graded component
  at handoff. This makes U/K equilibrium real (concave graded profiles,
  sustained ridge-valley relief) and **finally makes the thickness/isostasy
  machinery meaningful** (genuine rebound, root cause #3). Must be balanced
  against `EROSION_K`. Effort: large; risk: high (re-tunes everything).

- **Climate→erosion feedback** `[codex]`. Recompute orographic precipitation as
  relief evolves, and let lakes become evaporation sources for the climate.
  Wet/dry dissection is currently one-way (`moisture.rs:68` → precipitation-
  weighted area, never recomputed). This is only meaningful once relief is
  *moving*, so it belongs here, not in the one-shot. Effort: medium (on top of
  the loop).

- **Flexural (non-local) isostasy** `[me]`. Spatial rebound spread by an
  elastic plate, not pointwise Airy (see `docs/specs/flexure.md`). Only matters
  once mass moves over time. Effort: medium-large.

---

## Recommended sequence

Phase 0 → Phase 1 → Phase 3, with Phase 2's sediment-spreading pulled forward
if deltas look bad now. Rationale: Phase 0 makes tuning honest; Phase 1 is the
biggest payoff *and* a prerequisite for the loop; Phase 3 subsumes climate
feedback, flexure, and the inert-isostasy problem in one move.

**The one strategic fork:** after Phase 1, do you *polish the one-shot
relaxation to a high bar* (Phase 2, lower risk) or *commit to the coupled loop*
(Phase 3, higher reward but re-tunes everything)? Decide after seeing Phase 1
on real maps.

## Validation

Each phase ships independently; validate against references, not targets.

```bash
cargo fmt && cargo test
cargo run --release --bin hex3 -- --headless --seed 12345 --export /tmp/w.json.gz
cargo run --release --bin diagnose -- --seed 12345 --fine-max 300000
```

- `cargo test` green, including field smoothness (Moran's I) — no phase may
  introduce cell-scale speckle.
- Maps before/after on the affected stage; the user signs off on appearance.
  Erosion: valleys follow the flow-accumulation network; drainage density
  visibly higher in wet regions than arid at similar uplift.
- diagnose probes report the deltas (roughness percentiles, local-extrema %,
  denudation % of land volume, drainage density wet-vs-arid, hypsometry stays
  bimodal). Report magnitudes in the commit message; magnitudes are reviewed,
  not chased.
- Mass sanity: eroded ≈ deposited + lost; once Phase 0 lands, diffusion's
  contribution is in the ledger too.
- Phase 1 specifically: retiring the paint shifts the elevation distribution,
  so the sea-level solve re-centers automatically — re-check land fraction
  within 0.5pp of `LAND_FRACTION` and that hypsometry stays bimodal. The change
  in terrain *character* (smoother pre-erosion, grain from erosion + `K(x)`) is
  expected and reviewed on maps, not a regression.
- Phase 3 specifically: river long-profiles concave-up in active orogens (the
  U/K-equilibrium signature that relaxation-only cannot produce).

## Non-goals

- Glacial / periglacial / aeolian / coastal erosion; stratigraphy; submarine
  fluvial erosion (sea level stays a fixed datum).
- Retuning unrelated constants, or tuning erosion constants against numeric
  targets.
- GPU-side erosion compute.
- Force-derived Euler poles and other tectonics reworks (separate roadmap
  items); Phase 3 couples *erosion* to existing uplift forcing, not a new
  tectonics model.
