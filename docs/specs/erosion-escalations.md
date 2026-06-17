# Erosion Escalations — structural roadmap (post-tuning)

**Provenance.** Trio discussion 2026-06-16 (user + Claude + an independent
`codex exec` voice), after the routing ladder and the valleys-not-channels work.
Both independent analyses converged on the sequence and dependencies below; the
deltas Codex added (the evaluation gate after step 2, transport-before-
channelization, per-step traps) are folded in. This is the master ordering for
the remaining *structural* erosion work — tuning (K / diffusivity / channel
support, calibrated against the headless relief + carved-dissection metrics) is
handled separately and assumed to proceed in parallel.

See also: [`erosion.md`](erosion.md), [`erosion-routing-ladder.md`](erosion-routing-ladder.md)
(MFD, dormant), [`erosion-valleys-not-channels.md`](erosion-valleys-not-channels.md)
(regime gate Phase 1 dormant; thalweg coupling = step 5 here).

**Philosophy (unchanged).** Unreasonably physically inspired, not Earth-accurate;
mechanisms over hacks; parameters as playground knobs; emergent interactions are
the point.

## State of play

The visible "swiss cheese" is **real incision-carved dissection that is too BUSY
(too many finely-spaced valleys) and too SHARP (relief above Earth)** — *not*
routing-driven (MFD didn't change it), *not* a diffusion-magnitude or reroute
problem. Component isolation: incision creates it (but also the real valleys —
can't just kill it); **uplift is the biggest single source of the mountain-top
cell-scale bumps** (it re-injects the arc/collision/rift forcing every step).
Headless metrics now exist for both axes: local relief in meters vs Earth
("sharp"), and carved-dissection density / incision depth ("busy").

## The sequence

| # | Escalation | Why here / depends on | Scope | Status |
|---|---|---|---|---|
| 1 | **Uplift-forcing smoothing** | Targets the *confirmed* bump source; regularizes the tectonic forcing scale. Precede everything — else every later mechanism reacts to a noisy source. | small | **built; A/B = no-op on bumps** — source already smooth (Moran's I 0.992); uplift prickle is *magnitude*, not source speckle. Default-off infra; the de-prickle lever is #2. See erosion-uplift-smoothing.md §Result. |
| 2 | **Nonlinear (Roering) hillslope diffusion** | Planar slopes + crisp, physically-limited ridges vs linear-diffusion mush. Gives channel-initiation a *credible* hillslope to compete against → must precede #4. | medium | — |
| — | **▶ EVALUATION GATE** | Build 1+2, then stop and look. Mountains "still busy but no longer prickly" → proceed. Still prickly → the issue is forcing/incision scale, not river mechanics — rethink before building more. | — | — |
| 3 | **Transport-limited ⇄ detachment-limited blend** | Makes alluvial behaviour a first-class regime (floodplains/fans/deltas/valley-fill), not a downstream afterthought. Subsumes the Phase-1 regime gate. Precede #4 and #5. | med–large | — |
| 4 | **Channelization-instability initiation** | Drainage density *emerges* (replaces the channel-support knob). Only after #2 (credible hillslope) and #3 (credible sediment regime) — it's a feedback amplifier balancing processes that must be worth balancing. Cell-network based first; doesn't need #5. | medium | — |
| 5 | **Phase 2 thalweg coupling ("two elevations")** | The rivers endgame: path carries thalweg/width/discharge/sediment, coupled back to cell terrain; unlocks levees/terraces/meanders/deltas. Needs a good cell-terrain regime (#3) underneath. | large | — |
| 6 | **Time-coupled tectonics ⇄ erosion** | Water gaps, drainage capture, antecedent rivers, foreland basins. The prize, last: changes staged generation → co-evolution. Only once erosion is "boringly stable." | largest | — |
| 7 | **Max adaptive resolution + live Voronoi** (s2-voronoi) | Parallel infrastructure, **not a fix** — shrinks artifacts, rarely changes their nature. Use for iteration speed / better defaults; don't let it compete with structural work or substitute for validation. | infra | parallel |

## Per-step traps (heed before building)

- **Uplift smoothing:** smooth the *source*, not the final elevation — preserve
  integrated uplift. Smooth arc / collision / rift **separately** (signed rift
  thinning must not cancel or bleed into positive orogen uplift). Don't spread
  uplift across coast/ocean masks (reintroduces the submerged-uplift bug the code
  already guards). And audit the feature fields at the coarse source: if they're
  speckled before erosion sees them, fix the field, not just the forcing.
- **Roering diffusion:** explicit schemes are vicious near the critical slope
  (flux denominator → 0 is a numerical cliff). Use slope caps / regularization,
  prefer implicit/semi-implicit, log mass residuals, and watch for
  "critical-slope-everywhere" tiled-facet worlds.
- **Transport-limited:** the double-counting trap — incision makes sediment,
  deposition raises terrain, routing shifts and re-incises the fill. Needs clear
  transport-capacity / load accounting and conservative mass logs; too-eager
  deposition erases relief, too-high capacity starves floodplains.
- **Channelization-instability:** feedback → over-dense networks unless the
  initiation criterion has hysteresis / smoothing. The criterion must be physical
  area/discharge based, **not** cells-upstream (resolution-dependent).
- **Thalweg coupling:** do NOT scale incision by channel/cell width (the
  steady-state trap — broad cells demand absurd slopes). The thalweg keeps a
  plausible long profile; only the *transfer into cell-mean terrain* scales by
  valley/confinement fraction. Settle the routing negotiation (does the path
  follow the cell surface, or vice versa, and when).
- **Time-coupling:** staging explosion — every knob becomes a loop gain and
  interpretability is the first casualty. Start with slow tectonic *outer* loops,
  not per-step plate remeshing.
- **Resolution:** it hides process errors by shrinking them — a bad mechanism at
  8M cells is an expensive bad mechanism. Speed/quality, never validation.

## What to build first / what to skip now

- **First:** #1 (uplift source). Then #2 (Roering). **Then stop at the gate and
  look** before committing to #3+.
- **Defer hard:** #6 (time-coupling) unless specifically chasing the
  emergent-tectonics payoff; #5 (thalweg) until the cell-terrain regime is solid.
- **Skip:** OCN as an engine; further MFD work as a swiss-cheese fix (keep MFD
  dormant/toggleable as correct routing infra).
