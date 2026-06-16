# Erosion Routing Ladder — SFD → MFD (standing roadmap)

**Provenance.** A trio discussion on 2026-06-16 (user + Claude + an independent
`codex exec` voice) on the back of the "fractal spiral ridge / swiss-cheese"
artifact investigation and a recent-paper search. Two independent analyses
(Claude's and Codex's) converged on the same ladder and the same further-beyond
priorities; that convergence is the main reason to trust this path. This is a
**standing roadmap we intend to climb**, not a one-shot plan — start on
well-trodden ground, escalate rung by rung, measure at each.

See also: [`erosion.md`](erosion.md) (current loop + the "Roughness counters"
section), [`erosion-v2.md`](erosion-v2.md), and
[`../physically-inspired-roadmap.md`](../physically-inspired-roadmap.md).

**Philosophy constraint (unchanged).** Add the *mechanism* with named constants
as knobs; do not tune constants to chase probe numbers. The roughness counters
are references for judgement, not optimisation targets; the user judges maps by
eye, we review magnitudes. hex3 is "unreasonably physically inspired," not
Earth-accurate — favour mechanisms that widen the space of interesting worlds,
and prefer a knob that spans fantasy→Earth over a hardcoded choice.

---

## Diagnosis: two artifacts, two places, two measurements

The "spiral" and the "swiss-cheese" are **distinct problems in distinct places**;
conflating them leads to over-correcting. Attack and measure them separately.

| Artifact | Where | Root cause | Counter that sees it |
|---|---|---|---|
| **Spiral grooves** | on flats | priority-flood `flood_parent` wavefront drains flats with no convergence term → parallel/spiral flow, then incised | drainage-aspect (local); judge on the map |
| **Perforation** ("swiss-cheese") | on slopes | single-flow-direction (SFD) incision cuts 1-cell channels, leaving 1-cell ridges that near-zero diffusion (`EROSION_DIFFUSIVITY = 2e-8`) never erases | `pit%`, `checkerboard%` |

Literature backing (see References): Barnes 2014b documents the parallel/spiral
flat-flow and its convergent fix; the Salles (eSCAPE) and Anand/Porporato lines
show SFD drainage is partly a triangulation artifact that MFD removes, and that
D8 misses the smooth→channelized transition MFD reproduces.

---

## The core insight that de-risks the climb

**Braun–Willett does not need a *tree* — it needs a downstream *ordering*.** MFD
breaks "one receiver," but if every flow edge points to lower hydraulic
potential the graph is an acyclic **downhill DAG**; a topological sort still
yields an order where every receiver is solved before the node, so the implicit
`n = 1` incision stays an **O(edges) single pass** with weighted receivers:

```text
h_i_new = (h_i_old + F_i · Σ_j w_ij · h_j_new / d_ij)
          / (1 + F_i · Σ_j w_ij / d_ij)
```

where `j` ranges over downstream neighbours, `w_ij` are the MFD flow fractions,
and `F_i = dt · K · A_i^m`. No exotic solver machinery (no eSCAPE matrix, no
Anand linear-layout) is required for `n = 1`; generalise the existing
single-receiver routing to a weighted DAG. (Caveat: this incision form is a
modelling *choice* — incise toward a flow-weighted blend of downstream
neighbours — not a unique derivation. Acceptable under the philosophy.)

**MFD as the playground knob.** The flow-partition exponent `p` in
`w_ij ∝ S_ij^p` (Freeman/Quinn-style; Tarboton D∞ is the two-neighbour limit)
*is* the SFD↔MFD dial: `p → ∞` recovers crisp SFD canyons, `p ≈ 1` gives diffuse
Earth-like divergence. One named constant spanning fantasy-sharp → soft, falling
out of the MFD machinery for free.

---

## The ladder (climb in order; measure each rung)

| Rung | Change | Value | Effort | Status |
|---|---|---|---|---|
| 0 | Diffusivity sweep + `reroute=1` — **diagnostic only** | (info) | ~free | **done** (both knobs ruled out — perforation is structural) |
| 1 | Convergent flat resolution (Barnes 2014b) | very high | low–med | **signed off** (more natural; slightly too clean in places — convergent routing concentrates flow; expect MFD to relax it) |
| 2 | MFD drainage-*area* only (SFD incision kept) | high | med | **landed as infra** (knob `mfd_exponent`, default **off**) — MFD area *alone* is a near-no-op for perforation (curv-rms −1.5..3.5%, elevation unchanged); single-receiver incision still carves 1-cell. Substrate for Rung 3. |
| 3 | Full MFD-DAG implicit incision | highest | high | **next** (the real perforation fix — distribute the carving, turn `mfd_exponent` on here) |
| 4 | MFD sediment / deposition (same fractions) | high | med–high | TODO |
| 5 | Channelization-instability initiation | very high (philosophy) | high | future |

**Rung 0 — diagnostic, not a milestone. DONE (seed 12345, fine-max 600k).** Both
cheap knobs are ruled out, so we skip straight to the structural rungs:
- **Diffusion is not the fix.** Raising `EROSION_DIFFUSIVITY` ×1000 (2e-8→2e-5)
  leaves the cell-scale banding amplitude (`curv-rms`) *flat* (~2.3e-2) and pushes
  `checker%` *up* toward 50% (white-noise) — at high D the under-converged Jacobi
  injects its own cell-scale noise. Only `pit%` nibbles down (0.78→0.51). Incision
  regenerates the perforation every step faster than diffusion removes it.
- **Reroute frequency is irrelevant.** `reroute=1` vs `=6` is identical to 3
  decimals on every counter — stale routing contributes nothing.
- Conclusion: the perforation is **structural to SFD incision** (→ Rung 3 MFD),
  not a diffusion/reroute tuning problem. **Do not adopt diffusion as the fix** —
  if routing keeps cutting 1-cell channels, diffusion is only a texture knob hiding
  the wound (confirmed: it can't fill them).

**Rung 1 — convergent flat resolution.** Replace the bare `flood_parent`
direction on flats with Barnes' superimposed gradients (away-from-higher +
toward-lower outlet). Use it for routing/ordering only. Surgical, low-risk,
operates inside the existing priority-flood; kills the spiral at its source.
**Trap: route water with the flat potential, never carve with it** — the
artificial gradient is not erosive slope.

**Rung 2 — MFD drainage-area only.** Keep the SFD single-receiver incision; compute
`A` with MFD fractions on the hydraulic potential. Low-risk bridge that doesn't
touch the implicit solve. Tests whether smoother discharge alone reduces
perforation, and is the on-ramp to Rung 3. A natural "stop and evaluate" point:
Barnes flats + MFD area may already be "good enough" before the bigger swing.

**Rung 3 — full MFD-DAG incision.** Generalise `incise_step` from one receiver to
weighted downstream neighbours over the downhill DAG (see the core insight
above). The real fix for both artifacts: divergent flow, no 1-cell channels,
mesh-insensitive dendritic networks. More invasive but tractable for `n = 1`.

**Rung 4 — MFD deposition.** Split sediment by the same flow fractions. If
incision goes MFD but deposition stays tree-routed, you get unphysical
single-thread fans/deltas — incision and deposition must move together.

**Rung 5 — channelization-instability initiation.** Replace/augment the fixed
`EROSION_CHANNEL_SUPPORT_KM2` threshold with an emergent criterion from the
fluvial-incision-vs-hillslope-smoothing competition (discharge, slope,
diffusivity, lithology — a local Péclet-like balance). Turns drainage density
into an *emergent result* of climate+uplift+lithology rather than a magic
constant. Sits **on top of** a sane drainage substrate, so it comes after the
MFD rungs.

---

## Further beyond (frontier — know it, climb later)

Ranked by fit with the "emergent interactions" philosophy:

1. **Channelization-instability initiation** (Rung 5 above) — the big
   "unreasonably physical" knob; drainage density emerges.
2. **Nonlinear / threshold hillslope diffusion (Roering).** Flux → ∞ near a
   critical slope → planar hillslopes + crisp ridgelines instead of rounded
   mush. Directly addresses "ridges look wrong" with real physics; pairs with
   MFD. Linear `D` can't erase 1-cell ridges without smoothing everything if the
   artifact is constantly regenerated — this is the principled alternative.
3. **Transport-limited ⇄ detachment-limited blend.** We already have
   transport-aware deposition; going further lets sediment supply armour valleys,
   fill basins, and make fans/floodplains *compete* with incision. On-theme.
4. **Time-coupled tectonics ⇄ erosion.** The most magical upgrade: water gaps,
   drainage capture, antecedent rivers, range asymmetry, migrating divides — all
   emergent. **But the biggest blast radius**: erosion is currently a one-shot
   stage-4 pass (see [`staging.md`](staging.md)); this changes the staging model.
   Do it last, after routing is sane.

**Know-it-but-not-the-engine: Optimal Channel Networks** (minimum
energy-dissipation; Rinaldo & Rodríguez-Iturbe). Conceptually gorgeous —
self-organised fractal dendritic networks by construction — but both independent
voices flagged it as an analysis/initialisation idea, not the main engine: it
risks imposing a global drainage *aesthetic* instead of letting geology, climate,
uplift, and lithology fight it out locally ("network as output," a sibling of the
"noise as output" the philosophy resists).

---

## Exit ramps: non-physical "outs" (the off-ramp at any rung)

The ladder is physics; it does not have to be climbed to the top. At any rung we
can decide the *remaining* artifact is cosmetic and reach for an established
graphical technique instead of more physics — "enough is enough." This is a
legitimate **terminal** choice, not a failure: past some rung the marginal realism
stops paying for the effort, and the residual sits below the scale anyone reads as
"wrong physics."

**The philosophy gate is input-vs-output** (the standing rule): procedural detail
as an *input/modulator on top of the physical skeleton* is fine — it supplies the
irregularity and sub-grid detail the mechanism can't resolve. Noise as the
*output terrain* (replacing the erosion result with a procedural field) is the
thing the project exists to avoid. An "out" is acceptable to the degree it
*decorates* the physical skeleton rather than *substituting* for it.

**Micro/colour vs structural/macro — the higher bar.** Outs span a spectrum.
*Micro* outs (surface roughness, talus texture, shading/colour detail) are
low-stakes — they decorate and little rides on them. *Structural/macro* outs
(landform-scale ridge grain, basins, whole drainage features placed procedurally)
are higher-stakes but still acceptable **when they stand in for a real physical
process or gap we can't or won't simulate** — not when they paper over our own
discretization artifact. The test: does the macro out represent physics that
genuinely exists and we've chosen not to model (e.g. fold-belt lithologic grain →
ridge-and-valley texture), or is it masking a routing/mesh bug we'd rather fix?
The former is a legitimate gap-filler; the latter is makeup on a wound — and the
counters tell which.

Acceptable outs (skeleton stays physical; noise is keyed to physical fields):

- **Ridged multifractal noise (Musgrave) keyed to flow/slope/relief.** Adds
  ridge-and-valley texture where the mesh is too coarse to dissect, *modulated by
  the physical fields* (drainage, slope, lithologic grain) so it only appears
  where plausible — a field-keyed layer, not a global one.
- **Domain warping (Quílez).** Warp the sampling field with low-frequency noise to
  break mesh-aligned / spiral regularity. A cheap disguise aimed precisely at
  *discretization* artifacts (which is what the spiral/grooves are).
- **Slope-/curvature-keyed detail noise.** High-frequency roughness gated by
  steepness (talus on steep faces, smooth valley floors) — sub-mesh detail the
  fine mesh has a hard floor on. The resolution floor is real; noise is the right
  tool below it.
- **Spectral sub-grid synthesis.** Synthesize detail below the mesh scale to a
  target power spectrum (the fluvial ~β≈2 slope): "fill in below what the
  mechanism determines."
- **Game-dev hydraulic erosion as a cosmetic finishing pass** (droplet/grid, Mei
  et al.). "Erosion-like" texturing, not the LEM — a heavier out; use only if it
  reads better than the physical result at the same cost.

The trap and the not-this:

- **Ad-hoc blur / bilateral smoothing of the field** is the technique the
  philosophy explicitly resists (a neighbour-blur was once called out as ad-hoc) —
  it destroys emergent structure to hide a wound. Prefer a field-keyed modulator
  over a destructive filter.
- **Don't let a cosmetic out hide a structural bug we'd rather fix.** Measure with
  the roughness counters before and after, so we know what we're masking: residual
  sub-mesh texture is fine to decorate; a 5% pit field is a routing bug wearing
  makeup.

An out can also be **complementary, not terminal**: even after the MFD rungs,
slope-keyed detail noise is the right tool for sub-mesh roughness the mesh
fundamentally can't carry. "Physics down to the mesh floor, keyed noise below it"
is a perfectly on-philosophy resting place.

---

## Open decisions (user is the deciding vote)

1. **Appetite for the full MFD-DAG rewrite (Rung 3)** now that it's de-risked, or
   stop at Rung 2 (Barnes flats + MFD area) and evaluate "good enough" first?
2. **Is one-shot erosion a hard architectural constraint?** It gates the
   time-coupled frontier (#4). Closed door, or just not-yet?
3. **Run the Rung-0 diffusivity sweep first** (diagnostic instinct), or go
   straight to Barnes flats (don't-hide-wounds instinct)?

---

## References

- Barnes, Lehman & Mulla (2014). *An efficient assignment of drainage direction
  over flat surfaces in raster DEMs.* Computers & Geosciences 62:128–135.
  (Convergent flat resolution; the parallel/spiral-flow fix.)
- Salles et al. (2019). *eSCAPE: Regional to Global Scale Landscape Evolution
  Model.* Geosci. Model Dev. 12:4165. (Implicit MFD on unstructured global meshes
  at millions of nodes — hex3's scale and geometry.)
- Anand, Hooshyar & Porporato (2020). *Linear layout of multiple flow-direction
  networks for landscape-evolution simulations.* Env. Modelling & Software 133;
  and *Channelization cascade in landscape evolution*, PNAS 117. (MFD captures the
  smooth→channelized transition D8 misses; channelization-instability theory.)
- Liu et al. (2025). *A new Multiple Flow Direction (MFD) algorithm for modeling
  ridge and valley evolution.* Geomorphology.
- Tarboton (1997). *A new method for the determination of flow directions and
  upslope areas in grid DEMs* (D∞). Water Resources Research 33:309.
- Roering, Kirchner & Dietrich (1999). *Evidence for nonlinear, diffusive
  sediment transport on hillslopes…* Water Resources Research 35:853.
- Rinaldo, Rodríguez-Iturbe et al. — Optimal Channel Networks / minimum energy
  dissipation (self-organised fractal river networks).
- Braun & Willett (2013). *A very efficient O(n), implicit and parallel method to
  solve the stream power equation…* Geomorphology 180–181:170. (The implicit
  `n = 1` incision this generalises.)
