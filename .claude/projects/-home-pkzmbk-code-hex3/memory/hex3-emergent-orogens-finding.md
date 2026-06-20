---
name: hex3-emergent-orogens-finding
description: June 2026 — erosion-v3 "let erosion BUILD mountains from uplift" premise FALSIFIED for this solver (n=1 stream power can't dissect a smooth dome); this erosion dissects GIVEN relief, so painted substrate (P1) is the pragmatic path
metadata:
  type: project
---

**erosion-v3 emergent-orogens investigation — verdict (branch `erosion-v3-emergent-orogens`, 2026-06-20).** Spec: `docs/specs/erosion-v3-emergent-orogens.md`.

The chain: P1 painted substrate ([[hex3-fine-synthesis-p1]]) hit a "dissected noise / sand dunes" ceiling. Codex physical + architecture reviews diagnosed the root cause: the fine erosion stage is a POSTPROCESSOR — coarse hands it the finished saturated flat-topped plateau, and uplift is calibrated "hold & carve" (tiny, must not rebuild, because the coarse elevation already bakes in the height). So erosion can only dissect what it's given; it never builds. The physically-ideal fix: hand erosion a low ENVELOPE (`coarse − λ·(arc+collision)`, exactly separable since isostasy is linear) + an active uplift-RATE field, and let ranges EMERGE (Perron/Dietrich channelization).

Prototype built (behind `emergent_lambda`, default 0): envelope demotion + self-calibrating builder uplift (rebuild `target−base` over the epoch ×`EMERGENT_REBUILD_GAIN`, so height tracks target and `steps` is a build-vs-carve dial), builder gated on the coarse-target mask, rift excluded, terminal lakes off. Mechanically sound (height rebuilds ~90% target, land flips ~2% at λ=0.5).

**FALSIFIED on morphology.** The emergent orogen does NOT dissect — it stays a smooth swell. Numerical pre-screen (seed 12345, λ=0.5): more steps → SMOOTHER (summit slope 2.41e-4→2.28e-4 over 120→400 steps), aggressive channelization (channel-support 3, diffusivity 5e-9, SFD) doesn't help (drainage spacing ~1000 km vs Earth 0.5-5 km/km²). Emergent summits ~2.3e-4 < global land 3.8e-4 (smooth); painted P1a ~2.8e-3 (~10× more dissected). **Cause: n=1 stream-power incision can't manufacture relief from a smooth uplift dome (gentle slopes → weak stream power → no channel cutting; hillslope diffusion then smooths it). No channelization instability / threshold incision.** This solver dissects relief it is GIVEN, can't BUILD it.

**Implication:** true emergent terrain needs an EROSION-SOLVER rewrite (threshold or n>1 stream-power incision + proper hillslope–fluvial competition) — a weeks-not-days landscape-evolution project, NOT the interface swap. The envelope/uplift/self-calibrating scaffolding is retained (default-off) and reusable IF that solver work is done. Pragmatic path: painted P1 (give this erosion good rough substrate). Decision pending user.

**Also banked this session — RENDERING FIX (keep!):** the relief view shaded from the smooth SPHERE normal (relief displaced in the vertex shader but normal never recomputed) → terrain slopes caught no light, peaks blew out to flat white. Fixed with a `slope_shading` uniform: shade from the displaced FACE normal (screen-space `dpdx/dpdy` of world_pos). The sweep enables it + simple directional light (hemisphere was washing peaks). Interactive app unchanged (flag default off) — but it has the SAME washed-out problem and would benefit from the flag on. ALL prior visual judgments (P1a "noisy", P1b "dunes") were made on the BROKEN lighting — re-judge on hillshaded renders. Sweep folders: `C:\code\hex3\sweeps\{p1_hillshade,v3_hillshade}`.
