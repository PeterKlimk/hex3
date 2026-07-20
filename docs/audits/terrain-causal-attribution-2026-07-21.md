# Terrain causal attribution

Status: **bounded causal decision**, 2026-07-21.

This audit asks why long smooth mountain roofs recur in the Legacy product
terrain. It does not promote a replacement model. Current policy is summarized
in [landscape strategy](../landscape-strategy.md).

## Fixed observation

- revisions: trace `9db4f5f`, counterfactual `b468677`, matched capture
  `19014fa`;
- seeds: `12345`, `8675309`, `9001`;
- mesh: 100,000 requested coarse cells and about 255,000 fine cells;
- pipeline: Legacy, Stage 4, default physical parameters;
- objects: highest-peak and largest distinct component above 1.5 km and
  20,000 km² in each world;
- views: Physical/Diagnostic evidence for validity, with one common `0.04`
  relief scale used only for matched visual discrimination.

The trace records real contributing collision fronts and history episodes, then
follows their signal through raw seed, normalization, diffusion, square-root
amplitude, cap, Gaussian distance kernel and collision response. It also
constructs two no-erosion, positive-work-matched counterfactuals:

1. propagate the nearest real collision seed without the Legacy forcing
   smoother; and
2. replace seed amplitudes with their real episode means as a low-rank null.

Both retain the Legacy square-root amplitude and Gaussian cross-front kernel.
They isolate compiler information without pretending to be replacement terrain
models.

## Results

Collision response supports 99.9–100% of every selected range and contributes
roughly 46–70% of positive coarse-interpolant elevation-area. Exact
`capped amplitude × Gaussian kernel` reconstruction error is zero. Hard response
saturation is zero in all six ranges, so the cap is not active here. The fine
base and coarse interpolant are exactly equal.

Source variation is real and the compiler compresses it substantially:

| Seed / range | Raw seed CV | Smoothed CV | sqrt CV | Legacy response CV | nearest-source response CV |
|---|---:|---:|---:|---:|---:|
| 12345 / largest | 0.848 | 0.497 | 0.258 | 0.619 | 0.635 |
| 12345 / highest | 0.423 | 0.131 | 0.067 | 0.778 | 0.790 |
| 8675309 / highest | 0.504 | 0.239 | 0.127 | 0.586 | 0.592 |
| 8675309 / largest | 0.557 | 0.137 | 0.069 | 0.424 | 0.440 |
| 9001 / largest | 0.797 | 0.323 | 0.167 | 0.634 | 0.701 |
| 9001 / highest | 0.410 | 0.210 | 0.120 | 0.602 | 0.599 |

The nearest-source compiler meaningfully reduces the 500 m summit-cap area in
the seed-9001 largest range (593,375 to 122,790 km²) and the seed-8675309
largest range (17,614 to 8,332 km²). It makes little difference or reverses
direction in the other four objects. Whole-component area below 1% grade remains
66–91% in every nearest-source surface. The episode-mean null generally
preserves or worsens the roof.

Matched diagnostic inspection reaches the same discriminator: nearest-source
forcing changes local amplitude and texture, but the first three compiler rows
remain recognizably the same broad distance-band object in at least two worlds.
Only the final eroded row supplies substantial surface dissection. Numerically,
erosion lowers the whole-component gentle fraction in all six ranges, from
72–93% before erosion to 26–68% final, but seed 9001 remains visibly broad.

## Causal verdict

The immediate owner is the Legacy representation:

```text
long connected collision front
  -> normalized and diffused scalar forcing
  -> compressed amplitude
  × broad Gaussian distance-from-front kernel
  -> direct finished height over nearly the whole range
```

Normalization, diffusion and square-root response erase useful amplitude
variation, but merely removing diffusion does not generally reveal a missing
range grammar. Cross-front distance geometry and direct-height ownership supply
too few regional degrees of freedom. Fine interpolation is neutral. Erosion is
a powerful mitigator and detail generator, not a reliable first-order range
owner. Prior same-world ancestry already established repeated uplift as a
secondary amplifier rather than the original cause.

This does not imply that every long continuous belt is wrong. One reviewed
3,200 km parent genuinely has one continuous opportunity maximum and no
defensible internal low. Reality can produce broad plateaus; Hex3's defect is
using one smooth belt response generically even when available forcing varies.

## Decision

Stop tuning or further decomposing Legacy. Keep it as the usable control. Do not
send the nearest-source counterfactual through another terrain stage: it fails
the preregistered test of exposing coherent organization in two of three worlds.

The next bounded comparison should replace direct finished-height ownership.
It should consume finite deformation/material opportunity where current state
honestly supplies it, then couple that opportunity to drainage/divide
organization and nonlinear hillslope response. A reduced causal model or
authentic structural hack is acceptable; deeper physics is justified only when
it preserves a visible or downstream consequence more cheaply approximated
models cannot.

## Reproduction and limits

The numeric packets are under
`artifacts/terrain-causal-trace-v1/seed-*-counterfactual.json`. Matched rows
(baseline interpolant, nearest-source, episode-mean null, final terrain) are
under `artifacts/captures/terrain-causal-trace-v1/seed-*/`.

The compiler counterfactuals are globally work-matched and intentionally omit
erosion. Component masks come from final terrain, so this is attribution over
fixed visible objects rather than a competitive replacement evaluation. CV is
an amplitude statistic, not evidence of coherent spatial organization; the
matched images provide that separate check. Human visual review remains
authoritative and may revise the morphology reading. Results establish the cause
in this fixed ordinary-world corpus, not Earth calibration or universal geology.
