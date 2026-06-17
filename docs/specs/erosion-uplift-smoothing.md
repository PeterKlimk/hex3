# Erosion Escalation #1 — Uplift-Forcing Smoothing

**Provenance.** First structural escalation from
[`erosion-escalations.md`](erosion-escalations.md) (trio, 2026-06-16/17). Targets
the source the component-isolation probe confirmed as the biggest single
contributor to the mountain-top cell-scale "swiss cheese": the **erosion uplift
source**.

See also: [`erosion.md`](erosion.md), [`erosion-escalations.md`](erosion-escalations.md).

**Philosophy.** Mechanism with a physically-named knob, not a cosmetic blur of the
output. Smooth the *forcing*, never the final elevation.

## The problem

`ErosionState::new` builds a per-cell uplift source, applied every step:

```text
u_thick[i] = uplift_scale * ((arc[i] + collision[i]) * inv_slope + rift_delta[i])   // land only
...
thick[i] += u_thick[i] * dt      // every step, ×EROSION_STEPS
```

`arc`, `collision`, `rift_delta` are fine-mesh `ElevationFields` (interpolated from
the coarse boundary analysis). If they carry **wavelengths below the landform
scale**, erosion equilibrates against a noisy source and re-injects high-frequency
tectonic "work" every step — so the steady-state terrain contains cell-scale
chatter on the orogens. Isolation (uplift off → curv-rms −13%, fewer peaks)
pinpoints this as the dominant mountain-bump source. This is **not** a failure of
diffusion to clean up afterward; it is continuous high-frequency injection.

## Step 0 — diagnose WHERE the speckle originates (before any fix)

The fix locus depends on this, so measure first (headless):

- Visualise / probe `u_thick` and each component (`arc`, `collision`, `rift_delta`)
  on the fine mesh — `Tessellation::morans_i` per field + a render.
- **If the coarse feature field is already speckled** before erosion sees it → fix
  the *feature field* (or its generation), not the erosion forcing.
- **If the coarse field is smooth but the coarse→fine transfer speckles it** (e.g.
  nearest-cell / blocky interpolation) → smooth the erosion forcing at the fine
  handoff. ← expected locus, but confirm.

## Mechanism

Smooth the uplift **source** field `u_thick` (or its components) once, in
`ErosionState::new`, over the fine mesh, with a **physically-named length scale**
— sub-grid orogenic forcing width / flexural-process smoothing (real tectonic
uplift is smooth at cell scale; sub-cell variation is unmodelled detail). Apply a
neighbour/area-weighted diffusion or kernel of that width to `u_thick` before the
step loop; the per-step application is unchanged.

### Knob

`EROSION_UPLIFT_SMOOTH_KM` (or `_RADIANS`) — the forcing smoothing length.
`0` = off (current behaviour). Exposed via `ErosionParams` + diagnose/app
(`--erosion-uplift-smooth`) for A/B, like the other erosion knobs.

## Traps (from the trio)

- **Smooth the source, not the final elevation** — preserve *integrated* uplift
  (a smoothing kernel that conserves the area-weighted sum does this; don't let it
  bleed total uplift away).
- **Smooth `arc` / `collision` / `rift_delta` separately**, or at least be careful:
  `rift_delta` is a *signed* thickness delta (negative in the axial valley); a
  naive blur of the combined field can let signed rift thinning cancel or smear
  into positive orogen uplift. Prefer smoothing the components, then combine.
- **Don't spread uplift across the coast/ocean mask.** The source is gated to land
  (`base >= 0`); a smoothing kernel that pulls uplift onto submerged cells
  reintroduces the spurious-land / submerged-uplift problem the gate exists to
  prevent. Smooth within the land mask (no-flux at the coastline).
- Don't over-smooth: the goal is to remove sub-landform chatter, not to round off
  the genuine orogen front. Calibrate the length to the cell/landform scale.

## Validation (headless first)

- **Source speckle:** `morans_i(u_thick)` rises toward smooth after the kernel.
- **Outcome:** the carved-dissection + local-relief metrics
  ([`erosion.md`](erosion.md) roughness counters + the diagnose calibration probes)
  — mountain-top bumps (peak%, cell-scale curvature) should fall while the
  macro orogen (relief p90/p99, land volume) is preserved. Smoothing the source
  must NOT flatten mountains (that would be over-smoothing or wrong locus).
- Visual confirm on Windows (zoomed mountain terrain): prickle gone, ridgelines
  intact.

## Result (2026-06-17, built + A/B, seed 12345, full fine res ~1.3M land)

Mechanism implemented as specified: conservative, land-masked, area-weighted
diffusion of `u_thick` once before the step loop (`EROSION_UPLIFT_SMOOTH_KM` /
`--erosion-uplift-smooth`, default off). Codex-reviewed: conservation,
`w_ij=w_ji` symmetry, FV stability, length math, no-flux land mask, and the
signed-`rift_delta` linearity all check out; non-finite/huge-length guard added.

**But the A/B disconfirms the premise — it is a near-no-op on the bumps.**
The Step-0 question (is the source speckled at the fine handoff?) answers **no**:
`morans_i(u_thick)` is already **0.992** before any kernel (smoothing nudges it
to ~0.990 — global Moran's I on a mostly-zero, narrow-orogen field has no
headroom and even reads slightly *down*; it is not a usable speckle meter here).

Disentangling uplift *presence* from uplift *spatial frequency* (eroded surface):

| metric        | uplift OFF | baseline | smooth 20 km | smooth 50 km |
|---------------|-----------:|---------:|-------------:|-------------:|
| peak %        | 1.665      | 1.801    | 1.817        | 1.825        |
| curv-rms ×e-2 | 1.808      | 2.046    | 2.061        | 2.065        |
| R=10 p90 (m)  | 1305       | 1489     | 1469         | 1452         |
| R=25 p90 (m)  | 2046       | 2316     | 2305         | 2280         |

Turning uplift **off** drops curv-rms −11.6% and peak% −7.5% (reproduces the
isolation result the spec was built on — and lowers the orogen, the held-height
cost). But **smoothing the source spatially does not** — curv-rms/peak% are flat
to slightly *worse* even at 50 km; only a faint macro-relief softening (R=10/R=25
p90 drift down ~2%) appears, i.e. mild over-smoothing of the front, not
de-prickling. So the uplift bump contribution is in the forcing **magnitude ×
erosion-dynamics interaction**, NOT sub-landform spatial speckle in the source —
because the coarse→fine transfer keeps the source smooth (Moran's I 0.992), there
is no speckle for a spatial kernel to remove.

**Verdict:** land the mechanism as correct, cheap, default-off infra (a real
physical knob), but it is the **wrong locus** for the swiss-cheese bumps. The
lever on the uplift-attributable prickle is its *magnitude* (`uplift_scale` /
`steps` calibration), and the structural de-prickling lever is **#2 (Roering
nonlinear diffusion)**, not source smoothing. Escalation #1 does not pass its own
validation gate as a bump fix; proceed to #2 and re-evaluate there.

## After this — the evaluation gate

Per the roadmap: build #1, then #2 (Roering nonlinear diffusion), then **stop and
look**. If uplift-smoothing + Roering leaves mountains *busy but no longer
prickly*, proceed to the transport regime (#3). If still prickly, the problem is
still in forcing/incision scale — rethink before building river mechanics.
