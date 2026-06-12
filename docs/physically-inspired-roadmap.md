# Physically-Inspired Replacements for Ad-Hoc Mechanisms

Survey of places where a painted/prescribed mechanism could be replaced by a
physical principle (June 2026). Pattern to look for, learned from the uplift
fix: *the physical quantity is often already computed, just not used* (the
projection potential phi was documented as "useful as an uplift proxy" years
before anything used it).

Philosophy reminder: the goal is "unreasonably physically inspired," not
Earth simulation — mechanisms should be physical because that makes worlds
coherent; parameters stay playground knobs.

## 1. Crust thickness + Airy isostasy (highest leverage)

**Current ad-hoc:** hypsometry painted by constants — `CONTINENTAL_BASE`,
`ABYSSAL_DEPTH`, `MARGIN_DEPTH`, linear shelf blends, Gaussian collision
band, tanh volcanic-island cap.

**Physical replacement:** crust columns float on the mantle; elevation
derives from thickness x density contrast. Upgrade the crust field's
continuous `signed_margin_distance` to an actual thickness field. Then:
- continental base / abyssal depth / margin profile derive from the
  thickness ramp (shelf and slope shapes come from how thickness tapers)
- collision = crust thickening -> isostasy raises it -> broad plateaus
  emerge (the "collisions are narrow ridgelines" critique dissolves)
- rifts = thinning -> subsidence for free
- volcanic island cap becomes load-induced subsidence

Thickness is also the natural state variable for future time evolution
(collision and erosion both modify it). Effort: medium. Composes with #2.

## 2. Elastic plate flexure for trenches

**Current ad-hoc:** symmetric `exp_decay` on a distance field; the review
wishlist wanted asymmetry, forearc structure, and outer rise added by hand.

**Physical replacement:** the flexure profile of an elastic plate under an
end load (analytic, `e^-x (cos x + sin x)` family) produces a deep narrow
trench, gradual rebound on the overriding side, and a subtle seaward outer
bulge — all from two parameters, evaluated along the existing distance field
using the already-computed subduction polarity. Effort: low-medium.

## 3. One overturning circulation instead of three prescriptions

**Current ad-hoc:** zonal wind bands, the subsidence desert belt, and the
ITCZ wet band are three independently prescribed latitude functions.

**Physical replacement:** prescribe the meridional overturning mass flux
(one curve in latitude: the Hadley/Ferrel/Polar cell structure). Zonal winds
derive from Coriolis turning of the meridional flow; subsidence drying at
the cell boundaries and equatorial ascent/rain are the divergence of the
same flux. Three mechanisms collapse into one, and exotic worlds (one giant
cell, five narrow ones) get self-consistent winds and rain belts from one
knob set. Effort: medium.

## 4. Cheaper items

- **Land-ocean thermal contrast:** land has lower heat capacity, should run
  hotter/colder than ocean at the same latitude. One term; the missing
  driver for continental climate. Effort: low.
- **Per-basin evaporation from temperature:** lake equilibrium uses a global
  climate_ratio; scaling the evaporation side by basin temperature makes
  hot-desert terminal lakes vs full alpine lakes emerge. Effort: low.
- **Spreading rate -> ocean age:** depth uses distance-from-ridge at
  implicit constant spreading rate; the divergence rate per ridge segment is
  already computed in boundary.rs, so age = distance / rate. Effort: low.
- **Force-derived Euler poles:** plate motions are pure RNG; least-squares
  fit of poles to slab-pull + ridge-push forces would make kinematics match
  morphology. Effort: medium-high; defer to the time-evolution project.

## Future craziness (noted June 2026)

Zonally-asymmetric circulations — ocean gyres / "Gulf Stream" heat
transport, Walker-type cells — are deliberately outside the overturning
model (#3), which is zonal-mean by construction. They would slot in as a
separate longitude-dependent layer later, not as a rework of Psi.

## Defended as-is

- **Noise-layer modulation:** its physical replacement is erosion, which is
  resolution-gated (see algorithm-review-2026-06.md); modulated fBm is the
  honest stopgap until the high-density mesh work lands.
