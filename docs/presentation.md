# Presentation contract

This is a canonical policy for the boundary between modeled state and visual
communication. It should be read with the [project thesis](thesis.md) and
[semantic and presentation architecture](semantic-presentation.md). This file
records the currently implemented relief/river contract; the architecture
document defines the broader semantic, generalization and profile model. The
preset named `Authentic` is the current code/UI name; in architecture prose it
is the authentic/cartographic preset, not physical scale.

Physical elevation, datum and true-scale displacement are defined by the
[elevation and unit contract](units.md).

Hex3 stores and audits terrain in physical elevation units. Rendering is a
cartographic presentation layer: displacement, line width, color, and lighting
may be intentionally exaggerated, but they do not feed back into world
generation or terrain acceptance decisions.

## Relief presets

| Preset | Scale | Approx. exaggeration | Purpose |
|---|---:|---:|---|
| Flat | 0 | 0x | map/data inspection |
| Physical | `ELEVATION_UNIT_KM / PLANET_RADIUS_KM` | 1x | scientific geometry inspection |
| Authentic | 0.04 | 25x | product/default presentation |
| Dramatic | 0.08 | 51x | optional showcase view |

`Authentic` was selected by a Windows renderer-only A/B on identical seed-12345
terrain and camera. The old 0.20 scale (~127x) turned a broad 7.6 km massif into
an apparent thousand-kilometre tower. At 0.04, secondary ranges remain legible
and the same massif has a plausible visual silhouette.

Normal interactive use:

```powershell
cargo run --release --bin hex3 -- --relief-preset authentic
cargo run --release --bin hex3 -- --relief-preset physical
cargo run --release --bin hex3 -- --relief-scale 0.025
```

`--relief-scale` is an explicit custom override and takes precedence over the
named preset. Press `X` to cycle Flat → Physical → Authentic → Dramatic.
The selected scale is shared by terrain, relief edges, and surface-wind
particles.

## Rivers

River topology and selection remain physical: catchment area determines which
channels qualify. Visible width is cartographic and screen-space because even
the largest real rivers are subpixel in globe views.

```powershell
cargo run --release --bin hex3 -- --river-width-scale 0.75
```

This multiplier changes only the stroke width. It does not alter hydrology,
flow, catchment thresholds, or downstream widening.

## Validation separation

Terrain gates use exported/numeric quantities: kilometres of elevation and
relief, range width and taper, drainage structure, coarse-to-fine fidelity, and
component volume ledgers. They must not depend on a render scale.

Presentation gates record the preset and fixed camera. They cover apparent wall
angle, silhouettes, river screen width, snow/color coverage, and lighting.
Every visual A/B must state its relief preset; a screenshot without one is not
evidence for changing terrain generation.
