# Elevation and unit contract

This document defines the units Hex3 currently supports and names quantities
that remain normalized or dimensionless. It is a contract for interpretation,
not a claim that every model is dimensionally physical.

The machine-readable counterpart is `RunManifest.units` with contract version
`1`. Canonical conversions live in `src/world/units.rs`.

## Coordinate systems

| Quantity | Stored coordinate | Physical interpretation |
|---|---|---|
| Surface position | Unit `Vec3` | Direction from planet center |
| Angular distance | Arc radians on unit sphere | radians × `PLANET_RADIUS_KM` |
| Chord distance | Euclidean distance between unit vectors | approximates arc for short distances; convert deliberately |
| Cell area | Steradians | steradians × radius² = km² |
| Elevation | Normalized elevation unit | one unit = `ELEVATION_UNIT_KM` = 10 km |
| Crust thickness | Reference-column thickness | no direct kilometre conversion; mapped to elevation by isostasy |
| Native terrain slope | Elevation units per arc radian | calibrated simulation coordinate, not physical grade |
| Physical surface grade | Vertical km / horizontal km | `tan(slope angle)` |
| Render displacement | Fraction of globe radius | elevation × renderer relief scale |

`PLANET_RADIUS_KM` is currently 6371 km. This gives Earth-scale horizontal
measurements; it does not require every generated process to be Earth-like.

## Elevation

Elevation values use one canonical conversion:

```text
elevation_km = model_elevation × ELEVATION_UNIT_KM
ELEVATION_UNIT_KM = 10 km
```

This is a defined project interpretation rather than a value inferred separately
by every diagnostic. Code should use `elevation_to_km`, `elevation_to_meters`,
`km_to_elevation` or `meters_to_elevation` instead of multiplying by `10` or
`10_000`.

Positive elevation is above the solved sea-level datum; negative elevation is
below it. Elevation differences use the same conversion.

## Sea-level datum

The coarse elevation assembly selects a uniform shift so the area-weighted land
fraction matches `LAND_FRACTION`. The resulting sea level is defined as
elevation zero.

This is:

- one global datum solved on the coarse mesh;
- inherited by fine interpolation and erosion;
- area-weighted, not cell-count-weighted;
- unchanged by cartographic relief;
- not a solution of conserved ocean-water volume, geoid, ice volume or dynamic
  sea-level feedback.

Fine surfaces must not solve a new zero independently. Erosion, lakes and
rendering operate relative to the inherited datum.

## Crust thickness and isostasy

`CRUST_THICKNESS_CONTINENTAL = 1.0` and
`CRUST_THICKNESS_OCEANIC = 0.25` define a reference-column coordinate. They do
not mean 1 km, 10 km or a literal thickness ratio suitable for reporting.

The Airy-style relation is calibrated by continental and oceanic
thickness/elevation anchors. `isostasy_slope()` converts a change in reference
thickness into a change in elevation units. Only after that conversion may
`ELEVATION_UNIT_KM` be applied.

Consequences:

- crust volume ledgers in thickness × steradians are model/reference volumes;
- their isostatic surface expression can be reported in kilometres;
- code must not multiply raw crust thickness by 10 km;
- a future physical crust-thickness model may fundamentally replace this
  coordinate rather than merely rename it.

## Distance and area

Canonical conversions are:

```text
distance_km = arc_radians × PLANET_RADIUS_KM
area_km² = solid_angle_sr × PLANET_RADIUS_KM²
```

Short-distance code sometimes uses unit-sphere chord length to avoid precision
loss from `acos(dot)` on dense meshes. Chord and arc are close locally but are
not identical globally; documentation and APIs must state which is stored.

## Two slope coordinates

Hex3 currently has two legitimate but very different slope quantities.

### Native simulation slope

```text
S_native = Δelevation_units / Δarc_radians
```

Erosion diffusion, depositional thresholds, fine-density allocation and several
calibrated atmosphere/presentation heuristics use this coordinate. It is useful
and resolution-independent when distances are angular, but it is not
`tan(angle)`.

### Physical grade

```text
grade = S_native × ELEVATION_UNIT_KM / PLANET_RADIUS_KM
angle = atan(grade)
```

At current scales, native slope is approximately physical grade × 637. A native
slope of 1 therefore is not a 45° physical slope; it is a grade of about 0.00157,
or 0.09°.

Use `elevation_per_radian_to_grade`,
`grade_to_elevation_per_radian`, and `grade_to_degrees` whenever making a
physical slope claim.

### Existing calibrated heuristics

Terrain-wind blocking, wind permeability and terrain coloring historically use
native slope with empirical response functions. This unit pass labels them as
calibrated heuristics and preserves their output. It does not silently reinterpret
their parameters as physical angles.

Future work may replace those responses with physical-grade models. Such a
change is a model A/B requiring climate, hydrology and visual validation—not a
unit-only refactor.

## Relief rendering

Shaders displace a unit-sphere vertex by:

```text
radius = 1 + model_elevation × relief_scale
```

True-scale radial geometry therefore requires:

```text
PHYSICAL_RELIEF_SCALE = ELEVATION_UNIT_KM / PLANET_RADIUS_KM
```

The vertical-exaggeration factor for any renderer scale is:

```text
exaggeration = relief_scale / PHYSICAL_RELIEF_SCALE
```

At current constants:

- Physical: 1×;
- Authentic/cartographic (`0.04`): about 25.5×;
- Dramatic (`0.08`): about 51×.

The scale changes only radial displacement, relief-aware lines and wind-particle
surface following. It cannot change model elevation, physical validation,
hydrology or erosion.

## Time and rates

Plate motion and tectonic history expose explicit km/Myr, rad/Myr and Myr
conversions. The accepted erosion epoch does not: `steps × dt`, erodibility,
diffusivity and direct uplift form a calibrated dimensionless regime.

Therefore:

- tectonic rates may be reported in their defined physical units;
- the default 200 erosion steps must not be reported as geological duration;
- coupling tectonics and erosion on one clock requires a new validated model,
  not unit labels on the current parameters.

## Contract invariants

Tests protect:

- elevation/metre/kilometre round trips;
- arc/kilometre round trips;
- native-slope/physical-grade round trips;
- exact 1× physical relief;
- render-preset use of the canonical physical scale;
- the coarse sea-level datum remaining the inherited zero for fine stages.

