> **Archived:** Historical design or experiment record. It does not define current architecture, defaults, or priorities. Start at [`docs/README.md`](../../README.md).

# Spec: Three Cheap Physical Mechanisms (climate & ocean floor)

Roadmap item #4 (`docs/physically-inspired-roadmap.md`): three small,
independent mechanisms. Each replaces a place where a physical driver is
missing or implicit. They share one philosophy constraint: add the
*mechanism* with named constants as knobs; do not tune constants to chase
specific numbers (tuning happens after review). Where this spec is silent,
match the existing style of the module and prefer the simplest faithful
version. Leave `// SPEC:` comments for ambiguities instead of resolving
them ad hoc.

## A. Land-ocean thermal contrast

Water's high heat capacity moderates ocean surface temperature; land
deviates further from the global driving temperature, and continental
interiors deviate most (continentality). Today `atmosphere.rs` computes
surface temperature from latitude + elevation lapse only — land and ocean
at the same latitude are identical.

Mechanism: define a per-cell continentality factor in [0, 1] — 0 on open
ocean, rising with distance from the nearest ocean cell (ocean = below sea
level, so shelf seas moderate their surroundings; a saturating distance
scale constant, order ~0.05-0.15 rad). Let continentality *amplify* the
cell's deviation from a reference temperature (suggest: the global mean of
the latitude-only base temperature) by a factor `1 + CONTINENTALITY_AMP *
continentality`. Hot latitudes get hotter inland, cold latitudes colder —
one mechanism, both effects. Apply to surface temperature only (not the
upper layer driving pressure, to keep the wind solve unchanged for now;
note with a comment that land-driven pressure anomalies are future work).

Knobs: amplification strength, continentality distance scale.

## B. Per-basin evaporation from temperature

Lake equilibrium (`hydrology.rs`) uses one global `climate_ratio`
(precipitation/evaporation) for every basin. Physically, evaporation rises
steeply with temperature: hot basins should hold shrunken terminal lakes,
cold mountain basins fill to the sill.

Mechanism: compute each basin's mean surface temperature (area-weighted
over its catchment or its lake cells — implementer's choice, note which)
and scale the evaporation side of the balance by a monotonic function of
it, normalized so a basin at the global mean temperature behaves exactly
as today (the global `climate_ratio` knob keeps its meaning, including the
interactive Up/Down adjustment). Suggest a simple linear or exponential
factor with one sensitivity constant.

Knobs: evaporation-temperature sensitivity.

## C. Spreading rate -> ocean age

Ocean floor age currently equals distance-from-ridge at an implicit
constant spreading rate (`oceanic_age_factor_from_ridge_distance`). The
actual divergence rate per ridge segment is already computed in
`boundary.rs` (`convergence`, negative = opening). Fast-spreading ridges
should have broad young swells; slow ridges narrow ones.

Mechanism: carry the local opening rate along with the ridge distance
field (seed it at ridge cells the same way ridge strength/dist0 are
seeded, propagate with the distance field or as a smoothed forcing —
follow whichever existing pattern fits), and compute age ~ distance /
rate, normalized by a reference spreading rate constant so the current
behavior is recovered when all ridges open at the reference rate. Feed the
result through the existing age factor everywhere it is consumed (thermal
subsidence, trench depth multiplier, flexure alpha) — the consumers should
not change, only the age input. Guard the slow-spreading limit (rate -> 0
must saturate to "old", not divide by zero).

Knobs: reference spreading rate.

## Validation

```bash
cargo fmt && cargo test
cargo run --release --bin hex3 -- --headless --seed 12345 --export /tmp/w.json.gz
cargo run --release --bin diagnose -- --seed 12345
```

- All existing tests pass, including field smoothness (Moran's I) — none
  of the three mechanisms may introduce cell-scale speckle.
- New unit tests where a property is checkable in isolation (e.g. age
  saturation as rate -> 0; continentality = 0 on ocean cells; basin at
  mean temperature reproduces the global ratio).
- diagnose: add one line per mechanism if a natural summary exists (e.g.
  land-ocean mean temperature delta; lake count vs before; age field
  min/median/max). Keep it light.
- Briefly report what changed on seed 12345 (climate arid %, lake %,
  trench/flexure stats) in the commit message — changes are expected and
  fine; we review magnitudes, not targets.

## Non-goals

- No changes to the wind solve, pressure, moisture transport, or rivers
  beyond what flows naturally through the modified inputs.
- No retuning of existing constants.
- No changes under `scripts/`; no new dependencies.
- Mechanism A explicitly excludes seasonal cycles and land-driven pressure.
