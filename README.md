# Hex3

Hex3 is a real-time procedural planet system built around spherical Voronoi
geometry, plate tectonics, climate, hydrology and erosion. It aims to produce
worlds that are visually striking because their geography has coherent causes:
part globe simulator, part cartographic planet demo, and potentially part
strategy-game world.

The project favors physically based systems and authentic shortcuts where they
create useful emergence, then uses an explicit cartographic presentation layer
to make mountains, rivers and other features legible at planetary scale.

## Current capabilities

- spherical Voronoi planets with coarse and adaptive fine tessellations;
- independent tectonic plates and continental/oceanic crust;
- Euler-pole motion, plate boundaries and tectonic terrain features;
- crustal/isostatic elevation and ocean bathymetry;
- global temperature, circulation, wind and moisture transport;
- fine drainage, basins, lakes and river networks;
- fluvial and hillslope erosion with retained pre/post surfaces;
- globe and equirectangular map views;
- relief, terrain, elevation, plate, hydrology, tectonic and climate views;
- GPU wind particles, cartographic river rendering and controllable relief;
- diagnostics, scorecards, controlled render sweeps and world export.

World exports include an effective-run manifest with source revision, dirty
state, model parameters, stage, backend and fine-cache provenance.

The implemented pipeline currently reaches an eroded surface with final
hydrology. Biomes, vegetation, persistent sediment, oceans/ice and human-world
systems are roadmap candidates rather than current capabilities.

## Build and run

Development uses WSL2, but the application must run on Windows: compute shaders
used by wind particles do not work correctly through WSL2 GPU passthrough.

In WSL2:

```bash
cargo build
cargo test
cargo clippy
cargo fmt
```

From a Windows terminal:

```powershell
cargo run --release --bin hex3
```

Release mode is strongly recommended for generation and rendering performance.

Useful reproducible/headless options include:

```powershell
cargo run --release --bin hex3 -- --seed 12345
cargo run --release --bin hex3 -- --stage 4
cargo run --release --bin hex3 -- --headless --seed 12345 --export world.json.gz
```

Run `cargo run --release --bin hex3 -- --help` for the current model, diagnostic and
presentation options.

## Interaction

| Input | Action |
|---|---|
| Drag / scroll | Rotate globe / zoom |
| Tab | Globe / map |
| `1`–`8` | Relief, Terrain, Elevation, Plates, Noise, Hydrology, Features, Climate |
| `W` | Surface/upper wind visualization |
| `E` | Cell edges |
| `V` | Off/Major/All rivers |
| `X` | Relief presentation preset |
| `H` | Lighting style |
| Space / Backspace | Advance computation / view an earlier retained stage |
| Up / Down | Hydrologic climate ratio |
| `R` | Regenerate |
| `D` | Export latest computed world |
| Escape | Quit |

## Documentation

Start at [`docs/README.md`](docs/README.md). In particular:

- [`docs/thesis.md`](docs/thesis.md) — project goals and fidelity philosophy;
- [`docs/architecture.md`](docs/architecture.md) — current architecture;
- [`docs/pipeline.md`](docs/pipeline.md) — stages and retained state;
- [`docs/roadmap.md`](docs/roadmap.md) — active horizons and decision gates;
- [`docs/validation.md`](docs/validation.md) — evidence and promotion policy.

Historical specs and experiments are preserved outside the canonical reading
path and should not be treated as current behavior without checking code and the
experiment registry.

## Related project

The optional kNN-clipping backend uses
[`s2-voronoi`](https://github.com/PeterKlimk/s2-voronoi), maintained separately.
