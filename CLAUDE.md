# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope

This file covers the hex3 app. The s2-voronoi library lives in its own repository: https://github.com/PeterKlimk/s2-voronoi

## Development Environment

Development is done in WSL2, but the application must be run on Windows. Compute shaders (used for the particle system) do not work properly under WSL2's GPU passthrough.

- Use `cargo build` in WSL2 to verify code compiles
- Build and run on Windows for actual execution (`cargo run --release` from Windows terminal)

## Build & Run Commands

```bash
cargo build              # Build debug
cargo build --release    # Build release
cargo run --release      # Run (release recommended for performance)
cargo test               # Run all tests
cargo test voronoi       # Run tests matching "voronoi"
cargo test lloyd         # Run tests matching "lloyd"
cargo clippy             # Lint
cargo fmt                # Format
```

## CLI Options

```bash
cargo run --release                              # Interactive mode, stage 1
cargo run --release -- --seed 12345              # Specific seed
cargo run --release -- --stage 2                 # Start at stage 2
cargo run --release -- --headless --export out.json.gz  # Headless + export
```

- `--headless` - Generate without window, quit when done (defaults to max stage)
- `--stage N` - Target stage (1=Lithosphere, 2=Atmosphere, 3=Hydrosphere)
- `--seed N` - Random seed for reproducible generation
- `--export FILE` - Export world data to JSON (supports .json.gz)
- `--voronoi-backend <convex-hull|knn-clipping>` - Select Voronoi algorithm (default: convex-hull)
- `D` key - Export current world in interactive mode

## Data Analysis

Export world data and analyze with Python:

```bash
# Generate and export
cargo run --release -- --headless --seed 12345 --export world.json.gz

# Analyze (requires numpy, matplotlib)
uv venv scripts/.venv
uv pip install -r scripts/requirements.txt --python scripts/.venv/bin/python
scripts/.venv/bin/python scripts/analyze_terrain.py world.json.gz --show
```

The analysis script generates:
- Elevation histogram (land vs ocean distribution)
- Hypsometric curve (cumulative area vs elevation)
- Elevation by latitude heatmap
- Tectonic feature distributions
- Plate size chart

## Project Overview

Hex3 is a spherical Voronoi-based planet generator with tectonic plate simulation, rendered using wgpu. It generates procedural worlds with realistic terrain based on plate tectonics.

## Architecture

### Module Structure

- **`src/lib.rs`** - Library crate entry (re-exports `geometry`, `render`, `world`)
- **`src/geometry/`** - Computational geometry for spherical surfaces
  - `voronoi.rs` - Spherical Voronoi diagram via convex hull duality
  - `convex_hull.rs` - 3D convex hull using qhull
  - `lloyd.rs` - Lloyd relaxation for point distribution
  - `sphere.rs` - Uniform random points on a unit sphere
  - `mesh.rs` - Voronoi to triangle mesh conversion, map projection
  - `validation.rs` - Voronoi diagram validation utilities

- **`src/world/`** - World generation and simulation
  - `tessellation.rs` - Spherical tessellation with Voronoi cells and adjacency
  - `plates.rs` - Tectonic plate assignment via flood fill (motion units)
  - `crust.rs` - Per-cell continental/oceanic crust, grown as cratons independently of plates
  - `dynamics.rs` - Plate dynamics (Euler poles, velocities)
  - `boundary.rs` - Plate boundary classification and analysis
  - `features.rs` - Tectonic feature fields (trenches, arcs, ridges, collisions)
  - `elevation.rs` - Elevation generation from features + noise
  - `atmosphere.rs` - Atmosphere simulation (temperature, pressure, wind fields)
  - `hydrology.rs` - River networks, drainage basins, lakes
  - `constants.rs` - Tunable simulation parameters

- **`src/app/`** - Application layer
  - `state.rs` - Application state and rendering
  - `view.rs` - Render modes and view modes
  - `coloring.rs` - Per-cell color functions for visualization
  - `world.rs` - World buffer generation for GPU
  - `visualization.rs` - Debug visualization helpers
  - `export.rs` - World data export to JSON

- **`src/render/`** - wgpu rendering infrastructure
  - `context.rs` - GPU device/surface setup
  - `pipeline.rs` - Render pipeline builder
  - `renderer.rs` - Main renderer with multiple pipelines
  - `camera.rs` - Orbit camera with controller
  - `buffer.rs`, `uniform.rs`, `vertex.rs` - GPU buffer utilities
  - `particles.rs` - GPU wind particle system (compute shader-based)

- **`src/shaders/`** - WGSL shaders
  - `unified.wgsl` - Main terrain rendering
  - `wind_particles.wgsl` - Compute shader for particle physics
  - `particle_render.wgsl` - Particle trail rendering

### Key Data Flow

1. Random points on unit sphere → Lloyd relaxation → evenly distributed points
2. Convex hull of points → dual graph → SphericalVoronoi (cells, vertices)
3. Spaced seeds + varied target sizes → weighted flood fill → Plates (cell assignments)
4. Craton seeds → capped noisy flood fill → per-cell Crust (independent of plates; overlay creates passive vs active margins)
5. Euler pole velocities → plate dynamics → boundary classification (crust types per cell pair)
6. Boundary analysis → feature fields (trench, arc, ridge, collision, activity)
7. Features + crust type → elevation via decay functions + multi-layer fBm noise
8. Elevation + latitude → atmosphere simulation → temperature, pressure, wind fields
9. Elevation + atmosphere → hydrological simulation → rivers, lakes, drainage basins
10. World state → hypsometric coloring → VoronoiMesh → GPU buffers
11. Wind field → GPU particle system → animated wind visualization
12. Relief view: vertices displaced radially by averaged elevation

### Core Types

- `SphericalVoronoi` - Voronoi diagram with generators, vertices, and cells
- `Tessellation` - Voronoi + adjacency graph + cell area computation
- `Plates` - Cell-to-plate assignments (motion units; plates carry mixed crust)
- `Crust` - Per-cell crust type (Continental/Oceanic) + signed margin distance field
- `Dynamics` - Euler poles, velocities
- `FeatureFields` - Per-cell tectonic feature magnitudes (trench, arc, ridge, collision, activity)
- `Atmosphere` - Temperature, pressure, wind vectors, uplift per cell
- `Hydrology` - River network, drainage basins, lake levels
- `World` - Complete world state (tessellation, plates, dynamics, features, elevation, atmosphere, hydrology)
- `VoronoiMesh` - Triangle mesh with per-vertex colors for rendering
- `GpuContext` - wgpu device, queue, surface configuration
- `WindParticleSystem` - GPU compute-based particle system for wind visualization

### Tectonic Simulation

Plates rotate around Euler poles. At boundaries, relative velocity determines feature type:
- **Convergent**: subduction (trenches + volcanic arcs) or collision (mountain ranges)
- **Divergent**: mid-ocean ridges (oceanic) or rifts (continental)
- **Transform**: lateral motion (no elevation features)

Eight plate interaction multipliers (4 convergent + 4 divergent) in `world/constants.rs`:
- Convergent: `CONV_CONT_CONT`, `CONV_CONT_OCEAN`, `CONV_OCEAN_CONT`, `CONV_OCEAN_OCEAN`
- Divergent: `DIV_CONT_CONT`, `DIV_CONT_OCEAN`, `DIV_OCEAN_CONT`, `DIV_OCEAN_OCEAN`

Boundary forcing is weighted by edge arc length and normalized by cell area for resolution-independent results. Elevation response differs by plate type:
- **Continental**: asymmetric (compression → mountains, tension → rifts capped above ocean floor)
- **Oceanic**: thermal subsidence from ridge distance + feature-driven uplift

## Controls (Runtime)

- Drag: rotate globe
- Scroll: zoom
- Tab: toggle globe/map view
- 1-8: Render modes:
  - 1: Relief (default) - 3D terrain + lakes + wind particles
  - 2: Terrain - flat terrain + lakes
  - 3: Elevation - raw elevation only
  - 4: Plates - plate boundaries and velocities
  - 5: Noise - fBm noise contribution (press again to cycle layers)
  - 6: Hydrology - flow accumulation coloring
  - 7: Features - tectonic feature fields (press again to cycle: Trench/Arc/Ridge/Collision/Activity)
  - 8: Climate - atmosphere visualization (press again to cycle: Temperature/Wind (Surface)/Wind (Upper)/Uplift) - Stage 2+
- W: toggle between surface and upper wind (enters Climate mode if not already) - Stage 2+
- E: toggle edge visibility
- V: cycle river visibility (Off/Major/All) - Stage 3+
- H: toggle hemisphere lighting
- R: regenerate world with new seed
- Space: advance to next stage (1=Lithosphere → 2=Atmosphere → 3=Hydrosphere → 4=Erosion); if viewing an earlier already-computed stage, moves the view forward instead of recomputing
- Backspace: view the previous stage (no recompute; data is retained). Stage 3 = pre-erosion fine terrain, stage 4 = eroded — Space/Backspace snap between them for an instant before/after on erosion.
- Up/Down: adjust climate ratio (wetter/drier) - controls lake levels (Stage 3+)
- D: export world data to JSON file (exports the latest computed stage)
- Esc: quit

Notes:
- Plates mode (globe view) overlays plate velocity arrows and Euler pole markers.
- Rivers follow terrain elevation in Relief mode, flat in other modes.
- Stage navigation (Space/Backspace) renders already-computed stages without recompute; once both stage 3 (pre-erosion) and stage 4 (eroded) are visited, snapping between them is instant (per-stage GPU buffers are cached). The fine-mesh base is disk-cached (`.cache/finebase/`, keyed by seed + mesh params) so a recompile of erosion/downstream code reloads it instead of regenerating (`--no-fine-cache` / `--rebuild-fine-cache`). See docs/specs/staging.md.

## Voronoi Backends

Two backends for spherical Voronoi computation:
- **convex-hull** (default): qhull-based convex hull duality, mathematically exact
- **knn-clipping**: s2-voronoi crate, kNN-driven half-space clipping

The knn-clipping backend uses the external `s2-voronoi` crate (https://github.com/PeterKlimk/s2-voronoi, pulled in as a git dependency). Integration point is `Tessellation::generate_knn_clipping()` in `src/world/tessellation.rs`.

## Common Edit Points

- World resolution: `src/app/world.rs` (`NUM_CELLS`, `NUM_PLATES`; Lloyd relaxation count is fixed inside `Tessellation`)
- Continent layout: `src/world/constants.rs` (`NUM_CRATONS`, `CONTINENTAL_FRACTION`, `CRATON_*` — one big craton at high coverage = Pangaea world, many small at low coverage = archipelago)
- Tectonic feature tuning: `src/world/constants.rs`
- Elevation & noise tuning: `src/world/constants.rs` (noise layers, feature sensitivities)
- Plate generation heuristics: `src/world/plates.rs` (seed spacing, target sizes, noise)
- Coloring functions: `src/app/coloring.rs`
