# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope

This file covers the hex3 app and workspace-level guidance. For the s2-voronoi library crate, see `crates/s2-voronoi/CLAUDE.md`.

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
  - `plates.rs` - Tectonic plate assignment via flood fill
  - `dynamics.rs` - Plate dynamics (Euler poles, velocities)
  - `boundary.rs` - Plate boundary classification and analysis
  - `features.rs` - Tectonic feature fields (trenches, arcs, ridges, collisions)
  - `elevation.rs` - Elevation generation from features + noise
  - `atmosphere.rs` - Atmosphere simulation (temperature, pressure, wind fields)
  - `hydrology.rs` - River networks, drainage basins, lakes
  - `constants.rs` - Tunable simulation parameters
  - `gen.rs` - World generation orchestration

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
4. Euler pole velocities → plate dynamics → boundary classification
5. Boundary analysis → feature fields (trench, arc, ridge, collision, activity)
6. Features + plate type → elevation via decay functions + multi-layer fBm noise
7. Elevation + latitude → atmosphere simulation → temperature, pressure, wind fields
8. Elevation + atmosphere → hydrological simulation → rivers, lakes, drainage basins
9. World state → hypsometric coloring → VoronoiMesh → GPU buffers
10. Wind field → GPU particle system → animated wind visualization
11. Relief view: vertices displaced radially by averaged elevation

### Core Types

- `SphericalVoronoi` - Voronoi diagram with generators, vertices, and cells
- `Tessellation` - Voronoi + adjacency graph + cell area computation
- `Plates` - Cell-to-plate assignments
- `Dynamics` - Plate types (Continental/Oceanic), Euler poles, velocities
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
- Space: advance to next stage (1=Lithosphere → 2=Atmosphere → 3=Hydrosphere)
- Up/Down: adjust climate ratio (wetter/drier) - controls lake levels (Stage 3)
- D: export world data to JSON file
- Esc: quit

Notes:
- Plates mode (globe view) overlays plate velocity arrows and Euler pole markers.
- Rivers follow terrain elevation in Relief mode, flat in other modes.

## Voronoi Backends

Two backends for spherical Voronoi computation:
- **convex-hull** (default): qhull-based convex hull duality, mathematically exact
- **knn-clipping**: s2-voronoi crate, kNN-driven half-space clipping

The knn-clipping backend uses the `s2-voronoi` crate (see `crates/s2-voronoi/`). Integration point is `Tessellation::generate_knn_clipping()` in `src/world/tessellation.rs`.

## Common Edit Points

- World resolution: `src/app/world.rs` (`NUM_CELLS`, `LLOYD_ITERATIONS`, `NUM_PLATES`)
- Tectonic feature tuning: `src/world/constants.rs`
- Elevation & noise tuning: `src/world/constants.rs` (noise layers, feature sensitivities)
- Plate generation heuristics: `src/world/plates.rs` (seed spacing, target sizes, noise)
- Coloring functions: `src/app/coloring.rs`
