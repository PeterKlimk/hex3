# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Project Overview

Hex3 is a spherical Voronoi-based planet generator with tectonic plate simulation, rendered using wgpu. It generates procedural worlds with realistic terrain based on plate tectonics.

## Architecture

### Module Structure

- **`src/lib.rs`** - Library crate entry (re-exports `geometry` and `render`)
- **`src/geometry/`** - Computational geometry for spherical surfaces
  - `voronoi.rs` - Spherical Voronoi diagram via convex hull duality
  - `convex_hull.rs` - 3D convex hull using qhull
  - `lloyd.rs` - Lloyd relaxation for point distribution
  - `plates.rs` - Tectonic plate assignment via flood fill
  - `sphere.rs` - Uniform random points on a unit sphere
  - `tectonics.rs` - Euler pole rotation, stress propagation, elevation generation
  - `mesh.rs` - Voronoi to triangle mesh conversion, map projection

- **`src/render/`** - wgpu rendering infrastructure
  - `context.rs` - GPU device/surface setup
  - `pipeline.rs` - Render pipeline builder
  - `camera.rs` - Orbit camera with controller
  - `buffer.rs`, `uniform.rs`, `vertex.rs` - GPU buffer utilities

- **`src/shaders/`** - WGSL shaders (sphere.wgsl, edge.wgsl, colored_line.wgsl)

### Key Data Flow

1. Random points on unit sphere → Lloyd relaxation → evenly distributed points
2. Convex hull of points → dual graph → SphericalVoronoi (cells, vertices)
3. Spaced seeds + varied target sizes → weighted flood fill → TectonicPlates (cell assignments, Euler poles)
4. Euler pole velocities → boundary stress → propagated stress fields
5. Stress + plate type → elevation via sqrt response curves + fBm noise
6. Elevation → hypsometric coloring → VoronoiMesh → GPU buffers
7. Relief view: vertices displaced radially by averaged elevation

### Core Types

- `SphericalVoronoi` - Voronoi diagram with generators, vertices, and cells
- `TectonicPlates` - Plate assignments, types, Euler poles, stress/elevation per cell
- `VoronoiMesh` - Triangle mesh with per-vertex colors for rendering
- `GpuContext` - wgpu device, queue, surface configuration

### Tectonic Simulation

Plates rotate around Euler poles. At boundaries, relative velocity determines stress:
- **Convergent** (positive stress): mountains, volcanic arcs
- **Divergent** (negative stress): rifts (continental), mid-ocean ridges (oceanic)

Eight plate interaction multipliers (4 convergent + 4 divergent) in `tectonics::constants`:
- Convergent: `CONV_CONT_CONT`, `CONV_CONT_OCEAN`, `CONV_OCEAN_CONT`, `CONV_OCEAN_OCEAN`
- Divergent: `DIV_CONT_CONT`, `DIV_CONT_OCEAN`, `DIV_OCEAN_CONT`, `DIV_OCEAN_OCEAN`

Stress is weighted by edge arc length for density-independent results (`STRESS_SCALE` controls magnitude). Elevation response differs by plate type:
- **Continental**: asymmetric (compression → mountains, tension → rifts capped above ocean floor)
- **Oceanic**: both cause uplift, but compression can create islands while tension is capped underwater

## Controls (Runtime)

- Drag: rotate globe
- Scroll: zoom
- Tab: toggle globe/map view
- 1-5: Elevation/Plates/Stress/Relief/Noise render modes
- E: toggle edge visibility
- R: regenerate world with new seed
- Esc: quit

Notes:
- Plates mode (globe view) overlays plate velocity arrows and Euler pole markers.

## Common Edit Points

- World resolution: `src/main.rs` (`NUM_CELLS`, `LLOYD_ITERATIONS`, `NUM_PLATES`)
- Tectonics + terrain tuning: `src/geometry/tectonics.rs::constants`
- Plate generation heuristics: `src/geometry/plates.rs` (seed spacing, target sizes, noise)
