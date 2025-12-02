# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
cargo build              # Build debug
cargo build --release    # Build release
cargo run --release      # Run (release recommended for performance)
cargo test               # Run all tests
cargo test voronoi       # Run tests matching "voronoi"
cargo clippy             # Lint
cargo fmt                # Format
```

## Project Overview

Hex3 is a spherical Voronoi-based planet generator with tectonic plate simulation, rendered using wgpu. It generates procedural worlds with realistic terrain based on plate tectonics.

## Architecture

### Module Structure

- **`geometry/`** - Computational geometry for spherical surfaces
  - `voronoi.rs` - Spherical Voronoi diagram via convex hull duality
  - `convex_hull.rs` - 3D convex hull using qhull
  - `lloyd.rs` - Lloyd relaxation for point distribution
  - `plates.rs` - Tectonic plate assignment via flood fill
  - `tectonics.rs` - Euler pole rotation, stress propagation, elevation generation
  - `mesh.rs` - Voronoi to triangle mesh conversion, map projection

- **`render/`** - wgpu rendering infrastructure
  - `context.rs` - GPU device/surface setup
  - `pipeline.rs` - Render pipeline builder
  - `camera.rs` - Orbit camera with controller
  - `buffer.rs`, `uniform.rs`, `vertex.rs` - GPU buffer utilities

- **`shaders/`** - WGSL shaders (sphere.wgsl, edge.wgsl, colored_line.wgsl)

### Key Data Flow

1. Random points on unit sphere → Lloyd relaxation → evenly distributed points
2. Convex hull of points → dual graph → SphericalVoronoi (cells, vertices)
3. Flood fill from seeds → TectonicPlates (cell assignments, Euler poles)
4. Euler pole velocities → boundary stress → propagated stress fields
5. Stress + plate type → elevation via sqrt response curves
6. Elevation → hypsometric coloring → VoronoiMesh → GPU buffers

### Core Types

- `SphericalVoronoi` - Voronoi diagram with generators, vertices, and cells
- `TectonicPlates` - Plate assignments, types, Euler poles, stress/elevation per cell
- `VoronoiMesh` - Triangle mesh with per-vertex colors for rendering
- `GpuContext` - wgpu device, queue, surface configuration

### Tectonic Simulation

Plates rotate around Euler poles. At boundaries, relative velocity determines:
- **Convergent** (positive stress): mountains, volcanic arcs, trenches
- **Divergent** (negative stress): rifts (continental), mid-ocean ridges (oceanic)

Six plate interaction types with distinct multipliers in `tectonics::constants`.

## Controls (Runtime)

- Drag: rotate globe
- Scroll: zoom
- Tab: toggle globe/map view
- 1/2/3: Elevation/Plates/Stress render modes
- R: regenerate world with new seed
- Esc: quit
