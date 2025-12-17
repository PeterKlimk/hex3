# AGENTS.md

This file provides context for LLM coding assistants (ChatGPT Codex, Copilot, etc.) working with this repository.

## Development Environment

Development is done in WSL2, but the application must be run on Windows. Compute shaders (used for the particle system) do not work properly under WSL2's GPU passthrough.

- Use `cargo build` in WSL2 to verify code compiles
- Build and run on Windows for actual execution (`cargo run --release` from Windows terminal)

## Build & Run

```bash
cargo build              # Build debug
cargo build --release    # Build release
cargo run --release      # Run (release recommended for performance)
cargo test               # Run all tests
cargo clippy             # Lint
cargo fmt                # Format
```

## Project Overview

Hex3 is a spherical Voronoi-based planet generator with tectonic plate simulation, rendered using wgpu. It generates procedural worlds with realistic terrain based on plate tectonics.

## Architecture

### Module Structure

```
src/
├── lib.rs              # Library crate entry (re-exports geometry/render/world)
├── geometry/           # Computational geometry
│   ├── voronoi.rs      # Spherical Voronoi via convex hull duality
│   ├── convex_hull.rs  # 3D convex hull (qhull)
│   ├── lloyd.rs        # Lloyd relaxation for even point distribution
│   ├── sphere.rs       # Random points on unit sphere
│   ├── mesh.rs         # Voronoi → triangle mesh, map projection
│   └── gpu_voronoi.rs  # GPU-style Voronoi via half-space clipping
├── world/              # World generation and simulation
│   ├── tessellation.rs # Voronoi cells + adjacency + cell areas
│   ├── plates.rs       # Plate assignment via flood fill
│   ├── dynamics.rs     # Euler poles, plate velocities
│   ├── boundary.rs     # Plate boundary classification
│   ├── features.rs     # Tectonic features (trench, arc, ridge, collision)
│   ├── elevation.rs    # Elevation from features + noise
│   ├── atmosphere.rs   # Atmosphere (temperature, pressure, wind)
│   ├── hydrology.rs    # Rivers, drainage basins, lakes
│   ├── constants.rs    # Tunable parameters
│   └── gen.rs          # World generation orchestration
├── app/                # Application layer
│   ├── state.rs        # App state and rendering
│   ├── view.rs         # Render modes, view modes
│   ├── coloring.rs     # Cell coloring functions
│   ├── world.rs        # GPU buffer generation
│   ├── visualization.rs# Debug visualization
│   └── export.rs       # World data export to JSON
├── render/             # wgpu rendering
│   ├── context.rs      # GPU device/surface setup
│   ├── pipeline.rs     # Render pipeline builder
│   ├── renderer.rs     # Main renderer
│   ├── camera.rs       # Orbit camera with controller
│   ├── buffer.rs, uniform.rs, vertex.rs
│   └── particles.rs    # GPU wind particle system (compute shader)
├── shaders/            # WGSL shaders
│   ├── unified.wgsl        # Main terrain rendering
│   ├── wind_particles.wgsl # Compute shader for particle physics
│   └── particle_render.wgsl# Particle trail rendering
├── util.rs             # Utility types (timing, etc.)
└── main.rs             # Application entry, event loop
```

### Data Flow

1. Random points on unit sphere → Lloyd relaxation → even distribution
2. Convex hull → dual graph → `SphericalVoronoi` (cells, vertices)
3. Spaced seeds + varied target sizes → weighted flood fill → `Plates` (assignments)
4. Euler pole velocities → `Dynamics` → boundary classification
5. Boundary analysis → `FeatureFields` (trench, arc, ridge, collision, activity)
6. Features + plate type → elevation (decay functions + multi-layer fBm noise)
7. Elevation + latitude → `Atmosphere` (temperature, pressure, wind fields)
8. Elevation + atmosphere → hydrological simulation → rivers, lakes, drainage basins
9. World state → hypsometric coloring → `VoronoiMesh` → GPU buffers
10. Wind field → `WindParticleSystem` → animated wind visualization
11. Relief view: vertices displaced radially by averaged elevation

### Core Types

- `SphericalVoronoi` - Voronoi diagram: generators, vertices, cells with vertex_indices
- `Tessellation` - Voronoi + adjacency graph + cell area computation
- `Plates` - Cell-to-plate assignments
- `Dynamics` - Plate types (Continental/Oceanic), Euler poles, velocities
- `FeatureFields` - Per-cell tectonic feature magnitudes
- `Atmosphere` - Temperature, pressure, wind vectors, uplift per cell
- `Hydrology` - River network, drainage basins, lake simulation
- `World` - Complete world state (tessellation, plates, dynamics, features, elevation, atmosphere, hydrology)
- `VoronoiMesh` - Triangle mesh with per-vertex position/normal/color
- `GpuContext` - wgpu device, queue, surface
- `WindParticleSystem` - GPU compute-based particle system for wind visualization

## Tectonic Simulation

Plates rotate around Euler poles. At boundaries, relative velocity determines feature type:

- **Convergent**: subduction (trenches + volcanic arcs) or collision (mountain ranges)
- **Divergent**: mid-ocean ridges (oceanic) or rifts (continental)
- **Transform**: lateral motion (no elevation features)

### 8-Way Plate Interactions

All multipliers in `world/constants.rs`. Sign comes from convergence direction.

**Convergent:**
- `CONV_CONT_CONT` (1.5) - Himalayas-style collision
- `CONV_CONT_OCEAN` (1.2) - Andes-style mountains on continental side
- `CONV_OCEAN_CONT` (0.1) - Subducting oceanic plate (minimal uplift)
- `CONV_OCEAN_OCEAN` (0.8) - Island arc volcanism

**Divergent:**
- `DIV_CONT_CONT` (0.6) - Continental rift (East African Rift)
- `DIV_CONT_OCEAN` (0.1) - Passive margin rifting (modest)
- `DIV_OCEAN_CONT` (0.3) - Oceanic side thermal uplift
- `DIV_OCEAN_OCEAN` (0.5) - Mid-ocean ridge

### Feature Fields

- **Trench**: oceanic subducting side, exponential decay from boundary
- **Arc**: overriding plate (continental or oceanic), Gaussian band profile
- **Ridge**: mid-ocean divergent boundaries, exponential decay
- **Collision**: continental-continental convergent, Gaussian band profile
- **Activity**: general tectonic activity, screened diffusion from boundaries

### Elevation Response

- **Continental**: asymmetric - compression → mountains (up to 0.8), tension → rifts (down to -0.2)
- **Oceanic**: thermal subsidence from ridge distance + feature-driven uplift

### Resolution Independence

Boundary forcing is normalized by cell area for consistent results across resolutions. Smoothing and diffusion iterations scale adaptively with mean neighbor distance.

## Controls

- **Drag**: rotate globe
- **Scroll**: zoom
- **Tab**: toggle globe/map view
- **1-8**: Render modes:
  - 1: Relief (3D terrain + lakes + wind particles)
  - 2: Terrain (flat terrain + lakes)
  - 3: Elevation (raw elevation)
  - 4: Plates (boundaries + velocities)
  - 5: Noise (fBm layers, press again to cycle)
  - 6: Hydrology (flow accumulation)
  - 7: Features (tectonic fields, press again to cycle)
  - 8: Climate (atmosphere, press again to cycle: Temperature/Wind/Uplift) - Stage 2+
- **E**: toggle edge visibility
- **V**: cycle river visibility (Off/Major/All) - Stage 3+
- **H**: toggle hemisphere lighting
- **R**: regenerate world (new seed)
- **Space**: advance stage (1=Lithosphere → 2=Atmosphere → 3=Hydrosphere)
- **Up/Down**: adjust climate ratio (wetter/drier) - Stage 3
- **Esc**: quit

Notes:
- Plates mode (globe view) overlays plate velocity arrows and Euler pole markers.
- Rivers follow terrain elevation in Relief mode, flat in other modes.

## Key Constants (world/constants.rs)

```rust
// Plate generation
CONTINENTAL_FRACTION = 0.30   // Target continental coverage
SEED_SPACING_FRACTION = 0.5   // Seed minimum spacing vs ideal
TARGET_SIZE_SIGMA = 0.4       // Plate size variance (log-normal)
TARGET_SIZE_MAX_RATIO = 4.0   // Largest/smallest plate ratio cap
NOISE_WEIGHT = 1.0            // Boundary irregularity vs distance
NEIGHBOR_BONUS = 0.1          // Encourages compact plate shapes

// Base elevations
CONTINENTAL_BASE = 0.05       // Base elevation for continents
RIDGE_CREST_DEPTH = -0.25     // Mid-ocean ridge crest depth
ABYSSAL_DEPTH = -0.45         // Deep ocean floor

// Feature parameters
FEATURE_FORCE_SCALE = 35.0    // Global feature magnitude scale
TRENCH_DECAY = 0.020          // Trench decay distance (radians)
ARC_CONT_PEAK_DIST = 0.05     // Continental arc peak distance
ARC_OCEAN_PEAK_DIST = 0.04    // Oceanic arc peak distance
RIDGE_DECAY = 0.015           // Ridge decay distance
COLLISION_PEAK_DIST = 0.035   // Collision peak distance

// Noise layers (Macro, Hills, Ridge, Micro)
MACRO_AMPLITUDE = 0.12        // Large-scale terrain variation
HILLS_AMPLITUDE = 0.07        // Medium-scale hills
RIDGE_AMPLITUDE = 0.14        // Ridge/mountain detail
MICRO_AMPLITUDE = 0.02        // Fine detail

// Relief View
RELIEF_SCALE = 0.2            // Elevation displacement multiplier
```

## Common Tasks

### Tuning terrain appearance
Modify constants in `src/world/constants.rs`

### Changing cell count or plate count
Modify `NUM_CELLS` and `NUM_PLATES` in `src/main.rs` (and `LLOYD_ITERATIONS` if needed)

### Adding new render modes
1. Add variant to `RenderMode` enum in `src/app/view.rs`
2. Create color function in `src/app/coloring.rs`
3. Add mesh generation in `src/app/world.rs`
4. Add keyboard shortcut in `src/app/mod.rs`

### Existing render modes
- **Relief** (1): 3D displaced terrain with hypsometric colors + lakes + wind particles
- **Terrain** (2): Flat terrain with hypsometric colors + lakes
- **Elevation** (3): Raw elevation (ocean blue → land green/brown → mountain white)
- **Plates** (4): Plate colors + boundary convergence + velocity arrows + Euler poles
- **Noise** (5): Noise layer visualization (cycle through Macro/Hills/Ridge/Micro/Combined)
- **Hydrology** (6): Flow accumulation coloring
- **Features** (7): Tectonic feature fields (cycle through Trench/Arc/Ridge/Collision/Activity)
- **Climate** (8): Atmosphere visualization (cycle through Temperature/Wind/Uplift) - Stage 2+
