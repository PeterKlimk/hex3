# AGENTS.md

This file provides context for LLM coding assistants (ChatGPT Codex, Copilot, etc.) working with this repository.

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
├── geometry/           # Computational geometry
│   ├── voronoi.rs      # Spherical Voronoi via convex hull duality
│   ├── convex_hull.rs  # 3D convex hull (qhull)
│   ├── lloyd.rs        # Lloyd relaxation for even point distribution
│   ├── plates.rs       # TectonicPlates struct, flood fill assignment
│   ├── tectonics.rs    # Euler poles, stress calculation, elevation
│   └── mesh.rs         # Voronoi → triangle mesh, map projection
├── render/             # wgpu rendering
│   ├── context.rs      # GPU device/surface setup
│   ├── pipeline.rs     # Render pipeline builder
│   ├── camera.rs       # Orbit camera with controller
│   └── buffer.rs, uniform.rs, vertex.rs
├── shaders/            # WGSL shaders
└── main.rs             # Application entry, input handling, render loop
```

### Data Flow

1. Random points on unit sphere → Lloyd relaxation → even distribution
2. Convex hull → dual graph → `SphericalVoronoi` (cells, vertices)
3. Flood fill from seeds → `TectonicPlates` (assignments, Euler poles)
4. Euler pole velocities → boundary stress (edge-length weighted)
5. Stress propagation (plate-constrained, exponential decay)
6. Stress + plate type → elevation (sqrt response curves) + fBm noise
7. Elevation → hypsometric coloring → `VoronoiMesh` → GPU buffers
8. Relief view: vertices displaced radially by averaged elevation

### Core Types

- `SphericalVoronoi` - Voronoi diagram: generators, vertices, cells with vertex_indices
- `TectonicPlates` - Plate assignments, types (Continental/Oceanic), Euler poles, stress, elevation
- `VoronoiMesh` - Triangle mesh with per-vertex position/normal/color
- `GpuContext` - wgpu device, queue, surface

## Tectonic Simulation

Plates rotate around Euler poles. At boundaries, relative velocity determines stress:

- **Convergent** (stress > 0): compression → mountains, volcanic arcs
- **Divergent** (stress < 0): tension → rifts (continental), ridges (oceanic)

### 8-Way Plate Interactions

All multipliers in `tectonics::constants`. Sign comes from convergence direction.

**Convergent:**
- `CONV_CONT_CONT` (1.5) - Himalayas-style collision
- `CONV_CONT_OCEAN` (1.2) - Andes-style mountains on continental side
- `CONV_OCEAN_CONT` (0.15) - Subducting oceanic plate (minimal stress)
- `CONV_OCEAN_OCEAN` (0.4) - Island arc volcanism

**Divergent:**
- `DIV_CONT_CONT` (0.6) - Continental rift (East African Rift)
- `DIV_CONT_OCEAN` (0.15) - Passive margin rifting (modest)
- `DIV_OCEAN_CONT` (0.3) - Oceanic side thermal uplift
- `DIV_OCEAN_OCEAN` (0.5) - Mid-ocean ridge

### Elevation Response

- **Continental**: asymmetric - compression → mountains (up to 0.8), tension → rifts (down to -0.15)
- **Oceanic**: both cause uplift, but tension capped below sea level (can't create land from ridges)

## Controls

- **Drag**: rotate globe
- **Scroll**: zoom
- **Tab**: toggle globe/map view
- **1-5**: Elevation/Plates/Stress/Relief/Noise render modes
- **E**: toggle edge visibility
- **R**: regenerate world (new seed)
- **Esc**: quit

## Key Constants (tectonics.rs)

```rust
// Stress & Elevation
STRESS_SCALE = 25.0           // Global stress magnitude
STRESS_DECAY_LENGTH = 0.04    // Propagation distance (radians)
CONTINENTAL_BASE = 0.05       // Base elevation for continents
OCEANIC_BASE = -0.2           // Base elevation for oceans
CONT_MAX_RIFT = 0.2           // Max continental rift depth
OCEAN_TENSION_MAX = 0.12      // Max oceanic uplift from tension (stays underwater)
OCEAN_COMPRESSION_MAX = 0.25  // Max oceanic uplift from compression (can create islands)

// Elevation Noise (fBm)
ELEVATION_NOISE_STRESS = 0.2        // Additional noise per unit |stress|
ELEVATION_NOISE_CONTINENTAL = 0.1   // Base noise for continental plates
ELEVATION_NOISE_OCEANIC = 0.05      // Base noise for oceanic plates
ELEVATION_NOISE_FREQUENCY = 16.0    // fBm base frequency
ELEVATION_NOISE_OCTAVES = 4         // fBm octaves

// Relief View
RELIEF_SCALE = 0.1            // Elevation displacement multiplier
```

## Common Tasks

### Tuning terrain appearance
Modify constants in `src/geometry/tectonics.rs::constants`

### Changing cell count or plate count
Modify `NUM_CELLS` and `NUM_PLATES` in `src/main.rs`

### Adding new render modes
1. Add variant to `RenderMode` enum in main.rs
2. Create color function in plates.rs (like `cell_color_elevation`)
3. Add mesh generation in `generate_world_buffers`
4. Add keyboard shortcut

### Existing render modes
- **Elevation** (1): Hypsometric coloring (ocean blue → land green/brown → mountain white)
- **Plates** (2): Plate colors + boundary stress colors on edges
- **Stress** (3): Red = compression, blue = tension
- **Relief** (4): 3D displaced terrain with hypsometric colors
- **Noise** (5): Green = positive noise, magenta = negative noise
