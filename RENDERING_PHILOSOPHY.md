# Rendering Philosophy

This document outlines the rendering architecture for hex3's planet visualization.

## Goals

1. **Coherent physical object** - The planet should read as a 3D object with distinct materials, not a painted texture
2. **Unified lighting** - All features respond consistently to the same light source
3. **Material distinction** - Water, land, rivers, ice look and behave differently
4. **Clean architecture** - Easy to tune, extend, and maintain

## Unified Mesh Architecture

### Core Concept

All visible geometry (terrain, water, rivers) is rendered through a **single unified mesh** with a **single shader**. Different materials are distinguished by a material attribute per vertex, not by separate render passes.

```
Vertex Buffer: [terrain polygons...][river triangles...]
Draw: Single draw call
Shader: Material-aware lighting
```

### Vertex Format

```rust
struct UnifiedVertex {
    position: Vec3,     // World position (rivers offset slightly along normal)
    normal: Vec3,       // Surface normal for lighting
    color: Vec3,        // Base material color
    material: u8,       // Material type (see below)
}
```

### Material Types

| ID | Material | Diffuse | Specular | Fresnel | Alpha | Notes |
|----|----------|---------|----------|---------|-------|-------|
| 0  | Land     | Yes     | No       | No      | 1.0   | Matte terrain |
| 1  | Ocean    | Yes     | Yes      | Yes     | 1.0   | Deep water, reflective |
| 2  | Lake     | Yes     | Subtle   | Yes     | 1.0   | Shallower than ocean |
| 3  | River    | Yes     | No       | No      | 0.85  | Slight transparency |
| 4  | Ice/Snow | Yes     | Sharp    | No      | 1.0   | Future: high elevation + latitude |

## Lighting Model

Single directional light (sun) with consistent handling across all materials:

```
final_color = base_color * (ambient + diffuse) + specular + rim
```

### Components

- **Ambient** (~0.2): Constant, simulates indirect sky light
- **Diffuse**: `max(dot(N, L), 0)` - Standard Lambertian
- **Specular**: `pow(max(dot(R, V), 0), shininess)` - Water materials only
- **Rim** (optional): Subtle edge brightening for atmospheric effect

### Shader Pseudocode

```wgsl
fn fs_main(in: VertexInput) -> vec4<f32> {
    let N = normalize(in.normal);
    let L = uniforms.light_dir;
    let V = normalize(uniforms.camera_pos - in.world_pos);

    // Universal lighting
    let ambient = 0.2;
    let diffuse = max(dot(N, L), 0.0);
    var lighting = ambient + diffuse * (1.0 - ambient);
    var final_color = in.base_color * lighting;
    var alpha = 1.0;

    // Material-specific adjustments
    if (in.material == OCEAN || in.material == LAKE) {
        let R = reflect(-L, N);
        let spec = pow(max(dot(R, V), 0.0), 32.0);
        final_color += vec3(spec) * 0.3;

        // Fresnel - brighter at grazing angles
        let fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
        final_color = mix(final_color, vec3(0.8, 0.9, 1.0), fresnel * 0.2);
    }

    if (in.material == RIVER) {
        alpha = 0.85;
    }

    // Optional: rim lighting for atmosphere
    // let rim = pow(1.0 - max(dot(N, V), 0.0), 4.0);
    // final_color += rim * 0.1;

    return vec4(final_color, alpha);
}
```

## River Rendering

Rivers are rendered as **triangle strips** integrated into the unified mesh, not as separate line geometry.

### Why Not Lines?

- Lines don't respond to lighting
- Fixed width regardless of flow
- Feel like overlays, not part of terrain
- Z-fighting requires hacks

### Triangle Strip Generation

For each river segment (cell to downstream cell):

1. Get endpoints and terrain normals
2. Compute width based on flow
3. Generate quad perpendicular to flow direction
4. Offset vertices slightly along normal to prevent z-fighting

```rust
fn generate_river_segment(
    p1: Vec3, p2: Vec3,           // Segment endpoints
    n1: Vec3, n2: Vec3,           // Terrain normals
    flow: f32,                     // Flow accumulation
) -> [UnifiedVertex; 4] {
    let width = flow_to_width(flow);
    let tangent = (p2 - p1).normalize();

    // Perpendicular to flow, in tangent plane
    let bitangent1 = tangent.cross(n1).normalize();
    let bitangent2 = tangent.cross(n2).normalize();

    const Z_OFFSET: f32 = 0.002;

    // Four corners of the quad
    [
        vertex(p1 + bitangent1 * width/2 + n1 * Z_OFFSET, n1),
        vertex(p1 - bitangent1 * width/2 + n1 * Z_OFFSET, n1),
        vertex(p2 + bitangent2 * width/2 + n2 * Z_OFFSET, n2),
        vertex(p2 - bitangent2 * width/2 + n2 * Z_OFFSET, n2),
    ]
}
```

### River Width

Width varies with flow magnitude using square root scaling (gentler curve):

```rust
fn flow_to_width(flow: f32, max_flow: f32) -> f32 {
    let t = (flow / max_flow).sqrt();
    let min_width = 0.002;  // Smallest tributaries
    let max_width = 0.012;  // Major rivers
    min_width + t * (max_width - min_width)
}
```

### Future Considerations

- **Smooth curves**: River paths could be smoothed with Catmull-Rom splines
- **Animated flow**: Scrolling texture or vertex animation for water movement
- **Foam/rapids**: Different appearance in high-gradient sections

## Water Rendering

### Depth-Based Coloring

Water color varies with depth (already computed in hydrology):

- Shallow: Lighter, shows terrain tint through
- Deep: Darker, more saturated blue

### Specular Highlights

Ocean and lakes get specular reflection from the sun. Concentrated highlight gives sense of wet surface.

### Fresnel Effect

Water is more reflective at grazing angles. Horizon of ocean appears brighter/more sky-colored.

## Atmospheric Effects (Optional)

### Rim Lighting

Subtle brightening at planet edges facing away from camera. Simulates light scattering through atmosphere.

### Horizon Haze

Distant terrain slightly desaturated and blue-shifted. May be overkill for globe view but adds depth.

## Implementation Phases

### Phase 1: Unified Shader
- Consolidate terrain rendering to single shader
- Add material attribute to vertices
- Implement diffuse lighting for all materials
- Add specular for water

### Phase 2: River Geometry
- Replace line-based rivers with triangle strips
- Variable width based on flow
- Integrate into unified mesh

### Phase 3: Water Polish
- Fresnel effect for ocean/lakes
- Depth-based water coloring refinements
- Specular tuning

### Phase 4: Atmosphere & Details
- Rim lighting
- Snow/ice on peaks (elevation + latitude)
- Coastline emphasis

## File Structure

```
src/render/
    vertex.rs       - UnifiedVertex definition
    pipeline.rs     - Single unified pipeline

src/shaders/
    unified.wgsl    - Material-aware shader

src/app/
    world.rs        - Mesh generation (terrain + rivers)
    coloring.rs     - Color/material assignment
```

## Constants to Tune

```rust
// Lighting
const AMBIENT: f32 = 0.2;
const SPECULAR_POWER: f32 = 32.0;
const SPECULAR_INTENSITY: f32 = 0.3;
const FRESNEL_POWER: f32 = 3.0;
const FRESNEL_INTENSITY: f32 = 0.2;

// Rivers
const RIVER_MIN_WIDTH: f32 = 0.002;
const RIVER_MAX_WIDTH: f32 = 0.012;
const RIVER_Z_OFFSET: f32 = 0.002;
const RIVER_ALPHA: f32 = 0.85;

// Optional atmosphere
const RIM_POWER: f32 = 4.0;
const RIM_INTENSITY: f32 = 0.1;
```
