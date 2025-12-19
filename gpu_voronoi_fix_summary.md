## Fixes Applied

### 1. Increased Epsilon Tolerances

**File:** `src/geometry/gpu_voronoi/cell_builder.rs` (lines 10-16)

```rust
// Before: too small for f32 precision
pub(crate) const EPS_PLANE_CONTAINS: f32 = 1e-10;
pub(crate) const EPS_PLANE_CLIP: f32 = 1e-9;

// After: appropriate for f32
pub(crate) const EPS_PLANE_CONTAINS: f32 = 1e-6;
pub(crate) const EPS_PLANE_CLIP: f32 = 1e-6;
```

### 2. Generator Containment Check for Seed Triangles

**File:** `src/geometry/gpu_voronoi/cell_builder.rs` (lines 238-259 in `seed_from_triplet`)

Before accepting a seed triplet, verify the generator is geometrically inside the spherical triangle. A seed triangle that doesn't contain the generator can be clipped to nothing by later planes.

```rust
// Check if generator is inside the spherical triangle formed by v0, v1, v2
let edge_plane_01 = v0.cross(v1);
let edge_plane_12 = v1.cross(v2);
let edge_plane_20 = v2.cross(v0);

let inside_01 = edge_plane_01.dot(g);
let inside_12 = edge_plane_12.dot(g);
let inside_20 = edge_plane_20.dot(g);

// All should have the same sign for generator to be inside
let consistent_winding = (inside_01 > 0.0 && inside_12 > 0.0 && inside_20 > 0.0)
    || (inside_01 < 0.0 && inside_12 < 0.0 && inside_20 < 0.0);

if !consistent_winding {
    return false;
}
```
