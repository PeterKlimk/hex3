# Rendering Improvement Ideas

Visual enhancements to explore for the planet renderer.

## High Impact

### 1. Terrain-Based Normals
**Status:** ✅ Implemented

Lighting now uses terrain normals computed from elevation gradient at each vertex. Slopes facing the sun are brighter, slopes facing away are darker.

**Implementation:** `src/geometry/mesh.rs` - `compute_terrain_normals()` function estimates gradient from surrounding cell elevations and perturbs sphere normal.

### 2. Simple Ambient Occlusion
**Status:** ✅ Implemented

Valleys and depressions now appear darker due to less ambient light reaching them.

**Implementation:** `src/app/coloring.rs` - `compute_ambient_occlusion()` compares cell elevation to neighbor average, darkening valleys by up to 40%.

### 3. Slope-Based Coloring
**Status:** ✅ Implemented

Steep slopes now appear rocky/gray instead of vegetated green.

**Implementation:** `src/app/coloring.rs` - `compute_slope()` finds steepest neighbor gradient, blends toward rocky color above 30% slope threshold.

## Medium Impact

### 4. Snow Caps
**Status:** ✅ Implemented

High elevations have snow, with latitude modulation (lower snow line at poles).

**Implementation:** `src/app/coloring.rs` - `snow_line()` computes elevation threshold based on latitude, `apply_snow_cap()` blends to blue-white snow color.

### 5. Softer Lighting (Hemisphere)
**Status:** ✅ Implemented

Shadows are now blue-tinted rather than harsh black, using hemisphere lighting model.

**Implementation:** `src/shaders/unified.wgsl` - Ambient light blends between sky color (cool blue) and sun color (warm white) based on hemisphere factor.

## Attempted / Rejected

### Water Fresnel
**Status:** Rejected

Tried fresnel effect on water to make edges more reflective.

**Problem:** On a sphere, fresnel makes the entire rim white regardless of water position. Doesn't work well with globe geometry.

### Water Specular Highlights
**Status:** Rejected

Tried specular highlights on water surfaces.

**Problem:** Looked like plastic with a harsh white blob. Didn't improve realism.

## Implementation Notes

All five features have been implemented. The implementation order was:
1. Terrain normals (foundation for slope-based effects)
2. Snow caps (latitude-modulated)
3. Ambient occlusion (per-cell valley darkening)
4. Slope coloring (rocky steep areas)
5. Hemisphere lighting (softer shadows)
