// Unified material-aware shader
// All visible geometry (terrain, water) uses this single shader.
// Material attribute controls lighting behavior.
//
// Displacement: Vertices are on unit sphere. Shader displaces by:
//   pos * (1 + (elevation + micro_noise) * relief_scale)
// This ensures terrain and rivers use identical displacement.

// Material constants
const MATERIAL_LAND: u32 = 0u;
const MATERIAL_OCEAN: u32 = 1u;
const MATERIAL_LAKE: u32 = 2u;
const MATERIAL_RIVER: u32 = 3u;
const MATERIAL_ICE_SNOW: u32 = 4u;

// Lighting constants
const RIVER_ALPHA: f32 = 0.85;

// Hemisphere lighting - warm sun / cool sky for natural outdoor look
const SUN_COLOR: vec3<f32> = vec3<f32>(1.0, 0.92, 0.75);  // Warm golden sunlight
const SKY_COLOR: vec3<f32> = vec3<f32>(0.35, 0.5, 0.75);  // Cool sky blue ambient
const GROUND_COLOR: vec3<f32> = vec3<f32>(0.25, 0.2, 0.15); // Warm ground bounce

// Relief displacement
const MICRO_AMPLITUDE: f32 = 0.0;  // Disabled - micro noise affects color only (CPU-side)
const MICRO_FREQUENCY: f32 = 8.0;   // Micro noise frequency
const RIVER_Z_OFFSET: f32 = 0.002;  // Small offset to prevent z-fighting

struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding1: f32,
    light_dir: vec3<f32>,
    relief_scale: f32, // 0.0 = flat, >0 = 3D terrain displacement
    hemisphere_lighting: f32, // 1.0 = hemisphere, 0.0 = simple diffuse
    map_mode: f32, // 0.0 = globe view, 1.0 = equirectangular map view
    slope_shading: f32, // 1.0 = shade from displaced face normal (hillshade)
    rivers_enabled: f32, // 1.0 = blend the baked river texture into the surface
    river_major_only: f32, // 1.0 = major rivers only; 0.0 = all rivers
    river_width_scale: f32, // cartographic screen-space stroke multiplier
}

// River SDF: R = distance-to-river over [0, RIVER_SDF_RANGE_PX] px (must match the CPU bake),
// G = nearest river's flow factor, B = nearest river is major. Rivers are reconstructed THIN
// and crisp in-shader (width is flow-tapered, never exaggerated).
const RIVER_SDF_RANGE_PX: f32 = 6.0;
const RIVER_BASE_WIDTH_PX: f32 = 0.7;  // thin tributary half-width (px)
const RIVER_FLOW_WIDTH_PX: f32 = 1.4;  // extra half-width for max-flow trunks (px)
const RIVER_DEEP_COLOR: vec3<f32> = vec3<f32>(0.09, 0.20, 0.38);

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Per-world baked river network (equirectangular RGBA; alpha = river coverage). Rivers are
// drawn AS SURFACE SHADING here (perfectly draped) instead of floating quad ribbons.
@group(1) @binding(0) var river_tex: texture_2d<f32>;
@group(1) @binding(1) var river_samp: sampler;

// Constants for map projection
const PI: f32 = 3.14159265359;
const HALF_PI: f32 = 1.57079632679;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) elevation: f32,
    @location(4) material: u32,
    @location(5) wrap_offset: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) @interpolate(flat) material: u32,
    @location(4) river_uv: vec2<f32>,
}

// Simple 3D hash for procedural noise (fast, deterministic)
fn hash3(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Value noise - smooth interpolated noise
fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    // Smoothstep interpolation
    let u = f * f * (3.0 - 2.0 * f);

    // Sample 8 corners of cube
    let n000 = hash3(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash3(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash3(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash3(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash3(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash3(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash3(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash3(i + vec3<f32>(1.0, 1.0, 1.0));

    // Trilinear interpolation
    let n00 = mix(n000, n100, u.x);
    let n01 = mix(n001, n101, u.x);
    let n10 = mix(n010, n110, u.x);
    let n11 = mix(n011, n111, u.x);
    let n0 = mix(n00, n10, u.y);
    let n1 = mix(n01, n11, u.y);
    return mix(n0, n1, u.z);
}

// Micro noise for terrain texture (centered around 0)
fn micro_noise(pos: vec3<f32>) -> f32 {
    let p = pos * MICRO_FREQUENCY;
    return (noise3(p) - 0.5) * 2.0 * MICRO_AMPLITUDE;
}

// Project sphere position to equirectangular map coordinates
fn sphere_to_map(pos: vec3<f32>, wrap_offset: f32) -> vec3<f32> {
    let lon = atan2(pos.z, pos.x); // -PI to PI
    let lat = asin(clamp(pos.y, -1.0, 1.0)); // -PI/2 to PI/2

    let x = lon / PI + wrap_offset; // -1 to 1, with wrap adjustment
    let y = lat / HALF_PI;          // -1 to 1

    return vec3<f32>(x, y, 0.0);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Compute micro noise from world position
    let micro = micro_noise(in.position);

    // Apply displacement: pos * (1 + (elevation + micro) * relief_scale)
    // Only apply micro noise to land, not water (which should be flat)
    var total_elevation = in.elevation;
    if (in.material == MATERIAL_LAND || in.material == MATERIAL_RIVER) {
        total_elevation += micro;
    }

    var displacement = 1.0 + total_elevation * uniforms.relief_scale;

    // Rivers get extra offset to prevent z-fighting with terrain
    if (in.material == MATERIAL_RIVER && uniforms.relief_scale > 0.0) {
        displacement += RIVER_Z_OFFSET;
    }

    var final_pos: vec3<f32>;
    if (uniforms.map_mode > 0.5) {
        // Map view: project to 2D equirectangular
        final_pos = sphere_to_map(in.position, in.wrap_offset);
    } else {
        // Globe view: apply 3D displacement
        final_pos = in.position * displacement;
    }

    out.clip_position = uniforms.view_proj * vec4<f32>(final_pos, 1.0);
    out.world_pos = final_pos;
    out.world_normal = in.normal;  // Normal is still the original sphere normal
    out.color = in.color;
    out.material = in.material;

    // River-texture UV from the base SPHERE position (works in globe + map mode); must
    // match the CPU bake convention (lon=atan2(z,x), lat=asin(y)).
    let r_lon = atan2(in.position.z, in.position.x);
    let r_lat = asin(clamp(in.position.y, -1.0, 1.0));
    out.river_uv = vec2<f32>(r_lon / (2.0 * PI) + 0.5, 0.5 - r_lat / PI);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Shade from the actual displaced surface (slope hillshade) when requested: the
    // vertex normal is the smooth SPHERE normal (relief is displaced in the vertex
    // shader but the normal isn't recomputed), so without this, slopes catch no light
    // and tall/snow-capped peaks read as flat white. The screen-space derivatives of
    // world_pos give the displaced facet normal; orient it outward via the sphere normal.
    var N = normalize(in.world_normal);
    if (uniforms.slope_shading > 0.5) {
        let face_n = normalize(cross(dpdx(in.world_pos), dpdy(in.world_pos)));
        N = face_n * sign(dot(face_n, N));
    }
    let L = uniforms.light_dir;
    let V = normalize(uniforms.camera_pos - in.world_pos);

    let NdotL = dot(N, L);

    var lighting: vec3<f32>;
    if (uniforms.hemisphere_lighting > 0.5) {
        // Three-point hemisphere lighting:
        // - Direct sunlight (warm golden)
        // - Sky ambient from above (cool blue)
        // - Ground bounce from below (warm brown)

        // Direct sun contribution with soft falloff
        let sun_intensity = max(NdotL, 0.0);
        let direct = sun_intensity * SUN_COLOR * 0.7;

        // Hemisphere ambient: blend sky (up) and ground (down) based on normal.y
        // normal.y > 0 = facing up = more sky, normal.y < 0 = facing down = more ground
        let up_factor = N.y * 0.5 + 0.5; // 0 = facing down, 1 = facing up
        let ambient_color = mix(GROUND_COLOR, SKY_COLOR, up_factor);

        // Shadow softening: even back-facing surfaces get some wrap lighting
        let wrap = max(NdotL * 0.5 + 0.5, 0.0); // Wrapped diffuse for softer shadows
        let ambient_intensity = 0.3 + 0.15 * wrap;
        let ambient = ambient_color * ambient_intensity;

        lighting = direct + ambient;
    } else {
        // Simple diffuse lighting (original)
        lighting = vec3<f32>(0.25 + max(NdotL, 0.0) * 0.75);
    }

    var final_color = in.color * lighting;
    var alpha = 1.0;

    // Material-specific adjustments
    if (in.material == MATERIAL_RIVER) {
        alpha = RIVER_ALPHA;
    }

    if (in.material == MATERIAL_ICE_SNOW) {
        // Ice gets a slightly stronger glint
        let H = normalize(L + V);
        let glint = pow(max(dot(N, H), 0.0), 128.0);
        final_color += vec3<f32>(glint * 0.2);
    }

    // Draped rivers: reconstruct a thin, crisp river from the distance field and shade it as
    // water. rivers_enabled is a uniform so this branch is uniform control flow.
    if (uniforms.rivers_enabled > 0.5) {
        let s = textureSample(river_tex, river_samp, in.river_uv);
        // Density mode: in "major only" hide rivers whose nearest river isn't major.
        let visible = uniforms.river_major_only < 0.5 || s.b > 0.5;
        let dist_px = s.r * RIVER_SDF_RANGE_PX; // 0 = on centerline
        let flow = s.g;
        // Convert the desired SCREEN-pixel width into SDF-texture pixels using
        // the local derivative. The old code compared a fixed 0.7..2.1 texture
        // pixels directly with dist_px, making rivers physically 7..21 km wide
        // at this 8192x4096 bake and increasingly fat when zoomed in.
        let texels_per_screen_px = max(fwidth(dist_px), 1.0 / 255.0);
        let width_screen_px = (RIVER_BASE_WIDTH_PX + flow * RIVER_FLOW_WIDTH_PX)
            * uniforms.river_width_scale;
        let width = width_screen_px * texels_per_screen_px;
        let aa = 0.75 * texels_per_screen_px;
        let river_a = select(0.0, 1.0 - smoothstep(width - aa, width + aa, dist_px), visible);
        if (river_a > 0.001) {
            // Water look: sky-reflective (fresnel) deep blue + a sun glint, distinct from
            // the flat ocean, partially lit by the terrain shading.
            let fres = pow(1.0 - max(dot(N, V), 0.0), 3.0);
            var water = mix(RIVER_DEEP_COLOR, SKY_COLOR, fres * 0.5);
            let Hr = normalize(L + V);
            water += vec3<f32>(pow(max(dot(N, Hr), 0.0), 64.0) * 0.3);
            water *= 0.55 + 0.45 * max(NdotL, 0.0);
            final_color = mix(final_color, water, river_a);
        }
    }

    return vec4<f32>(final_color, alpha);
}
