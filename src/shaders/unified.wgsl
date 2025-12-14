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
    _padding2: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) elevation: f32,
    @location(4) material: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) @interpolate(flat) material: u32,
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

    let displaced_pos = in.position * displacement;

    out.clip_position = uniforms.view_proj * vec4<f32>(displaced_pos, 1.0);
    out.world_pos = displaced_pos;
    out.world_normal = in.normal;  // Normal is still the original sphere normal
    out.color = in.color;
    out.material = in.material;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
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

    return vec4<f32>(final_color, alpha);
}
