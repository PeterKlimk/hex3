// Particle trail rendering shader
//
// Renders particles as line segments from trail_end to position.
// Trail length is based on wind magnitude, not movement distance.
// Uses instancing: each instance is one particle, 2 vertices per instance.
//
// Surface wind particles are displaced by sampled elevation.
// Upper wind particles float at a fixed height above the terrain.

struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad0: f32,
    light_dir: vec3<f32>,
    _pad1: f32,
    flags: u32,          // bit 0: relief, bit 1: hemisphere_lighting, bit 2: map_mode
    relief_scale: f32,
    _pad2: vec2<f32>,
}

// Particle data from compute shader (must match exactly)
struct Particle {
    position: vec3<f32>,
    cell: u32,
    trail_end: vec3<f32>,
    age: f32,
}

struct ParticleUniforms {
    max_age: f32,
    relief_scale: f32,      // Elevation displacement scale (matches terrain)
    upper_wind_height: f32, // Fixed height for upper wind particles
    is_surface_wind: u32,   // 1 = surface wind (sample elevation), 0 = upper wind
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> particles: array<Particle>;
@group(1) @binding(1) var<uniform> particle_uniforms: ParticleUniforms;
@group(1) @binding(2) var elevation_map: texture_2d_array<f32>;
@group(1) @binding(3) var elevation_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

// Sample elevation at a sphere position using explicit face/UV mapping.
fn sample_elevation(pos: vec3<f32>) -> f32 {
    let cube = cubemap_dir_to_face_uv01(pos);
    return textureSampleLevel(elevation_map, elevation_sampler, cube.yz, i32(cube.x), 0.0).r;
}

// Displace a sphere position outward based on elevation.
fn displace_by_elevation(pos: vec3<f32>, elevation: f32) -> vec3<f32> {
    // Only displace if above sea level
    let disp = max(elevation, 0.0) * particle_uniforms.relief_scale;
    return pos * (1.0 + disp);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let p = particles[instance_idx];

    // vertex_idx 0 = trail_end (tail), vertex_idx 1 = position (head)
    var sphere_pos: vec3<f32>;
    if (vertex_idx == 0u) {
        sphere_pos = p.trail_end;
    } else {
        sphere_pos = p.position;
    }

    // Compute displaced position based on wind layer type
    var world_pos: vec3<f32>;
    if (particle_uniforms.is_surface_wind == 1u) {
        // Surface wind: sample elevation and displace
        let elevation = sample_elevation(sphere_pos);
        world_pos = displace_by_elevation(sphere_pos, elevation);
        // Lift above terrain to prevent clipping
        world_pos = world_pos + sphere_pos * 0.001;
    } else {
        // Upper wind: float at fixed height above sea level
        world_pos = sphere_pos * particle_uniforms.upper_wind_height;
    }

    // Age-based fade (young = bright, old = dim)
    let age_factor = 1.0 - (p.age / particle_uniforms.max_age);
    let alpha = pow(age_factor, 0.5); // sqrt for slower fade

    // Skip drawing if no trail (trail_end == position)
    let has_trail = length(p.position - p.trail_end) > 1e-6;

    var out: VertexOutput;

    if (has_trail && p.age > 0.0) {
        out.position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
        // Cyan/white color for wind
        out.color = vec3<f32>(0.3 + 0.7 * alpha, 0.8 + 0.2 * alpha, 1.0) * alpha;
    } else {
        // Move offscreen to effectively skip this particle
        out.position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.color = vec3<f32>(0.0);
    }

    return out;
}

// Alpha value matching colored_line.wgsl
const LINE_ALPHA: f32 = 0.2;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Discard nearly transparent pixels
    let brightness = max(max(in.color.r, in.color.g), in.color.b);
    if (brightness < 0.01) {
        discard;
    }
    return vec4<f32>(in.color, LINE_ALPHA);
}
