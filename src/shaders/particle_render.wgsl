// Particle trail rendering shader
//
// Renders particles as line segments from prev_position to position.
// Uses instancing: each instance is one particle, 2 vertices per instance.

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
    prev_position: vec3<f32>,
    age: f32,
}

struct ParticleUniforms {
    max_age: f32,
    lift: f32,              // How much to lift particles above surface
    _padding: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> particles: array<Particle>;
@group(1) @binding(1) var<uniform> particle_uniforms: ParticleUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let p = particles[instance_idx];

    // vertex_idx 0 = prev_position (tail), vertex_idx 1 = position (head)
    var world_pos: vec3<f32>;
    if (vertex_idx == 0u) {
        world_pos = p.prev_position;
    } else {
        world_pos = p.position;
    }

    // Lift above surface to prevent z-fighting
    world_pos = world_pos * particle_uniforms.lift;

    // Age-based fade (young = bright, old = dim)
    let age_factor = 1.0 - (p.age / particle_uniforms.max_age);
    let alpha = pow(age_factor, 0.5); // sqrt for slower fade

    // Skip drawing if particle just respawned (prev == current)
    let moved = length(p.position - p.prev_position) > 1e-6;

    var out: VertexOutput;

    if (moved && p.age > 0.0) {
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Discard nearly transparent pixels
    let brightness = max(max(in.color.r, in.color.g), in.color.b);
    if (brightness < 0.01) {
        discard;
    }
    return vec4<f32>(in.color, brightness);
}
