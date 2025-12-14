// Surface line shader - for rivers, roads, and other terrain features
// Vertices are displaced based on relief_scale uniform

struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding1: f32,
    light_dir: vec3<f32>,
    relief_scale: f32,  // 0.0 = flat, ~0.1 = relief mode
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,   // base position on unit sphere
    @location(1) elevation: f32,         // terrain height for displacement
    @location(2) color: vec4<f32>,       // RGBA color with alpha
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

// Small offset to prevent z-fighting with terrain
const Z_OFFSET: f32 = 0.003;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Displace vertex based on elevation and relief_scale
    let normalized_pos = normalize(in.position);
    let height = 1.0 + in.elevation * uniforms.relief_scale + Z_OFFSET;
    let world_pos = normalized_pos * height;

    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
