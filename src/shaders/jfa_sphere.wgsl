// Simplified sphere rendering shader - no texture sampling for debugging.

struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
// Texture binding removed for debugging - will add back once basic rendering works

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.normal = in.normal;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple color based on position (no texture)
    let sphere_pos = normalize(in.world_pos);

    // Color based on position for visualization
    let color = (sphere_pos + 1.0) * 0.5; // Map -1..1 to 0..1

    // Simple lighting
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let ndotl = max(dot(in.normal, light_dir), 0.0);
    let ambient = 0.3;
    let diffuse = 0.7;
    let lit_color = color * (ambient + diffuse * ndotl);

    return vec4<f32>(lit_color, 1.0);
}
