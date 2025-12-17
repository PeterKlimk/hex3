// Elevation cubemap rendering shader
//
// Renders terrain mesh to cubemap faces. Each face uses a standard
// perspective projection looking outward from the origin.
// The fragment shader outputs the elevation value for that point.

struct Uniforms {
    view_proj: mat4x4<f32>,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) elevation: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) elevation: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.elevation = in.elevation;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    return in.elevation;
}
