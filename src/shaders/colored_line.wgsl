// Colored line rendering shader - uses vertex colors

// Constants for map projection
const PI: f32 = 3.14159265359;
const HALF_PI: f32 = 1.57079632679;

struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding1: f32,
    light_dir: vec3<f32>,
    relief_scale: f32,
    hemisphere_lighting: f32,
    map_mode: f32, // 0.0 = globe view, 1.0 = equirectangular map view
    _padding2: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) wrap_offset: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
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

    var final_pos: vec3<f32>;
    if (uniforms.map_mode > 0.5) {
        // Map view: project to 2D equirectangular
        final_pos = sphere_to_map(in.position, in.wrap_offset);
    } else {
        // Globe view: use 3D position directly
        final_pos = in.position;
    }

    out.clip_position = uniforms.view_proj * vec4<f32>(final_pos, 1.0);
    out.color = in.color;
    return out;
}

// Alpha value for edge lines (0.0 = invisible, 1.0 = solid)
const EDGE_ALPHA: f32 = 0.2;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, EDGE_ALPHA);
}
