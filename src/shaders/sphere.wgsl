// Sphere rendering shader with basic lighting

// Constants for map projection
const PI: f32 = 3.14159265359;
const HALF_PI: f32 = 1.57079632679;

struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding1: f32,
    light_dir: vec3<f32>,
    map_mode: f32, // 0.0 = globe view, 1.0 = equirectangular map view
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
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec3<f32>,
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
    out.world_normal = in.normal;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ambient = 0.25;
    let diffuse = max(dot(normalize(in.world_normal), uniforms.light_dir), 0.0);
    let lighting = ambient + diffuse * 0.75;
    return vec4<f32>(in.color * lighting, 1.0);
}
