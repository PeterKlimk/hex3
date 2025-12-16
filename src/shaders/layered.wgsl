// Layered visualization shader for noise/feature modes.
// Stores 6 layer values per vertex; uniform selects which to display.
// Enables instant layer switching without buffer regeneration.

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

struct LayerUniforms {
    // Which layer to display (0-5)
    layer_index: u32,
    // Colormap mode: 0 = noise (green/magenta), 1 = feature (per-layer colors)
    colormap_mode: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> layer_uniforms: LayerUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) wrap_offset: f32,
    @location(2) normal: vec3<f32>,
    @location(3) layers: vec4<f32>,  // First 4 layers
    @location(4) layer4: f32,         // 5th layer
    @location(5) layer5: f32,         // 6th layer
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) layer_value: f32,
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

    // Select the layer value based on layer_index
    switch layer_uniforms.layer_index {
        case 0u: { out.layer_value = in.layers.x; }
        case 1u: { out.layer_value = in.layers.y; }
        case 2u: { out.layer_value = in.layers.z; }
        case 3u: { out.layer_value = in.layers.w; }
        case 4u: { out.layer_value = in.layer4; }
        case 5u: { out.layer_value = in.layer5; }
        default: { out.layer_value = in.layers.x; }
    }

    return out;
}

// Noise colormap: green for positive, magenta for negative
fn noise_color(value: f32, scale: f32) -> vec3<f32> {
    let t = clamp(abs(value) * scale, 0.0, 1.0);
    if value >= 0.0 {
        // Positive: gray -> green
        return vec3<f32>(0.3 * (1.0 - t), 0.3 + 0.7 * t, 0.3 * (1.0 - t));
    } else {
        // Negative: gray -> magenta
        return vec3<f32>(0.3 + 0.7 * t, 0.3 * (1.0 - t), 0.3 + 0.7 * t);
    }
}

// Feature colormap: different color per feature type
fn feature_color(value: f32, layer_index: u32) -> vec3<f32> {
    switch layer_index {
        // Trench: blue scale (deeper = more blue)
        case 0u: {
            let t = clamp(value / 0.2, 0.0, 1.0);
            return vec3<f32>(0.1, 0.15 + 0.15 * t, 0.3 + 0.6 * t);
        }
        // Arc: orange/red scale (higher = more red)
        case 1u: {
            let t = clamp(value / 0.25, 0.0, 1.0);
            return vec3<f32>(0.3 + 0.6 * t, 0.2 + 0.3 * t, 0.1);
        }
        // Ridge: green scale (higher = more green)
        case 2u: {
            let t = clamp(value / 0.2, 0.0, 1.0);
            return vec3<f32>(0.1, 0.3 + 0.6 * t, 0.15 + 0.15 * t);
        }
        // Collision: purple scale (higher = more purple)
        case 3u: {
            let t = clamp(value / 0.35, 0.0, 1.0);
            return vec3<f32>(0.3 + 0.5 * t, 0.1, 0.3 + 0.5 * t);
        }
        // Activity: grayscale (0 = dark, 1 = white)
        case 4u: {
            let gray = 0.1 + 0.9 * value;
            return vec3<f32>(gray, gray, gray);
        }
        default: {
            return vec3<f32>(0.5, 0.5, 0.5);
        }
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Apply colormap based on mode
    var color: vec3<f32>;
    if layer_uniforms.colormap_mode == 0u {
        // Noise mode: scale varies by layer
        var scale: f32;
        switch layer_uniforms.layer_index {
            case 0u: { scale = 10.0; }  // Combined
            case 1u: { scale = 15.0; }  // Macro
            case 2u: { scale = 12.0; }  // Hills
            case 3u: { scale = 10.0; }  // Ridges
            case 4u: { scale = 50.0; }  // Micro (very small values)
            case 5u: { scale = 1.0; }   // Island chain noise (already ~[-1, 1])
            default: { scale = 10.0; }
        }
        color = noise_color(in.layer_value, scale);
    } else {
        // Feature mode
        color = feature_color(in.layer_value, layer_uniforms.layer_index);
    }

    // Simple diffuse lighting
    let ambient = 0.25;
    let diffuse = max(dot(normalize(in.world_normal), uniforms.light_dir), 0.0);
    let lighting = ambient + diffuse * 0.75;

    return vec4<f32>(color * lighting, 1.0);
}
