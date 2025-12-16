// Render shader for visualizing the JFA spherical Voronoi cube map.
//
// Displays an equirectangular projection of the sphere with Voronoi cells
// colored by seed ID.

@group(0) @binding(0) var cube_map: texture_2d_array<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle - 3 vertices cover the screen
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Generate fullscreen triangle positions
    // Uses oversized triangle trick: 3 vertices create a triangle that covers the viewport
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);

    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y); // Flip Y for correct orientation

    return out;
}

const PI: f32 = 3.14159265359;

// Convert equirectangular UV to 3D sphere position
fn equirect_to_sphere(uv: vec2<f32>) -> vec3<f32> {
    let theta = uv.x * 2.0 * PI - PI;  // Longitude: -PI to PI
    let phi = uv.y * PI;                // Latitude: 0 (north) to PI (south)

    return vec3<f32>(
        sin(phi) * cos(theta),
        cos(phi),
        sin(phi) * sin(theta)
    );
}

// Convert 3D sphere position to cube map face + UV
fn sphere_to_cube_uv(pos: vec3<f32>) -> vec3<f32> {
    let abs_pos = abs(pos);
    var face: f32;
    var u: f32;
    var v: f32;

    if abs_pos.x >= abs_pos.y && abs_pos.x >= abs_pos.z {
        if pos.x > 0.0 {
            face = 0.0;
            u = -pos.z / abs_pos.x;
            v = -pos.y / abs_pos.x;
        } else {
            face = 1.0;
            u = pos.z / abs_pos.x;
            v = -pos.y / abs_pos.x;
        }
    } else if abs_pos.y >= abs_pos.x && abs_pos.y >= abs_pos.z {
        if pos.y > 0.0 {
            face = 2.0;
            u = pos.x / abs_pos.y;
            v = pos.z / abs_pos.y;
        } else {
            face = 3.0;
            u = pos.x / abs_pos.y;
            v = -pos.z / abs_pos.y;
        }
    } else {
        if pos.z > 0.0 {
            face = 4.0;
            u = pos.x / abs_pos.z;
            v = -pos.y / abs_pos.z;
        } else {
            face = 5.0;
            u = -pos.x / abs_pos.z;
            v = -pos.y / abs_pos.z;
        }
    }

    return vec3<f32>(face, u, v);
}

// Hash function for generating colors from seed ID
fn hash_color(seed_id: u32) -> vec3<f32> {
    // Use golden ratio based hash for good color distribution
    let golden = 0.618033988749895;
    let h = fract(f32(seed_id) * golden);
    let s = 0.7;
    let l = 0.6;

    // HSL to RGB conversion
    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    let x = c * (1.0 - abs(fract(h * 6.0) * 2.0 - 1.0));
    let m = l - c / 2.0;

    var rgb: vec3<f32>;
    let h6 = h * 6.0;
    if h6 < 1.0 {
        rgb = vec3<f32>(c, x, 0.0);
    } else if h6 < 2.0 {
        rgb = vec3<f32>(x, c, 0.0);
    } else if h6 < 3.0 {
        rgb = vec3<f32>(0.0, c, x);
    } else if h6 < 4.0 {
        rgb = vec3<f32>(0.0, x, c);
    } else if h6 < 5.0 {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }

    return rgb + vec3<f32>(m);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Convert screen UV to 3D sphere position (equirectangular projection)
    let sphere_pos = equirect_to_sphere(in.uv);

    // Convert to cube map coordinates
    let cube = sphere_to_cube_uv(sphere_pos);
    let face = u32(cube.x);

    // Get texture dimensions
    let dims = textureDimensions(cube_map);
    let size = f32(dims.x);

    // Convert UV [-1,1] to pixel coordinates
    let px = (cube.y + 1.0) * 0.5 * size - 0.5;
    let py = (cube.z + 1.0) * 0.5 * size - 0.5;

    // Clamp to valid range
    let x = u32(clamp(px, 0.0, size - 1.0));
    let y = u32(clamp(py, 0.0, size - 1.0));

    // Sample the cube map
    let data = textureLoad(cube_map, vec2<u32>(x, y), face, 0);
    let seed_id = data.w;

    // Color by seed ID
    if seed_id < 0.0 {
        // No seed assigned yet - show as dark gray
        return vec4<f32>(0.1, 0.1, 0.1, 1.0);
    }

    let color = hash_color(u32(seed_id));

    // Add subtle edge detection for cell boundaries
    // Check if neighbors have different seed IDs
    var is_edge = false;
    let offsets = array<vec2<i32>, 4>(
        vec2<i32>(1, 0),
        vec2<i32>(-1, 0),
        vec2<i32>(0, 1),
        vec2<i32>(0, -1)
    );

    for (var i = 0u; i < 4u; i = i + 1u) {
        let nx = i32(x) + offsets[i].x;
        let ny = i32(y) + offsets[i].y;

        if nx >= 0 && nx < i32(size) && ny >= 0 && ny < i32(size) {
            let neighbor = textureLoad(cube_map, vec2<u32>(u32(nx), u32(ny)), face, 0);
            if abs(neighbor.w - seed_id) > 0.5 {
                is_edge = true;
                break;
            }
        }
    }

    if is_edge {
        // Darken edges slightly
        return vec4<f32>(color * 0.5, 1.0);
    }

    return vec4<f32>(color, 1.0);
}
