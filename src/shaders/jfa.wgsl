// Jump Flooding Algorithm compute shader for spherical Voronoi on cube map.
//
// Each pixel stores:
// - R,G,B = position of closest seed (3D on unit sphere)
// - A = seed ID (or -1 if no seed found yet)
//
// Uses pixel-based stepping on cube map with sphere conversion for face boundaries.

@group(0) @binding(0) var input_texture: texture_2d_array<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d_array<rgba32float, write>;

struct Uniforms {
    step_size: u32,      // Step size in pixels
    face_size: u32,      // Size of each cube face in pixels
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

// Convert cube map face + UV to 3D unit sphere position
fn cube_uv_to_sphere(face: u32, u: f32, v: f32) -> vec3<f32> {
    var dir: vec3<f32>;
    switch face {
        case 0u: { dir = vec3<f32>(1.0, -v, -u); }   // +X
        case 1u: { dir = vec3<f32>(-1.0, -v, u); }   // -X
        case 2u: { dir = vec3<f32>(u, 1.0, v); }     // +Y
        case 3u: { dir = vec3<f32>(u, -1.0, -v); }   // -Y
        case 4u: { dir = vec3<f32>(u, -v, 1.0); }    // +Z
        case 5u: { dir = vec3<f32>(-u, -v, -1.0); }  // -Z
        default: { dir = vec3<f32>(0.0, 0.0, 1.0); }
    }
    return normalize(dir);
}

// Convert 3D sphere position to cube map face + pixel coordinates
fn sphere_to_cube_pixel(pos: vec3<f32>, size: u32) -> vec3<u32> {
    let abs_pos = abs(pos);
    var face: u32;
    var u: f32;
    var v: f32;

    if abs_pos.x >= abs_pos.y && abs_pos.x >= abs_pos.z {
        if pos.x > 0.0 {
            face = 0u; // +X
            u = -pos.z / abs_pos.x;
            v = -pos.y / abs_pos.x;
        } else {
            face = 1u; // -X
            u = pos.z / abs_pos.x;
            v = -pos.y / abs_pos.x;
        }
    } else if abs_pos.y >= abs_pos.x && abs_pos.y >= abs_pos.z {
        if pos.y > 0.0 {
            face = 2u; // +Y
            u = pos.x / abs_pos.y;
            v = pos.z / abs_pos.y;
        } else {
            face = 3u; // -Y
            u = pos.x / abs_pos.y;
            v = -pos.z / abs_pos.y;
        }
    } else {
        if pos.z > 0.0 {
            face = 4u; // +Z
            u = pos.x / abs_pos.z;
            v = -pos.y / abs_pos.z;
        } else {
            face = 5u; // -Z
            u = -pos.x / abs_pos.z;
            v = -pos.y / abs_pos.z;
        }
    }

    // Convert UV [-1,1] to pixel coordinates
    let fsize = f32(size);
    let px = u32(clamp((u + 1.0) * 0.5 * fsize, 0.0, fsize - 1.0));
    let py = u32(clamp((v + 1.0) * 0.5 * fsize, 0.0, fsize - 1.0));

    return vec3<u32>(px, py, face);
}

// Compute arc distance (angle) between two points on unit sphere
fn arc_distance(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return acos(clamp(dot(a, b), -1.0, 1.0));
}

// Sample at a pixel location, handling face boundary crossings via sphere conversion
fn sample_neighbor(base_face: u32, base_x: i32, base_y: i32, dx: i32, dy: i32, step: i32, size: u32) -> vec4<f32> {
    let nx = base_x + dx * step;
    let ny = base_y + dy * step;
    let isize = i32(size);

    // Check if we're still on the same face
    if nx >= 0 && nx < isize && ny >= 0 && ny < isize {
        // Simple case: same face
        return textureLoad(input_texture, vec2<u32>(u32(nx), u32(ny)), base_face, 0);
    }

    // Off the edge - convert to sphere coords and back to find correct face/pixel
    // First get the 3D position of this pixel if it were on the base face (extrapolated)
    let fsize = f32(size);
    let u = (f32(nx) + 0.5) / fsize * 2.0 - 1.0;
    let v = (f32(ny) + 0.5) / fsize * 2.0 - 1.0;
    let sphere_pos = cube_uv_to_sphere(base_face, u, v);

    // Now convert back to find the actual face and pixel
    let dest = sphere_to_cube_pixel(sphere_pos, size);
    return textureLoad(input_texture, vec2<u32>(dest.x, dest.y), dest.z, 0);
}

@compute @workgroup_size(8, 8, 1)
fn jfa_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let face = global_id.z;
    let x = global_id.x;
    let y = global_id.y;
    let size = uniforms.face_size;
    let step = i32(uniforms.step_size);

    if x >= size || y >= size || face >= 6u {
        return;
    }

    // Get this pixel's 3D position on the sphere
    let fsize = f32(size);
    let u = (f32(x) + 0.5) / fsize * 2.0 - 1.0;
    let v = (f32(y) + 0.5) / fsize * 2.0 - 1.0;
    let my_pos = cube_uv_to_sphere(face, u, v);

    // Current best seed from input
    let current = textureLoad(input_texture, vec2<u32>(x, y), face, 0);
    var best_seed_pos = current.xyz;
    var best_seed_id = current.w;
    var best_dist = 1e10f;

    // If we already have a seed, calculate its distance
    if best_seed_id >= 0.0 {
        best_dist = arc_distance(my_pos, best_seed_pos);
    }

    let ix = i32(x);
    let iy = i32(y);

    // JFA: check 9 neighbors in a 3x3 grid at step distance
    for (var dy = -1i; dy <= 1i; dy = dy + 1i) {
        for (var dx = -1i; dx <= 1i; dx = dx + 1i) {
            if dx == 0i && dy == 0i {
                continue; // Skip self
            }

            let neighbor_data = sample_neighbor(face, ix, iy, dx, dy, step, size);
            let neighbor_seed_id = neighbor_data.w;

            // If neighbor has a seed, check if it's closer to us
            if neighbor_seed_id >= 0.0 {
                let neighbor_seed_pos = neighbor_data.xyz;
                let dist = arc_distance(my_pos, neighbor_seed_pos);

                if dist < best_dist {
                    best_dist = dist;
                    best_seed_pos = neighbor_seed_pos;
                    best_seed_id = neighbor_seed_id;
                }
            }
        }
    }

    // Write result
    textureStore(output_texture, vec2<u32>(x, y), face, vec4<f32>(best_seed_pos, best_seed_id));
}
