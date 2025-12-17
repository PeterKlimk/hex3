// Cubemap mapping helpers.
//
// Convention:
// - Cubemap is a 2D array texture with 6 layers in this order:
//   0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z
// - Each face uses a right-handed basis (forward, up, right = cross(forward, up)).
// - We map a world direction `dir` to face-space coordinates:
//     x = dot(dir, right) / dot(dir, forward)
//     y_ndc = dot(dir, up) / dot(dir, forward)
//   Rasterization maps NDC Y up to texture Y down, so the sampling "down" axis is -up:
//     y = dot(dir, down) / dot(dir, forward), where down = -up
//
// This is designed to match camera-style cubemap capture using the same forward/up vectors.

const CUBEMAP_FACE_POS_X: u32 = 0u;
const CUBEMAP_FACE_NEG_X: u32 = 1u;
const CUBEMAP_FACE_POS_Y: u32 = 2u;
const CUBEMAP_FACE_NEG_Y: u32 = 3u;
const CUBEMAP_FACE_POS_Z: u32 = 4u;
const CUBEMAP_FACE_NEG_Z: u32 = 5u;

fn cubemap_face_from_dir(dir: vec3<f32>) -> u32 {
    let a = abs(dir);

    if a.x >= a.y && a.x >= a.z {
        if dir.x >= 0.0 {
            return CUBEMAP_FACE_POS_X;
        }
        return CUBEMAP_FACE_NEG_X;
    }

    if a.y >= a.x && a.y >= a.z {
        if dir.y >= 0.0 {
            return CUBEMAP_FACE_POS_Y;
        }
        return CUBEMAP_FACE_NEG_Y;
    }

    if dir.z >= 0.0 {
        return CUBEMAP_FACE_POS_Z;
    }
    return CUBEMAP_FACE_NEG_Z;
}

fn cubemap_forward(face: u32) -> vec3<f32> {
    switch face {
        case CUBEMAP_FACE_POS_X: { return vec3<f32>(1.0, 0.0, 0.0); }
        case CUBEMAP_FACE_NEG_X: { return vec3<f32>(-1.0, 0.0, 0.0); }
        case CUBEMAP_FACE_POS_Y: { return vec3<f32>(0.0, 1.0, 0.0); }
        case CUBEMAP_FACE_NEG_Y: { return vec3<f32>(0.0, -1.0, 0.0); }
        case CUBEMAP_FACE_POS_Z: { return vec3<f32>(0.0, 0.0, 1.0); }
        case CUBEMAP_FACE_NEG_Z: { return vec3<f32>(0.0, 0.0, -1.0); }
        default: { return vec3<f32>(0.0, 0.0, 1.0); }
    }
}

fn cubemap_up(face: u32) -> vec3<f32> {
    // Common capture convention:
    // - Side faces (+/-X, +/-Z): up = -Y
    // - +Y: up = +Z
    // - -Y: up = -Z
    switch face {
        case CUBEMAP_FACE_POS_X: { return vec3<f32>(0.0, -1.0, 0.0); }
        case CUBEMAP_FACE_NEG_X: { return vec3<f32>(0.0, -1.0, 0.0); }
        case CUBEMAP_FACE_POS_Y: { return vec3<f32>(0.0, 0.0, 1.0); }
        case CUBEMAP_FACE_NEG_Y: { return vec3<f32>(0.0, 0.0, -1.0); }
        case CUBEMAP_FACE_POS_Z: { return vec3<f32>(0.0, -1.0, 0.0); }
        case CUBEMAP_FACE_NEG_Z: { return vec3<f32>(0.0, -1.0, 0.0); }
        default: { return vec3<f32>(0.0, -1.0, 0.0); }
    }
}

// Returns (face, u, v) where u,v are in [-1, 1] and v is "down" in texture space.
fn cubemap_dir_to_face_uv(dir_in: vec3<f32>) -> vec3<f32> {
    let dir = normalize(dir_in);
    let face = cubemap_face_from_dir(dir);

    let fwd = cubemap_forward(face);
    let up = cubemap_up(face);
    let right = normalize(cross(fwd, up));
    let down = -up;

    let denom = dot(dir, fwd);
    // denom should be positive by construction (face chosen by dominant axis sign),
    // but guard against numerical issues.
    let inv = select(0.0, 1.0 / denom, denom > 1e-6);

    let u = dot(dir, right) * inv;
    let v = dot(dir, down) * inv;
    return vec3<f32>(f32(face), u, v);
}

fn cubemap_dir_to_face_uv01(dir: vec3<f32>) -> vec3<f32> {
    let cube = cubemap_dir_to_face_uv(dir);
    let uv01 = (cube.yz + vec2<f32>(1.0)) * 0.5;
    return vec3<f32>(cube.x, uv01.x, uv01.y);
}

