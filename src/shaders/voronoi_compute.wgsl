// Spherical Voronoi cell computation via half-space (great circle) clipping
//
// This compute shader computes Voronoi cells in parallel, one workgroup per cell.
// Each cell is computed independently from its k nearest neighbors.

// Constants - must match Rust side
const MAX_NEIGHBORS: u32 = 32u;
const MAX_PLANES: u32 = 24u;
const MAX_VERTICES: u32 = 32u;

// Input: generator points on the unit sphere
struct PointData {
    position: vec3<f32>,
    _padding: f32,
}

// Input: k-NN indices for each point (precomputed on CPU)
struct KnnData {
    neighbors: array<u32, 32>,  // Indices of k nearest neighbors
}

// Output: cell vertices
struct CellOutput {
    vertex_count: u32,
    vertices: array<vec4<f32>, 32>,  // xyz = position, w = unused
}

// Uniforms
struct Uniforms {
    num_points: u32,
    k: u32,  // Number of neighbors to use
}

@group(0) @binding(0) var<storage, read> points: array<PointData>;
@group(0) @binding(1) var<storage, read> knn: array<KnnData>;
@group(0) @binding(2) var<storage, read_write> cells: array<CellOutput>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

// A great circle (bisector plane) represented by its normal
struct GreatCircle {
    normal: vec3<f32>,
}

// Create bisector great circle between two points
fn bisector(a: vec3<f32>, b: vec3<f32>) -> GreatCircle {
    var gc: GreatCircle;
    gc.normal = normalize(a - b);
    return gc;
}

// Check if point is on positive side of great circle
fn contains(gc: GreatCircle, point: vec3<f32>) -> bool {
    return dot(gc.normal, point) >= -1e-6;
}

// Signed distance from point to great circle plane
fn signed_distance(gc: GreatCircle, point: vec3<f32>) -> f32 {
    return dot(gc.normal, point);
}

// Compute intersection of two great circles on the unit sphere
// Returns the point on the generator's side, or vec3(0) if invalid
fn plane_intersection(
    gc_i: GreatCircle,
    gc_j: GreatCircle,
    generator: vec3<f32>
) -> vec3<f32> {
    let cross_val = cross(gc_i.normal, gc_j.normal);
    let len = length(cross_val);

    if (len < 1e-10) {
        return vec3<f32>(0.0);  // Parallel planes
    }

    let v1 = cross_val / len;
    let v2 = -v1;

    // Check which is on generator's side of both planes
    let v1_ok = dot(gc_i.normal, v1) >= -1e-7 && dot(gc_j.normal, v1) >= -1e-7;
    let v2_ok = dot(gc_i.normal, v2) >= -1e-7 && dot(gc_j.normal, v2) >= -1e-7;

    if (v1_ok && !v2_ok) {
        return v1;
    } else if (v2_ok && !v1_ok) {
        return v2;
    } else if (v1_ok && v2_ok) {
        // Both valid, pick closer to generator
        if (dot(v1, generator) > dot(v2, generator)) {
            return v1;
        } else {
            return v2;
        }
    }

    return vec3<f32>(0.0);  // Neither valid
}

// Main compute shader - one invocation per cell
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;

    if (cell_idx >= uniforms.num_points) {
        return;
    }

    let generator = points[cell_idx].position;
    let k = min(uniforms.k, MAX_NEIGHBORS);

    // Collect bisector planes from neighbors
    var planes: array<GreatCircle, 32>;
    var num_planes: u32 = 0u;

    for (var i: u32 = 0u; i < k; i = i + 1u) {
        let neighbor_idx = knn[cell_idx].neighbors[i];
        if (neighbor_idx >= uniforms.num_points) {
            continue;
        }
        let neighbor = points[neighbor_idx].position;
        planes[num_planes] = bisector(generator, neighbor);
        num_planes = num_planes + 1u;

        if (num_planes >= MAX_PLANES) {
            break;
        }
    }

    // Find vertices: intersections of plane pairs that are inside all other planes
    var vertices: array<vec3<f32>, 32>;
    var num_vertices: u32 = 0u;

    if (num_planes >= 3u) {
        for (var i: u32 = 0u; i < num_planes; i = i + 1u) {
            for (var j: u32 = i + 1u; j < num_planes; j = j + 1u) {
                let v = plane_intersection(planes[i], planes[j], generator);

                // Check if valid (non-zero) and inside all other planes
                if (length(v) < 0.5) {
                    continue;
                }

                var inside_all = true;
                for (var k_idx: u32 = 0u; k_idx < num_planes; k_idx = k_idx + 1u) {
                    if (k_idx != i && k_idx != j) {
                        if (signed_distance(planes[k_idx], v) < -1e-6) {
                            inside_all = false;
                            break;
                        }
                    }
                }

                if (inside_all && num_vertices < MAX_VERTICES) {
                    // Check for near-duplicates
                    var is_duplicate = false;
                    for (var d: u32 = 0u; d < num_vertices; d = d + 1u) {
                        if (length(vertices[d] - v) < 1e-5) {
                            is_duplicate = true;
                            break;
                        }
                    }

                    if (!is_duplicate) {
                        vertices[num_vertices] = v;
                        num_vertices = num_vertices + 1u;
                    }
                }
            }
        }
    }

    // Order vertices CCW around generator (simple bubble sort by angle)
    if (num_vertices > 2u) {
        // Create tangent frame
        var up = vec3<f32>(0.0, 1.0, 0.0);
        if (abs(generator.y) > 0.9) {
            up = vec3<f32>(1.0, 0.0, 0.0);
        }
        let tangent_x = normalize(cross(generator, up));
        let tangent_y = normalize(cross(generator, tangent_x));

        // Compute angles
        var angles: array<f32, 32>;
        for (var i: u32 = 0u; i < num_vertices; i = i + 1u) {
            let to_point = vertices[i] - generator * dot(generator, vertices[i]);
            let x = dot(to_point, tangent_x);
            let y = dot(to_point, tangent_y);
            angles[i] = atan2(y, x);
        }

        // Bubble sort by angle
        for (var i: u32 = 0u; i < num_vertices - 1u; i = i + 1u) {
            for (var j: u32 = 0u; j < num_vertices - 1u - i; j = j + 1u) {
                if (angles[j] > angles[j + 1u]) {
                    let tmp_angle = angles[j];
                    angles[j] = angles[j + 1u];
                    angles[j + 1u] = tmp_angle;

                    let tmp_vert = vertices[j];
                    vertices[j] = vertices[j + 1u];
                    vertices[j + 1u] = tmp_vert;
                }
            }
        }
    }

    // Write output
    cells[cell_idx].vertex_count = num_vertices;
    for (var i: u32 = 0u; i < num_vertices; i = i + 1u) {
        cells[cell_idx].vertices[i] = vec4<f32>(vertices[i], 0.0);
    }
}
