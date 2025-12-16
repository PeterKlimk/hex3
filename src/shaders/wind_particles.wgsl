// GPU wind particle simulation with adjacency-based cell tracking
//
// Each particle:
// 1. Walks the Voronoi adjacency graph to find its current cell
// 2. Samples wind velocity from that cell
// 3. Integrates position on the sphere surface

const MAX_NEIGHBORS: u32 = 12u;
const INVALID_CELL: u32 = 0xFFFFFFFFu;

// Per-particle data (32 bytes, suitable for vertex input)
struct Particle {
    position: vec3<f32>,    // Position on unit sphere
    cell: u32,              // Current Voronoi cell index
    prev_position: vec3<f32>, // Previous position (for trail rendering)
    age: f32,               // Particle age (for respawning/fading)
}

// Cell center positions (generators)
struct CellCenter {
    position: vec3<f32>,
    _padding: f32,
}

// Wind velocity per cell (tangent to sphere)
struct WindVector {
    velocity: vec3<f32>,
    _padding: f32,
}

// Adjacency data: for cell i, neighbors are at adjacency_data[offsets[i]..offsets[i]+counts[i]]
struct AdjacencyOffsets {
    offset: u32,
    count: u32,
}

struct Uniforms {
    dt: f32,                // Time step
    num_particles: u32,
    num_cells: u32,
    max_age: f32,           // Max particle age before respawn
    wind_scale: f32,        // Wind velocity multiplier
    time: f32,              // Total elapsed time (for seeding RNG)
    _padding: vec2<f32>,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> cell_centers: array<CellCenter>;
@group(0) @binding(2) var<storage, read> wind: array<WindVector>;
@group(0) @binding(3) var<storage, read> adjacency_offsets: array<AdjacencyOffsets>;
@group(0) @binding(4) var<storage, read> adjacency_data: array<u32>;
@group(0) @binding(5) var<uniform> uniforms: Uniforms;

// Find which cell a position belongs to by walking from current cell
fn find_cell(pos: vec3<f32>, current_cell: u32) -> u32 {
    var cell = current_cell;

    // Handle invalid starting cell
    if (cell >= uniforms.num_cells) {
        cell = 0u;
    }

    let current_center = cell_centers[cell].position;
    var best_dot = dot(pos, current_center);
    var best_cell = cell;

    // Check all neighbors, pick the one with highest dot product
    let adj = adjacency_offsets[cell];
    for (var i: u32 = 0u; i < adj.count && i < MAX_NEIGHBORS; i = i + 1u) {
        let neighbor = adjacency_data[adj.offset + i];
        if (neighbor < uniforms.num_cells) {
            let neighbor_center = cell_centers[neighbor].position;
            let d = dot(pos, neighbor_center);
            if (d > best_dot) {
                best_dot = d;
                best_cell = neighbor;
            }
        }
    }

    return best_cell;
}

// Sample wind at a cell
fn sample_wind(cell: u32) -> vec3<f32> {
    if (cell >= uniforms.num_cells) {
        return vec3<f32>(0.0);
    }
    return wind[cell].velocity;
}

// Project vector onto tangent plane at position (remove radial component)
fn project_tangent(pos: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    return v - pos * dot(pos, v);
}

// PCG hash for random number generation
fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Generate random float in [0, 1)
fn rand_float(seed: u32) -> f32 {
    return f32(pcg_hash(seed)) / 4294967296.0;
}

// Generate random point on unit sphere
fn random_on_sphere(seed: u32) -> vec3<f32> {
    let h1 = pcg_hash(seed);
    let h2 = pcg_hash(h1);

    let u1 = f32(h1) / 4294967296.0;
    let u2 = f32(h2) / 4294967296.0;

    // Uniform sphere sampling
    let z = 2.0 * u1 - 1.0;
    let r = sqrt(max(0.0, 1.0 - z * z));
    let phi = 2.0 * 3.14159265359 * u2;

    return vec3<f32>(r * cos(phi), z, r * sin(phi));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= uniforms.num_particles) {
        return;
    }

    var p = particles[idx];

    // Store previous position for trail rendering
    p.prev_position = p.position;

    // Update age
    p.age += uniforms.dt;

    // Respawn if too old
    if (p.age > uniforms.max_age) {
        // Use particle index + time for unique seed each respawn
        let seed = idx * 1337u + u32(uniforms.time * 1000.0);
        p.position = random_on_sphere(seed);
        p.prev_position = p.position;
        p.cell = 0u;
        p.age = rand_float(seed + 7u) * 2.0; // Small random initial age to stagger respawns
    }

    // Find current cell (walk from previous cell)
    p.cell = find_cell(p.position, p.cell);

    // Sample wind velocity
    let wind_vel = sample_wind(p.cell) * uniforms.wind_scale;

    // Integrate position (Euler step on sphere)
    // Wind is already tangent, but project to be safe
    let tangent_vel = project_tangent(p.position, wind_vel);
    p.position = normalize(p.position + tangent_vel * uniforms.dt);

    // Write back
    particles[idx] = p;
}
