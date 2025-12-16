// GPU wind particle simulation with adjacency-based cell tracking
//
// Each particle:
// 1. Samples wind velocity from its current cell
// 2. Integrates position on the sphere surface
// 3. Computes trail end based on wind direction (independent of frame time)
// 4. Walks the Voronoi adjacency graph to update its cell

const MAX_NEIGHBORS: u32 = 12u;

// Per-particle data (32 bytes, suitable for vertex input)
struct Particle {
    position: vec3<f32>,    // Position on unit sphere
    cell: u32,              // Current Voronoi cell index
    trail_end: vec3<f32>,   // Trail end position (computed from wind direction)
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
// Padded to 16 bytes for proper alignment
struct AdjacencyOffsets {
    offset: u32,
    count: u32,
    _padding: vec2<u32>,
}

struct Uniforms {
    dt: f32,                // Frame time in seconds (for movement + aging)
    speed_scale: f32,       // How much wind moves particles per second
    trail_scale: f32,       // Trail length per unit wind speed (independent of dt)
    time: f32,              // Total elapsed time (for seeding RNG)
    num_particles: u32,
    num_cells: u32,
    _pad0: vec2<u32>,
    max_age: f32,           // Max particle age before respawn (in seconds)
    _pad1: vec3<f32>,
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= uniforms.num_particles) {
        return;
    }

    var p = particles[idx];

    // Update age
    p.age += uniforms.dt;

    // Respawn if too old
    if (p.age > uniforms.max_age) {
        // Use particle index + time for unique seed each respawn
        let seed = idx * 1337u + u32(uniforms.time * 1000.0);

        // Spawn at a random cell center
        let random_cell = pcg_hash(seed) % uniforms.num_cells;
        p.position = cell_centers[random_cell].position;
        p.trail_end = p.position;
        p.cell = random_cell;
        // Small random initial age (stagger)
        p.age = rand_float(seed + 7u) * (uniforms.max_age * 0.5);
    }

    // Sample wind at current cell
    let wind_vel = sample_wind(p.cell);
    let wind_mag = length(wind_vel);

    // Compute trail end based on wind direction (independent of dt)
    // Trail points backward from current position in opposite direction of wind
    if (wind_mag > 0.001) {
        let wind_dir = wind_vel / wind_mag;
        // Trail length proportional to wind magnitude
        p.trail_end = normalize(p.position - wind_dir * wind_mag * uniforms.trail_scale);
    } else {
        // No wind = no trail
        p.trail_end = p.position;
    }

    // Move particle (time-normalized)
    p.position = normalize(p.position + wind_vel * uniforms.dt * uniforms.speed_scale);

    // Update cell tracking after moving
    p.cell = find_cell(p.position, p.cell);

    // Write back
    particles[idx] = p;
}
