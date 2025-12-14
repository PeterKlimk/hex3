//! Tectonic plate assignment via flood fill.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use glam::Vec3;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use super::constants::*;
use super::Tessellation;

/// Tectonic plate assignments for each cell.
pub struct Plates {
    /// For each cell index, which plate does it belong to.
    pub cell_plate: Vec<u32>,

    /// Number of plates.
    pub num_plates: usize,

    /// Indices of cells that are on plate boundaries.
    pub boundary_cells: Vec<usize>,

    /// Target sizes used during generation (needed for type assignment).
    pub(crate) target_sizes: Vec<f32>,
}

impl Plates {
    /// Generate tectonic plates from a tessellation using flood fill.
    pub fn generate<R: Rng>(tessellation: &Tessellation, num_plates: usize, rng: &mut R) -> Self {
        // Select spaced seed cells
        let seeds = select_seeds_spaced(tessellation, num_plates, rng);

        // Generate target sizes for varied plate sizes
        let target_sizes = generate_target_sizes(num_plates, rng);

        // Weighted flood fill
        let cell_plate = flood_fill_weighted(tessellation, &seeds, &target_sizes, rng);

        // Find boundary cells
        let boundary_cells = find_boundary_cells(&tessellation.adjacency, &cell_plate);

        Self {
            cell_plate,
            num_plates,
            boundary_cells,
            target_sizes,
        }
    }

    /// Check if a cell is on a plate boundary.
    pub fn is_boundary(&self, cell_idx: usize) -> bool {
        self.boundary_cells.binary_search(&cell_idx).is_ok()
    }
}

/// Find cells that are on plate boundaries (adjacent to cells on different plates).
fn find_boundary_cells(adjacency: &[Vec<usize>], cell_plate: &[u32]) -> Vec<usize> {
    let mut boundary = Vec::new();
    for (cell_idx, neighbors) in adjacency.iter().enumerate() {
        let my_plate = cell_plate[cell_idx];
        if neighbors.iter().any(|&n| cell_plate[n] != my_plate) {
            boundary.push(cell_idx);
        }
    }
    boundary.sort(); // Sort for binary search
    boundary
}

/// Compute minimum angular distance between seeds based on plate count.
fn min_seed_distance(num_plates: usize) -> f32 {
    // Ideal angular radius if N seeds perfectly tile the sphere
    let ideal = (1.0 - 2.0 / num_plates as f32).acos();
    ideal * SEED_SPACING_FRACTION
}

/// Select plate seeds with minimum angular distance between them.
fn select_seeds_spaced<R: Rng>(
    tessellation: &Tessellation,
    num_plates: usize,
    rng: &mut R,
) -> Vec<usize> {
    let base_min_dist = min_seed_distance(num_plates);

    let mut indices: Vec<usize> = (0..tessellation.num_cells()).collect();
    indices.shuffle(rng);

    let mut seeds: Vec<usize> = Vec::with_capacity(num_plates);
    let mut seed_positions: Vec<Vec3> = Vec::with_capacity(num_plates);

    // First pass: use full minimum distance
    for &cell_idx in &indices {
        if seeds.len() >= num_plates {
            break;
        }

        let pos = tessellation.cell_center(cell_idx);

        let far_enough = seed_positions
            .iter()
            .all(|&seed_pos| pos.dot(seed_pos).clamp(-1.0, 1.0).acos() >= base_min_dist);

        if far_enough {
            seeds.push(cell_idx);
            seed_positions.push(pos);
        }
    }

    // Fallback: progressively relax distance constraint
    let mut relaxation = 0.75;
    while seeds.len() < num_plates && relaxation > 0.0 {
        let relaxed_dist = base_min_dist * relaxation;
        indices.shuffle(rng);

        for &cell_idx in &indices {
            if seeds.len() >= num_plates {
                break;
            }

            if seeds.contains(&cell_idx) {
                continue;
            }

            let pos = tessellation.cell_center(cell_idx);

            let far_enough = seed_positions
                .iter()
                .all(|&seed_pos| pos.dot(seed_pos).clamp(-1.0, 1.0).acos() >= relaxed_dist);

            if far_enough {
                seeds.push(cell_idx);
                seed_positions.push(pos);
            }
        }

        relaxation -= 0.25;
    }

    // Final fallback: add any remaining cells
    if seeds.len() < num_plates {
        for &cell_idx in &indices {
            if seeds.len() >= num_plates {
                break;
            }
            if !seeds.contains(&cell_idx) {
                seeds.push(cell_idx);
                seed_positions.push(tessellation.cell_center(cell_idx));
            }
        }
    }

    seeds
}

/// Generate target sizes for plates using log-normal distribution.
fn generate_target_sizes<R: Rng>(num_plates: usize, rng: &mut R) -> Vec<f32> {
    let normal = Normal::new(0.0, TARGET_SIZE_SIGMA as f64).unwrap();

    let mut sizes: Vec<f32> = (0..num_plates)
        .map(|_| (normal.sample(rng) as f32).exp())
        .collect();

    // Clamp ratio between max and min
    let max_val = sizes.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_val = sizes.iter().cloned().fold(f32::INFINITY, f32::min);

    if max_val > min_val * TARGET_SIZE_MAX_RATIO {
        let target_min = max_val / TARGET_SIZE_MAX_RATIO;
        for s in &mut sizes {
            if *s < target_min {
                *s = target_min;
            }
        }
    }

    // Normalize so sum = num_plates
    let sum: f32 = sizes.iter().sum();
    let scale = num_plates as f32 / sum;
    for s in &mut sizes {
        *s *= scale;
    }

    sizes
}

/// Per-plate state for flood fill.
struct PlateState {
    seed_pos: Vec3,
    target_size: f32,
    noise_offset: Vec3,
}

impl PlateState {
    fn new<R: Rng>(seed_pos: Vec3, target_size: f32, rng: &mut R) -> Self {
        let noise_offset = Vec3::new(
            rng.gen::<f32>() * 100.0,
            rng.gen::<f32>() * 100.0,
            rng.gen::<f32>() * 100.0,
        );
        PlateState {
            seed_pos,
            target_size,
            noise_offset,
        }
    }
}

/// Calculate priority for a cell (lower = claim sooner).
fn compute_priority(
    cell_pos: Vec3,
    plate: &PlateState,
    total_neighbors: usize,
    same_plate_neighbors: usize,
    cell_plate: &[u32],
    plate_id: usize,
    adjacency: &[Vec<usize>],
    cell_idx: usize,
    fbm: &Fbm<Perlin>,
) -> f32 {
    let _ = (cell_plate, adjacency, cell_idx); // Mark as intentionally unused for now

    // Arc distance from seed, scaled by target_size
    let distance = plate.seed_pos.dot(cell_pos).clamp(-1.0, 1.0).acos();
    let scaled_distance = distance / plate.target_size;

    // Noise at this position
    let noise_pos = cell_pos * NOISE_FREQUENCY as f32 + plate.noise_offset;
    let noise_val = fbm.get([noise_pos.x as f64, noise_pos.y as f64, noise_pos.z as f64]) as f32;

    // Net perimeter change: lower = better
    let perimeter_delta = total_neighbors as f32 - 2.0 * same_plate_neighbors as f32;

    let _ = plate_id; // Mark as intentionally unused

    scaled_distance + NOISE_WEIGHT * noise_val * 0.5 + NEIGHBOR_BONUS * perimeter_delta
}

/// Weighted flood fill with single global priority queue.
fn flood_fill_weighted<R: Rng>(
    tessellation: &Tessellation,
    seeds: &[usize],
    target_sizes: &[f32],
    rng: &mut R,
) -> Vec<u32> {
    let num_cells = tessellation.num_cells();
    let mut cell_plate = vec![u32::MAX; num_cells];

    // Create fBm noise generator
    let fbm: Fbm<Perlin> = Fbm::new(rng.gen())
        .set_frequency(NOISE_FREQUENCY)
        .set_octaves(NOISE_OCTAVES);

    // Initialize plate states
    let plates: Vec<PlateState> = seeds
        .iter()
        .enumerate()
        .map(|(plate_id, &seed_cell)| {
            let seed_pos = tessellation.cell_center(seed_cell);
            PlateState::new(seed_pos, target_sizes[plate_id], rng)
        })
        .collect();

    // Single global priority queue
    let mut queue: BinaryHeap<Reverse<(OrderedFloat<f32>, usize, usize)>> = BinaryHeap::new();

    // Initialize: claim seeds and add their neighbors
    for (plate_id, &seed_cell) in seeds.iter().enumerate() {
        cell_plate[seed_cell] = plate_id as u32;

        for &neighbor in tessellation.neighbors(seed_cell) {
            if cell_plate[neighbor] == u32::MAX {
                let neighbor_pos = tessellation.cell_center(neighbor);
                let total_neighbors = tessellation.neighbors(neighbor).len();
                let same_plate_neighbors = tessellation
                    .neighbors(neighbor)
                    .iter()
                    .filter(|&&n| cell_plate[n] == plate_id as u32)
                    .count();
                let priority = compute_priority(
                    neighbor_pos,
                    &plates[plate_id],
                    total_neighbors,
                    same_plate_neighbors,
                    &cell_plate,
                    plate_id,
                    &tessellation.adjacency,
                    neighbor,
                    &fbm,
                );
                queue.push(Reverse((OrderedFloat(priority), neighbor, plate_id)));
            }
        }
    }

    // Main loop
    while let Some(Reverse((_, cell_idx, plate_id))) = queue.pop() {
        if cell_plate[cell_idx] != u32::MAX {
            continue;
        }

        cell_plate[cell_idx] = plate_id as u32;

        for &neighbor in tessellation.neighbors(cell_idx) {
            if cell_plate[neighbor] == u32::MAX {
                let neighbor_pos = tessellation.cell_center(neighbor);
                let total_neighbors = tessellation.neighbors(neighbor).len();
                let same_plate_neighbors = tessellation
                    .neighbors(neighbor)
                    .iter()
                    .filter(|&&n| cell_plate[n] == plate_id as u32)
                    .count();
                let priority = compute_priority(
                    neighbor_pos,
                    &plates[plate_id],
                    total_neighbors,
                    same_plate_neighbors,
                    &cell_plate,
                    plate_id,
                    &tessellation.adjacency,
                    neighbor,
                    &fbm,
                );
                queue.push(Reverse((OrderedFloat(priority), neighbor, plate_id)));
            }
        }
    }

    cell_plate
}
