//! Per-cell crust type, generated independently of plates.
//!
//! Crust (the material: continental vs oceanic) and plates (the motion units)
//! are separate layers. Continents are grown as "cratons" by noisy weighted
//! flood fill to a target coverage, then overlaid on whatever plate layout
//! exists. Where a craton edge falls mid-plate the margin is passive (wide
//! shelf); where it falls near a convergent plate boundary it is active
//! (narrow shelf, trench offshore). Both arise from the overlay rather than
//! being modeled explicitly.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use glam::Vec3;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use super::constants::*;
use super::plates::PLATE_NOISE_OFFSET_RANGE;
use super::Tessellation;

/// Type of crust at a cell - affects base elevation and boundary behavior.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CrustType {
    Continental,
    Oceanic,
}

/// Per-cell crust data.
///
/// The continuous field is `signed_margin_distance`; `types` is the
/// thresholded view of it (continental where positive).
pub struct Crust {
    /// Crust type per cell.
    pub types: Vec<CrustType>,

    /// Signed angular distance (radians) to the nearest crust boundary:
    /// positive inland on continental crust, negative seaward on oceanic
    /// crust, ~0 at the coast-of-record. This is the continuous crust field
    /// used for margin profiles and available for future isostasy/age work.
    pub signed_margin_distance: Vec<f32>,

    /// Craton index per cell (u32::MAX for oceanic cells).
    pub cell_craton: Vec<u32>,

    /// Number of cratons grown.
    pub num_cratons: usize,
}

impl Crust {
    /// Generate crust by growing `num_cratons` cratons to `target_fraction`
    /// of total surface cells.
    pub fn generate<R: Rng>(
        tessellation: &Tessellation,
        num_cratons: usize,
        target_fraction: f32,
        rng: &mut R,
    ) -> Self {
        let seeds = select_craton_seeds(tessellation, num_cratons, rng);
        let target_sizes = generate_craton_sizes(seeds.len(), rng);
        let cell_craton = grow_cratons(tessellation, &seeds, &target_sizes, target_fraction, rng);

        let types: Vec<CrustType> = cell_craton
            .iter()
            .map(|&c| {
                if c == u32::MAX {
                    CrustType::Oceanic
                } else {
                    CrustType::Continental
                }
            })
            .collect();

        let signed_margin_distance = compute_signed_margin_distance(tessellation, &types);

        Self {
            types,
            signed_margin_distance,
            cell_craton,
            num_cratons: seeds.len(),
        }
    }

    /// Crust type at a cell.
    pub fn crust_type(&self, cell_idx: usize) -> CrustType {
        self.types[cell_idx]
    }

    /// Whether a cell is continental crust.
    pub fn is_continental(&self, cell_idx: usize) -> bool {
        self.types[cell_idx] == CrustType::Continental
    }

    /// Unsigned angular distance (radians) from a cell to the nearest crust
    /// boundary (continent-ocean margin).
    pub fn margin_distance(&self, cell_idx: usize) -> f32 {
        self.signed_margin_distance[cell_idx].abs()
    }

    /// Fraction of cells that are continental.
    pub fn continental_fraction(&self) -> f32 {
        let count = self
            .types
            .iter()
            .filter(|&&t| t == CrustType::Continental)
            .count();
        count as f32 / self.types.len() as f32
    }
}

/// Pick spaced seed cells for cratons, relaxing the spacing requirement if
/// the ideal spacing cannot be satisfied (same scheme as plate seeds).
fn select_craton_seeds<R: Rng>(
    tessellation: &Tessellation,
    num_cratons: usize,
    rng: &mut R,
) -> Vec<usize> {
    let num_cells = tessellation.num_cells();

    // Ideal angular spacing for evenly distributed seeds on the sphere: the
    // spherical-cap radius whose N equal caps tile the unit sphere, `acos(1 -
    // 2/N)`. This matches plate seeding (`plates.rs`) and is compared against
    // angular distances below — the old `sqrt(4π/N)` was a flat-area (linear)
    // spacing, geometrically inconsistent with the angular comparison, which
    // over-spaced the seeds and leaned on the relaxation loop to recover.
    let ideal_spacing = (1.0 - 2.0 / num_cratons as f32).clamp(-1.0, 1.0).acos();
    let base_min_dist = ideal_spacing * CRATON_SEED_SPACING_FRACTION;

    let mut indices: Vec<usize> = (0..num_cells).collect();
    indices.shuffle(rng);

    let mut seeds: Vec<usize> = Vec::with_capacity(num_cratons);
    let mut seed_positions: Vec<Vec3> = Vec::with_capacity(num_cratons);

    let mut relaxation = 1.0_f32;
    while seeds.len() < num_cratons && relaxation > 0.0 {
        let min_dist = base_min_dist * relaxation;
        for &cell_idx in &indices {
            if seeds.len() >= num_cratons {
                break;
            }
            if seeds.contains(&cell_idx) {
                continue;
            }
            let pos = tessellation.cell_center(cell_idx);
            let far_enough = seed_positions
                .iter()
                .all(|&sp| pos.dot(sp).clamp(-1.0, 1.0).acos() >= min_dist);
            if far_enough {
                seeds.push(cell_idx);
                seed_positions.push(pos);
            }
        }
        relaxation -= 0.25;
    }

    seeds
}

/// Log-normal target sizes (relative weights) for cratons.
fn generate_craton_sizes<R: Rng>(num_cratons: usize, rng: &mut R) -> Vec<f32> {
    let normal = Normal::new(0.0, CRATON_SIZE_SIGMA as f64).unwrap();

    let mut sizes: Vec<f32> = (0..num_cratons)
        .map(|_| (normal.sample(rng) as f32).exp())
        .collect();

    let max_val = sizes.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_floor = max_val / CRATON_SIZE_MAX_RATIO;
    for s in &mut sizes {
        if *s < min_floor {
            *s = min_floor;
        }
    }

    // Normalize so sizes average to 1 (they shape growth, not absolute counts).
    let sum: f32 = sizes.iter().sum();
    let scale = num_cratons as f32 / sum;
    for s in &mut sizes {
        *s *= scale;
    }

    sizes
}

/// Grow cratons with a noisy weighted flood fill, stopping when the total
/// claimed cell count reaches the coverage target.
///
/// Unlike plate flood fill (which partitions the whole sphere), craton growth
/// is capped globally: unclaimed cells remain oceanic. Per-craton variety
/// comes from the distance/target_size scaling in the priority, not from hard
/// per-craton caps (hard caps can strand the queue before reaching coverage).
fn grow_cratons<R: Rng>(
    tessellation: &Tessellation,
    seeds: &[usize],
    target_sizes: &[f32],
    target_fraction: f32,
    rng: &mut R,
) -> Vec<u32> {
    let num_cells = tessellation.num_cells();
    let target_total = ((num_cells as f32 * target_fraction) as usize).max(seeds.len());

    let mut cell_craton = vec![u32::MAX; num_cells];
    let fbm: Fbm<Perlin> = Fbm::new(rng.gen()).set_octaves(CRATON_NOISE_OCTAVES);

    // Per-craton noise offsets so cratons sample decorrelated noise regions.
    let noise_offsets: Vec<Vec3> = (0..seeds.len())
        .map(|_| {
            Vec3::new(
                rng.gen::<f32>() * PLATE_NOISE_OFFSET_RANGE,
                rng.gen::<f32>() * PLATE_NOISE_OFFSET_RANGE,
                rng.gen::<f32>() * PLATE_NOISE_OFFSET_RANGE,
            )
        })
        .collect();
    let seed_positions: Vec<Vec3> = seeds.iter().map(|&s| tessellation.cell_center(s)).collect();

    let priority_of = |cell_pos: Vec3, craton: usize| -> f32 {
        let distance = seed_positions[craton].dot(cell_pos).clamp(-1.0, 1.0).acos();
        let scaled_distance = distance / target_sizes[craton];
        let noise_pos = cell_pos * CRATON_NOISE_FREQUENCY as f32 + noise_offsets[craton];
        let noise_val =
            fbm.get([noise_pos.x as f64, noise_pos.y as f64, noise_pos.z as f64]) as f32;
        scaled_distance + CRATON_NOISE_WEIGHT * noise_val * 0.5
    };

    let mut queue: BinaryHeap<Reverse<(OrderedFloat<f32>, usize, usize)>> = BinaryHeap::new();
    let mut claimed = 0usize;

    for (craton, &seed_cell) in seeds.iter().enumerate() {
        cell_craton[seed_cell] = craton as u32;
        claimed += 1;
        for &neighbor in tessellation.neighbors(seed_cell) {
            if cell_craton[neighbor] == u32::MAX {
                let p = priority_of(tessellation.cell_center(neighbor), craton);
                queue.push(Reverse((OrderedFloat(p), neighbor, craton)));
            }
        }
    }

    while claimed < target_total {
        let Some(Reverse((_, cell_idx, craton))) = queue.pop() else {
            break; // queue exhausted (only possible if target_fraction ~ 1.0)
        };
        if cell_craton[cell_idx] != u32::MAX {
            continue;
        }

        cell_craton[cell_idx] = craton as u32;
        claimed += 1;

        for &neighbor in tessellation.neighbors(cell_idx) {
            if cell_craton[neighbor] == u32::MAX {
                let p = priority_of(tessellation.cell_center(neighbor), craton);
                queue.push(Reverse((OrderedFloat(p), neighbor, craton)));
            }
        }
    }

    cell_craton
}

/// Dijkstra from the crust boundary outward in both directions, producing a
/// signed distance field (positive on continental crust, negative on oceanic).
fn compute_signed_margin_distance(tessellation: &Tessellation, types: &[CrustType]) -> Vec<f32> {
    let num_cells = tessellation.num_cells();
    let mut distance: Vec<f32> = vec![f32::MAX; num_cells];
    let mut heap: BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>> = BinaryHeap::new();

    // Seed: cells adjacent to the opposite crust type sit on the boundary.
    for i in 0..num_cells {
        for &neighbor in tessellation.neighbors(i) {
            if types[i] != types[neighbor] {
                distance[i] = 0.0;
                heap.push(Reverse((OrderedFloat(0.0), i)));
                break;
            }
        }
    }

    while let Some(Reverse((OrderedFloat(dist), cell))) = heap.pop() {
        if dist > distance[cell] {
            continue;
        }
        let pos = tessellation.cell_center(cell);
        for &neighbor in tessellation.neighbors(cell) {
            let arc = pos
                .dot(tessellation.cell_center(neighbor))
                .clamp(-1.0, 1.0)
                .acos();
            let new_dist = dist + arc;
            if new_dist < distance[neighbor] {
                distance[neighbor] = new_dist;
                heap.push(Reverse((OrderedFloat(new_dist), neighbor)));
            }
        }
    }

    distance
        .into_iter()
        .zip(types.iter())
        .map(|(d, &t)| {
            // All-ocean or all-continent worlds have no boundary; keep distances finite.
            let d = if d == f32::MAX {
                std::f32::consts::PI
            } else {
                d
            };
            if t == CrustType::Continental {
                d
            } else {
                -d
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn test_crust(seed: u64, num_cells: usize, cratons: usize, fraction: f32) -> Crust {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let tessellation = Tessellation::generate(num_cells, 1, &mut rng);
        Crust::generate(&tessellation, cratons, fraction, &mut rng)
    }

    #[test]
    fn coverage_hits_target() {
        let crust = test_crust(7, 2000, 4, 0.3);
        let frac = crust.continental_fraction();
        assert!(
            (frac - 0.3).abs() < 0.02,
            "continental fraction {frac} should be ~0.30"
        );
    }

    #[test]
    fn all_cratons_survive() {
        let crust = test_crust(7, 2000, 4, 0.3);
        let mut seen = vec![false; crust.num_cratons];
        for &c in &crust.cell_craton {
            if c != u32::MAX {
                seen[c as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "every craton should claim cells");
    }

    #[test]
    fn signed_distance_sign_matches_type() {
        let crust = test_crust(7, 2000, 4, 0.3);
        for i in 0..crust.types.len() {
            let d = crust.signed_margin_distance[i];
            match crust.types[i] {
                CrustType::Continental => assert!(d >= 0.0, "continental cell {i} has d={d}"),
                CrustType::Oceanic => assert!(d <= 0.0, "oceanic cell {i} has d={d}"),
            }
        }
    }

    #[test]
    fn extreme_fractions_are_safe() {
        let waterworld = test_crust(7, 1000, 3, 0.02);
        assert!(waterworld.continental_fraction() < 0.05);

        let pangaea = test_crust(7, 1000, 1, 0.45);
        let frac = pangaea.continental_fraction();
        assert!((frac - 0.45).abs() < 0.03, "pangaea fraction {frac}");
    }
}
