//! Terrain elevation generation from stress with multi-layer noise.
//!
//! Four noise layers create realistic terrain:
//! - Macro: continental-scale smooth variation
//! - Hills: regional rolling terrain (suppressed in mountains)
//! - Ridges: drainage divides (amplified in mountains)
//! - Micro: fine surface texture

use noise::{Fbm, MultiFractal, NoiseFn, Perlin, RidgedMulti};
use rand::Rng;

use super::constants::*;
use super::boundary::{collect_plate_boundaries, BoundaryKind, SubductionPolarity};
use super::dynamics::{Dynamics, PlateType};
use super::{Plates, StressField, Tessellation};

/// Terrain elevation data.
pub struct Elevation {
    /// Elevation at each cell.
    pub values: Vec<f32>,

    /// Combined noise contribution at each cell (for visualization).
    pub noise_contribution: Vec<f32>,

    /// Individual noise layer contributions (for visualization).
    pub noise_layers: NoiseLayerData,
}

/// Individual noise layer contributions for visualization.
pub struct NoiseLayerData {
    /// Macro layer (continental tilt).
    pub macro_layer: Vec<f32>,
    /// Hills layer (regional terrain).
    pub hills_layer: Vec<f32>,
    /// Ridge layer (drainage divides).
    pub ridge_layer: Vec<f32>,
    /// Micro layer (surface texture).
    pub micro_layer: Vec<f32>,
}

/// Collection of noise generators for the four terrain layers.
struct TerrainNoise {
    macro_fbm: Fbm<Perlin>,
    hills_fbm: Fbm<Perlin>,
    ridges: RidgedMulti<Perlin>,
    micro_fbm: Fbm<Perlin>,
}

impl TerrainNoise {
    fn new<R: Rng>(rng: &mut R) -> Self {
        Self {
            macro_fbm: Fbm::new(rng.gen()).set_octaves(MACRO_OCTAVES),
            hills_fbm: Fbm::new(rng.gen()).set_octaves(HILLS_OCTAVES),
            ridges: RidgedMulti::new(rng.gen()).set_octaves(RIDGE_OCTAVES),
            micro_fbm: Fbm::new(rng.gen()).set_octaves(MICRO_OCTAVES),
        }
    }

    /// Sample all four layers at a position, with modulation.
    /// Returns (combined, macro, hills, ridge, micro) contributions.
    fn sample(
        &self,
        pos: glam::Vec3,
        stress_factor: f32,
        is_continental: bool,
        is_underwater: bool,
    ) -> (f32, f32, f32, f32, f32) {
        // Macro layer: continental tilt - PRIMARY vertical contributor
        let macro_pos = pos * MACRO_FREQUENCY as f32;
        let macro_sample =
            self.macro_fbm
                .get([macro_pos.x as f64, macro_pos.y as f64, macro_pos.z as f64]) as f32;
        let macro_amp = MACRO_AMPLITUDE * if is_continental { 1.0 } else { MACRO_OCEANIC_MULT };
        let macro_contrib = macro_sample * macro_amp;

        // Hills layer: regional terrain, suppressed by stress
        let hills_pos = pos * HILLS_FREQUENCY as f32;
        let hills_sample =
            self.hills_fbm
                .get([hills_pos.x as f64, hills_pos.y as f64, hills_pos.z as f64]) as f32;
        let hills_plate_mult = if is_continental { 1.0 } else { HILLS_OCEANIC_MULT };
        let hills_amp = HILLS_AMPLITUDE * hills_plate_mult * (1.0 - stress_factor * 0.8);
        let hills_contrib = hills_sample * hills_amp;

        // Ridge layer: mountain grain, amplified by stress, weaker offshore
        let ridge_pos = pos * RIDGE_FREQUENCY as f32;
        let ridge_sample =
            self.ridges
                .get([ridge_pos.x as f64, ridge_pos.y as f64, ridge_pos.z as f64]) as f32;
        // RidgedMulti outputs ~0-1, bias slightly upward (ridges add more "up" than "down")
        let ridge_centered = ridge_sample - 0.4;
        let ridge_plate_mult = if is_continental { 1.0 } else { RIDGE_OCEANIC_MULT };
        let ridge_stress_mult = RIDGE_MIN_FACTOR + (1.0 - RIDGE_MIN_FACTOR) * stress_factor;
        let ridge_amp = RIDGE_AMPLITUDE * ridge_plate_mult * ridge_stress_mult;
        let ridge_contrib = ridge_centered * ridge_amp;

        // Micro layer: surface texture (cosmetic)
        let micro_pos = pos * MICRO_FREQUENCY as f32;
        let micro_sample =
            self.micro_fbm
                .get([micro_pos.x as f64, micro_pos.y as f64, micro_pos.z as f64]) as f32;
        let micro_amp = MICRO_AMPLITUDE * if is_underwater { MICRO_UNDERWATER_MULT } else { 1.0 };
        let micro_contrib = micro_sample * micro_amp;

        let combined = macro_contrib + hills_contrib + ridge_contrib + micro_contrib;
        (combined, macro_contrib, hills_contrib, ridge_contrib, micro_contrib)
    }
}

/// Smoothstep function for gradual transitions.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn dijkstra_with_source(
    tessellation: &Tessellation,
    plates: &Plates,
    seed_strength: &[f32],
    restrict_to_plate: bool,
) -> (Vec<f32>, Vec<Option<usize>>) {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(Clone, Copy, PartialEq)]
    struct State {
        dist: f32,
        cell: usize,
    }

    impl Eq for State {}

    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            other
                .dist
                .partial_cmp(&self.dist)
                .unwrap_or(Ordering::Equal)
        }
    }

    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let n = tessellation.num_cells();
    let mut dist = vec![f32::INFINITY; n];
    let mut src: Vec<Option<usize>> = vec![None; n];
    let mut heap = BinaryHeap::new();

    for i in 0..n {
        if seed_strength[i] > 0.0 {
            dist[i] = 0.0;
            src[i] = Some(i);
            heap.push(State { dist: 0.0, cell: i });
        }
    }

    while let Some(State { dist: d, cell }) = heap.pop() {
        if d > dist[cell] {
            continue;
        }

        let pos = tessellation.cell_center(cell);
        let plate = plates.cell_plate[cell];

        for &neighbor in tessellation.neighbors(cell) {
            if restrict_to_plate && plates.cell_plate[neighbor] != plate {
                continue;
            }

            let neighbor_pos = tessellation.cell_center(neighbor);
            let arc_dist = pos.dot(neighbor_pos).clamp(-1.0, 1.0).acos();
            let nd = d + arc_dist;

            if nd < dist[neighbor] {
                dist[neighbor] = nd;
                src[neighbor] = src[cell];
                heap.push(State { dist: nd, cell: neighbor });
            }
        }
    }

    (dist, src)
}

/// Arc distance (radians) for the continental shelf transition zone.
/// ~0.03 radians ≈ 190 km on Earth-scale, independent of cell resolution.
const SHELF_TRANSITION_DIST: f32 = 0.03;

/// Compute interior factor for each cell: 0 at continent-ocean boundary, 1 at plate interior.
/// Used to blend elevation from sea level (at coast) to plate base elevation (at interior).
/// Uses actual arc distance for resolution-independent transition width.
fn compute_interior_factors(
    tessellation: &Tessellation,
    plates: &Plates,
    dynamics: &Dynamics,
) -> Vec<f32> {
    use std::collections::BinaryHeap;
    use std::cmp::Ordering;

    #[derive(PartialEq)]
    struct State {
        dist: f32,
        cell: usize,
    }

    impl Eq for State {}

    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reverse for min-heap
            other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
        }
    }

    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let num_cells = tessellation.num_cells();

    // Determine plate type for each cell
    let is_continental: Vec<bool> = (0..num_cells)
        .map(|i| dynamics.plate_type(plates.cell_plate[i] as usize) == PlateType::Continental)
        .collect();

    // Dijkstra from boundary cells using arc distance
    let mut distance: Vec<f32> = vec![f32::MAX; num_cells];
    let mut heap = BinaryHeap::new();

    // Seed: cells at continent-ocean boundaries
    for i in 0..num_cells {
        for &neighbor in tessellation.neighbors(i) {
            if is_continental[i] != is_continental[neighbor] {
                distance[i] = 0.0;
                heap.push(State { dist: 0.0, cell: i });
                break;
            }
        }
    }

    // Dijkstra to compute arc distance from boundary
    while let Some(State { dist, cell }) = heap.pop() {
        if dist > distance[cell] {
            continue; // Already found a shorter path
        }
        if dist >= SHELF_TRANSITION_DIST {
            continue; // Don't propagate beyond transition zone
        }

        let pos = tessellation.cell_center(cell);
        for &neighbor in tessellation.neighbors(cell) {
            let neighbor_pos = tessellation.cell_center(neighbor);
            // Arc distance = angle between unit vectors
            let arc_dist = pos.dot(neighbor_pos).clamp(-1.0, 1.0).acos();
            let new_dist = dist + arc_dist;

            if new_dist < distance[neighbor] {
                distance[neighbor] = new_dist;
                heap.push(State { dist: new_dist, cell: neighbor });
            }
        }
    }

    // Convert distance to interior factor: 0 at boundary, 1 at interior
    // This is used to blend elevation from sea level (at coast) to plate base (at interior)
    (0..num_cells)
        .map(|i| {
            let d = distance[i].min(SHELF_TRANSITION_DIST);
            d / SHELF_TRANSITION_DIST // 0 at boundary, 1 at interior
        })
        .collect()
}

impl Elevation {
    /// Generate elevation from stress and plate types.
    pub fn generate<R: Rng>(
        tessellation: &Tessellation,
        plates: &Plates,
        dynamics: &Dynamics,
        stress: &StressField,
        rng: &mut R,
    ) -> Self {
        let noise = TerrainNoise::new(rng);

        let (values, noise_contribution, noise_layers) =
            generate_heightmap_with_noise(tessellation, plates, dynamics, stress, &noise);

        Self {
            values,
            noise_contribution,
            noise_layers,
        }
    }

    /// Get elevation at a cell.
    pub fn at(&self, cell_idx: usize) -> f32 {
        self.values[cell_idx]
    }
}

/// Convert stress to elevation using sqrt response.
/// `interior_factor` is 0 at continent-ocean boundary, 1 at plate interior.
/// Continental base blends from sea level (0) at coast to CONTINENTAL_BASE inland.
/// Oceanic base stays at OCEANIC_BASE - no blending toward sea level.
fn stress_to_elevation(stress: f32, plate_type: PlateType, interior_factor: f32) -> f32 {
    // Base elevation depends on plate type:
    // Continental: 0 at coast (shelf edge), +CONTINENTAL_BASE inland
    // Oceanic: Always at OCEANIC_BASE (oceanic crust is always deep)
    //
    // The continental shelf/slope are continental crust underwater, not oceanic
    // crust rising up. Oceanic crust only rises above sea level with volcanic
    // activity (island arcs, hotspots) which comes from stress effects.
    let blended_base = match plate_type {
        PlateType::Continental => interior_factor * CONTINENTAL_BASE,
        PlateType::Oceanic => OCEANIC_BASE, // No blending - always at oceanic depth
    };

    // Compute stress effect based on plate type
    let effect = match plate_type {
        PlateType::Continental => {
            let (sens, max) = if stress >= 0.0 {
                (CONT_COMPRESSION_SENS, CONT_MAX_MOUNTAIN)
            } else {
                (CONT_TENSION_SENS, CONT_MAX_RIFT)
            };
            let effect = (stress.abs() * sens).sqrt().min(max);
            effect * stress.signum()
        }
        PlateType::Oceanic => {
            let max = if stress >= 0.0 {
                OCEAN_COMPRESSION_MAX
            } else {
                OCEAN_TENSION_MAX
            };
            (stress.abs() * OCEAN_SENSITIVITY).sqrt().min(max)
        }
    };

    blended_base + effect
}

fn sqrt_response(value: f32, sensitivity: f32, max: f32) -> f32 {
    (value.max(0.0) * sensitivity).sqrt().min(max)
}

fn gaussian_band(dist: f32, peak: f32, width: f32, gap: f32) -> f32 {
    // Suppress directly at the boundary (forearc gap) so arcs don't sit right on the interface.
    let gap_t = smoothstep(0.0, gap, dist);
    let w = width.max(1e-6);
    let z = (dist - peak) / w;
    gap_t * (-0.5 * z * z).exp()
}

fn exp_decay(dist: f32, decay: f32) -> f32 {
    let d = dist.max(0.0);
    let k = decay.max(1e-6);
    (-(d / k)).exp()
}

/// Generate heightmap with multi-layer noise modulated by stress and plate type.
fn generate_heightmap_with_noise(
    tessellation: &Tessellation,
    plates: &Plates,
    dynamics: &Dynamics,
    stress: &StressField,
    noise: &TerrainNoise,
) -> (Vec<f32>, Vec<f32>, NoiseLayerData) {
    let num_cells = tessellation.num_cells();

    let boundaries = collect_plate_boundaries(tessellation, plates, dynamics);

    // Pre-compute interior factors for coast-to-interior elevation blending
    let interior_factors = compute_interior_factors(tessellation, plates, dynamics);

    // --- Boundary-anchored feature fields ---
    //
    // For each field we compute:
    // - `seed_strength[cell]`: boundary forcing intensity for cells on the relevant side
    // - `dist[cell]`: arc distance from cell to nearest seed (within the same plate)
    // - `src[cell]`: which seed cell was nearest (used to pull strength across the band)

    let mut trench_seed = vec![0.0f32; num_cells];
    let mut arc_seed_cont = vec![0.0f32; num_cells];
    let mut arc_seed_ocean = vec![0.0f32; num_cells];
    let mut ridge_seed_ocean = vec![0.0f32; num_cells];

    for b in &boundaries {
        match b.kind {
            BoundaryKind::Convergent => {
                let Some(polarity) = b.subduction else {
                    continue;
                };

                // Boundary forcing uses the same per-side multipliers and scale as stress.
                let mult_a = match (b.type_a, b.type_b) {
                    (PlateType::Continental, PlateType::Continental) => CONV_CONT_CONT,
                    (PlateType::Oceanic, PlateType::Oceanic) => CONV_OCEAN_OCEAN,
                    (PlateType::Continental, PlateType::Oceanic) => CONV_CONT_OCEAN,
                    (PlateType::Oceanic, PlateType::Continental) => CONV_OCEAN_CONT,
                };
                let mult_b = match (b.type_b, b.type_a) {
                    (PlateType::Continental, PlateType::Continental) => CONV_CONT_CONT,
                    (PlateType::Oceanic, PlateType::Oceanic) => CONV_OCEAN_OCEAN,
                    (PlateType::Continental, PlateType::Oceanic) => CONV_CONT_OCEAN,
                    (PlateType::Oceanic, PlateType::Continental) => CONV_OCEAN_CONT,
                };

                let force_a = b.convergence.abs() * mult_a * b.edge_length * STRESS_SCALE;
                let force_b = b.convergence.abs() * mult_b * b.edge_length * STRESS_SCALE;

                match polarity {
                    SubductionPolarity::ASubducts => {
                        // A subducts: trench on A if oceanic; arc on B (overriding side).
                        if b.type_a == PlateType::Oceanic {
                            trench_seed[b.cell_a] += force_a;
                        }
                        match b.type_b {
                            PlateType::Continental => arc_seed_cont[b.cell_b] += force_b,
                            PlateType::Oceanic => arc_seed_ocean[b.cell_b] += force_b,
                        }
                    }
                    SubductionPolarity::BSubducts => {
                        if b.type_b == PlateType::Oceanic {
                            trench_seed[b.cell_b] += force_b;
                        }
                        match b.type_a {
                            PlateType::Continental => arc_seed_cont[b.cell_a] += force_a,
                            PlateType::Oceanic => arc_seed_ocean[b.cell_a] += force_a,
                        }
                    }
                }
            }
            BoundaryKind::Divergent => {
                // Only model mid-ocean ridges for ocean–ocean divergence (for now).
                if b.type_a == PlateType::Oceanic && b.type_b == PlateType::Oceanic {
                    let force = b.convergence.abs() * DIV_OCEAN_OCEAN * b.edge_length * STRESS_SCALE;
                    ridge_seed_ocean[b.cell_a] += force;
                    ridge_seed_ocean[b.cell_b] += force;
                }
            }
            BoundaryKind::Transform => {}
        }
    }

    let (trench_dist, trench_src) = dijkstra_with_source(
        tessellation,
        plates,
        &trench_seed,
        true,
    );
    let (arc_dist_cont, arc_src_cont) = dijkstra_with_source(
        tessellation,
        plates,
        &arc_seed_cont,
        true,
    );
    let (arc_dist_ocean, arc_src_ocean) = dijkstra_with_source(
        tessellation,
        plates,
        &arc_seed_ocean,
        true,
    );
    let (ridge_dist_ocean, ridge_src_ocean) = dijkstra_with_source(
        tessellation,
        plates,
        &ridge_seed_ocean,
        true,
    );

    let mut elevations = Vec::with_capacity(num_cells);
    let mut noise_contributions = Vec::with_capacity(num_cells);
    let mut macro_layer = Vec::with_capacity(num_cells);
    let mut hills_layer = Vec::with_capacity(num_cells);
    let mut ridge_layer = Vec::with_capacity(num_cells);
    let mut micro_layer = Vec::with_capacity(num_cells);

    for i in 0..num_cells {
        let cell_stress = stress.cell_stress[i];
        let plate_type = dynamics.plate_type(plates.cell_plate[i] as usize);
        let is_continental = plate_type == PlateType::Continental;

        // Start from the original stress-based elevation.
        //
        // Then add boundary-anchored features (trenches/arcs/ridges) to improve
        // physical plausibility, especially for ocean basins and subduction margins.
        let mut base = stress_to_elevation(cell_stress, plate_type, interior_factors[i]);

        // Trench: oceanic subducting side of convergent boundaries.
        if plate_type == PlateType::Oceanic {
            if let Some(src) = trench_src[i] {
                let d = trench_dist[i];
                if d.is_finite() {
                    let strength = trench_seed[src];
                    let depth = sqrt_response(strength, TRENCH_SENSITIVITY, TRENCH_MAX_DEPTH);
                    base -= depth * exp_decay(d, TRENCH_DECAY);
                }
            }
        }

        // Arc uplift: overriding side of convergent boundaries.
        // Continental arcs are broader; oceanic arcs produce island chains.
        if let Some(src) = if is_continental { arc_src_cont[i] } else { arc_src_ocean[i] } {
            let d = if is_continental { arc_dist_cont[i] } else { arc_dist_ocean[i] };
            if d.is_finite() {
                let seed_strength = if is_continental { arc_seed_cont[src] } else { arc_seed_ocean[src] };
                let uplift = sqrt_response(seed_strength, ARC_SENSITIVITY, ARC_MAX_UPLIFT);
                base += uplift * gaussian_band(d, ARC_PEAK_DIST, ARC_WIDTH, ARC_GAP);
            }
        }

        // Ridge uplift: ocean–ocean divergent boundaries.
        if plate_type == PlateType::Oceanic {
            if let Some(src) = ridge_src_ocean[i] {
                let d = ridge_dist_ocean[i];
                if d.is_finite() {
                    let strength = ridge_seed_ocean[src];
                    let uplift = sqrt_response(strength, RIDGE_SENSITIVITY, RIDGE_MAX_UPLIFT);
                    base += uplift * exp_decay(d, RIDGE_DECAY);
                }
            }
        }

        // Stress factor for noise modulation (0 = calm plains, 1 = intense mountains)
        // Using smoothstep for gradual transition instead of brittle linear normalization
        let stress_factor = smoothstep(STRESS_LOW_THRESHOLD, STRESS_HIGH_THRESHOLD, cell_stress.abs());

        // Sample noise (is_underwater determined by base elevation for now)
        let is_underwater = base < 0.0;
        let pos = tessellation.cell_center(i);
        let (combined, macro_c, hills_c, ridge_c, micro_c) =
            noise.sample(pos, stress_factor, is_continental, is_underwater);

        // Simulation elevation excludes micro noise (micro is cosmetic only)
        let simulation_noise = macro_c + hills_c + ridge_c;
        elevations.push(base + simulation_noise);
        noise_contributions.push(combined); // Visual includes all layers
        macro_layer.push(macro_c);
        hills_layer.push(hills_c);
        ridge_layer.push(ridge_c);
        micro_layer.push(micro_c);
    }

    let noise_layers = NoiseLayerData {
        macro_layer,
        hills_layer,
        ridge_layer,
        micro_layer,
    };

    (elevations, noise_contributions, noise_layers)
}
