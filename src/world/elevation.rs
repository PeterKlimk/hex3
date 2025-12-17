//! Terrain elevation generation from tectonic features with multi-layer noise.
//!
//! Elevation is built from:
//! - Isostatic base (continental shelf vs oceanic depth)
//! - Tectonic features (trench, arc, ridge, collision) from FeatureFields
//! - Multi-layer noise modulated by tectonic activity
//!
//! Four noise layers create realistic terrain:
//! - Macro: continental-scale smooth variation
//! - Hills: regional rolling terrain (suppressed in active areas)
//! - Ridges: drainage divides (amplified in active areas)
//! - Micro: fine surface texture

use noise::{Fbm, MultiFractal, NoiseFn, Perlin, RidgedMulti};
use rand::Rng;

use super::constants::*;
use super::dynamics::{Dynamics, PlateType};
use super::{FeatureFields, Plates, Tessellation};

/// Terrain elevation data.
pub struct Elevation {
    /// Elevation at each cell.
    pub values: Vec<f32>,

    /// Combined simulation noise contribution at each cell (macro + hills + ridges).
    ///
    /// This excludes micro noise, which is cosmetic-only and stored separately in `noise_layers`.
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
        convergent: f32,
        divergent: f32,
        is_continental: bool,
        is_underwater: bool,
        ridge_along_dir: glam::Vec3,
    ) -> (f32, f32, f32, f32, f32) {
        let comp_driver = convergent.clamp(0.0, 1.0);
        let ext_driver = divergent.clamp(0.0, 1.0);

        // Macro layer: continental tilt - PRIMARY vertical contributor
        let macro_pos = pos * MACRO_FREQUENCY as f32;
        let macro_sample =
            self.macro_fbm
                .get([macro_pos.x as f64, macro_pos.y as f64, macro_pos.z as f64])
                as f32;
        let macro_amp = MACRO_AMPLITUDE
            * if is_continental {
                1.0
            } else {
                MACRO_OCEANIC_MULT
            };
        let macro_contrib = macro_sample * macro_amp;

        // Hills layer: regional terrain.
        // Suppressed in active compressional orogens.
        let hills_pos = pos * HILLS_FREQUENCY as f32;
        let mut hills_sample =
            self.hills_fbm
                .get([hills_pos.x as f64, hills_pos.y as f64, hills_pos.z as f64])
                as f32;
        // In extension on continents, bias hills slightly downward to suggest rift basins/grabens.
        if is_continental {
            hills_sample -= HILLS_EXT_BIAS * ext_driver;
        }
        let hills_plate_mult = if is_continental {
            1.0
        } else {
            HILLS_OCEANIC_MULT
        };
        let hills_orogen_suppress = 1.0 - comp_driver * 0.8;
        let hills_amp = HILLS_AMPLITUDE * hills_plate_mult * hills_orogen_suppress;
        let hills_contrib = hills_sample * hills_amp;

        // Ridge layer: mountain grain, amplified by compressional activity, weaker offshore.
        let ridge_pos = pos * RIDGE_FREQUENCY as f32;
        let ridge_sample = {
            let n0 = self
                .ridges
                .get([ridge_pos.x as f64, ridge_pos.y as f64, ridge_pos.z as f64])
                as f32;

            let aniso_strength = (RIDGE_ANISO_STRENGTH * comp_driver).clamp(0.0, 1.0);
            if aniso_strength <= 0.0 || ridge_along_dir.length_squared() < 1e-10 {
                n0
            } else {
                // Elongate ridge noise along the inferred along-belt direction.
                //
                // NOTE: If this shows discontinuities from cell-to-cell sign flips, consider
                // enforcing a globally consistent sign on the direction field by propagating
                // over adjacency and flipping neighbor vectors to keep dot(v_i, v_j) >= 0.
                // See `NOISE_NOTES.md`.
                let along = ridge_along_dir * RIDGE_FREQUENCY as f32;
                let dp = along * RIDGE_ANISO_OFFSET;
                let p1 = ridge_pos + dp;
                let p2 = ridge_pos - dp;

                let n1 = self.ridges.get([p1.x as f64, p1.y as f64, p1.z as f64]) as f32;
                let n2 = self.ridges.get([p2.x as f64, p2.y as f64, p2.z as f64]) as f32;
                let blurred = (n0 + n1 + n2) / 3.0;
                n0 + aniso_strength * (blurred - n0)
            }
        };
        // RidgedMulti outputs ~0-1, bias slightly upward (ridges add more "up" than "down")
        let ridge_centered = ridge_sample - 0.4;
        let ridge_plate_mult = if is_continental {
            1.0
        } else {
            RIDGE_OCEANIC_MULT
        };
        let ridge_stress_mult = RIDGE_MIN_FACTOR + (1.0 - RIDGE_MIN_FACTOR) * comp_driver;
        let ridge_amp = RIDGE_AMPLITUDE * ridge_plate_mult * ridge_stress_mult;
        let ridge_contrib = ridge_centered * ridge_amp;

        // Micro layer: surface texture (cosmetic)
        let micro_pos = pos * MICRO_FREQUENCY as f32;
        let micro_sample =
            self.micro_fbm
                .get([micro_pos.x as f64, micro_pos.y as f64, micro_pos.z as f64])
                as f32;
        let micro_amp = MICRO_AMPLITUDE
            * if is_underwater {
                MICRO_UNDERWATER_MULT
            } else {
                1.0
            };
        let micro_contrib = micro_sample * micro_amp;

        let combined = macro_contrib + hills_contrib + ridge_contrib + micro_contrib;
        (
            combined,
            macro_contrib,
            hills_contrib,
            ridge_contrib,
            micro_contrib,
        )
    }
}

fn scalar_field_gradient(
    tessellation: &Tessellation,
    values: &[f32],
    cell_idx: usize,
) -> glam::Vec3 {
    use glam::Vec3;

    let v0 = values[cell_idx];
    let pos = tessellation.cell_center(cell_idx);
    let neighbors = tessellation.neighbors(cell_idx);
    if neighbors.is_empty() {
        return Vec3::ZERO;
    }

    let mut gradient = Vec3::ZERO;
    for &n in neighbors {
        let vn = values[n];
        let neighbor_pos = tessellation.cell_center(n);

        let to_neighbor = neighbor_pos - pos;
        let tangent_dir = to_neighbor - pos * pos.dot(to_neighbor);
        let tangent_len = tangent_dir.length();
        if tangent_len < 1e-6 {
            continue;
        }

        let arc_dist = pos.dot(neighbor_pos).clamp(-1.0, 1.0).acos();
        if arc_dist < 1e-6 {
            continue;
        }

        let dv = vn - v0;
        gradient += tangent_dir.normalize() * (dv / arc_dist);
    }

    gradient
}

/// Compute distance from continent-ocean boundary for each cell.
/// Returns raw arc distance in radians (not normalized).
/// Used for asymmetric blending with different widths for continental vs oceanic.
fn compute_margin_distances(
    tessellation: &Tessellation,
    plates: &Plates,
    dynamics: &Dynamics,
) -> Vec<f32> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(PartialEq)]
    struct State {
        dist: f32,
        cell: usize,
    }

    impl Eq for State {}

    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reverse for min-heap
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

    let num_cells = tessellation.num_cells();

    // Determine plate type for each cell
    let is_continental: Vec<bool> = (0..num_cells)
        .map(|i| dynamics.plate_type(plates.cell_plate[i] as usize) == PlateType::Continental)
        .collect();

    // Dijkstra from boundary cells using arc distance
    // Use max transition width as cutoff for propagation
    let max_transition = CONTINENTAL_SHELF_WIDTH.max(OCEANIC_TRANSITION_WIDTH);
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
        if dist >= max_transition {
            continue; // Don't propagate beyond max transition zone
        }

        let pos = tessellation.cell_center(cell);
        for &neighbor in tessellation.neighbors(cell) {
            let neighbor_pos = tessellation.cell_center(neighbor);
            // Arc distance = angle between unit vectors
            let arc_dist = pos.dot(neighbor_pos).clamp(-1.0, 1.0).acos();
            let new_dist = dist + arc_dist;

            if new_dist < distance[neighbor] {
                distance[neighbor] = new_dist;
                heap.push(State {
                    dist: new_dist,
                    cell: neighbor,
                });
            }
        }
    }

    distance
}

impl Elevation {
    /// Generate elevation from tectonic features and plate types.
    pub fn generate<R: Rng>(
        tessellation: &Tessellation,
        plates: &Plates,
        dynamics: &Dynamics,
        features: &FeatureFields,
        rng: &mut R,
    ) -> Self {
        let noise = TerrainNoise::new(rng);

        let (values, noise_contribution, noise_layers) =
            generate_heightmap_with_noise(tessellation, plates, dynamics, features, &noise);

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

    /// Compute elevation gradient (uphill direction) at a cell.
    ///
    /// Returns a Vec3 tangent to the sphere surface pointing in the direction
    /// of steepest ascent. Magnitude roughly indicates slope steepness.
    /// Returns zero vector for flat areas or cells with no neighbors.
    pub fn gradient(&self, tessellation: &Tessellation, cell_idx: usize) -> glam::Vec3 {
        use glam::Vec3;

        let cell_elev = self.values[cell_idx];
        let cell_pos = tessellation.cell_center(cell_idx);
        let neighbors = tessellation.neighbors(cell_idx);

        if neighbors.is_empty() {
            return Vec3::ZERO;
        }

        // Accumulate gradient as weighted sum of directions to neighbors
        let mut gradient = Vec3::ZERO;

        for &n in neighbors {
            let neighbor_elev = self.values[n];
            let neighbor_pos = tessellation.cell_center(n);

            // Direction from cell to neighbor (on sphere surface)
            let to_neighbor = neighbor_pos - cell_pos;

            // Project onto tangent plane (remove radial component)
            let tangent_dir = to_neighbor - cell_pos * cell_pos.dot(to_neighbor);
            let tangent_len = tangent_dir.length();
            if tangent_len < 1e-6 {
                continue;
            }

            // Arc distance between cells
            let arc_dist = cell_pos.dot(neighbor_pos).clamp(-1.0, 1.0).acos();
            if arc_dist < 1e-6 {
                continue;
            }

            // Elevation difference (positive = neighbor is higher)
            let elev_diff = neighbor_elev - cell_elev;

            // Slope in this direction
            let slope = elev_diff / arc_dist;

            // Accumulate: direction weighted by slope
            // Positive slope = uphill toward neighbor, so add that direction
            gradient += tangent_dir.normalize() * slope;
        }

        gradient
    }

    /// Compute gradient magnitude (slope steepness) at a cell.
    pub fn slope(&self, tessellation: &Tessellation, cell_idx: usize) -> f32 {
        self.gradient(tessellation, cell_idx).length()
    }
}

/// Compute thermal depth for oceanic crust based on distance from ridge.
/// Uses sqrt decay to model lithospheric cooling (depth ∝ √age ∝ √distance).
fn thermal_oceanic_depth(ridge_distance: f32) -> f32 {
    if !ridge_distance.is_finite() {
        // No ridge on this plate - use abyssal depth
        return ABYSSAL_DEPTH;
    }
    // Sqrt decay: young crust near ridge is shallow, old crust far from ridge is deep
    let thermal_factor = (ridge_distance / THERMAL_SUBSIDENCE_WIDTH).sqrt().min(1.0);
    RIDGE_CREST_DEPTH + thermal_factor * (ABYSSAL_DEPTH - RIDGE_CREST_DEPTH)
}

/// Compute isostatic base elevation for a cell.
///
/// Continental: blends from MARGIN_DEPTH at coast to CONTINENTAL_BASE inland.
/// Oceanic: thermal subsidence based on ridge distance, with margin effect near continents.
fn isostatic_base(plate_type: PlateType, margin_distance: f32, ridge_distance: f32) -> f32 {
    match plate_type {
        PlateType::Continental => {
            // Continental: blend from margin depth to continental base
            let interior_factor = (margin_distance / CONTINENTAL_SHELF_WIDTH).min(1.0);
            MARGIN_DEPTH + interior_factor * (CONTINENTAL_BASE - MARGIN_DEPTH)
        }
        PlateType::Oceanic => {
            // Oceanic: thermal depth based on ridge distance
            let thermal_depth = thermal_oceanic_depth(ridge_distance);

            // Near margins, blend toward MARGIN_DEPTH (continental rise effect)
            let margin_factor = (margin_distance / OCEANIC_TRANSITION_WIDTH).min(1.0);
            // At margin (factor=0): use MARGIN_DEPTH
            // At interior (factor=1): use thermal_depth
            MARGIN_DEPTH + margin_factor * (thermal_depth - MARGIN_DEPTH)
        }
    }
}

/// Generate heightmap using tectonic features and multi-layer noise.
fn generate_heightmap_with_noise(
    tessellation: &Tessellation,
    plates: &Plates,
    dynamics: &Dynamics,
    features: &FeatureFields,
    noise: &TerrainNoise,
) -> (Vec<f32>, Vec<f32>, NoiseLayerData) {
    let num_cells = tessellation.num_cells();

    // Pre-compute distances from continent-ocean margins
    let margin_distances = compute_margin_distances(tessellation, plates, dynamics);

    let mut elevations = Vec::with_capacity(num_cells);
    let mut noise_contributions = Vec::with_capacity(num_cells);
    let mut macro_layer = Vec::with_capacity(num_cells);
    let mut hills_layer = Vec::with_capacity(num_cells);
    let mut ridge_layer = Vec::with_capacity(num_cells);
    let mut micro_layer = Vec::with_capacity(num_cells);

    for i in 0..num_cells {
        let plate_type = dynamics.plate_type(plates.cell_plate[i] as usize);
        let is_continental = plate_type == PlateType::Continental;

        // 1. Isostatic base elevation
        // Continental: margin-based shelf transition
        // Oceanic: thermal subsidence from ridge distance + margin effect
        let base = isostatic_base(plate_type, margin_distances[i], features.ridge_distance[i]);

        // 2. Tectonic feature contributions (from FeatureFields)
        // Trench is negative (depression), others are positive (uplift)
        let tectonic =
            -features.trench[i] + features.arc[i] + features.ridge[i] + features.collision[i];

        let structural_elevation = base + tectonic;

        // 3. Regime-aware noise modulation.
        // Use separate convergent/divergent influence scalars derived from boundary kinematics.
        let convergent = features.convergent[i];
        let divergent = features.divergent[i];

        let is_underwater = structural_elevation < 0.0;
        let pos = tessellation.cell_center(i);
        let ridge_along_dir = {
            let grad = scalar_field_gradient(tessellation, &features.convergent, i);
            let g2 = grad.length_squared();
            if g2 < 1e-10 {
                glam::Vec3::ZERO
            } else {
                let across = grad / g2.sqrt();
                let along = pos.cross(across);
                if along.length_squared() < 1e-10 {
                    glam::Vec3::ZERO
                } else {
                    along.normalize()
                }
            }
        };
        let (_visual_combined, macro_c, hills_c, ridge_c, micro_c) = noise.sample(
            pos,
            convergent,
            divergent,
            is_continental,
            is_underwater,
            ridge_along_dir,
        );

        // Simulation elevation excludes micro noise (micro is cosmetic only)
        let simulation_noise = macro_c + hills_c + ridge_c;
        let mut elevation = structural_elevation + simulation_noise;

        // Cap volcanic island heights using tanh soft clamp.
        // Oceanic crust above sea level can't grow indefinitely - erosion/subsidence limits height.
        if !is_continental && elevation > 0.0 {
            let max_island = VOLCANIC_ISLAND_MAX_HEIGHT;
            elevation = max_island * (elevation / max_island).tanh();
        }

        elevations.push(elevation);
        noise_contributions.push(simulation_noise);
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
