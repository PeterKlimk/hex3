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

use glam::Vec3;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use rand::Rng;

use super::constants::*;
use super::crust::{Crust, CrustType};
use super::{FeatureFields, Tessellation};

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
    ridge_fbm: Fbm<Perlin>,
    micro_fbm: Fbm<Perlin>,
}

impl TerrainNoise {
    fn new<R: Rng>(rng: &mut R) -> Self {
        Self {
            macro_fbm: Fbm::new(rng.gen()).set_octaves(MACRO_OCTAVES),
            hills_fbm: Fbm::new(rng.gen()).set_octaves(HILLS_OCTAVES),
            ridge_fbm: Fbm::new(rng.gen()).set_octaves(RIDGE_OCTAVES),
            micro_fbm: Fbm::new(rng.gen()).set_octaves(MICRO_OCTAVES),
        }
    }

    /// Sample all four layers at a position, with modulation.
    /// Returns (combined, macro, hills, ridge, micro) contributions.
    fn sample(
        &self,
        pos: Vec3,
        convergent: f32,
        divergent: f32,
        is_continental: bool,
        is_underwater: bool,
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

        // Ridge layer: simple 3D noise, biased upward, modulated by convergence
        let ridge_contrib = {
            let ridge_pos = pos * RIDGE_FREQUENCY as f32;
            let ridge_raw =
                self.ridge_fbm
                    .get([ridge_pos.x as f64, ridge_pos.y as f64, ridge_pos.z as f64])
                    as f32;

            // Remap from observed range [-0.5, 0.5] to [0, 1]
            let ridge_biased = (ridge_raw + 0.5).clamp(0.0, 1.0);

            // Modulate by convergence so it only affects mountains
            let mountain_factor = comp_driver;

            let plate_mult = if is_continental {
                1.0
            } else {
                RIDGE_OCEANIC_MULT
            };

            ridge_biased * mountain_factor * RIDGE_AMPLITUDE * plate_mult
        };

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

impl Elevation {
    /// Generate elevation from tectonic features and crust.
    pub fn generate<R: Rng>(
        tessellation: &Tessellation,
        crust: &Crust,
        features: &FeatureFields,
        rng: &mut R,
    ) -> Self {
        let noise = TerrainNoise::new(rng);

        let (values, noise_contribution, noise_layers) =
            generate_heightmap_with_noise(tessellation, crust, features, &noise);

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
///
/// `convergent_influence` (0-1, from boundary kinematics) selects the margin
/// regime: passive margins (mid-plate craton edges) get wide gentle shelves,
/// active margins (near convergent plate boundaries) get narrow steep ones.
fn isostatic_base(
    crust_type: CrustType,
    margin_distance: f32,
    ridge_distance: f32,
    convergent_influence: f32,
) -> f32 {
    let activity = convergent_influence.clamp(0.0, 1.0);
    match crust_type {
        CrustType::Continental => {
            let shelf_width =
                PASSIVE_SHELF_WIDTH + activity * (ACTIVE_SHELF_WIDTH - PASSIVE_SHELF_WIDTH);
            // Continental: blend from margin depth to continental base
            let interior_factor = (margin_distance / shelf_width).min(1.0);
            MARGIN_DEPTH + interior_factor * (CONTINENTAL_BASE - MARGIN_DEPTH)
        }
        CrustType::Oceanic => {
            // Oceanic: thermal depth based on ridge distance
            let thermal_depth = thermal_oceanic_depth(ridge_distance);

            let transition_width = PASSIVE_OCEANIC_TRANSITION_WIDTH
                + activity * (ACTIVE_OCEANIC_TRANSITION_WIDTH - PASSIVE_OCEANIC_TRANSITION_WIDTH);
            // Near margins, blend toward MARGIN_DEPTH (continental rise effect)
            let margin_factor = (margin_distance / transition_width).min(1.0);
            // At margin (factor=0): use MARGIN_DEPTH
            // At interior (factor=1): use thermal_depth
            MARGIN_DEPTH + margin_factor * (thermal_depth - MARGIN_DEPTH)
        }
    }
}

/// Generate heightmap using tectonic features and multi-layer noise.
fn generate_heightmap_with_noise(
    tessellation: &Tessellation,
    crust: &Crust,
    features: &FeatureFields,
    noise: &TerrainNoise,
) -> (Vec<f32>, Vec<f32>, NoiseLayerData) {
    let num_cells = tessellation.num_cells();

    let mut elevations = Vec::with_capacity(num_cells);
    let mut noise_contributions = Vec::with_capacity(num_cells);
    let mut macro_layer = Vec::with_capacity(num_cells);
    let mut hills_layer = Vec::with_capacity(num_cells);
    let mut ridge_layer = Vec::with_capacity(num_cells);
    let mut micro_layer = Vec::with_capacity(num_cells);

    for i in 0..num_cells {
        let crust_type = crust.crust_type(i);
        let is_continental = crust_type == CrustType::Continental;

        // 1. Isostatic base elevation
        // Continental: margin-based shelf transition (narrow on active margins)
        // Oceanic: thermal subsidence from ridge distance + margin effect
        let base = isostatic_base(
            crust_type,
            crust.margin_distance(i),
            features.ridge_distance[i],
            features.convergent[i],
        );

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

        let (_visual_combined, macro_c, hills_c, ridge_c, micro_c) =
            noise.sample(pos, convergent, divergent, is_continental, is_underwater);

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
