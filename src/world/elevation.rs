//! Terrain elevation generation: isostasy over a crust thickness field.
//!
//! Decomposition (each term a distinct physical reason ground sits where
//! it does):
//!
//!   elevation = isostatic(thickness) + thermal(ocean age)
//!             + dynamic(trench flexure/outer rise) + surface noise
//!
//! Thickness = margin ramp (continental thick, oceanic thin) + macro-scale
//! thickness noise (cratonic cores / interior basins) + tectonic thickening
//! (collision, arcs) - rift thinning. The Airy relation (linear in
//! thickness for uniform densities) converts thickness to base elevation,
//! so plateaus, rift subsidence, and margin profiles all follow from one
//! principle. Thermal subsidence stays separate (young ocean floor is high
//! because it is hot, not thick), and trenches stay separate (held out of
//! isostatic equilibrium by slab pull).
//!
//! Sea level is solved (uniform shift) so land fraction hits LAND_FRACTION.
//!
//! Surface noise layers:
//! - Hills: regional rolling terrain (suppressed in active areas)
//! - Ridges: drainage divides (amplified in active areas)
//! - Micro: fine surface texture (cosmetic)

use glam::Vec3;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use rand::Rng;
#[cfg(not(feature = "single-threaded"))]
use rayon::prelude::*;

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

/// Structural inputs to elevation assembly.
///
/// These are deliberately separated from mesh-native `Crust`/`FeatureFields`
/// so the fine mesh can rebuild elevation from transferred physical fields
/// without fabricating coarse-only domain objects.
pub struct ElevationFields {
    pub crust_thickness: Vec<f32>,
    pub continentality: Vec<f32>,
    pub ridge_age_distance: Vec<f32>,
    pub trench: Vec<f32>,
    pub ridge: Vec<f32>,
    pub convergent: Vec<f32>,
    pub divergent: Vec<f32>,
    pub is_continental: Vec<bool>,
}

#[derive(Clone, Copy)]
pub(crate) enum NoiseAssembly {
    /// Coarse world: macro + hills + ridge are simulation elevation; micro is cosmetic.
    Coarse,
    /// Fine world: macro only in simulation elevation; micro remains cosmetic.
    FineMacroMicroOnly,
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

    /// Macro-scale crust thickness variation (thickness units).
    /// `continentality` (0-1, the margin ramp parameter) scales amplitude
    /// down over oceanic crust.
    fn macro_thickness(&self, pos: Vec3, continentality: f32) -> f32 {
        let macro_pos = pos * MACRO_FREQUENCY as f32;
        let sample =
            self.macro_fbm
                .get([macro_pos.x as f64, macro_pos.y as f64, macro_pos.z as f64])
                as f32;
        let mult = MACRO_OCEANIC_MULT + (1.0 - MACRO_OCEANIC_MULT) * continentality;
        sample * MACRO_THICKNESS_AMPLITUDE * mult
    }

    /// Sample the surface noise layers at a position, with modulation.
    /// Returns (hills, ridge, micro) contributions.
    fn sample(
        &self,
        pos: Vec3,
        convergent: f32,
        divergent: f32,
        is_continental: bool,
        is_underwater: bool,
    ) -> (f32, f32, f32) {
        let comp_driver = convergent.clamp(0.0, 1.0);
        let ext_driver = divergent.clamp(0.0, 1.0);

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

        (hills_contrib, ridge_contrib, micro_contrib)
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

    pub(crate) fn generate_from_fields<R: Rng>(
        tessellation: &Tessellation,
        fields: &ElevationFields,
        rng: &mut R,
        assembly: NoiseAssembly,
    ) -> Self {
        let noise = TerrainNoise::new(rng);
        let (values, noise_contribution, noise_layers) =
            assemble_heightmap_with_noise(tessellation, fields, &noise, assembly);

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

/// Thermal elevation anomaly for oceanic crust (positive near ridges).
/// Young lithosphere is hot and buoyant; depth approaches the abyssal
/// reference as sqrt(age), with spreading-adjusted ridge age distance as
/// the age proxy (Parsons-Sclater). This is thermal buoyancy, deliberately
/// separate from crust thickness.
fn thermal_anomaly(ridge_age_distance: f32) -> f32 {
    if !ridge_age_distance.is_finite() {
        // No ridge on this plate: old basin of unknown age, mild residual
        // anomaly so these basins are not uniformly maximal-depth.
        return NO_RIDGE_DEPTH - ABYSSAL_DEPTH;
    }
    let thermal_factor = (ridge_age_distance / THERMAL_SUBSIDENCE_WIDTH)
        .sqrt()
        .min(1.0);
    (1.0 - thermal_factor) * (RIDGE_CREST_DEPTH - ABYSSAL_DEPTH)
}

/// Isostatic elevation from crust thickness (Airy, uniform densities).
/// Linear relation through the two anchor points defined in constants.
fn isostatic_elevation(thickness: f32) -> f32 {
    let slope = (CONTINENTAL_BASE - ABYSSAL_DEPTH)
        / (CRUST_THICKNESS_CONTINENTAL - CRUST_THICKNESS_OCEANIC);
    let offset = CONTINENTAL_BASE - slope * CRUST_THICKNESS_CONTINENTAL;
    slope * thickness + offset
}

/// Elevation change per unit crust thickness (the Airy slope), used to
/// express feature forcing magnitudes (calibrated in elevation units) as
/// thickness changes.
fn isostasy_slope() -> f32 {
    (CONTINENTAL_BASE - ABYSSAL_DEPTH) / (CRUST_THICKNESS_CONTINENTAL - CRUST_THICKNESS_OCEANIC)
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Continentality: the margin ramp parameter (0 = full oceanic crust,
/// 1 = full continental), from the signed margin distance. Ramp widths
/// narrow on active margins (near convergent boundaries).
fn continentality(signed_margin_distance: f32, convergent_influence: f32) -> f32 {
    let activity = convergent_influence.clamp(0.0, 1.0);
    let land_width = PASSIVE_SHELF_WIDTH + activity * (ACTIVE_SHELF_WIDTH - PASSIVE_SHELF_WIDTH);
    let ocean_width = PASSIVE_OCEANIC_TRANSITION_WIDTH
        + activity * (ACTIVE_OCEANIC_TRANSITION_WIDTH - PASSIVE_OCEANIC_TRANSITION_WIDTH);
    smoothstep((signed_margin_distance + ocean_width) / (ocean_width + land_width))
}

pub(crate) fn coarse_elevation_fields(
    tessellation: &Tessellation,
    crust: &Crust,
    features: &FeatureFields,
) -> ElevationFields {
    let num_cells = tessellation.num_cells();
    let slope = isostasy_slope();

    let mut crust_thickness = Vec::with_capacity(num_cells);
    let mut continentality_field = Vec::with_capacity(num_cells);
    let mut is_continental = Vec::with_capacity(num_cells);

    for i in 0..num_cells {
        let crust_type = crust.crust_type(i);
        let continental = crust_type == CrustType::Continental;
        let pos = tessellation.cell_center(i);
        let convergent = features.convergent[i];

        let cont = continentality(crust.signed_margin_distance[i], convergent);
        let base_thickness = CRUST_THICKNESS_OCEANIC
            + cont * (CRUST_THICKNESS_CONTINENTAL - CRUST_THICKNESS_OCEANIC);
        let macro_dt = 0.0;
        let thickening = (features.arc[i] + features.collision[i]) / slope;
        let rift = features.rift_delta[i] * cont;
        let thickness = (base_thickness + macro_dt + thickening + rift).max(0.05);

        // `pos` is intentionally touched here so the loop mirrors the full
        // assembly inputs; macro thickness is added in the noise assembly step.
        let _ = pos;
        crust_thickness.push(thickness);
        continentality_field.push(cont);
        is_continental.push(continental);
    }

    ElevationFields {
        crust_thickness,
        continentality: continentality_field,
        ridge_age_distance: features.ridge_age_distance.clone(),
        trench: features.trench.clone(),
        ridge: features.ridge.clone(),
        convergent: features.convergent.clone(),
        divergent: features.divergent.clone(),
        is_continental,
    }
}

/// Generate heightmap: thickness field -> isostasy -> thermal/dynamic
/// terms -> surface noise -> sea-level solve.
fn generate_heightmap_with_noise(
    tessellation: &Tessellation,
    crust: &Crust,
    features: &FeatureFields,
    noise: &TerrainNoise,
) -> (Vec<f32>, Vec<f32>, NoiseLayerData) {
    let fields = coarse_elevation_fields(tessellation, crust, features);
    assemble_heightmap_with_noise(tessellation, &fields, noise, NoiseAssembly::Coarse)
}

fn assemble_heightmap_with_noise(
    tessellation: &Tessellation,
    fields: &ElevationFields,
    noise: &TerrainNoise,
    assembly: NoiseAssembly,
) -> (Vec<f32>, Vec<f32>, NoiseLayerData) {
    let num_cells = tessellation.num_cells();
    let slope = isostasy_slope();

    #[cfg(not(feature = "single-threaded"))]
    let assembled: Vec<AssembledElevationCell> = (0..num_cells)
        .into_par_iter()
        .map(|i| assemble_elevation_cell(tessellation, fields, noise, assembly, slope, i))
        .collect();

    #[cfg(feature = "single-threaded")]
    let assembled: Vec<AssembledElevationCell> = (0..num_cells)
        .map(|i| assemble_elevation_cell(tessellation, fields, noise, assembly, slope, i))
        .collect();

    let mut elevations = Vec::with_capacity(num_cells);
    let mut noise_contributions = Vec::with_capacity(num_cells);
    let mut macro_layer = Vec::with_capacity(num_cells);
    let mut hills_layer = Vec::with_capacity(num_cells);
    let mut ridge_layer = Vec::with_capacity(num_cells);
    let mut micro_layer = Vec::with_capacity(num_cells);

    for cell in assembled {
        elevations.push(cell.elevation);
        noise_contributions.push(cell.noise_contribution);
        macro_layer.push(cell.macro_layer);
        hills_layer.push(cell.hills_layer);
        ridge_layer.push(cell.ridge_layer);
        micro_layer.push(cell.micro_layer);
    }

    // --- 4. Sea-level solve: uniform shift so land fraction is exact ---
    let mut sorted = elevations.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let idx = (((1.0 - LAND_FRACTION) * num_cells as f32) as usize).min(num_cells - 1);
    let sea_level = sorted[idx];
    log::debug!("sea level solve: shift={:.4}", -sea_level);

    for e in &mut elevations {
        *e -= sea_level;
    }

    // --- 5. Volcanic island soft cap (relative to true sea level) ---
    // Oceanic crust above sea level can't grow indefinitely; erosion and
    // subsidence limit island height. Kept as a transitional safeguard.
    for i in 0..num_cells {
        if !fields.is_continental[i] && elevations[i] > 0.0 {
            let max_island = VOLCANIC_ISLAND_MAX_HEIGHT;
            elevations[i] = max_island * (elevations[i] / max_island).tanh();
        }
    }

    let noise_layers = NoiseLayerData {
        macro_layer,
        hills_layer,
        ridge_layer,
        micro_layer,
    };

    (elevations, noise_contributions, noise_layers)
}

struct AssembledElevationCell {
    elevation: f32,
    noise_contribution: f32,
    macro_layer: f32,
    hills_layer: f32,
    ridge_layer: f32,
    micro_layer: f32,
}

fn assemble_elevation_cell(
    tessellation: &Tessellation,
    fields: &ElevationFields,
    noise: &TerrainNoise,
    assembly: NoiseAssembly,
    slope: f32,
    i: usize,
) -> AssembledElevationCell {
    let is_continental = fields.is_continental[i];
    let pos = tessellation.cell_center(i);
    let convergent = fields.convergent[i];
    let divergent = fields.divergent[i];

    // --- 1. Crust thickness ---
    let cont = fields.continentality[i];
    let base_thickness = fields.crust_thickness[i];

    // Macro-scale thickness variation: cratonic cores and interior basins.
    let macro_dt = noise.macro_thickness(pos, cont);

    let thickness = (base_thickness + macro_dt).max(0.05);

    // --- 2. Isostatic base + thermal + dynamic terms ---
    // Thermal anomaly applies to the oceanic part of the column;
    // trench flexure is dynamic topography (slab pull holds it out of
    // isostatic equilibrium, with signed outer-rise uplift); the small
    // ridge feature rides on the thermal swell.
    let structural_elevation = isostatic_elevation(thickness)
        + thermal_anomaly(fields.ridge_age_distance[i]) * (1.0 - cont)
        + fields.ridge[i]
        - fields.trench[i];

    // --- 3. Surface noise (hills / ridge / micro) ---
    let is_underwater = structural_elevation < 0.0;
    let (hills_c, ridge_c, micro_c) =
        noise.sample(pos, convergent, divergent, is_continental, is_underwater);

    // Simulation elevation excludes micro noise (micro is cosmetic only).
    // The macro layer is reported as its isostatic elevation contribution
    // for visualization continuity.
    let macro_c = macro_dt * slope;
    let (hills_sim, ridge_sim) = match assembly {
        NoiseAssembly::Coarse => (hills_c, ridge_c),
        NoiseAssembly::FineMacroMicroOnly => (0.0, 0.0),
    };
    let simulation_noise = macro_c + hills_sim + ridge_sim;
    let elevation = structural_elevation + hills_sim + ridge_sim;

    AssembledElevationCell {
        elevation,
        noise_contribution: simulation_noise,
        macro_layer: macro_c,
        hills_layer: hills_c,
        ridge_layer: ridge_c,
        micro_layer: micro_c,
    }
}
