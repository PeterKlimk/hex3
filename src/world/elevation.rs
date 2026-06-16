//! Terrain elevation generation: isostasy over a crust thickness field.
//!
//! Decomposition (each term a distinct physical reason ground sits where
//! it does):
//!
//!   elevation = isostatic(thickness) + thermal(ocean age)
//!             + dynamic(trench flexure/outer rise)
//!
//! Thickness = margin ramp (continental thick, oceanic thin) + macro-scale
//! craton-structure thickness (thick shield interiors tapering to margins,
//! with intracratonic basins) + tectonic thickening
//! (collision, arcs) - rift thinning. The Airy relation (linear in
//! thickness for uniform densities) converts thickness to base elevation,
//! so plateaus, rift subsidence, and margin profiles all follow from one
//! principle. Thermal subsidence stays separate (young ocean floor is high
//! because it is hot, not thick), and trenches stay separate (held out of
//! isostatic equilibrium by slab pull).
//!
//! Sea level is solved (uniform shift) so land fraction hits LAND_FRACTION.
//!
//! The only surface noise in the simulation is the macro thickness
//! perturbation (folded into `thickness` above, isostatically compensated).
//! Hills/ridge were retired and cosmetic micro texture was removed entirely;
//! erosion supplies real fine relief on the fine mesh.

use glam::Vec3;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use rand::Rng;
#[cfg(not(feature = "single-threaded"))]
use rayon::prelude::*;

use super::constants::*;
use super::crust::{Crust, CrustType};
use super::plates::PLATE_NOISE_OFFSET_RANGE;
use super::{FeatureFields, Tessellation};

/// Terrain elevation data.
pub struct Elevation {
    /// Elevation at each cell.
    pub values: Vec<f32>,

    /// Simulation noise contribution at each cell (macro only; hills/ridge retired).
    pub noise_contribution: Vec<f32>,

    /// Individual noise layer contributions (for visualization).
    pub noise_layers: NoiseLayerData,
}

/// Individual noise layer contributions for visualization (macro only;
/// hills/ridge retired, micro removed — see
/// docs/specs/erosion-v2.md "Noise philosophy").
pub struct NoiseLayerData {
    /// Macro layer (continental tilt).
    pub macro_layer: Vec<f32>,
}

/// Structural inputs to elevation assembly.
///
/// These are deliberately separated from mesh-native `Crust`/`FeatureFields`
/// so the fine mesh can rebuild elevation from transferred physical fields
/// without fabricating coarse-only domain objects.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ElevationFields {
    pub crust_thickness: Vec<f32>,
    pub continentality: Vec<f32>,
    pub ridge_age_distance: Vec<f32>,
    pub trench: Vec<f32>,
    pub ridge: Vec<f32>,
    pub convergent: Vec<f32>,
    pub divergent: Vec<f32>,
    pub is_continental: Vec<bool>,
    /// Volcanic-arc uplift forcing (elevation magnitude). Carried so the fine
    /// mesh can drive erosion uplift from the same feature machinery elevation
    /// uses; already folded into `crust_thickness` for the static surface.
    pub arc: Vec<f32>,
    /// Continental-collision uplift forcing (elevation magnitude). See `arc`.
    pub collision: Vec<f32>,
    /// Signed continental-rift thickness delta (thickness units, not elevation):
    /// negative in the axial valley, positive on the shoulders.
    pub rift_delta: Vec<f32>,
}

/// Per-cell macro crust-thickness perturbation (thickness units), derived from
/// craton structure rather than free position noise. Each continental craton is
/// a thick interior that tapers to its margins (an isostatic dome → elevated
/// shield), with a per-craton base amplitude (older/thicker shields vs thinner
/// cratons) and a decorrelated interior fBm for intracratonic basins and swells.
/// Oceanic cells get no perturbation — their relief is thermal + ridge driven.
/// Added to base thickness during assembly; Airy isostasy turns it into relief.
fn macro_craton_thickness<R: Rng>(
    tessellation: &Tessellation,
    crust: &Crust,
    rng: &mut R,
) -> Vec<f32> {
    let num_cells = tessellation.num_cells();
    let num_cratons = crust.num_cratons.max(1);

    // Per-craton base amplitude and a decorrelated noise-domain offset, so each
    // craton's interior relief is independent (mirrors craton/plate growth).
    let craton_amp: Vec<f32> = (0..num_cratons)
        .map(|_| rng.gen_range(MACRO_CRATON_AMP_MIN..=MACRO_CRATON_AMP_MAX))
        .collect();
    let craton_offset: Vec<Vec3> = (0..num_cratons)
        .map(|_| {
            Vec3::new(
                rng.gen_range(-PLATE_NOISE_OFFSET_RANGE..PLATE_NOISE_OFFSET_RANGE),
                rng.gen_range(-PLATE_NOISE_OFFSET_RANGE..PLATE_NOISE_OFFSET_RANGE),
                rng.gen_range(-PLATE_NOISE_OFFSET_RANGE..PLATE_NOISE_OFFSET_RANGE),
            )
        })
        .collect();
    let interior_fbm = Fbm::<Perlin>::new(rng.gen()).set_octaves(MACRO_INTERIOR_OCTAVES);

    let sample = |i: usize| -> f32 {
        let craton = crust.cell_craton[i];
        if craton == u32::MAX {
            return 0.0; // oceanic: no cratonic structure
        }
        let craton = craton as usize;
        // Distance into the continent (≥0 inland); the dome saturates over
        // MACRO_CRATON_DECAY so margins stay thin and interiors reach full amp.
        let d = crust.signed_margin_distance[i].max(0.0);
        let dome = 1.0 - (-d / MACRO_CRATON_DECAY).exp();
        let p =
            tessellation.cell_center(i) * MACRO_INTERIOR_FREQUENCY as f32 + craton_offset[craton];
        let interior = interior_fbm.get([p.x as f64, p.y as f64, p.z as f64]) as f32;
        let amp = craton_amp[craton] + MACRO_INTERIOR_RELIEF * interior;
        MACRO_THICKNESS_AMPLITUDE * dome * amp
    };

    #[cfg(not(feature = "single-threaded"))]
    {
        (0..num_cells).into_par_iter().map(sample).collect()
    }
    #[cfg(feature = "single-threaded")]
    {
        (0..num_cells).map(sample).collect()
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
        let macro_field = macro_craton_thickness(tessellation, crust, rng);

        let (values, noise_contribution, noise_layers) =
            generate_heightmap(tessellation, crust, features, &macro_field);

        Self {
            values,
            noise_contribution,
            noise_layers,
        }
    }

    /// Build a fine-mesh elevation by refining an already-solved base elevation
    /// (interpolated from the coarse mesh, so it is on the fixed sea-level datum).
    /// There is NO sea-level re-solve: sea level is a global planet datum, chosen
    /// once on the coarse mesh, and inherited here. The simulation values are the
    /// smooth interpolated base (erosion carves the fine detail later); there is
    /// no cosmetic surface noise.
    pub(crate) fn refine_from_base(tessellation: &Tessellation, base_elevation: &[f32]) -> Self {
        let n = tessellation.num_cells();
        Self {
            values: base_elevation.to_vec(),
            noise_contribution: vec![0.0; n],
            noise_layers: NoiseLayerData {
                macro_layer: vec![0.0; n],
            },
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
pub(crate) fn isostatic_elevation(thickness: f32) -> f32 {
    let slope = (CONTINENTAL_BASE - ABYSSAL_DEPTH)
        / (CRUST_THICKNESS_CONTINENTAL - CRUST_THICKNESS_OCEANIC);
    let offset = CONTINENTAL_BASE - slope * CRUST_THICKNESS_CONTINENTAL;
    slope * thickness + offset
}

/// Elevation change per unit crust thickness (the Airy slope), used to
/// express feature forcing magnitudes (calibrated in elevation units) as
/// thickness changes.
pub(crate) fn isostasy_slope() -> f32 {
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
        let convergent = features.convergent[i];

        let cont = continentality(crust.signed_margin_distance[i], convergent);
        let base_thickness = CRUST_THICKNESS_OCEANIC
            + cont * (CRUST_THICKNESS_CONTINENTAL - CRUST_THICKNESS_OCEANIC);
        let thickening = (features.arc[i] + features.collision[i]) / slope;
        let rift = features.rift_delta[i] * cont;
        // Macro craton-thickness is added later, in assembly (it needs craton
        // structure + per-craton RNG), so it stays out of this transferred field.
        let thickness = (base_thickness + thickening + rift).max(0.05);

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
        arc: features.arc.clone(),
        collision: features.collision.clone(),
        rift_delta: features.rift_delta.clone(),
    }
}

/// Generate heightmap: thickness field (incl. macro craton perturbation) ->
/// isostasy -> thermal/dynamic terms -> sea-level solve.
fn generate_heightmap(
    tessellation: &Tessellation,
    crust: &Crust,
    features: &FeatureFields,
    macro_field: &[f32],
) -> (Vec<f32>, Vec<f32>, NoiseLayerData) {
    let fields = coarse_elevation_fields(tessellation, crust, features);
    assemble_heightmap(tessellation, &fields, macro_field)
}

fn assemble_heightmap(
    tessellation: &Tessellation,
    fields: &ElevationFields,
    macro_field: &[f32],
) -> (Vec<f32>, Vec<f32>, NoiseLayerData) {
    let num_cells = tessellation.num_cells();
    let slope = isostasy_slope();

    #[cfg(not(feature = "single-threaded"))]
    let assembled: Vec<AssembledElevationCell> = (0..num_cells)
        .into_par_iter()
        .map(|i| assemble_elevation_cell(fields, macro_field, slope, i))
        .collect();

    #[cfg(feature = "single-threaded")]
    let assembled: Vec<AssembledElevationCell> = (0..num_cells)
        .map(|i| assemble_elevation_cell(fields, macro_field, slope, i))
        .collect();

    let mut elevations = Vec::with_capacity(num_cells);
    let mut noise_contributions = Vec::with_capacity(num_cells);
    let mut macro_layer = Vec::with_capacity(num_cells);

    for cell in assembled {
        elevations.push(cell.elevation);
        noise_contributions.push(cell.noise_contribution);
        macro_layer.push(cell.macro_layer);
    }

    // --- 4. Sea-level solve: uniform shift so the land AREA fraction is exact ---
    // Area-weighted, not cell-count-weighted: the adaptive fine mesh has wildly
    // unequal cell areas (sparse huge ocean cells, dense tiny land cells), so a
    // count-based percentile would place sea level up among the land cells and
    // drown most of the landmass. Sorting by elevation and accumulating area
    // gives the correct sea level on any mesh (and matches the old behaviour on
    // the ~equal-area coarse mesh).
    let areas = tessellation.cell_areas();
    let total_area: f32 = areas.iter().sum();
    let target_submerged_area = (1.0 - LAND_FRACTION) * total_area;
    let mut order: Vec<usize> = (0..num_cells).collect();
    order.sort_by(|&a, &b| elevations[a].total_cmp(&elevations[b]));
    let mut accumulated = 0.0f32;
    let mut sea_level = elevations[*order.last().unwrap()];
    for &i in &order {
        accumulated += areas[i];
        if accumulated >= target_submerged_area {
            sea_level = elevations[i];
            break;
        }
    }
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

    let noise_layers = NoiseLayerData { macro_layer };

    (elevations, noise_contributions, noise_layers)
}

struct AssembledElevationCell {
    elevation: f32,
    noise_contribution: f32,
    macro_layer: f32,
}

fn assemble_elevation_cell(
    fields: &ElevationFields,
    macro_field: &[f32],
    slope: f32,
    i: usize,
) -> AssembledElevationCell {
    // --- 1. Crust thickness ---
    let cont = fields.continentality[i];
    let base_thickness = fields.crust_thickness[i];

    // Macro-scale thickness variation: craton-structure cores and interior basins
    // (precomputed per cell; see `macro_craton_thickness`).
    let macro_dt = macro_field[i];

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

    // --- 3. Macro thickness is reported as its isostatic elevation contribution
    // for the noise viz (it does not enter the simulation elevation directly; it
    // acts through `thickness`).
    let macro_c = macro_dt * slope;

    AssembledElevationCell {
        elevation: structural_elevation,
        noise_contribution: macro_c,
        macro_layer: macro_c,
    }
}
