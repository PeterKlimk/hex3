//! Atmosphere simulation (Stage 2).
//!
//! This module computes atmospheric properties for each cell:
//! - Temperature: based on latitude and elevation lapse rate
//! - Pressure: from temperature (hot = low pressure)
//! - Wind: from pressure gradients, zonal flow, Coriolis, terrain effects
//! - Uplift: from projection solver (for future precipitation)

use glam::Vec3;

use super::constants::*;
use super::{Elevation, Tessellation};

/// Temperature lapse rate: temperature drop per unit elevation.
/// In our normalized system (temp 0-1, elevation 0-0.8 for extreme mountains),
/// LAPSE_RATE = 1.5 means elevation 0.3 drops temp by 0.45.
pub const LAPSE_RATE: f32 = 1.5;

/// Base temperature at equator at sea level (normalized 0-1 scale, where 1 = hottest).
pub const EQUATOR_TEMP: f32 = 1.0;

/// Base temperature at poles at sea level.
pub const POLAR_TEMP: f32 = 0.0;

/// Atmosphere data for the world.
#[derive(Debug, Clone)]
pub struct Atmosphere {
    /// Temperature per cell (normalized 0-1, can go negative at high elevation).
    pub temperature: Vec<f32>,

    /// Atmospheric pressure per cell (1 - temperature: hot = low pressure).
    pub pressure: Vec<f32>,

    /// Wind direction per cell (tangent to sphere surface).
    pub wind: Vec<Vec3>,

    /// Uplift per cell (from correction potential φ, for precipitation).
    pub uplift: Vec<f32>,

    /// Correction potential from projection (for debugging/visualization).
    pub phi: Vec<f32>,
}

impl Atmosphere {
    /// Generate atmosphere data from tessellation and elevation.
    pub fn generate(tessellation: &Tessellation, elevation: &Elevation) -> Self {
        let _num_cells = tessellation.num_cells();

        // Step 1: Temperature from latitude + elevation lapse
        let temperature = generate_temperature(tessellation, &elevation.values);

        // Step 2: Pressure from temperature (hot = low pressure)
        let pressure: Vec<f32> = temperature.iter().map(|&t| 1.0 - t).collect();

        // Steps 2-4: Initial wind (pressure gradient + zonal + Coriolis)
        let mut wind = generate_initial_wind(tessellation, &pressure);

        // Step 5: Terrain effects (uphill blocking + katabatic acceleration)
        apply_terrain_effects(tessellation, elevation, &mut wind);

        // Step 6: Terrain-aware projection (SOR solver)
        let phi = project_wind_field(tessellation, elevation, &mut wind);

        // Step 7: Extract uplift from correction potential
        let uplift = extract_uplift(&phi);

        Self {
            temperature,
            pressure,
            wind,
            uplift,
            phi,
        }
    }

    /// Get temperature for a cell (normalized, can be negative).
    pub fn temperature(&self, cell_idx: usize) -> f32 {
        self.temperature[cell_idx]
    }

    /// Get temperature statistics.
    pub fn stats(&self) -> AtmosphereStats {
        let min_temp = self
            .temperature
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let max_temp = self
            .temperature
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = self.temperature.iter().sum();
        let mean_temp = sum / self.temperature.len() as f32;

        let max_wind = self.wind.iter().map(|w| w.length()).fold(0.0_f32, f32::max);
        let max_uplift = self.uplift.iter().copied().fold(0.0_f32, f32::max);

        AtmosphereStats {
            min_temp,
            max_temp,
            mean_temp,
            max_wind,
            max_uplift,
        }
    }
}

/// Summary statistics for atmosphere data.
#[derive(Debug, Clone, Copy)]
pub struct AtmosphereStats {
    pub min_temp: f32,
    pub max_temp: f32,
    pub mean_temp: f32,
    pub max_wind: f32,
    pub max_uplift: f32,
}

/// Generate temperature field from latitude and elevation.
fn generate_temperature(tessellation: &Tessellation, elevation: &[f32]) -> Vec<f32> {
    let num_cells = tessellation.num_cells();
    let mut temperature = vec![0.0; num_cells];

    for i in 0..num_cells {
        let pos = tessellation.cell_center(i);

        // Latitude: y-coordinate on unit sphere gives sin(latitude)
        let sin_lat = pos.y;
        let lat_factor = sin_lat.abs(); // 0 at equator, 1 at poles

        // Base temperature from latitude (cos²-like distribution)
        let base_temp = EQUATOR_TEMP - (EQUATOR_TEMP - POLAR_TEMP) * lat_factor * lat_factor;

        // Elevation lapse rate (only for positive elevation)
        let elev = elevation[i].max(0.0);
        let lapse = elev * LAPSE_RATE;

        temperature[i] = base_temp - lapse;
    }

    temperature
}

/// Compute pressure gradient at a cell (points toward HIGH pressure).
fn compute_pressure_gradient(
    tessellation: &Tessellation,
    pressure: &[f32],
    cell_idx: usize,
) -> Vec3 {
    let cell_pressure = pressure[cell_idx];
    let cell_pos = tessellation.cell_center(cell_idx);
    let neighbors = tessellation.neighbors(cell_idx);

    if neighbors.is_empty() {
        return Vec3::ZERO;
    }

    let mut gradient = Vec3::ZERO;

    for &n in neighbors {
        let neighbor_pressure = pressure[n];
        let neighbor_pos = tessellation.cell_center(n);

        // Direction from cell to neighbor
        let to_neighbor = neighbor_pos - cell_pos;

        // Project onto tangent plane
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

        // Pressure difference (positive = neighbor is higher pressure)
        let pressure_diff = neighbor_pressure - cell_pressure;
        let slope = pressure_diff / arc_dist;

        gradient += tangent_dir.normalize() * slope;
    }

    gradient
}

/// Get the east-pointing tangent vector at a position on the sphere.
fn tangent_east(pos: Vec3) -> Vec3 {
    // East = north × radial = (0,1,0) × pos (approximately, for non-polar points)
    // More precisely: east points in direction of increasing longitude
    let up = Vec3::Y;
    let east = up.cross(pos);
    let len = east.length();
    if len < 1e-6 {
        // At poles, pick arbitrary tangent
        return Vec3::X;
    }
    east / len
}

/// Compute zonal wind component at a position (trade winds, westerlies, polar easterlies).
fn zonal_wind(pos: Vec3) -> Vec3 {
    let latitude = pos.y; // sin(lat) on unit sphere
    let abs_lat = latitude.abs();

    // Zonal wind patterns (Earth-like):
    // - Trade winds: 0-30° latitude, easterly (blow from east to west)
    // - Westerlies: 30-60° latitude, westerly (blow from west to east)
    // - Polar easterlies: 60-90°, easterly

    // sin(30°) ≈ 0.5, sin(60°) ≈ 0.866
    const LAT_30: f32 = 0.5;
    const LAT_60: f32 = 0.866;

    let (direction, strength) = if abs_lat < LAT_30 {
        // Trade winds: easterly, strongest at equator
        (-1.0, 1.0 - abs_lat / LAT_30 * 0.3)
    } else if abs_lat < LAT_60 {
        // Westerlies: strongest around 45° (sin(45°) ≈ 0.707)
        let t = (abs_lat - LAT_30) / (LAT_60 - LAT_30);
        let westerly_strength = 4.0 * t * (1.0 - t); // peaks at t=0.5
        (1.0, westerly_strength)
    } else {
        // Polar easterlies: weak
        let t = (abs_lat - LAT_60) / (1.0 - LAT_60);
        (-1.0, 0.3 * (1.0 - t))
    };

    let east = tangent_east(pos);
    east * direction * strength * ZONAL_STRENGTH
}

/// Rotate a tangent vector around the surface normal by an angle (for Coriolis).
fn rotate_tangent(vec: Vec3, normal: Vec3, angle: f32) -> Vec3 {
    // Rodrigues' rotation formula
    let (sin_a, cos_a) = angle.sin_cos();
    vec * cos_a + normal.cross(vec) * sin_a + normal * normal.dot(vec) * (1.0 - cos_a)
}

/// Generate initial wind field from pressure gradient, zonal flow, and Coriolis.
fn generate_initial_wind(tessellation: &Tessellation, pressure: &[f32]) -> Vec<Vec3> {
    let num_cells = tessellation.num_cells();
    let mut wind = vec![Vec3::ZERO; num_cells];

    for i in 0..num_cells {
        let pos = tessellation.cell_center(i);

        // Step 2: Wind from pressure gradient
        // Gradient points toward HIGH pressure; wind flows toward LOW
        let pressure_grad = compute_pressure_gradient(tessellation, pressure, i);
        let pressure_wind = -pressure_grad * PRESSURE_WIND_SCALE;

        // Step 3: Zonal base flow
        let zonal = zonal_wind(pos);

        // Blend pressure and zonal winds
        let mut cell_wind = ZONAL_WEIGHT * zonal + PRESSURE_WEIGHT * pressure_wind;

        // Step 4: Coriolis deflection (partial, ~45° for surface wind)
        // Northern hemisphere: deflect right (clockwise from above)
        // Southern hemisphere: deflect left (counterclockwise)
        let latitude = pos.y;
        if cell_wind.length_squared() > 1e-10 {
            let coriolis_angle = SURFACE_CORIOLIS_ANGLE * latitude;
            cell_wind = rotate_tangent(cell_wind, pos, coriolis_angle);
        }

        wind[i] = cell_wind;
    }

    wind
}

/// Apply terrain effects to wind (uphill blocking + katabatic acceleration).
fn apply_terrain_effects(tessellation: &Tessellation, elevation: &Elevation, wind: &mut [Vec3]) {
    let num_cells = tessellation.num_cells();
    let mean_spacing = tessellation.mean_cell_area().sqrt();

    for i in 0..num_cells {
        let elev = elevation.values[i];
        if elev <= 0.0 {
            // Only apply terrain effects on land
            continue;
        }

        let gradient = elevation.gradient(tessellation, i);
        let gradient_mag = gradient.length();
        if gradient_mag < 1e-6 {
            continue;
        }

        let gradient_norm = gradient / gradient_mag;

        // `gradient_mag` is roughly Δelev / Δangle (radians). For terrain effects we want a
        // dimensionless "typical height difference" scale that doesn't blow up as resolution
        // increases, so we scale by mean cell spacing and clamp.
        let slope = (gradient_mag * mean_spacing).min(1.0);
        if slope < 1e-6 {
            continue;
        }

        // Uphill blocking: remove portion of uphill component
        let uphill_component = wind[i].dot(gradient_norm);
        if uphill_component > 0.0 {
            let block_factor = (slope * UPHILL_BLOCKING).min(1.0);
            wind[i] -= gradient_norm * uphill_component * block_factor;
        }

        // Katabatic acceleration: add downhill component
        let katabatic = -gradient_norm * slope * KATABATIC_STRENGTH;
        wind[i] += katabatic;
    }
}

/// Compute edge weight for projection solver (terrain-aware).
/// High terrain = low weight (hard to flow through).
fn edge_weight(elevation: &Elevation, i: usize, j: usize) -> f32 {
    let max_elev = elevation.values[i].max(elevation.values[j]).max(0.0);
    (-max_elev * TERRAIN_RESISTANCE).exp()
}

/// Compute edge length between two adjacent Voronoi cells.
/// This is the length of the great circle arc forming their shared boundary.
fn compute_edge_length(tessellation: &Tessellation, cell_a: usize, cell_b: usize) -> f32 {
    let voronoi = &tessellation.voronoi;
    let verts_a: std::collections::HashSet<usize> = voronoi.cell(cell_a)
        .vertex_indices
        .iter()
        .copied()
        .collect();
    let verts_b: std::collections::HashSet<usize> = voronoi.cell(cell_b)
        .vertex_indices
        .iter()
        .copied()
        .collect();

    // Find shared vertices
    let shared: Vec<usize> = verts_a.intersection(&verts_b).copied().collect();

    if shared.len() == 2 {
        // Edge length = arc distance between the two shared Voronoi vertices
        let v0 = voronoi.vertices[shared[0]];
        let v1 = voronoi.vertices[shared[1]];
        v0.dot(v1).clamp(-1.0, 1.0).acos()
    } else {
        // Fallback: approximate from mean cell spacing
        tessellation.mean_cell_area().sqrt()
    }
}

/// Precompute edge lengths for all neighbor pairs.
/// Returns a vec where edge_lengths[i] contains lengths for each neighbor of cell i.
fn precompute_edge_lengths(tessellation: &Tessellation) -> Vec<Vec<f32>> {
    let num_cells = tessellation.num_cells();
    let mut edge_lengths = Vec::with_capacity(num_cells);

    for i in 0..num_cells {
        let neighbors = tessellation.neighbors(i);
        let lengths: Vec<f32> = neighbors
            .iter()
            .map(|&n| compute_edge_length(tessellation, i, n))
            .collect();
        edge_lengths.push(lengths);
    }

    edge_lengths
}

/// Precompute permeability weights for all edges.
/// High terrain = low permeability (hard to flow through).
fn precompute_permeability(tessellation: &Tessellation, elevation: &Elevation) -> Vec<Vec<f32>> {
    let num_cells = tessellation.num_cells();
    let mut permeability = Vec::with_capacity(num_cells);

    for i in 0..num_cells {
        let neighbors = tessellation.neighbors(i);
        let weights: Vec<f32> = neighbors
            .iter()
            .map(|&n| edge_weight(elevation, i, n))
            .collect();
        permeability.push(weights);
    }

    permeability
}

fn tangent_toward(from: Vec3, to: Vec3) -> Vec3 {
    // Project `to` onto the tangent plane at `from` (direction of the geodesic to `to`).
    let tangent = to - from * from.dot(to);
    let len = tangent.length();
    if len < 1e-6 {
        Vec3::ZERO
    } else {
        tangent / len
    }
}

fn reverse_neighbor_indices(tessellation: &Tessellation) -> Vec<Vec<usize>> {
    let num_cells = tessellation.num_cells();
    let mut reverse = Vec::with_capacity(num_cells);

    for i in 0..num_cells {
        let neighbors = tessellation.neighbors(i);
        let mut indices = Vec::with_capacity(neighbors.len());
        for &j in neighbors {
            let rev = tessellation
                .neighbors(j)
                .iter()
                .position(|&n| n == i)
                .expect("Adjacency must be symmetric");
            indices.push(rev);
        }
        reverse.push(indices);
    }

    reverse
}

/// Project wind field to be (approximately) divergence-free (mass-conserving) on the sphere.
///
/// We treat `wind` as a cell-centered tangential velocity field and enforce zero net flux
/// per cell using a finite-volume pressure projection on the spherical Voronoi mesh:
/// 1. Compute cell flux divergence from symmetric edge-normal velocities.
/// 2. Solve a variable-coefficient Poisson equation for scalar potential `phi`:
///    Σ (k_ij * L_ij / d_ij) (phi_j - phi_i) = divergence_i
/// 3. Apply the induced edge-normal correction on edges, then reconstruct a cell-centered field.
///
/// `k_ij` comes from terrain permeability (mountains impede correction/flow routing).
///
/// Returns the correction potential `phi` (useful as an uplift proxy).
fn project_wind_field(
    tessellation: &Tessellation,
    elevation: &Elevation,
    wind: &mut [Vec3],
) -> Vec<f32> {
    let num_cells = tessellation.num_cells();
    const EPSILON: f32 = 1e-6;

    // Precompute geometric data
    let edge_lengths = precompute_edge_lengths(tessellation);
    let permeability = precompute_permeability(tessellation, elevation);
    let reverse = reverse_neighbor_indices(tessellation);

    // --- STEP 1: Compute edge-normal velocities and per-cell divergence (net flux) ---
    // Edge-normal velocity is estimated symmetrically so it is anti-symmetric across each edge:
    // u_n(i->j) = 0.5 * (u_i·n_ij - u_j·n_ji), with u_n(j->i) = -u_n(i->j).
    let mut edge_u = Vec::with_capacity(num_cells);
    for i in 0..num_cells {
        edge_u.push(vec![0.0_f32; tessellation.neighbors(i).len()]);
    }

    let mut divergence = vec![0.0; num_cells];
    for i in 0..num_cells {
        let pos_i = tessellation.cell_center(i);
        let neighbors = tessellation.neighbors(i);

        for (n_idx, &j) in neighbors.iter().enumerate() {
            if i >= j {
                continue; // process each undirected edge once
            }

            let pos_j = tessellation.cell_center(j);

            let rev_idx = reverse[i][n_idx];
            let edge_len = 0.5 * (edge_lengths[i][n_idx] + edge_lengths[j][rev_idx]);

            let n_ij = tangent_toward(pos_i, pos_j);
            let n_ji = tangent_toward(pos_j, pos_i);
            if n_ij.length_squared() < EPSILON || n_ji.length_squared() < EPSILON {
                continue;
            }

            let u_edge_n = 0.5 * (wind[i].dot(n_ij) - wind[j].dot(n_ji));
            edge_u[i][n_idx] = u_edge_n;
            edge_u[j][rev_idx] = -u_edge_n;
            let flux = u_edge_n * edge_len;

            // Add to i (outward), subtract from j.
            // This guarantees Σ divergence == 0 (up to float error) on a closed surface.
            divergence[i] += flux;
            divergence[j] -= flux;
        }
    }

    // --- STEP 2: Solve for pressure using standard Gauss-Seidel/SOR ---
    // We apply a variable-coefficient Laplacian with conductance:
    // w_ij = (k_ij * L_ij) / d_ij
    // and solve: Σ w_ij (phi_j - phi_i) = divergence_i
    let mut phi = vec![0.0; num_cells];

    for _ in 0..PROJECTION_ITERATIONS {
        for i in 0..num_cells {
            let pos = tessellation.cell_center(i);
            let neighbors = tessellation.neighbors(i);
            if neighbors.is_empty() {
                continue;
            }

            let mut weighted_neighbor_sum = 0.0;
            let mut total_weight = 0.0;

            for (n_idx, &neighbor_id) in neighbors.iter().enumerate() {
                let neighbor_pos = tessellation.cell_center(neighbor_id);
                let dist = pos.dot(neighbor_pos).clamp(-1.0, 1.0).acos().max(EPSILON);

                let rev_idx = reverse[i][n_idx];
                let edge_len = 0.5 * (edge_lengths[i][n_idx] + edge_lengths[neighbor_id][rev_idx]);
                let perm = 0.5 * (permeability[i][n_idx] + permeability[neighbor_id][rev_idx]);

                // Conductance weight: (k_ij * edge_len) / dist
                let weight = (edge_len * perm) / dist;

                weighted_neighbor_sum += weight * phi[neighbor_id];
                total_weight += weight;
            }

            if total_weight < EPSILON {
                continue;
            }

            // From Σw(phi_j - phi_i) = div:
            // phi_i = (Σw*phi_j - div) / Σw
            let gs_value = (weighted_neighbor_sum - divergence[i]) / total_weight;

            // SOR relaxation (omega=1.0 is pure Gauss-Seidel)
            phi[i] = (1.0 - SOR_OMEGA) * phi[i] + SOR_OMEGA * gs_value;
        }

        // Fix gauge: keep zero-mean potential (constant shifts don't affect the correction).
        let mean_phi = phi.iter().sum::<f32>() / num_cells as f32;
        for v in &mut phi {
            *v -= mean_phi;
        }
    }

    // --- STEP 3: Apply correction to edge-normal velocities ---
    for i in 0..num_cells {
        let neighbors = tessellation.neighbors(i);

        for (n_idx, &j) in neighbors.iter().enumerate() {
            if i >= j {
                continue;
            }

            let rev_idx = reverse[i][n_idx];
            let perm = 0.5 * (permeability[i][n_idx] + permeability[j][rev_idx]);
            let pos_i = tessellation.cell_center(i);
            let pos_j = tessellation.cell_center(j);
            let dist = pos_i.dot(pos_j).clamp(-1.0, 1.0).acos().max(EPSILON);

            let delta_phi = phi[j] - phi[i];
            let corr = perm * (delta_phi / dist);

            edge_u[i][n_idx] -= corr;
            edge_u[j][rev_idx] += corr;
        }
    }

    // --- STEP 4: Reconstruct a cell-centered tangent field from the corrected edge normals ---
    for i in 0..num_cells {
        let pos_i = tessellation.cell_center(i);
        let helper = if pos_i.y.abs() < 0.9 {
            Vec3::Y
        } else {
            Vec3::X
        };
        let t1 = helper.cross(pos_i).normalize();
        let t2 = pos_i.cross(t1).normalize();

        let neighbors = tessellation.neighbors(i);
        let mut m00 = 0.0_f32;
        let mut m01 = 0.0_f32;
        let mut m11 = 0.0_f32;
        let mut b0 = 0.0_f32;
        let mut b1 = 0.0_f32;

        for (n_idx, &j) in neighbors.iter().enumerate() {
            let pos_j = tessellation.cell_center(j);
            let dir = tangent_toward(pos_i, pos_j);
            if dir.length_squared() < EPSILON {
                continue;
            }

            let rev_idx = reverse[i][n_idx];
            let edge_len = 0.5 * (edge_lengths[i][n_idx] + edge_lengths[j][rev_idx]);
            let weight = edge_len.max(EPSILON);

            let c1 = dir.dot(t1);
            let c2 = dir.dot(t2);
            let target = edge_u[i][n_idx];

            m00 += weight * c1 * c1;
            m01 += weight * c1 * c2;
            m11 += weight * c2 * c2;
            b0 += weight * c1 * target;
            b1 += weight * c2 * target;
        }

        let det = m00 * m11 - m01 * m01;
        if det.abs() < EPSILON {
            wind[i] = Vec3::ZERO;
            continue;
        }

        let a = (b0 * m11 - b1 * m01) / det;
        let b = (b1 * m00 - b0 * m01) / det;
        wind[i] = t1 * a + t2 * b;
    }

    phi
}

/// Extract uplift from correction potential phi.
/// High phi = air backing up = uplift.
fn extract_uplift(phi: &[f32]) -> Vec<f32> {
    let phi_max = phi.iter().copied().fold(0.0_f32, f32::max);
    let phi_min = phi.iter().copied().fold(0.0_f32, f32::min);
    let phi_range = (phi_max - phi_min).max(0.001);

    phi.iter()
        .map(|&p| ((p - phi_min) / phi_range).max(0.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::world::NoiseLayerData;

    fn divergence_rms(tessellation: &Tessellation, wind: &[Vec3]) -> f32 {
        let num_cells = tessellation.num_cells();
        let edge_lengths = precompute_edge_lengths(tessellation);
        let reverse = reverse_neighbor_indices(tessellation);

        let mut divergence = vec![0.0f32; num_cells];
        for i in 0..num_cells {
            let pos_i = tessellation.cell_center(i);
            let neighbors = tessellation.neighbors(i);
            for (n_idx, &j) in neighbors.iter().enumerate() {
                if i >= j {
                    continue;
                }

                let pos_j = tessellation.cell_center(j);
                let rev_idx = reverse[i][n_idx];
                let edge_len = 0.5 * (edge_lengths[i][n_idx] + edge_lengths[j][rev_idx]);

                let n_ij = tangent_toward(pos_i, pos_j);
                let n_ji = tangent_toward(pos_j, pos_i);
                if n_ij.length_squared() < 1e-6 || n_ji.length_squared() < 1e-6 {
                    continue;
                }

                let u_edge_n = 0.5 * (wind[i].dot(n_ij) - wind[j].dot(n_ji));
                let flux = u_edge_n * edge_len;
                divergence[i] += flux;
                divergence[j] -= flux;
            }
        }

        let mean_sq = divergence.iter().map(|&d| d * d).sum::<f32>() / num_cells as f32;
        mean_sq.sqrt()
    }

    #[test]
    fn wind_projection_reduces_divergence() {
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let num_cells = 800;
        let tess = Tessellation::generate(num_cells, 0, &mut rng);

        let elevation = Elevation {
            values: vec![0.0; num_cells],
            noise_contribution: vec![0.0; num_cells],
            noise_layers: NoiseLayerData {
                macro_layer: vec![0.0; num_cells],
                hills_layer: vec![0.0; num_cells],
                ridge_layer: vec![0.0; num_cells],
                micro_layer: vec![0.0; num_cells],
            },
        };

        // Construct a clearly divergent field: mostly meridional flow.
        let mut wind = (0..num_cells)
            .map(|i| {
                let pos = tess.cell_center(i);
                let north = Vec3::Y;
                let tangent = north - pos * pos.dot(north);
                if tangent.length_squared() < 1e-6 {
                    Vec3::ZERO
                } else {
                    tangent.normalize() * 0.5
                }
            })
            .collect::<Vec<_>>();

        // Add a small random perturbation so the solver exercises the full stencil.
        for v in &mut wind {
            let jitter = Vec3::new(
                rng.gen_range(-0.05..0.05),
                rng.gen_range(-0.05..0.05),
                rng.gen_range(-0.05..0.05),
            );
            *v = (*v + jitter) * 0.9;
        }

        let before = divergence_rms(&tess, &wind);

        let _phi = project_wind_field(&tess, &elevation, &mut wind);

        let after = divergence_rms(&tess, &wind);

        assert!(
            after < before * 0.5,
            "projection should significantly reduce divergence (before={before:.4}, after={after:.4})"
        );
    }
}
