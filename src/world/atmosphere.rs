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

        let max_wind = self
            .wind
            .iter()
            .map(|w| w.length())
            .fold(0.0_f32, f32::max);
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
fn compute_pressure_gradient(tessellation: &Tessellation, pressure: &[f32], cell_idx: usize) -> Vec3 {
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
            let coriolis_angle = SURFACE_CORIOLIS_ANGLE * latitude.signum();
            cell_wind = rotate_tangent(cell_wind, pos, coriolis_angle);
        }

        wind[i] = cell_wind;
    }

    wind
}

/// Apply terrain effects to wind (uphill blocking + katabatic acceleration).
fn apply_terrain_effects(tessellation: &Tessellation, elevation: &Elevation, wind: &mut [Vec3]) {
    let num_cells = tessellation.num_cells();

    for i in 0..num_cells {
        let elev = elevation.values[i];
        if elev <= 0.0 {
            // Only apply terrain effects on land
            continue;
        }

        let gradient = elevation.gradient(tessellation, i);
        let slope = gradient.length();
        if slope < 1e-6 {
            continue;
        }

        let gradient_norm = gradient / slope;

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

/// Compute divergence of wind field at a cell.
fn compute_divergence(tessellation: &Tessellation, wind: &[Vec3], cell_idx: usize) -> f32 {
    let pos = tessellation.cell_center(cell_idx);
    let neighbors = tessellation.neighbors(cell_idx);
    let mut div = 0.0;

    for &n in neighbors {
        let neighbor_pos = tessellation.cell_center(n);
        let edge_dir = (neighbor_pos - pos).normalize();

        // Outflow from cell
        let outflow = wind[cell_idx].dot(edge_dir);
        // Inflow from neighbor
        let inflow = wind[n].dot(-edge_dir);

        div += outflow - inflow;
    }

    div
}

/// Compute gradient of phi (correction potential) at a cell.
/// Uses same approach as pressure gradient for consistency.
fn compute_phi_gradient(
    tessellation: &Tessellation,
    phi: &[f32],
    cell_idx: usize,
) -> Vec3 {
    let cell_phi = phi[cell_idx];
    let cell_pos = tessellation.cell_center(cell_idx);
    let neighbors = tessellation.neighbors(cell_idx);

    if neighbors.is_empty() {
        return Vec3::ZERO;
    }

    let mut gradient = Vec3::ZERO;

    for &n in neighbors {
        let neighbor_phi = phi[n];
        let neighbor_pos = tessellation.cell_center(n);

        let to_neighbor = neighbor_pos - cell_pos;
        let tangent_dir = to_neighbor - cell_pos * cell_pos.dot(to_neighbor);
        let tangent_len = tangent_dir.length();
        if tangent_len < 1e-6 {
            continue;
        }

        let arc_dist = cell_pos.dot(neighbor_pos).clamp(-1.0, 1.0).acos();
        if arc_dist < 1e-6 {
            continue;
        }

        let phi_diff = neighbor_phi - cell_phi;
        let slope = phi_diff / arc_dist;

        gradient += tangent_dir.normalize() * slope;
    }

    gradient
}

/// Compute divergence statistics for logging.
fn divergence_stats(tessellation: &Tessellation, wind: &[Vec3]) -> (f32, f32, f32) {
    let num_cells = tessellation.num_cells();
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut max_abs = 0.0f32;

    for i in 0..num_cells {
        let div = compute_divergence(tessellation, wind, i);
        sum += div.abs();
        sum_sq += div * div;
        max_abs = max_abs.max(div.abs());
    }

    let mean = sum / num_cells as f32;
    let rms = (sum_sq / num_cells as f32).sqrt();
    (mean, rms, max_abs)
}

/// Project wind field to be divergence-free using SOR solver.
/// Returns the correction potential phi.
fn project_wind_field(
    tessellation: &Tessellation,
    elevation: &Elevation,
    wind: &mut [Vec3],
) -> Vec<f32> {
    let num_cells = tessellation.num_cells();

    // Log divergence before projection
    let (_, rms_before, _) = divergence_stats(tessellation, wind);

    // Iterative projection: solve Poisson and apply correction multiple times
    let correction_scale = 0.0015;
    let outer_iterations = 5;
    let mut final_phi = vec![0.0; num_cells];

    for _ in 0..outer_iterations {
        // Compute current divergence
        let divergence: Vec<f32> = (0..num_cells)
            .map(|i| compute_divergence(tessellation, wind, i))
            .collect();

        // Subtract mean to ensure solvability on closed surface
        let mean_div = divergence.iter().sum::<f32>() / num_cells as f32;
        let divergence: Vec<f32> = divergence.iter().map(|&d| d - mean_div).collect();

        // Solve Poisson equation with SOR
        final_phi.fill(0.0);
        for _ in 0..PROJECTION_ITERATIONS {
            for i in 0..num_cells {
                let neighbors = tessellation.neighbors(i);
                if neighbors.is_empty() {
                    continue;
                }

                let mut weighted_sum = 0.0;
                let mut weight_total = 0.0;

                for &n in neighbors {
                    let w = edge_weight(elevation, i, n);
                    weighted_sum += w * final_phi[n];
                    weight_total += w;
                }

                if weight_total > 0.0 {
                    let gs_value = (weighted_sum - divergence[i]) / weight_total;
                    final_phi[i] = (1.0 - SOR_OMEGA) * final_phi[i] + SOR_OMEGA * gs_value;
                }
            }
        }

        // Apply correction
        for i in 0..num_cells {
            let phi_gradient = compute_phi_gradient(tessellation, &final_phi, i);
            wind[i] -= phi_gradient * correction_scale;
        }
    }

    // Log divergence after projection
    let (_, rms_after, _) = divergence_stats(tessellation, wind);
    log::info!(
        "Wind projection: divergence rms {:.3} -> {:.3} ({:.1}x reduction)",
        rms_before, rms_after, rms_before / rms_after.max(1e-10)
    );

    final_phi
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
