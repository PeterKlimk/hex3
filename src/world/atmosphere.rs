//! Atmosphere simulation (Stage 2).
//!
//! This module computes atmospheric properties for each cell:
//! - Surface temperature: based on latitude and elevation lapse rate
//! - Pressure forcing: from upper-layer temperature (latitudinal only; no lapse rate)
//! - Wind: from pressure gradients, zonal flow, Coriolis-like geostrophic balance, terrain effects
//! - Uplift: proxy from convergence (pre-projection) + orographic upslope flow

use glam::Vec3;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::circulation::Circulation;
use super::constants::*;
use super::moisture::simulate_moisture;
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
    /// Surface temperature per cell (normalized 0-1, can go negative at high elevation).
    pub temperature: Vec<f32>,

    /// Land-ocean thermal contrast factor: 0 at ocean cells, saturating inland.
    pub continentality: Vec<f32>,

    /// Upper-layer temperature used to derive pressure forcing (latitude-only).
    pub upper_temperature: Vec<f32>,

    /// Pressure forcing per cell (1 - upper_temperature: hot = low pressure).
    pub pressure: Vec<f32>,

    /// Upper wind per cell (terrain-unaware, free-flowing atmospheric wind).
    pub upper_wind: Vec<Vec3>,

    /// Surface wind per cell (terrain-influenced, tangent to sphere surface).
    pub wind: Vec<Vec3>,

    /// Uplift per cell (proxy from convergence + orographic upslope flow).
    pub uplift: Vec<f32>,

    /// Steady-state airborne moisture per cell.
    pub moisture: Vec<f32>,

    /// Precipitation per cell, normalized to mean 1.0 over the sphere.
    /// Drives hydrology flow accumulation and lake equilibria.
    pub precipitation: Vec<f32>,

    /// Correction potential from projection (for debugging/visualization).
    pub phi: Vec<f32>,
}

impl Atmosphere {
    /// Generate atmosphere data from tessellation and elevation.
    pub fn generate(tessellation: &Tessellation, elevation: &Elevation) -> Self {
        // Step 1: Surface temperature from latitude + elevation lapse (display/climate field)
        let (temperature, continentality) =
            generate_surface_temperature(tessellation, &elevation.values);

        // Step 2: Upper-layer temperature (latitude-only; avoids "mountain high pressure" artifacts)
        let upper_temperature = generate_upper_temperature(tessellation);

        // Step 3: Pressure forcing from upper temperature (hot = low pressure)
        let pressure: Vec<f32> = upper_temperature.iter().map(|&t| 1.0 - t).collect();

        let circulation = Circulation::default_planet();

        // Step 4: Initial wind (pressure gradient + mean overturning circulation)
        let initial_wind = generate_initial_wind(tessellation, &pressure, &circulation);

        // Step 5: Upper wind - visualization-quality return flow from the circulation.
        let upper_wind = generate_upper_wind(tessellation, &circulation);

        // Step 6: Surface wind - apply terrain effects to the surface branch.
        let mut wind = initial_wind;
        apply_terrain_effects(tessellation, elevation, &mut wind);

        // Step 7: Terrain-aware projection for surface wind (SOR solver)
        let phi = project_wind_field(tessellation, Some(elevation), &mut wind);

        // Step 8: Uplift proxy from convergence + orographic upslope flow.
        // The convergence part comes from the projection potential phi: the
        // solver inverted div = laplacian(phi), so phi is the elliptically
        // integrated (large-scale, mesh-noise-free) divergence field, with
        // convergence centers at phi maxima.
        let uplift = compute_uplift(tessellation, elevation, &phi, &wind, &circulation);

        // Step 9: Moisture transport and precipitation
        let moisture_result = simulate_moisture(
            tessellation,
            &elevation.values,
            &temperature,
            &wind,
            &uplift,
        );

        log::debug!(
            "field smoothness (Moran's I): uplift={:.3}, precipitation={:.3}",
            tessellation.morans_i(&uplift),
            tessellation.morans_i(&moisture_result.precipitation),
        );

        Self {
            temperature,
            continentality,
            upper_temperature,
            pressure,
            upper_wind,
            wind,
            uplift,
            moisture: moisture_result.moisture,
            precipitation: moisture_result.precipitation,
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

        let max_upper_wind = self
            .upper_wind
            .iter()
            .map(|w| w.length())
            .fold(0.0_f32, f32::max);
        let max_wind = self.wind.iter().map(|w| w.length()).fold(0.0_f32, f32::max);
        let max_uplift = self.uplift.iter().map(|u| u.abs()).fold(0.0_f32, f32::max);

        AtmosphereStats {
            min_temp,
            max_temp,
            mean_temp,
            max_upper_wind,
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
    pub max_upper_wind: f32,
    pub max_wind: f32,
    pub max_uplift: f32,
}

/// Generate temperature field from latitude, land-ocean contrast, and elevation.
fn generate_surface_temperature(
    tessellation: &Tessellation,
    elevation: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let num_cells = tessellation.num_cells();
    let mut temperature = vec![0.0; num_cells];
    let continentality = compute_continentality(tessellation, elevation);
    let reference_temp = latitude_reference_temperature(tessellation);

    for i in 0..num_cells {
        let pos = tessellation.cell_center(i);

        // Latitude: y-coordinate on unit sphere gives sin(latitude)
        let sin_lat = pos.y;
        let lat_factor = sin_lat.abs(); // 0 at equator, 1 at poles

        // Base temperature from latitude (cos²-like distribution)
        let base_temp = EQUATOR_TEMP - (EQUATOR_TEMP - POLAR_TEMP) * lat_factor * lat_factor;
        let deviation = base_temp - reference_temp;
        let contrast = 1.0 + CONTINENTALITY_AMP * continentality[i];
        let contrasted_base = reference_temp + deviation * contrast;

        // Elevation lapse rate (only for positive elevation)
        let elev = elevation[i].max(0.0);
        let lapse = elev * LAPSE_RATE;

        temperature[i] = contrasted_base - lapse;
    }

    // Land-driven pressure anomalies are intentionally not fed into
    // upper_temperature/pressure yet; keeping the wind solve latitude-driven
    // isolates this cheap thermal-contrast mechanism.
    (temperature, continentality)
}

fn latitude_reference_temperature(tessellation: &Tessellation) -> f32 {
    let num_cells = tessellation.num_cells();
    if num_cells == 0 {
        return 0.0;
    }

    let sum: f32 = (0..num_cells)
        .map(|i| {
            let lat_factor = tessellation.cell_center(i).y.abs();
            EQUATOR_TEMP - (EQUATOR_TEMP - POLAR_TEMP) * lat_factor * lat_factor
        })
        .sum();
    sum / num_cells as f32
}

fn compute_continentality(tessellation: &Tessellation, elevation: &[f32]) -> Vec<f32> {
    let ocean_distance = distance_from_ocean_cells(tessellation, elevation);
    ocean_distance
        .iter()
        .map(|&d| continentality_from_ocean_distance(d, CONTINENTALITY_DISTANCE_SCALE))
        .collect()
}

fn continentality_from_ocean_distance(dist: f32, scale: f32) -> f32 {
    if !dist.is_finite() {
        return 1.0;
    }
    if dist <= 0.0 {
        return 0.0;
    }
    let scale = scale.max(1e-6);
    (1.0 - (-(dist / scale)).exp()).clamp(0.0, 1.0)
}

fn distance_from_ocean_cells(tessellation: &Tessellation, elevation: &[f32]) -> Vec<f32> {
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
    let mut heap = BinaryHeap::new();

    for (cell, &elev) in elevation.iter().enumerate().take(n) {
        if elev < 0.0 {
            dist[cell] = 0.0;
            heap.push(State { dist: 0.0, cell });
        }
    }

    while let Some(State { dist: d, cell }) = heap.pop() {
        if d > dist[cell] {
            continue;
        }

        let pos = tessellation.cell_center(cell);
        for &neighbor in tessellation.neighbors(cell) {
            let neighbor_pos = tessellation.cell_center(neighbor);
            let step = pos.dot(neighbor_pos).clamp(-1.0, 1.0).acos();
            let nd = d + step;
            if nd < dist[neighbor] {
                dist[neighbor] = nd;
                heap.push(State {
                    dist: nd,
                    cell: neighbor,
                });
            }
        }
    }

    dist
}

/// Generate an upper-layer temperature field from latitude only.
fn generate_upper_temperature(tessellation: &Tessellation) -> Vec<f32> {
    let num_cells = tessellation.num_cells();
    let mut temperature = vec![0.0; num_cells];

    for i in 0..num_cells {
        let pos = tessellation.cell_center(i);

        // Latitude: y-coordinate on unit sphere gives sin(latitude)
        let sin_lat = pos.y;
        let lat_factor = sin_lat.abs(); // 0 at equator, 1 at poles

        // Base temperature from latitude (cos²-like distribution)
        temperature[i] = EQUATOR_TEMP - (EQUATOR_TEMP - POLAR_TEMP) * lat_factor * lat_factor;
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

/// Generate initial wind field from pressure gradient and overturning circulation.
fn generate_initial_wind(
    tessellation: &Tessellation,
    pressure: &[f32],
    circulation: &Circulation,
) -> Vec<Vec3> {
    let num_cells = tessellation.num_cells();
    let mut wind = vec![Vec3::ZERO; num_cells];

    for i in 0..num_cells {
        let pos = tessellation.cell_center(i);

        // Pressure gradient points toward HIGH pressure; pressure forcing points toward LOW.
        let pressure_grad = compute_pressure_gradient(tessellation, pressure, i);
        let to_low = -pressure_grad;

        // Geostrophic-like component: flow parallel to isobars.
        // We blend between down-gradient flow (equator) and geostrophic balance (poles),
        // since the geostrophic approximation breaks down as f -> 0 near the equator.
        let abs_lat = pos.y.abs().clamp(0.0, 1.0);
        // Geostrophic flow direction flips with the sign of the Coriolis
        // parameter: u ~ (k x grad p) / f, and f < 0 in the southern
        // hemisphere. Without the flip, SH mid-latitude flow around the
        // equatorial thermal low runs easterly and cancels the circulation's
        // westerlies. (The mix already fades this to zero at the equator.)
        let hemisphere = if pos.y < 0.0 { -1.0 } else { 1.0 };
        let geostrophic_dir = -pos.cross(pressure_grad) * hemisphere;
        let geostrophic_mix = (abs_lat * GEOSTROPHIC_BALANCE).clamp(0.0, 1.0);
        let pressure_wind = (1.0 - geostrophic_mix) * to_low + geostrophic_mix * geostrophic_dir;
        let pressure_wind = pressure_wind * PRESSURE_WIND_SCALE;

        let circulation_wind = circulation.surface_wind_at(pos);

        let cell_wind = circulation_wind + PRESSURE_WEIGHT * pressure_wind;

        wind[i] = cell_wind;
    }

    wind
}

fn generate_upper_wind(tessellation: &Tessellation, circulation: &Circulation) -> Vec<Vec3> {
    (0..tessellation.num_cells())
        .map(|i| circulation.upper_wind_at(tessellation.cell_center(i)))
        .collect()
}

/// Apply terrain effects to wind (uphill blocking + katabatic acceleration).
fn apply_terrain_effects(tessellation: &Tessellation, elevation: &Elevation, wind: &mut [Vec3]) {
    let num_cells = tessellation.num_cells();

    // Debug: track modifications
    let mut cells_modified = 0;
    let mut total_blocking = 0.0_f32;
    let mut total_katabatic = 0.0_f32;
    let mut max_slope_seen = 0.0_f32;

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

        // gradient_mag is the true terrain gradient dz/ds = tan(θ) for slope angle θ.
        let gradient = gradient_mag;
        max_slope_seen = max_slope_seen.max(gradient);

        // Uphill blocking: remove portion of uphill component
        // Use gradient directly - blocking saturates via the min(1.0) clamp
        let uphill_component = wind[i].dot(gradient_norm);
        if uphill_component > 0.0 {
            let block_factor = (gradient * UPHILL_BLOCKING).min(1.0);
            let blocked = gradient_norm * uphill_component * block_factor;
            wind[i] -= blocked;
            total_blocking += blocked.length();
            cells_modified += 1;
        }

        // Katabatic acceleration: gravity component parallel to slope is g*sin(θ)
        // sin(atan(gradient)) = gradient / sqrt(1 + gradient²), bounded to [0, 1]
        let sin_slope = gradient / (1.0 + gradient * gradient).sqrt();
        let katabatic = -gradient_norm * sin_slope * KATABATIC_STRENGTH;
        wind[i] += katabatic;
        total_katabatic += katabatic.length();
    }

    log::debug!(
        "terrain_effects: cells_modified={}, avg_blocking={:.4}, avg_katabatic={:.4}, max_slope={:.2}",
        cells_modified,
        if cells_modified > 0 { total_blocking / cells_modified as f32 } else { 0.0 },
        total_katabatic / num_cells as f32,
        max_slope_seen
    );
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
            .map(|&n| tessellation.shared_edge_length(i, n))
            .collect();
        edge_lengths.push(lengths);
    }

    edge_lengths
}

/// Precompute permeability weights for all edges.
/// Steep terrain transitions = low permeability (hard to flow through).
/// Uses cosine power law: perm = cos^p(atan(gradient)) = 1/(1+g²)^(p/2)
fn precompute_permeability(tessellation: &Tessellation, elevation: &Elevation) -> Vec<Vec<f32>> {
    let num_cells = tessellation.num_cells();
    let mut permeability = Vec::with_capacity(num_cells);

    // Debug: track edge slope statistics
    let mut all_slopes = Vec::new();
    let mut all_deltas = Vec::new();

    let half_power = PERMEABILITY_POWER / 2.0;

    for i in 0..num_cells {
        let pos_i = tessellation.cell_center(i);
        let neighbors = tessellation.neighbors(i);
        let elev_i = elevation.values[i].max(0.0);

        let weights: Vec<f32> = neighbors
            .iter()
            .map(|&j| {
                let elev_j = elevation.values[j].max(0.0);
                let delta = (elev_i - elev_j).abs();

                let pos_j = tessellation.cell_center(j);
                let dist = pos_i.dot(pos_j).clamp(-1.0, 1.0).acos().max(1e-6);

                // True gradient: elevation change per unit arc length (dz/ds) = tan(θ)
                let gradient = delta / dist;

                // Debug: collect slope statistics
                if delta > 0.0 {
                    all_slopes.push(gradient);
                    all_deltas.push(delta);
                }

                // Cosine power law: perm = cos^p(atan(g)) = 1/(1+g²)^(p/2)
                // For p=2: perm = 1/(1+g²), giving 0.5 at 45° slope
                1.0 / (1.0 + gradient * gradient).powf(half_power)
            })
            .collect();
        permeability.push(weights);
    }

    // Debug: print slope statistics
    if !all_slopes.is_empty() && log::log_enabled!(log::Level::Debug) {
        all_slopes.sort_by(|a, b| a.total_cmp(b));
        all_deltas.sort_by(|a, b| a.total_cmp(b));
        let p50 = all_slopes[all_slopes.len() / 2];
        let p90 = all_slopes[all_slopes.len() * 9 / 10];
        let p99 = all_slopes[all_slopes.len() * 99 / 100];
        let max_slope = all_slopes.last().copied().unwrap_or(0.0);
        let max_delta = all_deltas.last().copied().unwrap_or(0.0);

        // Cosine power permeability: 1/(1+g²)^(p/2)
        let perm = |g: f32| 1.0 / (1.0 + g * g).powf(half_power);

        log::debug!(
            "gradient: p50={:.4}, p90={:.4}, p99={:.4}, max={:.4}, max_delta={:.4}",
            p50,
            p90,
            p99,
            max_slope,
            max_delta
        );
        log::debug!(
            "gradient: perm at p50={:.4}, p90={:.4}, p99={:.4}, max={:.4}",
            perm(p50),
            perm(p90),
            perm(p99),
            perm(max_slope)
        );
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
/// When `elevation` is provided, `k_ij` comes from terrain permeability (mountains impede
/// correction/flow routing). When `None`, uniform permeability is used (for upper wind).
///
/// Returns the correction potential `phi` (useful as an uplift proxy).
fn project_wind_field(
    tessellation: &Tessellation,
    elevation: Option<&Elevation>,
    wind: &mut [Vec3],
) -> Vec<f32> {
    let num_cells = tessellation.num_cells();
    const EPSILON: f32 = 1e-6;

    // Precompute geometric data
    let edge_lengths = precompute_edge_lengths(tessellation);
    let permeability = if let Some(elev) = elevation {
        let perm = precompute_permeability(tessellation, elev);
        if log::log_enabled!(log::Level::Debug) {
            let all_perms: Vec<f32> = perm.iter().flat_map(|v| v.iter().copied()).collect();
            let min_perm = all_perms.iter().copied().fold(f32::INFINITY, f32::min);
            let max_perm = all_perms.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mean_perm = all_perms.iter().sum::<f32>() / all_perms.len() as f32;
            let low_count = all_perms.iter().filter(|&&p| p < 0.5).count();
            let very_low_count = all_perms.iter().filter(|&&p| p < 0.1).count();
            log::debug!(
                "permeability: min={:.4}, max={:.4}, mean={:.4}, <0.5: {}/{} ({:.1}%), <0.1: {}",
                min_perm,
                max_perm,
                mean_perm,
                low_count,
                all_perms.len(),
                100.0 * low_count as f32 / all_perms.len() as f32,
                very_low_count
            );
        }
        perm
    } else {
        // Uniform permeability (1.0) for upper wind - no terrain effects
        (0..num_cells)
            .map(|i| vec![1.0_f32; tessellation.neighbors(i).len()])
            .collect()
    };
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

fn normalize_positive_field(mut values: Vec<f32>, percentile: f32) -> Vec<f32> {
    let mut samples: Vec<f32> = values
        .iter()
        .copied()
        .filter(|&v| v.is_finite() && v > 0.0)
        .collect();
    if samples.is_empty() {
        values.fill(0.0);
        return values;
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p = percentile.clamp(0.5, 0.999);
    let idx = ((samples.len() - 1) as f32 * p).round() as usize;
    let scale = samples[idx].max(1e-6);

    for v in &mut values {
        if !v.is_finite() || *v <= 0.0 {
            *v = 0.0;
        } else {
            *v = (*v / scale).clamp(0.0, 1.0);
        }
    }

    values
}

fn normalize_signed_field(mut values: Vec<f32>, percentile: f32) -> Vec<f32> {
    let mut samples: Vec<f32> = values
        .iter()
        .copied()
        .filter(|&v| v.is_finite() && v != 0.0)
        .map(f32::abs)
        .collect();
    if samples.is_empty() {
        values.fill(0.0);
        return values;
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p = percentile.clamp(0.5, 0.999);
    let idx = ((samples.len() - 1) as f32 * p).round() as usize;
    let scale = samples[idx].max(1e-6);

    for v in &mut values {
        if !v.is_finite() {
            *v = 0.0;
        } else {
            *v = (*v / scale).clamp(-1.0, 1.0);
        }
    }

    values
}

fn compute_uplift(
    tessellation: &Tessellation,
    elevation: &Elevation,
    phi: &[f32],
    wind_final: &[Vec3],
    circulation: &Circulation,
) -> Vec<f32> {
    let num_cells = tessellation.num_cells();
    let mean_spacing = tessellation.mean_cell_area().sqrt();

    // Convergence proxy from the projection potential: phi solves
    // div = laplacian(phi), so it is the smooth integral of the divergence
    // the projection removed. Convergence (negative divergence) shows up as
    // phi maxima; the positive part above the mean is the convergence field.
    let phi_mean: f32 = phi.iter().sum::<f32>() / num_cells.max(1) as f32;
    let convergence: Vec<f32> = phi.iter().map(|&p| (p - phi_mean).max(0.0)).collect();

    // Orographic uplift proxy from upslope flow (terrain-following kinematics).
    let mut orographic = vec![0.0_f32; num_cells];
    for i in 0..num_cells {
        if elevation.values[i] <= 0.0 {
            continue;
        }
        let grad = elevation.gradient(tessellation, i);
        if grad.length_squared() < 1e-10 {
            continue;
        }
        // `grad` is roughly Δelev / Δangle; multiply by a typical angular scale for stability.
        // Weight by elevation so high terrain wrings air harder than the
        // coastal shelf step (otherwise uplift is a ring hugging every
        // coastline and "orographic" rain is mostly coastal rain).
        let height_factor = (elevation.values[i] / OROGRAPHIC_FULL_HEIGHT).clamp(0.0, 1.0);
        let w = wind_final[i].dot(grad) * mean_spacing * height_factor;
        orographic[i] = w.max(0.0);
    }

    // Convergence and orographic components have arbitrary relative units;
    // normalize each before weighting so the weights mean what they say.
    let convergence = normalize_positive_field(convergence, UPLIFT_NORM_PERCENTILE);
    let orographic = normalize_positive_field(orographic, UPLIFT_NORM_PERCENTILE);
    let circulation_vertical = normalize_signed_field(
        (0..num_cells)
            .map(|i| circulation.vertical_motion_at(tessellation.cell_center(i)))
            .collect(),
        UPLIFT_NORM_PERCENTILE,
    );

    let mut uplift = vec![0.0_f32; num_cells];
    for i in 0..num_cells {
        uplift[i] = UPLIFT_CONVERGENCE_WEIGHT * convergence[i]
            + UPLIFT_OROGRAPHIC_WEIGHT * orographic[i]
            + UPLIFT_CIRCULATION_WEIGHT * circulation_vertical[i];
    }

    uplift
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

        let _phi = project_wind_field(&tess, Some(&elevation), &mut wind);

        let after = divergence_rms(&tess, &wind);

        assert!(
            after < before * 0.5,
            "projection should significantly reduce divergence (before={before:.4}, after={after:.4})"
        );
    }

    #[test]
    fn continentality_is_zero_at_ocean_and_saturates_inland() {
        assert_eq!(
            continentality_from_ocean_distance(0.0, CONTINENTALITY_DISTANCE_SCALE),
            0.0
        );
        assert!(
            continentality_from_ocean_distance(
                10.0 * CONTINENTALITY_DISTANCE_SCALE,
                CONTINENTALITY_DISTANCE_SCALE,
            ) > 0.99
        );
    }
}
