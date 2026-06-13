//! Moisture transport and precipitation.
//!
//! A single-layer moisture field advected over the wind field: water cells
//! evaporate moisture toward a temperature-dependent carrying capacity, wind
//! carries it downwind, and rain falls where moisture meets uplift (orographic
//! + convergence) plus a baseline rainout so air dries over long fetches.
//!
//! Emergent behavior: rain shadows behind mountains, continental interiors
//! drying with distance from coast, wet ascent bands, and dry subsidence belts.
//! All rates are worldbuilding knobs in `constants.rs`.

use glam::Vec3;

use super::constants::*;
use super::Tessellation;

/// Result of the moisture simulation.
pub struct MoistureResult {
    /// Steady-state airborne moisture per cell (raw units).
    pub moisture: Vec<f32>,
    /// Precipitation rate per cell, normalized to mean 1.0 over the sphere
    /// (so hydrology flow units stay "average-cell equivalents").
    pub precipitation: Vec<f32>,
}

/// Temperature-dependent moisture carrying capacity (warm air holds more).
fn carrying_capacity(temperature: f32) -> f32 {
    let t = temperature.clamp(0.0, 1.0);
    MOISTURE_CAP_COLD + t * (MOISTURE_CAP_WARM - MOISTURE_CAP_COLD)
}

fn static_rain_rate(uplift: f32) -> f32 {
    (RAINOUT_BASE + RAINOUT_OROGRAPHIC * uplift).clamp(0.0, 1.0)
}

/// Simulate moisture transport to steady state and return the precipitation
/// field.
///
/// `elevation < 0` marks evaporation sources (stage-2 proxy for water; lakes
/// are not yet known at this point and their area is negligible for the
/// global budget).
pub fn simulate_moisture(
    tessellation: &Tessellation,
    elevation: &[f32],
    temperature: &[f32],
    wind: &[Vec3],
    uplift: &[f32],
) -> MoistureResult {
    let num_cells = tessellation.num_cells();

    // Per-iteration mixing fraction from the physical diffusivity
    // (explicit-scheme stability requires < ~0.5).
    let mean_spacing_sq = tessellation.mean_cell_area();
    let diffusion_frac = (MOISTURE_DIFFUSIVITY / mean_spacing_sq.max(1e-12)).min(0.45);

    // --- Precompute per-cell static data ---
    let capacity: Vec<f32> = temperature.iter().map(|&t| carrying_capacity(t)).collect();
    let is_water: Vec<bool> = elevation.iter().map(|&e| e < 0.0).collect();

    // Static part of the rainout rate: baseline + signed uplift.
    // Subsidence can suppress orographic/convergence rain, but rainout is
    // clamped non-negative so it never manufactures negative precipitation.
    let rain_rate: Vec<f32> = (0..num_cells)
        .map(|i| static_rain_rate(uplift[i]))
        .collect();
    let temp01: Vec<f32> = temperature.iter().map(|&t| t.clamp(0.0, 1.0)).collect();

    // Finite-volume upwind transport over shared Voronoi edges.
    //
    // For each edge (i, j) the volume flux is u_n * L (edge-normal wind times
    // edge length); the moisture carried is the upwind cell's. This is the
    // standard consistent FV discretization: for the (divergence-free,
    // post-projection) wind field the net transport per cell vanishes by
    // construction, instead of manufacturing cell-scale convergence noise
    // the way per-donor alignment weights do.
    struct Edge {
        a: usize,
        b: usize,
        /// Signed volume flux a->b: edge-normal wind (face-averaged) x length.
        flux: f32,
    }

    let areas = tessellation.cell_areas();
    let mut edges: Vec<Edge> = Vec::new();
    let mut outflow_rate = vec![0.0f32; num_cells]; // sum of outgoing flux / area

    for a in 0..num_cells {
        let pos_a = tessellation.cell_center(a);
        for &b in tessellation.neighbors(a) {
            if b <= a {
                continue;
            }
            let pos_b = tessellation.cell_center(b);
            // Outward edge normal at the shared face (tangent-plane component
            // of the direction a -> b).
            let chord = pos_b - pos_a;
            let mid = (pos_a + pos_b).normalize();
            let normal = chord - mid * chord.dot(mid);
            let len = normal.length();
            if len < 1e-9 {
                continue;
            }
            let normal = normal / len;

            let u_n = 0.5 * (wind[a] + wind[b]).dot(normal);
            let flux = u_n * tessellation.shared_edge_length(a, b);
            if flux.abs() < 1e-12 {
                continue;
            }
            if flux > 0.0 {
                outflow_rate[a] += flux / areas[a].max(1e-12);
            } else {
                outflow_rate[b] += -flux / areas[b].max(1e-12);
            }
            edges.push(Edge { a, b, flux });
        }
    }

    // CFL-limited timestep: no cell may export more than MOISTURE_CFL of its
    // content per iteration.
    let max_outflow = outflow_rate.iter().cloned().fold(0.0f32, f32::max);
    let dt = if max_outflow > 0.0 {
        MOISTURE_CFL / max_outflow
    } else {
        0.0
    };

    // --- Iterate to steady state ---
    let mut moisture = vec![0.0f32; num_cells];
    let mut next = vec![0.0f32; num_cells];
    let mut precip_accum = vec![0.0f32; num_cells];
    let avg_start = MOISTURE_ITERATIONS.saturating_sub(MOISTURE_AVG_WINDOW);

    for iter in 0..MOISTURE_ITERATIONS {
        let averaging = iter >= avg_start;

        for i in 0..num_cells {
            let mut m = moisture[i];

            // Evaporation: water cells relax toward carrying capacity.
            if is_water[i] {
                m += EVAPORATION_RATE * (capacity[i] - m).max(0.0);
            }

            // Rainout: static rate (baseline + orographic) plus convective
            // rain from warm humid air, plus gradual rainout of any moisture
            // above carrying capacity (air cooling as it climbs or chills).
            // Convective rain is land-only: maritime rain recycles into the
            // ocean anyway, so modeling it would only drain moisture that
            // should make landfall.
            let humidity = (m / capacity[i]).clamp(0.0, 1.0);
            let convective = if is_water[i] {
                0.0
            } else {
                RAINOUT_CONVECTIVE * humidity * humidity * temp01[i]
            };
            let over_capacity = (m - capacity[i]).max(0.0);
            let rain = (m * (rain_rate[i] + convective) + over_capacity * OVERFLOW_RAINOUT).min(m);
            m -= rain;
            // Evapotranspiration recycling returns part of land rain to the air.
            if !is_water[i] {
                m += rain * MOISTURE_RECYCLE_FRACTION;
            }
            if averaging {
                precip_accum[i] += rain;
            }

            moisture[i] = m;
        }

        // Transport pass: upwind edge fluxes (separate so all rain and
        // evaporation uses pre-transport values).
        next.copy_from_slice(&moisture);
        for e in &edges {
            // Upwind donor: the cell the flux leaves.
            let (donor, amount) = if e.flux > 0.0 {
                (e.a, e.flux * moisture[e.a])
            } else {
                (e.b, -e.flux * moisture[e.b])
            };
            let receiver = e.a + e.b - donor;
            let transported = dt * amount;
            next[donor] -= transported / areas[donor].max(1e-12);
            next[receiver] += transported / areas[receiver].max(1e-12);
        }
        std::mem::swap(&mut moisture, &mut next);

        // Eddy diffusion: horizontal turbulent mixing alongside advection
        // (standard in atmospheric transport models). Keeps the moisture
        // field smooth at the mesh scale for physical reasons rather than
        // post-hoc filtering.
        next.copy_from_slice(&moisture);
        for i in 0..num_cells {
            let neighbors = tessellation.neighbors(i);
            if neighbors.is_empty() {
                continue;
            }
            let mean: f32 =
                neighbors.iter().map(|&n| moisture[n]).sum::<f32>() / neighbors.len() as f32;
            next[i] = moisture[i] + diffusion_frac * (mean - moisture[i]);
        }
        std::mem::swap(&mut moisture, &mut next);
    }

    // Normalize precipitation so the LAND mean is 1.0: land rainfall is what
    // hydrology consumes (river thresholds and lake budgets are calibrated in
    // average-land-cell units), and ocean rain would otherwise dilute it.
    let (land_total, land_count) = precip_accum
        .iter()
        .zip(is_water.iter())
        .filter(|(_, &w)| !w)
        .fold((0.0f32, 0usize), |(t, c), (&p, _)| (t + p, c + 1));
    let mean = if land_count > 0 {
        land_total / land_count as f32
    } else {
        precip_accum.iter().sum::<f32>() / num_cells as f32 // waterworld fallback
    };
    let precipitation: Vec<f32> = if mean > 1e-12 {
        precip_accum.iter().map(|&p| p / mean).collect()
    } else {
        vec![1.0; num_cells] // degenerate (no rain anywhere): fall back to uniform
    };

    MoistureResult {
        moisture,
        precipitation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn precipitation_normalized_and_finite() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let tessellation = Tessellation::generate(2000, 1, &mut rng);
        let n = tessellation.num_cells();

        // Half-ocean world with a simple temperature gradient and eastward wind.
        let elevation: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { -0.3 } else { 0.1 })
            .collect();
        let temperature: Vec<f32> = (0..n)
            .map(|i| 1.0 - tessellation.cell_center(i).y.abs())
            .collect();
        let wind: Vec<Vec3> = (0..n)
            .map(|i| {
                let pos = tessellation.cell_center(i);
                let east = Vec3::Y.cross(pos);
                if east.length() > 1e-6 {
                    east.normalize() * 0.3
                } else {
                    Vec3::ZERO
                }
            })
            .collect();
        let uplift = vec![0.1f32; n];

        let result = simulate_moisture(&tessellation, &elevation, &temperature, &wind, &uplift);

        assert_eq!(result.precipitation.len(), n);
        assert!(result
            .precipitation
            .iter()
            .all(|p| p.is_finite() && *p >= 0.0));
        let land_precip: Vec<f32> = (0..n)
            .filter(|&i| elevation[i] >= 0.0)
            .map(|i| result.precipitation[i])
            .collect();
        let mean: f32 = land_precip.iter().sum::<f32>() / land_precip.len() as f32;
        assert!(
            (mean - 1.0).abs() < 1e-3,
            "land precip mean {mean} should be ~1.0"
        );
        assert!(result.moisture.iter().all(|m| m.is_finite() && *m >= 0.0));
    }

    #[test]
    fn signed_uplift_suppresses_static_rainout_without_going_negative() {
        assert!(static_rain_rate(1.0) > static_rain_rate(0.0));
        assert_eq!(static_rain_rate(-10.0), 0.0);
    }
}
