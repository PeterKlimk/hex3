//! Moisture transport and precipitation.
//!
//! A single-layer moisture field advected over the wind field: water cells
//! evaporate moisture toward a temperature-dependent carrying capacity, wind
//! carries it downwind, and rain falls where moisture meets uplift (orographic
//! + convergence) plus a baseline rainout so air dries over long fetches.
//! A prescribed subsidence belt suppresses rain near the horse latitudes
//! (the divergence-free wind projection removes the convergence signal that
//! would otherwise produce it - prescribed here in the same spirit as the
//! zonal wind bands).
//!
//! Emergent behavior: rain shadows behind mountains, continental interiors
//! drying with distance from coast, a wet equator (trade winds converge at
//! the thermal low and the pre-projection uplift proxy captures it), and
//! desert belts. All rates are worldbuilding knobs in `constants.rs`.

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

/// Latitudinal rain modulation: suppressed near the subsidence desert belt,
/// enhanced in the equatorial ITCZ convergence band.
fn latitude_rain_factor(sin_lat: f32) -> f32 {
    let a = sin_lat.abs();
    let d = a - DESERT_BELT_SIN_LAT;
    let suppression = DESERT_BELT_STRENGTH * (-(d * d) / (2.0 * DESERT_BELT_WIDTH.powi(2))).exp();
    let itcz = ITCZ_STRENGTH * (-(a * a) / (2.0 * ITCZ_WIDTH.powi(2))).exp();
    (1.0 - suppression) * (1.0 + itcz)
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

    // --- Precompute per-cell static data ---
    let capacity: Vec<f32> = temperature.iter().map(|&t| carrying_capacity(t)).collect();
    let is_water: Vec<bool> = elevation.iter().map(|&e| e < 0.0).collect();

    // Static part of the rainout rate: baseline + orographic/convergence
    // uplift. The convective part depends on the evolving humidity and is
    // added per iteration. Both are suppressed in the subsidence belt.
    let belt: Vec<f32> = (0..num_cells)
        .map(|i| latitude_rain_factor(tessellation.cell_center(i).y))
        .collect();
    let rain_rate: Vec<f32> = (0..num_cells)
        .map(|i| ((RAINOUT_BASE + RAINOUT_OROGRAPHIC * uplift[i]) * belt[i]).clamp(0.0, 1.0))
        .collect();
    let temp01: Vec<f32> = temperature.iter().map(|&t| t.clamp(0.0, 1.0)).collect();

    // Downwind transport: per cell, the fraction of moisture leaving per step
    // and its distribution over downwind neighbors.
    let mut transport_frac = vec![0.0f32; num_cells];
    let mut transport_targets: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_cells];

    for i in 0..num_cells {
        let pos = tessellation.cell_center(i);
        let w = wind[i];
        let speed = w.length();
        if speed < 1e-6 {
            continue;
        }
        let wind_dir = w / speed;

        let mut weights: Vec<(usize, f32)> = Vec::new();
        let mut total = 0.0f32;
        for &n in tessellation.neighbors(i) {
            let to_n = tessellation.cell_center(n) - pos;
            let tangent = to_n - pos * pos.dot(to_n);
            let len = tangent.length();
            if len < 1e-6 {
                continue;
            }
            let alignment = wind_dir.dot(tangent / len).max(0.0);
            if alignment > 0.0 {
                weights.push((n, alignment));
                total += alignment;
            }
        }
        if total > 0.0 {
            for (_, w) in &mut weights {
                *w /= total;
            }
            transport_frac[i] = (speed * MOISTURE_ADVECTION_SCALE).min(MOISTURE_MAX_TRANSPORT);
            transport_targets[i] = weights;
        }
    }

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
                RAINOUT_CONVECTIVE * humidity * humidity * temp01[i] * belt[i]
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

        // Transport pass (separate so all rain/evaporation uses pre-transport values).
        next.copy_from_slice(&moisture);
        for i in 0..num_cells {
            let frac = transport_frac[i];
            if frac <= 0.0 {
                continue;
            }
            let outgoing = moisture[i] * frac;
            next[i] -= outgoing;
            for &(target, weight) in &transport_targets[i] {
                next[target] += outgoing * weight;
            }
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
    fn latitude_bands_shape_rain() {
        // Desert belt is drier than both the equator and high latitudes.
        assert!(latitude_rain_factor(DESERT_BELT_SIN_LAT) < latitude_rain_factor(0.0));
        assert!(latitude_rain_factor(DESERT_BELT_SIN_LAT) < latitude_rain_factor(0.95));
        // ITCZ makes the equator the wettest band.
        assert!(latitude_rain_factor(0.0) > latitude_rain_factor(0.95));
    }
}
