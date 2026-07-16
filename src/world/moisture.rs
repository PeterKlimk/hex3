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
    /// Steady-state airborne moisture per cell (raw units). Retained as a solver output
    /// validated by tests; production consumes only `precipitation` (the atmosphere no
    /// longer stores the moisture field).
    #[cfg_attr(not(test), allow(dead_code))]
    pub moisture: Vec<f32>,
    /// Precipitation rate per cell, normalized to an area-weighted mean of 1.0
    /// over non-ocean land so hydrology receives a stable global supply.
    pub precipitation: Vec<f32>,
    /// Solver and trailing-window budget evidence. This is diagnostic state;
    /// product consumers should continue to use `precipitation`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) diagnostics: MoistureDiagnostics,
}

/// Diagnostics for the exact trailing window used to construct precipitation.
///
/// Every mass is area-integrated. The closure identity is
///
/// `end = start + evaporation - ocean_rain - land_rain + land_recycle
///              + advection_change + diffusion_change + closure_residual`.
#[derive(Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct MoistureDiagnostics {
    pub dt: f32,
    pub react_dt: f32,
    pub iterations: usize,
    pub converged: bool,
    pub forced_window: bool,
    pub measurement_count: usize,
    /// Factor mapping `raw_mean_precipitation` to product precipitation.
    pub normalization_factor: f64,
    pub normalization_fallback: bool,
    pub measurement_start_airborne_mass: f64,
    pub measurement_end_airborne_mass: f64,
    pub evaporation_mass: f64,
    pub ocean_rain_mass: f64,
    pub land_rain_mass: f64,
    pub land_recycle_mass: f64,
    pub gross_advected_mass: f64,
    pub advection_mass_change: f64,
    pub diffusion_mass_change: f64,
    pub closure_residual: f64,
    pub raw_mean_precipitation: Vec<f32>,
}

fn area_integrated_mass(field: &[f32], areas: &[f32]) -> f64 {
    field
        .iter()
        .zip(areas)
        .map(|(&value, &area)| f64::from(value) * f64::from(area))
        .sum()
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
/// `is_ocean` marks connected ocean evaporation sources. Lakes are not yet
/// known at this point; inland below-datum basins deliberately remain land.
pub fn simulate_moisture(
    tessellation: &Tessellation,
    is_ocean: &[bool],
    temperature: &[f32],
    wind: &[Vec3],
    uplift: &[f32],
) -> MoistureResult {
    let num_cells = tessellation.num_cells();
    assert_eq!(is_ocean.len(), num_cells);

    // Per-iteration mixing fraction from the physical diffusivity
    // (explicit-scheme stability requires < ~0.5).
    let mean_spacing_sq = tessellation.mean_cell_area();
    let diffusion_frac = (MOISTURE_DIFFUSIVITY / mean_spacing_sq.max(1e-12)).min(0.45);

    // --- Precompute per-cell static data ---
    let capacity: Vec<f32> = temperature.iter().map(|&t| carrying_capacity(t)).collect();

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
    // standard consistent FV discretization. Paired transfers conserve global
    // area-integrated moisture; local convergence still depends on the supplied
    // face fluxes (whose discretization is not identical to wind projection).
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
    //
    // Operator splitting with a consistent timestep. Advection advances physical
    // time `dt = MOISTURE_CFL/max_outflow` per iteration. The reaction and mixing
    // constants (evaporation, rainout, eddy diffusion) were tuned as per-iteration
    // fractions (an implicit dt=1), so we scale them by `react_dt = dt/DT_REF` to
    // put them on the same physical clock. This makes the reaction:advection
    // balance — hence how far moisture penetrates inland — independent of mesh
    // resolution and wind speed (which move `dt`); at the 100k design resolution
    // `dt ≈ DT_REF` so `react_dt ≈ 1` and the generated climate is unchanged.
    // Because `react_dt` shrinks with resolution, we iterate to a tolerance rather
    // than a fixed count, and average precip over a trailing window once steady.
    let react_dt = if MOISTURE_DT_REF > 0.0 {
        dt / MOISTURE_DT_REF
    } else {
        0.0
    };
    // Saturating relaxation factor for evaporation (a monotone relax toward
    // capacity is stable for factor <= 1; coarse meshes can push react_dt>1).
    let evap_factor = (react_dt * EVAPORATION_RATE).min(1.0);
    // Eddy-mixing fraction, kept < 0.5 for explicit stability after scaling.
    let mix_frac = (react_dt * diffusion_frac).min(0.45);

    // Warm start: water cells (the evaporative source) begin saturated, land dry.
    // This skips the ocean's fill-up transient (~1/(react_dt·EVAPORATION_RATE)
    // iterations) without changing the steady state the loop converges to.
    let mut moisture: Vec<f32> = (0..num_cells)
        .map(|i| if is_ocean[i] { capacity[i] } else { 0.0 })
        .collect();
    let mut next = vec![0.0f32; num_cells];
    let mut precip_accum = vec![0.0f32; num_cells];

    // Once the field is steady (or we hit the cap) average precip over the next
    // MOISTURE_AVG_WINDOW iterations.
    let force_at = MOISTURE_MAX_ITERATIONS.saturating_sub(MOISTURE_AVG_WINDOW);
    let mut measuring = false;
    let mut measure_count = 0usize;
    let mut prev = vec![0.0f32; num_cells];
    let mut iterations = 0usize;
    let mut converged = false;
    let mut forced_window = false;
    let mut measurement_start_airborne_mass = 0.0f64;
    let mut measurement_end_airborne_mass = 0.0f64;
    let mut evaporation_mass = 0.0f64;
    let mut ocean_rain_mass = 0.0f64;
    let mut land_rain_mass = 0.0f64;
    let mut land_recycle_mass = 0.0f64;
    let mut gross_advected_mass = 0.0f64;
    let mut advection_mass_change = 0.0f64;
    let mut diffusion_mass_change = 0.0f64;
    for iter in 0..MOISTURE_MAX_ITERATIONS {
        // Reserve the entire trailing window when convergence has not yet been
        // observed. Starting at the beginning of this iteration avoids the old
        // capped-run off-by-one (which collected only window - 1 samples).
        if !measuring && iter >= force_at {
            measuring = true;
            forced_window = true;
        }
        if measuring && measure_count == 0 {
            measurement_start_airborne_mass = area_integrated_mass(&moisture, &areas);
        }

        prev.copy_from_slice(&moisture);

        for i in 0..num_cells {
            let mut m = moisture[i];

            // Evaporation: water cells relax toward carrying capacity.
            if is_ocean[i] {
                let evaporation = evap_factor * (capacity[i] - m).max(0.0);
                m += evaporation;
                if measuring {
                    evaporation_mass += f64::from(evaporation) * f64::from(areas[i]);
                }
            }

            // Rainout: static rate (baseline + orographic) plus convective
            // rain from warm humid air, plus gradual rainout of any moisture
            // above carrying capacity (air cooling as it climbs or chills), all
            // scaled by react_dt. Convective rain is land-only: maritime rain
            // recycles into the ocean anyway, so modeling it would only drain
            // moisture that should make landfall.
            let humidity = (m / capacity[i]).clamp(0.0, 1.0);
            let convective = if is_ocean[i] {
                0.0
            } else {
                RAINOUT_CONVECTIVE * humidity * humidity * temp01[i]
            };
            let over_capacity = (m - capacity[i]).max(0.0);
            let rain = (react_dt
                * (m * (rain_rate[i] + convective) + over_capacity * OVERFLOW_RAINOUT))
                .min(m);
            m -= rain;
            // Evapotranspiration recycling returns part of land rain to the air.
            if !is_ocean[i] {
                let recycle = rain * MOISTURE_RECYCLE_FRACTION;
                m += recycle;
                if measuring {
                    land_rain_mass += f64::from(rain) * f64::from(areas[i]);
                    land_recycle_mass += f64::from(recycle) * f64::from(areas[i]);
                }
            } else if measuring {
                ocean_rain_mass += f64::from(rain) * f64::from(areas[i]);
            }
            if measuring {
                precip_accum[i] += rain;
            }

            moisture[i] = m;
        }

        // Transport pass: upwind edge fluxes (separate so all rain and
        // evaporation uses pre-transport values).
        let mass_before_advection = if measuring {
            area_integrated_mass(&moisture, &areas)
        } else {
            0.0
        };
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
            if measuring {
                gross_advected_mass += f64::from(transported);
            }
            next[donor] -= transported / areas[donor].max(1e-12);
            next[receiver] += transported / areas[receiver].max(1e-12);
        }
        std::mem::swap(&mut moisture, &mut next);
        if measuring {
            advection_mass_change +=
                area_integrated_mass(&moisture, &areas) - mass_before_advection;
        }

        // Eddy diffusion: horizontal turbulent mixing alongside advection
        // (standard in atmospheric transport models). Keeps the moisture
        // field smooth at the mesh scale for physical reasons rather than
        // post-hoc filtering. Scaled by react_dt (via mix_frac) so total mixing
        // is resolution-consistent.
        let mass_before_diffusion = if measuring {
            area_integrated_mass(&moisture, &areas)
        } else {
            0.0
        };
        next.copy_from_slice(&moisture);
        for i in 0..num_cells {
            let neighbors = tessellation.neighbors(i);
            if neighbors.is_empty() {
                continue;
            }
            let mean: f32 =
                neighbors.iter().map(|&n| moisture[n]).sum::<f32>() / neighbors.len() as f32;
            next[i] = moisture[i] + mix_frac * (mean - moisture[i]);
        }
        std::mem::swap(&mut moisture, &mut next);
        if measuring {
            diffusion_mass_change +=
                area_integrated_mass(&moisture, &areas) - mass_before_diffusion;
            measurement_end_airborne_mass = area_integrated_mass(&moisture, &areas);
        }

        // Convergence: max per-cell change relative to mean moisture.
        let mut max_delta = 0.0f32;
        let mut total_m = 0.0f32;
        for i in 0..num_cells {
            max_delta = max_delta.max((moisture[i] - prev[i]).abs());
            total_m += moisture[i];
        }
        let mean_m = total_m / num_cells as f32;
        let converged_now = mean_m > 1e-12 && max_delta < MOISTURE_CONV_TOL * mean_m;
        converged |= converged_now;
        iterations = iter + 1;

        if measuring {
            measure_count += 1;
            if measure_count >= MOISTURE_AVG_WINDOW {
                break;
            }
        } else if converged_now {
            measuring = true;
        }
    }

    // Normalize precipitation so the LAND mean is PRECIP_GLOBAL_SCALE (planet
    // wetness): land rainfall is what hydrology consumes (river thresholds and
    // lake budgets are calibrated in average-land-cell units), and ocean rain
    // would otherwise dilute it. The scale sets the absolute water level the
    // relative pattern is multiplied up to.
    let (land_total, land_area) = precip_accum
        .iter()
        .zip(is_ocean.iter())
        .zip(areas.iter())
        .filter(|((_, &ocean), _)| !ocean)
        .fold((0.0f32, 0.0f32), |(rain, area), ((&p, _), &a)| {
            (rain + p * a, area + a)
        });
    let mean = if land_area > 0.0 {
        land_total / land_area
    } else {
        let total_area: f32 = areas.iter().sum();
        precip_accum
            .iter()
            .zip(areas.iter())
            .map(|(&p, &a)| p * a)
            .sum::<f32>()
            / total_area.max(1e-12) // waterworld fallback
    };
    let (precipitation, normalization_factor, normalization_fallback): (Vec<f32>, f64, bool) =
        if mean > 1e-12 {
            let k = PRECIP_GLOBAL_SCALE / mean;
            (
                precip_accum.iter().map(|&p| p * k).collect(),
                f64::from(k) * measure_count as f64,
                false,
            )
        } else {
            (vec![PRECIP_GLOBAL_SCALE; num_cells], 0.0, true) // degenerate (no rain): uniform
        };

    let raw_mean_precipitation = if measure_count > 0 {
        let inverse_count = 1.0 / measure_count as f32;
        precip_accum
            .iter()
            .map(|&rain| rain * inverse_count)
            .collect()
    } else {
        vec![0.0; num_cells]
    };
    let expected_end =
        measurement_start_airborne_mass + evaporation_mass - ocean_rain_mass - land_rain_mass
            + land_recycle_mass
            + advection_mass_change
            + diffusion_mass_change;
    let closure_residual = measurement_end_airborne_mass - expected_end;

    MoistureResult {
        moisture,
        precipitation,
        diagnostics: MoistureDiagnostics {
            dt,
            react_dt,
            iterations,
            converged,
            forced_window,
            measurement_count: measure_count,
            normalization_factor,
            normalization_fallback,
            measurement_start_airborne_mass,
            measurement_end_airborne_mass,
            evaporation_mass,
            ocean_rain_mass,
            land_rain_mass,
            land_recycle_mass,
            gross_advected_mass,
            advection_mass_change,
            diffusion_mass_change,
            closure_residual,
            raw_mean_precipitation,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::f32::consts::PI;

    struct ManufacturedArm {
        name: &'static str,
        result: MoistureResult,
    }

    fn manufactured_arm(
        tessellation: &Tessellation,
        name: &'static str,
        direction: f32,
        with_ridge: bool,
    ) -> ManufacturedArm {
        let n = tessellation.num_cells();
        let is_ocean: Vec<bool> = (0..n)
            .map(|i| tessellation.cell_center(i).z > 0.0)
            .collect();
        let temperature = vec![0.7; n];

        // Rotation about +Y carries equatorial air toward decreasing longitude.
        // Scale by mean spacing so the manufactured mesh runs near the product
        // reaction timestep instead of accidentally testing an extreme CFL arm.
        let speed = MOISTURE_CFL * tessellation.mean_cell_area().sqrt() / MOISTURE_DT_REF;
        let wind: Vec<Vec3> = (0..n)
            .map(|i| direction * speed * Vec3::Y.cross(tessellation.cell_center(i)))
            .collect();

        // Analytic ridge height h(theta), centered halfway across the land
        // hemisphere. Uplift is dh/ds along the imposed wind, normalized so its
        // maximum absolute value is one. Reversing wind therefore swaps ascent
        // and descent without changing the ridge itself.
        let sigma = 0.25f32;
        let max_abs_derivative = (-0.5f32).exp() / sigma;
        let uplift: Vec<f32> = (0..n)
            .map(|i| {
                let p = tessellation.cell_center(i);
                if !with_ridge || p.z > 0.0 {
                    return 0.0;
                }
                let theta = p.z.atan2(p.x);
                let offset = theta + PI * 0.5;
                let height = (-0.5 * (offset / sigma).powi(2)).exp();
                let dh_dtheta = -offset * height / sigma.powi(2);
                -direction * dh_dtheta / max_abs_derivative
            })
            .collect();

        ManufacturedArm {
            name,
            result: simulate_moisture(tessellation, &is_ocean, &temperature, &wind, &uplift),
        }
    }

    fn band_mean(
        tessellation: &Tessellation,
        field: &[f32],
        theta_min: f32,
        theta_max: f32,
    ) -> f64 {
        let areas = tessellation.cell_areas();
        let (weighted, area) = (0..tessellation.num_cells())
            .filter_map(|i| {
                let p = tessellation.cell_center(i);
                let theta = p.z.atan2(p.x);
                (p.y.abs() < 0.25 && theta >= theta_min && theta < theta_max)
                    .then_some((f64::from(field[i] * areas[i]), f64::from(areas[i])))
            })
            .fold((0.0, 0.0), |(sum, area), (value, cell_area)| {
                (sum + value, area + cell_area)
            });
        assert!(area > 0.0, "manufactured analysis band has no cells");
        weighted / area
    }

    fn assert_arm_contract(tessellation: &Tessellation, arm: &ManufacturedArm) {
        let n = tessellation.num_cells();
        let areas = tessellation.cell_areas();
        let is_ocean = |i: usize| tessellation.cell_center(i).z > 0.0;
        assert!(arm
            .result
            .moisture
            .iter()
            .all(|m| m.is_finite() && *m >= 0.0));
        assert!(arm
            .result
            .precipitation
            .iter()
            .chain(&arm.result.diagnostics.raw_mean_precipitation)
            .all(|p| p.is_finite() && *p >= 0.0));

        let (land_rain, land_area) =
            (0..n)
                .filter(|&i| !is_ocean(i))
                .fold((0.0f64, 0.0f64), |(rain, area), i| {
                    (
                        rain + f64::from(arm.result.precipitation[i] * areas[i]),
                        area + f64::from(areas[i]),
                    )
                });
        let land_mean = land_rain / land_area;
        assert!(
            (land_mean - f64::from(PRECIP_GLOBAL_SCALE)).abs() < 1.0e-5,
            "{} normalized land mean = {land_mean}",
            arm.name
        );

        let d = &arm.result.diagnostics;
        assert!(d.converged, "{} did not converge", arm.name);
        assert!(!d.forced_window, "{} used forced window", arm.name);
        assert!(
            !d.normalization_fallback,
            "{} normalized by fallback",
            arm.name
        );
        assert!(d.normalization_factor.is_finite() && d.normalization_factor > 0.0);
        assert!(
            d.measurement_start_airborne_mass.is_finite()
                && d.measurement_start_airborne_mass >= 0.0
                && d.measurement_end_airborne_mass.is_finite()
                && d.measurement_end_airborne_mass >= 0.0
        );
        assert_eq!(d.measurement_count, MOISTURE_AVG_WINDOW);
        for (&raw, &normalized) in d
            .raw_mean_precipitation
            .iter()
            .zip(&arm.result.precipitation)
        {
            assert!(
                (f64::from(raw) * d.normalization_factor - f64::from(normalized)).abs() < 1.0e-5,
                "{} normalization diagnostic does not reproduce product precipitation",
                arm.name
            );
        }

        let (raw_land_rain, raw_ocean_rain) = (0..n).fold((0.0f64, 0.0f64), |sum, i| {
            let rain = f64::from(d.raw_mean_precipitation[i]) * f64::from(areas[i]);
            if is_ocean(i) {
                (sum.0, sum.1 + rain)
            } else {
                (sum.0 + rain, sum.1)
            }
        });
        let sample_count = d.measurement_count as f64;
        let ledger_match =
            |measured: f64, ledger: f64| (measured - ledger).abs() / ledger.abs().max(f64::EPSILON);
        assert!(
            ledger_match(raw_land_rain * sample_count, d.land_rain_mass) < 1.0e-6,
            "{} raw land rain does not reproduce the ledger",
            arm.name
        );
        assert!(
            ledger_match(raw_ocean_rain * sample_count, d.ocean_rain_mass) < 1.0e-6,
            "{} raw ocean rain does not reproduce the ledger",
            arm.name
        );
        assert!(
            ledger_match(
                d.land_recycle_mass,
                d.land_rain_mass * f64::from(MOISTURE_RECYCLE_FRACTION),
            ) < 1.0e-6,
            "{} land recycling does not match its declared fraction",
            arm.name
        );

        let reaction_throughput =
            d.evaporation_mass + d.ocean_rain_mass + d.land_rain_mass + d.land_recycle_mass;
        let closure_ratio = d.closure_residual.abs() / reaction_throughput.max(f64::EPSILON);
        let advection_ratio =
            d.advection_mass_change.abs() / d.gross_advected_mass.max(f64::EPSILON);
        let diffusion_ratio = d.diffusion_mass_change.abs() / reaction_throughput.max(f64::EPSILON);
        println!(
            "{}: iterations={} dt={:.6} react_dt={:.4} closure={:+.3e} ({:.3e}) advection={:+.3e}/{:.3e} ({:.3e}) diffusion={:+.3e} ({:.3e} reaction throughput)",
            arm.name,
            d.iterations,
            d.dt,
            d.react_dt,
            d.closure_residual,
            closure_ratio,
            d.advection_mass_change,
            d.gross_advected_mass,
            advection_ratio,
            d.diffusion_mass_change,
            diffusion_ratio,
        );
        assert!(
            closure_ratio < 1.0e-5,
            "{} explicit moisture ledger does not close: {closure_ratio:e}",
            arm.name
        );
        assert!(
            advection_ratio < 1.0e-6,
            "{} finite-volume advection changed global mass: {advection_ratio:e}",
            arm.name
        );
        // Deliberately no acceptance assertion on diffusion drift. The printed
        // ratio is evidence for deciding whether this discretization is viable.
    }

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

        let is_ocean: Vec<bool> = elevation.iter().map(|&e| e < 0.0).collect();
        let result = simulate_moisture(&tessellation, &is_ocean, &temperature, &wind, &uplift);

        assert_eq!(result.precipitation.len(), n);
        assert!(result
            .precipitation
            .iter()
            .all(|p| p.is_finite() && *p >= 0.0));
        let areas = tessellation.cell_areas();
        let (rain, area) = (0..n)
            .filter(|&i| !is_ocean[i])
            .fold((0.0f32, 0.0f32), |(rain, area), i| {
                (rain + result.precipitation[i] * areas[i], area + areas[i])
            });
        let mean = rain / area;
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

    #[test]
    fn manufactured_moisture_correspondence() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x5eed_cafe);
        let tessellation = Tessellation::generate(4000, 1, &mut rng);
        let flat_forward = manufactured_arm(&tessellation, "flat-forward", 1.0, false);
        let flat_reverse = manufactured_arm(&tessellation, "flat-reverse", -1.0, false);
        let ridge_forward = manufactured_arm(&tessellation, "ridge-forward", 1.0, true);
        let ridge_reverse = manufactured_arm(&tessellation, "ridge-reverse", -1.0, true);

        for arm in [&flat_forward, &flat_reverse, &ridge_forward, &ridge_reverse] {
            assert_arm_contract(&tessellation, arm);
        }

        let coast_width = 0.55;
        let coast_gap = 0.15;
        let forward_entry = band_mean(
            &tessellation,
            &flat_forward.result.diagnostics.raw_mean_precipitation,
            -coast_width,
            -coast_gap,
        );
        let forward_exit = band_mean(
            &tessellation,
            &flat_forward.result.diagnostics.raw_mean_precipitation,
            -PI + coast_gap,
            -PI + coast_width,
        );
        let reverse_entry = band_mean(
            &tessellation,
            &flat_reverse.result.diagnostics.raw_mean_precipitation,
            -PI + coast_gap,
            -PI + coast_width,
        );
        let reverse_exit = band_mean(
            &tessellation,
            &flat_reverse.result.diagnostics.raw_mean_precipitation,
            -coast_width,
            -coast_gap,
        );
        println!(
            "fetch: forward entry/exit={forward_entry:.6}/{forward_exit:.6} ({:.3}x), reverse entry/exit={reverse_entry:.6}/{reverse_exit:.6} ({:.3}x)",
            forward_entry / forward_exit,
            reverse_entry / reverse_exit,
        );
        assert!(
            forward_entry > forward_exit && reverse_entry > reverse_exit,
            "fetch drying did not reverse with wind"
        );

        let ridge_gap = 0.12;
        let ridge_width = 0.5;
        let forward_windward = (-PI * 0.5 + ridge_gap, -PI * 0.5 + ridge_width);
        let forward_leeward = (-PI * 0.5 - ridge_width, -PI * 0.5 - ridge_gap);
        let raw = |arm: &ManufacturedArm, band: (f32, f32)| {
            band_mean(
                &tessellation,
                &arm.result.diagnostics.raw_mean_precipitation,
                band.0,
                band.1,
            )
        };
        let ff_windward_delta =
            raw(&ridge_forward, forward_windward) - raw(&flat_forward, forward_windward);
        let ff_leeward_delta =
            raw(&ridge_forward, forward_leeward) - raw(&flat_forward, forward_leeward);
        let fr_windward_delta =
            raw(&ridge_reverse, forward_leeward) - raw(&flat_reverse, forward_leeward);
        let fr_leeward_delta =
            raw(&ridge_reverse, forward_windward) - raw(&flat_reverse, forward_windward);
        println!(
            "ridge-minus-flat raw rain: forward windward={ff_windward_delta:+.6} leeward={ff_leeward_delta:+.6}; reverse windward={fr_windward_delta:+.6} leeward={fr_leeward_delta:+.6}"
        );
        assert!(
            ff_windward_delta > 0.0
                && ff_leeward_delta < 0.0
                && fr_windward_delta > 0.0
                && fr_leeward_delta < 0.0,
            "ridge enhancement/shadow did not switch sides with wind"
        );
    }
}
