//! Conservative cell-mean hillslope transport for the C0 landscape model.
//!
//! Internal faces use the nonlinear threshold law
//! `q = D S / (1 - (|S| / S_c)^2)`. Each reciprocal face is visited once,
//! so its solid-volume flux is exactly antisymmetric. Closed boundary faces
//! carry no flux. Open portal faces deliberately use the simpler linear
//! Dirichlet control shared with the analytic boundary fixture: extending the
//! threshold law through a prescribed base-level face would assert unresolved
//! boundary material semantics that the testbed has not yet chosen.

use std::fmt;

use super::{
    linear_diffusive_boundary_flux_km3_myr, BoundaryFaceCondition, LandscapeMesh, OutletPortalId,
};

/// Dimensioned parameters for explicit nonlinear hillslope transport.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConservativeHillslopeParams {
    /// Material diffusivity (km²/Myr).
    pub diffusivity_km2_myr: f64,
    /// Threshold material grade (dimensionless rise/run).
    pub critical_slope_grade: f64,
    /// Floor applied to the nonlinear denominator near and above threshold.
    pub nonlinear_denominator_floor: f64,
    /// Safety factor applied to the local explicit Jacobian limit.
    pub timestep_safety: f64,
}

impl Default for ConservativeHillslopeParams {
    fn default() -> Self {
        Self {
            diffusivity_km2_myr: 0.1,
            critical_slope_grade: 0.7,
            nonlinear_denominator_floor: 1.0e-3,
            timestep_safety: 0.4,
        }
    }
}

/// One portal's signed solid-volume transfer during a step.
///
/// Positive values are export from the domain; negative values are import.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PortalSolidTransfer {
    pub portal_id: OutletPortalId,
    pub volume_km3: f64,
}

/// Conservation and stability evidence returned by one accepted step.
#[derive(Debug, Clone, PartialEq)]
pub struct ConservativeHillslopeStep {
    pub dt_myr: f64,
    pub explicit_dt_limit_myr: f64,
    pub initial_volume_km3: f64,
    pub final_volume_km3: f64,
    /// Roundoff residual from summing the antisymmetric internal increments.
    pub internal_conservation_error_km3: f64,
    pub portal_transfers: Vec<PortalSolidTransfer>,
    pub total_boundary_export_km3: f64,
    /// `final - initial + boundary export`; zero is exact closure.
    pub volume_closure_error_km3: f64,
    pub maximum_slope_ratio: f64,
    pub regularized_internal_faces: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConservativeHillslopeError {
    InvalidParameter(&'static str),
    InvalidMesh(String),
    LengthMismatch { expected: usize, actual: usize },
    NonFiniteElevation,
    UnstableTimestep { requested_myr: f64, limit_myr: f64 },
    BoundaryFlux(String),
    NonFiniteResult,
}

impl fmt::Display for ConservativeHillslopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(name) => write!(f, "invalid hillslope parameter: {name}"),
            Self::InvalidMesh(message) => write!(f, "invalid landscape mesh: {message}"),
            Self::LengthMismatch { expected, actual } => {
                write!(f, "elevation length {actual}, expected {expected}")
            }
            Self::NonFiniteElevation => f.write_str("elevation must be finite"),
            Self::UnstableTimestep {
                requested_myr,
                limit_myr,
            } => write!(
                f,
                "requested hillslope step {requested_myr} Myr exceeds explicit limit {limit_myr} Myr"
            ),
            Self::BoundaryFlux(message) => write!(f, "invalid boundary flux: {message}"),
            Self::NonFiniteResult => f.write_str("hillslope step produced a non-finite result"),
        }
    }
}

impl std::error::Error for ConservativeHillslopeError {}

/// Apply one conservative explicit step to cell-mean elevation.
///
/// This function never pins a cell. A portal base level exists at its physical
/// boundary face, at `center_distance_km` from the adjacent cell center.
pub fn apply_conservative_hillslope_step(
    mesh: &LandscapeMesh,
    elevation_km: &mut [f64],
    params: ConservativeHillslopeParams,
    dt_myr: f64,
) -> Result<ConservativeHillslopeStep, ConservativeHillslopeError> {
    validate_inputs(mesh, elevation_km, params, dt_myr)?;

    let n = mesh.cell_count();
    let mut volume_rate = vec![0.0; n];
    let mut conductance_sum = vec![0.0; n];
    let mut maximum_slope_ratio: f64 = 0.0;
    let mut regularized_internal_faces = 0;

    for cell in 0..n {
        let start = mesh.edge_offsets[cell] as usize;
        let end = mesh.edge_offsets[cell + 1] as usize;
        for edge in start..end {
            let neighbor = mesh.edge_neighbor[edge] as usize;
            if neighbor <= cell {
                continue;
            }
            let distance = f64::from(mesh.edge_distance_km[edge]);
            let width = f64::from(mesh.edge_face_width_km[edge]);
            let signed_slope = (elevation_km[cell] - elevation_km[neighbor]) / distance;
            let ratio = signed_slope.abs() / params.critical_slope_grade;
            maximum_slope_ratio = maximum_slope_ratio.max(ratio);
            let raw_denominator = 1.0 - ratio * ratio;
            if raw_denominator < params.nonlinear_denominator_floor {
                regularized_internal_faces += 1;
            }
            let denominator = raw_denominator.max(params.nonlinear_denominator_floor);
            let outward_rate = params.diffusivity_km2_myr * signed_slope * width / denominator;
            volume_rate[cell] -= outward_rate;
            volume_rate[neighbor] += outward_rate;

            // A conservative upper bound on the face-flux Jacobian. Retaining
            // the numerator when the denominator is floored is intentionally
            // cautious in the regularized regime.
            let derivative = (1.0 + ratio * ratio) / (denominator * denominator);
            let conductance = params.diffusivity_km2_myr * width / distance * derivative;
            conductance_sum[cell] += conductance;
            conductance_sum[neighbor] += conductance;
        }
    }

    let mut portal_rates: Vec<PortalSolidTransfer> = mesh
        .outlet_portals
        .iter()
        .map(|portal| PortalSolidTransfer {
            portal_id: portal.id,
            volume_km3: 0.0,
        })
        .collect();
    for face in &mesh.boundary_faces {
        let BoundaryFaceCondition::OpenBaseLevel {
            portal_id,
            elevation_km: base_level_km,
        } = face.condition
        else {
            continue;
        };
        let cell = face.cell as usize;
        let rate = linear_diffusive_boundary_flux_km3_myr(
            elevation_km[cell],
            f64::from(base_level_km),
            face.center_distance_km,
            face.width_km,
            params.diffusivity_km2_myr,
        )
        .map_err(|error| ConservativeHillslopeError::BoundaryFlux(error.to_string()))?;
        volume_rate[cell] -= rate;
        let portal = portal_rates
            .iter_mut()
            .find(|entry| entry.portal_id == portal_id)
            .ok_or_else(|| {
                ConservativeHillslopeError::InvalidMesh(format!(
                    "boundary face references absent portal {:?}",
                    portal_id
                ))
            })?;
        portal.volume_km3 += rate;
        conductance_sum[cell] +=
            params.diffusivity_km2_myr * face.width_km / face.center_distance_km;

        let boundary_ratio = (elevation_km[cell] - f64::from(base_level_km)).abs()
            / face.center_distance_km
            / params.critical_slope_grade;
        maximum_slope_ratio = maximum_slope_ratio.max(boundary_ratio);
    }

    let explicit_dt_limit_myr = conductance_sum
        .iter()
        .zip(&mesh.cell_area_km2)
        .filter(|(conductance, _)| **conductance > 0.0)
        .map(|(conductance, area)| params.timestep_safety * area / conductance)
        .fold(f64::INFINITY, f64::min);
    if dt_myr > explicit_dt_limit_myr {
        return Err(ConservativeHillslopeError::UnstableTimestep {
            requested_myr: dt_myr,
            limit_myr: explicit_dt_limit_myr,
        });
    }

    let initial_volume_km3 = volume(mesh, elevation_km);
    let total_boundary_rate: f64 = portal_rates.iter().map(|entry| entry.volume_km3).sum();
    let internal_rate_error = volume_rate.iter().sum::<f64>() + total_boundary_rate;
    // Construct and validate the complete candidate before mutating caller
    // state. A numerical failure is therefore transactional.
    let candidate: Vec<f64> = elevation_km
        .iter()
        .enumerate()
        .map(|(cell, elevation)| elevation + volume_rate[cell] * dt_myr / mesh.cell_area_km2[cell])
        .collect();
    if candidate.iter().any(|value| !value.is_finite()) {
        return Err(ConservativeHillslopeError::NonFiniteResult);
    }
    elevation_km.copy_from_slice(&candidate);
    for entry in &mut portal_rates {
        entry.volume_km3 *= dt_myr;
    }
    let total_boundary_export_km3 = total_boundary_rate * dt_myr;
    let final_volume_km3 = volume(mesh, elevation_km);
    let volume_closure_error_km3 =
        final_volume_km3 - initial_volume_km3 + total_boundary_export_km3;

    Ok(ConservativeHillslopeStep {
        dt_myr,
        explicit_dt_limit_myr,
        initial_volume_km3,
        final_volume_km3,
        internal_conservation_error_km3: internal_rate_error * dt_myr,
        portal_transfers: portal_rates,
        total_boundary_export_km3,
        volume_closure_error_km3,
        maximum_slope_ratio,
        regularized_internal_faces,
    })
}

fn validate_inputs(
    mesh: &LandscapeMesh,
    elevation_km: &[f64],
    params: ConservativeHillslopeParams,
    dt_myr: f64,
) -> Result<(), ConservativeHillslopeError> {
    mesh.validate()
        .map_err(|error| ConservativeHillslopeError::InvalidMesh(error.to_string()))?;
    if elevation_km.len() != mesh.cell_count() {
        return Err(ConservativeHillslopeError::LengthMismatch {
            expected: mesh.cell_count(),
            actual: elevation_km.len(),
        });
    }
    if elevation_km.iter().any(|value| !value.is_finite()) {
        return Err(ConservativeHillslopeError::NonFiniteElevation);
    }
    for (name, value) in [
        ("diffusivity_km2_myr", params.diffusivity_km2_myr),
        ("critical_slope_grade", params.critical_slope_grade),
        (
            "nonlinear_denominator_floor",
            params.nonlinear_denominator_floor,
        ),
        ("timestep_safety", params.timestep_safety),
        ("dt_myr", dt_myr),
    ] {
        if !value.is_finite() {
            return Err(ConservativeHillslopeError::InvalidParameter(name));
        }
    }
    if params.diffusivity_km2_myr < 0.0 {
        return Err(ConservativeHillslopeError::InvalidParameter(
            "diffusivity_km2_myr",
        ));
    }
    if params.critical_slope_grade <= 0.0 {
        return Err(ConservativeHillslopeError::InvalidParameter(
            "critical_slope_grade",
        ));
    }
    if !(0.0..=1.0).contains(&params.nonlinear_denominator_floor)
        || params.nonlinear_denominator_floor == 0.0
    {
        return Err(ConservativeHillslopeError::InvalidParameter(
            "nonlinear_denominator_floor",
        ));
    }
    if params.timestep_safety <= 0.0 || params.timestep_safety > 1.0 {
        return Err(ConservativeHillslopeError::InvalidParameter(
            "timestep_safety",
        ));
    }
    if dt_myr < 0.0 {
        return Err(ConservativeHillslopeError::InvalidParameter("dt_myr"));
    }
    Ok(())
}

fn volume(mesh: &LandscapeMesh, elevation_km: &[f64]) -> f64 {
    elevation_km
        .iter()
        .zip(&mesh.cell_area_km2)
        .map(|(elevation, area)| elevation * area)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::landscape::{
        linear_diffusive_boundary_flux_km3_myr, BoundarySide, OutletPortal,
    };

    fn closed_mesh(spacing: f64) -> LandscapeMesh {
        LandscapeMesh::uniform_planar_hex_with_portals(96.0, 64.0, spacing, &[]).unwrap()
    }

    fn params() -> ConservativeHillslopeParams {
        ConservativeHillslopeParams {
            diffusivity_km2_myr: 0.08,
            critical_slope_grade: 0.7,
            nonlinear_denominator_floor: 1.0e-3,
            timestep_safety: 0.4,
        }
    }

    #[test]
    fn closed_affine_and_perturbed_fields_conserve_internal_volume() {
        let mesh = closed_mesh(4.0);
        for perturb in [false, true] {
            let mut z: Vec<f64> = mesh
                .cell_center_km
                .iter()
                .enumerate()
                .map(|(i, point)| {
                    1.2 + 0.002 * point.x - 0.003 * point.y
                        + if perturb {
                            0.02 * ((i * 37 % 101) as f64 / 100.0 - 0.5)
                        } else {
                            0.0
                        }
                })
                .collect();
            let before = volume(&mesh, &z);
            let result = apply_conservative_hillslope_step(&mesh, &mut z, params(), 0.01).unwrap();
            assert!(result.internal_conservation_error_km3.abs() < 1.0e-12);
            assert!(result.total_boundary_export_km3.abs() < 1.0e-15);
            assert!((volume(&mesh, &z) - before).abs() < 2.0e-11);
            assert!(result.volume_closure_error_km3.abs() < 2.0e-11);
        }
    }

    #[test]
    fn open_face_uses_exact_linear_dirichlet_fixture_and_stable_portal_id() {
        let portal = OutletPortal {
            id: OutletPortalId(17),
            side: BoundarySide::South,
            span_start_km: -12.0,
            span_end_km: 12.0,
            base_level_km: 0.35,
        };
        let mesh = LandscapeMesh::uniform_planar_hex_with_portals(
            96.0,
            64.0,
            4.0,
            std::slice::from_ref(&portal),
        )
        .unwrap();
        let inward_grade = 0.0125;
        let mut z = vec![0.35; mesh.cell_count()];
        for face in &mesh.boundary_faces {
            if let BoundaryFaceCondition::OpenBaseLevel { portal_id, .. } = face.condition {
                assert_eq!(portal_id, portal.id);
                z[face.cell as usize] =
                    f64::from(portal.base_level_km) + inward_grade * face.center_distance_km;
            }
        }
        let mut expected_rate = 0.0;
        for face in &mesh.boundary_faces {
            if let BoundaryFaceCondition::OpenBaseLevel {
                elevation_km: base_level_km,
                ..
            } = face.condition
            {
                expected_rate += linear_diffusive_boundary_flux_km3_myr(
                    z[face.cell as usize],
                    f64::from(base_level_km),
                    face.center_distance_km,
                    face.width_km,
                    params().diffusivity_km2_myr,
                )
                .unwrap();
            }
        }
        let dt = 0.01;
        let result = apply_conservative_hillslope_step(&mesh, &mut z, params(), dt).unwrap();
        assert_eq!(result.portal_transfers.len(), 1);
        assert_eq!(result.portal_transfers[0].portal_id, portal.id);
        assert!((result.portal_transfers[0].volume_km3 - expected_rate * dt).abs() < 1e-14);
        assert!(result.volume_closure_error_km3.abs() < 2e-11);
    }

    fn quadratic_center_response(spacing: f64) -> f64 {
        let mesh = closed_mesh(spacing);
        let curvature = 2.0e-3;
        let mut z: Vec<f64> = mesh
            .cell_center_km
            .iter()
            .map(|point| 1.0 + curvature * (point.x * point.x + point.y * point.y))
            .collect();
        let center = mesh
            .cell_center_km
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.length_squared().total_cmp(&b.length_squared()))
            .unwrap()
            .0;
        let before = z[center];
        let dt = 1.0e-2;
        apply_conservative_hillslope_step(&mesh, &mut z, params(), dt).unwrap();
        (z[center] - before) / dt
    }

    #[test]
    fn manufactured_smooth_response_converges_at_8_4_2_km() {
        let expected = 4.0 * params().diffusivity_km2_myr * 2.0e-3;
        let responses = [8.0, 4.0, 2.0].map(quadratic_center_response);
        let errors = responses.map(|response| (response - expected).abs());
        assert!(
            errors[1] < errors[0],
            "responses={responses:?}, errors={errors:?}"
        );
        assert!(
            errors[2] < errors[1],
            "responses={responses:?}, errors={errors:?}"
        );
        assert!(
            errors[2] < 2.0e-7,
            "responses={responses:?}, errors={errors:?}"
        );
    }

    #[test]
    fn deterministic_finite_and_rejects_unstable_step() {
        let mesh = closed_mesh(4.0);
        let initial: Vec<f64> = mesh
            .cell_center_km
            .iter()
            .map(|point| 0.5 + 0.01 * (0.1 * point.x).sin() * (0.1 * point.y).cos())
            .collect();
        let mut a = initial.clone();
        let mut b = initial;
        let result_a = apply_conservative_hillslope_step(&mesh, &mut a, params(), 0.01).unwrap();
        let result_b = apply_conservative_hillslope_step(&mesh, &mut b, params(), 0.01).unwrap();
        assert_eq!(a, b);
        assert_eq!(result_a, result_b);
        assert!(a.iter().all(|value| value.is_finite()));
        assert!(result_a.explicit_dt_limit_myr.is_finite());

        let mut unstable = a;
        let before_rejected_step = unstable.clone();
        let requested = result_a.explicit_dt_limit_myr * 1.01;
        assert!(matches!(
            apply_conservative_hillslope_step(&mesh, &mut unstable, params(), requested),
            Err(ConservativeHillslopeError::UnstableTimestep { .. })
        ));
        assert_eq!(unstable, before_rejected_step);
    }
}
