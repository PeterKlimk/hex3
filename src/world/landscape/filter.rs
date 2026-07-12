//! Conservative scalar Helmholtz filtering on the landscape finite-volume mesh.
//!
//! The filter solves
//!
//! ```text
//! (M + alpha^2 L) y = M x,
//! ```
//!
//! where `M` contains cell areas and `L` is the reciprocal-face finite-volume
//! Laplacian.  This is an experimental C0 foundation: it filters scalar
//! cell-mean fields without changing the supplied field.  Vector-component
//! filtering is deliberately not part of this API.

use super::{BoundaryFaceCondition, LandscapeMesh};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelmholtzBoundaryMode {
    /// Zero normal derivative on every exposed face.
    HomogeneousNeumann,
    /// Portal faces use their declared base level; all other faces are closed.
    OpenPortalDirichlet,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HelmholtzFilterParams {
    pub alpha_km: f64,
    pub relative_tolerance: f64,
    pub absolute_tolerance: f64,
    pub max_iterations: usize,
    pub boundary_mode: HelmholtzBoundaryMode,
}

impl Default for HelmholtzFilterParams {
    fn default() -> Self {
        Self {
            alpha_km: 16.0,
            relative_tolerance: 1.0e-12,
            absolute_tolerance: 1.0e-14,
            max_iterations: 10_000,
            boundary_mode: HelmholtzBoundaryMode::HomogeneousNeumann,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HelmholtzFilterAudit {
    pub iterations: usize,
    pub initial_residual_l2: f64,
    pub final_residual_l2: f64,
    pub convergence_threshold_l2: f64,
    pub converged: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HelmholtzFilterResult {
    pub field: Vec<f64>,
    pub audit: HelmholtzFilterAudit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelmholtzFilterError(pub String);

impl fmt::Display for HelmholtzFilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for HelmholtzFilterError {}

/// Apply a deterministic scalar finite-volume Helmholtz filter.
///
/// Failure to reach the requested tolerance is reported in the returned audit
/// rather than hidden by changing the tolerance or the physical filter scale.
pub fn apply_scalar_helmholtz_filter(
    mesh: &LandscapeMesh,
    input: &[f64],
    params: HelmholtzFilterParams,
) -> Result<HelmholtzFilterResult, HelmholtzFilterError> {
    validate(mesh, input, params)?;
    let n = mesh.cell_count();

    if params.alpha_km == 0.0 {
        return Ok(HelmholtzFilterResult {
            field: input.to_vec(),
            audit: HelmholtzFilterAudit {
                iterations: 0,
                initial_residual_l2: 0.0,
                final_residual_l2: 0.0,
                convergence_threshold_l2: params.absolute_tolerance,
                converged: true,
            },
        });
    }

    let alpha2 = params.alpha_km * params.alpha_km;
    let mut diagonal = mesh.cell_area_km2.clone();
    let mut rhs = Vec::with_capacity(n);
    for (&area, &value) in mesh.cell_area_km2.iter().zip(input) {
        rhs.push(area * value);
    }

    for (cell, cell_diagonal) in diagonal.iter_mut().enumerate() {
        for edge in mesh.edge_offsets[cell] as usize..mesh.edge_offsets[cell + 1] as usize {
            let transmissibility =
                mesh.edge_face_width_km[edge] as f64 / mesh.edge_distance_km[edge] as f64;
            *cell_diagonal += alpha2 * transmissibility;
        }
    }
    if params.boundary_mode == HelmholtzBoundaryMode::OpenPortalDirichlet {
        for face in &mesh.boundary_faces {
            if let BoundaryFaceCondition::OpenBaseLevel { elevation_km, .. } = face.condition {
                let transmissibility = face.width_km / face.center_distance_km;
                let cell = face.cell as usize;
                diagonal[cell] += alpha2 * transmissibility;
                rhs[cell] += alpha2 * transmissibility * elevation_km as f64;
            }
        }
    }

    // Starting at the unfiltered field is deterministic and is exact when the
    // input is constant under homogeneous Neumann boundaries.
    let mut solution = input.to_vec();
    let mut product = vec![0.0; n];
    apply_operator(mesh, alpha2, &diagonal, &solution, &mut product);
    let mut residual: Vec<f64> = rhs
        .iter()
        .zip(&product)
        .map(|(&right, &left)| right - left)
        .collect();
    let initial_residual = l2_norm(&residual);
    let threshold = params
        .absolute_tolerance
        .max(params.relative_tolerance * l2_norm(&rhs));
    if initial_residual <= threshold {
        return Ok(HelmholtzFilterResult {
            field: solution,
            audit: HelmholtzFilterAudit {
                iterations: 0,
                initial_residual_l2: initial_residual,
                final_residual_l2: initial_residual,
                convergence_threshold_l2: threshold,
                converged: true,
            },
        });
    }

    let mut preconditioned: Vec<f64> = residual
        .iter()
        .zip(&diagonal)
        .map(|(&r, &d)| r / d)
        .collect();
    let mut direction = preconditioned.clone();
    let mut rz = dot(&residual, &preconditioned);
    let mut iterations = 0;

    for iteration in 1..=params.max_iterations {
        apply_operator(mesh, alpha2, &diagonal, &direction, &mut product);
        let denominator = dot(&direction, &product);
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(HelmholtzFilterError(
                "Helmholtz operator lost positive definiteness".into(),
            ));
        }
        let step = rz / denominator;
        for cell in 0..n {
            solution[cell] += step * direction[cell];
            residual[cell] -= step * product[cell];
        }
        iterations = iteration;
        let recursive_residual = l2_norm(&residual);
        if recursive_residual <= threshold {
            break;
        }

        for cell in 0..n {
            preconditioned[cell] = residual[cell] / diagonal[cell];
        }
        let next_rz = dot(&residual, &preconditioned);
        if !next_rz.is_finite() || next_rz < 0.0 {
            return Err(HelmholtzFilterError(
                "non-finite Helmholtz conjugate-gradient state".into(),
            ));
        }
        let beta = next_rz / rz;
        for cell in 0..n {
            direction[cell] = preconditioned[cell] + beta * direction[cell];
        }
        rz = next_rz;
    }

    // Audit the true algebraic residual, not only the recursively updated PCG
    // residual (which can drift by roundoff during long solves).
    apply_operator(mesh, alpha2, &diagonal, &solution, &mut product);
    let final_residual = rhs
        .iter()
        .zip(&product)
        .map(|(&right, &left)| (right - left).powi(2))
        .sum::<f64>()
        .sqrt();
    let converged = final_residual <= threshold;

    Ok(HelmholtzFilterResult {
        field: solution,
        audit: HelmholtzFilterAudit {
            iterations,
            initial_residual_l2: initial_residual,
            final_residual_l2: final_residual,
            convergence_threshold_l2: threshold,
            converged,
        },
    })
}

fn validate(
    mesh: &LandscapeMesh,
    input: &[f64],
    params: HelmholtzFilterParams,
) -> Result<(), HelmholtzFilterError> {
    mesh.validate()
        .map_err(|error| HelmholtzFilterError(format!("invalid mesh: {error}")))?;
    if input.len() != mesh.cell_count() {
        return Err(HelmholtzFilterError(format!(
            "field length {} does not match mesh cell count {}",
            input.len(),
            mesh.cell_count()
        )));
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(HelmholtzFilterError(
            "scalar field must contain only finite values".into(),
        ));
    }
    if !params.alpha_km.is_finite() || params.alpha_km < 0.0 {
        return Err(HelmholtzFilterError(
            "alpha_km must be finite and nonnegative".into(),
        ));
    }
    if !params.relative_tolerance.is_finite() || params.relative_tolerance < 0.0 {
        return Err(HelmholtzFilterError(
            "relative_tolerance must be finite and nonnegative".into(),
        ));
    }
    if !params.absolute_tolerance.is_finite() || params.absolute_tolerance < 0.0 {
        return Err(HelmholtzFilterError(
            "absolute_tolerance must be finite and nonnegative".into(),
        ));
    }
    if params.relative_tolerance == 0.0 && params.absolute_tolerance == 0.0 {
        return Err(HelmholtzFilterError(
            "at least one convergence tolerance must be positive".into(),
        ));
    }
    if params.max_iterations == 0 {
        return Err(HelmholtzFilterError(
            "max_iterations must be positive".into(),
        ));
    }

    // PCG requires a symmetric graph operator. Mesh validation guarantees a
    // reverse edge exists; here we additionally ensure both directed records
    // describe the same reciprocal-face transmissibility.
    for cell in 0..mesh.cell_count() {
        for edge in mesh.edge_offsets[cell] as usize..mesh.edge_offsets[cell + 1] as usize {
            let neighbor = mesh.edge_neighbor[edge] as usize;
            let reverse = (mesh.edge_offsets[neighbor] as usize
                ..mesh.edge_offsets[neighbor + 1] as usize)
                .find(|&candidate| mesh.edge_neighbor[candidate] as usize == cell)
                .expect("mesh validation guaranteed reciprocal adjacency");
            let forward_t =
                mesh.edge_face_width_km[edge] as f64 / mesh.edge_distance_km[edge] as f64;
            let reverse_t =
                mesh.edge_face_width_km[reverse] as f64 / mesh.edge_distance_km[reverse] as f64;
            let scale = forward_t.abs().max(reverse_t.abs()).max(1.0);
            if (forward_t - reverse_t).abs() > 32.0 * f64::EPSILON * scale {
                return Err(HelmholtzFilterError(format!(
                    "non-reciprocal transmissibility on face {cell}<->{neighbor}"
                )));
            }
        }
    }
    Ok(())
}

fn apply_operator(
    mesh: &LandscapeMesh,
    alpha2: f64,
    diagonal: &[f64],
    field: &[f64],
    output: &mut [f64],
) {
    for cell in 0..mesh.cell_count() {
        let mut value = diagonal[cell] * field[cell];
        for edge in mesh.edge_offsets[cell] as usize..mesh.edge_offsets[cell + 1] as usize {
            let transmissibility =
                mesh.edge_face_width_km[edge] as f64 / mesh.edge_distance_km[edge] as f64;
            value -= alpha2 * transmissibility * field[mesh.edge_neighbor[edge] as usize];
        }
        // Dirichlet boundary contributions are already in the diagonal.  The
        // prescribed value belongs only on the right-hand side.
        output[cell] = value;
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
}

fn l2_norm(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::landscape::{BoundarySide, OutletPortal, OutletPortalId};
    use std::f64::consts::TAU;

    fn params(alpha_km: f64, boundary_mode: HelmholtzBoundaryMode) -> HelmholtzFilterParams {
        HelmholtzFilterParams {
            alpha_km,
            boundary_mode,
            ..HelmholtzFilterParams::default()
        }
    }

    fn area_integral(mesh: &LandscapeMesh, field: &[f64]) -> f64 {
        mesh.cell_area_km2
            .iter()
            .zip(field)
            .map(|(&area, &value)| area * value)
            .sum()
    }

    #[test]
    fn alpha_zero_is_bitwise_identity_and_does_not_mutate_input() {
        let mesh = LandscapeMesh::uniform_planar_hex(64.0, 48.0, 4.0).unwrap();
        let input: Vec<_> = mesh
            .cell_center_km
            .iter()
            .map(|center| center.x.sin() + center.y.cos())
            .collect();
        let original = input.clone();
        let result = apply_scalar_helmholtz_filter(
            &mesh,
            &input,
            params(0.0, HelmholtzBoundaryMode::OpenPortalDirichlet),
        )
        .unwrap();
        assert_eq!(result.field, original);
        assert_eq!(input, original);
        assert_eq!(result.audit.iterations, 0);
    }

    #[test]
    fn neumann_preserves_constants_and_area_integral() {
        let mesh = LandscapeMesh::uniform_planar_hex(96.0, 64.0, 4.0).unwrap();
        let constant = vec![3.25; mesh.cell_count()];
        let constant_result = apply_scalar_helmholtz_filter(
            &mesh,
            &constant,
            params(16.0, HelmholtzBoundaryMode::HomogeneousNeumann),
        )
        .unwrap();
        assert!(constant_result.audit.converged);
        assert_eq!(constant_result.field, constant);

        let input: Vec<_> = mesh
            .cell_center_km
            .iter()
            .map(|center| 0.5 + (center.x / 17.0).sin() * (center.y / 13.0).cos())
            .collect();
        let result = apply_scalar_helmholtz_filter(
            &mesh,
            &input,
            params(12.0, HelmholtzBoundaryMode::HomogeneousNeumann),
        )
        .unwrap();
        assert!(result.audit.converged);
        let scale = area_integral(&mesh, &input).abs().max(1.0);
        assert!(
            (area_integral(&mesh, &result.field) - area_integral(&mesh, &input)).abs()
                < 2.0e-11 * scale
        );
    }

    #[test]
    fn nonnegative_impulse_remains_nonnegative_within_solver_tolerance() {
        let mesh = LandscapeMesh::uniform_planar_hex(96.0, 64.0, 4.0).unwrap();
        let mut input = vec![0.0; mesh.cell_count()];
        let center = mesh
            .cell_center_km
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.length_squared().total_cmp(&b.length_squared()))
            .unwrap()
            .0;
        input[center] = 1.0;
        let result = apply_scalar_helmholtz_filter(
            &mesh,
            &input,
            params(16.0, HelmholtzBoundaryMode::HomogeneousNeumann),
        )
        .unwrap();
        assert!(result.audit.converged);
        assert!(result.field.iter().all(|&value| value >= -2.0e-12));
        assert!((area_integral(&mesh, &result.field) - area_integral(&mesh, &input)).abs() < 1e-10);
    }

    #[test]
    fn affine_field_is_unchanged_in_neumann_buffered_interior() {
        let mesh = LandscapeMesh::uniform_planar_hex(256.0, 192.0, 4.0).unwrap();
        let input: Vec<_> = mesh
            .cell_center_km
            .iter()
            .map(|center| 2.0 + 0.003 * center.x - 0.002 * center.y)
            .collect();
        let result = apply_scalar_helmholtz_filter(
            &mesh,
            &input,
            params(8.0, HelmholtzBoundaryMode::HomogeneousNeumann),
        )
        .unwrap();
        assert!(result.audit.converged);
        let max_error = mesh
            .cell_center_km
            .iter()
            .zip(&input)
            .zip(&result.field)
            // An affine field is harmonic, but its nonzero normal derivative
            // is incompatible with the homogeneous-Neumann exterior. Sample
            // beyond eight physical filter lengths from every boundary so the
            // analytic exponentially decaying boundary layer is negligible.
            .filter(|((center, _), _)| center.x.abs() < 56.0 && center.y.abs() < 24.0)
            .map(|((_, &expected), &actual)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(max_error < 2.0e-5, "buffered affine error {max_error}");
    }

    #[test]
    fn sinusoidal_transfer_converges_at_fixed_physical_scale() {
        let wavelength_km = 128.0;
        let alpha_km = 16.0;
        let continuum_gain = 1.0 / (1.0 + (alpha_km * TAU / wavelength_km).powi(2));
        let mut errors = Vec::new();
        for spacing in [8.0, 4.0, 2.0] {
            let mesh = LandscapeMesh::uniform_planar_hex(512.0, 256.0, spacing).unwrap();
            let input: Vec<_> = mesh
                .cell_center_km
                .iter()
                .map(|center| (TAU * center.x / wavelength_km).cos())
                .collect();
            let result = apply_scalar_helmholtz_filter(
                &mesh,
                &input,
                params(alpha_km, HelmholtzBoundaryMode::HomogeneousNeumann),
            )
            .unwrap();
            assert!(result.audit.converged);
            let mut squared_error = 0.0;
            let mut count = 0;
            for ((center, &source), &filtered) in
                mesh.cell_center_km.iter().zip(&input).zip(&result.field)
            {
                if center.x.abs() < 160.0 && center.y.abs() < 64.0 {
                    squared_error += (filtered - continuum_gain * source).powi(2);
                    count += 1;
                }
            }
            errors.push((squared_error / count as f64).sqrt());
        }
        eprintln!("alpha=16 km sinusoidal RMS transfer errors at 8/4/2 km: {errors:?}");
        assert!(errors[1] < errors[0], "errors: {errors:?}");
        assert!(errors[2] < errors[1], "errors: {errors:?}");
        // At alpha=16 km the 8 km mesh is usable but visibly discretized; keep
        // that fact exposed instead of relaxing the physical scale.
        assert!(errors[0] < 0.015, "8 km transfer error: {}", errors[0]);
    }

    #[test]
    fn portal_dirichlet_condition_has_a_localized_boundary_effect() {
        let portal = OutletPortal {
            id: OutletPortalId(7),
            side: BoundarySide::South,
            span_start_km: -32.0,
            span_end_km: 32.0,
            base_level_km: 0.0,
        };
        let mesh =
            LandscapeMesh::uniform_planar_hex_with_portals(256.0, 192.0, 4.0, &[portal]).unwrap();
        let input = vec![1.0; mesh.cell_count()];
        let neumann = apply_scalar_helmholtz_filter(
            &mesh,
            &input,
            params(12.0, HelmholtzBoundaryMode::HomogeneousNeumann),
        )
        .unwrap();
        let portal = apply_scalar_helmholtz_filter(
            &mesh,
            &input,
            params(12.0, HelmholtzBoundaryMode::OpenPortalDirichlet),
        )
        .unwrap();
        assert!(portal.audit.converged);
        assert_eq!(neumann.field, input);
        let near_min = mesh
            .cell_center_km
            .iter()
            .zip(&portal.field)
            .filter(|(center, _)| center.y < -80.0 && center.x.abs() < 24.0)
            .map(|(_, &value)| value)
            .fold(f64::INFINITY, f64::min);
        let far_max_error = mesh
            .cell_center_km
            .iter()
            .zip(&portal.field)
            .filter(|(center, _)| center.y > 48.0)
            .map(|(_, &value)| (value - 1.0).abs())
            .fold(0.0, f64::max);
        assert!(
            near_min < 0.5,
            "portal did not impose base level: {near_min}"
        );
        assert!(
            far_max_error < 2.0e-5,
            "portal leaked through buffer: {far_max_error}"
        );
    }

    #[test]
    fn repeated_solves_are_bitwise_deterministic() {
        let mesh = LandscapeMesh::uniform_planar_hex(96.0, 64.0, 4.0).unwrap();
        let input: Vec<_> = mesh
            .cell_center_km
            .iter()
            .map(|center| (center.x * 0.19).sin() + (center.y * 0.13).cos())
            .collect();
        let first = apply_scalar_helmholtz_filter(
            &mesh,
            &input,
            params(10.0, HelmholtzBoundaryMode::HomogeneousNeumann),
        )
        .unwrap();
        let second = apply_scalar_helmholtz_filter(
            &mesh,
            &input,
            params(10.0, HelmholtzBoundaryMode::HomogeneousNeumann),
        )
        .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn rejects_invalid_parameters_and_fields() {
        let mesh = LandscapeMesh::uniform_planar_hex(32.0, 24.0, 4.0).unwrap();
        let field = vec![0.0; mesh.cell_count()];
        let mut invalid = params(-1.0, HelmholtzBoundaryMode::HomogeneousNeumann);
        assert!(apply_scalar_helmholtz_filter(&mesh, &field, invalid).is_err());
        invalid.alpha_km = 1.0;
        invalid.max_iterations = 0;
        assert!(apply_scalar_helmholtz_filter(&mesh, &field, invalid).is_err());
        assert!(apply_scalar_helmholtz_filter(
            &mesh,
            &field[..field.len() - 1],
            params(1.0, HelmholtzBoundaryMode::HomogeneousNeumann)
        )
        .is_err());
        let mut nonfinite = field;
        nonfinite[0] = f64::NAN;
        assert!(apply_scalar_helmholtz_filter(
            &mesh,
            &nonfinite,
            params(1.0, HelmholtzBoundaryMode::HomogeneousNeumann)
        )
        .is_err());
    }
}
