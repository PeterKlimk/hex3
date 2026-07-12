//! Cell-mean physical surface gradients for the C0 landscape representation.
//!
//! The reconstruction is deliberately independent of the drainage routing
//! surface.  For each cell it fits one planar gradient to differences between
//! the physical cell-mean elevation and the means of its graph neighbors.
//! Boundary cells use only real neighboring cells; portal or other ghost
//! elevations are not invented here.

use std::fmt;

use glam::DVec3;
use serde::{Deserialize, Serialize};

use super::LandscapeMesh;

/// Reconstructed gradient of the physical cell-mean bedrock surface.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeanSurfaceGradient {
    /// Horizontal elevation gradient `(dz/dx, dz/dy, 0)`, in km/km.
    pub vector: Vec<DVec3>,
    /// Magnitude of `vector`, a dimensionless surface grade.
    pub grade: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SurfaceGradientError {
    InvalidMesh(String),
    LengthMismatch { expected: usize, actual: usize },
    NonFiniteElevation { cell: usize },
    SingularStencil { cell: usize },
    NonFiniteResult { cell: usize },
}

impl fmt::Display for SurfaceGradientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMesh(message) => write!(f, "invalid landscape mesh: {message}"),
            Self::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "mean bedrock elevation has length {actual}, expected {expected}"
                )
            }
            Self::NonFiniteElevation { cell } => {
                write!(f, "mean bedrock elevation is non-finite at cell {cell}")
            }
            Self::SingularStencil { cell } => {
                write!(f, "surface-gradient stencil is singular at cell {cell}")
            }
            Self::NonFiniteResult { cell } => {
                write!(f, "surface-gradient result is non-finite at cell {cell}")
            }
        }
    }
}

impl std::error::Error for SurfaceGradientError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlowAlignedGradeError {
    LengthMismatch {
        gradient_count: usize,
        discharge_count: usize,
    },
    NonFiniteGradient {
        cell: usize,
    },
    NonFiniteDischarge {
        cell: usize,
    },
    NonFiniteGrade {
        cell: usize,
    },
}

impl fmt::Display for FlowAlignedGradeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                gradient_count,
                discharge_count,
            } => write!(
                f,
                "physical gradient has length {gradient_count}, but specific discharge has length {discharge_count}"
            ),
            Self::NonFiniteGradient { cell } => {
                write!(f, "physical surface gradient is non-finite at cell {cell}")
            }
            Self::NonFiniteDischarge { cell } => {
                write!(f, "specific-discharge vector is non-finite at cell {cell}")
            }
            Self::NonFiniteGrade { cell } => {
                write!(f, "flow-aligned physical grade is non-finite at cell {cell}")
            }
        }
    }
}

impl std::error::Error for FlowAlignedGradeError {}

/// Project physical surface gradient onto the local discharge direction.
///
/// For horizontal specific-discharge vector `q` and physical surface gradient
/// `grad(z)`, this returns the nonnegative downhill grade
/// `S = max(-q_hat dot grad(z), 0)`.  A vector whose horizontal magnitude is
/// **exactly zero** receives exactly zero grade; this helper deliberately has
/// no scale-dependent near-zero threshold.  Routing/fill elevation is not an
/// input and therefore cannot create physical slope.
pub fn flow_aligned_physical_grade(
    physical_gradient: &[DVec3],
    specific_discharge_vector_km2_myr: &[DVec3],
) -> Result<Vec<f64>, FlowAlignedGradeError> {
    if physical_gradient.len() != specific_discharge_vector_km2_myr.len() {
        return Err(FlowAlignedGradeError::LengthMismatch {
            gradient_count: physical_gradient.len(),
            discharge_count: specific_discharge_vector_km2_myr.len(),
        });
    }

    let mut grade = Vec::with_capacity(physical_gradient.len());
    for (cell, (gradient, discharge)) in physical_gradient
        .iter()
        .zip(specific_discharge_vector_km2_myr)
        .enumerate()
    {
        if !gradient.is_finite() {
            return Err(FlowAlignedGradeError::NonFiniteGradient { cell });
        }
        if !discharge.is_finite() {
            return Err(FlowAlignedGradeError::NonFiniteDischarge { cell });
        }

        let horizontal_magnitude = discharge.x.hypot(discharge.y);
        let cell_grade = if horizontal_magnitude == 0.0 {
            0.0
        } else {
            (-(discharge.x * gradient.x + discharge.y * gradient.y) / horizontal_magnitude).max(0.0)
        };
        if !cell_grade.is_finite() {
            return Err(FlowAlignedGradeError::NonFiniteGrade { cell });
        }
        grade.push(cell_grade);
    }
    Ok(grade)
}

/// Reconstruct the gradient of a physical cell-mean elevation field.
///
/// Each cell solves the weighted two-dimensional least-squares problem
///
/// `min_g sum_j T_ij (z_j - z_i - g dot (x_j - x_i))^2`,
///
/// where the finite-volume transmissibility weight is
/// `T_ij = face_width_ij / |x_j - x_i|`.  The 2x2 normal equation is assembled
/// and solved in `f64` from cell-center geometry.  An affine physical surface
/// is therefore reproduced exactly up to floating-point roundoff wherever the
/// real-neighbor stencil spans the plane.
pub fn reconstruct_mean_surface_gradient(
    mesh: &LandscapeMesh,
    mean_bedrock_elevation_km: &[f64],
) -> Result<MeanSurfaceGradient, SurfaceGradientError> {
    mesh.validate()
        .map_err(|error| SurfaceGradientError::InvalidMesh(error.to_string()))?;

    let cell_count = mesh.cell_count();
    if mean_bedrock_elevation_km.len() != cell_count {
        return Err(SurfaceGradientError::LengthMismatch {
            expected: cell_count,
            actual: mean_bedrock_elevation_km.len(),
        });
    }
    if let Some(cell) = mean_bedrock_elevation_km
        .iter()
        .position(|elevation| !elevation.is_finite())
    {
        return Err(SurfaceGradientError::NonFiniteElevation { cell });
    }

    let mut vector = Vec::with_capacity(cell_count);
    let mut grade = Vec::with_capacity(cell_count);
    for cell in 0..cell_count {
        let center = mesh.cell_center_km[cell];
        let elevation = mean_bedrock_elevation_km[cell];
        let start = mesh.edge_offsets[cell] as usize;
        let end = mesh.edge_offsets[cell + 1] as usize;

        // Symmetric 2x2 normal matrix and right-hand side.
        let (mut a_xx, mut a_xy, mut a_yy) = (0.0, 0.0, 0.0);
        let (mut b_x, mut b_y) = (0.0, 0.0);
        for edge in start..end {
            let neighbor = mesh.edge_neighbor[edge] as usize;
            let delta = mesh.cell_center_km[neighbor] - center;
            let distance = delta.length();
            let weight = f64::from(mesh.edge_face_width_km[edge]) / distance;
            let dz = mean_bedrock_elevation_km[neighbor] - elevation;

            a_xx += weight * delta.x * delta.x;
            a_xy += weight * delta.x * delta.y;
            a_yy += weight * delta.y * delta.y;
            b_x += weight * delta.x * dz;
            b_y += weight * delta.y * dz;
        }

        let determinant = a_xx * a_yy - a_xy * a_xy;
        // A relative test makes the degeneracy decision independent of the
        // physical units and resolution of an otherwise similar mesh.
        let matrix_scale = a_xx.abs().max(a_yy.abs()).max(a_xy.abs());
        if matrix_scale == 0.0
            || !matrix_scale.is_finite()
            || !determinant.is_finite()
            || determinant.abs() <= 64.0 * f64::EPSILON * matrix_scale * matrix_scale
        {
            return Err(SurfaceGradientError::SingularStencil { cell });
        }

        let gradient_x = (a_yy * b_x - a_xy * b_y) / determinant;
        let gradient_y = (a_xx * b_y - a_xy * b_x) / determinant;
        let cell_vector = DVec3::new(gradient_x, gradient_y, 0.0);
        let cell_grade = cell_vector.length();
        if !cell_vector.is_finite() || !cell_grade.is_finite() {
            return Err(SurfaceGradientError::NonFiniteResult { cell });
        }
        vector.push(cell_vector);
        grade.push(cell_grade);
    }

    Ok(MeanSurfaceGradient { vector, grade })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mesh(spacing_km: f64) -> LandscapeMesh {
        LandscapeMesh::uniform_planar_hex(240.0, 200.0, spacing_km).unwrap()
    }

    fn has_full_stencil(mesh: &LandscapeMesh, cell: usize) -> bool {
        mesh.edge_offsets[cell + 1] - mesh.edge_offsets[cell] == 6
    }

    #[test]
    fn affine_planes_are_reproduced_in_a_fixed_physical_interior() {
        for spacing_km in [8.0, 4.0, 2.0] {
            let mesh = mesh(spacing_km);
            for angle in [0.0_f64, 0.37, 1.1, 2.4] {
                let expected = DVec3::new(0.23 * angle.cos(), 0.23 * angle.sin(), 0.0);
                let elevation: Vec<_> = mesh
                    .cell_center_km
                    .iter()
                    .map(|center| 1.7 + expected.x * center.x + expected.y * center.y)
                    .collect();
                let reconstructed = reconstruct_mean_surface_gradient(&mesh, &elevation).unwrap();

                let mut tested = 0;
                for (cell, center) in mesh.cell_center_km.iter().enumerate() {
                    if center.x.abs() <= 70.0
                        && center.y.abs() <= 55.0
                        && has_full_stencil(&mesh, cell)
                    {
                        let error = (reconstructed.vector[cell] - expected).length();
                        assert!(
                            error <= 2.0e-14,
                            "h={spacing_km}, angle={angle}, error={error}"
                        );
                        assert!((reconstructed.grade[cell] - 0.23).abs() <= 2.0e-14);
                        tested += 1;
                    }
                }
                assert!(tested > 100);
            }
        }
    }

    #[test]
    fn smooth_radial_surface_converges_monotonically_on_fixed_mask() {
        let mut errors = Vec::new();
        for spacing_km in [8.0, 4.0, 2.0] {
            let mesh = mesh(spacing_km);
            let core_km = 18.0;
            let elevation: Vec<_> = mesh
                .cell_center_km
                .iter()
                .map(|center| {
                    (center.x * center.x + center.y * center.y + core_km * core_km).sqrt()
                })
                .collect();
            let reconstructed = reconstruct_mean_surface_gradient(&mesh, &elevation).unwrap();

            let mut squared_error = 0.0;
            let mut count = 0;
            for (cell, center) in mesh.cell_center_km.iter().enumerate() {
                let radius = center.truncate().length();
                if (25.0..=70.0).contains(&radius) && has_full_stencil(&mesh, cell) {
                    let denominator = (radius * radius + core_km * core_km).sqrt();
                    let expected = DVec3::new(center.x / denominator, center.y / denominator, 0.0);
                    squared_error += (reconstructed.vector[cell] - expected).length_squared();
                    count += 1;
                }
            }
            assert!(count > 50);
            errors.push((squared_error / count as f64).sqrt());
        }

        assert!(errors[1] < errors[0], "errors={errors:?}");
        assert!(errors[2] < errors[1], "errors={errors:?}");
        assert!(errors[2] < 0.4 * errors[0], "errors={errors:?}");
    }

    #[test]
    fn reconstruction_is_finite_and_bit_deterministic() {
        let mesh = mesh(8.0);
        let elevation: Vec<_> = mesh
            .cell_center_km
            .iter()
            .map(|center| 0.03 * center.x + 0.02 * center.y + (center.x * 0.07).sin() * 0.1)
            .collect();
        let first = reconstruct_mean_surface_gradient(&mesh, &elevation).unwrap();
        let second = reconstruct_mean_surface_gradient(&mesh, &elevation).unwrap();
        assert_eq!(first, second);
        assert!(first.vector.iter().all(|value| value.is_finite()));
        assert!(first
            .grade
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
    }

    #[test]
    fn invalid_fields_are_rejected_before_reconstruction() {
        let mesh = mesh(8.0);
        let error = reconstruct_mean_surface_gradient(&mesh, &[0.0]).unwrap_err();
        assert!(matches!(error, SurfaceGradientError::LengthMismatch { .. }));

        let mut elevation = vec![0.0; mesh.cell_count()];
        elevation[7] = f64::NAN;
        assert_eq!(
            reconstruct_mean_surface_gradient(&mesh, &elevation).unwrap_err(),
            SurfaceGradientError::NonFiniteElevation { cell: 7 }
        );
    }

    #[test]
    fn flow_alignment_preserves_downhill_grade_and_rejects_other_directions() {
        let gradient = vec![DVec3::new(3.0, 4.0, 0.0); 4];
        let discharge = vec![
            DVec3::new(-3.0, -4.0, 0.0),
            DVec3::new(-4.0, 3.0, 0.0),
            DVec3::new(3.0, 4.0, 0.0),
            DVec3::ZERO,
        ];
        let grade = flow_aligned_physical_grade(&gradient, &discharge).unwrap();
        assert_eq!(grade, vec![5.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn routing_over_flat_or_uphill_physical_paths_does_not_invent_grade() {
        // These discharge directions could have come from a depression-filled
        // routing surface.  Only the physical gradients are projected here.
        let physical_gradient = [DVec3::ZERO, DVec3::new(0.2, 0.0, 0.0)];
        let routed_discharge = [DVec3::new(7.0, 0.0, 0.0), DVec3::new(7.0, 0.0, 0.0)];
        assert_eq!(
            flow_aligned_physical_grade(&physical_gradient, &routed_discharge).unwrap(),
            vec![0.0, 0.0]
        );
    }

    #[test]
    fn flow_aligned_grade_is_deterministic_and_validates_inputs() {
        let gradient = [DVec3::new(0.1, -0.3, 0.0), DVec3::new(-0.2, 0.4, 0.0)];
        let discharge = [DVec3::new(-2.0, 5.0, 0.0), DVec3::new(3.0, -1.0, 0.0)];
        let first = flow_aligned_physical_grade(&gradient, &discharge).unwrap();
        let second = flow_aligned_physical_grade(&gradient, &discharge).unwrap();
        assert_eq!(first, second);

        assert!(matches!(
            flow_aligned_physical_grade(&gradient, &discharge[..1]),
            Err(FlowAlignedGradeError::LengthMismatch { .. })
        ));
        let invalid_gradient = [DVec3::new(f64::NAN, 0.0, 0.0)];
        assert_eq!(
            flow_aligned_physical_grade(&invalid_gradient, &[DVec3::ZERO]).unwrap_err(),
            FlowAlignedGradeError::NonFiniteGradient { cell: 0 }
        );
        let invalid_discharge = [DVec3::new(f64::INFINITY, 0.0, 0.0)];
        assert_eq!(
            flow_aligned_physical_grade(&[DVec3::ZERO], &invalid_discharge).unwrap_err(),
            FlowAlignedGradeError::NonFiniteDischarge { cell: 0 }
        );
    }
}
