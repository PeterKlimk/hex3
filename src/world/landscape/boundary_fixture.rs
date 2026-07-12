//! Analytic boundary-face flux control for the landscape testbed.
//!
//! This deliberately small fixture states the finite-volume sign and unit
//! convention at a Dirichlet boundary. It is not the production hillslope
//! boundary operator.

use std::fmt;

/// Invalid input to [`linear_diffusive_boundary_flux_km3_myr`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryFluxFixtureError {
    NonFinite(&'static str),
    NonPositive(&'static str),
    Negative(&'static str),
    NonFiniteFlux,
}

impl fmt::Display for BoundaryFluxFixtureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite(field) => write!(f, "{field} must be finite"),
            Self::NonPositive(field) => write!(f, "{field} must be positive"),
            Self::Negative(field) => write!(f, "{field} must be non-negative"),
            Self::NonFiniteFlux => f.write_str("boundary flux is not finite"),
        }
    }
}

impl std::error::Error for BoundaryFluxFixtureError {}

/// Signed diffusive volume flux through a Dirichlet boundary face (km³/Myr).
///
/// `cell_elevation_km` is the value at the adjacent cell center and
/// `boundary_elevation_km` is the prescribed value at the face. The returned
/// flux follows
///
/// `F_out = diffusivity * (cell_elevation - boundary_elevation)
///          / center_to_face_distance * face_width`.
///
/// Positive flux therefore exports solid volume from the cell; negative flux
/// imports it. Elevations and lengths are in km and diffusivity is in km²/Myr.
pub fn linear_diffusive_boundary_flux_km3_myr(
    cell_elevation_km: f64,
    boundary_elevation_km: f64,
    center_to_face_distance_km: f64,
    face_width_km: f64,
    diffusivity_km2_myr: f64,
) -> Result<f64, BoundaryFluxFixtureError> {
    for (field, value) in [
        ("cell_elevation_km", cell_elevation_km),
        ("boundary_elevation_km", boundary_elevation_km),
        ("center_to_face_distance_km", center_to_face_distance_km),
        ("face_width_km", face_width_km),
        ("diffusivity_km2_myr", diffusivity_km2_myr),
    ] {
        if !value.is_finite() {
            return Err(BoundaryFluxFixtureError::NonFinite(field));
        }
    }
    if center_to_face_distance_km <= 0.0 {
        return Err(BoundaryFluxFixtureError::NonPositive(
            "center_to_face_distance_km",
        ));
    }
    if face_width_km <= 0.0 {
        return Err(BoundaryFluxFixtureError::NonPositive("face_width_km"));
    }
    if diffusivity_km2_myr < 0.0 {
        return Err(BoundaryFluxFixtureError::Negative("diffusivity_km2_myr"));
    }

    let flux = diffusivity_km2_myr * (cell_elevation_km - boundary_elevation_km)
        / center_to_face_distance_km
        * face_width_km;
    if !flux.is_finite() {
        return Err(BoundaryFluxFixtureError::NonFiniteFlux);
    }
    Ok(flux)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RESOLUTIONS_KM: [f64; 3] = [8.0, 4.0, 2.0];

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual:.16e}, expected {expected:.16e}, tolerance {tolerance:.3e}"
        );
    }

    #[test]
    fn affine_dirichlet_flux_is_exact_and_resolution_independent() {
        let boundary_elevation_km = 0.35;
        let inward_grade = 0.0125;
        let diffusivity_km2_myr = 0.08;
        let strip_width_km = 37.0;
        let expected_flux = diffusivity_km2_myr * inward_grade * strip_width_km;

        for spacing_km in RESOLUTIONS_KM {
            let center_to_face_distance_km = 0.5 * spacing_km;
            // Exact affine sample at the center of the boundary-adjacent cell.
            let cell_elevation_km =
                boundary_elevation_km + inward_grade * center_to_face_distance_km;
            let flux = linear_diffusive_boundary_flux_km3_myr(
                cell_elevation_km,
                boundary_elevation_km,
                center_to_face_distance_km,
                strip_width_km,
                diffusivity_km2_myr,
            )
            .unwrap();

            assert_close(flux, expected_flux, 1.0e-15);
        }
    }

    #[test]
    fn finite_volume_strip_closes_storage_plus_boundary_export() {
        let boundary_elevation_km = -0.1;
        let inward_grade = 0.02;
        let diffusivity_km2_myr = 0.15;
        let strip_width_km = 24.0;
        let dt_myr = 0.25;

        for spacing_km in RESOLUTIONS_KM {
            let center_to_face_distance_km = 0.5 * spacing_km;
            let cell_elevation_km =
                boundary_elevation_km + inward_grade * center_to_face_distance_km;
            let cell_area_km2 = spacing_km * strip_width_km;
            let initial_storage_km3 = cell_elevation_km * cell_area_km2;
            let outward_flux_km3_myr = linear_diffusive_boundary_flux_km3_myr(
                cell_elevation_km,
                boundary_elevation_km,
                center_to_face_distance_km,
                strip_width_km,
                diffusivity_km2_myr,
            )
            .unwrap();
            let boundary_export_km3 = outward_flux_km3_myr * dt_myr;
            let final_elevation_km = cell_elevation_km - boundary_export_km3 / cell_area_km2;
            let final_storage_km3 = final_elevation_km * cell_area_km2;

            assert_close(
                final_storage_km3 - initial_storage_km3 + boundary_export_km3,
                0.0,
                2.0e-14,
            );
        }
    }

    #[test]
    fn sign_convention_and_inputs_are_explicit() {
        let export = linear_diffusive_boundary_flux_km3_myr(2.0, 1.0, 0.5, 3.0, 0.2).unwrap();
        let import = linear_diffusive_boundary_flux_km3_myr(1.0, 2.0, 0.5, 3.0, 0.2).unwrap();
        assert_close(export, 1.2, 4.0e-16);
        assert_close(import, -1.2, 4.0e-16);

        assert_eq!(
            linear_diffusive_boundary_flux_km3_myr(f64::NAN, 0.0, 1.0, 1.0, 1.0),
            Err(BoundaryFluxFixtureError::NonFinite("cell_elevation_km"))
        );
        assert_eq!(
            linear_diffusive_boundary_flux_km3_myr(0.0, 0.0, 0.0, 1.0, 1.0),
            Err(BoundaryFluxFixtureError::NonPositive(
                "center_to_face_distance_km"
            ))
        );
        assert_eq!(
            linear_diffusive_boundary_flux_km3_myr(0.0, 0.0, 1.0, 0.0, 1.0),
            Err(BoundaryFluxFixtureError::NonPositive("face_width_km"))
        );
        assert_eq!(
            linear_diffusive_boundary_flux_km3_myr(0.0, 0.0, 1.0, 1.0, -1.0),
            Err(BoundaryFluxFixtureError::Negative("diffusivity_km2_myr"))
        );
    }
}
