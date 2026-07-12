//! Effective areal fluvial denudation for the C0 landscape representation.
//!
//! This operator evolves a finite-volume **cell-mean** bedrock elevation.  Its
//! specific discharge and slope inputs are prescribed cell fields; routing is
//! deliberately outside this module.  Consequently this is a coarse-grained
//! areal process law, not a resolved channel-bed or channel-width model.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::LandscapeMesh;

/// Parameters for `E = K q^m S^n`.
///
/// `q` is specific discharge in km²/Myr, `S` is dimensionless grade and `E`
/// is effective denudation in km/Myr.  Therefore `k` has units
/// `(km/Myr) / (km²/Myr)^m`; the slope factor is dimensionless.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EffectiveArealDenudationParams {
    pub k: f64,
    pub discharge_exponent_m: f64,
    pub slope_exponent_n: f64,
}

impl EffectiveArealDenudationParams {
    pub fn validate(self) -> Result<(), EffectiveArealDenudationError> {
        if !self.k.is_finite() || self.k < 0.0 {
            return Err(EffectiveArealDenudationError::InvalidParameter("k"));
        }
        if !self.discharge_exponent_m.is_finite() || self.discharge_exponent_m < 0.0 {
            return Err(EffectiveArealDenudationError::InvalidParameter(
                "discharge_exponent_m",
            ));
        }
        if !self.slope_exponent_n.is_finite() || self.slope_exponent_n <= 0.0 {
            return Err(EffectiveArealDenudationError::InvalidParameter(
                "slope_exponent_n",
            ));
        }
        Ok(())
    }
}

/// Result of one simultaneous finite-volume denudation update.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectiveArealDenudationResult {
    /// Cell-mean lowering rate `E` (km/Myr).
    pub rate_km_myr: Vec<f64>,
    /// Solid volume exported by this update (km³).
    ///
    /// This is accumulated from exactly the depths applied to the state:
    /// `sum(E[cell] * cell_area[cell] * dt)`.
    pub exported_solid_volume_km3: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectiveArealDenudationError {
    InvalidParameter(&'static str),
    InvalidTimestep,
    InvalidMesh(String),
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidCellValue {
        field: &'static str,
        cell: usize,
    },
    NonFiniteRate {
        cell: usize,
    },
    NonFiniteUpdate,
}

impl fmt::Display for EffectiveArealDenudationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(name) => write!(f, "invalid denudation parameter {name}"),
            Self::InvalidTimestep => {
                f.write_str("denudation timestep must be finite and nonnegative")
            }
            Self::InvalidMesh(message) => write!(f, "invalid landscape mesh: {message}"),
            Self::LengthMismatch {
                field,
                expected,
                actual,
            } => write!(f, "{field} has length {actual}, expected {expected}"),
            Self::InvalidCellValue { field, cell } => {
                write!(f, "{field} has an invalid value at cell {cell}")
            }
            Self::NonFiniteRate { cell } => {
                write!(f, "effective denudation rate is non-finite at cell {cell}")
            }
            Self::NonFiniteUpdate => {
                f.write_str("effective denudation depth or exported volume is non-finite")
            }
        }
    }
}

impl std::error::Error for EffectiveArealDenudationError {}

/// Lower a cell-mean bedrock surface using prescribed C0 forcing fields.
///
/// All inputs are validated before the elevation slice is mutated.  Negative
/// elevations are permitted: zero elevation has no special status in this
/// operator, and base level belongs to the boundary/routing contract.
pub fn apply_effective_areal_denudation(
    params: EffectiveArealDenudationParams,
    mesh: &LandscapeMesh,
    mean_bedrock_elevation_km: &mut [f64],
    specific_discharge_km2_myr: &[f64],
    slope_grade: &[f64],
    dt_myr: f64,
) -> Result<EffectiveArealDenudationResult, EffectiveArealDenudationError> {
    params.validate()?;
    mesh.validate()
        .map_err(|error| EffectiveArealDenudationError::InvalidMesh(error.to_string()))?;
    if !dt_myr.is_finite() || dt_myr < 0.0 {
        return Err(EffectiveArealDenudationError::InvalidTimestep);
    }

    let cell_count = mesh.cell_count();
    validate_length(
        "mean_bedrock_elevation_km",
        mean_bedrock_elevation_km.len(),
        cell_count,
    )?;
    validate_length(
        "specific_discharge_km2_myr",
        specific_discharge_km2_myr.len(),
        cell_count,
    )?;
    validate_length("slope_grade", slope_grade.len(), cell_count)?;

    let mut rate_km_myr = Vec::with_capacity(cell_count);
    for cell in 0..cell_count {
        validate_finite(
            "mean_bedrock_elevation_km",
            mean_bedrock_elevation_km[cell],
            cell,
        )?;
        validate_nonnegative(
            "specific_discharge_km2_myr",
            specific_discharge_km2_myr[cell],
            cell,
        )?;
        validate_nonnegative("slope_grade", slope_grade[cell], cell)?;

        let rate = params.k
            * specific_discharge_km2_myr[cell].powf(params.discharge_exponent_m)
            * slope_grade[cell].powf(params.slope_exponent_n);
        if !rate.is_finite() || rate < 0.0 {
            return Err(EffectiveArealDenudationError::NonFiniteRate { cell });
        }
        rate_km_myr.push(rate);
    }

    let mut removed_depth_km = Vec::with_capacity(cell_count);
    let mut updated_elevation_km = Vec::with_capacity(cell_count);
    let mut exported_solid_volume_km3 = 0.0;
    for cell in 0..cell_count {
        let depth = rate_km_myr[cell] * dt_myr;
        exported_solid_volume_km3 += rate_km_myr[cell] * mesh.cell_area_km2[cell] * dt_myr;
        removed_depth_km.push(depth);
        updated_elevation_km.push(mean_bedrock_elevation_km[cell] - depth);
    }
    if removed_depth_km.iter().any(|depth| !depth.is_finite())
        || updated_elevation_km
            .iter()
            .any(|elevation| !elevation.is_finite())
        || !exported_solid_volume_km3.is_finite()
    {
        return Err(EffectiveArealDenudationError::NonFiniteUpdate);
    }
    mean_bedrock_elevation_km.copy_from_slice(&updated_elevation_km);

    Ok(EffectiveArealDenudationResult {
        rate_km_myr,
        exported_solid_volume_km3,
    })
}

fn validate_length(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), EffectiveArealDenudationError> {
    if actual != expected {
        return Err(EffectiveArealDenudationError::LengthMismatch {
            field,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_finite(
    field: &'static str,
    value: f64,
    cell: usize,
) -> Result<(), EffectiveArealDenudationError> {
    if !value.is_finite() {
        return Err(EffectiveArealDenudationError::InvalidCellValue { field, cell });
    }
    Ok(())
}

fn validate_nonnegative(
    field: &'static str,
    value: f64,
    cell: usize,
) -> Result<(), EffectiveArealDenudationError> {
    if !value.is_finite() || value < 0.0 {
        return Err(EffectiveArealDenudationError::InvalidCellValue { field, cell });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mesh(spacing_km: f64) -> LandscapeMesh {
        LandscapeMesh::uniform_planar_hex(96.0, 64.0, spacing_km).unwrap()
    }

    fn params() -> EffectiveArealDenudationParams {
        EffectiveArealDenudationParams {
            k: 0.02,
            discharge_exponent_m: 0.5,
            slope_exponent_n: 1.0,
        }
    }

    #[test]
    fn validates_parameters_fields_and_timestep_before_mutating() {
        let mesh = mesh(8.0);
        let n = mesh.cell_count();
        let initial = vec![1.0; n];
        let q = vec![4.0; n];
        let slope = vec![0.1; n];

        for invalid in [
            EffectiveArealDenudationParams {
                k: -1.0,
                ..params()
            },
            EffectiveArealDenudationParams {
                discharge_exponent_m: f64::NAN,
                ..params()
            },
            EffectiveArealDenudationParams {
                slope_exponent_n: 0.0,
                ..params()
            },
        ] {
            assert!(invalid.validate().is_err());
        }

        let mut elevation = initial.clone();
        assert!(apply_effective_areal_denudation(
            params(),
            &mesh,
            &mut elevation,
            &q[..n - 1],
            &slope,
            1.0,
        )
        .is_err());
        assert_eq!(elevation, initial);

        let mut bad_q = q;
        bad_q[n / 2] = -1.0;
        assert!(apply_effective_areal_denudation(
            params(),
            &mesh,
            &mut elevation,
            &bad_q,
            &slope,
            1.0,
        )
        .is_err());
        assert_eq!(elevation, initial);

        assert!(apply_effective_areal_denudation(
            params(),
            &mesh,
            &mut elevation,
            &vec![4.0; n],
            &slope,
            -1.0,
        )
        .is_err());
        assert_eq!(elevation, initial);

        let overflow_params = EffectiveArealDenudationParams {
            k: f64::MAX,
            discharge_exponent_m: 0.0,
            slope_exponent_n: 1.0,
        };
        assert!(apply_effective_areal_denudation(
            overflow_params,
            &mesh,
            &mut elevation,
            &vec![1.0; n],
            &vec![1.0; n],
            2.0,
        )
        .is_err());
        assert_eq!(elevation, initial);
    }

    #[test]
    fn rates_are_finite_nonnegative_and_zero_on_zero_slope() {
        let mesh = mesh(8.0);
        let n = mesh.cell_count();
        let mut elevation = vec![1.0; n];
        let q: Vec<_> = (0..n).map(|cell| cell as f64).collect();
        let mut slope = vec![0.2; n];
        slope[0] = 0.0;

        let result =
            apply_effective_areal_denudation(params(), &mesh, &mut elevation, &q, &slope, 0.25)
                .unwrap();

        assert_eq!(result.rate_km_myr[0], 0.0);
        assert!(result
            .rate_km_myr
            .iter()
            .all(|rate| rate.is_finite() && *rate >= 0.0));
        assert!(elevation.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn export_ledger_matches_the_applied_cell_mean_lowering() {
        let mesh = mesh(8.0);
        let n = mesh.cell_count();
        let mut elevation: Vec<_> = (0..n).map(|cell| 10.0 + cell as f64 * 1.0e-3).collect();
        let before = elevation.clone();
        let q: Vec<_> = (0..n).map(|cell| 1.0 + cell as f64 * 0.01).collect();
        let slope: Vec<_> = (0..n).map(|cell| 0.01 + cell as f64 * 1.0e-4).collect();
        let dt = 0.125;

        let result =
            apply_effective_areal_denudation(params(), &mesh, &mut elevation, &q, &slope, dt)
                .unwrap();

        let declared: f64 = result
            .rate_km_myr
            .iter()
            .zip(&mesh.cell_area_km2)
            .map(|(rate, area)| rate * area * dt)
            .sum();
        let realized: f64 = before
            .iter()
            .zip(&elevation)
            .zip(&mesh.cell_area_km2)
            .map(|((before, after), area)| (before - after) * area)
            .sum();
        assert_eq!(result.exported_solid_volume_km3, declared);
        assert!((realized - declared).abs() <= 2.0e-12 * declared.max(1.0));
    }

    #[test]
    fn smooth_manufactured_areal_export_converges_at_8_4_2_km() {
        // q is a smooth separable field that vanishes with zero gradient at
        // the exact rectangular boundary.  With K=1, m=1, S=1 and dt=1 its
        // analytic exported volume is W*H*(8/15)^2 km³.  Boundary cells are
        // still whole hex control volumes, so this test also detects leakage
        // from the current approximate boundary tiling.
        const WIDTH: f64 = 96.0;
        const HEIGHT: f64 = 64.0;
        let exact = WIDTH * HEIGHT * (8.0 / 15.0_f64).powi(2);
        let manufactured_params = EffectiveArealDenudationParams {
            k: 1.0,
            discharge_exponent_m: 1.0,
            slope_exponent_n: 1.0,
        };
        let mut errors = Vec::new();

        for spacing in [8.0, 4.0, 2.0] {
            let mesh = LandscapeMesh::uniform_planar_hex(WIDTH, HEIGHT, spacing).unwrap();
            let mut elevation = vec![100.0; mesh.cell_count()];
            let q: Vec<_> = mesh
                .cell_center_km
                .iter()
                .map(|center| {
                    let x = 2.0 * center.x / WIDTH;
                    let y = 2.0 * center.y / HEIGHT;
                    (1.0 - x * x).max(0.0).powi(2) * (1.0 - y * y).max(0.0).powi(2)
                })
                .collect();
            let slope = vec![1.0; mesh.cell_count()];
            let result = apply_effective_areal_denudation(
                manufactured_params,
                &mesh,
                &mut elevation,
                &q,
                &slope,
                1.0,
            )
            .unwrap();

            for (rate, expected) in result.rate_km_myr.iter().zip(&q) {
                assert!((rate - expected).abs() <= 2.0e-15);
            }
            errors.push((result.exported_solid_volume_km3 - exact).abs());
        }

        assert!(errors[1] < errors[0], "4 km did not improve: {errors:?}");
        assert!(errors[2] < errors[1], "2 km did not improve: {errors:?}");
        assert!(
            errors[2] / exact < 5.0e-4,
            "2 km error too large: {errors:?}"
        );
    }
}
