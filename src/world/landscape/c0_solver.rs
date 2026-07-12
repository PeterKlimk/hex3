//! Transactional C0 landscape solver for a finite-volume cell-mean surface.
//!
//! This family intentionally does not reuse the Slice 1 path-incision solver:
//! water is routed as face flux, fluvial lowering is effective **areal**
//! denudation, and portal base levels live on boundary faces rather than pinning
//! cells.  The elevation integral below is an explicit volume moment; it is not
//! a claim about absolute crustal volume.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::{
    apply_conservative_hillslope_step, apply_effective_areal_denudation, BoundaryFaceCondition,
    ConservativeHillslopeError, ConservativeHillslopeParams, DeformationFrame,
    EffectiveArealDenudationParams, FaceFlowCache, FlowPartition, LandscapeMesh, OutletPortalId,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct C0LandscapeParams {
    pub effective_areal_denudation: EffectiveArealDenudationParams,
    /// Uniform runoff depth rate; local water supply is this rate times cell area.
    pub runoff_depth_rate_km_myr: f64,
    pub hillslope: ConservativeHillslopeParams,
    /// Accuracy limit, not a stability clamp.
    pub maximum_uplift_depth_km: f64,
    /// Accuracy limit on effective cell-mean lowering in one Lie-split step.
    pub maximum_effective_denudation_depth_km: f64,
    /// Courant fraction for the explicit slope-dependent denudation response.
    pub max_effective_denudation_courant: f64,
    pub minimum_dt_myr: f64,
    pub maximum_adaptive_attempts: u32,
}

impl Default for C0LandscapeParams {
    fn default() -> Self {
        Self {
            effective_areal_denudation: EffectiveArealDenudationParams {
                // No coupled C0 regime has been dimensionally justified yet.
                k: 0.0,
                discharge_exponent_m: 1.0,
                slope_exponent_n: 1.0,
            },
            runoff_depth_rate_km_myr: 500.0,
            hillslope: ConservativeHillslopeParams::default(),
            maximum_uplift_depth_km: 0.02,
            maximum_effective_denudation_depth_km: 0.02,
            max_effective_denudation_courant: 0.25,
            minimum_dt_myr: 1.0e-8,
            maximum_adaptive_attempts: 16,
        }
    }
}

impl C0LandscapeParams {
    fn validate(self) -> Result<(), C0LandscapeError> {
        self.effective_areal_denudation
            .validate()
            .map_err(|error| C0LandscapeError::Operator(error.to_string()))?;
        for (name, value) in [
            ("runoff_depth_rate_km_myr", self.runoff_depth_rate_km_myr),
            ("maximum_uplift_depth_km", self.maximum_uplift_depth_km),
            (
                "maximum_effective_denudation_depth_km",
                self.maximum_effective_denudation_depth_km,
            ),
            (
                "max_effective_denudation_courant",
                self.max_effective_denudation_courant,
            ),
            ("minimum_dt_myr", self.minimum_dt_myr),
        ] {
            if !value.is_finite()
                || if name == "runoff_depth_rate_km_myr" {
                    value < 0.0
                } else {
                    value <= 0.0
                }
            {
                return Err(C0LandscapeError::InvalidParameter(name));
            }
        }
        // A slope-change CFL has not yet been derived for sublinear slope
        // laws in this coupled explicit family.
        if self.effective_areal_denudation.slope_exponent_n < 1.0 {
            return Err(C0LandscapeError::InvalidParameter(
                "effective_areal_denudation.slope_exponent_n",
            ));
        }
        if self.max_effective_denudation_courant > 1.0 {
            return Err(C0LandscapeError::InvalidParameter(
                "max_effective_denudation_courant",
            ));
        }
        if self.maximum_adaptive_attempts == 0 {
            return Err(C0LandscapeError::InvalidParameter(
                "maximum_adaptive_attempts",
            ));
        }
        // The hillslope operator owns the complete validation of this bundle;
        // obvious invalid values are rejected here so construction is useful.
        if !self.hillslope.diffusivity_km2_myr.is_finite()
            || self.hillslope.diffusivity_km2_myr < 0.0
            || !self.hillslope.critical_slope_grade.is_finite()
            || self.hillslope.critical_slope_grade <= 0.0
            || !self.hillslope.nonlinear_denominator_floor.is_finite()
            || self.hillslope.nonlinear_denominator_floor <= 0.0
            || self.hillslope.nonlinear_denominator_floor > 1.0
            || !self.hillslope.timestep_safety.is_finite()
            || self.hillslope.timestep_safety <= 0.0
            || self.hillslope.timestep_safety > 1.0
        {
            return Err(C0LandscapeError::InvalidParameter("hillslope"));
        }
        Ok(())
    }
}

/// Cumulative accounting for the integral `sum(z_mean * cell_area)` (km3).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct C0ElevationVolumeMomentLedger {
    pub initial_elevation_volume_moment_km3: f64,
    pub rock_uplift_moment_km3: f64,
    pub effective_areal_denudation_export_km3: f64,
    /// Signed: positive is export through portal faces, negative is import.
    pub hillslope_portal_transfer_km3: f64,
    pub final_elevation_volume_moment_km3: f64,
    pub closure_error_km3: f64,
}

impl C0ElevationVolumeMomentLedger {
    pub fn expected_final_moment_km3(self) -> f64 {
        self.initial_elevation_volume_moment_km3 + self.rock_uplift_moment_km3
            - self.effective_areal_denudation_export_km3
            - self.hillslope_portal_transfer_km3
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct C0LandscapeState {
    pub time_myr: f64,
    pub revision: u64,
    pub mean_bedrock_elevation_km: Vec<f64>,
    pub elevation_volume_moment_ledger: C0ElevationVolumeMomentLedger,
}

impl C0LandscapeState {
    pub fn new(
        mesh: &LandscapeMesh,
        mean_bedrock_elevation_km: Vec<f64>,
    ) -> Result<Self, C0LandscapeError> {
        mesh.validate()
            .map_err(|error| C0LandscapeError::InvalidMesh(error.to_string()))?;
        validate_elevation(mesh, &mean_bedrock_elevation_km)?;
        let initial = elevation_volume_moment(mesh, &mean_bedrock_elevation_km);
        Ok(Self {
            time_myr: 0.0,
            revision: 0,
            mean_bedrock_elevation_km,
            elevation_volume_moment_ledger: C0ElevationVolumeMomentLedger {
                initial_elevation_volume_moment_km3: initial,
                final_elevation_volume_moment_km3: initial,
                ..C0ElevationVolumeMomentLedger::default()
            },
        })
    }

    pub fn elevation_volume_moment_km3(&self, mesh: &LandscapeMesh) -> f64 {
        elevation_volume_moment(mesh, &self.mean_bedrock_elevation_km)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum C0TimestepLimiter {
    Requested,
    UpliftAccuracy,
    EffectiveDenudationAccuracy,
    EffectiveDenudationSlopeCourant,
    HillslopeStability,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct C0OperatorLimits {
    pub requested_dt_myr: f64,
    pub accepted_dt_myr: f64,
    pub uplift_accuracy_limit_myr: f64,
    pub effective_denudation_accuracy_limit_myr: f64,
    /// Explicit slope-response limit derived from routed physical faces.
    pub effective_denudation_slope_cfl_limit_myr: Option<f64>,
    pub hillslope_stability_limit_myr: f64,
    pub limiting_operator: C0TimestepLimiter,
    pub attempted_steps: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct C0WaterDiagnostics {
    pub total_supply_km3_myr: f64,
    pub portal_outflow_km3_myr: Vec<(OutletPortalId, f64)>,
    pub total_portal_outflow_km3_myr: f64,
    pub total_sink_storage_km3_myr: f64,
    pub water_balance_error_km3_myr: f64,
    /// Cells carrying accumulated flow for which the least-squares vector is
    /// exactly zero. This exposes the unresolved junction/cancellation limit;
    /// the solver does not replace it with raw discharge or a hidden floor.
    pub unresolved_specific_discharge_cells: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct C0StepDiagnostics {
    pub time_start_myr: f64,
    pub time_end_myr: f64,
    pub operator_limits: C0OperatorLimits,
    pub rock_uplift_moment_km3: f64,
    pub effective_areal_denudation_export_km3: f64,
    pub hillslope_portal_transfer_km3: f64,
    pub elevation_volume_moment_change_km3: f64,
    pub closure_error_km3: f64,
    pub maximum_effective_denudation_rate_km_myr: f64,
    pub maximum_hillslope_slope_ratio: f64,
    pub regularized_hillslope_faces: usize,
    pub water: C0WaterDiagnostics,
}

#[derive(Debug, Clone, PartialEq)]
pub enum C0LandscapeError {
    InvalidParameter(&'static str),
    InvalidTimestep,
    InvalidMesh(String),
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFinite(&'static str),
    StableStepTooSmall {
        stable_dt_myr: f64,
        minimum_dt_myr: f64,
    },
    TimestepDidNotConverge,
    Operator(String),
}

impl fmt::Display for C0LandscapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(name) => write!(f, "invalid C0 parameter {name}"),
            Self::InvalidTimestep => f.write_str("C0 timestep must be finite and positive"),
            Self::InvalidMesh(message) => write!(f, "invalid landscape mesh: {message}"),
            Self::LengthMismatch {
                field,
                expected,
                actual,
            } => write!(f, "{field} has length {actual}, expected {expected}"),
            Self::NonFinite(field) => write!(f, "{field} contains a non-finite value"),
            Self::StableStepTooSmall {
                stable_dt_myr,
                minimum_dt_myr,
            } => write!(
                f,
                "stable C0 step {stable_dt_myr} Myr is below minimum {minimum_dt_myr} Myr"
            ),
            Self::TimestepDidNotConverge => f.write_str("C0 adaptive timestep did not converge"),
            Self::Operator(message) => write!(f, "C0 operator failed: {message}"),
        }
    }
}

impl std::error::Error for C0LandscapeError {}

#[derive(Debug, Clone)]
pub struct C0LandscapeSolver {
    pub params: C0LandscapeParams,
}

impl C0LandscapeSolver {
    pub fn new(params: C0LandscapeParams) -> Result<Self, C0LandscapeError> {
        params.validate()?;
        Ok(Self { params })
    }

    /// Adaptive step treating the supplied frame as constant over all
    /// candidate trials. Use `step_with_forcing` across forcing ramps.
    pub fn step(
        &self,
        mesh: &LandscapeMesh,
        midpoint_forcing: &DeformationFrame,
        requested_dt_myr: f64,
        state: &mut C0LandscapeState,
    ) -> Result<C0StepDiagnostics, C0LandscapeError> {
        self.step_adaptive(mesh, requested_dt_myr, state, |_| midpoint_forcing.clone())
    }

    /// Adaptive step with forcing re-sampled at every candidate midpoint.
    pub fn step_with_forcing<F>(
        &self,
        mesh: &LandscapeMesh,
        requested_dt_myr: f64,
        state: &mut C0LandscapeState,
        evaluate_midpoint: F,
    ) -> Result<C0StepDiagnostics, C0LandscapeError>
    where
        F: FnMut(f64) -> DeformationFrame,
    {
        self.step_adaptive(mesh, requested_dt_myr, state, evaluate_midpoint)
    }

    fn step_adaptive<F>(
        &self,
        mesh: &LandscapeMesh,
        requested_dt_myr: f64,
        state: &mut C0LandscapeState,
        mut evaluate_midpoint: F,
    ) -> Result<C0StepDiagnostics, C0LandscapeError>
    where
        F: FnMut(f64) -> DeformationFrame,
    {
        if !requested_dt_myr.is_finite() || requested_dt_myr <= 0.0 {
            return Err(C0LandscapeError::InvalidTimestep);
        }
        mesh.validate()
            .map_err(|error| C0LandscapeError::InvalidMesh(error.to_string()))?;
        validate_elevation(mesh, &state.mean_bedrock_elevation_km)?;
        let original_time = state.time_myr;
        let mut candidate_dt = requested_dt_myr;
        let mut limiting_operator = C0TimestepLimiter::Requested;

        for attempt in 1..=self.params.maximum_adaptive_attempts {
            if candidate_dt < self.params.minimum_dt_myr {
                return Err(C0LandscapeError::StableStepTooSmall {
                    stable_dt_myr: candidate_dt,
                    minimum_dt_myr: self.params.minimum_dt_myr,
                });
            }
            let forcing = evaluate_midpoint(original_time + 0.5 * candidate_dt);
            match self.trial(mesh, &forcing, candidate_dt, state)? {
                Trial::Accepted(accepted) => {
                    let (mut candidate_state, mut diagnostics) = *accepted;
                    diagnostics.operator_limits.requested_dt_myr = requested_dt_myr;
                    diagnostics.operator_limits.accepted_dt_myr = candidate_dt;
                    diagnostics.operator_limits.limiting_operator = limiting_operator;
                    diagnostics.operator_limits.attempted_steps = attempt;
                    // The whole trial commits exactly once.
                    std::mem::swap(state, &mut candidate_state);
                    return Ok(diagnostics);
                }
                Trial::Retry { limit_myr, limiter } => {
                    if !limit_myr.is_finite() || limit_myr <= 0.0 {
                        return Err(C0LandscapeError::Operator(format!(
                            "{limiter:?} produced invalid timestep limit {limit_myr}"
                        )));
                    }
                    limiting_operator = limiter;
                    candidate_dt = candidate_dt.min(limit_myr);
                }
            }
        }
        Err(C0LandscapeError::TimestepDidNotConverge)
    }

    fn trial(
        &self,
        mesh: &LandscapeMesh,
        forcing: &DeformationFrame,
        dt_myr: f64,
        state: &C0LandscapeState,
    ) -> Result<Trial, C0LandscapeError> {
        validate_forcing(mesh, forcing)?;
        let maximum_uplift_rate = forcing
            .rock_vertical_rate_km_myr
            .iter()
            .map(|rate| f64::from(*rate).abs())
            .fold(0.0, f64::max);
        let uplift_limit = if maximum_uplift_rate == 0.0 {
            f64::INFINITY
        } else {
            self.params.maximum_uplift_depth_km / maximum_uplift_rate
        };
        if dt_myr > uplift_limit {
            return Ok(Trial::Retry {
                limit_myr: uplift_limit,
                limiter: C0TimestepLimiter::UpliftAccuracy,
            });
        }

        let mut candidate = state.clone();
        let volume_before = candidate.elevation_volume_moment_km3(mesh);
        let mut uplift_moment_km3 = 0.0;
        for cell in 0..mesh.cell_count() {
            let depth = f64::from(forcing.rock_vertical_rate_km_myr[cell]) * dt_myr;
            candidate.mean_bedrock_elevation_km[cell] += depth;
            uplift_moment_km3 += depth * mesh.cell_area_km2[cell];
        }

        let local_supply_km3_myr: Vec<f64> = mesh
            .cell_area_km2
            .iter()
            .map(|area| self.params.runoff_depth_rate_km_myr * area)
            .collect();
        let flow = FaceFlowCache::route_with_depressions(
            mesh,
            &candidate.mean_bedrock_elevation_km,
            &local_supply_km3_myr,
            FlowPartition::MfdSlope,
        )
        .map_err(|error| C0LandscapeError::Operator(error.to_string()))?;
        let (flow_grade, directional_length_km) = face_consistent_routed_grade_and_length(
            mesh,
            &candidate.mean_bedrock_elevation_km,
            &flow,
        )?;
        let slope_cfl_limit = effective_denudation_slope_cfl_limit(
            self.params,
            &flow.specific_discharge_km2_myr,
            &flow_grade,
            &directional_length_km,
        );
        if dt_myr > slope_cfl_limit {
            return Ok(Trial::Retry {
                limit_myr: slope_cfl_limit,
                limiter: C0TimestepLimiter::EffectiveDenudationSlopeCourant,
            });
        }
        let denudation = apply_effective_areal_denudation(
            self.params.effective_areal_denudation,
            mesh,
            &mut candidate.mean_bedrock_elevation_km,
            &flow.specific_discharge_km2_myr,
            &flow_grade,
            dt_myr,
        )
        .map_err(|error| C0LandscapeError::Operator(error.to_string()))?;
        let maximum_denudation_rate = denudation.rate_km_myr.iter().copied().fold(0.0, f64::max);
        let denudation_limit = if maximum_denudation_rate == 0.0 {
            f64::INFINITY
        } else {
            self.params.maximum_effective_denudation_depth_km / maximum_denudation_rate
        };
        if dt_myr > denudation_limit {
            return Ok(Trial::Retry {
                limit_myr: denudation_limit,
                limiter: C0TimestepLimiter::EffectiveDenudationAccuracy,
            });
        }

        let hillslope = match apply_conservative_hillslope_step(
            mesh,
            &mut candidate.mean_bedrock_elevation_km,
            self.params.hillslope,
            dt_myr,
        ) {
            Ok(result) => result,
            Err(ConservativeHillslopeError::UnstableTimestep { limit_myr, .. }) => {
                return Ok(Trial::Retry {
                    limit_myr,
                    limiter: C0TimestepLimiter::HillslopeStability,
                });
            }
            Err(error) => return Err(C0LandscapeError::Operator(error.to_string())),
        };

        let volume_after = candidate.elevation_volume_moment_km3(mesh);
        let change = volume_after - volume_before;
        let expected_change = uplift_moment_km3
            - denudation.exported_solid_volume_km3
            - hillslope.total_boundary_export_km3;
        let closure_error = change - expected_change;
        candidate.time_myr += dt_myr;
        candidate.revision = candidate
            .revision
            .checked_add(1)
            .ok_or_else(|| C0LandscapeError::Operator("revision overflow".into()))?;
        let ledger = &mut candidate.elevation_volume_moment_ledger;
        ledger.rock_uplift_moment_km3 += uplift_moment_km3;
        ledger.effective_areal_denudation_export_km3 += denudation.exported_solid_volume_km3;
        ledger.hillslope_portal_transfer_km3 += hillslope.total_boundary_export_km3;
        ledger.final_elevation_volume_moment_km3 = volume_after;
        ledger.closure_error_km3 = volume_after - ledger.expected_final_moment_km3();

        let unresolved_specific_discharge_cells = flow
            .available_supply_km3_myr
            .iter()
            .zip(&flow.local_supply_km3_myr)
            .zip(&flow.specific_discharge_km2_myr)
            .filter(|((available, local), q)| **available > **local && **q == 0.0)
            .count();
        let water_balance_error_km3_myr = flow.water_balance_error_km3_myr();
        let diagnostics = C0StepDiagnostics {
            time_start_myr: state.time_myr,
            time_end_myr: candidate.time_myr,
            operator_limits: C0OperatorLimits {
                requested_dt_myr: dt_myr,
                accepted_dt_myr: dt_myr,
                uplift_accuracy_limit_myr: uplift_limit,
                effective_denudation_accuracy_limit_myr: denudation_limit,
                effective_denudation_slope_cfl_limit_myr: Some(slope_cfl_limit),
                hillslope_stability_limit_myr: hillslope.explicit_dt_limit_myr,
                limiting_operator: C0TimestepLimiter::Requested,
                attempted_steps: 1,
            },
            rock_uplift_moment_km3: uplift_moment_km3,
            effective_areal_denudation_export_km3: denudation.exported_solid_volume_km3,
            hillslope_portal_transfer_km3: hillslope.total_boundary_export_km3,
            elevation_volume_moment_change_km3: change,
            closure_error_km3: closure_error,
            maximum_effective_denudation_rate_km_myr: maximum_denudation_rate,
            maximum_hillslope_slope_ratio: hillslope.maximum_slope_ratio,
            regularized_hillslope_faces: hillslope.regularized_internal_faces,
            water: C0WaterDiagnostics {
                total_supply_km3_myr: flow.total_supply_km3_myr,
                portal_outflow_km3_myr: flow.portal_outflow_km3_myr,
                total_portal_outflow_km3_myr: flow.total_portal_outflow_km3_myr,
                total_sink_storage_km3_myr: flow.total_sink_storage_km3_myr,
                water_balance_error_km3_myr,
                unresolved_specific_discharge_cells,
            },
        };
        Ok(Trial::Accepted(Box::new((candidate, diagnostics))))
    }
}

enum Trial {
    Accepted(Box<(C0LandscapeState, C0StepDiagnostics)>),
    Retry {
        limit_myr: f64,
        limiter: C0TimestepLimiter,
    },
}

/// Physical routed grade and one directional length for every finite volume.
///
/// Grade is the outgoing-water-flux-weighted mean of strictly downhill
/// physical face grades. A route created only by equal filled elevation may
/// carry water, but its physical grade contribution is exactly zero. The
/// directional length is the flux-weighted harmonic mean
/// `sum(Q) / sum(Q / distance)` over those same outgoing internal and portal
/// faces. Cells with no outgoing flux receive grade zero and infinite length.
fn face_consistent_routed_grade_and_length(
    mesh: &LandscapeMesh,
    physical_elevation_km: &[f64],
    flow: &FaceFlowCache,
) -> Result<(Vec<f64>, Vec<f64>), C0LandscapeError> {
    validate_elevation(mesh, physical_elevation_km)?;
    if flow.directed_edge_flux_km3_myr.len() != mesh.edge_neighbor.len()
        || flow.boundary_face_flux_km3_myr.len() != mesh.boundary_faces.len()
    {
        return Err(C0LandscapeError::Operator(
            "face-flow cache does not match mesh geometry".into(),
        ));
    }
    let n = mesh.cell_count();
    let mut outgoing_flux = vec![0.0; n];
    let mut flux_grade = vec![0.0; n];
    let mut flux_inverse_distance = vec![0.0; n];
    for cell in 0..n {
        let start = mesh.edge_offsets[cell] as usize;
        let end = mesh.edge_offsets[cell + 1] as usize;
        for edge in start..end {
            let flux = flow.directed_edge_flux_km3_myr[edge];
            if flux == 0.0 {
                continue;
            }
            let neighbor = mesh.edge_neighbor[edge] as usize;
            let distance = f64::from(mesh.edge_distance_km[edge]);
            let physical_grade = ((physical_elevation_km[cell] - physical_elevation_km[neighbor])
                / distance)
                .max(0.0);
            outgoing_flux[cell] += flux;
            flux_grade[cell] += flux * physical_grade;
            flux_inverse_distance[cell] += flux / distance;
        }
    }
    for (face_index, face) in mesh.boundary_faces.iter().enumerate() {
        let BoundaryFaceCondition::OpenBaseLevel { elevation_km, .. } = face.condition else {
            continue;
        };
        let flux = flow.boundary_face_flux_km3_myr[face_index];
        if flux == 0.0 {
            continue;
        }
        let cell = face.cell as usize;
        let physical_grade = ((physical_elevation_km[cell] - f64::from(elevation_km))
            / face.center_distance_km)
            .max(0.0);
        outgoing_flux[cell] += flux;
        flux_grade[cell] += flux * physical_grade;
        flux_inverse_distance[cell] += flux / face.center_distance_km;
    }

    let mut grade = vec![0.0; n];
    let mut directional_length_km = vec![f64::INFINITY; n];
    for cell in 0..n {
        if outgoing_flux[cell] > 0.0 {
            grade[cell] = flux_grade[cell] / outgoing_flux[cell];
            directional_length_km[cell] = outgoing_flux[cell] / flux_inverse_distance[cell];
        }
        if !grade[cell].is_finite()
            || grade[cell] < 0.0
            || !(directional_length_km[cell].is_finite()
                || directional_length_km[cell] == f64::INFINITY)
        {
            return Err(C0LandscapeError::Operator(format!(
                "invalid face-consistent grade geometry at cell {cell}"
            )));
        }
    }
    Ok((grade, directional_length_km))
}

fn effective_denudation_slope_cfl_limit(
    params: C0LandscapeParams,
    specific_discharge_km2_myr: &[f64],
    physical_grade: &[f64],
    directional_length_km: &[f64],
) -> f64 {
    let law = params.effective_areal_denudation;
    specific_discharge_km2_myr
        .iter()
        .zip(physical_grade)
        .zip(directional_length_km)
        .filter_map(|((q, grade), length)| {
            let slope_response = law.slope_exponent_n
                * law.k
                * q.powf(law.discharge_exponent_m)
                * grade.powf(law.slope_exponent_n - 1.0);
            (slope_response > 0.0 && length.is_finite())
                .then_some(params.max_effective_denudation_courant * length / slope_response)
        })
        .fold(f64::INFINITY, f64::min)
}

fn validate_elevation(mesh: &LandscapeMesh, elevation: &[f64]) -> Result<(), C0LandscapeError> {
    if elevation.len() != mesh.cell_count() {
        return Err(C0LandscapeError::LengthMismatch {
            field: "mean_bedrock_elevation_km",
            expected: mesh.cell_count(),
            actual: elevation.len(),
        });
    }
    if elevation.iter().any(|value| !value.is_finite()) {
        return Err(C0LandscapeError::NonFinite("mean_bedrock_elevation_km"));
    }
    Ok(())
}

fn validate_forcing(
    mesh: &LandscapeMesh,
    forcing: &DeformationFrame,
) -> Result<(), C0LandscapeError> {
    let expected = mesh.cell_count();
    for (field, actual) in [
        (
            "rock_vertical_rate_km_myr",
            forcing.rock_vertical_rate_km_myr.len(),
        ),
        (
            "horizontal_velocity_km_myr",
            forcing.horizontal_velocity_km_myr.len(),
        ),
        ("dominant_episode", forcing.dominant_episode.len()),
    ] {
        if actual != expected {
            return Err(C0LandscapeError::LengthMismatch {
                field,
                expected,
                actual,
            });
        }
    }
    if forcing
        .rock_vertical_rate_km_myr
        .iter()
        .any(|rate| !rate.is_finite())
        || forcing
            .horizontal_velocity_km_myr
            .iter()
            .any(|velocity| !velocity.is_finite())
    {
        return Err(C0LandscapeError::NonFinite("deformation frame"));
    }
    Ok(())
}

fn elevation_volume_moment(mesh: &LandscapeMesh, elevation: &[f64]) -> f64 {
    elevation
        .iter()
        .zip(&mesh.cell_area_km2)
        .map(|(z, area)| z * area)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn mesh() -> LandscapeMesh {
        LandscapeMesh::uniform_planar_hex(96.0, 64.0, 8.0).unwrap()
    }

    fn frame(mesh: &LandscapeMesh, rate: f32) -> DeformationFrame {
        DeformationFrame {
            rock_vertical_rate_km_myr: vec![rate; mesh.cell_count()],
            horizontal_velocity_km_myr: vec![Vec3::ZERO; mesh.cell_count()],
            dominant_episode: vec![None; mesh.cell_count()],
        }
    }

    fn inactive_params() -> C0LandscapeParams {
        C0LandscapeParams {
            effective_areal_denudation: EffectiveArealDenudationParams {
                k: 0.0,
                discharge_exponent_m: 0.5,
                slope_exponent_n: 1.0,
            },
            hillslope: ConservativeHillslopeParams {
                diffusivity_km2_myr: 0.0,
                ..ConservativeHillslopeParams::default()
            },
            ..C0LandscapeParams::default()
        }
    }

    #[test]
    fn zero_and_uplift_only_steps_are_exact() {
        let mesh = mesh();
        let solver = C0LandscapeSolver::new(inactive_params()).unwrap();
        let initial = vec![1.25; mesh.cell_count()];
        let mut zero = C0LandscapeState::new(&mesh, initial.clone()).unwrap();
        let diagnostics = solver
            .step(&mesh, &frame(&mesh, 0.0), 0.01, &mut zero)
            .unwrap();
        assert_eq!(zero.mean_bedrock_elevation_km, initial);
        assert_eq!(zero.time_myr, 0.01);
        assert_eq!(zero.revision, 1);
        assert_eq!(diagnostics.elevation_volume_moment_change_km3, 0.0);

        let mut uplift = C0LandscapeState::new(&mesh, vec![1.25; mesh.cell_count()]).unwrap();
        solver
            .step(&mesh, &frame(&mesh, 0.5), 0.01, &mut uplift)
            .unwrap();
        assert!(uplift.mean_bedrock_elevation_km.iter().all(|z| *z == 1.255));
        assert!(
            uplift
                .elevation_volume_moment_ledger
                .closure_error_km3
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn failed_step_is_byte_unchanged() {
        let mesh = mesh();
        let mut params = inactive_params();
        params.minimum_dt_myr = 0.1;
        let solver = C0LandscapeSolver::new(params).unwrap();
        let mut state = C0LandscapeState::new(&mesh, vec![1.0; mesh.cell_count()]).unwrap();
        let before = bincode::serialize(&state).unwrap();
        assert!(matches!(
            solver.step(&mesh, &frame(&mesh, 100.0), 1.0, &mut state),
            Err(C0LandscapeError::StableStepTooSmall { .. })
        ));
        assert_eq!(bincode::serialize(&state).unwrap(), before);
    }

    #[test]
    fn portal_cells_are_not_pinned() {
        let mesh = mesh();
        let solver = C0LandscapeSolver::new(inactive_params()).unwrap();
        let mut state = C0LandscapeState::new(&mesh, vec![0.75; mesh.cell_count()]).unwrap();
        solver
            .step(&mesh, &frame(&mesh, 0.25), 0.01, &mut state)
            .unwrap();
        for face in &mesh.boundary_faces {
            if matches!(
                face.condition,
                super::super::BoundaryFaceCondition::OpenBaseLevel { .. }
            ) {
                assert_eq!(state.mean_bedrock_elevation_km[face.cell as usize], 0.7525);
            }
        }
    }

    #[test]
    fn direct_operator_composition_matches_solver() {
        let mesh = mesh();
        let mut params = C0LandscapeParams::default();
        params.effective_areal_denudation.k = 0.02;
        params.maximum_uplift_depth_km = 10.0;
        params.maximum_effective_denudation_depth_km = 10.0;
        params.hillslope.diffusivity_km2_myr = 0.001;
        let solver = C0LandscapeSolver::new(params).unwrap();
        let initial: Vec<_> = mesh
            .cell_center_km
            .iter()
            .map(|p| 1.0 + 0.003 * p.y + 0.01 * (0.1 * p.x).sin())
            .collect();
        let forcing = frame(&mesh, 0.02);
        let dt = 1.0e-4;
        let mut state = C0LandscapeState::new(&mesh, initial.clone()).unwrap();
        solver.step(&mesh, &forcing, dt, &mut state).unwrap();

        let mut direct = initial;
        for (z, rate) in direct.iter_mut().zip(&forcing.rock_vertical_rate_km_myr) {
            *z += f64::from(*rate) * dt;
        }
        let supply: Vec<_> = mesh
            .cell_area_km2
            .iter()
            .map(|a| params.runoff_depth_rate_km_myr * a)
            .collect();
        let flow =
            FaceFlowCache::route_with_depressions(&mesh, &direct, &supply, FlowPartition::MfdSlope)
                .unwrap();
        let (grade, _) = face_consistent_routed_grade_and_length(&mesh, &direct, &flow).unwrap();
        apply_effective_areal_denudation(
            params.effective_areal_denudation,
            &mesh,
            &mut direct,
            &flow.specific_discharge_km2_myr,
            &grade,
            dt,
        )
        .unwrap();
        apply_conservative_hillslope_step(&mesh, &mut direct, params.hillslope, dt).unwrap();
        assert_eq!(state.mean_bedrock_elevation_km, direct);
    }

    #[test]
    fn ledger_closes_for_all_active_operators() {
        let mesh = mesh();
        let mut params = C0LandscapeParams::default();
        params.effective_areal_denudation.k = 0.02;
        params.maximum_uplift_depth_km = 1.0;
        params.maximum_effective_denudation_depth_km = 1.0;
        params.hillslope.diffusivity_km2_myr = 0.002;
        let solver = C0LandscapeSolver::new(params).unwrap();
        let initial: Vec<_> = mesh
            .cell_center_km
            .iter()
            .map(|p| 1.0 + 0.004 * p.y + 0.002 * p.x)
            .collect();
        let mut state = C0LandscapeState::new(&mesh, initial).unwrap();
        let diagnostics = solver
            .step(&mesh, &frame(&mesh, 0.03), 1e-4, &mut state)
            .unwrap();
        assert!(diagnostics.closure_error_km3.abs() < 2e-10);
        assert!(state.elevation_volume_moment_ledger.closure_error_km3.abs() < 2e-10);
        assert!(diagnostics.water.water_balance_error_km3_myr.abs() < 1e-8);
    }

    #[test]
    fn smooth_denudation_limit_halves_step_and_resamples_midpoint() {
        let mesh = mesh();
        let mut params = C0LandscapeParams::default();
        params.effective_areal_denudation.k = 1.0;
        // Preserve this test's original moderate-q manufactured regime; the
        // inactive production default now deliberately declares direct q.
        params.effective_areal_denudation.discharge_exponent_m = 0.5;
        params.maximum_effective_denudation_depth_km = 1.0e-5;
        params.hillslope.diffusivity_km2_myr = 0.0;
        let solver = C0LandscapeSolver::new(params).unwrap();
        let initial: Vec<_> = mesh
            .cell_center_km
            .iter()
            .map(|p| 2.0 + 0.01 * p.y)
            .collect();
        let mut state = C0LandscapeState::new(&mesh, initial).unwrap();
        let mut sample_times = Vec::new();
        let diagnostics = solver
            .step_with_forcing(&mesh, 0.01, &mut state, |time| {
                sample_times.push(time);
                frame(&mesh, 0.0)
            })
            .unwrap();
        assert!(diagnostics.operator_limits.accepted_dt_myr < 0.005);
        assert!(diagnostics.operator_limits.attempted_steps >= 2);
        assert_eq!(
            diagnostics.operator_limits.limiting_operator,
            C0TimestepLimiter::EffectiveDenudationAccuracy
        );
        assert!(sample_times.len() >= 2);
        assert_eq!(sample_times[0], 0.005);
        assert_eq!(
            *sample_times.last().unwrap(),
            0.5 * diagnostics.operator_limits.accepted_dt_myr
        );
    }

    #[test]
    fn tilted_plane_uses_analytic_outgoing_face_grade() {
        let mesh = mesh();
        let physical_slope = 0.01;
        let elevation: Vec<_> = mesh
            .cell_center_km
            .iter()
            .map(|point| 2.0 - physical_slope * point.y)
            .collect();
        let supply: Vec<_> = mesh.cell_area_km2.iter().map(|area| 5.0 * area).collect();
        let flow = FaceFlowCache::route_with_depressions(
            &mesh,
            &elevation,
            &supply,
            FlowPartition::MfdSlope,
        )
        .unwrap();
        let (grade, length) =
            face_consistent_routed_grade_and_length(&mesh, &elevation, &flow).unwrap();
        let center = mesh
            .cell_center_km
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.length_squared().total_cmp(&b.length_squared()))
            .unwrap()
            .0;
        // The two downhill pointy-hex faces are 30 degrees from the physical
        // gradient; both therefore have the same directional grade.
        let expected_grade = physical_slope * 3.0_f64.sqrt() * 0.5;
        assert!((grade[center] - expected_grade).abs() < 2.0e-10);
        assert!((length[center] - 8.0).abs() < 2.0e-6);
    }

    #[test]
    fn below_base_portal_route_does_not_invent_erosive_grade_or_pin_cell() {
        let mesh = mesh();
        let elevation = vec![-1.0; mesh.cell_count()];
        let supply: Vec<_> = mesh.cell_area_km2.iter().map(|area| 5.0 * area).collect();
        let flow = FaceFlowCache::route_with_depressions(
            &mesh,
            &elevation,
            &supply,
            FlowPartition::MfdSlope,
        )
        .unwrap();
        let (grade, _) = face_consistent_routed_grade_and_length(&mesh, &elevation, &flow).unwrap();
        assert!(mesh
            .boundary_faces
            .iter()
            .enumerate()
            .any(|(index, face)| matches!(
                face.condition,
                BoundaryFaceCondition::OpenBaseLevel { .. }
            ) && flow.boundary_face_flux_km3_myr[index] > 0.0));
        assert!(grade.iter().all(|value| *value == 0.0));

        let mut params = inactive_params();
        params.effective_areal_denudation.k = 100.0;
        let solver = C0LandscapeSolver::new(params).unwrap();
        let mut state = C0LandscapeState::new(&mesh, elevation.clone()).unwrap();
        solver
            .step(&mesh, &frame(&mesh, 0.0), 0.01, &mut state)
            .unwrap();
        assert_eq!(state.mean_bedrock_elevation_km, elevation);
    }

    #[test]
    fn slope_courant_rejection_resamples_midpoint() {
        let mesh = mesh();
        let initial: Vec<_> = mesh
            .cell_center_km
            .iter()
            .map(|point| 2.0 - 0.01 * point.y)
            .collect();
        let mut params = inactive_params();
        params.effective_areal_denudation.k = 100.0;
        params.maximum_effective_denudation_depth_km = 100.0;
        params.max_effective_denudation_courant = 1.0e-4;
        let solver = C0LandscapeSolver::new(params).unwrap();
        let mut state = C0LandscapeState::new(&mesh, initial).unwrap();
        let mut samples = Vec::new();
        let diagnostics = solver
            .step_with_forcing(&mesh, 0.01, &mut state, |time| {
                samples.push(time);
                frame(&mesh, 0.0)
            })
            .unwrap();
        assert_eq!(
            diagnostics.operator_limits.limiting_operator,
            C0TimestepLimiter::EffectiveDenudationSlopeCourant
        );
        let limit = diagnostics
            .operator_limits
            .effective_denudation_slope_cfl_limit_myr
            .unwrap();
        assert_eq!(diagnostics.operator_limits.accepted_dt_myr, limit);
        assert!(samples.len() >= 2);
        assert_eq!(samples[0], 0.005);
        assert_eq!(*samples.last().unwrap(), 0.5 * limit);
    }

    #[test]
    fn smooth_c0_solution_converges_under_temporal_refinement() {
        let mesh = mesh();
        let initial: Vec<_> = mesh
            .cell_center_km
            .iter()
            .map(|point| 2.0 - 0.006 * point.y + 0.02 * (0.08 * point.x).sin())
            .collect();
        let mut params = inactive_params();
        params.effective_areal_denudation.k = 0.1;
        params.maximum_effective_denudation_depth_km = 100.0;
        params.max_effective_denudation_courant = 1.0;
        let solver = C0LandscapeSolver::new(params).unwrap();
        let forcing = frame(&mesh, 0.0);
        let integrate = |steps: usize| {
            let mut state = C0LandscapeState::new(&mesh, initial.clone()).unwrap();
            let dt = 0.02 / steps as f64;
            for _ in 0..steps {
                let diagnostics = solver.step(&mesh, &forcing, dt, &mut state).unwrap();
                assert_eq!(diagnostics.operator_limits.accepted_dt_myr, dt);
            }
            state.mean_bedrock_elevation_km
        };
        let one = integrate(1);
        let two = integrate(2);
        let four = integrate(4);
        let rms = |a: &[f64], b: &[f64]| {
            (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
        };
        let coarse_change = rms(&one, &two);
        let fine_change = rms(&two, &four);
        assert!(coarse_change > 0.0);
        assert!(
            fine_change < coarse_change,
            "{fine_change} !< {coarse_change}"
        );
    }
}
