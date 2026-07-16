//! Frozen active-process kernel shared by the thin H and C owners.
//!
//! This module deliberately owns only the numerical operator. Adaptive attempt
//! chronology, forcing, endpoint clipping, artifact encoding, and publication
//! remain responsibilities of the arm drivers.

use std::fmt;

use super::{BoundaryFaceCondition, FaceFlowCache, FlowPartition, LandscapeMesh};

const EFFECTIVE_DENUDATION_K_KM_INVERSE_V0: f64 = 1.0e-4;
const LINEAR_HILLSLOPE_DIFFUSIVITY_KM2_MYR_V0: f64 = 0.1;
const LINEAR_HILLSLOPE_TIMESTEP_SAFETY_V0: f64 = 0.4;
const MAXIMUM_EFFECTIVE_DENUDATION_DEPTH_KM_V0: f64 = 0.02;
const EFFECTIVE_DENUDATION_SLOPE_COURANT_V0: f64 = 0.25;
const SOLID_CLOSURE_ABSOLUTE_KM3_V0: f64 = 1.0e-8;
const WATER_CLOSURE_ABSOLUTE_KM3_MYR_V0: f64 = 1.0e-6;
const CLOSURE_RELATIVE_V0: f64 = 5.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProcessLimiterV0 {
    EffectiveDenudationSlopeCourant,
    EffectiveDenudationAccuracy,
    HillslopeStability,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ProcessPortalRateV0 {
    pub portal_id: u32,
    pub rate_km3_myr: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ProcessPortalVolumeV0 {
    pub portal_id: u32,
    pub volume_km3: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ProcessWaterRateV0 {
    pub total_supply_km3_myr: f64,
    pub portal_outflow_km3_myr: Vec<ProcessPortalRateV0>,
    pub total_portal_outflow_km3_myr: f64,
    pub unresolved_sink_rate_km3_myr: f64,
    pub balance_error_km3_myr: f64,
    pub unresolved_specific_discharge_cell_count: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ProcessStepV0 {
    pub final_elevation_km: Vec<f64>,
    pub effective_denudation_export_km3: f64,
    pub hillslope_portal_transfers_km3: Vec<ProcessPortalVolumeV0>,
    pub hillslope_portal_transfer_km3: f64,
    pub hillslope_internal_conservation_error_km3: f64,
    pub maximum_effective_denudation_rate_km_myr: f64,
    pub maximum_linear_hillslope_abs_grade: f64,
    pub initial_elevation_moment_km3: f64,
    pub final_elevation_moment_km3: f64,
    pub elevation_moment_change_km3: f64,
    pub expected_elevation_moment_change_km3: f64,
    pub process_solid_closure_error_km3: f64,
    pub water: ProcessWaterRateV0,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ProcessAttemptV0 {
    Accepted(ProcessStepV0),
    Retry {
        limiter: ProcessLimiterV0,
        limit_myr: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ThinProcessErrorV0(pub String);

impl fmt::Display for ThinProcessErrorV0 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ThinProcessErrorV0 {}

/// Attempt one frozen route-denude-linear-hillslope transaction.
///
/// A retry contains no surface or physical-ledger mutation. Equality with a
/// finite limiter is accepted; only a strictly larger candidate retries.
pub(crate) fn attempt_active_process_v0(
    mesh: &LandscapeMesh,
    elevation_after_uplift_km: &[f64],
    local_supply_km3_myr: &[f64],
    candidate_dt_myr: f64,
) -> Result<ProcessAttemptV0, ThinProcessErrorV0> {
    validate_inputs_v0(
        mesh,
        elevation_after_uplift_km,
        local_supply_km3_myr,
        Some(candidate_dt_myr),
    )?;

    let flow = FaceFlowCache::route_with_depressions(
        mesh,
        elevation_after_uplift_km,
        local_supply_km3_myr,
        FlowPartition::MfdSlope,
    )
    .map_err(|error| fail_v0(format!("active-process routing failed: {error}")))?;
    let water = water_diagnostics_v0(&flow)?;
    let (physical_grade, directional_length_km) =
        face_consistent_grade_and_length_v0(mesh, elevation_after_uplift_km, &flow)?;

    let slope_courant_limit_myr =
        slope_courant_limit_v0(&flow.specific_discharge_km2_myr, &directional_length_km)?;
    if let Some(limit_myr) = slope_courant_limit_myr {
        if candidate_dt_myr > limit_myr {
            return Ok(ProcessAttemptV0::Retry {
                limiter: ProcessLimiterV0::EffectiveDenudationSlopeCourant,
                limit_myr,
            });
        }
    }

    // The complete denudation candidate and its export are evaluated before
    // the realized-depth retry decision, preserving scratch-attempt ordering.
    let denudation = specialized_denudation_v0(
        mesh,
        elevation_after_uplift_km,
        &flow.specific_discharge_km2_myr,
        &physical_grade,
        candidate_dt_myr,
    )?;
    let denudation_depth_limit_myr = positive_limit_v0(
        MAXIMUM_EFFECTIVE_DENUDATION_DEPTH_KM_V0,
        denudation.maximum_rate_km_myr,
        "effective-denudation depth",
    )?;
    if let Some(limit_myr) = denudation_depth_limit_myr {
        if candidate_dt_myr > limit_myr {
            return Ok(ProcessAttemptV0::Retry {
                limiter: ProcessLimiterV0::EffectiveDenudationAccuracy,
                limit_myr,
            });
        }
    }

    let hillslope = linear_hillslope_plan_v0(mesh, &denudation.elevation_km)?;
    if let Some(limit_myr) = hillslope.stability_limit_myr {
        if candidate_dt_myr > limit_myr {
            return Ok(ProcessAttemptV0::Retry {
                limiter: ProcessLimiterV0::HillslopeStability,
                limit_myr,
            });
        }
    }
    let final_elevation_km = apply_linear_hillslope_plan_v0(
        mesh,
        &denudation.elevation_km,
        &hillslope,
        candidate_dt_myr,
    )?;

    let initial_elevation_moment_km3 = elevation_moment_v0(mesh, elevation_after_uplift_km)?;
    let final_elevation_moment_km3 = elevation_moment_v0(mesh, &final_elevation_km)?;
    let elevation_moment_change_km3 = final_elevation_moment_km3 - initial_elevation_moment_km3;
    let hillslope_portal_transfer_km3 = hillslope.total_portal_rate_km3_myr * candidate_dt_myr;
    require_finite_v0(
        hillslope_portal_transfer_km3,
        "linear-hillslope total portal transfer",
    )?;
    let expected_elevation_moment_change_km3 =
        0.0 - denudation.export_km3 - hillslope_portal_transfer_km3;
    let process_solid_closure_error_km3 =
        elevation_moment_change_km3 - expected_elevation_moment_change_km3;
    require_finite_v0(
        process_solid_closure_error_km3,
        "active-process solid closure",
    )?;
    if !close_v0(
        elevation_moment_change_km3,
        expected_elevation_moment_change_km3,
        SOLID_CLOSURE_ABSOLUTE_KM3_V0,
        CLOSURE_RELATIVE_V0,
    )? {
        return Err(fail_v0(format!(
            "active-process solid closure failed: actual {elevation_moment_change_km3}, expected {expected_elevation_moment_change_km3}"
        )));
    }

    let hillslope_portal_transfers_km3 = hillslope
        .portal_rates_km3_myr
        .iter()
        .map(|rate| ProcessPortalVolumeV0 {
            portal_id: rate.portal_id,
            volume_km3: rate.rate_km3_myr * candidate_dt_myr,
        })
        .collect::<Vec<_>>();
    if hillslope_portal_transfers_km3
        .iter()
        .any(|entry| !entry.volume_km3.is_finite())
    {
        return Err(fail_v0("linear-hillslope portal transfer is non-finite"));
    }
    let hillslope_internal_conservation_error_km3 =
        hillslope.internal_rate_error_km3_myr * candidate_dt_myr;
    require_finite_v0(
        hillslope_internal_conservation_error_km3,
        "linear-hillslope internal conservation",
    )?;

    Ok(ProcessAttemptV0::Accepted(ProcessStepV0 {
        final_elevation_km,
        effective_denudation_export_km3: denudation.export_km3,
        hillslope_portal_transfers_km3,
        hillslope_portal_transfer_km3,
        hillslope_internal_conservation_error_km3,
        maximum_effective_denudation_rate_km_myr: denudation.maximum_rate_km_myr,
        maximum_linear_hillslope_abs_grade: hillslope.maximum_abs_grade,
        initial_elevation_moment_km3,
        final_elevation_moment_km3,
        elevation_moment_change_km3,
        expected_elevation_moment_change_km3,
        process_solid_closure_error_km3,
        water,
    }))
}

/// Fresh final-surface routing evidence. This never writes physical elevation.
pub(crate) fn fresh_routing_diagnostics_v0(
    mesh: &LandscapeMesh,
    elevation_km: &[f64],
    local_supply_km3_myr: &[f64],
) -> Result<ProcessWaterRateV0, ThinProcessErrorV0> {
    validate_inputs_v0(mesh, elevation_km, local_supply_km3_myr, None)?;
    let flow = FaceFlowCache::route_with_depressions(
        mesh,
        elevation_km,
        local_supply_km3_myr,
        FlowPartition::MfdSlope,
    )
    .map_err(|error| fail_v0(format!("fresh routing failed: {error}")))?;
    water_diagnostics_v0(&flow)
}

/// Stored-cell-order physical elevation-volume moment.
pub(crate) fn elevation_moment_v0(
    mesh: &LandscapeMesh,
    elevation_km: &[f64],
) -> Result<f64, ThinProcessErrorV0> {
    if elevation_km.len() != mesh.cell_count() {
        return Err(fail_v0(format!(
            "elevation length {}, expected {}",
            elevation_km.len(),
            mesh.cell_count()
        )));
    }
    let mut moment_km3 = 0.0;
    for (cell, &elevation) in elevation_km.iter().enumerate() {
        require_finite_v0(elevation, "physical elevation")?;
        let contribution = elevation * mesh.cell_area_km2[cell];
        require_finite_v0(contribution, "elevation-moment contribution")?;
        moment_km3 += contribution;
        require_finite_v0(moment_km3, "elevation moment")?;
    }
    Ok(moment_km3)
}

#[derive(Debug)]
struct DenudationCandidateV0 {
    elevation_km: Vec<f64>,
    export_km3: f64,
    maximum_rate_km_myr: f64,
}

fn specialized_denudation_v0(
    mesh: &LandscapeMesh,
    elevation_km: &[f64],
    specific_discharge_km2_myr: &[f64],
    physical_grade: &[f64],
    dt_myr: f64,
) -> Result<DenudationCandidateV0, ThinProcessErrorV0> {
    let n = mesh.cell_count();
    if elevation_km.len() != n || specific_discharge_km2_myr.len() != n || physical_grade.len() != n
    {
        return Err(fail_v0("specialized denudation length mismatch"));
    }
    let mut candidate = Vec::with_capacity(n);
    let mut export_km3 = 0.0;
    let mut maximum_rate_km_myr: f64 = 0.0;
    for cell in 0..n {
        let q = specific_discharge_km2_myr[cell];
        let grade = physical_grade[cell];
        if !q.is_finite() || q < 0.0 || !grade.is_finite() || grade < 0.0 {
            return Err(fail_v0(format!("invalid denudation input at cell {cell}")));
        }
        // The registered m=n=1 law is intentionally specialized and contains
        // no powf call.
        let rate = (EFFECTIVE_DENUDATION_K_KM_INVERSE_V0 * q) * grade;
        if !rate.is_finite() || rate < 0.0 {
            return Err(fail_v0(format!("invalid denudation rate at cell {cell}")));
        }
        maximum_rate_km_myr = maximum_rate_km_myr.max(rate);
        let depth = rate * dt_myr;
        let value = elevation_km[cell] - depth;
        require_finite_v0(value, "post-denudation elevation")?;
        candidate.push(value);
        export_km3 += (rate * mesh.cell_area_km2[cell]) * dt_myr;
        require_finite_v0(export_km3, "denudation export")?;
    }
    Ok(DenudationCandidateV0 {
        elevation_km: candidate,
        export_km3,
        maximum_rate_km_myr,
    })
}

fn slope_courant_limit_v0(
    specific_discharge_km2_myr: &[f64],
    directional_length_km: &[f64],
) -> Result<Option<f64>, ThinProcessErrorV0> {
    if specific_discharge_km2_myr.len() != directional_length_km.len() {
        return Err(fail_v0("slope-Courant input length mismatch"));
    }
    let mut limit_myr = f64::INFINITY;
    for cell in 0..specific_discharge_km2_myr.len() {
        let q = specific_discharge_km2_myr[cell];
        let length = directional_length_km[cell];
        if !q.is_finite() || q < 0.0 {
            return Err(fail_v0(format!(
                "invalid specific discharge at cell {cell}"
            )));
        }
        if !(length.is_finite() || length == f64::INFINITY) || length <= 0.0 {
            return Err(fail_v0(format!(
                "invalid directional length at cell {cell}"
            )));
        }
        let slope_response = EFFECTIVE_DENUDATION_K_KM_INVERSE_V0 * q;
        if !slope_response.is_finite() || slope_response < 0.0 {
            return Err(fail_v0(format!("invalid slope response at cell {cell}")));
        }
        if slope_response > 0.0 && length.is_finite() {
            let candidate = (EFFECTIVE_DENUDATION_SLOPE_COURANT_V0 * length) / slope_response;
            if !candidate.is_finite() {
                return Err(fail_v0(format!(
                    "non-finite slope-Courant candidate at cell {cell}"
                )));
            }
            if candidate <= 0.0 {
                return Err(fail_v0(format!(
                    "non-positive slope-Courant candidate at cell {cell}"
                )));
            }
            limit_myr = limit_myr.min(candidate);
        }
    }
    Ok(limit_myr.is_finite().then_some(limit_myr))
}

fn face_consistent_grade_and_length_v0(
    mesh: &LandscapeMesh,
    physical_elevation_km: &[f64],
    flow: &FaceFlowCache,
) -> Result<(Vec<f64>, Vec<f64>), ThinProcessErrorV0> {
    if physical_elevation_km.len() != mesh.cell_count()
        || flow.directed_edge_flux_km3_myr.len() != mesh.edge_neighbor.len()
        || flow.boundary_face_flux_km3_myr.len() != mesh.boundary_faces.len()
    {
        return Err(fail_v0("face-flow cache does not match mesh geometry"));
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
            if !flux.is_finite() || flux < 0.0 {
                return Err(fail_v0(format!("invalid routed edge flux {edge}")));
            }
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
        if !flux.is_finite() || flux < 0.0 {
            return Err(fail_v0(format!(
                "invalid routed boundary flux {face_index}"
            )));
        }
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
            || directional_length_km[cell] <= 0.0
        {
            return Err(fail_v0(format!(
                "invalid routed grade geometry at cell {cell}"
            )));
        }
    }
    Ok((grade, directional_length_km))
}

#[derive(Debug)]
struct LinearHillslopePlanV0 {
    volume_rate_km3_myr: Vec<f64>,
    portal_rates_km3_myr: Vec<ProcessPortalRateV0>,
    total_portal_rate_km3_myr: f64,
    internal_rate_error_km3_myr: f64,
    maximum_abs_grade: f64,
    stability_limit_myr: Option<f64>,
}

fn linear_hillslope_plan_v0(
    mesh: &LandscapeMesh,
    elevation_km: &[f64],
) -> Result<LinearHillslopePlanV0, ThinProcessErrorV0> {
    if elevation_km.len() != mesh.cell_count() {
        return Err(fail_v0("linear-hillslope elevation length mismatch"));
    }
    let n = mesh.cell_count();
    let mut volume_rate_km3_myr = vec![0.0; n];
    let mut conductance_km2_myr = vec![0.0; n];
    let mut maximum_abs_grade: f64 = 0.0;

    for cell in 0..n {
        require_finite_v0(elevation_km[cell], "linear-hillslope elevation")?;
        let start = mesh.edge_offsets[cell] as usize;
        let end = mesh.edge_offsets[cell + 1] as usize;
        for edge in start..end {
            let neighbor = mesh.edge_neighbor[edge] as usize;
            if neighbor <= cell {
                continue;
            }
            let distance_km = f64::from(mesh.edge_distance_km[edge]);
            let width_km = f64::from(mesh.edge_face_width_km[edge]);
            let grade = (elevation_km[cell] - elevation_km[neighbor]) / distance_km;
            let rate = (LINEAR_HILLSLOPE_DIFFUSIVITY_KM2_MYR_V0 * grade) * width_km;
            require_finite_v0(rate, "linear-hillslope internal rate")?;
            volume_rate_km3_myr[cell] -= rate;
            volume_rate_km3_myr[neighbor] += rate;
            let conductance = (LINEAR_HILLSLOPE_DIFFUSIVITY_KM2_MYR_V0 * width_km) / distance_km;
            conductance_km2_myr[cell] += conductance;
            conductance_km2_myr[neighbor] += conductance;
            maximum_abs_grade = maximum_abs_grade.max(grade.abs());
        }
    }

    // Portal slots and the later total reduction are semantic portal-ID order.
    // Boundary-face contributions still accumulate in stored face order.
    let mut portal_rates_km3_myr = mesh
        .outlet_portals
        .iter()
        .map(|portal| ProcessPortalRateV0 {
            portal_id: portal.id.0,
            rate_km3_myr: 0.0,
        })
        .collect::<Vec<_>>();
    portal_rates_km3_myr.sort_unstable_by_key(|entry| entry.portal_id);
    if portal_rates_km3_myr
        .windows(2)
        .any(|pair| pair[0].portal_id == pair[1].portal_id)
    {
        return Err(fail_v0("duplicate portal ID in linear hillslopes"));
    }
    for face in &mesh.boundary_faces {
        let BoundaryFaceCondition::OpenBaseLevel {
            portal_id,
            elevation_km: base_level_km,
        } = face.condition
        else {
            continue;
        };
        let cell = face.cell as usize;
        let grade = (elevation_km[cell] - f64::from(base_level_km)) / face.center_distance_km;
        let rate = (LINEAR_HILLSLOPE_DIFFUSIVITY_KM2_MYR_V0 * grade) * face.width_km;
        require_finite_v0(rate, "linear-hillslope portal rate")?;
        volume_rate_km3_myr[cell] -= rate;
        let portal = portal_rates_km3_myr
            .binary_search_by_key(&portal_id.0, |entry| entry.portal_id)
            .map_err(|_| {
                fail_v0(format!(
                    "boundary face references unknown portal {portal_id:?}"
                ))
            })?;
        portal_rates_km3_myr[portal].rate_km3_myr += rate;
        require_finite_v0(
            portal_rates_km3_myr[portal].rate_km3_myr,
            "accumulated linear-hillslope portal rate",
        )?;
        conductance_km2_myr[cell] +=
            (LINEAR_HILLSLOPE_DIFFUSIVITY_KM2_MYR_V0 * face.width_km) / face.center_distance_km;
        maximum_abs_grade = maximum_abs_grade.max(grade.abs());
    }

    let mut total_portal_rate_km3_myr = 0.0;
    for portal in &portal_rates_km3_myr {
        total_portal_rate_km3_myr += portal.rate_km3_myr;
        require_finite_v0(
            total_portal_rate_km3_myr,
            "total linear-hillslope portal rate",
        )?;
    }
    let mut cell_rate_sum_km3_myr = 0.0;
    for &rate in &volume_rate_km3_myr {
        require_finite_v0(rate, "linear-hillslope cell rate")?;
        cell_rate_sum_km3_myr += rate;
        require_finite_v0(
            cell_rate_sum_km3_myr,
            "linear-hillslope cell-rate reduction",
        )?;
    }
    let internal_rate_error_km3_myr = cell_rate_sum_km3_myr + total_portal_rate_km3_myr;
    require_finite_v0(
        internal_rate_error_km3_myr,
        "linear-hillslope internal rate closure",
    )?;

    let mut stability_limit_myr = f64::INFINITY;
    for cell in 0..n {
        let conductance = conductance_km2_myr[cell];
        if !conductance.is_finite() || conductance < 0.0 {
            return Err(fail_v0(format!(
                "invalid linear-hillslope conductance at cell {cell}"
            )));
        }
        if conductance > 0.0 {
            let cell_limit =
                (LINEAR_HILLSLOPE_TIMESTEP_SAFETY_V0 * mesh.cell_area_km2[cell]) / conductance;
            if !cell_limit.is_finite() {
                return Err(fail_v0(format!(
                    "non-finite hillslope stability candidate at cell {cell}"
                )));
            }
            if cell_limit <= 0.0 {
                return Err(fail_v0(format!(
                    "non-positive hillslope stability candidate at cell {cell}"
                )));
            }
            stability_limit_myr = stability_limit_myr.min(cell_limit);
        }
    }

    Ok(LinearHillslopePlanV0 {
        volume_rate_km3_myr,
        portal_rates_km3_myr,
        total_portal_rate_km3_myr,
        internal_rate_error_km3_myr,
        maximum_abs_grade,
        stability_limit_myr: stability_limit_myr
            .is_finite()
            .then_some(stability_limit_myr),
    })
}

fn apply_linear_hillslope_plan_v0(
    mesh: &LandscapeMesh,
    elevation_km: &[f64],
    plan: &LinearHillslopePlanV0,
    dt_myr: f64,
) -> Result<Vec<f64>, ThinProcessErrorV0> {
    if elevation_km.len() != mesh.cell_count()
        || plan.volume_rate_km3_myr.len() != mesh.cell_count()
    {
        return Err(fail_v0("linear-hillslope candidate length mismatch"));
    }
    let mut candidate = Vec::with_capacity(mesh.cell_count());
    for cell in 0..mesh.cell_count() {
        let value = elevation_km[cell]
            + ((plan.volume_rate_km3_myr[cell] * dt_myr) / mesh.cell_area_km2[cell]);
        require_finite_v0(value, "linear-hillslope candidate elevation")?;
        candidate.push(value);
    }
    Ok(candidate)
}

fn water_diagnostics_v0(flow: &FaceFlowCache) -> Result<ProcessWaterRateV0, ThinProcessErrorV0> {
    let mut portal_outflow_km3_myr = flow
        .portal_outflow_km3_myr
        .iter()
        .map(|(portal_id, rate_km3_myr)| ProcessPortalRateV0 {
            portal_id: portal_id.0,
            rate_km3_myr: *rate_km3_myr,
        })
        .collect::<Vec<_>>();
    portal_outflow_km3_myr.sort_unstable_by_key(|entry| entry.portal_id);
    if portal_outflow_km3_myr
        .iter()
        .any(|entry| !entry.rate_km3_myr.is_finite() || entry.rate_km3_myr < 0.0)
    {
        return Err(fail_v0("invalid routed portal outflow"));
    }
    let mut total_portal_outflow_km3_myr = 0.0;
    for portal in &portal_outflow_km3_myr {
        total_portal_outflow_km3_myr += portal.rate_km3_myr;
        require_finite_v0(total_portal_outflow_km3_myr, "total portal outflow")?;
    }
    let unresolved_specific_discharge_cell_count = flow
        .available_supply_km3_myr
        .iter()
        .zip(&flow.local_supply_km3_myr)
        .zip(&flow.specific_discharge_km2_myr)
        .filter(|((available, local), q)| **available > **local && **q == 0.0)
        .count() as u64;
    let accounted_water_km3_myr = total_portal_outflow_km3_myr + flow.total_sink_storage_km3_myr;
    let balance_error_km3_myr = flow.total_supply_km3_myr - accounted_water_km3_myr;
    for (value, label) in [
        (flow.total_supply_km3_myr, "total water supply"),
        (
            flow.total_sink_storage_km3_myr,
            "unresolved sink-storage rate",
        ),
        (balance_error_km3_myr, "water balance error"),
    ] {
        require_finite_v0(value, label)?;
    }
    require_finite_v0(accounted_water_km3_myr, "accounted water rate")?;
    if !close_v0(
        flow.total_supply_km3_myr,
        accounted_water_km3_myr,
        WATER_CLOSURE_ABSOLUTE_KM3_MYR_V0,
        CLOSURE_RELATIVE_V0,
    )? {
        return Err(fail_v0(format!(
            "routing water closure failed: supply {}, accounted {}",
            flow.total_supply_km3_myr, accounted_water_km3_myr
        )));
    }
    Ok(ProcessWaterRateV0 {
        total_supply_km3_myr: flow.total_supply_km3_myr,
        portal_outflow_km3_myr,
        total_portal_outflow_km3_myr,
        unresolved_sink_rate_km3_myr: flow.total_sink_storage_km3_myr,
        balance_error_km3_myr,
        unresolved_specific_discharge_cell_count,
    })
}

fn positive_limit_v0(
    maximum_depth_km: f64,
    maximum_rate_km_myr: f64,
    label: &str,
) -> Result<Option<f64>, ThinProcessErrorV0> {
    if !maximum_rate_km_myr.is_finite() || maximum_rate_km_myr < 0.0 {
        return Err(fail_v0(format!("invalid {label} maximum rate")));
    }
    if maximum_rate_km_myr == 0.0 {
        return Ok(None);
    }
    let limit_myr = maximum_depth_km / maximum_rate_km_myr;
    if !limit_myr.is_finite() {
        return Err(fail_v0(format!("non-finite {label} limit")));
    }
    if limit_myr <= 0.0 {
        return Err(fail_v0(format!("non-positive {label} limit")));
    }
    Ok(Some(limit_myr))
}

fn validate_inputs_v0(
    mesh: &LandscapeMesh,
    elevation_km: &[f64],
    local_supply_km3_myr: &[f64],
    candidate_dt_myr: Option<f64>,
) -> Result<(), ThinProcessErrorV0> {
    mesh.validate()
        .map_err(|error| fail_v0(format!("invalid process mesh: {error}")))?;
    let n = mesh.cell_count();
    if elevation_km.len() != n {
        return Err(fail_v0(format!(
            "elevation length {}, expected {n}",
            elevation_km.len()
        )));
    }
    if local_supply_km3_myr.len() != n {
        return Err(fail_v0(format!(
            "supply length {}, expected {n}",
            local_supply_km3_myr.len()
        )));
    }
    if elevation_km.iter().any(|value| !value.is_finite()) {
        return Err(fail_v0("physical elevation must be finite"));
    }
    if local_supply_km3_myr
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(fail_v0(
            "stored local runoff supply must be finite and non-negative",
        ));
    }
    if let Some(dt_myr) = candidate_dt_myr {
        if !dt_myr.is_finite() || dt_myr <= 0.0 {
            return Err(fail_v0(
                "active-process candidate dt must be finite and positive",
            ));
        }
    }
    Ok(())
}

fn require_finite_v0(value: f64, label: &str) -> Result<(), ThinProcessErrorV0> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(fail_v0(format!("{label} is non-finite")))
    }
}

fn close_v0(
    actual: f64,
    expected: f64,
    absolute: f64,
    relative: f64,
) -> Result<bool, ThinProcessErrorV0> {
    if !actual.is_finite()
        || !expected.is_finite()
        || !absolute.is_finite()
        || !relative.is_finite()
        || absolute < 0.0
        || relative < 0.0
    {
        return Err(fail_v0("invalid closure predicate operand"));
    }
    let difference = (actual - expected).abs();
    let scale = actual.abs().max(expected.abs());
    Ok(difference <= absolute + (relative * scale))
}

fn fail_v0(message: impl Into<String>) -> ThinProcessErrorV0 {
    ThinProcessErrorV0(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::landscape::{BoundarySide, OutletPortal, OutletPortalId};

    fn mesh_v0() -> LandscapeMesh {
        LandscapeMesh::uniform_planar_hex(16.0, 12.0, 4.0).unwrap()
    }

    #[test]
    fn specialized_k_q_s_is_exact_and_simultaneous() {
        let mesh = mesh_v0();
        let n = mesh.cell_count();
        let elevation = vec![2.0; n];
        let mut q = vec![0.0; n];
        let mut grade = vec![0.0; n];
        q[0] = 8.0;
        grade[0] = 0.25;
        q[1] = 3.0;
        grade[1] = 0.5;
        let dt = 0.125;
        let result = specialized_denudation_v0(&mesh, &elevation, &q, &grade, dt).unwrap();
        let rate0 = (1.0e-4 * q[0]) * grade[0];
        let rate1 = (1.0e-4 * q[1]) * grade[1];
        assert_eq!(result.elevation_km[0], elevation[0] - rate0 * dt);
        assert_eq!(result.elevation_km[1], elevation[1] - rate1 * dt);
        assert_eq!(result.maximum_rate_km_myr, rate0.max(rate1));
        assert_eq!(
            result.export_km3,
            ((rate0 * mesh.cell_area_km2[0]) * dt) + ((rate1 * mesh.cell_area_km2[1]) * dt)
        );
        assert_eq!(elevation, vec![2.0; n]);
    }

    #[test]
    fn linear_hillslope_conserves_internal_exchange_and_orders_portals() {
        let portals = [
            OutletPortal {
                id: OutletPortalId(9),
                side: BoundarySide::North,
                span_start_km: -8.0,
                span_end_km: 8.0,
                base_level_km: 0.0,
            },
            OutletPortal {
                id: OutletPortalId(3),
                side: BoundarySide::South,
                span_start_km: -8.0,
                span_end_km: 8.0,
                base_level_km: 0.0,
            },
        ];
        let mesh =
            LandscapeMesh::uniform_planar_hex_with_portals(16.0, 12.0, 4.0, &portals).unwrap();
        let elevation: Vec<f64> = mesh
            .cell_center_km
            .iter()
            .map(|center| 1.0 + 0.01 * center.x + 0.02 * center.y)
            .collect::<Vec<_>>();
        let plan = linear_hillslope_plan_v0(&mesh, &elevation).unwrap();
        assert_eq!(
            plan.portal_rates_km3_myr
                .iter()
                .map(|entry| entry.portal_id)
                .collect::<Vec<_>>(),
            vec![3, 9]
        );
        let mut expected = vec![0.0; 2];
        for face in &mesh.boundary_faces {
            let BoundaryFaceCondition::OpenBaseLevel {
                portal_id,
                elevation_km: base,
            } = face.condition
            else {
                continue;
            };
            let grade = (elevation[face.cell as usize] - f64::from(base)) / face.center_distance_km;
            let rate = (0.1 * grade) * face.width_km;
            expected[usize::from(portal_id.0 == 9)] += rate;
        }
        assert_eq!(plan.portal_rates_km3_myr[0].rate_km3_myr, expected[0]);
        assert_eq!(plan.portal_rates_km3_myr[1].rate_km3_myr, expected[1]);
        assert!(plan.internal_rate_error_km3_myr.abs() < 1.0e-13);
    }

    #[test]
    fn linear_hillslope_accepts_stability_equality_and_rejects_next_up() {
        let mesh = mesh_v0();
        let elevation = mesh
            .cell_center_km
            .iter()
            .map(|center| 0.8 + 0.03 * center.x - 0.02 * center.y)
            .collect::<Vec<_>>();
        let supply = vec![0.0; mesh.cell_count()];
        let plan = linear_hillslope_plan_v0(&mesh, &elevation).unwrap();
        let limit = plan.stability_limit_myr.unwrap();
        let at_limit = attempt_active_process_v0(&mesh, &elevation, &supply, limit).unwrap();
        assert!(matches!(at_limit, ProcessAttemptV0::Accepted(_)));
        let next = f64::from_bits(limit.to_bits() + 1);
        assert_eq!(
            attempt_active_process_v0(&mesh, &elevation, &supply, next).unwrap(),
            ProcessAttemptV0::Retry {
                limiter: ProcessLimiterV0::HillslopeStability,
                limit_myr: limit,
            }
        );
    }

    #[test]
    fn depression_routing_never_mutates_physical_bedrock() {
        let mesh = mesh_v0();
        let elevation: Vec<f64> = mesh
            .cell_center_km
            .iter()
            .map(|center| {
                let radius2 = center.x * center.x + center.y * center.y;
                if radius2 < 20.0 {
                    0.1
                } else {
                    1.0
                }
            })
            .collect::<Vec<_>>();
        let before_bits = elevation
            .iter()
            .map(|value| (*value).to_bits())
            .collect::<Vec<_>>();
        let supply = vec![1.0; mesh.cell_count()];
        let diagnostics = fresh_routing_diagnostics_v0(&mesh, &elevation, &supply).unwrap();
        assert_eq!(
            before_bits,
            elevation
                .iter()
                .map(|value| (*value).to_bits())
                .collect::<Vec<_>>()
        );
        assert!(diagnostics.balance_error_km3_myr.abs() < 1.0e-10);
    }
}
