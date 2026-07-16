//! Compact numerical output for the disposable 4 km H/C/G discriminator.
//!
//! The output deliberately summarizes the already-retained physical surfaces
//! and common S0/D0/O0a evidence. It is JSON-friendly evidence for human
//! comparison, not a campaign packet, score, or promotion decision.

use super::organization_artifact::OrganizationArmV0;
use super::organization_comparison::{ThinArmCommonEvidenceV0, ThinHcgCommonEvidenceV0};
use super::organization_owner::{ThinG4KmObservationV0, ThinGQueueCountersV0};
use super::organization_owner_c::{ThinC4KmObservationV0, ThinCLimiterHistogramV0};
use super::organization_owner_h::{ThinH4KmObservationV0, ThinHLimiterHistogramV0};
use super::{
    reconstruct_mean_surface_gradient, LinkedResolutionInputV0, LinkedSharedInputBundleV0,
};
use crate::world::landforms::{
    BoundaryFaceRoleKindV0, EvaluationDrainageV0, HighlandMeasurementsV0, LandformRelationshipsV0,
    SurfaceHierarchyV0,
};
use serde::Serialize;
use std::fmt;

pub const THIN_HCG_NUMERICAL_OUTPUT_SCHEMA_V0: &str = "orogen-owner-thin-hcg-numerical-output-v0";
const TARGET_SPACING_KM: f64 = 4.0;
const PLATEAU_GENTLE_GRADE: f64 = 0.01;
const REFERENCE_SUMMIT_CAP_DEPTH_KM: f64 = 0.5;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ThinHcgNumericalOutputV0 {
    pub schema_version: String,
    pub warning: String,
    pub input_bundle_hash: u64,
    pub input_resolution_hash: u64,
    pub quantile_definition: String,
    pub h: ThinArmNumericalOutputV0,
    pub c: ThinArmNumericalOutputV0,
    pub g: ThinArmNumericalOutputV0,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ThinArmNumericalOutputV0 {
    pub arm: OrganizationArmV0,
    pub physical_elevation_component_hash: u64,
    pub elevation_km: AreaWeightedDistributionV0,
    pub physical_grade: AreaWeightedDistributionV0,
    pub broad_highland: BroadHighlandSummaryV0,
    pub surface_objects: SurfaceObjectSummaryV0,
    pub drainage: DrainageSummaryV0,
    pub relationships: RelationshipSummaryV0,
    pub numerical_work: ArmNumericalWorkV0,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AreaWeightedDistributionV0 {
    pub sample_count: u64,
    pub area_km2: f64,
    pub minimum: f64,
    pub p05: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub maximum: f64,
    pub mean: f64,
}

/// Direct surface measures intended to expose broad, high, gentle terrain.
/// The upper-quartile threshold is computed over physical land (`z > 0`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BroadHighlandSummaryV0 {
    pub total_area_km2: f64,
    pub land_area_km2: f64,
    pub land_area_fraction: f64,
    pub land_p75_elevation_km: Option<f64>,
    pub land_upper_quartile_area_km2: f64,
    pub land_upper_quartile_gentle_grade_threshold: f64,
    pub land_upper_quartile_gentle_area_km2: f64,
    pub land_upper_quartile_gentle_fraction: Option<f64>,
    /// Union of reference S0 footprints; nested footprints are counted once.
    pub reference_highland_union_area_km2: f64,
    pub reference_highland_union_fraction: f64,
    pub largest_reference_highland_area_km2: Option<f64>,
    /// Object-weighted aggregate; nested S0 caps can overlap physically.
    pub reference_cap_depth_km: f64,
    pub reference_cap_object_count: u64,
    pub reference_cap_summed_area_km2: f64,
    pub reference_cap_area_weighted_valid_grade_fraction: Option<f64>,
    pub reference_cap_area_weighted_gentle_fraction: Option<f64>,
    pub reference_cap_merge_censored_count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SurfaceObjectSummaryV0 {
    pub peak_count: u64,
    pub saddle_count: u64,
    pub root_count: u64,
    pub reference_highland_count: u64,
    pub persistence_low_count: u64,
    pub persistence_high_count: u64,
    pub footprint_low_count: u64,
    pub footprint_high_count: u64,
    /// Sum over S0 objects; nested footprints can overlap physically.
    pub reference_summed_footprint_area_km2: f64,
    pub largest_reference_footprint_area_km2: Option<f64>,
    pub reference_persistence_p50_km: Option<f64>,
    pub reference_persistence_p90_km: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DrainageSummaryV0 {
    pub depression_count: u64,
    /// Sum over depression objects; nested affected regions can overlap.
    pub depression_summed_affected_area_km2: f64,
    pub virtual_fill_volume_km3: f64,
    pub maximum_fill_depth_km: Option<f64>,
    pub portal_count: u64,
    pub structural_area_residual_km2: f64,
    pub supplied_runoff_residual: f64,
    pub scales: Vec<DrainageScaleSummaryV0>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DrainageScaleSummaryV0 {
    pub support_threshold_km2: f64,
    pub reach_count: u64,
    pub catchment_count: u64,
    pub portal_role_count: u64,
    pub raw_catchment_boundary_face_count: u64,
    pub raw_catchment_boundary_length_km: f64,
    pub total_reach_length_km: f64,
    pub longest_single_reach_km: Option<f64>,
    pub longest_portal_trunk_km: Option<f64>,
    pub greatest_supply_portal_trunk_km: Option<f64>,
    pub maximum_strahler_order: Option<u32>,
    pub largest_nested_catchment_area_km2: Option<f64>,
    pub largest_nested_catchment_area_fraction: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RelationshipSummaryV0 {
    pub backed_boundary_face_count: u64,
    pub lateral_boundary_candidate_count: u64,
    pub flow_transition_count: u64,
    pub backed_bilateral_descent_count: u64,
    pub highland_boundary_relationship_count: u64,
    pub saddle_boundary_association_count: u64,
    pub saddle_association_within_covering_radius_count: u64,
    pub saddle_association_bilateral_descent_count: u64,
    pub reach_cross_section_probe_count: u64,
    pub reach_cross_section_station_count: u64,
    pub reach_cross_section_available_span_count: u64,
    pub raw_boundary_faces_examined: u64,
    pub receiver_trace_segments: u64,
    pub regular_cross_section_samples: u64,
    pub candidate_face_tests: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ArmNumericalWorkV0 {
    H(HNumericalWorkV0),
    C(CNumericalWorkV0),
    G(GNumericalWorkV0),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HNumericalWorkV0 {
    pub completed_pass_count: u32,
    pub accepted_step_count: u64,
    pub candidate_attempt_count: u64,
    pub maximum_attempts_for_one_step: u32,
    pub minimum_accepted_dt_myr: Option<f64>,
    pub maximum_accepted_dt_myr: Option<f64>,
    pub limiter_histogram: ThinHLimiterHistogramV0,
    pub maximum_denudation_rate_km_myr: f64,
    pub maximum_hillslope_grade: f64,
    pub maximum_unresolved_discharge_cells: u64,
    pub solid_closure_error_km3: f64,
    pub integrated_water_balance_error_km3: f64,
    pub final_water_balance_error_km3_myr: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CNumericalWorkV0 {
    pub reached_time_myr: f64,
    pub accepted_step_count: u64,
    pub candidate_attempt_count: u64,
    pub maximum_attempts_for_one_step: u32,
    pub minimum_accepted_dt_myr: f64,
    pub maximum_accepted_dt_myr: f64,
    pub limiter_histogram: ThinCLimiterHistogramV0,
    pub maximum_denudation_rate_km_myr: f64,
    pub maximum_hillslope_grade: f64,
    pub maximum_unresolved_discharge_cells: u64,
    pub solid_closure_error_km3: f64,
    pub integrated_water_balance_error_km3: f64,
    pub final_water_balance_error_km3_myr: f64,
    pub control_maximum_displacement_error_km: f64,
    pub control_area_weighted_rms_displacement_error_km: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GNumericalWorkV0 {
    pub calibration_bracket_expansion_count: u32,
    pub calibration_iteration_count: u32,
    pub calibration_signed_volume_residual_km3: f64,
    pub queue: ThinGQueueCountersV0,
    pub reconstruction_moment_identity_error_km3: f64,
    pub work_volume_residual_km3: f64,
    pub runoff_balance_error_km3_myr: f64,
    pub area_balance_error_km2: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinNumericalOutputErrorV0(pub String);

impl fmt::Display for ThinNumericalOutputErrorV0 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ThinNumericalOutputErrorV0 {}

pub fn build_thin_hcg_numerical_output_v0(
    bundle: &LinkedSharedInputBundleV0,
    h: &ThinH4KmObservationV0,
    c: &ThinC4KmObservationV0,
    g: &ThinG4KmObservationV0,
    evidence: &ThinHcgCommonEvidenceV0,
) -> Result<ThinHcgNumericalOutputV0, ThinNumericalOutputErrorV0> {
    let input = bundle
        .resolutions
        .iter()
        .find(|value| value.nominal_spacing_km.to_bits() == TARGET_SPACING_KM.to_bits())
        .ok_or_else(|| fail("accepted bundle has no exact 4 km resolution"))?;
    if evidence.input_bundle_hash != bundle.derived_bundle_hash
        || evidence.input_resolution_hash != input.derived_resolution_hash
    {
        return Err(fail(
            "common evidence does not bind the accepted 4 km input",
        ));
    }

    let h = build_arm(
        input,
        OrganizationArmV0::H,
        h.final_elevation_component_hash,
        &h.final_elevation_km,
        &evidence.h,
        ArmNumericalWorkV0::H(h_work(h)),
    )?;
    let c = build_arm(
        input,
        OrganizationArmV0::C,
        c.final_elevation_component_hash,
        &c.final_elevation_km,
        &evidence.c,
        ArmNumericalWorkV0::C(c_work(c)),
    )?;
    let g = build_arm(
        input,
        OrganizationArmV0::G,
        g.final_elevation_component_hash,
        &g.final_elevation_km,
        &evidence.g,
        ArmNumericalWorkV0::G(g_work(g)),
    )?;

    Ok(ThinHcgNumericalOutputV0 {
        schema_version: THIN_HCG_NUMERICAL_OUTPUT_SCHEMA_V0.into(),
        warning: "DISPOSABLE COMPARISON: descriptive evidence, not a score or promotion result"
            .into(),
        input_bundle_hash: bundle.derived_bundle_hash,
        input_resolution_hash: input.derived_resolution_hash,
        quantile_definition:
            "left-continuous inverse of cumulative physical cell area; no interpolation".into(),
        h,
        c,
        g,
    })
}

fn build_arm(
    input: &LinkedResolutionInputV0,
    arm: OrganizationArmV0,
    elevation_hash: u64,
    elevation: &[f64],
    evidence: &ThinArmCommonEvidenceV0,
    numerical_work: ArmNumericalWorkV0,
) -> Result<ThinArmNumericalOutputV0, ThinNumericalOutputErrorV0> {
    if evidence.arm != arm || evidence.physical_elevation_component_hash != elevation_hash {
        return Err(fail(format!("{arm:?} common evidence binding mismatch")));
    }
    if elevation.len() != input.mesh.cell_count() {
        return Err(fail(format!("{arm:?} physical elevation length mismatch")));
    }
    let grade = reconstruct_mean_surface_gradient(&input.mesh, elevation).map_err(|error| {
        fail(format!(
            "{arm:?} physical grade reconstruction failed: {error}"
        ))
    })?;
    let elevation_distribution = weighted_distribution(elevation, &input.mesh.cell_area_km2)?;
    let grade_distribution = weighted_distribution(&grade.grade, &input.mesh.cell_area_km2)?;

    Ok(ThinArmNumericalOutputV0 {
        arm,
        physical_elevation_component_hash: elevation_hash,
        elevation_km: elevation_distribution,
        physical_grade: grade_distribution,
        broad_highland: broad_highland_summary(
            elevation,
            &grade.grade,
            &input.mesh.cell_area_km2,
            &evidence.surface_hierarchy,
        )?,
        surface_objects: surface_object_summary(&evidence.surface_hierarchy)?,
        drainage: drainage_summary(&evidence.drainage, input.summary.actual_domain_area_km2),
        relationships: relationship_summary(&evidence.relationships),
        numerical_work,
    })
}

fn broad_highland_summary(
    elevation: &[f64],
    grade: &[f64],
    area: &[f64],
    hierarchy: &SurfaceHierarchyV0,
) -> Result<BroadHighlandSummaryV0, ThinNumericalOutputErrorV0> {
    let total_area = compensated_sum(area.iter().copied());
    let land_samples = elevation
        .iter()
        .copied()
        .zip(area.iter().copied())
        .filter(|(z, _)| *z > 0.0)
        .collect::<Vec<_>>();
    let land_area = compensated_sum(land_samples.iter().map(|(_, a)| *a));
    let land_p75 = if land_samples.is_empty() {
        None
    } else {
        Some(weighted_quantile(&land_samples, 0.75)?)
    };
    let mut upper_area = 0.0;
    let mut upper_gentle_area = 0.0;
    if let Some(threshold) = land_p75 {
        for ((&z, &g), &a) in elevation.iter().zip(grade).zip(area) {
            if z > 0.0 && z >= threshold {
                upper_area += a;
                if g <= PLATEAU_GENTLE_GRADE {
                    upper_gentle_area += a;
                }
            }
        }
    }

    let mut reference_union = vec![false; elevation.len()];
    let mut largest_reference_area: Option<f64> = None;
    for &peak_id in &hierarchy.populations.reference {
        let peak = hierarchy
            .peaks
            .iter()
            .find(|peak| peak.id == peak_id)
            .ok_or_else(|| fail(format!("reference peak {peak_id} is absent")))?;
        largest_reference_area = Some(
            largest_reference_area.map_or(peak.footprint_area_km2, |old| {
                old.max(peak.footprint_area_km2)
            }),
        );
        for &cell in &peak.footprint_members {
            let member = reference_union
                .get_mut(cell as usize)
                .ok_or_else(|| fail(format!("reference peak {peak_id} has invalid member")))?;
            *member = true;
        }
    }
    let union_area = compensated_sum(
        reference_union
            .iter()
            .zip(area)
            .filter_map(|(&member, &area)| member.then_some(area)),
    );

    let mut cap_count = 0_u64;
    let mut cap_area = 0.0;
    let mut valid_area = 0.0;
    let mut gentle_area = 0.0;
    let mut merge_censored_count = 0_u64;
    for highland in &hierarchy.reference_highlands {
        let caps = match &highland.measurements {
            HighlandMeasurementsV0::Planar(value) => &value.summit_caps,
            HighlandMeasurementsV0::Spherical(value) => &value.summit_caps,
        };
        let cap = caps
            .iter()
            .find(|cap| cap.depth_km.to_bits() == REFERENCE_SUMMIT_CAP_DEPTH_KM.to_bits())
            .ok_or_else(|| fail("reference highland lacks the registered 0.5 km summit cap"))?;
        let gentle = cap
            .gentle_fractions
            .iter()
            .find(|value| value.grade_threshold.to_bits() == PLATEAU_GENTLE_GRADE.to_bits())
            .ok_or_else(|| fail("reference summit cap lacks the registered 0.01 grade"))?;
        cap_count += 1;
        cap_area += cap.area_km2;
        valid_area += cap.area_km2 * cap.valid_grade_fraction;
        gentle_area += cap.area_km2 * gentle.fraction;
        merge_censored_count += u64::from(cap.cap_merge_censored);
    }

    Ok(BroadHighlandSummaryV0 {
        total_area_km2: total_area,
        land_area_km2: land_area,
        land_area_fraction: ratio(land_area, total_area),
        land_p75_elevation_km: land_p75,
        land_upper_quartile_area_km2: upper_area,
        land_upper_quartile_gentle_grade_threshold: PLATEAU_GENTLE_GRADE,
        land_upper_quartile_gentle_area_km2: upper_gentle_area,
        land_upper_quartile_gentle_fraction: optional_ratio(upper_gentle_area, upper_area),
        reference_highland_union_area_km2: union_area,
        reference_highland_union_fraction: ratio(union_area, total_area),
        largest_reference_highland_area_km2: largest_reference_area,
        reference_cap_depth_km: REFERENCE_SUMMIT_CAP_DEPTH_KM,
        reference_cap_object_count: cap_count,
        reference_cap_summed_area_km2: cap_area,
        reference_cap_area_weighted_valid_grade_fraction: optional_ratio(valid_area, cap_area),
        reference_cap_area_weighted_gentle_fraction: optional_ratio(gentle_area, cap_area),
        reference_cap_merge_censored_count: merge_censored_count,
    })
}

fn surface_object_summary(
    hierarchy: &SurfaceHierarchyV0,
) -> Result<SurfaceObjectSummaryV0, ThinNumericalOutputErrorV0> {
    let mut areas = Vec::new();
    let mut persistence = Vec::new();
    for &peak_id in &hierarchy.populations.reference {
        let peak = hierarchy
            .peaks
            .iter()
            .find(|peak| peak.id == peak_id)
            .ok_or_else(|| fail(format!("reference peak {peak_id} is absent")))?;
        areas.push(peak.footprint_area_km2);
        persistence.push(peak.persistence_km);
    }
    let object_weight = vec![1.0; areas.len()];
    let persistence_samples = persistence
        .iter()
        .copied()
        .zip(object_weight.iter().copied())
        .collect::<Vec<_>>();
    Ok(SurfaceObjectSummaryV0 {
        peak_count: hierarchy.peaks.len() as u64,
        saddle_count: hierarchy.saddles.len() as u64,
        root_count: hierarchy.roots.len() as u64,
        reference_highland_count: hierarchy.populations.reference.len() as u64,
        persistence_low_count: hierarchy.populations.persistence_low.len() as u64,
        persistence_high_count: hierarchy.populations.persistence_high.len() as u64,
        footprint_low_count: hierarchy.populations.footprint_low.len() as u64,
        footprint_high_count: hierarchy.populations.footprint_high.len() as u64,
        reference_summed_footprint_area_km2: compensated_sum(areas.iter().copied()),
        largest_reference_footprint_area_km2: areas.iter().copied().max_by(f64::total_cmp),
        reference_persistence_p50_km: (!persistence_samples.is_empty())
            .then(|| weighted_quantile(&persistence_samples, 0.50))
            .transpose()?,
        reference_persistence_p90_km: (!persistence_samples.is_empty())
            .then(|| weighted_quantile(&persistence_samples, 0.90))
            .transpose()?,
    })
}

fn drainage_summary(drainage: &EvaluationDrainageV0, total_area: f64) -> DrainageSummaryV0 {
    let depression_area = compensated_sum(
        drainage
            .depressions
            .iter()
            .map(|value| value.affected_area_km2),
    );
    let fill_volume = compensated_sum(
        drainage
            .depressions
            .iter()
            .map(|value| value.virtual_fill_volume_km3),
    );
    let maximum_fill_depth = drainage
        .depressions
        .iter()
        .map(|value| value.maximum_fill_depth_km)
        .max_by(f64::total_cmp);
    let scales = drainage
        .scales
        .iter()
        .map(|scale| {
            let reaches = &scale.reach_graph.reaches;
            let reach_length = |id: u32| {
                reaches
                    .iter()
                    .find(|reach| reach.id == id)
                    .map(|reach| reach.physical_length_km)
                    .unwrap_or(0.0)
            };
            let longest_trunk = scale
                .reach_graph
                .portal_roles
                .iter()
                .map(|role| compensated_sum(role.longest_trunk.iter().map(|&id| reach_length(id))))
                .max_by(f64::total_cmp);
            let supply_trunk = scale
                .reach_graph
                .portal_roles
                .iter()
                .map(|role| {
                    compensated_sum(role.greatest_supply.iter().map(|&id| reach_length(id)))
                })
                .max_by(f64::total_cmp);
            let largest_catchment = scale
                .basin_graph
                .catchments
                .iter()
                .map(|value| value.nested_structural_area_km2)
                .max_by(f64::total_cmp);
            DrainageScaleSummaryV0 {
                support_threshold_km2: scale.support_threshold_km2,
                reach_count: reaches.len() as u64,
                catchment_count: scale.basin_graph.catchments.len() as u64,
                portal_role_count: scale.reach_graph.portal_roles.len() as u64,
                raw_catchment_boundary_face_count: scale.basin_graph.raw_catchment_boundaries.len()
                    as u64,
                raw_catchment_boundary_length_km: compensated_sum(
                    scale
                        .basin_graph
                        .raw_catchment_boundaries
                        .iter()
                        .map(|value| value.physical_length_km),
                ),
                total_reach_length_km: compensated_sum(
                    reaches.iter().map(|value| value.physical_length_km),
                ),
                longest_single_reach_km: reaches
                    .iter()
                    .map(|value| value.physical_length_km)
                    .max_by(f64::total_cmp),
                longest_portal_trunk_km: longest_trunk,
                greatest_supply_portal_trunk_km: supply_trunk,
                maximum_strahler_order: reaches.iter().map(|value| value.strahler_order).max(),
                largest_nested_catchment_area_km2: largest_catchment,
                largest_nested_catchment_area_fraction: largest_catchment
                    .and_then(|area| optional_ratio(area, total_area)),
            }
        })
        .collect();
    DrainageSummaryV0 {
        depression_count: drainage.depressions.len() as u64,
        depression_summed_affected_area_km2: depression_area,
        virtual_fill_volume_km3: fill_volume,
        maximum_fill_depth_km: maximum_fill_depth,
        portal_count: drainage.routing.portal_ledgers.len() as u64,
        structural_area_residual_km2: drainage.routing.structural_area_residual_km2,
        supplied_runoff_residual: drainage.routing.supplied_runoff_residual,
        scales,
    }
}

fn relationship_summary(value: &LandformRelationshipsV0) -> RelationshipSummaryV0 {
    let lateral = value
        .backed_boundary_faces
        .iter()
        .filter(|face| face.role == BoundaryFaceRoleKindV0::LateralBoundaryCandidate)
        .count() as u64;
    let flow = value.backed_boundary_faces.len() as u64 - lateral;
    let station_count = value
        .reach_cross_section_probes
        .iter()
        .map(|probe| probe.stations.len() as u64)
        .sum();
    RelationshipSummaryV0 {
        backed_boundary_face_count: value.backed_boundary_faces.len() as u64,
        lateral_boundary_candidate_count: lateral,
        flow_transition_count: flow,
        backed_bilateral_descent_count: value
            .backed_boundary_faces
            .iter()
            .filter(|face| {
                face.bilateral_descent
                    .as_ref()
                    .is_some_and(|result| result.bilateral_physical_descent)
            })
            .count() as u64,
        highland_boundary_relationship_count: value.highland_boundary_relationships.len() as u64,
        saddle_boundary_association_count: value.saddle_boundary_associations.len() as u64,
        saddle_association_within_covering_radius_count: value
            .saddle_boundary_associations
            .iter()
            .filter(|item| item.within_covering_radius == Some(true))
            .count() as u64,
        saddle_association_bilateral_descent_count: value
            .saddle_boundary_associations
            .iter()
            .filter(|item| {
                item.bilateral_descent
                    .as_ref()
                    .is_some_and(|result| result.bilateral_physical_descent)
            })
            .count() as u64,
        reach_cross_section_probe_count: value.reach_cross_section_probes.len() as u64,
        reach_cross_section_station_count: station_count,
        reach_cross_section_available_span_count: value
            .reach_cross_section_probes
            .iter()
            .flat_map(|probe| &probe.stations)
            .filter(|station| station.relative_relief_span_km.is_some())
            .count() as u64,
        raw_boundary_faces_examined: value.work_counts.raw_boundary_faces,
        receiver_trace_segments: value.work_counts.receiver_trace_segments,
        regular_cross_section_samples: value.work_counts.regular_cross_section_samples,
        candidate_face_tests: value.work_counts.candidate_face_tests,
    }
}

fn h_work(value: &ThinH4KmObservationV0) -> HNumericalWorkV0 {
    HNumericalWorkV0 {
        completed_pass_count: value.completion.completed_pass_count,
        accepted_step_count: value.completion.accepted_step_count,
        candidate_attempt_count: value.completion.total_candidate_attempt_count,
        maximum_attempts_for_one_step: value.completion.maximum_attempts_for_one_step,
        minimum_accepted_dt_myr: value.completion.minimum_accepted_dt_myr,
        maximum_accepted_dt_myr: value.completion.maximum_accepted_dt_myr,
        limiter_histogram: value.completion.limiter_histogram.clone(),
        maximum_denudation_rate_km_myr: value.maximum_effective_denudation_rate_km_myr,
        maximum_hillslope_grade: value.maximum_linear_hillslope_abs_grade,
        maximum_unresolved_discharge_cells: value.maximum_unresolved_specific_discharge_cell_count,
        solid_closure_error_km3: value.ledger.closure_error_km3,
        integrated_water_balance_error_km3: value.ledger.process.water.balance_error_km3,
        final_water_balance_error_km3_myr: value.final_routing.balance_error_km3_myr,
    }
}

fn c_work(value: &ThinC4KmObservationV0) -> CNumericalWorkV0 {
    CNumericalWorkV0 {
        reached_time_myr: value.completion.reached_time_myr,
        accepted_step_count: value.completion.accepted_step_count,
        candidate_attempt_count: value.completion.total_candidate_attempt_count,
        maximum_attempts_for_one_step: value.completion.maximum_attempts_for_one_step,
        minimum_accepted_dt_myr: value.completion.minimum_accepted_dt_myr,
        maximum_accepted_dt_myr: value.completion.maximum_accepted_dt_myr,
        limiter_histogram: value.completion.limiter_histogram.clone(),
        maximum_denudation_rate_km_myr: value.process.maximum_denudation_rate_km_myr,
        maximum_hillslope_grade: value.process.maximum_linear_hillslope_abs_grade,
        maximum_unresolved_discharge_cells: value
            .process
            .maximum_unresolved_specific_discharge_cells,
        solid_closure_error_km3: value.solid.closure_error_km3,
        integrated_water_balance_error_km3: value.water.balance_error_km3,
        final_water_balance_error_km3_myr: value.process.final_routing_water_balance_error_km3_myr,
        control_maximum_displacement_error_km: value.opportunity.maximum_displacement_error_km,
        control_area_weighted_rms_displacement_error_km: value
            .opportunity
            .area_weighted_rms_displacement_error_km,
    }
}

fn g_work(value: &ThinG4KmObservationV0) -> GNumericalWorkV0 {
    GNumericalWorkV0 {
        calibration_bracket_expansion_count: value.calibration.bracket_expansion_count,
        calibration_iteration_count: value.calibration.iteration_count,
        calibration_signed_volume_residual_km3: value.calibration.signed_volume_residual_km3,
        queue: value.queue.clone(),
        reconstruction_moment_identity_error_km3: value.ledger.moment_identity_error_km3,
        work_volume_residual_km3: value.ledger.work_volume_residual_km3,
        runoff_balance_error_km3_myr: value.ledger.runoff_balance_error_km3_myr,
        area_balance_error_km2: value.ledger.area_balance_error_km2,
    }
}

fn weighted_distribution(
    values: &[f64],
    weights: &[f64],
) -> Result<AreaWeightedDistributionV0, ThinNumericalOutputErrorV0> {
    if values.is_empty() || values.len() != weights.len() {
        return Err(fail("weighted distribution has empty or mismatched inputs"));
    }
    if values.iter().any(|value| !value.is_finite())
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return Err(fail("weighted distribution has invalid values or weights"));
    }
    let samples = values
        .iter()
        .copied()
        .zip(weights.iter().copied())
        .collect::<Vec<_>>();
    let area = compensated_sum(weights.iter().copied());
    let mean = compensated_sum(
        values
            .iter()
            .copied()
            .zip(weights.iter().copied())
            .map(|(value, weight)| value * weight),
    ) / area;
    Ok(AreaWeightedDistributionV0 {
        sample_count: values.len() as u64,
        area_km2: area,
        minimum: values.iter().copied().min_by(f64::total_cmp).unwrap(),
        p05: weighted_quantile(&samples, 0.05)?,
        p25: weighted_quantile(&samples, 0.25)?,
        p50: weighted_quantile(&samples, 0.50)?,
        p75: weighted_quantile(&samples, 0.75)?,
        p90: weighted_quantile(&samples, 0.90)?,
        p95: weighted_quantile(&samples, 0.95)?,
        p99: weighted_quantile(&samples, 0.99)?,
        maximum: values.iter().copied().max_by(f64::total_cmp).unwrap(),
        mean,
    })
}

fn weighted_quantile(
    samples: &[(f64, f64)],
    probability: f64,
) -> Result<f64, ThinNumericalOutputErrorV0> {
    if samples.is_empty() || !(0.0..=1.0).contains(&probability) {
        return Err(fail("invalid weighted quantile request"));
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    if sorted
        .iter()
        .any(|(value, weight)| !value.is_finite() || !weight.is_finite() || *weight <= 0.0)
    {
        return Err(fail("weighted quantile has invalid samples"));
    }
    let total = compensated_sum(sorted.iter().map(|(_, weight)| *weight));
    let target = probability * total;
    let mut cumulative = 0.0;
    for (index, (value, weight)) in sorted.iter().enumerate() {
        cumulative += weight;
        if cumulative >= target || index + 1 == sorted.len() {
            return Ok(*value);
        }
    }
    unreachable!("a nonempty weighted sample reaches its total weight")
}

fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for value in values {
        let adjusted = value - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    sum
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

fn optional_ratio(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator > 0.0).then_some(numerator / denominator)
}

fn fail(message: impl Into<String>) -> ThinNumericalOutputErrorV0 {
    ThinNumericalOutputErrorV0(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_distribution_uses_physical_area_without_interpolation() {
        let summary = weighted_distribution(&[10.0, 20.0, 30.0], &[1.0, 8.0, 1.0]).unwrap();
        assert_eq!(summary.area_km2, 10.0);
        assert_eq!(summary.p05, 10.0);
        assert_eq!(summary.p25, 20.0);
        assert_eq!(summary.p90, 20.0);
        assert_eq!(summary.p95, 30.0);
        assert_eq!(summary.mean, 20.0);
    }

    #[test]
    fn compensated_sum_retains_small_tail() {
        assert_eq!(compensated_sum([1.0e16, 1.0, -1.0e16]), 0.0);
        assert_eq!(compensated_sum([1.0e16, -1.0e16, 1.0]), 1.0);
    }
}
