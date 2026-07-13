//! Path-local P0/M0 tracing and disclosed C0/F0 geometry for R1a.
//!
//! Tracing consumes one immutable registered route. The context indexes
//! boundary faces once; each arm then recomputes both ranks only at visited
//! cells and stops explicitly at the required semantic portal.

use std::collections::HashSet;

use glam::{DVec2, DVec3};

use super::{
    channel_extraction_r1a_fixture::rank_r1_case_cell, BoundaryFaceCondition, OutletPortalId,
    R1CaseConfig, R1CaseError, R1LocalRankObservation, R1RankTieDecision, R1RegisteredCase,
    R1SelectedFace, VoronoiCapFixture, R1_CAP_HEIGHT_KM, R1_CAP_PORTAL_ID, R1_HEAD_ALONG_TRACK_KM,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum R1ReceiverArm {
    P0PhysicalGrade,
    M0MfdFraction,
}

#[derive(Debug, Clone, PartialEq)]
pub struct R1TraceStep {
    pub cell: usize,
    pub ranks: R1LocalRankObservation,
    pub selected_face: R1SelectedFace,
}

#[derive(Debug, Clone, PartialEq)]
pub struct R1PathMetrics {
    pub vertex_count: usize,
    pub minimum_cross_track_km: f64,
    pub maximum_cross_track_km: f64,
    pub maximum_absolute_cross_track_km: f64,
    pub polyline_arclength_km: f64,
    pub relative_arclength_error: f64,
    pub total_backtracking_km: f64,
    pub outlet_midpoint_error_km: f64,
    pub terminal_along_track_km: f64,
    pub terminal_cross_track_km: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct R1PathGeometry {
    pub vertices_km: Vec<DVec3>,
    pub metrics: R1PathMetrics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct R1TracedPath {
    pub arm: R1ReceiverArm,
    pub head_index: usize,
    pub required_portal: OutletPortalId,
    pub steps: Vec<R1TraceStep>,
    pub cell_count: usize,
    pub internal_face_count: usize,
    pub selected_face_count: usize,
    pub exact_score_tie_selections: usize,
    pub build_index_tie_selections: usize,
    pub c0: R1PathGeometry,
    pub f0: R1PathGeometry,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum R1TraceTermination {
    Sink { cell: usize },
    Cycle { cell: usize },
    CellCountGuard,
}

#[derive(Debug, Clone, PartialEq)]
pub struct R1FailedTrace {
    pub arm: R1ReceiverArm,
    pub head_index: usize,
    pub required_portal: OutletPortalId,
    pub termination: R1TraceTermination,
    pub steps: Vec<R1TraceStep>,
    pub visited_cell_count: usize,
    pub internal_face_count: usize,
    pub selected_face_count: usize,
    pub exact_score_tie_selections: usize,
    pub build_index_tie_selections: usize,
    pub partial_c0_vertices_km: Vec<DVec3>,
    pub partial_f0_vertices_km: Vec<DVec3>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum R1TraceOutcome {
    ReachedPortal(R1TracedPath),
    Failed(R1FailedTrace),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct R1TraceContext {
    boundary_faces_by_cell: Vec<Vec<usize>>,
    cell_count: usize,
    boundary_face_count: usize,
}

impl R1TraceContext {
    pub fn new(cap: &VoronoiCapFixture) -> Result<Self, R1CaseError> {
        let cell_count = cap.mesh.cell_count();
        let mut boundary_faces_by_cell = vec![Vec::new(); cell_count];
        for (face_index, face) in cap.mesh.boundary_faces.iter().enumerate() {
            let cell = face.cell as usize;
            let Some(owner_faces) = boundary_faces_by_cell.get_mut(cell) else {
                return Err(R1CaseError(format!(
                    "boundary face {face_index} has invalid owner {cell}"
                )));
            };
            owner_faces.push(face_index);
        }
        Ok(Self {
            boundary_faces_by_cell,
            cell_count,
            boundary_face_count: cap.mesh.boundary_faces.len(),
        })
    }

    pub fn indexed_boundary_face_count(&self) -> usize {
        self.boundary_face_count
    }

    fn validate_cap(&self, cap: &VoronoiCapFixture) -> Result<(), R1CaseError> {
        if self.cell_count != cap.mesh.cell_count()
            || self.boundary_face_count != cap.mesh.boundary_faces.len()
        {
            return Err(R1CaseError(
                "trace context is incompatible with the supplied cap".into(),
            ));
        }
        Ok(())
    }
}

pub fn trace_r1_path(
    cap: &VoronoiCapFixture,
    case: &R1RegisteredCase,
    context: &R1TraceContext,
    head_index: usize,
    arm: R1ReceiverArm,
    required_portal: OutletPortalId,
) -> Result<R1TraceOutcome, R1CaseError> {
    context.validate_cap(cap)?;
    validate_trace_inputs(cap, case, head_index, required_portal)?;
    let head = &case.heads[head_index];
    let mut current = head.cell;
    let mut seen = HashSet::new();
    let mut steps = Vec::new();
    let mut c0_vertices = Vec::new();
    let mut f0_vertices = vec![DVec3::new(head.point_km.x, head.point_km.y, 0.0)];
    let mut exact_score_tie_selections = 0usize;
    let mut build_index_tie_selections = 0usize;

    loop {
        if current >= cap.mesh.cell_count() {
            return Err(R1CaseError(format!("trace reached invalid cell {current}")));
        }
        if !seen.insert(current) {
            return Ok(R1TraceOutcome::Failed(failed_trace(
                arm,
                head_index,
                required_portal,
                R1TraceTermination::Cycle { cell: current },
                steps,
                seen.len(),
                exact_score_tie_selections,
                build_index_tie_selections,
                c0_vertices,
                f0_vertices,
            )));
        }
        if steps.len() >= cap.mesh.cell_count() {
            return Ok(R1TraceOutcome::Failed(failed_trace(
                arm,
                head_index,
                required_portal,
                R1TraceTermination::CellCountGuard,
                steps,
                seen.len(),
                exact_score_tie_selections,
                build_index_tie_selections,
                c0_vertices,
                f0_vertices,
            )));
        }
        c0_vertices.push(cap.mesh.cell_center_km[current]);

        let recomputed_ranks =
            rank_r1_case_cell(cap, case, &context.boundary_faces_by_cell[current], current)?;
        if case.local_ranks.get(current) != Some(&recomputed_ranks) {
            return Err(R1CaseError(format!(
                "path-local ranks at cell {current} disagree with the immutable case audit"
            )));
        }
        let Some(ranks) = recomputed_ranks else {
            return Ok(R1TraceOutcome::Failed(failed_trace(
                arm,
                head_index,
                required_portal,
                R1TraceTermination::Sink { cell: current },
                steps,
                seen.len(),
                exact_score_tie_selections,
                build_index_tie_selections,
                c0_vertices,
                f0_vertices,
            )));
        };
        let active_rank = match arm {
            R1ReceiverArm::P0PhysicalGrade => &ranks.p0_physical_grade,
            R1ReceiverArm::M0MfdFraction => &ranks.m0_mfd_fraction,
        };
        if !active_rank.score.is_finite() || active_rank.score <= 0.0 {
            return Err(R1CaseError(format!(
                "cell {current} selected a non-positive or non-finite receiver"
            )));
        }
        if !matches!(
            active_rank.tie_decision,
            R1RankTieDecision::SoleEligibleFace | R1RankTieDecision::Score
        ) {
            exact_score_tie_selections += 1;
        }
        if active_rank.tie_decision == R1RankTieDecision::CombinedFaceIndex {
            build_index_tie_selections += 1;
        }

        let selected_face = active_rank.face;
        match selected_face {
            R1SelectedFace::Internal {
                directed_edge,
                neighbor,
            } => {
                validate_internal_selection(cap, case, current, directed_edge, neighbor, arm)?;
                let midpoint = cap.edge_face_midpoint_km[directed_edge];
                if midpoint != active_rank.midpoint_km {
                    return Err(R1CaseError(format!(
                        "cell {current} rank midpoint disagrees with edge {directed_edge}"
                    )));
                }
                f0_vertices.push(midpoint);
                steps.push(R1TraceStep {
                    cell: current,
                    ranks,
                    selected_face,
                });
                if seen.contains(&neighbor) {
                    return Ok(R1TraceOutcome::Failed(failed_trace(
                        arm,
                        head_index,
                        required_portal,
                        R1TraceTermination::Cycle { cell: neighbor },
                        steps,
                        seen.len(),
                        exact_score_tie_selections,
                        build_index_tie_selections,
                        c0_vertices,
                        f0_vertices,
                    )));
                }
                current = neighbor;
            }
            R1SelectedFace::Portal {
                boundary_face,
                portal_id,
            } => {
                let midpoint = validate_portal_selection(
                    cap,
                    case,
                    current,
                    boundary_face,
                    portal_id,
                    arm,
                    required_portal,
                )?;
                if midpoint != active_rank.midpoint_km {
                    return Err(R1CaseError(format!(
                        "cell {current} rank midpoint disagrees with portal face {boundary_face}"
                    )));
                }
                c0_vertices.push(midpoint);
                f0_vertices.push(midpoint);
                steps.push(R1TraceStep {
                    cell: current,
                    ranks,
                    selected_face,
                });
                break;
            }
        }
    }

    let cell_count = steps.len();
    let selected_face_count = steps.len();
    let internal_face_count = selected_face_count.saturating_sub(1);
    let c0 = measure_geometry(case.config, c0_vertices)?;
    let f0 = measure_geometry(case.config, f0_vertices)?;
    Ok(R1TraceOutcome::ReachedPortal(R1TracedPath {
        arm,
        head_index,
        required_portal,
        steps,
        cell_count,
        internal_face_count,
        selected_face_count,
        exact_score_tie_selections,
        build_index_tie_selections,
        c0,
        f0,
    }))
}

#[allow(clippy::too_many_arguments)]
fn failed_trace(
    arm: R1ReceiverArm,
    head_index: usize,
    required_portal: OutletPortalId,
    termination: R1TraceTermination,
    steps: Vec<R1TraceStep>,
    visited_cell_count: usize,
    exact_score_tie_selections: usize,
    build_index_tie_selections: usize,
    partial_c0_vertices_km: Vec<DVec3>,
    partial_f0_vertices_km: Vec<DVec3>,
) -> R1FailedTrace {
    let selected_face_count = steps.len();
    R1FailedTrace {
        arm,
        head_index,
        required_portal,
        termination,
        internal_face_count: selected_face_count,
        selected_face_count,
        steps,
        visited_cell_count,
        exact_score_tie_selections,
        build_index_tie_selections,
        partial_c0_vertices_km,
        partial_f0_vertices_km,
    }
}

fn validate_trace_inputs(
    cap: &VoronoiCapFixture,
    case: &R1RegisteredCase,
    head_index: usize,
    required_portal: OutletPortalId,
) -> Result<(), R1CaseError> {
    if required_portal != R1_CAP_PORTAL_ID {
        return Err(R1CaseError(format!(
            "R1a paths require portal {}, got {}",
            R1_CAP_PORTAL_ID.0, required_portal.0
        )));
    }
    if case.config.spacing_km != cap.config.spacing_km
        || case.local_ranks.len() != cap.mesh.cell_count()
        || case.elevation_km.len() != cap.mesh.cell_count()
        || case.local_supply_km3_myr.len() != cap.mesh.cell_count()
    {
        return Err(R1CaseError(
            "registered case is incompatible with the supplied cap".into(),
        ));
    }
    if head_index >= case.heads.len() {
        return Err(R1CaseError(format!(
            "head index {head_index} is out of range"
        )));
    }
    Ok(())
}

fn validate_internal_selection(
    cap: &VoronoiCapFixture,
    case: &R1RegisteredCase,
    cell: usize,
    edge: usize,
    neighbor: usize,
    arm: R1ReceiverArm,
) -> Result<(), R1CaseError> {
    let start = cap.mesh.edge_offsets[cell] as usize;
    let end = cap.mesh.edge_offsets[cell + 1] as usize;
    if !(start..end).contains(&edge)
        || cap
            .mesh
            .edge_neighbor
            .get(edge)
            .map(|&value| value as usize)
            != Some(neighbor)
    {
        return Err(R1CaseError(format!(
            "cell {cell} selected foreign or malformed edge {edge}"
        )));
    }
    let grade = (case.elevation_km[cell] - case.elevation_km[neighbor])
        / f64::from(cap.mesh.edge_distance_km[edge]);
    let fraction = case.flow.directed_edge_fraction[edge];
    let score = match arm {
        R1ReceiverArm::P0PhysicalGrade => grade,
        R1ReceiverArm::M0MfdFraction => fraction,
    };
    let stored = match arm {
        R1ReceiverArm::P0PhysicalGrade => case.local_ranks[cell]
            .as_ref()
            .map(|ranks| ranks.p0_physical_grade.score),
        R1ReceiverArm::M0MfdFraction => case.local_ranks[cell]
            .as_ref()
            .map(|ranks| ranks.m0_mfd_fraction.score),
    };
    if grade <= 0.0 || fraction <= 0.0 || stored != Some(score) {
        return Err(R1CaseError(format!(
            "cell {cell} selected edge {edge} that is not strictly eligible for {arm:?}"
        )));
    }
    Ok(())
}

fn validate_portal_selection(
    cap: &VoronoiCapFixture,
    case: &R1RegisteredCase,
    cell: usize,
    face_index: usize,
    portal_id: OutletPortalId,
    arm: R1ReceiverArm,
    required_portal: OutletPortalId,
) -> Result<DVec3, R1CaseError> {
    let face = cap
        .mesh
        .boundary_faces
        .get(face_index)
        .ok_or_else(|| R1CaseError(format!("invalid portal face {face_index}")))?;
    if face.cell as usize != cell || portal_id != required_portal {
        return Err(R1CaseError(format!(
            "cell {cell} selected a foreign or wrong portal face {face_index}"
        )));
    }
    let BoundaryFaceCondition::OpenBaseLevel {
        portal_id: actual_portal,
        elevation_km: base,
    } = face.condition
    else {
        return Err(R1CaseError(format!(
            "cell {cell} selected closed boundary face {face_index}"
        )));
    };
    if actual_portal != required_portal {
        return Err(R1CaseError(format!(
            "boundary face {face_index} belongs to portal {} instead of {}",
            actual_portal.0, required_portal.0
        )));
    }
    let grade = (case.elevation_km[cell] - f64::from(base)) / face.center_distance_km;
    let fraction = case.flow.boundary_face_fraction[face_index];
    let score = match arm {
        R1ReceiverArm::P0PhysicalGrade => grade,
        R1ReceiverArm::M0MfdFraction => fraction,
    };
    let stored = match arm {
        R1ReceiverArm::P0PhysicalGrade => case.local_ranks[cell]
            .as_ref()
            .map(|ranks| ranks.p0_physical_grade.score),
        R1ReceiverArm::M0MfdFraction => case.local_ranks[cell]
            .as_ref()
            .map(|ranks| ranks.m0_mfd_fraction.score),
    };
    if grade <= 0.0 || fraction <= 0.0 || stored != Some(score) {
        return Err(R1CaseError(format!(
            "cell {cell} selected portal face {face_index} that is not strictly eligible"
        )));
    }
    Ok(face.center_km)
}

fn measure_geometry(
    config: R1CaseConfig,
    vertices_km: Vec<DVec3>,
) -> Result<R1PathGeometry, R1CaseError> {
    if vertices_km.len() < 2 || vertices_km.iter().any(|point| !point.is_finite()) {
        return Err(R1CaseError(
            "path geometry requires at least two finite vertices".into(),
        ));
    }
    let (outlet, along, transverse) = trace_frame(config);
    let coordinates: Vec<_> = vertices_km
        .iter()
        .map(|point| {
            let relative = DVec2::new(point.x, point.y) - outlet;
            (relative.dot(along), relative.dot(transverse))
        })
        .collect();
    let minimum_cross_track_km = coordinates
        .iter()
        .map(|(_, n)| *n)
        .fold(f64::INFINITY, f64::min);
    let maximum_cross_track_km = coordinates
        .iter()
        .map(|(_, n)| *n)
        .fold(f64::NEG_INFINITY, f64::max);
    let maximum_absolute_cross_track_km = minimum_cross_track_km
        .abs()
        .max(maximum_cross_track_km.abs());
    let polyline_arclength_km: f64 = vertices_km
        .windows(2)
        .map(|segment| segment[0].distance(segment[1]))
        .sum();
    let total_backtracking_km: f64 = coordinates
        .windows(2)
        .map(|segment| (segment[1].0 - segment[0].0).max(0.0))
        .sum();
    let (terminal_along_track_km, terminal_cross_track_km) = coordinates
        .last()
        .copied()
        .ok_or_else(|| R1CaseError("path geometry has no endpoint".into()))?;
    let outlet_midpoint_error_km = terminal_along_track_km.hypot(terminal_cross_track_km);
    let metrics = R1PathMetrics {
        vertex_count: vertices_km.len(),
        minimum_cross_track_km,
        maximum_cross_track_km,
        maximum_absolute_cross_track_km,
        polyline_arclength_km,
        relative_arclength_error: (polyline_arclength_km - R1_HEAD_ALONG_TRACK_KM).abs()
            / R1_HEAD_ALONG_TRACK_KM,
        total_backtracking_km,
        outlet_midpoint_error_km,
        terminal_along_track_km,
        terminal_cross_track_km,
    };
    Ok(R1PathGeometry {
        vertices_km,
        metrics,
    })
}

fn trace_frame(config: R1CaseConfig) -> (DVec2, DVec2, DVec2) {
    let along = DVec2::new(config.theta_rad.sin(), config.theta_rad.cos());
    let transverse = DVec2::new(-config.theta_rad.cos(), config.theta_rad.sin());
    let outlet = DVec2::new(config.outlet_offset_km, -0.5 * R1_CAP_HEIGHT_KM);
    (outlet, along, transverse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::landscape::{
        build_r1_registered_case, build_r1_voronoi_cap, R1CaseConfig, R1SurfaceKind,
        VoronoiCapConfig,
    };

    #[test]
    fn path_metrics_use_frozen_frame_length_and_backtracking() {
        let config = R1CaseConfig {
            spacing_km: 8.0,
            theta_rad: 0.0,
            outlet_offset_km: 0.0,
            surface: R1SurfaceKind::Valley,
        };
        let geometry = measure_geometry(
            config,
            vec![
                DVec3::new(0.0, 64.0, 0.0),
                DVec3::new(0.0, 50.0, 0.0),
                DVec3::new(0.0, 55.0, 0.0),
                DVec3::new(1.0, -112.0, 0.0),
            ],
        )
        .unwrap();
        assert_eq!(geometry.metrics.minimum_cross_track_km, -1.0);
        assert_eq!(geometry.metrics.maximum_cross_track_km, 0.0);
        assert_eq!(geometry.metrics.maximum_absolute_cross_track_km, 1.0);
        assert_eq!(geometry.metrics.total_backtracking_km, 5.0);
        assert_eq!(geometry.metrics.outlet_midpoint_error_km, 1.0);
        assert_eq!(geometry.metrics.terminal_along_track_km, 0.0);
        assert_eq!(geometry.metrics.terminal_cross_track_km, -1.0);
    }

    #[test]
    fn eight_km_valley_paths_are_immutable_and_bit_deterministic() {
        let cap = build_r1_voronoi_cap(VoronoiCapConfig::r1(8.0)).unwrap();
        let case = build_r1_registered_case(
            &cap,
            R1CaseConfig {
                spacing_km: 8.0,
                theta_rad: 0.0,
                outlet_offset_km: 0.0,
                surface: R1SurfaceKind::Valley,
            },
        )
        .unwrap();
        let context = R1TraceContext::new(&cap).unwrap();
        let cap_before = cap.clone();
        let case_before = case.clone();
        for arm in [R1ReceiverArm::P0PhysicalGrade, R1ReceiverArm::M0MfdFraction] {
            let first_outcome =
                trace_r1_path(&cap, &case, &context, 0, arm, R1_CAP_PORTAL_ID).unwrap();
            let second_outcome =
                trace_r1_path(&cap, &case, &context, 0, arm, R1_CAP_PORTAL_ID).unwrap();
            assert_eq!(first_outcome, second_outcome);
            let R1TraceOutcome::ReachedPortal(first) = first_outcome else {
                panic!("registered valley path did not reach portal")
            };
            assert_eq!(first.selected_face_count, first.cell_count);
            assert_eq!(first.internal_face_count + 1, first.selected_face_count);
            assert_eq!(first.c0.vertices_km.len(), first.cell_count + 1);
            assert_eq!(first.f0.vertices_km.len(), first.selected_face_count + 1);
            assert_eq!(first.build_index_tie_selections, 0);
        }
        assert_eq!(cap, cap_before);
        assert_eq!(case, case_before);
        assert_eq!(context.indexed_boundary_face_count(), 256);
    }

    #[test]
    fn affine_sink_retains_a_deterministic_auditable_prefix() {
        let cap = build_r1_voronoi_cap(VoronoiCapConfig::r1(8.0)).unwrap();
        let case = build_r1_registered_case(
            &cap,
            R1CaseConfig {
                spacing_km: 8.0,
                theta_rad: 0.31,
                outlet_offset_km: 0.0,
                surface: R1SurfaceKind::Affine,
            },
        )
        .unwrap();
        let context = R1TraceContext::new(&cap).unwrap();
        let first = trace_r1_path(
            &cap,
            &case,
            &context,
            0,
            R1ReceiverArm::P0PhysicalGrade,
            R1_CAP_PORTAL_ID,
        )
        .unwrap();
        let second = trace_r1_path(
            &cap,
            &case,
            &context,
            0,
            R1ReceiverArm::P0PhysicalGrade,
            R1_CAP_PORTAL_ID,
        )
        .unwrap();
        assert_eq!(first, second);
        let R1TraceOutcome::Failed(failed) = first else {
            panic!("registered rotated affine control unexpectedly reached its portal")
        };
        let R1TraceTermination::Sink { cell } = failed.termination else {
            panic!("registered rotated affine control failed for the wrong reason")
        };
        assert!(!failed.steps.is_empty());
        assert_eq!(failed.steps.len(), failed.selected_face_count);
        assert_eq!(failed.internal_face_count, failed.selected_face_count);
        assert_eq!(
            failed.partial_c0_vertices_km.len(),
            failed.visited_cell_count
        );
        assert_eq!(
            failed.partial_f0_vertices_km.len(),
            failed.selected_face_count + 1
        );
        assert_eq!(failed.build_index_tie_selections, 0);

        let mut inconsistent = case.clone();
        inconsistent.local_ranks[cell] = inconsistent.local_ranks[case.heads[0].cell].clone();
        assert!(trace_r1_path(
            &cap,
            &inconsistent,
            &context,
            0,
            R1ReceiverArm::P0PhysicalGrade,
            R1_CAP_PORTAL_ID,
        )
        .is_err());
        assert!(trace_r1_path(
            &cap,
            &case,
            &context,
            0,
            R1ReceiverArm::P0PhysicalGrade,
            OutletPortalId(999),
        )
        .is_err());
    }

    #[derive(Debug, Clone, Copy)]
    struct GateObservation {
        spacing_km: f64,
        surface: R1SurfaceKind,
        arm: R1ReceiverArm,
        maximum_cross_track_km: Option<f64>,
        relative_arclength_error: Option<f64>,
        backtracking_km: Option<f64>,
        outlet_error_km: Option<f64>,
        cell_count: Option<usize>,
        build_index_ties: usize,
        terminated: bool,
    }

    #[derive(Debug, Clone, Copy)]
    struct BroadObservation {
        spacing_km: f64,
        theta_rad: f64,
        outlet_offset_km: f64,
        arm: R1ReceiverArm,
        head_index: usize,
        minimum_cross_track_km: Option<f64>,
        maximum_cross_track_km: Option<f64>,
        terminal_cross_track_km: Option<f64>,
    }

    #[test]
    #[ignore = "full registered 8/4/2 km P0/M0 path matrix is an audit test"]
    fn registered_full_path_matrix_reports_frozen_f0_gates() {
        let mut observations = Vec::new();
        let mut broad_observations = Vec::new();
        for spacing_km in [8.0, 4.0, 2.0] {
            let cap = build_r1_voronoi_cap(VoronoiCapConfig::r1(spacing_km)).unwrap();
            let context = R1TraceContext::new(&cap).unwrap();
            for surface in [
                R1SurfaceKind::Affine,
                R1SurfaceKind::Valley,
                R1SurfaceKind::Broad,
            ] {
                for theta_rad in [0.0, 0.31] {
                    for outlet_offset_km in [0.0, 0.7] {
                        let case = build_r1_registered_case(
                            &cap,
                            R1CaseConfig {
                                spacing_km,
                                theta_rad,
                                outlet_offset_km,
                                surface,
                            },
                        )
                        .unwrap();
                        let case_before = case.clone();
                        for head_index in 0..case.heads.len() {
                            for arm in
                                [R1ReceiverArm::P0PhysicalGrade, R1ReceiverArm::M0MfdFraction]
                            {
                                let path = trace_r1_path(
                                    &cap,
                                    &case,
                                    &context,
                                    head_index,
                                    arm,
                                    R1_CAP_PORTAL_ID,
                                );
                                let repeated = trace_r1_path(
                                    &cap,
                                    &case,
                                    &context,
                                    head_index,
                                    arm,
                                    R1_CAP_PORTAL_ID,
                                );
                                assert_eq!(path, repeated);
                                let outcome = path.unwrap_or_else(|error| {
                                    panic!("malformed trace state: {error}")
                                });
                                let path = match outcome {
                                    R1TraceOutcome::ReachedPortal(path) => path,
                                    R1TraceOutcome::Failed(failed) => {
                                        let p0_margin = failed
                                            .steps
                                            .iter()
                                            .map(|step| {
                                                step.ranks.p0_physical_grade.normalized_margin
                                            })
                                            .fold(f64::INFINITY, f64::min);
                                        let m0_margin = failed
                                            .steps
                                            .iter()
                                            .map(|step| {
                                                step.ranks.m0_mfd_fraction.normalized_margin
                                            })
                                            .fold(f64::INFINITY, f64::min);
                                        eprintln!(
                                            "TRACE_FAILURE spacing={spacing_km} theta={theta_rad} delta={outlet_offset_km} surface={surface:?} head={head_index} arm={arm:?} termination={:?} visited={} selected={} exact_ties={} build_ties={} p0_margin_min={p0_margin:.6e} m0_margin_min={m0_margin:.6e} partial_c0_vertices={} partial_f0_vertices={}",
                                            failed.termination,
                                            failed.visited_cell_count,
                                            failed.selected_face_count,
                                            failed.exact_score_tie_selections,
                                            failed.build_index_tie_selections,
                                            failed.partial_c0_vertices_km.len(),
                                            failed.partial_f0_vertices_km.len(),
                                        );
                                        if surface == R1SurfaceKind::Broad {
                                            broad_observations.push(BroadObservation {
                                                spacing_km,
                                                theta_rad,
                                                outlet_offset_km,
                                                arm,
                                                head_index,
                                                minimum_cross_track_km: None,
                                                maximum_cross_track_km: None,
                                                terminal_cross_track_km: None,
                                            });
                                        } else {
                                            observations.push(GateObservation {
                                                spacing_km,
                                                surface,
                                                arm,
                                                maximum_cross_track_km: None,
                                                relative_arclength_error: None,
                                                backtracking_km: None,
                                                outlet_error_km: None,
                                                cell_count: None,
                                                build_index_ties: failed.build_index_tie_selections,
                                                terminated: false,
                                            });
                                        }
                                        continue;
                                    }
                                };
                                let p0_margin = path
                                    .steps
                                    .iter()
                                    .map(|step| step.ranks.p0_physical_grade.normalized_margin)
                                    .fold(f64::INFINITY, f64::min);
                                let m0_margin = path
                                    .steps
                                    .iter()
                                    .map(|step| step.ranks.m0_mfd_fraction.normalized_margin)
                                    .fold(f64::INFINITY, f64::min);
                                eprintln!(
                                    "spacing={spacing_km} theta={theta_rad} delta={outlet_offset_km} surface={surface:?} head={head_index} arm={arm:?} cells={} f0_n={:.6}..{:.6} f0_cross={:.6} f0_len={:.6} f0_len_err={:.6} f0_back={:.6} outlet={:.6} end_s={:.6} end_n={:.6} c0_cross={:.6} c0_len_err={:.6} exact_ties={} build_ties={} p0_margin_min={:.6e} m0_margin_min={:.6e}",
                                    path.cell_count,
                                    path.f0.metrics.minimum_cross_track_km,
                                    path.f0.metrics.maximum_cross_track_km,
                                    path.f0.metrics.maximum_absolute_cross_track_km,
                                    path.f0.metrics.polyline_arclength_km,
                                    path.f0.metrics.relative_arclength_error,
                                    path.f0.metrics.total_backtracking_km,
                                    path.f0.metrics.outlet_midpoint_error_km,
                                    path.f0.metrics.terminal_along_track_km,
                                    path.f0.metrics.terminal_cross_track_km,
                                    path.c0.metrics.maximum_absolute_cross_track_km,
                                    path.c0.metrics.relative_arclength_error,
                                    path.exact_score_tie_selections,
                                    path.build_index_tie_selections,
                                    p0_margin,
                                    m0_margin,
                                );
                                if surface == R1SurfaceKind::Broad {
                                    broad_observations.push(BroadObservation {
                                        spacing_km,
                                        theta_rad,
                                        outlet_offset_km,
                                        arm,
                                        head_index,
                                        minimum_cross_track_km: Some(
                                            path.f0.metrics.minimum_cross_track_km,
                                        ),
                                        maximum_cross_track_km: Some(
                                            path.f0.metrics.maximum_cross_track_km,
                                        ),
                                        terminal_cross_track_km: Some(
                                            path.f0.metrics.terminal_cross_track_km,
                                        ),
                                    });
                                } else {
                                    observations.push(GateObservation {
                                        spacing_km,
                                        surface,
                                        arm,
                                        maximum_cross_track_km: Some(
                                            path.f0.metrics.maximum_absolute_cross_track_km,
                                        ),
                                        relative_arclength_error: Some(
                                            path.f0.metrics.relative_arclength_error,
                                        ),
                                        backtracking_km: Some(
                                            path.f0.metrics.total_backtracking_km,
                                        ),
                                        outlet_error_km: Some(
                                            path.f0.metrics.outlet_midpoint_error_km,
                                        ),
                                        cell_count: Some(path.cell_count),
                                        build_index_ties: path.build_index_tie_selections,
                                        terminated: true,
                                    });
                                }
                            }
                        }
                        assert_eq!(case, case_before);
                    }
                }
            }
        }

        let mut passed_surface_arms = 0usize;
        for surface in [R1SurfaceKind::Affine, R1SurfaceKind::Valley] {
            for arm in [R1ReceiverArm::P0PhysicalGrade, R1ReceiverArm::M0MfdFraction] {
                let selected: Vec<_> = observations
                    .iter()
                    .filter(|item| item.surface == surface && item.arm == arm)
                    .copied()
                    .collect();
                assert_eq!(selected.len(), 12);
                let cross_at = |spacing: f64| {
                    let values: Vec<_> = selected
                        .iter()
                        .filter(|item| item.spacing_km == spacing && item.terminated)
                        .filter_map(|item| item.maximum_cross_track_km)
                        .collect();
                    (
                        values.iter().copied().fold(f64::INFINITY, f64::min),
                        values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    )
                };
                let (cross_8_min, cross_8_max) = cross_at(8.0);
                let (cross_4_min, cross_4_max) = cross_at(4.0);
                let (cross_2_min, cross_2_max) = cross_at(2.0);
                let length_2_max = selected
                    .iter()
                    .filter(|item| item.spacing_km == 2.0 && item.terminated)
                    .filter_map(|item| item.relative_arclength_error)
                    .fold(f64::NEG_INFINITY, f64::max);
                let backtracking_2_max = selected
                    .iter()
                    .filter(|item| item.spacing_km == 2.0 && item.terminated)
                    .filter_map(|item| item.backtracking_km)
                    .fold(f64::NEG_INFINITY, f64::max);
                let build_index_ties: usize =
                    selected.iter().map(|item| item.build_index_ties).sum();
                let termination_failures = selected.iter().filter(|item| !item.terminated).count();
                let termination_gate = termination_failures == 0;
                let cross_gate = cross_2_max <= 3.0 && cross_2_max < cross_8_max;
                let length_gate = length_2_max < 0.05;
                let backtracking_gate = backtracking_2_max <= 2.0;
                let robustness_gate =
                    !(cross_2_max > 2.0 * cross_2_min && cross_2_max - cross_2_min > 2.0);
                let tie_gate = build_index_ties == 0;
                let pass_all = termination_gate
                    && cross_gate
                    && length_gate
                    && backtracking_gate
                    && robustness_gate
                    && tie_gate;
                passed_surface_arms += usize::from(pass_all);
                eprintln!(
                    "GATE surface={surface:?} arm={arm:?} termination_failures={termination_failures} cross8={cross_8_min:.6}..{cross_8_max:.6} cross4={cross_4_min:.6}..{cross_4_max:.6} cross2={cross_2_min:.6}..{cross_2_max:.6} len2max={length_2_max:.6} back2max={backtracking_2_max:.6} build_ties={build_index_ties} pass_termination={termination_gate} pass_cross={cross_gate} pass_length={length_gate} pass_back={backtracking_gate} pass_robust={robustness_gate} pass_tie={tie_gate} pass_all={}",
                    pass_all,
                );

                for spacing_km in [8.0, 4.0, 2.0] {
                    let at_spacing: Vec<_> = selected
                        .iter()
                        .filter(|item| item.spacing_km == spacing_km)
                        .collect();
                    let range = |values: Vec<f64>| {
                        (
                            values.iter().copied().fold(f64::INFINITY, f64::min),
                            values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                        )
                    };
                    let (cross_min, cross_max) = range(
                        at_spacing
                            .iter()
                            .filter_map(|item| item.maximum_cross_track_km)
                            .collect(),
                    );
                    let (length_min, length_max) = range(
                        at_spacing
                            .iter()
                            .filter_map(|item| item.relative_arclength_error)
                            .collect(),
                    );
                    let (back_min, back_max) = range(
                        at_spacing
                            .iter()
                            .filter_map(|item| item.backtracking_km)
                            .collect(),
                    );
                    let (outlet_min, outlet_max) = range(
                        at_spacing
                            .iter()
                            .filter_map(|item| item.outlet_error_km)
                            .collect(),
                    );
                    let cell_min = at_spacing
                        .iter()
                        .filter_map(|item| item.cell_count)
                        .min()
                        .unwrap_or(0);
                    let cell_max = at_spacing
                        .iter()
                        .filter_map(|item| item.cell_count)
                        .max()
                        .unwrap_or(0);
                    let failures = at_spacing.iter().filter(|item| !item.terminated).count();
                    eprintln!(
                        "SUMMARY spacing={spacing_km} surface={surface:?} arm={arm:?} failures={failures} cross={cross_min:.6}..{cross_max:.6} spread={:.6} length_error={length_min:.6}..{length_max:.6} spread={:.6} backtracking={back_min:.6}..{back_max:.6} spread={:.6} outlet={outlet_min:.6}..{outlet_max:.6} spread={:.6} cells={cell_min}..{cell_max}",
                        cross_max - cross_min,
                        length_max - length_min,
                        back_max - back_min,
                        outlet_max - outlet_min,
                    );
                }
            }
        }
        assert_eq!(passed_surface_arms, 0);

        for spacing_km in [8.0, 4.0, 2.0] {
            for theta_rad in [0.0, 0.31] {
                for outlet_offset_km in [0.0, 0.7] {
                    for arm in [R1ReceiverArm::P0PhysicalGrade, R1ReceiverArm::M0MfdFraction] {
                        let selected: Vec<_> = broad_observations
                            .iter()
                            .filter(|item| {
                                item.spacing_km == spacing_km
                                    && item.theta_rad == theta_rad
                                    && item.outlet_offset_km == outlet_offset_km
                                    && item.arm == arm
                            })
                            .collect();
                        assert_eq!(selected.len(), 2);
                        assert_eq!(selected[0].head_index, 0);
                        assert_eq!(selected[1].head_index, 1);
                        let union_min = selected
                            .iter()
                            .filter_map(|item| item.minimum_cross_track_km)
                            .reduce(f64::min);
                        let union_max = selected
                            .iter()
                            .filter_map(|item| item.maximum_cross_track_km)
                            .reduce(f64::max);
                        let union = union_min.zip(union_max);
                        let failures = selected
                            .iter()
                            .filter(|item| item.minimum_cross_track_km.is_none())
                            .count();
                        eprintln!(
                            "BROAD spacing={spacing_km} theta={theta_rad} delta={outlet_offset_km} arm={arm:?} failures={failures} head0_end_n={:?} head1_end_n={:?} union_n={union:?}",
                            selected[0].terminal_cross_track_km,
                            selected[1].terminal_cross_track_km,
                        );
                    }
                }
            }
        }
    }
}
