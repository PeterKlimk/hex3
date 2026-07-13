//! Report-only affine generator-point causal control for R1a.
//!
//! This module is test-only by design. Generator samples are not conservative
//! polygon means and cannot become a product terrain representation or a third
//! receiver arm.

use glam::{DVec2, DVec3};

use super::{
    build_r1_registered_case, build_r1_voronoi_cap,
    channel_extraction_r1a_fixture::build_r1_affine_generator_point_control, trace_r1_path,
    R1CaseConfig, R1PathMetrics, R1ReceiverArm, R1SurfaceKind, R1TraceContext, R1TraceOutcome,
    R1TraceTermination, VoronoiCapConfig, R1_CAP_HEIGHT_KM, R1_CAP_PORTAL_ID,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AffineRepresentation {
    PolygonMean,
    GeneratorPoint,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ControlTermination {
    ReachedPortal,
    Failed(R1TraceTermination),
}

#[derive(Debug, Clone)]
struct ControlObservation {
    spacing_km: f64,
    theta_rad: f64,
    outlet_offset_km: f64,
    representation: AffineRepresentation,
    arm: R1ReceiverArm,
    termination: ControlTermination,
    metrics: Option<R1PathMetrics>,
    cell_count: usize,
    exact_score_ties: usize,
    build_index_ties: usize,
    minimum_p0_margin: f64,
    minimum_m0_margin: f64,
    partial_maximum_absolute_cross_track_km: Option<f64>,
    partial_terminal_along_track_km: Option<f64>,
    partial_terminal_cross_track_km: Option<f64>,
}

impl ControlObservation {
    fn reached_portal(&self) -> bool {
        self.termination == ControlTermination::ReachedPortal
    }
}

#[test]
fn generator_control_is_explicit_deterministic_and_immutable() {
    let cap = build_r1_voronoi_cap(VoronoiCapConfig::r1(8.0)).unwrap();
    let config = R1CaseConfig {
        spacing_km: 8.0,
        theta_rad: 0.31,
        outlet_offset_km: 0.7,
        surface: R1SurfaceKind::Affine,
    };
    let cap_before = cap.clone();
    let registered = build_r1_registered_case(&cap, config).unwrap();
    let first = build_r1_affine_generator_point_control(&cap, config).unwrap();
    let second = build_r1_affine_generator_point_control(&cap, config).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.case.config, registered.config);
    assert_eq!(first.case.heads, registered.heads);
    assert_eq!(
        first.case.local_supply_km3_myr,
        registered.local_supply_km3_myr
    );
    assert_ne!(first.case.elevation_km, registered.elevation_km);
    assert!(first.case.audit.water_balance_relative_error <= 1.0e-12);

    let context = R1TraceContext::new(&cap).unwrap();
    let case_before = first.case.clone();
    let context_before = context.clone();
    for arm in [R1ReceiverArm::P0PhysicalGrade, R1ReceiverArm::M0MfdFraction] {
        let a = trace_r1_path(&cap, &first.case, &context, 0, arm, R1_CAP_PORTAL_ID).unwrap();
        let b = trace_r1_path(&cap, &first.case, &context, 0, arm, R1_CAP_PORTAL_ID).unwrap();
        assert_eq!(a, b);
    }
    assert_eq!(first.case, case_before);
    assert_eq!(context, context_before);
    assert_eq!(cap, cap_before);
}

#[test]
#[ignore = "full paired polygon-mean/generator-point affine matrix is an audit test"]
fn registered_affine_generator_control_reports_paired_causal_matrix() {
    let mut observations = Vec::new();
    for spacing_km in [8.0, 4.0, 2.0] {
        let cap = build_r1_voronoi_cap(VoronoiCapConfig::r1(spacing_km)).unwrap();
        let cap_before = cap.clone();
        let context = R1TraceContext::new(&cap).unwrap();
        for theta_rad in [0.0, 0.31] {
            for outlet_offset_km in [0.0, 0.7] {
                let config = R1CaseConfig {
                    spacing_km,
                    theta_rad,
                    outlet_offset_km,
                    surface: R1SurfaceKind::Affine,
                };
                let registered = build_r1_registered_case(&cap, config).unwrap();
                assert_eq!(registered, build_r1_registered_case(&cap, config).unwrap());
                let control = build_r1_affine_generator_point_control(&cap, config).unwrap();
                assert_eq!(
                    control,
                    build_r1_affine_generator_point_control(&cap, config).unwrap()
                );
                assert_eq!(registered.config, control.case.config);
                assert_eq!(registered.heads, control.case.heads);
                assert_eq!(
                    registered.local_supply_km3_myr,
                    control.case.local_supply_km3_myr
                );
                assert_ne!(registered.elevation_km, control.case.elevation_km);

                for (representation, case) in [
                    (AffineRepresentation::PolygonMean, &registered),
                    (AffineRepresentation::GeneratorPoint, &control.case),
                ] {
                    eprintln!(
                        "CONTROL_ROUTE spacing={spacing_km} theta={theta_rad} delta={outlet_offset_km} representation={representation:?} water_rel={:.6e} portal={:.6e} sink={:.6e}",
                        case.audit.water_balance_relative_error,
                        case.audit.target_portal_outflow_km3_myr,
                        case.audit.total_sink_storage_km3_myr,
                    );
                    for arm in [R1ReceiverArm::P0PhysicalGrade, R1ReceiverArm::M0MfdFraction] {
                        let first =
                            trace_r1_path(&cap, case, &context, 0, arm, R1_CAP_PORTAL_ID).unwrap();
                        let second =
                            trace_r1_path(&cap, case, &context, 0, arm, R1_CAP_PORTAL_ID).unwrap();
                        assert_eq!(first, second);
                        let observation = observe(
                            spacing_km,
                            theta_rad,
                            outlet_offset_km,
                            representation,
                            arm,
                            first,
                        );
                        report_observation(&observation);
                        observations.push(observation);
                    }
                }
            }
        }
        assert_eq!(cap, cap_before);
    }
    assert_eq!(observations.len(), 48);

    for representation in [
        AffineRepresentation::PolygonMean,
        AffineRepresentation::GeneratorPoint,
    ] {
        for arm in [R1ReceiverArm::P0PhysicalGrade, R1ReceiverArm::M0MfdFraction] {
            report_independent_summary(&observations, representation, arm);
        }
    }
    for arm in [R1ReceiverArm::P0PhysicalGrade, R1ReceiverArm::M0MfdFraction] {
        report_paired_summary(&observations, arm);
    }
    assert_frozen_control_result(&observations);
}

fn observe(
    spacing_km: f64,
    theta_rad: f64,
    outlet_offset_km: f64,
    representation: AffineRepresentation,
    arm: R1ReceiverArm,
    outcome: R1TraceOutcome,
) -> ControlObservation {
    match outcome {
        R1TraceOutcome::ReachedPortal(path) => {
            let (minimum_p0_margin, minimum_m0_margin) = minimum_margins(&path.steps);
            ControlObservation {
                spacing_km,
                theta_rad,
                outlet_offset_km,
                representation,
                arm,
                termination: ControlTermination::ReachedPortal,
                metrics: Some(path.f0.metrics),
                cell_count: path.cell_count,
                exact_score_ties: path.exact_score_tie_selections,
                build_index_ties: path.build_index_tie_selections,
                minimum_p0_margin,
                minimum_m0_margin,
                partial_maximum_absolute_cross_track_km: None,
                partial_terminal_along_track_km: None,
                partial_terminal_cross_track_km: None,
            }
        }
        R1TraceOutcome::Failed(failed) => {
            let (minimum_p0_margin, minimum_m0_margin) = minimum_margins(&failed.steps);
            let partial =
                partial_geometry(theta_rad, outlet_offset_km, &failed.partial_f0_vertices_km);
            ControlObservation {
                spacing_km,
                theta_rad,
                outlet_offset_km,
                representation,
                arm,
                termination: ControlTermination::Failed(failed.termination),
                metrics: None,
                cell_count: failed.visited_cell_count,
                exact_score_ties: failed.exact_score_tie_selections,
                build_index_ties: failed.build_index_tie_selections,
                minimum_p0_margin,
                minimum_m0_margin,
                partial_maximum_absolute_cross_track_km: partial
                    .map(|item| item.maximum_absolute_cross_track_km),
                partial_terminal_along_track_km: partial.map(|item| item.terminal_along_track_km),
                partial_terminal_cross_track_km: partial.map(|item| item.terminal_cross_track_km),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PartialGeometry {
    maximum_absolute_cross_track_km: f64,
    terminal_along_track_km: f64,
    terminal_cross_track_km: f64,
}

fn partial_geometry(
    theta_rad: f64,
    outlet_offset_km: f64,
    vertices: &[DVec3],
) -> Option<PartialGeometry> {
    let outlet = DVec2::new(outlet_offset_km, -0.5 * R1_CAP_HEIGHT_KM);
    let along = DVec2::new(theta_rad.sin(), theta_rad.cos());
    let transverse = DVec2::new(-theta_rad.cos(), theta_rad.sin());
    let mut maximum_absolute_cross_track_km: f64 = 0.0;
    for vertex in vertices {
        let relative = DVec2::new(vertex.x, vertex.y) - outlet;
        maximum_absolute_cross_track_km =
            maximum_absolute_cross_track_km.max(relative.dot(transverse).abs());
    }
    let terminal = vertices.last()?;
    let relative = DVec2::new(terminal.x, terminal.y) - outlet;
    Some(PartialGeometry {
        maximum_absolute_cross_track_km,
        terminal_along_track_km: relative.dot(along),
        terminal_cross_track_km: relative.dot(transverse),
    })
}

fn minimum_margins(steps: &[super::R1TraceStep]) -> (f64, f64) {
    (
        steps
            .iter()
            .map(|step| step.ranks.p0_physical_grade.normalized_margin)
            .fold(f64::INFINITY, f64::min),
        steps
            .iter()
            .map(|step| step.ranks.m0_mfd_fraction.normalized_margin)
            .fold(f64::INFINITY, f64::min),
    )
}

fn report_observation(observation: &ControlObservation) {
    if let Some(metrics) = &observation.metrics {
        eprintln!(
            "CONTROL_PATH spacing={} theta={} delta={} representation={:?} arm={:?} outcome={:?} cells={} cross={:.6} length_error={:.6} backtracking={:.6} outlet={:.6} exact_ties={} build_ties={} p0_margin_min={:.6e} m0_margin_min={:.6e}",
            observation.spacing_km,
            observation.theta_rad,
            observation.outlet_offset_km,
            observation.representation,
            observation.arm,
            observation.termination,
            observation.cell_count,
            metrics.maximum_absolute_cross_track_km,
            metrics.relative_arclength_error,
            metrics.total_backtracking_km,
            metrics.outlet_midpoint_error_km,
            observation.exact_score_ties,
            observation.build_index_ties,
            observation.minimum_p0_margin,
            observation.minimum_m0_margin,
        );
    } else {
        eprintln!(
            "CONTROL_PATH spacing={} theta={} delta={} representation={:?} arm={:?} outcome={:?} cells={} geometry=NON_EVALUABLE partial_cross={:?} partial_end_s={:?} partial_end_n={:?} exact_ties={} build_ties={} p0_margin_min={:.6e} m0_margin_min={:.6e}",
            observation.spacing_km,
            observation.theta_rad,
            observation.outlet_offset_km,
            observation.representation,
            observation.arm,
            observation.termination,
            observation.cell_count,
            observation.partial_maximum_absolute_cross_track_km,
            observation.partial_terminal_along_track_km,
            observation.partial_terminal_cross_track_km,
            observation.exact_score_ties,
            observation.build_index_ties,
            observation.minimum_p0_margin,
            observation.minimum_m0_margin,
        );
    }
}

fn selected<'a>(
    observations: &'a [ControlObservation],
    representation: AffineRepresentation,
    arm: R1ReceiverArm,
) -> Vec<&'a ControlObservation> {
    observations
        .iter()
        .filter(|item| item.representation == representation && item.arm == arm)
        .collect()
}

fn metric_range(items: impl Iterator<Item = f64>) -> Option<(f64, f64)> {
    items.fold(None, |range, value| {
        Some(match range {
            Some((minimum, maximum)) => (minimum.min(value), maximum.max(value)),
            None => (value, value),
        })
    })
}

fn report_independent_summary(
    observations: &[ControlObservation],
    representation: AffineRepresentation,
    arm: R1ReceiverArm,
) {
    let selected = selected(observations, representation, arm);
    assert_eq!(selected.len(), 12);
    let failures = selected
        .iter()
        .filter(|item| !item.reached_portal())
        .count();
    let build_ties: usize = selected.iter().map(|item| item.build_index_ties).sum();
    let mut cross_ranges = Vec::new();
    for spacing_km in [8.0, 4.0, 2.0] {
        let at_spacing: Vec<_> = selected
            .iter()
            .copied()
            .filter(|item| item.spacing_km == spacing_km)
            .collect();
        let cross = metric_range(at_spacing.iter().filter_map(|item| {
            item.metrics
                .as_ref()
                .map(|metrics| metrics.maximum_absolute_cross_track_km)
        }));
        let length = metric_range(at_spacing.iter().filter_map(|item| {
            item.metrics
                .as_ref()
                .map(|metrics| metrics.relative_arclength_error)
        }));
        let backtracking = metric_range(at_spacing.iter().filter_map(|item| {
            item.metrics
                .as_ref()
                .map(|metrics| metrics.total_backtracking_km)
        }));
        let censored = at_spacing
            .iter()
            .filter(|item| !item.reached_portal())
            .count();
        eprintln!(
            "CONTROL_SUMMARY spacing={spacing_km} representation={representation:?} arm={arm:?} censored={censored} cross={cross:?} length_error={length:?} backtracking={backtracking:?}"
        );
        cross_ranges.push((spacing_km, cross));
    }
    let cross_8 = cross_ranges[0].1.map(|(_, maximum)| maximum);
    let cross_2 = cross_ranges[2].1;
    let at_2: Vec<_> = selected
        .iter()
        .copied()
        .filter(|item| item.spacing_km == 2.0 && item.reached_portal())
        .collect();
    let length_2 = metric_range(at_2.iter().filter_map(|item| {
        item.metrics
            .as_ref()
            .map(|metrics| metrics.relative_arclength_error)
    }));
    let backtracking_2 = metric_range(at_2.iter().filter_map(|item| {
        item.metrics
            .as_ref()
            .map(|metrics| metrics.total_backtracking_km)
    }));
    let cross_gate = cross_8
        .zip(cross_2)
        .map(|(cross_8_max, (_, cross_2_max))| cross_2_max <= 3.0 && cross_2_max < cross_8_max);
    let length_gate = length_2.map(|(_, maximum)| maximum < 0.05);
    let backtracking_gate = backtracking_2.map(|(_, maximum)| maximum <= 2.0);
    let robustness_gate =
        cross_2.map(|(minimum, maximum)| !(maximum > 2.0 * minimum && maximum - minimum > 2.0));
    let termination_gate = failures == 0;
    let tie_gate = build_ties == 0;
    let pass_all = termination_gate
        && cross_gate == Some(true)
        && length_gate == Some(true)
        && backtracking_gate == Some(true)
        && robustness_gate == Some(true)
        && tie_gate;
    eprintln!(
        "CONTROL_GATE representation={representation:?} arm={arm:?} failures={failures} pass_termination={termination_gate} pass_cross={cross_gate:?} pass_length={length_gate:?} pass_backtracking={backtracking_gate:?} pass_robustness={robustness_gate:?} build_ties={build_ties} pass_tie={tie_gate} pass_all={pass_all}"
    );
}

fn report_paired_summary(observations: &[ControlObservation], arm: R1ReceiverArm) {
    let polygon = selected(observations, AffineRepresentation::PolygonMean, arm);
    let generator = selected(observations, AffineRepresentation::GeneratorPoint, arm);
    let mut repaired_terminations = 0usize;
    let mut remaining_generator_failures = 0usize;
    for original in &polygon {
        let paired = generator
            .iter()
            .copied()
            .find(|candidate| {
                candidate.spacing_km == original.spacing_km
                    && candidate.theta_rad == original.theta_rad
                    && candidate.outlet_offset_km == original.outlet_offset_km
            })
            .unwrap();
        repaired_terminations += usize::from(!original.reached_portal() && paired.reached_portal());
        remaining_generator_failures += usize::from(!paired.reached_portal());
        if let (Some(original_metrics), Some(paired_metrics)) = (&original.metrics, &paired.metrics)
        {
            eprintln!(
                "CONTROL_PAIR spacing={} theta={} delta={} arm={arm:?} cross_polygon={:.6} cross_generator={:.6} delta_cross={:.6} length_polygon={:.6} length_generator={:.6} delta_length={:.6}",
                original.spacing_km,
                original.theta_rad,
                original.outlet_offset_km,
                original_metrics.maximum_absolute_cross_track_km,
                paired_metrics.maximum_absolute_cross_track_km,
                paired_metrics.maximum_absolute_cross_track_km
                    - original_metrics.maximum_absolute_cross_track_km,
                original_metrics.relative_arclength_error,
                paired_metrics.relative_arclength_error,
                paired_metrics.relative_arclength_error
                    - original_metrics.relative_arclength_error,
            );
        }
    }
    let common_success_count = |spacing_km: f64| {
        polygon
            .iter()
            .filter(|original| {
                original.spacing_km == spacing_km
                    && original.reached_portal()
                    && generator.iter().any(|candidate| {
                        candidate.spacing_km == original.spacing_km
                            && candidate.theta_rad == original.theta_rad
                            && candidate.outlet_offset_km == original.outlet_offset_km
                            && candidate.reached_portal()
                    })
            })
            .count()
    };
    let common_8 = common_success_count(8.0);
    let common_4 = common_success_count(4.0);
    let common_2 = common_success_count(2.0);
    let cross_gate_evaluable = common_8 == 4 && common_2 == 4;
    let finest_gates_evaluable = common_2 == 4;
    eprintln!(
        "CONTROL_PAIRED_GATE arm={arm:?} repaired_terminations={repaired_terminations} remaining_generator_failures={remaining_generator_failures} common_success_8={common_8} common_success_4={common_4} common_success_2={common_2} cross_gate_evaluable={cross_gate_evaluable} finest_gates_evaluable={finest_gates_evaluable}"
    );
}

fn assert_frozen_control_result(observations: &[ControlObservation]) {
    for arm in [R1ReceiverArm::P0PhysicalGrade, R1ReceiverArm::M0MfdFraction] {
        let polygon = selected(observations, AffineRepresentation::PolygonMean, arm);
        let generator = selected(observations, AffineRepresentation::GeneratorPoint, arm);
        assert_eq!(generator.len(), 12);
        assert_eq!(
            generator
                .iter()
                .filter(|item| !item.reached_portal())
                .count(),
            6
        );
        assert!(generator
            .iter()
            .filter(|item| !item.reached_portal())
            .all(|item| item.theta_rad == 0.31));
        assert_eq!(
            polygon
                .iter()
                .filter(|original| {
                    !original.reached_portal()
                        && generator.iter().any(|candidate| {
                            candidate.spacing_km == original.spacing_km
                                && candidate.theta_rad == original.theta_rad
                                && candidate.outlet_offset_km == original.outlet_offset_km
                                && candidate.reached_portal()
                        })
                })
                .count(),
            0
        );
        for spacing_km in [8.0, 4.0, 2.0] {
            assert_eq!(
                generator
                    .iter()
                    .filter(|item| item.spacing_km == spacing_km && item.reached_portal())
                    .count(),
                2
            );
        }
        let finest: Vec<_> = generator
            .iter()
            .filter(|item| item.spacing_km == 2.0 && item.reached_portal())
            .filter_map(|item| item.metrics.as_ref())
            .collect();
        assert_eq!(finest.len(), 2);
        assert!(finest
            .iter()
            .any(|metrics| metrics.maximum_absolute_cross_track_km > 3.0));
        assert!(finest
            .iter()
            .any(|metrics| metrics.relative_arclength_error >= 0.05));
        assert!(generator.iter().all(|item| item.build_index_ties == 0));
        assert!(generator
            .iter()
            .filter_map(|item| item.metrics.as_ref())
            .all(|metrics| metrics.total_backtracking_km == 0.0));
    }

    let p0_polygon = selected(
        observations,
        AffineRepresentation::PolygonMean,
        R1ReceiverArm::P0PhysicalGrade,
    );
    let p0_generator = selected(
        observations,
        AffineRepresentation::GeneratorPoint,
        R1ReceiverArm::P0PhysicalGrade,
    );
    let mut comparable_p0 = 0usize;
    for original in p0_polygon.iter().filter(|item| item.reached_portal()) {
        let paired = p0_generator
            .iter()
            .copied()
            .find(|candidate| {
                candidate.spacing_km == original.spacing_km
                    && candidate.theta_rad == original.theta_rad
                    && candidate.outlet_offset_km == original.outlet_offset_km
                    && candidate.reached_portal()
            })
            .unwrap();
        comparable_p0 += 1;
        assert!(
            paired
                .metrics
                .as_ref()
                .unwrap()
                .maximum_absolute_cross_track_km
                < original
                    .metrics
                    .as_ref()
                    .unwrap()
                    .maximum_absolute_cross_track_km
        );
    }
    assert_eq!(comparable_p0, 6);
}
