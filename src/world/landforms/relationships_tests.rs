//! End-to-end public-API gates for the O0a relationship packet.

use super::*;
use crate::world::landscape::{BoundarySide, OutletPortal};

struct RelationshipFixture {
    graph: EvaluationSurfaceGraphV0,
    elevation_km: Vec<f64>,
    scored_cell: Vec<bool>,
    runoff: Vec<f64>,
    surface_config: SurfaceHierarchyConfigV0,
    drainage_config: DrainageConfigV0,
    hierarchy: SurfaceHierarchyV0,
    drainage: EvaluationDrainageV0,
    geometry_identity: PacketGeometryIdentityV0,
}

fn segment_network_cost(point: DVec3, start: DVec3, end: DVec3, offset: f64) -> f64 {
    let segment = end - start;
    let fraction = ((point - start).dot(segment) / segment.length_squared()).clamp(0.0, 1.0);
    let projected = start + fraction * segment;
    offset + fraction * segment.length() + 2.0 * point.distance(projected)
}

fn assembled_fixture(
    graph: EvaluationSurfaceGraphV0,
    spacing_km: f64,
    elevation_km: Vec<f64>,
    runoff: Vec<f64>,
) -> RelationshipFixture {
    let surface_config = SurfaceHierarchyConfigV0::default();
    let scored_cell = vec![true; graph.cell_count()];
    let drainage_config = DrainageConfigV0::default();
    let hierarchy =
        build_surface_hierarchy_v0(&graph, &elevation_km, &scored_cell, surface_config).unwrap();
    let drainage =
        build_evaluation_drainage_v0(&graph, &elevation_km, &runoff, drainage_config).unwrap();
    let geometry_identity = PacketGeometryIdentityV0::LandscapeRegularPlanar {
        nominal_spacing_km: spacing_km,
        canonical_graph_hash: relationship_graph_hash_v0(&graph).unwrap(),
    };
    RelationshipFixture {
        graph,
        elevation_km,
        scored_cell,
        runoff,
        surface_config,
        drainage_config,
        hierarchy,
        drainage,
        geometry_identity,
    }
}

fn asymmetric_y_fixture(spacing_km: f64) -> RelationshipFixture {
    let portal = OutletPortal {
        id: OutletPortalId(23),
        side: BoundarySide::South,
        span_start_km: -1.0,
        span_end_km: 1.0,
        base_level_km: 0.0,
    };
    let mesh = LandscapeMesh::uniform_planar_hex_with_portals(
        128.0,
        96.0,
        spacing_km,
        std::slice::from_ref(&portal),
    )
    .unwrap();
    let surface_config = SurfaceHierarchyConfigV0::default();
    let controls = build_regular_hex_control_volumes_v0(&mesh, &surface_config).unwrap();
    let graph = adapt_landscape_graph_v0(&mesh, &controls, &surface_config).unwrap();

    let outlet = DVec3::new(0.0, -48.0, 0.0);
    let junction = DVec3::ZERO;
    let left_head = DVec3::new(-48.0, 48.0, 0.0);
    let right_head = DVec3::new(48.0, 48.0, 0.0);
    let trunk_length = outlet.distance(junction);
    let elevation_km = graph
        .cell_center_km
        .iter()
        .map(|&point| {
            0.01 * segment_network_cost(point, outlet, junction, 0.0)
                .min(segment_network_cost(
                    point,
                    junction,
                    left_head,
                    trunk_length,
                ))
                .min(segment_network_cost(
                    point,
                    junction,
                    right_head,
                    trunk_length,
                ))
        })
        .collect::<Vec<_>>();
    let runoff = graph
        .cell_area_km2
        .iter()
        .zip(&graph.cell_center_km)
        .map(|(&area, center)| 0.3 * (1.0 + 0.002 * center.x) * area)
        .collect::<Vec<_>>();
    assembled_fixture(graph, spacing_km, elevation_km, runoff)
}

fn west_east_fixture(spacing_km: f64, flat: bool) -> RelationshipFixture {
    let base_level_km = if flat { 1.0 } else { 0.0 };
    let portals = [
        OutletPortal {
            id: OutletPortalId(1),
            side: BoundarySide::West,
            span_start_km: -96.0,
            span_end_km: 96.0,
            base_level_km,
        },
        OutletPortal {
            id: OutletPortalId(2),
            side: BoundarySide::East,
            span_start_km: -96.0,
            span_end_km: 96.0,
            base_level_km,
        },
    ];
    let mesh =
        LandscapeMesh::uniform_planar_hex_with_portals(256.0, 192.0, spacing_km, &portals).unwrap();
    let surface_config = SurfaceHierarchyConfigV0::default();
    let controls = build_regular_hex_control_volumes_v0(&mesh, &surface_config).unwrap();
    let graph = adapt_landscape_graph_v0(&mesh, &controls, &surface_config).unwrap();
    let elevation_km = graph
        .cell_center_km
        .iter()
        .map(|center| {
            if flat {
                1.0
            } else {
                0.25 + 0.005 * (128.0 - center.x.abs())
            }
        })
        .collect::<Vec<_>>();
    let runoff = graph.cell_area_km2.iter().map(|area| 0.1 * area).collect();
    assembled_fixture(graph, spacing_km, elevation_km, runoff)
}

fn build_relationships(
    fixture: &RelationshipFixture,
    config: LandformRelationshipConfigV0,
) -> Result<LandformRelationshipsV0, RelationshipErrorV0> {
    build_landform_relationships_v0(
        &fixture.graph,
        &fixture.elevation_km,
        &fixture.scored_cell,
        &fixture.runoff,
        fixture.surface_config,
        fixture.drainage_config,
        &fixture.hierarchy,
        &fixture.drainage,
        fixture.geometry_identity,
        config,
    )
}

fn assert_face_partition(fixture: &RelationshipFixture, relationships: &LandformRelationshipsV0) {
    let reference = fixture
        .drainage
        .scales
        .iter()
        .find(|scale| scale.support_threshold_km2 == 2_000.0)
        .unwrap();
    let raw = &reference.basin_graph.raw_catchment_boundaries;
    assert!(!raw.is_empty());
    assert_eq!(relationships.backed_boundary_faces.len(), raw.len());
    assert_eq!(
        relationships.work_counts.raw_boundary_faces as usize,
        raw.len()
    );

    let mut flow_transitions = 0usize;
    let mut lateral_candidates = 0usize;
    for face in &relationships.backed_boundary_faces {
        match face.role {
            BoundaryFaceRoleKindV0::FlowTransition => {
                flow_transitions += 1;
                assert!(face.receiver_direction.is_some());
                assert!(face.bilateral_descent.is_none());
            }
            BoundaryFaceRoleKindV0::LateralBoundaryCandidate => {
                lateral_candidates += 1;
                assert!(face.receiver_direction.is_none());
                assert!(face.bilateral_descent.is_some());
            }
        }
    }
    assert_eq!(
        flow_transitions + lateral_candidates,
        relationships.backed_boundary_faces.len()
    );

    let raw_length: f64 = raw.iter().map(|face| face.physical_length_km).sum();
    let backed_length: f64 = relationships
        .backed_boundary_faces
        .iter()
        .map(|face| face.physical_length_km)
        .sum();
    let tolerance = 1.0e-12 * raw_length.abs().max(1.0);
    assert!((backed_length - raw_length).abs() <= tolerance);
}

fn central_lateral_faces(relationships: &LandformRelationshipsV0) -> Vec<&BackedBoundaryFaceV0> {
    relationships
        .backed_boundary_faces
        .iter()
        .filter(|face| face.role == BoundaryFaceRoleKindV0::LateralBoundaryCandidate)
        .filter(|face| {
            let midpoint = 0.5 * (face.endpoints_km[0] + face.endpoints_km[1]);
            midpoint.x.abs() <= face.covering_radius_km
        })
        .collect()
}

#[test]
fn o0a_asymmetric_y_is_deterministic_and_partitions_all_raw_faces() {
    for spacing_km in [8.0, 4.0, 2.0] {
        let fixture = asymmetric_y_fixture(spacing_km);
        let first = build_relationships(&fixture, LandformRelationshipConfigV0::default()).unwrap();
        let second =
            build_relationships(&fixture, LandformRelationshipConfigV0::default()).unwrap();

        assert_eq!(first, second);
        assert_ne!(first.derived_evidence_hash, 0);
        assert_eq!(
            first.surface_hierarchy_input_hash,
            fixture.hierarchy.derived_evidence_hash
        );
        assert_eq!(
            first.drainage_input_hash,
            fixture.drainage.derived_evidence_hash
        );
        assert_face_partition(&fixture, &first);
        assert!(first
            .backed_boundary_faces
            .iter()
            .any(|face| face.role == BoundaryFaceRoleKindV0::FlowTransition));
        assert!(first
            .backed_boundary_faces
            .iter()
            .any(|face| face.role == BoundaryFaceRoleKindV0::LateralBoundaryCandidate));
        eprintln!(
            "O0a asymmetric-Y {spacing_km} km: cells={}, faces={}, stations={}, samples={}, candidate_tests={}",
            fixture.graph.cell_count(),
            first.work_counts.raw_boundary_faces,
            first.work_counts.reach_stations,
            first.work_counts.regular_cross_section_samples,
            first.work_counts.candidate_face_tests,
        );
    }
}

#[test]
fn o0a_symmetric_shed_and_translated_flat_control_bilateral_descent_at_8_4_2_km() {
    for spacing_km in [8.0, 4.0, 2.0] {
        let shed_fixture = west_east_fixture(spacing_km, false);
        let shed =
            build_relationships(&shed_fixture, LandformRelationshipConfigV0::default()).unwrap();
        let shed_central = central_lateral_faces(&shed);
        assert!(
            !shed_central.is_empty(),
            "no midpoint-near-x=0 lateral faces at {spacing_km} km"
        );
        for face in &shed_central {
            let probe = face.bilateral_descent.as_ref().unwrap();
            assert!(probe.bilateral_physical_descent);
            assert!(probe.unconditioned_bilateral_descent);
            for side in &probe.sides {
                assert!(side.target_drop_km > 0.0);
                assert!(side.minimum_segment_drop_km.is_some_and(|drop| drop > 0.0));
                assert_eq!(side.remote_maximum_excess_km, 0.0);
                assert!(side.physically_descending);
            }
        }

        let flat_fixture = west_east_fixture(spacing_km, true);
        let flat =
            build_relationships(&flat_fixture, LandformRelationshipConfigV0::default()).unwrap();
        let flat_central = central_lateral_faces(&flat);
        assert!(
            !flat_central.is_empty(),
            "no flat midpoint-near-x=0 lateral faces at {spacing_km} km"
        );
        for face in &flat_central {
            let probe = face.bilateral_descent.as_ref().unwrap();
            assert!(!probe.bilateral_physical_descent);
            assert!(!probe.unconditioned_bilateral_descent);
            for side in &probe.sides {
                assert_eq!(side.target_drop_km, 0.0);
                assert!(side.minimum_segment_drop_km.is_none_or(|drop| drop == 0.0));
                assert_eq!(side.remote_maximum_excess_km, 0.0);
                assert!(!side.physically_descending);
            }
        }
        eprintln!(
            "O0a shed/flat {spacing_km} km: cells={}, shed_central={}, flat_central={}, shed_candidate_tests={}, flat_candidate_tests={}",
            shed_fixture.graph.cell_count(),
            shed_central.len(),
            flat_central.len(),
            shed.work_counts.candidate_face_tests,
            flat.work_counts.candidate_face_tests,
        );
    }
}

#[test]
fn o0a_rejects_negative_zero_and_tampered_predecessor_hashes() {
    let fixture = asymmetric_y_fixture(4.0);
    let config = LandformRelationshipConfigV0::default();

    let mut elevation = fixture.elevation_km.clone();
    elevation[0] = -0.0;
    assert_eq!(
        build_landform_relationships_v0(
            &fixture.graph,
            &elevation,
            &fixture.scored_cell,
            &fixture.runoff,
            fixture.surface_config,
            fixture.drainage_config,
            &fixture.hierarchy,
            &fixture.drainage,
            fixture.geometry_identity,
            config,
        ),
        Err(RelationshipErrorV0::NonCanonicalZero {
            field: "physical_elevation_km",
            index: 0,
        })
    );

    let mut runoff = fixture.runoff.clone();
    runoff[0] = -0.0;
    assert_eq!(
        build_landform_relationships_v0(
            &fixture.graph,
            &fixture.elevation_km,
            &fixture.scored_cell,
            &runoff,
            fixture.surface_config,
            fixture.drainage_config,
            &fixture.hierarchy,
            &fixture.drainage,
            fixture.geometry_identity,
            config,
        ),
        Err(RelationshipErrorV0::NonCanonicalZero {
            field: "local_runoff_supply",
            index: 0,
        })
    );

    let mut hierarchy = fixture.hierarchy.clone();
    hierarchy.derived_evidence_hash ^= 1;
    assert!(matches!(
        build_landform_relationships_v0(
            &fixture.graph,
            &fixture.elevation_km,
            &fixture.scored_cell,
            &fixture.runoff,
            fixture.surface_config,
            fixture.drainage_config,
            &hierarchy,
            &fixture.drainage,
            fixture.geometry_identity,
            config,
        ),
        Err(RelationshipErrorV0::SurfaceHierarchyHashMismatch { .. })
    ));

    let mut drainage = fixture.drainage.clone();
    drainage.derived_evidence_hash ^= 1;
    assert!(matches!(
        build_landform_relationships_v0(
            &fixture.graph,
            &fixture.elevation_km,
            &fixture.scored_cell,
            &fixture.runoff,
            fixture.surface_config,
            fixture.drainage_config,
            &fixture.hierarchy,
            &drainage,
            fixture.geometry_identity,
            config,
        ),
        Err(RelationshipErrorV0::DrainageHashMismatch { .. })
    ));
}

#[test]
fn o0a_relationship_config_accepts_only_registered_one_factor_populations() {
    let fixture = asymmetric_y_fixture(4.0);

    let registered = LandformRelationshipConfigV0 {
        station_spacing_km: 10.0,
        ..LandformRelationshipConfigV0::default()
    };
    let result = build_relationships(&fixture, registered).unwrap();
    assert_eq!(result.config, registered);
    assert_eq!(
        result.run_namespace,
        RelationshipRunNamespaceV0::StationSpacingLow
    );

    let unknown = LandformRelationshipConfigV0 {
        station_spacing_km: 12.0,
        ..LandformRelationshipConfigV0::default()
    };
    assert_eq!(
        build_relationships(&fixture, unknown),
        Err(RelationshipErrorV0::UnregisteredConfiguration)
    );

    let multiple = LandformRelationshipConfigV0 {
        station_spacing_km: 10.0,
        cross_section_half_length_km: 50.0,
        ..LandformRelationshipConfigV0::default()
    };
    assert_eq!(
        build_relationships(&fixture, multiple),
        Err(RelationshipErrorV0::UnregisteredConfiguration)
    );
}

#[test]
fn o0a_registered_one_factor_sensitivities_are_separate_from_reference() {
    let fixture = asymmetric_y_fixture(8.0);
    let reference_config = LandformRelationshipConfigV0::default();
    let reference = build_relationships(&fixture, reference_config).unwrap();
    assert_eq!(
        reference.run_namespace,
        RelationshipRunNamespaceV0::Reference
    );
    let mut sensitivities = Vec::new();
    for station_spacing_km in [10.0, 40.0] {
        let mut config = reference_config;
        config.station_spacing_km = station_spacing_km;
        sensitivities.push(config);
    }
    for cross_section_half_length_km in [50.0, 150.0] {
        let mut config = reference_config;
        config.cross_section_half_length_km = cross_section_half_length_km;
        sensitivities.push(config);
    }
    for cross_section_sample_step_km in [2.0, 8.0] {
        let mut config = reference_config;
        config.cross_section_sample_step_km = cross_section_sample_step_km;
        sensitivities.push(config);
    }
    for relative_height_fraction in [0.15, 0.35] {
        let mut config = reference_config;
        config.relative_height_fraction = relative_height_fraction;
        sensitivities.push(config);
    }
    for maximum_downstream_support_km in [200.0, 600.0] {
        let mut config = reference_config;
        config.maximum_downstream_support_km = maximum_downstream_support_km;
        sensitivities.push(config);
    }

    let mut hashes = std::collections::BTreeSet::from([reference.derived_evidence_hash]);
    let mut namespaces = std::collections::BTreeSet::from([reference.run_namespace]);
    for config in sensitivities {
        let result = build_relationships(&fixture, config).unwrap();
        assert_eq!(result.config, config);
        assert_ne!(result.config, reference.config);
        assert_ne!(result.run_namespace, reference.run_namespace);
        assert!(namespaces.insert(result.run_namespace));
        assert_eq!(
            result.surface_hierarchy_input_hash,
            reference.surface_hierarchy_input_hash
        );
        assert_eq!(result.drainage_input_hash, reference.drainage_input_hash);
        assert!(hashes.insert(result.derived_evidence_hash));
    }
    assert_eq!(hashes.len(), 11);
    assert_eq!(namespaces.len(), 11);
}

fn run_asymmetric_y_isolated_cost_probe(spacing_km: f64) {
    // Fixture construction includes G0, S0 and D0 and intentionally occurs
    // before the isolated O0a timer starts.
    let fixture = asymmetric_y_fixture(spacing_km);
    let start = std::time::Instant::now();
    let relationships =
        build_relationships(&fixture, LandformRelationshipConfigV0::default()).unwrap();
    let elapsed = start.elapsed();
    eprintln!(
        "O0a isolated asymmetric-Y: spacing_km={spacing_km}, cells={}, raw_faces={}, trace_segments={}, stations={}, regular_samples={}, candidate_tests={}, elapsed={elapsed:?}",
        fixture.graph.cell_count(),
        relationships.work_counts.raw_boundary_faces,
        relationships.work_counts.receiver_trace_segments,
        relationships.work_counts.reach_stations,
        relationships.work_counts.regular_cross_section_samples,
        relationships.work_counts.candidate_face_tests,
    );
}

#[test]
#[ignore = "isolated 8 km O0a cost probe; run explicitly for dated audit evidence"]
fn o0a_asymmetric_y_isolated_cost_probe_8_km() {
    run_asymmetric_y_isolated_cost_probe(8.0);
}

#[test]
#[ignore = "isolated 4 km O0a cost probe; run explicitly for dated audit evidence"]
fn o0a_asymmetric_y_isolated_cost_probe_4_km() {
    run_asymmetric_y_isolated_cost_probe(4.0);
}

#[test]
#[ignore = "isolated 2 km O0a cost probe; run explicitly for dated audit evidence"]
fn o0a_asymmetric_y_isolated_cost_probe_2_km() {
    run_asymmetric_y_isolated_cost_probe(2.0);
}

fn compensated_test_sum(values: impl IntoIterator<Item = f64>) -> f64 {
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

fn finite_distribution(mut values: Vec<f64>) -> (usize, Option<f64>, Option<f64>, Option<f64>) {
    assert!(values.iter().all(|value| value.is_finite()));
    values.sort_by(f64::total_cmp);
    let count = values.len();
    let minimum = values.first().copied();
    let maximum = values.last().copied();
    let median = if count == 0 {
        None
    } else if count % 2 == 1 {
        Some(values[count / 2])
    } else {
        Some(0.5 * (values[count / 2 - 1] + values[count / 2]))
    };
    (count, minimum, median, maximum)
}

fn print_o0a_evidence_summary(fixture_name: &str, spacing_km: f64, fixture: &RelationshipFixture) {
    let result = build_relationships(fixture, LandformRelationshipConfigV0::default()).unwrap();
    let mut role_count = [0usize; 2];
    let mut role_length = [0.0; 2];
    let mut ancestry_count = [0usize; 4];
    let mut ancestry_length = [0.0; 4];
    let mut bilateral_count = 0usize;
    let mut bilateral_length = Vec::new();
    let mut unconditioned_count = 0usize;
    let mut unconditioned_length = Vec::new();
    for face in &result.backed_boundary_faces {
        let role = match face.role {
            BoundaryFaceRoleKindV0::FlowTransition => 0,
            BoundaryFaceRoleKindV0::LateralBoundaryCandidate => 1,
        };
        role_count[role] += 1;
        role_length[role] += face.physical_length_km;
        let ancestry = match face.owner_ancestry {
            OwnerAncestryV0::Same => 0,
            OwnerAncestryV0::FirstIsAncestor => 1,
            OwnerAncestryV0::SecondIsAncestor => 2,
            OwnerAncestryV0::Incomparable => 3,
        };
        ancestry_count[ancestry] += 1;
        ancestry_length[ancestry] += face.physical_length_km;
        if let Some(probe) = &face.bilateral_descent {
            if probe.bilateral_physical_descent {
                bilateral_count += 1;
                bilateral_length.push(face.physical_length_km);
            }
            if probe.unconditioned_bilateral_descent {
                unconditioned_count += 1;
                unconditioned_length.push(face.physical_length_km);
            }
        }
    }
    assert_eq!(
        role_count.iter().sum::<usize>(),
        result.backed_boundary_faces.len()
    );
    assert_eq!(
        ancestry_count.iter().sum::<usize>(),
        result.backed_boundary_faces.len()
    );
    assert_eq!(
        result.work_counts.raw_boundary_faces as usize,
        result.backed_boundary_faces.len()
    );
    assert!(role_length.iter().all(|length| length.is_finite()));
    assert!(ancestry_length.iter().all(|length| length.is_finite()));

    let bilateral_length = compensated_test_sum(bilateral_length);
    let unconditioned_length = compensated_test_sum(unconditioned_length);
    assert!(bilateral_length.is_finite() && unconditioned_length.is_finite());
    let highland_ratios = finite_distribution(
        result
            .highland_boundary_relationships
            .iter()
            .filter_map(|relationship| relationship.unconditioned_length_ratio)
            .collect(),
    );

    let saddle_total = result.saddle_boundary_associations.len();
    let saddle_present = result
        .saddle_boundary_associations
        .iter()
        .filter(|association| association.boundary_face_index.is_some())
        .count();
    let saddle_within = result
        .saddle_boundary_associations
        .iter()
        .filter(|association| association.within_covering_radius == Some(true))
        .count();
    let saddle_bilateral = result
        .saddle_boundary_associations
        .iter()
        .filter(|association| {
            association
                .bilateral_descent
                .as_ref()
                .is_some_and(|descent| descent.bilateral_physical_descent)
        })
        .count();
    let saddle_unconditioned = result
        .saddle_boundary_associations
        .iter()
        .filter(|association| {
            association
                .bilateral_descent
                .as_ref()
                .is_some_and(|descent| descent.unconditioned_bilateral_descent)
        })
        .count();

    let mut censor_count = [0usize; 7];
    let mut uncensored_sides = 0usize;
    let mut positive_relief = 0usize;
    let mut available_relief = 0usize;
    let mut spans = Vec::new();
    let mut station_count = 0usize;
    for station in result
        .reach_cross_section_probes
        .iter()
        .flat_map(|probe| &probe.stations)
    {
        station_count += 1;
        if let Some(span) = station.relative_relief_span_km {
            spans.push(span);
        }
        for side in [&station.left, &station.right] {
            match side.censor_reason {
                None => uncensored_sides += 1,
                Some(reason) => {
                    let index = match reason {
                        SectionCensorReasonV0::AxisOutsideCatchment => 0,
                        SectionCensorReasonV0::AxisOnBoundary => 1,
                        SectionCensorReasonV0::CollinearBoundary => 2,
                        SectionCensorReasonV0::DomainBoundary => 3,
                        SectionCensorReasonV0::FlowTransition => 4,
                        SectionCensorReasonV0::AmbiguousFaceGeometry => 5,
                        SectionCensorReasonV0::NoCatchmentExitWithinSupport => 6,
                    };
                    censor_count[index] += 1;
                }
            }
            if let Some(positive) = side.positive_boundary_relief {
                available_relief += 1;
                positive_relief += usize::from(positive);
            }
        }
    }
    assert_eq!(station_count as u64, result.work_counts.reach_stations);
    assert_eq!(
        uncensored_sides + censor_count.iter().sum::<usize>(),
        2 * station_count
    );
    let span_distribution = finite_distribution(spans);

    eprintln!(
        "O0a evidence: fixture={fixture_name}, spacing_km={spacing_km}, cells={}, role_count=[flow:{},lateral:{}], role_length_km=[flow:{:.12},lateral:{:.12}], ancestry_count=[same:{},first_ancestor:{},second_ancestor:{},incomparable:{}], ancestry_length_km=[same:{:.12},first_ancestor:{:.12},second_ancestor:{:.12},incomparable:{:.12}], lateral_descent=[bilateral_count:{bilateral_count},bilateral_length_km:{bilateral_length:.12},unconditioned_count:{unconditioned_count},unconditioned_length_km:{unconditioned_length:.12}], highlands=[count:{},ratio_distribution:{highland_ratios:?}], saddles=[total:{saddle_total},associated:{saddle_present},within_radius:{saddle_within},bilateral:{saddle_bilateral},unconditioned:{saddle_unconditioned}], reaches=[count:{},stations:{station_count}], sides=[uncensored:{uncensored_sides},censors_axis_outside:{},axis_on_boundary:{},collinear:{},domain:{},flow_transition:{},ambiguous:{},no_exit:{},positive_relief:{positive_relief},relief_available:{available_relief}], span_distribution={span_distribution:?}, candidate_tests={}",
        fixture.graph.cell_count(),
        role_count[0],
        role_count[1],
        role_length[0],
        role_length[1],
        ancestry_count[0],
        ancestry_count[1],
        ancestry_count[2],
        ancestry_count[3],
        ancestry_length[0],
        ancestry_length[1],
        ancestry_length[2],
        ancestry_length[3],
        result.highland_boundary_relationships.len(),
        result.reach_cross_section_probes.len(),
        censor_count[0],
        censor_count[1],
        censor_count[2],
        censor_count[3],
        censor_count[4],
        censor_count[5],
        censor_count[6],
        result.work_counts.candidate_face_tests,
    );
}

#[test]
#[ignore = "O0a evidence summary probe; run explicitly for dated audit evidence"]
fn o0a_asymmetric_y_and_shed_evidence_summary_at_8_4_2_km() {
    for spacing_km in [8.0, 4.0, 2.0] {
        let asymmetric = asymmetric_y_fixture(spacing_km);
        print_o0a_evidence_summary("asymmetric_y", spacing_km, &asymmetric);
        let shed = west_east_fixture(spacing_km, false);
        print_o0a_evidence_summary("symmetric_shed", spacing_km, &shed);
    }
}
