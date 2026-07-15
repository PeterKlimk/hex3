//! End-to-end manufactured gates for O0b common-packet assembly.

use super::*;
use crate::world::landscape::{BoundarySide, OutletPortal, OutletPortalId};

struct PacketFixture {
    graph: EvaluationSurfaceGraphV0,
    elevation_km: Vec<f64>,
    scored_cell: Vec<bool>,
    runoff: Vec<f64>,
    surface_config: SurfaceHierarchyConfigV0,
    drainage_config: DrainageConfigV0,
    relationship_configs: [LandformRelationshipConfigV0; 11],
    hierarchy: SurfaceHierarchyV0,
    drainage: EvaluationDrainageV0,
    relationships: Vec<LandformRelationshipsV0>,
    geometry_identity: PacketGeometryIdentityV0,
    population: CommonEvaluationPopulationV0,
}

impl PacketFixture {
    fn input(&self) -> LandformPacketAssemblyInputV0<'_> {
        LandformPacketAssemblyInputV0 {
            graph: &self.graph,
            physical_elevation_km: &self.elevation_km,
            scored_cell: &self.scored_cell,
            local_runoff_supply: &self.runoff,
            surface_config: self.surface_config,
            drainage_config: self.drainage_config,
            relationship_configs: &self.relationship_configs,
            surface_hierarchy: &self.hierarchy,
            drainage: &self.drainage,
            relationship_payloads: &self.relationships,
            geometry_identity: self.geometry_identity,
            population: self.population.clone(),
        }
    }
}

fn segment_network_cost(point: DVec3, start: DVec3, end: DVec3, offset: f64) -> f64 {
    let segment = end - start;
    let fraction = ((point - start).dot(segment) / segment.length_squared()).clamp(0.0, 1.0);
    offset + fraction * segment.length() + 2.0 * point.distance(start + fraction * segment)
}

fn asymmetric_y_population() -> CommonEvaluationPopulationV0 {
    let mut population = CommonEvaluationPopulationV0 {
        coordinate_frame: CoordinateFrameV0::LandscapeTestbedCartesianXyKmV0,
        declared_domain: DeclaredDomainV0::RequestedRegularPatchV0 {
            width_km: 128.0,
            height_km: 96.0,
        },
        scored_policy: ScoredPolicyV0::WholeGraphSupportV0,
        runoff_policy: RunoffPolicyV0::AsymmetricYAffinePerAreaV0 {
            base_rate: 0.3,
            x_gradient_per_km: 0.002,
        },
        semantic_portals: vec![DeclaredPortalV0 {
            id: 23,
            side: DeclaredPortalSideV0::South,
            span_start_km: -1.0,
            span_end_km: 1.0,
            base_level_km: 0.0,
        }],
        population_definition_hash: 0,
    };
    population.population_definition_hash = population_definition_hash_v0(&population).unwrap();
    population
}

fn asymmetric_y_packet_fixture(spacing_km: f64) -> PacketFixture {
    const WIDTH_KM: f64 = 128.0;
    const HEIGHT_KM: f64 = 96.0;
    let portal = OutletPortal {
        id: OutletPortalId(23),
        side: BoundarySide::South,
        span_start_km: -1.0,
        span_end_km: 1.0,
        base_level_km: 0.0,
    };
    let mesh = LandscapeMesh::uniform_planar_hex_with_portals(
        WIDTH_KM,
        HEIGHT_KM,
        spacing_km,
        std::slice::from_ref(&portal),
    )
    .unwrap();
    let surface_config = SurfaceHierarchyConfigV0::default();
    let controls = build_regular_hex_control_volumes_v0(&mesh, &surface_config).unwrap();
    let graph = adapt_landscape_graph_v0(&mesh, &controls, &surface_config).unwrap();

    let outlet = DVec3::new(0.0, -48.0, 0.0);
    let junction = DVec3::ZERO;
    let west_head = DVec3::new(-48.0, 48.0, 0.0);
    let east_head = DVec3::new(48.0, 48.0, 0.0);
    let trunk_length = outlet.distance(junction);
    let elevation_km = graph
        .cell_center_km
        .iter()
        .map(|&point| {
            0.01 * segment_network_cost(point, outlet, junction, 0.0)
                .min(segment_network_cost(
                    point,
                    junction,
                    west_head,
                    trunk_length,
                ))
                .min(segment_network_cost(
                    point,
                    junction,
                    east_head,
                    trunk_length,
                ))
        })
        .collect::<Vec<_>>();
    let scored_cell = vec![true; graph.cell_count()];
    let runoff = graph
        .cell_area_km2
        .iter()
        .zip(&graph.cell_center_km)
        .map(|(&area, center)| 0.3 * (1.0 + 0.002 * center.x) * area)
        .collect::<Vec<_>>();
    let drainage_config = DrainageConfigV0::default();
    let hierarchy =
        build_surface_hierarchy_v0(&graph, &elevation_km, &scored_cell, surface_config).unwrap();
    let drainage =
        build_evaluation_drainage_v0(&graph, &elevation_km, &runoff, drainage_config).unwrap();
    let geometry_identity = PacketGeometryIdentityV0::LandscapeRegularPlanar {
        nominal_spacing_km: spacing_km,
        canonical_graph_hash: relationship_graph_hash_v0(&graph).unwrap(),
    };
    let relationship_configs = registered_relationship_configs_v0();
    let relationships = relationship_configs
        .iter()
        .map(|&config| {
            build_landform_relationships_v0(
                &graph,
                &elevation_km,
                &scored_cell,
                &runoff,
                surface_config,
                drainage_config,
                &hierarchy,
                &drainage,
                geometry_identity,
                config,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let population = asymmetric_y_population();

    PacketFixture {
        graph,
        elevation_km,
        scored_cell,
        runoff,
        surface_config,
        drainage_config,
        relationship_configs,
        hierarchy,
        drainage,
        relationships,
        geometry_identity,
        population,
    }
}

pub(super) fn assembled_asymmetric_y_packet_at(spacing_km: f64) -> LandformObjectPacketCoreV0 {
    let fixture = asymmetric_y_packet_fixture(spacing_km);
    assemble_landform_object_packet_v0(fixture.input()).unwrap()
}

const ISOLATED_FOUR_CONE_WIDTH_KM: f64 = 720.0;
const ISOLATED_FOUR_CONE_HEIGHT_KM: f64 = 240.0 * 1.732_050_807_568_877_2;
const ISOLATED_FOUR_CONES: [(DVec3, f64); 4] = [
    (DVec3::new(-200.0, 0.0, 0.0), 0.50),
    (DVec3::new(-65.0, 0.0, 0.0), 0.45),
    (DVec3::new(65.0, 0.0, 0.0), 0.55),
    (DVec3::new(200.0, 0.0, 0.0), 0.48),
];

fn isolated_four_cone_packet_fixture(spacing_km: f64) -> PacketFixture {
    let portal = OutletPortal {
        id: OutletPortalId(41),
        side: BoundarySide::South,
        span_start_km: -16.0,
        span_end_km: 16.0,
        base_level_km: 0.0,
    };
    let mesh = LandscapeMesh::uniform_planar_hex_with_portals(
        ISOLATED_FOUR_CONE_WIDTH_KM,
        ISOLATED_FOUR_CONE_HEIGHT_KM,
        spacing_km,
        std::slice::from_ref(&portal),
    )
    .unwrap();
    let surface_config = SurfaceHierarchyConfigV0::default();
    let controls = build_regular_hex_control_volumes_v0(&mesh, &surface_config).unwrap();
    let graph = adapt_landscape_graph_v0(&mesh, &controls, &surface_config).unwrap();
    let elevation_km = graph
        .cell_center_km
        .iter()
        .map(|point| {
            let value = ISOLATED_FOUR_CONES
                .iter()
                .map(|(center, height)| height - 0.010 * point.distance(*center))
                .fold(f64::NEG_INFINITY, f64::max)
                .max(0.0);
            if value == 0.0 {
                0.0
            } else {
                value
            }
        })
        .collect::<Vec<_>>();
    let scored_cell = vec![true; graph.cell_count()];
    let runoff = graph
        .cell_area_km2
        .iter()
        .map(|area| 0.1 * area)
        .collect::<Vec<_>>();
    let drainage_config = DrainageConfigV0::default();
    let hierarchy =
        build_surface_hierarchy_v0(&graph, &elevation_km, &scored_cell, surface_config).unwrap();
    let drainage =
        build_evaluation_drainage_v0(&graph, &elevation_km, &runoff, drainage_config).unwrap();
    let geometry_identity = PacketGeometryIdentityV0::LandscapeRegularPlanar {
        nominal_spacing_km: spacing_km,
        canonical_graph_hash: relationship_graph_hash_v0(&graph).unwrap(),
    };
    let relationship_configs = registered_relationship_configs_v0();
    let relationships = relationship_configs
        .iter()
        .map(|&config| {
            build_landform_relationships_v0(
                &graph,
                &elevation_km,
                &scored_cell,
                &runoff,
                surface_config,
                drainage_config,
                &hierarchy,
                &drainage,
                geometry_identity,
                config,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut population = CommonEvaluationPopulationV0 {
        coordinate_frame: CoordinateFrameV0::LandscapeTestbedCartesianXyKmV0,
        declared_domain: DeclaredDomainV0::RequestedRegularPatchV0 {
            width_km: ISOLATED_FOUR_CONE_WIDTH_KM,
            height_km: ISOLATED_FOUR_CONE_HEIGHT_KM,
        },
        scored_policy: ScoredPolicyV0::WholeGraphSupportV0,
        runoff_policy: RunoffPolicyV0::UniformPerAreaV0 { rate: 0.1 },
        semantic_portals: vec![DeclaredPortalV0 {
            id: 41,
            side: DeclaredPortalSideV0::South,
            span_start_km: -16.0,
            span_end_km: 16.0,
            base_level_km: 0.0,
        }],
        population_definition_hash: 0,
    };
    population.population_definition_hash = population_definition_hash_v0(&population).unwrap();

    PacketFixture {
        graph,
        elevation_km,
        scored_cell,
        runoff,
        surface_config,
        drainage_config,
        relationship_configs,
        hierarchy,
        drainage,
        relationships,
        geometry_identity,
        population,
    }
}

pub(super) fn assembled_isolated_four_cone_packet_at(
    spacing_km: f64,
) -> LandformObjectPacketCoreV0 {
    let fixture = isolated_four_cone_packet_fixture(spacing_km);
    assemble_landform_object_packet_v0(fixture.input()).unwrap()
}

pub(super) fn assembled_linked_four_cone_packet_at(spacing_km: f64) -> LandformObjectPacketCoreV0 {
    const WIDTH_KM: f64 = 1120.0;
    const HEIGHT_KM: f64 = 480.0 * 1.732_050_807_568_877_2;
    const CONES: [(DVec3, f64); 4] = [
        (DVec3::new(-180.0, -40.0, 0.0), 2.4),
        (DVec3::new(-60.0, 0.0, 0.0), 2.0),
        (DVec3::new(60.0, 0.0, 0.0), 2.2),
        (DVec3::new(180.0, 40.0, 0.0), 1.9),
    ];
    let portal = OutletPortal {
        id: OutletPortalId(41),
        side: BoundarySide::South,
        span_start_km: -16.0,
        span_end_km: 16.0,
        base_level_km: 0.0,
    };
    let mesh = LandscapeMesh::uniform_planar_hex_with_portals(
        WIDTH_KM,
        HEIGHT_KM,
        spacing_km,
        std::slice::from_ref(&portal),
    )
    .unwrap();
    let surface_config = SurfaceHierarchyConfigV0::default();
    let controls = build_regular_hex_control_volumes_v0(&mesh, &surface_config).unwrap();
    let graph = adapt_landscape_graph_v0(&mesh, &controls, &surface_config).unwrap();
    let elevation_km = graph
        .cell_center_km
        .iter()
        .map(|point| {
            let value = CONES
                .iter()
                .map(|(center, height)| height - 0.010 * point.distance(*center))
                .fold(f64::NEG_INFINITY, f64::max)
                .max(0.0);
            if value == 0.0 {
                0.0
            } else {
                value
            }
        })
        .collect::<Vec<_>>();
    let scored_cell = vec![true; graph.cell_count()];
    let runoff = graph
        .cell_area_km2
        .iter()
        .map(|area| 0.1 * area)
        .collect::<Vec<_>>();
    let drainage_config = DrainageConfigV0::default();
    let hierarchy =
        build_surface_hierarchy_v0(&graph, &elevation_km, &scored_cell, surface_config).unwrap();
    let drainage =
        build_evaluation_drainage_v0(&graph, &elevation_km, &runoff, drainage_config).unwrap();
    let geometry_identity = PacketGeometryIdentityV0::LandscapeRegularPlanar {
        nominal_spacing_km: spacing_km,
        canonical_graph_hash: relationship_graph_hash_v0(&graph).unwrap(),
    };
    let relationship_configs = registered_relationship_configs_v0();
    let relationships = relationship_configs
        .iter()
        .map(|&config| {
            build_landform_relationships_v0(
                &graph,
                &elevation_km,
                &scored_cell,
                &runoff,
                surface_config,
                drainage_config,
                &hierarchy,
                &drainage,
                geometry_identity,
                config,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut population = CommonEvaluationPopulationV0 {
        coordinate_frame: CoordinateFrameV0::LandscapeTestbedCartesianXyKmV0,
        declared_domain: DeclaredDomainV0::RequestedRegularPatchV0 {
            width_km: WIDTH_KM,
            height_km: HEIGHT_KM,
        },
        scored_policy: ScoredPolicyV0::WholeGraphSupportV0,
        runoff_policy: RunoffPolicyV0::UniformPerAreaV0 { rate: 0.1 },
        semantic_portals: vec![DeclaredPortalV0 {
            id: 41,
            side: DeclaredPortalSideV0::South,
            span_start_km: -16.0,
            span_end_km: 16.0,
            base_level_km: 0.0,
        }],
        population_definition_hash: 0,
    };
    population.population_definition_hash = population_definition_hash_v0(&population).unwrap();
    assemble_landform_object_packet_v0(LandformPacketAssemblyInputV0 {
        graph: &graph,
        physical_elevation_km: &elevation_km,
        scored_cell: &scored_cell,
        local_runoff_supply: &runoff,
        surface_config,
        drainage_config,
        relationship_configs: &relationship_configs,
        surface_hierarchy: &hierarchy,
        drainage: &drainage,
        relationship_payloads: &relationships,
        geometry_identity,
        population,
    })
    .unwrap()
}

fn assert_isolated_four_cone_predecessors(spacing_km: f64, expected_cells: usize) {
    let fixture = isolated_four_cone_packet_fixture(spacing_km);
    assert_eq!(fixture.graph.cell_count(), expected_cells);
    assert_eq!(fixture.hierarchy.roots.len(), 4);
    assert_eq!(fixture.hierarchy.populations.reference.len(), 4);
    assert_eq!(fixture.hierarchy.reference_highlands.len(), 4);

    let mut labels = std::collections::BTreeSet::new();
    for &peak_id in &fixture.hierarchy.populations.reference {
        let peak = &fixture.hierarchy.peaks[peak_id as usize];
        assert!(!peak.equal_elder_ambiguous);
        let anchor = fixture.graph.cell_center_km[peak.anchor_cell as usize];
        let mut distances = ISOLATED_FOUR_CONES
            .iter()
            .enumerate()
            .map(|(label, (center, _))| (anchor.distance_squared(*center), label))
            .collect::<Vec<_>>();
        distances.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        assert!(distances[0].0 < distances[1].0);
        assert!(labels.insert(distances[0].1));
    }
    assert_eq!(labels, std::collections::BTreeSet::from([0, 1, 2, 3]));

    assert!(fixture
        .drainage
        .routing
        .filled_elevation_km
        .iter()
        .zip(&fixture.elevation_km)
        .all(|(filled, physical)| filled.to_bits() == physical.to_bits()));
    assert!(fixture
        .drainage
        .routing
        .fill_supported
        .iter()
        .all(|supported| !supported));
    assert!(fixture.drainage.depressions.is_empty());
    assert_eq!(fixture.relationship_configs.len(), 11);
    assert_eq!(fixture.relationships.len(), 11);

    let packet = assemble_landform_object_packet_v0(fixture.input()).unwrap();
    assert_eq!(packet.relationship_payloads.len(), 11);
    assert_ne!(packet.derived_common_packet_hash, 0);
    assert!(!landform_object_packet_bytes_v0(&packet).unwrap().is_empty());
}

#[test]
#[ignore = "explicit preregistered isolated-four-cone 8 km predecessor audit"]
fn isolated_four_cone_8km_predecessors_are_packet_admissible() {
    assert_isolated_four_cone_predecessors(8.0, 5_400);
}

#[test]
#[ignore = "explicit preregistered isolated-four-cone 4 km predecessor audit"]
fn isolated_four_cone_4km_predecessors_are_packet_admissible() {
    assert_isolated_four_cone_predecessors(4.0, 21_600);
}

#[test]
#[ignore = "explicit preregistered isolated-four-cone 2 km predecessor audit"]
fn isolated_four_cone_2km_predecessors_are_packet_admissible() {
    assert_isolated_four_cone_predecessors(2.0, 86_400);
}

#[test]
fn o0b_assembly_is_deterministic_and_canonicalizes_namespace_enumeration() {
    let mut fixture = asymmetric_y_packet_fixture(4.0);
    let reference = assemble_landform_object_packet_v0(fixture.input()).unwrap();
    let repeated = assemble_landform_object_packet_v0(fixture.input()).unwrap();
    assert_eq!(reference, repeated);
    assert_eq!(
        landform_object_packet_bytes_v0(&reference).unwrap(),
        landform_object_packet_bytes_v0(&repeated).unwrap()
    );
    assert_ne!(reference.derived_common_packet_hash, 0);
    assert_eq!(reference.relationship_payloads.len(), 11);
    assert_eq!(
        reference
            .relationship_payloads
            .iter()
            .map(|payload| payload.run_namespace)
            .collect::<Vec<_>>(),
        vec![
            RelationshipRunNamespaceV0::Reference,
            RelationshipRunNamespaceV0::StationSpacingLow,
            RelationshipRunNamespaceV0::StationSpacingHigh,
            RelationshipRunNamespaceV0::CrossSectionHalfLengthLow,
            RelationshipRunNamespaceV0::CrossSectionHalfLengthHigh,
            RelationshipRunNamespaceV0::CrossSectionSampleStepLow,
            RelationshipRunNamespaceV0::CrossSectionSampleStepHigh,
            RelationshipRunNamespaceV0::RelativeHeightFractionLow,
            RelationshipRunNamespaceV0::RelativeHeightFractionHigh,
            RelationshipRunNamespaceV0::MaximumDownstreamSupportLow,
            RelationshipRunNamespaceV0::MaximumDownstreamSupportHigh,
        ]
    );

    fixture.relationships.reverse();
    fixture.relationship_configs.reverse();
    let reordered = assemble_landform_object_packet_v0(fixture.input()).unwrap();
    assert_eq!(reference, reordered);
    assert_eq!(
        landform_object_packet_bytes_v0(&reference).unwrap(),
        landform_object_packet_bytes_v0(&reordered).unwrap()
    );

    let envelope_a = LandformPacketEnvelopeV0 {
        core: reference.clone(),
        run_id: Some("run-a".into()),
        revision: Some("revision-a".into()),
        dirty: Some(false),
        arm_label: Some("arbitrary-h".into()),
        native_provenance_hash: Some(1),
    };
    let envelope_b = LandformPacketEnvelopeV0 {
        core: reference.clone(),
        run_id: Some("run-b".into()),
        revision: Some("revision-b".into()),
        dirty: Some(true),
        arm_label: Some("arbitrary-c".into()),
        native_provenance_hash: Some(u64::MAX),
    };
    assert_ne!(envelope_a, envelope_b);
    assert_eq!(
        landform_object_packet_bytes_v0(&envelope_a.core).unwrap(),
        landform_object_packet_bytes_v0(&envelope_b.core).unwrap()
    );
}

#[test]
fn o0b_assembly_rejects_missing_duplicate_and_mutated_predecessors() {
    let mut fixture = asymmetric_y_packet_fixture(4.0);

    let missing = fixture.relationships.pop().unwrap();
    assert!(matches!(
        assemble_landform_object_packet_v0(fixture.input()),
        Err(PacketAssemblyErrorV0::MissingRelationshipNamespace(_))
    ));

    fixture.relationships.push(fixture.relationships[0].clone());
    assert!(matches!(
        assemble_landform_object_packet_v0(fixture.input()),
        Err(PacketAssemblyErrorV0::DuplicateRelationshipNamespace(_))
    ));

    fixture.relationships.pop();
    fixture.relationships.push(missing);

    let mut nonrepresentable_population = fixture.population.clone();
    nonrepresentable_population.semantic_portals[0].base_level_km = 0.1;
    nonrepresentable_population.population_definition_hash =
        population_definition_hash_v0(&nonrepresentable_population).unwrap();
    let mut nonrepresentable_input = fixture.input();
    nonrepresentable_input.population = nonrepresentable_population;
    assert_eq!(
        assemble_landform_object_packet_v0(nonrepresentable_input),
        Err(PacketAssemblyErrorV0::InvalidPopulation(
            "semantic_portal_base_level_not_mesh_representable"
        ))
    );

    fixture.elevation_km[0] = f64::from_bits(fixture.elevation_km[0].to_bits() ^ 1);
    assert!(matches!(
        assemble_landform_object_packet_v0(fixture.input()),
        Err(PacketAssemblyErrorV0::SurfaceHierarchyHashMismatch { .. })
            | Err(PacketAssemblyErrorV0::SurfaceHierarchy(_))
    ));
}

#[test]
fn o0b_hash_helpers_reject_noncanonical_arrays_and_bind_population_fields() {
    assert!(canonical_array_hash_v0(&[0.0, 1.0, 2.0]).is_ok());
    assert!(matches!(
        canonical_array_hash_v0(&[-0.0]),
        Err(PacketAssemblyErrorV0::NonCanonicalZero { .. })
    ));
    assert!(matches!(
        canonical_array_hash_v0(&[f64::NAN]),
        Err(PacketAssemblyErrorV0::NonFiniteInput(_))
    ));

    let population = asymmetric_y_population();
    let original = population_definition_hash_v0(&population).unwrap();
    let mut changed = population.clone();
    changed.semantic_portals[0].base_level_km = 1.0;
    assert_ne!(population_definition_hash_v0(&changed).unwrap(), original);
    changed = population;
    changed.declared_domain = DeclaredDomainV0::RequestedRegularPatchV0 {
        width_km: 129.0,
        height_km: 96.0,
    };
    assert_ne!(population_definition_hash_v0(&changed).unwrap(), original);
}

fn assert_common_core_split_round_trip(packet: &LandformObjectPacketCoreV0) {
    let original_bytes = landform_object_packet_bytes_v0(packet).unwrap();
    let original_hash = packet.derived_common_packet_hash;
    let (core, reference, suite) = split_landform_object_packet_v0(packet).unwrap();

    validate_common_planar_evidence_core_v0(&core).unwrap();
    validate_reference_relationship_evidence_v0(&reference).unwrap();
    validate_reference_relationship_evidence_against_core_v0(&core, &reference).unwrap();
    validate_relationship_sensitivity_suite_v0(&suite).unwrap();
    validate_relationship_sensitivity_suite_against_core_v0(&core, &suite).unwrap();
    assert_eq!(reference.core_hash, core.derived_core_hash);
    assert_eq!(suite.core_hash, core.derived_core_hash);

    let core_bytes = common_planar_evidence_core_bytes_v0(&core).unwrap();
    let reference_bytes = reference_relationship_evidence_bytes_v0(&reference).unwrap();
    let suite_bytes = relationship_sensitivity_suite_bytes_v0(&suite).unwrap();
    assert_eq!(
        decode_common_planar_evidence_core_v0(&core_bytes).unwrap(),
        core
    );
    assert_eq!(
        decode_reference_relationship_evidence_v0(&reference_bytes).unwrap(),
        reference
    );
    assert_eq!(
        decode_relationship_sensitivity_suite_v0(&suite_bytes).unwrap(),
        suite
    );

    let materialized = materialize_landform_object_packet_v0(&core, &reference, &suite).unwrap();
    assert_eq!(&materialized, packet);
    assert_eq!(materialized.derived_common_packet_hash, original_hash);
    assert_eq!(
        landform_object_packet_bytes_v0(&materialized).unwrap(),
        original_bytes
    );

    let repeated = split_landform_object_packet_v0(&materialized).unwrap();
    assert_eq!(repeated.0, core);
    assert_eq!(repeated.1, reference);
    assert_eq!(repeated.2, suite);
    assert_eq!(
        common_planar_evidence_core_bytes_v0(&repeated.0).unwrap(),
        core_bytes
    );
    assert_eq!(
        reference_relationship_evidence_bytes_v0(&repeated.1).unwrap(),
        reference_bytes
    );
    assert_eq!(
        relationship_sensitivity_suite_bytes_v0(&repeated.2).unwrap(),
        suite_bytes
    );
}

#[test]
fn common_core_split_is_an_exact_inverse_for_bounded_asymmetric_y() {
    assert_common_core_split_round_trip(&assembled_asymmetric_y_packet_at(4.0));
}

#[test]
#[ignore = "explicit common-core 8/4/2 exact decomposition matrix"]
fn common_core_split_is_an_exact_inverse_for_registered_fixture_matrix() {
    for spacing_km in [8.0, 4.0, 2.0] {
        assert_common_core_split_round_trip(&assembled_asymmetric_y_packet_at(spacing_km));
        assert_common_core_split_round_trip(&assembled_isolated_four_cone_packet_at(spacing_km));
    }
    for spacing_km in [8.0, 4.0] {
        assert_common_core_split_round_trip(&assembled_linked_four_cone_packet_at(spacing_km));
    }
}

fn rehash_suite(suite: &mut RelationshipSensitivitySuiteV0) {
    suite.derived_suite_hash = relationship_sensitivity_suite_hash_v0(suite).unwrap();
}

fn rehash_reference(reference: &mut ReferenceRelationshipEvidenceV0) {
    reference.derived_reference_hash = reference_relationship_evidence_hash_v0(reference).unwrap();
}

#[test]
fn common_core_sidecars_reject_wrong_shape_binding_and_rehashed_semantic_mutation() {
    let packet = assembled_asymmetric_y_packet_at(4.0);
    let (core, reference, suite) = split_landform_object_packet_v0(&packet).unwrap();

    let mut missing = suite.clone();
    missing.payloads.pop();
    rehash_suite(&mut missing);
    assert!(validate_relationship_sensitivity_suite_v0(&missing).is_err());

    let mut duplicate = suite.clone();
    duplicate.payloads[1] = duplicate.payloads[0].clone();
    rehash_suite(&mut duplicate);
    assert!(validate_relationship_sensitivity_suite_v0(&duplicate).is_err());

    let mut extra = suite.clone();
    extra.payloads.push(extra.payloads[0].clone());
    rehash_suite(&mut extra);
    assert!(validate_relationship_sensitivity_suite_v0(&extra).is_err());

    let mut reordered = suite.clone();
    reordered.payloads.swap(0, 1);
    rehash_suite(&mut reordered);
    assert!(validate_relationship_sensitivity_suite_v0(&reordered).is_err());

    let mut reference_in_suite = suite.clone();
    reference_in_suite.payloads[0] = reference.payload.clone();
    rehash_suite(&mut reference_in_suite);
    assert!(validate_relationship_sensitivity_suite_v0(&reference_in_suite).is_err());

    let mut sensitivity_in_reference = reference.clone();
    sensitivity_in_reference.payload = suite.payloads[0].clone();
    rehash_reference(&mut sensitivity_in_reference);
    assert!(validate_reference_relationship_evidence_v0(&sensitivity_in_reference).is_err());

    let mut wrong_config = suite.clone();
    wrong_config.payloads[0].config = suite.payloads[1].config.clone();
    rehash_suite(&mut wrong_config);
    assert!(validate_relationship_sensitivity_suite_v0(&wrong_config).is_err());

    let foreign_packet = assembled_asymmetric_y_packet_at(8.0);
    let (foreign_core, _, _) = split_landform_object_packet_v0(&foreign_packet).unwrap();
    assert!(
        validate_reference_relationship_evidence_against_core_v0(&foreign_core, &reference)
            .is_err()
    );
    assert!(
        validate_relationship_sensitivity_suite_against_core_v0(&foreign_core, &suite).is_err()
    );

    let mut wrong_geometry = reference.clone();
    match &mut wrong_geometry.payload.geometry_identity {
        PacketGeometryIdentityV0::LandscapeRegularPlanar {
            canonical_graph_hash,
            ..
        } => *canonical_graph_hash ^= 1,
        PacketGeometryIdentityV0::ProjectedR1VoronoiCap { .. } => unreachable!(),
    }
    rehash_reference(&mut wrong_geometry);
    assert!(
        validate_reference_relationship_evidence_against_core_v0(&core, &wrong_geometry).is_err()
    );

    let mut wrong_surface_predecessor = reference.clone();
    wrong_surface_predecessor
        .payload
        .surface_hierarchy_input_hash ^= 1;
    rehash_reference(&mut wrong_surface_predecessor);
    assert!(validate_reference_relationship_evidence_against_core_v0(
        &core,
        &wrong_surface_predecessor
    )
    .is_err());

    let mut wrong_drainage_predecessor = reference.clone();
    wrong_drainage_predecessor.payload.drainage_input_hash ^= 1;
    rehash_reference(&mut wrong_drainage_predecessor);
    assert!(validate_reference_relationship_evidence_against_core_v0(
        &core,
        &wrong_drainage_predecessor
    )
    .is_err());

    let mut malformed = reference.clone();
    malformed.payload.backed_boundary_faces.clear();
    rehash_reference(&mut malformed);
    assert!(validate_reference_relationship_evidence_v0(&malformed).is_ok());
    assert!(validate_reference_relationship_evidence_against_core_v0(&core, &malformed).is_err());

    let mut trailing = common_planar_evidence_core_bytes_v0(&core).unwrap();
    trailing.push(0);
    assert!(decode_common_planar_evidence_core_v0(&trailing).is_err());
    let mut trailing = reference_relationship_evidence_bytes_v0(&reference).unwrap();
    trailing.push(0);
    assert!(decode_reference_relationship_evidence_v0(&trailing).is_err());
    let mut trailing = relationship_sensitivity_suite_bytes_v0(&suite).unwrap();
    trailing.push(0);
    assert!(decode_relationship_sensitivity_suite_v0(&trailing).is_err());
}

fn assert_rehashed_core_mutation_rejected(
    core: &CommonPlanarEvidenceCoreV0,
    mutate: impl FnOnce(&mut CommonPlanarEvidenceCoreV0),
) {
    let mut witness = core.clone();
    mutate(&mut witness);
    witness.derived_core_hash = common_planar_evidence_core_hash_v0(&witness).unwrap();
    assert!(validate_common_planar_evidence_core_v0(&witness).is_err());
}

#[test]
fn common_core_finite_mutation_matrix_binds_every_retained_field_class() {
    let packet = assembled_asymmetric_y_packet_at(4.0);
    let (core, _, _) = split_landform_object_packet_v0(&packet).unwrap();

    let mut wrong_schema = core.clone();
    wrong_schema.schema_version = "foreign".into();
    assert!(validate_common_planar_evidence_core_v0(&wrong_schema).is_err());
    let mut wrong_hash_version = core.clone();
    wrong_hash_version.hash_version = "foreign".into();
    assert!(validate_common_planar_evidence_core_v0(&wrong_hash_version).is_err());

    assert_rehashed_core_mutation_rejected(&core, |value| {
        let DeclaredDomainV0::RequestedRegularPatchV0 { width_km, .. } =
            &mut value.population.declared_domain;
        *width_km += 4.0;
    });
    assert_rehashed_core_mutation_rejected(&core, |value| match &mut value.geometry_identity {
        PacketGeometryIdentityV0::LandscapeRegularPlanar {
            canonical_graph_hash,
            ..
        } => *canonical_graph_hash ^= 1,
        PacketGeometryIdentityV0::ProjectedR1VoronoiCap { .. } => unreachable!(),
    });
    assert_rehashed_core_mutation_rejected(&core, |value| {
        value.graph.cell_center_km[0].x += 1.0e-6;
    });
    assert_rehashed_core_mutation_rejected(&core, |value| {
        value.physical_elevation_km[0] += 1.0e-6;
    });
    assert_rehashed_core_mutation_rejected(&core, |value| value.scored_cell[0] = false);
    assert_rehashed_core_mutation_rejected(&core, |value| {
        value.local_runoff_supply[0] += 1.0e-6;
    });
    assert_rehashed_core_mutation_rejected(&core, |value| {
        value.surface_config.closure_level_km += 1.0e-6;
    });
    assert_rehashed_core_mutation_rejected(&core, |value| {
        value.drainage_config.support_thresholds_km2[0] += 1.0;
    });
    assert_rehashed_core_mutation_rejected(&core, |value| {
        value.surface_hierarchy.derived_evidence_hash ^= 1;
    });
    assert_rehashed_core_mutation_rejected(&core, |value| {
        value.drainage.derived_evidence_hash ^= 1;
    });

    assert_rehashed_core_mutation_rejected(&core, |value| {
        value.population.semantic_portals[0].base_level_km = -0.0;
        value.population.population_definition_hash =
            population_definition_hash_v0(&value.population).unwrap();
    });
    assert_rehashed_core_mutation_rejected(&core, |value| {
        value.physical_elevation_km[0] = -0.0;
    });
    assert_rehashed_core_mutation_rejected(&core, |value| {
        value.local_runoff_supply[0] = -0.0;
    });
    assert_rehashed_core_mutation_rejected(&core, |value| {
        value.graph.cell_center_km.pop();
    });
    assert_rehashed_core_mutation_rejected(&core, |value| {
        value.physical_elevation_km.pop();
    });
    assert_rehashed_core_mutation_rejected(&core, |value| {
        value.local_runoff_supply[0] = f64::NAN;
    });

    let mut wrong_outer_hash = core;
    wrong_outer_hash.derived_core_hash ^= 1;
    assert!(matches!(
        validate_common_planar_evidence_core_v0(&wrong_outer_hash),
        Err(PacketAssemblyErrorV0::CoreHashMismatch { .. })
    ));
}

#[test]
#[ignore = "explicit common-core deterministic-repeat matrix"]
fn common_core_split_and_materialization_repeat_deterministically() {
    for packet in [
        assembled_asymmetric_y_packet_at(4.0),
        assembled_asymmetric_y_packet_at(8.0),
        assembled_isolated_four_cone_packet_at(4.0),
        assembled_isolated_four_cone_packet_at(2.0),
    ] {
        let first = split_landform_object_packet_v0(&packet).unwrap();
        let second = split_landform_object_packet_v0(&packet).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            common_planar_evidence_core_bytes_v0(&first.0).unwrap(),
            common_planar_evidence_core_bytes_v0(&second.0).unwrap()
        );
        assert_eq!(
            reference_relationship_evidence_bytes_v0(&first.1).unwrap(),
            reference_relationship_evidence_bytes_v0(&second.1).unwrap()
        );
        assert_eq!(
            relationship_sensitivity_suite_bytes_v0(&first.2).unwrap(),
            relationship_sensitivity_suite_bytes_v0(&second.2).unwrap()
        );
        let materialized =
            materialize_landform_object_packet_v0(&first.0, &first.1, &first.2).unwrap();
        assert_eq!(materialized, packet);
    }
}
