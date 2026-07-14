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
