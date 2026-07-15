//! Owned O0b common-packet assembly.
//!
//! This module deliberately stops at predecessor validation, canonical owned
//! serialization, and hashing.  Correspondence geometry is implemented by a
//! later module over two successfully assembled cores.

use std::fmt;

use bincode::Options;
use serde::{Deserialize, Serialize};

use super::*;
use crate::world::landscape::{BoundarySide, LandscapeMesh, OutletPortal, OutletPortalId};

pub const O0B_PACKET_SCHEMA_VERSION: &str = "landform-object-packet-o0b-v0";
pub const O0B_PACKET_HASH_VERSION: &str = "fnv1a64-bincode-fixint-le-v0";
pub const COMMON_PLANAR_EVIDENCE_CORE_SCHEMA_VERSION: &str =
    "landform-common-planar-evidence-core-v0";
pub const REFERENCE_RELATIONSHIP_EVIDENCE_SCHEMA_VERSION: &str =
    "landform-reference-relationship-evidence-v0";
pub const RELATIONSHIP_SENSITIVITY_SUITE_SCHEMA_VERSION: &str =
    "landform-relationship-sensitivity-suite-v0";
pub const COMMON_PLANAR_ARTIFACT_HASH_VERSION: &str = "fnv1a64-bincode-fixint-le-v0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinateFrameV0 {
    LandscapeTestbedCartesianXyKmV0,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DeclaredDomainV0 {
    RequestedRegularPatchV0 { width_km: f64, height_km: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoredPolicyV0 {
    WholeGraphSupportV0,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RunoffPolicyV0 {
    ExactSameMeshArrayV0 {
        canonical_array_hash: u64,
    },
    UniformPerAreaV0 {
        rate: f64,
    },
    AsymmetricYAffinePerAreaV0 {
        base_rate: f64,
        x_gradient_per_km: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DeclaredPortalSideV0 {
    North,
    East,
    South,
    West,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DeclaredPortalV0 {
    pub id: u32,
    pub side: DeclaredPortalSideV0,
    pub span_start_km: f64,
    pub span_end_km: f64,
    pub base_level_km: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommonEvaluationPopulationV0 {
    pub coordinate_frame: CoordinateFrameV0,
    pub declared_domain: DeclaredDomainV0,
    pub scored_policy: ScoredPolicyV0,
    pub runoff_policy: RunoffPolicyV0,
    pub semantic_portals: Vec<DeclaredPortalV0>,
    pub population_definition_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceHierarchyConfigWireV0 {
    pub closure_level_km: f64,
    pub reference_persistence_km: f64,
    pub reference_min_footprint_km2: f64,
    pub persistence_sensitivity_km: [f64; 2],
    pub footprint_sensitivity_km2: [f64; 2],
    pub local_relief_radii_km: [f64; 3],
    pub summit_cap_depths_km: [f64; 3],
    pub gentle_grade_thresholds: [f64; 3],
    pub endpoint_match_abs_km: f64,
    pub planar_area_match_relative: f64,
    pub sphere_area_closure_relative: f64,
    pub linear_rank_relative: f64,
    pub orientation_ambiguity_anisotropy: f64,
    pub spherical_nonlocal_radius_rad: f64,
    pub schema_version: String,
    pub hash_version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DrainageConfigWireV0 {
    pub support_thresholds_km2: [f64; 3],
    pub balance_absolute_tolerance: f64,
    pub balance_relative_tolerance: f64,
    pub schema_version: String,
    pub hash_version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LandformRelationshipConfigWireV0 {
    pub station_spacing_km: f64,
    pub cross_section_half_length_km: f64,
    pub cross_section_sample_step_km: f64,
    pub relative_height_fraction: f64,
    pub maximum_downstream_support_km: f64,
    pub schema_version: String,
    pub hash_version: String,
}

/// Owned mirror of O0a's frozen source-declaration field order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LandformRelationshipsWireV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub geometry_identity: PacketGeometryIdentityV0,
    pub config: LandformRelationshipConfigWireV0,
    pub run_namespace: RelationshipRunNamespaceV0,
    pub surface_hierarchy_input_hash: u64,
    pub drainage_input_hash: u64,
    pub backed_boundary_faces: Vec<BackedBoundaryFaceV0>,
    pub highland_boundary_relationships: Vec<HighlandBoundaryRelationshipV0>,
    pub saddle_boundary_associations: Vec<SaddleBoundaryAssociationV0>,
    pub reach_cross_section_probes: Vec<ReachCrossSectionProbeV0>,
    pub work_counts: RelationshipWorkCountsV0,
    pub derived_evidence_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelationshipEvidenceHashV0 {
    pub run_namespace: RelationshipRunNamespaceV0,
    pub evidence_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredecessorEvidenceHashesV0 {
    pub surface_hierarchy_hash: u64,
    pub drainage_hash: u64,
    pub relationship_hashes: Vec<RelationshipEvidenceHashV0>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LandformObjectPacketCoreV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub population: CommonEvaluationPopulationV0,
    pub geometry_identity: PacketGeometryIdentityV0,
    pub graph: EvaluationSurfaceGraphV0,
    pub physical_elevation_km: Vec<f64>,
    pub scored_cell: Vec<bool>,
    pub local_runoff_supply: Vec<f64>,
    pub surface_config: SurfaceHierarchyConfigWireV0,
    pub drainage_config: DrainageConfigWireV0,
    pub relationship_configs: Vec<LandformRelationshipConfigWireV0>,
    pub surface_hierarchy: SurfaceHierarchyV0,
    pub drainage: EvaluationDrainageV0,
    pub relationship_payloads: Vec<LandformRelationshipsWireV0>,
    pub surface_hierarchy_input_hash: u64,
    pub drainage_input_hash: u64,
    pub predecessor_evidence_hashes: PredecessorEvidenceHashesV0,
    pub derived_common_packet_hash: u64,
}

/// S0/D0 evidence and the complete physical inputs needed to rebuild it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommonPlanarEvidenceCoreV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub population: CommonEvaluationPopulationV0,
    pub geometry_identity: PacketGeometryIdentityV0,
    pub graph: EvaluationSurfaceGraphV0,
    pub physical_elevation_km: Vec<f64>,
    pub scored_cell: Vec<bool>,
    pub local_runoff_supply: Vec<f64>,
    pub surface_config: SurfaceHierarchyConfigWireV0,
    pub drainage_config: DrainageConfigWireV0,
    pub surface_hierarchy: SurfaceHierarchyV0,
    pub drainage: EvaluationDrainageV0,
    pub derived_core_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceRelationshipEvidenceV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub core_hash: u64,
    pub payload: LandformRelationshipsWireV0,
    pub derived_reference_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelationshipSensitivitySuiteV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub core_hash: u64,
    pub payloads: Vec<LandformRelationshipsWireV0>,
    pub derived_suite_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LandformPacketEnvelopeV0 {
    pub core: LandformObjectPacketCoreV0,
    pub run_id: Option<String>,
    pub revision: Option<String>,
    pub dirty: Option<bool>,
    pub arm_label: Option<String>,
    pub native_provenance_hash: Option<u64>,
}

pub struct LandformPacketAssemblyInputV0<'a> {
    pub graph: &'a EvaluationSurfaceGraphV0,
    pub physical_elevation_km: &'a [f64],
    pub scored_cell: &'a [bool],
    pub local_runoff_supply: &'a [f64],
    pub surface_config: SurfaceHierarchyConfigV0,
    pub drainage_config: DrainageConfigV0,
    pub relationship_configs: &'a [LandformRelationshipConfigV0],
    pub surface_hierarchy: &'a SurfaceHierarchyV0,
    pub drainage: &'a EvaluationDrainageV0,
    pub relationship_payloads: &'a [LandformRelationshipsV0],
    pub geometry_identity: PacketGeometryIdentityV0,
    pub population: CommonEvaluationPopulationV0,
}

pub struct CommonPlanarEvidenceCoreAssemblyInputV0<'a> {
    pub graph: &'a EvaluationSurfaceGraphV0,
    pub physical_elevation_km: &'a [f64],
    pub scored_cell: &'a [bool],
    pub local_runoff_supply: &'a [f64],
    pub surface_config: SurfaceHierarchyConfigV0,
    pub drainage_config: DrainageConfigV0,
    pub surface_hierarchy: &'a SurfaceHierarchyV0,
    pub drainage: &'a EvaluationDrainageV0,
    pub geometry_identity: PacketGeometryIdentityV0,
    pub population: CommonEvaluationPopulationV0,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PacketAssemblyErrorV0 {
    WrongSchemaOrHashVersion(&'static str),
    UnsupportedGeometry,
    InvalidPopulation(&'static str),
    PopulationHashMismatch {
        stored: u64,
        computed: u64,
    },
    LengthMismatch(&'static str),
    NonFiniteInput(&'static str),
    NonCanonicalZero {
        field: &'static str,
        index: usize,
    },
    GraphRebuild(String),
    GraphHashMismatch {
        declared: u64,
        rebuilt: u64,
    },
    GraphMismatch,
    MissingRelationshipNamespace(RelationshipRunNamespaceV0),
    DuplicateRelationshipNamespace(RelationshipRunNamespaceV0),
    ForeignRelationshipConfig,
    SurfaceHierarchy(String),
    SurfaceHierarchyHashMismatch {
        stored: u64,
        computed: u64,
    },
    Drainage(String),
    DrainageHashMismatch {
        stored: u64,
        computed: u64,
    },
    Relationship {
        namespace: RelationshipRunNamespaceV0,
        error: String,
    },
    RelationshipMismatch(RelationshipRunNamespaceV0),
    PacketHashMismatch {
        stored: u64,
        computed: u64,
    },
    CoreHashMismatch {
        stored: u64,
        computed: u64,
    },
    ReferenceHashMismatch {
        stored: u64,
        computed: u64,
    },
    SensitivitySuiteHashMismatch {
        stored: u64,
        computed: u64,
    },
    SidecarCoreHashMismatch {
        stored: u64,
        expected: u64,
    },
    InvalidRelationshipSidecar(&'static str),
    Serialization(String),
}

impl fmt::Display for PacketAssemblyErrorV0 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for PacketAssemblyErrorV0 {}

pub fn registered_relationship_configs_v0() -> [LandformRelationshipConfigV0; 11] {
    let reference = LandformRelationshipConfigV0::default();
    let mut values = [reference; 11];
    values[1].station_spacing_km = 10.0;
    values[2].station_spacing_km = 40.0;
    values[3].cross_section_half_length_km = 50.0;
    values[4].cross_section_half_length_km = 150.0;
    values[5].cross_section_sample_step_km = 2.0;
    values[6].cross_section_sample_step_km = 8.0;
    values[7].relative_height_fraction = 0.15;
    values[8].relative_height_fraction = 0.35;
    values[9].maximum_downstream_support_km = 200.0;
    values[10].maximum_downstream_support_km = 600.0;
    values
}

pub fn canonical_array_hash_v0(values: &[f64]) -> Result<u64, PacketAssemblyErrorV0> {
    validate_values(values, "canonical_array", true)?;
    Ok(fnv1a64(&fixed_bytes(&values.to_vec())?))
}

pub fn population_definition_hash_v0(
    population: &CommonEvaluationPopulationV0,
) -> Result<u64, PacketAssemblyErrorV0> {
    let mut population = population.clone();
    canonicalize_population_zeros(&mut population);
    Ok(fnv1a64(&fixed_bytes(&(
        &population.coordinate_frame,
        &population.declared_domain,
        &population.scored_policy,
        &population.runoff_policy,
        &population.semantic_portals,
    ))?))
}

pub fn assemble_landform_object_packet_v0(
    mut input: LandformPacketAssemblyInputV0<'_>,
) -> Result<LandformObjectPacketCoreV0, PacketAssemblyErrorV0> {
    canonicalize_population_zeros(&mut input.population);
    validate_population(
        &input.population,
        input.graph,
        input.scored_cell,
        input.local_runoff_supply,
    )?;
    validate_live_inputs(&input)?;

    let mut payload_by_namespace: Vec<Option<&LandformRelationshipsV0>> = vec![None; 11];
    for payload in input.relationship_payloads {
        let index = namespace_index(payload.run_namespace);
        if payload_by_namespace[index].replace(payload).is_some() {
            return Err(PacketAssemblyErrorV0::DuplicateRelationshipNamespace(
                payload.run_namespace,
            ));
        }
    }
    for (index, payload) in payload_by_namespace.iter().enumerate() {
        if payload.is_none() {
            return Err(PacketAssemblyErrorV0::MissingRelationshipNamespace(
                namespace_from_index(index),
            ));
        }
    }

    let registered = registered_relationship_configs_v0();
    if input.relationship_configs.len() != registered.len() {
        return Err(PacketAssemblyErrorV0::ForeignRelationshipConfig);
    }
    let mut supplied_configs = input
        .relationship_configs
        .iter()
        .map(LandformRelationshipConfigWireV0::from)
        .collect::<Vec<_>>();
    supplied_configs.sort_by(config_wire_order);
    let mut expected_configs = registered
        .iter()
        .map(LandformRelationshipConfigWireV0::from)
        .collect::<Vec<_>>();
    expected_configs.sort_by(config_wire_order);
    if supplied_configs != expected_configs {
        return Err(PacketAssemblyErrorV0::ForeignRelationshipConfig);
    }

    let computed_surface_hash = surface_hierarchy_evidence_hash_v0(
        input.graph,
        input.physical_elevation_km,
        input.scored_cell,
        input.surface_config,
        input.surface_hierarchy,
    )
    .map_err(|error| PacketAssemblyErrorV0::SurfaceHierarchy(error.to_string()))?;
    if computed_surface_hash != input.surface_hierarchy.derived_evidence_hash {
        return Err(PacketAssemblyErrorV0::SurfaceHierarchyHashMismatch {
            stored: input.surface_hierarchy.derived_evidence_hash,
            computed: computed_surface_hash,
        });
    }
    let rebuilt_surface = build_surface_hierarchy_v0(
        input.graph,
        input.physical_elevation_km,
        input.scored_cell,
        input.surface_config,
    )
    .map_err(|error| PacketAssemblyErrorV0::SurfaceHierarchy(error.to_string()))?;
    if rebuilt_surface != *input.surface_hierarchy {
        return Err(PacketAssemblyErrorV0::SurfaceHierarchy(
            "payload differs from deterministic predecessor rebuild".into(),
        ));
    }

    let computed_drainage_hash = drainage_evidence_hash_v0(
        input.graph,
        input.physical_elevation_km,
        input.local_runoff_supply,
        input.drainage_config,
        input.drainage,
    )
    .map_err(|error| PacketAssemblyErrorV0::Drainage(error.to_string()))?;
    if computed_drainage_hash != input.drainage.derived_evidence_hash {
        return Err(PacketAssemblyErrorV0::DrainageHashMismatch {
            stored: input.drainage.derived_evidence_hash,
            computed: computed_drainage_hash,
        });
    }
    let rebuilt_drainage = build_evaluation_drainage_v0(
        input.graph,
        input.physical_elevation_km,
        input.local_runoff_supply,
        input.drainage_config,
    )
    .map_err(|error| PacketAssemblyErrorV0::Drainage(error.to_string()))?;
    if rebuilt_drainage != *input.drainage {
        return Err(PacketAssemblyErrorV0::Drainage(
            "payload differs from deterministic predecessor rebuild".into(),
        ));
    }

    let mut relationship_payloads = Vec::with_capacity(11);
    let mut relationship_hashes = Vec::with_capacity(11);
    for (index, config) in registered.into_iter().enumerate() {
        let namespace = namespace_from_index(index);
        let supplied = payload_by_namespace[index].expect("namespace presence checked");
        if supplied.config != config {
            return Err(PacketAssemblyErrorV0::RelationshipMismatch(namespace));
        }
        let rebuilt = build_landform_relationships_v0(
            input.graph,
            input.physical_elevation_km,
            input.scored_cell,
            input.local_runoff_supply,
            input.surface_config,
            input.drainage_config,
            input.surface_hierarchy,
            input.drainage,
            input.geometry_identity,
            config,
        )
        .map_err(|error| PacketAssemblyErrorV0::Relationship {
            namespace,
            error: error.to_string(),
        })?;
        if rebuilt != *supplied {
            return Err(PacketAssemblyErrorV0::RelationshipMismatch(namespace));
        }
        relationship_hashes.push(RelationshipEvidenceHashV0 {
            run_namespace: namespace,
            evidence_hash: supplied.derived_evidence_hash,
        });
        relationship_payloads.push(LandformRelationshipsWireV0::from(supplied));
    }

    let mut core = LandformObjectPacketCoreV0 {
        schema_version: O0B_PACKET_SCHEMA_VERSION.into(),
        hash_version: O0B_PACKET_HASH_VERSION.into(),
        population: input.population,
        geometry_identity: input.geometry_identity,
        graph: input.graph.clone(),
        physical_elevation_km: input.physical_elevation_km.to_vec(),
        scored_cell: input.scored_cell.to_vec(),
        local_runoff_supply: input.local_runoff_supply.to_vec(),
        surface_config: SurfaceHierarchyConfigWireV0::from(&input.surface_config),
        drainage_config: DrainageConfigWireV0::from(&input.drainage_config),
        relationship_configs: registered
            .iter()
            .map(LandformRelationshipConfigWireV0::from)
            .collect(),
        surface_hierarchy: input.surface_hierarchy.clone(),
        drainage: input.drainage.clone(),
        relationship_payloads,
        surface_hierarchy_input_hash: computed_surface_hash,
        drainage_input_hash: computed_drainage_hash,
        predecessor_evidence_hashes: PredecessorEvidenceHashesV0 {
            surface_hierarchy_hash: computed_surface_hash,
            drainage_hash: computed_drainage_hash,
            relationship_hashes,
        },
        derived_common_packet_hash: 0,
    };
    core.derived_common_packet_hash = packet_preimage_hash(&core)?;
    Ok(core)
}

pub fn landform_object_packet_hash_v0(
    core: &LandformObjectPacketCoreV0,
) -> Result<u64, PacketAssemblyErrorV0> {
    validate_packet_version(core)?;
    packet_preimage_hash(core)
}

pub fn landform_object_packet_bytes_v0(
    core: &LandformObjectPacketCoreV0,
) -> Result<Vec<u8>, PacketAssemblyErrorV0> {
    validate_packet_version(core)?;
    let computed = packet_preimage_hash(core)?;
    if computed != core.derived_common_packet_hash {
        return Err(PacketAssemblyErrorV0::PacketHashMismatch {
            stored: core.derived_common_packet_hash,
            computed,
        });
    }
    fixed_bytes(core)
}

pub fn decode_landform_object_packet_v0(
    bytes: &[u8],
) -> Result<LandformObjectPacketCoreV0, PacketAssemblyErrorV0> {
    let core: LandformObjectPacketCoreV0 = bincode_options()
        .deserialize(bytes)
        .map_err(|error| PacketAssemblyErrorV0::Serialization(error.to_string()))?;
    validate_packet_version(&core)?;
    let computed = packet_preimage_hash(&core)?;
    if computed != core.derived_common_packet_hash {
        return Err(PacketAssemblyErrorV0::PacketHashMismatch {
            stored: core.derived_common_packet_hash,
            computed,
        });
    }
    validate_decoded_core(&core)?;
    Ok(core)
}

pub fn assemble_common_planar_evidence_core_v0(
    mut input: CommonPlanarEvidenceCoreAssemblyInputV0<'_>,
) -> Result<CommonPlanarEvidenceCoreV0, PacketAssemblyErrorV0> {
    canonicalize_population_zeros(&mut input.population);
    let mut core = CommonPlanarEvidenceCoreV0 {
        schema_version: COMMON_PLANAR_EVIDENCE_CORE_SCHEMA_VERSION.into(),
        hash_version: COMMON_PLANAR_ARTIFACT_HASH_VERSION.into(),
        population: input.population,
        geometry_identity: input.geometry_identity,
        graph: input.graph.clone(),
        physical_elevation_km: input.physical_elevation_km.to_vec(),
        scored_cell: input.scored_cell.to_vec(),
        local_runoff_supply: input.local_runoff_supply.to_vec(),
        surface_config: SurfaceHierarchyConfigWireV0::from(&input.surface_config),
        drainage_config: DrainageConfigWireV0::from(&input.drainage_config),
        surface_hierarchy: input.surface_hierarchy.clone(),
        drainage: input.drainage.clone(),
        derived_core_hash: 0,
    };
    validate_common_planar_evidence_core_semantics_v0(&core)?;
    core.derived_core_hash = common_core_preimage_hash(&core)?;
    Ok(core)
}

pub fn common_planar_evidence_core_hash_v0(
    core: &CommonPlanarEvidenceCoreV0,
) -> Result<u64, PacketAssemblyErrorV0> {
    validate_common_core_version(core)?;
    common_core_preimage_hash(core)
}

pub fn common_planar_evidence_core_bytes_v0(
    core: &CommonPlanarEvidenceCoreV0,
) -> Result<Vec<u8>, PacketAssemblyErrorV0> {
    validate_common_planar_evidence_core_v0(core)?;
    fixed_bytes(core)
}

pub fn decode_common_planar_evidence_core_v0(
    bytes: &[u8],
) -> Result<CommonPlanarEvidenceCoreV0, PacketAssemblyErrorV0> {
    let core: CommonPlanarEvidenceCoreV0 = bincode_options()
        .deserialize(bytes)
        .map_err(|error| PacketAssemblyErrorV0::Serialization(error.to_string()))?;
    validate_common_planar_evidence_core_v0(&core)?;
    Ok(core)
}

pub fn validate_common_planar_evidence_core_v0(
    core: &CommonPlanarEvidenceCoreV0,
) -> Result<(), PacketAssemblyErrorV0> {
    validate_common_core_version(core)?;
    let computed = common_core_preimage_hash(core)?;
    if computed != core.derived_core_hash {
        return Err(PacketAssemblyErrorV0::CoreHashMismatch {
            stored: core.derived_core_hash,
            computed,
        });
    }
    validate_common_planar_evidence_core_semantics_v0(core)
}

pub fn assemble_reference_relationship_evidence_v0(
    core: &CommonPlanarEvidenceCoreV0,
    payload: &LandformRelationshipsWireV0,
) -> Result<ReferenceRelationshipEvidenceV0, PacketAssemblyErrorV0> {
    validate_common_planar_evidence_core_v0(core)?;
    let mut artifact = ReferenceRelationshipEvidenceV0 {
        schema_version: REFERENCE_RELATIONSHIP_EVIDENCE_SCHEMA_VERSION.into(),
        hash_version: COMMON_PLANAR_ARTIFACT_HASH_VERSION.into(),
        core_hash: core.derived_core_hash,
        payload: payload.clone(),
        derived_reference_hash: 0,
    };
    validate_reference_relationship_shape_v0(&artifact)?;
    validate_relationship_payload_against_core_v0(core, &artifact.payload, 0)?;
    artifact.derived_reference_hash = reference_preimage_hash(&artifact)?;
    Ok(artifact)
}

pub fn reference_relationship_evidence_hash_v0(
    artifact: &ReferenceRelationshipEvidenceV0,
) -> Result<u64, PacketAssemblyErrorV0> {
    validate_reference_version(artifact)?;
    reference_preimage_hash(artifact)
}

pub fn reference_relationship_evidence_bytes_v0(
    artifact: &ReferenceRelationshipEvidenceV0,
) -> Result<Vec<u8>, PacketAssemblyErrorV0> {
    validate_reference_relationship_evidence_v0(artifact)?;
    fixed_bytes(artifact)
}

pub fn decode_reference_relationship_evidence_v0(
    bytes: &[u8],
) -> Result<ReferenceRelationshipEvidenceV0, PacketAssemblyErrorV0> {
    let artifact: ReferenceRelationshipEvidenceV0 = bincode_options()
        .deserialize(bytes)
        .map_err(|error| PacketAssemblyErrorV0::Serialization(error.to_string()))?;
    validate_reference_relationship_evidence_v0(&artifact)?;
    Ok(artifact)
}

/// Standalone validation binds the wrapper's registered shape and outer hash.
/// Use [`validate_reference_relationship_evidence_against_core_v0`] for full
/// O0a semantic validation.
pub fn validate_reference_relationship_evidence_v0(
    artifact: &ReferenceRelationshipEvidenceV0,
) -> Result<(), PacketAssemblyErrorV0> {
    validate_reference_relationship_shape_v0(artifact)?;
    let computed = reference_preimage_hash(artifact)?;
    if computed != artifact.derived_reference_hash {
        return Err(PacketAssemblyErrorV0::ReferenceHashMismatch {
            stored: artifact.derived_reference_hash,
            computed,
        });
    }
    Ok(())
}

pub fn validate_reference_relationship_evidence_against_core_v0(
    core: &CommonPlanarEvidenceCoreV0,
    artifact: &ReferenceRelationshipEvidenceV0,
) -> Result<(), PacketAssemblyErrorV0> {
    validate_common_planar_evidence_core_v0(core)?;
    validate_reference_relationship_evidence_v0(artifact)?;
    require_sidecar_core_hash(core, artifact.core_hash)?;
    validate_relationship_payload_against_core_v0(core, &artifact.payload, 0)
}

pub fn assemble_relationship_sensitivity_suite_v0(
    core: &CommonPlanarEvidenceCoreV0,
    payloads: &[LandformRelationshipsWireV0],
) -> Result<RelationshipSensitivitySuiteV0, PacketAssemblyErrorV0> {
    validate_common_planar_evidence_core_v0(core)?;
    let mut artifact = RelationshipSensitivitySuiteV0 {
        schema_version: RELATIONSHIP_SENSITIVITY_SUITE_SCHEMA_VERSION.into(),
        hash_version: COMMON_PLANAR_ARTIFACT_HASH_VERSION.into(),
        core_hash: core.derived_core_hash,
        payloads: payloads.to_vec(),
        derived_suite_hash: 0,
    };
    validate_sensitivity_suite_shape_v0(&artifact)?;
    for (offset, payload) in artifact.payloads.iter().enumerate() {
        validate_relationship_payload_against_core_v0(core, payload, offset + 1)?;
    }
    artifact.derived_suite_hash = sensitivity_suite_preimage_hash(&artifact)?;
    Ok(artifact)
}

pub fn relationship_sensitivity_suite_hash_v0(
    artifact: &RelationshipSensitivitySuiteV0,
) -> Result<u64, PacketAssemblyErrorV0> {
    validate_sensitivity_suite_version(artifact)?;
    sensitivity_suite_preimage_hash(artifact)
}

pub fn relationship_sensitivity_suite_bytes_v0(
    artifact: &RelationshipSensitivitySuiteV0,
) -> Result<Vec<u8>, PacketAssemblyErrorV0> {
    validate_relationship_sensitivity_suite_v0(artifact)?;
    fixed_bytes(artifact)
}

pub fn decode_relationship_sensitivity_suite_v0(
    bytes: &[u8],
) -> Result<RelationshipSensitivitySuiteV0, PacketAssemblyErrorV0> {
    let artifact: RelationshipSensitivitySuiteV0 = bincode_options()
        .deserialize(bytes)
        .map_err(|error| PacketAssemblyErrorV0::Serialization(error.to_string()))?;
    validate_relationship_sensitivity_suite_v0(&artifact)?;
    Ok(artifact)
}

pub fn validate_relationship_sensitivity_suite_v0(
    artifact: &RelationshipSensitivitySuiteV0,
) -> Result<(), PacketAssemblyErrorV0> {
    validate_sensitivity_suite_shape_v0(artifact)?;
    let computed = sensitivity_suite_preimage_hash(artifact)?;
    if computed != artifact.derived_suite_hash {
        return Err(PacketAssemblyErrorV0::SensitivitySuiteHashMismatch {
            stored: artifact.derived_suite_hash,
            computed,
        });
    }
    Ok(())
}

pub fn validate_relationship_sensitivity_suite_against_core_v0(
    core: &CommonPlanarEvidenceCoreV0,
    artifact: &RelationshipSensitivitySuiteV0,
) -> Result<(), PacketAssemblyErrorV0> {
    validate_common_planar_evidence_core_v0(core)?;
    validate_relationship_sensitivity_suite_v0(artifact)?;
    require_sidecar_core_hash(core, artifact.core_hash)?;
    for (offset, payload) in artifact.payloads.iter().enumerate() {
        validate_relationship_payload_against_core_v0(core, payload, offset + 1)?;
    }
    Ok(())
}

pub fn split_landform_object_packet_v0(
    packet: &LandformObjectPacketCoreV0,
) -> Result<
    (
        CommonPlanarEvidenceCoreV0,
        ReferenceRelationshipEvidenceV0,
        RelationshipSensitivitySuiteV0,
    ),
    PacketAssemblyErrorV0,
> {
    validate_decoded_core(packet)?;
    let core = assemble_common_planar_evidence_core_v0(CommonPlanarEvidenceCoreAssemblyInputV0 {
        graph: &packet.graph,
        physical_elevation_km: &packet.physical_elevation_km,
        scored_cell: &packet.scored_cell,
        local_runoff_supply: &packet.local_runoff_supply,
        surface_config: packet.surface_config.to_live()?,
        drainage_config: packet.drainage_config.to_live()?,
        surface_hierarchy: &packet.surface_hierarchy,
        drainage: &packet.drainage,
        geometry_identity: packet.geometry_identity,
        population: packet.population.clone(),
    })?;
    let reference = assemble_reference_relationship_evidence_v0(
        &core,
        packet.relationship_payloads.first().ok_or(
            PacketAssemblyErrorV0::InvalidRelationshipSidecar("missing reference payload"),
        )?,
    )?;
    let suite = assemble_relationship_sensitivity_suite_v0(
        &core,
        packet.relationship_payloads.get(1..).ok_or(
            PacketAssemblyErrorV0::InvalidRelationshipSidecar("missing sensitivity payloads"),
        )?,
    )?;
    Ok((core, reference, suite))
}

pub fn materialize_landform_object_packet_v0(
    core: &CommonPlanarEvidenceCoreV0,
    reference: &ReferenceRelationshipEvidenceV0,
    suite: &RelationshipSensitivitySuiteV0,
) -> Result<LandformObjectPacketCoreV0, PacketAssemblyErrorV0> {
    validate_reference_relationship_evidence_against_core_v0(core, reference)?;
    validate_relationship_sensitivity_suite_against_core_v0(core, suite)?;

    let relationship_configs = registered_relationship_configs_v0()
        .iter()
        .map(LandformRelationshipConfigWireV0::from)
        .collect::<Vec<_>>();
    let mut relationship_payloads = Vec::with_capacity(11);
    relationship_payloads.push(reference.payload.clone());
    relationship_payloads.extend(suite.payloads.iter().cloned());
    let relationship_hashes = relationship_payloads
        .iter()
        .map(|payload| RelationshipEvidenceHashV0 {
            run_namespace: payload.run_namespace,
            evidence_hash: payload.derived_evidence_hash,
        })
        .collect();
    let surface_hash = core.surface_hierarchy.derived_evidence_hash;
    let drainage_hash = core.drainage.derived_evidence_hash;
    let mut packet = LandformObjectPacketCoreV0 {
        schema_version: O0B_PACKET_SCHEMA_VERSION.into(),
        hash_version: O0B_PACKET_HASH_VERSION.into(),
        population: core.population.clone(),
        geometry_identity: core.geometry_identity,
        graph: core.graph.clone(),
        physical_elevation_km: core.physical_elevation_km.clone(),
        scored_cell: core.scored_cell.clone(),
        local_runoff_supply: core.local_runoff_supply.clone(),
        surface_config: core.surface_config.clone(),
        drainage_config: core.drainage_config.clone(),
        relationship_configs,
        surface_hierarchy: core.surface_hierarchy.clone(),
        drainage: core.drainage.clone(),
        relationship_payloads,
        surface_hierarchy_input_hash: surface_hash,
        drainage_input_hash: drainage_hash,
        predecessor_evidence_hashes: PredecessorEvidenceHashesV0 {
            surface_hierarchy_hash: surface_hash,
            drainage_hash,
            relationship_hashes,
        },
        derived_common_packet_hash: 0,
    };
    packet.derived_common_packet_hash = packet_preimage_hash(&packet)?;
    validate_decoded_core(&packet)?;
    Ok(packet)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommonPlanarEvidenceCoreFieldBytesV0 {
    pub versions: usize,
    pub population: usize,
    pub geometry_identity: usize,
    pub graph: usize,
    pub physical_elevation_km: usize,
    pub scored_cell: usize,
    pub local_runoff_supply: usize,
    pub surface_config: usize,
    pub drainage_config: usize,
    pub surface_hierarchy: usize,
    pub drainage: usize,
    pub derived_core_hash: usize,
}

/// Fixed-encoding byte counts for the fields in the registered core preimage.
pub fn common_planar_evidence_core_field_bytes_v0(
    core: &CommonPlanarEvidenceCoreV0,
) -> Result<CommonPlanarEvidenceCoreFieldBytesV0, PacketAssemblyErrorV0> {
    validate_common_core_version(core)?;
    Ok(CommonPlanarEvidenceCoreFieldBytesV0 {
        versions: fixed_bytes(&(&core.schema_version, &core.hash_version))?.len(),
        population: fixed_bytes(&core.population)?.len(),
        geometry_identity: fixed_bytes(&core.geometry_identity)?.len(),
        graph: fixed_bytes(&core.graph)?.len(),
        physical_elevation_km: fixed_bytes(&core.physical_elevation_km)?.len(),
        scored_cell: fixed_bytes(&core.scored_cell)?.len(),
        local_runoff_supply: fixed_bytes(&core.local_runoff_supply)?.len(),
        surface_config: fixed_bytes(&core.surface_config)?.len(),
        drainage_config: fixed_bytes(&core.drainage_config)?.len(),
        surface_hierarchy: fixed_bytes(&core.surface_hierarchy)?.len(),
        drainage: fixed_bytes(&core.drainage)?.len(),
        derived_core_hash: fixed_bytes(&core.derived_core_hash)?.len(),
    })
}

fn validate_common_planar_evidence_core_semantics_v0(
    core: &CommonPlanarEvidenceCoreV0,
) -> Result<(), PacketAssemblyErrorV0> {
    validate_common_core_version(core)?;
    validate_population(
        &core.population,
        &core.graph,
        &core.scored_cell,
        &core.local_runoff_supply,
    )?;
    validate_values(&core.physical_elevation_km, "physical_elevation_km", false)?;
    validate_values(&core.local_runoff_supply, "local_runoff_supply", true)?;
    let n = core.graph.cell_count();
    for (field, len) in [
        ("physical_elevation_km", core.physical_elevation_km.len()),
        ("scored_cell", core.scored_cell.len()),
        ("local_runoff_supply", core.local_runoff_supply.len()),
    ] {
        if len != n {
            return Err(PacketAssemblyErrorV0::LengthMismatch(field));
        }
    }

    let surface_config = core.surface_config.to_live()?;
    let drainage_config = core.drainage_config.to_live()?;
    validate_regular_planar_graph_v0(
        &core.population,
        core.geometry_identity,
        &core.graph,
        &surface_config,
    )?;

    if core.surface_hierarchy.schema_version != G0S0_SCHEMA_VERSION
        || core.surface_hierarchy.hash_version != G0S0_HASH_VERSION
    {
        return Err(PacketAssemblyErrorV0::WrongSchemaOrHashVersion("G0/S0"));
    }
    let computed_surface_hash = surface_hierarchy_evidence_hash_v0(
        &core.graph,
        &core.physical_elevation_km,
        &core.scored_cell,
        surface_config,
        &core.surface_hierarchy,
    )
    .map_err(|error| PacketAssemblyErrorV0::SurfaceHierarchy(error.to_string()))?;
    if computed_surface_hash != core.surface_hierarchy.derived_evidence_hash {
        return Err(PacketAssemblyErrorV0::SurfaceHierarchyHashMismatch {
            stored: core.surface_hierarchy.derived_evidence_hash,
            computed: computed_surface_hash,
        });
    }
    let rebuilt_surface = build_surface_hierarchy_v0(
        &core.graph,
        &core.physical_elevation_km,
        &core.scored_cell,
        surface_config,
    )
    .map_err(|error| PacketAssemblyErrorV0::SurfaceHierarchy(error.to_string()))?;
    if rebuilt_surface != core.surface_hierarchy {
        return Err(PacketAssemblyErrorV0::SurfaceHierarchy(
            "payload differs from deterministic predecessor rebuild".into(),
        ));
    }

    if core.drainage.schema_version != D0_SCHEMA_VERSION
        || core.drainage.hash_version != D0_HASH_VERSION
    {
        return Err(PacketAssemblyErrorV0::WrongSchemaOrHashVersion("D0"));
    }
    let computed_drainage_hash = drainage_evidence_hash_v0(
        &core.graph,
        &core.physical_elevation_km,
        &core.local_runoff_supply,
        drainage_config,
        &core.drainage,
    )
    .map_err(|error| PacketAssemblyErrorV0::Drainage(error.to_string()))?;
    if computed_drainage_hash != core.drainage.derived_evidence_hash {
        return Err(PacketAssemblyErrorV0::DrainageHashMismatch {
            stored: core.drainage.derived_evidence_hash,
            computed: computed_drainage_hash,
        });
    }
    let rebuilt_drainage = build_evaluation_drainage_v0(
        &core.graph,
        &core.physical_elevation_km,
        &core.local_runoff_supply,
        drainage_config,
    )
    .map_err(|error| PacketAssemblyErrorV0::Drainage(error.to_string()))?;
    if rebuilt_drainage != core.drainage {
        return Err(PacketAssemblyErrorV0::Drainage(
            "payload differs from deterministic predecessor rebuild".into(),
        ));
    }
    Ok(())
}

fn validate_regular_planar_graph_v0(
    population: &CommonEvaluationPopulationV0,
    geometry_identity: PacketGeometryIdentityV0,
    graph: &EvaluationSurfaceGraphV0,
    surface_config: &SurfaceHierarchyConfigV0,
) -> Result<(), PacketAssemblyErrorV0> {
    if graph.domain != EvaluationDomainV0::Planar {
        return Err(PacketAssemblyErrorV0::UnsupportedGeometry);
    }
    let (spacing, declared_graph_hash) = match geometry_identity {
        PacketGeometryIdentityV0::LandscapeRegularPlanar {
            nominal_spacing_km,
            canonical_graph_hash,
        } => (nominal_spacing_km, canonical_graph_hash),
        PacketGeometryIdentityV0::ProjectedR1VoronoiCap { .. } => {
            return Err(PacketAssemblyErrorV0::UnsupportedGeometry);
        }
    };
    if !spacing.is_finite() || spacing <= 0.0 {
        return Err(PacketAssemblyErrorV0::UnsupportedGeometry);
    }
    let DeclaredDomainV0::RequestedRegularPatchV0 {
        width_km,
        height_km,
    } = population.declared_domain;
    let portals = population
        .semantic_portals
        .iter()
        .map(|portal| OutletPortal {
            id: OutletPortalId(portal.id),
            side: match portal.side {
                DeclaredPortalSideV0::North => BoundarySide::North,
                DeclaredPortalSideV0::East => BoundarySide::East,
                DeclaredPortalSideV0::South => BoundarySide::South,
                DeclaredPortalSideV0::West => BoundarySide::West,
            },
            span_start_km: portal.span_start_km,
            span_end_km: portal.span_end_km,
            base_level_km: portal.base_level_km as f32,
        })
        .collect::<Vec<_>>();
    let mesh =
        LandscapeMesh::uniform_planar_hex_with_portals(width_km, height_km, spacing, &portals)
            .map_err(|error| PacketAssemblyErrorV0::GraphRebuild(error.to_string()))?;
    let controls = build_regular_hex_control_volumes_v0(&mesh, surface_config)
        .map_err(|error| PacketAssemblyErrorV0::GraphRebuild(error.to_string()))?;
    let rebuilt = adapt_landscape_graph_v0(&mesh, &controls, surface_config)
        .map_err(|error| PacketAssemblyErrorV0::GraphRebuild(error.to_string()))?;
    let rebuilt_hash = relationship_graph_hash_v0(&rebuilt)
        .map_err(|error| PacketAssemblyErrorV0::GraphRebuild(error.to_string()))?;
    if rebuilt_hash != declared_graph_hash {
        return Err(PacketAssemblyErrorV0::GraphHashMismatch {
            declared: declared_graph_hash,
            rebuilt: rebuilt_hash,
        });
    }
    if rebuilt != *graph {
        return Err(PacketAssemblyErrorV0::GraphMismatch);
    }
    Ok(())
}

fn validate_relationship_payload_against_core_v0(
    core: &CommonPlanarEvidenceCoreV0,
    payload: &LandformRelationshipsWireV0,
    namespace_index_expected: usize,
) -> Result<(), PacketAssemblyErrorV0> {
    let expected_namespace = namespace_from_index(namespace_index_expected);
    let registered = registered_relationship_configs_v0();
    let expected_config = registered[namespace_index_expected];
    if payload.run_namespace != expected_namespace || payload.config.to_live()? != expected_config {
        return Err(PacketAssemblyErrorV0::RelationshipMismatch(
            expected_namespace,
        ));
    }
    if payload.geometry_identity != core.geometry_identity
        || payload.surface_hierarchy_input_hash != core.surface_hierarchy.derived_evidence_hash
        || payload.drainage_input_hash != core.drainage.derived_evidence_hash
    {
        return Err(PacketAssemblyErrorV0::InvalidRelationshipSidecar(
            "geometry or predecessor mismatch",
        ));
    }
    let rebuilt = build_landform_relationships_v0(
        &core.graph,
        &core.physical_elevation_km,
        &core.scored_cell,
        &core.local_runoff_supply,
        core.surface_config.to_live()?,
        core.drainage_config.to_live()?,
        &core.surface_hierarchy,
        &core.drainage,
        core.geometry_identity,
        expected_config,
    )
    .map_err(|error| PacketAssemblyErrorV0::Relationship {
        namespace: expected_namespace,
        error: error.to_string(),
    })?;
    if rebuilt != payload.to_live()? {
        return Err(PacketAssemblyErrorV0::RelationshipMismatch(
            expected_namespace,
        ));
    }
    Ok(())
}

fn validate_reference_relationship_shape_v0(
    artifact: &ReferenceRelationshipEvidenceV0,
) -> Result<(), PacketAssemblyErrorV0> {
    validate_reference_version(artifact)?;
    if artifact.payload.schema_version != O0A_SCHEMA_VERSION
        || artifact.payload.hash_version != O0A_HASH_VERSION
        || artifact.payload.run_namespace != RelationshipRunNamespaceV0::Reference
        || artifact.payload.config.to_live()? != registered_relationship_configs_v0()[0]
    {
        return Err(PacketAssemblyErrorV0::InvalidRelationshipSidecar(
            "reference namespace or configuration",
        ));
    }
    Ok(())
}

fn validate_sensitivity_suite_shape_v0(
    artifact: &RelationshipSensitivitySuiteV0,
) -> Result<(), PacketAssemblyErrorV0> {
    validate_sensitivity_suite_version(artifact)?;
    if artifact.payloads.len() != 10 {
        return Err(PacketAssemblyErrorV0::InvalidRelationshipSidecar(
            "sensitivity suite length",
        ));
    }
    let registered = registered_relationship_configs_v0();
    for (offset, payload) in artifact.payloads.iter().enumerate() {
        let index = offset + 1;
        if payload.schema_version != O0A_SCHEMA_VERSION
            || payload.hash_version != O0A_HASH_VERSION
            || payload.run_namespace != namespace_from_index(index)
            || payload.config.to_live()? != registered[index]
        {
            return Err(PacketAssemblyErrorV0::InvalidRelationshipSidecar(
                "sensitivity namespace order or configuration",
            ));
        }
    }
    Ok(())
}

fn require_sidecar_core_hash(
    core: &CommonPlanarEvidenceCoreV0,
    stored: u64,
) -> Result<(), PacketAssemblyErrorV0> {
    if stored != core.derived_core_hash {
        return Err(PacketAssemblyErrorV0::SidecarCoreHashMismatch {
            stored,
            expected: core.derived_core_hash,
        });
    }
    Ok(())
}

fn validate_common_core_version(
    core: &CommonPlanarEvidenceCoreV0,
) -> Result<(), PacketAssemblyErrorV0> {
    if core.schema_version != COMMON_PLANAR_EVIDENCE_CORE_SCHEMA_VERSION
        || core.hash_version != COMMON_PLANAR_ARTIFACT_HASH_VERSION
    {
        return Err(PacketAssemblyErrorV0::WrongSchemaOrHashVersion(
            "common planar evidence core",
        ));
    }
    Ok(())
}

fn validate_reference_version(
    artifact: &ReferenceRelationshipEvidenceV0,
) -> Result<(), PacketAssemblyErrorV0> {
    if artifact.schema_version != REFERENCE_RELATIONSHIP_EVIDENCE_SCHEMA_VERSION
        || artifact.hash_version != COMMON_PLANAR_ARTIFACT_HASH_VERSION
    {
        return Err(PacketAssemblyErrorV0::WrongSchemaOrHashVersion(
            "reference relationship evidence",
        ));
    }
    Ok(())
}

fn validate_sensitivity_suite_version(
    artifact: &RelationshipSensitivitySuiteV0,
) -> Result<(), PacketAssemblyErrorV0> {
    if artifact.schema_version != RELATIONSHIP_SENSITIVITY_SUITE_SCHEMA_VERSION
        || artifact.hash_version != COMMON_PLANAR_ARTIFACT_HASH_VERSION
    {
        return Err(PacketAssemblyErrorV0::WrongSchemaOrHashVersion(
            "relationship sensitivity suite",
        ));
    }
    Ok(())
}

fn validate_decoded_core(core: &LandformObjectPacketCoreV0) -> Result<(), PacketAssemblyErrorV0> {
    let surface_config = core.surface_config.to_live()?;
    let drainage_config = core.drainage_config.to_live()?;
    let relationship_configs = core
        .relationship_configs
        .iter()
        .map(LandformRelationshipConfigWireV0::to_live)
        .collect::<Result<Vec<_>, _>>()?;
    let relationship_payloads = core
        .relationship_payloads
        .iter()
        .map(LandformRelationshipsWireV0::to_live)
        .collect::<Result<Vec<_>, _>>()?;
    let rebuilt = assemble_landform_object_packet_v0(LandformPacketAssemblyInputV0 {
        graph: &core.graph,
        physical_elevation_km: &core.physical_elevation_km,
        scored_cell: &core.scored_cell,
        local_runoff_supply: &core.local_runoff_supply,
        surface_config,
        drainage_config,
        relationship_configs: &relationship_configs,
        surface_hierarchy: &core.surface_hierarchy,
        drainage: &core.drainage,
        relationship_payloads: &relationship_payloads,
        geometry_identity: core.geometry_identity,
        population: core.population.clone(),
    })?;
    if rebuilt != *core {
        return Err(PacketAssemblyErrorV0::PacketHashMismatch {
            stored: core.derived_common_packet_hash,
            computed: rebuilt.derived_common_packet_hash,
        });
    }
    Ok(())
}

fn validate_packet_version(core: &LandformObjectPacketCoreV0) -> Result<(), PacketAssemblyErrorV0> {
    if core.schema_version != O0B_PACKET_SCHEMA_VERSION
        || core.hash_version != O0B_PACKET_HASH_VERSION
    {
        return Err(PacketAssemblyErrorV0::WrongSchemaOrHashVersion("O0b"));
    }
    Ok(())
}

fn validate_live_inputs(
    input: &LandformPacketAssemblyInputV0<'_>,
) -> Result<(), PacketAssemblyErrorV0> {
    if input.graph.domain != EvaluationDomainV0::Planar {
        return Err(PacketAssemblyErrorV0::UnsupportedGeometry);
    }
    let n = input.graph.cell_count();
    for (name, length) in [
        ("physical_elevation_km", input.physical_elevation_km.len()),
        ("scored_cell", input.scored_cell.len()),
        ("local_runoff_supply", input.local_runoff_supply.len()),
    ] {
        if length != n {
            return Err(PacketAssemblyErrorV0::LengthMismatch(name));
        }
    }
    validate_values(input.physical_elevation_km, "physical_elevation_km", false)?;
    validate_values(input.local_runoff_supply, "local_runoff_supply", true)?;

    if input.surface_hierarchy.schema_version != G0S0_SCHEMA_VERSION
        || input.surface_hierarchy.hash_version != G0S0_HASH_VERSION
    {
        return Err(PacketAssemblyErrorV0::WrongSchemaOrHashVersion("G0/S0"));
    }
    if input.drainage.schema_version != D0_SCHEMA_VERSION
        || input.drainage.hash_version != D0_HASH_VERSION
    {
        return Err(PacketAssemblyErrorV0::WrongSchemaOrHashVersion("D0"));
    }
    if input.relationship_payloads.iter().any(|payload| {
        payload.schema_version != O0A_SCHEMA_VERSION || payload.hash_version != O0A_HASH_VERSION
    }) {
        return Err(PacketAssemblyErrorV0::WrongSchemaOrHashVersion("O0a"));
    }

    let (spacing, declared_graph_hash) = match input.geometry_identity {
        PacketGeometryIdentityV0::LandscapeRegularPlanar {
            nominal_spacing_km,
            canonical_graph_hash,
        } => (nominal_spacing_km, canonical_graph_hash),
        PacketGeometryIdentityV0::ProjectedR1VoronoiCap { .. } => {
            return Err(PacketAssemblyErrorV0::UnsupportedGeometry);
        }
    };
    if !spacing.is_finite() || spacing <= 0.0 {
        return Err(PacketAssemblyErrorV0::UnsupportedGeometry);
    }
    let DeclaredDomainV0::RequestedRegularPatchV0 {
        width_km,
        height_km,
    } = input.population.declared_domain;
    let portals = input
        .population
        .semantic_portals
        .iter()
        .map(|portal| OutletPortal {
            id: OutletPortalId(portal.id),
            side: match portal.side {
                DeclaredPortalSideV0::North => BoundarySide::North,
                DeclaredPortalSideV0::East => BoundarySide::East,
                DeclaredPortalSideV0::South => BoundarySide::South,
                DeclaredPortalSideV0::West => BoundarySide::West,
            },
            span_start_km: portal.span_start_km,
            span_end_km: portal.span_end_km,
            base_level_km: portal.base_level_km as f32,
        })
        .collect::<Vec<_>>();
    let mesh =
        LandscapeMesh::uniform_planar_hex_with_portals(width_km, height_km, spacing, &portals)
            .map_err(|error| PacketAssemblyErrorV0::GraphRebuild(error.to_string()))?;
    let controls = build_regular_hex_control_volumes_v0(&mesh, &input.surface_config)
        .map_err(|error| PacketAssemblyErrorV0::GraphRebuild(error.to_string()))?;
    let rebuilt = adapt_landscape_graph_v0(&mesh, &controls, &input.surface_config)
        .map_err(|error| PacketAssemblyErrorV0::GraphRebuild(error.to_string()))?;
    let rebuilt_hash = relationship_graph_hash_v0(&rebuilt)
        .map_err(|error| PacketAssemblyErrorV0::GraphRebuild(error.to_string()))?;
    if rebuilt_hash != declared_graph_hash {
        return Err(PacketAssemblyErrorV0::GraphHashMismatch {
            declared: declared_graph_hash,
            rebuilt: rebuilt_hash,
        });
    }
    if rebuilt != *input.graph {
        return Err(PacketAssemblyErrorV0::GraphMismatch);
    }
    Ok(())
}

fn validate_population(
    population: &CommonEvaluationPopulationV0,
    graph: &EvaluationSurfaceGraphV0,
    scored_cell: &[bool],
    runoff: &[f64],
) -> Result<(), PacketAssemblyErrorV0> {
    let DeclaredDomainV0::RequestedRegularPatchV0 {
        width_km,
        height_km,
    } = population.declared_domain;
    for value in [width_km, height_km] {
        if !value.is_finite() || value <= 0.0 || (value == 0.0 && value.to_bits() != 0) {
            return Err(PacketAssemblyErrorV0::InvalidPopulation("declared_domain"));
        }
    }
    let mut sorted = population.semantic_portals.clone();
    sorted.sort_by(portal_order);
    if sorted != population.semantic_portals {
        return Err(PacketAssemblyErrorV0::InvalidPopulation(
            "semantic_portal_order",
        ));
    }
    for (index, portal) in population.semantic_portals.iter().enumerate() {
        if index > 0 && population.semantic_portals[index - 1].id == portal.id {
            return Err(PacketAssemblyErrorV0::InvalidPopulation(
                "duplicate_portal_id",
            ));
        }
        for value in [
            portal.span_start_km,
            portal.span_end_km,
            portal.base_level_km,
        ] {
            if !value.is_finite() || (value == 0.0 && value.to_bits() != 0) {
                return Err(PacketAssemblyErrorV0::InvalidPopulation("semantic_portal"));
            }
        }
        if portal.span_start_km >= portal.span_end_km {
            return Err(PacketAssemblyErrorV0::InvalidPopulation(
                "semantic_portal_span",
            ));
        }
        if f64::from(portal.base_level_km as f32).to_bits() != portal.base_level_km.to_bits() {
            return Err(PacketAssemblyErrorV0::InvalidPopulation(
                "semantic_portal_base_level_not_mesh_representable",
            ));
        }
    }
    let computed = population_definition_hash_v0(population)?;
    if computed != population.population_definition_hash {
        return Err(PacketAssemblyErrorV0::PopulationHashMismatch {
            stored: population.population_definition_hash,
            computed,
        });
    }
    if runoff.len() != graph.cell_count() {
        return Err(PacketAssemblyErrorV0::LengthMismatch("local_runoff_supply"));
    }
    if scored_cell.len() != graph.cell_count() {
        return Err(PacketAssemblyErrorV0::LengthMismatch("scored_cell"));
    }
    if scored_cell.iter().any(|&scored| !scored) {
        return Err(PacketAssemblyErrorV0::InvalidPopulation(
            "whole_graph_support",
        ));
    }
    match population.runoff_policy {
        RunoffPolicyV0::ExactSameMeshArrayV0 {
            canonical_array_hash,
        } => {
            if canonical_array_hash_v0(runoff)? != canonical_array_hash {
                return Err(PacketAssemblyErrorV0::InvalidPopulation(
                    "runoff_array_hash",
                ));
            }
        }
        RunoffPolicyV0::UniformPerAreaV0 { rate } => {
            if !rate.is_finite() || rate < 0.0 || (rate == 0.0 && rate.to_bits() != 0) {
                return Err(PacketAssemblyErrorV0::InvalidPopulation(
                    "uniform_runoff_rate",
                ));
            }
            for (actual, area) in runoff.iter().zip(&graph.cell_area_km2) {
                if actual.to_bits() != (rate * area).to_bits() {
                    return Err(PacketAssemblyErrorV0::InvalidPopulation(
                        "uniform_runoff_value",
                    ));
                }
            }
        }
        RunoffPolicyV0::AsymmetricYAffinePerAreaV0 {
            base_rate,
            x_gradient_per_km,
        } => {
            if base_rate.to_bits() != 0.3f64.to_bits()
                || x_gradient_per_km.to_bits() != 0.002f64.to_bits()
            {
                return Err(PacketAssemblyErrorV0::InvalidPopulation(
                    "asymmetric_y_policy",
                ));
            }
            for ((actual, center), area) in runoff
                .iter()
                .zip(&graph.cell_center_km)
                .zip(&graph.cell_area_km2)
            {
                let expected = 0.3 * (1.0 + 0.002 * center.x) * area;
                if actual.to_bits() != expected.to_bits() {
                    return Err(PacketAssemblyErrorV0::InvalidPopulation(
                        "asymmetric_y_runoff",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn canonicalize_population_zeros(population: &mut CommonEvaluationPopulationV0) {
    let DeclaredDomainV0::RequestedRegularPatchV0 {
        width_km,
        height_km,
    } = &mut population.declared_domain;
    *width_km = canonical_zero(*width_km);
    *height_km = canonical_zero(*height_km);
    match &mut population.runoff_policy {
        RunoffPolicyV0::ExactSameMeshArrayV0 { .. } => {}
        RunoffPolicyV0::UniformPerAreaV0 { rate } => *rate = canonical_zero(*rate),
        RunoffPolicyV0::AsymmetricYAffinePerAreaV0 {
            base_rate,
            x_gradient_per_km,
        } => {
            *base_rate = canonical_zero(*base_rate);
            *x_gradient_per_km = canonical_zero(*x_gradient_per_km);
        }
    }
    for portal in &mut population.semantic_portals {
        portal.span_start_km = canonical_zero(portal.span_start_km);
        portal.span_end_km = canonical_zero(portal.span_end_km);
        portal.base_level_km = canonical_zero(portal.base_level_km);
    }
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn validate_values(
    values: &[f64],
    field: &'static str,
    non_negative: bool,
) -> Result<(), PacketAssemblyErrorV0> {
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() || (non_negative && value < 0.0) {
            return Err(PacketAssemblyErrorV0::NonFiniteInput(field));
        }
        if value == 0.0 && value.to_bits() != 0 {
            return Err(PacketAssemblyErrorV0::NonCanonicalZero { field, index });
        }
    }
    Ok(())
}

#[derive(Serialize)]
struct CommonCoreHashPreimageV0<'a> {
    schema_version: &'a String,
    hash_version: &'a String,
    population: &'a CommonEvaluationPopulationV0,
    geometry_identity: &'a PacketGeometryIdentityV0,
    graph: &'a EvaluationSurfaceGraphV0,
    physical_elevation_km: &'a Vec<f64>,
    scored_cell: &'a Vec<bool>,
    local_runoff_supply: &'a Vec<f64>,
    surface_config: &'a SurfaceHierarchyConfigWireV0,
    drainage_config: &'a DrainageConfigWireV0,
    surface_hierarchy: &'a SurfaceHierarchyV0,
    drainage: &'a EvaluationDrainageV0,
}

fn common_core_preimage_hash(
    core: &CommonPlanarEvidenceCoreV0,
) -> Result<u64, PacketAssemblyErrorV0> {
    Ok(fnv1a64(&fixed_bytes(&CommonCoreHashPreimageV0 {
        schema_version: &core.schema_version,
        hash_version: &core.hash_version,
        population: &core.population,
        geometry_identity: &core.geometry_identity,
        graph: &core.graph,
        physical_elevation_km: &core.physical_elevation_km,
        scored_cell: &core.scored_cell,
        local_runoff_supply: &core.local_runoff_supply,
        surface_config: &core.surface_config,
        drainage_config: &core.drainage_config,
        surface_hierarchy: &core.surface_hierarchy,
        drainage: &core.drainage,
    })?))
}

#[derive(Serialize)]
struct ReferenceHashPreimageV0<'a> {
    schema_version: &'a String,
    hash_version: &'a String,
    core_hash: u64,
    payload: &'a LandformRelationshipsWireV0,
}

fn reference_preimage_hash(
    artifact: &ReferenceRelationshipEvidenceV0,
) -> Result<u64, PacketAssemblyErrorV0> {
    Ok(fnv1a64(&fixed_bytes(&ReferenceHashPreimageV0 {
        schema_version: &artifact.schema_version,
        hash_version: &artifact.hash_version,
        core_hash: artifact.core_hash,
        payload: &artifact.payload,
    })?))
}

#[derive(Serialize)]
struct SensitivitySuiteHashPreimageV0<'a> {
    schema_version: &'a String,
    hash_version: &'a String,
    core_hash: u64,
    payloads: &'a Vec<LandformRelationshipsWireV0>,
}

fn sensitivity_suite_preimage_hash(
    artifact: &RelationshipSensitivitySuiteV0,
) -> Result<u64, PacketAssemblyErrorV0> {
    Ok(fnv1a64(&fixed_bytes(&SensitivitySuiteHashPreimageV0 {
        schema_version: &artifact.schema_version,
        hash_version: &artifact.hash_version,
        core_hash: artifact.core_hash,
        payloads: &artifact.payloads,
    })?))
}

#[derive(Serialize)]
struct PacketHashPreimageV0<'a> {
    schema_version: &'a String,
    hash_version: &'a String,
    population: &'a CommonEvaluationPopulationV0,
    geometry_identity: &'a PacketGeometryIdentityV0,
    graph: &'a EvaluationSurfaceGraphV0,
    physical_elevation_km: &'a Vec<f64>,
    scored_cell: &'a Vec<bool>,
    local_runoff_supply: &'a Vec<f64>,
    surface_config: &'a SurfaceHierarchyConfigWireV0,
    drainage_config: &'a DrainageConfigWireV0,
    relationship_configs: &'a Vec<LandformRelationshipConfigWireV0>,
    surface_hierarchy: &'a SurfaceHierarchyV0,
    drainage: &'a EvaluationDrainageV0,
    relationship_payloads: &'a Vec<LandformRelationshipsWireV0>,
    surface_hierarchy_input_hash: u64,
    drainage_input_hash: u64,
    predecessor_evidence_hashes: &'a PredecessorEvidenceHashesV0,
}

fn packet_preimage_hash(core: &LandformObjectPacketCoreV0) -> Result<u64, PacketAssemblyErrorV0> {
    Ok(fnv1a64(&fixed_bytes(&PacketHashPreimageV0 {
        schema_version: &core.schema_version,
        hash_version: &core.hash_version,
        population: &core.population,
        geometry_identity: &core.geometry_identity,
        graph: &core.graph,
        physical_elevation_km: &core.physical_elevation_km,
        scored_cell: &core.scored_cell,
        local_runoff_supply: &core.local_runoff_supply,
        surface_config: &core.surface_config,
        drainage_config: &core.drainage_config,
        relationship_configs: &core.relationship_configs,
        surface_hierarchy: &core.surface_hierarchy,
        drainage: &core.drainage,
        relationship_payloads: &core.relationship_payloads,
        surface_hierarchy_input_hash: core.surface_hierarchy_input_hash,
        drainage_input_hash: core.drainage_input_hash,
        predecessor_evidence_hashes: &core.predecessor_evidence_hashes,
    })?))
}

fn fixed_bytes(value: &impl Serialize) -> Result<Vec<u8>, PacketAssemblyErrorV0> {
    bincode_options()
        .serialize(value)
        .map_err(|error| PacketAssemblyErrorV0::Serialization(error.to_string()))
}

fn bincode_options() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn namespace_index(namespace: RelationshipRunNamespaceV0) -> usize {
    namespace as usize
}

fn namespace_from_index(index: usize) -> RelationshipRunNamespaceV0 {
    use RelationshipRunNamespaceV0::*;
    [
        Reference,
        StationSpacingLow,
        StationSpacingHigh,
        CrossSectionHalfLengthLow,
        CrossSectionHalfLengthHigh,
        CrossSectionSampleStepLow,
        CrossSectionSampleStepHigh,
        RelativeHeightFractionLow,
        RelativeHeightFractionHigh,
        MaximumDownstreamSupportLow,
        MaximumDownstreamSupportHigh,
    ][index]
}

fn portal_order(a: &DeclaredPortalV0, b: &DeclaredPortalV0) -> std::cmp::Ordering {
    a.id.cmp(&b.id)
        .then(a.side.cmp(&b.side))
        .then_with(|| a.span_start_km.total_cmp(&b.span_start_km))
        .then_with(|| a.span_end_km.total_cmp(&b.span_end_km))
        .then_with(|| a.base_level_km.total_cmp(&b.base_level_km))
}

fn config_wire_order(
    a: &LandformRelationshipConfigWireV0,
    b: &LandformRelationshipConfigWireV0,
) -> std::cmp::Ordering {
    a.station_spacing_km
        .total_cmp(&b.station_spacing_km)
        .then_with(|| {
            a.cross_section_half_length_km
                .total_cmp(&b.cross_section_half_length_km)
        })
        .then_with(|| {
            a.cross_section_sample_step_km
                .total_cmp(&b.cross_section_sample_step_km)
        })
        .then_with(|| {
            a.relative_height_fraction
                .total_cmp(&b.relative_height_fraction)
        })
        .then_with(|| {
            a.maximum_downstream_support_km
                .total_cmp(&b.maximum_downstream_support_km)
        })
        .then_with(|| a.schema_version.cmp(&b.schema_version))
        .then_with(|| a.hash_version.cmp(&b.hash_version))
}

impl From<&SurfaceHierarchyConfigV0> for SurfaceHierarchyConfigWireV0 {
    fn from(value: &SurfaceHierarchyConfigV0) -> Self {
        Self {
            closure_level_km: value.closure_level_km,
            reference_persistence_km: value.reference_persistence_km,
            reference_min_footprint_km2: value.reference_min_footprint_km2,
            persistence_sensitivity_km: value.persistence_sensitivity_km,
            footprint_sensitivity_km2: value.footprint_sensitivity_km2,
            local_relief_radii_km: value.local_relief_radii_km,
            summit_cap_depths_km: value.summit_cap_depths_km,
            gentle_grade_thresholds: value.gentle_grade_thresholds,
            endpoint_match_abs_km: value.endpoint_match_abs_km,
            planar_area_match_relative: value.planar_area_match_relative,
            sphere_area_closure_relative: value.sphere_area_closure_relative,
            linear_rank_relative: value.linear_rank_relative,
            orientation_ambiguity_anisotropy: value.orientation_ambiguity_anisotropy,
            spherical_nonlocal_radius_rad: value.spherical_nonlocal_radius_rad,
            schema_version: value.schema_version.into(),
            hash_version: value.hash_version.into(),
        }
    }
}

impl SurfaceHierarchyConfigWireV0 {
    fn to_live(&self) -> Result<SurfaceHierarchyConfigV0, PacketAssemblyErrorV0> {
        if self.schema_version != G0S0_SCHEMA_VERSION || self.hash_version != G0S0_HASH_VERSION {
            return Err(PacketAssemblyErrorV0::WrongSchemaOrHashVersion(
                "G0/S0 config",
            ));
        }
        Ok(SurfaceHierarchyConfigV0 {
            closure_level_km: self.closure_level_km,
            reference_persistence_km: self.reference_persistence_km,
            reference_min_footprint_km2: self.reference_min_footprint_km2,
            persistence_sensitivity_km: self.persistence_sensitivity_km,
            footprint_sensitivity_km2: self.footprint_sensitivity_km2,
            local_relief_radii_km: self.local_relief_radii_km,
            summit_cap_depths_km: self.summit_cap_depths_km,
            gentle_grade_thresholds: self.gentle_grade_thresholds,
            endpoint_match_abs_km: self.endpoint_match_abs_km,
            planar_area_match_relative: self.planar_area_match_relative,
            sphere_area_closure_relative: self.sphere_area_closure_relative,
            linear_rank_relative: self.linear_rank_relative,
            orientation_ambiguity_anisotropy: self.orientation_ambiguity_anisotropy,
            spherical_nonlocal_radius_rad: self.spherical_nonlocal_radius_rad,
            schema_version: G0S0_SCHEMA_VERSION,
            hash_version: G0S0_HASH_VERSION,
        })
    }
}

impl From<&DrainageConfigV0> for DrainageConfigWireV0 {
    fn from(value: &DrainageConfigV0) -> Self {
        Self {
            support_thresholds_km2: value.support_thresholds_km2,
            balance_absolute_tolerance: value.balance_absolute_tolerance,
            balance_relative_tolerance: value.balance_relative_tolerance,
            schema_version: value.schema_version.into(),
            hash_version: value.hash_version.into(),
        }
    }
}

impl DrainageConfigWireV0 {
    fn to_live(&self) -> Result<DrainageConfigV0, PacketAssemblyErrorV0> {
        if self.schema_version != D0_SCHEMA_VERSION || self.hash_version != D0_HASH_VERSION {
            return Err(PacketAssemblyErrorV0::WrongSchemaOrHashVersion("D0 config"));
        }
        Ok(DrainageConfigV0 {
            support_thresholds_km2: self.support_thresholds_km2,
            balance_absolute_tolerance: self.balance_absolute_tolerance,
            balance_relative_tolerance: self.balance_relative_tolerance,
            schema_version: D0_SCHEMA_VERSION,
            hash_version: D0_HASH_VERSION,
        })
    }
}

impl From<&LandformRelationshipConfigV0> for LandformRelationshipConfigWireV0 {
    fn from(value: &LandformRelationshipConfigV0) -> Self {
        Self {
            station_spacing_km: value.station_spacing_km,
            cross_section_half_length_km: value.cross_section_half_length_km,
            cross_section_sample_step_km: value.cross_section_sample_step_km,
            relative_height_fraction: value.relative_height_fraction,
            maximum_downstream_support_km: value.maximum_downstream_support_km,
            schema_version: value.schema_version.into(),
            hash_version: value.hash_version.into(),
        }
    }
}

impl LandformRelationshipConfigWireV0 {
    fn to_live(&self) -> Result<LandformRelationshipConfigV0, PacketAssemblyErrorV0> {
        if self.schema_version != O0A_SCHEMA_VERSION || self.hash_version != O0A_HASH_VERSION {
            return Err(PacketAssemblyErrorV0::WrongSchemaOrHashVersion(
                "O0a config",
            ));
        }
        Ok(LandformRelationshipConfigV0 {
            station_spacing_km: self.station_spacing_km,
            cross_section_half_length_km: self.cross_section_half_length_km,
            cross_section_sample_step_km: self.cross_section_sample_step_km,
            relative_height_fraction: self.relative_height_fraction,
            maximum_downstream_support_km: self.maximum_downstream_support_km,
            schema_version: O0A_SCHEMA_VERSION,
            hash_version: O0A_HASH_VERSION,
        })
    }
}

impl From<&LandformRelationshipsV0> for LandformRelationshipsWireV0 {
    fn from(value: &LandformRelationshipsV0) -> Self {
        Self {
            schema_version: value.schema_version.clone(),
            hash_version: value.hash_version.clone(),
            geometry_identity: value.geometry_identity,
            config: LandformRelationshipConfigWireV0::from(&value.config),
            run_namespace: value.run_namespace,
            surface_hierarchy_input_hash: value.surface_hierarchy_input_hash,
            drainage_input_hash: value.drainage_input_hash,
            backed_boundary_faces: value.backed_boundary_faces.clone(),
            highland_boundary_relationships: value.highland_boundary_relationships.clone(),
            saddle_boundary_associations: value.saddle_boundary_associations.clone(),
            reach_cross_section_probes: value.reach_cross_section_probes.clone(),
            work_counts: value.work_counts.clone(),
            derived_evidence_hash: value.derived_evidence_hash,
        }
    }
}

impl LandformRelationshipsWireV0 {
    fn to_live(&self) -> Result<LandformRelationshipsV0, PacketAssemblyErrorV0> {
        if self.schema_version != O0A_SCHEMA_VERSION || self.hash_version != O0A_HASH_VERSION {
            return Err(PacketAssemblyErrorV0::WrongSchemaOrHashVersion(
                "O0a payload",
            ));
        }
        Ok(LandformRelationshipsV0 {
            schema_version: self.schema_version.clone(),
            hash_version: self.hash_version.clone(),
            geometry_identity: self.geometry_identity,
            config: self.config.to_live()?,
            run_namespace: self.run_namespace,
            surface_hierarchy_input_hash: self.surface_hierarchy_input_hash,
            drainage_input_hash: self.drainage_input_hash,
            backed_boundary_faces: self.backed_boundary_faces.clone(),
            highland_boundary_relationships: self.highland_boundary_relationships.clone(),
            saddle_boundary_associations: self.saddle_boundary_associations.clone(),
            reach_cross_section_probes: self.reach_cross_section_probes.clone(),
            work_counts: self.work_counts.clone(),
            derived_evidence_hash: self.derived_evidence_hash,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PacketFixture {
        graph: EvaluationSurfaceGraphV0,
        elevation: Vec<f64>,
        scored: Vec<bool>,
        runoff: Vec<f64>,
        surface_config: SurfaceHierarchyConfigV0,
        drainage_config: DrainageConfigV0,
        hierarchy: SurfaceHierarchyV0,
        drainage: EvaluationDrainageV0,
        geometry: PacketGeometryIdentityV0,
        configs: [LandformRelationshipConfigV0; 11],
        relationships: Vec<LandformRelationshipsV0>,
        population: CommonEvaluationPopulationV0,
    }

    fn fixture() -> PacketFixture {
        let width_km = 64.0;
        let height_km = 48.0;
        let spacing_km = 8.0;
        let declared_portal = DeclaredPortalV0 {
            id: 7,
            side: DeclaredPortalSideV0::South,
            span_start_km: -8.0,
            span_end_km: 8.0,
            base_level_km: 0.0,
        };
        let portal = OutletPortal {
            id: OutletPortalId(declared_portal.id),
            side: BoundarySide::South,
            span_start_km: declared_portal.span_start_km,
            span_end_km: declared_portal.span_end_km,
            base_level_km: 0.0,
        };
        let mesh = LandscapeMesh::uniform_planar_hex_with_portals(
            width_km,
            height_km,
            spacing_km,
            &[portal],
        )
        .unwrap();
        let surface_config = SurfaceHierarchyConfigV0::default();
        let drainage_config = DrainageConfigV0::default();
        let controls = build_regular_hex_control_volumes_v0(&mesh, &surface_config).unwrap();
        let graph = adapt_landscape_graph_v0(&mesh, &controls, &surface_config).unwrap();
        let elevation = graph
            .cell_center_km
            .iter()
            .map(|center| 0.4 + 0.01 * (center.y + 0.5 * height_km) + 0.001 * center.x.abs())
            .collect::<Vec<_>>();
        let scored = vec![true; graph.cell_count()];
        let runoff = graph
            .cell_area_km2
            .iter()
            .map(|area| 0.1 * area)
            .collect::<Vec<_>>();
        let hierarchy =
            build_surface_hierarchy_v0(&graph, &elevation, &scored, surface_config).unwrap();
        let drainage =
            build_evaluation_drainage_v0(&graph, &elevation, &runoff, drainage_config).unwrap();
        let geometry = PacketGeometryIdentityV0::LandscapeRegularPlanar {
            nominal_spacing_km: spacing_km,
            canonical_graph_hash: relationship_graph_hash_v0(&graph).unwrap(),
        };
        let configs = registered_relationship_configs_v0();
        let relationships = configs
            .iter()
            .map(|&config| {
                build_landform_relationships_v0(
                    &graph,
                    &elevation,
                    &scored,
                    &runoff,
                    surface_config,
                    drainage_config,
                    &hierarchy,
                    &drainage,
                    geometry,
                    config,
                )
                .unwrap()
            })
            .collect();
        let mut population = CommonEvaluationPopulationV0 {
            coordinate_frame: CoordinateFrameV0::LandscapeTestbedCartesianXyKmV0,
            declared_domain: DeclaredDomainV0::RequestedRegularPatchV0 {
                width_km,
                height_km,
            },
            scored_policy: ScoredPolicyV0::WholeGraphSupportV0,
            runoff_policy: RunoffPolicyV0::UniformPerAreaV0 { rate: 0.1 },
            semantic_portals: vec![declared_portal],
            population_definition_hash: 0,
        };
        population.population_definition_hash = population_definition_hash_v0(&population).unwrap();
        PacketFixture {
            graph,
            elevation,
            scored,
            runoff,
            surface_config,
            drainage_config,
            hierarchy,
            drainage,
            geometry,
            configs,
            relationships,
            population,
        }
    }

    fn assemble(fixture: &PacketFixture) -> LandformObjectPacketCoreV0 {
        assemble_landform_object_packet_v0(LandformPacketAssemblyInputV0 {
            graph: &fixture.graph,
            physical_elevation_km: &fixture.elevation,
            scored_cell: &fixture.scored,
            local_runoff_supply: &fixture.runoff,
            surface_config: fixture.surface_config,
            drainage_config: fixture.drainage_config,
            relationship_configs: &fixture.configs,
            surface_hierarchy: &fixture.hierarchy,
            drainage: &fixture.drainage,
            relationship_payloads: &fixture.relationships,
            geometry_identity: fixture.geometry,
            population: fixture.population.clone(),
        })
        .unwrap()
    }

    #[test]
    fn o0a_owned_wire_preserves_frozen_bytes() {
        let fixture = fixture();
        for live in &fixture.relationships {
            let owned = LandformRelationshipsWireV0::from(live);
            assert_eq!(fixed_bytes(live).unwrap(), fixed_bytes(&owned).unwrap());
            assert_eq!(owned.to_live().unwrap(), *live);
        }
    }

    #[test]
    fn owned_packet_round_trips_with_identical_bytes_and_hash() {
        let fixture = fixture();
        let packet = assemble(&fixture);
        let bytes = landform_object_packet_bytes_v0(&packet).unwrap();
        let decoded = decode_landform_object_packet_v0(&bytes).unwrap();
        assert_eq!(decoded, packet);
        assert_eq!(landform_object_packet_bytes_v0(&decoded).unwrap(), bytes);
        assert_eq!(
            landform_object_packet_hash_v0(&decoded).unwrap(),
            packet.derived_common_packet_hash
        );
    }

    #[test]
    fn whole_graph_policy_rejects_unscored_cells() {
        let fixture = fixture();
        let mut scored = fixture.scored.clone();
        scored[0] = false;
        let error = assemble_landform_object_packet_v0(LandformPacketAssemblyInputV0 {
            graph: &fixture.graph,
            physical_elevation_km: &fixture.elevation,
            scored_cell: &scored,
            local_runoff_supply: &fixture.runoff,
            surface_config: fixture.surface_config,
            drainage_config: fixture.drainage_config,
            relationship_configs: &fixture.configs,
            surface_hierarchy: &fixture.hierarchy,
            drainage: &fixture.drainage,
            relationship_payloads: &fixture.relationships,
            geometry_identity: fixture.geometry,
            population: fixture.population.clone(),
        })
        .unwrap_err();
        assert_eq!(
            error,
            PacketAssemblyErrorV0::InvalidPopulation("whole_graph_support")
        );
    }

    #[test]
    fn newly_introduced_population_zeros_are_canonicalized() {
        let mut fixture = fixture();
        fixture.population.semantic_portals[0].base_level_km = -0.0;
        fixture.population.population_definition_hash =
            population_definition_hash_v0(&fixture.population).unwrap();
        let packet = assemble(&fixture);
        assert_eq!(
            packet.population.semantic_portals[0]
                .base_level_km
                .to_bits(),
            0.0f64.to_bits()
        );
    }
}
