//! O0a relationships between the independent S0 surface hierarchy and D0
//! drainage tree.
//!
//! The records in this module are mechanical evidence.  In particular, a
//! lateral owner seam is not named a divide, a bilateral descent result is not
//! named a ridge, and a reach section is not a valley polygon.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use bincode::Options;
use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use super::drainage_evidence_hash_v0;
use super::relationships_geometry::{
    validate_planar_subdivision_v0, validate_projected_r1_voronoi_cap_identity_v0,
};
use super::{
    surface_hierarchy_evidence_hash_v0, DrainageConfigV0, DrainageReceiverV0,
    EvaluationBoundaryConditionV0, EvaluationDomainV0, EvaluationDrainageV0,
    EvaluationSurfaceGraphV0, HighlandMeasurementsV0, IncrementalCatchmentOwnerV0,
    RawCatchmentBoundaryFaceV0, SurfaceHierarchyConfigV0, SurfaceHierarchyV0, D0_HASH_VERSION,
    D0_SCHEMA_VERSION, G0S0_HASH_VERSION, G0S0_SCHEMA_VERSION,
};

pub const O0A_SCHEMA_VERSION: &str = "landform-relationships-o0a-v0";
pub const O0A_HASH_VERSION: &str = "fnv1a64-bincode-fixint-le-v0";
const REFERENCE_REACH_SUPPORT_KM2: f64 = 2_000.0;
const TANGENT_HALF_SUPPORT_KM: f64 = 10.0;
const LONGITUDINAL_STEP_KM: f64 = 4.0;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PacketGeometryIdentityV0 {
    LandscapeRegularPlanar {
        nominal_spacing_km: f64,
        canonical_graph_hash: u64,
    },
    ProjectedR1VoronoiCap {
        canonical_graph_hash: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LandformRelationshipConfigV0 {
    pub station_spacing_km: f64,
    pub cross_section_half_length_km: f64,
    pub cross_section_sample_step_km: f64,
    pub relative_height_fraction: f64,
    pub maximum_downstream_support_km: f64,
    pub schema_version: &'static str,
    pub hash_version: &'static str,
}

impl Default for LandformRelationshipConfigV0 {
    fn default() -> Self {
        Self {
            station_spacing_km: 20.0,
            cross_section_half_length_km: 100.0,
            cross_section_sample_step_km: 4.0,
            relative_height_fraction: 0.25,
            maximum_downstream_support_km: 400.0,
            schema_version: O0A_SCHEMA_VERSION,
            hash_version: O0A_HASH_VERSION,
        }
    }
}

impl LandformRelationshipConfigV0 {
    fn validate(self) -> Result<RelationshipRunNamespaceV0, RelationshipErrorV0> {
        if self.schema_version != O0A_SCHEMA_VERSION || self.hash_version != O0A_HASH_VERSION {
            return Err(RelationshipErrorV0::UnregisteredConfiguration);
        }
        let reference = Self::default();
        let allowed = [
            (
                self.station_spacing_km,
                reference.station_spacing_km,
                [10.0, 40.0],
            ),
            (
                self.cross_section_half_length_km,
                reference.cross_section_half_length_km,
                [50.0, 150.0],
            ),
            (
                self.cross_section_sample_step_km,
                reference.cross_section_sample_step_km,
                [2.0, 8.0],
            ),
            (
                self.relative_height_fraction,
                reference.relative_height_fraction,
                [0.15, 0.35],
            ),
            (
                self.maximum_downstream_support_km,
                reference.maximum_downstream_support_km,
                [200.0, 600.0],
            ),
        ];
        if allowed.iter().any(|(v, _, _)| !v.is_finite() || *v <= 0.0) {
            return Err(RelationshipErrorV0::NonFiniteOrNonPositiveConfiguration);
        }
        let mut changed = None;
        for (index, (value, base, alternatives)) in allowed.into_iter().enumerate() {
            if value != base {
                if !alternatives.contains(&value) {
                    return Err(RelationshipErrorV0::UnregisteredConfiguration);
                }
                if changed.replace((index, value == alternatives[0])).is_some() {
                    return Err(RelationshipErrorV0::UnregisteredConfiguration);
                }
            }
        }
        Ok(match changed {
            None => RelationshipRunNamespaceV0::Reference,
            Some((0, true)) => RelationshipRunNamespaceV0::StationSpacingLow,
            Some((0, false)) => RelationshipRunNamespaceV0::StationSpacingHigh,
            Some((1, true)) => RelationshipRunNamespaceV0::CrossSectionHalfLengthLow,
            Some((1, false)) => RelationshipRunNamespaceV0::CrossSectionHalfLengthHigh,
            Some((2, true)) => RelationshipRunNamespaceV0::CrossSectionSampleStepLow,
            Some((2, false)) => RelationshipRunNamespaceV0::CrossSectionSampleStepHigh,
            Some((3, true)) => RelationshipRunNamespaceV0::RelativeHeightFractionLow,
            Some((3, false)) => RelationshipRunNamespaceV0::RelativeHeightFractionHigh,
            Some((4, true)) => RelationshipRunNamespaceV0::MaximumDownstreamSupportLow,
            Some((4, false)) => RelationshipRunNamespaceV0::MaximumDownstreamSupportHigh,
            Some(_) => unreachable!("registered relationship factor index"),
        })
    }
}

/// Explicit namespace separating the reference packet from each registered
/// one-factor sensitivity population.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RelationshipRunNamespaceV0 {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OwnerAncestryV0 {
    Same,
    FirstIsAncestor,
    SecondIsAncestor,
    Incomparable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryFaceRoleKindV0 {
    FlowTransition,
    LateralBoundaryCandidate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackedBoundaryFaceV0 {
    pub cells: [u32; 2],
    pub directed_edges: [u32; 2],
    pub owners: [IncrementalCatchmentOwnerV0; 2],
    pub endpoints_km: [DVec3; 2],
    pub physical_length_km: f64,
    pub reconstructed_face_height_km: f64,
    pub covering_radius_km: f64,
    pub owner_ancestry: OwnerAncestryV0,
    pub role: BoundaryFaceRoleKindV0,
    /// `(donor, receiver)` for an exact receiver-crossed face.
    pub receiver_direction: Option<[u32; 2]>,
    pub bilateral_descent: Option<BilateralPhysicalDescentV0>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceTargetV0 {
    ReachCell {
        reach_id: u32,
        cell: u32,
    },
    Portal {
        portal_id: u32,
        boundary_segment: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReceiverTraceStatusV0 {
    ReachedTarget,
    TargetAtBoundaryCell,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReceiverTraceDescentV0 {
    pub adjacent_cell: u32,
    pub target: TraceTargetV0,
    pub status: ReceiverTraceStatusV0,
    pub adjacent_elevation_km: f64,
    pub target_elevation_km: f64,
    pub target_drop_km: f64,
    pub minimum_segment_drop_km: Option<f64>,
    pub remote_maximum_excess_km: f64,
    pub receiver_length_km: f64,
    pub endpoint_distance_km: f64,
    pub tortuosity: Option<f64>,
    pub fill_supported: bool,
    pub flat_supported: bool,
    pub physically_non_descending_segment: bool,
    pub physically_descending: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BilateralPhysicalDescentV0 {
    pub sides: [ReceiverTraceDescentV0; 2],
    pub bilateral_physical_descent: bool,
    pub unconditioned_bilateral_descent: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HighlandBoundaryRelationshipV0 {
    pub peak_id: u32,
    pub candidate_length_km: f64,
    pub unconditioned_bilateral_descent_length_km: f64,
    pub unconditioned_length_ratio: Option<f64>,
    pub fill_supported_length_share: Option<f64>,
    pub flat_supported_length_share: Option<f64>,
    pub physical_non_descent_length_share: Option<f64>,
    pub boundary_axial_orientation: Option<DVec3>,
    pub acute_axis_difference_rad: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SaddleBoundaryAssociationV0 {
    pub saddle_id: u32,
    pub elder_peak_id: u32,
    pub losing_peak_id: u32,
    pub owners: [IncrementalCatchmentOwnerV0; 2],
    pub owner_ancestry: OwnerAncestryV0,
    pub boundary_face_index: Option<u32>,
    pub separation_km: Option<f64>,
    pub effective_covering_radius_km: Option<f64>,
    pub within_covering_radius: Option<bool>,
    pub saddle_minus_face_height_km: Option<f64>,
    pub bilateral_descent: Option<BilateralPhysicalDescentV0>,
    pub equal_elder_ambiguous: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LongitudinalReachSampleV0 {
    pub arclength_km: f64,
    pub point_km: DVec3,
    pub physical_elevation_km: f64,
    pub donor_cell: u32,
    pub structural_area_km2: f64,
    pub supplied_runoff: f64,
    pub interval_grade: Option<f64>,
    pub interval_fill_supported: bool,
    pub interval_flat_supported: bool,
    pub interval_physically_non_descending: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SectionCensorReasonV0 {
    AxisOutsideCatchment,
    AxisOnBoundary,
    CollinearBoundary,
    DomainBoundary,
    FlowTransition,
    AmbiguousFaceGeometry,
    NoCatchmentExitWithinSupport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossSectionSampleV0 {
    pub signed_offset_km: f64,
    pub point_km: DVec3,
    pub physical_elevation_km: Option<f64>,
    pub height_provenance: Option<CrossSectionHeightProvenanceV0>,
    pub outside_domain: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrossSectionHeightProvenanceV0 {
    CellMean,
    BoundaryHeightProxy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelativeHeightBracketV0 {
    CellMeans,
    CellMeanToBoundaryProxy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossSectionSideV0 {
    pub censor_reason: Option<SectionCensorReasonV0>,
    pub boundary_face_index: Option<u32>,
    pub boundary_offset_km: Option<f64>,
    pub boundary_height_proxy_km: Option<f64>,
    pub samples: Vec<CrossSectionSampleV0>,
    pub boundary_relief_km: Option<f64>,
    pub maximum_sampled_relief_km: Option<f64>,
    pub boundary_maximum_separation_km: Option<f64>,
    pub minimum_elevation_offset_km: Option<f64>,
    pub positive_boundary_relief: Option<bool>,
    pub relative_height_crossing_km: Option<f64>,
    pub crossing_bracket: Option<RelativeHeightBracketV0>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReachCrossSectionStationV0 {
    pub station_arclength_km: f64,
    pub axis_point_km: DVec3,
    pub tangent: DVec3,
    pub left: CrossSectionSideV0,
    pub right: CrossSectionSideV0,
    pub relative_relief_span_km: Option<f64>,
    pub span_unavailable_reason: Option<RelativeReliefSpanUnavailableV0>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelativeReliefSpanUnavailableV0 {
    Censored(SectionCensorReasonV0),
    NonPositiveBoundaryRelief,
    MissingRelativeHeightCrossing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReachCrossSectionProbeV0 {
    pub reach_id: u32,
    pub retained_length_km: f64,
    pub source_truncated: bool,
    pub retained_polyline_km: Vec<DVec3>,
    pub longitudinal_samples: Vec<LongitudinalReachSampleV0>,
    pub stations: Vec<ReachCrossSectionStationV0>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelationshipWorkCountsV0 {
    pub raw_boundary_faces: u64,
    pub receiver_trace_segments: u64,
    pub reach_stations: u64,
    pub regular_cross_section_samples: u64,
    pub candidate_face_tests: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LandformRelationshipsV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub geometry_identity: PacketGeometryIdentityV0,
    pub config: LandformRelationshipConfigV0,
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

#[derive(Debug, Clone, PartialEq)]
pub enum RelationshipErrorV0 {
    UnsupportedDomain,
    LengthMismatch(&'static str),
    NonFiniteInput(&'static str),
    NonCanonicalZero { field: &'static str, index: usize },
    NonFiniteOrNonPositiveConfiguration,
    UnregisteredConfiguration,
    InvalidGeometryIdentity,
    InvalidGraph(String),
    InvalidSurfaceHierarchy(String),
    InvalidDrainage(String),
    SurfaceHierarchyHashMismatch { stored: u64, computed: u64 },
    DrainageHashMismatch { stored: u64, computed: u64 },
    MissingReferenceDrainageScale,
    DuplicateReferenceDrainageScale,
    InconsistentRawBoundaryFaces,
    AmbiguousBoundaryBacking,
    MissingOwnerAncestry(IncrementalCatchmentOwnerV0),
    InconsistentReceiverTarget { cell: usize },
    ReceiverCycle,
    DegenerateReachPolyline { reach_id: u32 },
    InvalidCrossSectionTangent { reach_id: u32 },
    PointLocationFailure { reach_id: u32 },
    NonFiniteDerivedEvidence,
    Serialization(String),
}

impl fmt::Display for RelationshipErrorV0 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RelationshipErrorV0 {}

#[derive(Clone)]
struct TraceSummary {
    target: TraceTargetV0,
    target_point: DVec3,
    target_z: f64,
    min_drop: Option<f64>,
    maximum_z: f64,
    length: f64,
    length_correction: f64,
    segment_count: u64,
    fill: bool,
    flat: bool,
    non_descending: bool,
}

/// Build the registered O0a common relationship evidence packet.
#[allow(clippy::too_many_arguments)]
pub fn build_landform_relationships_v0(
    graph: &EvaluationSurfaceGraphV0,
    physical_elevation_km: &[f64],
    scored_cell: &[bool],
    local_runoff_supply: &[f64],
    surface_config: SurfaceHierarchyConfigV0,
    drainage_config: DrainageConfigV0,
    hierarchy: &SurfaceHierarchyV0,
    drainage: &EvaluationDrainageV0,
    geometry_identity: PacketGeometryIdentityV0,
    config: LandformRelationshipConfigV0,
) -> Result<LandformRelationshipsV0, RelationshipErrorV0> {
    let run_namespace = config.validate()?;
    let mut surface_config = surface_config;
    surface_config.closure_level_km = canonical_zero(surface_config.closure_level_km);
    if graph.domain != EvaluationDomainV0::Planar {
        return Err(RelationshipErrorV0::UnsupportedDomain);
    }
    graph
        .validate(&surface_config)
        .map_err(|e| RelationshipErrorV0::InvalidGraph(e.to_string()))?;
    validate_planar_subdivision_v0(
        graph,
        surface_config.endpoint_match_abs_km,
        surface_config.planar_area_match_relative,
    )?;
    let n = graph.cell_count();
    for (name, len) in [
        ("physical_elevation_km", physical_elevation_km.len()),
        ("scored_cell", scored_cell.len()),
        ("local_runoff_supply", local_runoff_supply.len()),
    ] {
        if len != n {
            return Err(RelationshipErrorV0::LengthMismatch(name));
        }
    }
    validate_canonical_input(physical_elevation_km, "physical_elevation_km", false)?;
    validate_canonical_input(local_runoff_supply, "local_runoff_supply", true)?;
    validate_geometry_identity(
        graph,
        geometry_identity,
        surface_config.endpoint_match_abs_km,
        surface_config.planar_area_match_relative,
    )?;
    if hierarchy.schema_version != G0S0_SCHEMA_VERSION
        || hierarchy.hash_version != G0S0_HASH_VERSION
    {
        return Err(RelationshipErrorV0::InvalidSurfaceHierarchy(
            "schema/hash version mismatch".into(),
        ));
    }
    if drainage.schema_version != D0_SCHEMA_VERSION || drainage.hash_version != D0_HASH_VERSION {
        return Err(RelationshipErrorV0::InvalidDrainage(
            "schema/hash version mismatch".into(),
        ));
    }
    let s0_hash = surface_hierarchy_evidence_hash_v0(
        graph,
        physical_elevation_km,
        scored_cell,
        surface_config,
        hierarchy,
    )
    .map_err(|e| RelationshipErrorV0::InvalidSurfaceHierarchy(e.to_string()))?;
    if s0_hash != hierarchy.derived_evidence_hash {
        return Err(RelationshipErrorV0::SurfaceHierarchyHashMismatch {
            stored: hierarchy.derived_evidence_hash,
            computed: s0_hash,
        });
    }
    let d0_hash = drainage_evidence_hash_v0(
        graph,
        physical_elevation_km,
        local_runoff_supply,
        drainage_config,
        drainage,
    )
    .map_err(|e| RelationshipErrorV0::InvalidDrainage(e.to_string()))?;
    if d0_hash != drainage.derived_evidence_hash {
        return Err(RelationshipErrorV0::DrainageHashMismatch {
            stored: drainage.derived_evidence_hash,
            computed: d0_hash,
        });
    }
    let mut scales = drainage
        .scales
        .iter()
        .filter(|s| s.support_threshold_km2 == REFERENCE_REACH_SUPPORT_KM2);
    let scale = scales
        .next()
        .ok_or(RelationshipErrorV0::MissingReferenceDrainageScale)?;
    if scales.next().is_some() {
        return Err(RelationshipErrorV0::DuplicateReferenceDrainageScale);
    }
    validate_reference_scale(n, scale)?;

    let reach_parent = reach_parent_table(scale)?;
    let reach_outlet: Vec<u32> = scale
        .reach_graph
        .reaches
        .iter()
        .map(|reach| reach.outlet_portal_id)
        .collect();
    let ancestry = |a, b| owner_ancestry(a, b, &reach_parent, &reach_outlet);
    let mut faces = back_boundary_faces(
        graph,
        physical_elevation_km,
        &drainage.routing.receiver,
        &scale.basin_graph.exclusive_owner,
        &scale.basin_graph.raw_catchment_boundaries,
        &ancestry,
    )?;
    let (trace_summaries, trace_segments) =
        trace_summaries(graph, physical_elevation_km, drainage, scale)?;
    for face in &mut faces {
        if face.role == BoundaryFaceRoleKindV0::LateralBoundaryCandidate {
            let a = trace_record(
                graph,
                face.cells[0] as usize,
                physical_elevation_km,
                &trace_summaries,
            )?;
            let b = trace_record(
                graph,
                face.cells[1] as usize,
                physical_elevation_km,
                &trace_summaries,
            )?;
            let physical = a.physically_descending && b.physically_descending;
            let conditioned = a.fill_supported
                || b.fill_supported
                || a.flat_supported
                || b.flat_supported
                || a.physically_non_descending_segment
                || b.physically_non_descending_segment;
            face.bilateral_descent = Some(BilateralPhysicalDescentV0 {
                sides: [a, b],
                bilateral_physical_descent: physical,
                unconditioned_bilateral_descent: physical && !conditioned,
            });
        }
    }
    let highlands = highland_relationships(hierarchy, &faces);
    let saddles = saddle_relationships(graph, hierarchy, scale, &faces, &ancestry)?;
    let spatial_index = RelationshipSpatialIndex::build(graph, &faces)?;
    let (reaches, candidate_face_tests) = reach_probes(
        graph,
        physical_elevation_km,
        drainage,
        scale,
        &faces,
        &reach_parent,
        &spatial_index,
        config,
    )?;
    let reach_stations = reaches.iter().map(|r| r.stations.len() as u64).sum();
    let regular_cross_section_samples = reaches
        .iter()
        .flat_map(|r| &r.stations)
        .map(|s| {
            [&s.left, &s.right]
                .into_iter()
                .flat_map(|side| &side.samples)
                .filter(|sample| {
                    sample.height_provenance
                        != Some(CrossSectionHeightProvenanceV0::BoundaryHeightProxy)
                })
                .count() as u64
        })
        .sum();
    let mut result = LandformRelationshipsV0 {
        schema_version: O0A_SCHEMA_VERSION.into(),
        hash_version: O0A_HASH_VERSION.into(),
        geometry_identity,
        config,
        run_namespace,
        surface_hierarchy_input_hash: s0_hash,
        drainage_input_hash: d0_hash,
        backed_boundary_faces: faces,
        highland_boundary_relationships: highlands,
        saddle_boundary_associations: saddles,
        reach_cross_section_probes: reaches,
        work_counts: RelationshipWorkCountsV0 {
            raw_boundary_faces: scale.basin_graph.raw_catchment_boundaries.len() as u64,
            receiver_trace_segments: trace_segments,
            reach_stations,
            regular_cross_section_samples,
            candidate_face_tests,
        },
        derived_evidence_hash: 0,
    };
    reject_nonfinite_result(&result)?;
    result.derived_evidence_hash = relationship_hash(
        graph,
        physical_elevation_km,
        scored_cell,
        local_runoff_supply,
        &surface_config,
        &drainage_config,
        hierarchy,
        drainage,
        &result,
    )?;
    Ok(result)
}

fn validate_canonical_input(
    values: &[f64],
    field: &'static str,
    non_negative: bool,
) -> Result<(), RelationshipErrorV0> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() || (non_negative && value < 0.0) {
            return Err(RelationshipErrorV0::NonFiniteInput(field));
        }
        if value == 0.0 && value.to_bits() != 0 {
            return Err(RelationshipErrorV0::NonCanonicalZero { field, index });
        }
    }
    Ok(())
}

fn validate_geometry_identity(
    graph: &EvaluationSurfaceGraphV0,
    identity: PacketGeometryIdentityV0,
    endpoint_tolerance_km: f64,
    relative_tolerance: f64,
) -> Result<(), RelationshipErrorV0> {
    let bytes = fixed_bytes(graph)?;
    let actual = fnv1a64(&bytes);
    match identity {
        PacketGeometryIdentityV0::LandscapeRegularPlanar {
            nominal_spacing_km,
            canonical_graph_hash,
        } => {
            if !nominal_spacing_km.is_finite()
                || nominal_spacing_km <= 0.0
                || canonical_graph_hash != actual
            {
                return Err(RelationshipErrorV0::InvalidGeometryIdentity);
            }
            let vertex_radius = nominal_spacing_km / 3.0_f64.sqrt();
            let close_exact = |actual: f64, expected: f64| {
                let tolerance = endpoint_tolerance_km
                    .max(relative_tolerance * expected.abs().max(f64::MIN_POSITIVE));
                actual.is_finite() && (actual - expected).abs() <= tolerance
            };
            let close_operator = |actual: f64, expected: f64| {
                let tolerance = endpoint_tolerance_km
                    .max(relative_tolerance * expected.abs().max(f64::MIN_POSITIVE))
                    .max(2.0 * f64::from(f32::EPSILON) * expected.abs());
                actual.is_finite() && (actual - expected).abs() <= tolerance
            };
            for cell in 0..graph.cell_count() {
                let Some(corners) = regular_hex_corners(graph.polygon(cell), endpoint_tolerance_km)
                else {
                    return Err(RelationshipErrorV0::InvalidGeometryIdentity);
                };
                if corners.len() != 6
                    || corners.iter().any(|&vertex| {
                        !close_exact(vertex.distance(graph.cell_center_km[cell]), vertex_radius)
                    })
                {
                    return Err(RelationshipErrorV0::InvalidGeometryIdentity);
                }
                for edge in edge_range(graph, cell) {
                    let neighbor = graph.edge_neighbor[edge] as usize;
                    if !close_exact(
                        graph.cell_center_km[cell].distance(graph.cell_center_km[neighbor]),
                        nominal_spacing_km,
                    ) || !close_operator(graph.edge_distance_km[edge], nominal_spacing_km)
                        || !close_operator(graph.edge_shared_width_km[edge], vertex_radius)
                    {
                        return Err(RelationshipErrorV0::InvalidGeometryIdentity);
                    }
                }
            }
        }
        PacketGeometryIdentityV0::ProjectedR1VoronoiCap {
            canonical_graph_hash,
        } if canonical_graph_hash != actual => {
            return Err(RelationshipErrorV0::InvalidGeometryIdentity);
        }
        PacketGeometryIdentityV0::ProjectedR1VoronoiCap {
            canonical_graph_hash,
        } => validate_projected_r1_voronoi_cap_identity_v0(graph, canonical_graph_hash)?,
    }
    Ok(())
}

fn regular_hex_corners(polygon: &[DVec3], tolerance: f64) -> Option<Vec<DVec3>> {
    if polygon.len() < 6 {
        return None;
    }
    let mut corners = Vec::new();
    for index in 0..polygon.len() {
        let previous = polygon[(index + polygon.len() - 1) % polygon.len()].truncate();
        let current = polygon[index].truncate();
        let next = polygon[(index + 1) % polygon.len()].truncate();
        let incoming = current - previous;
        let outgoing = next - current;
        let cross = incoming.x * outgoing.y - incoming.y * outgoing.x;
        let scale = incoming.length() + outgoing.length();
        let same_direction = incoming.dot(outgoing) > 0.0;
        if !same_direction || cross.abs() > tolerance * scale {
            corners.push(polygon[index]);
        }
    }
    if corners.len() != 6 {
        return None;
    }
    for &point in polygon {
        if corners.contains(&point) {
            continue;
        }
        let point = point.truncate();
        let lies_on_regular_face = corners
            .iter()
            .copied()
            .zip(corners.iter().copied().cycle().skip(1))
            .take(corners.len())
            .any(|(start, end)| {
                point_on_segment_tol(point, start.truncate(), end.truncate(), tolerance)
            });
        if !lies_on_regular_face {
            return None;
        }
    }
    Some(corners)
}

fn point_on_segment_tol(point: DVec2, start: DVec2, end: DVec2, tolerance: f64) -> bool {
    let edge = end - start;
    let length = edge.length();
    if !length.is_finite() || length <= 0.0 {
        return false;
    }
    let offset = point - start;
    let cross = edge.x * offset.y - edge.y * offset.x;
    if cross.abs() > tolerance * length {
        return false;
    }
    let along = offset.dot(edge) / length;
    along >= -tolerance && along <= length + tolerance
}

/// Canonical hash to store in [`PacketGeometryIdentityV0`].
pub fn relationship_graph_hash_v0(
    graph: &EvaluationSurfaceGraphV0,
) -> Result<u64, RelationshipErrorV0> {
    Ok(fnv1a64(&fixed_bytes(graph)?))
}

fn validate_reference_scale(
    n: usize,
    scale: &super::DrainageScaleV0,
) -> Result<(), RelationshipErrorV0> {
    if scale.basin_graph.exclusive_owner.len() != n || scale.reach_graph.cell_reach.len() != n {
        return Err(RelationshipErrorV0::InvalidDrainage(
            "reference-scale cell array length mismatch".into(),
        ));
    }
    for (index, reach) in scale.reach_graph.reaches.iter().enumerate() {
        if reach.id as usize != index
            || reach.cells.is_empty()
            || reach.cells.iter().any(|&cell| cell as usize >= n)
        {
            return Err(RelationshipErrorV0::InvalidDrainage(
                "malformed reference reach graph".into(),
            ));
        }
    }
    Ok(())
}

fn reach_parent_table(
    scale: &super::DrainageScaleV0,
) -> Result<Vec<Option<u32>>, RelationshipErrorV0> {
    let n = scale.reach_graph.reaches.len();
    let mut result = vec![None; n];
    for reach in &scale.reach_graph.reaches {
        if let Some(parent) = reach.downstream_reach {
            if parent as usize >= n || parent == reach.id {
                return Err(RelationshipErrorV0::MissingOwnerAncestry(
                    IncrementalCatchmentOwnerV0::Reach(reach.id),
                ));
            }
            result[reach.id as usize] = Some(parent);
        } else if reach.terminal_portal_id.is_none() {
            return Err(RelationshipErrorV0::MissingOwnerAncestry(
                IncrementalCatchmentOwnerV0::Reach(reach.id),
            ));
        }
    }
    for start in 0..n {
        let mut seen = BTreeSet::new();
        let mut current = Some(start as u32);
        while let Some(reach) = current {
            if !seen.insert(reach) {
                return Err(RelationshipErrorV0::ReceiverCycle);
            }
            current = result[reach as usize];
        }
    }
    Ok(result)
}

fn owner_ancestry(
    a: IncrementalCatchmentOwnerV0,
    b: IncrementalCatchmentOwnerV0,
    parents: &[Option<u32>],
    outlets: &[u32],
) -> OwnerAncestryV0 {
    if a == b {
        return OwnerAncestryV0::Same;
    }
    let is_ancestor = |ancestor: IncrementalCatchmentOwnerV0,
                       descendant: IncrementalCatchmentOwnerV0| {
        match (ancestor, descendant) {
            (IncrementalCatchmentOwnerV0::Reach(a), IncrementalCatchmentOwnerV0::Reach(mut b)) => {
                loop {
                    if a == b {
                        return true;
                    }
                    let Some(next) = parents.get(b as usize).copied().flatten() else {
                        return false;
                    };
                    b = next;
                }
            }
            (IncrementalCatchmentOwnerV0::Portal(p), IncrementalCatchmentOwnerV0::Reach(r)) => {
                outlets.get(r as usize).copied() == Some(p)
            }
            _ => false,
        }
    };
    if is_ancestor(a, b) {
        OwnerAncestryV0::FirstIsAncestor
    } else if is_ancestor(b, a) {
        OwnerAncestryV0::SecondIsAncestor
    } else {
        OwnerAncestryV0::Incomparable
    }
}

fn edge_range(graph: &EvaluationSurfaceGraphV0, cell: usize) -> std::ops::Range<usize> {
    graph.edge_offsets[cell] as usize..graph.edge_offsets[cell + 1] as usize
}

#[derive(Debug, Clone, Copy)]
struct SpatialBounds {
    min: DVec2,
    max: DVec2,
}

impl SpatialBounds {
    fn points(points: impl IntoIterator<Item = DVec2>) -> Option<Self> {
        let mut points = points.into_iter();
        let first = points.next()?;
        let mut bounds = Self {
            min: first,
            max: first,
        };
        for point in points {
            bounds.min = bounds.min.min(point);
            bounds.max = bounds.max.max(point);
        }
        Some(bounds)
    }

    fn segment(a: DVec3, b: DVec3) -> Self {
        let a = a.truncate();
        let b = b.truncate();
        Self {
            min: a.min(b),
            max: a.max(b),
        }
    }

    fn contains(self, point: DVec2) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
    }
}

#[derive(Debug, Clone, Copy)]
enum IndexedPhysicalFaceKind {
    Internal {
        cells: [u32; 2],
        backed_boundary: Option<u32>,
    },
    DomainBoundary {
        segment: u32,
    },
}

#[derive(Debug, Clone, Copy)]
struct IndexedPhysicalFace {
    endpoints_km: [DVec3; 2],
    kind: IndexedPhysicalFaceKind,
}

/// Deterministic uniform bounding-box index shared by all O0a point and ray
/// queries. Bucket contents and query answers retain stable numeric ordering.
struct RelationshipSpatialIndex {
    bounds: SpatialBounds,
    bins_x: usize,
    bins_y: usize,
    polygon_bounds: Vec<SpatialBounds>,
    polygon_buckets: Vec<Vec<u32>>,
    physical_faces: Vec<IndexedPhysicalFace>,
    face_bounds: Vec<SpatialBounds>,
    face_buckets: Vec<Vec<u32>>,
}

impl RelationshipSpatialIndex {
    fn build(
        graph: &EvaluationSurfaceGraphV0,
        backed_faces: &[BackedBoundaryFaceV0],
    ) -> Result<Self, RelationshipErrorV0> {
        let bounds = SpatialBounds::points(
            graph
                .cell_polygon_vertices_km
                .iter()
                .map(|point| point.truncate()),
        )
        .ok_or_else(|| RelationshipErrorV0::InvalidGraph("empty planar subdivision".into()))?;
        let width = bounds.max.x - bounds.min.x;
        let height = bounds.max.y - bounds.min.y;
        if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
            return Err(RelationshipErrorV0::InvalidGraph(
                "degenerate planar subdivision bounds".into(),
            ));
        }
        let count = graph.cell_count().max(1) as f64;
        let bins_x = (count * width / height).sqrt().ceil().max(1.0) as usize;
        let bins_y = (count / bins_x as f64).ceil().max(1.0) as usize;
        let mut index = Self {
            bounds,
            bins_x,
            bins_y,
            polygon_bounds: Vec::with_capacity(graph.cell_count()),
            polygon_buckets: vec![Vec::new(); bins_x * bins_y],
            physical_faces: Vec::new(),
            face_bounds: Vec::new(),
            face_buckets: vec![Vec::new(); bins_x * bins_y],
        };
        for cell in 0..graph.cell_count() {
            let cell_bounds =
                SpatialBounds::points(graph.polygon(cell).iter().map(|point| point.truncate()))
                    .ok_or_else(|| {
                        RelationshipErrorV0::InvalidGraph("empty cell polygon".into())
                    })?;
            index.polygon_bounds.push(cell_bounds);
            for bucket in index.bucket_range(cell_bounds) {
                index.polygon_buckets[bucket].push(cell as u32);
            }
        }

        let mut backed_by_edge = BTreeMap::<u32, u32>::new();
        for (face_index, face) in backed_faces.iter().enumerate() {
            for edge in face.directed_edges {
                if backed_by_edge.insert(edge, face_index as u32).is_some() {
                    return Err(RelationshipErrorV0::AmbiguousBoundaryBacking);
                }
            }
        }
        for cell in 0..graph.cell_count() {
            for edge in edge_range(graph, cell) {
                let reciprocal = graph.edge_reciprocal[edge] as usize;
                if edge > reciprocal {
                    continue;
                }
                let neighbor = graph.edge_neighbor[edge];
                index.physical_faces.push(IndexedPhysicalFace {
                    endpoints_km: graph.edge_face_endpoints_km[edge],
                    kind: IndexedPhysicalFaceKind::Internal {
                        cells: [cell as u32, neighbor],
                        backed_boundary: backed_by_edge.get(&(edge as u32)).copied(),
                    },
                });
            }
        }
        for (segment, boundary) in graph.boundary_segments.iter().enumerate() {
            index.physical_faces.push(IndexedPhysicalFace {
                endpoints_km: boundary.endpoints_km,
                kind: IndexedPhysicalFaceKind::DomainBoundary {
                    segment: segment as u32,
                },
            });
        }
        for face_index in 0..index.physical_faces.len() {
            let face = index.physical_faces[face_index];
            let face_bounds = SpatialBounds::segment(face.endpoints_km[0], face.endpoints_km[1]);
            index.face_bounds.push(face_bounds);
            for bucket in index.bucket_range(face_bounds) {
                index.face_buckets[bucket].push(face_index as u32);
            }
        }
        Ok(index)
    }

    fn bucket_coord(&self, point: DVec2) -> (usize, usize) {
        let normalized_x = (point.x - self.bounds.min.x) / (self.bounds.max.x - self.bounds.min.x);
        let normalized_y = (point.y - self.bounds.min.y) / (self.bounds.max.y - self.bounds.min.y);
        let x = (normalized_x * self.bins_x as f64).floor() as isize;
        let y = (normalized_y * self.bins_y as f64).floor() as isize;
        (
            x.clamp(0, self.bins_x as isize - 1) as usize,
            y.clamp(0, self.bins_y as isize - 1) as usize,
        )
    }

    fn bucket_range(&self, bounds: SpatialBounds) -> Vec<usize> {
        let (min_x, min_y) = self.bucket_coord(bounds.min);
        let (max_x, max_y) = self.bucket_coord(bounds.max);
        let mut result = Vec::with_capacity((max_x - min_x + 1) * (max_y - min_y + 1));
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                result.push(y * self.bins_x + x);
            }
        }
        result
    }

    fn locate_cell(&self, graph: &EvaluationSurfaceGraphV0, point: DVec3) -> Option<usize> {
        let point2 = point.truncate();
        if !self.bounds.contains(point2) {
            return None;
        }
        let (x, y) = self.bucket_coord(point2);
        let bucket = y * self.bins_x + x;
        let mut matches: Vec<usize> = self.polygon_buckets[bucket]
            .iter()
            .copied()
            .map(|cell| cell as usize)
            .filter(|&cell| {
                self.polygon_bounds[cell].contains(point2)
                    && polygon_contains(graph.polygon(cell), point2)
            })
            .collect();
        matches.sort_by(|&a, &b| point_cmp(graph.cell_center_km[a], graph.cell_center_km[b]));
        matches.into_iter().next()
    }

    fn ray_face_candidates(&self, axis: DVec3, endpoint: DVec3) -> Vec<usize> {
        let query = SpatialBounds::segment(axis, endpoint);
        let mut candidates = BTreeSet::<u32>::new();
        for bucket in self.bucket_range(query) {
            candidates.extend(self.face_buckets[bucket].iter().copied());
        }
        candidates
            .into_iter()
            .map(|face| face as usize)
            .filter(|&face| bounds_overlap(self.face_bounds[face], query))
            .collect()
    }
}

fn bounds_overlap(a: SpatialBounds, b: SpatialBounds) -> bool {
    a.max.x >= b.min.x && b.max.x >= a.min.x && a.max.y >= b.min.y && b.max.y >= a.min.y
}

fn back_boundary_faces(
    graph: &EvaluationSurfaceGraphV0,
    elevation: &[f64],
    receivers: &[DrainageReceiverV0],
    owners: &[IncrementalCatchmentOwnerV0],
    raw: &[RawCatchmentBoundaryFaceV0],
    ancestry: &impl Fn(IncrementalCatchmentOwnerV0, IncrementalCatchmentOwnerV0) -> OwnerAncestryV0,
) -> Result<Vec<BackedBoundaryFaceV0>, RelationshipErrorV0> {
    if owners.len() != graph.cell_count() || receivers.len() != graph.cell_count() {
        return Err(RelationshipErrorV0::InvalidDrainage(
            "routing/owner cell count mismatch".into(),
        ));
    }
    let mut derived = Vec::<(RawCatchmentBoundaryFaceV0, BackedBoundaryFaceV0)>::new();
    for cell in 0..graph.cell_count() {
        for edge in edge_range(graph, cell) {
            let reciprocal = graph.edge_reciprocal[edge] as usize;
            if edge > reciprocal {
                continue;
            }
            let neighbor = graph.edge_neighbor[edge] as usize;
            if owners[cell] == owners[neighbor] {
                continue;
            }
            let canonical_edge = if endpoints_cmp(
                graph.edge_face_endpoints_km[edge],
                graph.edge_face_endpoints_km[reciprocal],
            ) == Ordering::Greater
            {
                reciprocal
            } else {
                edge
            };
            let (owner_pair, backed_cells, backed_edges) = if owners[cell] <= owners[neighbor] {
                (
                    [owners[cell], owners[neighbor]],
                    [cell as u32, neighbor as u32],
                    [edge as u32, reciprocal as u32],
                )
            } else {
                (
                    [owners[neighbor], owners[cell]],
                    [neighbor as u32, cell as u32],
                    [reciprocal as u32, edge as u32],
                )
            };
            let endpoints = graph.edge_face_endpoints_km[canonical_edge];
            let length = graph.edge_shared_width_km[canonical_edge];
            let raw_face = RawCatchmentBoundaryFaceV0 {
                owners: owner_pair,
                endpoints_km: endpoints,
                physical_length_km: length,
            };
            let a_to_b = matches!(
                receivers[cell],
                DrainageReceiverV0::Cell {
                    cell: target,
                    directed_edge
                } if target as usize == neighbor && directed_edge as usize == edge
            );
            let b_to_a = matches!(
                receivers[neighbor],
                DrainageReceiverV0::Cell {
                    cell: target,
                    directed_edge
                } if target as usize == cell && directed_edge as usize == reciprocal
            );
            if a_to_b && b_to_a {
                return Err(RelationshipErrorV0::ReceiverCycle);
            }
            let role = if a_to_b || b_to_a {
                BoundaryFaceRoleKindV0::FlowTransition
            } else {
                BoundaryFaceRoleKindV0::LateralBoundaryCandidate
            };
            let radius = covering_radius(graph, cell).max(covering_radius(graph, neighbor));
            derived.push((
                raw_face,
                BackedBoundaryFaceV0 {
                    cells: backed_cells,
                    directed_edges: backed_edges,
                    owners: owner_pair,
                    endpoints_km: endpoints,
                    physical_length_km: canonical_zero(length),
                    reconstructed_face_height_km: canonical_zero(
                        0.5 * (elevation[cell] + elevation[neighbor]),
                    ),
                    covering_radius_km: canonical_zero(radius),
                    owner_ancestry: ancestry(owner_pair[0], owner_pair[1]),
                    role,
                    receiver_direction: if a_to_b {
                        Some([cell as u32, neighbor as u32])
                    } else if b_to_a {
                        Some([neighbor as u32, cell as u32])
                    } else {
                        None
                    },
                    bilateral_descent: None,
                },
            ));
        }
    }
    derived.sort_by(|a, b| raw_face_cmp(&a.0, &b.0));
    for pair in derived.windows(2) {
        if pair[0].0 == pair[1].0 && pair[0].1.cells != pair[1].1.cells {
            return Err(RelationshipErrorV0::AmbiguousBoundaryBacking);
        }
    }
    let mut raw_sorted = raw.to_vec();
    raw_sorted.sort_by(raw_face_cmp);
    if derived.len() != raw_sorted.len()
        || derived.iter().zip(&raw_sorted).any(|((a, _), b)| a != b)
    {
        return Err(RelationshipErrorV0::InconsistentRawBoundaryFaces);
    }
    let mut backed = derived
        .into_iter()
        .map(|(_, face)| face)
        .collect::<Vec<_>>();
    backed.sort_by(|a, b| {
        endpoints_cmp(a.endpoints_km, b.endpoints_km)
            .then_with(|| a.owners.cmp(&b.owners))
            .then_with(|| a.cells.cmp(&b.cells))
            .then_with(|| a.directed_edges.cmp(&b.directed_edges))
    });
    Ok(backed)
}

fn raw_face_cmp(a: &RawCatchmentBoundaryFaceV0, b: &RawCatchmentBoundaryFaceV0) -> Ordering {
    a.owners
        .cmp(&b.owners)
        .then_with(|| endpoints_cmp(a.endpoints_km, b.endpoints_km))
}

fn covering_radius(graph: &EvaluationSurfaceGraphV0, cell: usize) -> f64 {
    graph
        .polygon(cell)
        .iter()
        .map(|&p| p.distance(graph.cell_center_km[cell]))
        .fold(0.0, f64::max)
}

fn trace_summaries(
    graph: &EvaluationSurfaceGraphV0,
    elevation: &[f64],
    drainage: &EvaluationDrainageV0,
    scale: &super::DrainageScaleV0,
) -> Result<(Vec<TraceSummary>, u64), RelationshipErrorV0> {
    let n = graph.cell_count();
    let owners = &scale.basin_graph.exclusive_owner;
    let mut memo = vec![None::<TraceSummary>; n];
    let mut state = vec![0u8; n];
    let mut total_segments = 0u64;
    for start in 0..n {
        if memo[start].is_some() {
            continue;
        }
        let mut path = Vec::<usize>::new();
        let mut current = start;
        loop {
            if memo[current].is_some() {
                break;
            }
            if state[current] == 1 {
                return Err(RelationshipErrorV0::ReceiverCycle);
            }
            state[current] = 1;
            path.push(current);
            match owners[current] {
                IncrementalCatchmentOwnerV0::Reach(reach)
                    if scale.reach_graph.cell_reach[current] == Some(reach) =>
                {
                    memo[current] = Some(TraceSummary {
                        target: TraceTargetV0::ReachCell {
                            reach_id: reach,
                            cell: current as u32,
                        },
                        target_point: graph.cell_center_km[current],
                        target_z: elevation[current],
                        min_drop: None,
                        maximum_z: elevation[current],
                        length: 0.0,
                        length_correction: 0.0,
                        segment_count: 0,
                        fill: false,
                        flat: false,
                        non_descending: false,
                    });
                    break;
                }
                IncrementalCatchmentOwnerV0::Portal(portal) => {
                    if let DrainageReceiverV0::Portal {
                        boundary_segment,
                        portal_id,
                    } = drainage.routing.receiver[current]
                    {
                        if portal_id != portal {
                            return Err(RelationshipErrorV0::InconsistentReceiverTarget {
                                cell: current,
                            });
                        }
                        let segment = graph
                            .boundary_segments
                            .get(boundary_segment as usize)
                            .ok_or(RelationshipErrorV0::InconsistentReceiverTarget {
                                cell: current,
                            })?;
                        let EvaluationBoundaryConditionV0::OpenBaseLevel {
                            portal_id: declared,
                            elevation_km: target_z,
                        } = segment.condition
                        else {
                            return Err(RelationshipErrorV0::InconsistentReceiverTarget {
                                cell: current,
                            });
                        };
                        if declared != portal {
                            return Err(RelationshipErrorV0::InconsistentReceiverTarget {
                                cell: current,
                            });
                        }
                        let target_point =
                            0.5 * (segment.endpoints_km[0] + segment.endpoints_km[1]);
                        let drop = elevation[current] - target_z;
                        memo[current] = Some(TraceSummary {
                            target: TraceTargetV0::Portal {
                                portal_id: portal,
                                boundary_segment,
                            },
                            target_point,
                            target_z,
                            min_drop: Some(drop),
                            maximum_z: elevation[current].max(target_z),
                            length: drainage.routing.segment_length_km[current],
                            length_correction: 0.0,
                            segment_count: 1,
                            fill: drainage.routing.fill_supported[current],
                            flat: drainage.routing.flat_supported[current],
                            non_descending: drainage.routing.physically_non_descending[current],
                        });
                        total_segments += 1;
                        break;
                    }
                    let DrainageReceiverV0::Cell { cell: next, .. } =
                        drainage.routing.receiver[current]
                    else {
                        unreachable!()
                    };
                    let next = next as usize;
                    if next >= n || owners[next] != owners[current] {
                        return Err(RelationshipErrorV0::InconsistentReceiverTarget {
                            cell: current,
                        });
                    }
                    current = next;
                }
                IncrementalCatchmentOwnerV0::Reach(_) => {
                    let DrainageReceiverV0::Cell { cell: next, .. } =
                        drainage.routing.receiver[current]
                    else {
                        return Err(RelationshipErrorV0::InconsistentReceiverTarget {
                            cell: current,
                        });
                    };
                    let next = next as usize;
                    if next >= n || owners[next] != owners[current] {
                        return Err(RelationshipErrorV0::InconsistentReceiverTarget {
                            cell: current,
                        });
                    }
                    current = next;
                }
            }
        }
        while let Some(cell) = path.pop() {
            if memo[cell].is_some() {
                state[cell] = 2;
                continue;
            }
            let DrainageReceiverV0::Cell { cell: next, .. } = drainage.routing.receiver[cell]
            else {
                return Err(RelationshipErrorV0::InconsistentReceiverTarget { cell });
            };
            let tail = memo[next as usize]
                .clone()
                .ok_or(RelationshipErrorV0::InconsistentReceiverTarget { cell })?;
            let drop = elevation[cell] - elevation[next as usize];
            let (length, length_correction) = kahan_add(
                tail.length,
                tail.length_correction,
                drainage.routing.segment_length_km[cell],
            );
            let summary = TraceSummary {
                target: tail.target,
                target_point: tail.target_point,
                target_z: tail.target_z,
                min_drop: Some(tail.min_drop.map_or(drop, |v| v.min(drop))),
                maximum_z: elevation[cell].max(tail.maximum_z),
                length: canonical_zero(length),
                length_correction,
                segment_count: tail.segment_count + 1,
                fill: drainage.routing.fill_supported[cell] || tail.fill,
                flat: drainage.routing.flat_supported[cell] || tail.flat,
                non_descending: drainage.routing.physically_non_descending[cell]
                    || tail.non_descending,
            };
            total_segments += 1;
            memo[cell] = Some(summary);
            state[cell] = 2;
        }
    }
    Ok((
        memo.into_iter().map(Option::unwrap).collect(),
        total_segments,
    ))
}

fn trace_record(
    graph: &EvaluationSurfaceGraphV0,
    cell: usize,
    elevation: &[f64],
    summaries: &[TraceSummary],
) -> Result<ReceiverTraceDescentV0, RelationshipErrorV0> {
    let summary = summaries
        .get(cell)
        .ok_or(RelationshipErrorV0::InconsistentReceiverTarget { cell })?;
    let zero = summary.segment_count == 0;
    let endpoint_distance = summary.target_point.distance(graph.cell_center_km[cell]);
    if !endpoint_distance.is_finite() {
        return Err(RelationshipErrorV0::NonFiniteDerivedEvidence);
    }
    let adjacent_z = elevation[cell];
    let target_drop = canonical_zero(adjacent_z - summary.target_z);
    let excess = canonical_zero(summary.maximum_z - adjacent_z);
    let min = summary.min_drop.map(canonical_zero);
    let descending =
        !zero && target_drop > 0.0 && min.is_some_and(|drop| drop > 0.0) && excess == 0.0;
    Ok(ReceiverTraceDescentV0 {
        adjacent_cell: cell as u32,
        target: summary.target,
        status: if zero {
            ReceiverTraceStatusV0::TargetAtBoundaryCell
        } else {
            ReceiverTraceStatusV0::ReachedTarget
        },
        adjacent_elevation_km: adjacent_z,
        target_elevation_km: summary.target_z,
        target_drop_km: target_drop,
        minimum_segment_drop_km: min,
        remote_maximum_excess_km: excess,
        receiver_length_km: summary.length,
        endpoint_distance_km: canonical_zero(endpoint_distance),
        tortuosity: if zero || endpoint_distance == 0.0 {
            None
        } else {
            Some(canonical_zero(summary.length / endpoint_distance))
        },
        fill_supported: summary.fill,
        flat_supported: summary.flat,
        physically_non_descending_segment: summary.non_descending,
        physically_descending: descending,
    })
}

fn highland_relationships(
    hierarchy: &SurfaceHierarchyV0,
    faces: &[BackedBoundaryFaceV0],
) -> Vec<HighlandBoundaryRelationshipV0> {
    let mut output = Vec::with_capacity(hierarchy.reference_highlands.len());
    for feature in &hierarchy.reference_highlands {
        let peak = &hierarchy.peaks[feature.peak_id as usize];
        let footprint: BTreeSet<u32> = peak.footprint_members.iter().copied().collect();
        let incident: Vec<_> = faces
            .iter()
            .filter(|face| {
                face.role == BoundaryFaceRoleKindV0::LateralBoundaryCandidate
                    && (footprint.contains(&face.cells[0]) || footprint.contains(&face.cells[1]))
            })
            .collect();
        let total = kahan_sum(incident.iter().map(|face| face.physical_length_km));
        let unconditioned = kahan_sum(incident.iter().filter_map(|face| {
            face.bilateral_descent
                .as_ref()
                .filter(|probe| probe.unconditioned_bilateral_descent)
                .map(|_| face.physical_length_km)
        }));
        let supported_length = |predicate: fn(&ReceiverTraceDescentV0) -> bool| {
            kahan_sum(incident.iter().filter_map(|face| {
                let probe = face.bilateral_descent.as_ref()?;
                (probe.sides.iter().any(predicate)).then_some(face.physical_length_km)
            }))
        };
        let (orientation, difference) = match &feature.measurements {
            HighlandMeasurementsV0::Planar(measurements)
                if !measurements.footprint_geometry.orientation_ambiguous =>
            {
                boundary_orientation(
                    incident.iter().copied().filter(|face| {
                        face.bilateral_descent
                            .as_ref()
                            .is_some_and(|p| p.unconditioned_bilateral_descent)
                    }),
                    measurements.footprint_geometry.principal_axis,
                )
            }
            _ => (None, None),
        };
        output.push(HighlandBoundaryRelationshipV0 {
            peak_id: feature.peak_id,
            candidate_length_km: total,
            unconditioned_bilateral_descent_length_km: unconditioned,
            unconditioned_length_ratio: ratio(unconditioned, total),
            fill_supported_length_share: ratio(supported_length(|side| side.fill_supported), total),
            flat_supported_length_share: ratio(supported_length(|side| side.flat_supported), total),
            physical_non_descent_length_share: ratio(
                supported_length(|side| side.physically_non_descending_segment),
                total,
            ),
            boundary_axial_orientation: orientation,
            acute_axis_difference_rad: difference,
        });
    }
    output.sort_by_key(|record| record.peak_id);
    output
}

fn boundary_orientation<'a>(
    faces: impl Iterator<Item = &'a BackedBoundaryFaceV0>,
    principal_axis: DVec3,
) -> (Option<DVec3>, Option<f64>) {
    let mut c = 0.0;
    let mut c_correction = 0.0;
    let mut s = 0.0;
    let mut s_correction = 0.0;
    let mut any = false;
    for face in faces {
        let delta = face.endpoints_km[1] - face.endpoints_km[0];
        let angle = delta.y.atan2(delta.x);
        (c, c_correction) = kahan_add(
            c,
            c_correction,
            face.physical_length_km * (2.0 * angle).cos(),
        );
        (s, s_correction) = kahan_add(
            s,
            s_correction,
            face.physical_length_km * (2.0 * angle).sin(),
        );
        any = true;
    }
    if !any || (c == 0.0 && s == 0.0) {
        return (None, None);
    }
    let angle = 0.5 * s.atan2(c);
    let mut axis = DVec3::new(angle.cos(), angle.sin(), 0.0);
    if axis.x < 0.0 || (axis.x == 0.0 && axis.y < 0.0) {
        axis = -axis;
    }
    let p = DVec2::new(principal_axis.x, principal_axis.y).normalize();
    let acute = DVec2::new(axis.x, axis.y)
        .dot(p)
        .abs()
        .clamp(0.0, 1.0)
        .acos();
    (Some(canonical_point(axis)), Some(canonical_zero(acute)))
}

fn saddle_relationships(
    graph: &EvaluationSurfaceGraphV0,
    hierarchy: &SurfaceHierarchyV0,
    scale: &super::DrainageScaleV0,
    faces: &[BackedBoundaryFaceV0],
    ancestry: &impl Fn(IncrementalCatchmentOwnerV0, IncrementalCatchmentOwnerV0) -> OwnerAncestryV0,
) -> Result<Vec<SaddleBoundaryAssociationV0>, RelationshipErrorV0> {
    let mut output = Vec::new();
    for saddle in &hierarchy.saddles {
        let elder = hierarchy
            .peaks
            .get(saddle.elder_peak as usize)
            .ok_or_else(|| {
                RelationshipErrorV0::InvalidSurfaceHierarchy("unknown elder peak".into())
            })?;
        let elder_owner = *scale
            .basin_graph
            .exclusive_owner
            .get(elder.anchor_cell as usize)
            .ok_or_else(|| {
                RelationshipErrorV0::InvalidSurfaceHierarchy("elder anchor out of range".into())
            })?;
        for &losing_id in &saddle.losing_peaks {
            let losing = hierarchy.peaks.get(losing_id as usize).ok_or_else(|| {
                RelationshipErrorV0::InvalidSurfaceHierarchy("unknown losing peak".into())
            })?;
            let losing_owner = scale.basin_graph.exclusive_owner[losing.anchor_cell as usize];
            let mut owners = [elder_owner, losing_owner];
            owners.sort();
            let nearest = faces
                .iter()
                .enumerate()
                .filter(|(_, face)| {
                    face.role == BoundaryFaceRoleKindV0::LateralBoundaryCandidate
                        && face.owners == owners
                })
                .map(|(index, face)| {
                    (
                        index,
                        point_segment_distance(saddle.flat_centroid_km, face.endpoints_km),
                        face,
                    )
                })
                .min_by(|a, b| {
                    a.1.total_cmp(&b.1)
                        .then_with(|| endpoints_cmp(a.2.endpoints_km, b.2.endpoints_km))
                });
            let record = if let Some((index, distance, face)) = nearest {
                SaddleBoundaryAssociationV0 {
                    saddle_id: saddle.id,
                    elder_peak_id: saddle.elder_peak,
                    losing_peak_id: losing_id,
                    owners,
                    owner_ancestry: ancestry(owners[0], owners[1]),
                    boundary_face_index: Some(index as u32),
                    separation_km: Some(canonical_zero(distance)),
                    effective_covering_radius_km: Some(face.covering_radius_km),
                    within_covering_radius: Some(distance <= face.covering_radius_km),
                    saddle_minus_face_height_km: Some(canonical_zero(
                        saddle.elevation_km - face.reconstructed_face_height_km,
                    )),
                    bilateral_descent: face.bilateral_descent.clone(),
                    equal_elder_ambiguous: saddle.equal_elder_ambiguous,
                }
            } else {
                SaddleBoundaryAssociationV0 {
                    saddle_id: saddle.id,
                    elder_peak_id: saddle.elder_peak,
                    losing_peak_id: losing_id,
                    owners,
                    owner_ancestry: ancestry(owners[0], owners[1]),
                    boundary_face_index: None,
                    separation_km: None,
                    effective_covering_radius_km: None,
                    within_covering_radius: None,
                    saddle_minus_face_height_km: None,
                    bilateral_descent: None,
                    equal_elder_ambiguous: saddle.equal_elder_ambiguous,
                }
            };
            output.push(record);
        }
    }
    output.sort_by_key(|record| {
        (
            record.saddle_id,
            record.elder_peak_id,
            record.losing_peak_id,
        )
    });
    let _ = graph;
    Ok(output)
}

fn point_segment_distance(point: DVec3, endpoints: [DVec3; 2]) -> f64 {
    let p = point.truncate();
    let a = endpoints[0].truncate();
    let b = endpoints[1].truncate();
    let d = b - a;
    let t = if d.length_squared() == 0.0 {
        0.0
    } else {
        ((p - a).dot(d) / d.length_squared()).clamp(0.0, 1.0)
    };
    p.distance(a + t * d)
}

#[allow(clippy::too_many_arguments)]
fn reach_probes(
    graph: &EvaluationSurfaceGraphV0,
    elevation: &[f64],
    drainage: &EvaluationDrainageV0,
    scale: &super::DrainageScaleV0,
    faces: &[BackedBoundaryFaceV0],
    reach_parent: &[Option<u32>],
    spatial_index: &RelationshipSpatialIndex,
    config: LandformRelationshipConfigV0,
) -> Result<(Vec<ReachCrossSectionProbeV0>, u64), RelationshipErrorV0> {
    let mut output = Vec::with_capacity(scale.reach_graph.reaches.len());
    let mut candidate_face_tests = 0u64;
    for reach in &scale.reach_graph.reaches {
        let mut original: Vec<DVec3> = reach
            .cells
            .iter()
            .map(|&cell| graph.cell_center_km[cell as usize])
            .collect();
        let tail_cell = *reach.cells.last().unwrap() as usize;
        let target = match drainage.routing.receiver[tail_cell] {
            DrainageReceiverV0::Cell { cell, .. } => graph.cell_center_km[cell as usize],
            DrainageReceiverV0::Portal {
                boundary_segment, ..
            } => {
                let segment = graph
                    .boundary_segments
                    .get(boundary_segment as usize)
                    .ok_or(RelationshipErrorV0::InconsistentReceiverTarget { cell: tail_cell })?;
                0.5 * (segment.endpoints_km[0] + segment.endpoints_km[1])
            }
        };
        original.push(target);
        let original_lengths = cumulative_lengths(&original);
        let total = *original_lengths.last().unwrap_or(&0.0);
        if !total.is_finite() || total <= 0.0 {
            return Err(RelationshipErrorV0::DegenerateReachPolyline { reach_id: reach.id });
        }
        let source_offset = (total - config.maximum_downstream_support_km).max(0.0);
        let source_truncated = source_offset > 0.0;
        let retained_length = total - source_offset;
        let retained = truncate_source(&original, &original_lengths, source_offset);
        let longitudinal_samples = longitudinal_samples(
            graph,
            elevation,
            drainage,
            reach,
            &original,
            &original_lengths,
            source_offset,
            retained_length,
            spatial_index,
        )?;
        let mut stations = Vec::new();
        let mut station_s = 0.5 * config.station_spacing_km;
        while station_s < retained_length {
            let absolute_s = source_offset + station_s;
            let axis = arclength_point(&original, &original_lengths, absolute_s).0;
            let before = arclength_point(
                &original,
                &original_lengths,
                (absolute_s - TANGENT_HALF_SUPPORT_KM).max(source_offset),
            )
            .0;
            let after = arclength_point(
                &original,
                &original_lengths,
                (absolute_s + TANGENT_HALF_SUPPORT_KM).min(total),
            )
            .0;
            let chord = (after - before).truncate();
            if !chord.is_finite() || chord.length_squared() == 0.0 {
                return Err(RelationshipErrorV0::InvalidCrossSectionTangent { reach_id: reach.id });
            }
            let tangent2 = chord.normalize();
            let tangent = DVec3::new(tangent2.x, tangent2.y, 0.0);
            let left_normal = DVec3::new(-tangent.y, tangent.x, 0.0);
            let axis_cell = spatial_index.locate_cell(graph, axis);
            let axis_inside = axis_cell.is_some_and(|cell| {
                nested_owner(
                    scale.basin_graph.exclusive_owner[cell],
                    reach.id,
                    reach_parent,
                )
            });
            let left = section_side(
                graph,
                elevation,
                scale,
                faces,
                spatial_index,
                reach.id,
                reach_parent,
                axis,
                left_normal,
                axis_cell,
                axis_inside,
                1.0,
                config,
                &mut candidate_face_tests,
            );
            let right = section_side(
                graph,
                elevation,
                scale,
                faces,
                spatial_index,
                reach.id,
                reach_parent,
                axis,
                -left_normal,
                axis_cell,
                axis_inside,
                -1.0,
                config,
                &mut candidate_face_tests,
            );
            let span = match (
                left.relative_height_crossing_km,
                right.relative_height_crossing_km,
            ) {
                (Some(a), Some(b)) => Some(canonical_zero(a + b)),
                _ => None,
            };
            let reason = if span.is_some() {
                None
            } else if let Some(reason) = left.censor_reason.or(right.censor_reason) {
                Some(RelativeReliefSpanUnavailableV0::Censored(reason))
            } else if left.positive_boundary_relief == Some(false)
                || right.positive_boundary_relief == Some(false)
            {
                Some(RelativeReliefSpanUnavailableV0::NonPositiveBoundaryRelief)
            } else {
                Some(RelativeReliefSpanUnavailableV0::MissingRelativeHeightCrossing)
            };
            stations.push(ReachCrossSectionStationV0 {
                station_arclength_km: canonical_zero(station_s),
                axis_point_km: canonical_point(axis),
                tangent: canonical_point(tangent),
                left,
                right,
                relative_relief_span_km: span,
                span_unavailable_reason: reason,
            });
            station_s += config.station_spacing_km;
        }
        output.push(ReachCrossSectionProbeV0 {
            reach_id: reach.id,
            retained_length_km: canonical_zero(retained_length),
            source_truncated,
            retained_polyline_km: retained.into_iter().map(canonical_point).collect(),
            longitudinal_samples,
            stations,
        });
    }
    output.sort_by_key(|probe| probe.reach_id);
    Ok((output, candidate_face_tests))
}

fn cumulative_lengths(points: &[DVec3]) -> Vec<f64> {
    let mut output = Vec::with_capacity(points.len());
    output.push(0.0);
    let mut sum = 0.0;
    let mut correction = 0.0;
    for pair in points.windows(2) {
        (sum, correction) = kahan_add(sum, correction, pair[0].distance(pair[1]));
        output.push(canonical_zero(sum));
    }
    output
}

fn arclength_point(points: &[DVec3], lengths: &[f64], s: f64) -> (DVec3, usize) {
    let index = lengths
        .partition_point(|&value| value <= s)
        .saturating_sub(1)
        .min(points.len() - 2);
    let span = lengths[index + 1] - lengths[index];
    let t = if span == 0.0 {
        0.0
    } else {
        ((s - lengths[index]) / span).clamp(0.0, 1.0)
    };
    (points[index].lerp(points[index + 1], t), index)
}

fn truncate_source(points: &[DVec3], lengths: &[f64], source: f64) -> Vec<DVec3> {
    if source == 0.0 {
        return points.to_vec();
    }
    let (first, segment) = arclength_point(points, lengths, source);
    let mut output = vec![first];
    output.extend_from_slice(&points[segment + 1..]);
    output
}

#[allow(clippy::too_many_arguments)]
fn longitudinal_samples(
    graph: &EvaluationSurfaceGraphV0,
    elevation: &[f64],
    drainage: &EvaluationDrainageV0,
    reach: &super::RiverReachV0,
    points: &[DVec3],
    lengths: &[f64],
    source_offset: f64,
    retained_length: f64,
    spatial_index: &RelationshipSpatialIndex,
) -> Result<Vec<LongitudinalReachSampleV0>, RelationshipErrorV0> {
    let mut lattice = Vec::new();
    let mut s = 0.0;
    while s < retained_length {
        lattice.push(s);
        s += LONGITUDINAL_STEP_KM;
    }
    if lattice.last().copied() != Some(retained_length) {
        lattice.push(retained_length);
    }
    let mut output = Vec::with_capacity(lattice.len());
    for (sample_index, &local_s) in lattice.iter().enumerate() {
        let absolute = source_offset + local_s;
        let (point, segment) = arclength_point(points, lengths, absolute);
        let donor = reach.cells[segment.min(reach.cells.len() - 1)] as usize;
        let containing = spatial_index
            .locate_cell(graph, point)
            .ok_or(RelationshipErrorV0::PointLocationFailure { reach_id: reach.id })?;
        let (grade, fill, flat, non_descending) = if sample_index == 0 {
            (None, false, false, false)
        } else {
            let previous_s = source_offset + lattice[sample_index - 1];
            let mut fill = false;
            let mut flat = false;
            let mut non_descending = false;
            let interval_end = absolute;
            for seg in 0..reach.cells.len() {
                if !positive_segment_overlap(
                    lengths[seg],
                    lengths[seg + 1],
                    previous_s,
                    interval_end,
                ) {
                    continue;
                }
                let cell = reach.cells[seg] as usize;
                fill |= drainage.routing.fill_supported[cell];
                flat |= drainage.routing.flat_supported[cell];
                non_descending |= drainage.routing.physically_non_descending[cell];
            }
            let ds = local_s - lattice[sample_index - 1];
            let previous_z = output
                .last()
                .map(|record: &LongitudinalReachSampleV0| record.physical_elevation_km)
                .unwrap();
            (
                (ds > 0.0).then_some(canonical_zero((previous_z - elevation[containing]) / ds)),
                fill,
                flat,
                non_descending,
            )
        };
        output.push(LongitudinalReachSampleV0 {
            arclength_km: canonical_zero(local_s),
            point_km: canonical_point(point),
            physical_elevation_km: elevation[containing],
            donor_cell: donor as u32,
            structural_area_km2: drainage.routing.structural_area_km2[donor],
            supplied_runoff: drainage.routing.supplied_runoff[donor],
            interval_grade: grade,
            interval_fill_supported: fill,
            interval_flat_supported: flat,
            interval_physically_non_descending: non_descending,
        });
    }
    Ok(output)
}

fn positive_segment_overlap(
    segment_start: f64,
    segment_end: f64,
    interval_start: f64,
    interval_end: f64,
) -> bool {
    segment_start.max(interval_start) < segment_end.min(interval_end)
}

fn nested_owner(
    owner: IncrementalCatchmentOwnerV0,
    station_reach: u32,
    parents: &[Option<u32>],
) -> bool {
    let IncrementalCatchmentOwnerV0::Reach(mut reach) = owner else {
        return false;
    };
    loop {
        if reach == station_reach {
            return true;
        }
        let Some(parent) = parents.get(reach as usize).copied().flatten() else {
            return false;
        };
        reach = parent;
    }
}

#[derive(Clone, Copy)]
enum SectionRayEventKind {
    Lateral(usize),
    Censor(SectionCensorReasonV0),
}

#[derive(Clone, Copy)]
struct SectionRayEvent {
    distance: f64,
    physical_index: usize,
    kind: SectionRayEventKind,
}

#[allow(clippy::too_many_arguments)]
fn section_side(
    graph: &EvaluationSurfaceGraphV0,
    elevation: &[f64],
    scale: &super::DrainageScaleV0,
    faces: &[BackedBoundaryFaceV0],
    spatial_index: &RelationshipSpatialIndex,
    reach_id: u32,
    parents: &[Option<u32>],
    axis: DVec3,
    normal: DVec3,
    axis_cell: Option<usize>,
    axis_inside: bool,
    sign: f64,
    config: LandformRelationshipConfigV0,
    candidate_face_tests: &mut u64,
) -> CrossSectionSideV0 {
    let support = config.cross_section_half_length_km;
    let ray_endpoint = axis + support * normal;
    let candidates = spatial_index.ray_face_candidates(axis, ray_endpoint);
    let mut events = Vec::<SectionRayEvent>::new();
    for physical_index in candidates {
        *candidate_face_tests += 1;
        let physical = spatial_index.physical_faces[physical_index];
        if let IndexedPhysicalFaceKind::Internal {
            cells,
            backed_boundary: Some(face_index),
        } = physical.kind
        {
            let inside = cells.map(|cell| {
                nested_owner(
                    scale.basin_graph.exclusive_owner[cell as usize],
                    reach_id,
                    parents,
                )
            });
            if inside[0] != inside[1]
                && faces[face_index as usize].role
                    == BoundaryFaceRoleKindV0::LateralBoundaryCandidate
                && point_on_segment_tol(
                    axis.truncate(),
                    physical.endpoints_km[0].truncate(),
                    physical.endpoints_km[1].truncate(),
                    0.0,
                )
            {
                return censored_side(SectionCensorReasonV0::AxisOnBoundary);
            }
        }
        let intersection = ray_segment_intersection(axis, normal, physical.endpoints_km);
        let Some((distance, collinear)) = supported_ray_intersection(intersection, support) else {
            continue;
        };
        let kind = match physical.kind {
            IndexedPhysicalFaceKind::DomainBoundary { segment } => {
                let _ = segment;
                SectionRayEventKind::Censor(if collinear {
                    SectionCensorReasonV0::CollinearBoundary
                } else {
                    SectionCensorReasonV0::DomainBoundary
                })
            }
            IndexedPhysicalFaceKind::Internal {
                cells,
                backed_boundary,
            } => {
                let inside = cells.map(|cell| {
                    nested_owner(
                        scale.basin_graph.exclusive_owner[cell as usize],
                        reach_id,
                        parents,
                    )
                });
                if inside[0] == inside[1] {
                    continue;
                }
                let Some(face_index) = backed_boundary else {
                    events.push(SectionRayEvent {
                        distance,
                        physical_index,
                        kind: SectionRayEventKind::Censor(
                            SectionCensorReasonV0::AmbiguousFaceGeometry,
                        ),
                    });
                    continue;
                };
                let face = &faces[face_index as usize];
                if collinear {
                    SectionRayEventKind::Censor(SectionCensorReasonV0::CollinearBoundary)
                } else if face.role == BoundaryFaceRoleKindV0::FlowTransition {
                    SectionRayEventKind::Censor(SectionCensorReasonV0::FlowTransition)
                } else {
                    SectionRayEventKind::Lateral(face_index as usize)
                }
            }
        };
        events.push(SectionRayEvent {
            distance,
            physical_index,
            kind,
        });
    }
    events.sort_by(|a, b| {
        a.distance
            .total_cmp(&b.distance)
            .then_with(|| a.physical_index.cmp(&b.physical_index))
    });
    if events
        .iter()
        .take_while(|event| event.distance == 0.0)
        .any(|event| {
            matches!(event.kind, SectionRayEventKind::Lateral(_))
                || matches!(
                    event.kind,
                    SectionRayEventKind::Censor(SectionCensorReasonV0::CollinearBoundary)
                )
        })
    {
        return censored_side(SectionCensorReasonV0::AxisOnBoundary);
    }
    if !axis_inside {
        return censored_side(SectionCensorReasonV0::AxisOutsideCatchment);
    }
    let axis_z = elevation[axis_cell.unwrap()];
    if let Some(first) = events.first() {
        if events
            .iter()
            .take_while(|event| event.distance == first.distance)
            .any(|event| {
                matches!(
                    event.kind,
                    SectionRayEventKind::Censor(SectionCensorReasonV0::CollinearBoundary)
                )
            })
        {
            return censored_side_with_samples(
                SectionCensorReasonV0::CollinearBoundary,
                graph,
                elevation,
                spatial_index,
                axis,
                normal,
                sign,
                config,
            );
        }
    }
    if events.len() >= 2 && events[0].distance == events[1].distance {
        return censored_side_with_samples(
            SectionCensorReasonV0::AmbiguousFaceGeometry,
            graph,
            elevation,
            spatial_index,
            axis,
            normal,
            sign,
            config,
        );
    }
    let Some(event) = events.first() else {
        let mut result = censored_side(SectionCensorReasonV0::NoCatchmentExitWithinSupport);
        result.samples = regular_side_samples(
            graph,
            elevation,
            spatial_index,
            axis,
            normal,
            sign,
            config.cross_section_half_length_km,
            config.cross_section_sample_step_km,
        );
        return result;
    };
    let (boundary_distance, face_index) = match event.kind {
        SectionRayEventKind::Censor(reason) => {
            return censored_side_with_samples(
                reason,
                graph,
                elevation,
                spatial_index,
                axis,
                normal,
                sign,
                config,
            )
        }
        SectionRayEventKind::Lateral(face_index) => (event.distance, face_index),
    };
    let face = &faces[face_index];
    let mut samples = regular_side_samples(
        graph,
        elevation,
        spatial_index,
        axis,
        normal,
        sign,
        config.cross_section_half_length_km,
        config.cross_section_sample_step_km,
    );
    let regular_heights: Vec<(f64, f64)> = samples
        .iter()
        .filter(|sample| {
            sample.height_provenance == Some(CrossSectionHeightProvenanceV0::CellMean)
                && sample.signed_offset_km.abs() < boundary_distance
        })
        .filter_map(|sample| {
            sample
                .physical_elevation_km
                .map(|z| (sample.signed_offset_km.abs(), z))
        })
        .collect();
    samples.push(CrossSectionSampleV0 {
        signed_offset_km: canonical_zero(sign * boundary_distance),
        point_km: canonical_point(axis + boundary_distance * normal),
        physical_elevation_km: Some(face.reconstructed_face_height_km),
        height_provenance: Some(CrossSectionHeightProvenanceV0::BoundaryHeightProxy),
        outside_domain: false,
    });
    let relief = canonical_zero(face.reconstructed_face_height_km - axis_z);
    let maximum = regular_heights
        .iter()
        .map(|(_, z)| *z - axis_z)
        .max_by(f64::total_cmp)
        .unwrap_or(relief);
    let minimum_offset = regular_heights
        .iter()
        .min_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.total_cmp(&b.0)))
        .map(|(offset, _)| *offset);
    let threshold = axis_z + config.relative_height_fraction * relief;
    let (crossing, bracket) = if relief > 0.0 {
        let mut crossing_samples = regular_heights.clone();
        crossing_samples.push((boundary_distance, face.reconstructed_face_height_km));
        first_crossing(&crossing_samples, threshold, boundary_distance)
    } else {
        (None, None)
    };
    CrossSectionSideV0 {
        censor_reason: None,
        boundary_face_index: Some(face_index as u32),
        boundary_offset_km: Some(canonical_zero(boundary_distance)),
        boundary_height_proxy_km: Some(face.reconstructed_face_height_km),
        samples,
        boundary_relief_km: Some(relief),
        maximum_sampled_relief_km: Some(canonical_zero(maximum)),
        boundary_maximum_separation_km: Some(canonical_zero(maximum - relief)),
        minimum_elevation_offset_km: minimum_offset.map(canonical_zero),
        positive_boundary_relief: Some(relief > 0.0),
        relative_height_crossing_km: crossing,
        crossing_bracket: bracket,
    }
}

#[allow(clippy::too_many_arguments)]
fn regular_side_samples(
    graph: &EvaluationSurfaceGraphV0,
    elevation: &[f64],
    spatial_index: &RelationshipSpatialIndex,
    axis: DVec3,
    normal: DVec3,
    sign: f64,
    support: f64,
    step: f64,
) -> Vec<CrossSectionSampleV0> {
    let mut output = Vec::new();
    let mut distance = 0.0;
    while distance <= support {
        let point = axis + distance * normal;
        let cell = spatial_index.locate_cell(graph, point);
        output.push(CrossSectionSampleV0 {
            signed_offset_km: canonical_zero(sign * distance),
            point_km: canonical_point(point),
            physical_elevation_km: cell.map(|cell| elevation[cell]),
            height_provenance: cell.map(|_| CrossSectionHeightProvenanceV0::CellMean),
            outside_domain: cell.is_none(),
        });
        distance += step;
    }
    output
}

fn first_crossing(
    samples: &[(f64, f64)],
    threshold: f64,
    boundary_distance: f64,
) -> (Option<f64>, Option<RelativeHeightBracketV0>) {
    for pair in samples.windows(2) {
        let (x0, z0) = pair[0];
        let (x1, z1) = pair[1];
        if z0 < threshold && z1 >= threshold && z1 != z0 {
            let crossing = x0 + (threshold - z0) * (x1 - x0) / (z1 - z0);
            let bracket = if x1 == boundary_distance {
                RelativeHeightBracketV0::CellMeanToBoundaryProxy
            } else {
                RelativeHeightBracketV0::CellMeans
            };
            return (Some(canonical_zero(crossing)), Some(bracket));
        }
    }
    (None, None)
}

fn censored_side(reason: SectionCensorReasonV0) -> CrossSectionSideV0 {
    CrossSectionSideV0 {
        censor_reason: Some(reason),
        boundary_face_index: None,
        boundary_offset_km: None,
        boundary_height_proxy_km: None,
        samples: Vec::new(),
        boundary_relief_km: None,
        maximum_sampled_relief_km: None,
        boundary_maximum_separation_km: None,
        minimum_elevation_offset_km: None,
        positive_boundary_relief: None,
        relative_height_crossing_km: None,
        crossing_bracket: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn censored_side_with_samples(
    reason: SectionCensorReasonV0,
    graph: &EvaluationSurfaceGraphV0,
    elevation: &[f64],
    spatial_index: &RelationshipSpatialIndex,
    axis: DVec3,
    normal: DVec3,
    sign: f64,
    config: LandformRelationshipConfigV0,
) -> CrossSectionSideV0 {
    let mut result = censored_side(reason);
    result.samples = regular_side_samples(
        graph,
        elevation,
        spatial_index,
        axis,
        normal,
        sign,
        config.cross_section_half_length_km,
        config.cross_section_sample_step_km,
    );
    result
}

enum RayIntersection {
    None,
    Point(f64),
    Collinear { start: f64, end: f64 },
}

fn supported_ray_intersection(intersection: RayIntersection, support: f64) -> Option<(f64, bool)> {
    match intersection {
        RayIntersection::Point(distance) if distance <= support => Some((distance, false)),
        RayIntersection::Collinear { start, end } => {
            let overlap_start = start.max(0.0);
            let overlap_end = end.min(support);
            if overlap_start < overlap_end {
                Some((canonical_zero(overlap_start), true))
            } else if overlap_start == overlap_end && end >= 0.0 && start <= support {
                Some((canonical_zero(overlap_start), false))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn ray_segment_intersection(axis: DVec3, normal: DVec3, endpoints: [DVec3; 2]) -> RayIntersection {
    let p = axis.truncate();
    let r = normal.truncate();
    let q = endpoints[0].truncate();
    let s = (endpoints[1] - endpoints[0]).truncate();
    let cross = |a: DVec2, b: DVec2| a.x * b.y - a.y * b.x;
    let denominator = cross(r, s);
    let qp = q - p;
    if denominator == 0.0 {
        return if cross(qp, r) == 0.0 {
            let first = (q - p).dot(r);
            let second = (q + s - p).dot(r);
            RayIntersection::Collinear {
                start: canonical_zero(first.min(second)),
                end: canonical_zero(first.max(second)),
            }
        } else {
            RayIntersection::None
        };
    }
    let t = cross(qp, s) / denominator;
    let u = cross(qp, r) / denominator;
    if t >= 0.0 && (0.0..=1.0).contains(&u) {
        RayIntersection::Point(canonical_zero(t))
    } else {
        RayIntersection::None
    }
}

fn polygon_contains(polygon: &[DVec3], point: DVec2) -> bool {
    let mut inside = false;
    for index in 0..polygon.len() {
        let a = polygon[index].truncate();
        let b = polygon[(index + 1) % polygon.len()].truncate();
        let edge = b - a;
        let offset = point - a;
        let cross = edge.x * offset.y - edge.y * offset.x;
        if cross == 0.0 && offset.dot(edge) >= 0.0 && offset.dot(edge) <= edge.length_squared() {
            return true;
        }
        if (a.y > point.y) != (b.y > point.y) {
            let x = a.x + (point.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if x == point.x {
                return true;
            }
            if x > point.x {
                inside = !inside;
            }
        }
    }
    inside
}

fn reject_nonfinite_result(result: &LandformRelationshipsV0) -> Result<(), RelationshipErrorV0> {
    let finite_face = result.backed_boundary_faces.iter().all(|face| {
        face.endpoints_km.iter().all(|point| point.is_finite())
            && face.physical_length_km.is_finite()
            && face.reconstructed_face_height_km.is_finite()
            && face.covering_radius_km.is_finite()
            && face.bilateral_descent.as_ref().is_none_or(|probe| {
                probe.sides.iter().all(|side| {
                    side.adjacent_elevation_km.is_finite()
                        && side.target_elevation_km.is_finite()
                        && side.target_drop_km.is_finite()
                        && side.minimum_segment_drop_km.is_none_or(f64::is_finite)
                        && side.remote_maximum_excess_km.is_finite()
                        && side.receiver_length_km.is_finite()
                        && side.endpoint_distance_km.is_finite()
                        && side.tortuosity.is_none_or(f64::is_finite)
                })
            })
    });
    let finite_highlands = result.highland_boundary_relationships.iter().all(|record| {
        record.candidate_length_km.is_finite()
            && record.unconditioned_bilateral_descent_length_km.is_finite()
            && record.unconditioned_length_ratio.is_none_or(f64::is_finite)
            && record
                .fill_supported_length_share
                .is_none_or(f64::is_finite)
            && record
                .flat_supported_length_share
                .is_none_or(f64::is_finite)
            && record
                .physical_non_descent_length_share
                .is_none_or(f64::is_finite)
            && record
                .boundary_axial_orientation
                .is_none_or(|point| point.is_finite())
            && record.acute_axis_difference_rad.is_none_or(f64::is_finite)
    });
    let finite_saddles = result.saddle_boundary_associations.iter().all(|record| {
        record.separation_km.is_none_or(f64::is_finite)
            && record
                .effective_covering_radius_km
                .is_none_or(f64::is_finite)
            && record
                .saddle_minus_face_height_km
                .is_none_or(f64::is_finite)
    });
    let finite_reaches = result.reach_cross_section_probes.iter().all(|probe| {
        probe.retained_length_km.is_finite()
            && probe
                .retained_polyline_km
                .iter()
                .all(|point| point.is_finite())
            && probe.longitudinal_samples.iter().all(|sample| {
                sample.arclength_km.is_finite()
                    && sample.point_km.is_finite()
                    && sample.physical_elevation_km.is_finite()
                    && sample.structural_area_km2.is_finite()
                    && sample.supplied_runoff.is_finite()
                    && sample.interval_grade.is_none_or(f64::is_finite)
            })
            && probe.stations.iter().all(|station| {
                station.station_arclength_km.is_finite()
                    && station.axis_point_km.is_finite()
                    && station.tangent.is_finite()
                    && station.relative_relief_span_km.is_none_or(f64::is_finite)
                    && finite_section_side(&station.left)
                    && finite_section_side(&station.right)
            })
    });
    if finite_face && finite_highlands && finite_saddles && finite_reaches {
        Ok(())
    } else {
        Err(RelationshipErrorV0::NonFiniteDerivedEvidence)
    }
}

fn finite_section_side(side: &CrossSectionSideV0) -> bool {
    side.boundary_offset_km.is_none_or(f64::is_finite)
        && side.boundary_height_proxy_km.is_none_or(f64::is_finite)
        && side.boundary_relief_km.is_none_or(f64::is_finite)
        && side.maximum_sampled_relief_km.is_none_or(f64::is_finite)
        && side
            .boundary_maximum_separation_km
            .is_none_or(f64::is_finite)
        && side.minimum_elevation_offset_km.is_none_or(f64::is_finite)
        && side.relative_height_crossing_km.is_none_or(f64::is_finite)
        && side.samples.iter().all(|sample| {
            sample.signed_offset_km.is_finite()
                && sample.point_km.is_finite()
                && sample.physical_elevation_km.is_none_or(f64::is_finite)
        })
}

#[allow(clippy::too_many_arguments)]
fn relationship_hash(
    graph: &EvaluationSurfaceGraphV0,
    elevation: &[f64],
    scored_cell: &[bool],
    runoff: &[f64],
    surface_config: &SurfaceHierarchyConfigV0,
    drainage_config: &DrainageConfigV0,
    hierarchy: &SurfaceHierarchyV0,
    drainage: &EvaluationDrainageV0,
    result: &LandformRelationshipsV0,
) -> Result<u64, RelationshipErrorV0> {
    let inputs = fixed_bytes(&(
        graph,
        elevation,
        scored_cell,
        runoff,
        surface_config,
        drainage_config,
        hierarchy,
        drainage,
    ))?;
    let payload = fixed_bytes(&(
        &result.schema_version,
        &result.hash_version,
        &result.geometry_identity,
        &result.config,
        &result.run_namespace,
        result.surface_hierarchy_input_hash,
        result.drainage_input_hash,
        &result.backed_boundary_faces,
        &result.highland_boundary_relationships,
        &result.saddle_boundary_associations,
        &result.reach_cross_section_probes,
        &result.work_counts,
    ))?;
    let mut bytes = inputs;
    bytes.extend(payload);
    Ok(fnv1a64(&bytes))
}

fn fixed_bytes(value: &impl Serialize) -> Result<Vec<u8>, RelationshipErrorV0> {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .serialize(value)
        .map_err(|error| RelationshipErrorV0::Serialization(error.to_string()))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn kahan_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for value in values {
        let adjusted = value - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    canonical_zero(sum)
}

fn kahan_add(sum: f64, correction: f64, value: f64) -> (f64, f64) {
    let adjusted = value - correction;
    let next = sum + adjusted;
    (next, (next - sum) - adjusted)
}

fn ratio(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator > 0.0).then_some(canonical_zero(numerator / denominator))
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn canonical_point(point: DVec3) -> DVec3 {
    DVec3::new(
        canonical_zero(point.x),
        canonical_zero(point.y),
        canonical_zero(point.z),
    )
}

fn point_cmp(a: DVec3, b: DVec3) -> Ordering {
    a.x.total_cmp(&b.x)
        .then_with(|| a.y.total_cmp(&b.y))
        .then_with(|| a.z.total_cmp(&b.z))
}

fn endpoints_cmp(a: [DVec3; 2], b: [DVec3; 2]) -> Ordering {
    point_cmp(a[0], b[0]).then_with(|| point_cmp(a[1], b[1]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::landforms::{adapt_landscape_graph_v0, build_regular_hex_control_volumes_v0};
    use crate::world::landscape::LandscapeMesh;

    fn regular_graph(spacing: f64) -> EvaluationSurfaceGraphV0 {
        let config = SurfaceHierarchyConfigV0::default();
        let mesh = LandscapeMesh::uniform_planar_hex(48.0, 40.0, spacing).unwrap();
        let controls = build_regular_hex_control_volumes_v0(&mesh, &config).unwrap();
        adapt_landscape_graph_v0(&mesh, &controls, &config).unwrap()
    }

    #[test]
    fn collinear_contact_requires_positive_forward_support() {
        let axis = DVec3::ZERO;
        let normal = DVec3::X;
        let behind = ray_segment_intersection(
            axis,
            normal,
            [DVec3::new(-4.0, 0.0, 0.0), DVec3::new(-2.0, 0.0, 0.0)],
        );
        assert!(supported_ray_intersection(behind, 10.0).is_none());

        let forward = ray_segment_intersection(
            axis,
            normal,
            [DVec3::new(2.0, 0.0, 0.0), DVec3::new(4.0, 0.0, 0.0)],
        );
        assert_eq!(supported_ray_intersection(forward, 10.0), Some((2.0, true)));

        let support_endpoint = ray_segment_intersection(
            axis,
            normal,
            [DVec3::new(10.0, 0.0, 0.0), DVec3::new(12.0, 0.0, 0.0)],
        );
        assert_eq!(
            supported_ray_intersection(support_endpoint, 10.0),
            Some((10.0, false))
        );

        let axis_endpoint =
            ray_segment_intersection(axis, normal, [DVec3::new(-2.0, 0.0, 0.0), DVec3::ZERO]);
        assert_eq!(
            supported_ray_intersection(axis_endpoint, 10.0),
            Some((0.0, false))
        );
    }

    #[test]
    fn longitudinal_endpoint_contact_has_no_positive_segment_overlap() {
        assert!(positive_segment_overlap(0.0, 4.0, 0.0, 4.0));
        assert!(!positive_segment_overlap(4.0, 8.0, 0.0, 4.0));
        assert!(!positive_segment_overlap(0.0, 4.0, 4.0, 8.0));
        assert!(positive_segment_overlap(4.0, 8.0, 4.0, 8.0));
    }

    #[test]
    fn regular_identity_rejects_a_false_spacing_label() {
        let graph = regular_graph(4.0);
        let hash = relationship_graph_hash_v0(&graph).unwrap();
        let config = SurfaceHierarchyConfigV0::default();
        validate_geometry_identity(
            &graph,
            PacketGeometryIdentityV0::LandscapeRegularPlanar {
                nominal_spacing_km: 4.0,
                canonical_graph_hash: hash,
            },
            config.endpoint_match_abs_km,
            config.planar_area_match_relative,
        )
        .unwrap();
        assert_eq!(
            validate_geometry_identity(
                &graph,
                PacketGeometryIdentityV0::LandscapeRegularPlanar {
                    nominal_spacing_km: 4.1,
                    canonical_graph_hash: hash,
                },
                config.endpoint_match_abs_km,
                config.planar_area_match_relative,
            ),
            Err(RelationshipErrorV0::InvalidGeometryIdentity)
        );
    }

    #[test]
    fn bbox_index_matches_exhaustive_point_and_ray_oracles() {
        let graph = regular_graph(4.0);
        let index = RelationshipSpatialIndex::build(&graph, &[]).unwrap();
        for &point in graph.cell_center_km.iter().step_by(3) {
            let exhaustive = (0..graph.cell_count())
                .filter(|&cell| polygon_contains(graph.polygon(cell), point.truncate()))
                .min_by(|&a, &b| point_cmp(graph.cell_center_km[a], graph.cell_center_km[b]));
            assert_eq!(index.locate_cell(&graph, point), exhaustive);
        }

        let normal = DVec3::new(0.5, 3.0_f64.sqrt() * 0.5, 0.0);
        let support = 24.0;
        for &axis in graph.cell_center_km.iter().step_by(7) {
            let indexed_candidates = index.ray_face_candidates(axis, axis + support * normal);
            let indexed_hits: BTreeSet<_> = indexed_candidates
                .into_iter()
                .filter(|&face| {
                    supported_ray_intersection(
                        ray_segment_intersection(
                            axis,
                            normal,
                            index.physical_faces[face].endpoints_km,
                        ),
                        support,
                    )
                    .is_some()
                })
                .collect();
            let exhaustive_hits: BTreeSet<_> = index
                .physical_faces
                .iter()
                .enumerate()
                .filter_map(|(face, record)| {
                    supported_ray_intersection(
                        ray_segment_intersection(axis, normal, record.endpoints_km),
                        support,
                    )
                    .is_some()
                    .then_some(face)
                })
                .collect();
            assert_eq!(indexed_hits, exhaustive_hits);
        }
    }

    fn square(center: DVec3, half: f64) -> Vec<DVec3> {
        vec![
            DVec3::new(center.x - half, center.y - half, 0.0),
            DVec3::new(center.x + half, center.y - half, 0.0),
            DVec3::new(center.x + half, center.y + half, 0.0),
            DVec3::new(center.x - half, center.y + half, 0.0),
        ]
    }

    fn kernel_graph(
        centers: Vec<DVec3>,
        polygons: Vec<Vec<DVec3>>,
        edge_offsets: Vec<u32>,
        edge_neighbor: Vec<u32>,
        edge_reciprocal: Vec<u32>,
        endpoints: Vec<[DVec3; 2]>,
        widths: Vec<f64>,
    ) -> EvaluationSurfaceGraphV0 {
        let mut polygon_offsets = Vec::with_capacity(polygons.len() + 1);
        let mut vertices = Vec::new();
        for polygon in polygons {
            polygon_offsets.push(vertices.len() as u32);
            vertices.extend(polygon);
        }
        polygon_offsets.push(vertices.len() as u32);
        let edge_distance_km = edge_neighbor
            .iter()
            .enumerate()
            .map(|(edge, &neighbor)| {
                let cell = edge_offsets
                    .partition_point(|&offset| offset as usize <= edge)
                    .saturating_sub(1)
                    .min(centers.len() - 1);
                centers[cell].distance(centers[neighbor as usize])
            })
            .collect();
        EvaluationSurfaceGraphV0 {
            domain: EvaluationDomainV0::Planar,
            cell_area_km2: vec![1.0; centers.len()],
            cell_center_km: centers,
            cell_polygon_offsets: polygon_offsets,
            cell_polygon_vertices_km: vertices,
            edge_offsets,
            edge_neighbor,
            edge_reciprocal,
            edge_distance_km,
            edge_shared_width_km: widths,
            edge_face_endpoints_km: endpoints,
            boundary_segments: Vec::new(),
        }
    }

    #[test]
    fn frozen_three_face_table_separates_role_from_ancestry() {
        let centers = vec![
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(3.0, 0.0, 0.0),
            DVec3::new(4.0, 0.0, 0.0),
        ];
        let polygons = centers.iter().map(|&center| square(center, 0.4)).collect();
        let f0 = [DVec3::new(-0.5, -0.5, 0.0), DVec3::new(-0.5, 0.5, 0.0)];
        let f1 = [DVec3::new(2.5, -1.0, 0.0), DVec3::new(2.5, 1.0, 0.0)];
        let f2 = [DVec3::new(3.5, -1.5, 0.0), DVec3::new(3.5, 1.5, 0.0)];
        let reverse = |face: [DVec3; 2]| [face[1], face[0]];
        let graph = kernel_graph(
            centers,
            polygons,
            vec![0, 1, 2, 3, 5, 6],
            vec![1, 0, 3, 2, 4, 3],
            vec![1, 0, 3, 2, 5, 4],
            vec![f0, reverse(f0), f1, reverse(f1), f2, reverse(f2)],
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        );
        let owners = vec![
            IncrementalCatchmentOwnerV0::Reach(0),
            IncrementalCatchmentOwnerV0::Reach(1),
            IncrementalCatchmentOwnerV0::Reach(0),
            IncrementalCatchmentOwnerV0::Reach(1),
            IncrementalCatchmentOwnerV0::Reach(2),
        ];
        let mut receivers = vec![
            DrainageReceiverV0::Portal {
                boundary_segment: 0,
                portal_id: 9,
            };
            5
        ];
        receivers[1] = DrainageReceiverV0::Cell {
            cell: 0,
            directed_edge: 1,
        };
        let mut raw = vec![
            RawCatchmentBoundaryFaceV0 {
                owners: [owners[0], owners[1]],
                endpoints_km: f0,
                physical_length_km: 1.0,
            },
            RawCatchmentBoundaryFaceV0 {
                owners: [owners[2], owners[3]],
                endpoints_km: f1,
                physical_length_km: 2.0,
            },
            RawCatchmentBoundaryFaceV0 {
                owners: [owners[3], owners[4]],
                endpoints_km: f2,
                physical_length_km: 3.0,
            },
        ];
        raw.sort_by(raw_face_cmp);
        let parents = [None, Some(0), None];
        let outlets = [9, 9, 10];
        let faces = back_boundary_faces(&graph, &[0.0; 5], &receivers, &owners, &raw, &|a, b| {
            owner_ancestry(a, b, &parents, &outlets)
        })
        .unwrap();
        assert_eq!(faces.len(), 3);
        assert_eq!(faces[0].role, BoundaryFaceRoleKindV0::FlowTransition);
        assert_eq!(faces[0].owner_ancestry, OwnerAncestryV0::FirstIsAncestor);
        assert_eq!(
            faces[1].role,
            BoundaryFaceRoleKindV0::LateralBoundaryCandidate
        );
        assert_eq!(faces[1].owner_ancestry, OwnerAncestryV0::FirstIsAncestor);
        assert_eq!(
            faces[2].role,
            BoundaryFaceRoleKindV0::LateralBoundaryCandidate
        );
        assert_eq!(faces[2].owner_ancestry, OwnerAncestryV0::Incomparable);
        assert_eq!(
            kahan_sum(raw.iter().map(|face| face.physical_length_km)),
            kahan_sum(faces.iter().map(|face| face.physical_length_km))
        );
    }

    #[test]
    fn displaced_conditioned_and_zero_segment_trace_records_are_frozen() {
        let centers = vec![DVec3::ZERO, DVec3::new(2.0, 0.0, 0.0)];
        let graph = kernel_graph(
            centers.clone(),
            centers.iter().map(|&center| square(center, 0.4)).collect(),
            vec![0, 0, 0],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        let displaced = TraceSummary {
            target: TraceTargetV0::ReachCell {
                reach_id: 0,
                cell: 1,
            },
            target_point: centers[1],
            target_z: 0.0,
            min_drop: Some(-1.0),
            maximum_z: 2.0,
            length: 3.0,
            length_correction: 0.0,
            segment_count: 2,
            fill: true,
            flat: false,
            non_descending: true,
        };
        let record = trace_record(&graph, 0, &[1.0, 0.0], &[displaced.clone(), displaced]).unwrap();
        assert_eq!(record.target_drop_km, 1.0);
        assert_eq!(record.minimum_segment_drop_km, Some(-1.0));
        assert_eq!(record.remote_maximum_excess_km, 1.0);
        assert!(record.fill_supported);
        assert!(!record.physically_descending);

        let zero = TraceSummary {
            target: TraceTargetV0::ReachCell {
                reach_id: 0,
                cell: 1,
            },
            target_point: centers[1],
            target_z: 1.0,
            min_drop: None,
            maximum_z: 1.0,
            length: 0.0,
            length_correction: 0.0,
            segment_count: 0,
            fill: false,
            flat: false,
            non_descending: false,
        };
        let zero_record = trace_record(&graph, 1, &[1.0, 1.0], &[zero.clone(), zero]).unwrap();
        assert_eq!(
            zero_record.status,
            ReceiverTraceStatusV0::TargetAtBoundaryCell
        );
        assert_eq!(zero_record.minimum_segment_drop_km, None);
        assert_eq!(zero_record.tortuosity, None);
        assert!(!zero_record.physically_descending);
    }

    fn strip_cross_section_kernel() -> (
        EvaluationSurfaceGraphV0,
        super::super::DrainageScaleV0,
        Vec<BackedBoundaryFaceV0>,
        Vec<f64>,
    ) {
        let ranges = [
            (-100.0, -80.0),
            (-80.0, -30.0),
            (-30.0, 30.0),
            (30.0, 80.0),
            (80.0, 100.0),
        ];
        let centers: Vec<_> = ranges
            .iter()
            .map(|&(low, high)| DVec3::new(0.0, 0.5 * (low + high), 0.0))
            .collect();
        let polygons = ranges
            .iter()
            .map(|&(low, high)| {
                vec![
                    DVec3::new(-100.0, low, 0.0),
                    DVec3::new(100.0, low, 0.0),
                    DVec3::new(100.0, high, 0.0),
                    DVec3::new(-100.0, high, 0.0),
                ]
            })
            .collect();
        let levels = [-80.0, -30.0, 30.0, 80.0];
        let mut endpoints = Vec::new();
        for level in levels {
            let face = [
                DVec3::new(-100.0, level, 0.0),
                DVec3::new(100.0, level, 0.0),
            ];
            endpoints.push(face);
            endpoints.push([face[1], face[0]]);
        }
        let graph = kernel_graph(
            centers,
            polygons,
            vec![0, 1, 3, 5, 7, 8],
            vec![1, 0, 2, 1, 3, 2, 4, 3],
            vec![1, 0, 3, 2, 5, 4, 7, 6],
            endpoints.clone(),
            vec![200.0; 8],
        );
        let owners = vec![
            IncrementalCatchmentOwnerV0::Portal(2),
            IncrementalCatchmentOwnerV0::Reach(0),
            IncrementalCatchmentOwnerV0::Reach(1),
            IncrementalCatchmentOwnerV0::Reach(0),
            IncrementalCatchmentOwnerV0::Portal(3),
        ];
        let face_specs = [
            ([1, 0], [1, 0], [owners[1], owners[0]], -80.0, 3.0),
            ([1, 2], [2, 3], [owners[1], owners[2]], -30.0, 2.5),
            ([3, 2], [5, 4], [owners[3], owners[2]], 30.0, 2.5),
            ([3, 4], [6, 7], [owners[3], owners[4]], 80.0, 3.0),
        ];
        let faces = face_specs
            .into_iter()
            .map(
                |(cells, edges, owners, level, height)| BackedBoundaryFaceV0 {
                    cells,
                    directed_edges: edges,
                    owners,
                    endpoints_km: [
                        DVec3::new(-100.0, level, 0.0),
                        DVec3::new(100.0, level, 0.0),
                    ],
                    physical_length_km: 200.0,
                    reconstructed_face_height_km: height,
                    covering_radius_km: 100.0,
                    owner_ancestry: OwnerAncestryV0::Incomparable,
                    role: BoundaryFaceRoleKindV0::LateralBoundaryCandidate,
                    receiver_direction: None,
                    bilateral_descent: None,
                },
            )
            .collect();
        let scale = super::super::DrainageScaleV0 {
            support_threshold_km2: REFERENCE_REACH_SUPPORT_KM2,
            basin_graph: super::super::DrainageBasinGraphV0 {
                catchments: Vec::new(),
                exclusive_owner: owners,
                raw_catchment_boundaries: Vec::new(),
            },
            reach_graph: super::super::RiverReachGraphV0 {
                cell_reach: vec![None, Some(0), Some(1), Some(0), None],
                reaches: Vec::new(),
                portal_roles: Vec::new(),
            },
        };
        (graph, scale, faces, vec![3.0, 2.5, 2.0, 2.5, 3.0])
    }

    #[test]
    fn analytic_cross_section_and_nested_exit_kernels_are_frozen() {
        let mut quadratic = Vec::new();
        let mut offset = 0.0;
        while offset < 80.0 {
            quadratic.push((offset, 2.0 + (offset / 80.0_f64).powi(2)));
            offset += 4.0;
        }
        quadratic.push((80.0, 3.0));
        let (left, left_bracket) = first_crossing(&quadratic, 2.25, 80.0);
        let (right, right_bracket) = first_crossing(&quadratic, 2.25, 80.0);
        assert_eq!(left, Some(40.0));
        assert_eq!(right, Some(40.0));
        assert_eq!(left.unwrap() + right.unwrap(), 80.0);
        assert_eq!(left_bracket, Some(RelativeHeightBracketV0::CellMeans));
        assert_eq!(right_bracket, Some(RelativeHeightBracketV0::CellMeans));

        let (off_grid, bracket) =
            first_crossing(&[(0.0, 2.0), (76.0, 2.2), (79.0, 3.0)], 2.25, 79.0);
        assert!(off_grid.unwrap() > 76.0 && off_grid.unwrap() < 79.0);
        assert_eq!(
            bracket,
            Some(RelativeHeightBracketV0::CellMeanToBoundaryProxy)
        );

        let (graph, scale, faces, elevation) = strip_cross_section_kernel();
        let index = RelationshipSpatialIndex::build(&graph, &faces).unwrap();
        let parents = [None, Some(0)];
        let mut candidate_tests = 0;
        let reference = LandformRelationshipConfigV0::default();
        let positive = section_side(
            &graph,
            &elevation,
            &scale,
            &faces,
            &index,
            0,
            &parents,
            DVec3::ZERO,
            DVec3::Y,
            Some(2),
            true,
            1.0,
            reference,
            &mut candidate_tests,
        );
        let negative = section_side(
            &graph,
            &elevation,
            &scale,
            &faces,
            &index,
            0,
            &parents,
            DVec3::ZERO,
            -DVec3::Y,
            Some(2),
            true,
            -1.0,
            reference,
            &mut candidate_tests,
        );
        assert_eq!(positive.boundary_offset_km, Some(80.0));
        assert_eq!(negative.boundary_offset_km, Some(80.0));
        assert_eq!(positive.boundary_relief_km, Some(1.0));
        assert_eq!(negative.boundary_relief_km, Some(1.0));
        assert!(candidate_tests > 0);

        let mut short = reference;
        short.cross_section_half_length_km = 50.0;
        let short_side = section_side(
            &graph,
            &elevation,
            &scale,
            &faces,
            &index,
            0,
            &parents,
            DVec3::ZERO,
            DVec3::Y,
            Some(2),
            true,
            1.0,
            short,
            &mut candidate_tests,
        );
        assert_eq!(
            short_side.censor_reason,
            Some(SectionCensorReasonV0::NoCatchmentExitWithinSupport)
        );
    }

    fn peak(id: u32, anchor_cell: u32) -> super::super::PeakBranchV0 {
        super::super::PeakBranchV0 {
            id,
            peak_elevation_km: 2.0,
            anchor_cell,
            flat_centroid_km: DVec3::ZERO,
            flat_maximum_cells: vec![anchor_cell],
            parent_peak: None,
            key_saddle: None,
            persistence_km: 1.0,
            root_closure: false,
            equal_elder_ambiguous: false,
            exclusive_cells: vec![anchor_cell],
            footprint_members: vec![anchor_cell],
            footprint_area_km2: 1.0,
            union_boundary_edges: Vec::new(),
            physical_boundary_segments: Vec::new(),
            scored_boundary_contact: false,
        }
    }

    fn descent_probe(physical: bool) -> BilateralPhysicalDescentV0 {
        let side = |cell| ReceiverTraceDescentV0 {
            adjacent_cell: cell,
            target: TraceTargetV0::ReachCell {
                reach_id: cell,
                cell,
            },
            status: ReceiverTraceStatusV0::ReachedTarget,
            adjacent_elevation_km: 2.0,
            target_elevation_km: 1.0,
            target_drop_km: 1.0,
            minimum_segment_drop_km: Some(if physical { 1.0 } else { -1.0 }),
            remote_maximum_excess_km: if physical { 0.0 } else { 1.0 },
            receiver_length_km: 1.0,
            endpoint_distance_km: 1.0,
            tortuosity: Some(1.0),
            fill_supported: !physical,
            flat_supported: false,
            physically_non_descending_segment: !physical,
            physically_descending: physical,
        };
        BilateralPhysicalDescentV0 {
            sides: [side(0), side(1)],
            bilateral_physical_descent: physical,
            unconditioned_bilateral_descent: physical,
        }
    }

    fn saddle_hierarchy(equal_flags: [bool; 2]) -> SurfaceHierarchyV0 {
        SurfaceHierarchyV0 {
            schema_version: G0S0_SCHEMA_VERSION.into(),
            hash_version: G0S0_HASH_VERSION.into(),
            peaks: vec![peak(0, 0), peak(1, 1)],
            saddles: equal_flags
                .into_iter()
                .enumerate()
                .map(|(id, equal)| super::super::SaddleNodeV0 {
                    id: id as u32,
                    elevation_km: 1.0,
                    anchor_cell: 0,
                    flat_centroid_km: DVec3::ZERO,
                    flat_saddle_cells: vec![0],
                    elder_peak: 0,
                    losing_peaks: vec![1],
                    equal_elder_ambiguous: equal,
                })
                .collect(),
            roots: vec![0, 1],
            cell_peak_owner: vec![Some(0), Some(1)],
            populations: super::super::HighlandPopulationsV0 {
                reference: Vec::new(),
                persistence_low: Vec::new(),
                persistence_high: Vec::new(),
                footprint_low: Vec::new(),
                footprint_high: Vec::new(),
            },
            reference_highlands: Vec::new(),
            derived_evidence_hash: 0,
        }
    }

    fn saddle_scale() -> super::super::DrainageScaleV0 {
        super::super::DrainageScaleV0 {
            support_threshold_km2: REFERENCE_REACH_SUPPORT_KM2,
            basin_graph: super::super::DrainageBasinGraphV0 {
                catchments: Vec::new(),
                exclusive_owner: vec![
                    IncrementalCatchmentOwnerV0::Reach(0),
                    IncrementalCatchmentOwnerV0::Reach(1),
                ],
                raw_catchment_boundaries: Vec::new(),
            },
            reach_graph: super::super::RiverReachGraphV0 {
                cell_reach: vec![Some(0), Some(1)],
                reaches: Vec::new(),
                portal_roles: Vec::new(),
            },
        }
    }

    fn saddle_face(y: f64, proxy: f64, physical: bool) -> BackedBoundaryFaceV0 {
        BackedBoundaryFaceV0 {
            cells: [0, 1],
            directed_edges: [0, 1],
            owners: [
                IncrementalCatchmentOwnerV0::Reach(0),
                IncrementalCatchmentOwnerV0::Reach(1),
            ],
            endpoints_km: [DVec3::new(-1.0, y, 0.0), DVec3::new(1.0, y, 0.0)],
            physical_length_km: 2.0,
            reconstructed_face_height_km: proxy,
            covering_radius_km: 1.0,
            owner_ancestry: OwnerAncestryV0::Incomparable,
            role: BoundaryFaceRoleKindV0::LateralBoundaryCandidate,
            receiver_direction: None,
            bilateral_descent: Some(descent_probe(physical)),
        }
    }

    #[test]
    fn saddle_nearest_owner_pair_tie_and_negative_control_are_frozen() {
        let centers = vec![DVec3::new(-2.0, 0.0, 0.0), DVec3::new(2.0, 0.0, 0.0)];
        let graph = kernel_graph(
            centers.clone(),
            centers.iter().map(|&center| square(center, 0.4)).collect(),
            vec![0, 0, 0],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        let hierarchy = saddle_hierarchy([false, true]);
        let scale = saddle_scale();
        // Equal distances; coordinate-canonical endpoints at y=-1 win even
        // though that record appears second.
        let faces = vec![saddle_face(1.0, 0.75, true), saddle_face(-1.0, 0.75, true)];
        let records = saddle_relationships(&graph, &hierarchy, &scale, &faces, &|_, _| {
            OwnerAncestryV0::Incomparable
        })
        .unwrap();
        assert_eq!(records.len(), 2);
        for (record, equal) in records.iter().zip([false, true]) {
            assert_eq!(record.boundary_face_index, Some(1));
            assert_eq!(record.separation_km, Some(1.0));
            assert_eq!(record.effective_covering_radius_km, Some(1.0));
            assert_eq!(record.within_covering_radius, Some(true));
            assert_eq!(record.saddle_minus_face_height_km, Some(0.25));
            assert_eq!(record.bilateral_descent, faces[1].bilateral_descent);
            assert_eq!(record.equal_elder_ambiguous, equal);
        }

        let negative = vec![saddle_face(-1.0, 2.0, false)];
        let negative_record = saddle_relationships(
            &graph,
            &saddle_hierarchy([false, false]),
            &scale,
            &negative,
            &|_, _| OwnerAncestryV0::Incomparable,
        )
        .unwrap();
        assert_eq!(negative_record[0].saddle_minus_face_height_km, Some(-1.0));
        assert!(
            !negative_record[0]
                .bilateral_descent
                .as_ref()
                .unwrap()
                .bilateral_physical_descent
        );
    }

    fn probe_positive_side(
        graph: &EvaluationSurfaceGraphV0,
        scale: &super::super::DrainageScaleV0,
        faces: &[BackedBoundaryFaceV0],
        elevation: &[f64],
        config: LandformRelationshipConfigV0,
    ) -> CrossSectionSideV0 {
        let index = RelationshipSpatialIndex::build(graph, faces).unwrap();
        let mut tests = 0;
        section_side(
            graph,
            elevation,
            scale,
            faces,
            &index,
            0,
            &[None, Some(0)],
            DVec3::ZERO,
            DVec3::Y,
            Some(2),
            true,
            1.0,
            config,
            &mut tests,
        )
    }

    #[test]
    fn cross_section_negative_controls_retain_signed_and_censor_evidence() {
        let (graph, scale, base_faces, _) = strip_cross_section_kernel();

        let mut monotone_faces = base_faces.clone();
        monotone_faces[0].reconstructed_face_height_km = 1.0;
        monotone_faces[3].reconstructed_face_height_km = 3.0;
        let monotone_elevation = [1.0, 1.5, 2.0, 2.5, 3.0];
        let index = RelationshipSpatialIndex::build(&graph, &monotone_faces).unwrap();
        let mut tests = 0;
        let positive = section_side(
            &graph,
            &monotone_elevation,
            &scale,
            &monotone_faces,
            &index,
            0,
            &[None, Some(0)],
            DVec3::ZERO,
            DVec3::Y,
            Some(2),
            true,
            1.0,
            LandformRelationshipConfigV0::default(),
            &mut tests,
        );
        let negative = section_side(
            &graph,
            &monotone_elevation,
            &scale,
            &monotone_faces,
            &index,
            0,
            &[None, Some(0)],
            DVec3::ZERO,
            -DVec3::Y,
            Some(2),
            true,
            -1.0,
            LandformRelationshipConfigV0::default(),
            &mut tests,
        );
        assert_eq!(positive.positive_boundary_relief, Some(true));
        assert_eq!(negative.positive_boundary_relief, Some(false));
        assert_eq!(negative.relative_height_crossing_km, None);
        assert!(positive.relative_height_crossing_km.is_some());
        assert!(negative.relative_height_crossing_km.is_none());

        let shoulder = probe_positive_side(
            &graph,
            &scale,
            &base_faces,
            &[3.0, 2.5, 2.0, 1.0, 3.0],
            LandformRelationshipConfigV0::default(),
        );
        assert!(shoulder.minimum_elevation_offset_km.unwrap() > 0.0);

        let mut ridge_faces = base_faces.clone();
        ridge_faces[3].reconstructed_face_height_km = 1.0;
        let ridge = probe_positive_side(
            &graph,
            &scale,
            &ridge_faces,
            &[3.0, 2.5, 2.0, 2.5, 1.0],
            LandformRelationshipConfigV0::default(),
        );
        assert_eq!(ridge.boundary_relief_km, Some(-1.0));
        assert_eq!(ridge.positive_boundary_relief, Some(false));
        assert_eq!(ridge.relative_height_crossing_km, None);

        let short = LandformRelationshipConfigV0 {
            cross_section_half_length_km: 50.0,
            ..LandformRelationshipConfigV0::default()
        };
        let truncated = probe_positive_side(
            &graph,
            &scale,
            &base_faces,
            &[3.0, 2.5, 2.0, 2.5, 3.0],
            short,
        );
        assert_eq!(
            truncated.censor_reason,
            Some(SectionCensorReasonV0::NoCatchmentExitWithinSupport)
        );
    }

    fn rotate_30(point: DVec3) -> DVec3 {
        let angle = std::f64::consts::PI / 6.0;
        let (sin, cos) = angle.sin_cos();
        DVec3::new(
            cos * point.x - sin * point.y,
            sin * point.x + cos * point.y,
            point.z,
        )
    }

    fn rotated_kernel(
        graph: &EvaluationSurfaceGraphV0,
        faces: &[BackedBoundaryFaceV0],
    ) -> (EvaluationSurfaceGraphV0, Vec<BackedBoundaryFaceV0>) {
        let mut graph = graph.clone();
        graph
            .cell_center_km
            .iter_mut()
            .for_each(|point| *point = rotate_30(*point));
        graph
            .cell_polygon_vertices_km
            .iter_mut()
            .for_each(|point| *point = rotate_30(*point));
        graph.edge_face_endpoints_km.iter_mut().for_each(|edge| {
            edge[0] = rotate_30(edge[0]);
            edge[1] = rotate_30(edge[1]);
        });
        graph.boundary_segments.iter_mut().for_each(|segment| {
            segment.endpoints_km[0] = rotate_30(segment.endpoints_km[0]);
            segment.endpoints_km[1] = rotate_30(segment.endpoints_km[1]);
        });
        let mut faces = faces.to_vec();
        faces.iter_mut().for_each(|face| {
            face.endpoints_km[0] = rotate_30(face.endpoints_km[0]);
            face.endpoints_km[1] = rotate_30(face.endpoints_km[1]);
        });
        (graph, faces)
    }

    #[test]
    fn tie_free_descent_and_section_kernels_rotate_covariantly_30_degrees() {
        let (graph, scale, faces, elevation) = strip_cross_section_kernel();
        let base = probe_positive_side(
            &graph,
            &scale,
            &faces,
            &elevation,
            LandformRelationshipConfigV0::default(),
        );
        let (rotated_graph, rotated_faces) = rotated_kernel(&graph, &faces);
        let rotated_index =
            RelationshipSpatialIndex::build(&rotated_graph, &rotated_faces).unwrap();
        let mut tests = 0;
        let rotated = section_side(
            &rotated_graph,
            &elevation,
            &scale,
            &rotated_faces,
            &rotated_index,
            0,
            &[None, Some(0)],
            DVec3::ZERO,
            rotate_30(DVec3::Y),
            Some(2),
            true,
            1.0,
            LandformRelationshipConfigV0::default(),
            &mut tests,
        );
        assert!(
            (base.boundary_offset_km.unwrap() - rotated.boundary_offset_km.unwrap()).abs()
                < 1.0e-10
        );
        assert_eq!(base.boundary_relief_km, rotated.boundary_relief_km);
        assert_eq!(
            base.positive_boundary_relief,
            rotated.positive_boundary_relief
        );
        for (original, transformed) in base.samples.iter().zip(&rotated.samples) {
            assert!(rotate_30(original.point_km).distance(transformed.point_km) < 1.0e-9);
            // The regular lattice point exactly at the selected face is a
            // predecessor coordinate tie, so it is excluded from this
            // deliberately tie-free covariance comparison. The appended
            // boundary proxy at the same offset remains comparable.
            if original.height_provenance == Some(CrossSectionHeightProvenanceV0::CellMean)
                && original.signed_offset_km.abs() == base.boundary_offset_km.unwrap()
            {
                continue;
            }
            assert_eq!(
                original.physical_elevation_km,
                transformed.physical_elevation_km
            );
            assert_eq!(original.height_provenance, transformed.height_provenance);
        }

        let probe = descent_probe(true);
        let base_face = saddle_face(-1.0, 0.75, true);
        let mut rotated_face = base_face.clone();
        rotated_face.endpoints_km = base_face.endpoints_km.map(rotate_30);
        assert_eq!(base_face.bilateral_descent, Some(probe.clone()));
        assert_eq!(rotated_face.bilateral_descent, Some(probe));
        for index in 0..2 {
            assert!(
                rotate_30(base_face.endpoints_km[index]).distance(rotated_face.endpoints_km[index])
                    < 1.0e-12
            );
        }

        let (base_orientation, base_difference) =
            boundary_orientation(std::iter::once(&base_face), DVec3::X);
        let (rotated_orientation, rotated_difference) =
            boundary_orientation(std::iter::once(&rotated_face), rotate_30(DVec3::X));
        assert!(
            rotate_30(base_orientation.unwrap()).distance(rotated_orientation.unwrap()) < 1.0e-12
        );
        assert!((base_difference.unwrap() - rotated_difference.unwrap()).abs() < 1.0e-12);
    }

    fn reindex_strip_kernel(
        graph: &EvaluationSurfaceGraphV0,
        scale: &super::super::DrainageScaleV0,
        faces: &[BackedBoundaryFaceV0],
        elevation: &[f64],
    ) -> (
        EvaluationSurfaceGraphV0,
        super::super::DrainageScaleV0,
        Vec<BackedBoundaryFaceV0>,
        Vec<f64>,
        Vec<usize>,
    ) {
        let n = graph.cell_count();
        assert_eq!(n, 5);
        let old_to_new: Vec<usize> = (0..n).map(|old| (17 * old + 3) % n).collect();
        let mut new_to_old = vec![usize::MAX; n];
        for (old, &new) in old_to_new.iter().enumerate() {
            assert_eq!(new_to_old[new], usize::MAX);
            new_to_old[new] = old;
        }

        let mut centers = vec![DVec3::ZERO; n];
        let mut areas = vec![0.0; n];
        let mut polygon_offsets = Vec::with_capacity(n + 1);
        let mut polygon_vertices = Vec::new();
        for new in 0..n {
            let old = new_to_old[new];
            centers[new] = graph.cell_center_km[old];
            areas[new] = graph.cell_area_km2[old];
            polygon_offsets.push(polygon_vertices.len() as u32);
            polygon_vertices.extend_from_slice(graph.polygon(old));
        }
        polygon_offsets.push(polygon_vertices.len() as u32);

        let edge_count = graph.edge_neighbor.len();
        let mut old_edge_to_new = vec![usize::MAX; edge_count];
        let mut edge_offsets = Vec::with_capacity(n + 1);
        let mut edge_neighbor = Vec::with_capacity(edge_count);
        let mut edge_distance = Vec::with_capacity(edge_count);
        let mut edge_width = Vec::with_capacity(edge_count);
        let mut edge_endpoints = Vec::with_capacity(edge_count);
        for &old in &new_to_old {
            edge_offsets.push(edge_neighbor.len() as u32);
            for old_edge in edge_range(graph, old) {
                old_edge_to_new[old_edge] = edge_neighbor.len();
                edge_neighbor.push(old_to_new[graph.edge_neighbor[old_edge] as usize] as u32);
                edge_distance.push(graph.edge_distance_km[old_edge]);
                edge_width.push(graph.edge_shared_width_km[old_edge]);
                edge_endpoints.push(graph.edge_face_endpoints_km[old_edge]);
            }
        }
        edge_offsets.push(edge_neighbor.len() as u32);
        assert!(old_edge_to_new.iter().all(|&edge| edge != usize::MAX));
        let mut edge_reciprocal = vec![u32::MAX; edge_count];
        for old_edge in 0..edge_count {
            edge_reciprocal[old_edge_to_new[old_edge]] =
                old_edge_to_new[graph.edge_reciprocal[old_edge] as usize] as u32;
        }
        let mut boundary_segments = graph.boundary_segments.clone();
        for segment in &mut boundary_segments {
            segment.owner_cell = old_to_new[segment.owner_cell as usize] as u32;
        }
        let remapped_graph = EvaluationSurfaceGraphV0 {
            domain: graph.domain,
            cell_center_km: centers,
            cell_area_km2: areas,
            cell_polygon_offsets: polygon_offsets,
            cell_polygon_vertices_km: polygon_vertices,
            edge_offsets,
            edge_neighbor,
            edge_reciprocal,
            edge_distance_km: edge_distance,
            edge_shared_width_km: edge_width,
            edge_face_endpoints_km: edge_endpoints,
            boundary_segments,
        };

        let mut remapped_scale = scale.clone();
        let mut owners = vec![IncrementalCatchmentOwnerV0::Portal(0); n];
        let mut cell_reach = vec![None; n];
        let mut remapped_elevation = vec![0.0; n];
        for old in 0..n {
            let new = old_to_new[old];
            owners[new] = scale.basin_graph.exclusive_owner[old];
            cell_reach[new] = scale.reach_graph.cell_reach[old];
            remapped_elevation[new] = elevation[old];
        }
        remapped_scale.basin_graph.exclusive_owner = owners;
        remapped_scale.reach_graph.cell_reach = cell_reach;
        let mut remapped_faces = faces.to_vec();
        for face in &mut remapped_faces {
            face.cells = face.cells.map(|cell| old_to_new[cell as usize] as u32);
            face.directed_edges = face
                .directed_edges
                .map(|edge| old_edge_to_new[edge as usize] as u32);
        }
        (
            remapped_graph,
            remapped_scale,
            remapped_faces,
            remapped_elevation,
            old_to_new,
        )
    }

    fn assert_section_evidence_by_coordinates(
        expected: &CrossSectionSideV0,
        actual: &CrossSectionSideV0,
    ) {
        assert_eq!(expected.censor_reason, actual.censor_reason);
        assert_eq!(expected.boundary_offset_km, actual.boundary_offset_km);
        assert_eq!(
            expected.boundary_height_proxy_km,
            actual.boundary_height_proxy_km
        );
        assert_eq!(expected.boundary_relief_km, actual.boundary_relief_km);
        assert_eq!(
            expected.maximum_sampled_relief_km,
            actual.maximum_sampled_relief_km
        );
        assert_eq!(
            expected.boundary_maximum_separation_km,
            actual.boundary_maximum_separation_km
        );
        assert_eq!(
            expected.minimum_elevation_offset_km,
            actual.minimum_elevation_offset_km
        );
        assert_eq!(
            expected.positive_boundary_relief,
            actual.positive_boundary_relief
        );
        assert_eq!(
            expected.relative_height_crossing_km,
            actual.relative_height_crossing_km
        );
        assert_eq!(expected.crossing_bracket, actual.crossing_bracket);
        assert_eq!(expected.samples.len(), actual.samples.len());
        for (expected, actual) in expected.samples.iter().zip(&actual.samples) {
            assert_eq!(expected.signed_offset_km, actual.signed_offset_km);
            assert_eq!(expected.point_km, actual.point_km);
            assert_eq!(expected.physical_elevation_km, actual.physical_elevation_km);
            assert_eq!(expected.height_provenance, actual.height_provenance);
            assert_eq!(expected.outside_domain, actual.outside_domain);
        }
    }

    #[test]
    fn fixed_permutation_rebuilds_strip_csr_and_preserves_coordinate_evidence() {
        let (graph, scale, faces, elevation) = strip_cross_section_kernel();
        let (remapped_graph, remapped_scale, remapped_faces, remapped_elevation, old_to_new) =
            reindex_strip_kernel(&graph, &scale, &faces, &elevation);
        assert_eq!(old_to_new, vec![3, 0, 2, 4, 1]);
        for old in 0..graph.cell_count() {
            let new = old_to_new[old];
            assert_eq!(
                remapped_graph.cell_center_km[new],
                graph.cell_center_km[old]
            );
            assert_eq!(
                remapped_scale.basin_graph.exclusive_owner[new],
                scale.basin_graph.exclusive_owner[old]
            );
            assert_eq!(
                remapped_scale.reach_graph.cell_reach[new],
                scale.reach_graph.cell_reach[old]
            );
            assert_eq!(remapped_elevation[new], elevation[old]);
        }
        for new_edge in 0..remapped_graph.edge_neighbor.len() {
            let reciprocal = remapped_graph.edge_reciprocal[new_edge] as usize;
            assert_eq!(
                remapped_graph.edge_reciprocal[reciprocal] as usize,
                new_edge
            );
        }
        for (old, remapped) in faces.iter().zip(&remapped_faces) {
            assert_eq!(
                remapped.cells,
                old.cells.map(|cell| old_to_new[cell as usize] as u32)
            );
            assert_eq!(remapped.endpoints_km, old.endpoints_km);
        }

        let base_index = RelationshipSpatialIndex::build(&graph, &faces).unwrap();
        let remapped_index =
            RelationshipSpatialIndex::build(&remapped_graph, &remapped_faces).unwrap();
        for (normal, sign) in [(DVec3::Y, 1.0), (-DVec3::Y, -1.0)] {
            let mut base_tests = 0;
            let base = section_side(
                &graph,
                &elevation,
                &scale,
                &faces,
                &base_index,
                0,
                &[None, Some(0)],
                DVec3::ZERO,
                normal,
                Some(2),
                true,
                sign,
                LandformRelationshipConfigV0::default(),
                &mut base_tests,
            );
            let mut remapped_tests = 0;
            let remapped = section_side(
                &remapped_graph,
                &remapped_elevation,
                &remapped_scale,
                &remapped_faces,
                &remapped_index,
                0,
                &[None, Some(0)],
                DVec3::ZERO,
                normal,
                Some(old_to_new[2]),
                true,
                sign,
                LandformRelationshipConfigV0::default(),
                &mut remapped_tests,
            );
            assert_section_evidence_by_coordinates(&base, &remapped);
        }
    }
}
