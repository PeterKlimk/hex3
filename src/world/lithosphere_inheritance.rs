//! Source-only basement provinces and inherited structural contacts.
//!
//! This module is deliberately upstream of terrain. It subdivides the existing
//! continental envelope into coherent material provinces, compiles their exact
//! Voronoi contacts into a candidate graph, and exposes geometric relationships to
//! any plate boundary. It does not read or write elevation, drainage, erosion,
//! semantic landforms, or presentation state.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use glam::Vec3;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use super::constants::TRANSFORM_NORMAL_THRESHOLD;
use super::{
    BoundaryKind, CellEdgeId, Crust, CrustType, PlateBoundaryEdge, Tessellation, PLANET_RADIUS_KM,
};

pub const OCEANIC_BASEMENT_PROVINCE: u32 = u32::MAX;
pub const LITHOSPHERE_INHERITANCE_SEED_SALT: u64 = 5;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LithosphereInheritanceConfigV0 {
    /// Target province area. It controls count, not an exact final area.
    pub target_province_area_km2: f64,
    pub maximum_provinces_per_craton: usize,
    /// Province-count cap based on average cells, not a per-province guarantee.
    pub minimum_average_cells_per_province: usize,
}

impl Default for LithosphereInheritanceConfigV0 {
    fn default() -> Self {
        Self {
            target_province_area_km2: 2_000_000.0,
            maximum_provinces_per_craton: 24,
            minimum_average_cells_per_province: 16,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BasementProvinceV0 {
    pub id: u32,
    pub craton_id: u32,
    pub seed_cell: usize,
    pub cell_count: usize,
    pub area_km2: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InheritedStructureKindV0 {
    /// An exact material-province contact that has not been assigned history.
    BasementContact,
    Suture,
    InheritedRift,
    /// Explicit finite connector between offset inherited traces.
    TransferLink,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InheritedStructureEdgeV0 {
    pub id: CellEdgeId,
    pub cells: [usize; 2],
    /// Ordered in `cells[0]`'s polygon orientation.
    pub vertices: [u32; 2],
    pub endpoints: [Vec3; 2],
    pub length_km: f32,
    /// Canonically sorted province identity; V0 support carries no side polarity.
    pub provinces: [u32; 2],
    pub kind: InheritedStructureKindV0,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InheritedStructureSegmentV0 {
    pub id: u32,
    pub kind: InheritedStructureKindV0,
    pub provinces: [u32; 2],
    pub source_edges: Vec<CellEdgeId>,
    pub vertices_in_order: Vec<u32>,
    pub closed: bool,
    pub length_km: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InheritedStructureIncidenceKindV0 {
    Tip,
    MultiTrace,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InheritedStructureIncidenceV0 {
    pub id: u32,
    pub vertex: u32,
    pub kind: InheritedStructureIncidenceKindV0,
    pub incident_segments: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InheritedStructureSegmentEndV0 {
    Start,
    End,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct InheritedStructureSegmentEndRefV0 {
    pub segment_id: u32,
    pub end: InheritedStructureSegmentEndV0,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InheritedStructureRelationshipKindV0 {
    Continuation,
    Junction,
    OffsetTransfer,
    CrossingUnlinked,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InheritedStructureRelationshipTopologyV0 {
    Continuation {
        ends: [InheritedStructureSegmentEndRefV0; 2],
    },
    Junction {
        ends: Vec<InheritedStructureSegmentEndRefV0>,
    },
    OffsetTransfer {
        primary_ends: [InheritedStructureSegmentEndRefV0; 2],
        connector_segment_id: u32,
    },
    CrossingUnlinked {
        branches: [[InheritedStructureSegmentEndRefV0; 2]; 2],
    },
}

impl InheritedStructureRelationshipTopologyV0 {
    pub fn kind(&self) -> InheritedStructureRelationshipKindV0 {
        match self {
            Self::Continuation { .. } => InheritedStructureRelationshipKindV0::Continuation,
            Self::Junction { .. } => InheritedStructureRelationshipKindV0::Junction,
            Self::OffsetTransfer { .. } => InheritedStructureRelationshipKindV0::OffsetTransfer,
            Self::CrossingUnlinked { .. } => InheritedStructureRelationshipKindV0::CrossingUnlinked,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InheritedStructureRelationshipV0 {
    pub id: u32,
    pub topology: InheritedStructureRelationshipTopologyV0,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InheritedStructureGraphV0 {
    pub edges: Vec<InheritedStructureEdgeV0>,
    pub segments: Vec<InheritedStructureSegmentV0>,
    /// Geometric endpoint incidence only; this does not declare connectivity.
    pub incidences: Vec<InheritedStructureIncidenceV0>,
    /// Explicit geological connectivity. Candidate basement graphs leave this empty.
    pub relationships: Vec<InheritedStructureRelationshipV0>,
    pub total_length_km: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LithosphereInheritanceV0 {
    pub config: LithosphereInheritanceConfigV0,
    /// Province ID per coarse cell; `OCEANIC_BASEMENT_PROVINCE` over oceanic crust.
    pub cell_province: Vec<u32>,
    pub provinces: Vec<BasementProvinceV0>,
    pub graph: InheritedStructureGraphV0,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryInheritanceContactKindV0 {
    Unrelated,
    Coincident,
    VertexContact,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoundaryInheritanceRelationshipV0 {
    pub boundary: CellEdgeId,
    pub kind: BoundaryInheritanceContactKindV0,
    pub shared_vertices: Vec<u32>,
    pub structure_segment_ids: Vec<u32>,
    pub geometric_incidence_ids: Vec<u32>,
    pub structure_relationship_ids: Vec<u32>,
    pub structure_relationship_kinds: Vec<InheritedStructureRelationshipKindV0>,
    /// Smallest unoriented tangent angle at an exact shared vertex, in [0, 90].
    pub minimum_tangent_angle_deg: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryInheritanceApplicationV0 {
    Ineligible,
    ContinentalCollision,
    ContinentalRifting,
    OceanicSpreading,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryInheritanceGeologyV0 {
    None,
    CandidateBasementContact,
    NamedSuture,
    NamedInheritedRift,
    TransferLink,
    Mixed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundaryInheritanceAssessmentV0 {
    pub application: BoundaryInheritanceApplicationV0,
    pub geology: BoundaryInheritanceGeologyV0,
    pub geometric_contact: BoundaryInheritanceContactKindV0,
    pub structure_relationship_kinds: Vec<InheritedStructureRelationshipKindV0>,
}

struct ContactTopologyMetadataV0 {
    geometric_incidence_ids: Vec<u32>,
    relationship_ids: Vec<u32>,
    relationship_kinds: Vec<InheritedStructureRelationshipKindV0>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LithosphereInheritanceErrorV0 {
    InvalidConfig,
    CrustLengthMismatch,
    CrustIdentityMismatch(usize),
    MissingSharedTopology(CellEdgeId),
    InvalidBoundary(CellEdgeId),
    UnassignedContinentalCell(usize),
    InvalidProvinceContact(CellEdgeId),
    InvalidManufacturedGraph,
    InvalidStructureRelationship(u32),
    MissingStructureSegment(u32),
    InvalidRelationshipMetadata,
}

impl std::fmt::Display for LithosphereInheritanceErrorV0 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for LithosphereInheritanceErrorV0 {}

/// Generate a deterministic, terrain-blind basement subdivision and contact graph.
pub fn generate_lithosphere_inheritance_v0(
    seed: u64,
    tessellation: &Tessellation,
    crust: &Crust,
    config: LithosphereInheritanceConfigV0,
) -> Result<LithosphereInheritanceV0, LithosphereInheritanceErrorV0> {
    validate_inputs(tessellation, crust, config)?;
    let cell_areas = tessellation.cell_areas_ref();
    let mut craton_cells = BTreeMap::<u32, Vec<usize>>::new();
    for (cell, &craton) in crust.cell_craton.iter().enumerate() {
        if craton != u32::MAX {
            craton_cells.entry(craton).or_default().push(cell);
        }
    }

    let mut cell_province = vec![OCEANIC_BASEMENT_PROVINCE; tessellation.num_cells()];
    let mut province_seeds = Vec::<(u32, u32, usize)>::new();
    let mut next_province = 0_u32;
    for (&craton, cells) in &craton_cells {
        let area_km2 = cells
            .iter()
            .map(|&cell| f64::from(cell_areas[cell]) * f64::from(PLANET_RADIUS_KM).powi(2))
            .sum::<f64>();
        let count = province_count(cells.len(), area_km2, config);
        let seeds = farthest_point_seeds(
            seed ^ u64::from(craton).wrapping_mul(0x9e37_79b9_7f4a_7c15),
            tessellation,
            cells,
            count,
        );
        let ids: Vec<_> = (0..count)
            .map(|_| {
                let id = next_province;
                next_province += 1;
                id
            })
            .collect();
        assign_craton_provinces(
            tessellation,
            crust,
            craton,
            &seeds,
            &ids,
            &mut cell_province,
        );
        province_seeds.extend(
            ids.into_iter()
                .zip(seeds)
                .map(|(province, cell)| (province, craton, cell)),
        );
    }

    for (cell, &crust_type) in crust.types.iter().enumerate() {
        if crust_type == CrustType::Continental && cell_province[cell] == OCEANIC_BASEMENT_PROVINCE
        {
            return Err(LithosphereInheritanceErrorV0::UnassignedContinentalCell(
                cell,
            ));
        }
    }

    let provinces = province_seeds
        .into_iter()
        .map(|(id, craton_id, seed_cell)| {
            let members = cell_province
                .iter()
                .enumerate()
                .filter(|(_, province)| **province == id);
            let mut cell_count = 0;
            let mut area_km2 = 0.0;
            for (cell, _) in members {
                cell_count += 1;
                area_km2 += f64::from(cell_areas[cell]) * f64::from(PLANET_RADIUS_KM).powi(2);
            }
            BasementProvinceV0 {
                id,
                craton_id,
                seed_cell,
                cell_count,
                area_km2,
            }
        })
        .collect();
    let graph = compile_structure_graph(tessellation, crust, &cell_province)?;
    Ok(LithosphereInheritanceV0 {
        config,
        cell_province,
        provinces,
        graph,
    })
}

/// Describe the exact relationship between any plate boundary and inherited state.
///
/// Callers may use this for convergent organization or divergent/rift
/// localization. The query reports geometry and topology but does not assign a
/// deformation sign or terrain consequence.
pub fn query_boundary_inheritance_v0(
    tessellation: &Tessellation,
    inheritance: &LithosphereInheritanceV0,
    boundary: CellEdgeId,
) -> Result<BoundaryInheritanceRelationshipV0, LithosphereInheritanceErrorV0> {
    if boundary.cell_a >= tessellation.num_cells()
        || boundary.cell_b >= tessellation.num_cells()
        || boundary.cell_a == boundary.cell_b
    {
        return Err(LithosphereInheritanceErrorV0::InvalidBoundary(boundary));
    }
    let boundary = CellEdgeId::new(boundary.cell_a, boundary.cell_b);
    let boundary_vertices = tessellation
        .shared_edge_vertices(boundary.cell_a, boundary.cell_b)
        .ok_or(LithosphereInheritanceErrorV0::MissingSharedTopology(
            boundary,
        ))?;
    if inheritance
        .graph
        .edges
        .binary_search_by_key(&boundary, |edge| edge.id)
        .is_ok()
    {
        let structure_segment_ids = vec![segment_id_for_edge(&inheritance.graph, boundary)
            .ok_or(LithosphereInheritanceErrorV0::InvalidManufacturedGraph)?];
        let topology = contact_topology_metadata(
            &inheritance.graph,
            &boundary_vertices,
            &structure_segment_ids,
        )?;
        return Ok(BoundaryInheritanceRelationshipV0 {
            boundary,
            kind: BoundaryInheritanceContactKindV0::Coincident,
            shared_vertices: boundary_vertices.to_vec(),
            structure_segment_ids,
            geometric_incidence_ids: topology.geometric_incidence_ids,
            structure_relationship_ids: topology.relationship_ids,
            structure_relationship_kinds: topology.relationship_kinds,
            minimum_tangent_angle_deg: Some(0.0),
        });
    }

    let mut shared_vertices = BTreeSet::new();
    let mut structure_edges = BTreeSet::new();
    let mut minimum_angle = f32::INFINITY;
    let boundary_endpoints =
        boundary_vertices.map(|vertex| tessellation.voronoi.vertices[vertex as usize]);
    for (boundary_endpoint, &vertex) in boundary_vertices.iter().enumerate() {
        let vertex_position = boundary_endpoints[boundary_endpoint];
        let boundary_other = boundary_endpoints[1 - boundary_endpoint];
        let boundary_tangent = tangent_toward(vertex_position, boundary_other);
        for edge in inheritance
            .graph
            .edges
            .iter()
            .filter(|edge| edge.vertices.contains(&vertex))
        {
            shared_vertices.insert(vertex);
            structure_edges.insert(edge.id);
            let structure_endpoint = if edge.vertices[0] == vertex { 0 } else { 1 };
            let structure_tangent =
                tangent_toward(vertex_position, edge.endpoints[1 - structure_endpoint]);
            if boundary_tangent != Vec3::ZERO && structure_tangent != Vec3::ZERO {
                let angle = boundary_tangent
                    .dot(structure_tangent)
                    .abs()
                    .clamp(-1.0, 1.0)
                    .acos()
                    .to_degrees();
                minimum_angle = minimum_angle.min(angle);
            }
        }
    }
    let structure_segment_ids: BTreeSet<_> = structure_edges
        .iter()
        .filter_map(|&edge| segment_id_for_edge(&inheritance.graph, edge))
        .collect();
    let shared_vertices: Vec<_> = shared_vertices.into_iter().collect();
    let structure_segment_ids: Vec<_> = structure_segment_ids.into_iter().collect();
    let topology =
        contact_topology_metadata(&inheritance.graph, &shared_vertices, &structure_segment_ids)?;
    Ok(BoundaryInheritanceRelationshipV0 {
        boundary,
        kind: if shared_vertices.is_empty() {
            BoundaryInheritanceContactKindV0::Unrelated
        } else {
            BoundaryInheritanceContactKindV0::VertexContact
        },
        shared_vertices,
        structure_segment_ids,
        geometric_incidence_ids: topology.geometric_incidence_ids,
        structure_relationship_ids: topology.relationship_ids,
        structure_relationship_kinds: topology.relationship_kinds,
        minimum_tangent_angle_deg: minimum_angle.is_finite().then_some(minimum_angle),
    })
}

/// Classify which existing product boundary consumer can read the relationship.
///
/// This is an inert semantic assessment. It does not change kinematics, assign
/// localization strength, or write terrain.
pub fn assess_plate_boundary_inheritance_v0(
    boundary: &PlateBoundaryEdge,
    relationship: &BoundaryInheritanceRelationshipV0,
    graph: &InheritedStructureGraphV0,
) -> Result<BoundaryInheritanceAssessmentV0, LithosphereInheritanceErrorV0> {
    let boundary_id = CellEdgeId::new(boundary.cell_a, boundary.cell_b);
    if boundary.cell_a == boundary.cell_b || relationship.boundary != boundary_id {
        return Err(LithosphereInheritanceErrorV0::InvalidBoundary(
            relationship.boundary,
        ));
    }
    let topology = contact_topology_metadata(
        graph,
        &relationship.shared_vertices,
        &relationship.structure_segment_ids,
    )?;
    if relationship.structure_relationship_ids != topology.relationship_ids
        || relationship.structure_relationship_kinds != topology.relationship_kinds
    {
        return Err(LithosphereInheritanceErrorV0::InvalidRelationshipMetadata);
    }
    let locally_convergent = boundary.convergence > TRANSFORM_NORMAL_THRESHOLD;
    let locally_opening = -boundary.convergence > TRANSFORM_NORMAL_THRESHOLD;
    let continental_a = boundary.type_a == CrustType::Continental;
    let continental_b = boundary.type_b == CrustType::Continental;
    let application = match boundary.kind {
        BoundaryKind::Convergent if locally_convergent && continental_a && continental_b => {
            BoundaryInheritanceApplicationV0::ContinentalCollision
        }
        BoundaryKind::Divergent if locally_opening && !continental_a && !continental_b => {
            BoundaryInheritanceApplicationV0::OceanicSpreading
        }
        BoundaryKind::Divergent if locally_opening && (continental_a || continental_b) => {
            BoundaryInheritanceApplicationV0::ContinentalRifting
        }
        _ => BoundaryInheritanceApplicationV0::Ineligible,
    };

    let mut kinds = BTreeSet::new();
    for &segment_id in &relationship.structure_segment_ids {
        let segment = graph
            .segments
            .iter()
            .find(|segment| segment.id == segment_id)
            .ok_or(LithosphereInheritanceErrorV0::MissingStructureSegment(
                segment_id,
            ))?;
        kinds.insert(segment.kind);
    }
    let geology = if kinds.is_empty() {
        BoundaryInheritanceGeologyV0::None
    } else if kinds.len() > 1 {
        BoundaryInheritanceGeologyV0::Mixed
    } else {
        match *kinds.iter().next().expect("nonempty set") {
            InheritedStructureKindV0::BasementContact => {
                BoundaryInheritanceGeologyV0::CandidateBasementContact
            }
            InheritedStructureKindV0::Suture => BoundaryInheritanceGeologyV0::NamedSuture,
            InheritedStructureKindV0::InheritedRift => {
                BoundaryInheritanceGeologyV0::NamedInheritedRift
            }
            InheritedStructureKindV0::TransferLink => BoundaryInheritanceGeologyV0::TransferLink,
        }
    };
    Ok(BoundaryInheritanceAssessmentV0 {
        application,
        geology,
        geometric_contact: relationship.kind,
        structure_relationship_kinds: topology.relationship_kinds,
    })
}

fn segment_id_for_edge(graph: &InheritedStructureGraphV0, edge: CellEdgeId) -> Option<u32> {
    graph
        .segments
        .iter()
        .find(|segment| segment.source_edges.contains(&edge))
        .map(|segment| segment.id)
}

fn contact_topology_metadata(
    graph: &InheritedStructureGraphV0,
    shared_vertices: &[u32],
    contacted_segments: &[u32],
) -> Result<ContactTopologyMetadataV0, LithosphereInheritanceErrorV0> {
    let geometric_incidence_ids = graph
        .incidences
        .iter()
        .filter(|incidence| shared_vertices.contains(&incidence.vertex))
        .map(|incidence| incidence.id)
        .collect();
    let mut relationship_ids = Vec::new();
    let mut relationship_kinds = BTreeSet::new();
    for relationship in &graph.relationships {
        let mut endpoint_contact = false;
        for endpoint in relationship_end_refs(&relationship.topology) {
            if contacted_segments.contains(&endpoint.segment_id)
                && shared_vertices.contains(&segment_end_vertex(graph, endpoint, relationship.id)?)
            {
                endpoint_contact = true;
                break;
            }
        }
        let connector_contact = match relationship.topology {
            InheritedStructureRelationshipTopologyV0::OffsetTransfer {
                connector_segment_id,
                ..
            } => contacted_segments.contains(&connector_segment_id),
            _ => false,
        };
        if endpoint_contact || connector_contact {
            relationship_ids.push(relationship.id);
            relationship_kinds.insert(relationship.topology.kind());
        }
    }
    Ok(ContactTopologyMetadataV0 {
        geometric_incidence_ids,
        relationship_ids,
        relationship_kinds: relationship_kinds.into_iter().collect(),
    })
}

fn relationship_end_refs(
    topology: &InheritedStructureRelationshipTopologyV0,
) -> Vec<InheritedStructureSegmentEndRefV0> {
    match topology {
        InheritedStructureRelationshipTopologyV0::Continuation { ends } => ends.to_vec(),
        InheritedStructureRelationshipTopologyV0::Junction { ends } => ends.clone(),
        InheritedStructureRelationshipTopologyV0::OffsetTransfer { primary_ends, .. } => {
            primary_ends.to_vec()
        }
        InheritedStructureRelationshipTopologyV0::CrossingUnlinked { branches } => {
            branches.iter().flatten().copied().collect()
        }
    }
}

fn segment_end_vertex(
    graph: &InheritedStructureGraphV0,
    endpoint: InheritedStructureSegmentEndRefV0,
    relationship_id: u32,
) -> Result<u32, LithosphereInheritanceErrorV0> {
    let segment = graph
        .segments
        .iter()
        .find(|segment| segment.id == endpoint.segment_id)
        .ok_or(LithosphereInheritanceErrorV0::MissingStructureSegment(
            endpoint.segment_id,
        ))?;
    if segment.closed || segment.vertices_in_order.len() < 2 {
        return Err(LithosphereInheritanceErrorV0::InvalidStructureRelationship(
            relationship_id,
        ));
    }
    Ok(match endpoint.end {
        InheritedStructureSegmentEndV0::Start => segment.vertices_in_order[0],
        InheritedStructureSegmentEndV0::End => *segment
            .vertices_in_order
            .last()
            .expect("checked nonempty segment"),
    })
}

fn validate_inputs(
    tessellation: &Tessellation,
    crust: &Crust,
    config: LithosphereInheritanceConfigV0,
) -> Result<(), LithosphereInheritanceErrorV0> {
    if !config.target_province_area_km2.is_finite()
        || config.target_province_area_km2 <= 0.0
        || config.maximum_provinces_per_craton == 0
        || config.minimum_average_cells_per_province == 0
    {
        return Err(LithosphereInheritanceErrorV0::InvalidConfig);
    }
    let count = tessellation.num_cells();
    if crust.types.len() != count || crust.cell_craton.len() != count {
        return Err(LithosphereInheritanceErrorV0::CrustLengthMismatch);
    }
    for (cell, (&crust_type, &craton)) in crust.types.iter().zip(&crust.cell_craton).enumerate() {
        let identity_matches = match crust_type {
            CrustType::Continental => craton != u32::MAX,
            CrustType::Oceanic => craton == u32::MAX,
        };
        if !identity_matches {
            return Err(LithosphereInheritanceErrorV0::CrustIdentityMismatch(cell));
        }
    }
    Ok(())
}

fn province_count(
    cell_count: usize,
    area_km2: f64,
    config: LithosphereInheritanceConfigV0,
) -> usize {
    let desired = (area_km2 / config.target_province_area_km2)
        .round()
        .max(1.0) as usize;
    let cell_cap = (cell_count / config.minimum_average_cells_per_province).max(1);
    desired
        .min(config.maximum_provinces_per_craton)
        .min(cell_cap)
}

fn farthest_point_seeds(
    seed: u64,
    tessellation: &Tessellation,
    cells: &[usize],
    count: usize,
) -> Vec<usize> {
    debug_assert!(!cells.is_empty());
    let mut seeds = vec![cells[(splitmix64(seed) as usize) % cells.len()]];
    while seeds.len() < count {
        let next = cells
            .iter()
            .copied()
            .filter(|cell| !seeds.contains(cell))
            .max_by(|&left, &right| {
                minimum_seed_chord_squared(tessellation, left, &seeds)
                    .total_cmp(&minimum_seed_chord_squared(tessellation, right, &seeds))
                    .then_with(|| right.cmp(&left))
            })
            .expect("province count is capped by craton cell count");
        seeds.push(next);
    }
    seeds
}

fn minimum_seed_chord_squared(tessellation: &Tessellation, cell: usize, seeds: &[usize]) -> f32 {
    let center = tessellation.cell_center(cell);
    seeds
        .iter()
        .map(|&seed| center.distance_squared(tessellation.cell_center(seed)))
        .fold(f32::INFINITY, f32::min)
}

fn assign_craton_provinces(
    tessellation: &Tessellation,
    crust: &Crust,
    craton: u32,
    seeds: &[usize],
    province_ids: &[u32],
    cell_province: &mut [u32],
) {
    let mut distance = vec![f32::INFINITY; tessellation.num_cells()];
    let mut queue = BinaryHeap::new();
    for (&cell, &province) in seeds.iter().zip(province_ids) {
        distance[cell] = 0.0;
        cell_province[cell] = province;
        queue.push(Reverse((OrderedFloat(0.0), province, cell)));
    }
    while let Some(Reverse((OrderedFloat(current_distance), province, cell))) = queue.pop() {
        if current_distance > distance[cell]
            || (current_distance == distance[cell] && cell_province[cell] != province)
        {
            continue;
        }
        let center = tessellation.cell_center(cell);
        for &neighbor in tessellation.neighbors(cell) {
            if crust.cell_craton[neighbor] != craton {
                continue;
            }
            let chord = center.distance(tessellation.cell_center(neighbor));
            let edge_length = 2.0 * (0.5 * chord).clamp(0.0, 1.0).asin();
            let candidate = current_distance + edge_length;
            if candidate < distance[neighbor]
                || (candidate == distance[neighbor] && province < cell_province[neighbor])
            {
                distance[neighbor] = candidate;
                cell_province[neighbor] = province;
                queue.push(Reverse((OrderedFloat(candidate), province, neighbor)));
            }
        }
    }
}

fn compile_structure_graph(
    tessellation: &Tessellation,
    crust: &Crust,
    cell_province: &[u32],
) -> Result<InheritedStructureGraphV0, LithosphereInheritanceErrorV0> {
    let mut edges = Vec::new();
    for cell_a in 0..tessellation.num_cells() {
        for &cell_b in tessellation.neighbors(cell_a) {
            if cell_b <= cell_a
                || crust.types[cell_a] != CrustType::Continental
                || crust.types[cell_b] != CrustType::Continental
                || cell_province[cell_a] == cell_province[cell_b]
            {
                continue;
            }
            let id = CellEdgeId::new(cell_a, cell_b);
            let vertices = tessellation
                .shared_edge_vertices(cell_a, cell_b)
                .ok_or(LithosphereInheritanceErrorV0::MissingSharedTopology(id))?;
            let endpoints = vertices.map(|vertex| tessellation.voronoi.vertices[vertex as usize]);
            let chord = endpoints[0].distance(endpoints[1]);
            let length_km = 2.0 * (0.5 * chord).clamp(0.0, 1.0).asin() * PLANET_RADIUS_KM;
            let mut provinces = [cell_province[cell_a], cell_province[cell_b]];
            provinces.sort_unstable();
            if provinces[0] == OCEANIC_BASEMENT_PROVINCE || provinces[0] == provinces[1] {
                return Err(LithosphereInheritanceErrorV0::InvalidProvinceContact(id));
            }
            edges.push(InheritedStructureEdgeV0 {
                id,
                cells: [cell_a, cell_b],
                vertices,
                endpoints,
                length_km,
                provinces,
                kind: InheritedStructureKindV0::BasementContact,
            });
        }
    }
    compile_graph_from_edges(edges)
}

fn compile_graph_from_edges(
    mut edges: Vec<InheritedStructureEdgeV0>,
) -> Result<InheritedStructureGraphV0, LithosphereInheritanceErrorV0> {
    edges.sort_by_key(|edge| edge.id);
    if edges.windows(2).any(|pair| pair[0].id == pair[1].id) {
        return Err(LithosphereInheritanceErrorV0::InvalidManufacturedGraph);
    }
    let by_id: BTreeMap<_, _> = edges.iter().map(|edge| (edge.id, edge)).collect();
    let mut incidence = BTreeMap::<u32, Vec<CellEdgeId>>::new();
    for edge in &edges {
        incidence.entry(edge.vertices[0]).or_default().push(edge.id);
        incidence.entry(edge.vertices[1]).or_default().push(edge.id);
    }
    for incident in incidence.values_mut() {
        incident.sort_unstable();
    }

    let mut visited = BTreeSet::new();
    let mut segments = Vec::new();
    for edge in &edges {
        if visited.contains(&edge.id) {
            continue;
        }
        let component = chain_component(edge.id, edge.provinces, edge.kind, &incidence, &by_id);
        let mut terminal: Option<(u32, CellEdgeId)> = None;
        for id in &component {
            for vertex in by_id[id].vertices {
                if !continues_through(vertex, edge.provinces, edge.kind, &incidence, &by_id) {
                    terminal = Some(terminal.map_or((vertex, *id), |current| {
                        std::cmp::min(current, (vertex, *id))
                    }));
                }
            }
        }
        let (start_vertex, mut current) =
            terminal.unwrap_or_else(|| (edge.vertices[0].min(edge.vertices[1]), edge.id));
        let mut from_vertex = start_vertex;
        let mut source_edges = Vec::new();
        let mut vertices_in_order = vec![start_vertex];
        let mut length_km = 0.0;
        let mut closed = false;
        loop {
            if !visited.insert(current) {
                return Err(LithosphereInheritanceErrorV0::InvalidManufacturedGraph);
            }
            let current_edge = by_id[&current];
            if !current_edge.vertices.contains(&from_vertex)
                || current_edge.provinces != edge.provinces
                || current_edge.kind != edge.kind
            {
                return Err(LithosphereInheritanceErrorV0::InvalidManufacturedGraph);
            }
            let to_vertex = if current_edge.vertices[0] == from_vertex {
                current_edge.vertices[1]
            } else {
                current_edge.vertices[0]
            };
            source_edges.push(current);
            vertices_in_order.push(to_vertex);
            length_km += current_edge.length_km;
            if to_vertex == start_vertex {
                closed = true;
                break;
            }
            if !continues_through(to_vertex, edge.provinces, edge.kind, &incidence, &by_id) {
                break;
            }
            let next = incidence[&to_vertex]
                .iter()
                .copied()
                .find(|candidate| *candidate != current)
                .ok_or(LithosphereInheritanceErrorV0::InvalidManufacturedGraph)?;
            if visited.contains(&next) {
                return Err(LithosphereInheritanceErrorV0::InvalidManufacturedGraph);
            }
            current = next;
            from_vertex = to_vertex;
        }
        segments.push(InheritedStructureSegmentV0 {
            id: segments.len() as u32,
            kind: edge.kind,
            provinces: edge.provinces,
            source_edges,
            vertices_in_order,
            closed,
            length_km,
        });
    }

    let mut endpoint_segments = BTreeMap::<u32, Vec<u32>>::new();
    for segment in &segments {
        if !segment.closed {
            endpoint_segments
                .entry(segment.vertices_in_order[0])
                .or_default()
                .push(segment.id);
            endpoint_segments
                .entry(*segment.vertices_in_order.last().expect("nonempty segment"))
                .or_default()
                .push(segment.id);
        }
    }
    let incidences = endpoint_segments
        .into_iter()
        .enumerate()
        .map(|(id, (vertex, mut incident_segments))| {
            incident_segments.sort_unstable();
            incident_segments.dedup();
            InheritedStructureIncidenceV0 {
                id: id as u32,
                vertex,
                kind: if incident_segments.len() == 1 {
                    InheritedStructureIncidenceKindV0::Tip
                } else {
                    InheritedStructureIncidenceKindV0::MultiTrace
                },
                incident_segments,
            }
        })
        .collect();
    let total_length_km = edges.iter().map(|edge| f64::from(edge.length_km)).sum();
    Ok(InheritedStructureGraphV0 {
        edges,
        segments,
        incidences,
        relationships: Vec::new(),
        total_length_km,
    })
}

fn continues_through(
    vertex: u32,
    provinces: [u32; 2],
    kind: InheritedStructureKindV0,
    incidence: &BTreeMap<u32, Vec<CellEdgeId>>,
    by_id: &BTreeMap<CellEdgeId, &InheritedStructureEdgeV0>,
) -> bool {
    let Some(incident) = incidence.get(&vertex) else {
        return false;
    };
    incident.len() == 2
        && incident
            .iter()
            .all(|edge| by_id[edge].provinces == provinces && by_id[edge].kind == kind)
}

fn chain_component(
    seed: CellEdgeId,
    provinces: [u32; 2],
    kind: InheritedStructureKindV0,
    incidence: &BTreeMap<u32, Vec<CellEdgeId>>,
    by_id: &BTreeMap<CellEdgeId, &InheritedStructureEdgeV0>,
) -> BTreeSet<CellEdgeId> {
    let mut component = BTreeSet::from([seed]);
    let mut frontier = vec![seed];
    while let Some(edge) = frontier.pop() {
        for &vertex in &by_id[&edge].vertices {
            if !continues_through(vertex, provinces, kind, incidence, by_id) {
                continue;
            }
            for &next in &incidence[&vertex] {
                if by_id[&next].provinces == provinces
                    && by_id[&next].kind == kind
                    && component.insert(next)
                {
                    frontier.push(next);
                }
            }
        }
    }
    component
}

/// Validate explicit geological relationships independently of geometric incidence.
pub fn validate_structure_relationships_v0(
    graph: &InheritedStructureGraphV0,
) -> Result<(), LithosphereInheritanceErrorV0> {
    if graph
        .segments
        .windows(2)
        .any(|pair| pair[0].id >= pair[1].id)
        || graph
            .relationships
            .windows(2)
            .any(|pair| pair[0].id >= pair[1].id)
    {
        return Err(LithosphereInheritanceErrorV0::InvalidManufacturedGraph);
    }
    let mut used_ends = BTreeSet::new();
    for relationship in &graph.relationships {
        let invalid =
            || LithosphereInheritanceErrorV0::InvalidStructureRelationship(relationship.id);
        match &relationship.topology {
            InheritedStructureRelationshipTopologyV0::Continuation { ends } => {
                if ends[0] >= ends[1]
                    || segment_end_vertex(graph, ends[0], relationship.id)?
                        != segment_end_vertex(graph, ends[1], relationship.id)?
                    || !compatible_primary_ends(graph, ends[0], ends[1])?
                {
                    return Err(invalid());
                }
                insert_unique_end(&mut used_ends, ends[0], relationship.id)?;
                insert_unique_end(&mut used_ends, ends[1], relationship.id)?;
            }
            InheritedStructureRelationshipTopologyV0::Junction { ends } => {
                if ends.len() < 3
                    || ends.windows(2).any(|pair| pair[0] >= pair[1])
                    || !ends_are_primary(graph, ends)?
                {
                    return Err(invalid());
                }
                let vertices = ends
                    .iter()
                    .map(|&endpoint| segment_end_vertex(graph, endpoint, relationship.id))
                    .collect::<Result<BTreeSet<_>, _>>()?;
                if vertices.len() != 1 {
                    return Err(invalid());
                }
                for &endpoint in ends {
                    insert_unique_end(&mut used_ends, endpoint, relationship.id)?;
                }
            }
            InheritedStructureRelationshipTopologyV0::OffsetTransfer {
                primary_ends,
                connector_segment_id,
            } => {
                if primary_ends[0] >= primary_ends[1]
                    || !compatible_primary_ends(graph, primary_ends[0], primary_ends[1])?
                {
                    return Err(invalid());
                }
                let primary_vertices = [
                    segment_end_vertex(graph, primary_ends[0], relationship.id)?,
                    segment_end_vertex(graph, primary_ends[1], relationship.id)?,
                ];
                if primary_vertices[0] == primary_vertices[1] {
                    return Err(invalid());
                }
                let connector = graph
                    .segments
                    .iter()
                    .find(|segment| segment.id == *connector_segment_id)
                    .ok_or(LithosphereInheritanceErrorV0::MissingStructureSegment(
                        *connector_segment_id,
                    ))?;
                if connector.kind != InheritedStructureKindV0::TransferLink || connector.closed {
                    return Err(invalid());
                }
                let connector_ends = [
                    InheritedStructureSegmentEndRefV0 {
                        segment_id: *connector_segment_id,
                        end: InheritedStructureSegmentEndV0::Start,
                    },
                    InheritedStructureSegmentEndRefV0 {
                        segment_id: *connector_segment_id,
                        end: InheritedStructureSegmentEndV0::End,
                    },
                ];
                let mut expected_vertices = primary_vertices;
                expected_vertices.sort_unstable();
                let mut connector_vertices = [
                    segment_end_vertex(graph, connector_ends[0], relationship.id)?,
                    segment_end_vertex(graph, connector_ends[1], relationship.id)?,
                ];
                connector_vertices.sort_unstable();
                if expected_vertices != connector_vertices {
                    return Err(invalid());
                }
                for endpoint in primary_ends.iter().copied().chain(connector_ends) {
                    insert_unique_end(&mut used_ends, endpoint, relationship.id)?;
                }
            }
            InheritedStructureRelationshipTopologyV0::CrossingUnlinked { branches } => {
                if branches[0][0] >= branches[0][1]
                    || branches[1][0] >= branches[1][1]
                    || branches[0] >= branches[1]
                    || !compatible_primary_ends(graph, branches[0][0], branches[0][1])?
                    || !compatible_primary_ends(graph, branches[1][0], branches[1][1])?
                {
                    return Err(invalid());
                }
                let endpoints: Vec<_> = branches.iter().flatten().copied().collect();
                if endpoints.iter().copied().collect::<BTreeSet<_>>().len() != 4
                    || endpoints
                        .iter()
                        .map(|endpoint| endpoint.segment_id)
                        .collect::<BTreeSet<_>>()
                        .len()
                        != 4
                {
                    return Err(invalid());
                }
                let vertices = endpoints
                    .iter()
                    .map(|&endpoint| segment_end_vertex(graph, endpoint, relationship.id))
                    .collect::<Result<BTreeSet<_>, _>>()?;
                if vertices.len() != 1 {
                    return Err(invalid());
                }
                for endpoint in endpoints {
                    insert_unique_end(&mut used_ends, endpoint, relationship.id)?;
                }
            }
        }
    }
    Ok(())
}

/// Return deterministic segment components induced only by explicit relationships.
pub fn structure_relationship_components_v0(
    graph: &InheritedStructureGraphV0,
) -> Result<Vec<Vec<u32>>, LithosphereInheritanceErrorV0> {
    validate_structure_relationships_v0(graph)?;
    let by_id: BTreeMap<_, _> = graph
        .segments
        .iter()
        .enumerate()
        .map(|(index, segment)| (segment.id, index))
        .collect();
    let mut parent: Vec<_> = (0..graph.segments.len()).collect();
    for relationship in &graph.relationships {
        match &relationship.topology {
            InheritedStructureRelationshipTopologyV0::Continuation { ends } => {
                union_segment_ids(&by_id, &mut parent, ends[0].segment_id, ends[1].segment_id)?;
            }
            InheritedStructureRelationshipTopologyV0::Junction { ends } => {
                for pair in ends.windows(2) {
                    union_segment_ids(&by_id, &mut parent, pair[0].segment_id, pair[1].segment_id)?;
                }
            }
            InheritedStructureRelationshipTopologyV0::OffsetTransfer {
                primary_ends,
                connector_segment_id,
            } => {
                union_segment_ids(
                    &by_id,
                    &mut parent,
                    primary_ends[0].segment_id,
                    *connector_segment_id,
                )?;
                union_segment_ids(
                    &by_id,
                    &mut parent,
                    primary_ends[1].segment_id,
                    *connector_segment_id,
                )?;
            }
            InheritedStructureRelationshipTopologyV0::CrossingUnlinked { branches } => {
                for branch in branches {
                    union_segment_ids(
                        &by_id,
                        &mut parent,
                        branch[0].segment_id,
                        branch[1].segment_id,
                    )?;
                }
            }
        }
    }
    let mut components = BTreeMap::<usize, Vec<u32>>::new();
    for segment in &graph.segments {
        let index = by_id[&segment.id];
        let root = find_root(&mut parent, index);
        components.entry(root).or_default().push(segment.id);
    }
    let mut components: Vec<_> = components.into_values().collect();
    for component in &mut components {
        component.sort_unstable();
    }
    components.sort();
    Ok(components)
}

fn compatible_primary_ends(
    graph: &InheritedStructureGraphV0,
    first: InheritedStructureSegmentEndRefV0,
    second: InheritedStructureSegmentEndRefV0,
) -> Result<bool, LithosphereInheritanceErrorV0> {
    let first_kind = structure_segment_kind(graph, first.segment_id)?;
    let second_kind = structure_segment_kind(graph, second.segment_id)?;
    Ok(first.segment_id != second.segment_id
        && first_kind == second_kind
        && matches!(
            first_kind,
            InheritedStructureKindV0::Suture | InheritedStructureKindV0::InheritedRift
        ))
}

fn ends_are_primary(
    graph: &InheritedStructureGraphV0,
    ends: &[InheritedStructureSegmentEndRefV0],
) -> Result<bool, LithosphereInheritanceErrorV0> {
    for endpoint in ends {
        if !matches!(
            structure_segment_kind(graph, endpoint.segment_id)?,
            InheritedStructureKindV0::Suture | InheritedStructureKindV0::InheritedRift
        ) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn structure_segment_kind(
    graph: &InheritedStructureGraphV0,
    segment_id: u32,
) -> Result<InheritedStructureKindV0, LithosphereInheritanceErrorV0> {
    graph
        .segments
        .iter()
        .find(|segment| segment.id == segment_id)
        .map(|segment| segment.kind)
        .ok_or(LithosphereInheritanceErrorV0::MissingStructureSegment(
            segment_id,
        ))
}

fn insert_unique_end(
    used: &mut BTreeSet<InheritedStructureSegmentEndRefV0>,
    endpoint: InheritedStructureSegmentEndRefV0,
    relationship_id: u32,
) -> Result<(), LithosphereInheritanceErrorV0> {
    if used.insert(endpoint) {
        Ok(())
    } else {
        Err(LithosphereInheritanceErrorV0::InvalidStructureRelationship(
            relationship_id,
        ))
    }
}

fn union_segment_ids(
    by_id: &BTreeMap<u32, usize>,
    parent: &mut [usize],
    first: u32,
    second: u32,
) -> Result<(), LithosphereInheritanceErrorV0> {
    let first =
        *by_id
            .get(&first)
            .ok_or(LithosphereInheritanceErrorV0::MissingStructureSegment(
                first,
            ))?;
    let second =
        *by_id
            .get(&second)
            .ok_or(LithosphereInheritanceErrorV0::MissingStructureSegment(
                second,
            ))?;
    let first_root = find_root(parent, first);
    let second_root = find_root(parent, second);
    if first_root != second_root {
        let (lower, higher) = if first_root < second_root {
            (first_root, second_root)
        } else {
            (second_root, first_root)
        };
        parent[higher] = lower;
    }
    Ok(())
}

fn find_root(parent: &mut [usize], mut index: usize) -> usize {
    while parent[index] != index {
        parent[index] = parent[parent[index]];
        index = parent[index];
    }
    index
}

fn tangent_toward(origin: Vec3, target: Vec3) -> Vec3 {
    (target - origin * origin.dot(target)).normalize_or_zero()
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn generated(seed: u64, cells: usize) -> (Tessellation, Crust, LithosphereInheritanceV0) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let tessellation = Tessellation::generate(cells, 1, &mut rng);
        let crust = Crust::generate(&tessellation, 3, 0.45, &mut rng);
        let inheritance = generate_lithosphere_inheritance_v0(
            seed,
            &tessellation,
            &crust,
            LithosphereInheritanceConfigV0 {
                target_province_area_km2: 8_000_000.0,
                maximum_provinces_per_craton: 8,
                minimum_average_cells_per_province: 8,
            },
        )
        .unwrap();
        (tessellation, crust, inheritance)
    }

    #[test]
    fn generation_is_deterministic_and_preserves_craton_membership() {
        let (tessellation, crust, first) = generated(77, 2_000);
        let second =
            generate_lithosphere_inheritance_v0(77, &tessellation, &crust, first.config).unwrap();
        assert_eq!(first, second);
        assert!(first.provinces.len() > crust.num_cratons);
        for cell in 0..tessellation.num_cells() {
            if crust.types[cell] == CrustType::Oceanic {
                assert_eq!(first.cell_province[cell], OCEANIC_BASEMENT_PROVINCE);
            } else {
                let province = &first.provinces[first.cell_province[cell] as usize];
                assert_eq!(province.craton_id, crust.cell_craton[cell]);
            }
        }
    }

    #[test]
    fn generation_rejects_inconsistent_crust_identity() {
        let (tessellation, mut crust, inheritance) = generated(78, 1_000);
        let ocean = crust
            .types
            .iter()
            .position(|&crust_type| crust_type == CrustType::Oceanic)
            .unwrap();
        crust.cell_craton[ocean] = 0;
        assert_eq!(
            generate_lithosphere_inheritance_v0(78, &tessellation, &crust, inheritance.config,),
            Err(LithosphereInheritanceErrorV0::CrustIdentityMismatch(ocean))
        );
    }

    #[test]
    fn every_generated_province_is_connected() {
        let (tessellation, _, inheritance) = generated(91, 2_000);
        for province in &inheritance.provinces {
            let members: BTreeSet<_> = inheritance
                .cell_province
                .iter()
                .enumerate()
                .filter_map(|(cell, &id)| (id == province.id).then_some(cell))
                .collect();
            let start = *members.iter().next().unwrap();
            let mut reached = BTreeSet::from([start]);
            let mut frontier = vec![start];
            while let Some(cell) = frontier.pop() {
                for &neighbor in tessellation.neighbors(cell) {
                    if members.contains(&neighbor) && reached.insert(neighbor) {
                        frontier.push(neighbor);
                    }
                }
            }
            assert_eq!(reached, members);
            assert_eq!(province.cell_count, members.len());
        }
    }

    #[test]
    fn exact_contacts_compile_and_query_without_terrain() {
        let (tessellation, crust, inheritance) = generated(123, 2_000);
        assert!(!inheritance.graph.edges.is_empty());
        assert!(inheritance.graph.relationships.is_empty());
        validate_structure_relationships_v0(&inheritance.graph).unwrap();
        let declared: f64 = inheritance
            .graph
            .segments
            .iter()
            .map(|segment| f64::from(segment.length_km))
            .sum();
        assert!(
            (declared - inheritance.graph.total_length_km).abs()
                <= inheritance.graph.total_length_km * 1e-6
        );
        for edge in &inheritance.graph.edges {
            assert_ne!(edge.provinces[0], edge.provinces[1]);
            assert_eq!(crust.types[edge.cells[0]], CrustType::Continental);
            assert_eq!(crust.types[edge.cells[1]], CrustType::Continental);
            assert_eq!(
                tessellation.shared_edge_vertices(edge.cells[0], edge.cells[1]),
                Some(edge.vertices)
            );
        }
        let edge = inheritance.graph.edges[0].id;
        let relationship =
            query_boundary_inheritance_v0(&tessellation, &inheritance, edge).unwrap();
        assert_eq!(
            relationship.kind,
            BoundaryInheritanceContactKindV0::Coincident
        );
        assert_eq!(relationship.minimum_tangent_angle_deg, Some(0.0));
        let reversed = CellEdgeId {
            cell_a: edge.cell_b,
            cell_b: edge.cell_a,
        };
        let reversed_relationship =
            query_boundary_inheritance_v0(&tessellation, &inheritance, reversed).unwrap();
        assert_eq!(reversed_relationship.boundary, edge);
        assert_eq!(reversed_relationship.kind, relationship.kind);
        assert_eq!(
            query_boundary_inheritance_v0(
                &tessellation,
                &inheritance,
                CellEdgeId {
                    cell_a: edge.cell_a,
                    cell_b: edge.cell_a,
                },
            ),
            Err(LithosphereInheritanceErrorV0::InvalidBoundary(CellEdgeId {
                cell_a: edge.cell_a,
                cell_b: edge.cell_a,
            }))
        );
        let out_of_range = CellEdgeId {
            cell_a: edge.cell_a,
            cell_b: tessellation.num_cells(),
        };
        assert_eq!(
            query_boundary_inheritance_v0(&tessellation, &inheritance, out_of_range),
            Err(LithosphereInheritanceErrorV0::InvalidBoundary(out_of_range))
        );

        let mut vertex_contact = None;
        let mut unrelated = None;
        'cells: for cell_a in 0..tessellation.num_cells() {
            for &cell_b in tessellation.neighbors(cell_a) {
                if cell_b <= cell_a {
                    continue;
                }
                let relationship = query_boundary_inheritance_v0(
                    &tessellation,
                    &inheritance,
                    CellEdgeId::new(cell_a, cell_b),
                )
                .unwrap();
                match relationship.kind {
                    BoundaryInheritanceContactKindV0::VertexContact => {
                        vertex_contact.get_or_insert(relationship);
                    }
                    BoundaryInheritanceContactKindV0::Unrelated => {
                        unrelated.get_or_insert(relationship);
                    }
                    BoundaryInheritanceContactKindV0::Coincident => {}
                }
                if vertex_contact.is_some() && unrelated.is_some() {
                    break 'cells;
                }
            }
        }
        let vertex_contact = vertex_contact.expect("generated graph has an exact vertex contact");
        assert!(!vertex_contact.shared_vertices.is_empty());
        assert!(!vertex_contact.structure_segment_ids.is_empty());
        assert!(vertex_contact.minimum_tangent_angle_deg.is_some());
        let unrelated = unrelated.expect("generated graph leaves unrelated adjacencies");
        assert!(unrelated.shared_vertices.is_empty());
        assert!(unrelated.structure_segment_ids.is_empty());
        assert_eq!(unrelated.minimum_tangent_angle_deg, None);
    }

    #[test]
    fn candidate_chain_compiler_keeps_geometric_incidence_non_geological() {
        let edge = |a, b, vertices, provinces| InheritedStructureEdgeV0 {
            id: CellEdgeId::new(a, b),
            cells: [a.min(b), a.max(b)],
            vertices,
            endpoints: [Vec3::X, Vec3::Y],
            length_km: 10.0,
            provinces,
            kind: InheritedStructureKindV0::BasementContact,
        };
        let continuation = compile_graph_from_edges(vec![
            edge(0, 1, [10, 11], [0, 1]),
            edge(1, 2, [11, 12], [0, 1]),
        ])
        .unwrap();
        assert_eq!(continuation.segments.len(), 1);
        assert_eq!(continuation.segments[0].source_edges.len(), 2);
        assert_eq!(continuation.incidences.len(), 2);
        assert!(continuation
            .incidences
            .iter()
            .all(|incidence| incidence.kind == InheritedStructureIncidenceKindV0::Tip));
        assert!(continuation.relationships.is_empty());

        let junction = compile_graph_from_edges(vec![
            edge(0, 1, [10, 11], [0, 1]),
            edge(2, 3, [11, 12], [0, 2]),
            edge(4, 5, [11, 13], [1, 2]),
        ])
        .unwrap();
        assert_eq!(junction.segments.len(), 3);
        let incidence = junction
            .incidences
            .iter()
            .find(|incidence| incidence.vertex == 11)
            .unwrap();
        assert_eq!(
            incidence.kind,
            InheritedStructureIncidenceKindV0::MultiTrace
        );
        assert_eq!(incidence.incident_segments.len(), 3);
        assert!(junction.relationships.is_empty());
    }

    fn manufactured_segment(
        id: u32,
        start: u32,
        end: u32,
        kind: InheritedStructureKindV0,
    ) -> InheritedStructureSegmentV0 {
        InheritedStructureSegmentV0 {
            id,
            kind,
            provinces: [0, 1],
            source_edges: vec![CellEdgeId::new(id as usize * 2, id as usize * 2 + 1)],
            vertices_in_order: vec![start, end],
            closed: false,
            length_km: 10.0,
        }
    }

    fn manufactured_graph(
        segments: Vec<InheritedStructureSegmentV0>,
        relationships: Vec<InheritedStructureRelationshipV0>,
    ) -> InheritedStructureGraphV0 {
        InheritedStructureGraphV0 {
            edges: Vec::new(),
            segments,
            incidences: Vec::new(),
            relationships,
            total_length_km: 0.0,
        }
    }

    fn endpoint(
        segment_id: u32,
        end: InheritedStructureSegmentEndV0,
    ) -> InheritedStructureSegmentEndRefV0 {
        InheritedStructureSegmentEndRefV0 { segment_id, end }
    }

    #[test]
    fn explicit_crossing_and_junction_change_connectivity_with_identical_geometry() {
        let segments = vec![
            manufactured_segment(0, 10, 11, InheritedStructureKindV0::InheritedRift),
            manufactured_segment(1, 11, 12, InheritedStructureKindV0::InheritedRift),
            manufactured_segment(2, 13, 11, InheritedStructureKindV0::InheritedRift),
            manufactured_segment(3, 11, 14, InheritedStructureKindV0::InheritedRift),
        ];
        let crossing = manufactured_graph(
            segments.clone(),
            vec![InheritedStructureRelationshipV0 {
                id: 0,
                topology: InheritedStructureRelationshipTopologyV0::CrossingUnlinked {
                    branches: [
                        [
                            endpoint(0, InheritedStructureSegmentEndV0::End),
                            endpoint(1, InheritedStructureSegmentEndV0::Start),
                        ],
                        [
                            endpoint(2, InheritedStructureSegmentEndV0::End),
                            endpoint(3, InheritedStructureSegmentEndV0::Start),
                        ],
                    ],
                },
            }],
        );
        assert_eq!(
            structure_relationship_components_v0(&crossing).unwrap(),
            vec![vec![0, 1], vec![2, 3]]
        );
        let topology = contact_topology_metadata(&crossing, &[11], &[0]).unwrap();
        assert_eq!(topology.relationship_ids, vec![0]);
        assert_eq!(
            topology.relationship_kinds,
            vec![InheritedStructureRelationshipKindV0::CrossingUnlinked]
        );

        let mut third_trace = crossing.clone();
        third_trace.segments.push(manufactured_segment(
            4,
            11,
            15,
            InheritedStructureKindV0::InheritedRift,
        ));
        let unrelated_topology = contact_topology_metadata(&third_trace, &[11], &[4]).unwrap();
        assert!(unrelated_topology.relationship_ids.is_empty());

        let junction = manufactured_graph(
            segments,
            vec![InheritedStructureRelationshipV0 {
                id: 0,
                topology: InheritedStructureRelationshipTopologyV0::Junction {
                    ends: vec![
                        endpoint(0, InheritedStructureSegmentEndV0::End),
                        endpoint(1, InheritedStructureSegmentEndV0::Start),
                        endpoint(2, InheritedStructureSegmentEndV0::End),
                        endpoint(3, InheritedStructureSegmentEndV0::Start),
                    ],
                },
            }],
        );
        assert_eq!(
            structure_relationship_components_v0(&junction).unwrap(),
            vec![vec![0, 1, 2, 3]]
        );

        let reused_segment = manufactured_graph(
            vec![
                manufactured_segment(0, 11, 11, InheritedStructureKindV0::InheritedRift),
                manufactured_segment(1, 11, 12, InheritedStructureKindV0::InheritedRift),
                manufactured_segment(2, 11, 13, InheritedStructureKindV0::InheritedRift),
            ],
            vec![InheritedStructureRelationshipV0 {
                id: 0,
                topology: InheritedStructureRelationshipTopologyV0::CrossingUnlinked {
                    branches: [
                        [
                            endpoint(0, InheritedStructureSegmentEndV0::Start),
                            endpoint(1, InheritedStructureSegmentEndV0::Start),
                        ],
                        [
                            endpoint(0, InheritedStructureSegmentEndV0::End),
                            endpoint(2, InheritedStructureSegmentEndV0::Start),
                        ],
                    ],
                },
            }],
        );
        assert_eq!(
            validate_structure_relationships_v0(&reused_segment),
            Err(LithosphereInheritanceErrorV0::InvalidStructureRelationship(
                0
            ))
        );
    }

    #[test]
    fn continuation_and_offset_transfer_require_explicit_valid_links() {
        let continuation = manufactured_graph(
            vec![
                manufactured_segment(0, 10, 11, InheritedStructureKindV0::Suture),
                manufactured_segment(1, 11, 12, InheritedStructureKindV0::Suture),
            ],
            vec![InheritedStructureRelationshipV0 {
                id: 0,
                topology: InheritedStructureRelationshipTopologyV0::Continuation {
                    ends: [
                        endpoint(0, InheritedStructureSegmentEndV0::End),
                        endpoint(1, InheritedStructureSegmentEndV0::Start),
                    ],
                },
            }],
        );
        assert_eq!(
            structure_relationship_components_v0(&continuation).unwrap(),
            vec![vec![0, 1]]
        );

        let transfer = manufactured_graph(
            vec![
                manufactured_segment(0, 10, 11, InheritedStructureKindV0::InheritedRift),
                manufactured_segment(1, 20, 21, InheritedStructureKindV0::InheritedRift),
                manufactured_segment(2, 11, 20, InheritedStructureKindV0::TransferLink),
            ],
            vec![InheritedStructureRelationshipV0 {
                id: 0,
                topology: InheritedStructureRelationshipTopologyV0::OffsetTransfer {
                    primary_ends: [
                        endpoint(0, InheritedStructureSegmentEndV0::End),
                        endpoint(1, InheritedStructureSegmentEndV0::Start),
                    ],
                    connector_segment_id: 2,
                },
            }],
        );
        assert_eq!(
            structure_relationship_components_v0(&transfer).unwrap(),
            vec![vec![0, 1, 2]]
        );

        let mut invalid_transfer = transfer;
        invalid_transfer.segments[2].vertices_in_order = vec![11, 30];
        assert_eq!(
            validate_structure_relationships_v0(&invalid_transfer),
            Err(LithosphereInheritanceErrorV0::InvalidStructureRelationship(
                0
            ))
        );
    }

    fn manufactured_boundary(
        kind: BoundaryKind,
        convergence: f32,
        types: [CrustType; 2],
    ) -> PlateBoundaryEdge {
        PlateBoundaryEdge {
            cell_a: 0,
            cell_b: 1,
            plate_a: 0,
            plate_b: 1,
            type_a: types[0],
            type_b: types[1],
            boundary_point: Vec3::X,
            edge_length: 0.01,
            convergence,
            shear: 0.0,
            relative_speed: convergence.abs(),
            kind,
            subduction: None,
        }
    }

    #[test]
    fn collision_and_rift_consumers_share_geology_but_keep_distinct_applications() {
        let graph = manufactured_graph(
            vec![manufactured_segment(
                0,
                10,
                11,
                InheritedStructureKindV0::InheritedRift,
            )],
            Vec::new(),
        );
        let relationship = BoundaryInheritanceRelationshipV0 {
            boundary: CellEdgeId::new(0, 1),
            kind: BoundaryInheritanceContactKindV0::Coincident,
            shared_vertices: vec![10, 11],
            structure_segment_ids: vec![0],
            geometric_incidence_ids: Vec::new(),
            structure_relationship_ids: Vec::new(),
            structure_relationship_kinds: Vec::new(),
            minimum_tangent_angle_deg: Some(0.0),
        };
        let collision = assess_plate_boundary_inheritance_v0(
            &manufactured_boundary(
                BoundaryKind::Convergent,
                0.2,
                [CrustType::Continental, CrustType::Continental],
            ),
            &relationship,
            &graph,
        )
        .unwrap();
        let rift = assess_plate_boundary_inheritance_v0(
            &manufactured_boundary(
                BoundaryKind::Divergent,
                -0.2,
                [CrustType::Continental, CrustType::Continental],
            ),
            &relationship,
            &graph,
        )
        .unwrap();
        assert_eq!(
            collision.application,
            BoundaryInheritanceApplicationV0::ContinentalCollision
        );
        assert_eq!(
            rift.application,
            BoundaryInheritanceApplicationV0::ContinentalRifting
        );
        assert_eq!(
            collision.geology,
            BoundaryInheritanceGeologyV0::NamedInheritedRift
        );
        assert_eq!(rift.geology, collision.geology);
        assert_eq!(rift.geometric_contact, collision.geometric_contact);

        let collision_boundary = manufactured_boundary(
            BoundaryKind::Convergent,
            0.2,
            [CrustType::Continental, CrustType::Continental],
        );
        let mut stale_boundary = relationship.clone();
        stale_boundary.boundary = CellEdgeId::new(0, 2);
        assert_eq!(
            assess_plate_boundary_inheritance_v0(&collision_boundary, &stale_boundary, &graph,),
            Err(LithosphereInheritanceErrorV0::InvalidBoundary(
                CellEdgeId::new(0, 2)
            ))
        );
        let mut forged_topology = relationship.clone();
        forged_topology.structure_relationship_kinds =
            vec![InheritedStructureRelationshipKindV0::Junction];
        assert_eq!(
            assess_plate_boundary_inheritance_v0(&collision_boundary, &forged_topology, &graph,),
            Err(LithosphereInheritanceErrorV0::InvalidRelationshipMetadata)
        );

        let spreading = assess_plate_boundary_inheritance_v0(
            &manufactured_boundary(
                BoundaryKind::Divergent,
                -0.2,
                [CrustType::Oceanic, CrustType::Oceanic],
            ),
            &relationship,
            &graph,
        )
        .unwrap();
        assert_eq!(
            spreading.application,
            BoundaryInheritanceApplicationV0::OceanicSpreading
        );
        let locally_closing = assess_plate_boundary_inheritance_v0(
            &manufactured_boundary(
                BoundaryKind::Divergent,
                0.2,
                [CrustType::Continental, CrustType::Continental],
            ),
            &relationship,
            &graph,
        )
        .unwrap();
        assert_eq!(
            locally_closing.application,
            BoundaryInheritanceApplicationV0::Ineligible
        );
    }

    #[test]
    fn candidate_basement_contact_cannot_masquerade_as_geological_linkage() {
        let graph = manufactured_graph(
            vec![manufactured_segment(
                0,
                10,
                11,
                InheritedStructureKindV0::BasementContact,
            )],
            Vec::new(),
        );
        let relationship = BoundaryInheritanceRelationshipV0 {
            boundary: CellEdgeId::new(0, 1),
            kind: BoundaryInheritanceContactKindV0::Coincident,
            shared_vertices: vec![10, 11],
            structure_segment_ids: vec![0],
            geometric_incidence_ids: Vec::new(),
            structure_relationship_ids: Vec::new(),
            structure_relationship_kinds: Vec::new(),
            minimum_tangent_angle_deg: Some(0.0),
        };
        let assessment = assess_plate_boundary_inheritance_v0(
            &manufactured_boundary(
                BoundaryKind::Divergent,
                -0.2,
                [CrustType::Continental, CrustType::Continental],
            ),
            &relationship,
            &graph,
        )
        .unwrap();
        assert_eq!(
            assessment.geology,
            BoundaryInheritanceGeologyV0::CandidateBasementContact
        );

        let invalid = manufactured_graph(
            vec![
                manufactured_segment(0, 10, 11, InheritedStructureKindV0::BasementContact),
                manufactured_segment(1, 11, 12, InheritedStructureKindV0::BasementContact),
            ],
            vec![InheritedStructureRelationshipV0 {
                id: 0,
                topology: InheritedStructureRelationshipTopologyV0::Continuation {
                    ends: [
                        endpoint(0, InheritedStructureSegmentEndV0::End),
                        endpoint(1, InheritedStructureSegmentEndV0::Start),
                    ],
                },
            }],
        );
        assert_eq!(
            validate_structure_relationships_v0(&invalid),
            Err(LithosphereInheritanceErrorV0::InvalidStructureRelationship(
                0
            ))
        );
    }
}
