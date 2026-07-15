//! Packet-facing O0b correspondence assembly.
//!
//! This layer extracts the two registered object families from validated O0b
//! packet cores, performs the shared cell-intersection pass, and feeds the
//! frozen geometry and assignment kernels in [`super::correspondence`].

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use bincode::Options;
use glam::DVec3;
use serde::Serialize;

use super::*;

const REFERENCE_DRAINAGE_SUPPORT_KM2: f64 = 2_000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketCellClassV0 {
    EligibleObject(u32),
    IneligibleHighland {
        peak_id: u32,
        status: SupportStatusV0,
    },
    HighlandBackground,
    Portal(u32),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PacketAreaObjectV0 {
    pub id: u32,
    pub status: SupportStatusV0,
    pub nested_cells: Vec<u32>,
    pub exclusive_cells: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PacketAreaPopulationV0 {
    pub family: ObjectFamilyV0,
    pub objects: Vec<PacketAreaObjectV0>,
    pub cell_class: Vec<PacketCellClassV0>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PacketLineObjectV0 {
    pub id: u32,
    pub segments: Vec<LineSegmentInputV0>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PacketCorrespondenceErrorV0 {
    Incompatible(&'static str),
    Packet(String),
    MissingReferenceDrainageScale,
    DuplicateReferenceDrainageScale,
    InvalidObjectReference { family: ObjectFamilyV0, id: u32 },
    DuplicateObject { family: ObjectFamilyV0, id: u32 },
    InvalidExclusiveSupport { family: ObjectFamilyV0, cell: u32 },
    InvalidHierarchy(u32),
    InvalidReach(u32),
    AreaLedgerFailure { family: ObjectFamilyV0, id: u32 },
    Numerical(&'static str),
    Kernel(CorrespondenceErrorV0),
    Serialization(String),
}

impl fmt::Display for PacketCorrespondenceErrorV0 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for PacketCorrespondenceErrorV0 {}

impl From<CorrespondenceErrorV0> for PacketCorrespondenceErrorV0 {
    fn from(value: CorrespondenceErrorV0) -> Self {
        Self::Kernel(value)
    }
}

/// Extract nested and exclusive reference-highland support from frozen S0.
pub fn extract_highland_population_v0(
    packet: &LandformObjectPacketCoreV0,
) -> Result<PacketAreaPopulationV0, PacketCorrespondenceErrorV0> {
    let hierarchy = &packet.surface_hierarchy;
    let cell_count = packet.graph.cell_count();
    if hierarchy.cell_peak_owner.len() != cell_count {
        return Err(PacketCorrespondenceErrorV0::Incompatible(
            "surface hierarchy cell count",
        ));
    }
    let references = hierarchy
        .populations
        .reference
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if references.len() != hierarchy.populations.reference.len() {
        return Err(PacketCorrespondenceErrorV0::DuplicateObject {
            family: ObjectFamilyV0::Highland,
            id: 0,
        });
    }
    for &id in &references {
        if hierarchy
            .peaks
            .get(id as usize)
            .is_none_or(|peak| peak.id != id)
        {
            return Err(PacketCorrespondenceErrorV0::InvalidObjectReference {
                family: ObjectFamilyV0::Highland,
                id,
            });
        }
    }

    let mut exclusive = BTreeMap::<u32, Vec<u32>>::new();
    for (cell, owner) in hierarchy.cell_peak_owner.iter().copied().enumerate() {
        let Some(mut peak_id) = owner else { continue };
        let mut visited = BTreeSet::new();
        loop {
            if !visited.insert(peak_id) {
                return Err(PacketCorrespondenceErrorV0::InvalidHierarchy(peak_id));
            }
            if references.contains(&peak_id) {
                exclusive.entry(peak_id).or_default().push(cell as u32);
                break;
            }
            let peak = hierarchy
                .peaks
                .get(peak_id as usize)
                .filter(|peak| peak.id == peak_id)
                .ok_or(PacketCorrespondenceErrorV0::InvalidHierarchy(peak_id))?;
            let Some(parent) = peak.parent_peak else {
                break;
            };
            peak_id = parent;
        }
    }

    let mut objects = Vec::with_capacity(references.len());
    for &id in &references {
        let peak = &hierarchy.peaks[id as usize];
        let ambiguous = highland_support_is_ambiguous(hierarchy, id, &references)?;
        let exclusive_cells = exclusive.remove(&id).unwrap_or_default();
        let status = if ambiguous {
            SupportStatusV0::HierarchyAmbiguousSupport
        } else if exclusive_cells.is_empty() {
            SupportStatusV0::NoExclusiveSupport
        } else {
            SupportStatusV0::Eligible
        };
        let mut nested_cells = peak.footprint_members.clone();
        nested_cells.sort_unstable();
        nested_cells.dedup();
        if nested_cells.is_empty() || nested_cells.iter().any(|&cell| cell as usize >= cell_count) {
            return Err(PacketCorrespondenceErrorV0::InvalidHierarchy(id));
        }
        objects.push(PacketAreaObjectV0 {
            id,
            status,
            nested_cells,
            exclusive_cells,
        });
    }
    objects.sort_by_key(|object| object.id);

    let mut cell_class = vec![PacketCellClassV0::HighlandBackground; cell_count];
    for object in &objects {
        for &cell in &object.exclusive_cells {
            let class = match object.status {
                SupportStatusV0::Eligible => PacketCellClassV0::EligibleObject(object.id),
                SupportStatusV0::HierarchyAmbiguousSupport => {
                    PacketCellClassV0::IneligibleHighland {
                        peak_id: object.id,
                        status: object.status,
                    }
                }
                SupportStatusV0::NoExclusiveSupport => continue,
                SupportStatusV0::NoPositiveOverlap => {
                    return Err(PacketCorrespondenceErrorV0::InvalidHierarchy(object.id));
                }
            };
            if cell_class[cell as usize] != PacketCellClassV0::HighlandBackground {
                return Err(PacketCorrespondenceErrorV0::InvalidExclusiveSupport {
                    family: ObjectFamilyV0::Highland,
                    cell,
                });
            }
            cell_class[cell as usize] = class;
        }
    }
    Ok(PacketAreaPopulationV0 {
        family: ObjectFamilyV0::Highland,
        objects,
        cell_class,
    })
}

fn highland_support_is_ambiguous(
    hierarchy: &SurfaceHierarchyV0,
    id: u32,
    references: &BTreeSet<u32>,
) -> Result<bool, PacketCorrespondenceErrorV0> {
    let edge_ambiguous = |peak_id: u32| -> Result<bool, PacketCorrespondenceErrorV0> {
        let peak = hierarchy
            .peaks
            .get(peak_id as usize)
            .filter(|peak| peak.id == peak_id)
            .ok_or(PacketCorrespondenceErrorV0::InvalidHierarchy(peak_id))?;
        let saddle_ambiguous = peak.key_saddle.is_some_and(|saddle_id| {
            hierarchy
                .saddles
                .get(saddle_id as usize)
                .is_some_and(|saddle| saddle.id == saddle_id && saddle.equal_elder_ambiguous)
        });
        Ok(peak.equal_elder_ambiguous || saddle_ambiguous)
    };
    if edge_ambiguous(id)? {
        return Ok(true);
    }
    // The nearest retained-ancestor path of h.
    let mut cursor = id;
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(cursor) {
            return Err(PacketCorrespondenceErrorV0::InvalidHierarchy(id));
        }
        let peak = &hierarchy.peaks[cursor as usize];
        let Some(parent) = peak.parent_peak else {
            break;
        };
        if edge_ambiguous(cursor)? {
            return Ok(true);
        }
        cursor = parent;
        if references.contains(&cursor) {
            break;
        }
    }
    // Every retained-descendant path removed from F(h).
    for &candidate in references {
        if candidate == id {
            continue;
        }
        let mut cursor = candidate;
        let mut path = BTreeSet::new();
        let mut path_ambiguous = false;
        loop {
            if !path.insert(cursor) {
                return Err(PacketCorrespondenceErrorV0::InvalidHierarchy(candidate));
            }
            if cursor == id {
                if path_ambiguous {
                    return Ok(true);
                }
                break;
            }
            path_ambiguous |= edge_ambiguous(cursor)?;
            let peak = hierarchy
                .peaks
                .get(cursor as usize)
                .filter(|peak| peak.id == cursor)
                .ok_or(PacketCorrespondenceErrorV0::InvalidHierarchy(cursor))?;
            let Some(parent) = peak.parent_peak else {
                break;
            };
            cursor = parent;
        }
    }
    Ok(false)
}

/// Extract the exact 2,000 km2 D0 catchment population.
pub fn extract_drainage_population_v0(
    packet: &LandformObjectPacketCoreV0,
) -> Result<PacketAreaPopulationV0, PacketCorrespondenceErrorV0> {
    let mut scales = packet
        .drainage
        .scales
        .iter()
        .filter(|scale| scale.support_threshold_km2 == REFERENCE_DRAINAGE_SUPPORT_KM2);
    let scale = scales
        .next()
        .ok_or(PacketCorrespondenceErrorV0::MissingReferenceDrainageScale)?;
    if scales.next().is_some() {
        return Err(PacketCorrespondenceErrorV0::DuplicateReferenceDrainageScale);
    }
    let cell_count = packet.graph.cell_count();
    if scale.basin_graph.exclusive_owner.len() != cell_count {
        return Err(PacketCorrespondenceErrorV0::Incompatible(
            "drainage owner cell count",
        ));
    }
    let reaches = scale
        .reach_graph
        .reaches
        .iter()
        .map(|reach| reach.id)
        .collect::<BTreeSet<_>>();
    let catchments = scale
        .basin_graph
        .catchments
        .iter()
        .map(|catchment| (catchment.reach_id, catchment))
        .collect::<BTreeMap<_, _>>();
    if reaches.len() != scale.reach_graph.reaches.len()
        || catchments.len() != scale.basin_graph.catchments.len()
        || reaches.iter().any(|id| !catchments.contains_key(id))
        || catchments.keys().any(|id| !reaches.contains(id))
    {
        return Err(PacketCorrespondenceErrorV0::Incompatible(
            "reference reach/catchment bijection",
        ));
    }

    let mut exclusive = BTreeMap::<u32, Vec<u32>>::new();
    let mut cell_class = Vec::with_capacity(cell_count);
    for (cell, owner) in scale
        .basin_graph
        .exclusive_owner
        .iter()
        .copied()
        .enumerate()
    {
        match owner {
            IncrementalCatchmentOwnerV0::Reach(id) => {
                if !reaches.contains(&id) {
                    return Err(PacketCorrespondenceErrorV0::InvalidObjectReference {
                        family: ObjectFamilyV0::DrainageNode,
                        id,
                    });
                }
                exclusive.entry(id).or_default().push(cell as u32);
                cell_class.push(PacketCellClassV0::EligibleObject(id));
            }
            IncrementalCatchmentOwnerV0::Portal(id) => {
                cell_class.push(PacketCellClassV0::Portal(id));
            }
        }
    }
    let parent = catchments
        .iter()
        .map(|(&id, catchment)| (id, catchment.parent_reach))
        .collect::<BTreeMap<_, _>>();
    let mut objects = Vec::with_capacity(reaches.len());
    for &id in &reaches {
        let exclusive_cells = exclusive.remove(&id).unwrap_or_default();
        if exclusive_cells.is_empty() {
            return Err(PacketCorrespondenceErrorV0::InvalidReach(id));
        }
        let mut nested_cells = Vec::new();
        for (cell, owner) in scale
            .basin_graph
            .exclusive_owner
            .iter()
            .copied()
            .enumerate()
        {
            let IncrementalCatchmentOwnerV0::Reach(mut owner_id) = owner else {
                continue;
            };
            let mut visited = BTreeSet::new();
            loop {
                if !visited.insert(owner_id) {
                    return Err(PacketCorrespondenceErrorV0::InvalidReach(owner_id));
                }
                if owner_id == id {
                    nested_cells.push(cell as u32);
                    break;
                }
                let Some(Some(next)) = parent.get(&owner_id).copied() else {
                    break;
                };
                owner_id = next;
            }
        }
        objects.push(PacketAreaObjectV0 {
            id,
            status: SupportStatusV0::Eligible,
            nested_cells,
            exclusive_cells,
        });
    }
    objects.sort_by_key(|object| object.id);
    Ok(PacketAreaPopulationV0 {
        family: ObjectFamilyV0::DrainageNode,
        objects,
        cell_class,
    })
}

/// Validate the bounded common-population correspondence preconditions.
pub fn validate_correspondence_pair_v0(
    source: &LandformObjectPacketCoreV0,
    target: &LandformObjectPacketCoreV0,
    config: CorrespondenceConfigV0,
) -> Result<(), PacketCorrespondenceErrorV0> {
    config.validate()?;
    for packet in [source, target] {
        if packet.schema_version != O0B_PACKET_SCHEMA_VERSION
            || packet.hash_version != O0B_PACKET_HASH_VERSION
        {
            return Err(PacketCorrespondenceErrorV0::Incompatible(
                "packet schema/hash version",
            ));
        }
        let computed = landform_object_packet_hash_v0(packet)
            .map_err(|error| PacketCorrespondenceErrorV0::Packet(error.to_string()))?;
        if computed != packet.derived_common_packet_hash {
            return Err(PacketCorrespondenceErrorV0::Incompatible("packet hash"));
        }
        if packet.graph.domain != EvaluationDomainV0::Planar {
            return Err(PacketCorrespondenceErrorV0::Incompatible("planar domain"));
        }
        let PacketGeometryIdentityV0::LandscapeRegularPlanar {
            nominal_spacing_km, ..
        } = packet.geometry_identity
        else {
            return Err(PacketCorrespondenceErrorV0::Incompatible(
                "regular planar geometry",
            ));
        };
        if ![2.0, 4.0, 8.0].contains(&nominal_spacing_km) {
            return Err(PacketCorrespondenceErrorV0::Incompatible(
                "registered nominal spacing",
            ));
        }
    }
    if source.population != target.population {
        return Err(PacketCorrespondenceErrorV0::Incompatible(
            "common evaluation population",
        ));
    }
    let source_graph_hash = match source.geometry_identity {
        PacketGeometryIdentityV0::LandscapeRegularPlanar {
            canonical_graph_hash,
            ..
        } => canonical_graph_hash,
        _ => unreachable!("checked above"),
    };
    let target_graph_hash = match target.geometry_identity {
        PacketGeometryIdentityV0::LandscapeRegularPlanar {
            canonical_graph_hash,
            ..
        } => canonical_graph_hash,
        _ => unreachable!("checked above"),
    };
    if source_graph_hash == target_graph_hash {
        if source.graph != target.graph || source.scored_cell != target.scored_cell {
            return Err(PacketCorrespondenceErrorV0::Incompatible(
                "same-mesh graph/scored mask",
            ));
        }
        if matches!(
            source.population.runoff_policy,
            RunoffPolicyV0::ExactSameMeshArrayV0 { .. }
        ) && source.local_runoff_supply != target.local_runoff_supply
        {
            return Err(PacketCorrespondenceErrorV0::Incompatible(
                "same-mesh runoff array",
            ));
        }
    } else {
        if !source.scored_cell.iter().all(|&scored| scored)
            || !target.scored_cell.iter().all(|&scored| scored)
        {
            return Err(PacketCorrespondenceErrorV0::Incompatible(
                "cross-mesh whole scored support",
            ));
        }
        if matches!(
            source.population.runoff_policy,
            RunoffPolicyV0::ExactSameMeshArrayV0 { .. }
        ) {
            return Err(PacketCorrespondenceErrorV0::Incompatible(
                "cross-mesh reconstructible runoff",
            ));
        }
    }
    Ok(())
}

/// Build full D0 centre-to-receiver polylines with authoritative D0 measures.
pub fn extract_reference_reach_lines_v0(
    packet: &LandformObjectPacketCoreV0,
) -> Result<Vec<PacketLineObjectV0>, PacketCorrespondenceErrorV0> {
    let scale = reference_drainage_scale(packet)?;
    let graph = &packet.graph;
    let routing = &packet.drainage.routing;
    if routing.receiver.len() != graph.cell_count()
        || routing.segment_length_km.len() != graph.cell_count()
    {
        return Err(PacketCorrespondenceErrorV0::Incompatible(
            "routing cell count",
        ));
    }
    let covering_radii = (0..graph.cell_count())
        .map(|cell| {
            let center = graph.cell_center_km[cell];
            graph
                .polygon(cell)
                .iter()
                .map(|&point| center.distance(point))
                .max_by(f64::total_cmp)
                .filter(|radius| radius.is_finite() && *radius > 0.0)
                .ok_or(PacketCorrespondenceErrorV0::InvalidReach(cell as u32))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut output = Vec::with_capacity(scale.reach_graph.reaches.len());
    let mut seen = BTreeSet::new();
    for reach in &scale.reach_graph.reaches {
        if !seen.insert(reach.id) || reach.cells.is_empty() {
            return Err(PacketCorrespondenceErrorV0::InvalidReach(reach.id));
        }
        let mut segments = Vec::with_capacity(reach.cells.len());
        for &donor in &reach.cells {
            let donor_index = donor as usize;
            let start = *graph
                .cell_center_km
                .get(donor_index)
                .ok_or(PacketCorrespondenceErrorV0::InvalidReach(reach.id))?;
            let (end, local_radius) = match routing
                .receiver
                .get(donor_index)
                .copied()
                .ok_or(PacketCorrespondenceErrorV0::InvalidReach(reach.id))?
            {
                DrainageReceiverV0::Cell { cell, .. } => {
                    let receiver = cell as usize;
                    let end = *graph
                        .cell_center_km
                        .get(receiver)
                        .ok_or(PacketCorrespondenceErrorV0::InvalidReach(reach.id))?;
                    (
                        end,
                        covering_radii[donor_index].max(covering_radii[receiver]),
                    )
                }
                DrainageReceiverV0::Portal {
                    boundary_segment,
                    portal_id,
                } => {
                    let boundary = graph
                        .boundary_segments
                        .get(boundary_segment as usize)
                        .filter(|boundary| {
                            matches!(
                                boundary.condition,
                                EvaluationBoundaryConditionV0::OpenBaseLevel {
                                    portal_id: id,
                                    ..
                                } if id == portal_id
                            )
                        })
                        .ok_or(PacketCorrespondenceErrorV0::InvalidReach(reach.id))?;
                    (
                        0.5 * (boundary.endpoints_km[0] + boundary.endpoints_km[1]),
                        covering_radii[donor_index],
                    )
                }
            };
            let measure_length_km = *routing
                .segment_length_km
                .get(donor_index)
                .ok_or(PacketCorrespondenceErrorV0::InvalidReach(reach.id))?;
            if !start.is_finite()
                || !end.is_finite()
                || start == end
                || !measure_length_km.is_finite()
                || measure_length_km <= 0.0
            {
                return Err(PacketCorrespondenceErrorV0::InvalidReach(reach.id));
            }
            segments.push(LineSegmentInputV0 {
                endpoints_km: [start, end],
                measure_length_km,
                local_radius_km: local_radius,
            });
        }
        let reproduced = compensated_sum(segments.iter().map(|segment| segment.measure_length_km))?;
        if reproduced.to_bits() != reach.physical_length_km.to_bits() {
            return Err(PacketCorrespondenceErrorV0::InvalidReach(reach.id));
        }
        output.push(PacketLineObjectV0 {
            id: reach.id,
            segments,
        });
    }
    output.sort_by_key(|line| line.id);
    Ok(output)
}

fn reference_drainage_scale(
    packet: &LandformObjectPacketCoreV0,
) -> Result<&DrainageScaleV0, PacketCorrespondenceErrorV0> {
    let mut scales = packet
        .drainage
        .scales
        .iter()
        .filter(|scale| scale.support_threshold_km2 == REFERENCE_DRAINAGE_SUPPORT_KM2);
    let scale = scales
        .next()
        .ok_or(PacketCorrespondenceErrorV0::MissingReferenceDrainageScale)?;
    if scales.next().is_some() {
        return Err(PacketCorrespondenceErrorV0::DuplicateReferenceDrainageScale);
    }
    Ok(scale)
}

fn compensated_sum(
    values: impl IntoIterator<Item = f64>,
) -> Result<f64, PacketCorrespondenceErrorV0> {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for value in values {
        let adjusted = value - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
        if !sum.is_finite() || !correction.is_finite() {
            return Err(PacketCorrespondenceErrorV0::Numerical("compensated sum"));
        }
    }
    Ok(if sum == 0.0 { 0.0 } else { sum })
}

#[derive(Clone)]
struct CellGeometryV0 {
    area_km2: f64,
    centroid_km: DVec3,
    key: Vec<DVec3>,
    bounds: BoundsV0,
}

#[derive(Clone, Copy)]
struct BoundsV0 {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

impl BoundsV0 {
    fn overlaps(self, other: Self) -> bool {
        self.min_x <= other.max_x
            && other.min_x <= self.max_x
            && self.min_y <= other.max_y
            && other.min_y <= self.max_y
    }
}

#[derive(Clone)]
struct CellIntersectionV0 {
    source_cell: u32,
    target_cell: u32,
    area_km2: f64,
    low_key: Vec<DVec3>,
    high_key: Vec<DVec3>,
}

struct SharedAreaEvidenceV0 {
    source_cells: Vec<CellGeometryV0>,
    target_cells: Vec<CellGeometryV0>,
    intersections: Vec<CellIntersectionV0>,
    cell_box_candidates: u64,
}

struct IntervalIndexV0 {
    order: Vec<usize>,
    leaf_count: usize,
    subtree_max_x: Vec<f64>,
}

impl IntervalIndexV0 {
    fn new(cells: &[CellGeometryV0]) -> Self {
        let mut order = (0..cells.len()).collect::<Vec<_>>();
        order.sort_by(|&a, &b| {
            cells[a]
                .bounds
                .min_x
                .total_cmp(&cells[b].bounds.min_x)
                .then_with(|| point_sequence_cmp(&cells[a].key, &cells[b].key))
        });
        let leaf_count = order.len().next_power_of_two().max(1);
        let mut subtree_max_x = vec![f64::NEG_INFINITY; 2 * leaf_count];
        for (offset, &cell) in order.iter().enumerate() {
            subtree_max_x[leaf_count + offset] = cells[cell].bounds.max_x;
        }
        for node in (1..leaf_count).rev() {
            subtree_max_x[node] = subtree_max_x[2 * node].max(subtree_max_x[2 * node + 1]);
        }
        Self {
            order,
            leaf_count,
            subtree_max_x,
        }
    }

    fn query(&self, cells: &[CellGeometryV0], bounds: BoundsV0) -> Vec<usize> {
        let end = self
            .order
            .partition_point(|&cell| cells[cell].bounds.min_x <= bounds.max_x);
        let mut result = Vec::new();
        self.query_node(cells, bounds, 1, 0, self.leaf_count, end, &mut result);
        result.sort_by(|&a, &b| point_sequence_cmp(&cells[a].key, &cells[b].key));
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn query_node(
        &self,
        cells: &[CellGeometryV0],
        bounds: BoundsV0,
        node: usize,
        begin: usize,
        end: usize,
        query_end: usize,
        result: &mut Vec<usize>,
    ) {
        if begin >= query_end || self.subtree_max_x[node] < bounds.min_x {
            return;
        }
        if end - begin == 1 {
            if begin < self.order.len() {
                let cell = self.order[begin];
                if bounds.overlaps(cells[cell].bounds) {
                    result.push(cell);
                }
            }
            return;
        }
        let middle = (begin + end) / 2;
        self.query_node(cells, bounds, 2 * node, begin, middle, query_end, result);
        self.query_node(cells, bounds, 2 * node + 1, middle, end, query_end, result);
    }
}

fn shared_area_evidence_v0(
    source: &LandformObjectPacketCoreV0,
    target: &LandformObjectPacketCoreV0,
) -> Result<SharedAreaEvidenceV0, PacketCorrespondenceErrorV0> {
    let tolerance = source
        .surface_config
        .endpoint_match_abs_km
        .max(target.surface_config.endpoint_match_abs_km);
    shared_area_evidence_graphs_v0(&source.graph, &target.graph, tolerance)
}

fn shared_area_evidence_graphs_v0(
    source: &EvaluationSurfaceGraphV0,
    target: &EvaluationSurfaceGraphV0,
    tolerance: f64,
) -> Result<SharedAreaEvidenceV0, PacketCorrespondenceErrorV0> {
    let source_cells = cell_geometry_v0(source, tolerance)?;
    let target_cells = cell_geometry_v0(target, tolerance)?;
    let target_index = IntervalIndexV0::new(&target_cells);
    let mut intersections = Vec::new();
    let mut candidate_count = 0u64;
    for (source_cell, source_geometry) in source_cells.iter().enumerate() {
        for target_cell in target_index.query(&target_cells, source_geometry.bounds) {
            let target_geometry = &target_cells[target_cell];
            candidate_count =
                candidate_count
                    .checked_add(1)
                    .ok_or(PacketCorrespondenceErrorV0::Numerical(
                        "cell candidate overflow",
                    ))?;
            let Some(intersection) = convex_polygon_intersection_v0(
                source.polygon(source_cell),
                target.polygon(target_cell),
                tolerance,
            )?
            else {
                continue;
            };
            let (low_key, high_key) =
                if point_sequence_cmp(&source_geometry.key, &target_geometry.key)
                    != Ordering::Greater
                {
                    (source_geometry.key.clone(), target_geometry.key.clone())
                } else {
                    (target_geometry.key.clone(), source_geometry.key.clone())
                };
            intersections.push(CellIntersectionV0 {
                source_cell: source_cell as u32,
                target_cell: target_cell as u32,
                area_km2: intersection.area_km2,
                low_key,
                high_key,
            });
        }
    }
    intersections.sort_by(|a, b| {
        point_sequence_cmp(&a.low_key, &b.low_key)
            .then_with(|| point_sequence_cmp(&a.high_key, &b.high_key))
    });
    Ok(SharedAreaEvidenceV0 {
        source_cells,
        target_cells,
        intersections,
        cell_box_candidates: candidate_count,
    })
}

fn cell_geometry_v0(
    graph: &EvaluationSurfaceGraphV0,
    tolerance: f64,
) -> Result<Vec<CellGeometryV0>, PacketCorrespondenceErrorV0> {
    (0..graph.cell_count())
        .map(|cell| {
            let polygon = graph.polygon(cell);
            let self_clip = convex_polygon_intersection_v0(polygon, polygon, tolerance)?
                .ok_or(PacketCorrespondenceErrorV0::Numerical("cell self clip"))?;
            let mut key = polygon.to_vec();
            let start = key
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| point_cmp(**a, **b))
                .map(|(index, _)| index)
                .ok_or(PacketCorrespondenceErrorV0::Numerical("empty cell polygon"))?;
            key.rotate_left(start);
            let bounds = polygon_bounds(&key);
            Ok(CellGeometryV0 {
                area_km2: self_clip.area_km2,
                centroid_km: self_clip.centroid_km,
                key,
                bounds,
            })
        })
        .collect()
}

fn polygon_bounds(points: &[DVec3]) -> BoundsV0 {
    let mut bounds = BoundsV0 {
        min_x: f64::INFINITY,
        max_x: f64::NEG_INFINITY,
        min_y: f64::INFINITY,
        max_y: f64::NEG_INFINITY,
    };
    for point in points {
        bounds.min_x = bounds.min_x.min(point.x);
        bounds.max_x = bounds.max_x.max(point.x);
        bounds.min_y = bounds.min_y.min(point.y);
        bounds.max_y = bounds.max_y.max(point.y);
    }
    bounds
}

fn point_cmp(a: DVec3, b: DVec3) -> Ordering {
    a.x.total_cmp(&b.x)
        .then_with(|| a.y.total_cmp(&b.y))
        .then_with(|| a.z.total_cmp(&b.z))
}

fn point_sequence_cmp(a: &[DVec3], b: &[DVec3]) -> Ordering {
    a.len().cmp(&b.len()).then_with(|| {
        a.iter()
            .zip(b)
            .map(|(&a, &b)| point_cmp(a, b))
            .find(|ordering| *ordering != Ordering::Equal)
            .unwrap_or(Ordering::Equal)
    })
}

#[derive(Clone, Copy)]
struct SupportMeasureV0 {
    area_km2: f64,
    centroid_km: DVec3,
}

fn support_measure_v0(
    cells: &[u32],
    geometry: &[CellGeometryV0],
) -> Result<Option<SupportMeasureV0>, PacketCorrespondenceErrorV0> {
    if cells.is_empty() {
        return Ok(None);
    }
    let mut ordered = cells.to_vec();
    ordered
        .sort_by(|&a, &b| point_sequence_cmp(&geometry[a as usize].key, &geometry[b as usize].key));
    let area = compensated_sum(ordered.iter().map(|&cell| geometry[cell as usize].area_km2))?;
    if area <= 0.0 {
        return Err(PacketCorrespondenceErrorV0::Numerical(
            "nonpositive support area",
        ));
    }
    let x = compensated_sum(ordered.iter().map(|&cell| {
        let geometry = &geometry[cell as usize];
        geometry.area_km2 * geometry.centroid_km.x
    }))?;
    let y = compensated_sum(ordered.iter().map(|&cell| {
        let geometry = &geometry[cell as usize];
        geometry.area_km2 * geometry.centroid_km.y
    }))?;
    let centroid = DVec3::new(x / area, y / area, 0.0);
    if !centroid.is_finite() {
        return Err(PacketCorrespondenceErrorV0::Numerical("support centroid"));
    }
    Ok(Some(SupportMeasureV0 {
        area_km2: area,
        centroid_km: centroid,
    }))
}

struct FamilyAreaOutputV0 {
    nested_pairs: Vec<AreaPairV0>,
    exclusive_pairs: Vec<AreaPairV0>,
    context_records: Vec<ContextV0>,
    assignment_objects_source: Vec<AssignmentObjectInputV0>,
    assignment_objects_target: Vec<AssignmentObjectInputV0>,
    positive_scores: Vec<PositiveScoreV0>,
    nested_membership_contributions: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AreaPopulationKernelOutputV0 {
    pub nested_pairs: Vec<AreaPairV0>,
    pub exclusive_pairs: Vec<AreaPairV0>,
    pub context_records: Vec<ContextV0>,
    pub assignment_objects_source: Vec<AssignmentObjectInputV0>,
    pub assignment_objects_target: Vec<AssignmentObjectInputV0>,
    pub cell_box_candidates: u64,
    pub polygon_clips: u64,
    pub positive_cell_intersections: u64,
    pub nested_membership_contributions: u64,
}

/// Narrow production path for manufactured area/context/index fixtures.
/// Packet assembly uses the same shared-intersection and family distribution
/// functions after predecessor-ledger validation.
pub fn build_area_population_kernel_v0(
    source_graph: &EvaluationSurfaceGraphV0,
    source: &PacketAreaPopulationV0,
    target_graph: &EvaluationSurfaceGraphV0,
    target: &PacketAreaPopulationV0,
    endpoint_tolerance_km: f64,
    area_relative_tolerance: f64,
) -> Result<AreaPopulationKernelOutputV0, PacketCorrespondenceErrorV0> {
    if !endpoint_tolerance_km.is_finite()
        || endpoint_tolerance_km < 0.0
        || !area_relative_tolerance.is_finite()
        || area_relative_tolerance < 0.0
    {
        return Err(PacketCorrespondenceErrorV0::Incompatible(
            "area kernel tolerance",
        ));
    }
    let evidence =
        shared_area_evidence_graphs_v0(source_graph, target_graph, endpoint_tolerance_km)?;
    let output = build_family_area_v0(
        source,
        target,
        &evidence,
        endpoint_tolerance_km,
        area_relative_tolerance,
    )?;
    Ok(AreaPopulationKernelOutputV0 {
        nested_pairs: output.nested_pairs,
        exclusive_pairs: output.exclusive_pairs,
        context_records: output.context_records,
        assignment_objects_source: output.assignment_objects_source,
        assignment_objects_target: output.assignment_objects_target,
        cell_box_candidates: evidence.cell_box_candidates,
        polygon_clips: evidence.cell_box_candidates,
        positive_cell_intersections: evidence.intersections.len() as u64,
        nested_membership_contributions: output.nested_membership_contributions,
    })
}

#[derive(Default)]
struct ContextAccumulatorV0 {
    background: Vec<f64>,
    ineligible: BTreeMap<(u32, u8), Vec<f64>>,
    portals: BTreeMap<u32, Vec<f64>>,
    domain_intersection: Vec<f64>,
}

fn build_family_area_v0(
    source: &PacketAreaPopulationV0,
    target: &PacketAreaPopulationV0,
    evidence: &SharedAreaEvidenceV0,
    endpoint_tolerance_km: f64,
    area_relative_tolerance: f64,
) -> Result<FamilyAreaOutputV0, PacketCorrespondenceErrorV0> {
    if source.family != target.family
        || source.cell_class.len() != evidence.source_cells.len()
        || target.cell_class.len() != evidence.target_cells.len()
    {
        return Err(PacketCorrespondenceErrorV0::Incompatible(
            "area population geometry",
        ));
    }
    validate_area_population(source, evidence.source_cells.len())?;
    validate_area_population(target, evidence.target_cells.len())?;
    let source_nested = nested_memberships(source, evidence.source_cells.len())?;
    let target_nested = nested_memberships(target, evidence.target_cells.len())?;
    let source_measures = object_measures(source, &evidence.source_cells)?;
    let target_measures = object_measures(target, &evidence.target_cells)?;

    let mut nested = BTreeMap::<(u32, u32), Vec<f64>>::new();
    let mut exclusive = BTreeMap::<(u32, u32), Vec<f64>>::new();
    let mut source_context = source
        .objects
        .iter()
        .filter(|object| object.status == SupportStatusV0::Eligible)
        .map(|object| (object.id, ContextAccumulatorV0::default()))
        .collect::<BTreeMap<_, _>>();
    let mut target_context = target
        .objects
        .iter()
        .filter(|object| object.status == SupportStatusV0::Eligible)
        .map(|object| (object.id, ContextAccumulatorV0::default()))
        .collect::<BTreeMap<_, _>>();
    let mut nested_membership_contributions = 0u64;

    for intersection in &evidence.intersections {
        let source_cell = intersection.source_cell as usize;
        let target_cell = intersection.target_cell as usize;
        let source_members = &source_nested[source_cell];
        let target_members = &target_nested[target_cell];
        nested_membership_contributions = nested_membership_contributions
            .checked_add((source_members.len() * target_members.len()) as u64)
            .ok_or(PacketCorrespondenceErrorV0::Numerical(
                "nested contribution overflow",
            ))?;
        for &source_id in source_members {
            for &target_id in target_members {
                nested
                    .entry((source_id, target_id))
                    .or_default()
                    .push(intersection.area_km2);
            }
        }
        let source_class = source.cell_class[source_cell];
        let target_class = target.cell_class[target_cell];
        if let PacketCellClassV0::EligibleObject(source_id) = source_class {
            source_context
                .get_mut(&source_id)
                .ok_or(PacketCorrespondenceErrorV0::InvalidObjectReference {
                    family: source.family,
                    id: source_id,
                })?
                .domain_intersection
                .push(intersection.area_km2);
            accumulate_context(
                source_context.get_mut(&source_id).unwrap(),
                target_class,
                intersection.area_km2,
                target.family,
            )?;
        }
        if let PacketCellClassV0::EligibleObject(target_id) = target_class {
            target_context
                .get_mut(&target_id)
                .ok_or(PacketCorrespondenceErrorV0::InvalidObjectReference {
                    family: target.family,
                    id: target_id,
                })?
                .domain_intersection
                .push(intersection.area_km2);
            accumulate_context(
                target_context.get_mut(&target_id).unwrap(),
                source_class,
                intersection.area_km2,
                source.family,
            )?;
        }
        if let (
            PacketCellClassV0::EligibleObject(source_id),
            PacketCellClassV0::EligibleObject(target_id),
        ) = (source_class, target_class)
        {
            exclusive
                .entry((source_id, target_id))
                .or_default()
                .push(intersection.area_km2);
        }
    }

    let nested_pairs = area_rows(
        source.family,
        nested,
        AreaSupportV0::Nested,
        &source_measures.nested,
        &target_measures.nested,
        endpoint_tolerance_km,
        area_relative_tolerance,
    )?;
    let exclusive_pairs = area_rows(
        source.family,
        exclusive,
        AreaSupportV0::Exclusive,
        &source_measures.exclusive,
        &target_measures.exclusive,
        endpoint_tolerance_km,
        area_relative_tolerance,
    )?;
    let mut context_records = finish_contexts(
        PacketSideV0::Source,
        source.family,
        source_context,
        &source_measures.exclusive,
        endpoint_tolerance_km,
        area_relative_tolerance,
    )?;
    context_records.extend(finish_contexts(
        PacketSideV0::Target,
        target.family,
        target_context,
        &target_measures.exclusive,
        endpoint_tolerance_km,
        area_relative_tolerance,
    )?);
    let assignment_objects_source = assignment_objects(source, &source_measures)?;
    let assignment_objects_target = assignment_objects(target, &target_measures)?;
    let positive_scores = exclusive_pairs
        .iter()
        .map(|pair| PositiveScoreV0 {
            source_id: pair.source_id,
            target_id: pair.target_id,
            source_score: pair.intersection_area_km2,
            target_score: pair.intersection_area_km2,
        })
        .collect();
    Ok(FamilyAreaOutputV0 {
        nested_pairs,
        exclusive_pairs,
        context_records,
        assignment_objects_source,
        assignment_objects_target,
        positive_scores,
        nested_membership_contributions,
    })
}

fn validate_area_population(
    population: &PacketAreaPopulationV0,
    cell_count: usize,
) -> Result<(), PacketCorrespondenceErrorV0> {
    let mut objects = BTreeMap::new();
    let mut exclusive_owner = vec![None; cell_count];
    for object in &population.objects {
        if objects.insert(object.id, object.status).is_some() {
            return Err(PacketCorrespondenceErrorV0::DuplicateObject {
                family: population.family,
                id: object.id,
            });
        }
        if object.status == SupportStatusV0::NoPositiveOverlap {
            return Err(PacketCorrespondenceErrorV0::InvalidObjectReference {
                family: population.family,
                id: object.id,
            });
        }
        let nested = object.nested_cells.iter().copied().collect::<BTreeSet<_>>();
        let exclusive = object
            .exclusive_cells
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if nested.len() != object.nested_cells.len()
            || exclusive.len() != object.exclusive_cells.len()
            || nested.iter().any(|&cell| cell as usize >= cell_count)
            || exclusive.iter().any(|cell| !nested.contains(cell))
            || (object.status == SupportStatusV0::Eligible && exclusive.is_empty())
            || (object.status == SupportStatusV0::NoExclusiveSupport && !exclusive.is_empty())
        {
            return Err(PacketCorrespondenceErrorV0::InvalidObjectReference {
                family: population.family,
                id: object.id,
            });
        }
        for &cell in &exclusive {
            if exclusive_owner[cell as usize].replace(object.id).is_some() {
                return Err(PacketCorrespondenceErrorV0::InvalidExclusiveSupport {
                    family: population.family,
                    cell,
                });
            }
        }
    }
    for (cell, &class) in population.cell_class.iter().enumerate() {
        let expected = exclusive_owner[cell];
        match (population.family, class, expected) {
            (_, PacketCellClassV0::EligibleObject(id), Some(owner))
                if id == owner && objects.get(&id) == Some(&SupportStatusV0::Eligible) => {}
            (
                ObjectFamilyV0::Highland,
                PacketCellClassV0::IneligibleHighland { peak_id, status },
                Some(owner),
            ) if peak_id == owner
                && status == SupportStatusV0::HierarchyAmbiguousSupport
                && objects.get(&peak_id) == Some(&status) => {}
            (ObjectFamilyV0::Highland, PacketCellClassV0::HighlandBackground, None) => {}
            (ObjectFamilyV0::DrainageNode, PacketCellClassV0::Portal(_), None) => {}
            _ => {
                return Err(PacketCorrespondenceErrorV0::InvalidExclusiveSupport {
                    family: population.family,
                    cell: cell as u32,
                });
            }
        }
    }
    Ok(())
}

fn nested_memberships(
    population: &PacketAreaPopulationV0,
    cell_count: usize,
) -> Result<Vec<Vec<u32>>, PacketCorrespondenceErrorV0> {
    let mut result = vec![Vec::new(); cell_count];
    let mut ids = BTreeSet::new();
    for object in &population.objects {
        if !ids.insert(object.id) {
            return Err(PacketCorrespondenceErrorV0::DuplicateObject {
                family: population.family,
                id: object.id,
            });
        }
        for &cell in &object.nested_cells {
            let memberships = result.get_mut(cell as usize).ok_or(
                PacketCorrespondenceErrorV0::InvalidObjectReference {
                    family: population.family,
                    id: object.id,
                },
            )?;
            memberships.push(object.id);
        }
    }
    for memberships in &mut result {
        memberships.sort_unstable();
        memberships.dedup();
    }
    Ok(result)
}

struct ObjectMeasuresV0 {
    nested: BTreeMap<u32, SupportMeasureV0>,
    exclusive: BTreeMap<u32, SupportMeasureV0>,
}

fn object_measures(
    population: &PacketAreaPopulationV0,
    geometry: &[CellGeometryV0],
) -> Result<ObjectMeasuresV0, PacketCorrespondenceErrorV0> {
    let mut nested = BTreeMap::new();
    let mut exclusive = BTreeMap::new();
    for object in &population.objects {
        let nested_measure = support_measure_v0(&object.nested_cells, geometry)?.ok_or(
            PacketCorrespondenceErrorV0::AreaLedgerFailure {
                family: population.family,
                id: object.id,
            },
        )?;
        nested.insert(object.id, nested_measure);
        if let Some(measure) = support_measure_v0(&object.exclusive_cells, geometry)? {
            exclusive.insert(object.id, measure);
        }
        match object.status {
            SupportStatusV0::Eligible if !exclusive.contains_key(&object.id) => {
                return Err(PacketCorrespondenceErrorV0::AreaLedgerFailure {
                    family: population.family,
                    id: object.id,
                });
            }
            SupportStatusV0::NoExclusiveSupport if exclusive.contains_key(&object.id) => {
                return Err(PacketCorrespondenceErrorV0::AreaLedgerFailure {
                    family: population.family,
                    id: object.id,
                });
            }
            SupportStatusV0::NoPositiveOverlap => {
                return Err(PacketCorrespondenceErrorV0::AreaLedgerFailure {
                    family: population.family,
                    id: object.id,
                });
            }
            _ => {}
        }
    }
    Ok(ObjectMeasuresV0 { nested, exclusive })
}

fn area_rows(
    family: ObjectFamilyV0,
    contributions: BTreeMap<(u32, u32), Vec<f64>>,
    support_kind: AreaSupportV0,
    source_measures: &BTreeMap<u32, SupportMeasureV0>,
    target_measures: &BTreeMap<u32, SupportMeasureV0>,
    endpoint_tolerance_km: f64,
    area_relative_tolerance: f64,
) -> Result<Vec<AreaPairV0>, PacketCorrespondenceErrorV0> {
    let mut rows = Vec::with_capacity(contributions.len());
    for ((source_id, target_id), values) in contributions {
        let source = source_measures.get(&source_id).ok_or(
            PacketCorrespondenceErrorV0::InvalidObjectReference {
                family,
                id: source_id,
            },
        )?;
        let target = target_measures.get(&target_id).ok_or(
            PacketCorrespondenceErrorV0::InvalidObjectReference {
                family,
                id: target_id,
            },
        )?;
        let mut intersection = compensated_sum(values)?;
        if intersection <= 0.0 {
            continue;
        }
        let maximum = source.area_km2.min(target.area_km2);
        let tolerance = endpoint_tolerance_km * endpoint_tolerance_km
            + area_relative_tolerance * source.area_km2.max(target.area_km2);
        if intersection > maximum {
            if intersection - maximum > tolerance {
                return Err(PacketCorrespondenceErrorV0::Numerical(
                    "area intersection bound",
                ));
            }
            intersection = maximum;
        }
        let union = source.area_km2 + target.area_km2 - intersection;
        rows.push(AreaPairV0 {
            source_id,
            target_id,
            support_kind,
            intersection_area_km2: intersection,
            source_area_km2: source.area_km2,
            target_area_km2: target.area_km2,
            union_area_km2: union,
            source_coverage: intersection / source.area_km2,
            target_coverage: intersection / target.area_km2,
            jaccard: intersection / union,
            dice: 2.0 * intersection / (source.area_km2 + target.area_km2),
            source_centroid_km: source.centroid_km,
            target_centroid_km: target.centroid_km,
            centroid_displacement_km: source.centroid_km.distance(target.centroid_km),
        });
    }
    Ok(rows)
}

fn accumulate_context(
    accumulator: &mut ContextAccumulatorV0,
    opposite: PacketCellClassV0,
    area_km2: f64,
    family: ObjectFamilyV0,
) -> Result<(), PacketCorrespondenceErrorV0> {
    match opposite {
        PacketCellClassV0::EligibleObject(_) => {}
        PacketCellClassV0::IneligibleHighland { peak_id, status } => {
            if family != ObjectFamilyV0::Highland
                || status != SupportStatusV0::HierarchyAmbiguousSupport
            {
                return Err(PacketCorrespondenceErrorV0::Incompatible(
                    "ineligible highland context",
                ));
            }
            accumulator
                .ineligible
                .entry((peak_id, support_status_index(status)))
                .or_default()
                .push(area_km2);
        }
        PacketCellClassV0::HighlandBackground => {
            if family != ObjectFamilyV0::Highland {
                return Err(PacketCorrespondenceErrorV0::Incompatible(
                    "drainage background context",
                ));
            }
            accumulator.background.push(area_km2);
        }
        PacketCellClassV0::Portal(portal_id) => {
            if family != ObjectFamilyV0::DrainageNode {
                return Err(PacketCorrespondenceErrorV0::Incompatible(
                    "highland portal context",
                ));
            }
            accumulator
                .portals
                .entry(portal_id)
                .or_default()
                .push(area_km2);
        }
    }
    Ok(())
}

fn support_status_index(status: SupportStatusV0) -> u8 {
    match status {
        SupportStatusV0::Eligible => 0,
        SupportStatusV0::NoExclusiveSupport => 1,
        SupportStatusV0::HierarchyAmbiguousSupport => 2,
        SupportStatusV0::NoPositiveOverlap => 3,
    }
}

fn finish_contexts(
    side: PacketSideV0,
    family: ObjectFamilyV0,
    accumulators: BTreeMap<u32, ContextAccumulatorV0>,
    measures: &BTreeMap<u32, SupportMeasureV0>,
    endpoint_tolerance_km: f64,
    area_relative_tolerance: f64,
) -> Result<Vec<ContextV0>, PacketCorrespondenceErrorV0> {
    let mut records = Vec::with_capacity(accumulators.len());
    for (object_id, accumulator) in accumulators {
        let measure = measures.get(&object_id).ok_or(
            PacketCorrespondenceErrorV0::InvalidObjectReference {
                family,
                id: object_id,
            },
        )?;
        let domain_intersection = compensated_sum(accumulator.domain_intersection)?;
        let tolerance = endpoint_tolerance_km * endpoint_tolerance_km
            + area_relative_tolerance * measure.area_km2;
        let outside = if domain_intersection > measure.area_km2 {
            if domain_intersection - measure.area_km2 > tolerance {
                return Err(PacketCorrespondenceErrorV0::AreaLedgerFailure {
                    family,
                    id: object_id,
                });
            }
            0.0
        } else {
            let difference = measure.area_km2 - domain_intersection;
            if difference <= tolerance {
                0.0
            } else {
                difference
            }
        };
        let ineligible_highland_areas = accumulator
            .ineligible
            .into_iter()
            .map(|((peak_id, status), values)| {
                Ok(IneligibleHighlandAreaV0 {
                    peak_id,
                    support_status: match status {
                        2 => SupportStatusV0::HierarchyAmbiguousSupport,
                        _ => {
                            return Err(PacketCorrespondenceErrorV0::Incompatible(
                                "ineligible status",
                            ));
                        }
                    },
                    area_km2: compensated_sum(values)?,
                })
            })
            .collect::<Result<Vec<_>, PacketCorrespondenceErrorV0>>()?;
        let portal_areas_km2 = accumulator
            .portals
            .into_iter()
            .map(|(portal_id, values)| {
                Ok(PortalAreaV0 {
                    portal_id,
                    area_km2: compensated_sum(values)?,
                })
            })
            .collect::<Result<Vec<_>, PacketCorrespondenceErrorV0>>()?;
        records.push(ContextV0 {
            side,
            family,
            object_id,
            background_area_km2: compensated_sum(accumulator.background)?,
            ineligible_highland_areas,
            portal_areas_km2,
            outside_domain_area_km2: outside,
        });
    }
    Ok(records)
}

fn assignment_objects(
    population: &PacketAreaPopulationV0,
    measures: &ObjectMeasuresV0,
) -> Result<Vec<AssignmentObjectInputV0>, PacketCorrespondenceErrorV0> {
    population
        .objects
        .iter()
        .map(|object| {
            let nested = measures.nested.get(&object.id).ok_or(
                PacketCorrespondenceErrorV0::InvalidObjectReference {
                    family: population.family,
                    id: object.id,
                },
            )?;
            let exclusive = measures.exclusive.get(&object.id);
            Ok(AssignmentObjectInputV0 {
                object_id: object.id,
                object_measure: exclusive.map_or(0.0, |measure| measure.area_km2),
                anchor_km: exclusive.map_or(nested.centroid_km, |measure| measure.centroid_km),
                support_status: object.status,
            })
        })
        .collect()
}

fn validate_predecessor_area_ledgers(
    packet: &LandformObjectPacketCoreV0,
    population: &PacketAreaPopulationV0,
    geometry: &[CellGeometryV0],
) -> Result<(), PacketCorrespondenceErrorV0> {
    let polygon_allowance = |cells: &[u32]| -> Result<f64, PacketCorrespondenceErrorV0> {
        compensated_sum(cells.iter().map(|&cell| {
            packet.graph.cell_area_km2[cell as usize]
                * packet.surface_config.planar_area_match_relative
        }))
    };
    match population.family {
        ObjectFamilyV0::Highland => {
            for object in &population.objects {
                let measure = support_measure_v0(&object.nested_cells, geometry)?.ok_or(
                    PacketCorrespondenceErrorV0::AreaLedgerFailure {
                        family: population.family,
                        id: object.id,
                    },
                )?;
                let stored = packet.surface_hierarchy.peaks[object.id as usize].footprint_area_km2;
                let allowance = polygon_allowance(&object.nested_cells)?;
                if (measure.area_km2 - stored).abs() > allowance {
                    return Err(PacketCorrespondenceErrorV0::AreaLedgerFailure {
                        family: population.family,
                        id: object.id,
                    });
                }
            }
        }
        ObjectFamilyV0::DrainageNode => {
            let scale = reference_drainage_scale(packet)?;
            let catchments = scale
                .basin_graph
                .catchments
                .iter()
                .map(|catchment| (catchment.reach_id, catchment))
                .collect::<BTreeMap<_, _>>();
            for object in &population.objects {
                let catchment = catchments.get(&object.id).ok_or(
                    PacketCorrespondenceErrorV0::InvalidObjectReference {
                        family: population.family,
                        id: object.id,
                    },
                )?;
                for (cells, stored) in [
                    (&object.nested_cells, catchment.nested_structural_area_km2),
                    (
                        &object.exclusive_cells,
                        catchment.exclusive_physical_area_km2,
                    ),
                ] {
                    let measure = support_measure_v0(cells, geometry)?.ok_or(
                        PacketCorrespondenceErrorV0::AreaLedgerFailure {
                            family: population.family,
                            id: object.id,
                        },
                    )?;
                    let g0_allowance = polygon_allowance(cells)?;
                    let d0_allowance = packet
                        .drainage_config
                        .balance_absolute_tolerance
                        .max(packet.drainage_config.balance_relative_tolerance * stored.abs());
                    if (measure.area_km2 - stored).abs() > g0_allowance + d0_allowance {
                        return Err(PacketCorrespondenceErrorV0::AreaLedgerFailure {
                            family: population.family,
                            id: object.id,
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

struct LineOutputV0 {
    pairs: Vec<LinePairV0>,
    source_objects: Vec<AssignmentObjectInputV0>,
    target_objects: Vec<AssignmentObjectInputV0>,
    positive_scores: Vec<PositiveScoreV0>,
    source_segments: u64,
    target_segments: u64,
    segment_candidates: u64,
    segment_tests: u64,
}

fn build_line_output_v0(
    source: &[PacketLineObjectV0],
    target: &[PacketLineObjectV0],
) -> Result<LineOutputV0, PacketCorrespondenceErrorV0> {
    let source_objects = line_assignment_objects(source)?;
    let target_objects = line_assignment_objects(target)?;
    let source_inputs = source
        .iter()
        .map(|line| LineObjectInputV0 {
            object_id: line.id,
            segments: &line.segments,
        })
        .collect::<Vec<_>>();
    let target_inputs = target
        .iter()
        .map(|line| LineObjectInputV0 {
            object_id: line.id,
            segments: &line.segments,
        })
        .collect::<Vec<_>>();
    let population = build_line_population_v0(&source_inputs, &target_inputs)?;
    let positive_scores = population
        .pairs
        .iter()
        .map(|pair| PositiveScoreV0 {
            source_id: pair.source_id,
            target_id: pair.target_id,
            source_score: pair.source_covered_length_km,
            target_score: pair.target_covered_length_km,
        })
        .collect();
    Ok(LineOutputV0 {
        pairs: population.pairs,
        source_objects,
        target_objects,
        positive_scores,
        source_segments: population.source_segments,
        target_segments: population.target_segments,
        segment_candidates: population.segment_box_candidates,
        segment_tests: population.segment_pair_tests,
    })
}

fn line_assignment_objects(
    lines: &[PacketLineObjectV0],
) -> Result<Vec<AssignmentObjectInputV0>, PacketCorrespondenceErrorV0> {
    let mut seen = BTreeSet::new();
    let mut objects = Vec::with_capacity(lines.len());
    for line in lines {
        if !seen.insert(line.id) {
            return Err(PacketCorrespondenceErrorV0::DuplicateObject {
                family: ObjectFamilyV0::DrainageNode,
                id: line.id,
            });
        }
        let measure = measure_line_object_v0(&line.segments)?;
        objects.push(AssignmentObjectInputV0 {
            object_id: line.id,
            object_measure: measure.total_length_km,
            anchor_km: measure.half_arclength_anchor_km,
            support_status: SupportStatusV0::Eligible,
        });
    }
    objects.sort_by_key(|object| object.object_id);
    Ok(objects)
}

fn extract_highland_topology_v0(
    packet: &LandformObjectPacketCoreV0,
    population: &PacketAreaPopulationV0,
) -> Result<(Vec<TopologyEdgeInputV0>, Vec<TopologyObjectInputV0>), PacketCorrespondenceErrorV0> {
    let retained = population
        .objects
        .iter()
        .map(|object| object.id)
        .collect::<BTreeSet<_>>();
    let statuses = population
        .objects
        .iter()
        .map(|object| (object.id, object.status))
        .collect::<BTreeMap<_, _>>();
    let mut edges = Vec::with_capacity(population.objects.len());
    for object in &population.objects {
        let mut cursor = object.id;
        let mut visited = BTreeSet::new();
        let target = loop {
            if !visited.insert(cursor) {
                return Err(PacketCorrespondenceErrorV0::InvalidHierarchy(object.id));
            }
            let peak = packet
                .surface_hierarchy
                .peaks
                .get(cursor as usize)
                .filter(|peak| peak.id == cursor)
                .ok_or(PacketCorrespondenceErrorV0::InvalidHierarchy(cursor))?;
            let Some(parent) = peak.parent_peak else {
                break TopologyTargetV0::HighlandRoot;
            };
            if retained.contains(&parent) {
                break TopologyTargetV0::Highland(parent);
            }
            cursor = parent;
        };
        edges.push(TopologyEdgeInputV0 {
            from_id: object.id,
            target,
            hierarchy_ambiguous: statuses[&object.id] == SupportStatusV0::HierarchyAmbiguousSupport,
        });
    }
    let objects = edges
        .iter()
        .map(|edge| TopologyObjectInputV0 {
            object_id: edge.from_id,
            target: edge.target,
        })
        .collect();
    Ok((edges, objects))
}

fn extract_drainage_topology_v0(
    packet: &LandformObjectPacketCoreV0,
) -> Result<(Vec<TopologyEdgeInputV0>, Vec<TopologyObjectInputV0>), PacketCorrespondenceErrorV0> {
    let scale = reference_drainage_scale(packet)?;
    let ids = scale
        .reach_graph
        .reaches
        .iter()
        .map(|reach| reach.id)
        .collect::<BTreeSet<_>>();
    let mut edges = Vec::with_capacity(ids.len());
    for reach in &scale.reach_graph.reaches {
        let target = match (reach.downstream_reach, reach.terminal_portal_id) {
            (Some(id), None) if ids.contains(&id) => TopologyTargetV0::DrainageNode(id),
            (None, Some(id)) => TopologyTargetV0::Portal(id),
            _ => return Err(PacketCorrespondenceErrorV0::InvalidReach(reach.id)),
        };
        edges.push(TopologyEdgeInputV0 {
            from_id: reach.id,
            target,
            hierarchy_ambiguous: false,
        });
    }
    edges.sort_by_key(|edge| edge.from_id);
    let objects = edges
        .iter()
        .map(|edge| TopologyObjectInputV0 {
            object_id: edge.from_id,
            target: edge.target,
        })
        .collect();
    Ok((edges, objects))
}

#[allow(clippy::too_many_arguments)]
fn build_packet_topology_v0(
    source: &LandformObjectPacketCoreV0,
    target: &LandformObjectPacketCoreV0,
    source_highlands: &PacketAreaPopulationV0,
    target_highlands: &PacketAreaPopulationV0,
    assignments: &[AssignmentV0],
    components: &[BestComponentV0],
) -> Result<Vec<TopologyV0>, PacketCorrespondenceErrorV0> {
    let declared_portals = source
        .population
        .semantic_portals
        .iter()
        .map(|portal| portal.id)
        .collect::<Vec<_>>();
    let (source_highland_edges, source_highland_topology) =
        extract_highland_topology_v0(source, source_highlands)?;
    let (target_highland_edges, target_highland_topology) =
        extract_highland_topology_v0(target, target_highlands)?;
    let (source_drainage_edges, source_drainage_topology) = extract_drainage_topology_v0(source)?;
    let (target_drainage_edges, target_drainage_topology) = extract_drainage_topology_v0(target)?;
    let mut records = Vec::new();
    records.extend(build_topology_records_v0(
        PacketSideV0::Source,
        ObjectFamilyV0::Highland,
        AssignmentChannelV0::HighlandExclusiveArea,
        &source_highland_edges,
        &target_highland_topology,
        assignments,
        components,
        &declared_portals,
    )?);
    records.extend(build_topology_records_v0(
        PacketSideV0::Target,
        ObjectFamilyV0::Highland,
        AssignmentChannelV0::HighlandExclusiveArea,
        &target_highland_edges,
        &source_highland_topology,
        assignments,
        components,
        &declared_portals,
    )?);
    for channel in [
        AssignmentChannelV0::DrainageExclusiveArea,
        AssignmentChannelV0::DrainageLine,
    ] {
        records.extend(build_topology_records_v0(
            PacketSideV0::Source,
            ObjectFamilyV0::DrainageNode,
            channel,
            &source_drainage_edges,
            &target_drainage_topology,
            assignments,
            components,
            &declared_portals,
        )?);
        records.extend(build_topology_records_v0(
            PacketSideV0::Target,
            ObjectFamilyV0::DrainageNode,
            channel,
            &target_drainage_edges,
            &source_drainage_topology,
            assignments,
            components,
            &declared_portals,
        )?);
    }
    records.sort_by(|a, b| {
        a.side
            .cmp(&b.side)
            .then_with(|| a.family.cmp(&b.family))
            .then_with(|| a.channel.cmp(&b.channel))
            .then_with(|| a.from_id.cmp(&b.from_id))
    });
    Ok(records)
}

/// Build the ordered O0b correspondence artifact using the frozen config.
pub fn build_object_correspondence_v0(
    source: &LandformObjectPacketCoreV0,
    target: &LandformObjectPacketCoreV0,
) -> Result<ObjectCorrespondenceV0, PacketCorrespondenceErrorV0> {
    build_object_correspondence_with_config_v0(source, target, CorrespondenceConfigV0::default())
}

pub fn build_object_correspondence_with_config_v0(
    source: &LandformObjectPacketCoreV0,
    target: &LandformObjectPacketCoreV0,
    config: CorrespondenceConfigV0,
) -> Result<ObjectCorrespondenceV0, PacketCorrespondenceErrorV0> {
    validate_correspondence_pair_v0(source, target, config)?;
    let area_evidence = shared_area_evidence_v0(source, target)?;
    let endpoint_tolerance = source
        .surface_config
        .endpoint_match_abs_km
        .max(target.surface_config.endpoint_match_abs_km);
    let area_relative_tolerance = source
        .surface_config
        .planar_area_match_relative
        .max(target.surface_config.planar_area_match_relative);

    let source_highlands = extract_highland_population_v0(source)?;
    let target_highlands = extract_highland_population_v0(target)?;
    validate_predecessor_area_ledgers(source, &source_highlands, &area_evidence.source_cells)?;
    validate_predecessor_area_ledgers(target, &target_highlands, &area_evidence.target_cells)?;
    let highlands = build_family_area_v0(
        &source_highlands,
        &target_highlands,
        &area_evidence,
        endpoint_tolerance,
        area_relative_tolerance,
    )?;
    let source_drainage = extract_drainage_population_v0(source)?;
    let target_drainage = extract_drainage_population_v0(target)?;
    validate_predecessor_area_ledgers(source, &source_drainage, &area_evidence.source_cells)?;
    validate_predecessor_area_ledgers(target, &target_drainage, &area_evidence.target_cells)?;
    let drainage = build_family_area_v0(
        &source_drainage,
        &target_drainage,
        &area_evidence,
        endpoint_tolerance,
        area_relative_tolerance,
    )?;

    let source_lines = extract_reference_reach_lines_v0(source)?;
    let target_lines = extract_reference_reach_lines_v0(target)?;
    if source_lines.iter().map(|line| line.id).collect::<Vec<_>>()
        != source_drainage
            .objects
            .iter()
            .map(|object| object.id)
            .collect::<Vec<_>>()
        || target_lines.iter().map(|line| line.id).collect::<Vec<_>>()
            != target_drainage
                .objects
                .iter()
                .map(|object| object.id)
                .collect::<Vec<_>>()
    {
        return Err(PacketCorrespondenceErrorV0::Incompatible(
            "drainage area/line object bijection",
        ));
    }
    let lines = build_line_output_v0(&source_lines, &target_lines)?;

    let highland_assignment = build_assignment_kernel_v0(
        ObjectFamilyV0::Highland,
        AssignmentChannelV0::HighlandExclusiveArea,
        &highlands.assignment_objects_source,
        &highlands.assignment_objects_target,
        &highlands.positive_scores,
    )?;
    let drainage_area_assignment = build_assignment_kernel_v0(
        ObjectFamilyV0::DrainageNode,
        AssignmentChannelV0::DrainageExclusiveArea,
        &drainage.assignment_objects_source,
        &drainage.assignment_objects_target,
        &drainage.positive_scores,
    )?;
    let drainage_line_assignment = build_assignment_kernel_v0(
        ObjectFamilyV0::DrainageNode,
        AssignmentChannelV0::DrainageLine,
        &lines.source_objects,
        &lines.target_objects,
        &lines.positive_scores,
    )?;
    let mut assignment_records = highland_assignment.assignments;
    assignment_records.extend(drainage_area_assignment.assignments);
    assignment_records.extend(drainage_line_assignment.assignments);
    assignment_records
        .sort_by_key(|record| (record.side, record.family, record.object_id, record.channel));
    let mut best_components = highland_assignment.best_components;
    best_components.extend(drainage_area_assignment.best_components);
    best_components.extend(drainage_line_assignment.best_components);
    sort_merged_components(
        &mut best_components,
        &[
            (
                AssignmentChannelV0::HighlandExclusiveArea,
                ObjectFamilyV0::Highland,
                &highlands.assignment_objects_source,
                &highlands.assignment_objects_target,
            ),
            (
                AssignmentChannelV0::DrainageExclusiveArea,
                ObjectFamilyV0::DrainageNode,
                &drainage.assignment_objects_source,
                &drainage.assignment_objects_target,
            ),
            (
                AssignmentChannelV0::DrainageLine,
                ObjectFamilyV0::DrainageNode,
                &lines.source_objects,
                &lines.target_objects,
            ),
        ],
    )?;
    let metric_conflicts = build_metric_conflicts_v0(&assignment_records)?;
    let best_graph_edges = count_best_graph_edges(&assignment_records)?;
    let topology_records = build_packet_topology_v0(
        source,
        target,
        &source_highlands,
        &target_highlands,
        &assignment_records,
        &best_components,
    )?;

    let mut context_records = highlands.context_records;
    context_records.extend(drainage.context_records);
    context_records.sort_by_key(|record| (record.side, record.family, record.object_id));
    let nested_membership_contributions = highlands
        .nested_membership_contributions
        .checked_add(drainage.nested_membership_contributions)
        .ok_or(PacketCorrespondenceErrorV0::Numerical(
            "nested membership count overflow",
        ))?;
    let work_counts = CorrespondenceWorkCountsV0 {
        source_cells: source.graph.cell_count() as u64,
        target_cells: target.graph.cell_count() as u64,
        cell_box_candidates: area_evidence.cell_box_candidates,
        polygon_clips: area_evidence.cell_box_candidates,
        positive_cell_intersections: area_evidence.intersections.len() as u64,
        nested_membership_contributions,
        source_segments: lines.source_segments,
        target_segments: lines.target_segments,
        segment_box_candidates: lines.segment_candidates,
        segment_pair_tests: lines.segment_tests,
        positive_highland_nested_rows: highlands.nested_pairs.len() as u64,
        positive_highland_exclusive_rows: highlands.exclusive_pairs.len() as u64,
        positive_drainage_nested_rows: drainage.nested_pairs.len() as u64,
        positive_drainage_exclusive_rows: drainage.exclusive_pairs.len() as u64,
        positive_line_rows: lines.pairs.len() as u64,
        best_graph_edges,
    };
    let mut result = ObjectCorrespondenceV0 {
        schema_version: O0B_CORRESPONDENCE_SCHEMA_VERSION.into(),
        hash_version: O0B_CORRESPONDENCE_HASH_VERSION.into(),
        config: CorrespondenceConfigWireV0::from(&config),
        source_packet_hash: source.derived_common_packet_hash,
        target_packet_hash: target.derived_common_packet_hash,
        highland_nested_pairs: highlands.nested_pairs,
        highland_exclusive_pairs: highlands.exclusive_pairs,
        drainage_nested_pairs: drainage.nested_pairs,
        drainage_exclusive_pairs: drainage.exclusive_pairs,
        drainage_line_pairs: lines.pairs,
        context_records,
        assignment_records,
        best_components,
        metric_conflicts,
        topology_records,
        work_counts,
        derived_correspondence_hash: 0,
    };
    validate_object_correspondence_semantics_v0(&result)?;
    result.derived_correspondence_hash = correspondence_preimage_hash(&result)?;
    Ok(result)
}

fn count_best_graph_edges(
    assignments: &[AssignmentV0],
) -> Result<u64, PacketCorrespondenceErrorV0> {
    let mut edges = BTreeSet::new();
    for assignment in assignments {
        for &partner in &assignment.maximum_partner_ids {
            let (source, target) = match assignment.side {
                PacketSideV0::Source => (assignment.object_id, partner),
                PacketSideV0::Target => (partner, assignment.object_id),
            };
            edges.insert((assignment.channel, source, target));
        }
    }
    Ok(edges.len() as u64)
}

type ComponentAnchorPopulation<'a> = (
    AssignmentChannelV0,
    ObjectFamilyV0,
    &'a [AssignmentObjectInputV0],
    &'a [AssignmentObjectInputV0],
);

fn sort_merged_components(
    components: &mut [BestComponentV0],
    populations: &[ComponentAnchorPopulation<'_>],
) -> Result<(), PacketCorrespondenceErrorV0> {
    let mut anchors = BTreeMap::new();
    for &(channel, family, source, target) in populations {
        for (side, objects) in [
            (PacketSideV0::Source, source),
            (PacketSideV0::Target, target),
        ] {
            for object in objects {
                let key = (
                    channel,
                    BestMemberV0 {
                        side,
                        family,
                        object_id: object.object_id,
                    },
                );
                if anchors.insert(key, object.anchor_km).is_some() {
                    return Err(PacketCorrespondenceErrorV0::DuplicateObject {
                        family,
                        id: object.object_id,
                    });
                }
            }
        }
    }
    let member_cmp =
        |channel: AssignmentChannelV0, a: &BestMemberV0, b: &BestMemberV0| -> Ordering {
            let anchor_a = anchors[&(channel, *a)];
            let anchor_b = anchors[&(channel, *b)];
            point_cmp(anchor_a, anchor_b)
                .then_with(|| a.side.cmp(&b.side))
                .then_with(|| a.object_id.cmp(&b.object_id))
                .then_with(|| a.family.cmp(&b.family))
        };
    components.sort_by(|a, b| {
        let minimum_a = a
            .members
            .iter()
            .min_by(|left, right| member_cmp(a.channel, left, right))
            .expect("best component is nonempty");
        let minimum_b = b
            .members
            .iter()
            .min_by(|left, right| member_cmp(b.channel, left, right))
            .expect("best component is nonempty");
        let anchor_a = anchors[&(a.channel, *minimum_a)];
        let anchor_b = anchors[&(b.channel, *minimum_b)];
        point_cmp(anchor_a, anchor_b)
            .then_with(|| minimum_a.side.cmp(&minimum_b.side))
            .then_with(|| minimum_a.object_id.cmp(&minimum_b.object_id))
            .then_with(|| a.members.cmp(&b.members))
            .then_with(|| a.channel.cmp(&b.channel))
    });
    Ok(())
}

pub fn object_correspondence_hash_v0(
    correspondence: &ObjectCorrespondenceV0,
) -> Result<u64, PacketCorrespondenceErrorV0> {
    validate_correspondence_record_version(correspondence)?;
    validate_object_correspondence_semantics_v0(correspondence)?;
    correspondence_preimage_hash(correspondence)
}

pub fn object_correspondence_bytes_v0(
    correspondence: &ObjectCorrespondenceV0,
) -> Result<Vec<u8>, PacketCorrespondenceErrorV0> {
    validate_correspondence_record_version(correspondence)?;
    validate_object_correspondence_semantics_v0(correspondence)?;
    let computed = correspondence_preimage_hash(correspondence)?;
    if computed != correspondence.derived_correspondence_hash {
        return Err(PacketCorrespondenceErrorV0::Incompatible(
            "correspondence hash",
        ));
    }
    fixed_bytes(correspondence)
}

pub fn decode_object_correspondence_v0(
    bytes: &[u8],
) -> Result<ObjectCorrespondenceV0, PacketCorrespondenceErrorV0> {
    let correspondence: ObjectCorrespondenceV0 = bincode_options()
        .deserialize(bytes)
        .map_err(|error| PacketCorrespondenceErrorV0::Serialization(error.to_string()))?;
    validate_correspondence_record_version(&correspondence)?;
    validate_object_correspondence_semantics_v0(&correspondence)?;
    let computed = correspondence_preimage_hash(&correspondence)?;
    if computed != correspondence.derived_correspondence_hash {
        return Err(PacketCorrespondenceErrorV0::Incompatible(
            "correspondence hash",
        ));
    }
    Ok(correspondence)
}

fn validate_correspondence_record_version(
    correspondence: &ObjectCorrespondenceV0,
) -> Result<(), PacketCorrespondenceErrorV0> {
    if correspondence.schema_version != O0B_CORRESPONDENCE_SCHEMA_VERSION
        || correspondence.hash_version != O0B_CORRESPONDENCE_HASH_VERSION
        || correspondence.config
            != CorrespondenceConfigWireV0::from(&CorrespondenceConfigV0::default())
    {
        return Err(PacketCorrespondenceErrorV0::Incompatible(
            "correspondence schema/config",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct SerializedObjectEvidenceV0 {
    measure: f64,
    anchor: DVec3,
}

#[derive(Default)]
struct SerializedEvidenceIndexV0 {
    objects: BTreeMap<(AssignmentChannelV0, PacketSideV0, u32), SerializedObjectEvidenceV0>,
    scores: BTreeMap<(AssignmentChannelV0, u32, u32), (f64, f64)>,
}

fn validate_object_correspondence_semantics_v0(
    value: &ObjectCorrespondenceV0,
) -> Result<(), PacketCorrespondenceErrorV0> {
    let mut evidence = SerializedEvidenceIndexV0::default();
    validate_area_table(
        &value.highland_nested_pairs,
        AreaSupportV0::Nested,
        None,
        &mut evidence,
    )?;
    validate_area_table(
        &value.highland_exclusive_pairs,
        AreaSupportV0::Exclusive,
        Some(AssignmentChannelV0::HighlandExclusiveArea),
        &mut evidence,
    )?;
    validate_area_table(
        &value.drainage_nested_pairs,
        AreaSupportV0::Nested,
        None,
        &mut evidence,
    )?;
    validate_area_table(
        &value.drainage_exclusive_pairs,
        AreaSupportV0::Exclusive,
        Some(AssignmentChannelV0::DrainageExclusiveArea),
        &mut evidence,
    )?;
    validate_line_table(&value.drainage_line_pairs, &mut evidence)?;
    validate_context_records(&value.context_records)?;
    let reconstructed = reconstruct_assignment_evidence(value, &evidence)?;
    validate_context_completeness(&value.context_records, &value.assignment_records)?;
    validate_pair_assignment_references(value)?;
    validate_topology_semantics(
        &value.topology_records,
        &value.assignment_records,
        &reconstructed.components,
    )?;
    validate_work_counts(value)?;
    Ok(())
}

fn validate_area_table(
    rows: &[AreaPairV0],
    expected_support: AreaSupportV0,
    channel: Option<AssignmentChannelV0>,
    evidence: &mut SerializedEvidenceIndexV0,
) -> Result<(), PacketCorrespondenceErrorV0> {
    let mut previous = None;
    for row in rows {
        let key = (row.source_id, row.target_id);
        if previous.is_some_and(|previous| previous >= key) || row.support_kind != expected_support
        {
            return semantic_error("area pair ordering/support");
        }
        previous = Some(key);
        for scalar in [
            row.intersection_area_km2,
            row.source_area_km2,
            row.target_area_km2,
            row.union_area_km2,
            row.source_coverage,
            row.target_coverage,
            row.jaccard,
            row.dice,
            row.centroid_displacement_km,
        ] {
            validate_canonical_finite(scalar, "area pair scalar")?;
        }
        validate_planar_point(row.source_centroid_km, "source area centroid")?;
        validate_planar_point(row.target_centroid_km, "target area centroid")?;
        if row.intersection_area_km2 <= 0.0
            || row.source_area_km2 <= 0.0
            || row.target_area_km2 <= 0.0
            || row.intersection_area_km2 > row.source_area_km2.min(row.target_area_km2)
            || row.union_area_km2 <= 0.0
            || !(0.0 < row.source_coverage && row.source_coverage <= 1.0)
            || !(0.0 < row.target_coverage && row.target_coverage <= 1.0)
            || !(0.0 < row.jaccard && row.jaccard <= 1.0)
            || !(0.0 < row.dice && row.dice <= 1.0)
            || !same_float(
                row.union_area_km2,
                row.source_area_km2 + row.target_area_km2 - row.intersection_area_km2,
            )
            || !same_float(
                row.source_coverage,
                row.intersection_area_km2 / row.source_area_km2,
            )
            || !same_float(
                row.target_coverage,
                row.intersection_area_km2 / row.target_area_km2,
            )
            || !same_float(row.jaccard, row.intersection_area_km2 / row.union_area_km2)
            || !same_float(
                row.dice,
                2.0 * row.intersection_area_km2 / (row.source_area_km2 + row.target_area_km2),
            )
            || !same_float(
                row.centroid_displacement_km,
                row.source_centroid_km.distance(row.target_centroid_km),
            )
        {
            return semantic_error("area pair formula");
        }
        if let Some(channel) = channel {
            insert_serialized_object(
                evidence,
                channel,
                PacketSideV0::Source,
                row.source_id,
                row.source_area_km2,
                row.source_centroid_km,
            )?;
            insert_serialized_object(
                evidence,
                channel,
                PacketSideV0::Target,
                row.target_id,
                row.target_area_km2,
                row.target_centroid_km,
            )?;
            if evidence
                .scores
                .insert(
                    (channel, row.source_id, row.target_id),
                    (row.intersection_area_km2, row.intersection_area_km2),
                )
                .is_some()
            {
                return semantic_error("duplicate assignment score");
            }
        }
    }
    Ok(())
}

fn validate_line_table(
    rows: &[LinePairV0],
    evidence: &mut SerializedEvidenceIndexV0,
) -> Result<(), PacketCorrespondenceErrorV0> {
    let mut previous = None;
    for row in rows {
        let key = (row.source_id, row.target_id);
        if previous.is_some_and(|previous| previous >= key) {
            return semantic_error("line pair ordering");
        }
        previous = Some(key);
        for scalar in [
            row.source_covered_length_km,
            row.target_covered_length_km,
            row.source_coverage,
            row.target_coverage,
            row.source_length_km,
            row.target_length_km,
            row.anchor_displacement_km,
            row.minimum_positive_candidate_separation_km,
        ] {
            validate_canonical_finite(scalar, "line pair scalar")?;
        }
        validate_planar_point(row.source_anchor_km, "source line anchor")?;
        validate_planar_point(row.target_anchor_km, "target line anchor")?;
        if row.source_covered_length_km <= 0.0
            || row.target_covered_length_km <= 0.0
            || row.source_length_km <= 0.0
            || row.target_length_km <= 0.0
            || row.source_covered_length_km > row.source_length_km
            || row.target_covered_length_km > row.target_length_km
            || !(0.0 < row.source_coverage && row.source_coverage <= 1.0)
            || !(0.0 < row.target_coverage && row.target_coverage <= 1.0)
            || row.minimum_positive_candidate_separation_km < 0.0
            || !same_float(
                row.source_coverage,
                row.source_covered_length_km / row.source_length_km,
            )
            || !same_float(
                row.target_coverage,
                row.target_covered_length_km / row.target_length_km,
            )
            || !same_float(
                row.anchor_displacement_km,
                row.source_anchor_km.distance(row.target_anchor_km),
            )
        {
            return semantic_error("line pair formula");
        }
        let channel = AssignmentChannelV0::DrainageLine;
        insert_serialized_object(
            evidence,
            channel,
            PacketSideV0::Source,
            row.source_id,
            row.source_length_km,
            row.source_anchor_km,
        )?;
        insert_serialized_object(
            evidence,
            channel,
            PacketSideV0::Target,
            row.target_id,
            row.target_length_km,
            row.target_anchor_km,
        )?;
        if evidence
            .scores
            .insert(
                (channel, row.source_id, row.target_id),
                (row.source_covered_length_km, row.target_covered_length_km),
            )
            .is_some()
        {
            return semantic_error("duplicate line score");
        }
    }
    Ok(())
}

fn insert_serialized_object(
    evidence: &mut SerializedEvidenceIndexV0,
    channel: AssignmentChannelV0,
    side: PacketSideV0,
    id: u32,
    measure: f64,
    anchor: DVec3,
) -> Result<(), PacketCorrespondenceErrorV0> {
    let key = (channel, side, id);
    let object = SerializedObjectEvidenceV0 { measure, anchor };
    if let Some(previous) = evidence.objects.insert(key, object) {
        if !same_float(previous.measure, measure) || previous.anchor != anchor {
            return semantic_error("inconsistent serialized object evidence");
        }
    }
    Ok(())
}

fn validate_context_records(records: &[ContextV0]) -> Result<(), PacketCorrespondenceErrorV0> {
    let mut previous = None;
    for record in records {
        let key = (record.side, record.family, record.object_id);
        if previous.is_some_and(|previous| previous >= key) {
            return semantic_error("context ordering");
        }
        previous = Some(key);
        for scalar in [record.background_area_km2, record.outside_domain_area_km2] {
            validate_canonical_finite(scalar, "context scalar")?;
            if scalar < 0.0 {
                return semantic_error("negative context area");
            }
        }
        let mut previous_peak = None;
        for entry in &record.ineligible_highland_areas {
            if previous_peak.is_some_and(|previous| previous >= entry.peak_id)
                || entry.support_status != SupportStatusV0::HierarchyAmbiguousSupport
            {
                return semantic_error("ineligible context ordering/status");
            }
            previous_peak = Some(entry.peak_id);
            validate_canonical_finite(entry.area_km2, "ineligible context area")?;
            if entry.area_km2 <= 0.0 {
                return semantic_error("nonpositive ineligible context");
            }
        }
        let mut previous_portal = None;
        for entry in &record.portal_areas_km2 {
            if previous_portal.is_some_and(|previous| previous >= entry.portal_id) {
                return semantic_error("portal context ordering");
            }
            previous_portal = Some(entry.portal_id);
            validate_canonical_finite(entry.area_km2, "portal context area")?;
            if entry.area_km2 <= 0.0 {
                return semantic_error("nonpositive portal context");
            }
        }
        match record.family {
            ObjectFamilyV0::Highland if !record.portal_areas_km2.is_empty() => {
                return semantic_error("highland portal context");
            }
            ObjectFamilyV0::DrainageNode
                if record.background_area_km2 != 0.0
                    || !record.ineligible_highland_areas.is_empty() =>
            {
                return semantic_error("drainage highland context");
            }
            _ => {}
        }
    }
    Ok(())
}

struct ReconstructedCorrespondenceV0 {
    components: Vec<BestComponentV0>,
}

struct ReconstructedChannelV0 {
    family: ObjectFamilyV0,
    channel: AssignmentChannelV0,
    source_objects: Vec<AssignmentObjectInputV0>,
    target_objects: Vec<AssignmentObjectInputV0>,
    assignments: Vec<AssignmentV0>,
    components: Vec<BestComponentV0>,
}

fn reconstruct_assignment_evidence(
    value: &ObjectCorrespondenceV0,
    evidence: &SerializedEvidenceIndexV0,
) -> Result<ReconstructedCorrespondenceV0, PacketCorrespondenceErrorV0> {
    let mut previous = None;
    for assignment in &value.assignment_records {
        let key = (
            assignment.side,
            assignment.family,
            assignment.object_id,
            assignment.channel,
        );
        if previous.is_some_and(|previous| previous >= key)
            || !registered_family_channel(assignment.family, assignment.channel)
        {
            return semantic_error("assignment ordering/channel");
        }
        previous = Some(key);
        if matches!(
            assignment.support_status,
            SupportStatusV0::NoExclusiveSupport | SupportStatusV0::HierarchyAmbiguousSupport
        ) && (assignment.family != ObjectFamilyV0::Highland
            || assignment.channel != AssignmentChannelV0::HighlandExclusiveArea)
        {
            return semantic_error("assignment support status family");
        }
        validate_assignment_scalars(assignment)?;
    }

    let highland = reconstruct_channel(
        value,
        evidence,
        ObjectFamilyV0::Highland,
        AssignmentChannelV0::HighlandExclusiveArea,
    )?;
    let drainage_area = reconstruct_channel(
        value,
        evidence,
        ObjectFamilyV0::DrainageNode,
        AssignmentChannelV0::DrainageExclusiveArea,
    )?;
    let drainage_line = reconstruct_channel(
        value,
        evidence,
        ObjectFamilyV0::DrainageNode,
        AssignmentChannelV0::DrainageLine,
    )?;
    let channels = [&highland, &drainage_area, &drainage_line];
    let mut expected_assignments = channels
        .iter()
        .flat_map(|channel| channel.assignments.iter().cloned())
        .collect::<Vec<_>>();
    expected_assignments.sort_by_key(|assignment| {
        (
            assignment.side,
            assignment.family,
            assignment.object_id,
            assignment.channel,
        )
    });
    if expected_assignments != value.assignment_records {
        return semantic_error("assignment evidence mismatch");
    }

    let mut expected_components = channels
        .iter()
        .flat_map(|channel| channel.components.iter().cloned())
        .collect::<Vec<_>>();
    sort_merged_components(
        &mut expected_components,
        &[
            (
                highland.channel,
                highland.family,
                &highland.source_objects,
                &highland.target_objects,
            ),
            (
                drainage_area.channel,
                drainage_area.family,
                &drainage_area.source_objects,
                &drainage_area.target_objects,
            ),
            (
                drainage_line.channel,
                drainage_line.family,
                &drainage_line.source_objects,
                &drainage_line.target_objects,
            ),
        ],
    )?;
    if expected_components != value.best_components {
        return semantic_error("best component evidence mismatch");
    }
    let expected_conflicts = build_metric_conflicts_v0(&value.assignment_records)?;
    if expected_conflicts != value.metric_conflicts {
        return semantic_error("metric conflict evidence mismatch");
    }
    Ok(ReconstructedCorrespondenceV0 {
        components: expected_components,
    })
}

fn reconstruct_channel(
    value: &ObjectCorrespondenceV0,
    evidence: &SerializedEvidenceIndexV0,
    family: ObjectFamilyV0,
    channel: AssignmentChannelV0,
) -> Result<ReconstructedChannelV0, PacketCorrespondenceErrorV0> {
    let rows = value
        .assignment_records
        .iter()
        .filter(|assignment| assignment.family == family && assignment.channel == channel)
        .collect::<Vec<_>>();
    let source_objects =
        reconstructed_assignment_objects(&rows, evidence, channel, PacketSideV0::Source)?;
    let target_objects =
        reconstructed_assignment_objects(&rows, evidence, channel, PacketSideV0::Target)?;
    let scores = evidence
        .scores
        .iter()
        .filter(|((score_channel, _, _), _)| *score_channel == channel)
        .map(
            |((_, source_id, target_id), (source_score, target_score))| PositiveScoreV0 {
                source_id: *source_id,
                target_id: *target_id,
                source_score: *source_score,
                target_score: *target_score,
            },
        )
        .collect::<Vec<_>>();
    let output =
        build_assignment_kernel_v0(family, channel, &source_objects, &target_objects, &scores)?;
    Ok(ReconstructedChannelV0 {
        family,
        channel,
        source_objects,
        target_objects,
        assignments: output.assignments,
        components: output.best_components,
    })
}

fn reconstructed_assignment_objects(
    rows: &[&AssignmentV0],
    evidence: &SerializedEvidenceIndexV0,
    channel: AssignmentChannelV0,
    side: PacketSideV0,
) -> Result<Vec<AssignmentObjectInputV0>, PacketCorrespondenceErrorV0> {
    rows.iter()
        .filter(|assignment| assignment.side == side)
        .map(|assignment| {
            let serialized = evidence
                .objects
                .get(&(channel, side, assignment.object_id))
                .copied();
            let input_status = match assignment.support_status {
                SupportStatusV0::NoPositiveOverlap => SupportStatusV0::Eligible,
                status => status,
            };
            if assignment.support_status == SupportStatusV0::Eligible && serialized.is_none()
                || matches!(
                    assignment.support_status,
                    SupportStatusV0::NoExclusiveSupport
                        | SupportStatusV0::HierarchyAmbiguousSupport
                ) && serialized.is_some()
            {
                return semantic_error("assignment object support evidence");
            }
            let (object_measure, anchor_km) = serialized
                .map(|object| (object.measure, object.anchor))
                .unwrap_or((1.0, DVec3::ZERO));
            Ok(AssignmentObjectInputV0 {
                object_id: assignment.object_id,
                object_measure,
                anchor_km,
                support_status: input_status,
            })
        })
        .collect()
}

fn validate_assignment_scalars(
    assignment: &AssignmentV0,
) -> Result<(), PacketCorrespondenceErrorV0> {
    if !strictly_increasing(&assignment.positive_partner_ids)
        || !strictly_increasing(&assignment.maximum_partner_ids)
        || assignment
            .maximum_partner_ids
            .iter()
            .any(|id| assignment.positive_partner_ids.binary_search(id).is_err())
    {
        return semantic_error("assignment partner ordering");
    }
    for value in [
        assignment.best_score,
        assignment.second_distinct_score,
        assignment.normalized_margin,
    ]
    .into_iter()
    .flatten()
    {
        validate_canonical_finite(value, "assignment scalar")?;
        if value < 0.0 {
            return semantic_error("negative assignment scalar");
        }
    }
    match assignment.support_status {
        SupportStatusV0::Eligible => {
            if assignment.positive_partner_ids.is_empty()
                || assignment.maximum_partner_ids.is_empty()
                || assignment.best_score.is_none()
                || assignment.second_distinct_score.is_none()
                || assignment.normalized_margin.is_none()
                || assignment.exact_best_tie != (assignment.maximum_partner_ids.len() > 1)
            {
                return semantic_error("eligible assignment shape");
            }
        }
        SupportStatusV0::NoExclusiveSupport
        | SupportStatusV0::HierarchyAmbiguousSupport
        | SupportStatusV0::NoPositiveOverlap => {
            if !assignment.positive_partner_ids.is_empty()
                || !assignment.maximum_partner_ids.is_empty()
                || assignment.best_score.is_some()
                || assignment.second_distinct_score.is_some()
                || assignment.normalized_margin.is_some()
                || assignment.exact_best_tie
            {
                return semantic_error("unavailable assignment shape");
            }
        }
    }
    Ok(())
}

fn registered_family_channel(family: ObjectFamilyV0, channel: AssignmentChannelV0) -> bool {
    matches!(
        (family, channel),
        (
            ObjectFamilyV0::Highland,
            AssignmentChannelV0::HighlandExclusiveArea
        ) | (
            ObjectFamilyV0::DrainageNode,
            AssignmentChannelV0::DrainageExclusiveArea
        ) | (
            ObjectFamilyV0::DrainageNode,
            AssignmentChannelV0::DrainageLine
        )
    )
}

fn strictly_increasing(values: &[u32]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn validate_context_completeness(
    contexts: &[ContextV0],
    assignments: &[AssignmentV0],
) -> Result<(), PacketCorrespondenceErrorV0> {
    let expected = assignments
        .iter()
        .filter(|assignment| {
            matches!(
                assignment.channel,
                AssignmentChannelV0::HighlandExclusiveArea
                    | AssignmentChannelV0::DrainageExclusiveArea
            ) && matches!(
                assignment.support_status,
                SupportStatusV0::Eligible | SupportStatusV0::NoPositiveOverlap
            )
        })
        .map(|assignment| (assignment.side, assignment.family, assignment.object_id))
        .collect::<BTreeSet<_>>();
    let actual = contexts
        .iter()
        .map(|context| (context.side, context.family, context.object_id))
        .collect::<BTreeSet<_>>();
    if actual.len() != contexts.len() || actual != expected {
        return semantic_error("context object completeness");
    }
    Ok(())
}

fn validate_pair_assignment_references(
    value: &ObjectCorrespondenceV0,
) -> Result<(), PacketCorrespondenceErrorV0> {
    let assignments = value
        .assignment_records
        .iter()
        .map(|assignment| {
            (
                assignment.side,
                assignment.family,
                assignment.channel,
                assignment.object_id,
            )
        })
        .collect::<BTreeSet<_>>();
    let require = |side, family, channel, id| {
        if assignments.contains(&(side, family, channel, id)) {
            Ok(())
        } else {
            semantic_error("pair/context assignment reference")
        }
    };
    for row in value
        .highland_nested_pairs
        .iter()
        .chain(&value.highland_exclusive_pairs)
    {
        require(
            PacketSideV0::Source,
            ObjectFamilyV0::Highland,
            AssignmentChannelV0::HighlandExclusiveArea,
            row.source_id,
        )?;
        require(
            PacketSideV0::Target,
            ObjectFamilyV0::Highland,
            AssignmentChannelV0::HighlandExclusiveArea,
            row.target_id,
        )?;
    }
    for row in value
        .drainage_nested_pairs
        .iter()
        .chain(&value.drainage_exclusive_pairs)
    {
        for channel in [
            AssignmentChannelV0::DrainageExclusiveArea,
            AssignmentChannelV0::DrainageLine,
        ] {
            require(
                PacketSideV0::Source,
                ObjectFamilyV0::DrainageNode,
                channel,
                row.source_id,
            )?;
            require(
                PacketSideV0::Target,
                ObjectFamilyV0::DrainageNode,
                channel,
                row.target_id,
            )?;
        }
    }
    for row in &value.drainage_line_pairs {
        for channel in [
            AssignmentChannelV0::DrainageExclusiveArea,
            AssignmentChannelV0::DrainageLine,
        ] {
            require(
                PacketSideV0::Source,
                ObjectFamilyV0::DrainageNode,
                channel,
                row.source_id,
            )?;
            require(
                PacketSideV0::Target,
                ObjectFamilyV0::DrainageNode,
                channel,
                row.target_id,
            )?;
        }
    }
    for context in &value.context_records {
        let channel = match context.family {
            ObjectFamilyV0::Highland => AssignmentChannelV0::HighlandExclusiveArea,
            ObjectFamilyV0::DrainageNode => AssignmentChannelV0::DrainageExclusiveArea,
        };
        require(context.side, context.family, channel, context.object_id)?;
        if context.family == ObjectFamilyV0::DrainageNode {
            require(
                context.side,
                context.family,
                AssignmentChannelV0::DrainageLine,
                context.object_id,
            )?;
        }
    }
    Ok(())
}

fn validate_topology_semantics(
    records: &[TopologyV0],
    assignments: &[AssignmentV0],
    components: &[BestComponentV0],
) -> Result<(), PacketCorrespondenceErrorV0> {
    let assignment_by_key = assignments
        .iter()
        .map(|assignment| {
            (
                (
                    assignment.side,
                    assignment.family,
                    assignment.channel,
                    assignment.object_id,
                ),
                assignment,
            )
        })
        .collect::<BTreeMap<_, _>>();
    if assignment_by_key.len() != assignments.len() {
        return semantic_error("duplicate assignment topology key");
    }
    let mut topology_by_key = BTreeMap::new();
    let mut previous = None;
    for record in records {
        let key = (record.side, record.family, record.channel, record.from_id);
        if previous.is_some_and(|previous| previous >= key)
            || !registered_family_channel(record.family, record.channel)
            || !valid_topology_target(record.family, record.target)
            || topology_by_key.insert(key, record.target).is_some()
        {
            return semantic_error("topology ordering/target");
        }
        previous = Some(key);
    }
    if topology_by_key.keys().copied().collect::<BTreeSet<_>>()
        != assignment_by_key.keys().copied().collect::<BTreeSet<_>>()
    {
        return semantic_error("topology object completeness");
    }
    for record in records {
        let from = assignment_by_key
            .get(&(record.side, record.family, record.channel, record.from_id))
            .copied()
            .ok_or(PacketCorrespondenceErrorV0::Incompatible(
                "topology source assignment",
            ))?;
        let object_target = topology_target_object_id(record.family, record.target);
        let target_assignment = object_target.and_then(|id| {
            assignment_by_key
                .get(&(record.side, record.family, record.channel, id))
                .copied()
        });
        let missing_endpoint = from.maximum_partner_ids.is_empty()
            || object_target.is_some()
                && target_assignment.is_none_or(|target| target.maximum_partner_ids.is_empty());
        if record.family == ObjectFamilyV0::Highland
            && from.support_status == SupportStatusV0::HierarchyAmbiguousSupport
            && record.availability != TopologyAvailabilityV0::HierarchyAmbiguous
        {
            return semantic_error("ambiguous topology precedence");
        }
        match record.availability {
            TopologyAvailabilityV0::HierarchyAmbiguous => {
                if record.family != ObjectFamilyV0::Highland
                    || from.support_status != SupportStatusV0::HierarchyAmbiguousSupport
                    || record.mapped_adjacency.is_some()
                    || record.endpoints_in_same_best_component.is_some()
                {
                    return semantic_error("hierarchy-ambiguous topology shape");
                }
            }
            TopologyAvailabilityV0::NoMappedEndpoint => {
                if !missing_endpoint
                    || record.mapped_adjacency.is_some()
                    || record.endpoints_in_same_best_component.is_some()
                {
                    return semantic_error("unmapped topology shape");
                }
            }
            TopologyAvailabilityV0::Available => {
                if missing_endpoint {
                    return semantic_error("available topology endpoint");
                }
                let expected =
                    derive_mapped_adjacency(record, from, target_assignment, &topology_by_key)?;
                if record.mapped_adjacency != Some(expected) {
                    return semantic_error("mapped topology adjacency");
                }
                let expected_component = object_target.map(|target_id| {
                    topology_endpoints_share_component(
                        record.side,
                        record.family,
                        record.channel,
                        record.from_id,
                        target_id,
                        components,
                    )
                });
                if record.endpoints_in_same_best_component != expected_component {
                    return semantic_error("topology component evidence");
                }
            }
        }
    }
    // The same packet-local raw edge must not change between drainage channels.
    for side in [PacketSideV0::Source, PacketSideV0::Target] {
        let area = records
            .iter()
            .filter(|record| {
                record.side == side
                    && record.family == ObjectFamilyV0::DrainageNode
                    && record.channel == AssignmentChannelV0::DrainageExclusiveArea
            })
            .map(|record| (record.from_id, record.target))
            .collect::<BTreeMap<_, _>>();
        let line = records
            .iter()
            .filter(|record| {
                record.side == side
                    && record.family == ObjectFamilyV0::DrainageNode
                    && record.channel == AssignmentChannelV0::DrainageLine
            })
            .map(|record| (record.from_id, record.target))
            .collect::<BTreeMap<_, _>>();
        if area != line {
            return semantic_error("drainage topology channel mismatch");
        }
    }
    Ok(())
}

fn derive_mapped_adjacency(
    record: &TopologyV0,
    from: &AssignmentV0,
    target_assignment: Option<&AssignmentV0>,
    topology: &BTreeMap<(PacketSideV0, ObjectFamilyV0, AssignmentChannelV0, u32), TopologyTargetV0>,
) -> Result<MappedAdjacencyV0, PacketCorrespondenceErrorV0> {
    let opposite = match record.side {
        PacketSideV0::Source => PacketSideV0::Target,
        PacketSideV0::Target => PacketSideV0::Source,
    };
    let mut matching = 0usize;
    let total;
    if let Some(target_assignment) = target_assignment {
        total = from
            .maximum_partner_ids
            .len()
            .checked_mul(target_assignment.maximum_partner_ids.len())
            .ok_or(PacketCorrespondenceErrorV0::Numerical(
                "topology product overflow",
            ))?;
        for &mapped_from in &from.maximum_partner_ids {
            let mapped_target = topology
                .get(&(opposite, record.family, record.channel, mapped_from))
                .copied()
                .ok_or(PacketCorrespondenceErrorV0::Incompatible(
                    "opposite topology object",
                ))?;
            for &mapped_to in &target_assignment.maximum_partner_ids {
                if mapped_target == topology_object_target(record.family, mapped_to) {
                    matching += 1;
                }
            }
        }
    } else {
        total = from.maximum_partner_ids.len();
        for &mapped_from in &from.maximum_partner_ids {
            let mapped_target = topology
                .get(&(opposite, record.family, record.channel, mapped_from))
                .copied()
                .ok_or(PacketCorrespondenceErrorV0::Incompatible(
                    "opposite terminal topology object",
                ))?;
            if mapped_target == record.target {
                matching += 1;
            }
        }
    }
    Ok(if matching == total {
        MappedAdjacencyV0::All
    } else if matching == 0 {
        MappedAdjacencyV0::None
    } else {
        MappedAdjacencyV0::Some
    })
}

fn valid_topology_target(family: ObjectFamilyV0, target: TopologyTargetV0) -> bool {
    matches!(
        (family, target),
        (ObjectFamilyV0::Highland, TopologyTargetV0::Highland(_))
            | (ObjectFamilyV0::Highland, TopologyTargetV0::HighlandRoot)
            | (
                ObjectFamilyV0::DrainageNode,
                TopologyTargetV0::DrainageNode(_)
            )
            | (ObjectFamilyV0::DrainageNode, TopologyTargetV0::Portal(_))
    )
}

fn topology_target_object_id(family: ObjectFamilyV0, target: TopologyTargetV0) -> Option<u32> {
    match (family, target) {
        (ObjectFamilyV0::Highland, TopologyTargetV0::Highland(id))
        | (ObjectFamilyV0::DrainageNode, TopologyTargetV0::DrainageNode(id)) => Some(id),
        _ => None,
    }
}

fn topology_object_target(family: ObjectFamilyV0, id: u32) -> TopologyTargetV0 {
    match family {
        ObjectFamilyV0::Highland => TopologyTargetV0::Highland(id),
        ObjectFamilyV0::DrainageNode => TopologyTargetV0::DrainageNode(id),
    }
}

fn topology_endpoints_share_component(
    side: PacketSideV0,
    family: ObjectFamilyV0,
    channel: AssignmentChannelV0,
    from_id: u32,
    target_id: u32,
    components: &[BestComponentV0],
) -> bool {
    let from = BestMemberV0 {
        side,
        family,
        object_id: from_id,
    };
    let target = BestMemberV0 {
        side,
        family,
        object_id: target_id,
    };
    components.iter().any(|component| {
        component.channel == channel
            && component.members.contains(&from)
            && component.members.contains(&target)
    })
}

fn validate_work_counts(value: &ObjectCorrespondenceV0) -> Result<(), PacketCorrespondenceErrorV0> {
    let counts = &value.work_counts;
    let cell_product = counts.source_cells.checked_mul(counts.target_cells).ok_or(
        PacketCorrespondenceErrorV0::Numerical("cell work product overflow"),
    )?;
    let segment_product = counts
        .source_segments
        .checked_mul(counts.target_segments)
        .ok_or(PacketCorrespondenceErrorV0::Numerical(
            "segment work product overflow",
        ))?;
    let nested_rows = counts
        .positive_highland_nested_rows
        .checked_add(counts.positive_drainage_nested_rows)
        .ok_or(PacketCorrespondenceErrorV0::Numerical(
            "nested row count overflow",
        ))?;
    let exact_rows = [
        (
            counts.positive_highland_nested_rows,
            value.highland_nested_pairs.len() as u64,
        ),
        (
            counts.positive_highland_exclusive_rows,
            value.highland_exclusive_pairs.len() as u64,
        ),
        (
            counts.positive_drainage_nested_rows,
            value.drainage_nested_pairs.len() as u64,
        ),
        (
            counts.positive_drainage_exclusive_rows,
            value.drainage_exclusive_pairs.len() as u64,
        ),
        (
            counts.positive_line_rows,
            value.drainage_line_pairs.len() as u64,
        ),
        (
            counts.best_graph_edges,
            count_best_graph_edges(&value.assignment_records)?,
        ),
    ];
    if exact_rows.iter().any(|(stored, derived)| stored != derived)
        || counts.source_cells == 0
        || counts.target_cells == 0
        || counts.source_segments == 0
        || counts.target_segments == 0
        || counts.polygon_clips != counts.cell_box_candidates
        || counts.segment_pair_tests != counts.segment_box_candidates
        || counts.positive_cell_intersections > counts.polygon_clips
        || counts.nested_membership_contributions < nested_rows
        || counts.cell_box_candidates > cell_product
        || counts.segment_box_candidates > segment_product
        || (!value.highland_nested_pairs.is_empty()
            || !value.highland_exclusive_pairs.is_empty()
            || !value.drainage_nested_pairs.is_empty()
            || !value.drainage_exclusive_pairs.is_empty())
            && counts.positive_cell_intersections == 0
        || !value.drainage_line_pairs.is_empty() && counts.segment_pair_tests == 0
    {
        return semantic_error("correspondence work counts");
    }
    Ok(())
}

fn validate_canonical_finite(
    value: f64,
    message: &'static str,
) -> Result<(), PacketCorrespondenceErrorV0> {
    if !value.is_finite() || value == 0.0 && value.to_bits() != 0 {
        return semantic_error(message);
    }
    Ok(())
}

fn validate_planar_point(
    point: DVec3,
    message: &'static str,
) -> Result<(), PacketCorrespondenceErrorV0> {
    for value in [point.x, point.y, point.z] {
        validate_canonical_finite(value, message)?;
    }
    if point.z != 0.0 {
        return semantic_error(message);
    }
    Ok(())
}

fn same_float(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

fn semantic_error<T>(message: &'static str) -> Result<T, PacketCorrespondenceErrorV0> {
    Err(PacketCorrespondenceErrorV0::Incompatible(message))
}

fn correspondence_preimage_hash(
    value: &ObjectCorrespondenceV0,
) -> Result<u64, PacketCorrespondenceErrorV0> {
    let bytes = fixed_bytes(&(
        &value.schema_version,
        &value.hash_version,
        &value.config,
        value.source_packet_hash,
        value.target_packet_hash,
        &value.highland_nested_pairs,
        &value.highland_exclusive_pairs,
        &value.drainage_nested_pairs,
        &value.drainage_exclusive_pairs,
        &value.drainage_line_pairs,
        &value.context_records,
        &value.assignment_records,
        &value.best_components,
        &value.metric_conflicts,
        &value.topology_records,
        &value.work_counts,
    ))?;
    Ok(fnv1a64(&bytes))
}

fn fixed_bytes(value: &impl Serialize) -> Result<Vec<u8>, PacketCorrespondenceErrorV0> {
    bincode_options()
        .serialize(value)
        .map_err(|error| PacketCorrespondenceErrorV0::Serialization(error.to_string()))
}

fn bincode_options() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .reject_trailing_bytes()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod packet_kernel_tests {
    use super::*;

    fn hierarchy_peak(id: u32, parent_peak: Option<u32>, ambiguous_edge: bool) -> PeakBranchV0 {
        PeakBranchV0 {
            id,
            peak_elevation_km: 2.0,
            anchor_cell: id,
            flat_centroid_km: DVec3::new(f64::from(id), 0.0, 0.0),
            flat_maximum_cells: vec![id],
            parent_peak,
            key_saddle: None,
            persistence_km: 1.0,
            root_closure: parent_peak.is_none(),
            equal_elder_ambiguous: ambiguous_edge,
            exclusive_cells: vec![id],
            footprint_members: (0..=id).collect(),
            footprint_area_km2: f64::from(id + 1),
            union_boundary_edges: Vec::new(),
            physical_boundary_segments: Vec::new(),
            scored_boundary_contact: false,
        }
    }

    fn hierarchy_for_ambiguity(peaks: Vec<PeakBranchV0>) -> SurfaceHierarchyV0 {
        let cell_count = peaks.len();
        SurfaceHierarchyV0 {
            schema_version: G0S0_SCHEMA_VERSION.into(),
            hash_version: G0S0_HASH_VERSION.into(),
            peaks,
            saddles: Vec::new(),
            roots: vec![0],
            cell_peak_owner: (0..cell_count).map(|cell| Some(cell as u32)).collect(),
            populations: HighlandPopulationsV0 {
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

    fn rectangle(x0: f64, x1: f64, y0: f64, y1: f64) -> Vec<DVec3> {
        vec![
            DVec3::new(x0, y0, 0.0),
            DVec3::new(x1, y0, 0.0),
            DVec3::new(x1, y1, 0.0),
            DVec3::new(x0, y1, 0.0),
        ]
    }

    fn graph(polygons: Vec<Vec<DVec3>>) -> EvaluationSurfaceGraphV0 {
        let mut offsets = Vec::with_capacity(polygons.len() + 1);
        let mut vertices = Vec::new();
        let mut centers = Vec::new();
        let mut areas = Vec::new();
        for polygon in polygons {
            offsets.push(vertices.len() as u32);
            let x0 = polygon[0].x;
            let y0 = polygon[0].y;
            let x1 = polygon[2].x;
            let y1 = polygon[2].y;
            centers.push(DVec3::new(0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.0));
            areas.push((x1 - x0) * (y1 - y0));
            vertices.extend(polygon);
        }
        offsets.push(vertices.len() as u32);
        EvaluationSurfaceGraphV0 {
            domain: EvaluationDomainV0::Planar,
            cell_center_km: centers,
            cell_area_km2: areas,
            cell_polygon_offsets: offsets,
            cell_polygon_vertices_km: vertices,
            edge_offsets: vec![0; 1],
            edge_neighbor: Vec::new(),
            edge_reciprocal: Vec::new(),
            edge_distance_km: Vec::new(),
            edge_shared_width_km: Vec::new(),
            edge_face_endpoints_km: Vec::new(),
            boundary_segments: Vec::new(),
        }
    }

    fn object(id: u32, status: SupportStatusV0, cell: u32) -> PacketAreaObjectV0 {
        PacketAreaObjectV0 {
            id,
            status,
            nested_cells: vec![cell],
            exclusive_cells: vec![cell],
        }
    }

    fn exclusive_scores(rows: &[AreaPairV0]) -> Vec<PositiveScoreV0> {
        rows.iter()
            .map(|row| PositiveScoreV0 {
                source_id: row.source_id,
                target_id: row.target_id,
                source_score: row.source_coverage,
                target_score: row.target_coverage,
            })
            .collect()
    }

    #[test]
    fn ambiguity_stops_at_the_nearest_retained_ancestor_edge() {
        let hierarchy = hierarchy_for_ambiguity(vec![
            hierarchy_peak(0, None, true),
            hierarchy_peak(1, Some(0), false),
        ]);
        let references = BTreeSet::from([0, 1]);

        assert!(!highland_support_is_ambiguous(&hierarchy, 1, &references).unwrap());
        assert!(highland_support_is_ambiguous(&hierarchy, 0, &references).unwrap());
    }

    fn equal_elder_counterfactual(root_id: u32) -> (SurfaceHierarchyV0, PacketAreaPopulationV0) {
        assert!(root_id <= 1);
        let child_id = 1 - root_id;
        let (nested, exclusive, owners): ([Vec<u32>; 2], [Vec<u32>; 2], [u32; 3]) = if root_id == 0
        {
            (
                [vec![0, 1, 2], vec![1, 2]],
                [vec![0], vec![1, 2]],
                [0, 1, 1],
            )
        } else {
            (
                [vec![0, 1], vec![0, 1, 2]],
                [vec![0, 1], vec![2]],
                [0, 0, 1],
            )
        };
        let mut peaks = (0..2)
            .map(|id| {
                let is_child = id == child_id;
                let mut peak = hierarchy_peak(id, is_child.then_some(root_id), is_child);
                peak.anchor_cell = if id == 0 { 0 } else { 2 };
                peak.flat_centroid_km = DVec3::new(10.0 + 10.0 * f64::from(id), 5.0, 0.0);
                peak.flat_maximum_cells = vec![peak.anchor_cell];
                peak.key_saddle = is_child.then_some(0);
                peak.exclusive_cells = exclusive[id as usize].clone();
                peak.footprint_members = nested[id as usize].clone();
                peak.footprint_area_km2 = 100.0 * peak.footprint_members.len() as f64;
                peak
            })
            .collect::<Vec<_>>();
        peaks.sort_by_key(|peak| peak.id);
        let hierarchy = SurfaceHierarchyV0 {
            schema_version: G0S0_SCHEMA_VERSION.into(),
            hash_version: G0S0_HASH_VERSION.into(),
            peaks,
            saddles: vec![SaddleNodeV0 {
                id: 0,
                elevation_km: 1.0,
                anchor_cell: 1,
                flat_centroid_km: DVec3::new(15.0, 5.0, 0.0),
                flat_saddle_cells: vec![1],
                elder_peak: root_id,
                losing_peaks: vec![child_id],
                equal_elder_ambiguous: true,
            }],
            roots: vec![root_id],
            cell_peak_owner: owners.into_iter().map(Some).collect(),
            populations: HighlandPopulationsV0 {
                reference: vec![0, 1],
                persistence_low: Vec::new(),
                persistence_high: Vec::new(),
                footprint_low: Vec::new(),
                footprint_high: Vec::new(),
            },
            reference_highlands: Vec::new(),
            derived_evidence_hash: 0,
        };
        let references = BTreeSet::from([0, 1]);
        let objects = (0..2)
            .map(|id| {
                let status = if highland_support_is_ambiguous(&hierarchy, id, &references).unwrap()
                {
                    SupportStatusV0::HierarchyAmbiguousSupport
                } else {
                    SupportStatusV0::Eligible
                };
                PacketAreaObjectV0 {
                    id,
                    status,
                    nested_cells: nested[id as usize].clone(),
                    exclusive_cells: exclusive[id as usize].clone(),
                }
            })
            .collect::<Vec<_>>();
        let mut cell_class = vec![PacketCellClassV0::HighlandBackground; 3];
        for object in &objects {
            for &cell in &object.exclusive_cells {
                cell_class[cell as usize] = PacketCellClassV0::IneligibleHighland {
                    peak_id: object.id,
                    status: object.status,
                };
            }
        }
        (
            hierarchy,
            PacketAreaPopulationV0 {
                family: ObjectFamilyV0::Highland,
                objects,
                cell_class,
            },
        )
    }

    fn assert_equal_elder_counterfactual(
        geometry: &EvaluationSurfaceGraphV0,
        root_id: u32,
        expected_nested_area: [(u32, f64); 2],
        expected_exclusive_area: [(u32, f64); 2],
    ) {
        let child_id = 1 - root_id;
        let (hierarchy, ambiguous) = equal_elder_counterfactual(root_id);
        assert_eq!(hierarchy.roots, vec![root_id]);
        assert_eq!(
            hierarchy.peaks[child_id as usize].parent_peak,
            Some(root_id)
        );
        assert_eq!(hierarchy.peaks[child_id as usize].key_saddle, Some(0));
        assert!(hierarchy.peaks[child_id as usize].equal_elder_ambiguous);
        assert!(hierarchy.saddles[0].equal_elder_ambiguous);
        assert!(ambiguous
            .objects
            .iter()
            .all(|object| object.status == SupportStatusV0::HierarchyAmbiguousSupport));

        let eligible = PacketAreaPopulationV0 {
            family: ObjectFamilyV0::Highland,
            objects: vec![PacketAreaObjectV0 {
                id: 10,
                status: SupportStatusV0::Eligible,
                nested_cells: vec![0, 1, 2],
                exclusive_cells: vec![0, 1, 2],
            }],
            cell_class: vec![PacketCellClassV0::EligibleObject(10); 3],
        };
        let forward = build_area_population_kernel_v0(
            geometry, &ambiguous, geometry, &eligible, 1.0e-8, 1.0e-10,
        )
        .unwrap();
        let reverse = build_area_population_kernel_v0(
            geometry, &eligible, geometry, &ambiguous, 1.0e-8, 1.0e-10,
        )
        .unwrap();

        assert_eq!(
            forward
                .nested_pairs
                .iter()
                .map(|row| (row.source_id, row.intersection_area_km2))
                .collect::<Vec<_>>(),
            expected_nested_area
        );
        assert_eq!(
            reverse
                .nested_pairs
                .iter()
                .map(|row| (row.target_id, row.intersection_area_km2))
                .collect::<Vec<_>>(),
            expected_nested_area
        );
        assert!(forward.exclusive_pairs.is_empty());
        assert!(reverse.exclusive_pairs.is_empty());

        for (output, marked_side, context_side) in [
            (&forward, PacketSideV0::Source, PacketSideV0::Target),
            (&reverse, PacketSideV0::Target, PacketSideV0::Source),
        ] {
            let assignment = build_assignment_kernel_v0(
                ObjectFamilyV0::Highland,
                AssignmentChannelV0::HighlandExclusiveArea,
                &output.assignment_objects_source,
                &output.assignment_objects_target,
                &exclusive_scores(&output.exclusive_pairs),
            )
            .unwrap();
            assert!(assignment.best_components.is_empty());
            assert!(assignment.assignments.iter().all(|row| {
                row.maximum_partner_ids.is_empty()
                    && row.best_score.is_none()
                    && row.second_distinct_score.is_none()
                    && row.normalized_margin.is_none()
            }));
            assert!(assignment
                .assignments
                .iter()
                .filter(|row| row.side == marked_side)
                .all(|row| row.support_status == SupportStatusV0::HierarchyAmbiguousSupport));

            assert_eq!(output.context_records.len(), 1);
            let context = &output.context_records[0];
            assert_eq!(context.side, context_side);
            assert_eq!(context.object_id, 10);
            assert_eq!(
                context
                    .ineligible_highland_areas
                    .iter()
                    .map(|entry| (entry.peak_id, entry.area_km2))
                    .collect::<Vec<_>>(),
                expected_exclusive_area
            );
            assert_eq!(context.background_area_km2, 0.0);
            assert_eq!(context.outside_domain_area_km2, 0.0);

            let edges = [0, 1].map(|id| TopologyEdgeInputV0 {
                from_id: id,
                target: if id == root_id {
                    TopologyTargetV0::HighlandRoot
                } else {
                    TopologyTargetV0::Highland(root_id)
                },
                hierarchy_ambiguous: true,
            });
            let topology = build_topology_records_v0(
                marked_side,
                ObjectFamilyV0::Highland,
                AssignmentChannelV0::HighlandExclusiveArea,
                &edges,
                &[TopologyObjectInputV0 {
                    object_id: 10,
                    target: TopologyTargetV0::HighlandRoot,
                }],
                &assignment.assignments,
                &assignment.best_components,
                &[],
            )
            .unwrap();
            assert_eq!(topology.len(), 2);
            assert!(topology.iter().all(|record| {
                record.availability == TopologyAvailabilityV0::HierarchyAmbiguous
                    && record.mapped_adjacency.is_none()
                    && record.endpoints_in_same_best_component.is_none()
            }));
        }
    }

    #[test]
    fn both_equal_elder_counterfactuals_exclude_only_ambiguous_support_evidence() {
        let geometry = graph(vec![
            rectangle(0.0, 10.0, 0.0, 10.0),
            rectangle(10.0, 20.0, 0.0, 10.0),
            rectangle(20.0, 30.0, 0.0, 10.0),
        ]);
        assert_equal_elder_counterfactual(
            &geometry,
            0,
            [(0, 300.0), (1, 200.0)],
            [(0, 100.0), (1, 200.0)],
        );
        assert_equal_elder_counterfactual(
            &geometry,
            1,
            [(0, 200.0), (1, 300.0)],
            [(0, 200.0), (1, 100.0)],
        );
    }

    #[test]
    fn nested_parent_children_and_parent_only_target_keep_exact_exclusive_graph() {
        let geometry = graph(vec![
            rectangle(0.0, 40.0, 0.0, 100.0),
            rectangle(40.0, 60.0, 0.0, 100.0),
            rectangle(60.0, 100.0, 0.0, 100.0),
        ]);
        let hierarchy = PacketAreaPopulationV0 {
            family: ObjectFamilyV0::Highland,
            objects: vec![
                PacketAreaObjectV0 {
                    id: 0,
                    status: SupportStatusV0::Eligible,
                    nested_cells: vec![0, 1, 2],
                    exclusive_cells: vec![1],
                },
                object(1, SupportStatusV0::Eligible, 0),
                object(2, SupportStatusV0::Eligible, 2),
            ],
            cell_class: vec![
                PacketCellClassV0::EligibleObject(1),
                PacketCellClassV0::EligibleObject(0),
                PacketCellClassV0::EligibleObject(2),
            ],
        };
        let identical = build_area_population_kernel_v0(
            &geometry, &hierarchy, &geometry, &hierarchy, 1.0e-8, 1.0e-10,
        )
        .unwrap();
        assert_eq!(identical.nested_pairs.len(), 7);
        assert_eq!(identical.exclusive_pairs.len(), 3);
        assert_eq!(
            identical
                .exclusive_pairs
                .iter()
                .map(|row| (row.source_id, row.target_id, row.intersection_area_km2))
                .collect::<Vec<_>>(),
            vec![(0, 0, 2_000.0), (1, 1, 4_000.0), (2, 2, 4_000.0)]
        );

        let parent_only = PacketAreaPopulationV0 {
            family: ObjectFamilyV0::Highland,
            objects: vec![PacketAreaObjectV0 {
                id: 10,
                status: SupportStatusV0::Eligible,
                nested_cells: vec![0, 1, 2],
                exclusive_cells: vec![0, 1, 2],
            }],
            cell_class: vec![PacketCellClassV0::EligibleObject(10); 3],
        };
        let parent_result = build_area_population_kernel_v0(
            &geometry,
            &hierarchy,
            &geometry,
            &parent_only,
            1.0e-8,
            1.0e-10,
        )
        .unwrap();
        let assignment = build_assignment_kernel_v0(
            ObjectFamilyV0::Highland,
            AssignmentChannelV0::HighlandExclusiveArea,
            &parent_result.assignment_objects_source,
            &parent_result.assignment_objects_target,
            &exclusive_scores(&parent_result.exclusive_pairs),
        )
        .unwrap();
        let target = assignment
            .assignments
            .iter()
            .find(|row| row.side == PacketSideV0::Target && row.object_id == 10)
            .unwrap();
        assert_eq!(target.maximum_partner_ids, vec![1, 2]);
        assert!(target.exact_best_tie);
        assert_eq!(assignment.best_components.len(), 1);
        assert_eq!(
            assignment.best_components[0].kind,
            ComponentKindV0::ManyToOneBest
        );
    }

    #[test]
    fn fully_partitioning_children_leave_parent_with_typed_zero_exclusive_support() {
        let geometry = graph(vec![
            rectangle(0.0, 50.0, 0.0, 100.0),
            rectangle(50.0, 100.0, 0.0, 100.0),
        ]);
        let population = PacketAreaPopulationV0 {
            family: ObjectFamilyV0::Highland,
            objects: vec![
                PacketAreaObjectV0 {
                    id: 0,
                    status: SupportStatusV0::NoExclusiveSupport,
                    nested_cells: vec![0, 1],
                    exclusive_cells: Vec::new(),
                },
                object(1, SupportStatusV0::Eligible, 0),
                object(2, SupportStatusV0::Eligible, 1),
            ],
            cell_class: vec![
                PacketCellClassV0::EligibleObject(1),
                PacketCellClassV0::EligibleObject(2),
            ],
        };
        let result = build_area_population_kernel_v0(
            &geometry,
            &population,
            &geometry,
            &population,
            1.0e-8,
            1.0e-10,
        )
        .unwrap();
        assert!(result
            .exclusive_pairs
            .iter()
            .all(|row| row.source_id != 0 && row.target_id != 0));
        let assignment = build_assignment_kernel_v0(
            ObjectFamilyV0::Highland,
            AssignmentChannelV0::HighlandExclusiveArea,
            &result.assignment_objects_source,
            &result.assignment_objects_target,
            &exclusive_scores(&result.exclusive_pairs),
        )
        .unwrap();
        assert!(assignment
            .assignments
            .iter()
            .filter(|row| row.object_id == 0)
            .all(|row| {
                row.support_status == SupportStatusV0::NoExclusiveSupport
                    && row.maximum_partner_ids.is_empty()
                    && row.normalized_margin.is_none()
            }));
    }

    #[test]
    fn shared_area_path_reports_exact_background_and_portal_context() {
        let geometry = graph(vec![
            rectangle(0.0, 10.0, 0.0, 10.0),
            rectangle(10.0, 20.0, 0.0, 10.0),
        ]);
        let source = PacketAreaPopulationV0 {
            family: ObjectFamilyV0::Highland,
            objects: vec![object(1, SupportStatusV0::Eligible, 0)],
            cell_class: vec![
                PacketCellClassV0::EligibleObject(1),
                PacketCellClassV0::HighlandBackground,
            ],
        };
        let target = PacketAreaPopulationV0 {
            family: ObjectFamilyV0::Highland,
            objects: vec![object(2, SupportStatusV0::Eligible, 1)],
            cell_class: vec![
                PacketCellClassV0::HighlandBackground,
                PacketCellClassV0::EligibleObject(2),
            ],
        };
        let output = build_area_population_kernel_v0(
            &geometry, &source, &geometry, &target, 1.0e-8, 1.0e-10,
        )
        .unwrap();
        assert!(output.exclusive_pairs.is_empty());
        assert_eq!(output.context_records.len(), 2);
        assert!(output
            .context_records
            .iter()
            .all(|record| record.background_area_km2 == 100.0));

        let drainage_source = PacketAreaPopulationV0 {
            family: ObjectFamilyV0::DrainageNode,
            objects: vec![object(1, SupportStatusV0::Eligible, 0)],
            cell_class: vec![
                PacketCellClassV0::EligibleObject(1),
                PacketCellClassV0::Portal(7),
            ],
        };
        let drainage_target = PacketAreaPopulationV0 {
            family: ObjectFamilyV0::DrainageNode,
            objects: vec![object(2, SupportStatusV0::Eligible, 1)],
            cell_class: vec![
                PacketCellClassV0::Portal(7),
                PacketCellClassV0::EligibleObject(2),
            ],
        };
        let output = build_area_population_kernel_v0(
            &geometry,
            &drainage_source,
            &geometry,
            &drainage_target,
            1.0e-8,
            1.0e-10,
        )
        .unwrap();
        assert!(output.context_records.iter().all(|record| {
            record.portal_areas_km2
                == vec![PortalAreaV0 {
                    portal_id: 7,
                    area_km2: 100.0,
                }]
        }));
    }

    #[test]
    fn shared_area_path_types_ambiguity_and_two_way_outside_domain() {
        let two_cells = graph(vec![
            rectangle(0.0, 10.0, 0.0, 10.0),
            rectangle(10.0, 20.0, 0.0, 10.0),
        ]);
        let source = PacketAreaPopulationV0 {
            family: ObjectFamilyV0::Highland,
            objects: vec![
                object(1, SupportStatusV0::Eligible, 0),
                object(3, SupportStatusV0::HierarchyAmbiguousSupport, 1),
            ],
            cell_class: vec![
                PacketCellClassV0::EligibleObject(1),
                PacketCellClassV0::IneligibleHighland {
                    peak_id: 3,
                    status: SupportStatusV0::HierarchyAmbiguousSupport,
                },
            ],
        };
        let target = PacketAreaPopulationV0 {
            family: ObjectFamilyV0::Highland,
            objects: vec![
                object(2, SupportStatusV0::HierarchyAmbiguousSupport, 0),
                object(4, SupportStatusV0::Eligible, 1),
            ],
            cell_class: vec![
                PacketCellClassV0::IneligibleHighland {
                    peak_id: 2,
                    status: SupportStatusV0::HierarchyAmbiguousSupport,
                },
                PacketCellClassV0::EligibleObject(4),
            ],
        };
        let output = build_area_population_kernel_v0(
            &two_cells, &source, &two_cells, &target, 1.0e-8, 1.0e-10,
        )
        .unwrap();
        assert!(output.exclusive_pairs.is_empty());
        assert_eq!(output.context_records.len(), 2);
        assert!(output.context_records.iter().all(|record| {
            record.ineligible_highland_areas.len() == 1
                && record.ineligible_highland_areas[0].area_km2 == 100.0
        }));

        let source_graph = graph(vec![rectangle(0.0, 10.0, 0.0, 10.0)]);
        let target_graph = graph(vec![rectangle(5.0, 15.0, 0.0, 10.0)]);
        let source = PacketAreaPopulationV0 {
            family: ObjectFamilyV0::Highland,
            objects: vec![object(1, SupportStatusV0::Eligible, 0)],
            cell_class: vec![PacketCellClassV0::EligibleObject(1)],
        };
        let target = PacketAreaPopulationV0 {
            family: ObjectFamilyV0::Highland,
            objects: vec![object(2, SupportStatusV0::Eligible, 0)],
            cell_class: vec![PacketCellClassV0::EligibleObject(2)],
        };
        let output = build_area_population_kernel_v0(
            &source_graph,
            &source,
            &target_graph,
            &target,
            1.0e-8,
            1.0e-10,
        )
        .unwrap();
        assert_eq!(output.exclusive_pairs[0].intersection_area_km2, 50.0);
        assert!(output
            .context_records
            .iter()
            .all(|record| record.outside_domain_area_km2 == 50.0));
    }

    #[test]
    fn interval_index_returns_zero_candidates_for_separated_hundred_cell_sets() {
        let squares = |offset: f64| {
            (0..10)
                .flat_map(|y| {
                    (0..10).map(move |x| {
                        let x0 = offset + f64::from(x);
                        let y0 = f64::from(y);
                        rectangle(x0, x0 + 1.0, y0, y0 + 1.0)
                    })
                })
                .collect::<Vec<_>>()
        };
        let left = graph(squares(0.0));
        let right = graph(squares(100.0));
        let background = |count| PacketAreaPopulationV0 {
            family: ObjectFamilyV0::Highland,
            objects: Vec::new(),
            cell_class: vec![PacketCellClassV0::HighlandBackground; count],
        };
        for (source, target) in [(&left, &right), (&right, &left)] {
            let output = build_area_population_kernel_v0(
                source,
                &background(source.cell_count()),
                target,
                &background(target.cell_count()),
                1.0e-8,
                1.0e-10,
            )
            .unwrap();
            assert_eq!(output.cell_box_candidates, 0);
            assert_eq!(output.polygon_clips, 0);
            assert_eq!(output.positive_cell_intersections, 0);
        }
    }

    fn rehash_without_validation(value: &mut ObjectCorrespondenceV0) -> Vec<u8> {
        value.derived_correspondence_hash = correspondence_preimage_hash(value).unwrap();
        fixed_bytes(value).unwrap()
    }

    fn assert_rehashed_semantic_rejection(mut value: ObjectCorrespondenceV0) {
        let bytes = rehash_without_validation(&mut value);
        assert!(matches!(
            object_correspondence_bytes_v0(&value),
            Err(PacketCorrespondenceErrorV0::Incompatible(_))
        ));
        assert!(matches!(
            decode_object_correspondence_v0(&bytes),
            Err(PacketCorrespondenceErrorV0::Incompatible(_))
        ));
    }

    #[test]
    fn rehashed_semantically_malformed_artifacts_are_rejected() {
        let packet = super::super::packet_tests::assembled_asymmetric_y_packet_at(4.0);
        let valid = build_object_correspondence_v0(&packet, &packet).unwrap();
        object_correspondence_bytes_v0(&valid).unwrap();

        let mut malformed = valid.clone();
        malformed.highland_exclusive_pairs[0].union_area_km2 += 1.0;
        assert_rehashed_semantic_rejection(malformed);

        let mut malformed = valid.clone();
        malformed.assignment_records.swap(0, 1);
        assert_rehashed_semantic_rejection(malformed);

        let mut malformed = valid.clone();
        malformed.best_components[0].members.reverse();
        assert_rehashed_semantic_rejection(malformed);

        let mut malformed = valid.clone();
        malformed.metric_conflicts.push(MetricConflictV0 {
            side: PacketSideV0::Source,
            drainage_node_id: 0,
            area_maximum_ids: vec![0],
            line_maximum_ids: vec![1],
        });
        assert_rehashed_semantic_rejection(malformed);

        let mut malformed = valid.clone();
        let available = malformed
            .topology_records
            .iter_mut()
            .find(|record| record.availability == TopologyAvailabilityV0::Available)
            .unwrap();
        available.mapped_adjacency = None;
        assert_rehashed_semantic_rejection(malformed);

        let mut malformed = valid.clone();
        malformed.topology_records.pop();
        assert_rehashed_semantic_rejection(malformed);

        let mut malformed = valid.clone();
        malformed.context_records[0].background_area_km2 = -0.0;
        assert_rehashed_semantic_rejection(malformed);

        let mut malformed = valid;
        malformed.work_counts.positive_line_rows += 1;
        assert_rehashed_semantic_rejection(malformed);
    }
}
