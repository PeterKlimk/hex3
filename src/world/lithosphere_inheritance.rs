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

use super::{CellEdgeId, Crust, CrustType, Tessellation, PLANET_RADIUS_KM};

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
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InheritedStructureEdgeV0 {
    pub id: CellEdgeId,
    pub cells: [usize; 2],
    /// Ordered in `cells[0]`'s polygon orientation.
    pub vertices: [u32; 2],
    pub endpoints: [Vec3; 2],
    pub length_km: f32,
    /// Ordered by numeric province identity, because a suture has no polarity.
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
pub enum InheritedStructureNodeKindV0 {
    Tip,
    Junction,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InheritedStructureNodeV0 {
    pub vertex: u32,
    pub kind: InheritedStructureNodeKindV0,
    pub incident_segments: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InheritedStructureGraphV0 {
    pub edges: Vec<InheritedStructureEdgeV0>,
    pub segments: Vec<InheritedStructureSegmentV0>,
    pub nodes: Vec<InheritedStructureNodeV0>,
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
    pub at_structure_junction: bool,
    /// Smallest unoriented tangent angle at an exact shared vertex, in [0, 90].
    pub minimum_tangent_angle_deg: Option<f32>,
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
        return Ok(BoundaryInheritanceRelationshipV0 {
            boundary,
            kind: BoundaryInheritanceContactKindV0::Coincident,
            shared_vertices: boundary_vertices.to_vec(),
            structure_segment_ids: vec![segment_id_for_edge(&inheritance.graph, boundary)
                .ok_or(LithosphereInheritanceErrorV0::InvalidManufacturedGraph)?],
            at_structure_junction: boundary_vertices.iter().any(|vertex| {
                inheritance.graph.nodes.iter().any(|node| {
                    node.vertex == *vertex && node.kind == InheritedStructureNodeKindV0::Junction
                })
            }),
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
    let at_structure_junction = shared_vertices.iter().any(|vertex| {
        inheritance.graph.nodes.iter().any(|node| {
            node.vertex == *vertex && node.kind == InheritedStructureNodeKindV0::Junction
        })
    });
    Ok(BoundaryInheritanceRelationshipV0 {
        boundary,
        kind: if shared_vertices.is_empty() {
            BoundaryInheritanceContactKindV0::Unrelated
        } else {
            BoundaryInheritanceContactKindV0::VertexContact
        },
        shared_vertices: shared_vertices.into_iter().collect(),
        structure_segment_ids: structure_segment_ids.into_iter().collect(),
        at_structure_junction,
        minimum_tangent_angle_deg: minimum_angle.is_finite().then_some(minimum_angle),
    })
}

fn segment_id_for_edge(graph: &InheritedStructureGraphV0, edge: CellEdgeId) -> Option<u32> {
    graph
        .segments
        .iter()
        .find(|segment| segment.source_edges.contains(&edge))
        .map(|segment| segment.id)
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
    let nodes = endpoint_segments
        .into_iter()
        .map(|(vertex, mut incident_segments)| {
            incident_segments.sort_unstable();
            incident_segments.dedup();
            InheritedStructureNodeV0 {
                vertex,
                kind: if incident_segments.len() == 1 {
                    InheritedStructureNodeKindV0::Tip
                } else {
                    InheritedStructureNodeKindV0::Junction
                },
                incident_segments,
            }
        })
        .collect();
    let total_length_km = edges.iter().map(|edge| f64::from(edge.length_km)).sum();
    Ok(InheritedStructureGraphV0 {
        edges,
        segments,
        nodes,
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
    fn chain_compiler_preserves_continuations_and_junctions() {
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
        assert_eq!(continuation.nodes.len(), 2);
        assert!(continuation
            .nodes
            .iter()
            .all(|node| node.kind == InheritedStructureNodeKindV0::Tip));

        let junction = compile_graph_from_edges(vec![
            edge(0, 1, [10, 11], [0, 1]),
            edge(2, 3, [11, 12], [0, 2]),
            edge(4, 5, [11, 13], [1, 2]),
        ])
        .unwrap();
        assert_eq!(junction.segments.len(), 3);
        let node = junction
            .nodes
            .iter()
            .find(|node| node.vertex == 11)
            .unwrap();
        assert_eq!(node.kind, InheritedStructureNodeKindV0::Junction);
        assert_eq!(node.incident_segments.len(), 3);
    }
}
