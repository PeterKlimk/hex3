//! Legacy geometric source ancestry for a frozen Structural Mountain domain.
//!
//! Legacy arc/collision height is nonlinear and plate-diffuse, so this module
//! does not manufacture additive per-edge height shares. It recovers the finite
//! boundary sources that own response geometry and reports the larger diffuse
//! dependency roster separately.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};

use glam::Vec3;

use super::{
    BoundaryEdgeId, BoundaryKind, ConvergentFrontSet, Crust, CrustType, FeatureFields,
    PlateBoundaryEdge, Plates, StructuralRegime, SubductionPolarity, Tessellation,
    TRANSFORM_NORMAL_THRESHOLD,
};

const TIE_EPS: f32 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LegacyFeatureRole {
    ContinentalArc,
    OceanicArc,
    Collision,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegacyAttributedFront {
    pub edge_id: BoundaryEdgeId,
    /// Number of coarse interpolation-support cells geometrically owned by the
    /// edge, split by response role.
    pub owned_response_cells: Vec<(LegacyFeatureRole, usize)>,
    pub geometric_owner: bool,
    pub co_seed_contributor: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegacySourceAttribution {
    pub fine_domain_cells: Vec<usize>,
    pub coarse_read_cells: Vec<usize>,
    pub fronts: Vec<LegacyAttributedFront>,
    pub geometric_source_edges: Vec<BoundaryEdgeId>,
    pub co_seed_source_edges: Vec<BoundaryEdgeId>,
    /// Eligible exact-front edges on deterministic shortest topology paths
    /// between direct geometric/co-seed sources.
    pub bridge_source_edges: Vec<BoundaryEdgeId>,
    pub selected_source_edges: Vec<BoundaryEdgeId>,
    /// Compatible same-plate sources that can affect normalized legacy
    /// amplitude through non-finite-support diffusion. These are not belt
    /// identity and carry no per-edge share.
    pub diffuse_dependency_edges: Vec<BoundaryEdgeId>,
}

/// Fine-domain binding of an already-selected structural source to the legacy
/// arc/collision response geometry. Response values retain legacy feature units
/// and are not additive physical shares because legacy forcing is plate-diffuse.
#[derive(Clone, Debug, PartialEq)]
pub struct LegacySourceObservationBinding {
    pub source_edges: Vec<BoundaryEdgeId>,
    pub legacy_eligible_source_edges: Vec<BoundaryEdgeId>,
    pub unrepresented_source_edges: Vec<BoundaryEdgeId>,
    /// Non-target edges co-aggregated at a seed used by the target. They are
    /// disclosed amplitude dependencies, not imported belt identity.
    pub mixed_seed_external_edges: Vec<BoundaryEdgeId>,
    pub coarse_strict_response_cells: Vec<usize>,
    pub coarse_mixed_response_cells: Vec<usize>,
    /// Pure selected-source response exceeds mixed + other response.
    pub fine_strict_owned_cells: Vec<usize>,
    /// Selected-associated response exceeds other response, but pure selected
    /// response does not own a strict majority.
    pub fine_ambiguous_association_cells: Vec<usize>,
    pub strict_response: Vec<f32>,
    pub mixed_response: Vec<f32>,
    pub other_response: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LegacyAttributionError {
    LengthMismatch,
    EmptyDomain,
    EmptySourceSet,
    FineCellOutOfRange(usize),
    CoarseCellOutOfRange(usize),
    DuplicateBoundary(BoundaryEdgeId),
    MissingFront(BoundaryEdgeId),
    InconsistentFront(BoundaryEdgeId),
    NonFiniteGeometry(BoundaryEdgeId),
    UnresolvedResponse {
        cell: usize,
        role: LegacyFeatureRole,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResponseClass {
    Strict,
    Mixed,
    Other,
}

impl std::fmt::Display for LegacyAttributionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for LegacyAttributionError {}

#[derive(Clone, Copy, Debug)]
struct Seed {
    edge_id: BoundaryEdgeId,
    cell: usize,
    plate: usize,
    role: LegacyFeatureRole,
    midpoint: Vec3,
}

#[derive(Clone, Copy, Debug)]
struct OwnerState {
    distance: f32,
    cell: usize,
    plate: usize,
    edge_id: BoundaryEdgeId,
    seed_cell: usize,
}

impl PartialEq for OwnerState {
    fn eq(&self, other: &Self) -> bool {
        self.distance.to_bits() == other.distance.to_bits()
            && self.cell == other.cell
            && self.edge_id == other.edge_id
            && self.seed_cell == other.seed_cell
    }
}

impl Eq for OwnerState {}

impl Ord for OwnerState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .total_cmp(&self.distance)
            .then_with(|| other.edge_id.cmp(&self.edge_id))
            .then_with(|| other.seed_cell.cmp(&self.seed_cell))
            .then_with(|| other.cell.cmp(&self.cell))
    }
}

impl PartialOrd for OwnerState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Recover finite geometric ancestry from the frozen legacy response domain.
/// The boundary slice must be the same preclassified snapshot used to build
/// `features` and `fronts`.
#[allow(clippy::too_many_arguments)]
pub fn attribute_legacy_convergent_sources(
    coarse: &Tessellation,
    fine: &Tessellation,
    fine_coarse_cell: &[usize],
    plates: &Plates,
    crust: &Crust,
    features: &FeatureFields,
    boundaries: &[PlateBoundaryEdge],
    fronts: &ConvergentFrontSet,
    fine_domain_cells: &[usize],
) -> Result<LegacySourceAttribution, LegacyAttributionError> {
    let n = coarse.num_cells();
    if fine_coarse_cell.len() != fine.num_cells()
        || plates.cell_plate.len() != n
        || crust.types.len() != n
        || features.arc.len() != n
        || features.collision.len() != n
    {
        return Err(LegacyAttributionError::LengthMismatch);
    }
    let fine_domain: BTreeSet<_> = fine_domain_cells.iter().copied().collect();
    if fine_domain.is_empty() {
        return Err(LegacyAttributionError::EmptyDomain);
    }
    let mut coarse_read = BTreeSet::new();
    for fine_cell in &fine_domain {
        if *fine_cell >= fine.num_cells() {
            return Err(LegacyAttributionError::FineCellOutOfRange(*fine_cell));
        }
        let nearest = fine_coarse_cell[*fine_cell];
        if nearest >= n {
            return Err(LegacyAttributionError::CoarseCellOutOfRange(nearest));
        }
        coarse_read.insert(nearest);
        coarse_read.extend(coarse.neighbors(nearest).iter().copied());
    }

    let seeds = collect_legacy_seeds(boundaries, fronts)?;

    let mut co_seed_edges = BTreeMap::<(LegacyFeatureRole, usize), BTreeSet<BoundaryEdgeId>>::new();
    for seed in &seeds {
        co_seed_edges
            .entry((seed.role, seed.cell))
            .or_default()
            .insert(seed.edge_id);
    }

    let mut owner_fields = BTreeMap::new();
    for role in [
        LegacyFeatureRole::ContinentalArc,
        LegacyFeatureRole::OceanicArc,
        LegacyFeatureRole::Collision,
    ] {
        let role_seeds: Vec<_> = seeds
            .iter()
            .copied()
            .filter(|seed| seed.role == role)
            .collect();
        owner_fields.insert(role, propagate_owners(coarse, plates, &role_seeds));
    }

    let mut owner_counts = BTreeMap::<BoundaryEdgeId, BTreeMap<LegacyFeatureRole, usize>>::new();
    let mut geometric = BTreeSet::new();
    let mut owning_seed_cells = BTreeSet::new();
    let mut response_requirements = BTreeSet::<(LegacyFeatureRole, usize)>::new();
    for &cell in &coarse_read {
        let plate = plates.cell_plate[cell] as usize;
        let crust_type = crust.crust_type(cell);
        let mut roles = Vec::with_capacity(2);
        if features.arc[cell] > 0.0 {
            roles.push(match crust_type {
                CrustType::Continental => LegacyFeatureRole::ContinentalArc,
                CrustType::Oceanic => LegacyFeatureRole::OceanicArc,
            });
        }
        if crust_type == CrustType::Continental && features.collision[cell] > 0.0 {
            roles.push(LegacyFeatureRole::Collision);
        }
        for role in roles {
            response_requirements.insert((role, plate));
            let owner = owner_fields[&role][cell]
                .ok_or(LegacyAttributionError::UnresolvedResponse { cell, role })?;
            geometric.insert(owner.0);
            owning_seed_cells.insert((role, owner.1));
            *owner_counts
                .entry(owner.0)
                .or_default()
                .entry(role)
                .or_default() += 1;
        }
    }

    let mut co_seed = BTreeSet::new();
    for key in owning_seed_cells {
        if let Some(edges) = co_seed_edges.get(&key) {
            co_seed.extend(edges.iter().copied());
        }
    }
    let direct: BTreeSet<_> = geometric.union(&co_seed).copied().collect();
    let selected = topological_source_closure(fronts, &direct);
    let bridges: BTreeSet<_> = selected.difference(&direct).copied().collect();
    let diffuse: BTreeSet<_> = seeds
        .iter()
        .filter(|seed| response_requirements.contains(&(seed.role, seed.plate)))
        .map(|seed| seed.edge_id)
        .collect();

    let fronts = selected
        .iter()
        .map(|&edge_id| LegacyAttributedFront {
            edge_id,
            owned_response_cells: owner_counts
                .remove(&edge_id)
                .unwrap_or_default()
                .into_iter()
                .collect(),
            geometric_owner: geometric.contains(&edge_id),
            co_seed_contributor: co_seed.contains(&edge_id),
        })
        .collect();
    Ok(LegacySourceAttribution {
        fine_domain_cells: fine_domain.into_iter().collect(),
        coarse_read_cells: coarse_read.into_iter().collect(),
        fronts,
        geometric_source_edges: geometric.into_iter().collect(),
        co_seed_source_edges: co_seed.into_iter().collect(),
        bridge_source_edges: bridges.into_iter().collect(),
        selected_source_edges: selected.into_iter().collect(),
        diffuse_dependency_edges: diffuse.into_iter().collect(),
    })
}

/// Bind an already-fixed structural source to the legacy response geometry.
///
/// Each coarse arc/collision sample is assigned through the exact legacy
/// plate-restricted distance owner and its complete co-seed roster. Fine values
/// are then interpolated with the same nearest-coarse-plus-neighbours weights as
/// the product transfer. Elevation and hydrology do not participate.
#[allow(clippy::too_many_arguments)]
pub fn bind_legacy_observations_to_source(
    coarse: &Tessellation,
    fine: &Tessellation,
    fine_coarse_cell: &[usize],
    plates: &Plates,
    crust: &Crust,
    features: &FeatureFields,
    boundaries: &[PlateBoundaryEdge],
    fronts: &ConvergentFrontSet,
    source_edges: &[BoundaryEdgeId],
) -> Result<LegacySourceObservationBinding, LegacyAttributionError> {
    let n = coarse.num_cells();
    if fine_coarse_cell.len() != fine.num_cells()
        || plates.cell_plate.len() != n
        || crust.types.len() != n
        || features.arc.len() != n
        || features.collision.len() != n
    {
        return Err(LegacyAttributionError::LengthMismatch);
    }
    let source: BTreeSet<_> = source_edges.iter().copied().collect();
    if source.is_empty() {
        return Err(LegacyAttributionError::EmptySourceSet);
    }
    let front_ids: BTreeSet<_> = fronts.edges.iter().map(|edge| edge.id).collect();
    if let Some(missing) = source.difference(&front_ids).next().copied() {
        return Err(LegacyAttributionError::MissingFront(missing));
    }

    let seeds = collect_legacy_seeds(boundaries, fronts)?;
    let eligible: BTreeSet<_> = seeds.iter().map(|seed| seed.edge_id).collect();
    let mut co_seed_edges = BTreeMap::<(LegacyFeatureRole, usize), BTreeSet<BoundaryEdgeId>>::new();
    for seed in &seeds {
        co_seed_edges
            .entry((seed.role, seed.cell))
            .or_default()
            .insert(seed.edge_id);
    }
    let mut owner_fields = BTreeMap::new();
    for role in [
        LegacyFeatureRole::ContinentalArc,
        LegacyFeatureRole::OceanicArc,
        LegacyFeatureRole::Collision,
    ] {
        let role_seeds: Vec<_> = seeds
            .iter()
            .copied()
            .filter(|seed| seed.role == role)
            .collect();
        owner_fields.insert(role, propagate_owners(coarse, plates, &role_seeds));
    }

    let mut strict_coarse = vec![0.0f32; n];
    let mut mixed_coarse = vec![0.0f32; n];
    let mut other_coarse = vec![0.0f32; n];
    let mut mixed_external = BTreeSet::new();
    for cell in 0..n {
        let mut responses = Vec::with_capacity(2);
        if features.arc[cell] > 0.0 {
            responses.push((
                match crust.crust_type(cell) {
                    CrustType::Continental => LegacyFeatureRole::ContinentalArc,
                    CrustType::Oceanic => LegacyFeatureRole::OceanicArc,
                },
                features.arc[cell],
            ));
        }
        if crust.crust_type(cell) == CrustType::Continental && features.collision[cell] > 0.0 {
            responses.push((LegacyFeatureRole::Collision, features.collision[cell]));
        }
        for (role, value) in responses {
            let owner = owner_fields[&role][cell]
                .ok_or(LegacyAttributionError::UnresolvedResponse { cell, role })?;
            let roster = co_seed_edges
                .get(&(role, owner.1))
                .ok_or(LegacyAttributionError::UnresolvedResponse { cell, role })?;
            match classify_response_roster(roster, &source) {
                ResponseClass::Strict => strict_coarse[cell] += value,
                ResponseClass::Mixed => {
                    mixed_coarse[cell] += value;
                    mixed_external.extend(roster.difference(&source).copied());
                }
                ResponseClass::Other => other_coarse[cell] += value,
            }
        }
    }

    let coarse_strict_response_cells = majority_cells(&strict_coarse, &mixed_coarse, &other_coarse);
    let coarse_mixed_response_cells = ambiguous_cells(&strict_coarse, &mixed_coarse, &other_coarse);
    let mut strict_response = Vec::with_capacity(fine.num_cells());
    let mut mixed_response = Vec::with_capacity(fine.num_cells());
    let mut other_response = Vec::with_capacity(fine.num_cells());
    for cell in 0..fine.num_cells() {
        let nearest = fine_coarse_cell[cell];
        if nearest >= n {
            return Err(LegacyAttributionError::CoarseCellOutOfRange(nearest));
        }
        let position = fine.cell_center(cell);
        let mut weighted = [0.0f32; 3];
        let mut total_weight = 0.0;
        for coarse_cell in std::iter::once(nearest).chain(coarse.neighbors(nearest).iter().copied())
        {
            let weight = interpolation_weight(coarse.cell_center(coarse_cell), position);
            weighted[0] += strict_coarse[coarse_cell] * weight;
            weighted[1] += mixed_coarse[coarse_cell] * weight;
            weighted[2] += other_coarse[coarse_cell] * weight;
            total_weight += weight;
        }
        strict_response.push(weighted[0] / total_weight);
        mixed_response.push(weighted[1] / total_weight);
        other_response.push(weighted[2] / total_weight);
    }
    let fine_strict_owned_cells =
        majority_cells(&strict_response, &mixed_response, &other_response);
    let fine_ambiguous_association_cells =
        ambiguous_cells(&strict_response, &mixed_response, &other_response);

    Ok(LegacySourceObservationBinding {
        source_edges: source.iter().copied().collect(),
        legacy_eligible_source_edges: source.intersection(&eligible).copied().collect(),
        unrepresented_source_edges: source.difference(&eligible).copied().collect(),
        mixed_seed_external_edges: mixed_external.into_iter().collect(),
        coarse_strict_response_cells,
        coarse_mixed_response_cells,
        fine_strict_owned_cells,
        fine_ambiguous_association_cells,
        strict_response,
        mixed_response,
        other_response,
    })
}

fn classify_response_roster(
    roster: &BTreeSet<BoundaryEdgeId>,
    source: &BTreeSet<BoundaryEdgeId>,
) -> ResponseClass {
    if roster.is_disjoint(source) {
        ResponseClass::Other
    } else if roster.is_subset(source) {
        ResponseClass::Strict
    } else {
        ResponseClass::Mixed
    }
}

fn majority_cells(strict: &[f32], mixed: &[f32], other: &[f32]) -> Vec<usize> {
    strict
        .iter()
        .zip(mixed)
        .zip(other)
        .enumerate()
        .filter_map(|(cell, ((&strict, &mixed), &other))| {
            (strict > 0.0 && strict > mixed + other).then_some(cell)
        })
        .collect()
}

fn ambiguous_cells(strict: &[f32], mixed: &[f32], other: &[f32]) -> Vec<usize> {
    strict
        .iter()
        .zip(mixed)
        .zip(other)
        .enumerate()
        .filter_map(|(cell, ((&strict, &mixed), &other))| {
            let associated = strict + mixed;
            (associated > other && strict <= mixed + other).then_some(cell)
        })
        .collect()
}

fn interpolation_weight(coarse_position: Vec3, fine_position: Vec3) -> f32 {
    let distance = angular_distance(coarse_position, fine_position);
    1.0 / (distance * distance + 1e-8)
}

pub fn filter_attributed_fronts(
    fronts: &ConvergentFrontSet,
    attribution: &LegacySourceAttribution,
) -> Result<ConvergentFrontSet, LegacyAttributionError> {
    let selected: BTreeSet<_> = attribution.selected_source_edges.iter().copied().collect();
    let filtered: Vec<_> = fronts
        .edges
        .iter()
        .filter(|edge| selected.contains(&edge.id))
        .cloned()
        .collect();
    if filtered.len() != selected.len() {
        let present: BTreeSet<_> = filtered.iter().map(|edge| edge.id).collect();
        let missing = selected.difference(&present).next().copied().unwrap();
        return Err(LegacyAttributionError::MissingFront(missing));
    }
    Ok(ConvergentFrontSet {
        edges: filtered,
        all_boundary_vertex_degree: fronts.all_boundary_vertex_degree.clone(),
    })
}

fn collect_legacy_seeds(
    boundaries: &[PlateBoundaryEdge],
    fronts: &ConvergentFrontSet,
) -> Result<Vec<Seed>, LegacyAttributionError> {
    let front_by_id: HashMap<_, _> = fronts.edges.iter().map(|edge| (edge.id, edge)).collect();
    let mut boundary_ids = HashSet::new();
    let mut seeds = Vec::new();
    for boundary in boundaries {
        let id = BoundaryEdgeId::new(boundary.cell_a, boundary.cell_b);
        if !boundary_ids.insert(id) {
            return Err(LegacyAttributionError::DuplicateBoundary(id));
        }
        if boundary.kind != BoundaryKind::Convergent
            || boundary.convergence.max(0.0) < TRANSFORM_NORMAL_THRESHOLD
        {
            continue;
        }
        let front = front_by_id
            .get(&id)
            .ok_or(LegacyAttributionError::MissingFront(id))?;
        if !front.midpoint.is_finite() {
            return Err(LegacyAttributionError::NonFiniteGeometry(id));
        }
        match boundary.subduction {
            Some(SubductionPolarity::ASubducts) => {
                if front.regime != StructuralRegime::Subduction
                    || front.receiving_plate != Some(boundary.plate_b)
                {
                    return Err(LegacyAttributionError::InconsistentFront(id));
                }
                push_arc_seed(
                    &mut seeds,
                    id,
                    boundary.cell_b,
                    boundary.plate_b,
                    boundary.type_b,
                    front.midpoint,
                );
            }
            Some(SubductionPolarity::BSubducts) => {
                if front.regime != StructuralRegime::Subduction
                    || front.receiving_plate != Some(boundary.plate_a)
                {
                    return Err(LegacyAttributionError::InconsistentFront(id));
                }
                push_arc_seed(
                    &mut seeds,
                    id,
                    boundary.cell_a,
                    boundary.plate_a,
                    boundary.type_a,
                    front.midpoint,
                );
            }
            None if boundary.type_a == CrustType::Continental
                && boundary.type_b == CrustType::Continental =>
            {
                if front.regime != StructuralRegime::Collision {
                    return Err(LegacyAttributionError::InconsistentFront(id));
                }
                seeds.push(Seed {
                    edge_id: id,
                    cell: boundary.cell_a,
                    plate: boundary.plate_a,
                    role: LegacyFeatureRole::Collision,
                    midpoint: front.midpoint,
                });
                seeds.push(Seed {
                    edge_id: id,
                    cell: boundary.cell_b,
                    plate: boundary.plate_b,
                    role: LegacyFeatureRole::Collision,
                    midpoint: front.midpoint,
                });
            }
            None => {}
        }
    }
    seeds.sort_by_key(|seed| (seed.role, seed.cell, seed.edge_id));
    Ok(seeds)
}

fn push_arc_seed(
    seeds: &mut Vec<Seed>,
    edge_id: BoundaryEdgeId,
    cell: usize,
    plate: usize,
    crust: CrustType,
    midpoint: Vec3,
) {
    seeds.push(Seed {
        edge_id,
        cell,
        plate,
        role: match crust {
            CrustType::Continental => LegacyFeatureRole::ContinentalArc,
            CrustType::Oceanic => LegacyFeatureRole::OceanicArc,
        },
        midpoint,
    });
}

fn topological_source_closure(
    fronts: &ConvergentFrontSet,
    direct: &BTreeSet<BoundaryEdgeId>,
) -> BTreeSet<BoundaryEdgeId> {
    let by_id: BTreeMap<_, _> = fronts.edges.iter().map(|edge| (edge.id, edge)).collect();
    let mut by_vertex = BTreeMap::<u32, Vec<BoundaryEdgeId>>::new();
    for edge in &fronts.edges {
        for vertex in edge.vertices {
            by_vertex.entry(vertex).or_default().push(edge.id);
        }
    }
    let mut adjacency = BTreeMap::<BoundaryEdgeId, BTreeSet<BoundaryEdgeId>>::new();
    for edges in by_vertex.values_mut() {
        edges.sort();
        for left in 0..edges.len() {
            for right in (left + 1)..edges.len() {
                let a = by_id[&edges[left]];
                let b = by_id[&edges[right]];
                let mut a_plates = a.plates;
                let mut b_plates = b.plates;
                a_plates.sort_unstable();
                b_plates.sort_unstable();
                if a.episode_id == b.episode_id && a_plates == b_plates {
                    adjacency.entry(a.id).or_default().insert(b.id);
                    adjacency.entry(b.id).or_default().insert(a.id);
                }
            }
        }
    }

    let mut closure = direct.clone();
    let mut unassigned = direct.clone();
    while let Some(&anchor) = unassigned.first() {
        let mut queue = std::collections::VecDeque::from([anchor]);
        let mut predecessor = BTreeMap::<BoundaryEdgeId, BoundaryEdgeId>::new();
        let mut reached = BTreeSet::from([anchor]);
        while let Some(current) = queue.pop_front() {
            for &next in adjacency.get(&current).into_iter().flatten() {
                if reached.insert(next) {
                    predecessor.insert(next, current);
                    queue.push_back(next);
                }
            }
        }
        let targets: Vec<_> = direct.intersection(&reached).copied().collect();
        for target in targets {
            unassigned.remove(&target);
            let mut current = target;
            closure.insert(current);
            while current != anchor {
                current = predecessor[&current];
                closure.insert(current);
            }
        }
    }
    closure
}

fn propagate_owners(
    tessellation: &Tessellation,
    plates: &Plates,
    seeds: &[Seed],
) -> Vec<Option<(BoundaryEdgeId, usize)>> {
    let n = tessellation.num_cells();
    let mut distances = vec![f32::INFINITY; n];
    let mut owners = vec![None; n];
    let mut heap = BinaryHeap::new();
    for seed in seeds {
        let distance = angular_distance(tessellation.cell_center(seed.cell), seed.midpoint);
        let candidate = (seed.edge_id, seed.cell);
        if better_owner(distance, candidate, distances[seed.cell], owners[seed.cell]) {
            distances[seed.cell] = distance;
            owners[seed.cell] = Some(candidate);
            heap.push(OwnerState {
                distance,
                cell: seed.cell,
                plate: seed.plate,
                edge_id: seed.edge_id,
                seed_cell: seed.cell,
            });
        }
    }
    while let Some(state) = heap.pop() {
        if state.distance > distances[state.cell] + TIE_EPS
            || owners[state.cell] != Some((state.edge_id, state.seed_cell))
        {
            continue;
        }
        let position = tessellation.cell_center(state.cell);
        for &neighbor in tessellation.neighbors(state.cell) {
            if plates.cell_plate[neighbor] as usize != state.plate {
                continue;
            }
            let distance =
                state.distance + angular_distance(position, tessellation.cell_center(neighbor));
            let candidate = (state.edge_id, state.seed_cell);
            if better_owner(distance, candidate, distances[neighbor], owners[neighbor]) {
                distances[neighbor] = distance;
                owners[neighbor] = Some(candidate);
                heap.push(OwnerState {
                    distance,
                    cell: neighbor,
                    ..state
                });
            }
        }
    }
    owners
}

fn better_owner(
    candidate_distance: f32,
    candidate: (BoundaryEdgeId, usize),
    current_distance: f32,
    current: Option<(BoundaryEdgeId, usize)>,
) -> bool {
    candidate_distance + TIE_EPS < current_distance
        || ((candidate_distance - current_distance).abs() <= TIE_EPS
            && current.is_none_or(|owner| candidate < owner))
}

fn angular_distance(a: Vec3, b: Vec3) -> f32 {
    a.dot(b).clamp(-1.0, 1.0).acos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{collect_convergent_fronts, collect_plate_boundaries, World};

    #[test]
    fn lower_edge_identity_wins_distance_tie() {
        let low = (BoundaryEdgeId::new(1, 4), 9);
        let high = (BoundaryEdgeId::new(2, 3), 7);
        assert!(better_owner(1.0, low, 1.0, Some(high)));
        assert!(!better_owner(1.0, high, 1.0, Some(low)));
    }

    #[test]
    fn response_rosters_keep_pure_mixed_and_other_identity_distinct() {
        let selected = BTreeSet::from([BoundaryEdgeId::new(1, 2)]);
        assert_eq!(
            classify_response_roster(&selected, &selected),
            ResponseClass::Strict
        );
        assert_eq!(
            classify_response_roster(
                &BTreeSet::from([BoundaryEdgeId::new(1, 2), BoundaryEdgeId::new(2, 3)]),
                &selected,
            ),
            ResponseClass::Mixed
        );
        assert_eq!(
            classify_response_roster(&BTreeSet::from([BoundaryEdgeId::new(2, 3)]), &selected,),
            ResponseClass::Other
        );
    }

    #[test]
    fn filtered_fronts_preserve_complete_boundary_degrees() {
        let attribution = LegacySourceAttribution {
            fine_domain_cells: vec![],
            coarse_read_cells: vec![],
            fronts: vec![],
            geometric_source_edges: vec![],
            co_seed_source_edges: vec![],
            bridge_source_edges: vec![],
            selected_source_edges: vec![BoundaryEdgeId::new(1, 4)],
            diffuse_dependency_edges: vec![],
        };
        let fronts = ConvergentFrontSet {
            edges: vec![],
            all_boundary_vertex_degree: BTreeMap::from([(3, 2)]),
        };
        assert_eq!(
            filter_attributed_fronts(&fronts, &attribution),
            Err(LegacyAttributionError::MissingFront(BoundaryEdgeId::new(
                1, 4
            )))
        );
    }

    #[test]
    fn generated_attribution_is_permutation_deterministic_and_finite() {
        let mut world = World::new(12_345, 256, 0);
        world.generate_plates(6);
        world.generate_crust();
        world.generate_dynamics();
        world.generate_features();
        let mut boundaries = collect_plate_boundaries(
            &world.tessellation,
            world.plates.as_ref().unwrap(),
            world.crust.as_ref().unwrap(),
            world.dynamics.as_ref().unwrap(),
        );
        let fronts = collect_convergent_fronts(
            &world.tessellation,
            &boundaries,
            world.tectonic_history.as_ref().unwrap(),
        )
        .unwrap();
        let features = world.features.as_ref().unwrap();
        let cell = (0..world.tessellation.num_cells())
            .find(|&cell| features.arc[cell] > 0.0 || features.collision[cell] > 0.0)
            .unwrap();
        let coarse_cell: Vec<_> = (0..world.tessellation.num_cells()).collect();
        let original = attribute_legacy_convergent_sources(
            &world.tessellation,
            &world.tessellation,
            &coarse_cell,
            world.plates.as_ref().unwrap(),
            world.crust.as_ref().unwrap(),
            features,
            &boundaries,
            &fronts,
            &[cell],
        )
        .unwrap();
        boundaries.reverse();
        let reversed = attribute_legacy_convergent_sources(
            &world.tessellation,
            &world.tessellation,
            &coarse_cell,
            world.plates.as_ref().unwrap(),
            world.crust.as_ref().unwrap(),
            features,
            &boundaries,
            &fronts,
            &[cell, cell],
        )
        .unwrap();
        assert_eq!(original, reversed);
        assert!(!original.selected_source_edges.is_empty());
        assert!(original
            .geometric_source_edges
            .iter()
            .all(|edge| original.diffuse_dependency_edges.contains(edge)));
        assert!(original
            .co_seed_source_edges
            .iter()
            .all(|edge| original.diffuse_dependency_edges.contains(edge)));
        let selected = filter_attributed_fronts(&fronts, &original).unwrap();
        assert_eq!(selected.edges.len(), original.selected_source_edges.len());

        let binding = bind_legacy_observations_to_source(
            &world.tessellation,
            &world.tessellation,
            &coarse_cell,
            world.plates.as_ref().unwrap(),
            world.crust.as_ref().unwrap(),
            features,
            &boundaries,
            &fronts,
            &original.selected_source_edges,
        )
        .unwrap();
        assert!(!binding.fine_strict_owned_cells.is_empty());
        boundaries.reverse();
        let reversed_binding = bind_legacy_observations_to_source(
            &world.tessellation,
            &world.tessellation,
            &coarse_cell,
            world.plates.as_ref().unwrap(),
            world.crust.as_ref().unwrap(),
            features,
            &boundaries,
            &fronts,
            &original.selected_source_edges,
        )
        .unwrap();
        assert_eq!(binding, reversed_binding);
    }
}
