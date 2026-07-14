//! Arm-neutral D0 evaluation drainage over a validated physical surface graph.
//!
//! This module implements the bounded planar/testbed contract in
//! `docs/research/landform-object-packet-d0-2026-07-15.md`. It deliberately
//! does not adapt product hydrology or claim subcell channel geometry.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque};
use std::fmt;

use bincode::Options;
use glam::DVec3;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use super::{
    EvaluationBoundaryConditionV0, EvaluationDomainV0, EvaluationSurfaceGraphV0, LandformError,
    SurfaceHierarchyConfigV0,
};

pub const D0_SCHEMA_VERSION: &str = "landform-d0-v0";
pub const D0_HASH_VERSION: &str = "fnv1a64-bincode-fixint-le-v0";

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DrainageConfigV0 {
    pub support_thresholds_km2: [f64; 3],
    pub balance_absolute_tolerance: f64,
    pub balance_relative_tolerance: f64,
    pub schema_version: &'static str,
    pub hash_version: &'static str,
}

impl Default for DrainageConfigV0 {
    fn default() -> Self {
        Self {
            support_thresholds_km2: [1_000.0, 2_000.0, 4_000.0],
            balance_absolute_tolerance: 1.0e-9,
            balance_relative_tolerance: 1.0e-12,
            schema_version: D0_SCHEMA_VERSION,
            hash_version: D0_HASH_VERSION,
        }
    }
}

impl DrainageConfigV0 {
    fn validate(&self) -> Result<(), DrainageErrorV0> {
        if *self != Self::default() {
            return Err(DrainageErrorV0::UnregisteredConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DrainageReceiverV0 {
    Cell {
        cell: u32,
        directed_edge: u32,
    },
    Portal {
        boundary_segment: u32,
        portal_id: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortalDrainageLedgerV0 {
    pub portal_id: u32,
    pub structural_area_km2: f64,
    pub supplied_runoff: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DrainageRoutingV0 {
    pub receiver: Vec<DrainageReceiverV0>,
    pub filled_elevation_km: Vec<f64>,
    pub flat_potential: Vec<u32>,
    pub structural_area_km2: Vec<f64>,
    pub supplied_runoff: Vec<f64>,
    pub outlet_portal_id: Vec<u32>,
    pub segment_length_km: Vec<f64>,
    pub fill_supported: Vec<bool>,
    pub flat_supported: Vec<bool>,
    pub physically_non_descending: Vec<bool>,
    pub portal_ledgers: Vec<PortalDrainageLedgerV0>,
    pub structural_area_residual_km2: f64,
    pub supplied_runoff_residual: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DrainageDepressionV0 {
    pub id: u32,
    pub parent: Option<u32>,
    pub anchor_cell: u32,
    pub affected_cells: Vec<u32>,
    pub spill_elevation_km: f64,
    pub affected_area_km2: f64,
    pub maximum_fill_depth_km: f64,
    pub virtual_fill_volume_km3: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IncrementalCatchmentOwnerV0 {
    Reach(u32),
    Portal(u32),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiverReachV0 {
    pub id: u32,
    pub cells: Vec<u32>,
    pub upstream_reaches: Vec<u32>,
    pub downstream_reach: Option<u32>,
    /// Set only when this reach directly terminates at the portal.
    pub terminal_portal_id: Option<u32>,
    /// Ultimate semantic outlet for this reach and all of its nested catchment.
    pub outlet_portal_id: u32,
    pub physical_length_km: f64,
    pub head_structural_area_km2: f64,
    pub tail_structural_area_km2: f64,
    pub head_supplied_runoff: f64,
    pub tail_supplied_runoff: f64,
    pub strahler_order: u32,
    pub fill_supported_segment_count: u32,
    pub fill_supported_length_km: f64,
    pub flat_supported_segment_count: u32,
    pub flat_supported_length_km: f64,
    pub physically_non_descending_segment_count: u32,
    pub physically_non_descending_length_km: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetainedCatchmentV0 {
    pub reach_id: u32,
    pub parent_reach: Option<u32>,
    pub child_reaches: Vec<u32>,
    pub outlet_portal_id: u32,
    pub nested_structural_area_km2: f64,
    pub nested_supplied_runoff: f64,
    pub exclusive_physical_area_km2: f64,
    pub exclusive_local_runoff: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawCatchmentBoundaryFaceV0 {
    pub owners: [IncrementalCatchmentOwnerV0; 2],
    pub endpoints_km: [DVec3; 2],
    pub physical_length_km: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortalTrunkRolesV0 {
    pub portal_id: u32,
    pub terminal_reaches: Vec<u32>,
    /// Ordered source to outlet.
    pub greatest_supply: Vec<u32>,
    /// Ordered source to outlet.
    pub longest_trunk: Vec<u32>,
    /// Ordered source to outlet.
    pub highest_order: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiverReachGraphV0 {
    pub cell_reach: Vec<Option<u32>>,
    pub reaches: Vec<RiverReachV0>,
    pub portal_roles: Vec<PortalTrunkRolesV0>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DrainageBasinGraphV0 {
    pub catchments: Vec<RetainedCatchmentV0>,
    pub exclusive_owner: Vec<IncrementalCatchmentOwnerV0>,
    pub raw_catchment_boundaries: Vec<RawCatchmentBoundaryFaceV0>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DrainageScaleV0 {
    pub support_threshold_km2: f64,
    pub basin_graph: DrainageBasinGraphV0,
    pub reach_graph: RiverReachGraphV0,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationDrainageV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub routing: DrainageRoutingV0,
    pub depressions: Vec<DrainageDepressionV0>,
    pub scales: Vec<DrainageScaleV0>,
    pub derived_evidence_hash: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DrainageErrorV0 {
    InvalidGraph(String),
    UnsupportedDomain,
    LengthMismatch(&'static str),
    NonFiniteElevation,
    InvalidRunoff,
    UnregisteredConfiguration,
    MissingPortal,
    InvalidPortal { segment: usize },
    InvalidRoutingGeometry { cell: usize },
    DisconnectedFromPortal { cell: usize },
    MissingReceiver { cell: usize },
    ReceiverCycle,
    UnknownPortal(u32),
    DepressionHierarchyAmbiguity { depression: u32 },
    AreaBalanceFailure { residual: f64 },
    RunoffBalanceFailure { residual: f64 },
    NonFiniteAccumulation { cell: Option<usize> },
    NonFiniteConditioning { cell: usize },
    Overflow,
    Serialization(String),
}

impl fmt::Display for DrainageErrorV0 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for DrainageErrorV0 {}

#[derive(Debug, Clone, Copy)]
struct FillEntry {
    level: f64,
    cell: usize,
}

impl PartialEq for FillEntry {
    fn eq(&self, other: &Self) -> bool {
        self.level == other.level && self.cell == other.cell
    }
}

impl Eq for FillEntry {}

impl PartialOrd for FillEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FillEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .level
            .total_cmp(&self.level)
            .then_with(|| other.cell.cmp(&self.cell))
    }
}

#[derive(Clone)]
struct Candidate {
    receiver: DrainageReceiverV0,
    weight: f64,
    downstream_level: f64,
    downstream_potential: u32,
    midpoint: DVec3,
    destination_center: DVec3,
    internal: bool,
    portal_id: u32,
    length_km: f64,
    receiver_physical_elevation_km: f64,
    flat_supported: bool,
}

type SelectedReceivers = (
    Vec<DrainageReceiverV0>,
    Vec<f64>,
    Vec<bool>,
    Vec<bool>,
    Vec<bool>,
);

pub fn build_evaluation_drainage_v0(
    graph: &EvaluationSurfaceGraphV0,
    physical_elevation_km: &[f64],
    local_runoff_supply: &[f64],
    config: DrainageConfigV0,
) -> Result<EvaluationDrainageV0, DrainageErrorV0> {
    config.validate()?;
    if graph.domain != EvaluationDomainV0::Planar {
        return Err(DrainageErrorV0::UnsupportedDomain);
    }
    graph
        .validate(&SurfaceHierarchyConfigV0::default())
        .map_err(|error: LandformError| DrainageErrorV0::InvalidGraph(error.to_string()))?;
    let n = graph.cell_count();
    if physical_elevation_km.len() != n {
        return Err(DrainageErrorV0::LengthMismatch("physical_elevation_km"));
    }
    if local_runoff_supply.len() != n {
        return Err(DrainageErrorV0::LengthMismatch("local_runoff_supply"));
    }
    if physical_elevation_km.iter().any(|value| !value.is_finite()) {
        return Err(DrainageErrorV0::NonFiniteElevation);
    }
    if local_runoff_supply
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(DrainageErrorV0::InvalidRunoff);
    }

    let portals = collect_portals(graph)?;
    let (filled_elevation_km, flat_potential) =
        derive_conditioning(graph, physical_elevation_km, &portals)?;
    let order = routing_order(graph, &filled_elevation_km, &flat_potential);
    let (receiver, segment_length_km, fill_supported, flat_supported, non_descending) =
        select_receivers(
            graph,
            physical_elevation_km,
            &filled_elevation_km,
            &flat_potential,
            &portals,
        )?;
    let (
        structural_area_km2,
        supplied_runoff,
        outlet_portal_id,
        portal_ledgers,
        area_residual,
        runoff_residual,
    ) = accumulate(
        graph,
        local_runoff_supply,
        &receiver,
        &order,
        &portals,
        config,
    )?;
    let mut depressions = build_depressions(
        graph,
        physical_elevation_km,
        &filled_elevation_km,
        &receiver,
    )?;
    depressions.sort_by(|a, b| {
        a.spill_elevation_km
            .total_cmp(&b.spill_elevation_km)
            .then_with(|| {
                point_cmp(
                    graph.cell_center_km[a.anchor_cell as usize],
                    graph.cell_center_km[b.anchor_cell as usize],
                )
            })
    });
    let old_ids: BTreeMap<u32, u32> = depressions
        .iter()
        .enumerate()
        .map(|(new, depression)| (depression.id, new as u32))
        .collect();
    for (new, depression) in depressions.iter_mut().enumerate() {
        depression.id = new as u32;
        depression.parent = depression.parent.map(|old| old_ids[&old]);
    }

    let routing = DrainageRoutingV0 {
        receiver,
        filled_elevation_km,
        flat_potential,
        structural_area_km2,
        supplied_runoff,
        outlet_portal_id,
        segment_length_km,
        fill_supported,
        flat_supported,
        physically_non_descending: non_descending,
        portal_ledgers,
        structural_area_residual_km2: area_residual,
        supplied_runoff_residual: runoff_residual,
    };
    let mut scales = Vec::with_capacity(config.support_thresholds_km2.len());
    for threshold in config.support_thresholds_km2 {
        scales.push(build_scale(
            graph,
            physical_elevation_km,
            local_runoff_supply,
            &routing,
            threshold,
        )?);
    }
    let mut result = EvaluationDrainageV0 {
        schema_version: config.schema_version.to_owned(),
        hash_version: config.hash_version.to_owned(),
        routing,
        depressions,
        scales,
        derived_evidence_hash: 0,
    };
    let bytes = evidence_bytes(
        graph,
        physical_elevation_km,
        local_runoff_supply,
        &config,
        &result,
    )?;
    result.derived_evidence_hash = fnv1a64(&bytes);
    let repeated = evidence_bytes(
        graph,
        physical_elevation_km,
        local_runoff_supply,
        &config,
        &result,
    )?;
    if fnv1a64(&repeated) != result.derived_evidence_hash {
        return Err(DrainageErrorV0::Serialization(
            "D0 evidence hash is not stable".into(),
        ));
    }
    Ok(result)
}

#[derive(Clone, Copy)]
struct PortalSegment {
    segment: usize,
    owner: usize,
    portal_id: u32,
    base_elevation_km: f64,
    midpoint: DVec3,
    width_km: f64,
    center_distance_km: f64,
}

fn collect_portals(
    graph: &EvaluationSurfaceGraphV0,
) -> Result<Vec<PortalSegment>, DrainageErrorV0> {
    let mut portals = Vec::new();
    for (segment_index, segment) in graph.boundary_segments.iter().enumerate() {
        let EvaluationBoundaryConditionV0::OpenBaseLevel {
            portal_id,
            elevation_km,
        } = segment.condition
        else {
            continue;
        };
        let owner = segment.owner_cell as usize;
        let midpoint = 0.5 * (segment.endpoints_km[0] + segment.endpoints_km[1]);
        let center_distance_km = graph.cell_center_km[owner].distance(midpoint);
        if !elevation_km.is_finite()
            || !midpoint.is_finite()
            || !segment.physical_length_km.is_finite()
            || segment.physical_length_km <= 0.0
            || !center_distance_km.is_finite()
            || center_distance_km <= 0.0
        {
            return Err(DrainageErrorV0::InvalidPortal {
                segment: segment_index,
            });
        }
        portals.push(PortalSegment {
            segment: segment_index,
            owner,
            portal_id,
            base_elevation_km: elevation_km,
            midpoint,
            width_km: segment.physical_length_km,
            center_distance_km,
        });
    }
    if portals.is_empty() {
        return Err(DrainageErrorV0::MissingPortal);
    }
    portals.sort_by(|a, b| {
        a.portal_id
            .cmp(&b.portal_id)
            .then_with(|| point_cmp(a.midpoint, b.midpoint))
    });
    Ok(portals)
}

fn derive_conditioning(
    graph: &EvaluationSurfaceGraphV0,
    physical: &[f64],
    portals: &[PortalSegment],
) -> Result<(Vec<f64>, Vec<u32>), DrainageErrorV0> {
    let n = graph.cell_count();
    let mut filled = vec![f64::INFINITY; n];
    let mut heap = BinaryHeap::new();
    for portal in portals {
        let level = physical[portal.owner].max(portal.base_elevation_km);
        if level < filled[portal.owner] {
            filled[portal.owner] = level;
            heap.push(FillEntry {
                level,
                cell: portal.owner,
            });
        }
    }
    while let Some(FillEntry { level, cell }) = heap.pop() {
        if level != filled[cell] {
            continue;
        }
        for edge in edge_range(&graph.edge_offsets, cell) {
            let neighbor = graph.edge_neighbor[edge] as usize;
            let candidate = level.max(physical[neighbor]);
            if candidate < filled[neighbor] {
                filled[neighbor] = candidate;
                heap.push(FillEntry {
                    level: candidate,
                    cell: neighbor,
                });
            }
        }
    }
    if let Some(cell) = filled.iter().position(|level| !level.is_finite()) {
        return Err(DrainageErrorV0::DisconnectedFromPortal { cell });
    }

    let mut potential = vec![u32::MAX; n];
    let mut queue = VecDeque::new();
    let mut open_by_cell = vec![false; n];
    for portal in portals {
        if portal.base_elevation_km <= filled[portal.owner] {
            open_by_cell[portal.owner] = true;
        }
    }
    let mut seed_cells = Vec::new();
    for cell in 0..n {
        let has_lower = edge_range(&graph.edge_offsets, cell)
            .any(|edge| filled[graph.edge_neighbor[edge] as usize] < filled[cell]);
        if has_lower || open_by_cell[cell] {
            seed_cells.push(cell);
        }
    }
    seed_cells.sort_by(|&a, &b| point_cmp(graph.cell_center_km[a], graph.cell_center_km[b]));
    for cell in seed_cells {
        potential[cell] = 0;
        queue.push_back(cell);
    }
    while let Some(cell) = queue.pop_front() {
        let next = potential[cell]
            .checked_add(1)
            .ok_or(DrainageErrorV0::Overflow)?;
        let mut neighbors = edge_range(&graph.edge_offsets, cell)
            .map(|edge| graph.edge_neighbor[edge] as usize)
            .filter(|&neighbor| filled[neighbor] == filled[cell])
            .collect::<Vec<_>>();
        neighbors.sort_by(|&a, &b| point_cmp(graph.cell_center_km[a], graph.cell_center_km[b]));
        for neighbor in neighbors {
            if potential[neighbor] == u32::MAX {
                potential[neighbor] = next;
                queue.push_back(neighbor);
            }
        }
    }
    if let Some(cell) = potential.iter().position(|value| *value == u32::MAX) {
        return Err(DrainageErrorV0::DisconnectedFromPortal { cell });
    }
    Ok((filled, potential))
}

fn routing_order(
    graph: &EvaluationSurfaceGraphV0,
    filled: &[f64],
    potential: &[u32],
) -> Vec<usize> {
    let mut order: Vec<_> = (0..graph.cell_count()).collect();
    order.sort_by(|&a, &b| {
        filled[b]
            .total_cmp(&filled[a])
            .then_with(|| potential[b].cmp(&potential[a]))
            .then_with(|| point_cmp(graph.cell_center_km[a], graph.cell_center_km[b]))
    });
    order
}

fn select_receivers(
    graph: &EvaluationSurfaceGraphV0,
    physical: &[f64],
    filled: &[f64],
    potential: &[u32],
    portals: &[PortalSegment],
) -> Result<SelectedReceivers, DrainageErrorV0> {
    let n = graph.cell_count();
    let mut portals_by_cell = vec![Vec::new(); n];
    for portal in portals {
        portals_by_cell[portal.owner].push(*portal);
    }
    let mut receiver = Vec::with_capacity(n);
    let mut segment_length = Vec::with_capacity(n);
    let mut fill_supported = Vec::with_capacity(n);
    let mut flat_supported = Vec::with_capacity(n);
    let mut non_descending = Vec::with_capacity(n);
    for cell in 0..n {
        let mut best: Option<Candidate> = None;
        for edge in edge_range(&graph.edge_offsets, cell) {
            let neighbor = graph.edge_neighbor[edge] as usize;
            let drop = filled[cell] - filled[neighbor];
            let is_flat = drop == 0.0 && potential[neighbor] < potential[cell];
            if drop < 0.0 || (drop == 0.0 && !is_flat) {
                continue;
            }
            let distance = graph.edge_distance_km[edge];
            let width = graph.edge_shared_width_km[edge];
            let weight = if drop > 0.0 {
                width * drop / distance
            } else {
                width / distance
            };
            if !distance.is_finite()
                || distance <= 0.0
                || !width.is_finite()
                || width <= 0.0
                || !weight.is_finite()
                || weight <= 0.0
            {
                return Err(DrainageErrorV0::InvalidRoutingGeometry { cell });
            }
            let endpoints = graph.edge_face_endpoints_km[edge];
            let candidate = Candidate {
                receiver: DrainageReceiverV0::Cell {
                    cell: neighbor as u32,
                    directed_edge: edge as u32,
                },
                weight,
                downstream_level: filled[neighbor],
                downstream_potential: potential[neighbor],
                midpoint: 0.5 * (endpoints[0] + endpoints[1]),
                destination_center: graph.cell_center_km[neighbor],
                internal: true,
                portal_id: u32::MAX,
                length_km: distance,
                receiver_physical_elevation_km: physical[neighbor],
                flat_supported: is_flat,
            };
            if best
                .as_ref()
                .is_none_or(|current| candidate_better(&candidate, current))
            {
                best = Some(candidate);
            }
        }
        for portal in &portals_by_cell[cell] {
            let drop = filled[cell] - portal.base_elevation_km;
            let is_flat = drop == 0.0 && potential[cell] == 0;
            if drop < 0.0 || (drop == 0.0 && !is_flat) {
                continue;
            }
            let weight = if drop > 0.0 {
                portal.width_km * drop / portal.center_distance_km
            } else {
                portal.width_km / portal.center_distance_km
            };
            if !weight.is_finite() || weight <= 0.0 {
                return Err(DrainageErrorV0::InvalidRoutingGeometry { cell });
            }
            let candidate = Candidate {
                receiver: DrainageReceiverV0::Portal {
                    boundary_segment: portal.segment as u32,
                    portal_id: portal.portal_id,
                },
                weight,
                downstream_level: portal.base_elevation_km,
                downstream_potential: 0,
                midpoint: portal.midpoint,
                destination_center: DVec3::ZERO,
                internal: false,
                portal_id: portal.portal_id,
                length_km: portal.center_distance_km,
                receiver_physical_elevation_km: portal.base_elevation_km,
                flat_supported: is_flat,
            };
            if best
                .as_ref()
                .is_none_or(|current| candidate_better(&candidate, current))
            {
                best = Some(candidate);
            }
        }
        let best = best.ok_or(DrainageErrorV0::MissingReceiver { cell })?;
        receiver.push(best.receiver);
        segment_length.push(best.length_km);
        fill_supported.push(
            filled[cell] > physical[cell]
                || matches!(best.receiver, DrainageReceiverV0::Cell { cell: next, .. }
                    if filled[next as usize] > physical[next as usize]),
        );
        flat_supported.push(best.flat_supported);
        non_descending.push(physical[cell] <= best.receiver_physical_elevation_km);
    }
    Ok((
        receiver,
        segment_length,
        fill_supported,
        flat_supported,
        non_descending,
    ))
}

fn candidate_better(candidate: &Candidate, current: &Candidate) -> bool {
    candidate
        .weight
        .total_cmp(&current.weight)
        .then_with(|| {
            current
                .downstream_level
                .total_cmp(&candidate.downstream_level)
        })
        .then_with(|| {
            current
                .downstream_potential
                .cmp(&candidate.downstream_potential)
        })
        .then_with(|| point_cmp(current.midpoint, candidate.midpoint))
        .then_with(|| candidate.internal.cmp(&current.internal))
        .then_with(|| point_cmp(current.destination_center, candidate.destination_center))
        .then_with(|| current.portal_id.cmp(&candidate.portal_id))
        == Ordering::Greater
}

#[allow(clippy::type_complexity)]
fn accumulate(
    graph: &EvaluationSurfaceGraphV0,
    local_runoff: &[f64],
    receiver: &[DrainageReceiverV0],
    order: &[usize],
    portals: &[PortalSegment],
    config: DrainageConfigV0,
) -> Result<
    (
        Vec<f64>,
        Vec<f64>,
        Vec<u32>,
        Vec<PortalDrainageLedgerV0>,
        f64,
        f64,
    ),
    DrainageErrorV0,
> {
    let n = graph.cell_count();
    let mut structural_area = graph.cell_area_km2.clone();
    let mut runoff = local_runoff.to_vec();
    let mut area_correction = vec![0.0; n];
    let mut runoff_correction = vec![0.0; n];
    let portal_ids = portals
        .iter()
        .map(|portal| portal.portal_id)
        .collect::<BTreeSet<_>>();
    let mut portal_area = portal_ids
        .iter()
        .map(|&id| (id, 0.0))
        .collect::<BTreeMap<_, _>>();
    let mut portal_runoff = portal_area.clone();
    let mut portal_area_correction = portal_area.clone();
    let mut portal_runoff_correction = portal_area.clone();
    for &cell in order {
        match receiver[cell] {
            DrainageReceiverV0::Cell { cell: next, .. } => {
                let next = next as usize;
                let donor_area = structural_area[cell];
                let donor_runoff = runoff[cell];
                kahan_add(
                    &mut structural_area[next],
                    &mut area_correction[next],
                    donor_area,
                )
                .map_err(|_| DrainageErrorV0::NonFiniteAccumulation { cell: Some(next) })?;
                kahan_add(
                    &mut runoff[next],
                    &mut runoff_correction[next],
                    donor_runoff,
                )
                .map_err(|_| DrainageErrorV0::NonFiniteAccumulation { cell: Some(next) })?;
            }
            DrainageReceiverV0::Portal { portal_id, .. } => {
                let area = portal_area
                    .get_mut(&portal_id)
                    .ok_or(DrainageErrorV0::UnknownPortal(portal_id))?;
                let area_compensation = portal_area_correction
                    .get_mut(&portal_id)
                    .ok_or(DrainageErrorV0::UnknownPortal(portal_id))?;
                kahan_add(area, area_compensation, structural_area[cell])
                    .map_err(|_| DrainageErrorV0::NonFiniteAccumulation { cell: None })?;
                let supplied = portal_runoff
                    .get_mut(&portal_id)
                    .ok_or(DrainageErrorV0::UnknownPortal(portal_id))?;
                let supplied_compensation = portal_runoff_correction
                    .get_mut(&portal_id)
                    .ok_or(DrainageErrorV0::UnknownPortal(portal_id))?;
                kahan_add(supplied, supplied_compensation, runoff[cell])
                    .map_err(|_| DrainageErrorV0::NonFiniteAccumulation { cell: None })?;
            }
        }
    }

    let mut outlet = vec![u32::MAX; n];
    for &cell in order.iter().rev() {
        outlet[cell] = match receiver[cell] {
            DrainageReceiverV0::Cell { cell: next, .. } => {
                let value = outlet[next as usize];
                if value == u32::MAX {
                    return Err(DrainageErrorV0::ReceiverCycle);
                }
                value
            }
            DrainageReceiverV0::Portal { portal_id, .. } => portal_id,
        };
    }

    let total_area = kahan_sum(graph.cell_area_km2.iter().copied());
    let total_runoff = kahan_sum(local_runoff.iter().copied());
    let exported_area = kahan_sum(portal_area.values().copied());
    let exported_runoff = kahan_sum(portal_runoff.values().copied());
    let area_residual = total_area - exported_area;
    let runoff_residual = total_runoff - exported_runoff;
    if !total_area.is_finite()
        || !total_runoff.is_finite()
        || !exported_area.is_finite()
        || !exported_runoff.is_finite()
        || !area_residual.is_finite()
        || !runoff_residual.is_finite()
    {
        return Err(DrainageErrorV0::NonFiniteAccumulation { cell: None });
    }
    let area_tolerance = config
        .balance_absolute_tolerance
        .max(config.balance_relative_tolerance * total_area.abs());
    let runoff_tolerance = config
        .balance_absolute_tolerance
        .max(config.balance_relative_tolerance * total_runoff.abs());
    if area_residual.abs() > area_tolerance {
        return Err(DrainageErrorV0::AreaBalanceFailure {
            residual: area_residual,
        });
    }
    if runoff_residual.abs() > runoff_tolerance {
        return Err(DrainageErrorV0::RunoffBalanceFailure {
            residual: runoff_residual,
        });
    }
    let portal_ledgers = portal_ids
        .into_iter()
        .map(|portal_id| PortalDrainageLedgerV0 {
            portal_id,
            structural_area_km2: portal_area[&portal_id],
            supplied_runoff: portal_runoff[&portal_id],
        })
        .collect();
    Ok((
        structural_area,
        runoff,
        outlet,
        portal_ledgers,
        area_residual,
        runoff_residual,
    ))
}

#[derive(Clone)]
struct TempDepression {
    old_id: u32,
    cells: Vec<u32>,
    spill: f64,
    anchor: u32,
    area: f64,
    max_depth: f64,
    volume: f64,
}

fn build_depressions(
    graph: &EvaluationSurfaceGraphV0,
    physical: &[f64],
    filled: &[f64],
    receiver: &[DrainageReceiverV0],
) -> Result<Vec<DrainageDepressionV0>, DrainageErrorV0> {
    let n = graph.cell_count();
    let mut component = vec![None::<u32>; n];
    let mut temporary = Vec::<TempDepression>::new();
    for seed in 0..n {
        if filled[seed] <= physical[seed] || component[seed].is_some() {
            continue;
        }
        let old_id = temporary.len() as u32;
        let spill = filled[seed];
        let mut queue = VecDeque::from([seed]);
        component[seed] = Some(old_id);
        let mut cells = Vec::new();
        while let Some(cell) = queue.pop_front() {
            cells.push(cell as u32);
            let mut neighbors = edge_range(&graph.edge_offsets, cell)
                .map(|edge| graph.edge_neighbor[edge] as usize)
                .filter(|&neighbor| {
                    component[neighbor].is_none()
                        && filled[neighbor] > physical[neighbor]
                        && filled[neighbor] == spill
                })
                .collect::<Vec<_>>();
            neighbors.sort_by(|&a, &b| point_cmp(graph.cell_center_km[a], graph.cell_center_km[b]));
            for neighbor in neighbors {
                component[neighbor] = Some(old_id);
                queue.push_back(neighbor);
            }
        }
        cells.sort_unstable();
        let anchor = *cells
            .iter()
            .min_by(|&&a, &&b| {
                point_cmp(
                    graph.cell_center_km[a as usize],
                    graph.cell_center_km[b as usize],
                )
            })
            .expect("depression component is non-empty");
        let mut area = 0.0;
        let mut area_correction = 0.0;
        let mut max_depth: f64 = 0.0;
        let mut volume = 0.0;
        let mut volume_correction = 0.0;
        for &cell in &cells {
            let cell = cell as usize;
            let depth = filled[cell] - physical[cell];
            let contribution = graph.cell_area_km2[cell] * depth;
            if !depth.is_finite() || depth <= 0.0 || !contribution.is_finite() {
                return Err(DrainageErrorV0::NonFiniteConditioning { cell });
            }
            max_depth = max_depth.max(depth);
            kahan_add(&mut area, &mut area_correction, graph.cell_area_km2[cell])
                .map_err(|_| DrainageErrorV0::NonFiniteConditioning { cell })?;
            kahan_add(&mut volume, &mut volume_correction, contribution)
                .map_err(|_| DrainageErrorV0::NonFiniteConditioning { cell })?;
        }
        temporary.push(TempDepression {
            old_id,
            cells,
            spill,
            anchor,
            area,
            max_depth,
            volume,
        });
    }

    let first_positive = first_positive_component_downstream(&component, receiver)?;
    let mut parents = vec![None; temporary.len()];
    for depression in &temporary {
        let mut targets = BTreeSet::<Option<u32>>::new();
        for &cell in &depression.cells {
            let cell = cell as usize;
            let next = match receiver[cell] {
                DrainageReceiverV0::Cell { cell: next, .. } => next as usize,
                DrainageReceiverV0::Portal { .. } => {
                    targets.insert(None);
                    continue;
                }
            };
            if component[next] == Some(depression.old_id) {
                continue;
            }
            let target = match first_positive[next] {
                Some(target) if target == depression.old_id => {
                    first_different_component_from(next, depression.old_id, &component, receiver)?
                }
                other => other,
            };
            targets.insert(target);
        }
        if targets.len() > 1 {
            return Err(DrainageErrorV0::DepressionHierarchyAmbiguity {
                depression: depression.old_id,
            });
        }
        parents[depression.old_id as usize] = targets.into_iter().next().flatten();
    }

    Ok(temporary
        .into_iter()
        .map(|depression| DrainageDepressionV0 {
            id: depression.old_id,
            parent: parents[depression.old_id as usize],
            anchor_cell: depression.anchor,
            affected_cells: depression.cells,
            spill_elevation_km: depression.spill,
            affected_area_km2: depression.area,
            maximum_fill_depth_km: depression.max_depth,
            virtual_fill_volume_km3: depression.volume,
        })
        .collect())
}

fn first_different_component_from(
    start: usize,
    excluded: u32,
    component: &[Option<u32>],
    receiver: &[DrainageReceiverV0],
) -> Result<Option<u32>, DrainageErrorV0> {
    let mut current = start;
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current) {
            return Err(DrainageErrorV0::ReceiverCycle);
        }
        if let Some(target) = component[current] {
            if target != excluded {
                return Ok(Some(target));
            }
        }
        match receiver[current] {
            DrainageReceiverV0::Cell { cell, .. } => current = cell as usize,
            DrainageReceiverV0::Portal { .. } => return Ok(None),
        }
    }
}

fn first_positive_component_downstream(
    component: &[Option<u32>],
    receiver: &[DrainageReceiverV0],
) -> Result<Vec<Option<u32>>, DrainageErrorV0> {
    let n = receiver.len();
    let mut first = vec![None; n];
    let mut resolved = vec![false; n];
    let mut states = vec![0u8; n];
    for start in 0..n {
        if resolved[start] {
            continue;
        }
        let mut path = Vec::new();
        let mut current = start;
        let value = loop {
            if resolved[current] {
                break first[current];
            }
            if let Some(id) = component[current] {
                first[current] = Some(id);
                resolved[current] = true;
                break Some(id);
            }
            if states[current] == 1 {
                return Err(DrainageErrorV0::ReceiverCycle);
            }
            states[current] = 1;
            path.push(current);
            match receiver[current] {
                DrainageReceiverV0::Cell { cell, .. } => current = cell as usize,
                DrainageReceiverV0::Portal { .. } => break None,
            }
        };
        for cell in path {
            first[cell] = value;
            resolved[cell] = true;
            states[cell] = 2;
        }
    }
    Ok(first)
}

fn build_scale(
    graph: &EvaluationSurfaceGraphV0,
    _physical_elevation_km: &[f64],
    local_runoff: &[f64],
    routing: &DrainageRoutingV0,
    threshold: f64,
) -> Result<DrainageScaleV0, DrainageErrorV0> {
    let n = graph.cell_count();
    let supported: Vec<_> = routing
        .structural_area_km2
        .iter()
        .map(|&area| area >= threshold)
        .collect();
    let donors = receiver_donors(&routing.receiver, n)?;
    let upstream_degree: Vec<_> = donors
        .iter()
        .map(|cells| cells.iter().filter(|&&cell| supported[cell]).count())
        .collect();
    let mut starts: Vec<_> = (0..n)
        .filter(|&cell| supported[cell] && upstream_degree[cell] != 1)
        .collect();
    starts.sort_by(|&a, &b| point_cmp(graph.cell_center_km[a], graph.cell_center_km[b]));

    let mut cell_reach = vec![None::<u32>; n];
    let mut reach_cells = Vec::<Vec<u32>>::new();
    for start in starts {
        if cell_reach[start].is_some() {
            continue;
        }
        let id = reach_cells.len() as u32;
        let mut cells = Vec::new();
        let mut current = start;
        loop {
            if !supported[current] || cell_reach[current].is_some() {
                break;
            }
            cell_reach[current] = Some(id);
            cells.push(current as u32);
            let DrainageReceiverV0::Cell { cell: next, .. } = routing.receiver[current] else {
                break;
            };
            let next = next as usize;
            if !supported[next] || upstream_degree[next] != 1 {
                break;
            }
            current = next;
        }
        reach_cells.push(cells);
    }
    if let Some(cell) = supported
        .iter()
        .zip(&cell_reach)
        .position(|(&is_supported, owner)| is_supported && owner.is_none())
    {
        return Err(DrainageErrorV0::MissingReceiver { cell });
    }

    let mut downstream = vec![None::<u32>; reach_cells.len()];
    let mut terminal_portal = vec![None::<u32>; reach_cells.len()];
    for (reach, cells) in reach_cells.iter().enumerate() {
        let tail = *cells.last().expect("reach is non-empty") as usize;
        match routing.receiver[tail] {
            DrainageReceiverV0::Cell { cell: next, .. } => {
                downstream[reach] = cell_reach[next as usize];
                if downstream[reach].is_none() {
                    return Err(DrainageErrorV0::MissingReceiver { cell: tail });
                }
            }
            DrainageReceiverV0::Portal { portal_id, .. } => {
                terminal_portal[reach] = Some(portal_id);
            }
        }
    }
    let mut upstream = vec![Vec::<u32>::new(); reach_cells.len()];
    for (reach, &next) in downstream.iter().enumerate() {
        if let Some(next) = next {
            upstream[next as usize].push(reach as u32);
        }
    }
    for donors in &mut upstream {
        donors.sort_unstable();
    }

    let topological = reach_topological_order(graph, &reach_cells, &upstream, &downstream)?;
    let mut strahler = vec![0u32; reach_cells.len()];
    let mut longest_upstream = vec![0.0; reach_cells.len()];
    let mut lengths = vec![0.0; reach_cells.len()];
    for (reach, cells) in reach_cells.iter().enumerate() {
        lengths[reach] = kahan_sum(
            cells
                .iter()
                .map(|&cell| routing.segment_length_km[cell as usize]),
        );
    }
    for &reach in &topological {
        let reach = reach as usize;
        let mut maximum = 0;
        let mut maximum_count = 0;
        let mut longest: f64 = 0.0;
        for &donor in &upstream[reach] {
            let donor = donor as usize;
            if strahler[donor] > maximum {
                maximum = strahler[donor];
                maximum_count = 1;
            } else if strahler[donor] == maximum {
                maximum_count += 1;
            }
            longest = longest.max(longest_upstream[donor]);
        }
        strahler[reach] = if maximum == 0 {
            1
        } else if maximum_count >= 2 {
            maximum + 1
        } else {
            maximum
        };
        longest_upstream[reach] = lengths[reach] + longest;
    }

    let mut reaches = Vec::with_capacity(reach_cells.len());
    for (reach, cells) in reach_cells.iter().enumerate() {
        let head = cells[0] as usize;
        let tail = *cells.last().unwrap() as usize;
        let mut fill_count = 0u32;
        let mut fill_length = 0.0;
        let mut flat_count = 0u32;
        let mut flat_length = 0.0;
        let mut non_descending_count = 0u32;
        let mut non_descending_length = 0.0;
        for &cell in cells {
            let cell = cell as usize;
            if routing.fill_supported[cell] {
                fill_count += 1;
                fill_length += routing.segment_length_km[cell];
            }
            if routing.flat_supported[cell] {
                flat_count += 1;
                flat_length += routing.segment_length_km[cell];
            }
            if routing.physically_non_descending[cell] {
                non_descending_count += 1;
                non_descending_length += routing.segment_length_km[cell];
            }
        }
        reaches.push(RiverReachV0 {
            id: reach as u32,
            cells: cells.clone(),
            upstream_reaches: upstream[reach].clone(),
            downstream_reach: downstream[reach],
            terminal_portal_id: terminal_portal[reach],
            outlet_portal_id: routing.outlet_portal_id[tail],
            physical_length_km: lengths[reach],
            head_structural_area_km2: routing.structural_area_km2[head],
            tail_structural_area_km2: routing.structural_area_km2[tail],
            head_supplied_runoff: routing.supplied_runoff[head],
            tail_supplied_runoff: routing.supplied_runoff[tail],
            strahler_order: strahler[reach],
            fill_supported_segment_count: fill_count,
            fill_supported_length_km: fill_length,
            flat_supported_segment_count: flat_count,
            flat_supported_length_km: flat_length,
            physically_non_descending_segment_count: non_descending_count,
            physically_non_descending_length_km: non_descending_length,
        });
    }

    let exclusive_owner = exclusive_owners(&supported, &cell_reach, &routing.receiver)?;
    let mut exclusive_area = vec![0.0; reaches.len()];
    let mut exclusive_runoff = vec![0.0; reaches.len()];
    for cell in 0..n {
        if let IncrementalCatchmentOwnerV0::Reach(reach) = exclusive_owner[cell] {
            exclusive_area[reach as usize] += graph.cell_area_km2[cell];
            exclusive_runoff[reach as usize] += local_runoff[cell];
        }
    }
    let catchments = reaches
        .iter()
        .map(|reach| RetainedCatchmentV0 {
            reach_id: reach.id,
            parent_reach: reach.downstream_reach,
            child_reaches: reach.upstream_reaches.clone(),
            outlet_portal_id: reach.outlet_portal_id,
            nested_structural_area_km2: reach.tail_structural_area_km2,
            nested_supplied_runoff: reach.tail_supplied_runoff,
            exclusive_physical_area_km2: exclusive_area[reach.id as usize],
            exclusive_local_runoff: exclusive_runoff[reach.id as usize],
        })
        .collect();
    let raw_catchment_boundaries = raw_boundaries(graph, &exclusive_owner);
    let portal_roles = build_portal_roles(graph, &reaches, &longest_upstream);
    Ok(DrainageScaleV0 {
        support_threshold_km2: threshold,
        basin_graph: DrainageBasinGraphV0 {
            catchments,
            exclusive_owner,
            raw_catchment_boundaries,
        },
        reach_graph: RiverReachGraphV0 {
            cell_reach,
            reaches,
            portal_roles,
        },
    })
}

fn receiver_donors(
    receiver: &[DrainageReceiverV0],
    n: usize,
) -> Result<Vec<Vec<usize>>, DrainageErrorV0> {
    let mut donors = vec![Vec::new(); n];
    for (cell, value) in receiver.iter().enumerate() {
        if let DrainageReceiverV0::Cell { cell: next, .. } = *value {
            let next = next as usize;
            if next >= n {
                return Err(DrainageErrorV0::InvalidGraph(format!(
                    "receiver {next} out of range"
                )));
            }
            donors[next].push(cell);
        }
    }
    for cells in &mut donors {
        cells.sort_unstable();
    }
    Ok(donors)
}

fn reach_topological_order(
    graph: &EvaluationSurfaceGraphV0,
    reach_cells: &[Vec<u32>],
    upstream: &[Vec<u32>],
    downstream: &[Option<u32>],
) -> Result<Vec<u32>, DrainageErrorV0> {
    let mut remaining: Vec<_> = upstream.iter().map(Vec::len).collect();
    let mut ready = BTreeSet::<(PointKey, u32)>::new();
    for (reach, &degree) in remaining.iter().enumerate() {
        if degree == 0 {
            let head = reach_cells[reach][0] as usize;
            ready.insert((point_key(graph.cell_center_km[head]), reach as u32));
        }
    }
    let mut order = Vec::with_capacity(reach_cells.len());
    while let Some(key) = ready.pop_first() {
        let reach = key.1;
        order.push(reach);
        if let Some(next) = downstream[reach as usize] {
            let slot = &mut remaining[next as usize];
            *slot = slot.checked_sub(1).ok_or(DrainageErrorV0::ReceiverCycle)?;
            if *slot == 0 {
                let head = reach_cells[next as usize][0] as usize;
                ready.insert((point_key(graph.cell_center_km[head]), next));
            }
        }
    }
    if order.len() != reach_cells.len() {
        return Err(DrainageErrorV0::ReceiverCycle);
    }
    Ok(order)
}

fn exclusive_owners(
    supported: &[bool],
    cell_reach: &[Option<u32>],
    receiver: &[DrainageReceiverV0],
) -> Result<Vec<IncrementalCatchmentOwnerV0>, DrainageErrorV0> {
    let n = receiver.len();
    let mut owners = vec![None; n];
    let mut states = vec![0u8; n];
    fn resolve(
        start: usize,
        supported: &[bool],
        cell_reach: &[Option<u32>],
        receiver: &[DrainageReceiverV0],
        owners: &mut [Option<IncrementalCatchmentOwnerV0>],
        states: &mut [u8],
    ) -> Result<IncrementalCatchmentOwnerV0, DrainageErrorV0> {
        if let Some(owner) = owners[start] {
            return Ok(owner);
        }
        let mut path = Vec::new();
        let mut current = start;
        let owner = loop {
            if let Some(owner) = owners[current] {
                break owner;
            }
            if supported[current] {
                break IncrementalCatchmentOwnerV0::Reach(
                    cell_reach[current]
                        .ok_or(DrainageErrorV0::MissingReceiver { cell: current })?,
                );
            }
            if states[current] == 1 {
                return Err(DrainageErrorV0::ReceiverCycle);
            }
            states[current] = 1;
            path.push(current);
            match receiver[current] {
                DrainageReceiverV0::Cell { cell, .. } => current = cell as usize,
                DrainageReceiverV0::Portal { portal_id, .. } => {
                    break IncrementalCatchmentOwnerV0::Portal(portal_id)
                }
            }
        };
        if supported[current] {
            owners[current] = Some(owner);
            states[current] = 2;
        }
        for cell in path {
            owners[cell] = Some(owner);
            states[cell] = 2;
        }
        Ok(owner)
    }
    for cell in 0..n {
        let _ = resolve(
            cell,
            supported,
            cell_reach,
            receiver,
            &mut owners,
            &mut states,
        )?;
    }
    owners
        .into_iter()
        .enumerate()
        .map(|(cell, owner)| owner.ok_or(DrainageErrorV0::MissingReceiver { cell }))
        .collect()
}

fn raw_boundaries(
    graph: &EvaluationSurfaceGraphV0,
    owners: &[IncrementalCatchmentOwnerV0],
) -> Vec<RawCatchmentBoundaryFaceV0> {
    let mut result = Vec::new();
    for cell in 0..graph.cell_count() {
        for edge in edge_range(&graph.edge_offsets, cell) {
            let reciprocal = graph.edge_reciprocal[edge] as usize;
            if edge > reciprocal {
                continue;
            }
            let neighbor = graph.edge_neighbor[edge] as usize;
            if owners[cell] == owners[neighbor] {
                continue;
            }
            let mut owner_pair = [owners[cell], owners[neighbor]];
            owner_pair.sort();
            let a = graph.edge_face_endpoints_km[edge];
            let b = graph.edge_face_endpoints_km[reciprocal];
            let canonical_edge = if endpoints_cmp(a, b) == Ordering::Greater {
                reciprocal
            } else {
                edge
            };
            result.push(RawCatchmentBoundaryFaceV0 {
                owners: owner_pair,
                endpoints_km: graph.edge_face_endpoints_km[canonical_edge],
                physical_length_km: graph.edge_shared_width_km[canonical_edge],
            });
        }
    }
    result.sort_by(|a, b| {
        a.owners
            .cmp(&b.owners)
            .then_with(|| endpoints_cmp(a.endpoints_km, b.endpoints_km))
    });
    result
}

#[derive(Clone, Copy)]
enum TrunkCriterion {
    Supply,
    Length,
    Order,
}

fn build_portal_roles(
    graph: &EvaluationSurfaceGraphV0,
    reaches: &[RiverReachV0],
    longest_upstream: &[f64],
) -> Vec<PortalTrunkRolesV0> {
    let mut terminals = BTreeMap::<u32, Vec<u32>>::new();
    for reach in reaches {
        if let Some(portal) = reach.terminal_portal_id {
            terminals.entry(portal).or_default().push(reach.id);
        }
    }
    terminals
        .into_iter()
        .map(|(portal_id, mut terminal_reaches)| {
            terminal_reaches.sort_unstable();
            let greatest_supply = trunk_path(
                graph,
                reaches,
                longest_upstream,
                &terminal_reaches,
                TrunkCriterion::Supply,
            );
            let longest_trunk = trunk_path(
                graph,
                reaches,
                longest_upstream,
                &terminal_reaches,
                TrunkCriterion::Length,
            );
            let highest_order = trunk_path(
                graph,
                reaches,
                longest_upstream,
                &terminal_reaches,
                TrunkCriterion::Order,
            );
            PortalTrunkRolesV0 {
                portal_id,
                terminal_reaches,
                greatest_supply,
                longest_trunk,
                highest_order,
            }
        })
        .collect()
}

fn trunk_path(
    graph: &EvaluationSurfaceGraphV0,
    reaches: &[RiverReachV0],
    longest_upstream: &[f64],
    terminal_reaches: &[u32],
    criterion: TrunkCriterion,
) -> Vec<u32> {
    if terminal_reaches.is_empty() {
        return Vec::new();
    }
    let mut current = *terminal_reaches
        .iter()
        .max_by(|&&a, &&b| role_cmp(graph, reaches, longest_upstream, a, b, criterion))
        .unwrap();
    let mut path = vec![current];
    loop {
        let upstream = &reaches[current as usize].upstream_reaches;
        let Some(next) = upstream
            .iter()
            .copied()
            .max_by(|&a, &b| role_cmp(graph, reaches, longest_upstream, a, b, criterion))
        else {
            break;
        };
        current = next;
        path.push(current);
    }
    path.reverse();
    path
}

fn role_cmp(
    graph: &EvaluationSurfaceGraphV0,
    reaches: &[RiverReachV0],
    longest_upstream: &[f64],
    a: u32,
    b: u32,
    criterion: TrunkCriterion,
) -> Ordering {
    let a_reach = &reaches[a as usize];
    let b_reach = &reaches[b as usize];
    let primary = match criterion {
        TrunkCriterion::Supply => a_reach
            .tail_supplied_runoff
            .total_cmp(&b_reach.tail_supplied_runoff),
        TrunkCriterion::Length => {
            longest_upstream[a as usize].total_cmp(&longest_upstream[b as usize])
        }
        TrunkCriterion::Order => a_reach.strahler_order.cmp(&b_reach.strahler_order),
    };
    primary
        .then_with(|| {
            a_reach
                .tail_structural_area_km2
                .total_cmp(&b_reach.tail_structural_area_km2)
        })
        .then_with(|| {
            a_reach
                .physical_length_km
                .total_cmp(&b_reach.physical_length_km)
        })
        // max_by must prefer the lexicographically smaller head.
        .then_with(|| {
            let a_head = graph.cell_center_km[a_reach.cells[0] as usize];
            let b_head = graph.cell_center_km[b_reach.cells[0] as usize];
            point_cmp(b_head, a_head)
        })
}

type PointKey = (OrderedFloat<f64>, OrderedFloat<f64>, OrderedFloat<f64>);

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn point_key(point: DVec3) -> PointKey {
    (
        OrderedFloat(canonical_zero(point.x)),
        OrderedFloat(canonical_zero(point.y)),
        OrderedFloat(canonical_zero(point.z)),
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

fn edge_range(offsets: &[u32], cell: usize) -> std::ops::Range<usize> {
    offsets[cell] as usize..offsets[cell + 1] as usize
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

fn kahan_add(sum: &mut f64, correction: &mut f64, value: f64) -> Result<(), ()> {
    let adjusted = value - *correction;
    let next = *sum + adjusted;
    *correction = (next - *sum) - adjusted;
    *sum = next;
    if sum.is_finite() && correction.is_finite() {
        Ok(())
    } else {
        Err(())
    }
}

fn evidence_bytes(
    graph: &EvaluationSurfaceGraphV0,
    physical_elevation_km: &[f64],
    local_runoff_supply: &[f64],
    config: &DrainageConfigV0,
    result: &EvaluationDrainageV0,
) -> Result<Vec<u8>, DrainageErrorV0> {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .serialize(&(
            graph,
            physical_elevation_km,
            local_runoff_supply,
            config,
            &result.schema_version,
            &result.hash_version,
            &result.routing,
            &result.depressions,
            &result.scales,
        ))
        .map_err(|error| DrainageErrorV0::Serialization(error.to_string()))
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
mod tests {
    use super::*;

    fn role_reach(
        id: u32,
        upstream_reaches: Vec<u32>,
        terminal_portal_id: Option<u32>,
        runoff: f64,
        length: f64,
        order: u32,
    ) -> RiverReachV0 {
        RiverReachV0 {
            id,
            cells: vec![id],
            upstream_reaches,
            downstream_reach: if id < 3 { Some(3) } else { None },
            terminal_portal_id,
            outlet_portal_id: 9,
            physical_length_km: length,
            head_structural_area_km2: 1.0,
            tail_structural_area_km2: 1.0,
            head_supplied_runoff: runoff,
            tail_supplied_runoff: runoff,
            strahler_order: order,
            fill_supported_segment_count: 0,
            fill_supported_length_km: 0.0,
            flat_supported_segment_count: 0,
            flat_supported_length_km: 0.0,
            physically_non_descending_segment_count: 0,
            physically_non_descending_length_km: 0.0,
        }
    }

    #[test]
    fn trunk_roles_remain_distinct_when_fixture_makes_them_distinct() {
        let graph = EvaluationSurfaceGraphV0 {
            domain: EvaluationDomainV0::Planar,
            cell_center_km: vec![
                DVec3::new(-3.0, 0.0, 0.0),
                DVec3::new(-2.0, 0.0, 0.0),
                DVec3::new(-1.0, 0.0, 0.0),
                DVec3::ZERO,
            ],
            cell_area_km2: Vec::new(),
            cell_polygon_offsets: Vec::new(),
            cell_polygon_vertices_km: Vec::new(),
            edge_offsets: Vec::new(),
            edge_neighbor: Vec::new(),
            edge_reciprocal: Vec::new(),
            edge_distance_km: Vec::new(),
            edge_shared_width_km: Vec::new(),
            edge_face_endpoints_km: Vec::new(),
            boundary_segments: Vec::new(),
        };
        let reaches = vec![
            role_reach(0, Vec::new(), None, 100.0, 10.0, 1),
            role_reach(1, Vec::new(), None, 50.0, 20.0, 1),
            role_reach(2, Vec::new(), None, 30.0, 30.0, 3),
            role_reach(3, vec![0, 1, 2], Some(9), 180.0, 5.0, 4),
        ];
        let roles = build_portal_roles(&graph, &reaches, &[10.0, 200.0, 30.0, 205.0]);
        assert_eq!(roles.len(), 1);
        assert_eq!(roles[0].greatest_supply, vec![0, 3]);
        assert_eq!(roles[0].longest_trunk, vec![1, 3]);
        assert_eq!(roles[0].highest_order, vec![2, 3]);
    }

    #[test]
    fn depression_parent_rejects_mixed_parent_and_portal_exits() {
        let graph = EvaluationSurfaceGraphV0 {
            domain: EvaluationDomainV0::Planar,
            cell_center_km: vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(2.0, 0.0, 0.0),
            ],
            cell_area_km2: vec![1.0; 3],
            cell_polygon_offsets: Vec::new(),
            cell_polygon_vertices_km: Vec::new(),
            edge_offsets: vec![0, 1, 3, 4],
            edge_neighbor: vec![1, 0, 2, 1],
            edge_reciprocal: Vec::new(),
            edge_distance_km: Vec::new(),
            edge_shared_width_km: Vec::new(),
            edge_face_endpoints_km: Vec::new(),
            boundary_segments: Vec::new(),
        };
        let receiver = vec![
            DrainageReceiverV0::Portal {
                boundary_segment: 0,
                portal_id: 1,
            },
            DrainageReceiverV0::Cell {
                cell: 2,
                directed_edge: 2,
            },
            DrainageReceiverV0::Portal {
                boundary_segment: 1,
                portal_id: 1,
            },
        ];
        assert_eq!(
            build_depressions(&graph, &[0.0; 3], &[2.0, 2.0, 1.0], &receiver),
            Err(DrainageErrorV0::DepressionHierarchyAmbiguity { depression: 0 })
        );
    }

    #[test]
    fn disconnected_cycle_and_unknown_portal_are_typed() {
        let graph = EvaluationSurfaceGraphV0 {
            domain: EvaluationDomainV0::Planar,
            cell_center_km: vec![DVec3::ZERO, DVec3::X],
            cell_area_km2: vec![1.0; 2],
            cell_polygon_offsets: Vec::new(),
            cell_polygon_vertices_km: Vec::new(),
            edge_offsets: vec![0, 0, 0],
            edge_neighbor: Vec::new(),
            edge_reciprocal: Vec::new(),
            edge_distance_km: Vec::new(),
            edge_shared_width_km: Vec::new(),
            edge_face_endpoints_km: Vec::new(),
            boundary_segments: Vec::new(),
        };
        let portals = vec![PortalSegment {
            segment: 0,
            owner: 0,
            portal_id: 1,
            base_elevation_km: 0.0,
            midpoint: DVec3::new(0.0, -1.0, 0.0),
            width_km: 1.0,
            center_distance_km: 1.0,
        }];
        assert_eq!(
            derive_conditioning(&graph, &[1.0, 1.0], &portals),
            Err(DrainageErrorV0::DisconnectedFromPortal { cell: 1 })
        );

        let cycle = vec![
            DrainageReceiverV0::Cell {
                cell: 1,
                directed_edge: 0,
            },
            DrainageReceiverV0::Cell {
                cell: 0,
                directed_edge: 1,
            },
        ];
        assert_eq!(
            exclusive_owners(&[false, false], &[None, None], &cycle),
            Err(DrainageErrorV0::ReceiverCycle)
        );

        let unknown = vec![DrainageReceiverV0::Portal {
            boundary_segment: 0,
            portal_id: 99,
        }];
        let one_cell_graph = EvaluationSurfaceGraphV0 {
            domain: EvaluationDomainV0::Planar,
            cell_center_km: vec![DVec3::ZERO],
            cell_area_km2: vec![1.0],
            cell_polygon_offsets: Vec::new(),
            cell_polygon_vertices_km: Vec::new(),
            edge_offsets: vec![0, 0],
            edge_neighbor: Vec::new(),
            edge_reciprocal: Vec::new(),
            edge_distance_km: Vec::new(),
            edge_shared_width_km: Vec::new(),
            edge_face_endpoints_km: Vec::new(),
            boundary_segments: Vec::new(),
        };
        assert_eq!(
            accumulate(
                &one_cell_graph,
                &[1.0],
                &unknown,
                &[0],
                &portals,
                DrainageConfigV0::default()
            ),
            Err(DrainageErrorV0::UnknownPortal(99))
        );
    }
}
