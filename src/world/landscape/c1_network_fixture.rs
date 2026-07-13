//! Manufactured ownership, routing, and conservative remapping for `C1` reaches.
//!
//! This is deliberately a prescribed-network fixture. Stable reach identity and
//! physical geometry are inputs; terrain-derived receivers, width evolution,
//! sediment, and product integration are outside its contract.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use super::{
    apply_channel_only_excavation, apply_internal_interfluve_channel_transfer, C1CellGeometry,
    C1CellState, C1ExcavationLedger, C1FixtureError, C1InternalTransferLedger,
};

pub const REGISTERED_C1_K_PER_KM: f64 = 1.0e-4;
pub const REGISTERED_C1_DT_MYR: f64 = 0.1;

/// Stable semantic identity. Vector position is never an identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ReachId(pub u32);

/// Prescribed physical and forcing data for one semantic reach.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct C1ReachSpec {
    pub id: ReachId,
    pub length_km: f64,
    pub channel_width_km: f64,
    pub represented_swath_width_km: f64,
    pub grade: f64,
    pub headwater_discharge_km3_myr: f64,
    pub lateral_supply_km3_myr_per_km: f64,
    pub downstream: Option<ReachId>,
}

/// A segment is a physical half-open interval of a stable reach.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct C1Segment {
    pub reach_id: ReachId,
    pub s0_km: f64,
    pub s1_km: f64,
    pub geometry: C1CellGeometry,
}

impl C1Segment {
    pub fn length_km(self) -> f64 {
        self.s1_km - self.s0_km
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct C1ReachNetwork {
    reaches: Vec<C1ReachSpec>,
    segments: Vec<C1Segment>,
    /// Segment ownership ranges, indexed exactly like sorted `reaches`.
    segment_offsets: Vec<usize>,
    topological_order: Vec<ReachId>,
    nominal_spacing_km: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct C1SegmentFlow {
    pub reach_id: ReachId,
    pub s0_km: f64,
    pub s1_km: f64,
    pub inflow_km3_myr: f64,
    pub mean_discharge_km3_myr: f64,
    pub outflow_km3_myr: f64,
    pub closure_error_km3_myr: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct C1ReachAudit {
    pub reach_id: ReachId,
    pub upstream_inflow_km3_myr: f64,
    pub declared_headwater_km3_myr: f64,
    pub lateral_source_km3_myr: f64,
    pub outlet_flow_km3_myr: f64,
    pub closure_error_km3_myr: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct C1RoutingAudit {
    pub segment_flows: Vec<C1SegmentFlow>,
    pub reaches: Vec<C1ReachAudit>,
    pub outlets_km3_myr: BTreeMap<ReachId, f64>,
    pub total_source_km3_myr: f64,
    pub total_outlet_km3_myr: f64,
    pub network_closure_error_km3_myr: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct C1ResponseAudit {
    pub excavation: C1ExcavationLedger,
    pub exported_by_reach_km3: BTreeMap<ReachId, f64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct C1NetworkMoments {
    pub channel_elevation_volume_moment_km3: f64,
    pub interfluve_elevation_volume_moment_km3: f64,
    pub total_elevation_volume_moment_km3: f64,
}

/// Network plus state ownership, used to make topology replacement atomic.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct C1RoutedFixture {
    network: C1ReachNetwork,
    state: Vec<C1CellState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C1NetworkError {
    InvalidScalar {
        field: &'static str,
        reach: Option<ReachId>,
    },
    DuplicateReach(ReachId),
    UnknownReceiver {
        reach: ReachId,
        receiver: ReachId,
    },
    UnknownReach(ReachId),
    Cycle,
    StateLength {
        expected: usize,
        actual: usize,
    },
    InvalidState {
        segment: usize,
        source: C1FixtureError,
    },
    RoutingLength {
        expected: usize,
        actual: usize,
    },
    RoutingIdentity {
        segment: usize,
    },
    IncompatibleReachGeometry(ReachId),
    IncompleteOverlap {
        reach: ReachId,
        s0_bits: u64,
        s1_bits: u64,
    },
    NonFiniteComputation,
    C1(C1FixtureError),
}

impl fmt::Display for C1NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScalar { field, reach } => write!(f, "invalid {field} for {reach:?}"),
            Self::DuplicateReach(id) => write!(f, "duplicate reach {id:?}"),
            Self::UnknownReceiver { reach, receiver } => {
                write!(f, "reach {reach:?} has unknown receiver {receiver:?}")
            }
            Self::UnknownReach(id) => write!(f, "unknown reach {id:?}"),
            Self::Cycle => f.write_str("reach graph contains a cycle"),
            Self::StateLength { expected, actual } => {
                write!(f, "state has length {actual}, expected {expected}")
            }
            Self::InvalidState { segment, source } => {
                write!(f, "invalid state at segment {segment}: {source}")
            }
            Self::RoutingLength { expected, actual } => {
                write!(f, "routing has length {actual}, expected {expected}")
            }
            Self::RoutingIdentity { segment } => {
                write!(f, "routing identity mismatch at {segment}")
            }
            Self::IncompatibleReachGeometry(id) => {
                write!(f, "incompatible physical geometry for reach {id:?}")
            }
            Self::IncompleteOverlap { reach, .. } => {
                write!(f, "incomplete interval overlap for reach {reach:?}")
            }
            Self::NonFiniteComputation => f.write_str("non-finite network computation"),
            Self::C1(source) => source.fmt(f),
        }
    }
}

impl std::error::Error for C1NetworkError {}

impl From<C1FixtureError> for C1NetworkError {
    fn from(value: C1FixtureError) -> Self {
        Self::C1(value)
    }
}

impl C1ReachNetwork {
    pub fn new(
        mut reaches: Vec<C1ReachSpec>,
        nominal_spacing_km: f64,
    ) -> Result<Self, C1NetworkError> {
        if !nominal_spacing_km.is_finite() || nominal_spacing_km <= 0.0 {
            return Err(C1NetworkError::InvalidScalar {
                field: "nominal_spacing_km",
                reach: None,
            });
        }
        reaches.sort_by_key(|r| r.id);
        validate_specs(&reaches)?;
        let topological_order = deterministic_topological_order(&reaches)?;
        let mut segments = Vec::new();
        let mut segment_offsets = Vec::with_capacity(reaches.len() + 1);
        segment_offsets.push(0);
        for reach in &reaches {
            let mut s0 = 0.0;
            while s0 < reach.length_km {
                let s1 = (s0 + nominal_spacing_km).min(reach.length_km);
                let length = s1 - s0;
                segments.push(C1Segment {
                    reach_id: reach.id,
                    s0_km: s0,
                    s1_km: s1,
                    geometry: C1CellGeometry {
                        cell_area_km2: reach.represented_swath_width_km * length,
                        reach_length_km: length,
                        channel_width_km: reach.channel_width_km,
                    },
                });
                s0 = s1;
            }
            segment_offsets.push(segments.len());
        }
        Ok(Self {
            reaches,
            segments,
            segment_offsets,
            topological_order,
            nominal_spacing_km,
        })
    }

    pub fn reaches(&self) -> &[C1ReachSpec] {
        &self.reaches
    }
    pub fn segments(&self) -> &[C1Segment] {
        &self.segments
    }
    pub fn nominal_spacing_km(&self) -> f64 {
        self.nominal_spacing_km
    }

    pub fn reach(&self, id: ReachId) -> Option<&C1ReachSpec> {
        self.reach_index(id).map(|i| &self.reaches[i])
    }

    fn reach_index(&self, id: ReachId) -> Option<usize> {
        self.reaches.binary_search_by_key(&id, |r| r.id).ok()
    }

    fn segment_range(&self, reach_index: usize) -> std::ops::Range<usize> {
        self.segment_offsets[reach_index]..self.segment_offsets[reach_index + 1]
    }

    pub fn route(&self) -> Result<C1RoutingAudit, C1NetworkError> {
        let mut upstream = vec![0.0; self.reaches.len()];
        let mut segment_flows = vec![None; self.segments.len()];
        let mut reach_audits = BTreeMap::new();
        let mut outlets = BTreeMap::new();
        for id in &self.topological_order {
            let reach_index = self
                .reach_index(*id)
                .ok_or(C1NetworkError::UnknownReach(*id))?;
            let spec = self.reaches[reach_index];
            let upstream_flow = upstream[reach_index];
            let mut q = upstream_flow + spec.headwater_discharge_km3_myr;
            for segment_index in self.segment_range(reach_index) {
                let segment = self.segments[segment_index];
                let source = spec.lateral_supply_km3_myr_per_km * segment.length_km();
                let out = q + source;
                segment_flows[segment_index] = Some(C1SegmentFlow {
                    reach_id: *id,
                    s0_km: segment.s0_km,
                    s1_km: segment.s1_km,
                    inflow_km3_myr: q,
                    mean_discharge_km3_myr: q + 0.5 * source,
                    outflow_km3_myr: out,
                    closure_error_km3_myr: out - (q + source),
                });
                q = out;
            }
            let lateral = spec.lateral_supply_km3_myr_per_km * spec.length_km;
            reach_audits.insert(
                *id,
                C1ReachAudit {
                    reach_id: *id,
                    upstream_inflow_km3_myr: upstream_flow,
                    declared_headwater_km3_myr: spec.headwater_discharge_km3_myr,
                    lateral_source_km3_myr: lateral,
                    outlet_flow_km3_myr: q,
                    closure_error_km3_myr: q
                        - (upstream_flow + spec.headwater_discharge_km3_myr + lateral),
                },
            );
            if let Some(receiver) = spec.downstream {
                let receiver_index =
                    self.reach_index(receiver)
                        .ok_or(C1NetworkError::UnknownReceiver {
                            reach: spec.id,
                            receiver,
                        })?;
                upstream[receiver_index] += q;
            } else {
                outlets.insert(*id, q);
            }
        }
        let segment_flows = segment_flows
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(C1NetworkError::NonFiniteComputation)?;
        let total_source = self
            .reaches
            .iter()
            .map(|r| r.headwater_discharge_km3_myr + r.lateral_supply_km3_myr_per_km * r.length_km)
            .sum::<f64>();
        let total_outlet = outlets.values().sum::<f64>();
        if !total_source.is_finite() || !total_outlet.is_finite() {
            return Err(C1NetworkError::NonFiniteComputation);
        }
        Ok(C1RoutingAudit {
            segment_flows,
            reaches: reach_audits.into_values().collect(),
            outlets_km3_myr: outlets,
            total_source_km3_myr: total_source,
            total_outlet_km3_myr: total_outlet,
            network_closure_error_km3_myr: total_outlet - total_source,
        })
    }

    /// Replace one receiver only after the complete candidate graph validates.
    /// Segment geometry and ordering are not reconstructed.
    pub fn replace_receiver(
        &mut self,
        reach: ReachId,
        downstream: Option<ReachId>,
    ) -> Result<(), C1NetworkError> {
        let index = self
            .reaches
            .binary_search_by_key(&reach, |r| r.id)
            .map_err(|_| C1NetworkError::UnknownReach(reach))?;
        let mut candidate = self.reaches.clone();
        candidate[index].downstream = downstream;
        validate_specs(&candidate)?;
        let order = deterministic_topological_order(&candidate)?;
        self.reaches[index].downstream = downstream;
        self.topological_order = order;
        Ok(())
    }
}

impl C1RoutedFixture {
    pub fn new(network: C1ReachNetwork, state: Vec<C1CellState>) -> Result<Self, C1NetworkError> {
        validate_state(&network, &state)?;
        Ok(Self { network, state })
    }

    pub fn network(&self) -> &C1ReachNetwork {
        &self.network
    }

    pub fn state(&self) -> &[C1CellState] {
        &self.state
    }

    /// Mutable state values without exposing an operation that can change
    /// segment/state alignment.
    pub fn state_mut(&mut self) -> &mut [C1CellState] {
        &mut self.state
    }

    /// Atomic topology event: invalid candidates leave network and state untouched.
    pub fn replace_receiver(
        &mut self,
        reach: ReachId,
        downstream: Option<ReachId>,
    ) -> Result<(), C1NetworkError> {
        self.network.replace_receiver(reach, downstream)
    }
}

pub fn apply_unit_stream_power_response(
    network: &C1ReachNetwork,
    routing: &C1RoutingAudit,
    state: &mut [C1CellState],
    k_per_km: f64,
    dt_myr: f64,
) -> Result<C1ResponseAudit, C1NetworkError> {
    validate_state(network, state)?;
    if !k_per_km.is_finite() || k_per_km < 0.0 {
        return Err(C1NetworkError::InvalidScalar {
            field: "k_per_km",
            reach: None,
        });
    }
    if !dt_myr.is_finite() || dt_myr < 0.0 {
        return Err(C1NetworkError::InvalidScalar {
            field: "dt_myr",
            reach: None,
        });
    }
    if routing.segment_flows.len() != network.segments.len() {
        return Err(C1NetworkError::RoutingLength {
            expected: network.segments.len(),
            actual: routing.segment_flows.len(),
        });
    }
    let mut dz = Vec::with_capacity(state.len());
    let mut by_reach = BTreeMap::new();
    for (index, (segment, flow)) in network
        .segments
        .iter()
        .zip(&routing.segment_flows)
        .enumerate()
    {
        if segment.reach_id != flow.reach_id
            || segment.s0_km.to_bits() != flow.s0_km.to_bits()
            || segment.s1_km.to_bits() != flow.s1_km.to_bits()
        {
            return Err(C1NetworkError::RoutingIdentity { segment: index });
        }
        let spec = network
            .reach(segment.reach_id)
            .ok_or(C1NetworkError::UnknownReach(segment.reach_id))?;
        let lowering =
            k_per_km * (flow.mean_discharge_km3_myr / spec.channel_width_km) * spec.grade * dt_myr;
        if !lowering.is_finite() {
            return Err(C1NetworkError::NonFiniteComputation);
        }
        dz.push(-lowering);
        *by_reach.entry(segment.reach_id).or_insert(0.0) +=
            segment.geometry.channel_area_km2() * lowering;
    }
    let excavation = apply_channel_only_excavation(
        &network
            .segments
            .iter()
            .map(|s| s.geometry)
            .collect::<Vec<_>>(),
        state,
        &dz,
    )?;
    Ok(C1ResponseAudit {
        excavation,
        exported_by_reach_km3: by_reach,
    })
}

pub fn apply_internal_transfer_per_reach_length(
    network: &C1ReachNetwork,
    state: &mut [C1CellState],
    transfer_km3_per_km: &BTreeMap<ReachId, f64>,
) -> Result<C1InternalTransferLedger, C1NetworkError> {
    validate_state(network, state)?;
    for (id, density) in transfer_km3_per_km {
        if network.reach(*id).is_none() {
            return Err(C1NetworkError::UnknownReach(*id));
        }
        if !density.is_finite() || *density < 0.0 {
            return Err(C1NetworkError::InvalidScalar {
                field: "transfer_km3_per_km",
                reach: Some(*id),
            });
        }
    }
    let volumes = network
        .segments
        .iter()
        .map(|s| transfer_km3_per_km.get(&s.reach_id).copied().unwrap_or(0.0) * s.length_km())
        .collect::<Vec<_>>();
    Ok(apply_internal_interfluve_channel_transfer(
        &network
            .segments
            .iter()
            .map(|s| s.geometry)
            .collect::<Vec<_>>(),
        state,
        &volumes,
    )?)
}

pub fn network_moments(
    network: &C1ReachNetwork,
    state: &[C1CellState],
) -> Result<C1NetworkMoments, C1NetworkError> {
    validate_state(network, state)?;
    let mut result = C1NetworkMoments::default();
    for (index, segment) in network.segments.iter().enumerate() {
        let ac = segment.geometry.channel_area_km2();
        let ai = segment.geometry.cell_area_km2 - ac;
        let zi = state[index]
            .interfluve_mean_elevation_km(segment.geometry)
            .map_err(|source| C1NetworkError::InvalidState {
                segment: index,
                source,
            })?;
        result.channel_elevation_volume_moment_km3 +=
            ac * state[index].channel_surface_elevation_km;
        result.interfluve_elevation_volume_moment_km3 += ai * zi;
        result.total_elevation_volume_moment_km3 +=
            segment.geometry.cell_area_km2 * state[index].mean_elevation_km;
    }
    Ok(result)
}

pub fn remap_c1_state_by_reach_overlap(
    source: &C1ReachNetwork,
    source_state: &[C1CellState],
    destination: &C1ReachNetwork,
) -> Result<Vec<C1CellState>, C1NetworkError> {
    validate_state(source, source_state)?;
    for source_reach in source.reaches() {
        if destination.reach(source_reach.id).is_none() {
            return Err(C1NetworkError::IncompleteOverlap {
                reach: source_reach.id,
                s0_bits: 0.0f64.to_bits(),
                s1_bits: source_reach.length_km.to_bits(),
            });
        }
    }
    for destination_reach in destination.reaches() {
        let source_reach = source
            .reach(destination_reach.id)
            .ok_or(C1NetworkError::UnknownReach(destination_reach.id))?;
        if source_reach.length_km.to_bits() != destination_reach.length_km.to_bits()
            || source_reach.channel_width_km.to_bits()
                != destination_reach.channel_width_km.to_bits()
            || source_reach.represented_swath_width_km.to_bits()
                != destination_reach.represented_swath_width_km.to_bits()
        {
            return Err(C1NetworkError::IncompatibleReachGeometry(
                destination_reach.id,
            ));
        }
    }
    let mut result = Vec::with_capacity(destination.segments.len());
    for destination_reach_index in 0..destination.reaches.len() {
        let reach_id = destination.reaches[destination_reach_index].id;
        let source_reach_index = source
            .reach_index(reach_id)
            .ok_or(C1NetworkError::UnknownReach(reach_id))?;
        let source_range = source.segment_range(source_reach_index);
        let mut source_index = source_range.start;
        for destination_index in destination.segment_range(destination_reach_index) {
            let destination_segment = destination.segments[destination_index];
            let length = destination_segment.length_km();
            let tolerance = 64.0 * f64::EPSILON * length.max(1.0);
            let mut cursor = destination_segment.s0_km;
            let mut covered = 0.0;
            let mut channel_length_moment = 0.0;
            let mut interfluve_length_moment = 0.0;

            while source_index < source_range.end
                && source.segments[source_index].s1_km <= cursor + tolerance
            {
                source_index += 1;
            }
            let mut overlap_index = source_index;
            while cursor < destination_segment.s1_km - tolerance && overlap_index < source_range.end
            {
                let source_segment = source.segments[overlap_index];
                if source_segment.s0_km > cursor + tolerance {
                    break;
                }
                let overlap_end = destination_segment.s1_km.min(source_segment.s1_km);
                let overlap = overlap_end - cursor;
                if overlap > 0.0 {
                    let zi = source_state[overlap_index]
                        .interfluve_mean_elevation_km(source_segment.geometry)
                        .map_err(|source| C1NetworkError::InvalidState {
                            segment: overlap_index,
                            source,
                        })?;
                    covered += overlap;
                    channel_length_moment +=
                        overlap * source_state[overlap_index].channel_surface_elevation_km;
                    interfluve_length_moment += overlap * zi;
                    cursor = overlap_end;
                }
                if source_segment.s1_km <= cursor + tolerance {
                    overlap_index += 1;
                }
            }
            // A source segment may straddle the next destination boundary.
            source_index = overlap_index.saturating_sub(1).max(source_range.start);
            if (covered - length).abs() > tolerance {
                return Err(C1NetworkError::IncompleteOverlap {
                    reach: destination_segment.reach_id,
                    s0_bits: destination_segment.s0_km.to_bits(),
                    s1_bits: destination_segment.s1_km.to_bits(),
                });
            }
            let zc = channel_length_moment / covered;
            let zi = interfluve_length_moment / covered;
            let f = destination_segment.geometry.channel_fraction();
            let mean = f * zc + (1.0 - f) * zi;
            if !zc.is_finite() || !mean.is_finite() {
                return Err(C1NetworkError::NonFiniteComputation);
            }
            result.push(C1CellState {
                mean_elevation_km: mean,
                channel_surface_elevation_km: zc,
            });
        }
    }
    Ok(result)
}

fn validate_state(network: &C1ReachNetwork, state: &[C1CellState]) -> Result<(), C1NetworkError> {
    if state.len() != network.segments.len() {
        return Err(C1NetworkError::StateLength {
            expected: network.segments.len(),
            actual: state.len(),
        });
    }
    for (index, (segment, state)) in network.segments.iter().zip(state).enumerate() {
        state
            .interfluve_mean_elevation_km(segment.geometry)
            .map_err(|source| C1NetworkError::InvalidState {
                segment: index,
                source,
            })?;
    }
    Ok(())
}

fn validate_specs(reaches: &[C1ReachSpec]) -> Result<(), C1NetworkError> {
    let mut ids = BTreeSet::new();
    for r in reaches {
        if !ids.insert(r.id) {
            return Err(C1NetworkError::DuplicateReach(r.id));
        }
        for (field, value, positive) in [
            ("length_km", r.length_km, true),
            ("channel_width_km", r.channel_width_km, true),
            (
                "represented_swath_width_km",
                r.represented_swath_width_km,
                true,
            ),
            ("grade", r.grade, false),
            (
                "headwater_discharge_km3_myr",
                r.headwater_discharge_km3_myr,
                false,
            ),
            (
                "lateral_supply_km3_myr_per_km",
                r.lateral_supply_km3_myr_per_km,
                false,
            ),
        ] {
            if !value.is_finite() || if positive { value <= 0.0 } else { value < 0.0 } {
                return Err(C1NetworkError::InvalidScalar {
                    field,
                    reach: Some(r.id),
                });
            }
        }
        if r.channel_width_km >= r.represented_swath_width_km {
            return Err(C1NetworkError::InvalidScalar {
                field: "channel_width_km",
                reach: Some(r.id),
            });
        }
    }
    for r in reaches {
        if let Some(receiver) = r.downstream {
            if !ids.contains(&receiver) {
                return Err(C1NetworkError::UnknownReceiver {
                    reach: r.id,
                    receiver,
                });
            }
        }
    }
    Ok(())
}

fn deterministic_topological_order(
    reaches: &[C1ReachSpec],
) -> Result<Vec<ReachId>, C1NetworkError> {
    let mut indegree = reaches
        .iter()
        .map(|r| (r.id, 0usize))
        .collect::<BTreeMap<_, _>>();
    for r in reaches {
        if let Some(receiver) = r.downstream {
            *indegree
                .get_mut(&receiver)
                .ok_or(C1NetworkError::UnknownReceiver {
                    reach: r.id,
                    receiver,
                })? += 1;
        }
    }
    let mut ready = indegree
        .iter()
        .filter_map(|(id, degree)| (*degree == 0).then_some(*id))
        .collect::<BTreeSet<_>>();
    let by_id = reaches
        .iter()
        .map(|r| (r.id, r))
        .collect::<BTreeMap<_, _>>();
    let mut order = Vec::with_capacity(reaches.len());
    while let Some(id) = ready.pop_first() {
        order.push(id);
        if let Some(receiver) = by_id[&id].downstream {
            let degree = indegree.get_mut(&receiver).unwrap();
            *degree -= 1;
            if *degree == 0 {
                ready.insert(receiver);
            }
        }
    }
    if order.len() != reaches.len() {
        return Err(C1NetworkError::Cycle);
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: ReachId = ReachId(1);
    const B: ReachId = ReachId(2);
    const C: ReachId = ReachId(3);
    const D: ReachId = ReachId(4);

    fn specs(after: bool) -> Vec<C1ReachSpec> {
        vec![
            C1ReachSpec {
                id: A,
                length_km: 64.0,
                channel_width_km: 0.12,
                represented_swath_width_km: 12.0,
                grade: 0.012,
                headwater_discharge_km3_myr: 2.0,
                lateral_supply_km3_myr_per_km: 0.10,
                downstream: Some(C),
            },
            C1ReachSpec {
                id: B,
                length_km: 48.0,
                channel_width_km: 0.10,
                represented_swath_width_km: 10.0,
                grade: 0.015,
                headwater_discharge_km3_myr: 1.0,
                lateral_supply_km3_myr_per_km: 0.08,
                downstream: Some(if after { D } else { C }),
            },
            C1ReachSpec {
                id: C,
                length_km: 96.0,
                channel_width_km: 0.22,
                represented_swath_width_km: 16.0,
                grade: 0.008,
                headwater_discharge_km3_myr: 0.5,
                lateral_supply_km3_myr_per_km: 0.05,
                downstream: None,
            },
            C1ReachSpec {
                id: D,
                length_km: 80.0,
                channel_width_km: 0.18,
                represented_swath_width_km: 14.0,
                grade: 0.009,
                headwater_discharge_km3_myr: 0.75,
                lateral_supply_km3_myr_per_km: 0.04,
                downstream: None,
            },
        ]
    }
    fn net(h: f64, after: bool) -> C1ReachNetwork {
        C1ReachNetwork::new(specs(after), h).unwrap()
    }
    fn state(n: &C1ReachNetwork) -> Vec<C1CellState> {
        n.segments()
            .iter()
            .map(|s| {
                let x = s.reach_id.0 as f64 * 0.1 + s.s0_km * 0.001;
                let zc = 0.4 + x;
                let zi = 1.0 + 2.0 * x;
                let f = s.geometry.channel_fraction();
                C1CellState {
                    mean_elevation_km: f * zc + (1.0 - f) * zi,
                    channel_surface_elevation_km: zc,
                }
            })
            .collect()
    }
    fn close(a: f64, b: f64, t: f64) {
        assert!((a - b).abs() <= t, "{a:.17e} != {b:.17e}");
    }

    #[test]
    fn geometry_and_registered_water_are_invariant() {
        for h in [8., 4., 2.] {
            for after in [false, true] {
                let n = net(h, after);
                close(
                    n.segments().iter().map(|s| s.length_km()).sum(),
                    288.,
                    1e-13,
                );
                close(
                    n.segments()
                        .iter()
                        .map(|s| s.geometry.channel_area_km2())
                        .sum(),
                    48.,
                    1e-13,
                );
                for r in n.reaches() {
                    let reach_segments = n.segments().iter().filter(|s| s.reach_id == r.id);
                    close(
                        reach_segments.clone().map(|s| s.length_km()).sum(),
                        r.length_km,
                        1e-14,
                    );
                    close(
                        reach_segments.map(|s| s.geometry.channel_area_km2()).sum(),
                        r.channel_width_km * r.length_km,
                        3e-14,
                    );
                }
                let q = n.route().unwrap();
                close(q.total_outlet_km3_myr, 22.49, 5e-14);
                close(q.network_closure_error_km3_myr, 0., 5e-14);
                assert!(q
                    .segment_flows
                    .iter()
                    .all(|x| x.closure_error_km3_myr == 0.));
                assert!(q
                    .reaches
                    .iter()
                    .all(|x| x.closure_error_km3_myr.abs() <= 5e-14));
                let c_audit = q.reaches.iter().find(|x| x.reach_id == C).unwrap();
                let d_audit = q.reaches.iter().find(|x| x.reach_id == D).unwrap();
                close(
                    c_audit.upstream_inflow_km3_myr,
                    if after { 8.4 } else { 13.24 },
                    3e-14,
                );
                close(
                    d_audit.upstream_inflow_km3_myr,
                    if after { 4.84 } else { 0.0 },
                    3e-14,
                );
                let expected = if after {
                    [(C, 13.70), (D, 8.79)]
                } else {
                    [(C, 18.54), (D, 3.95)]
                };
                for (id, v) in expected {
                    close(q.outlets_km3_myr[&id], v, 5e-14);
                }
            }
        }
    }

    #[test]
    fn response_export_is_invariant_and_capture_only_changes_trunks() {
        let mut exports = [0.; 3];
        for (i, h) in [8., 4., 2.].into_iter().enumerate() {
            let n = net(h, false);
            let q = n.route().unwrap();
            let mut s = state(&n);
            let a = apply_unit_stream_power_response(
                &n,
                &q,
                &mut s,
                REGISTERED_C1_K_PER_KM,
                REGISTERED_C1_DT_MYR,
            )
            .unwrap();
            exports[i] = a.excavation.exported_solid_volume_km3;
            close(a.excavation.closure_error_km3, 0., 2e-10);
            close(exports[i], 0.0002018352, 3e-16);
        }
        close(exports[0], exports[1], 3e-18);
        close(exports[1], exports[2], 3e-18);
        let before = net(4., false);
        let after = net(4., true);
        let qb = before.route().unwrap();
        let qa = after.route().unwrap();
        for id in [A, B] {
            let before_flow = qb
                .segment_flows
                .iter()
                .filter(|flow| flow.reach_id == id)
                .collect::<Vec<_>>();
            let after_flow = qa
                .segment_flows
                .iter()
                .filter(|flow| flow.reach_id == id)
                .collect::<Vec<_>>();
            assert_eq!(before_flow, after_flow);
        }
        let mut sb = state(&before);
        let mut sa = sb.clone();
        let rb = apply_unit_stream_power_response(&before, &qb, &mut sb, 1e-4, 0.1).unwrap();
        let ra = apply_unit_stream_power_response(&after, &qa, &mut sa, 1e-4, 0.1).unwrap();
        assert_eq!(
            rb.exported_by_reach_km3[&A].to_bits(),
            ra.exported_by_reach_km3[&A].to_bits()
        );
        assert_eq!(
            rb.exported_by_reach_km3[&B].to_bits(),
            ra.exported_by_reach_km3[&B].to_bits()
        );
        assert_ne!(
            rb.exported_by_reach_km3[&C].to_bits(),
            ra.exported_by_reach_km3[&C].to_bits()
        );
        assert_ne!(
            rb.exported_by_reach_km3[&D].to_bits(),
            ra.exported_by_reach_km3[&D].to_bits()
        );
        for (index, segment) in before.segments().iter().enumerate() {
            if segment.reach_id == A || segment.reach_id == B {
                assert_eq!(sb[index], sa[index]);
            }
        }
    }

    #[test]
    fn topology_event_preserves_all_owned_bits_and_is_transactional() {
        let n = net(4., false);
        let s = state(&n);
        let mut f = C1RoutedFixture::new(n, s).unwrap();
        let seg = f.network().segments.clone();
        let bits = f
            .state()
            .iter()
            .map(|x| {
                (
                    x.mean_elevation_km.to_bits(),
                    x.channel_surface_elevation_km.to_bits(),
                )
            })
            .collect::<Vec<_>>();
        f.replace_receiver(B, Some(D)).unwrap();
        assert_eq!(f.network().segments, seg);
        assert_eq!(
            f.state()
                .iter()
                .map(|x| (
                    x.mean_elevation_km.to_bits(),
                    x.channel_surface_elevation_km.to_bits()
                ))
                .collect::<Vec<_>>(),
            bits
        );
        close(
            f.network().route().unwrap().outlets_km3_myr[&D],
            8.79,
            5e-14,
        );
        let before = f.clone();
        assert_eq!(f.replace_receiver(D, Some(B)), Err(C1NetworkError::Cycle));
        assert_eq!(f, before);
    }

    #[test]
    fn overlap_remap_preserves_three_moments_without_cross_reach_transfer() {
        let n8 = net(8., false);
        let s8 = state(&n8);
        let m8 = network_moments(&n8, &s8).unwrap();
        let n4 = net(4., true);
        let s4 = remap_c1_state_by_reach_overlap(&n8, &s8, &n4).unwrap();
        let m4 = network_moments(&n4, &s4).unwrap();
        let n2 = net(2., false);
        let s2 = remap_c1_state_by_reach_overlap(&n4, &s4, &n2).unwrap();
        let m2 = network_moments(&n2, &s2).unwrap();
        for (a, b) in [
            (
                m8.channel_elevation_volume_moment_km3,
                m4.channel_elevation_volume_moment_km3,
            ),
            (
                m8.interfluve_elevation_volume_moment_km3,
                m4.interfluve_elevation_volume_moment_km3,
            ),
            (
                m4.total_elevation_volume_moment_km3,
                m2.total_elevation_volume_moment_km3,
            ),
        ] {
            close(a, b, 3e-10)
        }
        for id in [A, B, C, D] {
            let src = network_moments_for(&n8, &s8, id);
            let dst = network_moments_for(&n2, &s2, id);
            close(
                src.channel_elevation_volume_moment_km3,
                dst.channel_elevation_volume_moment_km3,
                2e-12,
            );
            close(
                src.interfluve_elevation_volume_moment_km3,
                dst.interfluve_elevation_volume_moment_km3,
                2e-10,
            );
        }
    }
    fn network_moments_for(n: &C1ReachNetwork, s: &[C1CellState], id: ReachId) -> C1NetworkMoments {
        let mut m = C1NetworkMoments::default();
        for (i, g) in n
            .segments
            .iter()
            .enumerate()
            .filter(|(_, x)| x.reach_id == id)
        {
            let ac = g.geometry.channel_area_km2();
            let zi = s[i].interfluve_mean_elevation_km(g.geometry).unwrap();
            m.channel_elevation_volume_moment_km3 += ac * s[i].channel_surface_elevation_km;
            m.interfluve_elevation_volume_moment_km3 += (g.geometry.cell_area_km2 - ac) * zi;
            m.total_elevation_volume_moment_km3 +=
                g.geometry.cell_area_km2 * s[i].mean_elevation_km;
        }
        m
    }

    #[test]
    fn explicit_internal_transfer_scales_by_physical_length_and_cancels() {
        let n = net(4., false);
        let mut s = state(&n);
        let bits = s
            .iter()
            .map(|x| x.mean_elevation_km.to_bits())
            .collect::<Vec<_>>();
        let density = BTreeMap::from([(A, 0.002), (D, 0.003)]);
        let l = apply_internal_transfer_per_reach_length(&n, &mut s, &density).unwrap();
        close(
            l.channel_compartment_moment_change_km3,
            0.002 * 64. + 0.003 * 80.,
            1e-15,
        );
        assert_eq!(l.net_elevation_volume_moment_change_km3.to_bits(), 0);
        assert_eq!(
            bits,
            s.iter()
                .map(|x| x.mean_elevation_km.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn invalid_graph_geometry_and_overlap_fail_explicitly() {
        let mut x = specs(false);
        x.push(x[0]);
        assert_eq!(
            C1ReachNetwork::new(x, 4.).unwrap_err(),
            C1NetworkError::DuplicateReach(A)
        );
        let mut x = specs(false);
        x[0].downstream = Some(ReachId(99));
        assert_eq!(
            C1ReachNetwork::new(x, 4.).unwrap_err(),
            C1NetworkError::UnknownReceiver {
                reach: A,
                receiver: ReachId(99)
            }
        );
        let mut x = specs(false);
        x[2].downstream = Some(A);
        assert_eq!(
            C1ReachNetwork::new(x, 4.).unwrap_err(),
            C1NetworkError::Cycle
        );
        let mut x = specs(false);
        x[0].length_km = 0.;
        assert!(matches!(
            C1ReachNetwork::new(x, 4.),
            Err(C1NetworkError::InvalidScalar {
                field: "length_km",
                ..
            })
        ));
        let a = net(8., false);
        let s = state(&a);
        let mut x = specs(false);
        x[0].length_km = 63.;
        let b = C1ReachNetwork::new(x, 4.).unwrap();
        assert_eq!(
            remap_c1_state_by_reach_overlap(&a, &s, &b).unwrap_err(),
            C1NetworkError::IncompatibleReachGeometry(A)
        );
        let mut broken = a.clone();
        broken.segments[3].s0_km = 25.0;
        let d = net(4., false);
        assert!(matches!(
            remap_c1_state_by_reach_overlap(&broken, &state(&broken), &d),
            Err(C1NetworkError::IncompleteOverlap { reach: A, .. })
        ));

        let only_d = C1ReachNetwork::new(vec![specs(false)[3]], 4.).unwrap();
        assert!(matches!(
            remap_c1_state_by_reach_overlap(&a, &s, &only_d),
            Err(C1NetworkError::IncompleteOverlap { reach: A, .. })
        ));
    }

    #[test]
    fn response_and_transfer_fail_without_mutation() {
        let n = net(4., false);
        let mut s = state(&n);
        let before = s.clone();
        let mut bad = n.route().unwrap();
        bad.segment_flows[3].reach_id = D;
        assert!(matches!(
            apply_unit_stream_power_response(&n, &bad, &mut s, 1e-4, 0.1),
            Err(C1NetworkError::RoutingIdentity { segment: 3 })
        ));
        assert_eq!(s, before);
        let density = BTreeMap::from([(A, -1.)]);
        assert!(apply_internal_transfer_per_reach_length(&n, &mut s, &density).is_err());
        assert_eq!(s, before);
    }

    #[test]
    fn construction_routing_remap_and_response_are_bit_deterministic() {
        let run = || {
            let n8 = C1ReachNetwork::new(specs(false).into_iter().rev().collect(), 8.).unwrap();
            let s8 = state(&n8);
            let q = n8.route().unwrap();
            let n2 = net(2., true);
            let mut s2 = remap_c1_state_by_reach_overlap(&n8, &s8, &n2).unwrap();
            let r = apply_unit_stream_power_response(&n2, &n2.route().unwrap(), &mut s2, 1e-4, 0.1)
                .unwrap();
            (n8, q, s2, r)
        };
        assert_eq!(run(), run());
    }
}
