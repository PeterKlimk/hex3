//! Manufactured channel-promotion and lineage fixture.
//!
//! This is the deliberately narrow `M0` rung between the prescribed routed-C1
//! fixture and terrain-derived channel extraction. Physical channel observations
//! (including discharge, slope, resistance, support, and receiver candidates)
//! are inputs. The fixture tests hysteretic promotion, correspondence, stable
//! lineage, transactional capture, and conservative attachment of existing C1
//! state. It does not extract a thalweg from [`super::FaceFlowCache`], infer
//! width, or feed a channel back into terrain or water routing.

use std::{collections::BTreeMap, fmt};

use serde::{Deserialize, Serialize};

use super::{
    network_moments, remap_c1_state_by_reach_overlap, C1CellState, C1NetworkError,
    C1NetworkMoments, C1ReachNetwork, C1ReachSpec, C1RoutedFixture, ReachId,
};

/// Ephemeral identity supplied by one observation snapshot.
///
/// Rebuilds may reorder or replace these IDs. They are references within one
/// snapshot only and are never promoted to persistent reach identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ChannelCandidateId(pub u32);

/// One prescribed physical reach observation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChannelObservation {
    pub candidate_id: ChannelCandidateId,
    /// Physical upstream anchor used by the bounded correspondence fixture.
    pub source_anchor_km: f64,
    /// Physical downstream anchor. Together with the source anchor this is a
    /// reduced stand-in for centerline overlap, not a permanent identity.
    pub mouth_anchor_km: f64,
    pub length_km: f64,
    pub channel_width_km: f64,
    pub represented_swath_width_km: f64,
    pub grade: f64,
    pub headwater_discharge_km3_myr: f64,
    pub lateral_supply_km3_myr_per_km: f64,
    /// Continuum-derived specific discharge evidence in physical units.
    pub specific_discharge_km2_myr: f64,
    /// Dimensionless resistance closure for this manufactured observation.
    pub resistance: f64,
    pub downstream: Option<ChannelCandidateId>,
}

/// Frozen reduced channel-initiation and correspondence policy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChannelPromotionPolicy {
    pub reference_specific_discharge_km2_myr: f64,
    pub reference_grade: f64,
    pub discharge_exponent: f64,
    pub grade_exponent: f64,
    pub initiate_at: f64,
    pub retain_until_below: f64,
    pub maximum_anchor_shift_km: f64,
}

impl ChannelPromotionPolicy {
    pub fn initiation_evidence(self, observation: ChannelObservation) -> f64 {
        (observation.specific_discharge_km2_myr / self.reference_specific_discharge_km2_myr)
            .powf(self.discharge_exponent)
            * (observation.grade / self.reference_grade).powf(self.grade_exponent)
            / observation.resistance
    }
}

/// Snapshot-only reach assignment, retained as the no-memory control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotReach {
    pub id: ReachId,
    pub candidate_id: ChannelCandidateId,
}

/// Build-order IDs intentionally model current snapshot semantics. The same
/// physical reach can receive another ID when observation order changes.
pub fn snapshot_reaches(
    observations: &[ChannelObservation],
    policy: ChannelPromotionPolicy,
) -> Result<Vec<SnapshotReach>, ChannelOwnershipError> {
    validate_observations(observations, policy)?;
    Ok(observations
        .iter()
        .filter(|observation| policy.initiation_evidence(**observation) >= policy.initiate_at)
        .enumerate()
        .map(|(index, observation)| SnapshotReach {
            id: ReachId(index as u32),
            candidate_id: observation.candidate_id,
        })
        .collect())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelLineageEvent {
    Initiated {
        reach: ReachId,
    },
    Abandoned {
        reach: ReachId,
    },
    Captured {
        reach: ReachId,
        old_receiver: Option<ReachId>,
        new_receiver: Option<ReachId>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelCorrespondenceAudit {
    pub compared_anchor_pairs: usize,
    pub retained_reaches: usize,
    pub initiated_reaches: usize,
    pub abandoned_reaches: usize,
    pub events: Vec<ChannelLineageEvent>,
    pub moments_before: C1NetworkMoments,
    pub moments_after: C1NetworkMoments,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct OwnedReach {
    id: ReachId,
    observation: ChannelObservation,
    downstream: Option<ReachId>,
}

/// Persistent lineage plus the already validated routed-C1 state carrier.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PersistentChannelFixture {
    policy: ChannelPromotionPolicy,
    nominal_spacing_km: f64,
    reaches: Vec<OwnedReach>,
    next_id: u32,
    routed: C1RoutedFixture,
}

impl PersistentChannelFixture {
    pub fn new(
        observations: &[ChannelObservation],
        policy: ChannelPromotionPolicy,
        nominal_spacing_km: f64,
    ) -> Result<(Self, ChannelCorrespondenceAudit), ChannelOwnershipError> {
        validate_observations(observations, policy)?;
        let mut promoted: Vec<_> = observations
            .iter()
            .copied()
            .filter(|observation| policy.initiation_evidence(*observation) >= policy.initiate_at)
            .collect();
        promoted.sort_by(observation_physical_order);
        validate_unique_source_anchors(&promoted, policy)?;

        let mut candidate_to_reach = BTreeMap::new();
        let mut reaches = Vec::with_capacity(promoted.len());
        let mut events = Vec::with_capacity(promoted.len());
        for (index, observation) in promoted.iter().copied().enumerate() {
            let id = ReachId(index as u32);
            candidate_to_reach.insert(observation.candidate_id, id);
            reaches.push(OwnedReach {
                id,
                observation,
                downstream: None,
            });
            events.push(ChannelLineageEvent::Initiated { reach: id });
        }
        assign_receivers(&mut reaches, &candidate_to_reach)?;
        let network = build_network(&reaches, nominal_spacing_km)?;
        let state = initial_state(&network);
        let moments = network_moments(&network, &state)?;
        let routed = C1RoutedFixture::new(network, state)?;
        let audit = ChannelCorrespondenceAudit {
            compared_anchor_pairs: 0,
            retained_reaches: 0,
            initiated_reaches: reaches.len(),
            abandoned_reaches: 0,
            events,
            moments_before: moments,
            moments_after: moments,
        };
        Ok((
            Self {
                policy,
                nominal_spacing_km,
                next_id: reaches.len() as u32,
                reaches,
                routed,
            },
            audit,
        ))
    }

    pub fn network(&self) -> &C1ReachNetwork {
        self.routed.network()
    }

    pub fn state(&self) -> &[C1CellState] {
        self.routed.state()
    }

    pub fn reach_for_source_anchor(&self, source_anchor_km: f64) -> Option<ReachId> {
        let mut matches = self.reaches.iter().filter(|reach| {
            (reach.observation.source_anchor_km - source_anchor_km).abs()
                <= self.policy.maximum_anchor_shift_km
        });
        let id = matches.next()?.id;
        matches.next().is_none().then_some(id)
    }

    /// Reconcile one observation snapshot transactionally.
    ///
    /// M0 deliberately supports unchanged membership with jitter and receiver
    /// replacement. Birth/retirement is detected and reported as an unsupported
    /// state-ledger rung rather than silently discarding or inventing C1 state.
    pub fn reconcile(
        &mut self,
        observations: &[ChannelObservation],
    ) -> Result<ChannelCorrespondenceAudit, ChannelOwnershipError> {
        validate_observations(observations, self.policy)?;
        let moments_before = network_moments(self.routed.network(), self.routed.state())?;

        let mut old = self.reaches.clone();
        old.sort_by(owned_physical_order);
        let mut eligible: Vec<_> = observations
            .iter()
            .copied()
            .filter(|observation| {
                self.policy.initiation_evidence(*observation) >= self.policy.retain_until_below
            })
            .collect();
        eligible.sort_by(observation_physical_order);
        validate_unique_source_anchors(&eligible, self.policy)?;

        let mut compared_anchor_pairs = 0usize;
        let mut observation_to_reach = BTreeMap::new();
        let mut proposed = Vec::with_capacity(eligible.len());
        let mut events = Vec::new();
        let mut next_id = self.next_id;
        let mut old_index = 0usize;
        let mut new_index = 0usize;
        while old_index < old.len() && new_index < eligible.len() {
            compared_anchor_pairs += 1;
            let previous = old[old_index];
            let observation = eligible[new_index];
            let source_delta = observation.source_anchor_km - previous.observation.source_anchor_km;
            let mouth_delta = observation.mouth_anchor_km - previous.observation.mouth_anchor_km;
            if source_delta.abs() <= self.policy.maximum_anchor_shift_km
                && mouth_delta.abs() <= self.policy.maximum_anchor_shift_km
            {
                observation_to_reach.insert(observation.candidate_id, previous.id);
                proposed.push(OwnedReach {
                    id: previous.id,
                    observation,
                    downstream: None,
                });
                old_index += 1;
                new_index += 1;
            } else if observation.source_anchor_km
                < previous.observation.source_anchor_km - self.policy.maximum_anchor_shift_km
            {
                if self.policy.initiation_evidence(observation) >= self.policy.initiate_at {
                    let id = ReachId(next_id);
                    next_id = next_id
                        .checked_add(1)
                        .ok_or(ChannelOwnershipError::Invalid("reach identity overflow"))?;
                    observation_to_reach.insert(observation.candidate_id, id);
                    proposed.push(OwnedReach {
                        id,
                        observation,
                        downstream: None,
                    });
                    events.push(ChannelLineageEvent::Initiated { reach: id });
                }
                new_index += 1;
            } else {
                events.push(ChannelLineageEvent::Abandoned { reach: previous.id });
                old_index += 1;
            }
        }
        for previous in &old[old_index..] {
            events.push(ChannelLineageEvent::Abandoned { reach: previous.id });
        }
        for observation in eligible[new_index..].iter().copied() {
            if self.policy.initiation_evidence(observation) >= self.policy.initiate_at {
                let id = ReachId(next_id);
                next_id = next_id
                    .checked_add(1)
                    .ok_or(ChannelOwnershipError::Invalid("reach identity overflow"))?;
                observation_to_reach.insert(observation.candidate_id, id);
                proposed.push(OwnedReach {
                    id,
                    observation,
                    downstream: None,
                });
                events.push(ChannelLineageEvent::Initiated { reach: id });
            }
        }

        let initiated = events
            .iter()
            .filter(|event| matches!(event, ChannelLineageEvent::Initiated { .. }))
            .count();
        let abandoned = events
            .iter()
            .filter(|event| matches!(event, ChannelLineageEvent::Abandoned { .. }))
            .count();
        if initiated != 0 || abandoned != 0 {
            return Err(ChannelOwnershipError::MembershipStateLedgerRequired {
                initiated,
                abandoned,
            });
        }

        proposed.sort_by_key(|reach| reach.id);
        assign_receivers(&mut proposed, &observation_to_reach)?;
        let previous_by_id: BTreeMap<_, _> = self
            .reaches
            .iter()
            .map(|reach| (reach.id, reach.downstream))
            .collect();
        for reach in &proposed {
            let old_receiver = previous_by_id[&reach.id];
            if old_receiver != reach.downstream {
                events.push(ChannelLineageEvent::Captured {
                    reach: reach.id,
                    old_receiver,
                    new_receiver: reach.downstream,
                });
            }
        }

        // Construct and validate every candidate object before mutating `self`.
        let network = build_network(&proposed, self.nominal_spacing_km)?;
        let remapped =
            remap_c1_state_by_reach_overlap(self.routed.network(), self.routed.state(), &network)?;
        let moments_after = network_moments(&network, &remapped)?;
        let routed = C1RoutedFixture::new(network, remapped)?;

        self.reaches = proposed;
        self.routed = routed;
        self.next_id = next_id;
        Ok(ChannelCorrespondenceAudit {
            compared_anchor_pairs,
            retained_reaches: self.reaches.len(),
            initiated_reaches: initiated,
            abandoned_reaches: abandoned,
            events,
            moments_before,
            moments_after,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelOwnershipError {
    Invalid(&'static str),
    DuplicateCandidate(ChannelCandidateId),
    AmbiguousObservationPair {
        first: ChannelCandidateId,
        second: ChannelCandidateId,
    },
    UnknownReceiver {
        candidate: ChannelCandidateId,
        receiver: ChannelCandidateId,
    },
    MembershipStateLedgerRequired {
        initiated: usize,
        abandoned: usize,
    },
    C1(C1NetworkError),
}

impl fmt::Display for ChannelOwnershipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(field) => write!(f, "invalid {field}"),
            Self::DuplicateCandidate(id) => write!(f, "duplicate candidate {id:?}"),
            Self::AmbiguousObservationPair { first, second } => write!(
                f,
                "candidates {first:?} and {second:?} do not have uniquely matchable source anchors"
            ),
            Self::UnknownReceiver {
                candidate,
                receiver,
            } => write!(f, "candidate {candidate:?} has unknown receiver {receiver:?}"),
            Self::MembershipStateLedgerRequired {
                initiated,
                abandoned,
            } => write!(
                f,
                "M0 cannot transfer C1 state across membership change ({initiated} initiated, {abandoned} abandoned)"
            ),
            Self::C1(source) => source.fmt(f),
        }
    }
}

impl std::error::Error for ChannelOwnershipError {}

impl From<C1NetworkError> for ChannelOwnershipError {
    fn from(value: C1NetworkError) -> Self {
        Self::C1(value)
    }
}

fn validate_observations(
    observations: &[ChannelObservation],
    policy: ChannelPromotionPolicy,
) -> Result<(), ChannelOwnershipError> {
    let policy_scalars = [
        policy.reference_specific_discharge_km2_myr,
        policy.reference_grade,
        policy.discharge_exponent,
        policy.grade_exponent,
        policy.initiate_at,
        policy.retain_until_below,
        policy.maximum_anchor_shift_km,
    ];
    if policy_scalars.iter().any(|value| !value.is_finite())
        || policy.reference_specific_discharge_km2_myr <= 0.0
        || policy.reference_grade <= 0.0
        || policy.discharge_exponent < 0.0
        || policy.grade_exponent < 0.0
        || policy.initiate_at <= policy.retain_until_below
        || policy.retain_until_below < 0.0
        || policy.maximum_anchor_shift_km < 0.0
    {
        return Err(ChannelOwnershipError::Invalid("promotion policy"));
    }
    let mut ids = BTreeMap::new();
    for observation in observations {
        if ids.insert(observation.candidate_id, ()).is_some() {
            return Err(ChannelOwnershipError::DuplicateCandidate(
                observation.candidate_id,
            ));
        }
        let scalars = [
            observation.source_anchor_km,
            observation.mouth_anchor_km,
            observation.length_km,
            observation.channel_width_km,
            observation.represented_swath_width_km,
            observation.grade,
            observation.headwater_discharge_km3_myr,
            observation.lateral_supply_km3_myr_per_km,
            observation.specific_discharge_km2_myr,
            observation.resistance,
        ];
        if scalars.iter().any(|value| !value.is_finite())
            || observation.length_km <= 0.0
            || observation.channel_width_km <= 0.0
            || observation.represented_swath_width_km <= observation.channel_width_km
            || observation.grade < 0.0
            || observation.headwater_discharge_km3_myr < 0.0
            || observation.lateral_supply_km3_myr_per_km < 0.0
            || observation.specific_discharge_km2_myr < 0.0
            || observation.resistance <= 0.0
        {
            return Err(ChannelOwnershipError::Invalid("channel observation"));
        }
    }
    for observation in observations {
        if let Some(receiver) = observation.downstream {
            if !ids.contains_key(&receiver) {
                return Err(ChannelOwnershipError::UnknownReceiver {
                    candidate: observation.candidate_id,
                    receiver,
                });
            }
        }
    }
    Ok(())
}

fn validate_unique_source_anchors(
    physical_order: &[ChannelObservation],
    policy: ChannelPromotionPolicy,
) -> Result<(), ChannelOwnershipError> {
    // The caller supplies only the promoted/retained set in physical order.
    // Source-anchor neighborhoods may not overlap. This is intentionally
    // stricter than a future centerline-overlap matcher, but makes M0 identity
    // correspondence unique and checkable after one deterministic sort.
    for pair in physical_order.windows(2) {
        if pair[1].source_anchor_km - pair[0].source_anchor_km
            <= 2.0 * policy.maximum_anchor_shift_km
        {
            return Err(ChannelOwnershipError::AmbiguousObservationPair {
                first: pair[0].candidate_id,
                second: pair[1].candidate_id,
            });
        }
    }
    Ok(())
}

fn observation_physical_order(
    a: &ChannelObservation,
    b: &ChannelObservation,
) -> std::cmp::Ordering {
    a.source_anchor_km
        .total_cmp(&b.source_anchor_km)
        .then_with(|| a.mouth_anchor_km.total_cmp(&b.mouth_anchor_km))
        .then_with(|| a.candidate_id.cmp(&b.candidate_id))
}

fn owned_physical_order(a: &OwnedReach, b: &OwnedReach) -> std::cmp::Ordering {
    observation_physical_order(&a.observation, &b.observation).then_with(|| a.id.cmp(&b.id))
}

fn assign_receivers(
    reaches: &mut [OwnedReach],
    candidates: &BTreeMap<ChannelCandidateId, ReachId>,
) -> Result<(), ChannelOwnershipError> {
    for reach in reaches {
        reach.downstream = match reach.observation.downstream {
            Some(receiver) => Some(*candidates.get(&receiver).ok_or(
                ChannelOwnershipError::UnknownReceiver {
                    candidate: reach.observation.candidate_id,
                    receiver,
                },
            )?),
            None => None,
        };
    }
    Ok(())
}

fn build_network(
    reaches: &[OwnedReach],
    nominal_spacing_km: f64,
) -> Result<C1ReachNetwork, ChannelOwnershipError> {
    let specs = reaches
        .iter()
        .map(|reach| C1ReachSpec {
            id: reach.id,
            length_km: reach.observation.length_km,
            channel_width_km: reach.observation.channel_width_km,
            represented_swath_width_km: reach.observation.represented_swath_width_km,
            grade: reach.observation.grade,
            headwater_discharge_km3_myr: reach.observation.headwater_discharge_km3_myr,
            lateral_supply_km3_myr_per_km: reach.observation.lateral_supply_km3_myr_per_km,
            downstream: reach.downstream,
        })
        .collect();
    Ok(C1ReachNetwork::new(specs, nominal_spacing_km)?)
}

fn initial_state(network: &C1ReachNetwork) -> Vec<C1CellState> {
    network
        .segments()
        .iter()
        .map(|segment| {
            let channel = 0.4 + 0.001 * f64::from(segment.reach_id.0) + 1.0e-5 * segment.s0_km;
            let interfluve = channel + 0.6;
            let fraction = segment.geometry.channel_fraction();
            C1CellState {
                mean_elevation_km: fraction * channel + (1.0 - fraction) * interfluve,
                channel_surface_elevation_km: channel,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const POLICY: ChannelPromotionPolicy = ChannelPromotionPolicy {
        reference_specific_discharge_km2_myr: 1.0,
        reference_grade: 0.01,
        discharge_exponent: 1.0,
        grade_exponent: 1.0,
        initiate_at: 1.0,
        retain_until_below: 0.75,
        maximum_anchor_shift_km: 0.5,
    };

    fn observation(
        candidate: u32,
        source: f64,
        mouth: f64,
        length: f64,
        width: f64,
        evidence: f64,
        downstream: Option<u32>,
    ) -> ChannelObservation {
        ChannelObservation {
            candidate_id: ChannelCandidateId(candidate),
            source_anchor_km: source,
            mouth_anchor_km: mouth,
            length_km: length,
            channel_width_km: width,
            represented_swath_width_km: 12.0,
            grade: 0.01,
            headwater_discharge_km3_myr: 1.0 + 0.1 * f64::from(candidate % 4),
            lateral_supply_km3_myr_per_km: 0.02,
            specific_discharge_km2_myr: evidence,
            resistance: 1.0,
            downstream: downstream.map(ChannelCandidateId),
        }
    }

    fn stable() -> Vec<ChannelObservation> {
        vec![
            observation(10, -30.0, -10.0, 64.0, 0.12, 1.2, Some(30)),
            observation(20, -20.0, -10.0, 48.0, 0.10, 1.2, Some(30)),
            observation(30, -10.0, 0.0, 96.0, 0.22, 1.3, None),
            observation(40, 20.0, 30.0, 80.0, 0.18, 1.3, None),
        ]
    }

    fn threshold_dip() -> Vec<ChannelObservation> {
        // New candidate IDs and reversed build order are intentional. B dips
        // below initiation but remains above the registered retention margin.
        vec![
            observation(140, 20.2, 30.2, 80.0, 0.18, 1.3, None),
            observation(130, -10.2, 0.2, 96.0, 0.22, 1.3, None),
            observation(120, -19.8, -10.2, 48.0, 0.10, 0.9, Some(130)),
            observation(110, -30.2, -9.8, 64.0, 0.12, 1.2, Some(130)),
        ]
    }

    fn capture() -> Vec<ChannelObservation> {
        vec![
            observation(210, -30.0, -10.0, 64.0, 0.12, 1.2, Some(230)),
            observation(220, -20.0, -10.0, 48.0, 0.10, 1.2, Some(240)),
            observation(230, -10.0, 0.0, 96.0, 0.22, 1.3, None),
            observation(240, 20.0, 30.0, 80.0, 0.18, 1.3, None),
        ]
    }

    fn assert_moments_equal(a: C1NetworkMoments, b: C1NetworkMoments) {
        assert!(
            (a.channel_elevation_volume_moment_km3 - b.channel_elevation_volume_moment_km3).abs()
                < 1.0e-11
        );
        assert!(
            (a.interfluve_elevation_volume_moment_km3 - b.interfluve_elevation_volume_moment_km3)
                .abs()
                < 1.0e-9
        );
        assert!(
            (a.total_elevation_volume_moment_km3 - b.total_elevation_volume_moment_km3).abs()
                < 1.0e-9
        );
    }

    #[test]
    fn hysteresis_retains_a_marginal_reach_across_snapshot_rebuild() {
        for spacing in [8.0, 4.0, 2.0] {
            let snapshot0 = snapshot_reaches(&stable(), POLICY).unwrap();
            let snapshot1 = snapshot_reaches(&threshold_dip(), POLICY).unwrap();
            assert_eq!(snapshot0.len(), 4);
            assert_eq!(snapshot1.len(), 3);

            let (mut fixture, _) =
                PersistentChannelFixture::new(&stable(), POLICY, spacing).unwrap();
            let ids_before: Vec<_> = [-30.0, -20.0, -10.0, 20.0]
                .into_iter()
                .map(|anchor| fixture.reach_for_source_anchor(anchor).unwrap())
                .collect();
            let audit = fixture.reconcile(&threshold_dip()).unwrap();
            let ids_after: Vec<_> = [-30.0, -20.0, -10.0, 20.0]
                .into_iter()
                .map(|anchor| fixture.reach_for_source_anchor(anchor).unwrap())
                .collect();
            assert_eq!(ids_before, ids_after);
            assert!(audit.events.is_empty());
            assert_eq!(audit.retained_reaches, 4);
            assert!(audit.compared_anchor_pairs <= 4);
            assert_moments_equal(audit.moments_before, audit.moments_after);
        }
    }

    #[test]
    fn one_capture_is_transactional_and_preserves_state() {
        for spacing in [8.0, 4.0, 2.0] {
            let (mut fixture, _) =
                PersistentChannelFixture::new(&stable(), POLICY, spacing).unwrap();
            fixture.reconcile(&threshold_dip()).unwrap();
            let state_before = fixture.state().to_vec();
            let audit = fixture.reconcile(&capture()).unwrap();
            let capture_events: Vec<_> = audit
                .events
                .iter()
                .filter(|event| matches!(event, ChannelLineageEvent::Captured { .. }))
                .collect();
            assert_eq!(capture_events.len(), 1);
            assert_eq!(fixture.state(), state_before);
            assert_moments_equal(audit.moments_before, audit.moments_after);

            let before_network = fixture.network().clone();
            let before_state = fixture.state().to_vec();
            let mut cycle = capture();
            cycle[2].downstream = Some(ChannelCandidateId(210));
            assert!(matches!(
                fixture.reconcile(&cycle),
                Err(ChannelOwnershipError::C1(C1NetworkError::Cycle))
            ));
            assert_eq!(fixture.network(), &before_network);
            assert_eq!(fixture.state(), before_state);
        }
    }

    #[test]
    fn physical_support_is_exact_because_it_is_prescribed_not_extracted() {
        for spacing in [8.0, 4.0, 2.0] {
            let (fixture, _) = PersistentChannelFixture::new(&stable(), POLICY, spacing).unwrap();
            let length: f64 = fixture
                .network()
                .reaches()
                .iter()
                .map(|reach| reach.length_km)
                .sum();
            let area: f64 = fixture
                .network()
                .reaches()
                .iter()
                .map(|reach| reach.length_km * reach.channel_width_km)
                .sum();
            assert_eq!(length, 288.0);
            assert_eq!(area, 48.0);
        }
    }

    #[test]
    fn ambiguous_anchor_correspondence_is_rejected() {
        let mut ambiguous = stable();
        ambiguous[1].source_anchor_km = -29.5;
        ambiguous[1].mouth_anchor_km = -9.5;
        assert_eq!(snapshot_reaches(&ambiguous, POLICY).unwrap().len(), 4);
        assert!(matches!(
            PersistentChannelFixture::new(&ambiguous, POLICY, 4.0),
            Err(ChannelOwnershipError::AmbiguousObservationPair { .. })
        ));
    }

    #[test]
    fn membership_change_stops_before_discarding_attached_state() {
        let (mut fixture, _) = PersistentChannelFixture::new(&stable(), POLICY, 4.0).unwrap();
        let before_network = fixture.network().clone();
        let before_state = fixture.state().to_vec();
        let mut missing = stable();
        missing[1].specific_discharge_km2_myr = 0.5;
        assert!(matches!(
            fixture.reconcile(&missing),
            Err(ChannelOwnershipError::MembershipStateLedgerRequired {
                initiated: 0,
                abandoned: 1
            })
        ));
        assert_eq!(fixture.network(), &before_network);
        assert_eq!(fixture.state(), before_state);
    }

    #[test]
    fn repeated_correspondence_and_events_are_bit_deterministic() {
        let (mut a, init_a) = PersistentChannelFixture::new(&stable(), POLICY, 4.0).unwrap();
        let (mut b, init_b) = PersistentChannelFixture::new(&stable(), POLICY, 4.0).unwrap();
        assert_eq!(init_a, init_b);
        let dip_a = a.reconcile(&threshold_dip()).unwrap();
        let dip_b = b.reconcile(&threshold_dip()).unwrap();
        assert_eq!(dip_a, dip_b);
        let capture_a = a.reconcile(&capture()).unwrap();
        let capture_b = b.reconcile(&capture()).unwrap();
        assert_eq!(capture_a, capture_b);
        assert_eq!(a, b);
    }
}
