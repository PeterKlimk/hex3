//! Source-first target identity for Structural Mountain experiments.
//!
//! A target is an exact-link connected component of continental-relief-capable
//! finite parents inside one generated plate-pair/history-episode system.
//! Oceanic continuations remain source context but do not make a terrestrial
//! mountain belt global. Selection never reads elevation, drainage, rendering,
//! or legacy feature response.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    BoundaryEdgeId, CrustType, StructuralMountainGraph, StructuralReadiness, StructuralRegime,
    StructuralSegment,
};

#[derive(Clone, Debug, PartialEq)]
pub struct StructuralSourceBelt {
    /// Stable identity: the minimum finite-parent ID in the component.
    pub id: BoundaryEdgeId,
    pub episode_id: usize,
    pub plate_pair: [usize; 2],
    pub segment_ids: Vec<BoundaryEdgeId>,
    pub source_edges: Vec<BoundaryEdgeId>,
    pub length_km: f32,
    /// Sum of the compiler's shortening-area opportunity. This ranks source
    /// extent/forcing; it is not predicted uplift, work, or terrain height.
    pub declared_opportunity_km2: f64,
    pub collision_segment_count: usize,
    pub subduction_segment_count: usize,
    /// True when collision involves continental crust or a subduction segment
    /// has continental crust on its receiving plate.
    pub continental_relief_capable: bool,
    pub readiness: StructuralReadiness,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StructuralSourceTargetError {
    DuplicateSegment(BoundaryEdgeId),
    MissingLinkedSegment(BoundaryEdgeId),
    InvalidSegment(BoundaryEdgeId),
    NoContinentalReliefCapableBelt,
}

impl std::fmt::Display for StructuralSourceTargetError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for StructuralSourceTargetError {}

/// Enumerate exact-link components constrained to one plate pair, one history
/// episode, and a shared ability to build continental relief. Incapable
/// segments remain explicit singleton source systems but cannot carry
/// connectivity between terrestrial belts. Output order is stable identity
/// order, not desirability order.
pub fn catalog_structural_source_belts(
    graph: &StructuralMountainGraph,
) -> Result<Vec<StructuralSourceBelt>, StructuralSourceTargetError> {
    let mut by_id = BTreeMap::new();
    for segment in &graph.segments {
        if by_id.insert(segment.id, segment).is_some() {
            return Err(StructuralSourceTargetError::DuplicateSegment(segment.id));
        }
        validate_segment(segment)?;
    }

    let mut adjacency = BTreeMap::<BoundaryEdgeId, BTreeSet<BoundaryEdgeId>>::new();
    for &id in by_id.keys() {
        adjacency.entry(id).or_default();
    }
    for link in &graph.links {
        let left = by_id.get(&link.segments[0]).ok_or(
            StructuralSourceTargetError::MissingLinkedSegment(link.segments[0]),
        )?;
        let right = by_id.get(&link.segments[1]).ok_or(
            StructuralSourceTargetError::MissingLinkedSegment(link.segments[1]),
        )?;
        // A geometric junction is not enough to merge different generated
        // plate systems or history episodes into one mountain target.
        if left.episode_id == right.episode_id
            && left.plate_pair == right.plate_pair
            && segment_can_build_continental_relief(left)
            && segment_can_build_continental_relief(right)
        {
            adjacency
                .get_mut(&left.id)
                .expect("segment inserted")
                .insert(right.id);
            adjacency
                .get_mut(&right.id)
                .expect("segment inserted")
                .insert(left.id);
        }
    }

    let mut unseen: BTreeSet<_> = by_id.keys().copied().collect();
    let mut belts = Vec::new();
    while let Some(&start) = unseen.first() {
        unseen.remove(&start);
        let mut stack = vec![start];
        let mut segment_ids = Vec::new();
        while let Some(current) = stack.pop() {
            segment_ids.push(current);
            let next: Vec<_> = adjacency[&current].iter().copied().collect();
            for candidate in next.into_iter().rev() {
                if unseen.remove(&candidate) {
                    stack.push(candidate);
                }
            }
        }
        segment_ids.sort();
        let first = by_id[&segment_ids[0]];
        let mut source_edges = BTreeSet::new();
        let mut length_km = 0.0;
        let mut opportunity_km2 = 0.0;
        let mut collisions = 0;
        let mut subductions = 0;
        let mut continental_relief_capable = false;
        for id in &segment_ids {
            let segment = by_id[id];
            debug_assert_eq!(segment.episode_id, first.episode_id);
            debug_assert_eq!(segment.plate_pair, first.plate_pair);
            source_edges.extend(segment.source_edges.iter().copied());
            length_km += segment.length_km;
            opportunity_km2 += segment.declared_opportunity_km2;
            match segment.regime {
                StructuralRegime::Collision => collisions += 1,
                StructuralRegime::Subduction => subductions += 1,
            }
            continental_relief_capable |= segment_can_build_continental_relief(segment);
        }
        belts.push(StructuralSourceBelt {
            id: segment_ids[0],
            episode_id: first.episode_id,
            plate_pair: first.plate_pair,
            source_edges: source_edges.into_iter().collect(),
            length_km,
            declared_opportunity_km2: opportunity_km2,
            collision_segment_count: collisions,
            subduction_segment_count: subductions,
            continental_relief_capable,
            readiness: if segment_ids.len() == 1 {
                StructuralReadiness::FiniteParentOnly
            } else {
                StructuralReadiness::CausallySegmented
            },
            segment_ids,
        });
    }
    Ok(belts)
}

/// Select the source system with the greatest declared shortening-area
/// opportunity among systems capable of continental relief. Length and stable
/// identity break exact opportunity ties. No terrain observation participates.
pub fn select_primary_structural_source_belt(
    belts: &[StructuralSourceBelt],
) -> Result<&StructuralSourceBelt, StructuralSourceTargetError> {
    ranked_continental_source_belts(belts)
        .into_iter()
        .next()
        .ok_or(StructuralSourceTargetError::NoContinentalReliefCapableBelt)
}

/// Eligible systems in the exact order used for source-first selection.
pub fn ranked_continental_source_belts(
    belts: &[StructuralSourceBelt],
) -> Vec<&StructuralSourceBelt> {
    let mut ranked: Vec<_> = belts
        .iter()
        .filter(|belt| belt.continental_relief_capable)
        .collect();
    ranked.sort_by(|left, right| {
        right
            .declared_opportunity_km2
            .total_cmp(&left.declared_opportunity_km2)
            .then_with(|| right.length_km.total_cmp(&left.length_km))
            .then_with(|| left.id.cmp(&right.id))
    });
    ranked
}

fn validate_segment(segment: &StructuralSegment) -> Result<(), StructuralSourceTargetError> {
    if segment.source_edges.is_empty()
        || !segment.length_km.is_finite()
        || segment.length_km <= 0.0
        || !segment.declared_opportunity_km2.is_finite()
        || segment.declared_opportunity_km2 <= 0.0
        || segment.plate_pair[0] >= segment.plate_pair[1]
    {
        return Err(StructuralSourceTargetError::InvalidSegment(segment.id));
    }
    Ok(())
}

fn segment_can_build_continental_relief(segment: &StructuralSegment) -> bool {
    match segment.regime {
        StructuralRegime::Collision => segment
            .crust_on_plate_pair
            .iter()
            .all(|crust| *crust == CrustType::Continental),
        StructuralRegime::Subduction => segment.receiving_plate.is_some_and(|receiving| {
            segment
                .plate_pair
                .iter()
                .position(|plate| *plate == receiving)
                .is_some_and(|side| segment.crust_on_plate_pair[side] == CrustType::Continental)
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{
        OpportunityLedger, StructuralLink, StructuralLinkKind, StructuralMountainGraph,
        StructuralNode, StructuralSegment,
    };

    fn segment(
        id: usize,
        episode: usize,
        plate_pair: [usize; 2],
        regime: StructuralRegime,
        receiving: Option<usize>,
        crust: [CrustType; 2],
        opportunity: f64,
    ) -> StructuralSegment {
        let edge = BoundaryEdgeId::new(id, id + 100);
        StructuralSegment {
            id: edge,
            source_edges: vec![edge],
            vertices_in_order: vec![id as u32, id as u32 + 1],
            length_km: 100.0,
            episode_id: episode,
            regime,
            plate_pair,
            crust_on_plate_pair: crust,
            subducting_plate: receiving.map(|plate| {
                if plate == plate_pair[0] {
                    plate_pair[1]
                } else {
                    plate_pair[0]
                }
            }),
            receiving_plate: receiving,
            declared_opportunity_km2: opportunity,
            along_strike_taper: vec![1.0],
            compiled_opportunity_km2: vec![opportunity],
        }
    }

    fn graph(
        segments: Vec<StructuralSegment>,
        links: Vec<StructuralLink>,
    ) -> StructuralMountainGraph {
        StructuralMountainGraph {
            segments,
            nodes: Vec::<StructuralNode>::new(),
            links,
            omissions: vec![],
            ledger: OpportunityLedger {
                source_km2: 0.0,
                declared_km2: 0.0,
                compiled_km2: 0.0,
                omitted_km2: 0.0,
                residual_km2: 0.0,
                accounting_residual_km2: 0.0,
                segments: vec![],
            },
            readiness: StructuralReadiness::NoFiniteParent,
        }
    }

    fn link(left: &StructuralSegment, right: &StructuralSegment) -> StructuralLink {
        StructuralLink {
            node_id: 7,
            segments: [left.id, right.id],
            kind: StructuralLinkKind::Transfer,
        }
    }

    #[test]
    fn exact_links_join_only_the_same_plate_episode_system() {
        let a = segment(
            1,
            4,
            [2, 7],
            StructuralRegime::Subduction,
            Some(7),
            [CrustType::Oceanic, CrustType::Continental],
            5.0,
        );
        let b = segment(
            2,
            4,
            [2, 7],
            StructuralRegime::Collision,
            None,
            [CrustType::Continental, CrustType::Continental],
            8.0,
        );
        let other_episode = segment(
            3,
            5,
            [2, 7],
            StructuralRegime::Collision,
            None,
            [CrustType::Continental, CrustType::Continental],
            9.0,
        );
        let belts = catalog_structural_source_belts(&graph(
            vec![other_episode.clone(), b.clone(), a.clone()],
            vec![link(&a, &b), link(&b, &other_episode)],
        ))
        .unwrap();
        assert_eq!(belts.len(), 2);
        assert_eq!(belts[0].segment_ids, vec![a.id, b.id]);
        assert_eq!(belts[0].readiness, StructuralReadiness::CausallySegmented);
        assert_eq!(belts[0].declared_opportunity_km2, 13.0);
        assert_eq!(belts[1].segment_ids, vec![other_episode.id]);
    }

    #[test]
    fn selection_uses_opportunity_not_segment_count_or_height_proxy() {
        let oceanic = segment(
            1,
            1,
            [2, 7],
            StructuralRegime::Subduction,
            Some(7),
            [CrustType::Oceanic, CrustType::Oceanic],
            100.0,
        );
        let large = segment(
            2,
            1,
            [3, 8],
            StructuralRegime::Collision,
            None,
            [CrustType::Continental, CrustType::Continental],
            20.0,
        );
        let small_a = segment(
            3,
            1,
            [4, 9],
            StructuralRegime::Collision,
            None,
            [CrustType::Continental, CrustType::Continental],
            8.0,
        );
        let small_b = segment(
            4,
            1,
            [4, 9],
            StructuralRegime::Collision,
            None,
            [CrustType::Continental, CrustType::Continental],
            8.0,
        );
        let belts = catalog_structural_source_belts(&graph(
            vec![small_b.clone(), oceanic, large.clone(), small_a.clone()],
            vec![link(&small_a, &small_b)],
        ))
        .unwrap();
        let selected = select_primary_structural_source_belt(&belts).unwrap();
        assert_eq!(selected.segment_ids, vec![large.id]);
        assert_eq!(selected.readiness, StructuralReadiness::FiniteParentOnly);
    }

    #[test]
    fn oceanic_continuation_does_not_merge_terrestrial_belts() {
        let left = segment(
            1,
            1,
            [2, 7],
            StructuralRegime::Subduction,
            Some(7),
            [CrustType::Oceanic, CrustType::Continental],
            10.0,
        );
        let oceanic = segment(
            2,
            1,
            [2, 7],
            StructuralRegime::Subduction,
            Some(7),
            [CrustType::Oceanic, CrustType::Oceanic],
            100.0,
        );
        let right = segment(
            3,
            1,
            [2, 7],
            StructuralRegime::Collision,
            None,
            [CrustType::Continental, CrustType::Continental],
            9.0,
        );
        let belts = catalog_structural_source_belts(&graph(
            vec![right.clone(), oceanic.clone(), left.clone()],
            vec![link(&left, &oceanic), link(&oceanic, &right)],
        ))
        .unwrap();
        assert_eq!(belts.len(), 3);
        assert_eq!(
            select_primary_structural_source_belt(&belts).unwrap().id,
            left.id
        );
    }

    #[test]
    fn lower_identity_breaks_exact_ranking_ties() {
        let low = segment(
            1,
            1,
            [2, 7],
            StructuralRegime::Collision,
            None,
            [CrustType::Continental, CrustType::Continental],
            10.0,
        );
        let high = segment(
            2,
            1,
            [3, 8],
            StructuralRegime::Collision,
            None,
            [CrustType::Continental, CrustType::Continental],
            10.0,
        );
        let belts =
            catalog_structural_source_belts(&graph(vec![high, low.clone()], vec![])).unwrap();
        assert_eq!(
            select_primary_structural_source_belt(&belts).unwrap().id,
            low.id
        );
    }

    #[test]
    fn reports_no_continental_relief_capable_source() {
        let oceanic = segment(
            1,
            1,
            [2, 7],
            StructuralRegime::Subduction,
            Some(7),
            [CrustType::Oceanic, CrustType::Oceanic],
            10.0,
        );
        let belts = catalog_structural_source_belts(&graph(vec![oceanic], vec![])).unwrap();
        assert_eq!(
            select_primary_structural_source_belt(&belts),
            Err(StructuralSourceTargetError::NoContinentalReliefCapableBelt)
        );
    }
}
