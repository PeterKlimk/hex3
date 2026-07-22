//! Research-only source representation for the RDS0 deformation-support slice.
//!
//! This module stops at normalized deformation-opportunity rasters. It does not
//! read terrain, land, climate, drainage, erosion, or renderer state, and it
//! does not write any product field.

use std::collections::{BTreeMap, BTreeSet};

use glam::Vec3;

use super::features::smoothstep;
use super::{
    compile_structural_mountain, conservative_signed_flux_front_rates_v0, BoundaryEdgeId,
    ConservativeSignedFluxFrontErrorV0, ConvergentFrontEdge, ConvergentFrontSet, Crust, CrustType,
    Plates, StructuralMountainError, StructuralRegime, StructuralSegment, Tessellation,
    COLLISION_WIDTH, FINE_OROGEN_HINTERLAND_WIDTH, PLANET_RADIUS_KM,
};

/// The fixed number of equal-duration relative frames in RDS0.
pub const RDS0_FRAME_COUNT: usize = 4;
const RDS0_SIGNED_FLUX_SUBSTEPS: usize = 8;
const CLOSURE_RELATIVE_TOLERANCE: f64 = 2.0e-12;

/// Stable identity of one persistent sparse deformation element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct RegionalDeformationElementIdV0 {
    pub parent_segment_id: BoundaryEdgeId,
    /// `0` = left precursor, `1` = right precursor, `2` = linked successor.
    pub lineage: u8,
    /// Canonical side ordinal in the parent's canonical plate pair.
    pub side_ordinal: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
pub enum RegionalDeformationElementKindV0 {
    LeftPrecursor,
    RightPrecursor,
    LinkedSuccessor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
pub enum RegionalDeformationElementStateV0 {
    Growth,
    Overlap,
    Retirement,
    LinkedGrowth,
    Linked,
}

/// An explicit receiving material domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum RegionalDeformationRegimeV0 {
    Collision,
    Subduction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum RegionalDeformationMaterialV0 {
    Continental,
    Oceanic,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct RegionalDeformationSideV0 {
    pub side_ordinal: u8,
    pub plate: usize,
    pub material: RegionalDeformationMaterialV0,
    /// Collision partitions each edge 50/50. Subduction uses one receiver at 1.
    pub parent_share: f64,
}

/// Exact allocation from one corrected-positive source edge to one element-side.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct RegionalDeformationSourceAllocationV0 {
    pub source_edge: BoundaryEdgeId,
    /// Integral of `1 / active_element_count(u)` over this edge's normalized
    /// interval, divided by the interval length.
    pub continuous_partition: f64,
    pub source_flux_km2_per_myr: f64,
    pub allocated_flux_km2_per_myr: f64,
}

/// One active sparse element in one relative RDS0 frame.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct RegionalDeformationElementFrameV0 {
    pub id: RegionalDeformationElementIdV0,
    pub kind: RegionalDeformationElementKindV0,
    pub state: RegionalDeformationElementStateV0,
    pub frame_index: usize,
    /// Closed normalized interval along the selected finite parent.
    pub axial_interval: [f64; 2],
    /// Positive distance from the exact boundary into the receiving domain.
    pub cross_offset_km: f64,
    pub side: RegionalDeformationSideV0,
    pub source_allocations: Vec<RegionalDeformationSourceAllocationV0>,
    pub allocated_flux_km2_per_myr: f64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct RegionalDeformationFrameLedgerV0 {
    pub frame_index: usize,
    pub duration_myr: f64,
    pub source_flux_km2_per_myr: f64,
    pub allocated_flux_km2_per_myr: f64,
    pub closure_residual_km2_per_myr: f64,
    pub source_time_integral_km2: f64,
    pub allocated_time_integral_km2: f64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct RegionalDeformationFrameV0 {
    pub index: usize,
    pub start_fraction: f64,
    pub end_fraction: f64,
    pub elements: Vec<RegionalDeformationElementFrameV0>,
    pub ledger: RegionalDeformationFrameLedgerV0,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum RegionalDeformationOmissionReasonV0 {
    HistoricalSupportUnavailable,
    ClosedLoopWithoutFiniteParent,
    NoEligibleReceivingMaterial,
    AmbiguousSideSemantics,
    DisconnectedSupport,
    UnderresolvedSupport,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct RegionalDeformationOmissionV0 {
    pub reason: RegionalDeformationOmissionReasonV0,
    pub frame_index: Option<usize>,
    pub element_id: Option<RegionalDeformationElementIdV0>,
    pub source_edges: Vec<BoundaryEdgeId>,
    pub unallocated_flux_km2_per_myr: f64,
}

/// Sparse, deterministic source-only RDS0 program for one selected parent.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct RegionalDeformationCorrectedRateV0 {
    pub source_edge: BoundaryEdgeId,
    pub signed_rate_km_per_myr: f64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct RegionalDeformationProgramV0 {
    pub episode_id: usize,
    pub parent_segment_id: BoundaryEdgeId,
    pub parent_regime: RegionalDeformationRegimeV0,
    pub parent_source_edges: Vec<BoundaryEdgeId>,
    pub parent_length_km: f64,
    pub parent_duration_myr: f64,
    pub sigma_km: f64,
    pub corrected_signed_rates: Vec<RegionalDeformationCorrectedRateV0>,
    pub parent_positive_flux_km2_per_myr: f64,
    pub sides: Vec<RegionalDeformationSideV0>,
    pub frames: Vec<RegionalDeformationFrameV0>,
    pub omissions: Vec<RegionalDeformationOmissionV0>,
    /// Sum of equal-frame rate integrals over the parent's retained duration.
    pub source_time_integral_km2: f64,
    pub allocated_time_integral_km2: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegionalDeformationBuildErrorV0 {
    StructuralMountain(StructuralMountainError),
    SignedFlux(ConservativeSignedFluxFrontErrorV0),
    NoFiniteParentInEpisode(usize),
    NoPositiveParentInEpisode(usize),
    MissingSourceEdge(BoundaryEdgeId),
    InconsistentSourceGeometry(BoundaryEdgeId),
    AmbiguousSideSemantics(BoundaryEdgeId),
    NumericalFailure,
}

impl std::fmt::Display for RegionalDeformationBuildErrorV0 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for RegionalDeformationBuildErrorV0 {}

impl From<StructuralMountainError> for RegionalDeformationBuildErrorV0 {
    fn from(value: StructuralMountainError) -> Self {
        Self::StructuralMountain(value)
    }
}

impl From<ConservativeSignedFluxFrontErrorV0> for RegionalDeformationBuildErrorV0 {
    fn from(value: ConservativeSignedFluxFrontErrorV0) -> Self {
        Self::SignedFlux(value)
    }
}

/// Sparse per-cell contribution retained for source visualization/provenance.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct RegionalDeformationCellContributionV0 {
    pub element_id: RegionalDeformationElementIdV0,
    pub rate_density_per_myr: f64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct RegionalDeformationRasterLedgerV0 {
    /// `None` identifies the static control raster.
    pub frame_index: Option<usize>,
    pub requested_flux_km2_per_myr: f64,
    pub allocated_flux_km2_per_myr: f64,
    pub unallocated_flux_km2_per_myr: f64,
    pub closure_residual_km2_per_myr: f64,
    pub active_cell_count: usize,
    pub additive_overlap_cell_count: usize,
}

/// One area-normalized source-only spherical raster.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct RegionalDeformationRasterV0 {
    /// `None` identifies the static control raster.
    pub frame_index: Option<usize>,
    pub rate_density_per_myr: Vec<f64>,
    pub active_support_fraction: Vec<f32>,
    /// Opportunity-weighted axial direction, tangent to the sphere.
    pub axial_fabric: Vec<Vec3>,
    pub provenance: Vec<Vec<RegionalDeformationCellContributionV0>>,
    pub ledger: RegionalDeformationRasterLedgerV0,
    pub omissions: Vec<RegionalDeformationOmissionV0>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegionalDeformationRasterErrorV0 {
    FrameOutOfRange(usize),
    DomainLengthMismatch,
    MissingSourceEdge(BoundaryEdgeId),
    InvalidCellArea(usize),
    InvalidGeometry(BoundaryEdgeId),
    NumericalFailure,
}

impl std::fmt::Display for RegionalDeformationRasterErrorV0 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for RegionalDeformationRasterErrorV0 {}

#[derive(Clone, Copy)]
struct ElementTemplate {
    kind: RegionalDeformationElementKindV0,
    state: RegionalDeformationElementStateV0,
    interval: [f64; 2],
    offset_sigma: f64,
}

fn frame_templates(frame: usize) -> &'static [ElementTemplate] {
    const FRAME_0: [ElementTemplate; 2] = [
        ElementTemplate {
            kind: RegionalDeformationElementKindV0::LeftPrecursor,
            state: RegionalDeformationElementStateV0::Growth,
            interval: [0.0, 0.5],
            offset_sigma: 0.0,
        },
        ElementTemplate {
            kind: RegionalDeformationElementKindV0::RightPrecursor,
            state: RegionalDeformationElementStateV0::Growth,
            interval: [0.5, 1.0],
            offset_sigma: 0.0,
        },
    ];
    const FRAME_1: [ElementTemplate; 2] = [
        ElementTemplate {
            kind: RegionalDeformationElementKindV0::LeftPrecursor,
            state: RegionalDeformationElementStateV0::Overlap,
            interval: [0.0, 0.625],
            offset_sigma: 1.0 / 3.0,
        },
        ElementTemplate {
            kind: RegionalDeformationElementKindV0::RightPrecursor,
            state: RegionalDeformationElementStateV0::Overlap,
            interval: [0.375, 1.0],
            offset_sigma: 1.0 / 3.0,
        },
    ];
    const FRAME_2: [ElementTemplate; 3] = [
        ElementTemplate {
            kind: RegionalDeformationElementKindV0::LeftPrecursor,
            state: RegionalDeformationElementStateV0::Retirement,
            interval: [0.0, 0.625],
            offset_sigma: 2.0 / 3.0,
        },
        ElementTemplate {
            kind: RegionalDeformationElementKindV0::RightPrecursor,
            state: RegionalDeformationElementStateV0::Retirement,
            interval: [0.375, 1.0],
            offset_sigma: 2.0 / 3.0,
        },
        ElementTemplate {
            kind: RegionalDeformationElementKindV0::LinkedSuccessor,
            state: RegionalDeformationElementStateV0::LinkedGrowth,
            interval: [0.25, 0.75],
            offset_sigma: 1.0,
        },
    ];
    const FRAME_3: [ElementTemplate; 1] = [ElementTemplate {
        kind: RegionalDeformationElementKindV0::LinkedSuccessor,
        state: RegionalDeformationElementStateV0::Linked,
        interval: [0.0, 1.0],
        offset_sigma: 4.0 / 3.0,
    }];
    match frame {
        0 => &FRAME_0,
        1 => &FRAME_1,
        2 => &FRAME_2,
        3 => &FRAME_3,
        _ => &[],
    }
}

fn lineage(kind: RegionalDeformationElementKindV0) -> u8 {
    match kind {
        RegionalDeformationElementKindV0::LeftPrecursor => 0,
        RegionalDeformationElementKindV0::RightPrecursor => 1,
        RegionalDeformationElementKindV0::LinkedSuccessor => 2,
    }
}

/// Build the fixed four-frame dyadic-relay RDS0 source program.
///
/// The highest corrected-positive-flux finite parent in `episode_id` is chosen;
/// an exact tie is broken by canonical segment ID. Positive admission happens
/// after the fixed one-collision-width signed aggregation. Every source edge is
/// continuously partitioned among interval indicators active at each axial
/// position, so overlap never creates extra work.
pub fn build_regional_deformation_rds0_v0(
    fronts: &ConvergentFrontSet,
    episode_id: usize,
) -> Result<RegionalDeformationProgramV0, RegionalDeformationBuildErrorV0> {
    let sigma_km = f64::from(COLLISION_WIDTH) * f64::from(PLANET_RADIUS_KM);
    let corrected =
        conservative_signed_flux_front_rates_v0(fronts, sigma_km, RDS0_SIGNED_FLUX_SUBSTEPS)?;
    let graph = compile_structural_mountain(fronts)?;
    let edge_by_id: BTreeMap<_, _> = fronts.edges.iter().map(|edge| (edge.id, edge)).collect();

    let candidates: Vec<_> = graph
        .segments
        .iter()
        .filter(|segment| segment.episode_id == episode_id)
        .collect();
    if candidates.is_empty() {
        return Err(RegionalDeformationBuildErrorV0::NoFiniteParentInEpisode(
            episode_id,
        ));
    }
    let mut scored = Vec::with_capacity(candidates.len());
    for segment in candidates {
        let mut flux = 0.0;
        for edge_id in &segment.source_edges {
            let edge = edge_by_id
                .get(edge_id)
                .ok_or(RegionalDeformationBuildErrorV0::MissingSourceEdge(*edge_id))?;
            let rate = corrected
                .signed_rates_km_per_myr
                .get(edge_id)
                .copied()
                .ok_or(RegionalDeformationBuildErrorV0::MissingSourceEdge(*edge_id))?;
            flux += f64::from(edge.length_km) * rate.max(0.0);
        }
        scored.push((segment, flux));
    }
    scored.sort_by(|(left_segment, left_flux), (right_segment, right_flux)| {
        right_flux
            .total_cmp(left_flux)
            .then_with(|| left_segment.id.cmp(&right_segment.id))
    });
    let (parent, parent_flux) = scored[0];
    if !parent_flux.is_finite() || parent_flux <= 0.0 {
        return Err(RegionalDeformationBuildErrorV0::NoPositiveParentInEpisode(
            episode_id,
        ));
    }
    let parent_duration_myr = parent
        .source_edges
        .first()
        .and_then(|id| edge_by_id.get(id))
        .map(|edge| f64::from(edge.episode_duration_myr))
        .ok_or(RegionalDeformationBuildErrorV0::MissingSourceEdge(
            parent.id,
        ))?;
    if !parent_duration_myr.is_finite() || parent_duration_myr < 0.0 {
        return Err(RegionalDeformationBuildErrorV0::NumericalFailure);
    }
    let sides = receiving_sides(parent)?;
    let edge_spans = normalized_edge_spans(parent, &edge_by_id)?;
    let frame_duration = parent_duration_myr / RDS0_FRAME_COUNT as f64;
    let mut frames = Vec::with_capacity(RDS0_FRAME_COUNT);

    for frame_index in 0..RDS0_FRAME_COUNT {
        let templates = frame_templates(frame_index);
        let mut elements = Vec::new();
        for side in &sides {
            for template in templates {
                let mut allocations = Vec::new();
                let mut allocated_flux = 0.0;
                for (edge_id, edge_interval) in &edge_spans {
                    let edge = edge_by_id[edge_id];
                    let rate = corrected.signed_rates_km_per_myr[edge_id].max(0.0);
                    let source_flux = f64::from(edge.length_km) * rate;
                    let partition =
                        continuous_interval_partition(*edge_interval, template.interval, templates);
                    let allocation = source_flux * side.parent_share * partition;
                    allocations.push(RegionalDeformationSourceAllocationV0 {
                        source_edge: *edge_id,
                        continuous_partition: partition,
                        source_flux_km2_per_myr: source_flux,
                        allocated_flux_km2_per_myr: allocation,
                    });
                    allocated_flux += allocation;
                }
                elements.push(RegionalDeformationElementFrameV0 {
                    id: RegionalDeformationElementIdV0 {
                        parent_segment_id: parent.id,
                        lineage: lineage(template.kind),
                        side_ordinal: side.side_ordinal,
                    },
                    kind: template.kind,
                    state: template.state,
                    frame_index,
                    axial_interval: template.interval,
                    cross_offset_km: template.offset_sigma * sigma_km,
                    side: *side,
                    source_allocations: allocations,
                    allocated_flux_km2_per_myr: allocated_flux,
                });
            }
        }
        elements.sort_by_key(|element| element.id);
        let allocated = elements
            .iter()
            .map(|element| element.allocated_flux_km2_per_myr)
            .sum::<f64>();
        let residual = allocated - parent_flux;
        check_close(parent_flux, residual)?;
        frames.push(RegionalDeformationFrameV0 {
            index: frame_index,
            start_fraction: frame_index as f64 / RDS0_FRAME_COUNT as f64,
            end_fraction: (frame_index + 1) as f64 / RDS0_FRAME_COUNT as f64,
            elements,
            ledger: RegionalDeformationFrameLedgerV0 {
                frame_index,
                duration_myr: frame_duration,
                source_flux_km2_per_myr: parent_flux,
                allocated_flux_km2_per_myr: allocated,
                closure_residual_km2_per_myr: residual,
                source_time_integral_km2: parent_flux * frame_duration,
                allocated_time_integral_km2: allocated * frame_duration,
            },
        });
    }
    let source_time_integral = parent_flux * parent_duration_myr;
    let allocated_time_integral = frames
        .iter()
        .map(|frame| frame.ledger.allocated_time_integral_km2)
        .sum::<f64>();
    check_close(
        source_time_integral,
        allocated_time_integral - source_time_integral,
    )?;

    Ok(RegionalDeformationProgramV0 {
        episode_id,
        parent_segment_id: parent.id,
        parent_regime: match parent.regime {
            StructuralRegime::Collision => RegionalDeformationRegimeV0::Collision,
            StructuralRegime::Subduction => RegionalDeformationRegimeV0::Subduction,
        },
        parent_source_edges: parent.source_edges.clone(),
        parent_length_km: f64::from(parent.length_km),
        parent_duration_myr,
        sigma_km,
        corrected_signed_rates: corrected
            .signed_rates_km_per_myr
            .into_iter()
            .map(
                |(source_edge, signed_rate_km_per_myr)| RegionalDeformationCorrectedRateV0 {
                    source_edge,
                    signed_rate_km_per_myr,
                },
            )
            .collect(),
        parent_positive_flux_km2_per_myr: parent_flux,
        sides,
        frames,
        omissions: Vec::new(),
        source_time_integral_km2: source_time_integral,
        allocated_time_integral_km2: allocated_time_integral,
    })
}

fn receiving_sides(
    parent: &StructuralSegment,
) -> Result<Vec<RegionalDeformationSideV0>, RegionalDeformationBuildErrorV0> {
    match parent.regime {
        StructuralRegime::Collision => Ok((0..2)
            .map(|side| RegionalDeformationSideV0 {
                side_ordinal: side as u8,
                plate: parent.plate_pair[side],
                material: material(parent.crust_on_plate_pair[side]),
                parent_share: 0.5,
            })
            .collect()),
        StructuralRegime::Subduction => {
            let receiver = parent.receiving_plate.ok_or(
                RegionalDeformationBuildErrorV0::AmbiguousSideSemantics(parent.id),
            )?;
            let side = parent
                .plate_pair
                .iter()
                .position(|&plate| plate == receiver)
                .ok_or(RegionalDeformationBuildErrorV0::AmbiguousSideSemantics(
                    parent.id,
                ))?;
            Ok(vec![RegionalDeformationSideV0 {
                side_ordinal: side as u8,
                plate: receiver,
                material: material(parent.crust_on_plate_pair[side]),
                parent_share: 1.0,
            }])
        }
    }
}

fn normalized_edge_spans(
    parent: &StructuralSegment,
    edge_by_id: &BTreeMap<BoundaryEdgeId, &ConvergentFrontEdge>,
) -> Result<BTreeMap<BoundaryEdgeId, [f64; 2]>, RegionalDeformationBuildErrorV0> {
    let total = parent
        .source_edges
        .iter()
        .map(|id| {
            edge_by_id
                .get(id)
                .map(|edge| f64::from(edge.length_km))
                .ok_or(RegionalDeformationBuildErrorV0::MissingSourceEdge(*id))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(RegionalDeformationBuildErrorV0::NumericalFailure);
    }
    let mut cursor = 0.0;
    let mut result = BTreeMap::new();
    for edge_id in &parent.source_edges {
        let length = f64::from(edge_by_id[edge_id].length_km);
        let next = cursor + length / total;
        result.insert(*edge_id, [cursor, next]);
        cursor = next;
    }
    if let Some(last) = parent.source_edges.last() {
        result.get_mut(last).unwrap()[1] = 1.0;
    }
    Ok(result)
}

fn continuous_interval_partition(
    edge: [f64; 2],
    target: [f64; 2],
    active: &[ElementTemplate],
) -> f64 {
    let mut cuts = vec![edge[0], edge[1]];
    for template in active {
        for boundary in template.interval {
            if boundary > edge[0] && boundary < edge[1] {
                cuts.push(boundary);
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|left, right| *left == *right);
    let mut integral = 0.0;
    for pair in cuts.windows(2) {
        let midpoint = 0.5 * (pair[0] + pair[1]);
        if midpoint < target[0] || midpoint > target[1] {
            continue;
        }
        let count = active
            .iter()
            .filter(|template| midpoint >= template.interval[0] && midpoint <= template.interval[1])
            .count();
        debug_assert!(count > 0);
        integral += (pair[1] - pair[0]) / count as f64;
    }
    integral / (edge[1] - edge[0])
}

fn check_close(scale: f64, residual: f64) -> Result<(), RegionalDeformationBuildErrorV0> {
    if !residual.is_finite() || residual.abs() > CLOSURE_RELATIVE_TOLERANCE * scale.abs().max(1.0) {
        Err(RegionalDeformationBuildErrorV0::NumericalFailure)
    } else {
        Ok(())
    }
}

// Raster evaluation is implemented below; keeping the program construction and
// its exact allocation rules independent makes source ledgers testable without
// selecting any tessellation.

#[derive(Clone, Copy)]
struct ParentArc {
    id: BoundaryEdgeId,
    a: Vec3,
    b: Vec3,
    u: [f64; 2],
}

#[derive(Clone, Copy)]
struct ArcProjection {
    distance_radians: f64,
    u: f64,
    tangent: Vec3,
}

fn parent_arcs(
    program: &RegionalDeformationProgramV0,
    fronts: &ConvergentFrontSet,
) -> Result<Vec<ParentArc>, RegionalDeformationRasterErrorV0> {
    let graph = compile_structural_mountain(fronts)
        .map_err(|_| RegionalDeformationRasterErrorV0::NumericalFailure)?;
    let segment = graph
        .segments
        .iter()
        .find(|segment| segment.id == program.parent_segment_id)
        .ok_or(RegionalDeformationRasterErrorV0::MissingSourceEdge(
            program.parent_segment_id,
        ))?;
    let edge_by_id: BTreeMap<_, _> = fronts.edges.iter().map(|edge| (edge.id, edge)).collect();
    let spans = normalized_edge_spans(segment, &edge_by_id)
        .map_err(|_| RegionalDeformationRasterErrorV0::NumericalFailure)?;
    let mut arcs = Vec::with_capacity(segment.source_edges.len());
    for (index, edge_id) in segment.source_edges.iter().enumerate() {
        let edge =
            edge_by_id
                .get(edge_id)
                .ok_or(RegionalDeformationRasterErrorV0::MissingSourceEdge(
                    *edge_id,
                ))?;
        let wanted = [
            segment.vertices_in_order[index],
            segment.vertices_in_order[index + 1],
        ];
        let (a, b) = if edge.vertices == wanted {
            (edge.endpoints[0], edge.endpoints[1])
        } else if edge.vertices == [wanted[1], wanted[0]] {
            (edge.endpoints[1], edge.endpoints[0])
        } else {
            return Err(RegionalDeformationRasterErrorV0::InvalidGeometry(*edge_id));
        };
        arcs.push(ParentArc {
            id: *edge_id,
            a,
            b,
            u: spans[edge_id],
        });
    }
    Ok(arcs)
}

fn slerp_unit(a: Vec3, b: Vec3, t: f64) -> Vec3 {
    let angle = f64::from(a.dot(b).clamp(-1.0, 1.0)).acos();
    if angle <= 1.0e-10 {
        return (a * (1.0 - t as f32) + b * t as f32).normalize_or_zero();
    }
    let left = ((1.0 - t) * angle).sin() / angle.sin();
    let right = (t * angle).sin() / angle.sin();
    (a * left as f32 + b * right as f32).normalize()
}

fn clipped_arc(arc: ParentArc, interval: [f64; 2]) -> Option<ParentArc> {
    let lo = arc.u[0].max(interval[0]);
    let hi = arc.u[1].min(interval[1]);
    if hi <= lo {
        return None;
    }
    let span = arc.u[1] - arc.u[0];
    let t0 = (lo - arc.u[0]) / span;
    let t1 = (hi - arc.u[0]) / span;
    Some(ParentArc {
        id: arc.id,
        a: slerp_unit(arc.a, arc.b, t0),
        b: slerp_unit(arc.a, arc.b, t1),
        u: [lo, hi],
    })
}

fn angle_between(a: Vec3, b: Vec3) -> f64 {
    f64::from(a.dot(b).clamp(-1.0, 1.0)).acos()
}

fn project_to_arc(point: Vec3, arc: ParentArc) -> Option<ArcProjection> {
    let normal = arc.a.cross(arc.b).normalize_or_zero();
    if normal.length_squared() <= 1.0e-12 {
        return None;
    }
    let projected = (point - normal * point.dot(normal)).normalize_or_zero();
    let full = angle_between(arc.a, arc.b);
    let accepts = |candidate: Vec3| {
        let left = angle_between(arc.a, candidate);
        let right = angle_between(candidate, arc.b);
        (left + right - full).abs() <= 2.0e-5
    };
    let closest = if projected.length_squared() > 0.0 && accepts(projected) {
        projected
    } else if projected.length_squared() > 0.0 && accepts(-projected) {
        -projected
    } else if angle_between(point, arc.a) <= angle_between(point, arc.b) {
        arc.a
    } else {
        arc.b
    };
    let local_fraction = if full > 0.0 {
        (angle_between(arc.a, closest) / full).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let tangent = normal.cross(closest).normalize_or_zero();
    Some(ArcProjection {
        distance_radians: angle_between(point, closest),
        u: arc.u[0] + local_fraction * (arc.u[1] - arc.u[0]),
        tangent,
    })
}

fn side_normal(
    arc: ParentArc,
    side: RegionalDeformationSideV0,
    edge: &ConvergentFrontEdge,
    tessellation: &Tessellation,
) -> Option<Vec3> {
    let source_cell = edge
        .plates
        .iter()
        .position(|&plate| plate == side.plate)
        .map(|index| edge.cells[index])?;
    let normal = arc.a.cross(arc.b).normalize_or_zero();
    if normal.length_squared() <= 1.0e-12 {
        return None;
    }
    let sign = if tessellation.cell_center(source_cell).dot(normal) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    Some(normal * sign)
}

fn domain_is_eligible(
    cell: usize,
    side: RegionalDeformationSideV0,
    plates: &Plates,
    crust: &Crust,
) -> bool {
    plates.cell_plate[cell] as usize == side.plate && material(crust.types[cell]) == side.material
}

fn connected_domain_mask(
    program: &RegionalDeformationProgramV0,
    fronts: &ConvergentFrontSet,
    tessellation: &Tessellation,
    plates: &Plates,
    crust: &Crust,
    side: RegionalDeformationSideV0,
) -> Result<Vec<bool>, RegionalDeformationRasterErrorV0> {
    let edge_by_id: BTreeMap<_, _> = fronts.edges.iter().map(|edge| (edge.id, edge)).collect();
    let mut mask = vec![false; tessellation.num_cells()];
    let mut queue = std::collections::VecDeque::new();
    for edge_id in &program.parent_source_edges {
        let edge =
            edge_by_id
                .get(edge_id)
                .ok_or(RegionalDeformationRasterErrorV0::MissingSourceEdge(
                    *edge_id,
                ))?;
        if let Some(index) = edge.plates.iter().position(|&plate| plate == side.plate) {
            let cell = edge.cells[index];
            if cell < mask.len() && domain_is_eligible(cell, side, plates, crust) && !mask[cell] {
                mask[cell] = true;
                queue.push_back(cell);
            }
        }
    }
    while let Some(cell) = queue.pop_front() {
        for &neighbor in tessellation.neighbors(cell) {
            if !mask[neighbor] && domain_is_eligible(neighbor, side, plates, crust) {
                mask[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }
    Ok(mask)
}

fn material(crust: CrustType) -> RegionalDeformationMaterialV0 {
    match crust {
        CrustType::Continental => RegionalDeformationMaterialV0::Continental,
        CrustType::Oceanic => RegionalDeformationMaterialV0::Oceanic,
    }
}

fn physical_cell_areas(
    tessellation: &Tessellation,
) -> Result<Vec<f64>, RegionalDeformationRasterErrorV0> {
    tessellation
        .cell_areas_ref()
        .iter()
        .enumerate()
        .map(|(index, &area)| {
            let physical = f64::from(area) * f64::from(PLANET_RADIUS_KM).powi(2);
            if physical.is_finite() && physical > 0.0 {
                Ok(physical)
            } else {
                Err(RegionalDeformationRasterErrorV0::InvalidCellArea(index))
            }
        })
        .collect()
}

fn validate_domain(
    tessellation: &Tessellation,
    plates: &Plates,
    crust: &Crust,
) -> Result<(), RegionalDeformationRasterErrorV0> {
    let n = tessellation.num_cells();
    if plates.cell_plate.len() != n || crust.types.len() != n {
        Err(RegionalDeformationRasterErrorV0::DomainLengthMismatch)
    } else {
        Ok(())
    }
}

fn normalize_contribution(raw: &[f64], areas: &[f64], target: f64) -> Option<Vec<f64>> {
    let denominator = raw
        .iter()
        .zip(areas)
        .map(|(&weight, &area)| weight * area)
        .sum::<f64>();
    if !denominator.is_finite() || denominator <= 0.0 || target <= 0.0 {
        return None;
    }
    let scale = target / denominator;
    let mut normalized: Vec<_> = raw.iter().map(|weight| weight * scale).collect();
    let integrated = normalized
        .iter()
        .zip(areas)
        .map(|(&density, &area)| density * area)
        .sum::<f64>();
    let residual = target - integrated;
    let anchor = raw
        .iter()
        .enumerate()
        .filter(|(_, weight)| **weight > 0.0)
        .max_by(|(left_id, left), (right_id, right)| {
            left.total_cmp(right).then_with(|| right_id.cmp(left_id))
        })
        .map(|(index, _)| index)?;
    normalized[anchor] += residual / areas[anchor];
    Some(normalized)
}

/// Evaluate one RDS0 frame on the source (coarse) spherical material raster.
///
/// Each element-side is normalized independently to its exact source allocation;
/// overlapping elements then add. The compact axial kernel is nonperiodic and
/// vanishes at true interval ends. The compact cross kernel has radius `σ`.
pub fn evaluate_regional_deformation_frame_v0(
    program: &RegionalDeformationProgramV0,
    frame_index: usize,
    fronts: &ConvergentFrontSet,
    tessellation: &Tessellation,
    plates: &Plates,
    crust: &Crust,
) -> Result<RegionalDeformationRasterV0, RegionalDeformationRasterErrorV0> {
    validate_domain(tessellation, plates, crust)?;
    let frame = program.frames.get(frame_index).ok_or(
        RegionalDeformationRasterErrorV0::FrameOutOfRange(frame_index),
    )?;
    let n = tessellation.num_cells();
    let areas = physical_cell_areas(tessellation)?;
    let arcs = parent_arcs(program, fronts)?;
    let edge_by_id: BTreeMap<_, _> = fronts.edges.iter().map(|edge| (edge.id, edge)).collect();
    let connected_masks: BTreeMap<_, _> = program
        .sides
        .iter()
        .map(|&side| {
            connected_domain_mask(program, fronts, tessellation, plates, crust, side)
                .map(|mask| (side.side_ordinal, mask))
        })
        .collect::<Result<_, _>>()?;
    let mut density = vec![0.0; n];
    let mut support = vec![0.0f32; n];
    let mut fabric_sum = vec![Vec3::ZERO; n];
    let mut provenance = vec![Vec::new(); n];
    let mut omissions = Vec::new();
    let mut allocated_total = 0.0;
    let mut unallocated_total = 0.0;

    for element in &frame.elements {
        let source_ids: BTreeSet<_> = element
            .source_allocations
            .iter()
            .filter(|allocation| allocation.allocated_flux_km2_per_myr > 0.0)
            .map(|allocation| allocation.source_edge)
            .collect();
        let clipped: Vec<_> = arcs
            .iter()
            .filter_map(|&arc| clipped_arc(arc, element.axial_interval))
            .collect();
        let connected = &connected_masks[&element.side.side_ordinal];
        let eligible_count = (0..n).filter(|&cell| connected[cell]).count();
        let mut raw = vec![0.0; n];
        let mut raw_tangent = vec![Vec3::ZERO; n];
        let interval_length = element.axial_interval[1] - element.axial_interval[0];
        for cell in 0..n {
            if !connected[cell] {
                continue;
            }
            let point = tessellation.cell_center(cell);
            let mut best: Option<(ArcProjection, Vec3, BoundaryEdgeId)> = None;
            for &arc in &clipped {
                let Some(projection) = project_to_arc(point, arc) else {
                    continue;
                };
                let edge = edge_by_id
                    .get(&arc.id)
                    .ok_or(RegionalDeformationRasterErrorV0::MissingSourceEdge(arc.id))?;
                let Some(side_normal) = side_normal(arc, element.side, edge, tessellation) else {
                    continue;
                };
                if best.as_ref().is_none_or(|(current, _, current_edge)| {
                    projection.distance_radians < current.distance_radians
                        || (projection.distance_radians == current.distance_radians
                            && arc.id < *current_edge)
                }) {
                    best = Some((projection, side_normal, arc.id));
                }
            }
            let Some((projection, toward_side, _)) = best else {
                continue;
            };
            let cross_km = f64::from(point.dot(toward_side).clamp(-1.0, 1.0)).asin()
                * f64::from(PLANET_RADIUS_KM);
            if cross_km < 0.0 {
                continue;
            }
            let axial_t =
                ((projection.u - element.axial_interval[0]) / interval_length).clamp(0.0, 1.0);
            let axial = 16.0 * axial_t.powi(2) * (1.0 - axial_t).powi(2);
            let cross_z = (cross_km - element.cross_offset_km).abs() / program.sigma_km;
            let cross = if cross_z < 1.0 {
                (1.0 - cross_z * cross_z).powi(2)
            } else {
                0.0
            };
            raw[cell] = axial * cross;
            raw_tangent[cell] = projection.tangent;
        }
        let target = element.allocated_flux_km2_per_myr;
        let Some(normalized) = normalize_contribution(&raw, &areas, target) else {
            let reason = if eligible_count == 0 {
                RegionalDeformationOmissionReasonV0::NoEligibleReceivingMaterial
            } else {
                RegionalDeformationOmissionReasonV0::UnderresolvedSupport
            };
            omissions.push(RegionalDeformationOmissionV0 {
                reason,
                frame_index: Some(frame_index),
                element_id: Some(element.id),
                source_edges: source_ids.into_iter().collect(),
                unallocated_flux_km2_per_myr: target,
            });
            unallocated_total += target;
            continue;
        };
        allocated_total += target;
        for cell in 0..n {
            if normalized[cell] <= 0.0 {
                continue;
            }
            density[cell] += normalized[cell];
            support[cell] = support[cell].max(raw[cell] as f32);
            fabric_sum[cell] += raw_tangent[cell] * normalized[cell] as f32;
            provenance[cell].push(RegionalDeformationCellContributionV0 {
                element_id: element.id,
                rate_density_per_myr: normalized[cell],
            });
        }
    }
    let axial_fabric = fabric_sum
        .into_iter()
        .map(|value| value.normalize_or_zero())
        .collect();
    let integrated = density
        .iter()
        .zip(&areas)
        .map(|(&value, &area)| value * area)
        .sum::<f64>();
    let closure = integrated + unallocated_total - frame.ledger.source_flux_km2_per_myr;
    if !closure.is_finite()
        || closure.abs()
            > CLOSURE_RELATIVE_TOLERANCE * frame.ledger.source_flux_km2_per_myr.max(1.0)
    {
        return Err(RegionalDeformationRasterErrorV0::NumericalFailure);
    }
    Ok(RegionalDeformationRasterV0 {
        frame_index: Some(frame_index),
        rate_density_per_myr: density,
        active_support_fraction: support,
        axial_fabric,
        ledger: RegionalDeformationRasterLedgerV0 {
            frame_index: Some(frame_index),
            requested_flux_km2_per_myr: frame.ledger.source_flux_km2_per_myr,
            allocated_flux_km2_per_myr: allocated_total,
            unallocated_flux_km2_per_myr: unallocated_total,
            closure_residual_km2_per_myr: closure,
            active_cell_count: provenance
                .iter()
                .filter(|values| !values.is_empty())
                .count(),
            additive_overlap_cell_count: provenance
                .iter()
                .filter(|values| values.len() > 1)
                .count(),
        },
        provenance,
        omissions,
    })
}

/// Evaluate the static nearest-exact-arc control for the selected RDS0 parent.
///
/// This is the current gentle receiver-side support grammar without terrain or
/// Legacy fields: corrected positive owner rate times the hinterland smoothstep,
/// area-normalized independently on each receiving side to the same explicit
/// side partition used by every RDS0 frame.
pub fn evaluate_regional_deformation_static_control_v0(
    program: &RegionalDeformationProgramV0,
    fronts: &ConvergentFrontSet,
    tessellation: &Tessellation,
    plates: &Plates,
    crust: &Crust,
) -> Result<RegionalDeformationRasterV0, RegionalDeformationRasterErrorV0> {
    validate_domain(tessellation, plates, crust)?;
    let n = tessellation.num_cells();
    let areas = physical_cell_areas(tessellation)?;
    let arcs = parent_arcs(program, fronts)?;
    let connected_masks: BTreeMap<_, _> = program
        .sides
        .iter()
        .map(|&side| {
            connected_domain_mask(program, fronts, tessellation, plates, crust, side)
                .map(|mask| (side.side_ordinal, mask))
        })
        .collect::<Result<_, _>>()?;
    let mut density = vec![0.0; n];
    let mut support = vec![0.0f32; n];
    let mut fabric_sum = vec![Vec3::ZERO; n];
    let mut provenance = vec![Vec::new(); n];
    let mut omissions = Vec::new();
    let mut allocated_total = 0.0;
    let mut unallocated_total = 0.0;
    let width = f64::from(FINE_OROGEN_HINTERLAND_WIDTH);
    for &side in &program.sides {
        let connected = &connected_masks[&side.side_ordinal];
        let mut raw = vec![0.0; n];
        let mut tangent = vec![Vec3::ZERO; n];
        for cell in 0..n {
            if !connected[cell] {
                continue;
            }
            let point = tessellation.cell_center(cell);
            let mut best: Option<(f64, BoundaryEdgeId, ArcProjection)> = None;
            for &arc in &arcs {
                let Some(projection) = project_to_arc(point, arc) else {
                    continue;
                };
                if best.as_ref().is_none_or(|current| {
                    projection.distance_radians < current.0
                        || (projection.distance_radians == current.0 && arc.id < current.1)
                }) {
                    best = Some((projection.distance_radians, arc.id, projection));
                }
            }
            let Some((distance, edge_id, projection)) = best else {
                continue;
            };
            let profile = 1.0 - f64::from(smoothstep(0.0, width as f32, distance as f32));
            let rate = program
                .corrected_signed_rates
                .iter()
                .find(|rate| rate.source_edge == edge_id)
                .ok_or(RegionalDeformationRasterErrorV0::MissingSourceEdge(edge_id))?
                .signed_rate_km_per_myr
                .max(0.0);
            raw[cell] = rate * profile;
            tangent[cell] = projection.tangent;
        }
        let target = program.parent_positive_flux_km2_per_myr * side.parent_share;
        let Some(normalized) = normalize_contribution(&raw, &areas, target) else {
            let reason = if connected.iter().all(|&value| !value) {
                RegionalDeformationOmissionReasonV0::NoEligibleReceivingMaterial
            } else {
                RegionalDeformationOmissionReasonV0::UnderresolvedSupport
            };
            omissions.push(RegionalDeformationOmissionV0 {
                reason,
                frame_index: None,
                element_id: Some(RegionalDeformationElementIdV0 {
                    parent_segment_id: program.parent_segment_id,
                    lineage: u8::MAX,
                    side_ordinal: side.side_ordinal,
                }),
                source_edges: program.parent_source_edges.clone(),
                unallocated_flux_km2_per_myr: target,
            });
            unallocated_total += target;
            continue;
        };
        allocated_total += target;
        for cell in 0..n {
            if normalized[cell] <= 0.0 {
                continue;
            }
            density[cell] += normalized[cell];
            support[cell] = support[cell].max(raw[cell].min(1.0) as f32);
            fabric_sum[cell] += tangent[cell] * normalized[cell] as f32;
            provenance[cell].push(RegionalDeformationCellContributionV0 {
                element_id: RegionalDeformationElementIdV0 {
                    parent_segment_id: program.parent_segment_id,
                    lineage: u8::MAX,
                    side_ordinal: side.side_ordinal,
                },
                rate_density_per_myr: normalized[cell],
            });
        }
    }
    let integrated = density
        .iter()
        .zip(&areas)
        .map(|(&value, &area)| value * area)
        .sum::<f64>();
    let closure = integrated + unallocated_total - program.parent_positive_flux_km2_per_myr;
    if !closure.is_finite()
        || closure.abs()
            > CLOSURE_RELATIVE_TOLERANCE * program.parent_positive_flux_km2_per_myr.max(1.0)
    {
        return Err(RegionalDeformationRasterErrorV0::NumericalFailure);
    }
    Ok(RegionalDeformationRasterV0 {
        frame_index: None,
        rate_density_per_myr: density,
        active_support_fraction: support,
        axial_fabric: fabric_sum
            .into_iter()
            .map(|value| value.normalize_or_zero())
            .collect(),
        ledger: RegionalDeformationRasterLedgerV0 {
            frame_index: None,
            requested_flux_km2_per_myr: program.parent_positive_flux_km2_per_myr,
            allocated_flux_km2_per_myr: allocated_total,
            unallocated_flux_km2_per_myr: unallocated_total,
            closure_residual_km2_per_myr: closure,
            active_cell_count: provenance
                .iter()
                .filter(|values| !values.is_empty())
                .count(),
            additive_overlap_cell_count: 0,
        },
        provenance,
        omissions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{HistoryModel, World};

    fn edge(
        cells: [usize; 2],
        vertices: [u32; 2],
        episode: usize,
        length_km: f32,
        rate: f32,
    ) -> ConvergentFrontEdge {
        let position = |vertex: u32| {
            let angle = 0.01 * vertex as f32;
            Vec3::new(angle.cos(), angle.sin(), 0.08).normalize()
        };
        ConvergentFrontEdge {
            id: BoundaryEdgeId::new(cells[0], cells[1]),
            cells,
            vertices,
            endpoints: [position(vertices[0]), position(vertices[1])],
            midpoint: (position(vertices[0]) + position(vertices[1])).normalize(),
            length_km,
            plates: [2, 7],
            crust: [CrustType::Continental, CrustType::Continental],
            regime: StructuralRegime::Collision,
            subducting_plate: None,
            receiving_plate: None,
            convergence_km_per_myr: rate,
            shear_km_per_myr: 0.0,
            relative_speed_km_per_myr: rate.abs(),
            episode_id: episode,
            episode_duration_myr: 8.0,
            episode_normal_displacement_km: rate * 8.0,
            episode_shear_displacement_km: 0.0,
            history_model: HistoryModel::StationaryTopologyConstantVelocity,
            shortening_area_opportunity_km2: f64::from(length_km) * f64::from(rate.max(0.0)) * 8.0,
        }
    }

    fn manufactured_fronts() -> ConvergentFrontSet {
        let edges = vec![
            edge([0, 10], [1, 2], 9, 40.0, 1.0),
            edge([1, 11], [2, 3], 9, 40.0, 2.0),
            edge([2, 12], [3, 4], 9, 40.0, 3.0),
            edge([3, 13], [4, 5], 9, 40.0, 4.0),
        ];
        let mut degrees = BTreeMap::new();
        for edge in &edges {
            for vertex in edge.vertices {
                *degrees.entry(vertex).or_default() += 1;
            }
        }
        ConvergentFrontSet {
            edges,
            all_boundary_vertex_degree: degrees,
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = 5.0e-12 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:?} expected={expected:?} tolerance={tolerance:?}"
        );
    }

    fn assert_control_side_budgets(
        control: &RegionalDeformationRasterV0,
        program: &RegionalDeformationProgramV0,
        tessellation: &Tessellation,
    ) {
        let areas = physical_cell_areas(tessellation).unwrap();
        let mut by_side = BTreeMap::<u8, f64>::new();
        for (cell, contributions) in control.provenance.iter().enumerate() {
            for contribution in contributions {
                *by_side
                    .entry(contribution.element_id.side_ordinal)
                    .or_default() += contribution.rate_density_per_myr * areas[cell];
            }
        }
        for side in &program.sides {
            assert_close(
                by_side.get(&side.side_ordinal).copied().unwrap_or(0.0),
                program.parent_positive_flux_km2_per_myr * side.parent_share,
            );
        }
    }

    #[test]
    fn rds0_partitions_every_edge_frame_and_side_without_overlap_energy() {
        let program = build_regional_deformation_rds0_v0(&manufactured_fronts(), 9).unwrap();
        assert_eq!(program.frames.len(), RDS0_FRAME_COUNT);
        assert_eq!(program.sides.len(), 2);
        assert_eq!(program.parent_source_edges.len(), 4);

        for frame in &program.frames {
            assert_close(
                frame.ledger.allocated_flux_km2_per_myr,
                program.parent_positive_flux_km2_per_myr,
            );
            for side in &program.sides {
                for edge_id in &program.parent_source_edges {
                    let allocations: Vec<_> = frame
                        .elements
                        .iter()
                        .filter(|element| element.side.side_ordinal == side.side_ordinal)
                        .flat_map(|element| &element.source_allocations)
                        .filter(|allocation| allocation.source_edge == *edge_id)
                        .collect();
                    let partition = allocations
                        .iter()
                        .map(|allocation| allocation.continuous_partition)
                        .sum::<f64>();
                    assert_close(partition, 1.0);
                    let expected = allocations[0].source_flux_km2_per_myr * side.parent_share;
                    assert_close(
                        allocations
                            .iter()
                            .map(|allocation| allocation.allocated_flux_km2_per_myr)
                            .sum(),
                        expected,
                    );
                }
            }
        }

        // Frame 2 has two retiring precursors plus the linked successor. The
        // middle source edge participates in all three, but still closes once.
        let middle = program.parent_source_edges[1];
        let active = program.frames[2]
            .elements
            .iter()
            .filter(|element| element.side.side_ordinal == 0)
            .filter(|element| {
                element.source_allocations.iter().any(|allocation| {
                    allocation.source_edge == middle && allocation.continuous_partition > 0.0
                })
            })
            .count();
        assert_eq!(active, 3);
    }

    #[test]
    fn rds0_time_integral_and_repeat_are_deterministic() {
        let fronts = manufactured_fronts();
        let first = build_regional_deformation_rds0_v0(&fronts, 9).unwrap();
        let second = build_regional_deformation_rds0_v0(&fronts, 9).unwrap();
        assert_eq!(first, second);
        assert_close(
            first.allocated_time_integral_km2,
            first.source_time_integral_km2,
        );
        assert_close(
            first.source_time_integral_km2,
            first.parent_positive_flux_km2_per_myr * first.parent_duration_myr,
        );
        for side in 0..2 {
            let elements: Vec<_> = first.frames[0]
                .elements
                .iter()
                .filter(|element| element.side.side_ordinal == side)
                .map(|element| {
                    (
                        element.kind,
                        element.axial_interval,
                        element.cross_offset_km,
                    )
                })
                .collect();
            assert_eq!(
                elements,
                vec![
                    (
                        RegionalDeformationElementKindV0::LeftPrecursor,
                        [0.0, 0.5],
                        0.0
                    ),
                    (
                        RegionalDeformationElementKindV0::RightPrecursor,
                        [0.5, 1.0],
                        0.0
                    ),
                ]
            );
        }
    }

    #[test]
    fn subduction_has_only_the_declared_receiver() {
        let mut fronts = manufactured_fronts();
        for edge in &mut fronts.edges {
            edge.regime = StructuralRegime::Subduction;
            edge.subducting_plate = Some(2);
            edge.receiving_plate = Some(7);
        }
        let program = build_regional_deformation_rds0_v0(&fronts, 9).unwrap();
        assert_eq!(program.sides.len(), 1);
        assert_eq!(program.sides[0].plate, 7);
        assert_eq!(program.sides[0].parent_share, 1.0);
    }

    #[test]
    fn generated_source_program_and_coarse_rasters_close() {
        let mut world = World::new(8_675_309, 2_048, 0);
        world.generate_plates(8);
        world.generate_crust();
        world.generate_dynamics();
        world.generate_features();
        let boundaries = super::super::collect_plate_boundaries(
            &world.tessellation,
            world.plates.as_ref().unwrap(),
            world.crust.as_ref().unwrap(),
            world.dynamics.as_ref().unwrap(),
        );
        let fronts = super::super::collect_convergent_fronts(
            &world.tessellation,
            &boundaries,
            world.tectonic_history.as_ref().unwrap(),
        )
        .unwrap();
        let graph = compile_structural_mountain(&fronts).unwrap();
        let Some(episode) = graph
            .segments
            .iter()
            .map(|segment| segment.episode_id)
            .min()
        else {
            return;
        };
        let program = build_regional_deformation_rds0_v0(&fronts, episode).unwrap();
        let control = evaluate_regional_deformation_static_control_v0(
            &program,
            &fronts,
            &world.tessellation,
            world.plates.as_ref().unwrap(),
            world.crust.as_ref().unwrap(),
        )
        .unwrap();
        assert_close(
            control.ledger.allocated_flux_km2_per_myr,
            program.parent_positive_flux_km2_per_myr,
        );
        if program.sides.len() == 2 && control.omissions.is_empty() {
            assert_control_side_budgets(&control, &program, &world.tessellation);
        }
        for frame in 0..RDS0_FRAME_COUNT {
            let raster = evaluate_regional_deformation_frame_v0(
                &program,
                frame,
                &fronts,
                &world.tessellation,
                world.plates.as_ref().unwrap(),
                world.crust.as_ref().unwrap(),
            )
            .unwrap();
            assert_close(
                raster.ledger.allocated_flux_km2_per_myr
                    + raster.ledger.unallocated_flux_km2_per_myr,
                program.parent_positive_flux_km2_per_myr,
            );
            if frame == 2 {
                assert!(raster.ledger.additive_overlap_cell_count > 0);
            }
        }

        // The control must not introduce an area-dependent collision-side split.
        // Find a generated collision parent and prove both independently
        // normalized provenance ledgers receive exactly one half.
        let episodes: BTreeSet<_> = graph
            .segments
            .iter()
            .map(|segment| segment.episode_id)
            .collect();
        let collision = episodes.into_iter().find_map(|episode| {
            let candidate = build_regional_deformation_rds0_v0(&fronts, episode).ok()?;
            if candidate.sides.len() != 2 {
                return None;
            }
            let control = evaluate_regional_deformation_static_control_v0(
                &candidate,
                &fronts,
                &world.tessellation,
                world.plates.as_ref().unwrap(),
                world.crust.as_ref().unwrap(),
            )
            .ok()?;
            control.omissions.is_empty().then_some((candidate, control))
        });
        let (collision_program, collision_control) =
            collision.expect("generated source should contain a resolved collision parent");
        assert_control_side_budgets(&collision_control, &collision_program, &world.tessellation);
    }
}
