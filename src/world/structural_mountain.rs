//! Product-facing convergent-front organization for Structural Mountain V0.
//!
//! This module is deliberately inert: it compiles causal source evidence and a
//! closed shortening-opportunity ledger, but does not write terrain. It does
//! not reuse the experimental `OrogenFronts` chain coordinates or the research
//! landscape compiler.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::f64::consts::TAU;

use glam::Vec3;

use super::{
    BoundaryKind, CellEdgeId, CrustType, HistoryModel, PlateBoundaryEdge, SubductionPolarity,
    TectonicHistory, Tessellation, PLANET_RADIUS_KM,
};

/// Stable identity of one product boundary edge.
pub type BoundaryEdgeId = CellEdgeId;

/// Collision and subduction have different side semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum StructuralRegime {
    Collision,
    Subduction,
}

/// One exact Voronoi front arc with local kinematics and bounded history.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvergentFrontEdge {
    pub id: BoundaryEdgeId,
    pub cells: [usize; 2],
    pub vertices: [u32; 2],
    pub endpoints: [Vec3; 2],
    pub midpoint: Vec3,
    pub length_km: f32,
    pub plates: [usize; 2],
    pub crust: [CrustType; 2],
    pub regime: StructuralRegime,
    pub subducting_plate: Option<usize>,
    pub receiving_plate: Option<usize>,
    pub convergence_km_per_myr: f32,
    pub shear_km_per_myr: f32,
    pub relative_speed_km_per_myr: f32,
    pub episode_id: usize,
    pub episode_duration_myr: f32,
    pub episode_normal_displacement_km: f32,
    pub episode_shear_displacement_km: f32,
    pub history_model: HistoryModel,
    /// `length_km * max(local convergence, 0) * episode duration`.
    /// Units are km² of shortening-area opportunity, not uplift volume, work,
    /// elevation, or terrain response.
    pub shortening_area_opportunity_km2: f64,
}

/// Collected front arcs plus degree in the complete plate-boundary graph.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvergentFrontSet {
    pub edges: Vec<ConvergentFrontEdge>,
    pub all_boundary_vertex_degree: BTreeMap<u32, usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StructuralMountainError {
    NonCanonicalBoundary(BoundaryEdgeId),
    DuplicateBoundary(BoundaryEdgeId),
    MissingSharedTopology(BoundaryEdgeId),
    MissingEpisode(BoundaryEdgeId),
    InconsistentEpisode(BoundaryEdgeId),
    InvalidGeometry(BoundaryEdgeId),
    InvalidKinematics(BoundaryEdgeId),
    InvalidSideSemantics(BoundaryEdgeId),
    InvalidBoundaryDegree(u32),
    InconsistentVertexPosition(u32),
    EmptyTaperSupport(BoundaryEdgeId),
}

impl std::fmt::Display for StructuralMountainError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for StructuralMountainError {}

/// Invalid input to [`conservative_signed_flux_profile_v0`].
#[cfg(feature = "research-landscape")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConservativeSignedFluxProfileError {
    LengthRateCountMismatch,
    InvalidLength(usize),
    InvalidRate(usize),
    InvalidSigmaKm,
    InvalidSubsteps,
    NumericalFailure,
}

#[cfg(feature = "research-landscape")]
impl std::fmt::Display for ConservativeSignedFluxProfileError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

#[cfg(feature = "research-landscape")]
impl std::error::Error for ConservativeSignedFluxProfileError {}

/// Aggregate an ordered signed edge-rate profile without changing its topology.
///
/// This research operator applies backward-Euler finite-volume diffusion with
/// no-flux segment ends. Control-volume lengths are the conserved measure and
/// the conductance between adjacent controls is
/// `2 / (lengths_km[i] + lengths_km[i + 1])`. The total diffusion time is
/// `sigma_km.powi(2) / 2`, divided into the caller-declared `substeps`.
///
/// The result has exactly one value per input edge and preserves
/// `sum(length * signed_rate)` to floating-point roundoff. Signs are deliberately
/// not clipped here: any positive-part rectification belongs *after* this
/// conservative aggregation, so canceling edge-scale flux is not converted into
/// spurious positive work first. A zero sigma returns an exact copy and permits
/// zero substeps; a positive sigma requires at least one substep.
#[cfg(feature = "research-landscape")]
pub fn conservative_signed_flux_profile_v0(
    rates: &[f64],
    lengths_km: &[f64],
    sigma_km: f64,
    substeps: usize,
) -> Result<Vec<f64>, ConservativeSignedFluxProfileError> {
    if rates.len() != lengths_km.len() {
        return Err(ConservativeSignedFluxProfileError::LengthRateCountMismatch);
    }
    for (index, &length) in lengths_km.iter().enumerate() {
        if !length.is_finite() || length <= 0.0 {
            return Err(ConservativeSignedFluxProfileError::InvalidLength(index));
        }
    }
    for (index, &rate) in rates.iter().enumerate() {
        if !rate.is_finite() {
            return Err(ConservativeSignedFluxProfileError::InvalidRate(index));
        }
    }
    if !sigma_km.is_finite() || sigma_km < 0.0 {
        return Err(ConservativeSignedFluxProfileError::InvalidSigmaKm);
    }
    if sigma_km == 0.0 {
        return Ok(rates.to_vec());
    }
    if substeps == 0 {
        return Err(ConservativeSignedFluxProfileError::InvalidSubsteps);
    }
    if rates.len() <= 1 {
        return Ok(rates.to_vec());
    }

    let total_time_km2 = 0.5 * sigma_km * sigma_km;
    let step_time_km2 = total_time_km2 / substeps as f64;
    if !total_time_km2.is_finite() || !step_time_km2.is_finite() {
        return Err(ConservativeSignedFluxProfileError::InvalidSigmaKm);
    }

    let mut conductance = Vec::with_capacity(rates.len() - 1);
    for lengths in lengths_km.windows(2) {
        let value = 2.0 / (lengths[0] + lengths[1]);
        if !value.is_finite() || value <= 0.0 {
            return Err(ConservativeSignedFluxProfileError::NumericalFailure);
        }
        conductance.push(value);
    }

    let mut profile = rates.to_vec();
    let mut diagonal = vec![0.0; rates.len()];
    let mut rhs = vec![0.0; rates.len()];
    let mut off_diagonal = vec![0.0; rates.len() - 1];
    for _ in 0..substeps {
        for index in 0..profile.len() {
            let left = index
                .checked_sub(1)
                .map_or(0.0, |interface| conductance[interface]);
            let right = conductance.get(index).copied().unwrap_or(0.0);
            diagonal[index] = lengths_km[index] + step_time_km2 * (left + right);
            rhs[index] = lengths_km[index] * profile[index];
            if !diagonal[index].is_finite() || !rhs[index].is_finite() {
                return Err(ConservativeSignedFluxProfileError::NumericalFailure);
            }
        }
        for (coefficient, &interface_conductance) in off_diagonal.iter_mut().zip(&conductance) {
            *coefficient = -step_time_km2 * interface_conductance;
            if !coefficient.is_finite() {
                return Err(ConservativeSignedFluxProfileError::NumericalFailure);
            }
        }

        // Thomas elimination for the symmetric tridiagonal finite-volume
        // system. Positive lengths and conductances make every pivot positive.
        for index in 1..profile.len() {
            let previous_pivot = diagonal[index - 1];
            if !previous_pivot.is_finite() || previous_pivot <= 0.0 {
                return Err(ConservativeSignedFluxProfileError::NumericalFailure);
            }
            let multiplier = off_diagonal[index - 1] / previous_pivot;
            diagonal[index] -= multiplier * off_diagonal[index - 1];
            rhs[index] -= multiplier * rhs[index - 1];
        }

        let last = profile.len() - 1;
        if !diagonal[last].is_finite() || diagonal[last] <= 0.0 {
            return Err(ConservativeSignedFluxProfileError::NumericalFailure);
        }
        profile[last] = rhs[last] / diagonal[last];
        for index in (0..last).rev() {
            if !diagonal[index].is_finite() || diagonal[index] <= 0.0 {
                return Err(ConservativeSignedFluxProfileError::NumericalFailure);
            }
            profile[index] =
                (rhs[index] - off_diagonal[index] * profile[index + 1]) / diagonal[index];
        }
        if !profile.iter().all(|value| value.is_finite()) {
            return Err(ConservativeSignedFluxProfileError::NumericalFailure);
        }
    }

    Ok(profile)
}

/// Global conservation record for [`conservative_signed_flux_front_rates_v0`].
#[cfg(feature = "research-landscape")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConservativeSignedFluxFrontLedgerV0 {
    pub processed_segment_count: usize,
    pub processed_edge_count: usize,
    pub untouched_edge_count: usize,
    pub input_signed_flux_km2_per_myr: f64,
    pub output_signed_flux_km2_per_myr: f64,
    pub input_positive_clipped_flux_km2_per_myr: f64,
    pub output_positive_clipped_flux_km2_per_myr: f64,
    pub closure_residual_km2_per_myr: f64,
}

/// Signed rates for every exact source edge plus the aggregation ledger.
#[cfg(feature = "research-landscape")]
#[derive(Clone, Debug, PartialEq)]
pub struct ConservativeSignedFluxFrontRatesV0 {
    pub signed_rates_km_per_myr: BTreeMap<BoundaryEdgeId, f64>,
    pub ledger: ConservativeSignedFluxFrontLedgerV0,
}

/// Failure to build a topology-aware signed-rate bridge.
#[cfg(feature = "research-landscape")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConservativeSignedFluxFrontErrorV0 {
    StructuralMountain(StructuralMountainError),
    SignedFluxProfile(ConservativeSignedFluxProfileError),
    MissingSourceEdge(BoundaryEdgeId),
    DuplicateProcessedEdge(BoundaryEdgeId),
}

#[cfg(feature = "research-landscape")]
impl std::fmt::Display for ConservativeSignedFluxFrontErrorV0 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

#[cfg(feature = "research-landscape")]
impl std::error::Error for ConservativeSignedFluxFrontErrorV0 {}

#[cfg(feature = "research-landscape")]
impl From<StructuralMountainError> for ConservativeSignedFluxFrontErrorV0 {
    fn from(value: StructuralMountainError) -> Self {
        Self::StructuralMountain(value)
    }
}

#[cfg(feature = "research-landscape")]
impl From<ConservativeSignedFluxProfileError> for ConservativeSignedFluxFrontErrorV0 {
    fn from(value: ConservativeSignedFluxProfileError) -> Self {
        Self::SignedFluxProfile(value)
    }
}

/// Aggregate signed rates independently along each emitted structural segment.
///
/// The existing structural compiler remains the authority for uninterrupted
/// open-path identity and ordering. This adapter smooths raw signed convergence
/// using exact source-edge lengths before a caller performs positive
/// classification. It never reads the compiler's opportunity allocation or
/// along-strike taper. Edges omitted by the compiler (including closed loops)
/// remain at their exact input rate and are counted as untouched.
#[cfg(feature = "research-landscape")]
pub fn conservative_signed_flux_front_rates_v0(
    fronts: &ConvergentFrontSet,
    sigma_km: f64,
    substeps: usize,
) -> Result<ConservativeSignedFluxFrontRatesV0, ConservativeSignedFluxFrontErrorV0> {
    // Validate the declared operator even when this set has no emitted path.
    conservative_signed_flux_profile_v0(&[], &[], sigma_km, substeps)?;
    let graph = compile_structural_mountain(fronts)?;
    let by_id: BTreeMap<_, _> = fronts.edges.iter().map(|edge| (edge.id, edge)).collect();
    let mut signed_rates_km_per_myr: BTreeMap<_, _> = by_id
        .iter()
        .map(|(&id, edge)| (id, f64::from(edge.convergence_km_per_myr)))
        .collect();
    let mut processed = BTreeSet::new();

    for segment in &graph.segments {
        let mut input_rates = Vec::with_capacity(segment.source_edges.len());
        let mut lengths_km = Vec::with_capacity(segment.source_edges.len());
        for &edge_id in &segment.source_edges {
            let edge = by_id.get(&edge_id).ok_or(
                ConservativeSignedFluxFrontErrorV0::MissingSourceEdge(edge_id),
            )?;
            input_rates.push(f64::from(edge.convergence_km_per_myr));
            lengths_km.push(f64::from(edge.length_km));
        }
        let output_rates =
            conservative_signed_flux_profile_v0(&input_rates, &lengths_km, sigma_km, substeps)?;
        for (&edge_id, output_rate) in segment.source_edges.iter().zip(output_rates) {
            if !processed.insert(edge_id) {
                return Err(ConservativeSignedFluxFrontErrorV0::DuplicateProcessedEdge(
                    edge_id,
                ));
            }
            *signed_rates_km_per_myr.get_mut(&edge_id).ok_or(
                ConservativeSignedFluxFrontErrorV0::MissingSourceEdge(edge_id),
            )? = output_rate;
        }
    }

    let mut input_signed_flux = 0.0;
    let mut output_signed_flux = 0.0;
    let mut input_positive_clipped_flux = 0.0;
    let mut output_positive_clipped_flux = 0.0;
    for (&edge_id, edge) in &by_id {
        let length_km = f64::from(edge.length_km);
        let input_rate = f64::from(edge.convergence_km_per_myr);
        let output_rate = signed_rates_km_per_myr[&edge_id];
        input_signed_flux += length_km * input_rate;
        output_signed_flux += length_km * output_rate;
        input_positive_clipped_flux += length_km * input_rate.max(0.0);
        output_positive_clipped_flux += length_km * output_rate.max(0.0);
    }

    Ok(ConservativeSignedFluxFrontRatesV0 {
        signed_rates_km_per_myr,
        ledger: ConservativeSignedFluxFrontLedgerV0 {
            processed_segment_count: graph.segments.len(),
            processed_edge_count: processed.len(),
            untouched_edge_count: by_id.len() - processed.len(),
            input_signed_flux_km2_per_myr: input_signed_flux,
            output_signed_flux_km2_per_myr: output_signed_flux,
            input_positive_clipped_flux_km2_per_myr: input_positive_clipped_flux,
            output_positive_clipped_flux_km2_per_myr: output_positive_clipped_flux,
            closure_residual_km2_per_myr: output_signed_flux - input_signed_flux,
        },
    })
}

/// Why valid source evidence could not become a finite parent segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuralMountainOmissionReason {
    ClosedLoop,
    ZeroOpportunity,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructuralMountainOmission {
    pub reason: StructuralMountainOmissionReason,
    pub source_edges: Vec<BoundaryEdgeId>,
    pub declared_opportunity_km2: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuralNodeKind {
    Tip,
    Transfer,
    Junction,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructuralNode {
    /// Exact product Voronoi vertex ID.
    pub id: u32,
    pub position: Vec3,
    pub kind: StructuralNodeKind,
    pub incident_segments: Vec<BoundaryEdgeId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuralLinkKind {
    Transfer,
    Junction,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuralLink {
    pub node_id: u32,
    pub segments: [BoundaryEdgeId; 2],
    pub kind: StructuralLinkKind,
}

/// One causally uniform, finite parent. Its ID is its minimum source edge.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralSegment {
    pub id: BoundaryEdgeId,
    pub source_edges: Vec<BoundaryEdgeId>,
    pub vertices_in_order: Vec<u32>,
    pub length_km: f32,
    pub episode_id: usize,
    pub regime: StructuralRegime,
    pub plate_pair: [usize; 2],
    pub crust_on_plate_pair: [CrustType; 2],
    pub subducting_plate: Option<usize>,
    pub receiving_plate: Option<usize>,
    pub declared_opportunity_km2: f64,
    /// Full-cosine value at each source edge midpoint before normalization.
    pub along_strike_taper: Vec<f64>,
    /// Per-edge share after taper and renormalization. This sums to the
    /// segment's declared opportunity.
    pub compiled_opportunity_km2: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpportunityLedgerEntry {
    pub segment_id: BoundaryEdgeId,
    pub declared_km2: f64,
    pub compiled_km2: f64,
    pub residual_km2: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpportunityLedger {
    /// Opportunity carried by every valid input front, including omissions.
    pub source_km2: f64,
    pub declared_km2: f64,
    pub compiled_km2: f64,
    pub omitted_km2: f64,
    pub residual_km2: f64,
    /// `(declared + omitted) - source`.
    pub accounting_residual_km2: f64,
    pub segments: Vec<OpportunityLedgerEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuralReadiness {
    NoFiniteParent,
    FiniteParentOnly,
    CausallySegmented,
    DisconnectedFiniteParents,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructuralMountainGraph {
    pub segments: Vec<StructuralSegment>,
    pub nodes: Vec<StructuralNode>,
    pub links: Vec<StructuralLink>,
    pub omissions: Vec<StructuralMountainOmission>,
    pub ledger: OpportunityLedger,
    pub readiness: StructuralReadiness,
}

/// Collect exact convergent-front evidence from the same boundary snapshot used
/// to construct `history`. Missing topology/history is a typed error, never a
/// degenerate midpoint fallback.
pub fn collect_convergent_fronts(
    tessellation: &Tessellation,
    boundaries: &[PlateBoundaryEdge],
    history: &TectonicHistory,
) -> Result<ConvergentFrontSet, StructuralMountainError> {
    let mut seen = HashSet::new();
    let mut degrees = BTreeMap::<u32, usize>::new();
    let mut exact_arcs = HashMap::<BoundaryEdgeId, ([u32; 2], [Vec3; 2])>::new();

    // Count every plate-boundary arm. Otherwise a convergent/convergent/
    // transform triple junction can be mistaken for an ordinary continuation.
    for boundary in boundaries {
        let id = BoundaryEdgeId::new(boundary.cell_a, boundary.cell_b);
        if boundary.cell_a >= boundary.cell_b {
            return Err(StructuralMountainError::NonCanonicalBoundary(id));
        }
        if !seen.insert(id) {
            return Err(StructuralMountainError::DuplicateBoundary(id));
        }
        let vertices = tessellation
            .shared_edge_vertices(boundary.cell_a, boundary.cell_b)
            .ok_or(StructuralMountainError::MissingSharedTopology(id))?;
        let endpoints = [
            tessellation.voronoi.vertices[vertices[0] as usize],
            tessellation.voronoi.vertices[vertices[1] as usize],
        ];
        for vertex in vertices {
            *degrees.entry(vertex).or_default() += 1;
        }
        exact_arcs.insert(id, (vertices, endpoints));
    }

    let mut edges = Vec::new();
    for boundary in boundaries {
        if boundary.kind != BoundaryKind::Convergent {
            continue;
        }
        let id = BoundaryEdgeId::new(boundary.cell_a, boundary.cell_b);
        let (vertices, endpoints) = exact_arcs[&id];
        let chord = (endpoints[0] - endpoints[1]).length();
        let arc_radians = 2.0 * (0.5 * chord.clamp(0.0, 2.0)).asin();
        let length_km = arc_radians * PLANET_RADIUS_KM;
        let midpoint_sum = endpoints[0] + endpoints[1];
        if !length_km.is_finite()
            || length_km <= 0.0
            || !endpoints.iter().all(|p| p.is_finite())
            || midpoint_sum.length_squared() <= 1e-12
        {
            return Err(StructuralMountainError::InvalidGeometry(id));
        }
        let midpoint = midpoint_sum.normalize();
        let episode = history
            .episode_for_edge(boundary.cell_a, boundary.cell_b)
            .ok_or(StructuralMountainError::MissingEpisode(id))?;
        let boundary_plate_pair = canonical_pair(boundary.plate_a, boundary.plate_b);
        if episode.kind != BoundaryKind::Convergent
            || [episode.plate_a, episode.plate_b] != boundary_plate_pair
        {
            return Err(StructuralMountainError::InconsistentEpisode(id));
        }
        let convergence = boundary.convergence_km_per_myr();
        let shear = boundary.shear_km_per_myr();
        let relative = boundary.relative_speed_km_per_myr();
        if !convergence.is_finite()
            || !shear.is_finite()
            || !relative.is_finite()
            || !episode.duration_myr.is_finite()
            || episode.duration_myr < 0.0
        {
            return Err(StructuralMountainError::InvalidKinematics(id));
        }
        let (regime, subducting_plate, receiving_plate) = match boundary.subduction {
            Some(SubductionPolarity::ASubducts) => (
                StructuralRegime::Subduction,
                Some(boundary.plate_a),
                Some(boundary.plate_b),
            ),
            Some(SubductionPolarity::BSubducts) => (
                StructuralRegime::Subduction,
                Some(boundary.plate_b),
                Some(boundary.plate_a),
            ),
            None => (StructuralRegime::Collision, None, None),
        };
        let opportunity = f64::from(length_km)
            * f64::from(convergence.max(0.0))
            * f64::from(episode.duration_myr);
        edges.push(ConvergentFrontEdge {
            id,
            cells: [boundary.cell_a, boundary.cell_b],
            vertices,
            endpoints,
            midpoint,
            length_km,
            plates: [boundary.plate_a, boundary.plate_b],
            crust: [boundary.type_a, boundary.type_b],
            regime,
            subducting_plate,
            receiving_plate,
            convergence_km_per_myr: convergence,
            shear_km_per_myr: shear,
            relative_speed_km_per_myr: relative,
            episode_id: episode.id,
            episode_duration_myr: episode.duration_myr,
            episode_normal_displacement_km: episode.integrated_normal_displacement_km,
            episode_shear_displacement_km: episode.integrated_shear_displacement_km,
            history_model: episode.model,
            shortening_area_opportunity_km2: opportunity,
        });
    }
    edges.sort_by_key(|edge| edge.id);
    Ok(ConvergentFrontSet {
        edges,
        all_boundary_vertex_degree: degrees,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CausalKey {
    episode_id: usize,
    regime: StructuralRegime,
    plate_pair: [usize; 2],
    crust: [u8; 2],
    subducting_plate: Option<usize>,
    receiving_plate: Option<usize>,
}

impl CausalKey {
    fn from_edge(edge: &ConvergentFrontEdge) -> Self {
        let (plate_pair, crust) = if edge.plates[0] <= edge.plates[1] {
            (edge.plates, edge.crust)
        } else {
            (
                [edge.plates[1], edge.plates[0]],
                [edge.crust[1], edge.crust[0]],
            )
        };
        Self {
            episode_id: edge.episode_id,
            regime: edge.regime,
            plate_pair,
            crust: [crust_code(crust[0]), crust_code(crust[1])],
            subducting_plate: edge.subducting_plate,
            receiving_plate: edge.receiving_plate,
        }
    }
}

fn crust_code(crust: CrustType) -> u8 {
    match crust {
        CrustType::Continental => 0,
        CrustType::Oceanic => 1,
    }
}

fn crust_from_code(code: u8) -> CrustType {
    match code {
        0 => CrustType::Continental,
        _ => CrustType::Oceanic,
    }
}

/// Compile causally uniform paths into finite parents and a budget-closed
/// along-strike opportunity allocation. This remains source-domain evidence;
/// it does not compile a terrain field.
pub fn compile_structural_mountain(
    fronts: &ConvergentFrontSet,
) -> Result<StructuralMountainGraph, StructuralMountainError> {
    let mut edges = fronts.edges.clone();
    edges.sort_by_key(|edge| edge.id);
    for pair in edges.windows(2) {
        if pair[0].id == pair[1].id {
            return Err(StructuralMountainError::DuplicateBoundary(pair[0].id));
        }
    }

    let mut positions = BTreeMap::<u32, Vec3>::new();
    let mut incident = BTreeMap::<u32, Vec<usize>>::new();
    for (index, edge) in edges.iter().enumerate() {
        validate_front(edge)?;
        for endpoint in 0..2 {
            let vertex = edge.vertices[endpoint];
            if let Some(existing) = positions.insert(vertex, edge.endpoints[endpoint]) {
                if existing.distance(edge.endpoints[endpoint]) > 1e-5 {
                    return Err(StructuralMountainError::InconsistentVertexPosition(vertex));
                }
            }
            incident.entry(vertex).or_default().push(index);
        }
    }
    for values in incident.values_mut() {
        values.sort_by_key(|&index| edges[index].id);
    }
    for (&vertex, retained) in &incident {
        if fronts
            .all_boundary_vertex_degree
            .get(&vertex)
            .is_none_or(|&degree| degree < retained.len())
        {
            return Err(StructuralMountainError::InvalidBoundaryDegree(vertex));
        }
    }

    let keys: Vec<_> = edges.iter().map(CausalKey::from_edge).collect();
    let can_continue = |vertex: u32, left: usize, right: usize| {
        fronts
            .all_boundary_vertex_degree
            .get(&vertex)
            .copied()
            .unwrap_or(0)
            == 2
            && incident.get(&vertex).is_some_and(|at| at.len() == 2)
            && keys[left] == keys[right]
    };

    let mut adjacency = vec![Vec::<(usize, u32)>::new(); edges.len()];
    for (&vertex, at) in &incident {
        if at.len() == 2 && can_continue(vertex, at[0], at[1]) {
            adjacency[at[0]].push((at[1], vertex));
            adjacency[at[1]].push((at[0], vertex));
        }
    }

    let mut components = Vec::<Vec<usize>>::new();
    let mut unseen: BTreeSet<usize> = (0..edges.len()).collect();
    while let Some(&start) = unseen.first() {
        let mut stack = vec![start];
        unseen.remove(&start);
        let mut component = Vec::new();
        while let Some(current) = stack.pop() {
            component.push(current);
            let mut next: Vec<_> = adjacency[current].iter().map(|&(i, _)| i).collect();
            next.sort_by_key(|&i| edges[i].id);
            for candidate in next.into_iter().rev() {
                if unseen.remove(&candidate) {
                    stack.push(candidate);
                }
            }
        }
        component.sort_by_key(|&index| edges[index].id);
        components.push(component);
    }
    components.sort_by_key(|component| edges[component[0]].id);

    let mut segments = Vec::new();
    let mut omissions = Vec::new();
    for component in components {
        let component_opportunity = component
            .iter()
            .map(|&i| edges[i].shortening_area_opportunity_km2)
            .sum::<f64>();
        let terminals: Vec<_> = component
            .iter()
            .copied()
            .filter(|&index| adjacency[index].len() < 2)
            .collect();
        if terminals.is_empty() {
            omissions.push(StructuralMountainOmission {
                reason: StructuralMountainOmissionReason::ClosedLoop,
                source_edges: component.iter().map(|&i| edges[i].id).collect(),
                declared_opportunity_km2: component_opportunity,
            });
            continue;
        }
        let mut terminal_choices = Vec::new();
        for &edge_index in &terminals {
            for &vertex in &edges[edge_index].vertices {
                let continues_here = adjacency[edge_index]
                    .iter()
                    .any(|&(_, shared)| shared == vertex);
                if !continues_here {
                    terminal_choices.push((edges[edge_index].id, vertex, edge_index));
                }
            }
        }
        terminal_choices.sort();
        let (_, start_vertex, mut current) = terminal_choices[0];
        let mut previous = None;
        let mut current_vertex = start_vertex;
        let mut ordered = Vec::with_capacity(component.len());
        let mut vertices = vec![start_vertex];
        loop {
            ordered.push(current);
            let edge = &edges[current];
            let exit_vertex = if edge.vertices[0] == current_vertex {
                edge.vertices[1]
            } else {
                edge.vertices[0]
            };
            vertices.push(exit_vertex);
            let next = adjacency[current]
                .iter()
                .find(|&&(candidate, shared)| Some(candidate) != previous && shared == exit_vertex)
                .map(|&(candidate, _)| candidate);
            match next {
                Some(candidate) => {
                    previous = Some(current);
                    current = candidate;
                    current_vertex = exit_vertex;
                }
                None => break,
            }
        }
        debug_assert_eq!(ordered.len(), component.len());
        let declared = component_opportunity;
        if !declared.is_finite() || declared <= 0.0 {
            omissions.push(StructuralMountainOmission {
                reason: StructuralMountainOmissionReason::ZeroOpportunity,
                source_edges: component.iter().map(|&i| edges[i].id).collect(),
                declared_opportunity_km2: component_opportunity,
            });
            continue;
        }
        let length = ordered
            .iter()
            .map(|&i| edges[i].length_km as f64)
            .sum::<f64>();
        let mut distance = 0.0;
        let mut taper = Vec::with_capacity(ordered.len());
        let mut weighted = Vec::with_capacity(ordered.len());
        for &index in &ordered {
            let edge_length = f64::from(edges[index].length_km);
            let u = (distance + 0.5 * edge_length) / length;
            let value = 0.5 * (1.0 - (TAU * u).cos());
            taper.push(value);
            weighted.push(value * edges[index].shortening_area_opportunity_km2);
            distance += edge_length;
        }
        let weighted_total = weighted.iter().sum::<f64>();
        if !weighted_total.is_finite() || weighted_total <= 0.0 {
            return Err(StructuralMountainError::EmptyTaperSupport(
                component.iter().map(|&i| edges[i].id).min().unwrap(),
            ));
        }
        let compiled: Vec<_> = weighted
            .into_iter()
            .map(|value| value * declared / weighted_total)
            .collect();
        let key = keys[ordered[0]];
        let id = component.iter().map(|&i| edges[i].id).min().unwrap();
        segments.push(StructuralSegment {
            id,
            source_edges: ordered.iter().map(|&i| edges[i].id).collect(),
            vertices_in_order: vertices,
            length_km: length as f32,
            episode_id: key.episode_id,
            regime: key.regime,
            plate_pair: key.plate_pair,
            crust_on_plate_pair: [crust_from_code(key.crust[0]), crust_from_code(key.crust[1])],
            subducting_plate: key.subducting_plate,
            receiving_plate: key.receiving_plate,
            declared_opportunity_km2: declared,
            along_strike_taper: taper,
            compiled_opportunity_km2: compiled,
        });
    }
    segments.sort_by_key(|segment| segment.id);
    omissions.sort_by_key(|omission| omission.source_edges[0]);

    let mut node_incidence = BTreeMap::<u32, Vec<BoundaryEdgeId>>::new();
    for segment in &segments {
        for vertex in [
            segment.vertices_in_order[0],
            *segment.vertices_in_order.last().unwrap(),
        ] {
            node_incidence.entry(vertex).or_default().push(segment.id);
        }
    }
    let mut nodes = Vec::new();
    let mut links = Vec::new();
    for (vertex, mut segment_ids) in node_incidence {
        segment_ids.sort();
        segment_ids.dedup();
        let complete_degree = fronts.all_boundary_vertex_degree[&vertex];
        let kind = match (complete_degree, segment_ids.len()) {
            (3.., _) | (_, 3..) => StructuralNodeKind::Junction,
            (_, 2) => StructuralNodeKind::Transfer,
            _ => StructuralNodeKind::Tip,
        };
        let link_kind = match kind {
            StructuralNodeKind::Transfer => Some(StructuralLinkKind::Transfer),
            StructuralNodeKind::Junction => Some(StructuralLinkKind::Junction),
            StructuralNodeKind::Tip => None,
        };
        if let Some(link_kind) = link_kind {
            for left in 0..segment_ids.len() {
                for right in (left + 1)..segment_ids.len() {
                    links.push(StructuralLink {
                        node_id: vertex,
                        segments: [segment_ids[left], segment_ids[right]],
                        kind: link_kind,
                    });
                }
            }
        }
        nodes.push(StructuralNode {
            id: vertex,
            position: positions[&vertex],
            kind,
            incident_segments: segment_ids,
        });
    }

    let ledger_entries: Vec<_> = segments
        .iter()
        .map(|segment| {
            let compiled = segment.compiled_opportunity_km2.iter().sum::<f64>();
            OpportunityLedgerEntry {
                segment_id: segment.id,
                declared_km2: segment.declared_opportunity_km2,
                compiled_km2: compiled,
                residual_km2: compiled - segment.declared_opportunity_km2,
            }
        })
        .collect();
    let declared = ledger_entries.iter().map(|entry| entry.declared_km2).sum();
    let compiled = ledger_entries.iter().map(|entry| entry.compiled_km2).sum();
    let source = edges
        .iter()
        .map(|edge| edge.shortening_area_opportunity_km2)
        .sum::<f64>();
    let omitted = omissions
        .iter()
        .map(|omission| omission.declared_opportunity_km2)
        .sum::<f64>();
    let readiness = match segments.len() {
        0 => StructuralReadiness::NoFiniteParent,
        1 => StructuralReadiness::FiniteParentOnly,
        count if linked_segment_count(&segments, &links) == count => {
            StructuralReadiness::CausallySegmented
        }
        _ => StructuralReadiness::DisconnectedFiniteParents,
    };
    Ok(StructuralMountainGraph {
        segments,
        nodes,
        links,
        omissions,
        ledger: OpportunityLedger {
            source_km2: source,
            declared_km2: declared,
            compiled_km2: compiled,
            omitted_km2: omitted,
            residual_km2: compiled - declared,
            accounting_residual_km2: declared + omitted - source,
            segments: ledger_entries,
        },
        readiness,
    })
}

fn validate_front(edge: &ConvergentFrontEdge) -> Result<(), StructuralMountainError> {
    if edge.id != BoundaryEdgeId::new(edge.cells[0], edge.cells[1])
        || edge.cells[0] >= edge.cells[1]
    {
        return Err(StructuralMountainError::NonCanonicalBoundary(edge.id));
    }
    if !edge.length_km.is_finite()
        || edge.length_km <= 0.0
        || !edge.shortening_area_opportunity_km2.is_finite()
        || edge.shortening_area_opportunity_km2 < 0.0
        || !edge.endpoints.iter().all(|point| point.is_finite())
        || !edge.midpoint.is_finite()
        || edge.vertices[0] == edge.vertices[1]
    {
        return Err(StructuralMountainError::InvalidGeometry(edge.id));
    }
    if !edge.convergence_km_per_myr.is_finite()
        || !edge.shear_km_per_myr.is_finite()
        || !edge.relative_speed_km_per_myr.is_finite()
        || !edge.episode_duration_myr.is_finite()
        || edge.episode_duration_myr < 0.0
        || !edge.episode_normal_displacement_km.is_finite()
        || !edge.episode_shear_displacement_km.is_finite()
    {
        return Err(StructuralMountainError::InvalidKinematics(edge.id));
    }
    let valid_sides = match edge.regime {
        StructuralRegime::Collision => {
            edge.subducting_plate.is_none() && edge.receiving_plate.is_none()
        }
        StructuralRegime::Subduction => {
            edge.subducting_plate.is_some()
                && edge.receiving_plate.is_some()
                && edge.subducting_plate != edge.receiving_plate
                && edge.plates.contains(&edge.subducting_plate.unwrap())
                && edge.plates.contains(&edge.receiving_plate.unwrap())
        }
    };
    if !valid_sides {
        return Err(StructuralMountainError::InvalidSideSemantics(edge.id));
    }
    Ok(())
}

fn canonical_pair(a: usize, b: usize) -> [usize; 2] {
    [a.min(b), a.max(b)]
}

fn linked_segment_count(segments: &[StructuralSegment], links: &[StructuralLink]) -> usize {
    let Some(first) = segments.first() else {
        return 0;
    };
    let mut adjacency = BTreeMap::<BoundaryEdgeId, Vec<BoundaryEdgeId>>::new();
    for link in links {
        adjacency
            .entry(link.segments[0])
            .or_default()
            .push(link.segments[1]);
        adjacency
            .entry(link.segments[1])
            .or_default()
            .push(link.segments[0]);
    }
    let mut seen = BTreeSet::from([first.id]);
    let mut stack = vec![first.id];
    while let Some(current) = stack.pop() {
        for &next in adjacency.get(&current).into_iter().flatten() {
            if seen.insert(next) {
                stack.push(next);
            }
        }
    }
    seen.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{collect_plate_boundaries, World};

    fn edge(
        cells: [usize; 2],
        vertices: [u32; 2],
        episode: usize,
        opportunity: f64,
    ) -> ConvergentFrontEdge {
        let position = |vertex: u32| Vec3::new(vertex as f32, 1.0, 0.0).normalize();
        ConvergentFrontEdge {
            id: BoundaryEdgeId::new(cells[0], cells[1]),
            cells,
            vertices,
            endpoints: [position(vertices[0]), position(vertices[1])],
            midpoint: (position(vertices[0]) + position(vertices[1])).normalize(),
            length_km: 10.0,
            plates: [2, 7],
            crust: [CrustType::Continental, CrustType::Continental],
            regime: StructuralRegime::Collision,
            subducting_plate: None,
            receiving_plate: None,
            convergence_km_per_myr: 1.0,
            shear_km_per_myr: 0.0,
            relative_speed_km_per_myr: 1.0,
            episode_id: episode,
            episode_duration_myr: 1.0,
            episode_normal_displacement_km: 1.0,
            episode_shear_displacement_km: 0.0,
            history_model: HistoryModel::StationaryTopologyConstantVelocity,
            shortening_area_opportunity_km2: opportunity,
        }
    }

    #[cfg(feature = "research-landscape")]
    fn flux_edge(
        cells: [usize; 2],
        vertices: [u32; 2],
        episode: usize,
        length_km: f32,
        signed_rate_km_per_myr: f32,
    ) -> ConvergentFrontEdge {
        let mut result = edge(cells, vertices, episode, 0.0);
        result.length_km = length_km;
        result.convergence_km_per_myr = signed_rate_km_per_myr;
        result.relative_speed_km_per_myr = signed_rate_km_per_myr.abs();
        result.episode_normal_displacement_km = signed_rate_km_per_myr;
        result.shortening_area_opportunity_km2 =
            f64::from(length_km) * f64::from(signed_rate_km_per_myr.max(0.0));
        result
    }

    fn set(edges: Vec<ConvergentFrontEdge>) -> ConvergentFrontSet {
        let mut degree = BTreeMap::new();
        for edge in &edges {
            for vertex in edge.vertices {
                *degree.entry(vertex).or_default() += 1;
            }
        }
        ConvergentFrontSet {
            edges,
            all_boundary_vertex_degree: degree,
        }
    }

    #[test]
    fn uniform_chain_is_one_finite_parent_and_closes() {
        let graph = compile_structural_mountain(&set(vec![
            edge([0, 10], [1, 2], 4, 10.0),
            edge([1, 11], [2, 3], 4, 10.0),
            edge([2, 12], [3, 4], 4, 10.0),
        ]))
        .unwrap();
        assert_eq!(graph.readiness, StructuralReadiness::FiniteParentOnly);
        assert_eq!(graph.segments.len(), 1);
        assert_eq!(graph.nodes.len(), 2);
        assert!(graph
            .nodes
            .iter()
            .all(|node| node.kind == StructuralNodeKind::Tip));
        assert_eq!(graph.segments[0].vertices_in_order, vec![1, 2, 3, 4]);
        assert!((graph.segments[0].along_strike_taper[1] - 1.0).abs() < 1e-12);
        assert!(graph.segments[0].along_strike_taper[0] < 0.26);
        assert!(graph.ledger.residual_km2.abs() < 1e-12);
        for (actual, expected) in graph.segments[0]
            .compiled_opportunity_km2
            .iter()
            .zip([5.0, 20.0, 5.0])
        {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn episode_change_creates_linked_finite_segments() {
        let graph = compile_structural_mountain(&set(vec![
            edge([0, 10], [1, 2], 4, 10.0),
            edge([1, 11], [2, 3], 4, 10.0),
            edge([2, 12], [3, 4], 5, 20.0),
        ]))
        .unwrap();
        assert_eq!(graph.readiness, StructuralReadiness::CausallySegmented);
        assert_eq!(graph.segments.len(), 2);
        assert_eq!(graph.links.len(), 1);
        assert_eq!(graph.links[0].node_id, 3);
        assert_eq!(graph.links[0].kind, StructuralLinkKind::Transfer);
        assert!(graph.ledger.residual_km2.abs() < 1e-12);
    }

    #[test]
    fn input_and_endpoint_reversal_canonicalize_to_same_graph() {
        let original = set(vec![
            edge([0, 10], [1, 2], 4, 10.0),
            edge([1, 11], [2, 3], 4, 20.0),
        ]);
        let mut reversed_edges = original.edges.clone();
        reversed_edges.reverse();
        for edge in &mut reversed_edges {
            edge.vertices.reverse();
            edge.endpoints.reverse();
        }
        let reversed = ConvergentFrontSet {
            edges: reversed_edges,
            all_boundary_vertex_degree: original.all_boundary_vertex_degree.clone(),
        };
        assert_eq!(
            compile_structural_mountain(&original).unwrap(),
            compile_structural_mountain(&reversed).unwrap()
        );
    }

    #[test]
    fn closed_loop_and_zero_opportunity_are_explicit_omissions() {
        let graph = compile_structural_mountain(&set(vec![
            edge([0, 10], [1, 2], 4, 10.0),
            edge([1, 11], [2, 3], 4, 10.0),
            edge([2, 12], [3, 1], 4, 10.0),
            edge([3, 13], [4, 5], 7, 0.0),
        ]))
        .unwrap();
        assert_eq!(graph.segments.len(), 0);
        assert_eq!(graph.omissions.len(), 2);
        assert_eq!(
            graph.omissions[0].reason,
            StructuralMountainOmissionReason::ClosedLoop
        );
        assert_eq!(
            graph.omissions[1].reason,
            StructuralMountainOmissionReason::ZeroOpportunity
        );
        assert_eq!(graph.readiness, StructuralReadiness::NoFiniteParent);
        assert_eq!(graph.ledger.source_km2, 30.0);
        assert_eq!(graph.ledger.omitted_km2, 30.0);
        assert!(graph.ledger.accounting_residual_km2.abs() < 1e-12);
    }

    #[test]
    fn collision_and_subduction_side_semantics_are_not_interchangeable() {
        let mut invalid = edge([0, 10], [1, 2], 4, 10.0);
        invalid.receiving_plate = Some(7);
        assert_eq!(
            compile_structural_mountain(&set(vec![invalid])).unwrap_err(),
            StructuralMountainError::InvalidSideSemantics(BoundaryEdgeId::new(0, 10))
        );

        let mut subduction = edge([0, 10], [1, 2], 4, 10.0);
        subduction.regime = StructuralRegime::Subduction;
        subduction.crust = [CrustType::Oceanic, CrustType::Continental];
        subduction.subducting_plate = Some(2);
        subduction.receiving_plate = Some(7);
        assert!(compile_structural_mountain(&set(vec![subduction])).is_ok());
    }

    #[test]
    fn complete_boundary_degree_splits_a_hidden_third_arm() {
        let mut fronts = set(vec![
            edge([0, 10], [1, 2], 4, 10.0),
            edge([1, 11], [2, 3], 4, 10.0),
        ]);
        fronts.all_boundary_vertex_degree.insert(2, 3);
        let graph = compile_structural_mountain(&fronts).unwrap();
        assert_eq!(graph.segments.len(), 2);
        assert_eq!(
            graph.nodes.iter().find(|node| node.id == 2).unwrap().kind,
            StructuralNodeKind::Junction
        );
        assert_eq!(graph.links[0].kind, StructuralLinkKind::Junction);
    }

    #[test]
    fn disconnected_finite_parents_are_not_called_causal_segmentation() {
        let graph = compile_structural_mountain(&set(vec![
            edge([0, 10], [1, 2], 4, 10.0),
            edge([1, 11], [3, 4], 5, 10.0),
        ]))
        .unwrap();
        assert_eq!(
            graph.readiness,
            StructuralReadiness::DisconnectedFiniteParents
        );
    }

    #[test]
    fn generated_product_inputs_collect_exact_finite_fronts() {
        let mut world = World::new(12_345, 256, 0);
        world.generate_plates(6);
        world.generate_crust();
        world.generate_dynamics();
        world.generate_features();
        let boundaries = collect_plate_boundaries(
            &world.tessellation,
            world.plates.as_ref().unwrap(),
            world.crust.as_ref().unwrap(),
            world.dynamics.as_ref().unwrap(),
        );
        let expected = boundaries
            .iter()
            .filter(|edge| edge.kind == BoundaryKind::Convergent)
            .count();
        let fronts = collect_convergent_fronts(
            &world.tessellation,
            &boundaries,
            world.tectonic_history.as_ref().unwrap(),
        )
        .unwrap();
        assert_eq!(fronts.edges.len(), expected);
        assert!(expected > 0);
        assert!(fronts.edges.iter().all(|edge| {
            edge.length_km.is_finite()
                && edge.length_km > 0.0
                && edge.shortening_area_opportunity_km2.is_finite()
        }));
        let graph = compile_structural_mountain(&fronts).unwrap();
        assert!(graph.ledger.residual_km2.abs() <= 1e-9 * graph.ledger.declared_km2.max(1.0));
    }

    #[cfg(feature = "research-landscape")]
    fn weighted_signed_flux(rates: &[f64], lengths: &[f64]) -> f64 {
        rates
            .iter()
            .zip(lengths)
            .map(|(&rate, &length)| rate * length)
            .sum()
    }

    #[cfg(feature = "research-landscape")]
    fn weighted_positive_flux(rates: &[f64], lengths: &[f64]) -> f64 {
        rates
            .iter()
            .zip(lengths)
            .map(|(&rate, &length)| rate.max(0.0) * length)
            .sum()
    }

    #[cfg(feature = "research-landscape")]
    fn total_variation(rates: &[f64]) -> f64 {
        rates.windows(2).map(|pair| (pair[1] - pair[0]).abs()).sum()
    }

    #[cfg(feature = "research-landscape")]
    #[test]
    fn signed_flux_diffusion_preserves_an_irregular_constant_profile() {
        let lengths = [3.0, 17.0, 5.0, 29.0, 11.0];
        let rates = [2.75; 5];
        let result = conservative_signed_flux_profile_v0(&rates, &lengths, 127.0, 4).unwrap();
        assert!(result.iter().all(|&value| (value - 2.75).abs() < 1e-12));
        assert!(
            (weighted_signed_flux(&result, &lengths) - weighted_signed_flux(&rates, &lengths))
                .abs()
                < 1e-11
        );
    }

    #[cfg(feature = "research-landscape")]
    #[test]
    fn signed_flux_diffusion_closes_and_reduces_rectification_and_variation() {
        let lengths = [3.0, 11.0, 5.0, 17.0, 7.0, 23.0];
        let rates = [4.0, -3.0, 5.0, -2.0, 1.0, -0.5];
        let result = conservative_signed_flux_profile_v0(&rates, &lengths, 20.0, 3).unwrap();
        let before = weighted_signed_flux(&rates, &lengths);
        let after = weighted_signed_flux(&result, &lengths);
        assert!((after - before).abs() <= 1e-12 * before.abs().max(1.0));
        assert!(
            weighted_positive_flux(&result, &lengths) < weighted_positive_flux(&rates, &lengths)
        );
        assert!(total_variation(&result) < total_variation(&rates));
        assert!(result.iter().all(|&value| (-3.0..=5.0).contains(&value)));
    }

    #[cfg(feature = "research-landscape")]
    #[test]
    fn signed_flux_diffusion_is_reversal_invariant_to_roundoff() {
        let lengths = [2.0, 13.0, 7.0, 19.0, 5.0];
        let rates = [-1.0, 4.0, -2.0, 3.0, 0.5];
        let forward = conservative_signed_flux_profile_v0(&rates, &lengths, 31.0, 5).unwrap();
        let mut reversed_lengths = lengths;
        let mut reversed_rates = rates;
        reversed_lengths.reverse();
        reversed_rates.reverse();
        let mut reversed =
            conservative_signed_flux_profile_v0(&reversed_rates, &reversed_lengths, 31.0, 5)
                .unwrap();
        reversed.reverse();
        for (&left, &right) in forward.iter().zip(&reversed) {
            assert!((left - right).abs() < 1e-12);
        }
    }

    #[cfg(feature = "research-landscape")]
    #[test]
    fn signed_flux_diffusion_has_expected_two_cell_solution_and_is_positive_linear() {
        let fixture =
            conservative_signed_flux_profile_v0(&[1.0, 0.0], &[1.0, 3.0], 2.0_f64.sqrt(), 1)
                .unwrap();
        assert!((fixture[0] - 0.7).abs() < 1e-15);
        assert!((fixture[1] - 0.1).abs() < 1e-15);

        let lengths = [2.0, 13.0, 7.0, 19.0, 5.0];
        let impulse = [0.0, 0.0, 4.0, 0.0, 0.0];
        let positive = conservative_signed_flux_profile_v0(&impulse, &lengths, 9.0, 3).unwrap();
        assert!(positive.iter().all(|&value| (0.0..=4.0).contains(&value)));

        let negative_impulse = impulse.map(|value| -value);
        let negative =
            conservative_signed_flux_profile_v0(&negative_impulse, &lengths, 9.0, 3).unwrap();
        for (&positive_value, &negative_value) in positive.iter().zip(&negative) {
            assert!((positive_value + negative_value).abs() < 1e-14);
        }
    }

    #[cfg(feature = "research-landscape")]
    #[test]
    fn signed_flux_diffusion_zero_sigma_and_bad_inputs_are_explicit() {
        let rates = [-0.0, 2.0, -3.0];
        let lengths = [2.0, 5.0, 11.0];
        assert_eq!(
            conservative_signed_flux_profile_v0(&rates, &lengths, 0.0, 0).unwrap(),
            rates
        );
        assert_eq!(
            conservative_signed_flux_profile_v0(&rates[..2], &lengths, 1.0, 1),
            Err(ConservativeSignedFluxProfileError::LengthRateCountMismatch)
        );
        assert_eq!(
            conservative_signed_flux_profile_v0(&rates, &[2.0, 0.0, 11.0], 1.0, 1),
            Err(ConservativeSignedFluxProfileError::InvalidLength(1))
        );
        assert_eq!(
            conservative_signed_flux_profile_v0(&[0.0, f64::NAN, 1.0], &lengths, 1.0, 1),
            Err(ConservativeSignedFluxProfileError::InvalidRate(1))
        );
        assert_eq!(
            conservative_signed_flux_profile_v0(&rates, &lengths, -1.0, 1),
            Err(ConservativeSignedFluxProfileError::InvalidSigmaKm)
        );
        assert_eq!(
            conservative_signed_flux_profile_v0(&rates, &lengths, 1.0, 0),
            Err(ConservativeSignedFluxProfileError::InvalidSubsteps)
        );
    }

    #[cfg(feature = "research-landscape")]
    #[test]
    fn signed_flux_front_bridge_closes_mixed_irregular_chain_and_reduces_rectification() {
        let fronts = set(vec![
            flux_edge([0, 10], [1, 2], 4, 3.0, 4.0),
            flux_edge([1, 11], [2, 3], 4, 11.0, -3.0),
            flux_edge([2, 12], [3, 4], 4, 5.0, 5.0),
            flux_edge([3, 13], [4, 5], 4, 17.0, -2.0),
            flux_edge([4, 14], [5, 6], 4, 7.0, 1.0),
            flux_edge([5, 15], [6, 7], 4, 23.0, -0.5),
        ]);
        let result = conservative_signed_flux_front_rates_v0(&fronts, 20.0, 3).unwrap();
        assert_eq!(result.signed_rates_km_per_myr.len(), fronts.edges.len());
        assert_eq!(result.ledger.processed_segment_count, 1);
        assert_eq!(result.ledger.processed_edge_count, fronts.edges.len());
        assert_eq!(result.ledger.untouched_edge_count, 0);
        assert!(
            result.ledger.closure_residual_km2_per_myr.abs()
                <= 1e-12 * result.ledger.input_signed_flux_km2_per_myr.abs().max(1.0)
        );
        assert!(
            result.ledger.output_positive_clipped_flux_km2_per_myr
                < result.ledger.input_positive_clipped_flux_km2_per_myr
        );
    }

    #[cfg(feature = "research-landscape")]
    #[test]
    fn signed_flux_front_bridge_does_not_exchange_across_episode_transfer() {
        let fronts = set(vec![
            flux_edge([0, 10], [1, 2], 4, 1.0, 2.0),
            flux_edge([1, 11], [2, 3], 5, 3.0, 0.5),
        ]);
        let result = conservative_signed_flux_front_rates_v0(&fronts, 127.0, 4).unwrap();
        assert_eq!(result.ledger.processed_segment_count, 2);
        assert_eq!(result.ledger.processed_edge_count, 2);
        for edge in &fronts.edges {
            assert_eq!(
                result.signed_rates_km_per_myr[&edge.id],
                f64::from(edge.convergence_km_per_myr)
            );
        }
    }

    #[cfg(feature = "research-landscape")]
    #[test]
    fn signed_flux_front_bridge_leaves_closed_loop_omission_unchanged() {
        let fronts = set(vec![
            flux_edge([0, 10], [1, 2], 4, 3.0, 1.0),
            flux_edge([1, 11], [2, 3], 4, 5.0, -2.0),
            flux_edge([2, 12], [3, 1], 4, 7.0, 3.0),
        ]);
        let result = conservative_signed_flux_front_rates_v0(&fronts, 127.0, 4).unwrap();
        assert_eq!(result.ledger.processed_segment_count, 0);
        assert_eq!(result.ledger.processed_edge_count, 0);
        assert_eq!(result.ledger.untouched_edge_count, 3);
        for edge in &fronts.edges {
            assert_eq!(
                result.signed_rates_km_per_myr[&edge.id],
                f64::from(edge.convergence_km_per_myr)
            );
        }
        assert_eq!(
            result.ledger.input_signed_flux_km2_per_myr,
            result.ledger.output_signed_flux_km2_per_myr
        );
        assert_eq!(result.ledger.closure_residual_km2_per_myr, 0.0);
    }

    #[cfg(feature = "research-landscape")]
    #[test]
    fn signed_flux_front_bridge_is_deterministic_under_input_permutation() {
        let fronts = set(vec![
            flux_edge([0, 10], [1, 2], 4, 3.0, 4.0),
            flux_edge([1, 11], [2, 3], 4, 11.0, -3.0),
            flux_edge([2, 12], [3, 4], 4, 5.0, 5.0),
        ]);
        let mut permuted = fronts.clone();
        permuted.edges.reverse();
        assert_eq!(
            conservative_signed_flux_front_rates_v0(&fronts, 31.0, 5).unwrap(),
            conservative_signed_flux_front_rates_v0(&permuted, 31.0, 5).unwrap()
        );
    }
}
