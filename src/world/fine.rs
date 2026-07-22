//! Adaptive fine mesh refinement for Stage 3 hydrology and erosion.

use std::time::Instant;

use glam::Vec3;
use kiddo::{ImmutableKdTree, KdTree, SquaredEuclidean};
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
#[cfg(not(feature = "single-threaded"))]
use rayon::prelude::*;

use super::boundary::{collect_plate_boundaries, BoundaryKind, SubductionPolarity};
use super::constants::*;
use super::dynamics::Dynamics;
use super::elevation::{coarse_elevation_fields, isostasy_slope, ElevationFields, OrogenModel};
use super::erosion::ErosionParams;
#[cfg(feature = "research-landscape")]
use super::erosion::{
    FiniteAgeFluxModel, LegacyBudgetOpportunityAuditV0, LegacyBudgetOpportunityErrorV0,
};
use super::features::build_cell_pair_edge_endpoints;
use super::fine_cache::{self, FineCacheMode};
#[cfg(feature = "research-landscape")]
use super::structural_mountain::{
    collect_convergent_fronts, conservative_signed_flux_front_rates_v0,
    ConservativeSignedFluxFrontErrorV0,
};
use super::water::connected_ocean_cells;
use super::{
    Atmosphere, CellWaterState, Crust, Elevation, FeatureFields, FineCacheOutcome, FineCacheRecord,
    Hydrology, Plates, Tessellation,
};
#[cfg(feature = "research-landscape")]
use super::{CellEdgeId, RegionalDeformationRasterV0, TectonicHistory, RDS0_FRAME_COUNT};

type CoarseTree = ImmutableKdTree<f32, 3>;

/// Runtime-tunable knobs for the fine-mesh areal density prior (defaults from the
/// `FINE_*` consts). Lets tools sweep the ocean/plains/mountain cell-size budget
/// and the demand blend without a recompile — mirrors [`ErosionParams`]. Changing
/// any field changes the sampled mesh, so it is part of the fine-base cache key.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct FineDensityParams {
    pub plains_km: f32,
    pub mountain_km: f32,
    pub ocean_km: f32,
    pub exponent: f32,
    pub slope_weight: f32,
    pub flow_weight: f32,
    pub activity_weight: f32,
}

impl Default for FineDensityParams {
    fn default() -> Self {
        Self {
            plains_km: FINE_PLAINS_CELL_KM,
            mountain_km: FINE_MOUNTAIN_CELL_KM,
            ocean_km: FINE_OCEAN_CELL_KM,
            exponent: FINE_DENSITY_FEATURE_EXPONENT,
            slope_weight: FINE_SLOPE_DENSITY_WEIGHT,
            flow_weight: FINE_FLOW_DENSITY_WEIGHT,
            activity_weight: FINE_ACTIVITY_DENSITY_WEIGHT,
        }
    }
}

/// Runtime-tunable knobs for the pre-erosion fine STRUCTURAL relief (erosion-v2
/// Phase 1 / P1a). Mirrors [`FineDensityParams`]: these shape `FineBase`'s
/// `base_elevation` (the substrate erosion carves), so they are part of the fine-
/// base cache key, NOT erosion-stage knobs.
///
/// `fault_scarp_height` migrated here from [`ErosionParams`] (decision A in
/// docs/archive/specs/erosion-fine-synthesis.md): structural relief is part of the
/// pre-erosion base — applying it in the erosion stage left terminal-lake base
/// levels computed on an unfaulted base, and made the scarp a temporal knob over a
/// structural input. Moving it here also lets the disk cache key see it (a sweep
/// of `fault_scarp` / `interior_relief` now regenerates the base per value).
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct FineStructureParams {
    /// Fault range-front scarp relief imposed on the base. 0 = off.
    pub fault_scarp_height: f32,
    /// Amplitude of the zero-mean interior structural relief (fault-block / fold
    /// grain) that breaks the flat interpolated orogen summit so erosion has a
    /// gradient to organize. 0 = off (pure interpolant). The master P1a knob.
    pub interior_relief: f32,
    /// Blend (0..1) of strike-banded grain (aligned to the nearest orogen front)
    /// vs P1a isotropic grain, faded by distance to the front (P1b). 0 = pure
    /// isotropic. The master P1b knob.
    pub front_strike_weight: f32,
    /// Strength of the active/passive margin contrast: relief is sharpened toward an
    /// active (convergent) coast and damped toward a passive one (P1c). 0 = off (P1b).
    pub margin_contrast: f32,
    /// Emergent-orogens demotion fraction (erosion-v3): fraction of the solved
    /// tectonic crust load removed from the static base envelope and rebuilt by active
    /// uplift during erosion. 0 = off (painted/postprocessor path); >0 = emergent.
    pub emergent_lambda: f32,
    /// O0 (orogen-structure): blend of the structured (asymmetric + segmented) emergent
    /// uplift shape vs the uniform v3 rebuild. 0 = uniform; 1 = fully structured.
    pub emergent_structured: f32,
    /// Candidate A meso-band modulation depth for the emergent uplift shape. 0 = off.
    pub meso_relief: f32,
    /// Candidate A' meso-band relief painted into the base elevation. 0 = off.
    pub meso_base_relief: f32,
    /// Candidate A fold-train wavelength in km, converted to front-normal radians.
    pub meso_wavelength_km: f32,
    /// Fold-train irregularity 0..1: 0 = plain 1-D train (phase-locked ridges),
    /// >0 scales cross-strike decorrelation + second octave + crest sharpening.
    pub meso_irregularity: f32,
    /// Meso construction style: 0 = fold train (foreland preset), 1 = massif-corridor.
    pub meso_style: usize,
}

impl Default for FineStructureParams {
    fn default() -> Self {
        Self {
            fault_scarp_height: FAULT_SCARP_HEIGHT,
            interior_relief: FINE_INTERIOR_RELIEF,
            front_strike_weight: FINE_FRONT_STRIKE_WEIGHT,
            margin_contrast: FINE_MARGIN_CONTRAST,
            emergent_lambda: FINE_EMERGENT_LAMBDA,
            emergent_structured: FINE_EMERGENT_STRUCTURED,
            meso_relief: FINE_MESO_RELIEF,
            meso_base_relief: FINE_MESO_BASE_RELIEF,
            meso_wavelength_km: FINE_MESO_WAVELENGTH_KM,
            meso_irregularity: FINE_MESO_IRREGULARITY,
            meso_style: FINE_MESO_STYLE,
        }
    }
}

/// Coarse convergent-boundary primitives for the strike-aware structural relief
/// (P1b). Built once on the coarse mesh (where plates/dynamics live) and consumed
/// by the fine-base synthesis as ORIENTATION primitives — the distance to the nearest
/// *compatible* front orients the interior grain along the orogen (its iso-distance
/// contours run parallel to the front). Each front is a great-circle ARC (the shared
/// Voronoi edge), so the distance field is a true offset from the boundary polyline,
/// NOT a bullseye around a point anchor. NOT a serialized part of `FineBase` — it's a
/// generation input (like [`FineStructureParams`]), hashed into the cache key.
#[derive(Debug, Clone, Default)]
pub struct OrogenFronts {
    /// Edge-midpoint anchors (unit vectors) — the KD-tree query points used only to
    /// gather candidate arcs near a cell.
    pub points: Vec<Vec3>,
    /// The two Voronoi-vertex endpoints of each front's shared edge: distance is the
    /// great-circle distance to this ARC, not to `points[i]`.
    pub seg_a: Vec<Vec3>,
    pub seg_b: Vec<Vec3>,
    /// Per-front structural side: `Some(overriding_plate)` for a subduction front
    /// (fold grain belongs to the overriding plate), `None` for continent–continent
    /// collision (both sides build mountains).
    pub accept_plate: Vec<Option<u32>>,
    /// Coarse `plates.cell_plate`, so a fine cell's plate = `coarse_cell_plate[
    /// coarse_cell[i]]` resolves which fronts are on its side.
    pub coarse_cell_plate: Vec<u32>,
    /// Per-front ALONG-STRIKE coordinate (radians of arc length within its chain): the
    /// convergent edges are chained into ordered polylines (split at triple junctions),
    /// and this is each front's position along its chain. Drives PRINCIPLED segmentation
    /// (the range plunges/segments ALONG its length) instead of the 3D-noise proxy.
    pub arc_u: Vec<f32>,
    /// Per-front chain id, so segmentation phase is decorrelated between distinct ranges.
    pub chain_id: Vec<u32>,
    /// LINEAR along-strike coordinate for the massif-corridor style: endpoint-ordered
    /// (no mirror fold at the BFS seed like `arc_u`), one direction per chain. `arc_u`
    /// is kept untouched for the fold-train style (identity).
    pub u_lin: Vec<f32>,
    /// Per-front orientation: +1 if seg_a->seg_b points in +u_lin direction, else -1,
    /// so a query point can be PROJECTED within its segment (arc_u/u_lin are midpoint
    /// values; without projection u is quantized at coarse-segment ~70 km blocks).
    pub u_dir: Vec<f32>,
    /// Retained age of the present boundary component owning each front. This is
    /// source provenance for the research-only finite-age landscape candidate;
    /// it does not imply that the front moved through time.
    #[cfg(feature = "research-landscape")]
    pub episode_duration_myr: Vec<f32>,
    /// Exact [`TectonicHistory`] episode owning each front. `usize::MAX` denotes
    /// the defensive no-episode case and is aligned with a zero duration.
    #[cfg(feature = "research-landscape")]
    pub episode_id: Vec<usize>,
    /// Canonical coarse boundary edge represented by each front. This lets
    /// research diagnostics cross-check fallback geometry against exact arcs.
    #[cfg(feature = "research-landscape")]
    pub edge_id: Vec<CellEdgeId>,
    /// Positive local closing rate on each exact front arc. Slice A uses this as
    /// relative rock-uplift opportunity before one global budget calibration.
    #[cfg(feature = "research-landscape")]
    pub convergence_km_per_myr: Vec<f32>,
}

impl OrogenFronts {
    /// Extract the convergent fronts from the coarse plate boundaries. Filters to
    /// `BoundaryKind::Convergent` (collision + subduction — the orogen-builders) and
    /// stores each as its real shared-edge ARC (the two Voronoi-vertex endpoints), so
    /// the consumer measures distance to the boundary curve, not a single anchor.
    pub fn build(
        coarse: &Tessellation,
        plates: &Plates,
        crust: &Crust,
        dynamics: &Dynamics,
        #[cfg(feature = "research-landscape")] history: &TectonicHistory,
    ) -> Self {
        let boundaries = collect_plate_boundaries(coarse, plates, crust, dynamics);
        let endpoints = build_cell_pair_edge_endpoints(coarse);
        let mut points = Vec::new();
        let mut seg_a = Vec::new();
        let mut seg_b = Vec::new();
        let mut accept_plate = Vec::new();
        #[cfg(feature = "research-landscape")]
        let mut episode_duration_myr = Vec::new();
        #[cfg(feature = "research-landscape")]
        let mut episode_id = Vec::new();
        #[cfg(feature = "research-landscape")]
        let mut edge_id = Vec::new();
        #[cfg(feature = "research-landscape")]
        let mut convergence_km_per_myr = Vec::new();
        // Voronoi-vertex endpoint IDs per front (for chaining into polylines).
        let mut vids: Vec<(u32, u32)> = Vec::new();
        for e in &boundaries {
            if e.kind != BoundaryKind::Convergent {
                continue;
            }
            let key = if e.cell_a < e.cell_b {
                (e.cell_a, e.cell_b)
            } else {
                (e.cell_b, e.cell_a)
            };
            // Real Voronoi-edge arc endpoints (+ their vertex IDs); fall back to a
            // degenerate arc at the stored bisector point if the edge isn't found.
            let (va, vb, a, b) = endpoints.get(&key).copied().unwrap_or((
                u32::MAX,
                u32::MAX,
                e.boundary_point,
                e.boundary_point,
            ));
            let sum = a + b;
            let mid = if sum.length_squared() > 1e-10 {
                sum.normalize()
            } else {
                a
            };
            // Subduction → fold grain belongs to the OVERRIDING (non-subducting)
            // plate; collision (no polarity) → both sides build mountains.
            let accept = match e.subduction {
                Some(SubductionPolarity::ASubducts) => Some(e.plate_b as u32),
                Some(SubductionPolarity::BSubducts) => Some(e.plate_a as u32),
                None => None,
            };
            points.push(mid);
            seg_a.push(a);
            seg_b.push(b);
            accept_plate.push(accept);
            #[cfg(feature = "research-landscape")]
            {
                let episode = history.episode_for_edge(e.cell_a, e.cell_b);
                episode_duration_myr.push(episode.map_or(0.0, |episode| episode.duration_myr));
                episode_id.push(episode.map_or(usize::MAX, |episode| episode.id));
                edge_id.push(CellEdgeId::new(e.cell_a, e.cell_b));
                convergence_km_per_myr.push(e.convergence_km_per_myr().max(0.0));
            }
            vids.push((va, vb));
        }
        let (arc_u, chain_id, u_lin, u_dir) = chain_fronts(&seg_a, &seg_b, &vids);
        Self {
            points,
            seg_a,
            seg_b,
            accept_plate,
            coarse_cell_plate: plates.cell_plate.clone(),
            arc_u,
            chain_id,
            u_lin,
            u_dir,
            #[cfg(feature = "research-landscape")]
            episode_duration_myr,
            #[cfg(feature = "research-landscape")]
            episode_id,
            #[cfg(feature = "research-landscape")]
            edge_id,
            #[cfg(feature = "research-landscape")]
            convergence_km_per_myr,
        }
    }
}

/// Realized source ledger for the coupled conservative finite-age arm.
///
/// The operator closes its ledger in `f64`, while the final four fields disclose
/// the `f32` rates actually installed into [`OrogenFronts`] for fine-source
/// sampling. The downstream builder subsequently normalizes that spatial shape
/// to the unchanged demoted Legacy target volume.
#[cfg(feature = "research-landscape")]
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct FiniteAgeFluxAuditV0 {
    pub model: FiniteAgeFluxModel,
    pub status: &'static str,
    pub builder_budget_semantics: &'static str,
    pub sigma_km: f64,
    pub implicit_substeps: usize,
    pub processed_segment_count: usize,
    pub processed_edge_count: usize,
    pub untouched_omission_edge_count: usize,
    pub input_signed_flux_km2_per_myr: f64,
    pub output_signed_flux_km2_per_myr: f64,
    pub input_positive_clipped_flux_km2_per_myr: f64,
    pub output_positive_clipped_flux_km2_per_myr: f64,
    pub closure_residual_km2_per_myr: f64,
    pub installed_f32_signed_flux_km2_per_myr: f64,
    pub installed_f32_positive_clipped_flux_km2_per_myr: f64,
    pub installed_f32_cast_residual_km2_per_myr: f64,
    pub rectification_excess_reduction_km2_per_myr: f64,
}

/// Install the fixed one-collision-width signed-flux candidate into raw fronts.
///
/// Exact front topology, identity, age and ownership geometry stay unchanged.
/// The adapter only replaces the aligned positive convergence-rate vector after
/// conservative aggregation within the structural compiler's uninterrupted
/// causal segments. Compiler omissions retain their raw rate and are explicit
/// in the returned audit.
#[cfg(feature = "research-landscape")]
pub fn apply_conservative_finite_age_flux_v0(
    fronts: &mut OrogenFronts,
    coarse: &Tessellation,
    plates: &Plates,
    crust: &Crust,
    dynamics: &Dynamics,
    history: &TectonicHistory,
) -> Result<FiniteAgeFluxAuditV0, ConservativeSignedFluxFrontErrorV0> {
    const IMPLICIT_SUBSTEPS: usize = 8;
    let sigma_km = f64::from(COLLISION_WIDTH) * f64::from(PLANET_RADIUS_KM);
    let boundaries = collect_plate_boundaries(coarse, plates, crust, dynamics);
    let exact_fronts = collect_convergent_fronts(coarse, &boundaries, history)?;
    let candidate =
        conservative_signed_flux_front_rates_v0(&exact_fronts, sigma_km, IMPLICIT_SUBSTEPS)?;

    assert_eq!(
        fronts.edge_id.len(),
        fronts.convergence_km_per_myr.len(),
        "finite-age front provenance/rate alignment"
    );
    assert_eq!(
        fronts.edge_id.len(),
        candidate.signed_rates_km_per_myr.len(),
        "finite-age front roster must match exact signed source"
    );

    let length_by_id: std::collections::BTreeMap<_, _> = exact_fronts
        .edges
        .iter()
        .map(|edge| (edge.id, f64::from(edge.length_km)))
        .collect();
    let installed_signed_rates: Vec<f32> = fronts
        .edge_id
        .iter()
        .map(|edge_id| {
            candidate
                .signed_rates_km_per_myr
                .get(edge_id)
                .copied()
                .unwrap_or_else(|| panic!("candidate omitted exact front {edge_id:?}"))
                as f32
        })
        .collect();
    let mut installed_signed_flux = 0.0;
    let mut installed_positive_flux = 0.0;
    for ((edge_id, &signed_rate), convergence) in fronts
        .edge_id
        .iter()
        .zip(&installed_signed_rates)
        .zip(&mut fronts.convergence_km_per_myr)
    {
        let length_km = length_by_id
            .get(edge_id)
            .copied()
            .unwrap_or_else(|| panic!("missing exact length for front {edge_id:?}"));
        installed_signed_flux += length_km * f64::from(signed_rate);
        installed_positive_flux += length_km * f64::from(signed_rate.max(0.0));
        *convergence = signed_rate.max(0.0);
    }

    let ledger = candidate.ledger;
    let input_excess =
        ledger.input_positive_clipped_flux_km2_per_myr - ledger.input_signed_flux_km2_per_myr;
    let installed_excess = installed_positive_flux - installed_signed_flux;
    Ok(FiniteAgeFluxAuditV0 {
        model: FiniteAgeFluxModel::ConservativeSignedOneCollisionWidthV0,
        status: "research-selectable; nonpromoted",
        builder_budget_semantics: "changes finite-age uplift spatial grammar; the coupled builder retains its existing globally normalized demoted-Legacy target budget",
        sigma_km,
        implicit_substeps: IMPLICIT_SUBSTEPS,
        processed_segment_count: ledger.processed_segment_count,
        processed_edge_count: ledger.processed_edge_count,
        untouched_omission_edge_count: ledger.untouched_edge_count,
        input_signed_flux_km2_per_myr: ledger.input_signed_flux_km2_per_myr,
        output_signed_flux_km2_per_myr: ledger.output_signed_flux_km2_per_myr,
        input_positive_clipped_flux_km2_per_myr: ledger
            .input_positive_clipped_flux_km2_per_myr,
        output_positive_clipped_flux_km2_per_myr: ledger
            .output_positive_clipped_flux_km2_per_myr,
        closure_residual_km2_per_myr: ledger.closure_residual_km2_per_myr,
        installed_f32_signed_flux_km2_per_myr: installed_signed_flux,
        installed_f32_positive_clipped_flux_km2_per_myr: installed_positive_flux,
        installed_f32_cast_residual_km2_per_myr: installed_signed_flux
            - ledger.output_signed_flux_km2_per_myr,
        rectification_excess_reduction_km2_per_myr: input_excess - installed_excess,
    })
}

/// Chain convergent fronts into ordered polylines and assign each front an along-strike
/// coordinate (arc length within its chain) + a chain id. Two fronts are chained only
/// through a shared Voronoi vertex of DEGREE 2 (exactly those two fronts meet there), so
/// triple junctions split chains rather than merging unrelated ranges. Within a chain,
/// arc length accumulates from an arbitrary seed front via the front-graph (great-circle
/// edge lengths). Returns `(arc_u, chain_id)` per front.
fn chain_fronts(
    seg_a: &[Vec3],
    seg_b: &[Vec3],
    vids: &[(u32, u32)],
) -> (Vec<f32>, Vec<u32>, Vec<f32>, Vec<f32>) {
    let nf = seg_a.len();
    let mut arc_u = vec![0.0f32; nf];
    let mut chain_id = vec![0u32; nf];
    if nf == 0 {
        return (arc_u, chain_id, Vec::new(), Vec::new());
    }
    // vertex id -> fronts touching it (skip degenerate u32::MAX endpoints).
    let mut vert_fronts: std::collections::HashMap<u32, Vec<usize>> =
        std::collections::HashMap::new();
    for (f, &(va, vb)) in vids.iter().enumerate() {
        if va != u32::MAX {
            vert_fronts.entry(va).or_default().push(f);
        }
        if vb != u32::MAX {
            vert_fronts.entry(vb).or_default().push(f);
        }
    }
    // Front adjacency: f~g if they share a DEGREE-2 vertex.
    let arc_len = |f: usize| seg_a[f].dot(seg_b[f]).clamp(-1.0, 1.0).acos();
    let neighbors = |f: usize| -> Vec<usize> {
        let mut out = Vec::new();
        for &v in &[vids[f].0, vids[f].1] {
            if v == u32::MAX {
                continue;
            }
            if let Some(fs) = vert_fronts.get(&v) {
                if fs.len() == 2 {
                    out.push(if fs[0] == f { fs[1] } else { fs[0] });
                }
            }
        }
        out
    };
    // BFS each connected component (chain); arc_u = accumulated arc length from the seed.
    let mut visited = vec![false; nf];
    let mut next_chain = 0u32;
    for seed in 0..nf {
        if visited[seed] {
            continue;
        }
        let cid = next_chain;
        next_chain += 1;
        visited[seed] = true;
        chain_id[seed] = cid;
        arc_u[seed] = 0.0;
        let mut stack = vec![seed];
        while let Some(f) = stack.pop() {
            for g in neighbors(f) {
                if !visited[g] {
                    visited[g] = true;
                    chain_id[g] = cid;
                    // Position the neighbour half an edge further along the chain.
                    arc_u[g] = arc_u[f] + 0.5 * (arc_len(f) + arc_len(g));
                    stack.push(g);
                }
            }
        }
    }
    // Second pass — LINEAR walk per chain for the massif-corridor style: start at a
    // chain endpoint (degree-1 front; arbitrary for loops) and accumulate arc length
    // in ONE direction, recording each front's segment orientation relative to +u.
    // `arc_u` above is left untouched (fold-train identity): its BFS folds the two
    // directions from an arbitrary seed onto the same positive coordinate.
    let shared_vertex = |f: usize, g: usize| -> Option<u32> {
        for &vf in &[vids[f].0, vids[f].1] {
            if vf != u32::MAX && (vids[g].0 == vf || vids[g].1 == vf) {
                return Some(vf);
            }
        }
        None
    };
    let mut u_lin = vec![0.0f32; nf];
    let mut u_dir = vec![1.0f32; nf];
    let mut chain_members: std::collections::HashMap<u32, Vec<usize>> =
        std::collections::HashMap::new();
    for f in 0..nf {
        chain_members.entry(chain_id[f]).or_default().push(f);
    }
    for members in chain_members.values() {
        let start = members
            .iter()
            .copied()
            .find(|&f| neighbors(f).len() < 2)
            .unwrap_or(members[0]);
        let mut prev = usize::MAX;
        let mut cur = start;
        let mut u = 0.0f32;
        loop {
            u_lin[cur] = u;
            let next = neighbors(cur).into_iter().find(|&g| g != prev);
            match next {
                Some(g) if g != start => {
                    // Exit vertex of `cur` toward `g` tells its orientation vs +u.
                    let v_out = shared_vertex(cur, g);
                    u_dir[cur] = if v_out == Some(vids[cur].1) {
                        1.0
                    } else {
                        -1.0
                    };
                    u += 0.5 * (arc_len(cur) + arc_len(g));
                    prev = cur;
                    cur = g;
                }
                _ => {
                    // Last front (or closed loop): orient by the ENTRY vertex.
                    if prev != usize::MAX {
                        let v_in = shared_vertex(prev, cur);
                        u_dir[cur] = if v_in == Some(vids[cur].0) { 1.0 } else { -1.0 };
                    }
                    break;
                }
            }
        }
    }
    (arc_u, chain_id, u_lin, u_dir)
}

/// Fine Stage-3/4 world state. The expensive [`FineBase`] (mesh + transferred
/// fields + pre-erosion base elevation) is built once and shared by two cheap
/// [`FineSurface`] snapshots:
/// - `pre` — **stage 3** (Hydrosphere): hydrology on the un-eroded base.
/// - `eroded` — **stage 4** (Erosion): full fluvial erosion + hydrology;
///   computed only when stage 4 is reached.
///
/// Both are retained so the app can snap between pre/post erosion instantly
/// (the structural axis), rather than stepping erosion (the temporal axis,
/// which was slow and needs keyframes to be useful). Re-running with tweaked
/// `ErosionParams` replaces `eroded`, reusing the base.
pub struct FineWorld {
    pub base: FineBase,
    pub pre: FineSurface,
    pub eroded: Option<FineSurface>,
    pub cache_record: FineCacheRecord,
}

/// Expensive, reused base of the fine mesh (stage 3a): the adaptive tessellation,
/// the coarse-cell map, the transferred smooth fields, and the pre-erosion base
/// elevation. Built once; every erosion/hydrology variant reads it by reference.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct FineBase {
    pub tessellation: Tessellation,
    pub coarse_cell: Vec<usize>,
    pub fields: FineFields,
    /// Pre-erosion base the fluvial loop carves into: the interpolated coarse
    /// elevation PLUS the synthesized fine structural relief (interior fault/fold
    /// grain + range-front scarps; P1a). On the fixed coarse sea-level datum.
    /// Distinct from the eroded `surface.elevation`.
    pub base_elevation: Vec<f32>,
    /// The pure interpolated coarse elevation (no structural relief) — the datum
    /// the transferred coarse `temperature` field was lapse-baked against. The
    /// lapse correction in [`FineSurface::from_eroded`] measures relief against
    /// THIS, not `base_elevation`, so the added fine structure (and later the
    /// eroded relief) lapse temperature correctly. See erosion-fine-synthesis.md.
    /// When emergent (erosion-v3), this is ALSO the orogen-build TARGET and the
    /// coarse-target land mask the builder uplift gates on (base_elevation is the
    /// demoted envelope, which can dip below the target / sea level).
    pub coarse_base_elevation: Vec<f32>,
    /// Emergent-orogens demotion fraction used to build this base (erosion-v3). >0
    /// means `base_elevation` is the demoted envelope and erosion should run as an
    /// active builder (rift-excluded uplift gated on `coarse_base_elevation`).
    pub emergent_lambda: f32,
    /// O0 structured-emergent uplift SHAPE per cell (asymmetric front × segmentation ×
    /// demoted forcing), or `None` for the uniform v3 rebuild. When `Some`, the builder
    /// volume-normalizes this and uses it as the uplift source instead of `target−base`.
    pub emergent_uplift_shape: Option<Vec<f32>>,
    pub density: Vec<f32>,
    pub achieved_density_ratio: f32,
}

/// Cheap, per-variant surface over a [`FineBase`] (stages 3b+3c): the eroded
/// elevation and the hydrology derived from it. Re-generated to replace when
/// erosion knobs change.
pub struct FineSurface {
    pub elevation: Elevation,
    pub hydrology: Hydrology,
    /// Precipitation this surface's hydrology used. For the eroded surface this
    /// is the orographic precip recomputed on the eroded relief (rain shadows);
    /// for the pre-erosion surface it is the transferred coarse precip.
    pub precipitation: Vec<f32>,
    /// Temperature this surface's hydrology used: the transferred coarse field
    /// lapse-corrected for the relief above the coarse datum (the synthesized
    /// structural relief on the pre surface, the carved relief on the eroded one).
    /// Stored so rendering/export/diagnose see the same field hydrology did, not the
    /// uncorrected coarse-baked one.
    pub temperature: Vec<f32>,
}

/// Smooth fields transferred to the fine mesh.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct FineFields {
    pub elevation_fields: ElevationFields,
    pub temperature: Vec<f32>,
    pub precipitation: Vec<f32>,
    pub uplift: Vec<f32>,
    /// Surface wind (transferred), for modulating precip by orographic forcing
    /// on the eroded relief (climate↔erosion feedback).
    pub wind: Vec<Vec3>,
}

impl FineWorld {
    /// Build the fine base (cache-aware) plus the pre-erosion surface (stage 3).
    /// Erosion (stage 4) is computed later via [`Self::compute_eroded`].
    #[allow(clippy::too_many_arguments)]
    pub fn generate_pre(
        seed: u64,
        coarse_tessellation: &Tessellation,
        crust: &Crust,
        features: &FeatureFields,
        coarse_elevation: &Elevation,
        atmosphere: &Atmosphere,
        max_cells: usize,
        cache: FineCacheMode,
        density_params: FineDensityParams,
        structure_params: FineStructureParams,
        fronts: &OrogenFronts,
    ) -> Self {
        Self::generate_pre_with_model(
            seed,
            OrogenModel::Legacy,
            coarse_tessellation,
            crust,
            features,
            coarse_elevation,
            atmosphere,
            max_cells,
            cache,
            density_params,
            structure_params,
            fronts,
        )
    }

    /// Model-aware variant of [`Self::generate_pre`]. Experimental callers must
    /// opt in explicitly; the stable constructor keeps legacy product behavior.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_pre_with_model(
        seed: u64,
        orogen_model: OrogenModel,
        coarse_tessellation: &Tessellation,
        crust: &Crust,
        features: &FeatureFields,
        coarse_elevation: &Elevation,
        atmosphere: &Atmosphere,
        max_cells: usize,
        cache: FineCacheMode,
        density_params: FineDensityParams,
        structure_params: FineStructureParams,
        fronts: &OrogenFronts,
    ) -> Self {
        let total = Instant::now();
        let (base, cache_record) = FineBase::load_or_generate(
            cache,
            seed,
            orogen_model,
            coarse_tessellation,
            crust,
            features,
            coarse_elevation,
            atmosphere,
            max_cells,
            density_params,
            structure_params,
            fronts,
        );
        // Pre-erosion surface: hydrology rides the un-eroded interpolated base.
        let pre =
            FineSurface::from_eroded(&base, &base.base_elevation, &base.fields.precipitation, 0.0);
        log::info!(
            "fine mesh: stage-3 base+pre {:.2?}, cells={}, density_ratio={:.1}:1",
            total.elapsed(),
            base.tessellation.num_cells(),
            base.achieved_density_ratio
        );
        Self {
            base,
            pre,
            eroded: None,
            cache_record,
        }
    }

    /// Compute the eroded surface (stage 4) over the existing base if absent.
    /// The pre-erosion (stage-3) hydrology supplies terminal-lake base levels.
    pub fn compute_eroded(&mut self, seed: u64, params: ErosionParams) {
        if self.eroded.is_none() {
            let t = Instant::now();
            self.eroded = Some(FineSurface::generate(
                seed,
                &self.base,
                &self.pre.hydrology,
                params,
            ));
            log::info!("fine mesh: stage-4 erosion surface {:.2?}", t.elapsed());
        }
    }

    /// Re-run the eroded surface with (possibly tweaked) params, replacing it
    /// (no mesh recompute — reuses the base).
    pub fn rerun_eroded(&mut self, seed: u64, params: ErosionParams) {
        self.eroded = Some(FineSurface::generate(
            seed,
            &self.base,
            &self.pre.hydrology,
            params,
        ));
    }

    #[cfg(feature = "research-landscape")]
    pub(crate) fn rerun_eroded_finite_age(
        &mut self,
        seed: u64,
        params: ErosionParams,
        source: &FrozenSupportUplift,
        lookback_myr: f32,
    ) {
        self.eroded = Some(FineSurface::generate_finite_age(
            seed,
            &self.base,
            &self.pre.hydrology,
            params,
            source,
            lookback_myr,
        ));
    }

    /// Whether the eroded (stage-4) surface exists yet.
    pub fn has_eroded(&self) -> bool {
        self.eroded.is_some()
    }

    pub fn tessellation(&self) -> &Tessellation {
        &self.base.tessellation
    }
    pub fn coarse_cell(&self) -> &[usize] {
        &self.base.coarse_cell
    }
    pub fn fields(&self) -> &FineFields {
        &self.base.fields
    }
    pub fn density(&self) -> &[f32] {
        &self.base.density
    }
    pub fn achieved_density_ratio(&self) -> f32 {
        self.base.achieved_density_ratio
    }

    /// Surface for a given view stage: the eroded (stage-4) surface at view >= 4
    /// if computed, otherwise the pre-erosion (stage-3) surface.
    pub fn surface_for(&self, view_stage: u32) -> &FineSurface {
        if view_stage >= 4 {
            if let Some(eroded) = &self.eroded {
                return eroded;
            }
        }
        &self.pre
    }

    /// Adjust the climate ratio on the surface for `view_stage` (disjoint borrow
    /// of base.tessellation + the selected surface).
    pub fn set_climate_ratio(&mut self, view_stage: u32, ratio: f32) {
        let tess = &self.base.tessellation;
        let surface = if view_stage >= 4 && self.eroded.is_some() {
            self.eroded.as_mut().unwrap()
        } else {
            &mut self.pre
        };
        surface.hydrology.set_climate_ratio(tess, ratio);
    }
}

impl FineBase {
    /// Stage 3a with the disk cache: load a matching base if one is cached
    /// (mode `Enabled`), otherwise generate and (unless `Disabled`) save it. The
    /// cache key is a content hash of the inputs, so a changed coarse world / fine
    /// constant is a miss. See [`fine_cache`].
    #[allow(clippy::too_many_arguments)]
    pub fn load_or_generate(
        cache: FineCacheMode,
        seed: u64,
        orogen_model: OrogenModel,
        coarse_tessellation: &Tessellation,
        crust: &Crust,
        features: &FeatureFields,
        coarse_elevation: &Elevation,
        atmosphere: &Atmosphere,
        max_cells: usize,
        density_params: FineDensityParams,
        structure_params: FineStructureParams,
        fronts: &OrogenFronts,
    ) -> (Self, FineCacheRecord) {
        let key = fine_cache::fine_base_key(
            seed,
            orogen_model,
            coarse_tessellation,
            crust,
            features,
            coarse_elevation,
            atmosphere,
            max_cells,
            &density_params,
            &structure_params,
            fronts,
        );
        if cache == FineCacheMode::Enabled {
            if let Some(base) = fine_cache::load(key) {
                let actual_cells = base.tessellation.num_cells();
                return (
                    base,
                    FineCacheRecord {
                        mode: cache,
                        version: fine_cache::FINE_BASE_CACHE_VERSION,
                        key_hex: format!("{key:016x}"),
                        outcome: FineCacheOutcome::Hit,
                        write_succeeded: None,
                        max_cells,
                        actual_cells,
                    },
                );
            }
        }
        let base = Self::generate_with_target(
            seed,
            orogen_model,
            coarse_tessellation,
            crust,
            features,
            coarse_elevation,
            atmosphere,
            max_cells,
            density_params,
            structure_params,
            fronts,
        );
        let write_succeeded = matches!(cache, FineCacheMode::Enabled | FineCacheMode::Rebuild)
            .then(|| fine_cache::save(key, &base));
        let outcome = match cache {
            FineCacheMode::Disabled => FineCacheOutcome::DisabledGenerated,
            FineCacheMode::Enabled => FineCacheOutcome::MissGenerated,
            FineCacheMode::Rebuild => FineCacheOutcome::Rebuilt,
        };
        let actual_cells = base.tessellation.num_cells();
        (
            base,
            FineCacheRecord {
                mode: cache,
                version: fine_cache::FINE_BASE_CACHE_VERSION,
                key_hex: format!("{key:016x}"),
                outcome,
                write_succeeded,
                max_cells,
                actual_cells,
            },
        )
    }

    /// Stage 3a: build the expensive, reusable fine-mesh base (steps 1–7 of the
    /// old monolith). Stops short of erosion — that's [`FineSurface::generate`].
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_target(
        seed: u64,
        orogen_model: OrogenModel,
        coarse_tessellation: &Tessellation,
        crust: &Crust,
        features: &FeatureFields,
        coarse_elevation: &Elevation,
        atmosphere: &Atmosphere,
        max_cells: usize,
        density_params: FineDensityParams,
        structure_params: FineStructureParams,
        fronts: &OrogenFronts,
    ) -> Self {
        let t0 = Instant::now();
        let preview_hydrology = Hydrology::generate(
            coarse_tessellation,
            crust,
            coarse_elevation,
            &atmosphere.precipitation,
            &atmosphere.temperature,
        );
        log::info!("fine mesh: coarse hydrology preview {:.2?}", t0.elapsed());

        let t0 = Instant::now();
        let raw_density = compute_areal_density(
            coarse_tessellation,
            coarse_elevation,
            features,
            &preview_hydrology,
            &density_params,
        );
        // The cell count EMERGES from integrating the areal density over the
        // mesh; max_cells is a guardrail that uniformly coarsens if exceeded.
        let coarse_areas = coarse_tessellation.cell_areas();
        let emergent: f64 = raw_density
            .iter()
            .zip(coarse_areas.iter())
            .map(|(&g, &a)| (g * a) as f64)
            .sum();
        let scale = if emergent > max_cells as f64 {
            let s = (max_cells as f64 / emergent) as f32;
            log::warn!(
                "fine mesh: emergent count {:.0} exceeds cap {} -> coarsening uniformly ({:.2}x larger cells)",
                emergent,
                max_cells,
                (1.0 / s).sqrt()
            );
            s
        } else {
            1.0
        };
        let density: Vec<f32> = raw_density.iter().map(|&g| g * scale).collect();
        let density_min = density
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min)
            .max(1e-12);
        let density_max = density.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let achieved_density_ratio = density_max / density_min;
        log::info!(
            "fine mesh: density field {:.2?}, target sizes {:.1}-{:.1} km, emergent {:.0} cells (cap {})",
            t0.elapsed(),
            density_params.mountain_km,
            density_params.ocean_km,
            emergent * scale as f64,
            max_cells,
        );

        let t0 = Instant::now();
        let tree = build_coarse_tree(coarse_tessellation);
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(30));
        let points = sample_fine_points(coarse_tessellation, &density, &tree, &mut rng);
        log::info!(
            "fine mesh: sampling {:.2?} ({} cells)",
            t0.elapsed(),
            points.len()
        );

        let t0 = Instant::now();
        let tessellation = Tessellation::from_points_knn_clipping(points);
        log::info!("fine mesh: tessellation {:.2?}", t0.elapsed());

        let t0 = Instant::now();
        let coarse_cell = map_to_coarse(&tessellation, &tree);
        let fine_density: Vec<f32> = coarse_cell.iter().map(|&c| density[c]).collect();
        mesh_quality_probe(&tessellation);
        let fields = transfer_fields(
            coarse_tessellation,
            &tessellation,
            &coarse_cell,
            crust,
            features,
            orogen_model,
            coarse_elevation,
            atmosphere,
        );
        log::info!("fine mesh: field transfer {:.2?}", t0.elapsed());

        // Refine the coarse elevation onto the fine cells rather than recomputing
        // it from transferred structural fields. Sea level is a global datum
        // solved once on the coarse mesh; interpolating the (already sea-level-
        // shifted) coarse elevation inherits that datum exactly, so the fine mesh
        // never re-solves sea level — and the relief matches coarse instead of
        // collapsing toward zero.
        let t0 = Instant::now();
        let coarse_base_elevation = interpolate_coarse_elevation(
            coarse_tessellation,
            &tessellation,
            &coarse_cell,
            &coarse_elevation.values,
        );
        log::info!("fine mesh: elevation refine {:.2?}", t0.elapsed());

        // Synthesize the mid-band structural relief erosion will carve (P1a). The
        // interpolated base is smooth in orogen interiors (the coarse forcing
        // saturates there → flat-topped highs with no drainage gradient); this adds
        // fault-block / fold grain + range-front scarps onto it, BEFORE pre-
        // hydrology so terminal-lake base levels and the temperature lapse see the
        // real substrate. `coarse_base_elevation` is kept as the lapse baseline.
        let t0 = Instant::now();
        let mut base_elevation = coarse_base_elevation.clone();
        // Emergent orogens (erosion-v3): DEMOTE a fraction of the model-selected
        // tectonic load from the static envelope. Erosion rebuilds exactly this
        // load as active uplift. In the product baseline the load reproduces the
        // historical arc/collision response; experiments can supply conserved
        // crust evolution instead.
        // Erosion then rebuilds it as active uplift, carving dissected ranges instead
        // of dissecting a flat plateau. `coarse_base_elevation` is kept as the rebuild
        // TARGET and the coarse-target land mask. See erosion-v3-emergent-orogens.md.
        if structure_params.emergent_lambda > 0.0 {
            let lambda = structure_params.emergent_lambda;
            let ef = &fields.elevation_fields;
            if orogen_model == OrogenModel::Legacy {
                // BIT-EXACT historical demotion: λ·(arc+col) on the fine-interpolated
                // response fields — NOT λ·slope·thickening, whose slope round-trip
                // reseeds the erosion chaos cascade and breaks default identity.
                for (i, elevation) in base_elevation.iter_mut().enumerate() {
                    *elevation -= lambda * (ef.arc[i] + ef.collision[i]).max(0.0);
                }
            } else {
                for (elevation, &tectonic) in
                    base_elevation.iter_mut().zip(ef.tectonic_thickening.iter())
                {
                    *elevation -= lambda * isostasy_slope() * tectonic.max(0.0);
                }
            }
            report_envelope_land_flips(&tessellation, &coarse_base_elevation, &base_elevation);
        }
        // Interpolate the raw signed margin distance (radians from the coast; +
        // continental) onto the fine cells for the active/passive margin contrast
        // (P1c). Reuses the generic coarse-scalar interpolator.
        let margin_distance = interpolate_coarse_elevation(
            coarse_tessellation,
            &tessellation,
            &coarse_cell,
            &crust.signed_margin_distance,
        );
        add_interior_structural_relief(
            &tessellation,
            &coarse_cell,
            &fields.elevation_fields,
            fronts,
            &margin_distance,
            &mut base_elevation,
            seed,
            structure_params.interior_relief,
            structure_params.front_strike_weight,
            structure_params.margin_contrast,
        );
        add_meso_base_relief(
            &tessellation,
            &coarse_cell,
            &fields.elevation_fields,
            fronts,
            &mut base_elevation,
            seed,
            structure_params.meso_base_relief,
            structure_params.meso_wavelength_km,
            structure_params.meso_irregularity,
            structure_params.meso_style,
        );
        // Range-front scarps: sharpen active orogen margins (footwall up / basin
        // down). Deliberately ASYMMETRIC (real fronts express footwall uplift far
        // more than basin drop; basins fill), so — unlike the interior grain — they
        // are NOT coarse-cell zero-mean; the sea-level clamp adds a small positive
        // bias. The datum/land-fraction safety for the COMBINED structural edit is
        // verified by the area-weighted drift check below, not by zero-mean alone.
        apply_fault_scarps(
            &mut base_elevation,
            &fields.elevation_fields,
            structure_params.fault_scarp_height,
        );
        log::info!("fine mesh: structural relief {:.2?}", t0.elapsed());

        // Area-weighted land-fraction drift from the combined structural relief
        // (interior grain + scarps). The interior term is zero-mean per coarse cell
        // and gated well above sea level, so it should not move the land/ocean mask;
        // a non-trivial drift means a knob (likely a high `interior_relief` sweep, or
        // scarps) is flipping near-sea-level cells and silently invalidating the
        // coarse atmosphere assumptions (erosion-fine-synthesis.md, fix #2). Count-
        // based fractions are wrong on the adaptive mesh, so weight by cell area.
        report_land_fraction_drift(&tessellation, &coarse_base_elevation, &base_elevation);

        // O0 structured-emergent uplift shape (orogen-structure.md): an asymmetric
        // front profile × along-strike segmentation × demoted forcing, used by the
        // builder instead of the uniform `target−base` rebuild. Built here where the
        // fronts + demoted envelope are in hand; volume-normalized in the erosion stage.
        let emergent_uplift_shape = (structure_params.emergent_lambda > 0.0
            && structure_params.emergent_structured > 0.0)
            .then(|| {
                compute_emergent_uplift_shape(
                    &tessellation,
                    &coarse_cell,
                    fronts,
                    &coarse_base_elevation,
                    &base_elevation,
                    seed,
                    structure_params.emergent_structured,
                    structure_params.meso_relief,
                    structure_params.meso_wavelength_km,
                    structure_params.meso_irregularity,
                    structure_params.meso_style,
                )
            });

        Self {
            tessellation,
            coarse_cell,
            fields,
            base_elevation,
            coarse_base_elevation,
            emergent_lambda: structure_params.emergent_lambda,
            emergent_uplift_shape,
            density: fine_density,
            achieved_density_ratio,
        }
    }
}

impl FineSurface {
    /// Stages 3b+3c: carve the base into river valleys, then derive hydrology.
    /// Reads `base` by reference so it can be re-run cheaply with new erosion
    /// knobs (`params`). `seed` drives the stochastic erosion fields (lithologic
    /// erodibility).
    pub fn generate(
        seed: u64,
        base: &FineBase,
        pre_hydrology: &Hydrology,
        params: ErosionParams,
    ) -> Self {
        Self::generate_impl(
            seed,
            base,
            pre_hydrology,
            params,
            #[cfg(feature = "research-landscape")]
            None,
        )
    }

    /// Research-only finite-age candidate over the same mesh, climate supply
    /// and final hydrology contract as the ordinary erosion surface.
    #[cfg(feature = "research-landscape")]
    pub(crate) fn generate_finite_age(
        seed: u64,
        base: &FineBase,
        pre_hydrology: &Hydrology,
        params: ErosionParams,
        source: &FrozenSupportUplift,
        lookback_myr: f32,
    ) -> Self {
        Self::generate_impl(
            seed,
            base,
            pre_hydrology,
            params,
            Some((source, lookback_myr)),
        )
    }

    /// Research-only terrain discriminator for four process-mesh opportunity
    /// rasters. Supplying the same raster four times is the control; supplying
    /// RDS frames 0..3 is the counterfactual. Both therefore traverse this exact
    /// continuous erosion path and differ only in normalized spatial weights.
    #[cfg(feature = "research-landscape")]
    pub fn generate_regional_deformation_v0(
        seed: u64,
        base: &FineBase,
        _pre_hydrology: &Hydrology,
        params: ErosionParams,
        opportunity_frames: [&RegionalDeformationRasterV0; RDS0_FRAME_COUNT],
        parent_duration_myr: f64,
        lookback_myr: f64,
    ) -> Result<(Self, LegacyBudgetOpportunityAuditV0), LegacyBudgetOpportunityErrorV0> {
        let erodibility = lithology_erodibility(
            &base.tessellation,
            &base.fields.elevation_fields,
            seed,
            params.litho_sigma,
            EROSION_LITHO_GEO_STRENGTH,
            params.litho_grain_strength,
        );
        if base.emergent_lambda <= 0.0 {
            return Err(LegacyBudgetOpportunityErrorV0::RequiresEmergentTarget);
        }
        let lake_base = vec![f32::NEG_INFINITY; base.tessellation.num_cells()];
        let structured_base = &base.base_elevation;
        let geom = super::erosion::NeighborGeometry::build(&base.tessellation);
        let opportunity_weights: [&[f64]; RDS0_FRAME_COUNT] =
            std::array::from_fn(|index| opportunity_frames[index].rate_density_per_myr.as_slice());

        let iters = params.precip_outer_iters.max(1);
        let mut precip = normalize_fine_precipitation(
            &base.tessellation,
            structured_base,
            &base.fields.elevation_fields.continentality,
            &base.fields.precipitation,
        );
        let mut eroded = structured_base.clone();
        let mut final_audit = None;
        for outer in 0..iters {
            let t0 = Instant::now();
            let (candidate, audit) = super::erosion::erode_regional_deformation_v0(
                &base.tessellation,
                &base.fields.elevation_fields,
                structured_base,
                &precip,
                &erodibility,
                &lake_base,
                &geom,
                params,
                Some(&base.coarse_base_elevation),
                opportunity_weights,
                parent_duration_myr,
                lookback_myr,
            )?;
            eroded = candidate;
            final_audit = Some(audit);
            let t_erode = t0.elapsed();
            let t1 = Instant::now();
            precip = fine_precipitation(
                &base.tessellation,
                &eroded,
                &base.fields.wind,
                &base.fields.precipitation,
                params.orographic_precip_strength,
                params.downwind_shadow_strength,
            );
            precip = normalize_fine_precipitation(
                &base.tessellation,
                &eroded,
                &base.fields.elevation_fields.continentality,
                &precip,
            );
            log::info!(
                "fine mesh: RDS erode+precip pass {}/{} (erode {:.2?}, precip {:.2?})",
                outer + 1,
                iters,
                t_erode,
                t1.elapsed(),
            );
        }

        let t0 = Instant::now();
        super::erosion::glacial_erode(&base.tessellation, &mut eroded, &lake_base, &geom, params);
        log::info!("fine mesh: RDS glacial pass {:.2?}", t0.elapsed());
        let surface = Self::from_eroded(base, &eroded, &precip, params.lake_evap_strength);
        Ok((
            surface,
            final_audit.expect("at least one RDS erosion/precip pass"),
        ))
    }

    fn generate_impl(
        seed: u64,
        base: &FineBase,
        pre_hydrology: &Hydrology,
        params: ErosionParams,
        #[cfg(feature = "research-landscape")] finite_age: Option<(&FrozenSupportUplift, f32)>,
    ) -> Self {
        // Fluvial erosion: carve the interpolated base into real river valleys by
        // evolving crust thickness (isostasy responds). Runs on the fine mesh
        // before final hydrology; sea level is the fixed datum inherited via
        // `base_elevation`. See docs/archive/specs/erosion.md.
        let erodibility = lithology_erodibility(
            &base.tessellation,
            &base.fields.elevation_fields,
            seed,
            params.litho_sigma,
            EROSION_LITHO_GEO_STRENGTH,
            params.litho_grain_strength,
        );

        // Terminal (endorheic) lakes from the pre-erosion hydrology act as fixed
        // local base levels, so inflowing rivers grade to the lake surface and
        // closed basins drain internally instead of being carved over their spill.
        // EMERGENT (erosion-v3): lakes OFF — pre-hydrology on the LOW demoted envelope
        // would freeze low-envelope lakes as base levels that pin the orogen the uplift
        // is trying to build (codex). Re-enable once the build is warmed up (future).
        let emergent = base.emergent_lambda > 0.0;
        let lake_base = if emergent {
            vec![f32::NEG_INFINITY; base.tessellation.num_cells()]
        } else {
            terminal_lake_base_levels(&base.tessellation, pre_hydrology)
        };
        // Emergent: erosion gets the demoted envelope as `base` and the coarse TARGET
        // (full coarse elevation) as the land mask the builder uplift gates on — so a
        // demoted-below-sea orogen cell still uplifts back instead of dying.
        let coarse_target = emergent.then_some(base.coarse_base_elevation.as_slice());
        // O0: the structured uplift shape, if built (asymmetric/segmented); the builder
        // volume-normalizes it and uses it instead of the uniform target−base rebuild.
        let uplift_shape = base.emergent_uplift_shape.as_deref();

        // The base already carries the synthesized structural relief (interior
        // fault/fold grain + range-front scarps; built in `FineBase`), so erosion
        // carves directly into it — no separate scarp pass here.
        let structured_base = &base.base_elevation;

        // Coupled erode↔precip loop: each pass re-carves the base relief with the
        // rain-shadow precip from the previous pass (windward flanks, wetter,
        // dissect more than lee). Pass 1 erodes with the coarse precip; later
        // passes use the orographic-modulated precip. The precip is always a
        // modulation of the COARSE field on the CURRENT relief, so it tracks the
        // carved ranges. Converges in a couple of passes.
        let iters = params.precip_outer_iters.max(1);
        let mut precip = normalize_fine_precipitation(
            &base.tessellation,
            structured_base,
            &base.fields.elevation_fields.continentality,
            &base.fields.precipitation,
        );
        let mut eroded = structured_base.clone();
        // The neighbour geometry (chord distances + finite-volume edge weights) is
        // a function of the immutable tessellation, so build it once and reuse it
        // across every erode↔precip pass and the glacial pass instead of
        // rescanning each cell's Voronoi vertices for shared-edge lengths per call.
        let geom = super::erosion::NeighborGeometry::build(&base.tessellation);

        // A4 drainage pulse (meso-a4-drainage-pulse.md): a BURN-IN epoch with the
        // unmodified shape self-organizes drainage on this base; the extracted
        // trunk/interfluve modifier then redistributes the uplift shape for the
        // real (final) epoch below — valleys deepen by incision along the
        // organized network, servo-neutral per orogen. The modifier is FROZEN:
        // one feedback pass only (continuous feedback locks into exaggerated
        // spokes — consult §4.3). Dial 0 skips all of this (path untouched).
        let mut pulsed_shape: Option<Vec<f32>> = None;
        if params.drainage_pulse > 0.0 {
            if let (true, Some(shape)) = (emergent, uplift_shape) {
                let t0 = Instant::now();
                let mut burn_params = params;
                burn_params.steps = params.pulse_burnin_steps;
                let burn_eroded = super::erosion::erode(
                    &base.tessellation,
                    &base.fields.elevation_fields,
                    structured_base,
                    &precip,
                    &erodibility,
                    &lake_base,
                    &geom,
                    burn_params,
                    coarse_target,
                    uplift_shape,
                );
                pulsed_shape = super::erosion::drainage_pulse_modifier(
                    &burn_eroded,
                    &precip,
                    &lake_base,
                    &geom,
                    base.tessellation.cell_areas_ref(),
                    shape,
                    &base.coarse_base_elevation,
                    params,
                )
                .map(|modifier| shape.iter().zip(&modifier).map(|(&s, &m)| s * m).collect());
                log::info!(
                    "fine mesh: drainage-pulse burn-in ({} steps) + extraction {:.2?}",
                    params.pulse_burnin_steps,
                    t0.elapsed()
                );
            } else {
                log::warn!(
                    "drainage pulse requires the emergent structured-uplift path; dial ignored"
                );
            }
        }
        let uplift_shape = pulsed_shape.as_deref().or(uplift_shape);
        for outer in 0..iters {
            let t0 = Instant::now();
            #[cfg(feature = "research-landscape")]
            {
                if let Some((source, lookback_myr)) = finite_age {
                    let program = super::erosion::FiniteAgeUpliftProgram {
                        duration_myr: &source.duration_myr,
                        lookback_myr,
                        match_static_total: false,
                    };
                    let (candidate, audit) = super::erosion::erode_finite_age(
                        &base.tessellation,
                        &base.fields.elevation_fields,
                        structured_base,
                        &precip,
                        &erodibility,
                        &lake_base,
                        &geom,
                        params,
                        coarse_target,
                        Some(&source.shape),
                        program,
                    )
                    .expect("validated frozen-support finite-age uplift program");
                    log::info!(
                        "finite-age uplift: {} source cells, {} ages, active steps {}..{}, rate scale {:.3}, scheduled/target {:.6}/{:.6}, actual {:.6}",
                        source.owned_cells,
                        source.distinct_durations,
                        audit.min_active_steps,
                        audit.max_active_steps,
                        audit.rate_scale,
                        audit.expected_scheduled_uplift_volume,
                        audit.target_static_uplift_volume,
                        audit.actual_injected_uplift_volume,
                    );
                    eroded = candidate;
                } else {
                    eroded = super::erosion::erode(
                        &base.tessellation,
                        &base.fields.elevation_fields,
                        structured_base,
                        &precip,
                        &erodibility,
                        &lake_base,
                        &geom,
                        params,
                        coarse_target,
                        uplift_shape,
                    );
                }
            }
            #[cfg(not(feature = "research-landscape"))]
            {
                eroded = super::erosion::erode(
                    &base.tessellation,
                    &base.fields.elevation_fields,
                    structured_base,
                    &precip,
                    &erodibility,
                    &lake_base,
                    &geom,
                    params,
                    coarse_target,
                    uplift_shape,
                );
            }
            let t_erode = t0.elapsed();
            let t1 = Instant::now();
            precip = fine_precipitation(
                &base.tessellation,
                &eroded,
                &base.fields.wind,
                &base.fields.precipitation,
                params.orographic_precip_strength,
                params.downwind_shadow_strength,
            );
            precip = normalize_fine_precipitation(
                &base.tessellation,
                &eroded,
                &base.fields.elevation_fields.continentality,
                &precip,
            );
            log::info!(
                "fine mesh: erode+precip pass {}/{} (erode {:.2?}, precip {:.2?})",
                outer + 1,
                iters,
                t_erode,
                t1.elapsed(),
            );
        }

        // Glacial sculpting on the carved relief: snowline-driven ice over-
        // deepening (U-troughs, tarns) that sharpens the peaks between glaciers.
        let t0 = Instant::now();
        super::erosion::glacial_erode(&base.tessellation, &mut eroded, &lake_base, &geom, params);
        log::info!("fine mesh: glacial pass {:.2?}", t0.elapsed());

        Self::from_eroded(base, &eroded, &precip, params.lake_evap_strength)
    }

    /// Build the surface (eroded elevation + hydrology) from an already-eroded
    /// elevation. `lake_evap_strength > 0` adds the lakes-as-evaporation
    /// pass (re-runs hydrology once with lake-boosted precip).
    pub fn from_eroded(
        base: &FineBase,
        eroded: &[f32],
        precipitation: &[f32],
        lake_evap_strength: f32,
    ) -> Self {
        // The eroded surface is the elevation hydrology and rendering consume.
        let elevation = Elevation::refine_from_base(&base.tessellation, eroded);
        log_resolution_probe(&base.tessellation, &elevation);

        // Correct temperature for the relief above the coarse datum.
        // `fields.temperature` is the coarse field interpolated onto the fine mesh,
        // so its lapse is baked against the COARSE elevation (`coarse_base_elevation`
        // — the pure interpolant, NOT `base_elevation`, which now also carries the
        // synthesized fine structural relief). Re-apply the lapse delta against the
        // current relief — the structured base for the pre-erosion surface, or the
        // eroded relief for stage 4 — so the added structure and the carved valleys
        // both lapse temperature and basin evaporation sees the terrain it drains.
        // Only positive elevation lapses (matches `generate_surface_temperature`).
        let temperature: Vec<f32> = (0..base.tessellation.num_cells())
            .map(|i| {
                let delta = eroded[i].max(0.0) - base.coarse_base_elevation[i].max(0.0);
                base.fields.temperature[i] - super::atmosphere::LAPSE_RATE * delta
            })
            .collect();

        let hydro = |precip: &[f32]| {
            Hydrology::generate_from_continentality(
                &base.tessellation,
                &base.fields.elevation_fields.continentality,
                &elevation,
                precip,
                &temperature,
            )
        };

        let t0 = Instant::now();
        let mut precip = normalize_fine_precipitation(
            &base.tessellation,
            &elevation.values,
            &base.fields.elevation_fields.continentality,
            precipitation,
        );
        let mut hydrology = hydro(&precip);
        log::info!("fine mesh: hydrology {:.2?}", t0.elapsed());

        // Lakes as evaporation sources: standing water adds local humidity, so
        // boost precip in a halo around the lakes and re-run hydrology once. Local
        // (no transport on the fine mesh), one pass = no runaway lake growth.
        if lake_evap_strength > 0.0 {
            let t0 = Instant::now();
            let boosted = boost_precip_near_lakes(
                &base.tessellation,
                &elevation.values,
                &precip,
                &hydrology,
                lake_evap_strength,
            );
            precip = normalize_fine_precipitation(
                &base.tessellation,
                &elevation.values,
                &base.fields.elevation_fields.continentality,
                &boosted,
            );
            hydrology = hydro(&precip);
            log::info!("fine mesh: lake-evap + re-hydrology {:.2?}", t0.elapsed());
        }

        // Adopt the drainage-integrated (outlet-carved) elevation as the rendered surface so
        // the relief is consistent with where the rivers actually drain (carved channels are
        // real water gaps). `integrate_basins` is idempotent on an already-carved surface.
        Self {
            elevation: Elevation {
                values: hydrology.elevation.clone(),
                ..elevation
            },
            hydrology,
            precipitation: precip,
            temperature,
        }
    }
}

/// Normalize the precipitation actually supplied to fine hydrology against the
/// same connected-ocean rule hydrology uses. This must run even when every
/// optional fine-climate modifier is off: interpolation onto an adaptive mesh
/// does not preserve an area-weighted land integral by itself.
fn normalize_fine_precipitation(
    tessellation: &Tessellation,
    elevation: &[f32],
    continentality: &[f32],
    precipitation: &[f32],
) -> Vec<f32> {
    let areas = tessellation.cell_areas_ref();
    let ocean = connected_ocean_cells(tessellation, continentality, elevation, areas);
    let (weighted_supply, land_area) = (0..tessellation.num_cells())
        .filter(|&cell| !ocean[cell])
        .fold((0.0f64, 0.0f64), |(supply, area), cell| {
            (
                supply + precipitation[cell].max(0.0) as f64 * areas[cell] as f64,
                area + areas[cell] as f64,
            )
        });
    let mean = if land_area > 0.0 {
        weighted_supply / land_area
    } else {
        0.0
    };
    let scale = if mean > 1e-12 {
        PRECIP_GLOBAL_SCALE as f64 / mean
    } else {
        1.0
    };
    precipitation
        .iter()
        .map(|&value| (value.max(0.0) as f64 * scale) as f32)
        .collect()
}

#[cfg(test)]
mod precipitation_normalization_tests {
    use super::*;

    #[test]
    fn adaptive_transfer_is_normalized_over_hydrologic_land() {
        let mut rng = ChaCha8Rng::seed_from_u64(817);
        let tessellation = Tessellation::generate(800, 0, &mut rng);
        let n = tessellation.num_cells();
        let elevation: Vec<f32> = (0..n)
            .map(|cell| 0.2 * tessellation.cell_center(cell).y)
            .collect();
        let continentality = vec![0.0; n];
        let precipitation: Vec<f32> = (0..n)
            .map(|cell| 0.5 + 3.0 * tessellation.cell_center(cell).x.max(0.0))
            .collect();

        let normalized = normalize_fine_precipitation(
            &tessellation,
            &elevation,
            &continentality,
            &precipitation,
        );
        let areas = tessellation.cell_areas_ref();
        let ocean = connected_ocean_cells(&tessellation, &continentality, &elevation, areas);
        let (supply, area) =
            (0..n)
                .filter(|&cell| !ocean[cell])
                .fold((0.0f64, 0.0f64), |(supply, area), cell| {
                    (
                        supply + normalized[cell] as f64 * areas[cell] as f64,
                        area + areas[cell] as f64,
                    )
                });

        assert!((supply / area - PRECIP_GLOBAL_SCALE as f64).abs() < 1e-6);
    }
}

/// Per-cell lithologic erodibility multiplier on the fine mesh. Sums three role-1
/// "rock varies" log-contrasts that the incision step organizes into drainage-
/// aligned differential relief, then exponentiates:
///   - GEOLOGY (transferred fields): deep continental interiors are old, hard
///     cratonic basement (lower K); volcanic arcs are fresh, fractured terrain
///     (higher K). Tied to the same continentality/arc machinery elevation uses,
///     so the grain follows the world's geology, not just free noise.
///   - STRUCTURAL GRAIN: alternating hard/soft bands in convergent belts, striking
///     along the iso-convergent contours (≈ parallel to the collision front, i.e.
///     fold-axis strike), tightest/strongest near the suture and fading outward —
///     a fold-and-thrust fabric that the incision can express as ridge-and-valley
///     / trellis drainage. Experimental (`ideas.md`: K sets incision rate, not
///     geometry, so trellis is an outcome to TEST, not a promise).
///   - TEXTURE (fBm at fine cell centers, never interpolated): sub-unit variation
///     at terrane→formation scale.
/// Returned un-normalized; erosion normalizes it to unit land mean so it only
/// REDISTRIBUTES incision. All-ones when all knobs are 0.
fn lithology_erodibility(
    tess: &Tessellation,
    fields: &ElevationFields,
    seed: u64,
    sigma: f32,
    geo_strength: f32,
    grain_strength: f32,
) -> Vec<f32> {
    let n = tess.num_cells();
    if sigma <= 0.0 && geo_strength <= 0.0 && grain_strength <= 0.0 {
        return vec![1.0; n];
    }
    // Arc + convergence normalized to [0,1] by their land maxima (robust to weak/
    // strong worlds; the result is re-normalized to unit mean downstream anyway).
    let arc_max = fields.arc.iter().copied().fold(0.0f32, f32::max).max(1e-6);
    let conv_max = fields
        .convergent
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let fbm = (sigma > 0.0).then(|| {
        Fbm::<Perlin>::new(seed.wrapping_add(47) as u32).set_octaves(EROSION_LITHO_OCTAVES)
    });
    let sample = |i: usize| {
        let craton = fields.continentality[i].clamp(0.0, 1.0); // 1 = deep interior
        let arc_soft = (fields.arc[i].max(0.0) / arc_max).clamp(0.0, 1.0);
        // Geology: harder in cratons, softer in arcs (log-K contrast).
        let geo_log = geo_strength * (arc_soft - craton);
        // Structural grain: bands along iso-convergent contours (fold strike),
        // amplitude growing toward the suture so folds are tightest there.
        let grain_log = if grain_strength > 0.0 {
            let conv = (fields.convergent[i].max(0.0) / conv_max).clamp(0.0, 1.0);
            grain_strength * conv * (std::f32::consts::TAU * conv / EROSION_FOLD_WAVELENGTH).sin()
        } else {
            0.0
        };
        let fbm_log = match &fbm {
            Some(f) => {
                let p = tess.cell_center(i) * EROSION_LITHO_FREQUENCY as f32;
                sigma * f.get([p.x as f64, p.y as f64, p.z as f64]) as f32
            }
            None => 0.0,
        };
        (geo_log + grain_log + fbm_log).exp()
    };
    #[cfg(not(feature = "single-threaded"))]
    {
        (0..n).into_par_iter().map(sample).collect()
    }
    #[cfg(feature = "single-threaded")]
    {
        (0..n).map(sample).collect()
    }
}

struct MesoFieldSampler {
    wavelength: f32,
    irregularity: f32,
    phase_fbm: Fbm<Perlin>,
    amp_fbm: Fbm<Perlin>,
    iso_fbm: Fbm<Perlin>,
}

impl MesoFieldSampler {
    fn new(seed: u64, wavelength_km: f32, irregularity: f32) -> Self {
        Self {
            wavelength: (wavelength_km / PLANET_RADIUS_KM).max(1e-6),
            irregularity: irregularity.clamp(0.0, 1.0),
            phase_fbm: Fbm::<Perlin>::new(seed.wrapping_add(72) as u32).set_octaves(3),
            amp_fbm: Fbm::<Perlin>::new(seed.wrapping_add(73) as u32).set_octaves(3),
            iso_fbm: Fbm::<Perlin>::new(seed.wrapping_add(74) as u32).set_octaves(3),
        }
    }

    fn sample(&self, c: Vec3, u: f32, v: f32, chain_id: u32) -> f32 {
        let chain = chain_id as f32;
        let g = self.irregularity;
        // Cross-strike decorrelation coordinate: at g=0 this is 0 and the field
        // reduces exactly to the 1-D (u-only) train, where every ridge is a
        // phase-locked copy of its neighbor; at g>0 the phase/spur modulation
        // drifts across strike so ridges wobble and terminate individually.
        let v_dec = (v / (self.wavelength * FINE_MESO_DECOR_WAVELENGTHS)) as f64 * g as f64;
        let phase_raw = self.phase_fbm.get([
            (u * FINE_MESO_PHASE_FREQUENCY as f32) as f64 + chain as f64 * 41.71,
            v_dec,
            0.0,
        ]) as f32;
        let amp_raw = self.amp_fbm.get([
            (u * FINE_MESO_SPUR_FREQUENCY as f32) as f64 + chain as f64 * 67.19,
            17.0 + v_dec,
            0.0,
        ]) as f32;
        let iso = self.iso_fbm.get([
            c.x as f64 * FINE_MESO_ISO_FREQUENCY,
            c.y as f64 * FINE_MESO_ISO_FREQUENCY,
            c.z as f64 * FINE_MESO_ISO_FREQUENCY,
        ]) as f32;
        let phase = phase_raw * FINE_MESO_PHASE_WARP * std::f32::consts::TAU;
        let wave_jitter = 1.0 + (phase_raw * FINE_MESO_WAVELENGTH_JITTER).clamp(-0.45, 0.45);
        let mut fold =
            ((v / (self.wavelength * wave_jitter)) * std::f32::consts::TAU + phase).sin();
        if g > 0.0 {
            // Second, incommensurate fold octave: the beat against the primary
            // varies ridge spacing and prominence so the train never reads as a
            // metronome. Then sharpen crests/valleys away from the pure-sine
            // "dune" cross-section.
            let phase2_raw = self.phase_fbm.get([
                (u * FINE_MESO_PHASE_FREQUENCY as f32) as f64 + chain as f64 * 41.71 + 91.37,
                v_dec,
                0.0,
            ]) as f32;
            let phase2 = phase2_raw * FINE_MESO_PHASE_WARP * std::f32::consts::TAU;
            let jitter2 = 1.0 + (phase2_raw * FINE_MESO_WAVELENGTH_JITTER).clamp(-0.45, 0.45);
            let fold2 = ((v / (self.wavelength * FINE_MESO_OCTAVE2_RATIO * jitter2))
                * std::f32::consts::TAU
                + phase2)
                .sin();
            let w2 = g * FINE_MESO_OCTAVE2_AMP;
            fold = (fold + w2 * fold2) / (1.0 + w2);
            let k = 1.0 - g * FINE_MESO_SHARPEN;
            fold = fold.signum() * fold.abs().powf(k);
        }
        let spur = FINE_MESO_AMP_MIN
            + (1.0 - FINE_MESO_AMP_MIN) * super::features::smoothstep(-0.5, 0.5, amp_raw);
        let front_meso = fold * spur;
        (front_meso * (1.0 - FINE_MESO_ISO_WEIGHT) + iso * FINE_MESO_ISO_WEIGHT).clamp(-1.0, 1.0)
    }
}

/// A2+A3 massif-corridor meso field (relief-spectrum spec §13): irregular
/// anisotropic uplift massifs on a jittered along-strike lattice, minus branching
/// low-uplift valley corridors rooted at the outer hinterland — the object
/// vocabulary is "massifs separated by corridors", so the seeded valleys and the
/// emergent ridges agree (the fold train's measured grammar failure). Shares the
/// fold train's delivery, wavelength dial, and isotropic blend.
struct MassifCorridorSampler {
    wavelength: f32,
    iso_frequency: f64,
    seed: u64,
    wobble_fbm: Fbm<Perlin>,
    iso_fbm: Fbm<Perlin>,
}

/// Deterministic per-site hash -> [0, 1) (splitmix64 finalizer).
fn meso_site_hash(seed: u64, chain: u32, k: i64, salt: u64) -> f32 {
    let mut x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((chain as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add((k as u64).wrapping_mul(0x94D0_49BB_1331_11EB))
        .wrapping_add(salt.wrapping_mul(0xD6E8_FEB8_6659_FD93));
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 40) as f32 / (1u64 << 24) as f32
}

impl MassifCorridorSampler {
    fn new(seed: u64, wavelength_km: f32) -> Self {
        // Floor the wavelength well above mesh scale: metre-scale lattice periods
        // would make the finite site windows meaningless.
        let wl_km = wavelength_km.max(5.0);
        Self {
            wavelength: wl_km / PLANET_RADIUS_KM,
            // Scale the isotropic blend with the construction (one dial scales all).
            iso_frequency: FINE_MESO_ISO_FREQUENCY * (FINE_MESO_WAVELENGTH_KM / wl_km) as f64,
            seed,
            wobble_fbm: Fbm::<Perlin>::new(seed.wrapping_add(75) as u32).set_octaves(2),
            iso_fbm: Fbm::<Perlin>::new(seed.wrapping_add(74) as u32).set_octaves(3),
        }
    }

    fn sample(&self, c: Vec3, u: f32, v: f32, chain_id: u32) -> f32 {
        let lam = self.wavelength;
        let min_sigma = FINE_MESO_MIN_SIGMA_KM / PLANET_RADIUS_KM;
        let w_h = FINE_OROGEN_HINTERLAND_WIDTH;
        let w_f = FINE_OROGEN_FORELAND_WIDTH;

        // -- Massifs (A2): jittered u-lattice, anisotropic Gaussians, heavy-tailed
        // amplitudes, centers offset off the crest toward either flank.
        let m_period = FINE_MESO_MASSIF_PERIOD * lam;
        let mk = (u / m_period).floor() as i64;
        let mut massifs = 0.0f32;
        for k in (mk - 3)..=(mk + 3) {
            let h_pos = meso_site_hash(self.seed, chain_id, k, 1);
            let h_v = meso_site_hash(self.seed, chain_id, k, 2);
            let h_lu = meso_site_hash(self.seed, chain_id, k, 3);
            let h_lv = meso_site_hash(self.seed, chain_id, k, 4);
            let h_a = meso_site_hash(self.seed, chain_id, k, 5);
            let u_i = (k as f32 + 0.5 + (h_pos - 0.5) * 0.8) * m_period;
            let v_i = -0.15 * w_f + h_v * (0.6 * w_h + 0.15 * w_f);
            let l_u = ((0.4 + 0.8 * h_lu) * lam).max(min_sigma);
            let l_v = ((0.3 + 0.5 * h_lv) * lam).max(min_sigma);
            let a = 0.35 + 0.65 * h_a * h_a * h_a; // heavy tail: a few dominant
            let du = (u - u_i) / l_u;
            let dv = (v - v_i) / l_v;
            massifs += a * (-0.5 * (du * du + dv * dv)).exp();
        }

        // -- Corridors (A3): jittered u-lattice of transverse valley paths rooted
        // at the outer hinterland, drifting obliquely and wobbling as they descend
        // toward the crest; most fade below the crest (interdigitating heads), a
        // hashed minority crosses into the foreland (water gaps).
        let c_period = FINE_MESO_CORRIDOR_PERIOD * lam;
        // Corridors live on the hinterland flank; distance descended from the root.
        let descent = w_h - v;
        let mut corridors = 0.0f32;
        if v < w_h + 2.0 * lam {
            // A corridor's path drifts up to tan(40°)·W_h ≈ 6-7 lattice periods off
            // its root, so a window around the CELL's u misses it near the crest.
            // Instead, per obliquity sign, invert the mean drift to estimate the
            // root index and scan around THAT (window covers the 20-40° spread +
            // jitter + wobble + Gaussian support). Each site contributes only under
            // its own hashed sign, so overlapping windows cannot double-count.
            let (th_lo, th_hi) = FINE_MESO_CORRIDOR_OBLIQUITY_DEG;
            let tan_mid = 0.5 * (th_lo.to_radians().tan() + th_hi.to_radians().tan());
            for sign in [1.0f32, -1.0] {
                let root_est = u - sign * tan_mid * descent;
                let ck = (root_est / c_period).floor() as i64;
                for k in (ck - 4)..=(ck + 4) {
                    let site_sign = if meso_site_hash(self.seed, chain_id, k, 16) < 0.5 {
                        -1.0
                    } else {
                        1.0
                    };
                    if site_sign != sign {
                        continue;
                    }
                    let h_pos = meso_site_hash(self.seed, chain_id, k, 11);
                    let h_th = meso_site_hash(self.seed, chain_id, k, 12);
                    let h_w = meso_site_hash(self.seed, chain_id, k, 13);
                    let h_cross = meso_site_hash(self.seed, chain_id, k, 14);
                    let h_head = meso_site_hash(self.seed, chain_id, k, 15);
                    let u_k = (k as f32 + 0.5 + (h_pos - 0.5) * 0.8) * c_period;
                    let theta = (th_lo + (th_hi - th_lo) * h_th).to_radians() * site_sign;
                    let wobble = self.wobble_fbm.get([
                        (v / (2.0 * lam)) as f64,
                        (k as f64) * 7.31 + chain_id as f64 * 13.7,
                        0.0,
                    ]) as f32
                        * 0.3
                        * lam;
                    let u_path = u_k + theta.tan() * descent + wobble;
                    let w = ((0.35 + 0.25 * h_w) * lam).max(min_sigma);
                    // Head gate: crossers run through the crest into the foreland;
                    // the rest fade out at 0.1-0.3 W_h above the crest.
                    let v_end = if h_cross < FINE_MESO_CORRIDOR_CROSS_FRACTION {
                        -0.8 * w_f
                    } else {
                        (0.1 + 0.2 * h_head) * w_h
                    };
                    let head = super::features::smoothstep(v_end - lam, v_end + lam, v);
                    let du = (u - u_path) / w;
                    corridors += head * (-0.5 * du * du).exp();
                }
            }
        }

        let field = (massifs.min(FINE_MESO_MASSIF_CAP) - corridors * FINE_MESO_CORRIDOR_GAIN)
            .clamp(-1.3, 1.0);
        let iso = self.iso_fbm.get([
            c.x as f64 * self.iso_frequency,
            c.y as f64 * self.iso_frequency,
            c.z as f64 * self.iso_frequency,
        ]) as f32;
        (field * (1.0 - FINE_MESO_ISO_WEIGHT) + iso * FINE_MESO_ISO_WEIGHT).clamp(-1.0, 1.0)
    }
}

/// Style dispatch shared by the uplift-shape and base-relief delivery paths.
enum MesoSampler {
    FoldTrain(MesoFieldSampler),
    MassifCorridor(MassifCorridorSampler),
}

impl MesoSampler {
    fn new(seed: u64, wavelength_km: f32, irregularity: f32, style: usize) -> Self {
        match style {
            0 => Self::FoldTrain(MesoFieldSampler::new(seed, wavelength_km, irregularity)),
            _ => Self::MassifCorridor(MassifCorridorSampler::new(seed, wavelength_km)),
        }
    }

    fn sample(&self, c: Vec3, frame: &MesoFrontFrame) -> f32 {
        match self {
            Self::FoldTrain(s) => s.sample(c, frame.u, frame.v, frame.chain_id),
            Self::MassifCorridor(s) => s.sample(c, frame.u_lin, frame.v, frame.chain_id),
        }
    }
}

#[derive(Clone, Copy)]
struct MesoFrontFrame {
    #[cfg(feature = "research-landscape")]
    front_index: usize,
    u: f32,
    v: f32,
    /// Continuous along-strike coordinate (endpoint-ordered chain walk + within-
    /// segment projection) — the massif-corridor u axis. `u` above is the legacy
    /// midpoint-quantized fold-train coordinate (kept for identity).
    u_lin: f32,
    chain_id: u32,
}

/// Exact-arc coordinates recovered from a retained finite-age source owner.
///
/// This is a research-only provenance seam, not additional terrain state. It lets
/// diagnostics map a fine-cell source sample back onto the same continuous
/// along-strike coordinate and great-circle arc used by the terrain constructor.
#[cfg(feature = "research-landscape")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OwnedFrontProjection {
    /// Index into the aligned vectors in [`OrogenFronts`].
    pub front_index: u32,
    /// Ordered convergent-front chain containing `front_index`.
    pub chain_id: u32,
    /// Continuous endpoint-ordered coordinate along that chain, in radians.
    pub u_lin_radians: f32,
    /// Unsigned great-circle distance to the owned front arc, in radians.
    pub arc_distance_radians: f32,
}

#[derive(Clone, Copy)]
struct FrontArcProjection {
    u_lin: f32,
    #[cfg(feature = "research-landscape")]
    distance: f32,
}

/// Fraction [0, 1] along the great-circle arc a->b of the point closest to `c`
/// (clamped to the endpoints; 0.5 for degenerate arcs).
fn point_on_arc_param(c: Vec3, a: Vec3, b: Vec3) -> f32 {
    let n = a.cross(b);
    if n.length_squared() < 1e-18 {
        return 0.5;
    }
    let n = n.normalize();
    let p = c - n * c.dot(n);
    if p.length_squared() < 1e-12 {
        return 0.5;
    }
    let p = p.normalize();
    let full = a.dot(b).clamp(-1.0, 1.0).acos();
    if full < 1e-9 {
        return 0.5;
    }
    let ta = a.dot(p).clamp(-1.0, 1.0).acos();
    let tb = b.dot(p).clamp(-1.0, 1.0).acos();
    if ta + tb > full + 1e-6 {
        return if ta < tb { 0.0 } else { 1.0 };
    }
    (ta / full).clamp(0.0, 1.0)
}

/// Project a point onto one exact front arc. Both ownership and the public
/// research provenance seam below use this helper so their continuous coordinate
/// cannot acquire independent geometry conventions.
fn project_front_arc(c: Vec3, fronts: &OrogenFronts, front_index: usize) -> FrontArcProjection {
    let (a, b) = (fronts.seg_a[front_index], fronts.seg_b[front_index]);
    let t = point_on_arc_param(c, a, b);
    let seg_len = a.dot(b).clamp(-1.0, 1.0).acos();
    FrontArcProjection {
        u_lin: fronts.u_lin[front_index] + fronts.u_dir[front_index] * (t - 0.5) * seg_len,
        #[cfg(feature = "research-landscape")]
        distance: point_to_arc_distance(c, a, b),
    }
}

/// Recover continuous exact-front coordinates for a retained source owner.
///
/// `owner_front` is a value from [`FrozenSupportUplift::owner_front`]. The
/// sentinel `u32::MAX`, or a malformed/misaligned index, returns `None`. `center`
/// is expected to be a unit-vector fine-cell center.
#[cfg(feature = "research-landscape")]
pub fn project_owned_front(
    center: Vec3,
    owner_front: u32,
    fronts: &OrogenFronts,
) -> Option<OwnedFrontProjection> {
    let front_index = usize::try_from(owner_front).ok()?;
    if owner_front == u32::MAX
        || front_index >= fronts.seg_a.len()
        || front_index >= fronts.seg_b.len()
        || front_index >= fronts.chain_id.len()
        || front_index >= fronts.u_lin.len()
        || front_index >= fronts.u_dir.len()
    {
        return None;
    }
    let projection = project_front_arc(center, fronts, front_index);
    (projection.u_lin.is_finite() && projection.distance.is_finite()).then_some(
        OwnedFrontProjection {
            front_index: owner_front,
            chain_id: fronts.chain_id[front_index],
            u_lin_radians: projection.u_lin,
            arc_distance_radians: projection.distance,
        },
    )
}

fn meso_front_tree(fronts: &OrogenFronts) -> Option<CoarseTree> {
    (!fronts.points.is_empty()).then(|| {
        let entries: Vec<[f32; 3]> = fronts.points.iter().map(|p| [p.x, p.y, p.z]).collect();
        ImmutableKdTree::<f32, 3>::new_from_slice(&entries)
    })
}

fn meso_front_gather_r2() -> f32 {
    let gather_chord =
        2.0 * ((FINE_OROGEN_HINTERLAND_WIDTH + FINE_FRONT_GATHER_MARGIN) * 0.5).sin();
    gather_chord * gather_chord
}

fn nearest_meso_front_frame(
    c: Vec3,
    plate: u32,
    fronts: &OrogenFronts,
    tree: &CoarseTree,
    gather_r2: f32,
) -> Option<MesoFrontFrame> {
    let mut best_d = f32::INFINITY;
    let mut best_side = 1.0f32;
    let mut best_front = usize::MAX;
    for nn in tree.within_unsorted::<SquaredEuclidean>(&[c.x, c.y, c.z], gather_r2) {
        let item = nn.item as usize;
        let side = match fronts.accept_plate[item] {
            None => 1.0, // collision: treat as overriding (one gentle flank)
            Some(p) => {
                if p == plate {
                    1.0
                } else {
                    -1.0
                }
            }
        };
        let d = point_to_arc_distance(c, fronts.seg_a[item], fronts.seg_b[item]);
        if d < best_d {
            best_d = d;
            best_side = side;
            best_front = item;
        }
    }
    if !best_d.is_finite() {
        return None;
    }
    let projection = project_front_arc(c, fronts, best_front);
    Some(MesoFrontFrame {
        #[cfg(feature = "research-landscape")]
        front_index: best_front,
        u: fronts.arc_u[best_front],
        v: best_side * best_d,
        u_lin: projection.u_lin,
        chain_id: fronts.chain_id[best_front],
    })
}

/// Frozen present-front source for the first finite-age coupled-landscape slice.
/// Each cell has one nearest exact-arc owner. Relative uplift opportunity is the
/// owner's positive local convergence times a declared cross-front profile;
/// duration is the owner's retained component age. Legacy relief is not used as
/// the spatial source: its removed volume calibrates the hypothetical all-old
/// source rate, while actual ages determine the candidate's integrated work.
#[cfg(feature = "research-landscape")]
pub struct FrozenSupportUplift {
    pub shape: Vec<f32>,
    pub duration_myr: Vec<f32>,
    /// Per-fine-cell index into [`OrogenFronts`], or `u32::MAX` when the cell has
    /// no positive finite-age source.
    pub owner_front: Vec<u32>,
    pub owned_cells: usize,
    pub distinct_durations: usize,
}

#[cfg(feature = "research-landscape")]
pub fn frozen_support_uplift(base: &FineBase, fronts: &OrogenFronts) -> FrozenSupportUplift {
    let n = base.tessellation.num_cells();
    let Some(tree) = meso_front_tree(fronts) else {
        return FrozenSupportUplift {
            shape: vec![0.0; n],
            duration_myr: vec![0.0; n],
            owner_front: vec![u32::MAX; n],
            owned_cells: 0,
            distinct_durations: 0,
        };
    };
    let gather_r2 = meso_front_gather_r2();
    let sample = |i: usize| -> (f32, f32, u32) {
        let center = base.tessellation.cell_center(i);
        let plate = fronts.coarse_cell_plate[base.coarse_cell[i]];
        let Some(front) = nearest_meso_front_frame(center, plate, fronts, &tree, gather_r2) else {
            return (0.0, 0.0, u32::MAX);
        };
        let owner = front.front_index;
        if fronts.accept_plate[owner].is_some_and(|receiver| receiver != plate) {
            return (0.0, 0.0, u32::MAX);
        }
        let profile = if front.v < 0.0 {
            super::features::smoothstep(-FINE_OROGEN_FORELAND_WIDTH, 0.0, front.v)
        } else {
            1.0 - super::features::smoothstep(0.0, FINE_OROGEN_HINTERLAND_WIDTH, front.v)
        };
        let rate = fronts.convergence_km_per_myr[owner] * profile;
        let duration = fronts.episode_duration_myr[owner];
        if rate > 0.0 && duration > 0.0 {
            (
                rate,
                duration,
                u32::try_from(owner).expect("orogen front index exceeds u32 provenance capacity"),
            )
        } else {
            (0.0, 0.0, u32::MAX)
        }
    };
    #[cfg(not(feature = "single-threaded"))]
    let samples: Vec<(f32, f32, u32)> = (0..n).into_par_iter().map(sample).collect();
    #[cfg(feature = "single-threaded")]
    let samples: Vec<(f32, f32, u32)> = (0..n).map(sample).collect();
    let shape: Vec<f32> = samples.iter().map(|&(rate, _, _)| rate).collect();
    let duration_myr: Vec<f32> = samples.iter().map(|&(_, duration, _)| duration).collect();
    let owner_front: Vec<u32> = samples.iter().map(|&(_, _, owner)| owner).collect();
    let owned_cells = shape.iter().filter(|&&rate| rate > 0.0).count();
    let distinct_durations = duration_myr
        .iter()
        .filter(|&&duration| duration > 0.0)
        .map(|duration| duration.to_bits())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    FrozenSupportUplift {
        shape,
        duration_myr,
        owner_front,
        owned_cells,
        distinct_durations,
    }
}

/// Candidate A' base-elevation meso relief: the same front-coordinate meso field used
/// by the emergent uplift variant, painted into the pre-erosion substrate and
/// area-zeroed per coarse cell so it does not move the coarse datum.
fn add_meso_base_relief(
    tess: &Tessellation,
    coarse_cell: &[usize],
    fields: &ElevationFields,
    fronts: &OrogenFronts,
    base_elev: &mut [f32],
    seed: u64,
    amplitude: f32,
    wavelength_km: f32,
    meso_irregularity: f32,
    meso_style: usize,
) {
    if amplitude <= 0.0 {
        return;
    }
    let Some(front_tree) = meso_front_tree(fronts) else {
        return;
    };
    let n = tess.num_cells();
    let sstep = super::features::smoothstep;
    let meso = MesoSampler::new(seed, wavelength_km, meso_irregularity, meso_style);
    let gather_r2 = meso_front_gather_r2();

    // Same high-orogen gate shape as the interior grain: an elevation ramp keeps
    // negative excursions well above sea level, and an arc+collision envelope keeps
    // plains/cratons clean.
    let forcing: Vec<f32> = (0..n)
        .map(|i| (fields.arc[i] + fields.collision[i]).max(0.0))
        .collect();
    let fmax = (0..n)
        .filter(|&i| base_elev[i] >= 0.0)
        .map(|i| forcing[i])
        .fold(0.0f32, f32::max)
        .max(1e-6);

    let sample = |i: usize| -> (f32, bool) {
        let elev_gate = sstep(
            FINE_INTERIOR_MIN_ELEV,
            FINE_INTERIOR_MIN_ELEV + FINE_INTERIOR_ELEV_BAND,
            base_elev[i],
        );
        if elev_gate <= 0.0 {
            return (0.0, false);
        }
        let force_gate = sstep(
            FINE_INTERIOR_FORCING_THRESHOLD,
            FINE_INTERIOR_FORCING_THRESHOLD + FINE_INTERIOR_FORCING_BAND,
            forcing[i] / fmax,
        );
        let gate = elev_gate * force_gate;
        if gate <= 0.0 {
            return (0.0, false);
        }
        let c = tess.cell_center(i);
        let plate = fronts.coarse_cell_plate[coarse_cell[i]];
        let Some(front) = nearest_meso_front_frame(c, plate, fronts, &front_tree, gather_r2) else {
            return (0.0, false);
        };
        (amplitude * gate * meso.sample(c, &front), true)
    };
    #[cfg(not(feature = "single-threaded"))]
    let raw: Vec<(f32, bool)> = (0..n).into_par_iter().map(sample).collect();
    #[cfg(feature = "single-threaded")]
    let raw: Vec<(f32, bool)> = (0..n).map(sample).collect();

    let areas = tess.cell_areas();
    let ncoarse = coarse_cell.iter().copied().max().map_or(0, |m| m + 1);
    let mut sum_wd = vec![0.0f64; ncoarse];
    let mut sum_w = vec![0.0f64; ncoarse];
    for i in 0..n {
        if raw[i].1 {
            let c = coarse_cell[i];
            sum_wd[c] += (areas[i] * raw[i].0) as f64;
            sum_w[c] += areas[i] as f64;
        }
    }
    for i in 0..n {
        if raw[i].1 {
            let c = coarse_cell[i];
            let mean = if sum_w[c] > 0.0 {
                (sum_wd[c] / sum_w[c]) as f32
            } else {
                0.0
            };
            base_elev[i] += raw[i].0 - mean;
        }
    }
}

/// O0 structured-emergent uplift SHAPE (orogen-structure.md): per cell, the demoted
/// orogen forcing (`target − base`) shaped by an ASYMMETRIC front profile (steep narrow
/// foreland flank → crest at the front → gentle wide hinterland; side from overriding-vs-
/// foreland plate membership) × a low-frequency along-strike SEGMENTATION proxy (the range
/// plunges/segments). Returned UN-normalized — the erosion builder volume-normalizes it so
/// total uplift is preserved while the distribution becomes tectonic. `structured ∈ (0,1]`
/// blends the shaped field with the flat demoted forcing.
#[allow(clippy::too_many_arguments)]
fn compute_emergent_uplift_shape(
    tess: &Tessellation,
    coarse_cell: &[usize],
    fronts: &OrogenFronts,
    target: &[f32],
    base: &[f32],
    seed: u64,
    structured: f32,
    meso_relief: f32,
    meso_wavelength_km: f32,
    meso_irregularity: f32,
    meso_style: usize,
) -> Vec<f32> {
    let n = tess.num_cells();
    let blend = structured.clamp(0.0, 1.0);
    let meso_strength = meso_relief.clamp(0.0, 1.0);
    let sstep = super::features::smoothstep;
    let seg_fbm = Fbm::<Perlin>::new(seed.wrapping_add(71) as u32).set_octaves(2);
    let meso = MesoSampler::new(seed, meso_wavelength_km, meso_irregularity, meso_style);
    let front_tree = meso_front_tree(fronts);
    let gather_r2 = meso_front_gather_r2();

    let sample = |i: usize| -> f32 {
        let demoted = (target[i] - base[i]).max(0.0);
        if demoted <= 1e-6 {
            return 0.0; // not an (emergent) orogen cell
        }
        let Some(tree) = front_tree.as_ref() else {
            return demoted; // no fronts — fall back to flat forcing
        };
        let c = tess.cell_center(i);
        let plate = fronts.coarse_cell_plate[coarse_cell[i]];
        let Some(front) = nearest_meso_front_frame(c, plate, fronts, tree, gather_r2) else {
            return demoted;
        };
        let v = front.v; // signed front-normal distance
        let profile = if v < 0.0 {
            sstep(-FINE_OROGEN_FORELAND_WIDTH, 0.0, v) // steep narrow foreland rise
        } else {
            1.0 - sstep(0.0, FINE_OROGEN_HINTERLAND_WIDTH, v) // gentle wide hinterland
        };
        // Principled along-strike segmentation: 1-D noise of the nearest front's ARC-LENGTH
        // coordinate (decorrelated per chain), so the range plunges/segments ALONG its
        // length coherently — not the 3D-noise proxy (blobs/seams at kinks).
        let u = front.u;
        let chain = front.chain_id as f32;
        let raw = seg_fbm.get([
            (u * FINE_OROGEN_SEGMENT_FREQUENCY as f32) as f64 + chain as f64 * 53.13,
            0.0,
            0.0,
        ]) as f32;
        let seg = FINE_OROGEN_SEGMENT_MIN + (1.0 - FINE_OROGEN_SEGMENT_MIN) * sstep(-0.4, 0.4, raw);
        let mut shaped = demoted * profile * seg;
        if meso_strength > 0.0 {
            shaped *= (1.0 + meso_strength * meso.sample(c, &front)).max(0.0);
        }
        demoted * (1.0 - blend) + shaped * blend
    };
    #[cfg(not(feature = "single-threaded"))]
    {
        (0..n).into_par_iter().map(sample).collect()
    }
    #[cfg(feature = "single-threaded")]
    {
        (0..n).map(sample).collect()
    }
}

/// Great-circle distance (radians) from a unit point `p` to the minor great-circle
/// ARC between unit endpoints `a` and `b`. If the foot of the perpendicular lies on
/// the arc it's the cross-track distance; otherwise the nearer endpoint. Used so the
/// P1b front distance field is a true offset from the boundary polyline (parallel
/// iso-contours), not a bullseye around a point anchor.
fn point_to_arc_distance(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ang = |x: Vec3, y: Vec3| x.dot(y).clamp(-1.0, 1.0).acos();
    let n = a.cross(b);
    let nlen = n.length();
    if nlen < 1e-9 {
        // Degenerate arc (coincident/antipodal endpoints): fall back to an endpoint.
        return ang(p, a);
    }
    let n = n / nlen;
    let foot = p - n * p.dot(n); // project p onto the arc's great-circle plane
    if foot.length() > 1e-9 {
        let foot = foot.normalize();
        let ab = ang(a, b);
        // Foot is on the minor arc iff it doesn't overshoot either endpoint.
        if ang(a, foot) <= ab + 1e-4 && ang(b, foot) <= ab + 1e-4 {
            return p.dot(n).abs().clamp(0.0, 1.0).asin(); // cross-track distance
        }
    }
    ang(p, a).min(ang(p, b))
}

/// Synthesize the zero-mean interior structural relief (erosion-v2 Phase 1) and ADD
/// it to the fine base, in place. The interpolated coarse elevation is smooth in
/// orogen interiors (the distance-decay forcing saturates there → flat-topped highs),
/// so erosion has no drainage gradient to organize and degenerates into cottage-
/// cheese. This imposes a mid-band fault-block / fold-grain HEIGHT field — the
/// SUBSTRATE erosion dissects into real ranges — gated to high orogen terrain and
/// made coarse-cell-local zero-mean so it adds sub-coarse structure WITHOUT shifting
/// the coarse sea-level datum or land fraction (root cause #1 / fix #7,
/// docs/archive/specs/erosion-fine-synthesis.md).
///
/// Two grains, blended by `front_strike_weight` and proximity to a front:
/// - **Isotropic fBm (P1a):** soft, orientation-free. The fallback everywhere, and
///   the only grain where no convergent front is near.
/// - **Strike-banded (P1b):** a banded function of the signed great-circle distance
///   to the nearest *compatible* convergent front (`fronts`). The distance field's
///   iso-contours run parallel to the front, so the bands are ridge-and-valley grain
///   STRIKING ALONG the orogen (fold-and-thrust fabric) — no per-cell strike vector
///   (which would seam at kinks). "Compatible" = the front's overriding side for a
///   subduction front, either side for collision (so grain doesn't bleed onto the
///   subducting plate). The fronts drive ORIENTATION only; amplitude stays gated.
///
/// P1c layers an active/passive MARGIN contrast on top: a coastal-band amplitude
/// scale (from the interpolated `margin_distance` × convergent forcing) that sharpens
/// an active (convergent) coast and damps a passive one — amplitude only, so the land
/// mask is untouched. `margin_contrast == 0` reduces to P1b. (A high forced cell on
/// OCEANIC crust — a volcanic island arc — has a negative margin distance and so reads
/// as "coastal"; since it's convergent it gets the ACTIVE sharpening, which is the
/// right call for a steep arc. A rare non-convergent oceanic highland gets mild
/// passive damping, which is harmless.)
///
/// `amplitude == 0` is a no-op (pure interpolant — the old flat-top behaviour).
#[allow(clippy::too_many_arguments)]
fn add_interior_structural_relief(
    tess: &Tessellation,
    coarse_cell: &[usize],
    fields: &ElevationFields,
    fronts: &OrogenFronts,
    margin_distance: &[f32],
    base_elev: &mut [f32],
    seed: u64,
    amplitude: f32,
    front_strike_weight: f32,
    margin_contrast: f32,
) {
    if amplitude <= 0.0 {
        return;
    }
    let n = tess.num_cells();

    // A deformation model supplies accumulated strain directly. Legacy models
    // retain their historical arc+collision+influence eligibility field.
    let strain_driven = fields
        .tectonic_strain
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        > 1e-6;
    let forcing: Vec<f32> = if strain_driven {
        fields.tectonic_strain.clone()
    } else {
        (0..n)
            .map(|i| (fields.collision[i] + fields.convergent[i] + fields.arc[i]).max(0.0))
            .collect()
    };
    let fmax = (0..n)
        .filter(|&i| strain_driven || base_elev[i] >= 0.0)
        .map(|i| forcing[i])
        .fold(0.0f32, f32::max)
        .max(1e-6);

    // Convergent forcing land-max, for the active-margin (subduction) activity in the
    // P1c margin contrast (active = convergent front at the coast). The contrast is a
    // strength dial in [0,1] (the active/passive FACTORS set the magnitude); clamp so a
    // sweep can't push passive `margin_scale` negative and INVERT relief (codex).
    let margin_contrast = margin_contrast.clamp(0.0, 1.0);
    let conv_max = (0..n)
        .filter(|&i| base_elev[i] >= 0.0)
        .map(|i| fields.convergent[i].max(0.0))
        .fold(0.0f32, f32::max)
        .max(1e-6);

    // Blend knob is a weight in [0,1]; clamp so the band/isotropic mix (and hence the
    // ~[-1,1] boundedness the zero-mean + land-drift argument relies on) holds even if
    // a sweep passes a larger value.
    let strike_weight = front_strike_weight.clamp(0.0, 1.0);

    let fbm = Fbm::<Perlin>::new(seed.wrapping_add(61) as u32).set_octaves(FINE_INTERIOR_OCTAVES);
    let warp_fbm = Fbm::<Perlin>::new(seed.wrapping_add(62) as u32).set_octaves(2);
    let sstep = super::features::smoothstep;

    // KD-tree over the convergent-front edge-midpoint anchors (only when banding is
    // actually requested and there are fronts to band toward). The anchors gather
    // CANDIDATE arcs near a cell; the distance is then measured to the arc itself.
    let front_tree = (strike_weight > 0.0 && !fronts.points.is_empty()).then(|| {
        let entries: Vec<[f32; 3]> = fronts.points.iter().map(|p| [p.x, p.y, p.z]).collect();
        ImmutableKdTree::<f32, 3>::new_from_slice(&entries)
    });
    // Candidate gather radius (squared chord): the influence radius plus a margin of
    // one half-anchor-spacing, so an arc whose midpoint sits just past the influence
    // radius but whose body enters it is still considered.
    let gather_chord = 2.0 * ((FINE_FRONT_INFLUENCE_RADIUS + FINE_FRONT_GATHER_MARGIN) * 0.5).sin();
    let gather_r2 = gather_chord * gather_chord;

    // Great-circle distance (radians, unsigned) to the nearest front ARC on this
    // cell's structural side, or None if no compatible front is within reach. Unsigned
    // is fine: the band phase uses `cos` (even) and side-awareness already filters the
    // wrong (subducting) side, so a signed front-normal coordinate isn't needed.
    let nearest_front_dist = |i: usize, c: Vec3| -> Option<f32> {
        let tree = front_tree.as_ref()?;
        let plate = fronts.coarse_cell_plate[coarse_cell[i]];
        let mut best = f32::INFINITY;
        for nn in tree.within_unsorted::<SquaredEuclidean>(&[c.x, c.y, c.z], gather_r2) {
            let item = nn.item as usize;
            let compatible = match fronts.accept_plate[item] {
                None => true,          // collision: both sides build mountains
                Some(p) => p == plate, // subduction: overriding side only
            };
            if compatible {
                let d = point_to_arc_distance(c, fronts.seg_a[item], fronts.seg_b[item]);
                if d < best {
                    best = d;
                }
            }
        }
        best.is_finite().then_some(best)
    };

    // Raw gated relief + gate mask per cell. Two gates keep it on high orogen
    // interiors: an elevation ramp (so the zero-mean negative excursions stay well
    // above sea level → land fraction preserved by construction) and a forcing ramp
    // (so cratons stay quiet). `in_gate` records membership in the zero-mean set
    // independently of whether the noise sample happens to be exactly 0.
    let sample = |i: usize| -> (f32, bool) {
        // Strain is physical eligibility and remains valid for low/submarine
        // active structures. Legacy has no strain state, so preserve its land-
        // safety elevation ramp exactly.
        let elev_gate = if strain_driven {
            1.0
        } else {
            sstep(
                FINE_INTERIOR_MIN_ELEV,
                FINE_INTERIOR_MIN_ELEV + FINE_INTERIOR_ELEV_BAND,
                base_elev[i],
            )
        };
        if elev_gate <= 0.0 {
            return (0.0, false);
        }
        let force_gate = sstep(
            FINE_INTERIOR_FORCING_THRESHOLD,
            FINE_INTERIOR_FORCING_THRESHOLD + FINE_INTERIOR_FORCING_BAND,
            forcing[i] / fmax,
        );
        let gate = elev_gate * force_gate;
        if gate <= 0.0 {
            return (0.0, false);
        }
        let c = tess.cell_center(i);
        let p = c * FINE_INTERIOR_FREQUENCY as f32;
        let isotropic = fbm.get([p.x as f64, p.y as f64, p.z as f64]) as f32;

        // Blend toward the strike-banded grain near a compatible front, fading back
        // to isotropic at the influence radius.
        let value = match nearest_front_dist(i, c) {
            Some(d) if d < FINE_FRONT_INFLUENCE_RADIUS => {
                let pw = c * FINE_FRONT_WARP_FREQUENCY as f32;
                let warp =
                    FINE_FRONT_WARP * warp_fbm.get([pw.x as f64, pw.y as f64, pw.z as f64]) as f32;
                // cos of the (warped) front distance → ridges parallel to the front.
                let band =
                    (std::f32::consts::TAU * FINE_FRONT_BAND_FREQUENCY as f32 * (d + warp)).cos();
                let w = strike_weight * (1.0 - sstep(0.0, FINE_FRONT_INFLUENCE_RADIUS, d));
                w * band + (1.0 - w) * isotropic
            }
            _ => isotropic,
        };

        // P1c active/passive margin contrast: a coastal-band amplitude scale, full at
        // the coast and fading to neutral (1.0) inland, that sharpens an ACTIVE
        // (convergent) margin and damps a PASSIVE one. Modulates amplitude only (stays
        // within the elevation gate), so it never moves the land/ocean mask.
        let margin_scale = if margin_contrast > 0.0 {
            let coastal = 1.0 - sstep(0.0, FINE_MARGIN_WIDTH, margin_distance[i].max(0.0));
            let activity = (fields.convergent[i].max(0.0) / conv_max).clamp(0.0, 1.0);
            let target = FINE_MARGIN_PASSIVE_FACTOR
                + activity * (FINE_MARGIN_ACTIVE_FACTOR - FINE_MARGIN_PASSIVE_FACTOR);
            1.0 + margin_contrast * coastal * (target - 1.0)
        } else {
            1.0
        };
        (gate * amplitude * margin_scale * value, true)
    };
    #[cfg(not(feature = "single-threaded"))]
    let raw: Vec<(f32, bool)> = (0..n).into_par_iter().map(sample).collect();
    #[cfg(feature = "single-threaded")]
    let raw: Vec<(f32, bool)> = (0..n).map(sample).collect();

    // Coarse-cell-local zero-mean (area-weighted) over the GATED cells only:
    // subtract each coarse cell's gated mean from its gated fine cells, leaving
    // ungated cells at exactly 0. The area-weighted perturbation over every coarse
    // cell is then ~0, so the coarse datum and per-cell mean elevation are preserved
    // (no fine sea-level re-solve), and untouched lowland/ocean cells cannot drift
    // across the land threshold.
    let areas = tess.cell_areas();
    let ncoarse = coarse_cell.iter().copied().max().map_or(0, |m| m + 1);
    let mut sum_wd = vec![0.0f64; ncoarse];
    let mut sum_w = vec![0.0f64; ncoarse];
    for i in 0..n {
        if raw[i].1 {
            let c = coarse_cell[i];
            sum_wd[c] += (areas[i] * raw[i].0) as f64;
            sum_w[c] += areas[i] as f64;
        }
    }
    for i in 0..n {
        if raw[i].1 {
            let c = coarse_cell[i];
            let mean = if sum_w[c] > 0.0 {
                (sum_wd[c] / sum_w[c]) as f32
            } else {
                0.0
            };
            base_elev[i] += raw[i].0 - mean;
        }
    }
}

/// Area-weighted land-fraction drift from the structural relief: `land = elev >= 0`,
/// weighted by cell area (the fine mesh is adaptive, so a cell COUNT over-weights the
/// dense mountain cells). Logs the before/after land fraction and the drift; warns if
/// it exceeds a small tolerance — the structural relief is meant to add sub-coarse
/// detail WITHOUT moving the land/ocean mask the coarse atmosphere was solved on
/// (erosion-fine-synthesis.md). `~0` at the default knobs; a warning flags a knob
/// (high `interior_relief`, or scarps) flipping near-sea-level cells.
fn report_land_fraction_drift(tess: &Tessellation, before: &[f32], after: &[f32]) {
    let areas = tess.cell_areas();
    let mut total = 0.0f64;
    let mut land_before = 0.0f64;
    let mut land_after = 0.0f64;
    for i in 0..tess.num_cells() {
        let a = areas[i] as f64;
        total += a;
        if before[i] >= 0.0 {
            land_before += a;
        }
        if after[i] >= 0.0 {
            land_after += a;
        }
    }
    if total <= 0.0 {
        return;
    }
    let (fb, fa) = (land_before / total, land_after / total);
    let drift = fa - fb;
    log::info!(
        "fine mesh: land fraction (area-weighted) {:.4} -> {:.4} (drift {:+.2e})",
        fb,
        fa,
        drift
    );
    if drift.abs() > FINE_STRUCTURE_LAND_DRIFT_TOL as f64 {
        log::warn!(
            "fine mesh: structural relief shifted land fraction by {:+.2e} (> tol {:.0e}); a knob is flipping near-sea-level cells — the coarse atmosphere mask is no longer consistent",
            drift,
            FINE_STRUCTURE_LAND_DRIFT_TOL
        );
    }
}

/// Emergent-orogens audit (erosion-v3): how much intended-LAND the envelope demotion
/// pushed below sea level. The builder uplift gates on the coarse-target mask so these
/// cells DO rebuild, but a large area-weighted flip fraction means the demotion is
/// drowning real coast (the decomposition doesn't commute with the coarse sea-level
/// solve), so λ is too aggressive. Logs the area-weighted fraction of cells with
/// `target >= 0` but `envelope < 0`.
fn report_envelope_land_flips(tess: &Tessellation, target: &[f32], envelope: &[f32]) {
    let areas = tess.cell_areas();
    let mut land = 0.0f64;
    let mut flipped = 0.0f64;
    for i in 0..tess.num_cells() {
        if target[i] >= 0.0 {
            let a = areas[i] as f64;
            land += a;
            if envelope[i] < 0.0 {
                flipped += a;
            }
        }
    }
    let frac = if land > 0.0 { flipped / land } else { 0.0 };
    log::info!(
        "fine mesh: emergent envelope land flips (area-weighted) {:.2}% of land (target>=0, envelope<0; rebuilt by uplift but flags too-aggressive λ)",
        100.0 * frac
    );
}

/// Fault range-front scarps (v1 proxy): sharpen active orogen margins on the fine
/// base BEFORE erosion. The lithosphere is a smooth isostatic field, so orogen
/// margins grade out softly; this imposes a localized scarp at the contour of the
/// (land-normalized) collision+convergence forcing — which follows the plate
/// boundary, so the front is boundary-seeded. The displacement is the derivative-
/// of-Gaussian of the forcing across that contour: footwall edge up, hanging-wall/
/// basin edge down, tapering to zero in the deep interior and far basin (a
/// steepening, not a second uplift). Erosion then cuts canyons through the abrupt
/// front and triangular facets emerge between them. Land cells only; 0 = no-op.
fn apply_fault_scarps(base_elev: &mut [f32], fields: &ElevationFields, scarp_height: f32) {
    if scarp_height <= 0.0 {
        return;
    }
    let n = base_elev.len();
    let forcing: Vec<f32> = (0..n)
        .map(|i| (fields.collision[i] + fields.convergent[i]).max(0.0))
        .collect();
    // Normalize by the land maximum (robust across weak/strong-orogen worlds).
    let fmax = (0..n)
        .filter(|&i| base_elev[i] >= 0.0)
        .map(|i| forcing[i])
        .fold(0.0f32, f32::max)
        .max(1e-6);
    for i in 0..n {
        if base_elev[i] < 0.0 {
            continue; // land only
        }
        let z = (forcing[i] / fmax - FAULT_FRONT_THRESHOLD) / FAULT_FRONT_BAND;
        // +ve just inside the contour (footwall up), −ve just outside (basin down,
        // damped so it sharpens the front without drowning the lowland).
        let mut scarp = scarp_height * z * (-0.5 * z * z).exp();
        if scarp < 0.0 {
            scarp *= FAULT_BASIN_DROP_FRAC;
        }
        // Clamp to sea level: the basin-drop must never push a coastal land cell
        // below 0. Hydrology's lake/ocean set was derived from the UN-faulted
        // base, so a scarp silently submerging a cell it considered dry would
        // desync the erosion datum from the lake topology (audit H7).
        base_elev[i] = (base_elev[i] + scarp).max(0.0);
    }
}

/// Per-cell base level from terminal (endorheic) pre-erosion lakes: a lake whose
/// basin does NOT overflow is a true sink, so its surface elevation becomes a
/// fixed base level for the erosion loop — inflowing rivers grade to it and the
/// basin drains internally instead of being carved over its spill. Overflowing
/// (through-flowing) lakes are left to the priority-flood spill routing.
/// NEG_INFINITY for non-lake cells (no constraint beyond sea level), so a
/// lakeless world reproduces the sea-level-only behaviour exactly.
fn terminal_lake_base_levels(tess: &Tessellation, hydrology: &Hydrology) -> Vec<f32> {
    let n = tess.num_cells();
    let mut lake_base = vec![f32::NEG_INFINITY; n];
    for i in 0..n {
        if hydrology.water_state(i) != CellWaterState::LakeWater {
            continue;
        }
        if let Some(bid) = hydrology.basin_id[i] {
            let basin = &hydrology.basins[bid];
            if !basin.is_overflowing() {
                lake_base[i] = basin.water_level;
            }
        }
    }
    lake_base
}

/// Climate↔erosion feedback: modulate the (correct) coarse precipitation by the
/// orographic forcing on the eroded fine relief — windward slopes wetter, lee
/// drier (rain shadows behind the carved ranges). The coarse moisture model
/// already supplied the large-scale transport and interior drying; this adds the
/// fine-scale rain shadow the coarse mesh couldn't resolve. Re-running the full
/// transport on the adaptive fine mesh over-dries interiors (tiny land cells →
/// tiny CFL dt), so we modulate rather than re-transport. Renormalized to land
/// mean 1.0 so hydrology budgets are unchanged in the mean.
fn fine_precipitation(
    tess: &Tessellation,
    elevation: &[f32],
    wind: &[Vec3],
    coarse_precip: &[f32],
    strength: f32,
    downwind_strength: f32,
) -> Vec<f32> {
    let n = tess.num_cells();
    if strength <= 0.0 && downwind_strength <= 0.0 {
        return coarse_precip.to_vec();
    }

    // Signed orographic forcing: wind·∇elev (chord gradient — acos collapses on
    // the fine mesh), positive upslope (windward), negative downslope (lee),
    // weighted by height so high terrain forces hardest.
    let mut oro = vec![0.0f32; n];
    for i in 0..n {
        if elevation[i] <= 0.0 {
            continue;
        }
        let grad = chord_gradient(tess, elevation, i);
        if grad.length_squared() < 1e-10 {
            continue;
        }
        let height_factor = (elevation[i] / OROGRAPHIC_FULL_HEIGHT).clamp(0.0, 1.0);
        oro[i] = wind[i].dot(grad) * height_factor;
    }

    // Mesh-independent scale: a high percentile of |oro| maps to the full
    // modulation strength (so the knob means the same on any resolution).
    let mut mags: Vec<f32> = oro.iter().map(|o| o.abs()).filter(|m| *m > 0.0).collect();
    let scale = if mags.is_empty() {
        1.0
    } else {
        mags.sort_by(f32::total_cmp);
        mags[((mags.len() - 1) as f32 * UPLIFT_NORM_PERCENTILE) as usize].max(1e-12)
    };

    // Windward wetter / lee drier on top of the coarse precip.
    let mut precip: Vec<f32> = (0..n)
        .map(|i| {
            let f = (1.0 + strength * (oro[i] / scale).clamp(-1.0, 1.0))
                .clamp(OROGRAPHIC_PRECIP_MIN, OROGRAPHIC_PRECIP_MAX);
            (coarse_precip[i] * f).max(0.0)
        })
        .collect();

    // Downwind rain shadow (rain-shadow.md): propagate each cell's LOCAL lee dry-anomaly
    // downwind along the per-cell wind so lee basins keep drying instead of snapping back
    // to coarse precip. One ordered O(N) DAG sweep — no CFL advection (the over-dry trap).
    if downwind_strength > 0.0 {
        precip = downwind_lee_shadow(
            tess,
            elevation,
            wind,
            coarse_precip,
            &precip,
            downwind_strength,
        );
    }

    // Renormalize to AREA-weighted land mean 1.0: on the adaptive mesh the many
    // tiny land cells must not dominate the mean, and hydrology now integrates
    // precip × area, so the budget it calibrates against is the area-weighted one.
    let areas = tess.cell_areas();
    let (mut wsum, mut asum) = (0.0f64, 0.0f64);
    for i in 0..n {
        if elevation[i] >= 0.0 {
            wsum += (precip[i] * areas[i]) as f64;
            asum += areas[i] as f64;
        }
    }
    if asum > 0.0 {
        let mean = (wsum / asum) as f32;
        if mean > 1e-6 {
            // Renormalize to the planet-wetness mean (not 1.0), preserving the
            // absolute level the coarse moisture set rather than stripping it.
            let k = PRECIP_GLOBAL_SCALE / mean;
            for p in &mut precip {
                *p *= k;
            }
        }
    }

    // Over-dry audit (rain-shadow.md): the floor is applied pre-renorm, so it is soft after
    // rescale. Log the honest tripwires — the renorm pins the MEAN, so a sliding MEDIAN /
    // climbing floor-hit fraction (not the mean) signals over-drying.
    if downwind_strength > 0.0 {
        log_downwind_shadow_audit(elevation, &precip, coarse_precip);
    }
    precip
}

/// Diagnostics for the downwind rain shadow: how dry the lee got after renormalization.
fn log_downwind_shadow_audit(elevation: &[f32], precip: &[f32], coarse_precip: &[f32]) {
    let mut ratios: Vec<f32> = (0..elevation.len())
        .filter(|&i| elevation[i] >= 0.0 && coarse_precip[i] > 1e-6)
        .map(|i| precip[i] / coarse_precip[i])
        .collect();
    if ratios.is_empty() {
        return;
    }
    ratios.sort_by(f32::total_cmp);
    let pct = |p: f32| ratios[((ratios.len() - 1) as f32 * p) as usize];
    let n = ratios.len() as f32;
    let floor_hits = ratios
        .iter()
        .filter(|&&r| r <= DOWNWIND_SHADOW_FLOOR + 1e-4)
        .count() as f32;
    let over_max = ratios
        .iter()
        .filter(|&&r| r > OROGRAPHIC_PRECIP_MAX)
        .count() as f32;
    log::info!(
        "downwind shadow audit: precip/coarse p10={:.2} median={:.2} p90={:.2} | floor-hit {:.1}% over-MAX {:.1}%",
        pct(0.10),
        pct(0.50),
        pct(0.90),
        100.0 * floor_hits / n,
        100.0 * over_max / n,
    );
}

/// Propagate each land cell's LOCAL lee dry-anomaly DOWNWIND along the per-cell wind in one
/// ordered O(N) DAG sweep (rain-shadow.md). Each land cell points to its single most-downwind
/// land neighbour (local wind, not a global axis — the circulation is latitude-banded with
/// zonal reversal). The resulting functional graph is topo-sorted (Kahn); cells left in a
/// cycle (anti-parallel wind eddies) receive from outside but do not propagate further (their
/// outgoing edge is effectively cut). NOT CFL advection — no timestep, so the resolution-
/// dependent over-dry trap cannot occur. Mean is restored by the caller's renormalization.
fn downwind_lee_shadow(
    tess: &Tessellation,
    elevation: &[f32],
    wind: &[Vec3],
    coarse_precip: &[f32],
    precip_initial: &[f32],
    strength: f32,
) -> Vec<f32> {
    let n = tess.num_cells();
    let fetch_chord = (DOWNWIND_SHADOW_FETCH_KM / PLANET_RADIUS_KM).max(1e-6);

    // Seed: each land cell's local lee dry-anomaly (the moisture the per-cell term suppressed).
    let mut accum = vec![0.0f32; n];
    for i in 0..n {
        if elevation[i] > 0.0 {
            accum[i] = (coarse_precip[i] - precip_initial[i]).max(0.0);
        }
    }

    // Per-cell single downwind receiver: the land neighbour best aligned with wind[i]
    // (cos > cone). Near-zero wind or no qualifying neighbour → no receiver (anomaly rains
    // out locally). Out-degree ≤ 1 → a functional graph.
    let mut receiver = vec![usize::MAX; n];
    let mut indeg = vec![0u32; n];
    for i in 0..n {
        if elevation[i] <= 0.0 {
            continue;
        }
        let w = wind[i];
        if w.length_squared() < 1e-12 {
            continue;
        }
        let wn = w.normalize();
        let ci = tess.cell_center(i);
        let mut best_cos = DOWNWIND_CONE_COS;
        let mut best_j = usize::MAX;
        for &j in tess.neighbors(i) {
            if elevation[j] <= 0.0 {
                continue; // receivers must be land
            }
            let dir = tess.cell_center(j) - ci;
            let d2 = dir.length_squared();
            if d2 < 1e-20 {
                continue;
            }
            let cos = wn.dot(dir) / d2.sqrt();
            if cos > best_cos {
                best_cos = cos;
                best_j = j;
            }
        }
        if best_j != usize::MAX {
            receiver[i] = best_j;
            indeg[best_j] += 1;
        }
    }

    // Topological order (Kahn, deterministic FIFO). Only carriers (receiver != MAX) enter the
    // order. Cells whose indeg never reaches 0 are in a cycle → omitted (their outgoing edge
    // is cut); they still RECEIVE from acyclic upwind cells via the sweep below.
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut queue: std::collections::VecDeque<usize> = (0..n)
        .filter(|&i| indeg[i] == 0 && receiver[i] != usize::MAX)
        .collect();
    while let Some(i) = queue.pop_front() {
        order.push(i);
        let r = receiver[i];
        indeg[r] -= 1;
        if indeg[r] == 0 && receiver[r] != usize::MAX {
            queue.push_back(r);
        }
    }

    // One ordered sweep: each cell's anomaly is final (all upwind contributors are earlier in
    // `order`) before it carries a fetch-decayed fraction to its receiver. The remainder is
    // dropped (rained out — cannot amplify).
    let mut incoming = vec![0.0f32; n];
    for &i in &order {
        let r = receiver[i];
        let d = (tess.cell_center(r) - tess.cell_center(i)).length();
        let carried = accum[i] * (-d / fetch_chord).exp();
        accum[r] += carried;
        incoming[r] += carried;
    }

    // Apply the arriving downwind anomaly under a floor (≤ coarse precip · FLOOR).
    let mut precip = precip_initial.to_vec();
    for i in 0..n {
        if elevation[i] > 0.0 {
            let dried = precip_initial[i] - strength * incoming[i];
            precip[i] = dried.max(DOWNWIND_SHADOW_FLOOR * coarse_precip[i]);
        }
    }
    precip
}

/// Tangent-plane elevation gradient using CHORD neighbour distance (f32-robust
/// at the fine mesh's km scale, unlike acos(dot)). Magnitude ~ Δelev / Δarc.
fn chord_gradient(tess: &Tessellation, values: &[f32], i: usize) -> Vec3 {
    let cell_elev = values[i];
    let cell_pos = tess.cell_center(i);
    let mut gradient = Vec3::ZERO;
    for &nb in tess.neighbors(i) {
        let nb_pos = tess.cell_center(nb);
        let to_nb = nb_pos - cell_pos;
        let tangent = to_nb - cell_pos * cell_pos.dot(to_nb);
        let tlen = tangent.length();
        let dist = to_nb.length();
        if tlen < 1e-6 || dist < 1e-9 {
            continue;
        }
        gradient += tangent.normalize() * ((values[nb] - cell_elev) / dist);
    }
    gradient
}

/// Lakes-as-evaporation: boost precipitation in a halo around lake cells (standing
/// water adds local humidity). Lake presence is diffused into the surrounding
/// land, precip multiplied by (1 + strength * halo), then renormalized to land
/// mean 1.0. Local only — the fine mesh doesn't re-advect moisture, so this is a
/// halo, not a downwind plume.
fn boost_precip_near_lakes(
    tess: &Tessellation,
    elevation: &[f32],
    precip: &[f32],
    hydrology: &Hydrology,
    strength: f32,
) -> Vec<f32> {
    let n = tess.num_cells();
    let is_lake: Vec<bool> = (0..n)
        .map(|i| hydrology.water_state(i) == CellWaterState::LakeWater)
        .collect();

    // Diffuse lake presence into a halo (sources pinned at 1.0).
    let mut hum: Vec<f32> = is_lake.iter().map(|&l| if l { 1.0 } else { 0.0 }).collect();
    for _ in 0..LAKE_EVAP_DIFFUSE_STEPS {
        let prev = hum.clone();
        for i in 0..n {
            if is_lake[i] {
                continue;
            }
            let nbs = tess.neighbors(i);
            if nbs.is_empty() {
                continue;
            }
            let mean: f32 = nbs.iter().map(|&j| prev[j]).sum::<f32>() / nbs.len() as f32;
            hum[i] = 0.5 * prev[i] + 0.5 * mean;
        }
    }

    let mut out: Vec<f32> = (0..n)
        .map(|i| precip[i] * (1.0 + strength * hum[i]))
        .collect();
    // Area-weighted land-mean-1.0 renormalization (see fine_precipitation).
    let areas = tess.cell_areas();
    let (mut wsum, mut asum) = (0.0f64, 0.0f64);
    for i in 0..n {
        if elevation[i] >= 0.0 {
            wsum += (out[i] * areas[i]) as f64;
            asum += areas[i] as f64;
        }
    }
    if asum > 0.0 {
        let mean = (wsum / asum) as f32;
        if mean > 1e-6 {
            // Preserve the planet-wetness mean (see fine_precipitation).
            let k = PRECIP_GLOBAL_SCALE / mean;
            for p in &mut out {
                *p *= k;
            }
        }
    }
    out
}

fn build_coarse_tree(tessellation: &Tessellation) -> CoarseTree {
    let entries: Vec<[f32; 3]> = (0..tessellation.num_cells())
        .map(|i| {
            let p = tessellation.cell_center(i);
            [p.x, p.y, p.z]
        })
        .collect();
    ImmutableKdTree::new_from_slice(&entries)
}

fn nearest_coarse(tree: &CoarseTree, pos: Vec3) -> usize {
    tree.nearest_one::<SquaredEuclidean>(&[pos.x, pos.y, pos.z])
        .item as usize
}

fn map_to_coarse(fine: &Tessellation, tree: &CoarseTree) -> Vec<usize> {
    #[cfg(not(feature = "single-threaded"))]
    {
        (0..fine.num_cells())
            .into_par_iter()
            .map(|i| nearest_coarse(tree, fine.cell_center(i)))
            .collect()
    }

    #[cfg(feature = "single-threaded")]
    (0..fine.num_cells())
        .map(|i| nearest_coarse(tree, fine.cell_center(i)))
        .collect()
}

/// Convert a target cell size (km) to an areal cell density (cells per steradian
/// on the unit sphere): g = 1 / size_in_radians^2.
fn cell_size_km_to_density(km: f32) -> f32 {
    let rad = (km / PLANET_RADIUS_KM).max(1e-9);
    1.0 / (rad * rad)
}

/// Absolute areal cell density (cells/steradian) per coarse cell, derived from
/// the physical cell-size scales. Ocean is set directly; land interpolates
/// between the plains and mountain densities by a normalized refinement demand
/// (slope/flow/activity). The total cell count emerges from integrating this.
fn compute_areal_density(
    tessellation: &Tessellation,
    elevation: &Elevation,
    features: &FeatureFields,
    preview_hydrology: &Hydrology,
    params: &FineDensityParams,
) -> Vec<f32> {
    let n = tessellation.num_cells();
    let max_slope = (0..n)
        .map(|i| elevation.slope_elevation_per_radian(tessellation, i))
        .fold(0.0_f32, f32::max)
        .max(1e-6);
    // flow_count_equiv (not raw discharge) so the .max(1.0) log floor stays in
    // count units after hydrology became area-weighted.
    let max_flow_ln = (0..n)
        .map(|i| preview_hydrology.flow_count_equiv(i).max(1.0).ln())
        .fold(0.0_f32, f32::max)
        .max(1e-6);
    let max_strain = features
        .thin_sheet_strain
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        .max(1e-6);

    let g_plains = cell_size_km_to_density(params.plains_km);
    let g_mountain = cell_size_km_to_density(params.mountain_km);
    let g_ocean = cell_size_km_to_density(params.ocean_km);
    let e = params.exponent;
    let wsum = params.slope_weight + params.flow_weight + params.activity_weight;

    let mut density = Vec::with_capacity(n);
    for i in 0..n {
        if elevation.values[i] < 0.0 {
            density.push(g_ocean);
            continue;
        }
        // Each feature normalized to [0,1], raised to a concentration exponent
        // so gentle terrain stays near the plains size; combined into a single
        // demand in [0,1] (0 = flat plains, 1 = all features maxed). Weights are
        // relative importances; absolute scale comes from the cell-size scales.
        let slope = (elevation.slope_elevation_per_radian(tessellation, i) / max_slope).powf(e);
        let flow = (preview_hydrology.flow_count_equiv(i).max(1.0).ln() / max_flow_ln).powf(e);
        let strain = (features.thin_sheet_strain[i] / max_strain).clamp(0.0, 1.0);
        let activity = features.activity[i].clamp(0.0, 1.0).max(strain).powf(e);
        let demand = (params.slope_weight * slope
            + params.flow_weight * flow
            + params.activity_weight * activity)
            / wsum;
        density.push(g_plains + demand * (g_mountain - g_plains));
    }
    density
}

/// Blue-noise quality probe (a REFERENCE for judging relaxation passes, not a
/// target). Uses the canonical regularity metric from `sample_experiment.rs`:
///
/// ```text
/// rho = nearest-neighbour distance / sqrt(cell_area)
/// ```
///
/// ~1 and tight for blue noise; a tail toward 0 means slivers/clumps (a cell
/// whose nearest neighbour sits much closer than its size implies — the thing
/// relaxation exists to remove). Distance is chord (f32-robust at km scale).
/// Reports the low-end distribution, coefficient of variation, and the sliver
/// fractions (rho < 0.4 and < 0.25) so reducing FINE_RELAX_PASSES is judged on
/// the metric relaxation actually moves, not just cell area.
fn mesh_quality_probe(tessellation: &Tessellation) {
    let areas = tessellation.cell_areas();
    let n = tessellation.num_cells();
    if n == 0 {
        return;
    }
    let rho_of = |i: usize| -> f32 {
        let area = areas[i].max(1e-20);
        let pos = tessellation.cell_center(i);
        let mut nn = f32::INFINITY;
        for &nb in tessellation.neighbors(i) {
            nn = nn.min((pos - tessellation.cell_center(nb)).length());
        }
        if nn.is_finite() {
            nn / area.sqrt()
        } else {
            f32::NAN
        }
    };
    #[cfg(not(feature = "single-threaded"))]
    let mut rho: Vec<f32> = (0..n)
        .into_par_iter()
        .map(rho_of)
        .filter(|r| r.is_finite())
        .collect();
    #[cfg(feature = "single-threaded")]
    let mut rho: Vec<f32> = (0..n).map(rho_of).filter(|r| r.is_finite()).collect();
    if rho.is_empty() {
        return;
    }
    let m = rho.len();
    let mean = rho.iter().sum::<f32>() / m as f32;
    let var = rho.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / m as f32;
    let cov = var.sqrt() / mean.max(1e-9);
    let s40 = rho.iter().filter(|&&r| r < 0.4).count();
    let s25 = rho.iter().filter(|&&r| r < 0.25).count();
    rho.sort_by(f32::total_cmp);
    let pct = |p: f32| rho[(((m - 1) as f32) * p) as usize];
    log::info!(
        "fine mesh quality: rho=nn/sqrt(area) min={:.2} p1={:.2} p5={:.2} p50={:.2} mean={:.2} CoV={:.2} | slivers <0.4 {:.3}% <0.25 {:.3}%",
        rho[0],
        pct(0.01),
        pct(0.05),
        pct(0.50),
        mean,
        cov,
        100.0 * s40 as f32 / m as f32,
        100.0 * s25 as f32 / m as f32,
    );
}

/// Report the physical resolution of the generated fine mesh, by terrain tier,
/// so we can judge whether mountains are resolved finely enough for erosion
/// (target: low single-digit km). Cell width ~ sqrt(cell_area) on the unit
/// sphere, scaled by PLANET_RADIUS_KM.
fn log_resolution_probe(tessellation: &Tessellation, elevation: &Elevation) {
    let areas = tessellation.cell_areas();
    let n = areas.len().min(elevation.values.len());
    if n == 0 {
        return;
    }
    let spacing_km = |i: usize| areas[i].max(0.0).sqrt() * PLANET_RADIUS_KM;

    // Land vs ocean by elevation; the finest land cells ARE the mountains and
    // river channels, so land-spacing percentiles report the resolution where
    // erosion happens without picking an arbitrary "mountain" threshold.
    let mut land: Vec<f32> = Vec::new();
    let mut ocean: Vec<f32> = Vec::new();
    for i in 0..n {
        let s = spacing_km(i);
        if elevation.values[i] < 0.0 {
            ocean.push(s);
        } else {
            land.push(s);
        }
    }
    land.sort_by(f32::total_cmp);
    ocean.sort_by(f32::total_cmp);
    let pct = |v: &[f32], p: f32| -> f32 {
        if v.is_empty() {
            return f32::NAN;
        }
        v[(((v.len() - 1) as f32) * p) as usize]
    };

    log::info!(
        "fine mesh resolution (km, planet R={:.0}): land [{} cells] p1(finest mtns)={:.1} p10={:.1} p50={:.1} p90(plains)={:.1} | ocean [{} cells] median={:.1} max={:.1}",
        PLANET_RADIUS_KM,
        land.len(),
        pct(&land, 0.01),
        pct(&land, 0.10),
        pct(&land, 0.50),
        pct(&land, 0.90),
        ocean.len(),
        pct(&ocean, 0.50),
        ocean.last().copied().unwrap_or(f32::NAN),
    );
}

/// Sample fine points by directly allocating each coarse cell's expected count
/// (areal density x cell area) and scattering that many points within the cell,
/// then relaxing to blue noise. There is no global target/normalization: the
/// total count is the sum of per-cell counts (it emerges). `density` is already
/// scaled to honour the cap. O(N) — no rejection over-generation.
fn sample_fine_points<R: Rng>(
    coarse: &Tessellation,
    density: &[f32],
    tree: &CoarseTree,
    rng: &mut R,
) -> Vec<Vec3> {
    let areas = coarse.cell_areas();
    let n = coarse.num_cells();
    let seed = rng.gen::<u64>();

    // Golden angle for the sunflower/Fibonacci disk pattern.
    const GOLDEN_ANGLE: f32 = 2.399_963_2;
    // Disk radius factor: candidates are clipped to the Voronoi cell, so the disk
    // must over-cover it (corners) -- area f^2, hence f^2x candidates generated.
    const DISK_OVERFILL: f32 = 1.3;
    let place = |c: usize| -> Vec<Vec3> {
        let expected = density[c] * areas[c];
        // Stochastic rounding so the total count is unbiased, not floored.
        let extra = (hash_unit_f32(seed, c as u64, 1) < expected.fract()) as u64;
        let count = expected.floor() as u64 + extra;
        if count == 0 {
            return Vec::new();
        }
        let center = coarse.cell_center(c);
        let radius = (areas[c] / std::f32::consts::PI).sqrt() * DISK_OVERFILL;
        let (u, v) = tangent_basis(center);
        // Lay a Fibonacci sunflower over the oversized disk, then KEEP ONLY the
        // candidates that fall in this Voronoi cell (nearest coarse == c). This
        // makes each cell fill its own polygon so the cells TILE -- no disk-shaped
        // density patches/rings, no corner gaps, no cross-boundary bleed. The
        // sunflower gives a built-in min-distance (few slivers); small jitter
        // breaks the spiral; relaxation finishes spacing and cell seams.
        let n_cand = ((count as f32) * DISK_OVERFILL * DISK_OVERFILL).ceil() as u64;
        let jitter = radius / (n_cand as f32).sqrt() * 0.5;
        (0..n_cand)
            .filter_map(|k| {
                let rr = radius * ((k as f32 + 0.5) / n_cand as f32).sqrt();
                let theta = k as f32 * GOLDEN_ANGLE;
                let jr = jitter * hash_unit_f32(seed, c as u64, 2 * k + 2).sqrt();
                let ja = hash_unit_f32(seed, c as u64, 2 * k + 3) * std::f32::consts::TAU;
                let px = rr * theta.cos() + jr * ja.cos();
                let py = rr * theta.sin() + jr * ja.sin();
                let p = (center + u * px + v * py).normalize();
                (nearest_coarse(tree, p) == c).then_some(p)
            })
            .collect()
    };

    #[cfg(not(feature = "single-threaded"))]
    let points: Vec<Vec3> = (0..n).into_par_iter().flat_map_iter(place).collect();
    #[cfg(feature = "single-threaded")]
    let points: Vec<Vec3> = (0..n).flat_map(place).collect();

    relax_fine_points(points, density, tree)
}

const RELAX_K: usize = 8;

/// Two orthonormal tangent vectors at a point on the unit sphere.
fn tangent_basis(p: Vec3) -> (Vec3, Vec3) {
    let arbitrary = if p.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = p.cross(arbitrary).normalize();
    let v = p.cross(u);
    (u, v)
}

/// One density-aware particle-repulsion step for point `i`: push off neighbours
/// closer than its target spacing, projected onto the sphere's tangent plane.
fn relax_step(
    i: usize,
    p: Vec3,
    tree: &KdTree<f32, 3>,
    entries: &[[f32; 3]],
    points: &[Vec3],
    point_spacing: &[f32],
) -> Vec3 {
    let sep = point_spacing[i];
    let mut push = Vec3::ZERO;
    // Bounded nearest-K (not radius-within): keeps per-point work constant even
    // where the thinned input clumps, which is what blows up an unbounded radius
    // query on white-noise points.
    for nb in tree.nearest_n::<SquaredEuclidean>(&entries[i], RELAX_K + 1) {
        let j = nb.item as usize;
        if j == i {
            continue;
        }
        let d = nb.distance.sqrt();
        if d > 1e-7 && d < sep {
            push += (p - points[j]) / d * (sep - d) / sep;
        }
    }
    if push == Vec3::ZERO {
        return p;
    }
    let tangent = push - p * push.dot(p);
    (p + tangent * (0.4 * sep)).normalize()
}

/// Turn the white-noise (sliver-prone) thinned points into adaptive blue noise
/// via `FINE_RELAX_PASSES` Jacobi repulsion passes. Each pass rebuilds a mutable
/// kd-tree (a 0.4*sep move per pass makes the neighbour set too stale to reuse)
/// and moves every point off its too-close neighbours. Point count is preserved.
///
/// PERF (parked, revisit with the s2-voronoi rework): the kd-tree's serial
/// build is the portable bottleneck (~1s/pass; worse-relative on many cores
/// since the parallel query scales but the build doesn't). The clean fix is NOT
/// a hex3-side grid (a single flat grid loses to the 1153:1 density variation,
/// a hierarchy is overkill) nor reusing s2's full Voronoi per pass (clipping +
/// 16-24 candidate margin is a sledgehammer to read off ~6 neighbours). What we
/// actually need is a BARE k~8 nearest-neighbour query over s2's cube grid, no
/// clipping — that primitive exists inside s2's construction but isn't public
/// (`SphereLocator` is nearest-1, post-build, so it doesn't fit). If the s2
/// rework exposes `knn(&points, k)` over the cube grid, relaxation can call it:
/// ~0.2-0.4s/pass, density-correct for free, no second spatial index in hex3.
/// See the s2-voronoi-performance note. ImmutableKdTree was tried and regressed
/// (bulk build cost x per-pass rebuild). Passes were cut 5->3 (sharp quality
/// knee; see FINE_RELAX_PASSES and `mesh_quality_probe`).
fn relax_fine_points(
    mut points: Vec<Vec3>,
    density: &[f32],
    coarse_tree: &CoarseTree,
) -> Vec<Vec3> {
    if FINE_RELAX_PASSES == 0 {
        return points;
    }
    let t0 = Instant::now();

    // Fix each point's target spacing from its initial coarse cell once (points
    // move only ~0.4*spacing per pass). Spacing = 1/sqrt(areal density): the
    // absolute cell size the density field asks for at that location.
    let spacing_at = |p: Vec3| {
        let g = density[nearest_coarse(coarse_tree, p)].max(1e-12);
        1.0 / g.sqrt()
    };
    let point_spacing: Vec<f32> = {
        #[cfg(not(feature = "single-threaded"))]
        {
            points.par_iter().map(|&p| spacing_at(p)).collect()
        }
        #[cfg(feature = "single-threaded")]
        {
            points.iter().map(|&p| spacing_at(p)).collect()
        }
    };

    let (mut t_build, mut t_query) = (0.0f64, 0.0f64);
    for _ in 0..FINE_RELAX_PASSES {
        let s = Instant::now();
        let entries: Vec<[f32; 3]> = points.iter().map(|p| p.to_array()).collect();
        let mut tree = KdTree::<f32, 3>::with_capacity(entries.len());
        for (i, e) in entries.iter().enumerate() {
            tree.add(e, i as u64);
        }
        t_build += s.elapsed().as_secs_f64();
        let s = Instant::now();
        let new_points: Vec<Vec3> = {
            #[cfg(not(feature = "single-threaded"))]
            {
                points
                    .par_iter()
                    .enumerate()
                    .map(|(i, &p)| relax_step(i, p, &tree, &entries, &points, &point_spacing))
                    .collect()
            }
            #[cfg(feature = "single-threaded")]
            {
                points
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| relax_step(i, p, &tree, &entries, &points, &point_spacing))
                    .collect()
            }
        };
        points = new_points;
        t_query += s.elapsed().as_secs_f64();
    }

    log::info!(
        "fine mesh: relaxation {} passes {:.2?} (build {:.1}s, query+move {:.1}s)",
        FINE_RELAX_PASSES,
        t0.elapsed(),
        t_build,
        t_query,
    );
    points
}

fn hash_unit_f32(seed: u64, index: u64, stream: u64) -> f32 {
    let value = splitmix64(seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ stream);
    ((value >> 40) as f32) * (1.0 / (1u32 << 24) as f32)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Interpolate the coarse final elevation onto the fine cells, using the same
/// nearest-coarse + neighbours inverse-distance support as the field transfer.
/// The result is on the coarse sea-level datum (coarse values are already
/// shifted so 0 = sea level), so no re-solve is needed downstream.
fn interpolate_coarse_elevation(
    coarse: &Tessellation,
    fine: &Tessellation,
    coarse_cell: &[usize],
    coarse_elevation: &[f32],
) -> Vec<f32> {
    let interp = |i: usize| {
        let pos = fine.cell_center(i);
        let nearest = coarse_cell[i];
        let mut support = InterpolationSupport::new();
        support.push(
            nearest,
            interpolation_weight(coarse.cell_center(nearest), pos),
        );
        for &nb in coarse.neighbors(nearest) {
            support.push(nb, interpolation_weight(coarse.cell_center(nb), pos));
        }
        support.interpolate(coarse_elevation, 0.0)
    };

    #[cfg(not(feature = "single-threaded"))]
    {
        (0..fine.num_cells()).into_par_iter().map(interp).collect()
    }
    #[cfg(feature = "single-threaded")]
    {
        (0..fine.num_cells()).map(interp).collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn transfer_fields(
    coarse: &Tessellation,
    fine: &Tessellation,
    coarse_cell: &[usize],
    crust: &Crust,
    features: &FeatureFields,
    orogen_model: OrogenModel,
    coarse_elevation: &Elevation,
    atmosphere: &Atmosphere,
) -> FineFields {
    let coarse_fields = coarse_elevation_fields(coarse, crust, features, orogen_model);
    let n = fine.num_cells();

    #[cfg(not(feature = "single-threaded"))]
    let transferred: Vec<TransferredCell> = (0..n)
        .into_par_iter()
        .map(|i| {
            transfer_cell(
                coarse,
                fine.cell_center(i),
                coarse_cell[i],
                &coarse_fields,
                atmosphere,
            )
        })
        .collect();

    #[cfg(feature = "single-threaded")]
    let transferred: Vec<TransferredCell> = (0..n)
        .map(|i| {
            transfer_cell(
                coarse,
                fine.cell_center(i),
                coarse_cell[i],
                &coarse_fields,
                atmosphere,
            )
        })
        .collect();

    let _ = coarse_elevation;
    let mut elevation_fields = ElevationFields {
        crust_thickness: Vec::with_capacity(n),
        tectonic_thickening: Vec::with_capacity(n),
        legacy_uplift_source: coarse_fields.legacy_uplift_source,
        tectonic_strain: Vec::with_capacity(n),
        compression_axis: Vec::with_capacity(n),
        tectonic_uplift_rate: Vec::with_capacity(n),
        allow_source_craton_macro: coarse_fields.allow_source_craton_macro,
        continentality: Vec::with_capacity(n),
        ridge_age_distance: Vec::with_capacity(n),
        trench: Vec::with_capacity(n),
        ridge: Vec::with_capacity(n),
        convergent: Vec::with_capacity(n),
        divergent: Vec::with_capacity(n),
        is_continental: Vec::with_capacity(n),
        arc: Vec::with_capacity(n),
        collision: Vec::with_capacity(n),
        rift_delta: Vec::with_capacity(n),
    };
    let mut temperature = Vec::with_capacity(n);
    let mut precipitation = Vec::with_capacity(n);
    let mut uplift = Vec::with_capacity(n);
    let mut wind = Vec::with_capacity(n);

    for cell in transferred {
        elevation_fields.crust_thickness.push(cell.crust_thickness);
        elevation_fields
            .tectonic_thickening
            .push(cell.tectonic_thickening);
        elevation_fields.tectonic_strain.push(cell.tectonic_strain);
        elevation_fields
            .compression_axis
            .push(cell.compression_axis);
        elevation_fields
            .tectonic_uplift_rate
            .push(cell.tectonic_uplift_rate);
        elevation_fields.continentality.push(cell.continentality);
        elevation_fields
            .ridge_age_distance
            .push(cell.ridge_age_distance);
        elevation_fields.trench.push(cell.trench);
        elevation_fields.ridge.push(cell.ridge);
        elevation_fields.convergent.push(cell.convergent);
        elevation_fields.divergent.push(cell.divergent);
        elevation_fields
            .is_continental
            .push(cell.continentality >= 0.5);
        elevation_fields.arc.push(cell.arc);
        elevation_fields.collision.push(cell.collision);
        elevation_fields.rift_delta.push(cell.rift_delta);
        temperature.push(cell.temperature);
        precipitation.push(cell.precipitation);
        uplift.push(cell.uplift);
        wind.push(cell.wind);
    }

    FineFields {
        elevation_fields,
        temperature,
        precipitation,
        uplift,
        wind,
    }
}

struct TransferredCell {
    crust_thickness: f32,
    tectonic_thickening: f32,
    tectonic_strain: f32,
    compression_axis: Vec3,
    tectonic_uplift_rate: f32,
    continentality: f32,
    ridge_age_distance: f32,
    trench: f32,
    ridge: f32,
    convergent: f32,
    divergent: f32,
    arc: f32,
    collision: f32,
    rift_delta: f32,
    temperature: f32,
    precipitation: f32,
    uplift: f32,
    wind: Vec3,
}

fn transfer_cell(
    coarse: &Tessellation,
    pos: Vec3,
    nearest: usize,
    coarse_fields: &ElevationFields,
    atmosphere: &Atmosphere,
) -> TransferredCell {
    let mut support = InterpolationSupport::new();
    support.push(
        nearest,
        interpolation_weight(coarse.cell_center(nearest), pos),
    );
    for &nb in coarse.neighbors(nearest) {
        support.push(nb, interpolation_weight(coarse.cell_center(nb), pos));
    }

    let continentality = support.interpolate(&coarse_fields.continentality, 0.0);
    TransferredCell {
        crust_thickness: support.interpolate(&coarse_fields.crust_thickness, 0.0),
        tectonic_thickening: support.interpolate(&coarse_fields.tectonic_thickening, 0.0),
        tectonic_strain: support.interpolate(&coarse_fields.tectonic_strain, 0.0),
        compression_axis: support
            .interpolate_vec3(&coarse_fields.compression_axis)
            .normalize_or_zero(),
        tectonic_uplift_rate: support.interpolate(&coarse_fields.tectonic_uplift_rate, 0.0),
        continentality,
        ridge_age_distance: support.interpolate(&coarse_fields.ridge_age_distance, f32::INFINITY),
        trench: support.interpolate(&coarse_fields.trench, 0.0),
        ridge: support.interpolate(&coarse_fields.ridge, 0.0),
        convergent: support.interpolate(&coarse_fields.convergent, 0.0),
        divergent: support.interpolate(&coarse_fields.divergent, 0.0),
        arc: support.interpolate(&coarse_fields.arc, 0.0),
        collision: support.interpolate(&coarse_fields.collision, 0.0),
        rift_delta: support.interpolate(&coarse_fields.rift_delta, 0.0),
        temperature: support.interpolate(&atmosphere.temperature, 0.0),
        precipitation: support.interpolate(&atmosphere.precipitation, 0.0).max(0.0),
        uplift: support.interpolate(&atmosphere.uplift, 0.0),
        wind: support.interpolate_vec3(&atmosphere.wind),
    }
}

struct InterpolationSupport {
    len: usize,
    entries: [(usize, f32); 16],
    overflow: Vec<(usize, f32)>,
}

impl InterpolationSupport {
    fn new() -> Self {
        Self {
            len: 0,
            entries: [(0, 0.0); 16],
            overflow: Vec::new(),
        }
    }

    fn push(&mut self, idx: usize, weight: f32) {
        if self.len < self.entries.len() {
            self.entries[self.len] = (idx, weight);
            self.len += 1;
        } else {
            self.overflow.push((idx, weight));
        }
    }

    fn interpolate(&self, field: &[f32], fallback: f32) -> f32 {
        let mut weighted = 0.0;
        let mut total = 0.0;
        for &(idx, weight) in &self.entries[..self.len] {
            let value = field[idx];
            if value.is_finite() {
                weighted += value * weight;
                total += weight;
            }
        }
        for &(idx, weight) in &self.overflow {
            let value = field[idx];
            if value.is_finite() {
                weighted += value * weight;
                total += weight;
            }
        }
        if total > 0.0 {
            weighted / total
        } else {
            fallback
        }
    }

    /// Inverse-distance interpolate a Vec3 field (no renormalization — wind is a
    /// velocity, not a direction). ZERO fallback if no finite support.
    fn interpolate_vec3(&self, field: &[Vec3]) -> Vec3 {
        let mut weighted = Vec3::ZERO;
        let mut total = 0.0f32;
        for &(idx, weight) in &self.entries[..self.len] {
            let v = field[idx];
            if v.is_finite() {
                weighted += v * weight;
                total += weight;
            }
        }
        for &(idx, weight) in &self.overflow {
            let v = field[idx];
            if v.is_finite() {
                weighted += v * weight;
                total += weight;
            }
        }
        if total > 0.0 {
            weighted / total
        } else {
            Vec3::ZERO
        }
    }
}

fn interpolation_weight(coarse_pos: Vec3, fine_pos: Vec3) -> f32 {
    let dist = coarse_pos.dot(fine_pos).clamp(-1.0, 1.0).acos();
    1.0 / (dist * dist + 1e-8)
}
