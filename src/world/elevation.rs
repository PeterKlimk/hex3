//! Terrain elevation generation: isostasy over a crust thickness field.
//!
//! Decomposition (each term a distinct physical reason ground sits where
//! it does):
//!
//!   elevation = isostatic(thickness) + thermal(ocean age)
//!             + dynamic(trench flexure/outer rise)
//!
//! Thickness = margin ramp (continental thick, oceanic thin) + macro-scale
//! craton-structure thickness (thick shield interiors tapering to margins,
//! with intracratonic basins) + model-selected tectonic thickening
//! (the product baseline or an explicitly selected conservation experiment)
//! - rift thinning.
//!
//! The Airy relation (linear in
//! thickness for uniform densities) converts thickness to base elevation,
//! so plateaus, rift subsidence, and margin profiles all follow from one
//! principle. Thermal subsidence stays separate (young ocean floor is high
//! because it is hot, not thick), and trenches stay separate (held out of
//! isostatic equilibrium by slab pull).
//!
//! Sea level is solved (uniform shift) so land fraction hits LAND_FRACTION.
//!
//! The only surface noise in the simulation is the macro thickness
//! perturbation (folded into `thickness` above, isostatically compensated).
//! Hills/ridge were retired and cosmetic micro texture was removed entirely;
//! erosion supplies real fine relief on the fine mesh.

use glam::Vec3;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use ordered_float::OrderedFloat;
use rand::Rng;
#[cfg(not(feature = "single-threaded"))]
use rayon::prelude::*;

use super::constants::*;
use super::crust::{Crust, CrustType};
use super::plates::PLATE_NOISE_OFFSET_RANGE;
use super::{EpisodeCrustWork, FeatureFields, MaterialEpisodeWork, Tessellation};

/// Coarse convergent-orogen model.
///
/// Kept runtime-selectable so physical-model experiments can be evaluated on
/// identical seeds without silently replacing the product terrain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum OrogenModel {
    /// Historical additive arc/collision response fields. Product baseline.
    #[default]
    Legacy,
    /// Legacy source + gravitational yield relaxation: tectonic thickening
    /// above `OROGEN_YIELD_ELEV` spreads (conserved) over
    /// `OROGEN_YIELD_SPREAD_KM`; sub-yield terrain is bit-untouched. Targets
    /// the microplate mesa/pillar failure without the detail loss of the
    /// conserved experiments.
    LegacyYield,
    /// Conserved boundary volume left in its receiving boundary cells. Isolates
    /// source semantics from all lateral relaxation.
    ConservedLocal,
    /// Conserved total volume distributed through the historical arc/collision
    /// footprint. Causal bridge: old geometry, new mass budget.
    ConservedFeatureFootprint,
    /// Conserved boundary crust flux redistributed by isotropic thin-sheet
    /// diffusion. Retained as an explicit experiment, not the default.
    ConservedIsotropic,
    /// Physical-rate crust work integrated over connected boundary-episode duration,
    /// left in receiving boundary cells. First geological-time experiment.
    HistoryLocal,
    /// Episode-integrated crust work redistributed by dimensioned lower-crust mobility.
    HistoryDiffusive,
    /// Episode work first occupies a volume-derived finite-strain material
    /// footprint, then undergoes dimensioned lower-crust diffusion.
    HistoryMaterial,
    /// Physical-rate thin-sheet continuity integrated over boundary-episode
    /// intervals. No scalar closing-volume source and no yield-height tuning.
    HistoryThinSheet,
    /// History thin sheet driven by a moving low-resolution material carrier.
    HistoryCarrierThinSheet,
    /// Moving-boundary carrier replay with deformation accumulated on material
    /// parcels and projected back to the present terrain mesh.
    HistoryCarrierEvolved,
    /// Forward plate/crust lifecycle automaton on the fixed tectonic carrier.
    HistoryCarrierLifecycle,
    /// Velocity/continuity thin-sheet prototype (T0).
    ThinSheet,
}

impl std::fmt::Display for OrogenModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Legacy => write!(f, "legacy"),
            Self::LegacyYield => write!(f, "legacy-yield"),
            Self::ConservedLocal => write!(f, "conserved-local"),
            Self::ConservedFeatureFootprint => write!(f, "conserved-feature-footprint"),
            Self::ConservedIsotropic => write!(f, "conserved-isotropic"),
            Self::HistoryLocal => write!(f, "history-local"),
            Self::HistoryDiffusive => write!(f, "history-diffusive"),
            Self::HistoryMaterial => write!(f, "history-material"),
            Self::HistoryThinSheet => write!(f, "history-thin-sheet"),
            Self::HistoryCarrierThinSheet => write!(f, "history-carrier-thin-sheet"),
            Self::HistoryCarrierEvolved => write!(f, "history-carrier-evolved"),
            Self::HistoryCarrierLifecycle => write!(f, "history-carrier-lifecycle"),
            Self::ThinSheet => write!(f, "thin-sheet"),
        }
    }
}

/// Terrain elevation data.
pub struct Elevation {
    /// Elevation at each cell.
    pub values: Vec<f32>,

    /// Simulation noise contribution at each cell (macro only; hills/ridge retired).
    pub noise_contribution: Vec<f32>,

    /// Individual noise layer contributions (for visualization).
    pub noise_layers: NoiseLayerData,
}

/// Individual noise layer contributions for visualization (macro only;
/// hills/ridge retired, micro removed — see
/// docs/specs/erosion-v2.md "Noise philosophy").
pub struct NoiseLayerData {
    /// Macro layer (continental tilt).
    pub macro_layer: Vec<f32>,
}

/// Structural inputs to elevation assembly.
///
/// These are deliberately separated from mesh-native `Crust`/`FeatureFields`
/// so the fine mesh can rebuild elevation from transferred physical fields
/// without fabricating coarse-only domain objects.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ElevationFields {
    pub crust_thickness: Vec<f32>,
    /// Model-selected tectonic excess thickness. In the legacy product model
    /// this reproduces the historical `arc + collision` elevation response;
    /// experimental models derive it from conserved crust evolution.
    pub tectonic_thickening: Vec<f32>,
    /// Whether fine erosion may interpret tectonic state as the legacy per-step
    /// uplift source. False for history models until both clocks are physical.
    pub legacy_uplift_source: bool,
    /// Accumulated deformational strain; zero for models without a strain solve.
    pub tectonic_strain: Vec<f32>,
    /// Tangent principal-compression axis; zero where unresolved.
    pub compression_axis: Vec<Vec3>,
    /// Present physical tectonic thickness tendency (thickness units/Myr).
    pub tectonic_uplift_rate: Vec<f32>,
    /// Whether source-layout craton noise may enter assembly. Lifecycle worlds
    /// disable it because their generated layout is the oldest, not final, crust.
    pub allow_source_craton_macro: bool,
    pub continentality: Vec<f32>,
    pub ridge_age_distance: Vec<f32>,
    pub trench: Vec<f32>,
    pub ridge: Vec<f32>,
    pub convergent: Vec<f32>,
    pub divergent: Vec<f32>,
    pub is_continental: Vec<bool>,
    /// Volcanic-arc response magnitude. It prescribes height in the legacy
    /// product model and remains a spatial/process diagnostic in experiments.
    pub arc: Vec<f32>,
    /// Continental-collision response magnitude. See `arc`.
    pub collision: Vec<f32>,
    /// Signed continental-rift thickness delta (thickness units, not elevation):
    /// negative in the axial valley, positive on the shoulders.
    pub rift_delta: Vec<f32>,
}

/// Per-cell macro crust-thickness perturbation (thickness units), derived from
/// craton structure rather than free position noise. Each continental craton is
/// a thick interior that tapers to its margins (an isostatic dome → elevated
/// shield), with a per-craton base amplitude (older/thicker shields vs thinner
/// cratons) and a decorrelated interior fBm for intracratonic basins and swells.
/// Oceanic cells get no perturbation — their relief is thermal + ridge driven.
/// Added to base thickness during assembly; Airy isostasy turns it into relief.
fn macro_craton_thickness<R: Rng>(
    tessellation: &Tessellation,
    crust: &Crust,
    rng: &mut R,
) -> Vec<f32> {
    let num_cells = tessellation.num_cells();
    let num_cratons = crust.num_cratons.max(1);

    // Per-craton base amplitude and a decorrelated noise-domain offset, so each
    // craton's interior relief is independent (mirrors craton/plate growth).
    let craton_amp: Vec<f32> = (0..num_cratons)
        .map(|_| rng.gen_range(MACRO_CRATON_AMP_MIN..=MACRO_CRATON_AMP_MAX))
        .collect();
    let craton_offset: Vec<Vec3> = (0..num_cratons)
        .map(|_| {
            Vec3::new(
                rng.gen_range(-PLATE_NOISE_OFFSET_RANGE..PLATE_NOISE_OFFSET_RANGE),
                rng.gen_range(-PLATE_NOISE_OFFSET_RANGE..PLATE_NOISE_OFFSET_RANGE),
                rng.gen_range(-PLATE_NOISE_OFFSET_RANGE..PLATE_NOISE_OFFSET_RANGE),
            )
        })
        .collect();
    let interior_fbm = Fbm::<Perlin>::new(rng.gen()).set_octaves(MACRO_INTERIOR_OCTAVES);

    let sample = |i: usize| -> f32 {
        let craton = crust.cell_craton[i];
        if craton == u32::MAX {
            return 0.0; // oceanic: no cratonic structure
        }
        let craton = craton as usize;
        // Distance into the continent (≥0 inland); the dome saturates over
        // MACRO_CRATON_DECAY so margins stay thin and interiors reach full amp.
        let d = crust.signed_margin_distance[i].max(0.0);
        let dome = 1.0 - (-d / MACRO_CRATON_DECAY).exp();
        let p =
            tessellation.cell_center(i) * MACRO_INTERIOR_FREQUENCY as f32 + craton_offset[craton];
        let interior = interior_fbm.get([p.x as f64, p.y as f64, p.z as f64]) as f32;
        let amp = craton_amp[craton] + MACRO_INTERIOR_RELIEF * interior;
        MACRO_THICKNESS_AMPLITUDE * dome * amp
    };

    #[cfg(not(feature = "single-threaded"))]
    {
        (0..num_cells).into_par_iter().map(sample).collect()
    }
    #[cfg(feature = "single-threaded")]
    {
        (0..num_cells).map(sample).collect()
    }
}

impl Elevation {
    /// Generate elevation from tectonic features and crust.
    pub fn generate<R: Rng>(
        tessellation: &Tessellation,
        crust: &Crust,
        features: &FeatureFields,
        orogen_model: OrogenModel,
        rng: &mut R,
    ) -> Self {
        let macro_field = macro_craton_thickness(tessellation, crust, rng);

        let (values, noise_contribution, noise_layers) =
            generate_heightmap(tessellation, crust, features, orogen_model, &macro_field);

        Self {
            values,
            noise_contribution,
            noise_layers,
        }
    }

    /// Build a fine-mesh elevation by refining an already-solved base elevation
    /// (interpolated from the coarse mesh, so it is on the fixed sea-level datum).
    /// There is NO sea-level re-solve: sea level is a global planet datum, chosen
    /// once on the coarse mesh, and inherited here. The simulation values are the
    /// smooth interpolated base (erosion carves the fine detail later); there is
    /// no cosmetic surface noise.
    pub(crate) fn refine_from_base(tessellation: &Tessellation, base_elevation: &[f32]) -> Self {
        let n = tessellation.num_cells();
        Self {
            values: base_elevation.to_vec(),
            noise_contribution: vec![0.0; n],
            noise_layers: NoiseLayerData {
                macro_layer: vec![0.0; n],
            },
        }
    }

    /// Get elevation at a cell.
    pub fn at(&self, cell_idx: usize) -> f32 {
        self.values[cell_idx]
    }

    /// Compute elevation gradient (uphill direction) at a cell.
    ///
    /// Returns a Vec3 tangent to the sphere surface pointing in the direction
    /// of steepest ascent. Magnitude roughly indicates slope steepness.
    /// Returns zero vector for flat areas or cells with no neighbors.
    pub fn gradient(&self, tessellation: &Tessellation, cell_idx: usize) -> glam::Vec3 {
        use glam::Vec3;

        let cell_elev = self.values[cell_idx];
        let cell_pos = tessellation.cell_center(cell_idx);
        let neighbors = tessellation.neighbors(cell_idx);

        if neighbors.is_empty() {
            return Vec3::ZERO;
        }

        // Accumulate gradient as weighted sum of directions to neighbors
        let mut gradient = Vec3::ZERO;

        for &n in neighbors {
            let neighbor_elev = self.values[n];
            let neighbor_pos = tessellation.cell_center(n);

            // Direction from cell to neighbor (on sphere surface)
            let to_neighbor = neighbor_pos - cell_pos;

            // Project onto tangent plane (remove radial component)
            let tangent_dir = to_neighbor - cell_pos * cell_pos.dot(to_neighbor);
            let tangent_len = tangent_dir.length();
            if tangent_len < 1e-6 {
                continue;
            }

            // Arc distance between cells
            let arc_dist = cell_pos.dot(neighbor_pos).clamp(-1.0, 1.0).acos();
            if arc_dist < 1e-6 {
                continue;
            }

            // Elevation difference (positive = neighbor is higher)
            let elev_diff = neighbor_elev - cell_elev;

            // Slope in this direction
            let slope = elev_diff / arc_dist;

            // Accumulate: direction weighted by slope
            // Positive slope = uphill toward neighbor, so add that direction
            gradient += tangent_dir.normalize() * slope;
        }

        gradient
    }

    /// Compute gradient magnitude (slope steepness) at a cell.
    pub fn slope(&self, tessellation: &Tessellation, cell_idx: usize) -> f32 {
        self.gradient(tessellation, cell_idx).length()
    }
}

/// Thermal elevation anomaly for oceanic crust (positive near ridges).
/// Young lithosphere is hot and buoyant; depth approaches the abyssal
/// reference as sqrt(age), with spreading-adjusted ridge age distance as
/// the age proxy (Parsons-Sclater). This is thermal buoyancy, deliberately
/// separate from crust thickness.
fn thermal_anomaly(ridge_age_distance: f32) -> f32 {
    if !ridge_age_distance.is_finite() {
        // No ridge on this plate: old basin of unknown age, mild residual
        // anomaly so these basins are not uniformly maximal-depth.
        return NO_RIDGE_DEPTH - ABYSSAL_DEPTH;
    }
    let thermal_factor = (ridge_age_distance / THERMAL_SUBSIDENCE_WIDTH)
        .sqrt()
        .min(1.0);
    (1.0 - thermal_factor) * (RIDGE_CREST_DEPTH - ABYSSAL_DEPTH)
}

/// Isostatic elevation from crust thickness (Airy, uniform densities).
/// Linear relation through the two anchor points defined in constants.
pub(crate) fn isostatic_elevation(thickness: f32) -> f32 {
    let slope = (CONTINENTAL_BASE - ABYSSAL_DEPTH)
        / (CRUST_THICKNESS_CONTINENTAL - CRUST_THICKNESS_OCEANIC);
    let offset = CONTINENTAL_BASE - slope * CRUST_THICKNESS_CONTINENTAL;
    slope * thickness + offset
}

/// Elevation change per unit crust thickness (the Airy slope), used to
/// express feature forcing magnitudes (calibrated in elevation units) as
/// thickness changes.
pub(crate) fn isostasy_slope() -> f32 {
    (CONTINENTAL_BASE - ABYSSAL_DEPTH) / (CRUST_THICKNESS_CONTINENTAL - CRUST_THICKNESS_OCEANIC)
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Continentality: the margin ramp parameter (0 = full oceanic crust,
/// 1 = full continental), from the signed margin distance. Ramp widths
/// narrow on active margins (near convergent boundaries).
fn continentality(signed_margin_distance: f32, convergent_influence: f32) -> f32 {
    let activity = convergent_influence.clamp(0.0, 1.0);
    let land_width = PASSIVE_SHELF_WIDTH + activity * (ACTIVE_SHELF_WIDTH - PASSIVE_SHELF_WIDTH);
    let ocean_width = PASSIVE_OCEANIC_TRANSITION_WIDTH
        + activity * (ACTIVE_OCEANIC_TRANSITION_WIDTH - PASSIVE_OCEANIC_TRANSITION_WIDTH);
    smoothstep((signed_margin_distance + ocean_width) / (ocean_width + land_width))
}

#[derive(Clone, Copy)]
struct RelaxationEdge {
    a: usize,
    b: usize,
    conductance: f32,
}

/// Turn extensive convergent crust flux into a conservative thickness field.
///
/// The implicit finite-volume diffusion equation
///
/// ```text
/// (I - D t ∇²) H = H_source
/// ```
///
/// represents gravitational spreading over the same episode that accumulated
/// the shortening. No-flux boundaries follow crust-type boundaries. The solve
/// conserves `sum(area * H)`; its natural length is sqrt(D*t), so changing mesh
/// resolution does not change the physical footprint.
///
/// Local orogen belt width (radians): 2× the graph distance from each
/// footprint cell (thickening > threshold) to the nearest non-footprint cell —
/// the distance transform of the footprint. A 700-km-wide belt's crest reads
/// ~700 km; a 150-km salient reads ~150. Deterministic (index-tied Dijkstra);
/// non-footprint cells get 0 (their yield clamps to the minimum factor, which
/// never binds below the footprint threshold anyway).
fn orogen_width_field(tess: &Tessellation, thickening: &[f32], threshold: f32) -> Vec<f32> {
    let n = tess.num_cells();
    let in_footprint: Vec<bool> = thickening.iter().map(|&t| t > threshold).collect();
    let mut dist = vec![f32::INFINITY; n];
    let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(OrderedFloat<f32>, usize)>> =
        std::collections::BinaryHeap::new();
    for i in 0..n {
        if !in_footprint[i] {
            dist[i] = 0.0;
            heap.push(std::cmp::Reverse((OrderedFloat(0.0), i)));
        }
    }
    while let Some(std::cmp::Reverse((d, i))) = heap.pop() {
        if d.0 > dist[i] {
            continue;
        }
        let ci = tess.cell_center(i);
        for &nb in tess.neighbors(i) {
            let nd = d.0 + (tess.cell_center(nb) - ci).length();
            if nd < dist[nb] {
                dist[nb] = nd;
                heap.push(std::cmp::Reverse((OrderedFloat(nd), nb)));
            }
        }
    }
    (0..n)
        .map(|i| {
            if in_footprint[i] && dist[i].is_finite() {
                2.0 * dist[i]
            } else {
                0.0
            }
        })
        .collect()
}

fn solve_tectonic_thickening(
    tessellation: &Tessellation,
    crust: &Crust,
    volume_flux: &[f32],
    diffusivity: f32,
) -> Vec<f32> {
    let n = tessellation.num_cells();
    assert_eq!(volume_flux.len(), n);
    let areas = tessellation.cell_areas();
    let duration = TECTONIC_ACCUMULATION_TIME.max(0.0);
    let source: Vec<f32> = volume_flux
        .iter()
        .zip(areas.iter())
        .map(|(&flux, &area)| flux.max(0.0) * duration / area.max(1e-12))
        .collect();

    let tau = diffusivity.max(0.0) * duration;
    if tau <= 0.0 || !source.iter().any(|&x| x > 0.0) {
        return source;
    }

    let mut edges = Vec::with_capacity(tessellation.adjacency.total_neighbor_entries() / 2);
    for i in 0..n {
        for &j in tessellation.neighbors(i) {
            if j <= i || crust.crust_type(i) != crust.crust_type(j) {
                continue;
            }
            let center_distance =
                (tessellation.cell_center(i) - tessellation.cell_center(j)).length();
            if center_distance <= 1e-8 {
                continue;
            }
            let face_length = tessellation.shared_edge_length(i, j);
            if face_length > 0.0 {
                edges.push(RelaxationEdge {
                    a: i,
                    b: j,
                    conductance: face_length / center_distance,
                });
            }
        }
    }

    let relaxed = solve_screened_conservative(&areas, &edges, &source, tau);
    if log::log_enabled!(log::Level::Debug) {
        let source_max = source.iter().copied().fold(0.0f32, f32::max);
        let relaxed_max = relaxed.iter().copied().fold(0.0f32, f32::max);
        let volume: f64 = relaxed
            .iter()
            .zip(areas.iter())
            .map(|(&h, &a)| h as f64 * a as f64)
            .sum();
        log::debug!(
            "tectonic crust: volume={:.6}, local max={:.3}, relaxed max={:.3}, length={:.4} rad",
            volume,
            source_max,
            relaxed_max,
            tau.sqrt(),
        );
    }
    relaxed
}

fn solve_history_diffusive_thickening(
    tessellation: &Tessellation,
    crust: &Crust,
    episode_work: &[EpisodeCrustWork],
) -> Vec<f32> {
    solve_history_work_fields(
        tessellation,
        crust,
        episode_work
            .iter()
            .map(|episode| (episode.duration_myr, episode.cell_work.as_slice())),
    )
}

fn solve_history_material_thickening(
    tessellation: &Tessellation,
    crust: &Crust,
    episode_work: &[MaterialEpisodeWork],
) -> Vec<f32> {
    solve_history_work_fields(
        tessellation,
        crust,
        episode_work
            .iter()
            .map(|episode| (episode.duration_myr, episode.cell_work.as_slice())),
    )
}

fn solve_history_work_fields<'a>(
    tessellation: &Tessellation,
    crust: &Crust,
    episode_work: impl Iterator<Item = (f32, &'a [(usize, f32)])>,
) -> Vec<f32> {
    // Superpose the linear conservative solve episode by episode. This is more
    // expensive than diffusing the global sum once, but it is the correct causal
    // baseline: each contact relaxes over its own duration rather than an
    // area-weighted global mean age.
    let n = tessellation.num_cells();
    let mut total = vec![0.0f32; n];
    let shim_duration = TECTONIC_ACCUMULATION_TIME.max(1e-9);
    for (duration_myr, cell_work) in episode_work {
        let mut equivalent_flux = vec![0.0f32; n];
        for &(cell, work) in cell_work {
            equivalent_flux[cell] = work / shim_duration;
        }
        let tau_rad2 = CRUST_GRAVITATIONAL_DIFFUSIVITY_KM2_PER_MYR * duration_myr.max(0.0)
            / (PLANET_RADIUS_KM * PLANET_RADIUS_KM);
        let relaxed = solve_tectonic_thickening(
            tessellation,
            crust,
            &equivalent_flux,
            tau_rad2 / shim_duration,
        );
        for (sum, value) in total.iter_mut().zip(relaxed) {
            *sum += value;
        }
    }
    total
}

/// Conjugate-gradient solve for `(A + tau*L) x = A*source`, where A is cell
/// area and L is the symmetric finite-volume graph Laplacian.
fn solve_screened_conservative(
    areas: &[f32],
    edges: &[RelaxationEdge],
    source: &[f32],
    tau: f32,
) -> Vec<f32> {
    let n = source.len();
    let apply = |x: &[f32], out: &mut [f32]| {
        for i in 0..n {
            out[i] = areas[i] * x[i];
        }
        for edge in edges {
            let flux = tau * edge.conductance * (x[edge.a] - x[edge.b]);
            out[edge.a] += flux;
            out[edge.b] -= flux;
        }
    };
    let dot = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum()
    };

    let rhs: Vec<f32> = areas
        .iter()
        .zip(source.iter())
        .map(|(&area, &height)| area * height)
        .collect();
    let mut x = source.to_vec();
    let mut ax = vec![0.0f32; n];
    apply(&x, &mut ax);
    let mut residual: Vec<f32> = rhs.iter().zip(ax.iter()).map(|(&b, &a)| b - a).collect();
    let mut direction = residual.clone();
    let mut residual_sq = dot(&residual, &residual);
    let rhs_norm = dot(&rhs, &rhs).sqrt().max(1e-20);
    let tolerance = 1e-6 * rhs_norm;
    let mut applied = vec![0.0f32; n];

    // Numerical bound only; it is not a worldbuilding parameter.
    for _ in 0..256 {
        if residual_sq.sqrt() <= tolerance {
            break;
        }
        apply(&direction, &mut applied);
        let denom = dot(&direction, &applied);
        if denom <= 1e-30 {
            break;
        }
        let alpha = residual_sq / denom;
        for i in 0..n {
            x[i] += (alpha * direction[i] as f64) as f32;
            residual[i] -= (alpha * applied[i] as f64) as f32;
        }
        let next_residual_sq = dot(&residual, &residual);
        let beta = next_residual_sq / residual_sq.max(1e-30);
        for i in 0..n {
            direction[i] = residual[i] + (beta * direction[i] as f64) as f32;
        }
        residual_sq = next_residual_sq;
    }

    // The exact M-matrix solution is non-negative. Remove floating-point CG
    // undershoot, then restore the volume invariant exactly after roundoff.
    for value in &mut x {
        *value = value.max(0.0);
    }
    let source_volume: f64 = source
        .iter()
        .zip(areas.iter())
        .map(|(&h, &a)| h as f64 * a as f64)
        .sum();
    let solved_volume: f64 = x
        .iter()
        .zip(areas.iter())
        .map(|(&h, &a)| h as f64 * a as f64)
        .sum();
    if solved_volume > 0.0 {
        let scale = (source_volume / solved_volume) as f32;
        for value in &mut x {
            *value *= scale;
        }
    }
    x
}

pub(crate) fn coarse_elevation_fields(
    tessellation: &Tessellation,
    crust: &Crust,
    features: &FeatureFields,
    orogen_model: OrogenModel,
) -> ElevationFields {
    let num_cells = tessellation.num_cells();

    let tectonic_thickening = match orogen_model {
        OrogenModel::Legacy => {
            // BIT-EXACT reproduction of the historical thickening: `(arc+col)/slope`
            // — division (not `* inv_slope`), no `.max(0.0)` (the fields are
            // nonnegative by construction). The legacy default must stay
            // byte-identical through the fine/erosion chaos cascade, so every
            // float op here matches the pre-model-ladder expression.
            let slope = isostasy_slope();
            features
                .arc
                .iter()
                .zip(features.collision.iter())
                .map(|(&arc, &collision)| (arc + collision) / slope)
                .collect()
        }
        OrogenModel::LegacyYield => {
            // Same source as legacy, then strength-limited gravitational
            // spreading of ONLY the over-yield excess (see constants.rs,
            // "legacy-yield orogen rung"). Sub-yield cells keep their exact
            // legacy thickening; over-strength compact loads (the all-sides-
            // convergent salient) cap out and build a conserved foothill apron.
            // The threshold is WIDTH-AWARE: wide belts support taller loads,
            // narrow slivers cap Taiwan-class (Earth's peak-vs-width ≈ sqrt).
            let slope = isostasy_slope();
            let legacy: Vec<f32> = features
                .arc
                .iter()
                .zip(features.collision.iter())
                .map(|(&arc, &collision)| (arc + collision) / slope)
                .collect();
            let yield_ref = OROGEN_YIELD_ELEV / slope;
            let widths = orogen_width_field(
                tessellation,
                &legacy,
                OROGEN_YIELD_FOOTPRINT_FRAC * yield_ref,
            );
            let yields: Vec<f32> = widths
                .iter()
                .map(|&w_rad| {
                    let w_km = w_rad * PLANET_RADIUS_KM;
                    let factor = (w_km / OROGEN_YIELD_WIDTH_REF_KM)
                        .sqrt()
                        .clamp(OROGEN_YIELD_WIDTH_FACTOR_MIN, OROGEN_YIELD_WIDTH_FACTOR_MAX);
                    yield_ref * factor
                })
                .collect();
            let spread = OROGEN_YIELD_SPREAD_KM / PLANET_RADIUS_KM;
            super::deformation::yield_relax(
                tessellation,
                &legacy,
                &yields,
                0.5 * spread * spread,
                OROGEN_YIELD_PICARD_STEPS,
            )
        }
        OrogenModel::ConservedLocal => {
            solve_tectonic_thickening(tessellation, crust, &features.tectonic_crust_flux, 0.0)
        }
        OrogenModel::ConservedFeatureFootprint => {
            let inv_slope = 1.0 / isostasy_slope();
            let mut footprint: Vec<f32> = features
                .arc
                .iter()
                .zip(features.collision.iter())
                .map(|(&arc, &collision)| (arc + collision).max(0.0) * inv_slope)
                .collect();
            let areas = tessellation.cell_areas();
            let target_volume: f64 = features
                .tectonic_crust_flux
                .iter()
                .map(|&flux| flux.max(0.0) as f64 * TECTONIC_ACCUMULATION_TIME.max(0.0) as f64)
                .sum();
            let footprint_volume: f64 = footprint
                .iter()
                .zip(areas.iter())
                .map(|(&height, &area)| height as f64 * area as f64)
                .sum();
            let scale = if footprint_volume > 0.0 {
                (target_volume / footprint_volume) as f32
            } else {
                0.0
            };
            for height in &mut footprint {
                *height *= scale;
            }
            footprint
        }
        OrogenModel::ConservedIsotropic => solve_tectonic_thickening(
            tessellation,
            crust,
            &features.tectonic_crust_flux,
            CRUST_GRAVITATIONAL_DIFFUSIVITY,
        ),
        OrogenModel::HistoryLocal => features
            .tectonic_crust_work
            .iter()
            .zip(tessellation.cell_areas().iter())
            .map(|(&work, &area)| work.max(0.0) / area.max(1e-12))
            .collect(),
        OrogenModel::HistoryDiffusive => {
            solve_history_diffusive_thickening(tessellation, crust, &features.tectonic_episode_work)
        }
        OrogenModel::HistoryMaterial => solve_history_material_thickening(
            tessellation,
            crust,
            &features.tectonic_material_episode_work,
        ),
        OrogenModel::ThinSheet
        | OrogenModel::HistoryThinSheet
        | OrogenModel::HistoryCarrierThinSheet
        | OrogenModel::HistoryCarrierEvolved
        | OrogenModel::HistoryCarrierLifecycle => features.thin_sheet_thickness_delta.clone(),
    };

    let mut crust_thickness = Vec::with_capacity(num_cells);
    let mut continentality_field = Vec::with_capacity(num_cells);
    let mut is_continental = Vec::with_capacity(num_cells);

    // Legacy assembles thickness with the HISTORICAL grouping and clamp position
    // (`(base + thickening + rift).max(0.05)`) for byte-identity; experimental
    // models clamp the pre-tectonic column and add their (possibly negative)
    // model delta after, so a conserved model can thin below the legacy floor.
    // legacy-yield shares the grouping so its sub-yield cells stay bit-equal to
    // legacy at the coarse level.
    let legacy_grouping = matches!(orogen_model, OrogenModel::Legacy | OrogenModel::LegacyYield);
    let lifecycle = orogen_model == OrogenModel::HistoryCarrierLifecycle;
    #[allow(clippy::needless_range_loop)] // indexes 4 parallel sources; zip obscures
    for i in 0..num_cells {
        let continental = if lifecycle {
            features
                .lifecycle_final_continental
                .as_ref()
                .expect("lifecycle final crust projected")[i]
        } else {
            crust.crust_type(i) == CrustType::Continental
        };
        let cont = if lifecycle {
            if continental {
                1.0
            } else {
                0.0
            }
        } else {
            continentality(crust.signed_margin_distance[i], features.convergent[i])
        };
        let base_thickness = CRUST_THICKNESS_OCEANIC
            + cont * (CRUST_THICKNESS_CONTINENTAL - CRUST_THICKNESS_OCEANIC);
        let rift = if lifecycle {
            0.0
        } else {
            features.rift_delta[i] * cont
        };
        // Macro craton-thickness is added later, in assembly (it needs craton
        // structure + per-craton RNG), so it stays out of this transferred field.
        let thickness = if legacy_grouping {
            (base_thickness + tectonic_thickening[i] + rift).max(0.05)
        } else {
            (base_thickness + rift).max(0.05) + tectonic_thickening[i]
        };

        crust_thickness.push(thickness);
        continentality_field.push(cont);
        is_continental.push(continental);
    }

    ElevationFields {
        crust_thickness,
        tectonic_thickening,
        legacy_uplift_source: matches!(
            orogen_model,
            OrogenModel::Legacy | OrogenModel::LegacyYield
        ),
        tectonic_strain: features.thin_sheet_strain.clone(),
        compression_axis: features.thin_sheet_compression_axis.clone(),
        tectonic_uplift_rate: features.tectonic_uplift_rate.clone(),
        allow_source_craton_macro: !lifecycle,
        continentality: continentality_field,
        ridge_age_distance: if lifecycle {
            features
                .lifecycle_ocean_age_myr
                .iter()
                .map(|&age| {
                    age * OCEAN_SPREADING_REFERENCE_RATE * MAX_PLATE_ANGULAR_SPEED_RAD_PER_MYR
                })
                .collect()
        } else {
            features.ridge_age_distance.clone()
        },
        trench: if lifecycle {
            vec![0.0; num_cells]
        } else {
            features.trench.clone()
        },
        ridge: if lifecycle {
            vec![0.0; num_cells]
        } else {
            features.ridge.clone()
        },
        convergent: if lifecycle {
            vec![0.0; num_cells]
        } else {
            features.convergent.clone()
        },
        divergent: if lifecycle {
            vec![0.0; num_cells]
        } else {
            features.divergent.clone()
        },
        is_continental,
        arc: if lifecycle {
            vec![0.0; num_cells]
        } else {
            features.arc.clone()
        },
        collision: if lifecycle {
            vec![0.0; num_cells]
        } else {
            features.collision.clone()
        },
        rift_delta: if lifecycle {
            vec![0.0; num_cells]
        } else {
            features.rift_delta.clone()
        },
    }
}

/// Generate heightmap: thickness field (incl. macro craton perturbation) ->
/// isostasy -> thermal/dynamic terms -> sea-level solve.
fn generate_heightmap(
    tessellation: &Tessellation,
    crust: &Crust,
    features: &FeatureFields,
    orogen_model: OrogenModel,
    macro_field: &[f32],
) -> (Vec<f32>, Vec<f32>, NoiseLayerData) {
    let fields = coarse_elevation_fields(tessellation, crust, features, orogen_model);
    assemble_heightmap(tessellation, &fields, macro_field)
}

fn assemble_heightmap(
    tessellation: &Tessellation,
    fields: &ElevationFields,
    macro_field: &[f32],
) -> (Vec<f32>, Vec<f32>, NoiseLayerData) {
    let num_cells = tessellation.num_cells();
    let slope = isostasy_slope();

    #[cfg(not(feature = "single-threaded"))]
    let assembled: Vec<AssembledElevationCell> = (0..num_cells)
        .into_par_iter()
        .map(|i| assemble_elevation_cell(fields, macro_field, slope, i))
        .collect();

    #[cfg(feature = "single-threaded")]
    let assembled: Vec<AssembledElevationCell> = (0..num_cells)
        .map(|i| assemble_elevation_cell(fields, macro_field, slope, i))
        .collect();

    let mut elevations = Vec::with_capacity(num_cells);
    let mut noise_contributions = Vec::with_capacity(num_cells);
    let mut macro_layer = Vec::with_capacity(num_cells);

    for cell in assembled {
        elevations.push(cell.elevation);
        noise_contributions.push(cell.noise_contribution);
        macro_layer.push(cell.macro_layer);
    }

    // --- 4. Sea-level solve: uniform shift so the land AREA fraction is exact ---
    // Area-weighted, not cell-count-weighted: the adaptive fine mesh has wildly
    // unequal cell areas (sparse huge ocean cells, dense tiny land cells), so a
    // count-based percentile would place sea level up among the land cells and
    // drown most of the landmass. Sorting by elevation and accumulating area
    // gives the correct sea level on any mesh (and matches the old behaviour on
    // the ~equal-area coarse mesh).
    let areas = tessellation.cell_areas();
    let total_area: f32 = areas.iter().sum();
    let target_submerged_area = (1.0 - LAND_FRACTION) * total_area;
    let mut order: Vec<usize> = (0..num_cells).collect();
    order.sort_by(|&a, &b| elevations[a].total_cmp(&elevations[b]));
    let mut accumulated = 0.0f32;
    let mut sea_level = elevations[*order.last().unwrap()];
    for &i in &order {
        accumulated += areas[i];
        if accumulated >= target_submerged_area {
            sea_level = elevations[i];
            break;
        }
    }
    log::debug!("sea level solve: shift={:.4}", -sea_level);

    for e in &mut elevations {
        *e -= sea_level;
    }

    // --- 5. Volcanic island soft cap (relative to true sea level) ---
    // Oceanic crust above sea level can't grow indefinitely; erosion and
    // subsidence limit island height. Kept as a transitional safeguard.
    for i in 0..num_cells {
        if !fields.is_continental[i] && elevations[i] > 0.0 {
            let max_island = VOLCANIC_ISLAND_MAX_HEIGHT;
            elevations[i] = max_island * (elevations[i] / max_island).tanh();
        }
    }

    let noise_layers = NoiseLayerData { macro_layer };

    (elevations, noise_contributions, noise_layers)
}

struct AssembledElevationCell {
    elevation: f32,
    noise_contribution: f32,
    macro_layer: f32,
}

fn assemble_elevation_cell(
    fields: &ElevationFields,
    macro_field: &[f32],
    slope: f32,
    i: usize,
) -> AssembledElevationCell {
    // --- 1. Crust thickness ---
    let cont = fields.continentality[i];
    let base_thickness = fields.crust_thickness[i];

    // Macro-scale thickness variation: craton-structure cores and interior basins
    // (precomputed per cell; see `macro_craton_thickness`).
    let macro_dt = if fields.allow_source_craton_macro {
        macro_field[i]
    } else {
        0.0
    };

    let thickness = (base_thickness + macro_dt).max(0.05);

    // --- 2. Isostatic base + thermal + dynamic terms ---
    // Thermal anomaly applies to the oceanic part of the column;
    // trench flexure is dynamic topography (slab pull holds it out of
    // isostatic equilibrium, with signed outer-rise uplift); the small
    // ridge feature rides on the thermal swell.
    let structural_elevation = isostatic_elevation(thickness)
        + thermal_anomaly(fields.ridge_age_distance[i]) * (1.0 - cont)
        + fields.ridge[i]
        - fields.trench[i];

    // --- 3. Macro thickness is reported as its isostatic elevation contribution
    // for the noise viz (it does not enter the simulation elevation directly; it
    // acts through `thickness`).
    let macro_c = macro_dt * slope;

    AssembledElevationCell {
        elevation: structural_elevation,
        noise_contribution: macro_c,
        macro_layer: macro_c,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{TectonicCarrierConfig, World, NUM_PLATES_DEFAULT};

    #[test]
    fn legacy_orogeny_remains_the_product_default() {
        assert_eq!(OrogenModel::default(), OrogenModel::Legacy);
    }

    #[test]
    fn lifecycle_elevation_uses_only_evolved_crust_state() {
        let mut world = World::new(606, 512, 0);
        world.orogen_model = OrogenModel::HistoryCarrierLifecycle;
        world.tectonic_carrier_config = TectonicCarrierConfig {
            cells: 256,
            step_myr: 2.0,
            ..TectonicCarrierConfig::default()
        };
        world.generate_plates(NUM_PLATES_DEFAULT.min(6));
        world.generate_crust();
        world.generate_dynamics();
        world.dynamics.as_mut().unwrap().clock.lookback_myr = 8.0;
        world.generate_features();
        let fields = coarse_elevation_fields(
            &world.tessellation,
            world.crust.as_ref().unwrap(),
            world.features.as_ref().unwrap(),
            world.orogen_model,
        );
        assert!(!fields.allow_source_craton_macro);
        for field in [
            &fields.trench,
            &fields.ridge,
            &fields.arc,
            &fields.collision,
            &fields.rift_delta,
            &fields.convergent,
            &fields.divergent,
        ] {
            assert!(field.iter().all(|&value| value == 0.0));
        }
        assert!(world
            .features
            .as_ref()
            .unwrap()
            .lifecycle_final_continental
            .is_some());
    }

    #[test]
    fn crust_relaxation_conserves_volume_and_spreads_load() {
        let areas = vec![1.0, 1.0, 1.0];
        let edges = vec![
            RelaxationEdge {
                a: 0,
                b: 1,
                conductance: 1.0,
            },
            RelaxationEdge {
                a: 1,
                b: 2,
                conductance: 1.0,
            },
        ];
        let source = vec![1.0, 0.0, 0.0];
        let solved = solve_screened_conservative(&areas, &edges, &source, 1.0);

        let volume: f32 = solved.iter().zip(&areas).map(|(h, a)| h * a).sum();
        assert!((volume - 1.0).abs() < 1e-5);
        assert!(solved[0] < source[0]);
        assert!(solved[1] > 0.0 && solved[2] > 0.0);
        assert!(solved[0] > solved[1] && solved[1] > solved[2]);
    }

    #[test]
    fn zero_mobility_preserves_local_thickening() {
        let areas = vec![0.5, 1.5];
        let edges = vec![RelaxationEdge {
            a: 0,
            b: 1,
            conductance: 2.0,
        }];
        let source = vec![0.75, 0.1];
        let solved = solve_screened_conservative(&areas, &edges, &source, 0.0);
        assert_eq!(solved, source);
    }

    #[test]
    fn no_flux_boundary_keeps_disconnected_domain_empty() {
        let areas = vec![1.0, 1.0, 1.0];
        let edges = vec![RelaxationEdge {
            a: 0,
            b: 1,
            conductance: 1.0,
        }];
        let source = vec![1.0, 0.0, 0.0];
        let solved = solve_screened_conservative(&areas, &edges, &source, 1.0);
        assert_eq!(solved[2], 0.0);
    }
}
