//! Fluvial erosion on the fine mesh (docs/specs/erosion.md).
//!
//! Detachment-limited stream power (Braun & Willett 2013 implicit scheme, n = 1)
//! with linear hillslope diffusion, plus transport-aware deposition (en-route
//! aggradation toward a repose-slope surface — fans/floodplains/deltas — and
//! base-level sink fill). Terminal (endorheic) pre-erosion lakes act as fixed
//! local base levels. The loop runs once after fine-elevation refinement and
//! before final hydrology.
//!
//! State is CRUST THICKNESS, not elevation. Elevation is coupled to thickness as
//! an isostatic (Airy) delta on top of the interpolated coarse base:
//!
//! ```text
//! elev_i = base_i + isostasy_slope * (thick_i - thick_init_i)
//! ```
//!
//! At t=0 the delta is zero (elev = base, the fixed sea-level datum). Incision
//! thins the crust; the surface drops by `slope * dthick` (< dthick) — that
//! shortfall IS the isostatic rebound. Because `slope < 1`, realizing a given
//! surface drop removes MORE thickness than the drop (dividing by `slope`).
//! Tectonic uplift thickens the crust and raises the surface.
//!
//! The incision operator (`incise_step`) is kept pure in elevation space so it
//! reproduces the standard stream-power steady state (unit-tested); the
//! thickness bookkeeping lives in `erode`, which folds the eroded/diffused
//! surface back into thickness via `(elev - base) / slope` each step.
//!
//! NOTE on isostasy: the incision/diffusion operators work in elevation space,
//! and with `uplift_scale = 0` the fold-back and `derive_elev` are exact inverses
//! (`thick - thick_init ≡ (elev - base) / slope`), so the thickness state then
//! carries no shape information beyond elevation (the "isostatic rebound" above is
//! a volume reinterpretation only). With uplift ON — the Phase 3 "Hold & carve"
//! default — thickness becomes a live degree of freedom: the uplift source
//! thickens the crust and `derive_elev` raises the surface through the Airy slope,
//! so divides are held against incision and the orogen evolves rather than merely
//! decaying.

use std::cmp::Reverse;
use std::collections::VecDeque;
use std::time::Instant;

use glam::Vec3;
use ordered_float::OrderedFloat;
#[cfg(not(feature = "single-threaded"))]
use rayon::prelude::*;

use super::constants::*;
use super::elevation::{isostasy_slope, ElevationFields};
use super::Tessellation;

/// Runtime-tunable erosion knobs. Carried in world/app state so erosion can be
/// re-run with tweaked values without a recompile (staging tooling). `Default`
/// pulls today's `EROSION_*` constants — those stay the source of truth for the
/// defaults; this struct just makes them overridable at runtime.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErosionParams {
    /// Number of erosion steps to run.
    pub steps: usize,
    /// Time increment per step.
    pub dt: f32,
    /// Stream-power drainage-area exponent (m in K A^m).
    pub m: f32,
    /// Stream-power SLOPE exponent (n in K A^m S^n). 1 = linear (closed form); >1 =
    /// sharper valleys/divides (Newton-solved). Only affects SFD incision.
    pub n: f32,
    /// Stream-power erodibility coefficient K.
    pub k: f32,
    /// Linear hillslope diffusivity.
    pub diffusivity: f32,
    /// Roering nonlinear-hillslope critical slope S_c (Δelev/radian). Diffusivity
    /// blows up as the cell slope nears S_c → planar slopes + crisp ridges. 0 =
    /// off (linear creep). See `EROSION_HILLSLOPE_CRITICAL_SLOPE`.
    pub hillslope_critical_slope: f32,
    /// Jacobi sweeps per implicit diffusion solve.
    pub diffusion_iters: usize,
    /// Re-route (and re-accumulate drainage area) every this many steps.
    pub reroute_interval: usize,
    /// Barnes convergent flat resolution on the priority-flood surface (vs the
    /// bare `flood_parent` wavefront that spirals). See `EROSION_FLAT_RESOLUTION`.
    pub flat_resolution: bool,
    /// Multiple-flow-direction drainage-area exponent (`slope^p` flow split);
    /// 0 = single-flow. See `EROSION_MFD_EXPONENT`.
    pub mfd_exponent: f32,
    /// Plains alluvial regime gate: channel slope (elev/km) at/above which
    /// incision is full; gentler channels fade toward alluvial (deposition-graded
    /// floodplains). 0 = off. See `EROSION_CONFINEMENT_SLOPE`.
    pub confinement_slope: f32,
    /// Tectonic uplift source scale (thickness units).
    pub uplift_scale: f32,
    /// Forcing-smoothing length (km) for the uplift SOURCE, applied once before the
    /// step loop (conservative land-masked diffusion of `u_thick`). 0 = off. See
    /// `EROSION_UPLIFT_SMOOTH_KM` / docs/specs/erosion-uplift-smoothing.md.
    pub uplift_smooth_km: f32,
    /// Fraction of a sink's depth-to-base-level that deposition may fill.
    pub deposit_fill_fraction: f32,
    /// Depositional repose slope (elevation per radian). En route to the sea/lakes,
    /// sediment aggrades reaches whose bed lies below the surface grading to the
    /// receiver at this slope — building fans, floodplains, and deltas. 0 = no
    /// en-route deposition (terminal-sink fill only).
    pub deposition_slope: f32,
    /// Channel-initiation support area (km²) at mean land wetness. Below the
    /// equivalent discharge a cell is a hillslope (diffusion only, no
    /// stream-power incision). 0 = off (incise wherever downhill).
    pub channel_support_km2: f32,
    /// Log-amplitude of lithologic erodibility variation (contrast). The K
    /// multiplier is exp(sigma * fbm), normalized to unit mean over land so it
    /// only redistributes incision. 0 = uniform K.
    pub litho_sigma: f32,
    /// Log-amplitude of the structural-grain (fold-belt) erodibility contrast.
    /// 0 = no grain. Experimental (ridge-and-valley / trellis drainage).
    pub litho_grain_strength: f32,
    /// Strength of the orographic precip modulation on the eroded fine relief
    /// (climate↔erosion feedback: windward wetter, lee drier). 0 = coarse precip.
    pub orographic_precip_strength: f32,
    /// Coupled erode↔precip feedback passes (≥1). 2 = one feedback pass.
    pub precip_outer_iters: usize,
    /// Lakes-as-evaporation precip boost strength (local humidity halo). 0 = off.
    pub lake_evap_strength: f32,
    /// Glacial abrasion coefficient (ice flux → bed lowering). 0 = no glacial pass.
    pub glacial_k: f32,
    /// Glacial sub-steps (re-route + accumulate + abrade).
    pub glacial_steps: usize,
    /// Snowline elevation at the equator (only terrain above it glaciates).
    pub glacial_snowline_equator: f32,
    /// Snowline elevation at the poles (lower; uplands glaciate toward the coast).
    pub glacial_snowline_pole: f32,
    /// Ice ablation rate below the snowline (bounds glacier tongue length).
    pub glacial_ablation: f32,
    /// Max reverse gradient glacial abrasion may carve (tarn / over-deepening depth).
    pub glacial_overdeepen_max: f32,
}

impl Default for ErosionParams {
    fn default() -> Self {
        Self {
            steps: EROSION_STEPS,
            dt: EROSION_DT,
            m: EROSION_M,
            n: EROSION_N,
            k: EROSION_K,
            diffusivity: EROSION_DIFFUSIVITY,
            hillslope_critical_slope: EROSION_HILLSLOPE_CRITICAL_SLOPE,
            diffusion_iters: EROSION_DIFFUSION_ITERS,
            reroute_interval: EROSION_REROUTE_INTERVAL,
            flat_resolution: EROSION_FLAT_RESOLUTION,
            mfd_exponent: EROSION_MFD_EXPONENT,
            confinement_slope: EROSION_CONFINEMENT_SLOPE,
            uplift_scale: EROSION_UPLIFT_SCALE,
            uplift_smooth_km: EROSION_UPLIFT_SMOOTH_KM,
            deposit_fill_fraction: EROSION_DEPOSIT_FILL_FRACTION,
            deposition_slope: EROSION_DEPOSITION_SLOPE,
            channel_support_km2: EROSION_CHANNEL_SUPPORT_KM2,
            litho_sigma: EROSION_LITHO_SIGMA,
            litho_grain_strength: EROSION_LITHO_GRAIN_STRENGTH,
            orographic_precip_strength: OROGRAPHIC_PRECIP_STRENGTH,
            precip_outer_iters: EROSION_PRECIP_OUTER_ITERS,
            lake_evap_strength: LAKE_EVAP_STRENGTH,
            glacial_k: GLACIAL_K,
            glacial_steps: GLACIAL_STEPS,
            glacial_snowline_equator: GLACIAL_SNOWLINE_EQUATOR,
            glacial_snowline_pole: GLACIAL_SNOWLINE_POLE,
            glacial_ablation: GLACIAL_ABLATION,
            glacial_overdeepen_max: GLACIAL_OVERDEEPEN_MAX,
        }
    }
}

/// Run the erosion loop and return the eroded fine-mesh elevation (on the fixed
/// sea-level datum). `fields.crust_thickness` seeds the working thickness;
/// `base` is the interpolated coarse elevation; `precipitation` weights drainage
/// area so wet ranges dissect more finely than arid ones.
///
/// Thin wrapper over [`ErosionState`]: `new(..).step(EROSION_STEPS)` is the batch
/// run. The before/after roughness probes (and the diagnostics they print) live
/// here, where the `Tessellation` is in hand.
#[allow(clippy::too_many_arguments)]
pub(crate) fn erode(
    tess: &Tessellation,
    fields: &ElevationFields,
    base: &[f32],
    precipitation: &[f32],
    erodibility: &[f32],
    lake_base: &[f32],
    geom: &NeighborGeometry,
    params: ErosionParams,
    coarse_target: Option<&[f32]>,
    uplift_shape: Option<&[f32]>,
) -> Vec<f32> {
    roughness_report(tess, base, "base ");
    let mut state = ErosionState::new(
        tess,
        fields,
        base,
        precipitation,
        erodibility,
        lake_base,
        geom,
        params,
        coarse_target,
        uplift_shape,
    );
    state.step(params.steps);
    state.log_summary();
    let final_elev = state.elevation();
    roughness_report(tess, &final_elev, "eroded");
    final_elev
}

/// Glacial erosion pass (v1). A snowline-driven ice over-deepening applied on top
/// of the fluvial surface: terrain above a latitude-dependent snowline carries
/// ice; the ice routes downslope (reusing the fluvial [`Routing`]) and abrades its
/// bed in proportion to ice flux, so trunk glaciers over-deepen into troughs and
/// tarn basins while the low-flux peaks/divides between them erode little and stand
/// out sharper (arêtes, horns). Ice ablates below the snowline, giving glaciers
/// bounded tongues. Works in elevation space (no isostatic response in v1) and
/// never carves below sea level. U-shape valley *widening* is the deferred v2 part.
pub(crate) fn glacial_erode(
    tess: &Tessellation,
    elev: &mut [f32],
    lake_base: &[f32],
    geom: &NeighborGeometry,
    params: ErosionParams,
) {
    if params.glacial_k <= 0.0 || params.glacial_steps == 0 {
        return;
    }
    let n = tess.num_cells();
    let areas = tess.cell_areas();

    // Latitude-dependent snowline elevation: EQUATOR at the equator, POLE at the
    // poles, sin²(lat) between. cell_center.y is sin(latitude) on the unit sphere.
    // Elevation is deliberately kept out of the snowline (it is the ice-load term),
    // so the two don't double-count the way the lapse-baked temperature field would.
    let snowline: Vec<f32> = (0..n)
        .map(|i| {
            let s = tess.cell_center(i).y;
            let lat2 = (s * s).clamp(0.0, 1.0);
            params.glacial_snowline_equator
                + (params.glacial_snowline_pole - params.glacial_snowline_equator) * lat2
        })
        .collect();

    // Coverage diagnostic measured on the input (fluvial) surface.
    let (mut gl_land, mut land) = (0usize, 0usize);
    for i in 0..n {
        if elev[i] >= 0.0 {
            land += 1;
            if elev[i] > snowline[i] {
                gl_land += 1;
            }
        }
    }

    // Ice routes to the SAME base levels as the fluvial stage: sea level, or a
    // terminal lake's surface (audit H6). Without this the ice routed to sea
    // level everywhere and could gouge a glaciated cell below the lake floor the
    // fluvial pass graded to.
    let mut total_abraded = 0.0f64;

    for step in 0..params.glacial_steps {
        // Ice routes single-flow (SFD); MFD off for the glacial pass.
        let Some(routing) = Routing::build(elev, geom, lake_base, params.flat_resolution, 0.0)
        else {
            break;
        };

        // Ice flux: accumulate the above-snowline ice load downstream (upstream-
        // first), melting it in proportion to how far each cell sits below the
        // snowline, so glaciers reach a bounded distance past the snowline.
        let mut flux: Vec<f32> = (0..n).map(|i| (elev[i] - snowline[i]).max(0.0)).collect();
        for &cell in routing.order.iter().rev() {
            if routing.is_sink[cell] {
                continue;
            }
            let below = (snowline[cell] - elev[cell]).max(0.0);
            flux[cell] = (flux[cell] - params.glacial_ablation * below).max(0.0);
            flux[routing.receiver[cell]] += flux[cell];
        }

        // PROBE: spikiness of the SFD ice-discharge field that drives abrasion.
        // Reuse roughness_counters over the glaciated subgraph (mask non-ice cells
        // to <0 so they're skipped). checker% near ~50% = white-noise cell-scale
        // concentration (the single-flow point-routing artifact: all flux dumps
        // into one downstream cell); near the elevation base (~41%) = smooth
        // distributed flow. The physical-fix test for going MFD/diffusive on ice.
        if log::log_enabled!(log::Level::Info) && step + 1 == params.glacial_steps {
            let masked: Vec<f32> = flux
                .iter()
                .map(|&f| if f > 0.0 { f } else { -1.0 })
                .collect();
            let fr = roughness_counters(tess, &masked);
            log::info!(
                "glacial ice-flux (SFD) spikiness: checker {:.2}% (elev base ~41.6%) | pit {:.2}% peak {:.2}% | glaciated-flux cells {}",
                fr.checkerboard_pct,
                fr.pit_pct,
                fr.peak_pct,
                fr.land,
            );
        }

        // Abrasion: lower the bed by k·flux, allowing limited over-deepening below
        // the receiver (tarns) but never below sea level. Snapshot so every cell
        // uses the pre-step surface (consistent receiver elevations).
        let pre = elev.to_vec();
        for i in 0..n {
            if routing.is_sink[i] || flux[i] <= 0.0 {
                continue;
            }
            let r = routing.receiver[i];
            // Floor at the cell's terminal base level (sea level, or the surface
            // of the terminal lake it drains into), not just sea level — so ice
            // can't over-deepen below a carved lake floor (audit H6).
            let floor = (pre[r] - params.glacial_overdeepen_max).max(routing.base_floor[i]);
            let lowered = (pre[i] - params.glacial_k * flux[i]).max(floor);
            if lowered < pre[i] {
                total_abraded += ((pre[i] - lowered) * areas[i]) as f64;
                elev[i] = lowered;
            }
        }
    }

    log::info!(
        "glacial: {} steps, snowline {:.3}..{:.3} (pole..eq) | glaciated land {:.2}% | abraded volume(elev*sr)={:.3e}",
        params.glacial_steps,
        params.glacial_snowline_pole,
        params.glacial_snowline_equator,
        100.0 * gl_land as f32 / land.max(1) as f32,
        total_abraded,
    );
    // Post-glacial relief probe (compare to the "eroded" line for the glacial
    // delta: glaciated valleys deepen, peaks/divides between them stand out).
    roughness_report(tess, elev, "glacial");
}

/// Resumable erosion. `ErosionState::new(..).step(EROSION_STEPS)` reproduces the
/// batch `erode()` run bit-for-bit; the UI can instead `step()` in increments to
/// watch valleys carve. Holds every loop-carried value (working thickness, cached
/// routing + drainage area, step counter) plus the once-built geometry/uplift, so
/// stepping needs no `Tessellation` borrow — only construction does. The owned
/// input copies keep the state self-contained across frames.
pub(crate) struct ErosionState {
    n: usize,
    slope: f32,
    inv_slope: f32,
    /// Channel-initiation discharge threshold (precip x steradian); cells with
    /// less drainage than this are hillslopes (no stream-power incision).
    /// Derived in `new` from `params.channel_support_km2` and mean land precip.
    a_crit: f32,
    /// Per-cell incision-K multiplier (lithologic erodibility), normalized in
    /// `new` to unit area-weighted mean over land so it redistributes, not
    /// scales, total incision. All-ones when `litho_sigma = 0`.
    erodibility: Vec<f32>,

    // Immutable inputs (owned so the state stands alone after construction).
    base: Vec<f32>,
    thick_init: Vec<f32>,
    precipitation: Vec<f32>,
    /// Tectonic uplift source in THICKNESS units per step (see `new`).
    u_thick: Vec<f32>,
    /// Terminal (endorheic) lake surface per cell (finite => lake sink), or
    /// NEG_INFINITY. Frozen from the pre-erosion hydrology; defines extra sinks
    /// and the per-cell base level the routing grades to.
    lake_base: Vec<f32>,
    areas: Vec<f32>,
    geom: NeighborGeometry,
    params: ErosionParams,

    // Working state, carried across steps.
    thick: Vec<f32>,
    /// Re-routed only every EROSION_REROUTE_INTERVAL steps and reused (with the
    /// drainage area it determines) in between; the incision slope-guard
    /// tolerates a receiver that has gone non-downhill as the surface evolves.
    routing: Option<Routing>,
    area: Vec<f32>,
    /// `area[i].powf(m)` — the drainage-area term of the stream-power incision
    /// law. `area` only changes on a reroute, so this is memoized alongside it and
    /// reused across the steps in between rather than recomputing a `powf` per
    /// channel cell every step. Only filled for channel cells (`area >= a_crit`);
    /// other entries are 0 and never read (incision gates on the same threshold).
    area_pow: Vec<f32>,
    /// Per-edge hillslope-diffusion weights with sink-neighbour edges zeroed,
    /// aligned with the geometry CSR. Rebuilt each reroute (sink set is fixed
    /// between reroutes) and reused by every diffusion sweep in the interval to
    /// drop the per-edge `is_sink` lookup. Only the linear creep law uses it; the
    /// Roering law's conductivity is per-step (lagged slope) so it can't be cached
    /// and this stays empty.
    diff_w_land: Vec<f32>,
    /// Per-step scratch buffers (elevation working copy + two snapshots + the
    /// eroded-volume accumulator), reused across steps via `mem::take` to avoid
    /// reallocating ~4xN floats every step. Contents are overwritten each step.
    scratch_elev: Vec<f32>,
    scratch_pre: Vec<f32>,
    scratch_eroded: Vec<f32>,
    scratch_pre_diff: Vec<f32>,
    step: usize,
    /// Set once a route finds no sinks (all-land world): nothing to erode toward.
    halted: bool,

    // Mass-balance accounting (thickness-volume = thickness * area) and per-phase
    // timing, accumulated across steps.
    total_eroded: f64,
    total_deposited: f64,
    total_lost: f64,
    /// Net thickness-volume moved by hillslope diffusion (≈0 if conservative;
    /// nonzero = no-flux clamp + finite-Jacobi non-convergence). Audited so
    /// diffusion's otherwise-silent mass change is visible next to incision.
    total_diffused: f64,
    t_route: f64,
    t_accum: f64,
    t_incise: f64,
    t_diffuse: f64,
    t_deposit: f64,
    t_misc: f64,
}

impl ErosionState {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        tess: &Tessellation,
        fields: &ElevationFields,
        base: &[f32],
        precipitation: &[f32],
        erodibility_in: &[f32],
        lake_base: &[f32],
        geom: &NeighborGeometry,
        params: ErosionParams,
        // EMERGENT (erosion-v3): when Some, the full coarse-target elevation. The
        // builder uplift gates on THIS (not the demoted `base`), so a demoted-below-sea
        // orogen cell still uplifts back, and it EXCLUDES rift_delta (only the demoted
        // arc+collision orogen term is re-supplied). None = current painted behaviour.
        coarse_target: Option<&[f32]>,
        // O0 structured uplift shape (orogen-structure.md). When Some (+ emergent), the
        // builder volume-normalizes this to the demoted volume and uses it as the uplift
        // source — an asymmetric, segmented tectonic distribution — instead of the uniform
        // per-cell `target−base` rebuild.
        uplift_shape: Option<&[f32]>,
    ) -> Self {
        let n = tess.num_cells();
        let slope = isostasy_slope();
        let inv_slope = 1.0 / slope;
        let thick_init = fields.crust_thickness.clone();

        // Tectonic uplift source, in THICKNESS units per step. arc/collision are
        // elevation magnitudes (-> thickness via /slope); rift_delta is already a
        // signed thickness delta (negative in the axial valley). NOT atmospheric
        // uplift. Scaled by params.uplift_scale.
        //
        // Gated to land (base >= sea level). Incision and diffusion skip submerged
        // cells, so uplift there would be a one-way thickness addition with nothing
        // removing it — over many steps it lifts arc/collision-forced ocean cells
        // (e.g. subduction trench margins) across sea level and manufactures spurious
        // land, breaking the fixed sea-level datum. (Terminal-lake sinks are left
        // uplifting: those are either sub-sea-level — already excluded — or genuinely
        // tectonically active basins where uplift/inversion is physical.)
        let epoch = (params.steps as f32 * params.dt).max(1.0);
        // O0 (orogen-structure): volume-normalization constant for the STRUCTURED builder.
        // The shape redistributes uplift tectonically (asymmetric/segmented); normalize it
        // so the total uplifted elevation·area equals the gained demoted volume — height
        // stays sane while the distribution becomes tectonic. `shape_c = gain·Σ(demoted·a)/
        // Σ(shape·a)`; then uplifted-elev[i] = shape_c·shape[i].
        let shape_c: Option<f32> = match (coarse_target, uplift_shape) {
            (Some(target), Some(shape)) => {
                let a = tess.cell_areas();
                let mut dvol = 0.0f64;
                let mut svol = 0.0f64;
                for i in 0..n {
                    if target[i] >= 0.0 {
                        let ai = a[i] as f64;
                        dvol += (target[i] - base[i]).max(0.0) as f64 * ai;
                        svol += shape[i].max(0.0) as f64 * ai;
                    }
                }
                Some(if svol > 0.0 {
                    (EMERGENT_REBUILD_GAIN as f64 * dvol / svol) as f32
                } else {
                    0.0
                })
            }
            _ => None,
        };
        let mut u_thick: Vec<f32> = (0..n)
            .map(|i| {
                if let Some(target) = coarse_target {
                    // EMERGENT (erosion-v3): SELF-CALIBRATING builder. Rebuild the demoted
                    // envelope (target − base = the removed orogen, ≈ λ·(arc+collision))
                    // over the erosion epoch, ×gain to offset what erosion removes WHILE
                    // building, so the eroded orogen lands near the coarse target. Gated on
                    // the TARGET land mask so demoted-below-sea orogen cells still rebuild.
                    // `params.uplift_scale` is unused here (rate is auto-derived), so `steps`
                    // is a pure build-vs-carve dial.
                    if target[i] < 0.0 {
                        return 0.0;
                    }
                    if let (Some(c), Some(shape)) = (shape_c, uplift_shape) {
                        // O0 structured: the volume-normalized tectonic uplift shape.
                        c * shape[i].max(0.0) * inv_slope / epoch
                    } else {
                        // Uniform v3 rebuild: per-cell exact (height tracks target).
                        let demoted = (target[i] - base[i]).max(0.0);
                        EMERGENT_REBUILD_GAIN * demoted * inv_slope / epoch
                    }
                } else {
                    // Painted path (hold & carve): ongoing tectonic uplift from the
                    // arc/collision forcing + rift, gated to land (base ≥ sea level).
                    if base[i] < 0.0 {
                        return 0.0;
                    }
                    params.uplift_scale
                        * ((fields.arc[i] + fields.collision[i]) * inv_slope + fields.rift_delta[i])
                }
            })
            .collect();

        let areas = tess.cell_areas();
        // Built once by the caller and reused across the coupled erode↔precip
        // passes (and the glacial pass) — the geometry is a function of the
        // immutable tessellation alone. Cloning is a cheap CSR memcpy that avoids
        // re-scanning every cell's Voronoi vertices for shared-edge lengths.
        let geom = geom.clone();

        // Escalation #1: smooth the uplift FORCING (never the final elevation) over
        // a physically-named length — the sub-grid orogenic forcing width. The
        // coarse->fine handoff can speckle the source below the landform scale;
        // erosion then re-injects that high-frequency tectonic work every step and
        // the orogens carry cell-scale chatter. A conservative, land-masked
        // diffusion of `u_thick` removes the sub-landform speckle while preserving
        // the integrated (area-weighted) uplift exactly. See
        // docs/specs/erosion-uplift-smoothing.md.
        if params.uplift_smooth_km > 0.0 {
            // Emergent: smooth over the coarse-TARGET land mask, so the conservative
            // no-flux boundary sits at the intended coastline, not the demoted
            // envelope's (which would distort the broad builder uplift in the
            // demoted-below-sea land-flip region — codex).
            let smooth_mask = coarse_target.unwrap_or(base);
            let before = tess.morans_i(&u_thick);
            smooth_uplift_source(
                &mut u_thick,
                &geom,
                &areas,
                smooth_mask,
                params.uplift_smooth_km,
            );
            let after = tess.morans_i(&u_thick);
            log::info!(
                "erosion: uplift source smoothed over {:.1} km — Moran's I {:.4} -> {:.4}",
                params.uplift_smooth_km,
                before,
                after
            );
        }

        // Channel-initiation threshold. The knob is a geometric support area
        // (km²) at MEAN land wetness; convert it to a discharge (precip x
        // steradian) using the area-weighted mean land precipitation, so the
        // threshold is precip-scale-robust and the GEOMETRIC support shrinks
        // where it rains more than average -> denser channel networks in wet
        // regions, sparser in arid ones, without a separate knob.
        let a_crit = if params.channel_support_km2 > 0.0 {
            let (mut wp, mut wa) = (0.0f64, 0.0f64);
            for i in 0..n {
                if base[i] >= 0.0 {
                    wp += (precipitation[i].max(0.0) * areas[i]) as f64;
                    wa += areas[i] as f64;
                }
            }
            let mean_precip = if wa > 0.0 { (wp / wa) as f32 } else { 0.0 };
            let support_sr = params.channel_support_km2 / (PLANET_RADIUS_KM * PLANET_RADIUS_KM);
            support_sr * mean_precip
        } else {
            0.0
        };

        // Normalize the lithologic erodibility multiplier to unit area-weighted
        // mean over land, so it redistributes incision (harder/softer rock)
        // without changing total denudation from the uniform-K baseline.
        let erodibility = {
            let (mut wsum, mut asum) = (0.0f64, 0.0f64);
            for i in 0..n {
                if base[i] >= 0.0 {
                    wsum += (erodibility_in[i] * areas[i]) as f64;
                    asum += areas[i] as f64;
                }
            }
            let mean = if asum > 0.0 {
                (wsum / asum) as f32
            } else {
                1.0
            };
            let inv = if mean > 0.0 { 1.0 / mean } else { 1.0 };
            erodibility_in
                .iter()
                .map(|&e| e * inv)
                .collect::<Vec<f32>>()
        };

        let thick = thick_init.clone();

        Self {
            n,
            slope,
            inv_slope,
            a_crit,
            erodibility,
            base: base.to_vec(),
            thick_init,
            precipitation: precipitation.to_vec(),
            u_thick,
            lake_base: lake_base.to_vec(),
            areas,
            geom,
            params,
            thick,
            routing: None,
            area: Vec::new(),
            area_pow: Vec::new(),
            diff_w_land: Vec::new(),
            scratch_elev: vec![0.0f32; n],
            scratch_pre: vec![0.0f32; n],
            scratch_eroded: vec![0.0f32; n],
            scratch_pre_diff: vec![0.0f32; n],
            step: 0,
            halted: false,
            total_eroded: 0.0,
            total_deposited: 0.0,
            total_lost: 0.0,
            total_diffused: 0.0,
            t_route: 0.0,
            t_accum: 0.0,
            t_incise: 0.0,
            t_diffuse: 0.0,
            t_deposit: 0.0,
            t_misc: 0.0,
        }
    }

    /// Current eroded elevation on the fixed sea-level datum (the isostatic delta
    /// `slope * (thick - thick_init)` on top of the coarse base).
    pub(crate) fn elevation(&self) -> Vec<f32> {
        self.derive_elev()
    }

    fn derive_elev(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.n);
        self.derive_elev_into(&mut out);
        out
    }

    /// Write the derived elevation into `out` (reusing its allocation), avoiding
    /// the per-step Vec allocation of [`derive_elev`]. Byte-identical to it.
    fn derive_elev_into(&self, out: &mut Vec<f32>) {
        out.clear();
        out.extend(
            (0..self.n).map(|i| self.base[i] + self.slope * (self.thick[i] - self.thick_init[i])),
        );
    }

    /// Advance `n_steps` steps, stopping early once a route finds no sinks.
    pub(crate) fn step(&mut self, n_steps: usize) {
        for _ in 0..n_steps {
            if self.halted {
                break;
            }
            self.advance_one();
        }
    }

    fn advance_one(&mut self) {
        let n = self.n;
        // Reusable per-step scratch (taken before `routing` is borrowed so the
        // borrows stay disjoint); restored at the end / on the early return.
        let mut elev = std::mem::take(&mut self.scratch_elev);
        let mut pre = std::mem::take(&mut self.scratch_pre);
        let mut eroded_vol = std::mem::take(&mut self.scratch_eroded);
        let mut pre_diff = std::mem::take(&mut self.scratch_pre_diff);
        let mut s = Instant::now();
        self.derive_elev_into(&mut elev);
        self.t_misc += s.elapsed().as_secs_f64();

        // 1. Route: receivers across pits via priority-flood fill + steepest
        //    descent. Cells below sea level are fixed base-level sinks.
        if self.step % self.params.reroute_interval == 0 {
            s = Instant::now();
            let Some(r) = Routing::build(
                &elev,
                &self.geom,
                &self.lake_base,
                self.params.flat_resolution,
                self.params.mfd_exponent,
            ) else {
                // No sinks (e.g. an all-land world): nothing to erode toward.
                self.halted = true;
                self.scratch_elev = elev;
                self.scratch_pre = pre;
                self.scratch_eroded = eroded_vol;
                self.scratch_pre_diff = pre_diff;
                return;
            };
            self.t_route += s.elapsed().as_secs_f64();

            // 2. Accumulate precipitation-weighted drainage area ("wet area"):
            //    single-flow (SFD) or multiple-flow (MFD, Rung 2) per the exponent.
            s = Instant::now();
            self.area = match r.mfd.as_ref() {
                Some(mfd) => accumulate_wet_area_mfd(&r, mfd, &self.precipitation, &self.areas),
                None => accumulate_wet_area(&r, &self.precipitation, &self.areas),
            };
            // Memoize the A^m stream-power term while `area` is fresh: it is reused
            // unchanged by `incise_step` for every step until the next reroute.
            let m = self.params.m;
            let a_crit = self.a_crit;
            self.area_pow.clear();
            self.area_pow.resize(n, 0.0);
            for i in 0..n {
                if self.area[i] >= a_crit {
                    self.area_pow[i] = self.area[i].powf(m);
                }
            }
            // Cache the sink-zeroed diffusion edge weights for the linear creep
            // law (the only law whose conductivity is constant between reroutes).
            if self.params.hillslope_critical_slope <= 0.0 {
                self.diff_w_land = self.geom.land_weights(&r.is_sink);
            }
            self.t_accum += s.elapsed().as_secs_f64();

            self.routing = Some(r);
        }
        let routing = self.routing.as_ref().unwrap();

        // 3. Incise (implicit, receivers-first). Snapshot first so we can book
        //    the removed rock as eroded volume and route it downstream. MFD-DAG
        //    incision distributes the carving across downslope neighbours (Rung 3);
        //    SFD carves toward the single receiver.
        s = Instant::now();
        pre.clear();
        pre.extend_from_slice(&elev);
        match routing.mfd.as_ref() {
            Some(mfd) => incise_step_mfd(
                &mut elev,
                mfd,
                &routing.is_sink,
                &self.area,
                &self.area_pow,
                self.a_crit,
                &self.erodibility,
                self.params.k,
                self.params.dt,
                self.params.confinement_slope,
            ),
            None => incise_step(
                &mut elev,
                &routing.receiver,
                &routing.is_sink,
                &routing.dist,
                &self.area,
                &self.area_pow,
                self.a_crit,
                &self.erodibility,
                &routing.order,
                self.params.k,
                self.params.dt,
                self.params.n,
                self.params.confinement_slope,
            ),
        }
        // No fluvial erosion below the local base level (sea level, or a terminal
        // lake's surface for cells draining into it): land cells may incise down
        // to, but not past, that datum. `floor.min(pre)` stops incision without
        // ever raising terrain that already sits below the base level.
        eroded_vol.clear();
        eroded_vol.resize(n, 0.0);
        for i in 0..n {
            if routing.is_sink[i] {
                continue;
            }
            let floor = routing.base_floor[i].min(pre[i]);
            if elev[i] < floor {
                elev[i] = floor;
            }
            let drop = pre[i] - elev[i];
            if drop > 0.0 {
                eroded_vol[i] = drop * self.inv_slope * self.areas[i];
            }
        }
        self.t_incise += s.elapsed().as_secs_f64();

        // 4. Diffuse hillslope creep on land, implicit Jacobi. Linear by default;
        //    with a critical slope set, the edge conductivity follows the Roering
        //    nonlinear law (planar slopes + crisp ridges). Snapshot first so the
        //    net volume it moves is auditable: interior diffusion is conservative
        //    (symmetric edge fluxes — the κ multiplier is symmetric too), so this
        //    residual measures only the no-flux clamp + finite-Jacobi
        //    non-convergence and should stay small next to `eroded`.
        s = Instant::now();
        pre_diff.clear();
        pre_diff.extend_from_slice(&elev);
        // Linear creep reuses the per-reroute sink-zeroed edge weights; the
        // Roering law recomputes conductivity per step from the lagged slope.
        let w_land =
            (self.params.hillslope_critical_slope <= 0.0).then_some(self.diff_w_land.as_slice());
        diffuse_land(
            &mut elev,
            routing,
            &self.geom,
            &self.areas,
            w_land,
            self.params.dt,
            self.params.diffusivity,
            self.params.diffusion_iters,
            self.params.hillslope_critical_slope,
        );
        self.total_diffused += (0..n)
            .map(|i| ((elev[i] - pre_diff[i]) * self.inv_slope * self.areas[i]) as f64)
            .sum::<f64>();
        self.t_diffuse += s.elapsed().as_secs_f64();

        s = Instant::now();
        // 5. Fold the eroded/diffused surface back into thickness.
        for i in 0..n {
            self.thick[i] = self.thick_init[i] + (elev[i] - self.base[i]) * self.inv_slope;
        }
        // 6. Uplift source (thickness).
        for i in 0..n {
            self.thick[i] += self.u_thick[i] * self.params.dt;
        }
        self.t_misc += s.elapsed().as_secs_f64();

        // 7. Deposit: aggrade low-gradient reaches toward a repose-slope surface
        //    en route (floodplains, fans, deltas), then fill terminal sinks toward
        //    their local base level (sea level or a lake surface).
        s = Instant::now();
        let (deposited, lost) = deposit(
            routing,
            &eroded_vol,
            &elev,
            &self.areas,
            self.slope,
            self.params.deposit_fill_fraction,
            self.params.deposition_slope,
            &mut self.thick,
        );
        self.total_eroded += eroded_vol.iter().map(|&v| v as f64).sum::<f64>();
        self.total_deposited += deposited;
        self.total_lost += lost;
        self.t_deposit += s.elapsed().as_secs_f64();

        self.step += 1;

        // Return the scratch buffers (with their capacity) for reuse next step.
        self.scratch_elev = elev;
        self.scratch_pre = pre;
        self.scratch_eroded = eroded_vol;
        self.scratch_pre_diff = pre_diff;
    }

    /// Log mass-balance and per-phase timing for the run so far.
    fn log_summary(&self) {
        // Tectonic uplift thickness-volume injected over the run. The eroded /
        // deposited figures below are an INTERNAL transfer (incision moves rock
        // to sinks); this uplift is NET ADDITION and is NOT in that balance — so
        // a large positive uplift_in here means the orogen grew, even when
        // eroded ~= deposited makes the transfer look conserved.
        let per_step_uplift: f64 = (0..self.n)
            .map(|i| (self.u_thick[i] * self.areas[i]) as f64)
            .sum();
        let uplift_in = self.step as f64 * self.params.dt as f64 * per_step_uplift;
        log::info!(
            "erosion: {} steps, uplift-in {:.3e} | eroded {:.3e} deposited {:.3e} lost-to-ocean {:.3e} diffused-net {:.3e} (volume = thickness x steradian)",
            self.step,
            uplift_in,
            self.total_eroded,
            self.total_deposited,
            self.total_lost,
            self.total_diffused,
        );
        log::info!(
            "erosion phases (s): route {:.1} accum {:.1} incise {:.1} diffuse {:.1} deposit {:.1} misc {:.1}",
            self.t_route, self.t_accum, self.t_incise, self.t_diffuse, self.t_deposit, self.t_misc,
        );
    }
}

/// Artifact counters that separate genuine fluvial dissection from SFD +
/// flat-routing discretization texture (the "fractal spiral ridge / swiss-cheese"
/// look — see docs/specs/erosion.md "Roughness counters"). Computed over the land
/// cells of `elev`. Pulled out of [`roughness_report`] so the diagnose binary can
/// print them on any surface (e.g. pre-erosion vs eroded) in its own report style.
pub struct RoughnessCounters {
    /// Land cells (elev >= 0) considered.
    pub land: usize,
    /// % of land cells strictly lower than every land neighbour (closed sinks):
    /// the unambiguous "swiss-cheese perforation" signal. A drained fluvial
    /// surface is ~0% (every land cell routes to a sink), so this is the one
    /// Goodhart-safe target — lower is unambiguously better.
    pub pit_pct: f32,
    /// % of land cells strictly higher than every land neighbour. Split from pits
    /// because real horns/spires are peaks too (not purely an artifact).
    pub peak_pct: f32,
    /// % of land-land edges whose endpoints have OPPOSITE discrete-curvature sign
    /// (a bump beside a dip): the cell-scale high-low banding of single-flow
    /// grooves that hillslope diffusion failed to erase. Reference points: ~50%
    /// is spatially-uncorrelated (white-noise) curvature, a coherent smooth
    /// surface sits well below, so read the base→eroded *rise toward 50%*.
    pub checkerboard_pct: f32,
    /// RMS discrete curvature (elev-units; ~10 km/unit, so 1e-3 ≈ 10 m): the
    /// amplitude of that cell-scale banding, read beside `checkerboard_pct`.
    pub curv_rms: f32,
    /// Steepest-descent azimuth mean-resultant length in [0,1] (0 = isotropic
    /// dendritic drainage; up = mesh-direction locking / global spiral bias).
    pub aspect_r: f32,
    /// Steepest-descent azimuth entropy normalized by ln(bins) (1 = uniform across
    /// sectors; down = anisotropy). A *pure* spiral covers all azimuths, so this
    /// catches GLOBAL anisotropy, not local curl — judge spirals on the map too.
    pub aspect_entropy: f32,
    /// Land cells with a strictly-downhill land neighbour that fed the aspect stats.
    pub aspect_n: usize,
}

/// Compute the [`RoughnessCounters`] for `elev` on `tess`. Two cheap passes over
/// the mesh (per-cell curvature/extrema/aspect, then a curvature-sign edge pass).
pub fn roughness_counters(tess: &Tessellation, elev: &[f32]) -> RoughnessCounters {
    const ASPECT_BINS: usize = 16;
    let n = tess.num_cells();
    let mut land = 0usize;
    let mut pits = 0usize;
    let mut peaks = 0usize;
    // Discrete curvature per cell (elev minus mean land-neighbour elev): cell-scale
    // relief sign + amplitude. NaN marks ocean / no-land-neighbour cells so the
    // edge pass below can skip them.
    let mut curv = vec![f32::NAN; n];
    // Drainage-aspect accumulators: resultant (sum of unit azimuth vectors) and a
    // sector histogram, over land cells that have a strictly-lower land neighbour.
    let (mut aspect_cos, mut aspect_sin) = (0.0f64, 0.0f64);
    let mut aspect_hist = [0u32; ASPECT_BINS];
    let mut aspect_count = 0usize;
    for i in 0..n {
        if elev[i] < 0.0 {
            continue;
        }
        land += 1;
        let pos = tess.cell_center(i);
        let mut all_higher = true;
        let mut all_lower = true;
        let mut has_land_nb = false;
        let mut nb_sum = 0.0f32;
        let mut nb_cnt = 0u32;
        let mut lowest_nb: Option<(usize, f32)> = None; // (idx, elev)
        for &nb in tess.neighbors(i) {
            if elev[nb] < 0.0 {
                continue;
            }
            has_land_nb = true;
            nb_sum += elev[nb];
            nb_cnt += 1;
            if lowest_nb.is_none_or(|(_, e)| elev[nb] < e) {
                lowest_nb = Some((nb, elev[nb]));
            }
            if elev[nb] <= elev[i] {
                all_higher = false;
            }
            if elev[nb] >= elev[i] {
                all_lower = false;
            }
        }
        if !has_land_nb {
            continue;
        }
        if all_higher {
            pits += 1; // strictly lower than every land neighbour (closed sink)
        } else if all_lower {
            peaks += 1;
        }
        curv[i] = elev[i] - nb_sum / nb_cnt as f32;
        // Steepest-descent azimuth toward the lowest land neighbour, in the tangent
        // plane at this cell. Skip near the poles (basis degenerates) and cells
        // with no downhill land neighbour (local pits/flats).
        if let Some((r, re)) = lowest_nb {
            if re < elev[i] && pos.y.abs() < 0.999 {
                // East/north tangent basis matching the project's longitude
                // convention (atan2(z, x), see app/export.rs): at (1,0,0) east is
                // +Z. R/entropy are reflection-invariant, but keep real bearings.
                let east = pos.cross(Vec3::Y).normalize();
                let north = east.cross(pos);
                let chord = tess.cell_center(r) - pos;
                let tang = chord - pos * chord.dot(pos); // project to tangent plane
                let (e, no) = (tang.dot(east), tang.dot(north));
                if e.abs() + no.abs() > 1e-12 {
                    let theta = e.atan2(no); // bearing, (-pi, pi]
                    aspect_cos += theta.cos() as f64;
                    aspect_sin += theta.sin() as f64;
                    let b = (((theta + std::f32::consts::PI) / (2.0 * std::f32::consts::PI))
                        * ASPECT_BINS as f32) as usize;
                    aspect_hist[b.min(ASPECT_BINS - 1)] += 1;
                    aspect_count += 1;
                }
            }
        }
    }

    // Checkerboard index: land-land edges whose endpoints have opposite curvature
    // sign (a bump beside a dip), plus the RMS curvature amplitude.
    let (mut cb_edges, mut cb_total) = (0usize, 0usize);
    let (mut curv_sq, mut curv_cnt) = (0.0f64, 0usize);
    for i in 0..n {
        if curv[i].is_nan() {
            continue;
        }
        curv_sq += (curv[i] as f64).powi(2);
        curv_cnt += 1;
        for &nb in tess.neighbors(i) {
            if nb > i && !curv[nb].is_nan() {
                cb_total += 1;
                if curv[i] * curv[nb] < 0.0 {
                    cb_edges += 1;
                }
            }
        }
    }

    // Aspect isotropy. R = |mean unit azimuth| in [0,1] (0 = isotropic); entropy
    // normalized by ln(bins) (1 = uniform across sectors).
    let aspect_r = if aspect_count > 0 {
        ((aspect_cos * aspect_cos + aspect_sin * aspect_sin).sqrt() / aspect_count as f64) as f32
    } else {
        0.0
    };
    let aspect_entropy = if aspect_count > 0 {
        let inv = 1.0 / aspect_count as f64;
        let h: f64 = aspect_hist
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 * inv;
                -p * p.ln()
            })
            .sum();
        (h / (ASPECT_BINS as f64).ln()) as f32
    } else {
        0.0
    };

    let pct = |num: usize, den: usize| 100.0 * num as f32 / den.max(1) as f32;
    RoughnessCounters {
        land,
        pit_pct: pct(pits, land),
        peak_pct: pct(peaks, land),
        checkerboard_pct: pct(cb_edges, cb_total),
        curv_rms: (curv_sq / curv_cnt.max(1) as f64).sqrt() as f32,
        aspect_r,
        aspect_entropy,
        aspect_n: aspect_count,
    }
}

/// Numerical roughness probe (a REFERENCE, not an optimization target). On land
/// cells it reports the local-slope distribution (|d elev| / km to the steepest
/// neighbour), Moran's I of the whole field, and the artifact [`RoughnessCounters`]
/// (pit% / peak% / checkerboard% / drainage-aspect isotropy) that tell genuine
/// fluvial dissection apart from SFD + flat-routing discretization texture.
///
/// Compare `base ` vs `eroded` to see what erosion did to terrain coherence
/// rather than guessing from a render.
fn roughness_report(tess: &Tessellation, elev: &[f32], label: &str) {
    // Everything below is emitted through log::info!. When Info is disabled
    // (the default env_logger level, and the headless/sweep condition) the
    // whole body — a full-mesh slope/edge pass, roughness_counters, cell_areas,
    // morans_i, and two land-sized sorts (~1.3 M cells) — would be computed and
    // then discarded. Called 5x/erosion run, so skip it when nothing will print.
    // The diagnose binary uses roughness_counters() directly and is unaffected.
    if !log::log_enabled!(log::Level::Info) {
        return;
    }
    let n = tess.num_cells();
    let mut slopes: Vec<f32> = Vec::new();
    let mut land = 0usize;
    let mut min_d_km = f32::INFINITY;
    let mut edges_total = 0usize;
    let mut edges_sub_100m = 0usize;
    let mut edges_sub_1m = 0usize;
    // Absolute land-elevation distribution + area-weighted volume. None of the
    // shape probes (slope/concavity/extrema) measure net height, so an orogen
    // can grow under "erosion" with every shape metric still looking right —
    // compare these base-vs-eroded to catch that.
    let areas = tess.cell_areas();
    let mut elevs: Vec<f32> = Vec::new();
    let mut land_volume = 0.0f64;
    for i in 0..n {
        if elev[i] < 0.0 {
            continue;
        }
        land += 1;
        elevs.push(elev[i]);
        land_volume += elev[i] as f64 * areas[i] as f64;
        let pos = tess.cell_center(i);
        let mut steepest = 0.0f32;
        let mut has_land_nb = false;
        for &nb in tess.neighbors(i) {
            if elev[nb] < 0.0 {
                continue;
            }
            has_land_nb = true;
            // Chord distance (accurate in f32); acos(dot) collapses to 0 below ~3 km.
            let d_km = (pos - tess.cell_center(nb)).length() * PLANET_RADIUS_KM;
            if nb > i {
                edges_total += 1;
                min_d_km = min_d_km.min(d_km);
                if d_km < 0.1 {
                    edges_sub_100m += 1;
                }
                if d_km < 1e-3 {
                    edges_sub_1m += 1;
                }
            }
            steepest = steepest.max((elev[i] - elev[nb]).abs() / d_km.max(1e-6));
        }
        if has_land_nb {
            slopes.push(steepest);
        }
    }
    if slopes.is_empty() {
        return;
    }

    let c = roughness_counters(tess, elev);
    slopes.sort_by(f32::total_cmp);
    let pct = |p: f32| slopes[(((slopes.len() - 1) as f32) * p) as usize];
    log::info!(
        "erosion roughness [{}]: land {} | slope(elev/km) p50={:.3e} p90={:.3e} p99={:.3e} max={:.3e} | Moran's I {:.3}",
        label,
        land,
        pct(0.50),
        pct(0.90),
        pct(0.99),
        pct(1.0),
        tess.morans_i(elev),
    );
    log::info!(
        "erosion artifacts [{}]: pit {:.3}% peak {:.3}% | checkerboard {:.2}% (curv-rms {:.3e} elev) | aspect R={:.3} entropy={:.3} (n={})",
        label,
        c.pit_pct,
        c.peak_pct,
        c.checkerboard_pct,
        c.curv_rms,
        c.aspect_r,
        c.aspect_entropy,
        c.aspect_n,
    );
    log::info!(
        "erosion roughness [{}]: land-land edges {} | min spacing {:.4} km | <100m {} ({:.4}%) | <1m {}",
        label,
        edges_total,
        min_d_km,
        edges_sub_100m,
        100.0 * edges_sub_100m as f32 / edges_total.max(1) as f32,
        edges_sub_1m,
    );
    elevs.sort_by(f32::total_cmp);
    let epct = |p: f32| elevs[(((elevs.len() - 1) as f32) * p) as usize];
    let emean = elevs.iter().map(|&e| e as f64).sum::<f64>() / elevs.len() as f64;
    log::info!(
        "erosion height [{}]: land-elev mean={:.4} p50={:.4} p90={:.4} p99={:.4} max={:.4} | land-volume(elev*sr)={:.4e}",
        label,
        emean,
        epct(0.50),
        epct(0.90),
        epct(0.99),
        epct(1.0),
        land_volume,
    );
}

/// Plains alluvial regime gate (docs/specs/erosion-valleys-not-channels.md).
/// Detachment-vs-transport confinement factor in [0,1] from the channel slope
/// (elev-units per km): `C = smoothstep(0, confinement_slope, slope)`. 1 =
/// bedrock/confined (full stream-power incision); 0 = alluvial plain (no incision
/// — the deposition pass grades the lowland to a floodplain instead of erosion
/// gouging a cell-wide ditch). `confinement_slope <= 0` disables the gate (C=1).
#[inline]
fn confinement(slope_km: f32, confinement_slope: f32) -> f32 {
    if confinement_slope <= 0.0 {
        return 1.0;
    }
    let t = (slope_km / confinement_slope).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// One implicit Braun & Willett incision sweep (stream power, n = 1) in
/// ELEVATION space. `order` is downstream-first (each cell's receiver appears
/// before it), so the receiver already holds its updated height when a cell is
/// processed — the linear implicit solve each cell needs. Sinks are fixed base
/// level and are skipped. Pure: no sea-level clamp, no thickness coupling.
///
/// Detachment-limited incision only acts on a positive (downhill) slope. The
/// receiver is taken from the PIT-FILLED surface, so inside filled basins/flats
/// it points up toward the spill rim; without this guard the implicit step
/// would pull such cells UP toward a higher receiver (anti-erosion), pimpling
/// basin interiors with cell-scale bumps. Where the slope is non-positive there
/// is no stream incision (a transport/lake regime handled elsewhere), so skip.
/// Incision is scaled by the [`confinement`] regime gate (alluvial plains fade).
#[allow(clippy::too_many_arguments)]
fn incise_step(
    elev: &mut [f32],
    receiver: &[usize],
    is_sink: &[bool],
    dist: &[f32],
    area: &[f32],
    area_pow: &[f32],
    area_crit: f32,
    erodibility: &[f32],
    order: &[usize],
    k: f32,
    dt: f32,
    n: f32,
    confinement_slope: f32,
) {
    let linear_n = (n - 1.0).abs() < 1e-6;
    for &cell in order {
        if is_sink[cell] {
            continue;
        }
        if area[cell] < area_crit {
            continue; // below channel initiation -> hillslope (diffusion only)
        }
        let r = receiver[cell];
        let hr = elev[r];
        if hr >= elev[cell] {
            continue; // no downhill gradient -> no detachment-limited incision
        }
        let d = dist[cell].max(1e-12);
        // Regime gate: fade incision toward 0 on gentle (alluvial) channels.
        let slope_km = (elev[cell] - hr) / (d * PLANET_RADIUS_KM);
        let c = confinement(slope_km, confinement_slope);
        if c <= 0.0 {
            continue; // fully alluvial -> deposition handles it
        }
        // h_i = (h_i + dt K C A^m h_rcv / d) / (1 + dt K C A^m / d), K per-cell
        // (lithologic erodibility), C the confinement gate.
        let f = dt * k * c * erodibility[cell] * area_pow[cell] / d;
        if linear_n {
            elev[cell] = (elev[cell] + f * hr) / (1.0 + f);
        } else {
            // n != 1: detachment-limited stream power E = K C A^m S^n is NONLINEAR in
            // the new height, so the implicit step has no closed form. Newton-solve
            // x = h_old - a·(x − hr)^n  with  a = f / d^(n−1)  (so a·(x−hr)^n =
            // dt·K·C·A^m·((x−hr)/d)^n), over x ∈ [hr, h_old]. F is convex for n ≥ 1 and
            // F(h_old) > 0, so Newton from h_old descends monotonically to the root.
            let h_old = elev[cell];
            let a = f / d.powf(n - 1.0);
            let mut x = h_old;
            for _ in 0..EROSION_INCISION_NEWTON_ITERS {
                let s = (x - hr).max(0.0);
                let fx = x - h_old + a * s.powf(n);
                let dfx = 1.0 + a * n * s.powf(n - 1.0);
                let dx = fx / dfx;
                x -= dx;
                if x < hr {
                    x = hr;
                }
                if dx.abs() < 1e-10 {
                    break;
                }
            }
            elev[cell] = x;
        }
    }
}

/// Steepest-descent receivers over a priority-flood-filled surface, plus a
/// downstream-first processing order. Fixed sinks are cells below sea level and
/// cells flooded by a terminal (endorheic) pre-erosion lake.
struct Routing {
    /// receiver[i]; for sinks receiver[i] == i.
    receiver: Vec<usize>,
    /// Fixed base level (below sea level, or terminal-lake water); not
    /// incised/diffused.
    is_sink: Vec<bool>,
    /// Arc distance to receiver (radians on the unit sphere); 0 for sinks.
    dist: Vec<f32>,
    /// Downstream-first: every cell's receiver precedes it.
    order: Vec<usize>,
    /// Lowest elevation each cell may incise to: the base level of the terminal
    /// sink it drains into — 0 (sea level) for cells reaching the ocean, the lake
    /// surface for cells draining into a terminal lake. Propagated upstream.
    base_floor: Vec<f32>,
    /// Multiple-flow-direction flow DAG (Rung 2/3): downslope flow split + a
    /// topological order, shared by MFD drainage-area accumulation and MFD-DAG
    /// incision. `None` => single-flow (SFD) via `receiver`/`order` above.
    mfd: Option<MfdFlow>,
}

/// Multiple-flow-direction flow DAG over the pit-filled surface. Each non-sink
/// cell sends flow to all filled-downslope neighbours, split by `slope^p`
/// (normalised to sum 1); flats with no downslope neighbour carry their single
/// convergent SFD receiver. `order` is a topological order of the downhill DAG —
/// sources-first (filled descending, flats by mask) — so accumulation runs forward
/// (a cell finalised after its donors) and the implicit MFD incision runs in
/// reverse (every receiver solved before the cell). CSR-packed (aligned `nbr`/`w`/
/// `d`, sliced by `off`) to avoid per-cell allocation.
struct MfdFlow {
    order: Vec<usize>,
    off: Vec<usize>,
    nbr: Vec<usize>,
    w: Vec<f32>,
    d: Vec<f32>,
}

/// Build the [`MfdFlow`] DAG. `recv_dist[i]` is the SFD receiver distance (used for
/// the single flat-cell edge). Weights are `slope^exponent` normalised per cell.
fn build_mfd_flow(
    filled: &[f32],
    is_flat: &[bool],
    flat_mask: &[i64],
    is_sink: &[bool],
    geom: &NeighborGeometry,
    exponent: f32,
) -> MfdFlow {
    let n = filled.len();

    // Sources-first topological order: filled descending, then flat-mask descending
    // (a flat cell precedes those it drains into; non-flat "drain" cells sort last
    // within their filled level via MIN). Index tiebreak keeps it deterministic.
    let key_mask = |i: usize| -> i64 {
        if is_flat[i] {
            flat_mask[i]
        } else {
            i64::MIN
        }
    };
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        filled[b]
            .total_cmp(&filled[a])
            .then_with(|| key_mask(b).cmp(&key_mask(a)))
            .then(a.cmp(&b))
    });

    let mut off = vec![0usize; n + 1];
    let mut nbr: Vec<usize> = Vec::new();
    let mut w: Vec<f32> = Vec::new();
    let mut d: Vec<f32> = Vec::new();
    for i in 0..n {
        off[i] = nbr.len();
        if is_sink[i] {
            continue;
        }
        let fi = filled[i];
        let nbs = geom.tess_neighbors(i);
        let ds = geom.dists(i);
        let start = nbr.len();
        let mut total = 0.0f32;
        for (k, &nb) in nbs.iter().enumerate() {
            if filled[nb] < fi {
                let slope = (fi - filled[nb]) / ds[k].max(1e-12);
                let wk = slope.powf(exponent);
                nbr.push(nb);
                w.push(wk);
                d.push(ds[k]);
                total += wk;
            }
        }
        if nbr.len() > start {
            if total > 0.0 {
                for x in &mut w[start..] {
                    *x /= total;
                }
            }
        } else {
            // Flat: route by the Barnes mask, to the lowest-key same-level
            // neighbour — a non-flat drain (outlet or same-level sink, key MIN) or
            // the lowest-mask flat neighbour. This makes the flat edge consistent
            // with `order` (always strictly-lower key, since BFS guarantees a
            // lower-mask neighbour for non-outlet flats) INDEPENDENT of the SFD
            // `flat_resolution` toggle — so the DAG stays a valid topological order
            // even when that toggle is off. A stranded flat (no same-level
            // neighbour) contributes no edge.
            let mut best_k: Option<usize> = None;
            let mut best_key = i64::MAX;
            for (k, &nb) in nbs.iter().enumerate() {
                if filled[nb] != fi {
                    continue;
                }
                let key = if is_flat[nb] { flat_mask[nb] } else { i64::MIN };
                if key < best_key {
                    best_key = key;
                    best_k = Some(k);
                }
            }
            if let Some(k) = best_k {
                nbr.push(nbs[k]);
                w.push(1.0);
                d.push(ds[k].max(1e-12));
            }
        }
    }
    off[n] = nbr.len();
    MfdFlow {
        order,
        off,
        nbr,
        w,
        d,
    }
}

/// Barnes-style convergent flat resolution (Barnes, Lehman & Mulla 2014, *Computers
/// & Geosciences* 62:128–135). On a priority-flood-filled surface the interior of a
/// filled flat has no descending neighbour, so the bare priority-flood wavefront
/// (`flood_parent`) drains it in parallel/spiral paths that then get incised as
/// spiral grooves. This builds a synthetic descent over each flat from two integer
/// BFS distance fields — `to_low` (hops to the flat's outlets) and `from_high`
/// (hops from the surrounding higher walls) — combined as `mask = 2*to_low -
/// from_high`. Steepest descent on `mask` flows toward outlets (down `to_low`) and
/// away from walls (the `from_high` term): convergent drainage, not a spiral.
///
/// Returns `(is_flat, mask)`. `is_flat[i]` marks non-sink cells with no strictly
/// lower filled neighbour (the cells that otherwise fall back to `flood_parent`);
/// `mask` is meaningful only there. The `to_low` weight 2 > `from_high` weight 1
/// guarantees every non-outlet flat cell has a lower-`mask` neighbour, i.e. a
/// descending path to an outlet exists.
fn resolve_flats(
    filled: &[f32],
    geom: &NeighborGeometry,
    is_sink: &[bool],
) -> (Vec<bool>, Vec<i64>) {
    let n = geom.num_cells();

    // Flat = non-sink with no strictly-lower filled neighbour (would otherwise use
    // flood_parent). Such cells touch lower terrain only through a same-level
    // outlet. Per-cell, read-only over shared state -> parallel.
    let flat_of = |i: usize| -> bool {
        !is_sink[i]
            && !geom
                .tess_neighbors(i)
                .iter()
                .any(|&nb| filled[nb] < filled[i])
    };
    #[cfg(not(feature = "single-threaded"))]
    let is_flat: Vec<bool> = (0..n).into_par_iter().map(flat_of).collect();
    #[cfg(feature = "single-threaded")]
    let is_flat: Vec<bool> = (0..n).map(flat_of).collect();

    // Multi-source BFS over the flat subgraph (edges between equal-`filled` flat
    // cells). `to_low` seeds: flat cells adjacent to a same-level outlet (a
    // non-flat, non-sink cell at the same filled level, which drains lower).
    // `from_high` seeds: flat cells adjacent to a strictly-higher filled cell.
    let mut to_low = vec![u32::MAX; n];
    let mut from_high = vec![u32::MAX; n];
    let mut queue: VecDeque<usize> = VecDeque::new();

    for i in 0..n {
        if is_flat[i]
            && geom
                .tess_neighbors(i)
                .iter()
                .any(|&nb| filled[nb] == filled[i] && !is_flat[nb] && !is_sink[nb])
        {
            to_low[i] = 0;
            queue.push_back(i);
        }
    }
    bfs_flat(&mut to_low, &mut queue, &is_flat, filled, geom);

    for i in 0..n {
        if is_flat[i]
            && geom
                .tess_neighbors(i)
                .iter()
                .any(|&nb| filled[nb] > filled[i])
        {
            from_high[i] = 0;
            queue.push_back(i);
        }
    }
    bfs_flat(&mut from_high, &mut queue, &is_flat, filled, geom);

    let mask_of = |i: usize| -> i64 {
        if !is_flat[i] {
            return 0;
        }
        // Unreached (no outlet/wall in this flat) -> 0, a benign neutral level.
        let l = if to_low[i] == u32::MAX {
            0
        } else {
            to_low[i] as i64
        };
        let h = if from_high[i] == u32::MAX {
            0
        } else {
            from_high[i] as i64
        };
        2 * l - h
    };
    #[cfg(not(feature = "single-threaded"))]
    let mask: Vec<i64> = (0..n).into_par_iter().map(mask_of).collect();
    #[cfg(feature = "single-threaded")]
    let mask: Vec<i64> = (0..n).map(mask_of).collect();

    // Engagement / coverage signal (the global roughness counters are blind to the
    // spiral — it is a local pattern in filled basins). `flats` is how much of the
    // map the flat artifact even covers; `stranded` is flat cells the BFS reached
    // from neither an outlet nor a wall (they fall back to flood_parent).
    if log::log_enabled!(log::Level::Debug) {
        let flats = is_flat.iter().filter(|&&f| f).count();
        let stranded = (0..n)
            .filter(|&i| is_flat[i] && to_low[i] == u32::MAX && from_high[i] == u32::MAX)
            .count();
        log::debug!(
            "flat resolution: {} flat cells ({:.2}% of mesh), {} stranded (flood_parent fallback)",
            flats,
            100.0 * flats as f32 / n as f32,
            stranded,
        );
    }
    (is_flat, mask)
}

/// Multi-source BFS over the flat subgraph: relaxes `dist` across edges between
/// flat cells at the same `filled` level, starting from the seeds already enqueued.
fn bfs_flat(
    dist: &mut [u32],
    queue: &mut VecDeque<usize>,
    is_flat: &[bool],
    filled: &[f32],
    geom: &NeighborGeometry,
) {
    while let Some(c) = queue.pop_front() {
        let d = dist[c];
        for &nb in geom.tess_neighbors(c) {
            if is_flat[nb] && filled[nb] == filled[c] && dist[nb] == u32::MAX {
                dist[nb] = d + 1;
                queue.push_back(nb);
            }
        }
    }
}

impl Routing {
    /// `lake_base[i]` is the surface elevation of the terminal lake flooding cell
    /// `i` (finite => that cell is a lake sink), or NEG_INFINITY for non-lake
    /// cells. All-NEG_INFINITY reproduces the sea-level-only behaviour.
    fn build(
        elev: &[f32],
        geom: &NeighborGeometry,
        lake_base: &[f32],
        flat_resolution: bool,
        mfd_exponent: f32,
    ) -> Option<Routing> {
        let n = geom.num_cells();
        let is_sink: Vec<bool> = (0..n)
            .map(|i| elev[i] < 0.0 || lake_base[i].is_finite())
            .collect();
        if !is_sink.iter().any(|&s| s) {
            return None;
        }

        // Priority-flood fill (Barnes 2014, two-queue variant): every land cell
        // ends up able to drain to a sink. A cell whose neighbour fills it to the
        // current water level cannot introduce a lower level later, so it goes to
        // a plain FIFO (`pit`) instead of the heap (`open`); only cells that sit
        // strictly above the current level need the heap's ordering. This keeps
        // most of the wavefront off the O(log n) heap. The `filled` surface is the
        // per-cell minimal spill elevation — a path-max-min that is independent of
        // processing order, so it is byte-identical to the single-heap variant.
        let mut filled = elev.to_vec();
        let mut processed = vec![false; n];
        let mut open: std::collections::BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>> =
            std::collections::BinaryHeap::new();
        let mut pit: VecDeque<usize> = VecDeque::new();
        for i in 0..n {
            if is_sink[i] {
                processed[i] = true;
                open.push(Reverse((OrderedFloat(elev[i]), i)));
            }
        }
        while !open.is_empty() || !pit.is_empty() {
            // Drain the plain queue while its front is no higher than the heap top
            // (the pit holds cells at-or-below the current level, FIFO-ordered so
            // their levels are non-decreasing); otherwise take the lowest heap cell.
            let cell = match pit.front() {
                Some(&front)
                    if open
                        .peek()
                        .is_none_or(|Reverse((top, _))| filled[front] <= top.0) =>
                {
                    pit.pop_front().unwrap()
                }
                _ => open.pop().unwrap().0 .1,
            };
            let level = filled[cell];
            for &nb in geom.tess_neighbors(cell) {
                if processed[nb] {
                    continue;
                }
                processed[nb] = true;
                if elev[nb] <= level {
                    filled[nb] = level;
                    pit.push_back(nb);
                } else {
                    filled[nb] = elev[nb];
                    open.push(Reverse((OrderedFloat(elev[nb]), nb)));
                }
            }
        }

        // Descent direction across filled flats/pits. The single-heap flood set
        // `flood_parent[i]` to the first neighbour to reach `i`; on an ascending
        // priority flood that is exactly the neighbour with the smallest
        // (filled, index), so recompute it directly (order-independent, parallel)
        // rather than threading parent pointers through the two-queue wavefront.
        let fp_of = |i: usize| -> usize {
            if is_sink[i] {
                return i;
            }
            let mut best = i;
            let mut best_key = (OrderedFloat(f32::INFINITY), usize::MAX);
            for &nb in geom.tess_neighbors(i) {
                let key = (OrderedFloat(filled[nb]), nb);
                if key < best_key {
                    best_key = key;
                    best = nb;
                }
            }
            best
        };
        #[cfg(not(feature = "single-threaded"))]
        let flood_parent: Vec<usize> = (0..n).into_par_iter().map(fp_of).collect();
        #[cfg(feature = "single-threaded")]
        let flood_parent: Vec<usize> = (0..n).map(fp_of).collect();

        // Barnes convergent flat resolution (Rung 1): a synthetic descent over
        // filled flats that routes toward outlets and away from higher walls,
        // replacing the spiral-prone `flood_parent` wavefront. Always computed
        // (MFD accumulation needs the flat flag + mask to order across flats);
        // `flat_resolution` only gates whether `receiver_of` uses it below.
        let (is_flat, flat_mask) = resolve_flats(&filled, geom, &is_sink);

        // Receivers: steepest descent on the filled surface; on flats, convergent
        // flat resolution (or the flood-parent wavefront when disabled). Distance
        // is the chord to the receiver. Per-cell, read-only over shared state ->
        // parallel.
        let receiver_of = |i: usize| -> usize {
            if is_sink[i] {
                return i;
            }
            if !flat_resolution || !is_flat[i] {
                // Non-flat: steepest descent (a strictly-lower neighbour exists).
                // flood_parent covers the flat-resolution-off flat case.
                let mut best = flood_parent[i];
                let mut best_elev = filled[i];
                for &nb in geom.tess_neighbors(i) {
                    if filled[nb] < best_elev {
                        best_elev = filled[nb];
                        best = nb;
                    }
                }
                best
            } else {
                // Flat: route to an adjacent same-level DRAIN (a non-flat cell at
                // the same filled level — an outlet that drains lower, or a fixed
                // lake/sea sink) if present, else to the lowest-`mask` flat
                // neighbour (toward outlets / away from walls). flood_parent is the
                // last resort.
                let mut best = flood_parent[i];
                let mut best_key = i64::MAX;
                for &nb in geom.tess_neighbors(i) {
                    if filled[nb] != filled[i] {
                        continue; // only same-level (flat) edges
                    }
                    let key = if !is_flat[nb] {
                        i64::MIN // same-level drain (outlet or sink): most attractive
                    } else {
                        flat_mask[nb]
                    };
                    if key < best_key {
                        best_key = key;
                        best = nb;
                    }
                }
                best
            }
        };
        #[cfg(not(feature = "single-threaded"))]
        let receiver: Vec<usize> = (0..n).into_par_iter().map(receiver_of).collect();
        #[cfg(feature = "single-threaded")]
        let receiver: Vec<usize> = (0..n).map(receiver_of).collect();

        let dist_of = |i: usize| -> f32 {
            if is_sink[i] || receiver[i] == i {
                0.0
            } else {
                geom.arc_to(i, receiver[i])
            }
        };
        #[cfg(not(feature = "single-threaded"))]
        let dist: Vec<f32> = (0..n).into_par_iter().map(dist_of).collect();
        #[cfg(feature = "single-threaded")]
        let dist: Vec<f32> = (0..n).map(dist_of).collect();

        // Downstream-first order via BFS from sinks over the donor graph, stored
        // CSR-style (counting sort) to avoid n small Vec allocations per step.
        let mut donor_count = vec![0u32; n];
        for i in 0..n {
            if !is_sink[i] && receiver[i] != i {
                donor_count[receiver[i]] += 1;
            }
        }
        let mut donor_off = vec![0usize; n + 1];
        for i in 0..n {
            donor_off[i + 1] = donor_off[i] + donor_count[i] as usize;
        }
        let mut cursor = donor_off.clone();
        let mut donors = vec![0usize; donor_off[n]];
        for i in 0..n {
            if !is_sink[i] && receiver[i] != i {
                let r = receiver[i];
                donors[cursor[r]] = i;
                cursor[r] += 1;
            }
        }
        let mut order = Vec::with_capacity(n);
        let mut queue: VecDeque<usize> = (0..n).filter(|&i| is_sink[i]).collect();
        while let Some(cell) = queue.pop_front() {
            order.push(cell);
            for &d in &donors[donor_off[cell]..donor_off[cell + 1]] {
                queue.push_back(d);
            }
        }

        // Base level per cell: a sink's own datum (lake surface, else sea level),
        // inherited by every cell that drains into it. `order` is downstream-first
        // (a cell's receiver precedes it), so one forward pass propagates it.
        let mut base_floor = vec![0.0f32; n];
        for i in 0..n {
            if is_sink[i] && lake_base[i].is_finite() {
                base_floor[i] = lake_base[i];
            }
        }
        for &cell in &order {
            if !is_sink[cell] {
                base_floor[cell] = base_floor[receiver[cell]];
            }
        }

        // MFD flow DAG (Rung 2/3), when enabled. Built on the filled surface +
        // the convergent flat structure; reused by accumulation and incision.
        let mfd = if mfd_exponent > 0.0 {
            Some(build_mfd_flow(
                &filled,
                &is_flat,
                &flat_mask,
                &is_sink,
                geom,
                mfd_exponent,
            ))
        } else {
            None
        };

        Some(Routing {
            receiver,
            is_sink,
            dist,
            order,
            base_floor,
            mfd,
        })
    }
}

/// Precipitation-weighted drainage area A_i (units: rainfall x steradian).
/// Each cell seeds `precip * area`; flow accumulates upstream-first (reverse of
/// the downstream-first order).
fn accumulate_wet_area(routing: &Routing, precipitation: &[f32], areas: &[f32]) -> Vec<f32> {
    let n = areas.len();
    let mut acc: Vec<f32> = (0..n)
        .map(|i| precipitation[i].max(0.0) * areas[i])
        .collect();
    for &cell in routing.order.iter().rev() {
        if routing.is_sink[cell] {
            continue;
        }
        let r = routing.receiver[cell];
        acc[r] += acc[cell];
    }
    acc
}

/// Multiple-flow-direction drainage area (Rung 2): accumulate precip-weighted area
/// down the [`MfdFlow`] DAG. Each cell, finalised after its donors (sources-first
/// `order`), splits its flow to all downslope neighbours by the precomputed weights
/// — so discharge disperses on hillslopes and reconverges in valleys instead of
/// running in one SFD wire. Serial + deterministic (fixed order) for reproducible
/// float accumulation.
fn accumulate_wet_area_mfd(
    routing: &Routing,
    mfd: &MfdFlow,
    precipitation: &[f32],
    areas: &[f32],
) -> Vec<f32> {
    let n = areas.len();
    let mut acc: Vec<f32> = (0..n)
        .map(|i| precipitation[i].max(0.0) * areas[i])
        .collect();
    for &i in &mfd.order {
        if routing.is_sink[i] {
            continue;
        }
        let ai = acc[i];
        for k in mfd.off[i]..mfd.off[i + 1] {
            acc[mfd.nbr[k]] += ai * mfd.w[k];
        }
    }
    acc
}

/// MFD-DAG implicit incision (Rung 3). Generalises [`incise_step`] from one receiver
/// to the weighted multi-receiver stream-power solve (n = 1) over the downhill DAG.
/// Braun & Willett needs a downstream ORDERING, not a tree: processing `mfd.order`
/// in REVERSE (receivers-first) means every downstream neighbour is already solved,
/// so the per-cell update stays local and the whole sweep is O(edges):
///
/// ```text
/// h_i = (h_i + Σ_j c_ij h_j) / (1 + Σ_j c_ij),  c_ij = dt K A_i^m w_ij / d_ij
/// ```
///
/// over the cell's downslope neighbours `j`. Distributing the carving (not just the
/// drainage area) is what breaks the 1-cell SFD channel/ridge striping. The
/// per-receiver `elev[j] < elev[i]` guard keeps it erosive as the surface evolves
/// between re-routes (a stale receiver that has gone non-downhill is skipped, never
/// pulling the cell up). Pure: no sea-level clamp / thickness coupling (folded in by
/// the caller, as for SFD).
#[allow(clippy::too_many_arguments)]
fn incise_step_mfd(
    elev: &mut [f32],
    mfd: &MfdFlow,
    is_sink: &[bool],
    area: &[f32],
    area_pow: &[f32],
    area_crit: f32,
    erodibility: &[f32],
    k: f32,
    dt: f32,
    confinement_slope: f32,
) {
    for &cell in mfd.order.iter().rev() {
        if is_sink[cell] {
            continue;
        }
        if area[cell] < area_crit {
            continue; // below channel initiation -> hillslope (diffusion only)
        }
        let hi = elev[cell];
        // Regime gate from the steepest downslope channel (alluvial plains fade).
        let steepest_km = {
            let mut s = 0.0f32;
            for kk in mfd.off[cell]..mfd.off[cell + 1] {
                let drop = hi - elev[mfd.nbr[kk]];
                if drop > 0.0 {
                    s = s.max(drop / (mfd.d[kk].max(1e-12) * PLANET_RADIUS_KM));
                }
            }
            s
        };
        let cgate = confinement(steepest_km, confinement_slope);
        if cgate <= 0.0 {
            continue; // fully alluvial -> deposition handles it
        }
        let f_base = dt * k * cgate * erodibility[cell] * area_pow[cell];
        let mut num = hi;
        let mut den = 1.0f32;
        for kk in mfd.off[cell]..mfd.off[cell + 1] {
            let j = mfd.nbr[kk];
            let hj = elev[j];
            if hj < hi {
                let c = f_base * mfd.w[kk] / mfd.d[kk].max(1e-12);
                num += c * hj;
                den += c;
            }
        }
        if den > 1.0 {
            elev[cell] = num / den;
        }
    }
}

/// Maximum slope ratio `S/S_c` the Roering conductivity is evaluated at. Clamping
/// the ratio below 1 regularizes the `1/(1-(S/S_c)²)` singularity so the flux is
/// bounded (here κ ≤ `1/(1-0.95²)` ≈ 10.3) — the explicit "denominator → 0
/// cliff" the escalation trap warns about cannot fire. Slopes can still exceed
/// S_c (they just transport ~10× faster), so the world never hard-facets.
const ROERING_RATIO_MAX: f32 = 0.95;

/// Hillslope diffusion on land cells, solved IMPLICITLY (backward Euler) with
/// Jacobi sweeps. Edges to sinks are no-flux boundaries (no erosion into the
/// ocean). Implicit is unconditionally stable, so the finest sliver cells can't
/// force a substep blow-up the way an explicit CFL scheme would.
///
/// Per cell the backward-Euler update is
///   (1 + c_i) h_i^{new} = h_i^{old} + dt D / area_i * sum_j W_ij h_j^{new}
/// with c_i = dt D / area_i * sum_j W_ij over land neighbours j. We approximate
/// the coupled solve with a few Jacobi iterations (RHS h_old fixed).
///
/// `critical_slope` (S_c, Δelev/radian) selects the flux law via the edge
/// conductivity `W_ij = κ_ij · w_ij`:
/// - `0` → κ ≡ 1: linear creep `q = -D∇h` (the original behaviour).
/// - `>0` → Roering nonlinear `κ_ij = 1/(1-(S_ij/S_c)²)`, with `S_ij` the edge
///   slope from the lagged (pre-diffusion) surface `h_old`, regularized by
///   `ROERING_RATIO_MAX`. Diffusivity blows up toward S_c → planar slopes, crisp
///   ridges. κ is symmetric (`S_ij`, `w_ij` both symmetric) so the scheme stays
///   conservative; frozen at `h_old` it is a semi-implicit (lagged-conductivity)
///   linearization, and the diagonal stays strictly dominant so Jacobi converges
///   at any κ magnitude.
#[allow(clippy::too_many_arguments)]
fn diffuse_land(
    elev: &mut [f32],
    routing: &Routing,
    geom: &NeighborGeometry,
    areas: &[f32],
    w_land: Option<&[f32]>,
    dt: f32,
    diffusivity: f32,
    diffusion_iters: usize,
    critical_slope: f32,
) {
    if diffusivity <= 0.0 {
        return;
    }
    let n = elev.len();
    let dd = diffusivity * dt;

    // Lagged surface: the Roering edge conductivity is frozen at the slopes the
    // diffusion step starts from (semi-implicit). Also the fixed Jacobi RHS.
    let h_old = elev.to_vec();

    // Effective edge conductivity W_ij = κ_ij · w_ij for the edge from cell i to
    // its k-th neighbour nb. κ = 1 (linear) when critical_slope <= 0; otherwise
    // the regularized Roering multiplier from the lagged edge slope.
    let nonlinear = critical_slope > 0.0;
    let weff = |i: usize, k: usize, nb: usize| -> f32 {
        let w = geom.weight(i, k);
        if !nonlinear {
            return w;
        }
        let d = geom.dists(i)[k].max(1e-12);
        let s = (h_old[i] - h_old[nb]).abs() / d;
        let r = (s / critical_slope).min(ROERING_RATIO_MAX);
        w / (1.0 - r * r)
    };

    // Per-cell f_i = dt D / area_i and diagonal denominator (1 + c_i), counting
    // only land neighbours (sink edges are no-flux). Constant across sweeps. Each
    // cell is independent (read-only over shared state) -> parallel; the unzip
    // preserves index order, so f/denom are bit-identical to the serial build.
    let prep = |i: usize| -> (f32, f32) {
        if routing.is_sink[i] {
            return (0.0, 1.0);
        }
        let fi = dd / areas[i].max(1e-12);
        let mut wsum = 0.0f32;
        if let Some(wl) = w_land {
            // Linear: precomputed weights with sink edges already zeroed, so the
            // `+ 0.0` from a sink edge leaves the sum bit-identical to skipping it.
            let off = geom.edge_offset(i);
            for k in 0..geom.tess_neighbors(i).len() {
                wsum += wl[off + k];
            }
        } else {
            for (k, &nb) in geom.tess_neighbors(i).iter().enumerate() {
                if !routing.is_sink[nb] {
                    wsum += weff(i, k, nb);
                }
            }
        }
        (fi, 1.0 + fi * wsum)
    };
    #[cfg(not(feature = "single-threaded"))]
    let (f, denom): (Vec<f32>, Vec<f32>) = (0..n).into_par_iter().map(prep).unzip();
    #[cfg(feature = "single-threaded")]
    let (f, denom): (Vec<f32>, Vec<f32>) = (0..n).map(prep).unzip();

    let mut cur = h_old.clone();
    let mut next = vec![0.0f32; n];
    for _ in 0..diffusion_iters {
        let sweep = |i: usize| -> f32 {
            if routing.is_sink[i] {
                return cur[i];
            }
            let mut acc = 0.0f32;
            if let Some(wl) = w_land {
                // Linear: sink edges carry weight 0, so the per-edge `is_sink`
                // lookup is gone and the `0.0 * cur[nb]` terms are inert.
                let off = geom.edge_offset(i);
                for (k, &nb) in geom.tess_neighbors(i).iter().enumerate() {
                    acc += wl[off + k] * cur[nb];
                }
            } else {
                for (k, &nb) in geom.tess_neighbors(i).iter().enumerate() {
                    if !routing.is_sink[nb] {
                        acc += weff(i, k, nb) * cur[nb];
                    }
                }
            }
            // Clamp to the cell's LOCAL base level (sea level, or a terminal-lake
            // surface), not a hard sea level. Incision grades cells down to
            // `base_floor`; clamping diffusion only at 0.0 could push a cell graded
            // to a lake surface L>0 below L, undoing that grade.
            ((h_old[i] + f[i] * acc) / denom[i]).max(routing.base_floor[i])
        };
        // Sweep into the reused `next` buffer then ping-pong — identical Jacobi
        // math to collecting a fresh Vec each iter (every cell reads the previous
        // `cur`, writes a disjoint index), without the per-iteration allocation.
        #[cfg(not(feature = "single-threaded"))]
        {
            next.par_iter_mut()
                .enumerate()
                .for_each(|(i, slot)| *slot = sweep(i));
        }
        #[cfg(feature = "single-threaded")]
        {
            for (i, slot) in next.iter_mut().enumerate() {
                *slot = sweep(i);
            }
        }
        std::mem::swap(&mut cur, &mut next);
    }
    elev.copy_from_slice(&cur);
}

/// Smooth the tectonic uplift SOURCE in place over a physical length scale
/// (`smooth_km`), once, before the erosion step loop (escalation #1). Mechanism:
/// a conservative finite-volume graph diffusion of `u_thick` restricted to the
/// LAND mask (`base >= 0`), with no-flux at the coastline — so uplift never
/// spreads onto submerged cells (which would manufacture spurious land) and the
/// area-weighted integral `Σ area_i · u_thick_i` over land is preserved exactly
/// (the edge fluxes `w_ij·(u_j − u_i)` are antisymmetric; `w_ij = w_ji`).
///
/// The diffusion is linear, so smoothing the combined source is identical to
/// smoothing arc/collision/rift_delta separately and recombining — the signed
/// rift thinning superposes correctly with the positive orogen uplift (no
/// magnitude blur). `smooth_km` is the Gaussian std-dev of the kernel: for 2-D
/// isotropic diffusion the per-axis variance after "time" τ is `2·D·τ`, so we
/// target a diffusion product `D·τ = L²/2` with `L = smooth_km / R` in the
/// chord/radian units the geometry uses. The product is integrated with enough
/// explicit sub-steps to stay below the FV stability limit.
fn smooth_uplift_source(
    u_thick: &mut [f32],
    geom: &NeighborGeometry,
    areas: &[f32],
    base: &[f32],
    smooth_km: f32,
) {
    let n = u_thick.len();
    let land = |i: usize| base[i] >= 0.0;

    // Target diffusion product D·τ = L²/2 with L the smoothing length in the
    // geometry's units (chord ≈ radian on the unit sphere). Reject non-finite /
    // non-positive lengths (a NaN/inf knob would otherwise blow up the sub-step
    // count below).
    if !smooth_km.is_finite() || smooth_km <= 0.0 {
        return;
    }
    let l = smooth_km / PLANET_RADIUS_KM;
    let g_total = 0.5 * l * l;
    if g_total <= 0.0 {
        return;
    }

    // Lumped per-cell coefficient c_i = (1/area_i) Σ_{j land} w_ij (units 1/len²),
    // counting only land neighbours (no-flux at the coast). Explicit FV stability
    // wants g_sub · c_i ≤ ~0.5; pick the sub-step count from the worst cell.
    let mut c = vec![0.0f32; n];
    let mut c_max = 0.0f32;
    for i in 0..n {
        if !land(i) {
            continue;
        }
        let mut wsum = 0.0f32;
        for (k, &nb) in geom.tess_neighbors(i).iter().enumerate() {
            if land(nb) {
                wsum += geom.weight(i, k);
            }
        }
        let ci = wsum / areas[i].max(1e-12);
        c[i] = ci;
        c_max = c_max.max(ci);
    }
    if c_max <= 0.0 {
        return;
    }
    // Sub-step count from the worst cell's stability bound (g_sub·c_max ≤ 0.4).
    // Capped so an absurdly large length (typo, or calibration overshoot) can't
    // silently spin millions of full-mesh sweeps: above the cap we keep the
    // stable per-step amount and under-smooth (logged), rather than hang. At the
    // cap the achieved length is ~sqrt(2·N_SUB_MAX·0.4/c_max).
    const N_SUB_MAX: usize = 20_000;
    let g_sub_stable = 0.4 / c_max;
    let n_sub_ideal = (g_total / g_sub_stable).ceil() as usize;
    let n_sub = n_sub_ideal.clamp(1, N_SUB_MAX);
    // Never exceed the stable step: if capped, g_sub == g_sub_stable (under-smooth).
    let g_sub = (g_total / n_sub as f32).min(g_sub_stable);
    if n_sub_ideal > N_SUB_MAX {
        let achieved_l = (2.0 * n_sub as f32 * g_sub).sqrt() * PLANET_RADIUS_KM;
        log::warn!(
            "erosion: uplift smoothing {:.0} km exceeds the {} sub-step cap; \
             clamped to ~{:.0} km (raise N_SUB_MAX or lower the length)",
            smooth_km,
            N_SUB_MAX,
            achieved_l
        );
    }

    let mut cur = u_thick.to_vec();
    for _ in 0..n_sub {
        let sweep = |i: usize| -> f32 {
            if !land(i) {
                return cur[i];
            }
            let mut acc = 0.0f32;
            for (k, &nb) in geom.tess_neighbors(i).iter().enumerate() {
                if land(nb) {
                    acc += geom.weight(i, k) * (cur[nb] - cur[i]);
                }
            }
            cur[i] + g_sub / areas[i].max(1e-12) * acc
        };
        #[cfg(not(feature = "single-threaded"))]
        let next: Vec<f32> = (0..n).into_par_iter().map(sweep).collect();
        #[cfg(feature = "single-threaded")]
        let next: Vec<f32> = (0..n).map(sweep).collect();
        cur = next;
    }
    u_thick.copy_from_slice(&cur);
}

/// Transport-aware deposition. Routes eroded sediment downstream; along the way,
/// where the bed lies below the surface that grades to the receiver at the
/// depositional repose slope (low-gradient reaches: valley floors, alluvial fans
/// at range fronts, lake/coast margins), it aggrades the bed toward that surface.
/// This builds floodplains, fans, and deltas instead of dumping every catchment's
/// load at a single river mouth. Whatever still reaches a terminal sink fills it
/// toward the LOCAL base level (sea level, or a lake surface) up to
/// `deposit_fill_fraction`; the remainder is lost to the deep ocean. Returns
/// (deposited, lost) thickness-volumes.
///
/// `deposition_slope = 0` disables en-route aggradation (sink-fill only), which
/// reproduces the old behaviour generalized to the per-sink base level.
fn deposit(
    routing: &Routing,
    eroded_vol: &[f32],
    elev: &[f32],
    areas: &[f32],
    slope: f32,
    deposit_fill_fraction: f32,
    deposition_slope: f32,
    thick: &mut [f32],
) -> (f64, f64) {
    let n = eroded_vol.len();
    let mut sed = eroded_vol.to_vec();
    let mut deposited = 0.0f64;
    let mut lost = 0.0f64;

    // Upstream-first: every cell's donors have already routed their sediment into
    // it. Aggrade low-gradient reaches toward the repose-slope surface, then pass
    // the remainder to the receiver (processed later in this same pass).
    for &c in routing.order.iter().rev() {
        if routing.is_sink[c] {
            continue;
        }
        let r = routing.receiver[c];
        if sed[c] > 0.0 && deposition_slope > 0.0 {
            // Depositional surface grading to the receiver at the repose slope;
            // aggrade only where the current bed sits below it.
            let target = elev[r] + deposition_slope * routing.dist[c];
            if elev[c] < target {
                let room = (target - elev[c]) / slope * areas[c];
                let dep = sed[c].min(room.max(0.0));
                if dep > 0.0 {
                    thick[c] += dep / areas[c].max(1e-12);
                    deposited += dep as f64;
                    sed[c] -= dep;
                }
            }
        }
        sed[r] += sed[c];
    }

    // Terminal sinks: fill toward the local base level (sea level 0, or a terminal
    // lake's surface via base_floor) up to the fill fraction; lose the rest.
    for i in 0..n {
        if !routing.is_sink[i] || sed[i] <= 0.0 {
            continue;
        }
        let depth = (routing.base_floor[i] - elev[i]).max(0.0); // below local datum
        let cap = deposit_fill_fraction * depth / slope * areas[i];
        let take = sed[i].min(cap.max(0.0));
        if take > 0.0 {
            thick[i] += take / areas[i].max(1e-12);
            deposited += take as f64;
        }
        lost += (sed[i] - take).max(0.0) as f64;
    }
    (deposited, lost)
}

/// Per-cell neighbor geometry precomputed once: arc distance to each neighbor
/// and the finite-volume diffusion weight (shared edge length / center
/// distance). Aligned index-for-index with `tess.neighbors(i)`.
#[derive(Clone)]
pub(crate) struct NeighborGeometry {
    offsets: Vec<usize>,
    neighbors: Vec<usize>,
    dist: Vec<f32>,
    weight: Vec<f32>,
}

impl NeighborGeometry {
    pub(crate) fn build(tess: &Tessellation) -> Self {
        let n = tess.num_cells();
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        for i in 0..n {
            offsets.push(offsets[i] + tess.neighbors(i).len());
        }
        let total = *offsets.last().unwrap();

        let build_cell = |i: usize| -> (Vec<usize>, Vec<f32>, Vec<f32>) {
            let center = tess.cell_center(i);
            let nbs = tess.neighbors(i);
            let mut ns = Vec::with_capacity(nbs.len());
            let mut ds = Vec::with_capacity(nbs.len());
            let mut ws = Vec::with_capacity(nbs.len());
            for &nb in nbs {
                // Chord, not acos(dot): the latter collapses to 0 in f32 for the
                // sub-3km separations the fine mesh has (cos rounds to 1.0),
                // which would make incision (K A^m / d) and diffusion (edge / d)
                // blow up on the finest cells. Chord is accurate here and ~ arc.
                let d = (center - tess.cell_center(nb)).length();
                let edge = tess.shared_edge_length(i, nb);
                ns.push(nb);
                ds.push(d);
                ws.push(edge / d.max(1e-12));
            }
            (ns, ds, ws)
        };

        #[cfg(not(feature = "single-threaded"))]
        let per_cell: Vec<(Vec<usize>, Vec<f32>, Vec<f32>)> =
            (0..n).into_par_iter().map(build_cell).collect();
        #[cfg(feature = "single-threaded")]
        let per_cell: Vec<(Vec<usize>, Vec<f32>, Vec<f32>)> = (0..n).map(build_cell).collect();

        let mut neighbors = Vec::with_capacity(total);
        let mut dist = Vec::with_capacity(total);
        let mut weight = Vec::with_capacity(total);
        for (ns, ds, ws) in per_cell {
            neighbors.extend(ns);
            dist.extend(ds);
            weight.extend(ws);
        }

        Self {
            offsets,
            neighbors,
            dist,
            weight,
        }
    }

    fn num_cells(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Start of cell `i`'s edge slice in the flat CSR arrays — lets callers index
    /// an externally-built per-edge array (e.g. land weights) aligned with
    /// `tess_neighbors(i)`.
    fn edge_offset(&self, i: usize) -> usize {
        self.offsets[i]
    }

    /// Per-edge diffusion weights with sink-neighbour edges zeroed, aligned with
    /// the flat CSR (so `w[edge_offset(i)+k]` matches `tess_neighbors(i)[k]`).
    /// Lets the linear hillslope sweep drop the per-edge `is_sink[nb]` lookup: a
    /// zeroed sink edge contributes `0.0 * cur[nb]`, which leaves the running sum
    /// (and its term order) bit-identical to skipping the edge.
    fn land_weights(&self, is_sink: &[bool]) -> Vec<f32> {
        let zero = |(&nb, &w): (&usize, &f32)| if is_sink[nb] { 0.0 } else { w };
        #[cfg(not(feature = "single-threaded"))]
        {
            self.neighbors
                .par_iter()
                .zip(self.weight.par_iter())
                .map(zero)
                .collect()
        }
        #[cfg(feature = "single-threaded")]
        {
            self.neighbors
                .iter()
                .zip(self.weight.iter())
                .map(zero)
                .collect()
        }
    }

    fn tess_neighbors(&self, i: usize) -> &[usize] {
        &self.neighbors[self.offsets[i]..self.offsets[i + 1]]
    }

    /// Per-neighbor chord distances for cell `i`, aligned with `tess_neighbors(i)`.
    fn dists(&self, i: usize) -> &[f32] {
        &self.dist[self.offsets[i]..self.offsets[i + 1]]
    }

    fn weight(&self, i: usize, k: usize) -> f32 {
        self.weight[self.offsets[i] + k]
    }

    /// Arc distance from cell `i` to a specific neighbor `j` (linear scan over
    /// `i`'s short neighbor list).
    fn arc_to(&self, i: usize, j: usize) -> f32 {
        let start = self.offsets[i];
        let end = self.offsets[i + 1];
        for k in start..end {
            if self.neighbors[k] == j {
                return self.dist[k];
            }
        }
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1-D chain neighbour geometry (cell i adjacent to i±1) with unit edge
    /// lengths and distances, for testing the conservative source smoothing.
    fn chain_geom(n: usize) -> NeighborGeometry {
        let mut offsets = vec![0usize];
        let mut neighbors = Vec::new();
        let mut dist = Vec::new();
        let mut weight = Vec::new();
        for i in 0..n {
            if i > 0 {
                neighbors.push(i - 1);
                dist.push(1.0);
                weight.push(1.0);
            }
            if i + 1 < n {
                neighbors.push(i + 1);
                dist.push(1.0);
                weight.push(1.0);
            }
            offsets.push(neighbors.len());
        }
        NeighborGeometry {
            offsets,
            neighbors,
            dist,
            weight,
        }
    }

    /// A minimal all-land chain routing (no sinks, base level far below) for
    /// exercising `diffuse_land` in isolation.
    fn open_chain_routing(n: usize) -> Routing {
        Routing {
            receiver: (0..n).collect(),
            is_sink: vec![false; n],
            dist: vec![0.0; n],
            order: (0..n).collect(),
            base_floor: vec![-1.0e9; n],
            mfd: None,
        }
    }

    /// Roering nonlinear diffusion (escalation #2) transports steep slopes faster
    /// than linear creep — so a sharp step closes more in the same step — while
    /// staying mass-conserving (no sinks, no base-level clamp).
    #[test]
    fn roering_flattens_steep_step_faster_than_linear() {
        let n = 6usize;
        let geom = chain_geom(n);
        let areas = vec![1.0f32; n];
        let routing = open_chain_routing(n);
        // Flat-step-flat: only the middle edge (2->3) carries a slope of 1.0
        // (dist = 1 radian), so linear and Roering differ only there.
        let init = || vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let (dt, d, iters) = (1.0f32, 0.1f32, 30usize);

        let mut lin = init();
        // Exercise the cached-weights linear path (sink set is empty here).
        let w_land = geom.land_weights(&routing.is_sink);
        diffuse_land(
            &mut lin,
            &routing,
            &geom,
            &areas,
            Some(&w_land),
            dt,
            d,
            iters,
            0.0,
        );
        let mut roe = init();
        // S_c just above the step slope -> the middle edge sits near critical
        // (kappa ~10x), the flat edges stay ~linear. Roering recomputes per-step
        // conductivity, so it uses the weff path (None).
        diffuse_land(&mut roe, &routing, &geom, &areas, None, dt, d, iters, 1.05);

        let step_lin = lin[3] - lin[2];
        let step_roe = roe[3] - roe[2];
        assert!(
            step_roe < step_lin,
            "Roering should close the steep step faster: linear {step_lin}, roering {step_roe}"
        );

        let s0: f32 = init().iter().sum();
        let sl: f32 = lin.iter().sum();
        let sr: f32 = roe.iter().sum();
        assert!(
            (sl - s0).abs() < 1e-3 && (sr - s0).abs() < 1e-3,
            "diffusion must conserve mass: init {s0}, linear {sl}, roering {sr}"
        );
    }

    /// With the critical slope off (0), `diffuse_land` is bit-identical to the
    /// original linear scheme — escalation #2 is a no-op when disabled.
    #[test]
    fn roering_off_matches_linear() {
        let n = 6usize;
        let geom = chain_geom(n);
        let areas = vec![1.0f32; n];
        let routing = open_chain_routing(n);
        let init = vec![0.0, 0.2, 0.9, 0.3, 0.7, 0.1];

        let mut a = init.clone();
        diffuse_land(&mut a, &routing, &geom, &areas, None, 1.0, 0.1, 10, 0.0);
        // A huge S_c keeps every kappa ~1, so it must match linear closely.
        let mut b = init.clone();
        diffuse_land(&mut b, &routing, &geom, &areas, None, 1.0, 0.1, 10, 1.0e9);
        for i in 0..n {
            assert!(
                (a[i] - b[i]).abs() < 1e-6,
                "huge S_c should equal linear at cell {i}: {} vs {}",
                a[i],
                b[i]
            );
        }
    }

    /// Uplift-source smoothing (escalation #1) conserves the area-weighted integral
    /// of the source exactly and actually spreads an impulse (variance falls).
    #[test]
    fn uplift_smoothing_conserves_and_spreads() {
        let n = 9usize;
        let geom = chain_geom(n);
        let areas = vec![1.0f32; n];
        let base = vec![0.0f32; n]; // all land
        let mut u = vec![0.0f32; n];
        u[n / 2] = 1.0; // impulse at center

        let sum_before: f32 = u.iter().sum();
        let peak_before = u[n / 2];
        // Length is in km; the geometry's units are abstract here, so pick a km
        // that makes L = km/R = 1 (g_total = 0.5) — a moderate spread.
        smooth_uplift_source(&mut u, &geom, &areas, &base, PLANET_RADIUS_KM);
        let sum_after: f32 = u.iter().sum();

        assert!(
            (sum_after - sum_before).abs() < 1e-5,
            "integrated uplift not conserved: {sum_before} -> {sum_after}"
        );
        assert!(
            u[n / 2] < peak_before,
            "impulse peak did not spread: {peak_before} -> {}",
            u[n / 2]
        );
        assert!(
            u[n / 2 - 1] > 0.0 && u[n / 2 + 1] > 0.0,
            "neighbours not lifted"
        );
    }

    /// The land mask is no-flux: uplift never bleeds onto submerged cells, so the
    /// land integral is conserved without leaking into the ocean.
    #[test]
    fn uplift_smoothing_respects_land_mask() {
        let n = 6usize;
        let geom = chain_geom(n);
        let areas = vec![1.0f32; n];
        // Cells 0..3 land, 3..6 ocean (base < 0).
        let base = vec![0.0, 0.0, 0.0, -1.0, -1.0, -1.0];
        let mut u = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0];

        let land_sum_before: f32 = (0..3).map(|i| u[i]).sum();
        smooth_uplift_source(&mut u, &geom, &areas, &base, PLANET_RADIUS_KM);
        let land_sum_after: f32 = (0..3).map(|i| u[i]).sum();

        assert!(
            (land_sum_after - land_sum_before).abs() < 1e-5,
            "land uplift leaked across the coast: {land_sum_before} -> {land_sum_after}"
        );
        assert!(
            u[3] == 0.0 && u[4] == 0.0 && u[5] == 0.0,
            "uplift bled onto submerged cells: {:?}",
            &u[3..]
        );
    }

    /// A 1-D chain (cell i drains to i+1, last cell is fixed base level) relaxes
    /// to the stream-power steady state: adjacent drop = U d / (K A^m) with n=1.
    #[test]
    fn incision_reaches_analytic_steady_state() {
        let n = 20usize;
        let d = 1.0f32;
        let a = 1.0f32;
        let k = 0.1f32;
        let m = 1.0f32;
        let u = 0.01f32;
        let dt = 1.0f32;

        let receiver: Vec<usize> = (0..n).map(|i| (i + 1).min(n - 1)).collect();
        let mut is_sink = vec![false; n];
        is_sink[n - 1] = true; // outlet fixed at 0
        let dist = vec![d; n];
        let area = vec![a; n];
        // Downstream-first: outlet first, then upstream.
        let order: Vec<usize> = (0..n).rev().collect();

        let area_pow: Vec<f32> = area.iter().map(|&a| a.powf(m)).collect();
        let erod = vec![1.0f32; n];
        let mut elev = vec![0.0f32; n];
        for _ in 0..5000 {
            // Uplift interior cells in elevation space; outlet stays at 0.
            for (i, e) in elev.iter_mut().enumerate() {
                if !is_sink[i] {
                    *e += u * dt;
                }
            }
            incise_step(
                &mut elev, &receiver, &is_sink, &dist, &area, &area_pow, 0.0, &erod, &order, k, dt,
                1.0, 0.0,
            );
        }

        let expected_drop = u * d / (k * a.powf(m)); // 0.1
                                                     // Check an interior cell's drop to its receiver.
        let i = n / 2;
        let drop = elev[i] - elev[receiver[i]];
        assert!(
            (drop - expected_drop).abs() / expected_drop < 0.01,
            "steady-state drop {drop} != expected {expected_drop}"
        );
    }

    /// Steeper terrain (larger A) erodes faster: with the same uplift, a higher
    /// area gives a gentler equilibrium slope.
    #[test]
    fn higher_drainage_area_gives_gentler_slope() {
        let drop = |a: f32| {
            let n = 10usize;
            let (k, m, u, d, dt) = (0.1f32, 0.5f32, 0.01f32, 1.0f32, 1.0f32);
            let receiver: Vec<usize> = (0..n).map(|i| (i + 1).min(n - 1)).collect();
            let mut is_sink = vec![false; n];
            is_sink[n - 1] = true;
            let dist = vec![d; n];
            let area = vec![a; n];
            let area_pow: Vec<f32> = area.iter().map(|&a| a.powf(m)).collect();
            let order: Vec<usize> = (0..n).rev().collect();
            let erod = vec![1.0f32; n];
            let mut elev = vec![0.0f32; n];
            for _ in 0..5000 {
                for (i, e) in elev.iter_mut().enumerate() {
                    if !is_sink[i] {
                        *e += u * dt;
                    }
                }
                incise_step(
                    &mut elev, &receiver, &is_sink, &dist, &area, &area_pow, 0.0, &erod, &order, k,
                    dt, 1.0, 0.0,
                );
            }
            elev[n / 2] - elev[receiver[n / 2]]
        };
        assert!(drop(4.0) < drop(1.0));
    }

    /// Deposition conserves mass (deposited + lost == eroded) and aggrades a
    /// low-gradient reach en route instead of carrying everything to the sink.
    #[test]
    fn deposition_aggrades_flats_and_conserves_mass() {
        // chain: source 2 -> reach 1 -> sink 0 (sea). `dist` is a PRODUCTION-scale
        // receiver distance (~2e-3 rad ≈ 13 km), not a full radian — so the repose
        // offset `slope*dist` is small and the elevations are scaled to match.
        // Reach 1 sits just below its repose surface (aggrades); the steeper source
        // 2 passes its load on. (An earlier version used dist=1.0, which hid the
        // slope's true production magnitude — see EROSION_DEPOSITION_SLOPE docs.)
        let routing = Routing {
            receiver: vec![0, 0, 1],
            is_sink: vec![true, false, false],
            dist: vec![0.0, 2.0e-3, 2.0e-3],
            order: vec![0, 1, 2],            // downstream-first
            base_floor: vec![0.0, 0.0, 0.0], // sea level
            mfd: None,
        };
        // slope*dist = 6.0 * 2e-3 = 0.012 repose offset above the receiver.
        // reach 1 (elev -0.015, receiver 0 at -0.02): target -0.008 > -0.015 -> aggrades.
        // source 2 (elev 0.02, receiver 1 at -0.015): target -0.003 < 0.02 -> passes on.
        let elev = vec![-0.02, -0.015, 0.02];
        let areas = vec![1.0, 1.0, 1.0];
        let eroded_vol = vec![0.0, 0.0, 0.05];
        let mut thick = vec![0.0; 3];

        let (deposited, lost) = deposit(
            &routing,
            &eroded_vol,
            &elev,
            &areas,
            0.7, // isostasy slope
            0.5, // deposit_fill_fraction
            6.0, // deposition_slope (production default)
            &mut thick,
        );

        let total: f64 = eroded_vol.iter().map(|&v| v as f64).sum();
        assert!(
            (deposited + lost - total).abs() < 1e-6,
            "mass not conserved"
        );
        assert!(thick[1] > 0.0, "flat reach should aggrade");
        assert_eq!(thick[2], 0.0, "steep source should not aggrade");
    }

    /// With `deposition_slope = 0` there is no en-route aggradation: everything
    /// reaches the sink, where the base-level fill cap applies.
    #[test]
    fn deposition_off_is_sink_fill_only() {
        let routing = Routing {
            receiver: vec![0, 0, 1],
            is_sink: vec![true, false, false],
            dist: vec![0.0, 1.0, 1.0],
            order: vec![0, 1, 2],
            base_floor: vec![0.0, 0.0, 0.0],
            mfd: None,
        };
        let elev = vec![-0.05, 0.0, 0.1];
        let areas = vec![1.0, 1.0, 1.0];
        let eroded_vol = vec![0.0, 0.0, 0.05];
        let mut thick = vec![0.0; 3];

        let (deposited, lost) = deposit(
            &routing,
            &eroded_vol,
            &elev,
            &areas,
            0.7,
            0.5,
            0.0,
            &mut thick,
        );

        assert_eq!(thick[1], 0.0, "no en-route deposition when slope is 0");
        // Sink fill cap = 0.5 * depth(0.05) / slope(0.7) * area(1) = 0.0357.
        assert!((deposited - 0.5 * 0.05 / 0.7).abs() < 1e-6);
        assert!((deposited + lost - 0.05).abs() < 1e-6);
    }
}
