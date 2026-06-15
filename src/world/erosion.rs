//! Fluvial erosion on the fine mesh (docs/specs/erosion.md).
//!
//! Detachment-limited stream power (Braun & Willett 2013 implicit scheme, n = 1)
//! with linear hillslope diffusion, plus simple coastal/sink deposition. The
//! loop runs once after fine-elevation refinement and before final hydrology.
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
    /// Stream-power erodibility coefficient K.
    pub k: f32,
    /// Linear hillslope diffusivity.
    pub diffusivity: f32,
    /// Jacobi sweeps per implicit diffusion solve.
    pub diffusion_iters: usize,
    /// Re-route (and re-accumulate drainage area) every this many steps.
    pub reroute_interval: usize,
    /// Tectonic uplift source scale (thickness units).
    pub uplift_scale: f32,
    /// Fraction of a sink's depth-to-sea-level that deposition may fill.
    pub deposit_fill_fraction: f32,
    /// Channel-initiation support area (km²) at mean land wetness. Below the
    /// equivalent discharge a cell is a hillslope (diffusion only, no
    /// stream-power incision). 0 = off (incise wherever downhill).
    pub channel_support_km2: f32,
    /// Log-amplitude of lithologic erodibility variation (contrast). The K
    /// multiplier is exp(sigma * fbm), normalized to unit mean over land so it
    /// only redistributes incision. 0 = uniform K.
    pub litho_sigma: f32,
    /// Strength of the orographic precip modulation on the eroded fine relief
    /// (climate↔erosion feedback: windward wetter, lee drier). 0 = coarse precip.
    pub orographic_precip_strength: f32,
}

impl Default for ErosionParams {
    fn default() -> Self {
        Self {
            steps: EROSION_STEPS,
            dt: EROSION_DT,
            m: EROSION_M,
            k: EROSION_K,
            diffusivity: EROSION_DIFFUSIVITY,
            diffusion_iters: EROSION_DIFFUSION_ITERS,
            reroute_interval: EROSION_REROUTE_INTERVAL,
            uplift_scale: EROSION_UPLIFT_SCALE,
            deposit_fill_fraction: EROSION_DEPOSIT_FILL_FRACTION,
            channel_support_km2: EROSION_CHANNEL_SUPPORT_KM2,
            litho_sigma: EROSION_LITHO_SIGMA,
            orographic_precip_strength: OROGRAPHIC_PRECIP_STRENGTH,
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
pub(crate) fn erode(
    tess: &Tessellation,
    fields: &ElevationFields,
    base: &[f32],
    precipitation: &[f32],
    erodibility: &[f32],
    params: ErosionParams,
) -> Vec<f32> {
    roughness_report(tess, base, "base ");
    let mut state = ErosionState::new(tess, fields, base, precipitation, erodibility, params);
    state.step(params.steps);
    state.log_summary();
    let final_elev = state.elevation();
    roughness_report(tess, &final_elev, "eroded");
    final_elev
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
    pub(crate) fn new(
        tess: &Tessellation,
        fields: &ElevationFields,
        base: &[f32],
        precipitation: &[f32],
        erodibility_in: &[f32],
        params: ErosionParams,
    ) -> Self {
        let n = tess.num_cells();
        let slope = isostasy_slope();
        let inv_slope = 1.0 / slope;
        let thick_init = fields.crust_thickness.clone();

        // Tectonic uplift source, in THICKNESS units per step. arc/collision are
        // elevation magnitudes (-> thickness via /slope); rift_delta is already a
        // signed thickness delta (negative in the axial valley). NOT atmospheric
        // uplift. Scaled by params.uplift_scale.
        let u_thick: Vec<f32> = (0..n)
            .map(|i| {
                params.uplift_scale
                    * ((fields.arc[i] + fields.collision[i]) * inv_slope + fields.rift_delta[i])
            })
            .collect();

        let areas = tess.cell_areas();

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

        let geom = NeighborGeometry::build(tess);
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
            areas,
            geom,
            params,
            thick,
            routing: None,
            area: Vec::new(),
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
        (0..self.n)
            .map(|i| self.base[i] + self.slope * (self.thick[i] - self.thick_init[i]))
            .collect()
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
        let mut s = Instant::now();
        let mut elev = self.derive_elev();
        self.t_misc += s.elapsed().as_secs_f64();

        // 1. Route: receivers across pits via priority-flood fill + steepest
        //    descent. Cells below sea level are fixed base-level sinks.
        if self.step % self.params.reroute_interval == 0 {
            s = Instant::now();
            let Some(r) = Routing::build(&elev, &self.geom) else {
                // No sinks (e.g. an all-land world): nothing to erode toward.
                self.halted = true;
                return;
            };
            self.t_route += s.elapsed().as_secs_f64();

            // 2. Accumulate precipitation-weighted drainage area ("wet area").
            s = Instant::now();
            self.area = accumulate_wet_area(&r, &self.precipitation, &self.areas);
            self.t_accum += s.elapsed().as_secs_f64();

            self.routing = Some(r);
        }
        let routing = self.routing.as_ref().unwrap();

        // 3. Incise (implicit, downstream-first). Snapshot first so we can book
        //    the removed rock as eroded volume and route it downstream.
        s = Instant::now();
        let pre = elev.clone();
        incise_step(
            &mut elev,
            &routing.receiver,
            &routing.is_sink,
            &routing.dist,
            &self.area,
            self.a_crit,
            &self.erodibility,
            &routing.order,
            self.params.k,
            self.params.m,
            self.params.dt,
        );
        // No fluvial erosion below sea level (sea level is fixed): land cells
        // may incise down to, but not past, the datum.
        let mut eroded_vol = vec![0.0f32; n];
        for i in 0..n {
            if routing.is_sink[i] {
                continue;
            }
            if elev[i] < 0.0 {
                elev[i] = 0.0;
            }
            let drop = pre[i] - elev[i];
            if drop > 0.0 {
                eroded_vol[i] = drop * self.inv_slope * self.areas[i];
            }
        }
        self.t_incise += s.elapsed().as_secs_f64();

        // 4. Diffuse (linear hillslope creep) on land, implicit Jacobi. Snapshot
        //    first so the net volume it moves is auditable: interior diffusion is
        //    conservative (symmetric edge fluxes), so this residual measures only
        //    the no-flux clamp + finite-Jacobi non-convergence and should stay
        //    small next to `eroded`.
        s = Instant::now();
        let pre_diff = elev.clone();
        diffuse_land(
            &mut elev,
            routing,
            &self.geom,
            &self.areas,
            self.params.dt,
            self.params.diffusivity,
            self.params.diffusion_iters,
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

        // 7. Deposit: route sediment to the coastal sink it drains into and
        //    drop it there (capped so deltas don't breach sea level). Spreading
        //    over a basin's low cells is simplified to per-mouth deposition;
        //    full transport-limited routing is out of scope (see spec).
        s = Instant::now();
        let (deposited, lost) = deposit(
            routing,
            &eroded_vol,
            &elev,
            &self.areas,
            self.slope,
            self.params.deposit_fill_fraction,
            &mut self.thick,
        );
        self.total_eroded += eroded_vol.iter().map(|&v| v as f64).sum::<f64>();
        self.total_deposited += deposited;
        self.total_lost += lost;
        self.t_deposit += s.elapsed().as_secs_f64();

        self.step += 1;
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

/// Numerical roughness probe (a REFERENCE, not an optimization target). On land
/// cells it reports the local-slope distribution (|d elev| / km to the steepest
/// neighbour), the fraction of cells that are strict local extrema (a clean
/// cell-scale-speckle signal — smooth fluvial terrain has few interior pits/
/// peaks), and Moran's I of the whole field. Compare `base ` vs `eroded` to see
/// what erosion did to terrain coherence rather than guessing from a render.
fn roughness_report(tess: &Tessellation, elev: &[f32], label: &str) {
    let n = tess.num_cells();
    let mut slopes: Vec<f32> = Vec::new();
    let mut extrema = 0usize;
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
        let mut all_higher = true;
        let mut all_lower = true;
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
            if elev[nb] <= elev[i] {
                all_higher = false;
            }
            if elev[nb] >= elev[i] {
                all_lower = false;
            }
        }
        if has_land_nb {
            slopes.push(steepest);
            if all_higher || all_lower {
                extrema += 1;
            }
        }
    }
    if slopes.is_empty() {
        return;
    }
    slopes.sort_by(f32::total_cmp);
    let pct = |p: f32| slopes[(((slopes.len() - 1) as f32) * p) as usize];
    log::info!(
        "erosion roughness [{}]: land {} | slope(elev/km) p50={:.3e} p90={:.3e} p99={:.3e} max={:.3e} | local-extrema {:.2}% | Moran's I {:.3}",
        label,
        land,
        pct(0.50),
        pct(0.90),
        pct(0.99),
        pct(1.0),
        100.0 * extrema as f32 / land.max(1) as f32,
        tess.morans_i(elev),
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
fn incise_step(
    elev: &mut [f32],
    receiver: &[usize],
    is_sink: &[bool],
    dist: &[f32],
    area: &[f32],
    area_crit: f32,
    erodibility: &[f32],
    order: &[usize],
    k: f32,
    m: f32,
    dt: f32,
) {
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
        // h_i = (h_i + dt K A^m h_rcv / d) / (1 + dt K A^m / d), K per-cell
        // (lithologic erodibility).
        let f = dt * k * erodibility[cell] * area[cell].powf(m) / d;
        elev[cell] = (elev[cell] + f * hr) / (1.0 + f);
    }
}

/// Steepest-descent receivers over a priority-flood-filled surface, plus a
/// downstream-first processing order. Cells below sea level are fixed sinks.
struct Routing {
    /// receiver[i]; for sinks receiver[i] == i.
    receiver: Vec<usize>,
    /// Below sea level (fixed base level); not incised/diffused.
    is_sink: Vec<bool>,
    /// Arc distance to receiver (radians on the unit sphere); 0 for sinks.
    dist: Vec<f32>,
    /// Downstream-first: every cell's receiver precedes it.
    order: Vec<usize>,
}

impl Routing {
    fn build(elev: &[f32], geom: &NeighborGeometry) -> Option<Routing> {
        let n = geom.num_cells();
        let is_sink: Vec<bool> = elev.iter().map(|&e| e < 0.0).collect();
        if !is_sink.iter().any(|&s| s) {
            return None;
        }

        // Priority-flood fill (Barnes 2014): every land cell ends up able to
        // drain to a sink, and `flood_parent` carries a valid descent direction
        // across filled flats/pits.
        let mut filled = elev.to_vec();
        let mut processed = vec![false; n];
        let mut flood_parent: Vec<usize> = (0..n).collect();
        let mut heap: std::collections::BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>> =
            std::collections::BinaryHeap::new();
        for i in 0..n {
            if is_sink[i] {
                processed[i] = true;
                heap.push(Reverse((OrderedFloat(elev[i]), i)));
            }
        }
        while let Some(Reverse((level, cell))) = heap.pop() {
            let level = level.0;
            for &nb in geom.tess_neighbors(cell) {
                if processed[nb] {
                    continue;
                }
                processed[nb] = true;
                filled[nb] = elev[nb].max(level);
                flood_parent[nb] = cell;
                heap.push(Reverse((OrderedFloat(filled[nb]), nb)));
            }
        }

        // Receivers: steepest descent on the filled surface, flood-parent
        // fallback on flats. Distance is the chord to the receiver. Both are
        // per-cell and read-only over shared state -> parallel.
        let receiver_of = |i: usize| -> usize {
            if is_sink[i] {
                return i;
            }
            let mut best = flood_parent[i];
            let mut best_elev = filled[i];
            for &nb in geom.tess_neighbors(i) {
                if filled[nb] < best_elev {
                    best_elev = filled[nb];
                    best = nb;
                }
            }
            best
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

        Some(Routing {
            receiver,
            is_sink,
            dist,
            order,
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

/// Linear hillslope diffusion on land cells, solved IMPLICITLY (backward Euler)
/// with Jacobi sweeps. Edges to sinks are no-flux boundaries (no erosion into
/// the ocean). Implicit is unconditionally stable, so the finest sliver cells
/// can't force a substep blow-up the way an explicit CFL scheme would.
///
/// Per cell the backward-Euler update is
///   (1 + c_i) h_i^{new} = h_i^{old} + dt D / area_i * sum_j w_ij h_j^{new}
/// with c_i = dt D / area_i * sum_j w_ij over land neighbours j. We approximate
/// the coupled solve with a few Jacobi iterations (RHS h_old fixed).
fn diffuse_land(
    elev: &mut [f32],
    routing: &Routing,
    geom: &NeighborGeometry,
    areas: &[f32],
    dt: f32,
    diffusivity: f32,
    diffusion_iters: usize,
) {
    if diffusivity <= 0.0 {
        return;
    }
    let n = elev.len();
    let dd = diffusivity * dt;

    // Per-cell f_i = dt D / area_i and diagonal denominator (1 + c_i), counting
    // only land neighbours (sink edges are no-flux). Constant across sweeps.
    let mut f = vec![0.0f32; n];
    let mut denom = vec![1.0f32; n];
    let prep = |i: usize| -> (f32, f32) {
        if routing.is_sink[i] {
            return (0.0, 1.0);
        }
        let fi = dd / areas[i].max(1e-12);
        let mut wsum = 0.0f32;
        for (k, &nb) in geom.tess_neighbors(i).iter().enumerate() {
            if !routing.is_sink[nb] {
                wsum += geom.weight(i, k);
            }
        }
        (fi, 1.0 + fi * wsum)
    };
    for i in 0..n {
        let (fi, di) = prep(i);
        f[i] = fi;
        denom[i] = di;
    }

    let h_old = elev.to_vec(); // fixed RHS
    let mut cur = h_old.clone();
    for _ in 0..diffusion_iters {
        let sweep = |i: usize| -> f32 {
            if routing.is_sink[i] {
                return cur[i];
            }
            let mut acc = 0.0f32;
            for (k, &nb) in geom.tess_neighbors(i).iter().enumerate() {
                if !routing.is_sink[nb] {
                    acc += geom.weight(i, k) * cur[nb];
                }
            }
            ((h_old[i] + f[i] * acc) / denom[i]).max(0.0)
        };
        #[cfg(not(feature = "single-threaded"))]
        {
            cur = (0..n).into_par_iter().map(sweep).collect();
        }
        #[cfg(feature = "single-threaded")]
        {
            cur = (0..n).map(sweep).collect();
        }
    }
    elev.copy_from_slice(&cur);
}

/// Route eroded sediment to the coastal sink each catchment drains into and
/// deposit it there, capped at `EROSION_DEPOSIT_FILL_FRACTION` of the sink's
/// depth-to-sea-level (delta building). Returns (deposited, lost) volumes.
fn deposit(
    routing: &Routing,
    eroded_vol: &[f32],
    elev: &[f32],
    areas: &[f32],
    slope: f32,
    deposit_fill_fraction: f32,
    thick: &mut [f32],
) -> (f64, f64) {
    let n = eroded_vol.len();
    let mut sed = eroded_vol.to_vec();
    // Accumulate downstream into sinks (upstream-first).
    for &cell in routing.order.iter().rev() {
        if routing.is_sink[cell] {
            continue;
        }
        sed[routing.receiver[cell]] += sed[cell];
    }

    let mut deposited = 0.0f64;
    let mut lost = 0.0f64;
    for i in 0..n {
        if !routing.is_sink[i] || sed[i] <= 0.0 {
            continue;
        }
        // Max thickness-volume that fills this sink toward (but not past) sea
        // level, times the allowed fill fraction.
        let depth = (-elev[i]).max(0.0); // elevation units below datum
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
struct NeighborGeometry {
    offsets: Vec<usize>,
    neighbors: Vec<usize>,
    dist: Vec<f32>,
    weight: Vec<f32>,
}

impl NeighborGeometry {
    fn build(tess: &Tessellation) -> Self {
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

    fn tess_neighbors(&self, i: usize) -> &[usize] {
        &self.neighbors[self.offsets[i]..self.offsets[i + 1]]
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
                &mut elev, &receiver, &is_sink, &dist, &area, 0.0, &erod, &order, k, m, dt,
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
                    &mut elev, &receiver, &is_sink, &dist, &area, 0.0, &erod, &order, k, m, dt,
                );
            }
            elev[n / 2] - elev[receiver[n / 2]]
        };
        assert!(drop(4.0) < drop(1.0));
    }
}
