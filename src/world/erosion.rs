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

use std::cmp::Reverse;
use std::collections::VecDeque;

use ordered_float::OrderedFloat;
#[cfg(not(feature = "single-threaded"))]
use rayon::prelude::*;

use super::constants::*;
use super::elevation::{isostasy_slope, ElevationFields};
use super::Tessellation;

/// Run the erosion loop and return the eroded fine-mesh elevation (on the fixed
/// sea-level datum). `fields.crust_thickness` seeds the working thickness;
/// `base` is the interpolated coarse elevation; `precipitation` weights drainage
/// area so wet ranges dissect more finely than arid ones.
pub(crate) fn erode(
    tess: &Tessellation,
    fields: &ElevationFields,
    base: &[f32],
    precipitation: &[f32],
) -> Vec<f32> {
    let n = tess.num_cells();
    let slope = isostasy_slope();
    let inv_slope = 1.0 / slope;
    let thick_init = &fields.crust_thickness;

    // Tectonic uplift source, in THICKNESS units per step. arc/collision are
    // elevation magnitudes (-> thickness via /slope); rift_delta is already a
    // signed thickness delta (negative in the axial valley). NOT atmospheric
    // uplift. Scaled by EROSION_UPLIFT_SCALE.
    let u_thick: Vec<f32> = (0..n)
        .map(|i| {
            EROSION_UPLIFT_SCALE
                * ((fields.arc[i] + fields.collision[i]) * inv_slope + fields.rift_delta[i])
        })
        .collect();

    let areas = tess.cell_areas();
    let geom = NeighborGeometry::build(tess);

    roughness_report(tess, base, "base ");

    let mut thick = thick_init.clone();

    let derive_elev = |thick: &[f32]| -> Vec<f32> {
        (0..n)
            .map(|i| base[i] + slope * (thick[i] - thick_init[i]))
            .collect()
    };

    // Mass-balance accounting (thickness-volume = thickness * area).
    let mut total_eroded = 0.0f64;
    let mut total_deposited = 0.0f64;
    let mut total_lost = 0.0f64;

    for _ in 0..EROSION_STEPS {
        let mut elev = derive_elev(&thick);

        // 1. Route: receivers across pits via priority-flood fill + steepest
        //    descent. Cells below sea level are fixed base-level sinks.
        let Some(routing) = Routing::build(tess, &elev, &geom) else {
            // No sinks (e.g. an all-land world): nothing to erode toward.
            break;
        };

        // 2. Accumulate precipitation-weighted drainage area ("wet area").
        let area = accumulate_wet_area(&routing, precipitation, &areas);

        // 3. Incise (implicit, downstream-first). Snapshot first so we can book
        //    the removed rock as eroded volume and route it downstream.
        let pre = elev.clone();
        incise_step(
            &mut elev,
            &routing.receiver,
            &routing.is_sink,
            &routing.dist,
            &area,
            &routing.order,
            EROSION_K,
            EROSION_M,
            EROSION_DT,
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
                eroded_vol[i] = drop * inv_slope * areas[i];
            }
        }

        // 4. Diffuse (linear hillslope creep) on land, CFL-safe explicit substeps.
        diffuse_land(&mut elev, &routing, &geom, &areas, EROSION_DT);

        // 5. Fold the eroded/diffused surface back into thickness.
        for i in 0..n {
            thick[i] = thick_init[i] + (elev[i] - base[i]) * inv_slope;
        }

        // 6. Uplift source (thickness).
        for i in 0..n {
            thick[i] += u_thick[i] * EROSION_DT;
        }

        // 7. Deposit: route sediment to the coastal sink it drains into and
        //    drop it there (capped so deltas don't breach sea level). Spreading
        //    over a basin's low cells is simplified to per-mouth deposition;
        //    full transport-limited routing is out of scope (see spec).
        let (deposited, lost) =
            deposit(&routing, &eroded_vol, &elev, &areas, slope, &mut thick);
        total_eroded += eroded_vol.iter().map(|&v| v as f64).sum::<f64>();
        total_deposited += deposited;
        total_lost += lost;
    }

    log::info!(
        "erosion: {} steps, eroded {:.3e} deposited {:.3e} lost-to-ocean {:.3e} (volume = thickness x steradian)",
        EROSION_STEPS,
        total_eroded,
        total_deposited,
        total_lost,
    );

    let final_elev = derive_elev(&thick);
    roughness_report(tess, &final_elev, "eroded");
    final_elev
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
    for i in 0..n {
        if elev[i] < 0.0 {
            continue;
        }
        land += 1;
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
    order: &[usize],
    k: f32,
    m: f32,
    dt: f32,
) {
    for &cell in order {
        if is_sink[cell] {
            continue;
        }
        let r = receiver[cell];
        let hr = elev[r];
        if hr >= elev[cell] {
            continue; // no downhill gradient -> no detachment-limited incision
        }
        let d = dist[cell].max(1e-12);
        // h_i = (h_i + dt K A^m h_rcv / d) / (1 + dt K A^m / d)
        let f = dt * k * area[cell].powf(m) / d;
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
    fn build(tess: &Tessellation, elev: &[f32], geom: &NeighborGeometry) -> Option<Routing> {
        let n = tess.num_cells();
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
            for &nb in tess.neighbors(cell) {
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
        // fallback on flats. Distance is the great-circle arc to the receiver.
        let receiver: Vec<usize> = (0..n)
            .map(|i| {
                if is_sink[i] {
                    return i;
                }
                let mut best = flood_parent[i];
                let mut best_elev = filled[i];
                for &nb in tess.neighbors(i) {
                    if filled[nb] < best_elev {
                        best_elev = filled[nb];
                        best = nb;
                    }
                }
                best
            })
            .collect();

        let dist: Vec<f32> = (0..n)
            .map(|i| {
                if is_sink[i] || receiver[i] == i {
                    0.0
                } else {
                    geom.arc_to(i, receiver[i])
                }
            })
            .collect();

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
    let mut acc: Vec<f32> = (0..n).map(|i| precipitation[i].max(0.0) * areas[i]).collect();
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
) {
    if EROSION_DIFFUSIVITY <= 0.0 {
        return;
    }
    let n = elev.len();
    let dd = EROSION_DIFFUSIVITY * dt;

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
    for _ in 0..EROSION_DIFFUSION_ITERS {
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
        let cap = EROSION_DEPOSIT_FILL_FRACTION * depth / slope * areas[i];
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

        let mut elev = vec![0.0f32; n];
        for _ in 0..5000 {
            // Uplift interior cells in elevation space; outlet stays at 0.
            for (i, e) in elev.iter_mut().enumerate() {
                if !is_sink[i] {
                    *e += u * dt;
                }
            }
            incise_step(&mut elev, &receiver, &is_sink, &dist, &area, &order, k, m, dt);
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
            let mut elev = vec![0.0f32; n];
            for _ in 0..5000 {
                for (i, e) in elev.iter_mut().enumerate() {
                    if !is_sink[i] {
                        *e += u * dt;
                    }
                }
                incise_step(&mut elev, &receiver, &is_sink, &dist, &area, &order, k, m, dt);
            }
            elev[n / 2] - elev[receiver[n / 2]]
        };
        assert!(drop(4.0) < drop(1.0));
    }
}
