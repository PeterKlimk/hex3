//! Adaptive fine mesh refinement for Stage 3 hydrology and erosion.

use std::time::Instant;

use glam::Vec3;
use kiddo::{ImmutableKdTree, KdTree, SquaredEuclidean};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
#[cfg(not(feature = "single-threaded"))]
use rayon::prelude::*;

use super::constants::*;
use super::elevation::{coarse_elevation_fields, ElevationFields};
use super::erosion::ErosionParams;
use super::fine_cache::{self, FineCacheMode};
use super::{Atmosphere, Crust, Elevation, FeatureFields, Hydrology, Tessellation};

type CoarseTree = ImmutableKdTree<f32, 3>;

/// Fine Stage-3 world state, split into the expensive, reused [`FineBase`]
/// (stage 3a: mesh + transferred fields + pre-erosion base elevation) and the
/// cheap, per-variant [`FineSurface`] (stages 3b+3c: eroded elevation +
/// hydrology). Re-running erosion with tweaked knobs rebuilds only the surface
/// (`rerun_surface`), reusing the base — that split is the whole point.
///
/// The accessor methods hide the split so consumers don't care which half a
/// field lives in.
pub struct FineWorld {
    pub base: FineBase,
    pub surface: FineSurface,
}

/// Expensive, reused base of the fine mesh (stage 3a): the adaptive tessellation,
/// the coarse-cell map, the transferred smooth fields, and the pre-erosion base
/// elevation. Built once; every erosion/hydrology variant reads it by reference.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct FineBase {
    pub tessellation: Tessellation,
    pub coarse_cell: Vec<usize>,
    pub fields: FineFields,
    /// Coarse elevation interpolated onto the fine cells (the fixed sea-level
    /// datum erosion carves into). Distinct from the eroded `surface.elevation`.
    pub base_elevation: Vec<f32>,
    pub density: Vec<f32>,
    pub achieved_density_ratio: f32,
}

/// Cheap, per-variant surface over a [`FineBase`] (stages 3b+3c): the eroded
/// elevation and the hydrology derived from it. Re-generated to replace when
/// erosion knobs change.
pub struct FineSurface {
    pub elevation: Elevation,
    pub hydrology: Hydrology,
}

/// Smooth fields transferred to the fine mesh.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct FineFields {
    pub elevation_fields: ElevationFields,
    pub temperature: Vec<f32>,
    pub precipitation: Vec<f32>,
    pub uplift: Vec<f32>,
}

impl FineWorld {
    pub fn generate(
        seed: u64,
        coarse_tessellation: &Tessellation,
        crust: &Crust,
        features: &FeatureFields,
        coarse_elevation: &Elevation,
        atmosphere: &Atmosphere,
        params: ErosionParams,
        cache: FineCacheMode,
    ) -> Self {
        Self::generate_with_target(
            seed,
            coarse_tessellation,
            crust,
            features,
            coarse_elevation,
            atmosphere,
            FINE_MAX_CELLS,
            params,
            cache,
        )
    }

    /// `max_cells` is a guardrail ceiling, not a target: the count emerges from
    /// the resolution field and is only coarsened if it would exceed this.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_target(
        seed: u64,
        coarse_tessellation: &Tessellation,
        crust: &Crust,
        features: &FeatureFields,
        coarse_elevation: &Elevation,
        atmosphere: &Atmosphere,
        max_cells: usize,
        params: ErosionParams,
        cache: FineCacheMode,
    ) -> Self {
        let total = Instant::now();
        let base = FineBase::load_or_generate(
            cache,
            seed,
            coarse_tessellation,
            crust,
            features,
            coarse_elevation,
            atmosphere,
            max_cells,
        );
        let surface = FineSurface::generate(seed, &base, params);
        log::info!(
            "fine mesh: total {:.2?}, cells={}, density_ratio={:.1}:1",
            total.elapsed(),
            base.tessellation.num_cells(),
            base.achieved_density_ratio
        );
        Self { base, surface }
    }

    /// Re-run erosion + hydrology over the existing base, replacing the surface
    /// in place (no mesh recompute). Used when erosion knobs change.
    pub fn rerun_surface(&mut self, seed: u64, params: ErosionParams) {
        self.surface = FineSurface::generate(seed, &self.base, params);
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
    pub fn elevation(&self) -> &Elevation {
        &self.surface.elevation
    }
    pub fn hydrology(&self) -> &Hydrology {
        &self.surface.hydrology
    }
    pub fn hydrology_mut(&mut self) -> &mut Hydrology {
        &mut self.surface.hydrology
    }

    /// Adjust the climate ratio on the fine hydrology in place (disjoint borrow
    /// of base.tessellation + surface.hydrology).
    pub fn set_climate_ratio(&mut self, ratio: f32) {
        self.surface
            .hydrology
            .set_climate_ratio(&self.base.tessellation, ratio);
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
        coarse_tessellation: &Tessellation,
        crust: &Crust,
        features: &FeatureFields,
        coarse_elevation: &Elevation,
        atmosphere: &Atmosphere,
        max_cells: usize,
    ) -> Self {
        let key = fine_cache::fine_base_key(
            seed,
            coarse_tessellation,
            coarse_elevation,
            atmosphere,
            max_cells,
        );
        if cache == FineCacheMode::Enabled {
            if let Some(base) = fine_cache::load(key) {
                return base;
            }
        }
        let base = Self::generate_with_target(
            seed,
            coarse_tessellation,
            crust,
            features,
            coarse_elevation,
            atmosphere,
            max_cells,
        );
        if matches!(cache, FineCacheMode::Enabled | FineCacheMode::Rebuild) {
            fine_cache::save(key, &base);
        }
        base
    }

    /// Stage 3a: build the expensive, reusable fine-mesh base (steps 1–7 of the
    /// old monolith). Stops short of erosion — that's [`FineSurface::generate`].
    pub fn generate_with_target(
        seed: u64,
        coarse_tessellation: &Tessellation,
        crust: &Crust,
        features: &FeatureFields,
        coarse_elevation: &Elevation,
        atmosphere: &Atmosphere,
        max_cells: usize,
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
        let density_min = density.iter().copied().fold(f32::INFINITY, f32::min).max(1e-12);
        let density_max = density.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let achieved_density_ratio = density_max / density_min;
        log::info!(
            "fine mesh: density field {:.2?}, target sizes {:.1}-{:.1} km, emergent {:.0} cells (cap {})",
            t0.elapsed(),
            FINE_MOUNTAIN_CELL_KM,
            FINE_OCEAN_CELL_KM,
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
        let base_elevation = interpolate_coarse_elevation(
            coarse_tessellation,
            &tessellation,
            &coarse_cell,
            &coarse_elevation.values,
        );
        log::info!("fine mesh: elevation refine {:.2?}", t0.elapsed());

        Self {
            tessellation,
            coarse_cell,
            fields,
            base_elevation,
            density: fine_density,
            achieved_density_ratio,
        }
    }
}

impl FineSurface {
    /// Stages 3b+3c: carve the base into river valleys, then derive hydrology.
    /// Reads `base` by reference so it can be re-run cheaply with new erosion
    /// knobs (`params`). `seed` drives only the cosmetic micro-noise rng.
    pub fn generate(seed: u64, base: &FineBase, params: ErosionParams) -> Self {
        // Fluvial erosion: carve the interpolated base into real river valleys by
        // evolving crust thickness (isostasy responds). Runs on the fine mesh
        // before final hydrology; sea level is the fixed datum inherited via
        // `base_elevation`. See docs/specs/erosion.md.
        let t0 = Instant::now();
        let eroded_base = super::erosion::erode(
            &base.tessellation,
            &base.fields.elevation_fields,
            &base.base_elevation,
            &base.fields.precipitation,
            params,
        );
        log::info!("fine mesh: erosion {:.2?}", t0.elapsed());
        Self::from_eroded(seed, base, &eroded_base)
    }

    /// Build the surface (micro-noise elevation + hydrology) from an already-
    /// eroded elevation. Shared by full generation and the UI erosion stepper,
    /// which supplies the current elevation of a resumable `ErosionState` (step 0
    /// = the un-eroded `base.base_elevation`).
    pub fn from_eroded(seed: u64, base: &FineBase, eroded: &[f32]) -> Self {
        // Cosmetic micro noise rides on the eroded surface; this is the elevation
        // hydrology and rendering consume.
        let mut elev_rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(3));
        let elevation = Elevation::refine_from_base(&base.tessellation, eroded, &mut elev_rng);
        log_resolution_probe(&base.tessellation, &elevation);

        let t0 = Instant::now();
        let hydrology = Hydrology::generate_from_continentality(
            &base.tessellation,
            &base.fields.elevation_fields.continentality,
            &elevation,
            &base.fields.precipitation,
            &base.fields.temperature,
        );
        log::info!("fine mesh: hydrology {:.2?}", t0.elapsed());

        Self {
            elevation,
            hydrology,
        }
    }
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
) -> Vec<f32> {
    let n = tessellation.num_cells();
    let max_slope = (0..n)
        .map(|i| elevation.slope(tessellation, i))
        .fold(0.0_f32, f32::max)
        .max(1e-6);
    let max_flow_ln = preview_hydrology
        .flow_accumulation
        .iter()
        .map(|f| f.max(1.0).ln())
        .fold(0.0_f32, f32::max)
        .max(1e-6);

    let g_plains = cell_size_km_to_density(FINE_PLAINS_CELL_KM);
    let g_mountain = cell_size_km_to_density(FINE_MOUNTAIN_CELL_KM);
    let g_ocean = cell_size_km_to_density(FINE_OCEAN_CELL_KM);
    let e = FINE_DENSITY_FEATURE_EXPONENT;
    let wsum =
        FINE_SLOPE_DENSITY_WEIGHT + FINE_FLOW_DENSITY_WEIGHT + FINE_ACTIVITY_DENSITY_WEIGHT;

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
        let slope = (elevation.slope(tessellation, i) / max_slope).powf(e);
        let flow = (preview_hydrology.flow_accumulation[i].max(1.0).ln() / max_flow_ln).powf(e);
        let activity = features.activity[i].clamp(0.0, 1.0).powf(e);
        let demand = (FINE_SLOPE_DENSITY_WEIGHT * slope
            + FINE_FLOW_DENSITY_WEIGHT * flow
            + FINE_ACTIVITY_DENSITY_WEIGHT * activity)
            / wsum;
        density.push(g_plains + demand * (g_mountain - g_plains));
    }
    density
}

/// Blue-noise quality probe (a REFERENCE for judging relaxation passes, not a
/// target). Uses the canonical regularity metric from `sample_experiment.rs`:
///
///     rho = nearest-neighbour distance / sqrt(cell_area)
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
    let mut rho: Vec<f32> = (0..n).into_par_iter().map(rho_of).filter(|r| r.is_finite()).collect();
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
        support.push(nearest, interpolation_weight(coarse.cell_center(nearest), pos));
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

fn transfer_fields(
    coarse: &Tessellation,
    fine: &Tessellation,
    coarse_cell: &[usize],
    crust: &Crust,
    features: &FeatureFields,
    coarse_elevation: &Elevation,
    atmosphere: &Atmosphere,
) -> FineFields {
    let coarse_fields = coarse_elevation_fields(coarse, crust, features);
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

    for cell in transferred {
        elevation_fields.crust_thickness.push(cell.crust_thickness);
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
    }

    FineFields {
        elevation_fields,
        temperature,
        precipitation,
        uplift,
    }
}

struct TransferredCell {
    crust_thickness: f32,
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
}

fn interpolation_weight(coarse_pos: Vec3, fine_pos: Vec3) -> f32 {
    let dist = coarse_pos.dot(fine_pos).clamp(-1.0, 1.0).acos();
    1.0 / (dist * dist + 1e-8)
}
