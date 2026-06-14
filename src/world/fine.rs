//! Adaptive fine mesh refinement for Stage 3 hydrology and erosion.

use std::time::Instant;

use glam::Vec3;
use kiddo::{ImmutableKdTree, KdTree, SquaredEuclidean};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
#[cfg(not(feature = "single-threaded"))]
use rayon::prelude::*;

use super::constants::*;
use super::elevation::{coarse_elevation_fields, ElevationFields};
use super::{Atmosphere, Crust, Elevation, FeatureFields, Hydrology, Tessellation};

type CoarseTree = ImmutableKdTree<f32, 3>;
const FINE_FIBONACCI_JITTER: f32 = 0.25;
const PHI: f32 = 1.618_034;

/// Fine Stage-3 world state.
pub struct FineWorld {
    pub tessellation: Tessellation,
    pub coarse_cell: Vec<usize>,
    pub fields: FineFields,
    pub elevation: Elevation,
    pub hydrology: Hydrology,
    pub density: Vec<f32>,
    pub achieved_density_ratio: f32,
}

/// Smooth fields transferred to the fine mesh.
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
    ) -> Self {
        Self::generate_with_target(
            seed,
            coarse_tessellation,
            crust,
            features,
            coarse_elevation,
            atmosphere,
            FINE_NUM_CELLS,
        )
    }

    pub fn generate_with_target(
        seed: u64,
        coarse_tessellation: &Tessellation,
        crust: &Crust,
        features: &FeatureFields,
        coarse_elevation: &Elevation,
        atmosphere: &Atmosphere,
        target_cells: usize,
    ) -> Self {
        let total = Instant::now();

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
        let density = compute_density_prior(
            coarse_tessellation,
            coarse_elevation,
            features,
            &preview_hydrology,
        );
        let density_min = density.iter().copied().fold(f32::INFINITY, f32::min);
        let density_max = density.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let achieved_density_ratio = density_max / density_min.max(1e-6);
        log::info!(
            "fine mesh: density prior {:.2?}, ratio {:.1}:1",
            t0.elapsed(),
            achieved_density_ratio
        );

        let t0 = Instant::now();
        let tree = build_coarse_tree(coarse_tessellation);
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(30));
        let points =
            sample_fine_points(coarse_tessellation, &density, &tree, target_cells, &mut rng);
        log::info!(
            "fine mesh: weighted sampling {:.2?} ({} cells)",
            t0.elapsed(),
            points.len()
        );

        let t0 = Instant::now();
        let tessellation = Tessellation::from_points_knn_clipping(points);
        log::info!("fine mesh: tessellation {:.2?}", t0.elapsed());

        let t0 = Instant::now();
        let coarse_cell = map_to_coarse(&tessellation, &tree);
        let fine_density: Vec<f32> = coarse_cell.iter().map(|&c| density[c]).collect();
        log_resolution_probe(&tessellation, &fine_density);
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
        let mut elev_rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(3));
        let elevation = Elevation::refine_from_base(&tessellation, &base_elevation, &mut elev_rng);
        log::info!("fine mesh: elevation refine {:.2?}", t0.elapsed());

        let t0 = Instant::now();
        let hydrology = Hydrology::generate_from_continentality(
            &tessellation,
            &fields.elevation_fields.continentality,
            &elevation,
            &fields.precipitation,
            &fields.temperature,
        );
        log::info!("fine mesh: hydrology {:.2?}", t0.elapsed());
        log::info!(
            "fine mesh: total {:.2?}, cells={}, density_ratio={:.1}:1",
            total.elapsed(),
            tessellation.num_cells(),
            achieved_density_ratio
        );

        Self {
            tessellation,
            coarse_cell,
            fields,
            elevation,
            hydrology,
            density: fine_density,
            achieved_density_ratio,
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

fn compute_density_prior(
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

    // Land density is anchored to the plains floor (FINE_LAND_BASE_DENSITY) and
    // clamped to FINE_MAX_DENSITY_RATIO above it, so plains:mountains spans the
    // full ratio. Ocean is set independently as a small fraction of that floor,
    // OUTSIDE the clamp — otherwise (anchoring the clamp to the ocean min) the
    // ratio budget is spent on ocean->mountain and land contrast collapses.
    let land_floor = FINE_LAND_BASE_DENSITY;
    let land_ceiling = land_floor * FINE_MAX_DENSITY_RATIO;
    let ocean_density = land_floor * FINE_OCEAN_DENSITY_RATIO;

    let mut density = Vec::with_capacity(n);
    for i in 0..n {
        if elevation.values[i] < 0.0 {
            density.push(ocean_density);
            continue;
        }
        // Each feature normalized to [0,1], then raised to a concentration
        // exponent so gentle terrain stays near the plains floor and only the
        // steepest ground / strongest channels pull cells in.
        let e = FINE_DENSITY_FEATURE_EXPONENT;
        let slope = (elevation.slope(tessellation, i) / max_slope).powf(e);
        let flow = (preview_hydrology.flow_accumulation[i].max(1.0).ln() / max_flow_ln).powf(e);
        let activity = features.activity[i].clamp(0.0, 1.0).powf(e);
        let d = land_floor
            + FINE_SLOPE_DENSITY_WEIGHT * slope
            + FINE_FLOW_DENSITY_WEIGHT * flow
            + FINE_ACTIVITY_DENSITY_WEIGHT * activity;
        // Land floor is `land_floor` by construction (features are non-negative);
        // only the ceiling needs clamping.
        density.push(d.min(land_ceiling));
    }
    density
}

/// Report the physical resolution of the generated fine mesh, by terrain tier,
/// so we can judge whether mountains are resolved finely enough for erosion
/// (target: low single-digit km). Cell width ~ sqrt(cell_area) on the unit
/// sphere, scaled by PLANET_RADIUS_KM.
fn log_resolution_probe(tessellation: &Tessellation, fine_density: &[f32]) {
    let areas = tessellation.cell_areas();
    let n = areas.len().min(fine_density.len());
    if n == 0 {
        return;
    }
    let spacing_km = |i: usize| areas[i].max(0.0).sqrt() * PLANET_RADIUS_KM;

    // Land vs ocean by density floor; the finest land cells ARE the mountains and
    // river channels, so land-spacing percentiles report the resolution where
    // erosion happens without picking an arbitrary "mountain" threshold.
    let base = FINE_LAND_BASE_DENSITY;
    let mut land: Vec<f32> = Vec::new();
    let mut ocean: Vec<f32> = Vec::new();
    for i in 0..n {
        let s = spacing_km(i);
        if fine_density[i] < base {
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

fn sample_fine_points<R: Rng>(
    coarse: &Tessellation,
    density: &[f32],
    tree: &CoarseTree,
    target: usize,
    rng: &mut R,
) -> Vec<Vec3> {
    let mean_density = area_weighted_mean_density(coarse, density).max(1e-6);
    let max_density = density.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let candidate_count = ((target as f32 * max_density / mean_density) * 1.15).ceil() as usize;
    let mean_spacing = (4.0 * std::f32::consts::PI / candidate_count as f32).sqrt();
    let jitter = mean_spacing * FINE_FIBONACCI_JITTER;
    let sampling_seed = rng.gen::<u64>();

    let t0 = Instant::now();
    #[cfg(not(feature = "single-threaded"))]
    let mut accepted: Vec<Vec3> = {
        let threads = rayon::current_num_threads().max(1);
        let chunk_size = candidate_count.div_ceil(threads * 4).max(1);
        let ranges: Vec<(usize, usize)> = (0..candidate_count)
            .step_by(chunk_size)
            .map(|start| (start, (start + chunk_size).min(candidate_count)))
            .collect();
        let chunks: Vec<Vec<Vec3>> = ranges
            .par_iter()
            .map(|&(start, end)| {
                let mut local = Vec::new();
                for i in start..end {
                    let p = jittered_fibonacci_point(i, candidate_count, jitter, sampling_seed);
                    let coarse_idx = nearest_coarse(tree, p);
                    let accept_prob = (density[coarse_idx] / max_density).clamp(0.0, 1.0);
                    if hash_unit_f32(sampling_seed, i as u64, 1) <= accept_prob {
                        local.push(p);
                    }
                }
                local
            })
            .collect();
        chunks.into_iter().flatten().collect()
    };

    #[cfg(feature = "single-threaded")]
    let mut accepted: Vec<Vec3> = {
        let mut accepted = Vec::with_capacity(target);
        for i in 0..candidate_count {
            let p = jittered_fibonacci_point(i, candidate_count, jitter, sampling_seed);
            let coarse_idx = nearest_coarse(tree, p);
            let accept_prob = (density[coarse_idx] / max_density).clamp(0.0, 1.0);
            if hash_unit_f32(sampling_seed, i as u64, 1) <= accept_prob {
                accepted.push(p);
            }
        }
        accepted
    };

    log::info!(
        "fine mesh: sampling generated/classified {} candidates in {:.2?} ({} accepted)",
        candidate_count,
        t0.elapsed(),
        accepted.len(),
    );

    let t0 = Instant::now();
    while accepted.len() < target {
        let p = random_sphere_point(rng);
        let coarse_idx = nearest_coarse(tree, p);
        let accept_prob = (density[coarse_idx] / max_density).clamp(0.0, 1.0);
        if rng.gen::<f32>() <= accept_prob {
            accepted.push(p);
        }
    }

    accepted.shuffle(rng);
    accepted.truncate(target);
    log::info!(
        "fine mesh: sampling filled/shuffled target in {:.2?}",
        t0.elapsed()
    );

    relax_fine_points(accepted, density, tree, mean_density, target)
}

const RELAX_K: usize = 8;

/// Target nearest-neighbour spacing at a given density, for `target` total
/// points distributed with areal density proportional to the field.
fn target_spacing(density_value: f32, mean_density: f32, target: usize) -> f32 {
    let g = target as f32 * density_value / (mean_density * 4.0 * std::f32::consts::PI);
    (1.0 / g.max(1e-12)).sqrt()
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
fn relax_fine_points(
    mut points: Vec<Vec3>,
    density: &[f32],
    coarse_tree: &CoarseTree,
    mean_density: f32,
    target: usize,
) -> Vec<Vec3> {
    if FINE_RELAX_PASSES == 0 {
        return points;
    }
    let t0 = Instant::now();

    // Fix each point's target spacing from its initial coarse cell once: points
    // move only ~0.4*spacing per pass, so re-deriving density mid-relax isn't
    // worth a coarse-tree query per point per pass.
    let point_spacing: Vec<f32> = {
        #[cfg(not(feature = "single-threaded"))]
        {
            points
                .par_iter()
                .map(|&p| {
                    target_spacing(
                        density[nearest_coarse(coarse_tree, p)],
                        mean_density,
                        target,
                    )
                })
                .collect()
        }
        #[cfg(feature = "single-threaded")]
        {
            points
                .iter()
                .map(|&p| {
                    target_spacing(
                        density[nearest_coarse(coarse_tree, p)],
                        mean_density,
                        target,
                    )
                })
                .collect()
        }
    };

    for _ in 0..FINE_RELAX_PASSES {
        let entries: Vec<[f32; 3]> = points.iter().map(|p| p.to_array()).collect();
        let mut tree = KdTree::<f32, 3>::with_capacity(entries.len());
        for (i, e) in entries.iter().enumerate() {
            tree.add(e, i as u64);
        }
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
    }

    log::info!(
        "fine mesh: relaxation {} passes {:.2?}",
        FINE_RELAX_PASSES,
        t0.elapsed()
    );
    points
}

fn area_weighted_mean_density(coarse: &Tessellation, density: &[f32]) -> f32 {
    let areas = coarse.cell_areas();
    let total_area: f32 = areas.iter().sum();
    areas
        .iter()
        .zip(density.iter())
        .map(|(&a, &d)| a * d)
        .sum::<f32>()
        / total_area.max(1e-6)
}

fn random_sphere_point<R: Rng>(rng: &mut R) -> Vec3 {
    let y: f32 = rng.gen_range(-1.0..1.0);
    let theta: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
    let r = (1.0 - y * y).sqrt();
    Vec3::new(r * theta.cos(), y, r * theta.sin())
}

fn jittered_fibonacci_point(i: usize, n: usize, jitter: f32, seed: u64) -> Vec3 {
    let y = 1.0 - (2.0 * i as f32 + 1.0) / n as f32;
    let r = (1.0 - y * y).sqrt();
    let theta = std::f32::consts::TAU * i as f32 / PHI;

    let mut p = Vec3::new(r * theta.cos(), y, r * theta.sin());
    if jitter > 0.0 {
        let arbitrary = if p.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let u = p.cross(arbitrary).normalize();
        let v = p.cross(u);
        let angle = hash_unit_f32(seed, i as u64, 0) * std::f32::consts::TAU;
        let tangent = u * angle.cos() + v * angle.sin();
        p = (p + tangent * jitter).normalize();
    }
    p
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
