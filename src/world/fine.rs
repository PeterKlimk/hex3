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
use super::{Atmosphere, Crust, Elevation, FeatureFields, Hydrology, Tessellation};

type CoarseTree = ImmutableKdTree<f32, 3>;

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
            FINE_MAX_CELLS,
        )
    }

    /// `max_cells` is a guardrail ceiling, not a target: the count emerges from
    /// the resolution field and is only coarsened if it would exceed this.
    pub fn generate_with_target(
        seed: u64,
        coarse_tessellation: &Tessellation,
        crust: &Crust,
        features: &FeatureFields,
        coarse_elevation: &Elevation,
        atmosphere: &Atmosphere,
        max_cells: usize,
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
        log_resolution_probe(&tessellation, &elevation);

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
    let place = |c: usize| -> Vec<Vec3> {
        let expected = density[c] * areas[c];
        // Stochastic rounding so the total count is unbiased, not floored.
        let extra = (hash_unit_f32(seed, c as u64, 1) < expected.fract()) as u64;
        let count = expected.floor() as u64 + extra;
        if count == 0 {
            return Vec::new();
        }
        let center = coarse.cell_center(c);
        let radius = (areas[c] / std::f32::consts::PI).sqrt(); // equal-area disk
        let (u, v) = tangent_basis(center);
        // Fibonacci sunflower: locally even with a built-in minimum distance
        // (~0.5 of the local spacing), so the seed has few slivers and both the
        // relaxation and the tessellation stay fast. Small jitter breaks the
        // spiral regularity; relaxation finishes the job and fixes cell seams.
        let jitter = radius / (count as f32).sqrt() * 0.5;
        (0..count)
            .map(|k| {
                let rr = radius * ((k as f32 + 0.5) / count as f32).sqrt();
                let theta = k as f32 * GOLDEN_ANGLE;
                let jr = jitter * hash_unit_f32(seed, c as u64, 2 * k + 2).sqrt();
                let ja = hash_unit_f32(seed, c as u64, 2 * k + 3) * std::f32::consts::TAU;
                let px = rr * theta.cos() + jr * ja.cos();
                let py = rr * theta.sin() + jr * ja.sin();
                (center + u * px + v * py).normalize()
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
