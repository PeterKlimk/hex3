//! Adaptive fine mesh refinement for Stage 3 hydrology and erosion.

use std::time::Instant;

use glam::Vec3;
use kiddo::{ImmutableKdTree, KdTree, SquaredEuclidean};
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
#[cfg(not(feature = "single-threaded"))]
use rayon::prelude::*;

use super::constants::*;
use super::elevation::{coarse_elevation_fields, ElevationFields};
use super::erosion::ErosionParams;
use super::fine_cache::{self, FineCacheMode};
use super::{Atmosphere, CellWaterState, Crust, Elevation, FeatureFields, Hydrology, Tessellation};

type CoarseTree = ImmutableKdTree<f32, 3>;

/// Runtime-tunable knobs for the fine-mesh areal density prior (defaults from the
/// `FINE_*` consts). Lets tools sweep the ocean/plains/mountain cell-size budget
/// and the demand blend without a recompile — mirrors [`ErosionParams`]. Changing
/// any field changes the sampled mesh, so it is part of the fine-base cache key.
#[derive(Debug, Clone, Copy)]
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
    /// Precipitation this surface's hydrology used. For the eroded surface this
    /// is the orographic precip recomputed on the eroded relief (rain shadows);
    /// for the pre-erosion surface it is the transferred coarse precip.
    pub precipitation: Vec<f32>,
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
            density_params,
        );
        // Pre-erosion surface: hydrology rides the un-eroded interpolated base.
        let pre = FineSurface::from_eroded(
            seed,
            &base,
            &base.base_elevation,
            &base.fields.precipitation,
            0.0,
        );
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
        coarse_tessellation: &Tessellation,
        crust: &Crust,
        features: &FeatureFields,
        coarse_elevation: &Elevation,
        atmosphere: &Atmosphere,
        max_cells: usize,
        density_params: FineDensityParams,
    ) -> Self {
        let key = fine_cache::fine_base_key(
            seed,
            coarse_tessellation,
            crust,
            features,
            coarse_elevation,
            atmosphere,
            max_cells,
            &density_params,
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
            density_params,
        );
        if matches!(cache, FineCacheMode::Enabled | FineCacheMode::Rebuild) {
            fine_cache::save(key, &base);
        }
        base
    }

    /// Stage 3a: build the expensive, reusable fine-mesh base (steps 1–7 of the
    /// old monolith). Stops short of erosion — that's [`FineSurface::generate`].
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_target(
        seed: u64,
        coarse_tessellation: &Tessellation,
        crust: &Crust,
        features: &FeatureFields,
        coarse_elevation: &Elevation,
        atmosphere: &Atmosphere,
        max_cells: usize,
        density_params: FineDensityParams,
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
    pub fn generate(
        seed: u64,
        base: &FineBase,
        pre_hydrology: &Hydrology,
        params: ErosionParams,
    ) -> Self {
        // Fluvial erosion: carve the interpolated base into real river valleys by
        // evolving crust thickness (isostasy responds). Runs on the fine mesh
        // before final hydrology; sea level is the fixed datum inherited via
        // `base_elevation`. See docs/specs/erosion.md.
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
        let lake_base = terminal_lake_base_levels(&base.tessellation, pre_hydrology);

        // Fault range-front scarps: sharpen active orogen margins so ranges rise
        // along near-linear fronts; erosion then cuts canyons through them and the
        // triangular facets emerge. Applied to the base erosion carves into.
        let mut faulted_base = base.base_elevation.clone();
        apply_fault_scarps(
            &mut faulted_base,
            &base.fields.elevation_fields,
            params.fault_scarp_height,
        );

        // Coupled erode↔precip loop: each pass re-carves the base relief with the
        // rain-shadow precip from the previous pass (windward flanks, wetter,
        // dissect more than lee). Pass 1 erodes with the coarse precip; later
        // passes use the orographic-modulated precip. The precip is always a
        // modulation of the COARSE field on the CURRENT relief, so it tracks the
        // carved ranges. Converges in a couple of passes.
        let iters = params.precip_outer_iters.max(1);
        let mut precip = base.fields.precipitation.clone();
        let mut eroded = faulted_base.clone();
        for outer in 0..iters {
            let t0 = Instant::now();
            eroded = super::erosion::erode(
                &base.tessellation,
                &base.fields.elevation_fields,
                &faulted_base,
                &precip,
                &erodibility,
                &lake_base,
                params,
            );
            let t_erode = t0.elapsed();
            let t1 = Instant::now();
            precip = fine_precipitation(
                &base.tessellation,
                &eroded,
                &base.fields.wind,
                &base.fields.precipitation,
                params.orographic_precip_strength,
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
        super::erosion::glacial_erode(&base.tessellation, &mut eroded, &lake_base, params);
        log::info!("fine mesh: glacial pass {:.2?}", t0.elapsed());

        Self::from_eroded(seed, base, &eroded, &precip, params.lake_evap_strength)
    }

    /// Build the surface (micro-noise elevation + hydrology) from an already-
    /// eroded elevation. `lake_evap_strength > 0` adds the lakes-as-evaporation
    /// pass (re-runs hydrology once with lake-boosted precip).
    pub fn from_eroded(
        seed: u64,
        base: &FineBase,
        eroded: &[f32],
        precipitation: &[f32],
        lake_evap_strength: f32,
    ) -> Self {
        // Cosmetic micro noise rides on the eroded surface; this is the elevation
        // hydrology and rendering consume.
        let mut elev_rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(3));
        let elevation = Elevation::refine_from_base(&base.tessellation, eroded, &mut elev_rng);
        log_resolution_probe(&base.tessellation, &elevation);

        // Correct temperature for the relief erosion carved. `fields.temperature`
        // is the coarse field interpolated onto the fine mesh, so its lapse is
        // baked against the pre-erosion datum (`base_elevation`). Re-apply the
        // lapse delta against the eroded relief (held/sharpened peaks, carved
        // valleys) so basin evaporation sees the terrain it actually drains. Only
        // positive elevation lapses (matches `generate_surface_temperature`); this
        // is a no-op for the pre-erosion surface (eroded == base_elevation).
        let temperature: Vec<f32> = (0..base.tessellation.num_cells())
            .map(|i| {
                let delta = eroded[i].max(0.0) - base.base_elevation[i].max(0.0);
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
        let mut precip = precipitation.to_vec();
        let mut hydrology = hydro(&precip);
        log::info!("fine mesh: hydrology {:.2?}", t0.elapsed());

        // Lakes as evaporation sources: standing water adds local humidity, so
        // boost precip in a halo around the lakes and re-run hydrology once. Local
        // (no transport on the fine mesh), one pass = no runaway lake growth.
        if lake_evap_strength > 0.0 {
            let t0 = Instant::now();
            precip = boost_precip_near_lakes(
                &base.tessellation,
                &elevation.values,
                &precip,
                &hydrology,
                lake_evap_strength,
            );
            hydrology = hydro(&precip);
            log::info!("fine mesh: lake-evap + re-hydrology {:.2?}", t0.elapsed());
        }

        Self {
            elevation,
            hydrology,
            precipitation: precip,
        }
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
) -> Vec<f32> {
    let n = tess.num_cells();
    if strength <= 0.0 {
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
        .map(|i| elevation.slope(tessellation, i))
        .fold(0.0_f32, f32::max)
        .max(1e-6);
    // flow_count_equiv (not raw discharge) so the .max(1.0) log floor stays in
    // count units after hydrology became area-weighted.
    let max_flow_ln = (0..n)
        .map(|i| preview_hydrology.flow_count_equiv(i).max(1.0).ln())
        .fold(0.0_f32, f32::max)
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
        let slope = (elevation.slope(tessellation, i) / max_slope).powf(e);
        let flow = (preview_hydrology.flow_count_equiv(i).max(1.0).ln() / max_flow_ln).powf(e);
        let activity = features.activity[i].clamp(0.0, 1.0).powf(e);
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
    let mut wind = Vec::with_capacity(n);

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
